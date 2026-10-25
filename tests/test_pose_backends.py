"""Tests for pose_backends: the YOLO backend (with a fake model, no weights) and the stub backend."""

import json

import numpy as np
import pandas as pd
import pytest

from live_tracker import pose_backends as pb
from live_tracker import pose_model as pm
from live_tracker.calms21 import KEYPOINTS


class T:
    """Looks like a torch tensor for the two calls the backend makes."""
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Boxes:
    def __init__(self, cls, conf):
        self.cls, self.conf = T(cls), T(conf)

    def __len__(self):
        return len(self.cls.a)


class Result:
    def __init__(self, cls, conf, xy, kp_conf=None):
        self.boxes = Boxes(cls, conf)
        self.keypoints = type("K", (), {"xy": T(xy), "conf": None if kp_conf is None else T(kp_conf)})()


class FakeModel:
    def __init__(self, result):
        self.result, self.seen = result, []

    def predict(self, image, **kw):
        self.seen.append((image.shape, kw))
        return [self.result]


def two_mice():
    xy = np.zeros((2, 7, 2))
    xy[0] = [[100 + j, 200] for j in range(7)]                         # black mouse
    xy[1] = [[700 - j, 300] for j in range(7)]                         # white mouse
    return Result([0, 1], [0.9, 0.8], xy, kp_conf=np.full((2, 7), 0.7))


def install(monkeypatch, tmp_path, result, **settings):
    model = FakeModel(result)
    monkeypatch.setattr(pb, "_load_yolo", lambda path: model)
    (tmp_path / "w.pt").write_text("x")
    (tmp_path / "models/pose").mkdir(parents=True)
    (tmp_path / pm.DEFAULT_CONFIG).write_text(json.dumps({"backend": "yolo", "weights": "w.pt", **settings}))
    return model


def test_the_yolo_backend_gives_14_keypoints_that_pass_the_slot_check(monkeypatch, tmp_path):
    model = install(monkeypatch, tmp_path, two_mice(), device="intel:gpu", imgsz=[320, 512])
    fn, reason = pm.load(tmp_path, env={})
    assert reason is None
    out = fn(np.zeros((570, 1024, 3), np.uint8))
    assert len(out) == 14 and set(out["mouse"]) == {"a", "b"} and list(out["keypoint"][:7]) == list(KEYPOINTS)
    a, b = out[out["mouse"] == "a"], out[out["mouse"] == "b"]
    assert a["x"].iloc[0] == 100 and b["x"].iloc[0] == 700 and np.allclose(out["score"], 0.7)   # class 0 is mouse a (black)
    shape, kw = model.seen[0]
    assert kw["imgsz"] == (320, 512) and kw["device"] == "intel:gpu" and kw["conf"] == pb.DEFAULT_CONF


def test_a_picture_of_another_size_is_resized_to_the_calms21_frame(monkeypatch, tmp_path):
    model = install(monkeypatch, tmp_path, two_mice())
    fn, _ = pm.load(tmp_path, env={})
    fn(np.zeros((480, 640, 3), np.uint8))
    assert model.seen[0][0] == (570, 1024, 3)


def test_the_most_confident_detection_of_each_class_is_used_and_a_missing_mouse_is_left_out(monkeypatch, tmp_path):
    xy = np.zeros((3, 7, 2))
    xy[0], xy[1], xy[2] = 10.0, 500.0, 600.0
    res = Result([0, 0, 0], [0.3, 0.9, 0.5], xy)                        # three black mice, the second is the most confident
    install(monkeypatch, tmp_path, res)
    fn, _ = pm.load(tmp_path, env={})
    out = fn(np.zeros((570, 1024, 3), np.uint8))
    assert set(out["mouse"]) == {"a"} and len(out) == 7 and (out["x"] == 500.0).all()
    assert (out["score"] == 0.9).all()                                  # without per-keypoint confidence the box confidence is used


def test_no_detection_means_lost_tracking(monkeypatch, tmp_path):
    empty = Result(np.zeros(0), np.zeros(0), np.zeros((0, 7, 2)))
    install(monkeypatch, tmp_path, empty)
    fn, _ = pm.load(tmp_path, env={})
    assert fn(np.zeros((570, 1024, 3), np.uint8)) is None


def test_missing_weights_are_said_plainly(tmp_path):
    (tmp_path / "models/pose").mkdir(parents=True)
    (tmp_path / pm.DEFAULT_CONFIG).write_text(json.dumps({"backend": "yolo", "weights": "nothing_here"}))
    fn, reason = pm.load(tmp_path, env={})
    assert fn is None and "FileNotFoundError" in reason


def test_the_stub_backend_follows_the_clock_and_is_not_a_model(monkeypatch, tmp_path):
    rows = [{"mouse": m, "keypoint": k, "frame": f, "x": float(f), "y": 1.0, "score": 0.9}
            for f in range(100, 106) for m in "ab" for k in KEYPOINTS]
    pd.DataFrame(rows).to_parquet(tmp_path / "p.parquet", index=False)
    clock = iter([10.0, 10.0, 10.0 + 2 / 30.0, 99.0])
    monkeypatch.setattr(pb.time, "perf_counter", lambda: next(clock))
    fn = pb.stub_factory({"poses": str(tmp_path / "p.parquet"), "start_frame": 100}, tmp_path)
    img = np.zeros((4, 4, 3), np.uint8)
    assert fn(img)["x"].iloc[0] == 100.0                                # the first call is frame 100
    assert fn(img)["x"].iloc[0] == 100.0
    assert fn(img)["x"].iloc[0] == 102.0                                # two frame times later
    assert fn(img)["x"].iloc[0] == 105.0                                # past the end: the last frame
