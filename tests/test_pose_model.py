"""Tests for pose_model (the slot for the pose model) and camera_check."""

import json

import numpy as np
import pandas as pd
import pytest

from live_tracker import camera_check as cc
from live_tracker import pose_model as pm
from live_tracker import run_sources as rsrc
from live_tracker.calms21 import KEYPOINTS


def good():
    return pd.DataFrame([{"mouse": m, "keypoint": k, "x": 100.0 + i, "y": 200.0, "score": 0.9}
                         for m in "ab" for i, k in enumerate(KEYPOINTS)])


def write_settings(root, text):
    (root / "models/pose").mkdir(parents=True)
    (root / pm.DEFAULT_CONFIG).write_text(text)


def test_no_config_means_no_model_and_a_reason(tmp_path):
    fn, reason = pm.load(tmp_path, env={})
    assert fn is None and "no pose model" in reason


def test_unknown_backend_is_said_plainly(tmp_path):
    write_settings(tmp_path, json.dumps({"backend": "magic"}))
    fn, reason = pm.load(tmp_path, env={})
    assert fn is None and "'magic'" in reason


def test_broken_settings_file_is_said_plainly(tmp_path):
    write_settings(tmp_path, "not json")
    fn, reason = pm.load(tmp_path, env={})
    assert fn is None and "cannot be read" in reason


def test_a_registered_backend_is_loaded_and_its_output_is_checked(tmp_path, monkeypatch):
    monkeypatch.setitem(pm._BACKENDS, "fake", lambda settings, root: lambda image: good())
    write_settings(tmp_path, json.dumps({"backend": "fake"}))
    fn, reason = pm.load(tmp_path, env={})
    assert reason is None and len(fn(np.zeros((2, 2, 3)))) == 14


def test_a_backend_that_fails_to_load_is_reported_by_class_name(tmp_path, monkeypatch):
    def boom(settings, root):
        raise FileNotFoundError("secret path")
    monkeypatch.setitem(pm._BACKENDS, "boom", boom)
    write_settings(tmp_path, json.dumps({"backend": "boom"}))
    fn, reason = pm.load(tmp_path, env={})
    assert fn is None and "FileNotFoundError" in reason and "secret" not in reason


def test_env_names_another_settings_file(tmp_path):
    other = tmp_path / "other.json"
    assert pm.config_path(tmp_path, {pm.CONFIG_ENV: str(other)}) == other


@pytest.mark.parametrize("bad", [
    lambda d: d.drop(columns=["score"]),
    lambda d: d.assign(mouse="c"),
    lambda d: d.assign(keypoint="tail_tip"),
    lambda d: pd.concat([d, d.iloc[:1]]),
    lambda d: d.assign(score=1.5),
    lambda d: d.assign(x=5000.0),
])
def test_bad_output_is_refused(bad):
    with pytest.raises(pm.PoseModelError):
        pm.check_output(bad(good()))


def test_no_mouse_found_is_none():
    assert pm.check_output(None) is None and pm.check_output(good().iloc[0:0]) is None


# ---- the camera in make_source ----

def test_camera_without_a_model_opens_nothing(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(rsrc, "open_camera", lambda d=0: opened.append(d))
    with pytest.raises(rsrc.NoPoseModelError, match="no pose model"):
        rsrc.make_source({"kind": "camera", "device": 0}, tmp_path)
    assert opened == []


class FakeCapture:
    released = False

    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_camera_with_a_model_opens_the_camera_by_number(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(rsrc, "open_camera", lambda d=0: opened.append(d) or FakeCapture())
    src = rsrc.make_source({"kind": "camera", "device": "1", "pose_estimator": lambda im: None}, tmp_path)
    assert opened == [1] and src.kind == "pose" and src.reopen is not None
    src.close()                                          # stops the camera reader thread


def test_camera_device_number_or_address():
    assert rsrc.camera_device("0") == 0 and rsrc.camera_device(2) == 2
    assert rsrc.camera_device(" rtsp://cam/1 ") == "rtsp://cam/1"


# ---- camera_check ----

def test_snapshot_gives_png_and_releases_the_camera():
    cap = FakeCapture()
    png, why = cc.snapshot(0, opener=lambda d: cap)
    assert why is None and png[:4] == b"\x89PNG" and cap.released


def test_snapshot_reports_a_camera_that_will_not_open():
    def opener(d):
        raise rsrc.CameraUnavailableError("could not open the camera 0")
    assert cc.snapshot(0, opener) == (None, "could not open the camera 0")


def test_snapshot_reports_no_picture():
    class Empty(FakeCapture):
        def read(self):
            return False, None
    cap = Empty()
    png, why = cc.snapshot(0, opener=lambda d: cap)
    assert png is None and "no picture" in why and cap.released


def test_check_needs_all_three(tmp_path):
    result = cc.check(0, opener=lambda d: FakeCapture(), root=tmp_path, env={})
    assert result.opencv and result.camera_ok and not result.pose_model_ok and not result.can_analyse


def test_find_cameras_lists_only_working_numbers():
    def opener(d):
        if d != 1:
            raise rsrc.CameraUnavailableError("no")
        return FakeCapture()
    assert cc.find_cameras(opener) == [1]
