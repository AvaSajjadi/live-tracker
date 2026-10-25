"""Tests for run_sources: score and keypoint playback, camera frame numbering, reconnecting, refusals."""

import json

import numpy as np
import pandas as pd
import pytest

from live_tracker import run_sources as rsrc
from live_tracker.calms21 import KEYPOINTS, MICE
from live_tracker.fake_scores import DEMO_SESSION, demo_scores

from pose_fixtures import SESSION, make_poses


# ---- score playback ----

def test_score_file_source_plays_every_row_in_order():
    df = demo_scores()
    src = rsrc.ScoreFileSource(df)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    assert src.session == DEMO_SESSION and len(frames) == len(df)
    assert [f.frame for f in frames] == sorted(f.frame for f in frames)
    row = frames[0].scores
    assert tuple(row) == rsrc.SCORE_COLUMNS and type(row["frame"]) is int and type(row["tracking_ok"]) is bool
    assert frames[0].pose is None and src.kind == "scores" and src.paced


def test_score_file_source_reads_a_csv_and_picks_a_session(tmp_path):
    df = pd.concat([demo_scores(), demo_scores().assign(session="SYNTHETIC_other")])
    path = tmp_path / "scores.csv"
    df.to_csv(path, index=False)
    with pytest.raises(rsrc.SourceRefusedError, match="several sessions"):
        rsrc.ScoreFileSource(path)
    src = rsrc.ScoreFileSource(path, session="SYNTHETIC_other")
    assert src.session == "SYNTHETIC_other" and src.read().scores["session"] == "SYNTHETIC_other"


def test_score_file_source_refuses_bad_tables():
    with pytest.raises(rsrc.SourceRefusedError, match="lacks columns"):
        rsrc.ScoreFileSource(demo_scores().drop(columns=["attack"]))
    with pytest.raises(rsrc.SourceRefusedError, match="no rows"):
        rsrc.ScoreFileSource(demo_scores(), session="nope")


def test_synthetic_source_is_labeled_synthetic():
    src = rsrc.synthetic_source()
    assert src.session.startswith("SYNTHETIC_") and "SYNTHETIC" in src.description


# ---- keypoint playback ----

def test_keypoint_source_gives_one_frame_of_pose_rows_at_a_time():
    poses = make_poses(10, seed=1)
    src = rsrc.KeypointFileSource(poses, SESSION)
    first = src.read()
    assert first.frame == 0 and first.scores is None and len(first.pose) == len(MICE) * len(KEYPOINTS)
    assert set(first.pose["frame"]) == {0} and src.kind == "pose"
    n = 1
    while src.read() is not None:
        n += 1
    assert n == 10
    with pytest.raises(rsrc.SourceRefusedError):
        rsrc.KeypointFileSource(poses, "unknown")


# ---- camera ----

class FakeCapture:
    """Plays a list of (time, ok) steps; read() moves the fake clock to that time."""

    def __init__(self, steps, clock):
        self.steps, self.clock, self.released = list(steps), clock, False

    def read(self):
        t, ok = self.steps.pop(0)
        self.clock["t"] = t
        return ok, f"image@{t}"

    def release(self):
        self.released = True


def good_pose(image):
    rows = [{"mouse": m, "keypoint": k, "x": 100.0, "y": 200.0, "score": 0.9} for m in MICE for k in KEYPOINTS]
    return pd.DataFrame(rows)


def camera(steps, estimator=good_pose, **kw):
    clock = {"t": 0.0}
    return rsrc.CameraSource(FakeCapture(steps, clock), estimator, session="CAM", fps=30.0,
                             now_fn=lambda: clock["t"], **kw), clock


def test_camera_needs_a_pose_model():
    with pytest.raises(rsrc.NoPoseModelError, match="no mouse pose model"):
        rsrc.CameraSource(FakeCapture([], {}), None, session="CAM")
    with pytest.raises(rsrc.NoPoseModelError):
        rsrc.CameraSource(FakeCapture([], {}), "not callable", session="CAM")


def test_camera_frames_are_numbered_from_the_clock_not_counted():
    # jittery timestamps, one duplicate slot (0.001 s after the first) and a drop (frames 3-4 missing)
    steps = [(10.000, True), (10.001, True), (10.034, True), (10.066, True), (10.170, True)]
    cam, _ = camera(steps)
    frames = [cam.read().frame for _ in range(4)]
    assert frames == [0, 1, 2, 5]                       # the duplicate was skipped; frames 3-4 were dropped


def test_camera_pose_rows_have_the_scorer_format():
    cam, _ = camera([(5.0, True), (5.0334, True)])
    f = cam.read()
    assert tuple(f.pose.columns) == ("session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score")
    assert set(f.pose["session"]) == {"CAM"} and set(f.pose["frame"]) == {0}
    f = cam.read()
    assert f.frame == 1 and abs(f.pose["time_s"].iloc[0] - 1 / 30) < 1e-12


def test_no_mouse_found_is_a_lost_frame_not_a_skipped_one():
    cam, _ = camera([(1.0, True), (1.0334, True)], estimator=lambda image: None)
    f = cam.read()
    assert f.frame == 0 and len(f.pose) == 14 and f.pose["x"].isna().all() and (f.pose["score"] == 0).all()


def test_a_failed_read_raises_source_interrupted():
    cam, _ = camera([(1.0, True), (2.0, False)])
    cam.read()
    with pytest.raises(rsrc.SourceInterrupted, match="no picture"):
        cam.read()


def test_a_driver_error_is_an_interruption_too():
    class Broken:
        def read(self):
            raise OSError("usb reset")

        def release(self):
            pass

    cam = rsrc.CameraSource(Broken(), good_pose, session="CAM", now_fn=lambda: 0.0)
    with pytest.raises(rsrc.SourceInterrupted, match="usb reset"):
        cam.read()


def test_reconnect_swaps_the_capture_and_keeps_the_frame_numbers_going():
    clock = {"t": 0.0}
    old = FakeCapture([(1.0, True), (1.0334, False)], clock)
    new = FakeCapture([(4.0, True)], clock)
    cam = rsrc.CameraSource(old, good_pose, session="CAM", fps=30.0, now_fn=lambda: clock["t"], reopen=lambda: new)
    assert cam.read().frame == 0
    with pytest.raises(rsrc.SourceInterrupted):
        cam.read()
    assert cam.reconnect() is True and old.released
    assert cam.read().frame == 90                       # 3 s later: the numbers show the gap


def test_reconnect_fails_without_reopen_or_when_reopen_fails():
    cam, _ = camera([])
    assert cam.reconnect() is False
    cam2 = rsrc.CameraSource(FakeCapture([], {}), good_pose, session="CAM", reopen=lambda: None)
    assert cam2.reconnect() is False
    cam3 = rsrc.CameraSource(FakeCapture([], {}), good_pose, session="CAM", reopen=lambda: 1 / 0)
    assert cam3.reconnect() is False


def test_open_camera_reports_missing_opencv(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_cv2(name, *a, **k):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    with pytest.raises(rsrc.CameraUnavailableError, match="OpenCV is not installed"):
        rsrc.open_camera(0)


# ---- the factory ----

def manifest_root(tmp_path):
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "split_manifest.json").write_text(json.dumps(
        {"seed": 0, "train": ["task1/train/a"], "validation": ["task1/train/b"], "test": ["task1/test/c"]}), encoding="utf-8")
    return tmp_path


def test_make_source_synthetic_and_scores(tmp_path):
    assert rsrc.make_source({"kind": "synthetic"}).session == DEMO_SESSION
    path = tmp_path / "s.csv"
    demo_scores().to_csv(path, index=False)
    src = rsrc.make_source({"kind": "scores", "path": str(path)})
    assert src.session == DEMO_SESSION and "s.csv" in src.description


def test_make_source_refuses_test_sessions_and_unknown_things(tmp_path):
    root = manifest_root(tmp_path)
    with pytest.raises(rsrc.SourceRefusedError, match="never used"):
        rsrc.make_source({"kind": "keypoints", "json": "x.json", "session": "task1/test/c"}, root)
    with pytest.raises(rsrc.SourceRefusedError, match="never used"):
        rsrc.make_source({"kind": "keypoints", "json": "calms21_task1_test.json", "session": "task1/train/b"}, root)
    with pytest.raises(rsrc.SourceRefusedError, match="not in the train or validation"):
        rsrc.make_source({"kind": "keypoints", "json": "x.json", "session": "task1/other"}, root)
    with pytest.raises(rsrc.SourceRefusedError, match="unknown source kind"):
        rsrc.make_source({"kind": "telepathy"}, root)


def test_make_source_camera_says_a_pose_model_is_missing():
    with pytest.raises(rsrc.NoPoseModelError):
        rsrc.make_source({"kind": "camera", "device": 0})


# ---- one saved session (the demo source) ----

def test_make_source_poses_file_plays_a_saved_session(tmp_path):
    root = manifest_root(tmp_path)
    poses = make_poses(20, seed=2, session="task1/train/b")
    poses.to_parquet(tmp_path / "one.parquet", index=False)
    src = rsrc.make_source({"kind": "poses_file", "path": "one.parquet", "session": "task1/train/b"}, root)
    assert src.session == "task1/train/b" and src.kind == "pose" and "recorded keypoints" in src.description
    assert src.read().frame == 0


def test_poses_file_refuses_test_sessions_other_sessions_and_wrong_files(tmp_path):
    root = manifest_root(tmp_path)
    make_poses(5, session="task1/train/b").to_parquet(tmp_path / "one.parquet", index=False)
    with pytest.raises(rsrc.SourceRefusedError, match="never used"):
        rsrc.make_source({"kind": "poses_file", "path": "one.parquet", "session": "task1/test/c"}, root)
    with pytest.raises(rsrc.SourceRefusedError, match="never used"):
        rsrc.make_source({"kind": "poses_file", "path": "calms21_test_poses.parquet", "session": "task1/train/b"}, root)
    with pytest.raises(rsrc.SourceRefusedError, match="not in the train or validation"):
        rsrc.make_source({"kind": "poses_file", "path": "one.parquet", "session": "task1/other"}, root)
    with pytest.raises(rsrc.SourceRefusedError, match="does not hold exactly"):
        rsrc.make_source({"kind": "poses_file", "path": "one.parquet", "session": "task1/train/a"}, root)
