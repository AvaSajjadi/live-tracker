"""Tests for the Camera page: USB camera, live video, corners, record and analyze, staged demo (fake camera, no hardware)."""

from pathlib import Path

import cv2
import numpy as np

from streamlit.testing.v1 import AppTest

from live_tracker import camera_check as cc

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "7_Camera.py")
PNG = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()


def texts(elements):
    return " ".join(e.value for e in elements)


def button(at, key):
    return next(b for b in at.button if b.key == key)


class FakeCap:
    def __init__(self):
        self.released = False

    def read(self):
        import time
        time.sleep(0.01)
        return True, np.full((480, 640, 3), 80, dtype=np.uint8)

    def release(self):
        self.released = True


def open_usb(monkeypatch, snapshot):
    from live_tracker import run_sources as rsrc
    monkeypatch.setattr(cc, "snapshot", snapshot)
    monkeypatch.setattr(rsrc, "open_camera", lambda device: FakeCap())      # the live video never touches a real camera
    return AppTest.from_file(PAGE, default_timeout=30).run()


def test_a_working_camera_can_be_chosen_and_the_pose_model_gap_is_said(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRACKER_POSE_MODEL", str(tmp_path / "no_such_pose_model.json"))     # not the local model of this computer
    at = open_usb(monkeypatch, lambda device, opener=None: (PNG, None))
    assert "Press Test the camera to see live video" in texts(at.caption)
    button(at, "upload_test_camera").click().run()
    assert "The camera works." in texts(at.success)
    at.radio(key="upload_cam_mode").set_value("Live real pose (needs a fast pose model)").run()
    assert "pose model" in texts(at.warning)
    button(at, "upload_use_camera").click().run()
    source = at.session_state["flow_source"]
    assert source.kind == "camera" and source.device == "0"


def test_a_camera_that_does_not_open_is_not_offered(monkeypatch):
    at = open_usb(monkeypatch, lambda device, opener=None: (None, "could not open the camera '0'"))
    button(at, "upload_test_camera").click().run()
    assert "cannot be used" in texts(at.error)
    assert not any(b.key == "upload_use_camera" for b in at.button)
    assert "flow_source" not in at.session_state


def test_changing_the_number_asks_for_a_new_test(monkeypatch):
    at = open_usb(monkeypatch, lambda device, opener=None: (PNG, None))
    button(at, "upload_test_camera").click().run()
    at.text_input(key="upload_device").set_value("1").run()
    assert not any(b.key == "upload_use_camera" for b in at.button)


def test_opening_the_camera_page_drops_the_simulated_demo_camera_of_the_upload_page(monkeypatch):
    from live_tracker import flow
    from live_tracker import uploads as up
    monkeypatch.setattr(cc, "snapshot", lambda device, opener=None: (PNG, None))
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state[flow.SOURCE_KEY] = up.SessionSource("demo", "Demo camera (recorded-video simulation)", video="v.mp4", poses="p.parquet",
                                                         session="task1/train/x")
    at.run()
    button(at, "upload_test_camera").click().run()
    assert not at.exception and at.session_state[flow.SOURCE_KEY] is None if flow.SOURCE_KEY in at.session_state else True


def test_the_screen_corners_are_saved_with_the_camera_and_shown_stretched(monkeypatch):
    from live_tracker import screen_map as sm
    big = cv2.imencode(".png", np.full((480, 640, 3), 80, dtype=np.uint8))[1].tobytes()
    at = open_usb(monkeypatch, lambda device, opener=None: (big, None))
    button(at, "upload_test_camera").click().run()
    assert not at.exception
    at.radio(key="upload_cam_mode").set_value("Live real pose (needs a fast pose model)").run()
    button(at, "upload_use_camera").click().run()
    assert at.session_state["flow_source"].screen == sm.check_corners(sm.DEFAULT_CORNERS)
    at.checkbox(key="upload_screen_on").set_value(False).run()
    button(at, "upload_use_camera").click().run()
    assert at.session_state["flow_source"].screen is None


def test_the_staged_demo_can_be_chosen_and_keeps_the_camera_number(monkeypatch):
    from live_tracker import demo as demo_mod
    fake = demo_mod.Demo("clip", "task1/train/clip", Path("clip.mp4"), Path("clip_poses.parquet"))
    monkeypatch.setattr(demo_mod, "find_demos", lambda root=None: [fake])
    at = open_usb(monkeypatch, lambda device, opener=None: (PNG, None))
    button(at, "upload_test_camera").click().run()
    at.radio(key="upload_cam_mode").set_value("Staged demo (saved keypoints of the demo video)").run()
    assert "STAGED DEMO" in texts(at.warning)
    button(at, "upload_use_staged").click().run()
    source = at.session_state["flow_source"]
    assert source.staged_camera and source.device == "0" and source.session == "task1/train/clip"
    at.radio(key="upload_cam_mode").set_value("Live real pose (needs a fast pose model)").run()
    assert "flow_source" not in at.session_state or at.session_state["flow_source"] is None


def test_the_camera_test_shows_live_video_and_stop_releases_the_camera(monkeypatch):
    at = open_usb(monkeypatch, lambda device, opener=None: (PNG, None))
    button(at, "upload_test_camera").click().run()
    assert not at.exception and at.session_state["upload_live"] is True
    cap = at.session_state["camera_live_grabber"].capture
    assert isinstance(cap, FakeCap) and not cap.released                   # the live video holds the camera open
    button(at, "upload_stop_camera").click().run()
    assert cap.released and "camera_live_grabber" not in at.session_state and at.session_state["upload_live"] is False


# ---- record, then analyze with MARS (the real pose route, offline) ----

def test_record_then_analyze_then_use_the_result(monkeypatch, tmp_path):
    import json
    import time
    import cv2
    from live_tracker import mars_job as mj
    from pose_fixtures import make_poses

    monkeypatch.setenv("LIVE_TRACKER_RECORDINGS", str(tmp_path / "recordings"))
    monkeypatch.setenv("LIVE_TRACKER_UPLOADS", str(tmp_path / "uploads"))
    monkeypatch.setattr(mj, "available", lambda: (True, ""))
    launched = []
    monkeypatch.setattr(mj, "LAUNCHER", lambda cmd, job_dir: launched.append((cmd, job_dir)))

    at = open_usb(monkeypatch, lambda device, opener=None: (PNG, None))
    button(at, "upload_test_camera").click().run()
    assert "Record, then analyze" in at.radio(key="upload_cam_mode").value              # the default mode
    assert not at.exception and button(at, "upload_rec_start")                # the default mode is the offline real pose
    button(at, "upload_rec_start").click().run()
    assert at.session_state["camera_live_grabber"].recording
    time.sleep(1.2)
    at.run()
    button(at, "upload_rec_stop").click().run()
    rec = at.session_state["upload_recording"]
    assert rec["frames"] >= 20 and (rec["width"], rec["height"]) == (1024, 570)           # the default corners stretch the screen
    assert cv2.VideoCapture(rec["path"]).get(cv2.CAP_PROP_FRAME_COUNT) == rec["frames"]
    button(at, "upload_mars_start").click().run()
    assert len(launched) == 1 and "camera_live_grabber" not in at.session_state and at.session_state["upload_live"] is False
    job = at.session_state["upload_job"]
    # the job ends: the result file and the done file appear
    make_poses(60, session="camera").to_parquet(mj.result(job), index=False)
    (mj.result(job).parent / "keypoints.parquet.done.json").write_text(json.dumps({"ok": True, "frames": 60, "seconds": 3.0}), encoding="utf-8")
    at.run()
    button(at, "upload_use_recording").click().run()
    assert not at.exception
    source = at.session_state["flow_source"]
    assert source.kind == "video_keypoints" and source.fps == 30.0 and source.session
