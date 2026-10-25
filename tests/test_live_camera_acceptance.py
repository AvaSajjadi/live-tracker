"""Tests for the judging rules of scripts/live_camera_acceptance.py (the measurements are fed in; no camera, no run program)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_camera_acceptance", ROOT / "scripts" / "live_camera_acceptance.py")
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)


def run(**kw):
    base = {"started": True, "start_s": 12.0, "message": None, "final_state": "stopped", "frames": 900, "rows": 895, "rejected": 0,
            "alerts": 2, "asks": 1, "interruptions": 0, "seconds": 30.0, "frames_per_s": 29.8, "worst_window_fps": 28.0, "error": ""}
    return {**base, **kw}


def pose(**kw):
    return {**{"description": "trained model", "stub": False, "pictures": 60, "median_ms": 18.0, "p95_ms": 25.0, "found_share": 1.0}, **kw}


def levels(m):
    return {name: lv for name, lv, _ in acc.judge(m)}


def test_a_good_live_setup_passes_every_check():
    m = {"expect_events": True, "expect_mice": True, "camera": {"fps": 30.0, "size": "1280 x 720"}, "pose": pose(), "run": run()}
    lv = levels(m)
    assert set(lv.values()) == {"PASS"}


def test_a_slow_model_fails_the_speed_check_and_the_keep_up_check():
    m = {"camera": {"fps": 30.0, "size": "x"}, "pose": pose(median_ms=850.0, p95_ms=900.0), "run": run(frames_per_s=1.1, worst_window_fps=0.9, rows=30, frames=33)}
    lv = levels(m)
    assert lv["Pose model speed (33 ms or less)"] == "FAIL" and lv["Keeps up with 30 frames per second"] == "FAIL"


def test_a_15_fps_webcam_is_a_warning_not_a_failure():
    lv = levels({"camera": {"fps": 15.0, "size": "640 x 480"}, "pose": pose(), "run": run(frames_per_s=15.5)})
    assert lv["Camera gives pictures"] == "WARN" and lv["Keeps up with 30 frames per second"] == "WARN"


def test_no_pose_model_stops_the_test_with_the_reason():
    results = acc.judge({"pose": None, "pose_error": "There is no pose model.", "camera": {"fps": 30.0, "size": "x"}, "run": None})
    assert ("Pose model loads", "FAIL", "There is no pose model.") in results


def test_a_stub_skips_the_speed_check_so_a_rehearsal_cannot_pass_for_the_model():
    lv = levels({"pose": pose(stub=True), "run": run()})
    assert lv["Pose model speed (33 ms or less)"] == "SKIP"


def test_a_run_that_never_started_or_crashed_fails():
    assert levels({"pose": pose(), "run": {"started": False, "start_s": 240.0, "message": "boom"}})["Run program starts"] == "FAIL"
    lv = levels({"pose": pose(), "run": run(final_state="failed", error="OSError: camera")})
    assert lv["Runs without a crash or a camera drop-out"] == "FAIL"
    lv = levels({"pose": pose(), "run": run(interruptions=2)})
    assert lv["Runs without a crash or a camera drop-out"] == "FAIL"


def test_missing_scores_and_missing_alerts_are_caught():
    lv = levels({"expect_events": True, "pose": pose(), "run": run(rows=10, alerts=0, asks=0)})
    assert lv["Scores reach the monitor"] == "FAIL" and lv["Alerts or questions appear"] == "FAIL"
    assert levels({"pose": pose(), "run": run(alerts=0, asks=0)})["Alerts or questions appear"] == "INFO"


def test_a_model_that_does_not_find_the_mice_is_caught():
    lv = levels({"expect_mice": True, "pose": pose(found_share=0.2), "run": run()})
    assert lv["Both mice found in the pictures"] == "FAIL"
