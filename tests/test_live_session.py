"""Tests for live_session: starting a run from a source, and the motion numbers."""

import json

import numpy as np
import pandas as pd
import pytest

from live_tracker import live_session as ls
from live_tracker import monitor_runner as mr
from live_tracker import run_control as rc
from live_tracker.calms21 import KEYPOINTS, MICE
from live_tracker.rules import Rule
from live_tracker.uploads import SessionSource

from pose_fixtures import make_poses


def moving_poses(frames=90, step_a=3.0):
    """Mouse a walks 3 px per frame along x, mouse b stands still; all keypoints at the same spot."""
    rows = []
    for f in range(frames):
        for mouse in MICE:
            for kp in KEYPOINTS:
                x = (100.0 + step_a * f) if mouse == "a" else 400.0
                rows.append({"session": "S", "mouse": mouse, "keypoint": kp, "frame": f, "time_s": f / 30.0,
                             "x": x, "y": 200.0, "score": 1.0})
    return pd.DataFrame(rows)


def save(tmp_path, df, name="p.parquet"):
    path = tmp_path / name
    df.to_parquet(path, index=False)
    return path


def approved_rule():
    return Rule(rule_id="R1", behavior="attack", min_duration_s=0.5, ask_threshold=0.6, status="approved")


def source(tmp_path, kind="video_keypoints", frames=300):
    poses = save(tmp_path, make_poses(frames, session="UPLOAD_x"))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    if kind == "keypoints":
        return SessionSource("keypoints", "k", poses=str(poses), session="UPLOAD_x")
    return SessionSource(kind, "v", video=str(video), poses=str(poses), session="UPLOAD_x")


def test_length_comes_from_the_frames(tmp_path):
    assert ls.length_s(save(tmp_path, make_poses(300))) == pytest.approx(10.0)
    assert ls.length_s(save(tmp_path, make_poses(300)), fps=25.0) == pytest.approx(12.0)


def test_run_spec_checks_the_demo_against_the_split_and_uploads_by_name(tmp_path):
    assert ls.run_spec(source(tmp_path, "video_keypoints"))["kind"] == "upload_poses"
    demo = SessionSource("demo", "d", video="v.mp4", poses="p.parquet", session="task1/train/mouse064_task1_annotator1")
    assert ls.run_spec(demo) == {"kind": "poses_file", "path": "p.parquet", "session": "task1/train/mouse064_task1_annotator1"}
    with pytest.raises(mr.RunRefusedError, match="no keypoints"):
        ls.run_spec(SessionSource("centers", "c", centers="c.csv"))


def test_the_schedule_covers_the_recording_plus_a_margin_and_is_approved():
    s = ls.live_schedule(238.3)
    assert s.status == "approved" and s.schedule_id == "LIVE" and s.planned_s == pytest.approx(238.3 + ls.EXTRA_S, abs=1.0)


def test_start_live_writes_a_config_with_the_researchers_rules(tmp_path, monkeypatch):
    monkeypatch.setenv(rc.RUNS_ENV, str(tmp_path / "runs"))
    launched = []
    run_dir = ls.start_live(source(tmp_path), [approved_rule()], wait_for_go=True,
                            launcher=lambda path: launched.append(path) or 7)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert launched == [run_dir / "config.json"] and cfg["wait_for_go"] is True and cfg["model"] == ls.MODEL
    assert cfg["source"]["kind"] == "upload_poses" and cfg["source"]["session"] == "UPLOAD_x"
    assert [r["rule_id"] for r in cfg["rules"]] == ["R1"] and cfg["schedule"]["status"] == "approved"
    assert cfg["speed"] == 1.0 and cfg["fps"] == 30.0


def test_start_live_without_a_picture_does_not_wait_and_unapproved_rules_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(rc.RUNS_ENV, str(tmp_path / "runs"))
    run_dir = ls.start_live(source(tmp_path, "keypoints"), [approved_rule()], wait_for_go=False, launcher=lambda p: 1)
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["wait_for_go"] is False
    with pytest.raises(mr.RunRefusedError, match="every rule"):
        ls.start_live(source(tmp_path, "keypoints"), [Rule("R2", "attack", 1.0)], wait_for_go=False, launcher=lambda p: 1)


def test_go_alert_and_failure_helpers(tmp_path):
    assert not ls.go_if_ready(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"state": "ready"}), encoding="utf-8")
    assert ls.go_if_ready(tmp_path)
    rc.send_go(tmp_path)
    assert not ls.go_if_ready(tmp_path)
    events = [{"action": "alert", "start_s": 3.0}, {"action": "log", "start_s": 9.0}, {"action": "ask", "start_s": 7.0}]
    assert ls.latest_alert(events)["start_s"] == 7.0 and ls.latest_alert([]) is None
    assert ls.failed_message({"state": "failed", "message": "refused: x"}) == "refused: x"
    assert ls.failed_message({"state": "running"}) is None and ls.failed_message(None) is None


# ---- motion numbers ----

def test_speed_and_distance_of_a_walking_mouse_and_a_still_one(tmp_path):
    series = ls.motion_series(save(tmp_path, moving_poses(90, step_a=3.0)))
    assert len(series) == 90 and list(series.columns) == ["time_s", "speed_a", "speed_b", "distance", "tracking_ok"]
    late = series.iloc[10:]
    assert late["speed_a"].between(89.9, 90.1).all()                      # 3 px per frame at 30 fps is 90 px/s
    assert (late["speed_b"] == 0).all()
    assert series["distance"].iloc[0] == pytest.approx(300.0) and series["distance"].iloc[60] == pytest.approx(120.0)
    assert series["tracking_ok"].all() and series["time_s"].iloc[30] == pytest.approx(1.0)


def test_a_missing_mouse_means_tracking_is_not_ok_and_speed_is_unknown_not_zero(tmp_path):
    df = moving_poses(60)
    df.loc[(df.frame.between(20, 29)) & (df.mouse == "b"), ["x", "y"]] = np.nan
    series = ls.motion_series(save(tmp_path, df))
    assert not series.loc[20:29, "tracking_ok"].any() and series.loc[:19, "tracking_ok"].all() and series.loc[30:, "tracking_ok"].all()
    assert np.isnan(series.loc[22, "speed_b"]) and series.loc[22, "speed_a"] == pytest.approx(90.0)   # never "still" while missing
    assert np.isnan(series.loc[22, "distance"])


def test_speed_uses_only_the_past(tmp_path):
    a = moving_poses(90)
    b = a.copy()
    b.loc[b.frame >= 60, "x"] += 300.0                                    # change the future only
    sa, sb = ls.motion_series(save(tmp_path, a, "a.parquet")), ls.motion_series(save(tmp_path, b, "b.parquet"))
    pd.testing.assert_frame_equal(sa.iloc[:60], sb.iloc[:60])


def test_centers_table_gives_the_same_numbers():
    df = pd.DataFrame({"x_a": 100.0 + 3.0 * np.arange(60), "y_a": 200.0, "x_b": 400.0, "y_b": 200.0})
    series = ls.centers_series(df)
    assert series["speed_a"].iloc[20] == pytest.approx(90.0) and series["speed_b"].iloc[20] == 0
    assert series["distance"].iloc[0] == pytest.approx(300.0) and series["tracking_ok"].all()
