"""Tests for demo.py: finding demo videos, the approved demo rule and schedule, starting a demo."""

import json

import pytest

from live_tracker import demo
from live_tracker import run_control as rc
from live_tracker.evaluate import ATTACK_RULE_V1

from pose_fixtures import make_poses

NAME = "mouse001_task1_annotator1"
MANIFEST = {"seed": 0, "train": [f"task1/train/{NAME}"], "validation": ["task1/train/mouse002_task1_annotator1"],
            "test": ["task1/test/mouse003_task1_annotator1"]}


def make_root(tmp_path, names=(NAME,), with_poses=True, frames=300):
    (tmp_path / "splits").mkdir(parents=True, exist_ok=True)
    (tmp_path / "splits" / "split_manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    folder = tmp_path / demo.VIDEO_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / f"{n}.mp4").write_bytes(b"not really a video")
        if with_poses:
            make_poses(frames, session=f"task1/train/{n}").to_parquet(folder / f"{n}_poses.parquet", index=False)
    return tmp_path


def test_find_demos_needs_a_video_a_poses_file_and_an_allowed_session(tmp_path):
    root = make_root(tmp_path)
    (found,) = demo.find_demos(root)
    assert found.name == NAME and found.session == f"task1/train/{NAME}"
    assert found.video.name == f"{NAME}.mp4" and found.poses.name == f"{NAME}_poses.parquet"


def test_a_video_without_a_poses_file_is_not_a_demo(tmp_path):
    assert demo.find_demos(make_root(tmp_path, with_poses=False)) == []


def test_a_test_session_or_an_unknown_video_is_never_a_demo(tmp_path):
    root = make_root(tmp_path, names=("mouse003_task1_annotator1", "mouse099_task1_annotator1"))
    assert demo.find_demos(root) == []


def test_no_folder_or_no_manifest_means_no_demos(tmp_path):
    assert demo.find_demos(tmp_path) == []
    (tmp_path / demo.VIDEO_DIR).mkdir(parents=True)
    assert demo.find_demos(tmp_path) == []


def test_several_demos_are_listed_in_name_order(tmp_path):
    root = make_root(tmp_path, names=("mouse002_task1_annotator1", NAME))
    assert [d.name for d in demo.find_demos(root)] == [NAME, "mouse002_task1_annotator1"]


def test_model_available(tmp_path):
    assert not demo.model_available(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model_b.joblib").write_text("x", encoding="utf-8")
    assert not demo.model_available(tmp_path)
    (tmp_path / "models" / "model_b.json").write_text("{}", encoding="utf-8")
    assert demo.model_available(tmp_path)


def test_length_comes_from_the_frames(tmp_path):
    (found,) = demo.find_demos(make_root(tmp_path, frames=300))
    assert demo.length_s(found) == pytest.approx(10.0)


def test_the_demo_rule_is_the_validated_rule_and_approved():
    (rule,) = demo.demo_rule()
    assert rule.status == "approved" and rule.rule_id == demo.DEMO_RULE_ID
    assert (rule.behavior, rule.min_duration_s, rule.ask_threshold, rule.alert_threshold, rule.alert_mode) == (
        ATTACK_RULE_V1["behavior"], ATTACK_RULE_V1["min_duration_s"], ATTACK_RULE_V1["ask_threshold"],
        ATTACK_RULE_V1["alert_threshold"], ATTACK_RULE_V1["alert_mode"])
    assert "Attack for at least 0.5 s" in demo.rule_text() and "0.6" in demo.rule_text() and "0.8" in demo.rule_text()


def test_the_schedule_covers_the_video_plus_a_margin():
    s = demo.demo_schedule(238.3)
    assert s.status == "approved" and s.schedule_id == "DEMO"
    assert s.planned_s == pytest.approx(238.3 + demo.EXTRA_S, abs=1.0)


def test_start_demo_writes_a_config_that_waits_for_go_and_launches(tmp_path, monkeypatch):
    monkeypatch.setenv(rc.RUNS_ENV, str(tmp_path / "runs"))
    (found,) = demo.find_demos(make_root(tmp_path / "root"))
    launched = []
    run_dir = demo.start_demo(found, launcher=lambda path: launched.append(path) or 1)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert launched == [run_dir / "config.json"] and run_dir.parent == tmp_path / "runs"
    assert cfg["wait_for_go"] is True and cfg["speed"] == 1.0 and cfg["model"] == demo.MODEL
    assert cfg["source"] == {"kind": "poses_file", "path": str(found.poses), "session": found.session}
    assert [r["status"] for r in cfg["rules"]] == ["approved"] and cfg["schedule"]["status"] == "approved"


def test_go_if_ready_only_for_a_ready_run_that_has_not_been_started(tmp_path):
    assert not demo.go_if_ready(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"state": "waiting"}), encoding="utf-8")
    assert not demo.go_if_ready(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"state": "ready"}), encoding="utf-8")
    assert demo.go_if_ready(tmp_path)
    rc.send_go(tmp_path)
    assert not demo.go_if_ready(tmp_path)


def test_latest_alert_and_failed_message():
    events = [{"action": "alert", "start_s": 3.0}, {"action": "log", "start_s": 9.0}, {"action": "ask", "start_s": 7.0}]
    assert demo.latest_alert(events)["start_s"] == 7.0
    assert demo.latest_alert([{"action": "log", "start_s": 1.0}]) is None and demo.latest_alert([]) is None
    assert demo.failed_message({"state": "failed", "message": "refused: x"}) == "refused: x"
    assert demo.failed_message({"state": "failed"}) == "the run program failed"
    assert demo.failed_message({"state": "running"}) is None and demo.failed_message(None) is None
