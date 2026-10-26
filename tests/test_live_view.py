"""The live view (video one side, analysis the other), driven with Streamlit's AppTest. No program is started."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from live_tracker import live_view
from live_tracker import run_control as rc
from live_tracker import uploads as up
from live_tracker import video_web as vw
from live_tracker.rule_workflow import RuleWorkflow

from pose_fixtures import make_poses
from runner_fixtures import make_runner
from video_fixtures import fake_mp4


def texts(elements):
    return " ".join(e.value for e in elements)


def app():
    from live_tracker import live_view as lv
    lv.render()


@pytest.fixture
def env(tmp_path, monkeypatch):
    for ext in (".joblib", ".json"):
        (tmp_path / "models").mkdir(exist_ok=True)
        (tmp_path / "models" / f"model_b{ext}").write_text("x", encoding="utf-8")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    monkeypatch.setenv(rc.RUNS_ENV, str(tmp_path / "runs"))
    launched = []
    monkeypatch.setattr(rc, "LAUNCHER", lambda path: launched.append(path) or 99)
    return tmp_path, launched


def source(tmp_path, playable=True, with_video=True):
    poses = tmp_path / "up_poses.parquet"
    make_poses(300, seed=1, session="UPLOAD_x").to_parquet(poses, index=False)
    video = None
    if with_video:
        video = tmp_path / "clip.mp4"
        video.write_bytes(fake_mp4(b"avc1" if playable else b"mp4v", seconds=10.0))
    return up.SessionSource("video_keypoints" if with_video else "keypoints", "clip.mp4",
                            video=str(video) if video else None, poses=str(poses), session="UPLOAD_x")


def rule_workflow():
    wf = RuleWorkflow()
    wf.submit("alert me if the mice attack for at least 0.5 seconds")
    wf.confirm(wf.pending.token)
    return wf


def open_view(state=None):
    at = AppTest.from_function(app, default_timeout=60)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def with_source(src, rules=True, **extra):
    state = {"flow_source": src, **extra}
    if rules:
        state["rule_workflow"] = rule_workflow()
    return state


def button(at, key):
    return [b for b in at.button if b.key == key]


# ---- nothing to show yet ----

def test_no_source_points_back_to_upload(env):
    assert "Go back to the Upload step" in texts(open_view().info)


def test_a_camera_without_a_pose_model_says_so_and_has_no_start_button(env, monkeypatch):
    root, _ = env
    monkeypatch.setenv("LIVE_TRACKER_POSE_MODEL", str(root / "no_such_pose_model.json"))     # not the local model of this computer
    at = open_view(with_source(up.SessionSource("camera", "USB camera 0", device="0")))
    assert "There is no pose model" in texts(at.error) and not button(at, "live_start_camera")


def test_a_camera_with_a_pose_model_starts_a_camera_run(env, monkeypatch):
    root, launched = env
    monkeypatch.setenv("LIVE_TRACKER_POSE_MODEL", str(root / "pose_model.json"))
    (root / "pose_model.json").write_text(json.dumps({"backend": "fake"}), encoding="utf-8")
    from live_tracker import pose_model
    pose_model.register_backend("fake", lambda settings, r: (lambda image: None))
    at = open_view(with_source(up.SessionSource("camera", "USB camera 2", device="2")))
    assert "real camera" in texts(at.markdown)
    button(at, "live_start_camera")[0].click().run()
    assert not at.exception and len(launched) == 1
    cfg = json.loads(Path(launched[0]).read_text(encoding="utf-8"))
    assert cfg["source"]["kind"] == "camera" and cfg["source"]["device"] == "2" and cfg["wait_for_go"] is False


def test_a_centers_file_shows_motion_only(env):
    root, _ = env
    import pandas as pd
    path = root / "c.csv"
    pd.DataFrame({"x_a": [100.0 + 3 * i for i in range(90)], "y_a": 200.0, "x_b": 400.0, "y_b": 200.0}).to_csv(path, index=False)
    at = open_view({"flow_source": up.SessionSource("centers", "c.csv", centers=str(path))})
    assert "no behavior scores" in texts(at.caption)
    assert {m.label: m.value for m in at.metric} == {"Frames": "90", "Length": "0:03", "Both mice tracked": "100.0%"}
    assert len(at.get("vega_lite_chart")) == 2


def test_without_an_approved_rule_it_points_back_to_rules(env):
    root, _ = env
    at = open_view(with_source(source(root), rules=False))
    assert "No approved rule yet" in texts(at.warning) and "Rules step" in texts(at.warning)
    assert not button(at, "live_start")


def test_without_model_b_it_says_so(env):
    root, _ = env
    for f in (root / "models").iterdir():
        f.unlink()
    assert "Model B is not on this computer" in texts(open_view(with_source(source(root))).warning)


# ---- before starting ----

def test_the_intro_shows_the_recording_the_rules_and_a_start_button(env):
    root, _ = env
    at = open_view(with_source(source(root)))
    shown = texts(at.markdown)
    assert "Recorded-data simulation" in texts(at.caption) and "not camera images" in texts(at.caption)
    assert "clip.mp4, 0:10 long" in shown and "attack for at least 0.5 s" in shown and "not calibrated" in shown
    assert button(at, "live_start")[0].label == "Start recorded session"


def test_a_video_the_browser_cannot_play_is_explained_and_start_changes_its_label(env):
    root, _ = env
    at = open_view(with_source(source(root, playable=False)))
    assert "cannot be shown in the browser" in texts(at.warning) and "mp4v" in texts(at.warning)
    assert button(at, "live_start")[0].label == "Start without the picture"


def test_a_web_copy_makes_the_video_playable_again(env):
    root, _ = env
    src = source(root, playable=False)
    Path(vw.web_copy_path(src.video)).write_bytes(fake_mp4(b"avc1", seconds=10.0))
    at = open_view(with_source(src))
    assert not at.warning and button(at, "live_start")[0].label == "Start recorded session"


def test_start_with_a_picture_waits_for_go(env):
    root, launched = env
    at = open_view(with_source(source(root)))
    button(at, "live_start")[0].click().run()
    assert not at.exception and len(launched) == 1
    cfg = json.loads(Path(launched[0]).read_text(encoding="utf-8"))
    assert cfg["wait_for_go"] is True and cfg["source"]["kind"] == "upload_poses" and cfg["source"]["session"] == "UPLOAD_x"
    assert [r["behavior"] for r in cfg["rules"]] == ["attack"]
    assert at.session_state["live_go_sent"] is False and "Getting ready" in texts(at.info)


def test_start_without_a_picture_does_not_wait(env):
    root, launched = env
    at = open_view(with_source(source(root, with_video=False)))
    assert button(at, "live_start")[0].label == "Start recorded session"
    button(at, "live_start")[0].click().run()
    assert json.loads(Path(launched[0]).read_text(encoding="utf-8"))["wait_for_go"] is False
    assert at.session_state["live_go_sent"] is True


def write_status(run_dir, state, **extra):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({
        "state": state, "heartbeat_at": datetime.now(timezone.utc).isoformat(), "frames": 0, "alerts": 0, "asks": 0, **extra}),
        encoding="utf-8")


def test_when_ready_the_video_is_drawn_and_go_is_sent_once(env):
    root, launched = env
    at = open_view(with_source(source(root)))
    button(at, "live_start")[0].click().run()
    run_dir = Path(launched[0]).parent
    write_status(run_dir, "ready")
    at.session_state["live_offset"] = 0.0
    at.run()
    assert not at.exception and (run_dir / "GO").exists() and len(at.get("video")) == 1
    (run_dir / "GO").unlink()
    write_status(run_dir, "running", frames=30)
    at.run()
    assert not (run_dir / "GO").exists()


# ---- while it runs ----

def running_state(src, run_dir):
    return with_source(src, live_run=str(run_dir), live_video=True, live_go_sent=True, live_offset=0.0,
                       live_source=src.to_dict())


def demo_run(root):
    runner, _ = make_runner(root / "runs")
    runner.run()
    return runner.run_dir


def test_the_panel_shows_alert_counts_table_and_the_motion_chart_next_to_the_video(env):
    root, _ = env
    src = source(root)
    run_dir = demo_run(root)
    write_status(run_dir, "running", frames=240, alerts=1, asks=1, max_lag_s=0.1)
    at = open_view(running_state(src, run_dir))
    assert len(at.get("video")) == 1
    assert "Alerts: 1" in texts(at.markdown) and "Questions: 1" in texts(at.markdown)
    assert "ASK" in texts(at.warning) and "possible attack" in texts(at.warning)
    assert list(at.dataframe[0].value["action"]) == ["ask", "alert"]
    assert len(at.get("vega_lite_chart")) == 1 and "Motion, last 30 s" in texts(at.markdown)


def test_a_session_without_a_picture_still_shows_the_analysis(env):
    root, _ = env
    src = source(root, with_video=False)
    run_dir = demo_run(root)
    write_status(run_dir, "running", frames=240, alerts=1, asks=1, max_lag_s=0.0)
    at = open_view(running_state(src, run_dir))
    assert not at.get("video") and "Alerts: 1" in texts(at.markdown)


def test_falling_behind_is_a_warning(env):
    root, _ = env
    src = source(root)
    run_dir = demo_run(root)
    write_status(run_dir, "running", frames=240, alerts=1, asks=1, max_lag_s=3.2)
    assert "fell behind at times (slowest delay 3.2 s)" in texts(open_view(running_state(src, run_dir)).warning)


def test_a_failed_run_shows_why_and_start_again_returns_to_the_intro(env):
    root, _ = env
    src = source(root)
    run_dir = root / "runs" / "LIVE_x"
    write_status(run_dir, "failed", message="refused: x")
    at = open_view(running_state(src, run_dir))
    assert "The run could not start: refused: x" in texts(at.error)
    button(at, "live_again")[0].click().run()
    assert not at.exception and "live_run" not in at.session_state and button(at, "live_start")


def test_a_finished_session_shows_the_report(env):
    root, _ = env
    src = source(root)
    run_dir = demo_run(root)
    write_status(run_dir, "finished", frames=720, alerts=1, asks=1, max_lag_s=0.0)
    at = open_view(running_state(src, run_dir))
    assert "The session has finished" in texts(at.success) and "Run report: S1" in texts(at.markdown)


def test_when_the_chosen_time_is_over_the_page_says_the_report_is_ready(env):
    root, _ = env
    src = source(root)
    run_dir = demo_run(root)
    write_status(run_dir, "finished", frames=5400, alerts=1, asks=0, max_lag_s=0.0, message="The schedule ended.")
    at = open_view(running_state(src, run_dir))
    assert "chosen time is over" in texts(at.success) and "report is ready" in texts(at.success)


def test_the_biologist_report_is_written_on_a_click_and_can_be_downloaded(env, monkeypatch):
    from live_tracker import biologist_report as bio
    root, _ = env
    src = source(root)
    run_dir = demo_run(root)
    write_status(run_dir, "finished", frames=720, alerts=1, asks=0, max_lag_s=0.0)
    (run_dir / "report.md").write_text("# Run report", encoding="utf-8")
    (run_dir / "report.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(bio, "write", lambda report_json: calls.append(report_json) or ("# Report for the biologist Text.", None))
    monkeypatch.setattr(bio, "save_and_notify", lambda d, md: "The report is saved. The phone message is off (no ntfy topic is set).")
    at = open_view(running_state(src, run_dir))
    assert calls == [] and not at.get("download_button")                  # nothing is sent to Claude before the click
    button(at, "live_bio_write")[0].click().run()
    assert not at.exception and calls == ["{}"]
    assert "Report for the biologist" in texts(at.markdown) and len(at.get("download_button")) == 1
    assert "phone message is off" in texts(at.caption)


def test_a_failed_biologist_report_shows_the_reason(env, monkeypatch):
    from live_tracker import biologist_report as bio
    root, _ = env
    run_dir = demo_run(root)
    write_status(run_dir, "finished", frames=720, alerts=1, asks=0, max_lag_s=0.0)
    (run_dir / "report.md").write_text("# Run report", encoding="utf-8")
    (run_dir / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bio, "write", lambda report_json: (None, "ANTHROPIC_API_KEY is not set."))
    at = open_view(running_state(source(root), run_dir))
    button(at, "live_bio_write")[0].click().run()
    assert "Not written" in texts(at.error) and not at.get("download_button")


def test_stop_leaves_a_stop_file(env):
    root, _ = env
    src = source(root)
    run_dir = demo_run(root)
    write_status(run_dir, "running", frames=100)
    at = open_view(running_state(src, run_dir))
    button(at, "live_stop")[0].click().run()
    assert (run_dir / "STOP").exists()


def test_choosing_a_different_recording_resets_the_session(env):
    root, _ = env
    src = source(root)
    run_dir = demo_run(root)
    write_status(run_dir, "running", frames=100)
    at = open_view(running_state(src, run_dir))
    assert at.session_state["live_run"] == str(run_dir)
    other = source(root, with_video=False)
    at.session_state["flow_source"] = other
    at.run()
    assert "live_run" not in at.session_state and button(at, "live_start")


def test_without_claude_the_plain_report_is_shown(env, monkeypatch):
    from live_tracker import biologist_report as bio
    root, _ = env
    run_dir = demo_run(root)
    write_status(run_dir, "finished", frames=720, alerts=1, asks=0, max_lag_s=0.0)
    (run_dir / "report.md").write_text("# Run report", encoding="utf-8")
    (run_dir / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bio, "write", lambda report_json: (None, "ANTHROPIC_API_KEY is not set."))
    monkeypatch.setattr(bio, "write_plain", lambda report_json: ("# Report for the biologist plain", None))
    monkeypatch.setattr(bio, "save_and_notify", lambda d, md: "The report is saved.")
    at = open_view(running_state(source(root), run_dir))
    button(at, "live_bio_write")[0].click().run()
    assert not at.exception and "Report for the biologist plain" in texts(at.markdown)
    assert "plain report" in texts(at.caption) and len(at.get("download_button")) == 1


# ---- staged demo: a real camera films a screen, the numbers come from saved keypoints ----

def staged_source(tmp_path):
    base = source(tmp_path)
    return up.SessionSource("demo", "Staged demo: camera 0 films the screen", video=base.video, poses=base.poses,
                            session=base.session, device="0")


class FakeCap:
    def read(self):
        import time
        import numpy as np
        time.sleep(0.01)
        return True, np.full((60, 80, 3), 90, dtype=np.uint8)

    def release(self):
        pass


def test_a_staged_demo_says_what_it_is_and_waits_for_go(env, monkeypatch):
    from live_tracker import run_sources as rsrc
    monkeypatch.setattr(rsrc, "open_camera", lambda device: FakeCap())
    root, launched = env
    at = open_view(with_source(staged_source(root)))
    assert "STAGED DEMO" in texts(at.warning) and "SAVED keypoints" in texts(at.warning)
    button(at, "live_start_staged")[0].click().run()
    assert not at.exception and len(launched) == 1
    cfg = json.loads(Path(launched[0]).read_text(encoding="utf-8"))
    assert cfg["wait_for_go"] is True and cfg["source"]["kind"] == "poses_file"       # the saved keypoints, not the camera
    assert "Getting ready" in texts(at.info) and not button(at, "live_go_staged")


def test_a_staged_demo_shows_the_camera_and_go_is_sent_by_the_button(env, monkeypatch):
    from live_tracker import run_sources as rsrc
    monkeypatch.setattr(rsrc, "open_camera", lambda device: FakeCap())
    root, launched = env
    at = open_view(with_source(staged_source(root)))
    button(at, "live_start_staged")[0].click().run()
    run_dir = Path(launched[0]).parent
    write_status(run_dir, "ready")
    at.run()
    assert not at.exception and isinstance(at.session_state["camera_live_grabber"].capture, FakeCap)   # the camera is open and shown
    assert "STAGED DEMO" in texts(at.warning)
    assert not (run_dir / "GO").exists()                                              # GO waits for the researcher
    button(at, "live_go_staged")[0].click().run()
    assert (run_dir / "GO").exists() and at.session_state["live_go_sent"] is True


def test_finished_without_a_report_does_not_claim_it_is_ready(env):
    root, _ = env
    src = source(root)
    run_dir = root / "runs" / "NO_REPORT"
    write_status(run_dir, "finished", frames=300)
    at = open_view(running_state(src, run_dir))
    assert "report is not available yet" in texts(at.warning)
    assert "report is ready" not in texts(at.success)


def test_live_event_table_keeps_behavior_and_confidence_readable(env):
    root, _ = env
    src = source(root)
    run_dir = demo_run(root)
    write_status(run_dir, "running", frames=240, alerts=1, asks=1)
    at = open_view(running_state(src, run_dir))
    table = at.dataframe[0].value
    assert list(table["action"]) == ["ask", "alert"]
    assert list(table["behavior"]) == ["attack", "attack"]
    assert set(("time", "confidence")) <= set(table.columns)
    assert "model score, not verified behavior" in texts(at.warning)
