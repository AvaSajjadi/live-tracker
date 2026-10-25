"""The Live demo page, driven with Streamlit's AppTest. No program is started and no real video is played."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from live_tracker import demo
from live_tracker import demo_ui
from live_tracker import run_control as rc
from runner_fixtures import make_runner
from test_demo import MANIFEST, NAME, make_root

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "7_Demo.py")


def texts(elements):
    return " ".join(e.value for e in elements)


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = make_root(tmp_path / "root")
    (root / "models").mkdir()
    for ext in (".joblib", ".json"):
        (root / "models" / f"model_b{ext}").write_text("x", encoding="utf-8")
    monkeypatch.setattr(rc, "ROOT", root)
    monkeypatch.setenv(rc.RUNS_ENV, str(tmp_path / "runs"))
    launched = []
    monkeypatch.setattr(rc, "LAUNCHER", lambda path: launched.append(path) or 4321)
    return root, launched, tmp_path


def open_page(state=None):
    at = AppTest.from_file(PAGE, default_timeout=60)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    return at


def start_button(at):
    return [b for b in at.button if b.key == "demo_start"]


def write_status(run_dir, state, **extra):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({
        "state": state, "heartbeat_at": datetime.now(timezone.utc).isoformat(), "frames": 0, "alerts": 0, "asks": 0,
        **extra}), encoding="utf-8")


def started(env):
    """Open the page, click Start, and return (at, run_dir)."""
    root, launched, tmp = env
    at = open_page()
    start_button(at)[0].click().run()
    assert not at.exception and len(launched) == 1
    return at, Path(launched[0]).parent


# ---- before the demo ----

def test_without_a_demo_video_the_page_says_where_to_put_one(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    at = open_page()
    assert "No demo video found" in texts(at.info) and "scripts/prepare_demo_session.py" in texts(at.info)


def test_without_model_b_the_page_says_so(env):
    root, _, _ = env
    for f in (root / "models").iterdir():
        f.unlink()
    at = open_page()
    assert "Model B is not on this computer" in texts(at.warning) and not start_button(at)


def test_the_intro_shows_the_rule_the_length_and_a_start_button(env):
    at = open_page()
    shown = texts(at.markdown)
    assert "Simulated live stream" in texts(at.caption) and "No camera is used" in texts(at.caption)
    assert "Attack for at least 0.5 s" in shown and "0:10" in shown and f"task1/train/{NAME}" in shown
    assert "not calibrated" in shown
    assert start_button(at)[0].label == "Confirm the rule and start the demo"
    assert [e.label for e in at.expander] == ["Sync"]


def test_start_launches_the_run_and_waits_for_the_program(env):
    at, run_dir = started(env)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["wait_for_go"] is True and cfg["source"]["kind"] == "poses_file"
    assert "Getting ready" in texts(at.info) or "Waiting for the run program" in texts(at.caption)
    assert at.session_state["demo_run"] == str(run_dir) and at.session_state["demo_go_sent"] is False


# ---- the moment the video starts ----

def test_when_the_run_is_ready_the_video_is_drawn_and_go_is_sent(env):
    at, run_dir = started(env)
    write_status(run_dir, "ready")
    at.session_state["demo_offset"] = 0.0
    at.run()
    assert not at.exception
    assert at.session_state["demo_video"] is True and at.session_state["demo_go_sent"] is True
    assert (run_dir / "GO").exists()
    assert len(at.get("video")) == 1


def test_go_is_sent_only_once_and_not_before_the_run_is_ready(env):
    at, run_dir = started(env)
    write_status(run_dir, "waiting")
    at.run()
    assert not (run_dir / "GO").exists() and not at.session_state["demo_video"] and not at.get("video")
    write_status(run_dir, "ready")
    at.session_state["demo_offset"] = 0.0
    at.run()
    (run_dir / "GO").unlink()
    write_status(run_dir, "running")
    at.run()
    assert not (run_dir / "GO").exists()                                         # a later rerun does not send it again


# ---- while the demo runs ----

def finished_demo(root_for_runs):
    runner, _ = make_runner(root_for_runs)
    runner.run()
    return runner.run_dir


def running_state(run_dir):
    return {"demo_run": str(run_dir), "demo_name": NAME, "demo_video": True, "demo_go_sent": True, "demo_offset": 0.0}


def test_the_live_panel_shows_the_latest_alert_the_counts_and_the_table(env):
    root, _, tmp = env
    run_dir = finished_demo(tmp / "runs")
    write_status(run_dir, "running", frames=240, alerts=1, asks=1, max_lag_s=0.12)
    at = open_page(running_state(run_dir))
    assert {m.label: m.value for m in at.metric} == {"Alerts": "1", "Questions": "1"}
    assert "Question" in texts(at.warning) and "possible attack" in texts(at.warning)       # the ask at 0:07 is the latest
    assert "Keeping up with real time (slowest delay 0.1 s)" in texts(at.caption)
    table = at.dataframe[0].value
    assert list(table["action"]) == ["ask", "alert"] and list(table["time"]) == ["0:07", "0:02"]
    assert len(at.get("video")) == 1


def test_falling_behind_is_shown_as_a_warning(env):
    root, _, tmp = env
    run_dir = finished_demo(tmp / "runs")
    write_status(run_dir, "running", frames=240, alerts=1, asks=1, max_lag_s=2.4)
    at = open_page(running_state(run_dir))
    assert "fell behind at times (slowest delay 2.4 s)" in texts(at.warning)


def test_a_failed_run_shows_why_and_start_again_returns_to_the_intro(env):
    root, _, tmp = env
    run_dir = tmp / "runs" / "DEMO_x"
    write_status(run_dir, "failed", message="refused: the schedule is not approved")
    at = open_page(running_state(run_dir))
    assert "The run could not start: refused: the schedule is not approved" in texts(at.error)
    [b for b in at.button if b.key == "demo_again"][0].click().run()
    assert not at.exception and "demo_run" not in at.session_state and start_button(at)


def test_a_finished_demo_says_so_and_shows_the_report(env):
    root, _, tmp = env
    run_dir = finished_demo(tmp / "runs")
    write_status(run_dir, "finished", frames=720, alerts=1, asks=1, max_lag_s=0.0)
    at = open_page(running_state(run_dir))
    assert "The demo has finished" in texts(at.success)
    assert "Run report: S1" in texts(at.markdown)
    assert [b for b in at.button if b.key == "demo_again"]


def test_stop_the_demo_leaves_a_stop_file(env):
    root, _, tmp = env
    run_dir = finished_demo(tmp / "runs")
    write_status(run_dir, "running", frames=100, alerts=0, asks=0)
    at = open_page(running_state(run_dir))
    [b for b in at.button if b.key == "demo_stop"][0].click().run()
    assert (run_dir / "STOP").exists()
