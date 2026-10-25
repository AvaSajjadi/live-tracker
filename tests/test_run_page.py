"""The Run page, driven with Streamlit's AppTest. No program is started: the launcher is replaced."""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from live_tracker import run_control as rc
from live_tracker import run_schedule as rs
from live_tracker import schedule_workflow as sw
from live_tracker.rule_workflow import RuleWorkflow
from runner_fixtures import make_runner

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "6_Run.py")


def texts(elements):
    return " ".join(e.value for e in elements)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(rc.RUNS_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv(sw.SAVE_DIR_ENV, str(tmp_path / "schedules"))
    monkeypatch.setattr(rc, "ROOT", tmp_path)                              # no data or models here
    launched = []
    monkeypatch.setattr(rc, "LAUNCHER", lambda path: launched.append(path) or 777)
    return tmp_path, launched


def approved_workflow():
    wf = RuleWorkflow()
    wf.submit("alert me if the mice attack for at least 0.5 seconds")
    wf.confirm(wf.pending.token)
    assert wf.ruleset.active()
    return wf


def save_schedule(root, start, seconds=3600, schedule_id="S1"):
    s = rs.RunSchedule(schedule_id, "duration", "America/New_York", start, start + timedelta(seconds=seconds)).approve()
    folder = root / "schedules"
    folder.mkdir(exist_ok=True)
    (folder / f"{schedule_id}.json").write_text(json.dumps(s.to_dict()), encoding="utf-8")
    return s


def open_page(with_rule=False):
    at = AppTest.from_file(PAGE, default_timeout=60)
    if with_rule:
        at.session_state["rule_workflow"] = approved_workflow()
    at.run()
    assert not at.exception
    return at


def finished_demo(root):
    runner, _ = make_runner(root / "runs")
    runner.run()
    return runner.run_dir


def button(at, key):
    return [b for b in at.button if b.key == key]


# ---- start ----

def test_an_empty_page_says_what_is_missing(env):
    at = open_page()
    assert "No approved rule yet" in texts(at.warning) and "No approved schedule yet" in texts(at.warning)
    assert button(at, "run_start")[0].disabled
    assert "No runs yet." in texts(at.caption)


def test_only_available_sources_are_offered_and_the_others_say_why(env):
    at = open_page()
    assert [o for o in at.radio(key="run_source_choice").options] == ["Demo: made-up scores (SYNTHETIC)"]
    caption = texts(at.caption)
    assert "Camera: not available, needs a mouse pose model" in caption
    assert "Recorded validation session" in caption and "missing:" in caption


def test_start_is_enabled_with_a_rule_and_a_schedule_and_starts_a_run(env):
    root, launched = env
    save_schedule(root, datetime.now(timezone.utc))
    at = open_page(with_rule=True)
    assert not button(at, "run_start")[0].disabled
    button(at, "run_start")[0].click().run()
    assert not at.exception and "Started." in texts(at.success)
    (config_path,) = launched
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assert cfg["source"] == {"kind": "synthetic"} and cfg["schedule"]["status"] == "approved"
    assert [r["status"] for r in cfg["rules"]] == ["approved"]
    assert (Path(config_path).parent / "pid.txt").read_text().strip() == "777"


def test_a_schedule_that_has_ended_is_not_offered(env):
    root, _ = env
    save_schedule(root, datetime.now(timezone.utc) - timedelta(hours=5), seconds=3600)
    at = open_page(with_rule=True)
    assert "No approved schedule yet" in texts(at.warning)
    assert button(at, "run_start")[0].disabled


def test_start_needs_an_approved_rule(env):
    root, _ = env
    save_schedule(root, datetime.now(timezone.utc))
    at = open_page(with_rule=False)
    assert "No approved rule yet" in texts(at.warning) and button(at, "run_start")[0].disabled


# ---- watching a run ----

def test_a_finished_run_shows_status_events_report_and_messages(env):
    root, _ = env
    run_dir = finished_demo(root)
    (run_dir / "notifications.log").write_text("2026-10-24 22:00:00Z | high | ALERT: possible attack | Run S1: ...\n",
                                               encoding="utf-8")
    at = open_page()
    assert at.selectbox(key="run_pick").value == run_dir.name
    assert "Finished" in texts(at.markdown) and "complete" in texts(at.markdown)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Frames"] == "720" and metrics["Alerts"] == "1" and metrics["Questions"] == "1"
    assert metrics["Unanswered"] == "1" and metrics["Interruptions"] == "0"
    events = [df for df in at.dataframe if "action" in df.value.columns][0].value
    assert list(events["action"]) == ["alert", "ask"]
    assert "Run report: S1" in texts(at.markdown) and "Session summary" in texts(at.markdown)
    labels = [d.label for d in at.get("download_button")]
    assert labels == ["Download report (Markdown)", "Download report (JSON)"]
    assert "Messages sent" in [e.label for e in at.expander]


def test_answering_a_question_after_the_run_is_recorded_and_the_report_updates(env):
    root, _ = env
    run_dir = finished_demo(root)
    ask = [e for e in rc.read_events(run_dir) if e["action"] == "ask"][0]
    at = open_page()
    button(at, f"run_confirm_{ask['event_id']}")[0].click().run()
    assert not at.exception
    assert [e["researcher_answer"] for e in rc.read_events(run_dir) if e["action"] == "ask"] == ["confirmed"]
    assert "1 confirmed, 0 rejected, 0 unanswered" in (run_dir / "report.md").read_text(encoding="utf-8")
    assert not button(at, f"run_confirm_{ask['event_id']}") and not button(at, f"run_reject_{ask['event_id']}")


def write_status(run_dir, state, beat, **extra):
    (run_dir / "status.json").write_text(json.dumps({"state": state, "heartbeat_at": beat.isoformat(), **extra}), encoding="utf-8")


def live_run(root, beat=None):
    run_dir = finished_demo(root)
    write_status(run_dir, "running", beat or datetime.now(timezone.utc), frames=5, alerts=0, asks=0, unanswered=0,
                 interruptions=0, seconds_left=100, source="demo")
    (run_dir / "report.md").unlink()
    (run_dir / "report.json").unlink()
    return run_dir


def test_a_live_run_has_a_stop_button_and_stop_leaves_a_file(env):
    root, _ = env
    run_dir = live_run(root)
    at = open_page()
    assert "Running" in texts(at.markdown) and "has not reported" not in texts(at.error)
    button(at, f"run_stop_{run_dir.name}")[0].click().run()
    assert (run_dir / "STOP").exists()
    assert "Stop requested" in texts(at.caption) and not button(at, f"run_stop_{run_dir.name}")


def test_a_silent_run_gets_a_warning(env):
    root, _ = env
    run_dir = live_run(root, beat=datetime.now(timezone.utc) - timedelta(minutes=10))
    at = open_page()
    assert "has not reported for a while" in texts(at.error)


def test_a_dead_run_says_so_and_still_takes_answers(env):
    root, _ = env
    run_dir = live_run(root, beat=datetime.now(timezone.utc) - timedelta(minutes=10))
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    (run_dir / "pid.txt").write_text(str(gone.pid), encoding="utf-8")
    ask = [e for e in rc.read_events(run_dir) if e["action"] == "ask"][0]
    at = open_page()
    assert "no longer running" in texts(at.error)
    button(at, f"run_reject_{ask['event_id']}")[0].click().run()
    assert [e["researcher_answer"] for e in rc.read_events(run_dir) if e["action"] == "ask"] == ["rejected"]


def test_a_failed_run_shows_its_message(env):
    root, _ = env
    run_dir = finished_demo(root)
    write_status(run_dir, "failed", datetime.now(timezone.utc), outcome="failed", message="refused: the schedule is not approved")
    at = open_page()
    assert "Failed" in texts(at.markdown) and "refused: the schedule is not approved" in texts(at.caption)


def test_a_run_without_a_status_yet_waits_quietly(env):
    root, _ = env
    run_dir = root / "runs" / "S9_new"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    at = open_page()
    assert "Waiting for the program to write its first status" in texts(at.info)
