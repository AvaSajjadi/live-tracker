"""Tests for run_control: what the dashboard does to a run, with a fake launcher (nothing is started)."""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from live_tracker import monitor_runner as mr
from live_tracker import run_control as rc
from live_tracker import run_schedule as rs
from live_tracker.rules import Rule

NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 10, 24, 18, 0, tzinfo=NY)


def approved_schedule(seconds=3600):
    return rs.RunSchedule("S1", "duration", "America/New_York", T0, T0 + timedelta(seconds=seconds)).approve()


def approved_rule():
    return Rule(rule_id="R1", behavior="attack", min_duration_s=2.0, status="approved")


def config(tmp_path, **kw):
    return rc.build_config(tmp_path / "run1", approved_schedule(), [approved_rule()], {"kind": "synthetic"}, **kw)


# ---- building and starting ----

def test_build_config_refuses_what_is_not_approved(tmp_path):
    proposed = rs.RunSchedule("S1", "duration", "America/New_York", T0, T0 + timedelta(hours=1))
    with pytest.raises(mr.RunRefusedError, match="schedule is not approved"):
        rc.build_config(tmp_path, proposed, [approved_rule()], {"kind": "synthetic"})
    with pytest.raises(mr.RunRefusedError, match="every rule"):
        rc.build_config(tmp_path, approved_schedule(), [], {"kind": "synthetic"})
    with pytest.raises(mr.RunRefusedError, match="every rule"):
        rc.build_config(tmp_path, approved_schedule(), [Rule("R2", "attack", 2.0)], {"kind": "synthetic"})


def test_start_run_writes_the_config_and_calls_the_launcher(tmp_path):
    launched = []
    run_dir = rc.start_run(config(tmp_path), launcher=lambda path: launched.append(path) or 4242)
    assert launched == [run_dir / "config.json"] and (run_dir / "pid.txt").read_text().strip() == "4242"
    assert mr.RunConfig.load(run_dir / "config.json").source == {"kind": "synthetic"}


def test_start_run_fails_before_launching_when_the_config_is_bad(tmp_path):
    cfg = config(tmp_path)
    cfg.rules = [{**cfg.rules[0], "status": "proposed"}]
    launched = []
    with pytest.raises(mr.RunRefusedError):
        rc.start_run(cfg, launcher=lambda path: launched.append(path) or 1)
    assert launched == [] and not (tmp_path / "run1").exists()


def test_new_run_dirs_never_collide(tmp_path):
    now = datetime(2026, 10, 24, 22, 0, 5, tzinfo=timezone.utc)
    a = rc.new_run_dir("S1", now, tmp_path)
    a.mkdir(parents=True)
    b = rc.new_run_dir("S1", now, tmp_path)
    assert a.name == "S1_20261024T220005Z" and b.name == "S1_20261024T220005Z_2"


def test_runs_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(rc.RUNS_ENV, str(tmp_path / "elsewhere"))
    assert rc.runs_root() == tmp_path / "elsewhere"


# ---- stop and answers ----

def test_stop_run_writes_the_stop_file(tmp_path):
    assert not rc.stop_requested(tmp_path)
    rc.stop_run(tmp_path)
    assert rc.stop_requested(tmp_path)


def test_answer_event_appends_a_line_and_checks_the_answer(tmp_path):
    rc.answer_event(tmp_path, "E-1", "confirmed")
    rc.answer_event(tmp_path, "E-2", "rejected")
    lines = [json.loads(l) for l in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert [(l["event_id"], l["answer"]) for l in lines] == [("E-1", "confirmed"), ("E-2", "rejected")]
    with pytest.raises(ValueError):
        rc.answer_event(tmp_path, "E-3", "maybe")
    with pytest.raises(ValueError):
        rc.answer_event(tmp_path, "", "confirmed")


# ---- reading ----

def test_read_status_is_none_for_missing_or_half_written_files(tmp_path):
    assert rc.read_status(tmp_path) is None
    (tmp_path / "status.json").write_text('{"state": "run', encoding="utf-8")
    assert rc.read_status(tmp_path) is None
    (tmp_path / "status.json").write_text('{"state": "running"}', encoding="utf-8")
    assert rc.read_status(tmp_path) == {"state": "running"}


def test_is_stale_only_for_active_runs_without_a_recent_heartbeat():
    now = datetime(2026, 10, 24, 18, 5, tzinfo=timezone.utc)
    fresh = {"state": "running", "heartbeat_at": (now - timedelta(seconds=5)).isoformat()}
    old = {"state": "running", "heartbeat_at": (now - timedelta(minutes=2)).isoformat()}
    assert not rc.is_stale(fresh, now) and rc.is_stale(old, now)
    assert not rc.is_stale({"state": "finished", "heartbeat_at": old["heartbeat_at"]}, now)
    assert rc.is_stale({"state": "waiting"}, now)                              # an active run with no heartbeat at all
    assert not rc.is_stale(None, now)


def run_a_demo(tmp_path):
    """A real (fast) run of the demo, so the reading functions see genuine files."""
    from runner_fixtures import make_runner
    runner, _ = make_runner(tmp_path)
    runner.run()
    return runner.run_dir


def test_read_events_status_and_report_of_a_finished_run(tmp_path):
    run_dir = run_a_demo(tmp_path)
    events = rc.read_events(run_dir)
    assert [e["action"] for e in events] == ["alert", "ask"]
    status = rc.read_status(run_dir)
    assert status["state"] == "finished" and rc.progress(status) > 0
    md, js = rc.read_report(run_dir)
    assert "Run report: S1" in md and json.loads(js)["run"]["outcome"] == "complete"


def test_pending_answers_are_the_queued_ones_the_log_does_not_show_yet(tmp_path):
    run_dir = run_a_demo(tmp_path)
    write_status(run_dir, "running", datetime.now(timezone.utc))                    # the run is still going
    ask = [e for e in rc.read_events(run_dir) if e["action"] == "ask"][0]
    assert rc.answer_event(run_dir, ask["event_id"], "confirmed") == "queued"
    assert rc.pending_answers(run_dir) == {ask["event_id"]: "confirmed"}


def test_read_events_survives_missing_and_broken_logs(tmp_path):
    assert rc.read_events(tmp_path) == [] and rc.read_report(tmp_path) is None
    (tmp_path / "journal.jsonl").write_text("garbage\n", encoding="utf-8")
    assert rc.read_events(tmp_path) == []


def test_list_runs_newest_first_and_only_real_run_folders(tmp_path):
    import os
    for name, age in (("old", 100), ("new", 0)):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        os.utime(d, (1_700_000_000 - age, 1_700_000_000 - age))
    (tmp_path / "not_a_run").mkdir()
    assert [r["name"] for r in rc.list_runs(tmp_path)] == ["new", "old"]
    assert rc.list_runs(tmp_path / "missing") == []


def test_progress_is_the_share_of_the_window_that_has_passed():
    s = approved_schedule(3600)
    status = {"schedule": s.to_dict(), "clock_now": (T0 + timedelta(minutes=15)).isoformat()}
    assert rc.progress(status) == pytest.approx(0.25)
    assert rc.progress({**status, "clock_now": (T0 + timedelta(hours=3)).isoformat()}) == 1.0
    assert rc.progress({**status, "clock_now": (T0 - timedelta(hours=1)).isoformat()}) == 0.0
    assert rc.progress(None) == 0.0 and rc.progress({"schedule": {}, "clock_now": "x"}) == 0.0


# ---- what can be started ----

def test_available_sources_say_why_something_is_off(tmp_path):
    sources = {s["kind"]: s for s in rc.available_sources(tmp_path)}
    assert sources["synthetic"]["enabled"]
    assert not sources["keypoints"]["enabled"] and "CalMS21 train file" in sources["keypoints"]["reason"]
    assert not sources["camera"]["enabled"] and "pose model" in sources["camera"]["reason"]


def test_available_sources_enable_recorded_keypoints_when_everything_is_there(tmp_path):
    (tmp_path / "data/raw/calms21").mkdir(parents=True)
    (tmp_path / "data/raw/calms21/calms21_task1_train.json").write_text("{}", encoding="utf-8")
    (tmp_path / "models").mkdir()
    for name in ("model_b.joblib", "model_b.json"):
        (tmp_path / "models" / name).write_text("x", encoding="utf-8")
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits/split_manifest.json").write_text(json.dumps({"validation": ["task1/train/v1"]}), encoding="utf-8")
    assert {s["kind"]: s["enabled"] for s in rc.available_sources(tmp_path)}["keypoints"]
    assert rc.validation_sessions(tmp_path) == ["task1/train/v1"]
    assert rc.validation_sessions(tmp_path / "nothing") == []


# ---- is the program still alive? ----

def test_pid_alive():
    import os
    import subprocess
    import sys
    assert rc.pid_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not rc.pid_alive(proc.pid)
    assert not rc.pid_alive("not a pid") and not rc.pid_alive(None)


def write_status(run_dir, state, beat):
    (run_dir / "status.json").write_text(json.dumps({"state": state, "heartbeat_at": beat.isoformat()}), encoding="utf-8")


def test_run_is_over_cases(tmp_path):
    import os
    import subprocess
    import sys
    now = datetime.now(timezone.utc)
    assert not rc.run_is_over(tmp_path)                                               # no status yet
    write_status(tmp_path, "finished", now - timedelta(days=1))
    assert rc.run_is_over(tmp_path)
    write_status(tmp_path, "running", now)
    assert not rc.run_is_over(tmp_path)                                               # a fresh heartbeat
    write_status(tmp_path, "running", now - timedelta(minutes=5))
    assert not rc.run_is_over(tmp_path)                                               # stale, but no pid to check
    (tmp_path / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    assert not rc.run_is_over(tmp_path)                                               # stale, process still alive
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    (tmp_path / "pid.txt").write_text(str(gone.pid), encoding="utf-8")
    assert rc.run_is_over(tmp_path)                                                   # stale and the process is gone


# ---- answers ----

def test_an_answer_to_a_live_run_is_queued_and_the_log_is_not_touched(tmp_path):
    run_dir = run_a_demo(tmp_path)
    write_status(run_dir, "running", datetime.now(timezone.utc))
    before = (run_dir / "journal.jsonl").read_bytes()
    ask = [e for e in rc.read_events(run_dir) if e["action"] == "ask"][0]
    assert rc.answer_event(run_dir, ask["event_id"], "confirmed") == "queued"
    assert (run_dir / "journal.jsonl").read_bytes() == before
    assert json.loads((run_dir / "answers.jsonl").read_text().splitlines()[0])["event_id"] == ask["event_id"]


def test_a_late_answer_goes_into_the_log_and_the_report_is_rebuilt(tmp_path):
    run_dir = run_a_demo(tmp_path)
    assert "1 unanswered" in (run_dir / "report.md").read_text(encoding="utf-8")
    ask = [e for e in rc.read_events(run_dir) if e["action"] == "ask"][0]
    assert rc.answer_event(run_dir, ask["event_id"], "confirmed") == "applied"
    assert [e["researcher_answer"] for e in rc.read_events(run_dir) if e["action"] == "ask"] == ["confirmed"]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "1 confirmed, 0 rejected, 0 unanswered" in report and "Outcome: COMPLETE" in report
    assert not (run_dir / "answers.jsonl").exists()
    assert json.loads((run_dir / "report.json").read_text())["summary"] is not None
    assert rc.answer_event(run_dir, ask["event_id"], "confirmed") == "applied"        # the same answer again is harmless


def test_a_late_answer_cannot_change_an_earlier_one_or_name_a_missing_event(tmp_path):
    run_dir = run_a_demo(tmp_path)
    ask = [e for e in rc.read_events(run_dir) if e["action"] == "ask"][0]
    rc.answer_event(run_dir, ask["event_id"], "confirmed")
    with pytest.raises(ValueError, match="not recorded"):
        rc.answer_event(run_dir, ask["event_id"], "rejected")
    with pytest.raises(ValueError, match="not recorded"):
        rc.answer_event(run_dir, "E-nothing-1", "confirmed")


def test_an_answer_to_a_dead_run_goes_straight_into_the_log(tmp_path):
    import subprocess
    import sys
    run_dir = run_a_demo(tmp_path)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    (run_dir / "pid.txt").write_text(str(gone.pid), encoding="utf-8")
    write_status(run_dir, "running", datetime.now(timezone.utc) - timedelta(hours=1))
    ask = [e for e in rc.read_events(run_dir) if e["action"] == "ask"][0]
    assert rc.answer_event(run_dir, ask["event_id"], "rejected") == "applied"
    assert [e["researcher_answer"] for e in rc.read_events(run_dir) if e["action"] == "ask"] == ["rejected"]
