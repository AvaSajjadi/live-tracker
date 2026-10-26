"""Tests for monitor_runner: a whole run with a manual clock (a night takes no time)."""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from live_tracker import monitor_runner as mr
from live_tracker import notify
from live_tracker import run_schedule as rs
from live_tracker import run_sources as rsrc
from live_tracker.fake_scores import build_scores

from pose_fixtures import SESSION, features_of, make_poses, tiny_model
from runner_fixtures import NY, RULE, T0, ListNotifier, config, make_runner, read_status, schedule

# ---- a normal run ----

def test_a_complete_run_writes_the_log_the_report_and_the_messages(tmp_path):
    runner, notifier = make_runner(tmp_path)
    result = runner.run()
    assert result["state"] == "finished" and result["outcome"] == "complete"
    assert (result["alerts"], result["asks"], result["rows"]) == (1, 1, 720)         # the demo: one alert, one ask
    d = runner.run_dir
    assert (d / "journal.jsonl").exists() and (d / "report.md").exists() and (d / "report.json").exists()
    report = (d / "report.md").read_text(encoding="utf-8")
    assert "Outcome: COMPLETE" in report and "Run report: S1" in report and "Session summary" in report
    assert "The video source ended." in report
    body = json.loads((d / "report.json").read_text(encoding="utf-8"))
    assert body["run"]["outcome"] == "complete" and body["summary"] is not None
    assert notifier.titles() == ["ALERT: possible attack", "Question: possible attack", "Run finished"]
    assert "1 question waiting" in notifier.sent[-1][1]
    st = read_status(runner)
    assert st["state"] == "finished" and st["outcome"] == "complete" and st["unanswered"] == 1


def test_a_short_recording_in_a_long_window_is_partial(tmp_path):
    runner, notifier = make_runner(tmp_path, seconds=600)                   # the demo has only 24 s of data
    result = runner.run()
    assert result["outcome"] == "partial"
    report = (runner.run_dir / "report.md").read_text(encoding="utf-8")
    assert "Outcome: PARTIAL" in report and "missing at the end: 9 min 36 s" in report
    assert "Planned length: 10 min 00 s; recorded span: 0 min 24 s" in report
    assert notifier.sent[-1][0] == "Run finished (partial)"


def test_data_longer_than_the_window_is_cut_at_the_end_of_the_window(tmp_path):
    long_scores = build_scores("SYNTHETIC_long", 120.0, (), ())
    runner, _ = make_runner(tmp_path, source=rsrc.ScoreFileSource(long_scores), seconds=60)
    result = runner.run()
    assert result["outcome"] == "complete" and "The schedule ended." in result["message"]
    assert 1790 <= result["rows"] <= 1801                                    # 60 s at 30 fps, not 120 s


def test_the_run_waits_for_the_start_time(tmp_path):
    clock_start = T0 - timedelta(seconds=30)
    runner, _ = make_runner(tmp_path, clock_start=clock_start)
    runner.run()
    assert runner.first_at >= T0 and runner.first_at < T0 + timedelta(seconds=1)
    assert runner.result()["outcome"] == "complete"


def test_the_heartbeat_file_has_what_the_dashboard_needs(tmp_path):
    runner, _ = make_runner(tmp_path)
    runner.run()
    st = read_status(runner)
    for key in ("state", "outcome", "run_id", "schedule_id", "session", "source", "clock_now", "heartbeat_at",
                "frames", "alerts", "asks", "answered", "unanswered", "interruptions", "seconds_left", "schedule"):
        assert key in st
    assert st["session"] == "SYNTHETIC_demo_01" and "SYNTHETIC" in st["source"] and st["frames"] == 720


# ---- stopping ----

class Hook:
    """Wraps a source and calls hook(frame_number, runner) before returning each frame."""

    def __init__(self, inner, hook):
        self.inner, self.hook, self.runner = inner, hook, None
        self.kind, self.paced = inner.kind, inner.paced
        self.session, self.description, self.fps = inner.session, inner.description, inner.fps

    def read(self):
        frame = self.inner.read()
        if frame is not None:
            self.hook(frame.frame, self.runner)
        return frame

    def reconnect(self):
        return self.inner.reconnect()

    def close(self):
        self.inner.close()


def hooked(tmp_path, hook, **kw):
    src = Hook(rsrc.synthetic_source(), hook)
    runner, notifier = make_runner(tmp_path, source=src, **kw)
    src.runner = runner
    return runner, notifier


def test_stop_early_gives_a_stopped_run_with_a_report(tmp_path):
    runner, notifier = hooked(tmp_path, lambda f, r: r.stop() if f == 150 else None)
    result = runner.run()
    assert result["outcome"] == "stopped" and result["rows"] < 200
    assert "Outcome: STOPPED EARLY" in (runner.run_dir / "report.md").read_text(encoding="utf-8")
    assert notifier.sent[-1][0] == "Run stopped"


def test_a_stop_file_stops_the_run(tmp_path):
    def hook(frame, runner):
        if frame == 100:
            (runner.run_dir / "STOP").write_text("", encoding="utf-8")

    runner, _ = hooked(tmp_path, hook)
    assert runner.run()["outcome"] == "stopped"


def test_stop_while_waiting_for_the_start(tmp_path):
    runner, _ = make_runner(tmp_path, clock_start=T0 - timedelta(hours=1))
    runner.run_dir.mkdir(parents=True, exist_ok=True)
    (runner.run_dir / "STOP").write_text("", encoding="utf-8")
    result = runner.run()
    assert result["outcome"] == "no_data" or result["outcome"] == "stopped"
    assert result["rows"] == 0 and result["state"] == "finished"
    assert "no frames reached the monitor" in (runner.run_dir / "report.md").read_text(encoding="utf-8").lower()


# ---- answers ----

def test_an_answer_written_by_the_dashboard_reaches_the_log_and_the_report(tmp_path):
    def hook(frame, runner):
        if frame == 400:                                                      # the ask fired at about frame 270
            with open(runner.run_dir / "answers.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"event_id": f"E-{runner.run_id}-2", "answer": "confirmed"}) + "\n")
                f.write(json.dumps({"event_id": "E-nope-9", "answer": "confirmed"}) + "\n")
                f.write("not json\n")

    runner, notifier = hooked(tmp_path, hook)
    result = runner.run()
    assert result["answered"] == 1 and len(runner.ignored_answers) == 2
    journal = (runner.run_dir / "journal.jsonl").read_text(encoding="utf-8")
    assert '"event_answered"' in journal and '"confirmed"' in journal
    report = (runner.run_dir / "report.md").read_text(encoding="utf-8")
    assert "1 confirmed" in report and "2 answer(s) could not be applied" in report
    assert "waiting" not in notifier.sent[-1][1]


# ---- a source that drops out ----

class Flaky(Hook):
    """Fails at one frame, then needs `failures` reconnect attempts, then continues `jump` frames later."""

    def __init__(self, inner, at_frame, failures, jump):
        super().__init__(inner, lambda f, r: None)
        self.at_frame, self.failures, self.jump = at_frame, failures, jump
        self.broken = False
        self.tries = 0

    def read(self):
        if self.broken:
            raise rsrc.SourceInterrupted("usb reset")
        frame = self.inner.read()
        if frame is not None and frame.frame == self.at_frame:
            self.broken = True
            raise rsrc.SourceInterrupted("usb reset")
        return frame

    def reconnect(self):
        self.tries += 1
        if self.tries <= self.failures:
            return False
        self.broken = False
        for _ in range(self.jump):                                            # the frames that were lost
            self.inner.read()
        return True


def test_an_interruption_is_a_gap_that_is_reported_and_the_run_goes_on(tmp_path):
    src = Flaky(rsrc.synthetic_source(), at_frame=100, failures=2, jump=60)
    runner, notifier = make_runner(tmp_path, source=src)
    result = runner.run()
    assert result["interruptions"] == 1 and result["outcome"] == "complete"
    assert result["rows"] == 720 - 61                                         # frame 100 and the 60 lost frames are absent
    assert "Camera interrupted" in notifier.titles() and "Camera back" in notifier.titles()
    report = (runner.run_dir / "report.md").read_text(encoding="utf-8")
    assert "Interruptions of the video source" in report and "usb reset" in report
    assert "this is not a still mouse" in report


def test_a_source_that_never_comes_back_ends_the_run_at_the_window_end(tmp_path):
    src = Flaky(rsrc.synthetic_source(), at_frame=100, failures=10 ** 6, jump=0)
    runner, notifier = make_runner(tmp_path, source=src, seconds=120)
    result = runner.run()
    assert result["interruptions"] == 1 and result["outcome"] == "partial" and result["state"] == "finished"
    assert runner.interruptions[0]["end"] is not None                         # closed at the end of the run
    assert notifier.titles().count("Camera interrupted") == 1                 # told once, not at every attempt


# ---- bad frames ----

def test_a_frame_the_monitor_refuses_is_counted_and_skipped(tmp_path):
    class Bad(Hook):
        def read(self):
            frame = self.inner.read()
            if frame is not None and frame.frame == 50:
                frame = rsrc.Frame(frame=50, scores={**frame.scores, "time_s": 99.0})        # a time that does not fit
            return frame

    runner, _ = make_runner(tmp_path, source=Bad(rsrc.synthetic_source(), lambda f, r: None))
    result = runner.run()
    assert result["rejected_rows"] == 1 and result["rows"] == 719 and result["outcome"] == "complete"
    assert "1 frame(s) were refused" in (runner.run_dir / "report.md").read_text(encoding="utf-8")


def test_no_frames_at_all_is_no_data(tmp_path):
    class Empty(Hook):
        def read(self):
            return None

    runner, notifier = make_runner(tmp_path, source=Empty(rsrc.synthetic_source(), lambda f, r: None))
    result = runner.run()
    assert result["outcome"] == "no_data" and result["rows"] == 0
    assert notifier.sent[-1][0] == "Run ended with no data" and notifier.sent[-1][2] == "high"


# ---- pose source with live scoring ----

def test_a_pose_source_is_scored_live_and_reaches_the_monitor(tmp_path):
    from live_tracker.live_scoring import LiveScorer
    model, cols = tiny_model(features_of(make_poses(300, seed=3)))
    poses = make_poses(150, seed=4)
    src = rsrc.KeypointFileSource(poses, SESSION)
    scorer = LiveScorer(model, cols, session=SESSION)
    runner, _ = make_runner(tmp_path, source=src, scorer=scorer)
    result = runner.run()
    assert result["rows"] == 150 and result["rejected_rows"] == 0 and result["state"] == "finished"
    assert result["outcome"] in ("complete", "partial")


# ---- refusals ----

def test_refusals(tmp_path):
    with pytest.raises(mr.RunRefusedError, match="schedule is not approved"):
        make_runner(tmp_path, sched=schedule(status="proposed"))
    with pytest.raises(mr.RunRefusedError, match="at least one approved rule"):
        make_runner(tmp_path, rules=[])
    with pytest.raises(mr.RunRefusedError, match="not approved"):
        make_runner(tmp_path, rules=[{**RULE, "status": "proposed"}])
    with pytest.raises(mr.RunRefusedError, match="already ended"):
        make_runner(tmp_path, clock_start=T0 + timedelta(hours=2))
    with pytest.raises(mr.RunRefusedError, match="needed"):
        make_runner(tmp_path, source=rsrc.KeypointFileSource(make_poses(5), SESSION))
    with pytest.raises(mr.RunRefusedError, match="speed"):
        make_runner(tmp_path, speed=0)


def test_a_fast_speed_is_refused_for_a_camera_like_source(tmp_path):
    class Live(Hook):
        paced = False

    src = Live(rsrc.synthetic_source(), lambda f, r: None)
    src.paced = False
    with pytest.raises(mr.RunRefusedError, match="recorded sources"):
        make_runner(tmp_path, source=src, speed=60.0)


def test_an_unknown_config_field_is_refused(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({**json.loads(config(tmp_path).to_json()), "turbo": True}), encoding="utf-8")
    with pytest.raises(mr.RunRefusedError, match="unknown config fields"):
        mr.RunConfig.load(path)


def test_the_config_round_trips(tmp_path):
    cfg = config(tmp_path, speed=60.0)
    path = tmp_path / "c.json"
    path.write_text(cfg.to_json(), encoding="utf-8")
    assert mr.RunConfig.load(path) == cfg


# ---- the command line ----

def test_the_command_line_runs_a_whole_demo_quickly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mr.signal, "signal", lambda *a: None)
    now = datetime.now(timezone.utc)
    sched = rs.RunSchedule("S1", "duration", "America/New_York", now, now + timedelta(seconds=600)).approve()
    # speed 30: the 24 s demo plays in about a second; a much higher speed would let computing time eat the window
    cfg = mr.RunConfig(run_dir=str(tmp_path / "cli"), schedule=sched.to_dict(), rules=[RULE],
                       source={"kind": "synthetic"}, speed=30.0, status_every_s=0.0)
    path = tmp_path / "config.json"
    path.write_text(cfg.to_json(), encoding="utf-8")
    assert mr.main(["--config", str(path)]) == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["state"] == "finished" and result["alerts"] == 1 and result["asks"] == 1
    assert (tmp_path / "cli" / "report.md").exists() and (tmp_path / "cli" / "notifications.log").exists()


def test_the_command_line_refuses_a_bad_run_and_says_why(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mr.signal, "signal", lambda *a: None)
    cfg = config(tmp_path, sched=schedule(status="proposed"))
    path = tmp_path / "config.json"
    path.write_text(cfg.to_json(), encoding="utf-8")
    assert mr.main(["--config", str(path)]) == 2
    status = json.loads((tmp_path / "run1" / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed" and "not approved" in status["message"]
    assert "refused" in capsys.readouterr().err


def test_a_camera_run_is_refused_because_no_pose_model_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mr.signal, "signal", lambda *a: None)
    monkeypatch.setenv("LIVE_TRACKER_POSE_MODEL", str(tmp_path / "no_such_pose_model.json"))     # not the local model of this computer
    cfg = config(tmp_path, source={"kind": "camera", "device": 0})
    path = tmp_path / "config.json"
    path.write_text(cfg.to_json(), encoding="utf-8")
    assert mr.main(["--config", str(path)]) == 2
    assert "pose model" in capsys.readouterr().err


# ---- start on a signal (a video that must begin at the same moment) ----

class GoClock(mr.ManualClock):
    """Creates the GO file once the program has waited `after_s` seconds."""

    def __init__(self, start, run_dir, after_s):
        super().__init__(start)
        self.run_dir, self.after_s = run_dir, after_s

    def sleep(self, seconds):
        super().sleep(seconds)
        if self.real_seconds() >= self.after_s:
            (self.run_dir / mr.GO_FILE).write_text("go", encoding="utf-8")


def go_runner(tmp_path, after_s, **kw):
    cfg = config(tmp_path, wait_for_go=True, **kw)
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = mr.Runner(cfg, rsrc.synthetic_source(), clock=GoClock(T0, run_dir, after_s), notifier=ListNotifier(),
                       wall_now=lambda: datetime(2026, 10, 24, 18, 0, tzinfo=timezone.utc))
    states = []
    original = runner.write_status

    def spy(force=False):
        states.append(runner.state)
        original(force)

    runner.write_status = spy
    return runner, states


def test_a_run_that_waits_for_go_is_ready_first_and_starts_after_the_signal(tmp_path):
    runner, states = go_runner(tmp_path, after_s=3.0)
    result = runner.run()
    assert "ready" in states and states.index("ready") < states.index("running")
    assert runner.first_at >= T0 + timedelta(seconds=3.0)                       # nothing was processed before GO
    assert result["outcome"] == "complete" and result["alerts"] == 1


def test_stop_while_ready_ends_the_run_without_frames(tmp_path):
    runner, _ = go_runner(tmp_path, after_s=10 ** 9)
    (runner.run_dir / "STOP").write_text("", encoding="utf-8")
    result = runner.run()
    assert result["outcome"] == "stopped" and result["rows"] == 0 and result["state"] == "finished"


def test_a_missing_go_ends_the_run_at_the_window_end(tmp_path):
    runner, _ = go_runner(tmp_path, after_s=10 ** 9)
    result = runner.run()
    assert result["outcome"] == "no_data" and result["rows"] == 0 and "The schedule ended." in result["message"]


# ---- the delay meter ----

def test_a_run_that_keeps_up_says_so_and_a_slow_moment_is_reported(tmp_path):
    runner, _ = make_runner(tmp_path)
    runner.run()
    assert runner.max_lag_s == 0.0
    assert "Processing kept up with real time (slowest moment: 0.0 s behind)" in (runner.run_dir / "report.md").read_text(encoding="utf-8")
    assert read_status(runner)["max_lag_s"] == 0.0

    slow, _ = hooked(tmp_path / "slow", lambda f, r: r.clock.sleep(2.0) if f == 100 else None)
    slow.run()
    assert 1.9 <= slow.max_lag_s <= 2.1
    report = (slow.run_dir / "report.md").read_text(encoding="utf-8")
    assert "Processing FELL BEHIND real time at times (slowest moment: 2.0 s behind)" in report
    assert read_status(slow)["max_lag_s"] >= 1.9 and json.loads((slow.run_dir / "report.json").read_text())["run"]["max_lag_s"] >= 1.9


# ---- a model file that does not fit this scikit-learn ----

def model_files(tmp_path, info_extra=None, joblib_bytes=b"not a model"):
    (tmp_path / "models").mkdir(exist_ok=True)
    info = {"features": "basic,motion", "feature_columns": ["a"], **(info_extra or {})}
    (tmp_path / "models" / "model_x.json").write_text(json.dumps(info), encoding="utf-8")
    (tmp_path / "models" / "model_x.joblib").write_bytes(joblib_bytes)


def pose_source():
    return rsrc.KeypointFileSource(make_poses(5), SESSION)


def test_a_model_trained_with_another_scikit_learn_is_refused_with_the_fix(tmp_path):
    model_files(tmp_path, {"sklearn_version": "0.0.1"})
    cfg = config(tmp_path, model="models/model_x")
    with pytest.raises(mr.RunRefusedError) as err:
        mr.build_scorer(cfg, pose_source(), root=tmp_path)
    text = str(err.value)
    assert "trained with scikit-learn 0.0.1" in text and "does not load across versions" in text
    assert "live_tracker.classify train --name model_x" in text


def test_a_model_file_that_cannot_be_loaded_is_explained_not_a_raw_error(tmp_path):
    model_files(tmp_path)                                     # no version recorded, and the file is not a model
    with pytest.raises(mr.RunRefusedError) as err:
        mr.build_scorer(config(tmp_path, model="models/model_x"), pose_source(), root=tmp_path)
    assert "could not be loaded" in str(err.value) and "probably trained with another version" in str(err.value)


def test_a_missing_model_or_name_is_refused_and_a_score_source_needs_none(tmp_path):
    with pytest.raises(mr.RunRefusedError, match="needs a model"):
        mr.build_scorer(config(tmp_path), pose_source(), root=tmp_path)
    assert mr.build_scorer(config(tmp_path), rsrc.synthetic_source(), root=tmp_path) is None


def test_write_json_tries_again_when_windows_locks_the_file(tmp_path, monkeypatch):
    import os
    from live_tracker import monitor_runner
    real, calls = os.replace, []

    def locked_twice(a, b):
        calls.append(1)
        if len(calls) <= 2:
            raise PermissionError(5, "Access is denied")
        return real(a, b)

    monkeypatch.setattr(monitor_runner.os, "replace", locked_twice)
    monitor_runner._write_json(tmp_path / "status.json", {"a": 1})
    assert len(calls) == 3
    assert (tmp_path / "status.json").read_text(encoding="utf-8").count('"a": 1') == 1


def test_a_camera_run_starts_its_time_when_it_is_ready_not_when_it_was_launched(tmp_path):
    late = T0 + timedelta(seconds=45)                                       # loading the model and the camera took 45 s of a 60 s window
    runner, _ = make_runner(tmp_path, clock_start=late, start_when_ready=True)
    assert runner.schedule.start == late and (runner.schedule.end - runner.schedule.start) == timedelta(seconds=60)
    with pytest.raises(mr.RunRefusedError, match="already ended"):          # without the flag the same late start would lose the window
        make_runner(tmp_path / "other", clock_start=T0 + timedelta(seconds=90))
