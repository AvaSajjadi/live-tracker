"""The run program: it does the night's work on the lab computer (new file).

    python -m live_tracker.monitor_runner --config outputs/runs/<run>/config.json

It reads a run folder's config (an approved schedule, the approved rules, a frame source), waits for
the start time, then reads frames until the end time, one at a time:
    frame -> (pose -> live scores) -> monitor -> event log, messages to the researcher
At the end it writes the report. The dashboard only starts and stops it and shows what it wrote, so
closing the browser never stops a run.

Files in the run folder:
    config.json          what to run (written by whoever starts the run)
    journal.jsonl        the event log (append-only; this program is its only writer)
    status.json          a heartbeat for the dashboard, rewritten every few seconds
    answers.jsonl        the researcher's answers to "ask" events (written by the dashboard)
    STOP                 create this file to stop the run early
    report.md / .json    the end-of-run report (written from the event log, plus a run section)
    notifications.log    every message that was sent

Rules this program keeps:
  - it refuses a schedule that is not approved and rules that are not approved;
  - the monitor only sees frames inside the schedule window;
  - it never reads the CalMS21 test split (run_sources refuses it);
  - a lost camera is a gap in the frame numbers, never a still mouse; the report lists every gap;
  - a restart inside the window starts a NEW run folder and a new report (a run is not resumed).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import notify as notify_mod
from . import run_schedule as rs
from . import summary as summary_mod
from .event_log import JournalIntegrityError
from .monitor import Monitor, MonitorInputError
from .rules import RuleSet
from .run_sources import SourceInterrupted

ROOT = Path(__file__).resolve().parents[2]
STATES = ("waiting", "running", "finishing", "finished", "failed")
OUTCOMES = ("complete", "partial", "stopped", "no_data", "failed")


class RunRefusedError(ValueError):
    """The run must not start: the schedule, the rules or the source are not acceptable."""


# ---- clocks ----

class RealClock:
    """The wall clock. speed > 1 makes a demo run faster than real time (recorded sources only)."""

    def __init__(self, speed=1.0, now_fn=lambda: datetime.now(timezone.utc), mono=time.monotonic, sleep=time.sleep):
        self.speed, self._mono, self._sleep = float(speed), mono, sleep
        self._anchor, self._m0 = now_fn(), mono()

    def now(self):
        return self._anchor + timedelta(seconds=(self._mono() - self._m0) * self.speed)

    def sleep(self, seconds):
        self._sleep(max(0.0, seconds) / self.speed)

    def real_seconds(self):
        return self._mono()


class ManualClock:
    """A clock that only moves when the program sleeps; tests use it so a night takes no time."""

    def __init__(self, start):
        self._now, self._elapsed = start, 0.0

    def now(self):
        return self._now

    def sleep(self, seconds):
        self._now += timedelta(seconds=max(0.0, seconds))
        self._elapsed += max(0.0, seconds)

    def real_seconds(self):
        return self._elapsed


# ---- the config ----

@dataclass
class RunConfig:
    run_dir: str
    schedule: dict                      # RunSchedule.to_dict(), status "approved"
    rules: list                         # approved Rule dicts
    source: dict                        # run_sources.make_source spec
    model: str | None = None            # "models/model_b" for pose sources
    speed: float = 1.0
    fps: float = 30.0
    status_every_s: float = 5.0         # real seconds between heartbeats
    reconnect_wait_s: float = 2.0       # first wait before a reconnect attempt; doubles up to 60 s

    def to_json(self):
        return json.dumps(asdict(self), indent=2) + "\n"

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        extra = set(data) - known
        if extra:
            raise RunRefusedError(f"unknown config fields: {sorted(extra)}")
        return cls(**data)


def rules_from_dicts(dicts):
    """Approved rules from the dicts a dashboard saved. A file that is not marked approved is refused."""
    if not dicts:
        raise RunRefusedError("no rules: a run needs at least one approved rule")
    rule_set = RuleSet()
    for d in dicts:
        if d.get("status") != "approved":
            raise RunRefusedError(f"rule {d.get('rule_id', '?')} is not approved")
        rule_set.add({**d, "status": "proposed"})          # the only way in is add(), then approve()
        rule_set.approve(d["rule_id"])
    return rule_set.active()


def _write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _duration(seconds):
    """'8 h 05 min' for long spans, '9 min 36 s' for short ones."""
    seconds = int(round(max(0.0, seconds)))
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"
    return f"{seconds // 60} min {seconds % 60:02d} s"


def _safe_id(text):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in text)[:64] or "run"


# ---- the runner ----

class Runner:
    def __init__(self, config, source, *, scorer=None, clock=None, notifier=None,
                 wall_now=lambda: datetime.now(timezone.utc)):
        self.cfg = config
        self.run_dir = Path(config.run_dir)
        self.schedule = rs.RunSchedule.from_dict(config.schedule)
        self.rules = rules_from_dicts(config.rules)
        if self.schedule.status != "approved":
            raise RunRefusedError("the schedule is not approved")
        if not config.speed > 0:
            raise RunRefusedError("speed must be more than 0")
        if config.speed != 1.0 and not getattr(source, "paced", False):
            raise RunRefusedError("a faster-than-real-time speed only works for recorded sources, not a camera")
        if source.kind == "pose" and scorer is None:
            raise RunRefusedError("this source gives pose keypoints, so a scorer (model) is needed")
        self.source, self.scorer = source, scorer
        self.clock = clock or RealClock(config.speed)
        if self.schedule.status_at(self.clock.now()) == "finished":
            raise RunRefusedError("the schedule has already ended")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.notifier = notifier or notify_mod.from_env(self.run_dir / "notifications.log")
        self.wall_now = wall_now
        self.run_id = _safe_id(f"{self.schedule.schedule_id}_{self.schedule.start.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"
                               f"_{self.run_dir.name}")
        self.stop_requested = False
        self.state, self.outcome, self.message = "waiting", None, ""
        self.n_frames = self.n_rows = self.n_rejected = 0
        self.first_frame = self.last_frame = None
        self.first_time_s = self.last_time_s = None
        self.first_at = self.last_at = None
        self.alerts = self.asks = self.logs = self.answered = 0
        self.interruptions = []             # {"start", "end", "reason"} in clock times
        self.ignored_answers = []
        self._answers_offset = 0
        self._anchor = None
        self._last_beat = None
        self.monitor = None

    # ---- stop ----

    def stop(self):
        self.stop_requested = True

    def _should_stop(self):
        return self.stop_requested or (self.run_dir / "STOP").exists()

    # ---- the status file ----

    def write_status(self, force=False):
        beat = self.clock.real_seconds()
        if not force and self._last_beat is not None and beat - self._last_beat < self.cfg.status_every_s:
            return
        self._last_beat = beat
        _write_json(self.run_dir / "status.json", {
            "state": self.state, "outcome": self.outcome, "message": self.message,
            "run_id": self.run_id, "schedule_id": self.schedule.schedule_id,
            "schedule": self.schedule.to_dict(), "session": getattr(self.source, "session", None),
            "source": getattr(self.source, "description", ""), "speed": self.cfg.speed,
            "clock_now": self.clock.now().isoformat(), "heartbeat_at": self.wall_now().isoformat(),
            "frames": self.n_frames, "rows_to_monitor": self.n_rows, "rejected_rows": self.n_rejected,
            "last_frame": self.last_frame, "alerts": self.alerts, "asks": self.asks, "logs": self.logs,
            "answered": self.answered, "unanswered": max(0, self.asks - self.answered),
            "interruptions": len(self.interruptions), "interrupted_now": bool(self.interruptions and self.interruptions[-1]["end"] is None),
            "seconds_left": self.schedule.seconds_left(self.clock.now()),
        })

    # ---- answers from the dashboard ----

    def apply_answers(self):
        path = self.run_dir / "answers.jsonl"
        if self.monitor is None or not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            f.seek(self._answers_offset)
            text = f.read()
            self._answers_offset = f.tell()
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                self.monitor.answer(item["event_id"], item["answer"])
                self.answered += 1
            except (KeyError, ValueError, TypeError, JournalIntegrityError) as exc:
                self.ignored_answers.append(f"{line.strip()[:80]} ({type(exc).__name__})")

    # ---- the loop ----

    def run(self):
        try:
            return self._run()
        except Exception as exc:                          # a night run must leave a clear trace, not just die
            self.state, self.outcome = "failed", "failed"
            self.message = f"{type(exc).__name__}: {exc}"
            (self.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self.write_status(force=True)
            try:
                self.notifier.send("Run FAILED", f"Run {self.schedule.schedule_id} stopped with an error: {self.message}", "high")
            except Exception:
                pass
            return self.result()
        finally:
            try:
                self.source.close()
            except Exception:
                pass

    def _run(self):
        self.write_status(force=True)
        while self.schedule.status_at(self.clock.now()) == "pending":      # wait for the start time
            if self._should_stop():
                return self._finish("stopped")
            self.write_status()
            self.clock.sleep(min(1.0, self.schedule.seconds_to_start(self.clock.now())))

        self.state = "running"
        self.monitor = Monitor(self.rules, fps=self.cfg.fps, journal_path=self.run_dir / "journal.jsonl", run_id=self.run_id)
        self.write_status(force=True)
        reason = "window_end"
        while True:
            if self._should_stop():
                reason = "stopped"
                break
            if self.schedule.status_at(self.clock.now()) == "finished":
                break
            try:
                frame = self.source.read()
            except SourceInterrupted as exc:
                if not self._handle_interruption(str(exc)):
                    reason = "stopped" if self._should_stop() else "window_end"
                    break
                continue
            if frame is None:
                reason = "source_ended"
                break
            if getattr(self.source, "paced", False):
                if self._anchor is None:                        # recorded sources play at their own speed from here
                    self._anchor = (frame.frame, self.clock.now())
                anchor_frame, anchor_clock = self._anchor
                due = anchor_clock + timedelta(seconds=(frame.frame - anchor_frame) / self.cfg.fps)
                while self.clock.now() < due and not self._should_stop():
                    self.write_status()
                    self.clock.sleep(min(0.25, (due - self.clock.now()).total_seconds()))
                if self.schedule.status_at(self.clock.now()) == "finished":
                    break
            self._process(frame)
            self.write_status()
            self.apply_answers()
        return self._finish(reason)

    def _process(self, frame):
        self.n_frames += 1
        if frame.scores is not None:
            rows = [frame.scores]
        else:
            rows = self.scorer.push(frame.frame, frame.pose)
        self._feed(rows, frame.frame)

    def _feed(self, rows, _frame_hint=None):
        for row in rows:
            try:
                events = self.monitor.step(row)
            except MonitorInputError:
                self.n_rejected += 1
                continue
            now = self.clock.now()
            self.n_rows += 1
            if self.first_frame is None:
                self.first_frame, self.first_time_s, self.first_at = row["frame"], row["time_s"], now
            self.last_frame, self.last_time_s, self.last_at = row["frame"], row["time_s"], now
            for event in events:
                self._announce(event, now)

    def _announce(self, event, now):
        action = event.get("action")
        if action == "alert":
            self.alerts += 1
        elif action == "ask":
            self.asks += 1
        else:
            self.logs += 1
        if action in ("alert", "ask"):
            when = now.astimezone(self.schedule.start.tzinfo).strftime("%H:%M")
            title, message, priority = notify_mod.event_message(event, self.schedule.schedule_id, when)
            self.notifier.send(title, message, priority)
        self.write_status(force=True)

    def _handle_interruption(self, reason):
        """The source stopped. Try to reconnect until it works, the window ends, or the researcher stops the run."""
        entry = {"start": self.clock.now(), "end": None, "reason": reason}
        self.interruptions.append(entry)
        self.notifier.send("Camera interrupted", f"Run {self.schedule.schedule_id}: the video source stopped ({reason}). Trying to reconnect.", "high")
        self.write_status(force=True)
        wait = self.cfg.reconnect_wait_s
        while not self._should_stop() and self.schedule.status_at(self.clock.now()) != "finished":
            if self.source.reconnect():
                entry["end"] = self.clock.now()
                self.notifier.send("Camera back", f"Run {self.schedule.schedule_id}: the video source is back.", "default")
                self.write_status(force=True)
                return True
            self.write_status()
            self.clock.sleep(wait)
            wait = min(60.0, wait * 2)
        return False

    # ---- the end ----

    def _finish(self, reason):
        self.state = "finishing"
        self.write_status(force=True)
        if self.monitor is not None:
            if self.scorer is not None:
                self._feed(self.scorer.flush())
            for entry in self.interruptions:
                if entry["end"] is None:
                    entry["end"] = self.clock.now()            # still down at the end of the run
            self.monitor.finish()
            self.apply_answers()
        completeness = rs.completeness(self.schedule, self.first_at, self.last_at)
        self.outcome = "stopped" if reason == "stopped" else completeness["outcome"]
        self.message = {"window_end": "The schedule ended.", "source_ended": "The video source ended.",
                        "stopped": "Stopped by the researcher."}.get(reason, reason)
        self._write_report(reason, completeness)
        self.state = "finished"
        self.write_status(force=True)
        title, message, priority = notify_mod.finish_message(
            self.schedule.schedule_id, self.outcome, self.alerts, self.asks, max(0, self.asks - self.answered))
        self.notifier.send(title, message, priority)
        return self.result()

    def result(self):
        return {"state": self.state, "outcome": self.outcome, "message": self.message, "run_id": self.run_id,
                "alerts": self.alerts, "asks": self.asks, "answered": self.answered, "frames": self.n_frames,
                "rows": self.n_rows, "rejected_rows": self.n_rejected, "interruptions": len(self.interruptions)}

    def _write_report(self, reason, completeness):
        run_info = {"reason": reason, "outcome": self.outcome, "plan": self.schedule.describe(),
                    "completeness": completeness, "source": getattr(self.source, "description", ""),
                    "interruptions": [{"start": e["start"].isoformat(), "end": e["end"].isoformat() if e["end"] else None,
                                       "reason": e["reason"]} for e in self.interruptions],
                    "rejected_rows": self.n_rejected, "ignored_answers": self.ignored_answers}
        summary = None
        if self.n_rows:
            recording = {"n_rows": self.n_rows, "first_frame": self.first_frame, "last_frame": self.last_frame,
                         "first_time_s": self.first_time_s, "last_time_s": self.last_time_s, "fps": self.cfg.fps,
                         "source": run_info["source"] or "run"}
            answered = {e["event_id"]: None for e in self.monitor.journal.events(self.run_id)
                        if e.get("researcher_answer") is not None}
            summary = summary_mod.build_summary(self.monitor.journal.records, run_id=self.run_id,
                                                session=self.source.session, rules=self.rules,
                                                recording=recording, answer_positions=answered)
        (self.run_dir / "report.md").write_text(self._markdown(summary, run_info), encoding="utf-8")
        _write_json(self.run_dir / "report.json", {"run": run_info, "summary": summary.as_dict() if summary else None})

    def _markdown(self, summary, info):
        c = info["completeness"]
        label = {"complete": "COMPLETE", "partial": "PARTIAL", "stopped": "STOPPED EARLY", "no_data": "NO DATA",
                 "failed": "FAILED"}[info["outcome"]]
        lines = [f"# Run report: {self.schedule.schedule_id}", "", f"**Outcome: {label}.** {self.message}", "",
                 "## Run", "", f"- Plan: {info['plan']}", f"- Source: {info['source'] or 'unknown'}",
                 f"- Planned length: {_duration(c['planned_s'])}; recorded span: {_duration(c['recorded_s'])}",
                 f"- Missing at the start: {_duration(c['missing_at_start_s'])}; missing at the end: {_duration(c['missing_at_end_s'])}"]
        if info["interruptions"]:
            lines += ["- Interruptions of the video source (no frames were seen; this is not a still mouse):"]
            for e in info["interruptions"]:
                lines.append(f"  - from {e['start']} to {e['end'] or 'the end'}: {e['reason']}")
        else:
            lines.append("- No interruptions of the video source.")
        if info["rejected_rows"]:
            lines.append(f"- {info['rejected_rows']} frame(s) were refused by the monitor (bad timing or scores) and skipped.")
        if info["ignored_answers"]:
            lines.append(f"- {len(info['ignored_answers'])} answer(s) could not be applied.")
        lines += ["", "---", ""]
        lines.append(summary.to_markdown() if summary else "_No frames reached the monitor, so there is no event summary._")
        return "\n".join(lines) + "\n"


# ---- command line ----

def build_scorer(config, source, root=ROOT):
    if source.kind != "pose":
        return None
    if not config.model:
        raise RunRefusedError("a pose source needs a model, for example models/model_b")
    import joblib
    from .live_scoring import LiveScorer
    base = Path(config.model) if Path(config.model).is_absolute() else Path(root) / config.model
    info = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    model = joblib.load(base.with_suffix(".joblib"))
    return LiveScorer(model, info["feature_columns"], session=source.session, features=info.get("features", "basic,motion"),
                      fps=config.fps)


def main(argv=None):
    from .run_sources import make_source
    ap = argparse.ArgumentParser(description="Run the monitor for an approved schedule")
    ap.add_argument("--config", required=True, help="path to the run folder's config.json")
    args = ap.parse_args(argv)
    config = RunConfig.load(args.config)
    try:
        source = make_source({**config.source, "fps": config.fps})
        runner = Runner(config, source, scorer=build_scorer(config, source))
    except (RunRefusedError, rs.ScheduleError, ValueError, RuntimeError) as exc:
        run_dir = Path(config.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "status.json", {"state": "failed", "outcome": "failed", "message": f"refused: {exc}",
                                              "heartbeat_at": datetime.now(timezone.utc).isoformat()})
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: runner.stop())
    result = runner.run()
    print(json.dumps(result))
    return 1 if result["outcome"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
