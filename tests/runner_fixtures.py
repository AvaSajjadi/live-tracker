"""Shared helpers for the runner tests: an approved schedule, a rule, a list notifier, a manual clock."""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from live_tracker import monitor_runner as mr
from live_tracker import notify
from live_tracker import run_schedule as rs
from live_tracker import run_sources as rsrc

NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 10, 24, 18, 0, tzinfo=NY)
RULE = {"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0, "alert_threshold": 0.8,
        "ask_threshold": 0.5, "alert_mode": "alert", "description": "test rule", "status": "approved"}


class ListNotifier(notify.Notifier):
    def __init__(self):
        self.sent = []

    def send(self, title, message, priority="default"):
        self.sent.append((title, message, priority))

    def titles(self):
        return [t for t, _, _ in self.sent]


def schedule(seconds=60, start=T0, status="approved"):
    s = rs.RunSchedule("S1", "duration", "America/New_York", start, start + timedelta(seconds=seconds))
    return s.approve() if status == "approved" else s


def config(tmp_path, seconds=60, rules=None, source=None, sched=None, **kw):
    return mr.RunConfig(run_dir=str(tmp_path / "run1"), schedule=(sched or schedule(seconds)).to_dict(),
                        rules=[RULE] if rules is None else rules, source=source or {"kind": "synthetic"}, **kw)


def make_runner(tmp_path, source=None, scorer=None, clock_start=None, seconds=60, **kw):
    cfg = config(tmp_path, seconds=seconds, **kw)
    notifier = ListNotifier()
    runner = mr.Runner(cfg, source or rsrc.synthetic_source(), scorer=scorer,
                       clock=mr.ManualClock(clock_start or T0), notifier=notifier,
                       wall_now=lambda: datetime(2026, 10, 24, 18, 0, tzinfo=timezone.utc))
    return runner, notifier


def read_status(runner):
    return json.loads((runner.run_dir / "status.json").read_text(encoding="utf-8"))
