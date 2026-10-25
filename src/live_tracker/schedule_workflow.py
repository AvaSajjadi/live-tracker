"""Run-schedule workflow: propose -> Confirm / Reject, one proposal at a time (new file).

The state behind the "Run schedule" part of the Rules page, kept out of the page so it can be
tested without a browser (the same idea as rule_workflow.py). A schedule is only a proposal until
the researcher confirms it; a confirmed schedule is saved as a small JSON file that the monitor
program will read later. Nothing here starts or stops a run.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from . import run_schedule as rs

SAVE_DIR_ENV = "LIVE_TRACKER_SCHEDULE_DIR"
ROOT = Path(__file__).resolve().parents[2]


class DecisionError(RuntimeError):
    """Confirm / Reject could not be done; the proposal is still pending."""


class StaleActionError(RuntimeError):
    """The click belongs to a proposal that is no longer the pending one."""


def save_dir():
    return Path(os.environ.get(SAVE_DIR_ENV) or ROOT / "outputs" / "schedules")


def _utc_now():
    return datetime.now(timezone.utc)


class SchedulePending:
    def __init__(self, schedule, token, signature):
        self.schedule, self.token, self.signature = schedule, token, signature


class ScheduleWorkflow:
    def __init__(self, directory=None, now_fn=_utc_now):
        self.directory = Path(directory) if directory is not None else None   # None: save_dir() when saving
        self.now_fn = now_fn
        self.pending = None
        self.decided = []                 # approved or rejected RunSchedule, in order
        self.error = None                 # why the last proposal was refused
        self._count = 0

    # ---- proposing ----

    def _next_id(self):
        self._count += 1
        return f"S{self._count}"

    def _propose(self, build, signature):
        self.error = None
        self.pending = None
        try:
            schedule = build(self._next_id())
        except rs.ScheduleError as exc:
            self._count -= 1               # the number is only used up by a valid proposal
            self.error = str(exc)
            return None
        self.pending = SchedulePending(schedule, secrets.token_hex(4), signature)
        return schedule

    def propose_for_hours(self, hours, tz, start=None, signature=None):
        return self._propose(lambda sid: rs.for_hours(hours, now=self.now_fn(), tz=tz, start=start, schedule_id=sid),
                             signature)

    def propose_between(self, start_clock, end_clock, tz, signature=None):
        return self._propose(lambda sid: rs.between(start_clock, end_clock, now=self.now_fn(), tz=tz,
                                                    schedule_id=sid), signature)

    def sync(self, signature):
        """Changing an input after proposing drops the pending proposal (nothing is approved by accident)."""
        if self.pending is not None and self.pending.signature != signature:
            self.pending = None
            return True
        return False

    # ---- deciding ----

    def _check(self, token):
        if self.pending is None or self.pending.token != token:
            raise StaleActionError("this proposal is no longer pending")
        return self.pending

    def confirm(self, token):
        pending = self._check(token)
        approved = pending.schedule.approve()
        try:
            self._save(approved)
        except OSError as exc:
            raise DecisionError(f"could not save the schedule ({exc}); it is still pending") from exc
        self.decided.append(approved)
        self.pending = None
        return approved

    def reject(self, token):
        pending = self._check(token)
        self.decided.append(pending.schedule.reject())
        self.pending = None

    def _save(self, schedule):
        folder = self.directory if self.directory is not None else save_dir()
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{schedule.schedule_id}_{schedule.start.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        tmp = folder / (name + ".tmp")
        tmp.write_text(json.dumps(schedule.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, folder / name)

    # ---- for the page ----

    def table(self):
        return [{"schedule": s.schedule_id, "status": s.status, "mode": "for N hours" if s.mode == "duration" else "from - to",
                 "from": f"{s.start:%a %d %b %H:%M}", "to": f"{s.end:%a %d %b %H:%M}",
                 "length": s.length_text(), "time zone": s.tz} for s in self.decided]

    def approved(self):
        return [s for s in self.decided if s.status == "approved"]
