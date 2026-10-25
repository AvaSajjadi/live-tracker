"""Run schedule: when the monitor runs, and whether a finished run was complete.

New file for Live Tracker (written by us, not part of the starter).

A researcher sets a run in one of two ways:
  - for N hours (starting now, or at a chosen time), or
  - from one clock time to another ("from 18:00 to 06:00").
Either way the run ends by itself; the end-of-run report is written from the event log
(summary.py), not here. Like a rule, a schedule is only a proposal until the researcher
approves it: `may_run()` is false for anything that is not approved.

Plain deterministic Python with no clock of its own: every function that needs "now"
takes it as an argument, so tests can use fixed times.

Time rules:
  - All instants are time-zone aware and stored in the lab's IANA zone ("America/New_York").
  - "For N hours" counts real elapsed hours, so a night with a clock change is not shifted.
  - A clock time that does not exist (spring forward) moves to the next valid minute.
    A clock time that happens twice (fall back) means its first occurrence.
  - If the end clock time is not later than the start, the end is on the next day.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MODES = ("duration", "window")
STATUSES = ("proposed", "approved", "rejected")
MIN_RUN_S = 60.0                     # a run shorter than a minute is almost surely a typing mistake
MAX_RUN_S = 7 * 24 * 3600.0          # longer than a week: ask the researcher to check
DEFAULT_TOLERANCE_S = 60.0           # how far the first / last frame may be from the planned start / end


class ScheduleError(ValueError):
    """The schedule is not valid; the message says what to fix."""


def _zone(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ScheduleError(f"unknown time zone {name!r} (use a name like 'America/New_York')") from exc


def _aware(value, label):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ScheduleError(f"{label} must be a date and time with a time zone")
    return value


def _local(day, clock, zone):
    """The wall-clock time `clock` on `day` in `zone`; a time that does not exist moves forward."""
    wall = datetime.combine(day, clock.replace(tzinfo=None), tzinfo=zone)
    return wall.astimezone(timezone.utc).astimezone(zone)


def _utc(value):
    """Compare and subtract in UTC: inside one zone Python works on the wall clock, which is wrong at a clock change."""
    return value.astimezone(timezone.utc)


def _plus_hours(start, hours, zone):
    return (_utc(start) + timedelta(hours=hours)).astimezone(zone)


@dataclass(frozen=True)
class RunSchedule:
    schedule_id: str
    mode: str                 # "duration" or "window"
    tz: str                   # IANA name, the lab's zone
    start: datetime           # aware, in tz
    end: datetime             # aware, in tz
    status: str = "proposed"  # "proposed" until the researcher approves it
    description: str = ""

    def __post_init__(self):
        if self.mode not in MODES:
            raise ScheduleError(f"mode must be one of {list(MODES)}, got {self.mode!r}")
        if self.status not in STATUSES:
            raise ScheduleError(f"status must be one of {list(STATUSES)}, got {self.status!r}")
        zone = _zone(self.tz)
        start = _aware(self.start, "start").astimezone(zone)
        end = _aware(self.end, "end").astimezone(zone)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if self.planned_s <= 0:
            raise ScheduleError("the end must be after the start")
        if self.planned_s < MIN_RUN_S:
            raise ScheduleError(f"a run must last at least {MIN_RUN_S:g} seconds")
        if self.planned_s > MAX_RUN_S:
            raise ScheduleError(f"a run may last at most {MAX_RUN_S / 3600:g} hours; please check the times")

    # ---- what the schedule says ----

    @property
    def planned_s(self):
        return (_utc(self.end) - _utc(self.start)).total_seconds()      # real elapsed seconds

    def status_at(self, now):
        """'pending' before the start, 'running' from the start up to (not including) the end, then 'finished'."""
        now = _utc(_aware(now, "now"))
        if now < _utc(self.start):
            return "pending"
        return "running" if now < _utc(self.end) else "finished"

    def seconds_to_start(self, now):
        """Seconds until the start (0 once it has started)."""
        return max(0.0, (_utc(self.start) - _utc(_aware(now, "now"))).total_seconds())

    def seconds_left(self, now):
        """Seconds until the end (0 once finished)."""
        return max(0.0, (_utc(self.end) - _utc(_aware(now, "now"))).total_seconds())

    def length_text(self):
        hours, rest = divmod(round(self.planned_s), 3600)
        return f"{hours} h {rest // 60:02d} min"

    def describe(self):
        return (f"From {self.start:%a %d %b %H:%M} to {self.end:%a %d %b %H:%M} "
                f"({self.length_text()}, {self.tz})")

    # ---- the approval gate ----

    def approve(self):
        return replace(self, status="approved")

    def reject(self):
        return replace(self, status="rejected")

    def may_run(self, now):
        """True only for an approved schedule whose time has come and not gone."""
        return self.status == "approved" and self.status_at(now) == "running"

    # ---- saving ----

    def to_dict(self):
        return {"schedule_id": self.schedule_id, "mode": self.mode, "tz": self.tz,
                "start": self.start.isoformat(), "end": self.end.isoformat(),
                "status": self.status, "description": self.description}

    @classmethod
    def from_dict(cls, d):
        try:
            return cls(schedule_id=str(d["schedule_id"]), mode=d["mode"], tz=d["tz"],
                       start=datetime.fromisoformat(d["start"]), end=datetime.fromisoformat(d["end"]),
                       status=d.get("status", "proposed"), description=str(d.get("description", "")))
        except KeyError as exc:
            raise ScheduleError(f"missing field {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ScheduleError):
                raise
            raise ScheduleError(f"cannot read the schedule: {exc}") from exc


# ---- the two ways to set a run ----

def for_hours(hours, *, now, tz, start=None, schedule_id="S1", description=""):
    """Run for `hours` real hours, starting at `start` (an aware time; default: now)."""
    zone = _zone(tz)
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or hours != hours:
        raise ScheduleError("hours must be a number")
    if hours <= 0:
        raise ScheduleError("hours must be more than 0")
    begin = _aware(start if start is not None else now, "start").astimezone(zone)
    return RunSchedule(schedule_id, "duration", tz, begin, _plus_hours(begin, hours, zone),
                       description=description or f"Run for {hours:g} hours")


def between(start_clock, end_clock, *, now, tz, schedule_id="S1", description=""):
    """Run from `start_clock` to `end_clock` (datetime.time) in the lab's zone.

    Picks the first window that has not ended yet: one that began yesterday and is still going
    (a night window, asked for at 02:00), otherwise the one that starts today, otherwise tomorrow's.
    """
    zone = _zone(tz)
    if not isinstance(start_clock, time) or not isinstance(end_clock, time):
        raise ScheduleError("start and end must be clock times")
    if start_clock.replace(tzinfo=None) == end_clock.replace(tzinfo=None):
        raise ScheduleError("the start and end times are the same; for a full day use 'for 24 hours'")
    today = _aware(now, "now").astimezone(zone).date()
    for offset in (-1, 0, 1):
        first_day = today + timedelta(days=offset)
        begin = _local(first_day, start_clock, zone)
        last_day = first_day if end_clock.replace(tzinfo=None) > start_clock.replace(tzinfo=None) else first_day + timedelta(days=1)
        finish = _local(last_day, end_clock, zone)
        if _utc(finish) > _utc(now):
            return RunSchedule(schedule_id, "window", tz, begin, finish,
                               description=description or f"From {start_clock:%H:%M} to {end_clock:%H:%M}")
    raise ScheduleError("no window found")        # unreachable: tomorrow's window always ends after now


# ---- after the run: was it complete? ----

def completeness(schedule, first_frame_at, last_frame_at, *, tolerance_s=DEFAULT_TOLERANCE_S):
    """Compare the recorded span with the plan, for the end-of-run report.

    first_frame_at / last_frame_at: aware times of the first and last frame that was recorded
    (None when nothing was recorded). A gap in the middle is not seen here; the monitor already
    marks those frames as tracking not ok.
    """
    planned = schedule.planned_s
    if first_frame_at is None or last_frame_at is None:
        return {"outcome": "no_data", "planned_s": planned, "recorded_s": 0.0,
                "missing_at_start_s": planned, "missing_at_end_s": 0.0, "complete": False}
    start, end = _utc(schedule.start), _utc(schedule.end)
    first = max(_utc(_aware(first_frame_at, "first_frame_at")), start)
    last = min(_utc(_aware(last_frame_at, "last_frame_at")), end)
    if last < first:
        return {"outcome": "no_data", "planned_s": planned, "recorded_s": 0.0,
                "missing_at_start_s": planned, "missing_at_end_s": 0.0, "complete": False}
    late_start = (first - start).total_seconds()
    early_stop = (end - last).total_seconds()
    complete = late_start <= tolerance_s and early_stop <= tolerance_s
    return {"outcome": "complete" if complete else "partial", "planned_s": planned,
            "recorded_s": (last - first).total_seconds(),
            "missing_at_start_s": late_start, "missing_at_end_s": early_stop, "complete": complete}
