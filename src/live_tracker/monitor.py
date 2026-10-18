"""Streaming monitor: approved rules + per-frame scores -> events (new file).

Plain deterministic Python, no LLM. Frames go in one at a time, in order, like
a live feed; a decision at frame k only uses frames up to k (causal).

Input rows follow INTERFACES.md §2 (session, frame, time_s, <behavior scores>,
tracking_ok). Events are written to an append-only journal (event_log.py) and
exported in the INTERFACES.md §4 format.

Time contract (Day 1: regularly sampled input only)
---------------------------------------------------
- time_s is the *observation time* of a frame: the moment it was sampled, in
  seconds since the start of the session (INTERFACES.md), so it is never
  negative. CalMS21 (Igor's loader) gives time_s = frame / 30.
- Sampling must be regular: within a session, every row must satisfy
  time_s == t0 + (frame - f0) / fps (to 0.1 % of a frame period), where f0, t0
  are the session's first accepted row. Rows that break this are rejected.
  Skipped frame numbers are allowed and break any bout (a gap is not evidence).
- A frame is *evidence* for a rule when tracking_ok is true, the rule's score
  is present, and score >= ask_threshold (">=", same as events.py).
- A *bout* is a run of evidence frames with consecutive frame numbers in one
  session. It ends at the first frame that is not evidence. Frames with
  tracking_ok = false END the bout (they never pause it), as do missing
  scores, skipped frame numbers and the end of a session.
- first qualifying frame: the bout's first evidence frame; start_s is its
  observation time.
- Elapsed duration at frame k = time_s(k) - start_s, both *observed* times as
  given in the rows (not frame-derived), so allowed timestamp jitter can only
  delay firing, never make it early.
- Firing: the event fires on the first evidence frame whose elapsed duration is
  >= min_duration_s, while the bout is still going on. fired_frame is that
  frame and fired_s its observation time. At 30 fps a bout starting at 0.0 s
  with min_duration_s = 2.0 fires on frame 60 (fired_s = 2.0), never earlier;
  if frame 60 is stamped slightly early (e.g. 1.99998 s) it fires on frame 61.
- Exclusive end: end_s = nominal time of the frame slot right after the bout's
  last evidence frame, t0 + (last_frame + 1 - f0) / fps (so bout frames / fps ==
  end_s - start_s up to jitter, as in the §4 example). It is not an observation
  (that frame may not exist), so it is the nominal slot time.

Rule semantics (provisional; meaning from rules.py and plan/SYED.md)
--------------------------------------------------------------------
- Confidence = mean score over the bout's evidence frames up to and including
  the firing frame. It is fixed at firing; later frames never change it or the
  action. Statistics of the full bout are reported separately (bout_*).
- alert_mode "alert": confidence >= alert_threshold -> "alert", else "ask".
- alert_mode "ask":   always "ask" (the researcher confirms every event).
                      NEEDS TEAM AGREEMENT: no doc defines this mode yet.
- alert_mode "log":   always "log" (recorded, does not interrupt the UI).
- One continuous bout gives at most one event; a new event needs a new bout.
- Only classifier behaviors (attack, mount, investigation) are monitored;
  rules for immobility / tracking_loss are refused, not silently skipped.

Atomicity and persistence
-------------------------
step() validates the whole row first, then works out every state change and
journal record without touching the monitor, then writes the records (fired
events before returning them), and only then commits the new state. If the row
is invalid (MonitorInputError) or a write fails (EventPersistenceError), the
monitor is unchanged and the same row can be sent again. After a write failure
the monitor only accepts a retry of that same row (or of finish()), so a failed
event can never be skipped silently. If the journal reports that a failed
write could not be undone (JournalRecoveryRequired), call recover_journal()
and then retry the same row.

Replay and current state
------------------------
step() returns each event exactly as it was at firing: built from the record
this monitor just computed, never from the journal, so it cannot contain a
later end_s, later scores or a researcher answer, even when the same run is
replayed into a journal that already holds its future. finish() returns the
events this monitor instance computed (fired + closed) plus answers given
through this instance's answer(). The journal's current state, including
answers recorded at any time, is read separately with monitor.journal.events()
or event_log.export_events() (recovery / inspection).

Replaying an already persisted run (same run_id, same input, same journal)
recomputes identical records; they are confirmed, not written again, so the
file is unchanged. If the earlier run stopped early (crash), the replay appends
only the records that are missing. Different input under the same run_id is a
JournalConflictError.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, replace

import numpy as np

from live_tracker.event_log import (SAMPLING_TOL, EventJournal, EventPersistenceError,
                                    JournalState, answer_record, normalize_record,
                                    project_event)
from live_tracker.rules import CLASSIFIER_BEHAVIORS, Rule, RuleSet, validate_rule

DEFAULT_FPS = 30.0  # proposal-reported CalMS21 rate; to be confirmed by H1a
_TOL = 1e-9         # float tolerance for score and duration comparisons
_SAMPLING_TOL = SAMPLING_TOL  # allowed timestamp jitter, fraction of a frame period
_RUN_ID = re.compile(r"^[A-Za-z0-9_]{1,64}$")

__all__ = ["Monitor", "MonitorInputError", "EventPersistenceError", "run_monitor",
           "normalize_rule", "DEFAULT_FPS"]


class MonitorInputError(ValueError):
    """A score row was rejected. The monitor did not change."""


# ---- rules ----

def normalize_rule(rule):
    """Re-check an approved Rule before monitoring with it. Returns a private
    copy. Uses rules.validate_rule plus checks it does not do yet (finite
    numbers, classifier behavior, unchanged rule_id)."""
    if not isinstance(rule, Rule):
        raise TypeError("rules must be Rule objects from rules.py")
    if rule.status != "approved":
        raise ValueError(f"rule {rule.rule_id} is not approved (status {rule.status!r})")
    data = {**rule.to_dict(), "status": "proposed"}
    checked, errors = validate_rule(data)
    for key in ("min_duration_s", "alert_threshold", "ask_threshold"):
        v = data[key]
        if not isinstance(v, bool) and isinstance(v, (int, float)) and not math.isfinite(v):
            errors.append(f"{key} must be finite")
    if errors:
        raise ValueError(f"rule {rule.rule_id!r} is invalid: {'; '.join(errors)}")
    if checked.rule_id != rule.rule_id:
        raise ValueError(f"rule_id {rule.rule_id!r} has surrounding whitespace")
    if checked.behavior not in CLASSIFIER_BEHAVIORS:
        raise ValueError(f"rule {rule.rule_id}: behavior {rule.behavior!r} has no classifier "
                         f"score; the monitor supports {sorted(CLASSIFIER_BEHAVIORS)}")
    return replace(checked, status="approved")


def _decide_action(rule, confidence):
    if rule.alert_mode == "log":
        return "log"
    if rule.alert_mode == "ask":
        return "ask"
    return "alert" if confidence >= rule.alert_threshold - _TOL else "ask"


# ---- row validation ----

_TRUE = {"true", "1"}
_FALSE = {"false", "0"}


def _is_real(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _parse_tracking(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in _TRUE | _FALSE:
        return v.strip().lower() in _TRUE
    raise MonitorInputError(f"tracking_ok must be a boolean (True/False, 1/0, 'true'/'false'), got {v!r}")


def _parse_score(name, v):
    """Missing (None / NaN) -> None, which breaks the bout. Anything else must
    be a real number in [0, 1]."""
    if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
        return None
    if not _is_real(v):
        raise MonitorInputError(f"score {name!r} must be a number in [0, 1] or missing, got {v!r}")
    v = float(v)
    if not 0.0 <= v <= 1.0:
        raise MonitorInputError(f"score {name!r} must be in [0, 1], got {v!r}")
    return v


@dataclass(frozen=True)
class _Row:
    session: str
    frame: int
    time_s: float
    tracking_ok: bool
    scores: tuple  # (behavior, score or None), sorted by behavior


@dataclass(frozen=True)
class _Bout:
    first_frame: int
    start_s: float
    last_frame: int
    last_s: float
    count: int
    total: float
    lo: float
    hi: float
    event_id: str | None = None  # set once the bout has fired

    def add(self, frame, t, score):
        return replace(self, last_frame=frame, last_s=t, count=self.count + 1,
                       total=self.total + score, lo=min(self.lo, score), hi=max(self.hi, score))


# ---- monitor ----

class Monitor:
    """Streams score rows through the approved rules.

    monitor = Monitor(ruleset, journal_path="events.journal.jsonl")
    for row in rows:                       # one frame at a time, in order
        for event in monitor.step(row):    # §4 events that fired on this frame,
            show(event)                    # already written to the journal
    events = monitor.finish()              # close open bouts; §4 events of this run

    run_id makes event IDs unique across runs ("E-<run_id>-<n>"); by default it
    is random. Replaying the same input with the same run_id into the same
    journal is idempotent (nothing is written twice).
    """

    def __init__(self, rules, fps=DEFAULT_FPS, journal_path=None, run_id=None):
        if isinstance(rules, RuleSet):
            rules = rules.active()
        checked = [normalize_rule(r) for r in rules]
        if len({r.rule_id for r in checked}) != len(checked):
            raise ValueError("duplicate rule_id")
        if not (_is_real(fps) and math.isfinite(fps) and fps > 0):
            raise ValueError("fps must be a positive finite number")
        run_id = uuid.uuid4().hex[:12] if run_id is None else run_id
        if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
            raise ValueError("run_id must be 1-64 letters, digits or underscores")
        self._rules = tuple(checked)
        self._behaviors = tuple(sorted({r.behavior for r in checked}))
        self.fps = float(fps)
        self.period = 1.0 / self.fps
        self.run_id = run_id
        self.journal = EventJournal(journal_path)
        # streaming state; only _commit() changes it
        self._open = {}          # rule_id -> _Bout
        self._session = None
        self._origin = None      # (first frame, first time_s) of the current session
        self._last_frame = None
        self._finished_sessions = frozenset()
        self._next_n = 1
        self._done = False
        self._pending = None     # row (or "finish") whose journal write failed
        self._run = JournalState()  # records this instance computed (for finish())

    @property
    def rules(self):
        """Copies of the monitored rules (changing them has no effect)."""
        return tuple(replace(r) for r in self._rules)

    # ---- public API ----

    def step(self, row):
        """Process one frame. Returns the §4 events that fired on this frame
        (as seen at firing). Raises MonitorInputError for a bad row and
        EventPersistenceError if the journal write fails; in both cases the
        monitor is unchanged and the row may be sent again."""
        if self._done:
            raise RuntimeError("monitor already finished")
        r = self._validate(row)
        if self._pending is not None and self._pending != r:
            raise RuntimeError("the previous frame's events could not be saved; "
                               "retry that frame (or fix storage) before sending new ones")
        state, records, fired_ids = self._plan(r)
        records = [normalize_record(rec) for rec in records]
        self._persist(records, pending=r)
        self._commit(state, records)
        # causal snapshots: only the fired record computed now, never journal state
        return [project_event(rec) for rec in records if rec["record"] == "event_fired"]

    def finish(self):
        """End of the stream: close open bouts and return this run's §4 events.
        Safe to call again (returns the same events)."""
        if not self._done:
            if self._pending is not None and self._pending != "finish":
                raise RuntimeError("the previous frame's events could not be saved; retry it first")
            records = self._close_records(self._open, "stream_end")
            records = [normalize_record(rec) for rec in records]
            self._persist(records, pending="finish")
            self._commit({"open": {}}, records)
            self._done = True
        return self._run.events()

    def answer(self, event_id, answer):
        """Researcher's answer ("confirmed" / "rejected") to an "ask" event."""
        if not self._run.has("event_fired", event_id):
            raise KeyError(f"{event_id!r} is not an event of this monitor run")
        written = self.journal.record_answer(event_id, answer)
        self._run.add(answer_record(event_id, answer))
        return written

    def recover_journal(self):
        """After JournalRecoveryRequired: reconcile the journal with the file.
        Then retry the failed row (or finish())."""
        self.journal.recover()

    # ---- internals ----

    def _validate(self, row):
        """Check and normalize a whole row without changing anything."""
        if hasattr(row, "to_dict") and not isinstance(row, dict):
            row = row.to_dict()
        if not isinstance(row, dict):
            raise MonitorInputError("row must be a dict (or a pandas Series)")
        need = ("session", "frame", "time_s", "tracking_ok") + self._behaviors
        missing = [c for c in need if c not in row]
        if missing:
            raise MonitorInputError(f"row is missing columns {missing}")

        session = row["session"]
        if not isinstance(session, str) or not session.strip():
            raise MonitorInputError(f"session must be a non-empty string, got {session!r}")
        frame = row["frame"]
        if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)) or frame < 0:
            raise MonitorInputError(f"frame must be a non-negative integer, got {frame!r}")
        frame = int(frame)
        t = row["time_s"]
        if not _is_real(t) or not math.isfinite(t) or t < 0:
            raise MonitorInputError(f"time_s must be a finite number >= 0 "
                                    f"(seconds since session start), got {t!r}")
        t = float(t)
        ok = _parse_tracking(row["tracking_ok"])
        scores = tuple((b, _parse_score(b, row[b])) for b in self._behaviors)

        where = f"session {session}, frame {frame}"
        if session == self._session:
            if frame <= self._last_frame:
                raise MonitorInputError(f"{where}: frame not after previous frame {self._last_frame}")
            f0, t0 = self._origin
            expected = t0 + (frame - f0) * self.period
            if abs(t - expected) > _SAMPLING_TOL * self.period:
                raise MonitorInputError(
                    f"{where}: time_s {t} does not match regular sampling at {self.fps} fps "
                    f"(expected {expected}); irregular timestamps are not supported")
        elif session in self._finished_sessions:
            raise MonitorInputError(f"session {session} appeared again after it ended; "
                                    "send each session's frames together")
        return _Row(session, frame, t, ok, scores)

    def _plan(self, r):
        """New state + journal records for row r. Pure: changes nothing."""
        records = []
        open_ = dict(self._open)
        finished = self._finished_sessions
        origin = self._origin
        if r.session != self._session:
            records += self._close_records(open_, "session_end")
            open_ = {}
            if self._session is not None:
                finished = finished | {self._session}
            origin = (r.frame, r.time_s)
        elif r.frame != self._last_frame + 1:
            records += self._close_records(open_, "frame_gap")
            open_ = {}

        scores = dict(r.scores)
        next_n = self._next_n
        fired_ids = []
        for rule in self._rules:
            score = scores[rule.behavior]
            if not r.tracking_ok:
                reason = "tracking_lost"
            elif score is None:
                reason = "missing_score"
            elif score < rule.ask_threshold - _TOL:
                reason = "below_threshold"
            else:
                reason = None
            if reason is not None:
                bout = open_.pop(rule.rule_id, None)
                if bout is not None and bout.event_id is not None:
                    records.append(self._closed_record(bout, reason))
                continue
            bout = open_.get(rule.rule_id)
            if bout is None:
                bout = _Bout(r.frame, r.time_s, r.frame, r.time_s, 1, score, score, score)
            else:
                bout = bout.add(r.frame, r.time_s, score)
            # elapsed *observed* time: jitter can delay firing but never advance it
            elapsed = r.time_s - bout.start_s
            if bout.event_id is None and elapsed >= rule.min_duration_s - _TOL:
                event_id = f"E-{self.run_id}-{next_n}"
                next_n += 1
                bout = replace(bout, event_id=event_id)
                records.append(self._fired_record(rule, bout, r))
                fired_ids.append(event_id)
            open_[rule.rule_id] = bout

        state = {"open": open_, "session": r.session, "origin": origin,
                 "last_frame": r.frame, "finished": finished, "next_n": next_n}
        return state, records, fired_ids

    def _fired_record(self, rule, bout, r):
        confidence = bout.total / bout.count
        return {
            "record": "event_fired", "event_id": bout.event_id, "run_id": self.run_id,
            "session": r.session, "rule_id": rule.rule_id, "behavior": rule.behavior,
            "start_s": bout.start_s, "first_frame": bout.first_frame,
            "fired_frame": r.frame, "fired_s": r.time_s, "frame_period_s": self.period,
            "confidence": confidence, "action": _decide_action(rule, confidence),
            "decision_mean_score": confidence, "decision_min_score": bout.lo,
            "decision_frames": bout.count,
            "rule": {"min_duration_s": rule.min_duration_s, "ask_threshold": rule.ask_threshold,
                     "alert_threshold": rule.alert_threshold, "alert_mode": rule.alert_mode},
        }

    def _closed_record(self, bout, reason):
        return {
            "record": "event_closed", "event_id": bout.event_id,
            "end_s": self._slot_time(bout.last_frame + 1), "end_reason": reason,
            "last_frame": bout.last_frame, "bout_frames": bout.count,
            "bout_mean_score": bout.total / bout.count,
            "bout_min_score": bout.lo, "bout_max_score": bout.hi,
        }

    def _slot_time(self, frame):
        """Observation time of a frame slot in the current session (regular sampling)."""
        f0, t0 = self._origin
        return t0 + (frame - f0) * self.period

    def _close_records(self, open_, reason):
        return [self._closed_record(b, reason) for b in open_.values() if b.event_id is not None]

    def _persist(self, records, pending):
        try:
            for rec in records:
                self.journal.append(rec)  # idempotent: already-written records are skipped
        except EventPersistenceError:
            self._pending = pending
            raise
        self._pending = None

    def _commit(self, state, records):
        for rec in records:  # already accepted by the journal, so valid here too
            self._run.add(rec)
        self._open = state["open"]
        if "session" in state:
            self._session = state["session"]
            self._origin = state["origin"]
            self._last_frame = state["last_frame"]
            self._finished_sessions = state["finished"]
            self._next_n = state["next_n"]


def run_monitor(scores, rules, fps=DEFAULT_FPS, journal_path=None, run_id=None):
    """Feed a whole score table (DataFrame or iterable of dicts) through a
    Monitor in the given row order and return this run's §4 events. Rows are
    not sorted here: out-of-order input is an error, as in a live feed."""
    rows = scores.to_dict("records") if hasattr(scores, "to_dict") else scores
    monitor = Monitor(rules, fps=fps, journal_path=journal_path, run_id=run_id)
    for row in rows:
        monitor.step(row)
    return monitor.finish()
