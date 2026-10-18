"""Append-only event journal and the INTERFACES.md §4 event-log export (new file).

Two files, two formats:

1. The *journal* (this module's own format) is what the monitor writes while it
   runs. It is append-only JSONL; each line is one lifecycle record:

     {"record": "event_fired",    "event_id": ..., <decision made at firing>}
     {"record": "event_closed",   "event_id": ..., <how the bout ended>}
     {"record": "event_answered", "event_id": ..., "researcher_answer": ...}

   A record is durable once append() returns: the line is written, flushed
   and fsync'ed. The monitor writes "event_fired" BEFORE it hands the alert to
   the caller, so an alert that was shown is always on disk.

2. The *§4 event log* is a projection of the journal: one object per event in
   exactly the INTERFACES.md §4 shape (export_events / project_events). The
   summary and the replay read this. The shared §4 format is not changed.

One reduction path: JournalState is the only code that turns records into
journal state. Loading a file, appending, projecting and exporting all go
through it, so they always agree. It checks each record's fields, that closed
and answered records refer to an earlier fired event, and that counts and
durations are consistent. Identical duplicate records are dropped; records
with the same (record, event_id) but different content raise
JournalConflictError everywhere.

Idempotency: appending a record identical to one already stored does nothing
(returns False). So retrying a failed write, or replaying the same run (same
run_id, same input), never duplicates lines.

Write failures:
- If a write or its fsync fails, the journal truncates the file back to its
  previous size, fsyncs, and checks the size. If that rollback is verified,
  EventPersistenceError is raised: nothing was stored and a retry is safe.
- If the rollback fails or cannot be verified (the file may now hold a partial
  line, or a complete line whose fsync failed), JournalRecoveryRequired is
  raised and the journal refuses every further append until recover() has
  re-read the file from disk, cut any unterminated tail, re-validated all
  records, and fsync'ed the file. A record that did reach the file is then
  part of the state, so the retry finds it and does not write it again.
- A record is never acknowledged (append never returns) while the file state
  is uncertain.
- A crash in the middle of a write can leave an unterminated last line; it was
  never acknowledged, so it is ignored when reading and cut off (verified) when
  the journal is opened for writing.

Single writer: one process writes a journal file at a time (no file locking).
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

EVENT_KEYS = ("event_id", "session", "rule_id", "behavior", "start_s", "end_s",
              "confidence", "action", "researcher_answer", "evidence")
RECORD_TYPES = ("event_fired", "event_closed", "event_answered")
ACTIONS = ("alert", "ask", "log")
ANSWERS = ("confirmed", "rejected")
END_REASONS = ("below_threshold", "tracking_lost", "missing_score", "frame_gap",
               "session_end", "stream_end")
SAMPLING_TOL = 1e-3  # allowed timestamp jitter, as a fraction of a frame period
_TOL = 1e-9

_FIRED_FIELDS = {
    "record": str, "event_id": str, "run_id": str, "session": str, "rule_id": str,
    "behavior": str, "start_s": "num", "first_frame": int, "fired_frame": int,
    "fired_s": "num", "frame_period_s": "num", "confidence": "num", "action": str,
    "decision_mean_score": "num", "decision_min_score": "num", "decision_frames": int,
    "rule": dict,
}
_RULE_FIELDS = {"min_duration_s": "num", "ask_threshold": "num",
                "alert_threshold": "num", "alert_mode": str}
_CLOSED_FIELDS = {
    "record": str, "event_id": str, "end_s": "num", "end_reason": str, "last_frame": int,
    "bout_frames": int, "bout_mean_score": "num", "bout_min_score": "num",
    "bout_max_score": "num",
}
_ANSWERED_FIELDS = {"record": str, "event_id": str, "researcher_answer": str}
_FIELDS = {"event_fired": _FIRED_FIELDS, "event_closed": _CLOSED_FIELDS,
           "event_answered": _ANSWERED_FIELDS}


class JournalIntegrityError(ValueError):
    """A journal record is malformed or inconsistent with the rest of the journal."""


class JournalConflictError(JournalIntegrityError):
    """A record with the same (record, event_id) but different content exists."""


class EventPersistenceError(RuntimeError):
    """Writing to the journal failed. The record was not stored."""


class JournalRecoveryRequired(EventPersistenceError):
    """A failed write could not be rolled back verifiably. The journal accepts
    no appends until recover() succeeds."""


def _clean(value):
    """Plain JSON types (numpy numbers -> Python numbers)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _dumps(obj):
    return json.dumps(_clean(obj), allow_nan=False)


def normalize_record(record):
    """The record exactly as it will look after a write + read (plain JSON)."""
    return json.loads(_dumps(record))


# ---- the one validated reduction path ----

def _check_fields(rec, spec, what):
    if set(rec) != set(spec):
        raise JournalIntegrityError(f"{what}: fields {sorted(rec)} != {sorted(spec)}")
    for key, kind in spec.items():
        v = rec[key]
        if kind == "num":
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        elif kind is int:
            ok = isinstance(v, int) and not isinstance(v, bool)
        else:
            ok = isinstance(v, kind)
        if not ok:
            raise JournalIntegrityError(f"{what}: field {key!r} has a bad value {v!r}")


def _validate_shape(rec):
    if not isinstance(rec, dict) or rec.get("record") not in RECORD_TYPES:
        raise JournalIntegrityError("not a journal record")
    what = f"{rec['record']} {rec.get('event_id')!r}"
    _check_fields(rec, _FIELDS[rec["record"]], what)
    if not rec["event_id"]:
        raise JournalIntegrityError(f"{what}: empty event_id")
    if rec["record"] == "event_fired":
        _check_fields(rec["rule"], _RULE_FIELDS, what + " rule")
        r = rec["rule"]
        problems = [
            (rec["action"] not in ACTIONS, "unknown action"),
            (not 0.0 <= rec["confidence"] <= 1.0, "confidence outside [0, 1]"),
            (rec["confidence"] != rec["decision_mean_score"], "confidence != decision_mean_score"),
            (rec["decision_min_score"] > rec["confidence"] + _TOL, "decision_min_score > mean"),
            (rec["frame_period_s"] <= 0, "frame_period_s <= 0"),
            (rec["fired_frame"] < rec["first_frame"], "fired before first frame"),
            (rec["decision_frames"] != rec["fired_frame"] - rec["first_frame"] + 1,
             "decision_frames does not match first/fired frame"),
            (rec["fired_s"] - rec["start_s"] < r["min_duration_s"] - _TOL,
             "fired before min_duration_s of observed time"),
        ]
    elif rec["record"] == "event_closed":
        problems = [
            (rec["end_reason"] not in END_REASONS, "unknown end_reason"),
            (not rec["bout_min_score"] - _TOL <= rec["bout_mean_score"] <= rec["bout_max_score"] + _TOL,
             "bout min/mean/max out of order"),
        ]
    else:
        problems = [(rec["researcher_answer"] not in ANSWERS, "unknown researcher_answer")]
    for bad, msg in problems:
        if bad:
            raise JournalIntegrityError(f"{what}: {msg}")


def _validate_against(rec, by_key):
    """Checks that need the event's fired record."""
    eid = rec["event_id"]
    fired = by_key.get(("event_fired", eid))
    if fired is None:
        raise JournalIntegrityError(f"{rec['record']} for {eid!r} has no earlier event_fired")
    if rec["record"] == "event_answered":
        if fired["action"] != "ask":
            raise JournalIntegrityError(
                f"event {eid} is {fired['action']!r}; only 'ask' events take answers")
        return
    period = fired["frame_period_s"]
    problems = [
        (rec["last_frame"] < fired["fired_frame"], "last_frame before fired_frame"),
        (rec["bout_frames"] != rec["last_frame"] - fired["first_frame"] + 1,
         "bout_frames does not match first/last frame"),
        (abs(rec["end_s"] - fired["start_s"] - rec["bout_frames"] * period)
         > SAMPLING_TOL * period + _TOL, "end_s - start_s does not match bout_frames"),
        (rec["bout_min_score"] > fired["decision_min_score"] + _TOL,
         "bout_min_score above the decision minimum"),
    ]
    for bad, msg in problems:
        if bad:
            raise JournalIntegrityError(f"event_closed {eid!r}: {msg}")


class JournalState:
    """Validated, deduplicated journal records (the only reduction path)."""

    def __init__(self):
        self._records = []   # in file order
        self._by_key = {}    # (record, event_id) -> record

    @classmethod
    def from_records(cls, records):
        state = cls()
        for rec in records:
            state.add(rec)
        return state

    def check(self, rec):
        """Validate rec against this state. True if new, False if an identical
        record is already there. Raises on invalid or conflicting records."""
        _validate_shape(rec)
        key = (rec["record"], rec["event_id"])
        existing = self._by_key.get(key)
        if existing is not None:
            if existing == rec:
                return False
            raise JournalConflictError(f"journal already has a different {key[0]} for {key[1]}")
        if rec["record"] != "event_fired":
            _validate_against(rec, self._by_key)
        return True

    def add(self, rec):
        rec = normalize_record(rec)
        if not self.check(rec):
            return False
        self._by_key[(rec["record"], rec["event_id"])] = rec
        self._records.append(rec)
        return True

    def has(self, record, event_id):
        return (record, event_id) in self._by_key

    def get(self, record, event_id):
        rec = self._by_key.get((record, event_id))
        return copy.deepcopy(rec)

    @property
    def records(self):
        return copy.deepcopy(self._records)

    def event(self, event_id):
        fired = self._by_key.get(("event_fired", event_id))
        if fired is None:
            raise KeyError(event_id)
        return project_event(fired, self._by_key.get(("event_closed", event_id)),
                             self._by_key.get(("event_answered", event_id)))

    def events(self, run_id=None):
        """§4 projection of every fired event (optionally one run), in firing order."""
        return [self.event(r["event_id"]) for r in self._records
                if r["record"] == "event_fired" and (run_id is None or r["run_id"] == run_id)]


# ---- files ----

def read_journal(path):
    """Raw complete records of a journal file, in order (no validation, no
    changes to the file). An unterminated last line (torn write) is ignored."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    complete = data[:data.rfind(b"\n") + 1]
    records = []
    for n, line in enumerate(complete.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JournalIntegrityError(f"{path}: line {n} is not valid JSON") from exc
        if not isinstance(rec, dict) or rec.get("record") not in RECORD_TYPES:
            raise JournalIntegrityError(f"{path}: line {n} is not a journal record")
        records.append(rec)
    return records


def load_journal(path):
    """Validated JournalState of a journal file (read only)."""
    return JournalState.from_records(read_journal(path))


class EventJournal:
    """Append-only lifecycle journal. path=None keeps records in memory only."""

    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self._needs_recovery = False
        if self.path is not None:
            self._repair_tail()
            self._state = load_journal(self.path)
        else:
            self._state = JournalState()

    @property
    def needs_recovery(self):
        return self._needs_recovery

    @property
    def records(self):
        """Deep copies: changing them never changes the journal."""
        return self._state.records

    def get(self, record, event_id):
        return self._state.get(record, event_id)

    def has(self, record, event_id):
        return self._state.has(record, event_id)

    def event(self, event_id):
        """§4 projection of one event, current state (KeyError if it never fired)."""
        return self._state.event(event_id)

    def events(self, run_id=None):
        """§4 projection of the current journal state (recovery / inspection)."""
        return self._state.events(run_id)

    def append(self, record):
        """Store one record durably. Returns True if written, False if an
        identical record was already there (idempotent retry)."""
        if self._needs_recovery:
            raise JournalRecoveryRequired(f"{self.path} needs recover() after a failed write")
        rec = normalize_record(record)
        if not self._state.check(rec):
            return False
        if self.path is not None:
            self._write_line(_dumps(rec) + "\n")
        self._state.add(rec)
        return True

    def record_answer(self, event_id, answer):
        """The researcher's answer to an "ask" event (appends event_answered).
        Answering again with the same answer is a no-op; a different one is a
        conflict (changing an answer is not supported on Day 1)."""
        if answer not in ANSWERS:
            raise ValueError(f"answer must be one of {ANSWERS}")
        if not self._state.has("event_fired", event_id):
            raise KeyError(f"no fired event {event_id!r} in the journal")
        return self.append(answer_record(event_id, answer))

    def recover(self):
        """Reconcile the in-memory state with the file after a failed write:
        cut an unterminated tail, re-read and re-validate every record, fsync.
        Only then are appends allowed again. Raises (and stays blocked) if any
        step fails or the file holds invalid / conflicting records."""
        if self.path is None:
            self._needs_recovery = False
            return
        self._needs_recovery = True
        try:
            self._repair_tail()
            state = load_journal(self.path)
            if self.path.exists():
                with open(self.path, "r+b") as f:
                    os.fsync(f.fileno())
        except OSError as exc:
            raise JournalRecoveryRequired(f"recovery of {self.path} failed: {exc}") from exc
        self._state = state
        self._needs_recovery = False

    # ---- internals ----

    def _repair_tail(self):
        """Cut an unterminated last line (never acknowledged) and verify it."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        keep = data.rfind(b"\n") + 1
        if keep == len(data):
            return
        with open(self.path, "r+b") as f:
            f.truncate(keep)
            f.flush()
            os.fsync(f.fileno())
        if self.path.stat().st_size != keep:
            raise OSError(f"could not cut the torn tail of {self.path}")

    def _write_line(self, text):
        data = text.encode("utf-8")
        try:
            size = self.path.stat().st_size if self.path.exists() else 0
        except OSError as exc:  # nothing written yet
            raise EventPersistenceError(f"could not write to {self.path}: {exc}") from exc
        try:
            with open(self.path, "ab") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except BaseException as exc:
            if not self._rollback(size):
                self._needs_recovery = True
                if isinstance(exc, OSError):
                    raise JournalRecoveryRequired(
                        f"write to {self.path} failed ({exc}) and could not be undone; "
                        "call recover() before writing again") from exc
                raise
            if isinstance(exc, OSError):
                raise EventPersistenceError(
                    f"could not write to {self.path}: {exc} (nothing stored; safe to retry)") from exc
            raise

    def _rollback(self, size):
        """Truncate back to `size`, fsync, and verify. True only if verified."""
        try:
            if not self.path.exists():
                return size == 0
            if self.path.stat().st_size != size:
                with open(self.path, "r+b") as f:
                    f.truncate(size)
                    f.flush()
                    os.fsync(f.fileno())
            return self.path.stat().st_size == size
        except OSError:
            return False


def answer_record(event_id, answer):
    return {"record": "event_answered", "event_id": event_id, "researcher_answer": answer}


def project_event(fired, closed=None, answered=None):
    """One §4 event from its lifecycle records.

    Top-level confidence and evidence mean_score / min_score / frames are the
    values the decision was made on at firing; they never change. Statistics of
    the whole completed bout are in separate bout_* fields (None while open).
    While the bout is open, end_s is the end of what was seen at firing.
    """
    end_s = closed["end_s"] if closed else fired["fired_s"] + fired["frame_period_s"]
    evidence = {
        "mean_score": fired["decision_mean_score"],
        "min_score": fired["decision_min_score"],
        "frames": fired["decision_frames"],
        "run_id": fired["run_id"],
        "first_frame": fired["first_frame"],
        "fired_frame": fired["fired_frame"],
        "fired_s": fired["fired_s"],
        "min_duration_s": fired["rule"]["min_duration_s"],
        "ask_threshold": fired["rule"]["ask_threshold"],
        "alert_threshold": fired["rule"]["alert_threshold"],
        "alert_mode": fired["rule"]["alert_mode"],
        "bout_closed": closed is not None,
        "end_reason": closed["end_reason"] if closed else None,
        "last_frame": closed["last_frame"] if closed else None,
        "bout_frames": closed["bout_frames"] if closed else None,
        "bout_mean_score": closed["bout_mean_score"] if closed else None,
        "bout_min_score": closed["bout_min_score"] if closed else None,
        "bout_max_score": closed["bout_max_score"] if closed else None,
    }
    return {"event_id": fired["event_id"], "session": fired["session"],
            "rule_id": fired["rule_id"], "behavior": fired["behavior"],
            "start_s": fired["start_s"], "end_s": end_s,
            "confidence": fired["confidence"], "action": fired["action"],
            "researcher_answer": answered["researcher_answer"] if answered else None,
            "evidence": evidence}


def project_events(records, run_id=None):
    """§4 events from raw records, through the same validation as everything else."""
    return JournalState.from_records(records).events(run_id)


# ---- §4 event-log files ----

def event_to_json(event):
    """One §4 event as a single JSON line, keys in §4 order."""
    return _dumps({k: event.get(k) for k in EVENT_KEYS})


def write_jsonl(path, events):
    """Write (overwrite) a §4 event-log file. Written to a temp file first and
    renamed, so readers never see a half-written log."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(event_to_json(e) + "\n" for e in events), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_events(journal_path, out_path, run_id=None):
    """Journal -> §4 event-log JSONL (validated state). Returns the events."""
    events = load_journal(journal_path).events(run_id)
    write_jsonl(out_path, events)
    return events
