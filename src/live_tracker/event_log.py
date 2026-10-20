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

Run provenance (Stage 9, S9-2)
------------------------------
A run may start with one header record, written by the monitor before any of
the run's events when it is given verified score provenance:

     {"record": "run_started", "run_id": ..., "header_version": "run-header/1",
      "frame_period_s": ..., "rules": [<approved rules, sorted by rule_id>],
      "rules_digest": <sha256 of the canonical rules list>,
      "source": <score_drops.ScoreProvenance.to_json()>}

It is validated like every other record: exact fields, a rules digest that
recomputes, approved rules only, source digests as SHA-256 hex, the
validation split, the model label marked unverified and every model-artifact
field null (nothing about the model is claimed). It must precede the run's
events (a header for a run that already has events is a conflict), and every
event_fired of that run must use one of its rules, unchanged, and its frame
period. Journals written before S9-2 have no headers: they stay readable and
their runs are reported as "no_source_provenance" (run_provenance()).
The §4 export is unchanged (headers are not events).

Source binding: a header written by the public run_header() (or by a monitor
given a dict, a ScoreProvenance or a hand-built ScoreDrop) has
source_binding "declared". Only a Monitor handed a ScoreDrop issued by
score_drops.load_validation_scores writes "verified_load", and even then the
run counts as "verified_score_drop" only once its completion record shows that
the rows the monitor processed hash to the verified rows digest.

Completion: Monitor.finish() of a run with a header appends
     {"record": "run_finished", "run_id", "n_rows", "n_events",
      "events_sha256", "input_rows_sha256"}
after the run's last closed event. It is refused unless every event is
closed and n_events / events_sha256 equal the run's event_fired /
event_closed records; no event of the run may follow it.

Frozen evidence (freeze_run): the canonical JSON of one run's header (or
null), its event_fired / event_closed records in journal order and its
run_finished record (or null), with its SHA-256. Researcher answers and other
runs are not part of it. A run with a header but no run_finished is never
frozen; a run without a header (journals written before S9-2) is frozen with
completion "legacy_unrecorded", never "complete". load_snapshot rebuilds the
journal state from the snapshot, so deleting event pairs (or otherwise
editing records) without a matching completion record is refused even when
the snapshot digest is recomputed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

EVENT_KEYS = ("event_id", "session", "rule_id", "behavior", "start_s", "end_s",
              "confidence", "action", "researcher_answer", "evidence")
RECORD_TYPES = ("event_fired", "event_closed", "event_answered", "run_started", "run_finished")
EVENT_RECORDS = ("event_fired", "event_closed", "event_answered")
HEADER_VERSION = "run-header/1"
SNAPSHOT_VERSION = "journal-snapshot/1"
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
_HEADER_FIELDS = {"record": str, "run_id": str, "header_version": str, "frame_period_s": "num", "rules": list,
                  "rules_digest": str, "source_binding": str, "source": dict}
# "verified_load": the monitor was handed a ScoreDrop issued by score_drops.load_validation_scores
# (score_drops.trusted_rows_sha256); "declared": anything else (a dict, a ScoreProvenance, a hand-built drop)
SOURCE_BINDINGS = ("verified_load", "declared")
_FINISHED_FIELDS = {"record": str, "run_id": str, "n_rows": int, "n_events": int, "events_sha256": str,
                    "input_rows_sha256": str}
_HEADER_RULE_FIELDS = {"rule_id": str, "behavior": str, "status": str, **_RULE_FIELDS}
# score_drops.ScoreProvenance fields (S9-1a); kinds: "hex" SHA-256, "null" never claimed, int, str
_SOURCE_FIELDS = {"archive_name": str, "archive_sha256": "hex", "archive_sha256_check": str, "member": str,
                  "member_sha256": "hex", "member_sha256_check": str, "split": str, "manifest_sha256": "hex",
                  "n_sessions": int, "n_frames": int, "rows_sha256": "hex", "model_label": str,
                  "model_label_status": str,
                  "model_artifact_sha256": "null", "model_id": "null", "feature_configuration": "null",
                  "producing_commit": "null", "model_provenance_status": str}
_HEX64 = re.compile(r"[0-9a-f]{64}")


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


def canonical_bytes(obj):
    """Canonical serialization for digests: sorted keys, compact, UTF-8, no NaN."""
    return json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def rules_digest(rules):
    """SHA-256 of the canonical list of header rules (sorted by rule_id)."""
    return hashlib.sha256(canonical_bytes(sorted(rules, key=lambda r: r["rule_id"]))).hexdigest()


def run_header(run_id, rules, frame_period_s, source):
    """The run_started record for a monitor run whose source is only DECLARED
    (rules: approved rules.Rule objects; source: a ScoreProvenance or its
    to_json() dict). This public helper never produces a verified binding:
    only Monitor, handed a ScoreDrop that score_drops issued, writes
    source_binding "verified_load". The source is copied, never completed."""
    return _run_header(run_id, rules, frame_period_s, source, "declared")


def _run_header(run_id, rules, frame_period_s, source, binding):
    items = sorted(({"rule_id": r.rule_id, "behavior": r.behavior, "status": r.status,
                     "min_duration_s": r.min_duration_s, "ask_threshold": r.ask_threshold,
                     "alert_threshold": r.alert_threshold, "alert_mode": r.alert_mode} for r in rules),
                   key=lambda r: r["rule_id"])
    src = source.to_json() if hasattr(source, "to_json") else source
    header = normalize_record({"record": "run_started", "run_id": run_id, "header_version": HEADER_VERSION,
                               "frame_period_s": frame_period_s, "rules": items, "rules_digest": rules_digest(items),
                               "source_binding": binding, "source": copy.deepcopy(src)})
    _validate_header(header)
    return header


def _validate_header(rec):
    what = f"run_started {rec.get('run_id')!r}"
    _check_fields(rec, _HEADER_FIELDS, what)
    if not rec["run_id"]:
        raise JournalIntegrityError(f"{what}: empty run_id")
    if rec["header_version"] != HEADER_VERSION:
        raise JournalIntegrityError(f"{what}: unknown header_version {rec['header_version']!r}")
    if rec["frame_period_s"] <= 0:
        raise JournalIntegrityError(f"{what}: frame_period_s <= 0")
    if rec["source_binding"] not in SOURCE_BINDINGS:
        raise JournalIntegrityError(f"{what}: unknown source_binding {rec['source_binding']!r}")
    if not rec["rules"]:
        raise JournalIntegrityError(f"{what}: no rules")
    for r in rec["rules"]:
        if not isinstance(r, dict):
            raise JournalIntegrityError(f"{what}: a rule is not an object")
        _check_fields(r, _HEADER_RULE_FIELDS, f"{what} rule")
        if r["status"] != "approved":
            raise JournalIntegrityError(f"{what}: rule {r['rule_id']!r} is not approved")
    ids = [r["rule_id"] for r in rec["rules"]]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise JournalIntegrityError(f"{what}: rules must be unique and sorted by rule_id")
    if rec["rules_digest"] != rules_digest(rec["rules"]):
        raise JournalIntegrityError(f"{what}: rules_digest does not match the rules")
    src = rec["source"]
    if set(src) != set(_SOURCE_FIELDS):
        raise JournalIntegrityError(f"{what}: source fields {sorted(src)} != {sorted(_SOURCE_FIELDS)}")
    for key, kind in _SOURCE_FIELDS.items():
        v = src[key]
        ok = (v is None if kind == "null" else isinstance(v, str) and bool(_HEX64.fullmatch(v)) if kind == "hex"
              else isinstance(v, int) and not isinstance(v, bool) and v >= 0 if kind is int
              else isinstance(v, str) and bool(v))
        if not ok:
            raise JournalIntegrityError(f"{what}: source field {key!r} has a bad value {str(v)[:60]!r}"
                                        + (" (model identity is never claimed without verified provenance)"
                                           if kind == "null" else ""))
    if src["split"] != "validation":
        raise JournalIntegrityError(f"{what}: source split {src['split']!r} is not validation")
    if not src["model_label_status"].startswith("unverified"):
        raise JournalIntegrityError(f"{what}: the model label must be marked unverified")


def _record_key(rec):
    if rec["record"] in ("run_started", "run_finished"):
        return (rec["record"], rec["run_id"])
    return (rec["record"], rec["event_id"])


def events_digest(records):
    """SHA-256 of the canonical list of a run's event_fired / event_closed records, in order."""
    return hashlib.sha256(canonical_bytes(list(records))).hexdigest()


def _validate_finished(rec):
    what = f"run_finished {rec.get('run_id')!r}"
    _check_fields(rec, _FINISHED_FIELDS, what)
    if rec["n_rows"] < 0 or rec["n_events"] < 0:
        raise JournalIntegrityError(f"{what}: negative count")
    for key in ("events_sha256", "input_rows_sha256"):
        if not _HEX64.fullmatch(rec[key]):
            raise JournalIntegrityError(f"{what}: {key} is not a SHA-256 hex digest")


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
    if rec["record"] == "run_started":
        _validate_header(rec)
        return
    if rec["record"] == "run_finished":
        _validate_finished(rec)
        return
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
    if rec["record"] == "event_fired":
        header = by_key.get(("run_started", rec["run_id"]))
        if header is not None:                   # a run with a header: its events use its rules unchanged
            rule = next((r for r in header["rules"] if r["rule_id"] == rec["rule_id"]), None)
            if rule is None:
                raise JournalIntegrityError(f"event_fired {rec['event_id']!r}: rule {rec['rule_id']!r} is not in "
                                            f"the header of run {rec['run_id']!r}")
            if rule["behavior"] != rec["behavior"] or {k: rule[k] for k in _RULE_FIELDS} != rec["rule"] or \
                    rec["frame_period_s"] != header["frame_period_s"]:
                raise JournalIntegrityError(f"event_fired {rec['event_id']!r}: rule or frame period differs from "
                                            f"the header of run {rec['run_id']!r}")
        return
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
        key = _record_key(rec)
        existing = self._by_key.get(key)
        if existing is not None:
            if existing == rec:
                return False
            raise JournalConflictError(f"journal already has a different {key[0]} for {key[1]}")
        if rec["record"] == "run_started":
            if any(r["record"] == "event_fired" and r["run_id"] == rec["run_id"] for r in self._records):
                raise JournalConflictError(f"run {rec['run_id']!r} already has events; its header must come first")
            return True
        if rec["record"] == "run_finished":
            self._check_completion(rec)
            return True
        if rec["record"] in ("event_fired", "event_closed"):
            run = rec["run_id"] if rec["record"] == "event_fired" else \
                (self._by_key.get(("event_fired", rec["event_id"])) or {}).get("run_id")
            if run is not None and ("run_finished", run) in self._by_key:
                raise JournalIntegrityError(f"{rec['record']} {rec['event_id']!r} after run {run!r} finished")
        _validate_against(rec, self._by_key)
        return True

    def _run_event_records(self, run_id):
        ids = {r["event_id"] for r in self._records if r["record"] == "event_fired" and r["run_id"] == run_id}
        return [r for r in self._records if r["record"] in ("event_fired", "event_closed") and r["event_id"] in ids]

    def _check_completion(self, rec):
        """A completion record binds the run's finished lifecycle: it needs the
        run's header, every event closed, and the exact count and digest of the
        run's event_fired / event_closed records."""
        run = rec["run_id"]
        if ("run_started", run) not in self._by_key:
            raise JournalIntegrityError(f"run_finished {run!r} has no run_started header")
        records = self._run_event_records(run)
        fired = [r for r in records if r["record"] == "event_fired"]
        closed = {r["event_id"] for r in records if r["record"] == "event_closed"}
        if any(r["event_id"] not in closed for r in fired):
            raise JournalIntegrityError(f"run_finished {run!r} while events are still open")
        if rec["n_events"] != len(fired) or rec["events_sha256"] != events_digest(records):
            raise JournalIntegrityError(f"run_finished {run!r}: n_events / events_sha256 do not match the run's "
                                        "lifecycle records")

    def add(self, rec):
        rec = normalize_record(rec)
        if not self.check(rec):
            return False
        self._by_key[_record_key(rec)] = rec
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

    def run_ids(self):
        """Run ids with a header or an event, in first-appearance order."""
        out = []
        for r in self._records:
            if r["record"] in ("run_started", "event_fired", "run_finished") and r["run_id"] not in out:
                out.append(r["run_id"])
        return out

    def run_provenance(self, run_id):
        """{"status", "completion", "header", "finished"} of a run.

        status: "verified_score_drop" only when the header's source_binding is
        "verified_load" AND the run finished AND the rows the monitor actually
        processed (run_finished.input_rows_sha256 / n_rows) are exactly the
        verified rows (source.rows_sha256 / n_frames); "declared_unverified" for
        any other header; "no_source_provenance" without a header.
        completion: "complete" (a valid run_finished), "unfinished" (a header,
        no run_finished) or "legacy_unrecorded" (no header: completion was never
        recorded, so it is never presented as verified)."""
        if run_id not in self.run_ids():
            raise KeyError(f"no run {run_id!r} in the journal")
        header = self._by_key.get(("run_started", run_id))
        finished = self._by_key.get(("run_finished", run_id))
        if header is None:
            status, completion = "no_source_provenance", "legacy_unrecorded"
        else:
            completion = "complete" if finished else "unfinished"
            bound = (finished is not None and header["source_binding"] == "verified_load"
                     and finished["input_rows_sha256"] == header["source"]["rows_sha256"]
                     and finished["n_rows"] == header["source"]["n_frames"])
            status = "verified_score_drop" if bound else "declared_unverified"
        return {"status": status, "completion": completion, "header": copy.deepcopy(header),
                "finished": copy.deepcopy(finished)}


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

    def run_ids(self):
        return self._state.run_ids()

    def run_provenance(self, run_id):
        return self._state.run_provenance(run_id)

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


# ---- frozen run evidence (S9-2) ----

@dataclass(frozen=True)
class RunSnapshot:
    """One run's frozen journal evidence: canonical bytes and their SHA-256."""
    run_id: str
    provenance_status: str               # "verified_score_drop" | "declared_unverified" | "no_source_provenance"
    completion: str                      # "complete" | "legacy_unrecorded" (unfinished runs are never frozen)
    n_events: int
    sha256: str
    canonical: bytes

    def payload(self):
        """A fresh copy of the frozen content."""
        return json.loads(self.canonical)


def _state_of(source):
    if isinstance(source, JournalState):
        return source
    if isinstance(source, EventJournal):
        return source._state
    return load_journal(source)


def freeze_run(source, run_id):
    """The frozen evidence of one run (RunSnapshot) from a journal (path,
    EventJournal or JournalState). Refused if the run is unknown or any of its
    events is still open. Nothing missing is filled in: a run without a header
    is frozen with header null and status "no_source_provenance"."""
    state = _state_of(source)
    prov = state.run_provenance(run_id)
    if prov["completion"] == "unfinished":
        raise JournalIntegrityError(f"run {run_id!r} has no run_finished record: an unfinished run is never frozen")
    records = state._run_event_records(run_id)
    fired = [r for r in records if r["record"] == "event_fired"]
    open_ids = sorted({r["event_id"] for r in fired} - {r["event_id"] for r in records if r["record"] == "event_closed"})
    if open_ids:
        raise JournalIntegrityError(f"run {run_id!r} has open events {open_ids[:5]}; only finished runs are frozen")
    finished = prov["finished"]
    if finished is not None and (finished["n_events"] != len(fired) or finished["events_sha256"] != events_digest(records)):
        raise JournalIntegrityError(f"run {run_id!r}: its lifecycle records do not match its run_finished record")
    payload = {"snapshot_version": SNAPSHOT_VERSION, "run_id": run_id, "provenance_status": prov["status"],
               "completion": prov["completion"], "header": prov["header"], "records": copy.deepcopy(records),
               "finished": finished,
               "excluded": "researcher answers and other runs are not part of the frozen evidence"}
    canonical = canonical_bytes(payload)
    return RunSnapshot(run_id=run_id, provenance_status=prov["status"], completion=prov["completion"],
                       n_events=len(fired), sha256=hashlib.sha256(canonical).hexdigest(), canonical=canonical)


def load_snapshot(canonical, sha256):
    """A RunSnapshot from stored bytes: the digest must match and the content
    must be a valid, complete, duplicate-free frozen run (re-validated through
    JournalState)."""
    if not isinstance(canonical, bytes) or hashlib.sha256(canonical).hexdigest() != sha256:
        raise JournalIntegrityError("frozen snapshot bytes do not match their SHA-256")
    try:
        payload = json.loads(canonical)
    except ValueError:
        raise JournalIntegrityError("frozen snapshot is not JSON") from None
    if not isinstance(payload, dict) or set(payload) != {"snapshot_version", "run_id", "provenance_status",
                                                         "completion", "header", "records", "finished",
                                                         "excluded"} or \
            payload["snapshot_version"] != SNAPSHOT_VERSION or canonical_bytes(payload) != canonical:
        raise JournalIntegrityError("frozen snapshot is not in canonical journal-snapshot/1 form")
    records = payload["records"]
    if not isinstance(records, list) or any(not isinstance(r, dict) or r.get("record") not in
                                            ("event_fired", "event_closed") for r in records):
        raise JournalIntegrityError("frozen snapshot records must be event_fired / event_closed records")
    keys = [_record_key(r) for r in records]
    if len(set(keys)) != len(keys):
        raise JournalIntegrityError("frozen snapshot has duplicate records")
    if any(r["record"] == "event_fired" and r.get("run_id") != payload["run_id"] for r in records):
        raise JournalIntegrityError("frozen snapshot holds an event of another run")
    if payload["header"] is None and payload["finished"] is not None:
        raise JournalIntegrityError("frozen snapshot has a completion record without a header")
    # header, records, completion: the completion record re-checks the count and digest of the records
    state = JournalState.from_records(([payload["header"]] if payload["header"] is not None else []) + records
                                      + ([payload["finished"]] if payload["finished"] is not None else []))
    again = freeze_run(state, payload["run_id"]) if records or payload["header"] else None
    if again is None or again.sha256 != sha256:
        raise JournalIntegrityError("frozen snapshot is not consistent with its own records")
    return again


def verify_snapshot(snapshot, source):
    """Raise JournalIntegrityError unless the journal still holds exactly the
    frozen evidence of the snapshot's run (later answers and other runs are
    allowed; any change to the run's header or lifecycle records is not)."""
    current = freeze_run(source, snapshot.run_id)
    if current.sha256 != snapshot.sha256:
        raise JournalIntegrityError(f"the frozen evidence of run {snapshot.run_id!r} changed in the journal")
    return current


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
