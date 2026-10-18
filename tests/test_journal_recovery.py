"""Regression tests for the second Codex audit (new file).

Journal failure + rollback failure + recovery, one reduction path for
loading / projection / export, causal same-run replay, and deep-copied records.
"""

import builtins
import copy
import json
import os
from pathlib import Path

import pytest

import live_tracker.event_log as event_log
from live_tracker.event_log import (EventJournal, EventPersistenceError, JournalConflictError,
                                    JournalIntegrityError, JournalRecoveryRequired,
                                    export_events, load_journal, project_events, read_journal)
from live_tracker.monitor import Monitor, run_monitor
from live_tracker.rules import RuleSet

FPS = 30.0
REAL_OPEN = builtins.open
REAL_FSYNC = os.fsync


def rules(duration=2, both=False):
    rs = RuleSet()
    for i, behavior in enumerate(("attack", "mount") if both else ("attack",), 1):
        rs.add(dict(rule_id=f"R{i}", behavior=behavior, min_duration_s=duration))
        rs.approve(f"R{i}")
    return rs


def row(i, score=0.6, session="SYNTHETIC_review"):
    return dict(session=session, frame=i, time_s=i / FPS, attack=score, mount=score,
                investigation=0.0, tracking_ok=True)


def state(m):
    return copy.deepcopy((m._open, m._session, m._origin, m._last_frame,
                          m._finished_sessions, m._next_n, m._done))


def ready_to_fire(path, run_id="review", **kw):
    """Monitor that will fire on row(60)."""
    m = Monitor(rules(**kw), journal_path=path, run_id=run_id)
    for i in range(60):
        assert m.step(row(i)) == []
    return m


def fired_monitor(path):
    m = Monitor(rules(), journal_path=path, run_id="review")
    for i in range(61):
        fired = m.step(row(i))
    return m, fired[0]


class _PartialWrite:
    """File wrapper whose write() stores only the first `n` bytes, then fails."""

    def __init__(self, f, n):
        self.f, self.n = f, n

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.f.close()

    def write(self, data):
        self.f.write(data[:self.n])
        self.f.flush()
        raise OSError("simulated partial write")


def faulty_open(path, partial=None, rollback="fail"):
    """open() replacement for event_log: optional partial append; rollback
    (the "r+b" open) can "fail" or "noop" (truncate silently does nothing)."""
    def _open(p, mode="r", *a, **kw):
        if Path(p) == path and mode == "r+b":
            if rollback == "fail":
                raise OSError("simulated failed rollback")
            if rollback == "noop":
                f = REAL_OPEN(p, mode, *a, **kw)
                f.truncate = lambda size=None: size  # pretends to work
                return f
        f = REAL_OPEN(p, mode, *a, **kw)
        if partial is not None and Path(p) == path and mode == "ab":
            return _PartialWrite(f, partial)
        return f
    return _open


def fail_fsync(fd):
    raise OSError("simulated fsync failure")


# ---- Fix 1: failed rollback blocks the journal until recover() ----

def test_partial_write_and_failed_rollback_requires_recovery(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = ready_to_fire(path)
    before = state(m)
    with monkeypatch.context() as mp:
        mp.setattr(event_log, "open", faulty_open(path, partial=25), raising=False)
        with pytest.raises(JournalRecoveryRequired):
            m.step(row(60))
    assert state(m) == before
    assert m.journal.needs_recovery
    assert path.read_bytes()[-1:] != b"\n"  # the torn line is still on disk
    # storage is healthy again, but the journal stays blocked until recover()
    with pytest.raises(JournalRecoveryRequired):
        m.step(row(60))
    with pytest.raises(JournalRecoveryRequired):
        m.journal.append({"record": "event_answered", "event_id": "E-x-1",
                          "researcher_answer": "confirmed"})
    assert state(m) == before
    m.recover_journal()
    assert not m.journal.needs_recovery and path.read_bytes() == b""  # torn tail cut
    (event,) = m.step(row(60))
    assert event["evidence"]["fired_frame"] == 60
    assert [r["record"] for r in read_journal(path)] == ["event_fired"]
    assert len(load_journal(path).events()) == 1


def test_complete_write_fsync_failure_and_failed_rollback_no_duplicate(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = ready_to_fire(path)
    with monkeypatch.context() as mp:
        mp.setattr(event_log, "open", faulty_open(path, rollback="fail"), raising=False)
        mp.setattr(event_log.os, "fsync", fail_fsync)
        with pytest.raises(JournalRecoveryRequired):
            m.step(row(60))
    # the whole line reached the file, but was never acknowledged
    assert len(read_journal(path)) == 1
    with pytest.raises(JournalRecoveryRequired):
        m.step(row(60))
    m.recover_journal()  # adopts the complete record (and fsyncs it)
    (event,) = m.step(row(60))  # the retry finds it: not written again
    assert event["event_id"] == "E-review-1"
    assert len(read_journal(path)) == 1
    for i in range(61, 70):
        m.step(row(i))
    m.finish()
    assert [r["record"] for r in read_journal(path)] == ["event_fired", "event_closed"]


def test_rollback_that_silently_does_nothing_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = ready_to_fire(path)
    with monkeypatch.context() as mp:
        mp.setattr(event_log, "open", faulty_open(path, partial=10, rollback="noop"), raising=False)
        with pytest.raises(JournalRecoveryRequired):
            m.step(row(60))
    assert m.journal.needs_recovery


def test_rollback_fsync_failure_requires_recovery(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = ready_to_fire(path)
    with monkeypatch.context() as mp:
        mp.setattr(event_log.os, "fsync", fail_fsync)  # the write's and the rollback's fsync
        with pytest.raises(JournalRecoveryRequired):
            m.step(row(60))
    m.recover_journal()
    assert len(m.step(row(60))) == 1
    assert len(read_journal(path)) == 1


def test_verified_rollback_does_not_require_recovery(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = ready_to_fire(path)
    calls = {"n": 0}

    def fail_first(fd):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("first fsync fails")
        return REAL_FSYNC(fd)
    with monkeypatch.context() as mp:
        mp.setattr(event_log.os, "fsync", fail_first)
        with pytest.raises(EventPersistenceError) as exc:
            m.step(row(60))
    assert not isinstance(exc.value, JournalRecoveryRequired)
    assert not m.journal.needs_recovery and path.read_bytes() == b""
    assert len(m.step(row(60))) == 1


def test_recovery_failure_keeps_journal_blocked(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = ready_to_fire(path)
    with monkeypatch.context() as mp:
        mp.setattr(event_log, "open", faulty_open(path, partial=25), raising=False)
        with pytest.raises(JournalRecoveryRequired):
            m.step(row(60))
        with pytest.raises(JournalRecoveryRequired):
            m.recover_journal()  # cannot cut the tail while storage is still failing
    assert m.journal.needs_recovery
    m.recover_journal()
    assert len(m.step(row(60))) == 1


def test_recovery_refuses_conflicting_file(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = ready_to_fire(path)
    with monkeypatch.context() as mp:
        mp.setattr(event_log, "open", faulty_open(path, rollback="fail"), raising=False)
        mp.setattr(event_log.os, "fsync", fail_fsync)
        with pytest.raises(JournalRecoveryRequired):
            m.step(row(60))
    fired = read_journal(path)[0]
    with REAL_OPEN(path, "a") as f:  # someone else wrote a conflicting record
        f.write(json.dumps({**fired, "confidence": 0.99, "decision_mean_score": 0.99}) + "\n")
    with pytest.raises(JournalConflictError):
        m.recover_journal()
    assert m.journal.needs_recovery


def test_partial_failure_at_second_record_retries_consistently(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = ready_to_fire(path, both=True)
    before = state(m)
    calls = {"n": 0}

    def fail_second(fd):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("second record failed")
        return REAL_FSYNC(fd)
    with monkeypatch.context() as mp:
        mp.setattr(event_log.os, "fsync", fail_second)
        with pytest.raises(EventPersistenceError):
            m.step(row(60))
    assert state(m) == before
    with pytest.raises(RuntimeError):
        m.step(row(61))
    assert len(m.step(row(60))) == 2
    assert len(read_journal(path)) == 2 and m._last_frame == 60


# ---- Fix 2: one validated reduction path for load / events / export ----

def _finished_journal(path):
    m, _ = fired_monitor(path)
    m.finish()
    return read_journal(path)


def test_identical_duplicates_are_deduplicated_everywhere(tmp_path):
    path = tmp_path / "j.jsonl"
    records = _finished_journal(path)
    with REAL_OPEN(path, "ab") as f:
        for line in path.read_bytes().splitlines(keepends=True):
            f.write(line)  # every record twice
    assert len(read_journal(path)) == 2 * len(records)
    expected = load_journal(path).events()
    assert len(expected) == 1
    assert EventJournal(path).events() == expected
    assert export_events(path, tmp_path / "e.jsonl") == expected
    assert project_events(read_journal(path)) == expected


@pytest.mark.parametrize("change", [{"end_s": 999}, {"end_reason": "tracking_lost"},
                                    {"bout_frames": 31, "last_frame": 90}])
def test_conflicting_closure_is_rejected_everywhere(tmp_path, change):
    path = tmp_path / "j.jsonl"
    closed = _finished_journal(path)[-1]
    with REAL_OPEN(path, "a") as f:
        f.write(json.dumps({**closed, **change}) + "\n")
    for read in (lambda: EventJournal(path), lambda: load_journal(path),
                 lambda: export_events(path, tmp_path / "e.jsonl"),
                 lambda: project_events(read_journal(path))):
        with pytest.raises(JournalConflictError):
            read()
    assert not (tmp_path / "e.jsonl").exists()


def test_conflicting_answers_are_rejected_everywhere(tmp_path):
    path = tmp_path / "j.jsonl"
    m, first = fired_monitor(path)
    m.answer(first["event_id"], "confirmed")
    with REAL_OPEN(path, "a") as f:
        f.write(json.dumps({"record": "event_answered", "event_id": first["event_id"],
                            "researcher_answer": "rejected"}) + "\n")
    with pytest.raises(JournalConflictError):
        EventJournal(path)
    with pytest.raises(JournalConflictError):
        export_events(path, tmp_path / "e.jsonl")


@pytest.mark.parametrize("record, change, msg", [
    (1, {"bout_frames": 10}, "bout_frames"),        # count inconsistent with frames
    (1, {"end_s": 100.0}, "end_s"),                  # duration inconsistent with frames
    (0, {"decision_frames": 7}, "decision_frames"),
    (0, {"confidence": 0.7}, "confidence"),
    (0, {"fired_s": 1.5}, "min_duration_s"),         # fired before the duration was reached
    (0, {"extra": 1}, "fields"),
    (1, {"end_reason": "magic"}, "end_reason"),
])
def test_inconsistent_records_are_rejected_everywhere(tmp_path, record, change, msg):
    path = tmp_path / "j.jsonl"
    records = _finished_journal(path)
    records[record] = {**records[record], **change}
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    for read in (lambda: EventJournal(path), lambda: export_events(path, tmp_path / "e.jsonl")):
        with pytest.raises(JournalIntegrityError, match=msg):
            read()


def test_closure_before_fired_is_rejected(tmp_path):
    path = tmp_path / "j.jsonl"
    fired, closed = _finished_journal(path)
    path.write_text(json.dumps(closed) + "\n" + json.dumps(fired) + "\n")
    with pytest.raises(JournalIntegrityError, match="no earlier event_fired"):
        export_events(path, tmp_path / "e.jsonl")


def test_answer_to_non_ask_event_in_file_is_rejected(tmp_path):
    path = tmp_path / "j.jsonl"
    run_monitor([row(i, 0.95) for i in range(61)], rules(), journal_path=path, run_id="a")
    with REAL_OPEN(path, "a") as f:
        f.write(json.dumps({"record": "event_answered", "event_id": "E-a-1",
                            "researcher_answer": "confirmed"}) + "\n")
    with pytest.raises(JournalIntegrityError, match="only 'ask'"):
        load_journal(path)


# ---- Fix 3: causal same-run replay ----

def test_codex_counterexample_replay_returns_only_evidence_at_firing(tmp_path):
    path = tmp_path / "j.jsonl"
    m, first = fired_monitor(path)
    m.answer(first["event_id"], "confirmed")
    for i in range(61, 90):
        m.step(row(i, 0.95))
    m.finish()
    replay = Monitor(rules(), journal_path=path, run_id="review")
    for i in range(61):
        current = replay.step(row(i))
    assert current[0] == first


def test_replay_snapshot_hides_future_end_scores_and_answer(tmp_path):
    path = tmp_path / "j.jsonl"
    data = [row(i, 0.6) for i in range(61)] + [row(i, 0.95) for i in range(61, 90)]
    m = Monitor(rules(), journal_path=path, run_id="r")
    for r in data[:61]:
        m.step(r)
    m.answer("E-r-1", "confirmed")
    for r in data[61:]:
        m.step(r)
    m.finish()
    before = path.read_bytes()

    replay = Monitor(rules(), journal_path=path, run_id="r")
    snapshots = [e for r in data for e in replay.step(r)]
    (snap,) = snapshots
    assert snap["researcher_answer"] is None
    assert snap["end_s"] == pytest.approx(61 / FPS)  # not the future 3.0
    ev = snap["evidence"]
    assert ev["bout_closed"] is False and ev["bout_frames"] is None
    assert ev["bout_max_score"] is None  # the later 0.95 scores are not visible
    # finish(): what this replay computed; no answer was given through it
    (done,) = replay.finish()
    assert done["end_s"] == pytest.approx(3.0) and done["researcher_answer"] is None
    # current state (recovery / inspection) is separate and still has the answer
    assert replay.journal.events()[0]["researcher_answer"] == "confirmed"
    assert path.read_bytes() == before  # replay wrote nothing


def test_replay_can_answer_its_events_idempotently(tmp_path):
    path = tmp_path / "j.jsonl"
    m, first = fired_monitor(path)
    m.answer(first["event_id"], "confirmed")
    m.finish()
    replay = Monitor(rules(), journal_path=path, run_id="review")
    for i in range(61):
        replay.step(row(i))
    assert replay.answer(first["event_id"], "confirmed") is False  # already on disk
    with pytest.raises(JournalConflictError):
        replay.answer(first["event_id"], "rejected")
    assert replay.finish()[0]["researcher_answer"] == "confirmed"


def test_monitor_answer_only_for_its_own_events(tmp_path):
    path = tmp_path / "j.jsonl"
    run_monitor([row(i) for i in range(61)], rules(), journal_path=path, run_id="a")
    m = Monitor(rules(), journal_path=path, run_id="b")
    with pytest.raises(KeyError, match="not an event of this monitor run"):
        m.answer("E-a-1", "confirmed")
    EventJournal(path).record_answer("E-a-1", "confirmed")  # direct journal access still works


def test_replay_after_crash_appends_only_missing_records(tmp_path):
    path = tmp_path / "j.jsonl"
    data = [row(i) for i in range(90)] + [row(90, 0.1)]
    m = Monitor(rules(), journal_path=path, run_id="c")
    for r in data[:70]:
        m.step(r)
    del m  # crash before the bout closed
    events = run_monitor(data, rules(), journal_path=path, run_id="c")
    assert [r["record"] for r in read_journal(path)] == ["event_fired", "event_closed"]
    assert events[0]["evidence"]["bout_frames"] == 90


# ---- Fix 4: records cannot be mutated from outside ----

def test_nested_record_mutation_does_not_change_journal(tmp_path):
    path = tmp_path / "j.jsonl"
    m, first = fired_monitor(path)
    original = m.journal.events()
    recs = m.journal.records
    recs[0]["rule"]["min_duration_s"] = 999
    recs[0]["decision_frames"] = 1
    got = m.journal.get("event_fired", first["event_id"])
    got["rule"]["alert_threshold"] = 0.0
    ev = m.journal.event(first["event_id"])
    ev["evidence"]["frames"] = 0
    first["evidence"]["mean_score"] = 0.0
    assert m.journal.events() == original
    assert m.journal.records[0]["rule"]["min_duration_s"] == 2.0
    assert load_journal(path).events() == original
