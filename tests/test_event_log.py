"""Tests for the event journal, persistence failures and the §4 export (new file)."""

import json
import math
import os

import numpy as np
import pytest

import live_tracker.event_log as event_log
from live_tracker.event_log import (EVENT_KEYS, EventJournal, EventPersistenceError,
                                    JournalConflictError, JournalIntegrityError,
                                    JournalRecoveryRequired, export_events, load_journal,
                                    project_events, read_journal, read_jsonl, write_jsonl)
from live_tracker.fake_scores import demo_scores
from live_tracker.monitor import Monitor, run_monitor
from live_tracker.rules import RuleSet

FPS = 30.0
REAL_FSYNC = os.fsync  # tests patch os.fsync; keep the real one
R1 = {"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0}


def rule(**change):
    rs = RuleSet()
    rs.add({**R1, **change})
    rs.approve("R1")
    return rs


def rows(attack, session="SYNTHETIC_t", start_frame=0):
    return [{"session": session, "frame": start_frame + i, "time_s": (start_frame + i) / FPS,
             "attack": a, "mount": 0.0, "investigation": 0.0, "tracking_ok": True}
            for i, a in enumerate(attack)]


def kinds(path):
    return [(r["record"], r["event_id"]) for r in read_journal(path)]


class FailingFsync:
    """Makes os.fsync in event_log fail the next `times` calls."""

    def __init__(self, monkeypatch, times=1):
        self.left = times
        monkeypatch.setattr(event_log.os, "fsync", self)

    def __call__(self, fd):
        if self.left > 0:
            self.left -= 1
            raise OSError("simulated disk failure")
        return REAL_FSYNC(fd)


# ---- lifecycle records ----

def test_fired_is_persisted_before_it_is_returned(tmp_path):
    path = tmp_path / "j.jsonl"
    m = Monitor(rule(), journal_path=path, run_id="t")
    for r in rows([0.9] * 61):
        fired = m.step(r)
        if fired:
            # the alert is on disk at the moment the caller receives it
            assert kinds(path) == [("event_fired", fired[0]["event_id"])]
    assert len(fired) == 1
    m.finish()
    assert kinds(path) == [("event_fired", "E-t-1"), ("event_closed", "E-t-1")]


def test_lifecycle_records_content(tmp_path):
    path = tmp_path / "j.jsonl"
    events = run_monitor(rows([0.9] * 75 + [0.1]), rule(alert_mode="ask"),
                         journal_path=path, run_id="t")
    EventJournal(path).record_answer(events[0]["event_id"], "confirmed")
    fired, closed, answered = read_journal(path)
    assert fired["record"] == "event_fired" and fired["run_id"] == "t"
    assert fired["fired_frame"] == 60 and fired["fired_s"] == 2.0
    assert fired["decision_frames"] == 61 and fired["action"] == "ask"
    assert fired["rule"] == {"min_duration_s": 2.0, "ask_threshold": 0.5,
                             "alert_threshold": 0.8, "alert_mode": "ask"}
    assert closed["record"] == "event_closed" and closed["end_reason"] == "below_threshold"
    assert closed["end_s"] == pytest.approx(2.5) and closed["bout_frames"] == 75
    assert answered == {"record": "event_answered", "event_id": "E-t-1",
                        "researcher_answer": "confirmed"}


# ---- answers ----

def test_answer_updates_projection_and_persists(tmp_path):
    path = tmp_path / "j.jsonl"
    m = Monitor(rule(), journal_path=path, run_id="t")
    for r in rows([0.6] * 61):  # uncertain -> ask
        fired = m.step(r)
    eid = fired[0]["event_id"]
    assert m.answer(eid, "rejected") is True  # answered while the bout is still open
    assert m.answer(eid, "rejected") is False  # same answer again: no-op
    with pytest.raises(JournalConflictError):
        m.answer(eid, "confirmed")  # changing an answer is not supported
    (e,) = m.finish()
    assert e["researcher_answer"] == "rejected"
    assert EventJournal(path).events()[0]["researcher_answer"] == "rejected"  # survives reload
    assert [k for k, _ in kinds(path)] == ["event_fired", "event_answered", "event_closed"]


def test_answer_validation(tmp_path):
    path = tmp_path / "j.jsonl"
    (alert,) = run_monitor(rows([0.95] * 61), rule(), journal_path=path, run_id="t")
    j = EventJournal(path)
    with pytest.raises(ValueError, match="only 'ask'"):
        j.record_answer(alert["event_id"], "confirmed")
    with pytest.raises(KeyError):
        j.record_answer("E-nope-1", "confirmed")
    with pytest.raises(ValueError, match="answer must be"):
        j.record_answer(alert["event_id"], "yes")


# ---- crash and recovery ----

def test_crash_before_bout_closure(tmp_path):
    path = tmp_path / "j.jsonl"
    data = rows([0.9] * 90 + [0.1] * 10)
    m = Monitor(rule(), journal_path=path, run_id="r1")
    for r in data[:70]:
        m.step(r)
    del m  # crash: finish() never called, bout never closed
    assert kinds(path) == [("event_fired", "E-r1-1")]
    (e,) = EventJournal(path).events()  # the alert that was shown is not lost
    assert e["evidence"]["bout_closed"] is False and e["evidence"]["end_reason"] is None
    assert e["end_s"] == pytest.approx(61 / FPS)  # what was known at firing
    # recovery: replay the same input with the same run_id -> nothing duplicated
    events = run_monitor(data, rule(), journal_path=path, run_id="r1")
    assert kinds(path) == [("event_fired", "E-r1-1"), ("event_closed", "E-r1-1")]
    assert events[0]["evidence"]["bout_frames"] == 90 and events[0]["end_s"] == pytest.approx(3.0)


def test_torn_last_line_is_ignored_and_removed(tmp_path):
    path = tmp_path / "j.jsonl"
    run_monitor(rows([0.9] * 61), rule(), journal_path=path, run_id="a")
    good = path.read_bytes()
    path.write_bytes(good + b'{"record": "event_fir')  # crash in the middle of a write
    assert len(read_journal(path)) == 2
    run_monitor(rows([0.9] * 61), rule(), journal_path=path, run_id="b")
    assert [eid for _, eid in kinds(path)] == ["E-a-1", "E-a-1", "E-b-1", "E-b-1"]


def test_malformed_complete_line_is_an_error(tmp_path):
    path = tmp_path / "j.jsonl"
    path.write_text('{"record": "event_fired", "event_id": "E1"}\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        read_journal(path)
    path.write_text('{"something": "else"}\n')
    with pytest.raises(ValueError, match="not a journal record"):
        read_journal(path)


# ---- write failures and idempotent retry ----

def test_failed_write_at_firing_can_be_retried(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    data = rows([0.9] * 70)
    m = Monitor(rule(), journal_path=path, run_id="t")
    for r in data[:60]:
        assert m.step(r) == []
    FailingFsync(monkeypatch)
    with pytest.raises(EventPersistenceError):
        m.step(data[60])  # the firing frame
    assert path.read_bytes() == b""  # partial line was rolled back
    with pytest.raises(RuntimeError, match="retry"):
        m.step(data[61])  # skipping the failed frame is refused
    with pytest.raises(RuntimeError, match="retry"):
        m.finish()
    fired = m.step(data[60])  # retry the same frame: now it works
    assert [e["event_id"] for e in fired] == ["E-t-1"]
    for r in data[61:]:
        assert m.step(r) == []
    m.finish()
    assert kinds(path) == [("event_fired", "E-t-1"), ("event_closed", "E-t-1")]


def test_partial_failure_retry_writes_each_record_once(tmp_path, monkeypatch):
    # one frame both closes R1's event and fires R2's: the first write works,
    # the second fails; the retry must not write the first again
    path = tmp_path / "j.jsonl"
    rs = RuleSet()
    rs.add({"rule_id": "R1", "behavior": "attack", "min_duration_s": 1.0})
    rs.add({"rule_id": "R2", "behavior": "mount", "min_duration_s": 2.0})
    rs.approve("R1")
    rs.approve("R2")
    m = Monitor(rs, journal_path=path, run_id="t")
    data = [dict(r, mount=0.9) for r in rows([0.9] * 60 + [0.1] * 5)]
    for r in data[:60]:
        m.step(r)
    calls = {"n": 0}

    def fail_second(fd):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return REAL_FSYNC(fd)
    monkeypatch.setattr(event_log.os, "fsync", fail_second)
    with pytest.raises(EventPersistenceError):
        m.step(data[60])
    assert kinds(path) == [("event_fired", "E-t-1"), ("event_closed", "E-t-1")]
    fired = m.step(data[60])
    assert [e["rule_id"] for e in fired] == ["R2"]
    assert kinds(path) == [("event_fired", "E-t-1"), ("event_closed", "E-t-1"),
                           ("event_fired", "E-t-2")]


def test_failed_write_at_finish_can_be_retried(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    m = Monitor(rule(), journal_path=path, run_id="t")
    for r in rows([0.9] * 61):
        m.step(r)
    FailingFsync(monkeypatch)
    with pytest.raises(EventPersistenceError):
        m.finish()
    with pytest.raises(RuntimeError, match="retry"):
        m.step(rows([0.9], start_frame=61)[0])  # finish must be retried first
    (e,) = m.finish()
    assert e["evidence"]["bout_closed"] is True
    assert kinds(path) == [("event_fired", "E-t-1"), ("event_closed", "E-t-1")]
    assert m.finish() == [e]


def test_failed_answer_write_is_not_recorded(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    (e,) = run_monitor(rows([0.6] * 61), rule(), journal_path=path, run_id="t")
    j = EventJournal(path)
    FailingFsync(monkeypatch)
    with pytest.raises(EventPersistenceError):
        j.record_answer(e["event_id"], "confirmed")
    assert j.events()[0]["researcher_answer"] is None
    assert j.record_answer(e["event_id"], "confirmed") is True


def test_append_is_idempotent_and_detects_conflicts(tmp_path):
    path = tmp_path / "j.jsonl"
    run_monitor(rows([0.6] * 61), rule(), journal_path=path, run_id="x")
    j = EventJournal(path)
    rec = {"record": "event_answered", "event_id": "E-x-1", "researcher_answer": "confirmed"}
    assert j.append(rec) is True
    assert j.append(dict(rec)) is False
    with pytest.raises(JournalConflictError):
        j.append({**rec, "researcher_answer": "rejected"})
    assert len(read_journal(path)) == 3
    with pytest.raises(JournalIntegrityError):
        j.append({"record": "event_deleted", "event_id": "E-x-1"})
    with pytest.raises(JournalIntegrityError, match="no earlier event_fired"):
        j.append({**rec, "event_id": "E-x-9"})


# ---- IDs across runs ----

def test_event_ids_unique_across_runs(tmp_path):
    path = tmp_path / "j.jsonl"
    for session in ("SYNTHETIC_a", "SYNTHETIC_b", "SYNTHETIC_a"):
        run_monitor(rows([0.9] * 61, session=session), rule(), journal_path=path)  # random run_id
    ids = [e["event_id"] for e in EventJournal(path).events()]
    assert len(ids) == 3 and len(set(ids)) == 3


def test_same_run_id_replay_is_idempotent(tmp_path):
    path = tmp_path / "j.jsonl"
    run_monitor(demo_scores(), rule(), journal_path=path, run_id="demo")
    before = path.read_bytes()
    run_monitor(demo_scores(), rule(), journal_path=path, run_id="demo")
    assert path.read_bytes() == before


def test_same_run_id_with_different_data_is_a_conflict(tmp_path):
    path = tmp_path / "j.jsonl"
    run_monitor(rows([0.9] * 61), rule(), journal_path=path, run_id="r")
    with pytest.raises(JournalConflictError):
        run_monitor(rows([0.6] * 61), rule(), journal_path=path, run_id="r")


def test_events_can_be_filtered_by_run(tmp_path):
    path = tmp_path / "j.jsonl"
    run_monitor(rows([0.9] * 61), rule(), journal_path=path, run_id="a")
    (b,) = run_monitor(rows([0.9] * 61), rule(), journal_path=path, run_id="b")
    assert EventJournal(path).events(run_id="b") == [b]
    assert len(EventJournal(path).events()) == 2


# ---- §4 projection / export ----

def test_export_matches_interfaces_section_4(tmp_path):
    journal, out = tmp_path / "j.jsonl", tmp_path / "events.jsonl"
    events = run_monitor(demo_scores(), rule(), journal_path=journal, run_id="demo")
    exported = export_events(journal, out)
    assert exported == events and read_jsonl(out) == events
    for line in out.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        assert tuple(e) == EVENT_KEYS  # §4 keys only, in §4 order
        assert e["event_id"].startswith("E") and e["session"].startswith("SYNTHETIC_")
        assert e["action"] in {"alert", "ask", "log"} and e["researcher_answer"] is None
        assert 0.0 <= e["confidence"] <= 1.0
        ev = e["evidence"]
        assert {"mean_score", "min_score", "frames"} <= set(ev)
        assert e["confidence"] == ev["mean_score"]  # as in the §4 example
        assert e["start_s"] < ev["fired_s"] < e["end_s"]
        # end-exclusive: frames of the whole bout / fps == end_s - start_s
        assert ev["bout_frames"] / FPS == pytest.approx(e["end_s"] - e["start_s"])


def test_end_exclusive_never_overlaps_next_frame():
    for end_kind in ("low", "lost"):
        data = rows([0.9] * 70 + [0.1])
        if end_kind == "lost":
            data[-1] = dict(data[-1], attack=0.9, tracking_ok=False)
        (e,) = run_monitor(data, rule(), run_id="t")
        # end_s is the time of the first non-evidence frame: no overlap, no gap
        assert e["end_s"] == pytest.approx(data[-1]["time_s"], abs=1e-12)
        assert data[-2]["time_s"] < e["end_s"] <= data[-1]["time_s"] + 1e-12


def test_write_jsonl_handles_numpy_types_and_refuses_nan(tmp_path):
    path = tmp_path / "e.jsonl"
    e = {k: None for k in EVENT_KEYS}
    e.update(event_id="E1", confidence=np.float64(0.9),
             evidence={"frames": np.int64(60), "mean_score": np.float32(0.5)})
    write_jsonl(path, [e])
    (back,) = read_jsonl(path)
    assert back["confidence"] == 0.9 and back["evidence"]["frames"] == 60
    with pytest.raises(ValueError):
        write_jsonl(path, [dict(e, confidence=math.nan)])
    assert read_jsonl(path) == [back]  # failed write left the old file intact


def test_in_memory_journal_without_path():
    events = run_monitor(demo_scores(), rule(), run_id="t")
    assert len(events) == 2 and all(e["evidence"]["bout_closed"] for e in events)
