"""Concurrent researcher answers on one shared replay journal (D3-2 review). New file.

Two replay instances (threads) and independent processes answer the same ask
event of the SYNTHETIC demo session at the same moment. The per-run lock
(replay.run_lock) must make every answer atomic: refresh, check, journal
append and position record together. No network, no real data.
"""

import errno
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import live_tracker.event_log as el
import live_tracker.investigation_data as idata
import live_tracker.replay as rp
from live_tracker.fake_scores import DEMO_SESSION, demo_scores
from test_replay import R_ALERT, ruleset

ROOT = Path(__file__).resolve().parents[1]


def build(tmp_path):
    return rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT), journal_path=tmp_path / "j.jsonl",
                           answers_path=tmp_path / "a.jsonl")


def ask_id(r):
    return r.end_state().events[1]["event_id"]


def answered(tmp_path):
    state = el.load_journal(tmp_path / "j.jsonl")             # the journal stays valid (no conflicting records)
    records = [r for r in state.records if r["record"] == "event_answered"]
    positions = [json.loads(x) for x in (tmp_path / "a.jsonl").read_text().splitlines()] \
        if (tmp_path / "a.jsonl").exists() else []
    return records, positions


def race(replays, answers, frames):
    barrier = threading.Barrier(len(replays))
    results = [None] * len(replays)

    def go(i):
        barrier.wait()
        try:
            results[i] = ("written" if replays[i].answer(ask_id(replays[i]), answers[i], at_frame=frames[i])
                          else "no-op")
        except rp.ReplayError as exc:
            results[i] = f"refused: {exc}"
    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(replays))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return results


@pytest.mark.parametrize("attempt", range(5))
def test_contradictory_answers_at_the_same_moment(tmp_path, attempt):
    a, b = build(tmp_path), build(tmp_path)                    # two tabs, each with its own journal object
    results = race([a, b], ["confirmed", "rejected"], [280, 300])
    assert sorted(r.split(":")[0] for r in results) == ["refused", "written"], results
    assert "answers are final" in next(r for r in results if r.startswith("refused"))
    records, positions = answered(tmp_path)
    assert len(records) == 1 and len(positions) == 1
    assert positions[0]["researcher_answer"] == records[0]["researcher_answer"]
    winner = records[0]["researcher_answer"]
    assert build(tmp_path).answer_status(ask_id(a)) == {
        "researcher_answer": winner, "answered_frame": 280 if winner == "confirmed" else 300}


@pytest.mark.parametrize("attempt", range(5))
def test_the_same_answer_at_the_same_moment(tmp_path, attempt):
    a, b = build(tmp_path), build(tmp_path)
    results = race([a, b], ["confirmed", "confirmed"], [280, 290])
    assert sorted(results) == ["no-op", "written"], results
    records, positions = answered(tmp_path)
    assert len(records) == 1 and len(positions) == 1


WORKER = r"""
import json, sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/tests"]
import live_tracker.replay as rp
from live_tracker.fake_scores import DEMO_SESSION, demo_scores
from test_replay import R_ALERT, ruleset
d, answer, frame = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
r = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT), journal_path=d / "j.jsonl",
                    answers_path=d / "a.jsonl")
eid = r.end_state().events[1]["event_id"]
print("ready", flush=True)
while not (d / "go").exists():
    time.sleep(0.001)
try:
    print(json.dumps("written" if r.answer(eid, answer, at_frame=frame) else "no-op"))
except rp.ReplayError as exc:
    print(json.dumps("refused: " + str(exc)))
"""


def processes(tmp_path, answers):
    build(tmp_path)                                             # the journal exists before the race
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    procs = [subprocess.Popen([sys.executable, "-I", str(script), str(ROOT), str(tmp_path), a, str(280 + 5 * i)],
                              stdout=subprocess.PIPE, text=True) for i, a in enumerate(answers)]
    for p in procs:
        assert p.stdout.readline().strip() == "ready"
    (tmp_path / "go").write_text("")
    out = []
    for p in procs:
        stdout, _ = p.communicate(timeout=60)
        assert p.returncode == 0
        out.append(json.loads(stdout.strip().splitlines()[-1]))
    return out


def test_independent_processes_with_contradictory_answers(tmp_path):
    results = processes(tmp_path, ["confirmed", "rejected", "confirmed", "rejected"])
    written = [r for r in results if r == "written"]
    assert len(written) == 1, results
    records, positions = answered(tmp_path)
    winner = records[0]["researcher_answer"]
    for r, a in zip(results, ["confirmed", "rejected", "confirmed", "rejected"]):
        if r != "written":
            assert (r == "no-op") if a == winner else ("answers are final" in r), (r, a, winner)
    assert len(records) == 1 and len(positions) == 1 and positions[0]["researcher_answer"] == winner


def test_independent_processes_with_the_same_answer(tmp_path):
    results = processes(tmp_path, ["confirmed"] * 4)
    assert sorted(results) == ["no-op"] * 3 + ["written"], results
    records, positions = answered(tmp_path)
    assert len(records) == 1 and len(positions) == 1


def test_a_held_lock_times_out_without_writing_and_is_released(tmp_path, monkeypatch):
    r = build(tmp_path)
    monkeypatch.setattr(rp, "LOCK_TIMEOUT_S", 0.2)
    with rp.run_lock(tmp_path / "j.jsonl"):                     # another writer holds the run
        t = time.monotonic()
        with pytest.raises(rp.ReplayError, match="busy"):
            r.answer(ask_id(r), "confirmed", at_frame=280)
        assert time.monotonic() - t < 2.0
        with pytest.raises(rp.ReplayError, match="busy"):
            r.refresh()                                          # readers wait too (no repair mid-write)
    assert answered(tmp_path) == ([], [])
    assert r.answer(ask_id(r), "confirmed", at_frame=280) is True               # released: works now


def test_the_lock_is_released_after_a_refusal_and_an_error(tmp_path, monkeypatch):
    r = build(tmp_path)
    r.answer(ask_id(r), "confirmed", at_frame=280)
    with pytest.raises(rp.ReplayError, match="answers are final"):           # refused inside the lock
        r.answer(ask_id(r), "rejected", at_frame=290)
    with rp.run_lock(tmp_path / "j.jsonl", timeout_s=0.2):
        pass                                                                  # released
    d = tmp_path / "fail"
    d.mkdir()
    f = build(d)

    def broken(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(rp, "_append_line", broken)                          # fails after the journal write
    with pytest.raises(rp.ReplayError, match="position could not be recorded"):
        f.answer(ask_id(f), "confirmed", at_frame=280)
    with rp.run_lock(d / "j.jsonl", timeout_s=0.2):
        pass                                                                  # released after an error too
    records, positions = answered(d)
    assert len(records) == 1 and positions == []                              # the answer itself is kept
    assert build(d).answer_status(ask_id(f)) == {"researcher_answer": "confirmed", "answered_frame": None}


# ---- the Windows lock path (msvcrt), exercised on any platform with a stand-in module ----

class FakeMsvcrt:
    """SYNTHETIC stand-in for Windows' msvcrt: locking() raises the scripted errnos in order, then succeeds."""
    LK_UNLCK, LK_LOCK, LK_NBLCK = 0, 1, 2

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), []

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR)))
        if self.errors:
            code = self.errors.pop(0)
            raise OSError(code, os.strerror(code))


@pytest.fixture
def lock_fd(tmp_path):
    fd = os.open(tmp_path / "j.lock", os.O_RDWR | os.O_CREAT, 0o600)
    yield fd
    os.close(fd)


def test_windows_try_lock_is_non_blocking_from_byte_zero(lock_fd):
    fake = FakeMsvcrt()
    os.lseek(lock_fd, 5, os.SEEK_SET)
    assert rp._windows_try_lock(lock_fd, fake) is True
    rp._windows_unlock(lock_fd, fake)
    assert fake.calls == [(fake.LK_NBLCK, 1, 0), (fake.LK_UNLCK, 1, 0)]


@pytest.mark.parametrize("contention", sorted(idata.WINDOWS_LOCK_CONTENTION))
def test_windows_contention_means_busy_not_failure(lock_fd, contention):
    assert rp._windows_try_lock(lock_fd, FakeMsvcrt(contention)) is False


@pytest.mark.parametrize("permanent", [errno.EBADF, errno.EINVAL, errno.EIO])
def test_windows_other_lock_errors_propagate(lock_fd, permanent):
    with pytest.raises(OSError) as err:
        rp._windows_try_lock(lock_fd, FakeMsvcrt(permanent))
    assert err.value.errno == permanent


def test_run_lock_times_out_while_busy_and_releases_after(tmp_path, monkeypatch):
    journal = tmp_path / "run.journal.jsonl"
    monkeypatch.setattr(rp, "_try_lock", lambda fd: False)            # another holder never lets go
    with pytest.raises(rp.ReplayError, match="busy"):
        with rp.run_lock(journal, timeout_s=0.05):
            pass
    monkeypatch.undo()
    with rp.run_lock(journal, timeout_s=0.05):                       # the real lock is free again
        pass
