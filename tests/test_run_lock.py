"""The Live Agent's one-investigation lock (investigation_v1_runs.RunLock, fcntl.flock). New file.

Threads and separate processes; no network, no model, no data. The paid-path tests use SYNTHETIC fixtures and the
in-process OpenRouter emulator only.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import live_tracker.investigation_live_v1 as lv1
import live_tracker.investigation_v1_runs as runs
from test_live_agent_page import offline, provider, saved_dirs, synthetic  # noqa: F401 (fixtures)
from test_investigation_v1_validation import inputs, recorded_run  # noqa: F401 (fixture)

ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import sys, time
from live_tracker.investigation_v1_runs import RunBusy, RunLock
root, mode = sys.argv[1], sys.argv[2]
try:
    with RunLock(root):
        t0 = time.time()
        if mode == "hold":
            print("entered", flush=True)
            sys.stdin.readline()                 # held until the parent says so (or the process is killed)
        else:
            time.sleep(2.0)
        t1 = time.time()
    if mode != "hold":
        print("entered", t0, t1, flush=True)
except RunBusy:
    print("busy", flush=True)
"""


def busy(root):
    with pytest.raises(runs.RunBusy):
        with runs.RunLock(root):
            pass


def child(root, mode):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.Popen([sys.executable, "-c", CHILD, str(root), mode], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, env=env)


# ---- Codex's race ----

def test_codex_a_holder_paused_before_its_metadata_is_written_still_excludes_b(tmp_path, monkeypatch):
    acquired, go_on, leave = threading.Event(), threading.Event(), threading.Event()
    real = runs.RunLock._write_metadata

    def paused(self, fd):                    # A holds the OS lock but has written nothing yet
        acquired.set()
        go_on.wait(10)
        real(self, fd)
    monkeypatch.setattr(runs.RunLock, "_write_metadata", paused)
    entered = []

    def session_a():
        with runs.RunLock(tmp_path):
            entered.append("A")
            leave.wait(10)
    a = threading.Thread(target=session_a)
    a.start()
    assert acquired.wait(10)
    lock_file = tmp_path / runs.LOCK_NAME
    assert lock_file.read_bytes() == b""                                     # the empty file of the reproduction
    monkeypatch.setattr(runs.RunLock, "_write_metadata", real)
    busy(tmp_path)                                                           # B: refused, nothing deleted
    assert runs.RunLock(tmp_path).holder() is not None and lock_file.exists()
    go_on.set()
    for _ in range(100):
        if entered:
            break
        threading.Event().wait(0.05)
    assert entered == ["A"]
    busy(tmp_path)                                                           # still refused inside A's run
    leave.set()
    a.join(10)
    assert runs.RunLock(tmp_path).holder() is None


def test_a_release_never_releases_a_later_holder(tmp_path):
    a = runs.RunLock(tmp_path)
    with a:
        inode = (tmp_path / runs.LOCK_NAME).stat().st_ino
    b = runs.RunLock(tmp_path)
    b.__enter__()
    try:
        a.__exit__(None, None, None)                                         # A's late or repeated cleanup
        busy(tmp_path)                                                       # B still holds it
        assert (tmp_path / runs.LOCK_NAME).stat().st_ino == inode            # the same inode, never unlinked
    finally:
        b.__exit__(None, None, None)
    with runs.RunLock(tmp_path):
        pass


@pytest.mark.parametrize("content", [b"", b"not json", b"{\"pid\": 1}", b"\x00\xff", b"[]"])
def test_an_empty_or_malformed_file_never_bypasses_an_active_lock(tmp_path, content):
    with runs.RunLock(tmp_path):
        (tmp_path / runs.LOCK_NAME).write_bytes(content)                     # clobbered content
        busy(tmp_path)
        assert runs.RunLock(tmp_path).holder() is not None


@pytest.mark.parametrize("content", [b"", b"garbage", json.dumps({"pid": os.getpid()}).encode()])
def test_a_lock_file_left_on_disk_but_not_held_is_reusable(tmp_path, content):
    (tmp_path / runs.LOCK_NAME).write_bytes(content)
    inode = (tmp_path / runs.LOCK_NAME).stat().st_ino
    assert runs.RunLock(tmp_path).holder() is None
    with runs.RunLock(tmp_path):
        assert json.loads((tmp_path / runs.LOCK_NAME).read_text())["pid"] == os.getpid()
    assert (tmp_path / runs.LOCK_NAME).stat().st_ino == inode


def test_exceptions_release_the_lock(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        with runs.RunLock(tmp_path):
            raise ValueError("a failed run")
    assert runs.RunLock(tmp_path).holder() is None

    def broken(self, fd):
        raise OSError("disk full")
    monkeypatch.setattr(runs.RunLock, "_write_metadata", broken)
    with pytest.raises(runs.RunBusy):                                        # fail closed, and released
        with runs.RunLock(tmp_path):
            pytest.fail("entered without a recorded lock")
    monkeypatch.undo()
    with runs.RunLock(tmp_path):
        pass


def test_filesystem_problems_fail_closed(tmp_path):
    (tmp_path / "dir" / runs.LOCK_NAME).mkdir(parents=True)                  # the lock path is a directory
    busy(tmp_path / "dir")
    target = tmp_path / "elsewhere"
    target.write_text("")
    (tmp_path / "link").mkdir()
    (tmp_path / "link" / runs.LOCK_NAME).symlink_to(target)                  # or a symlink
    busy(tmp_path / "link")
    assert runs.RunLock(tmp_path / "link").holder() is not None
    blocker = tmp_path / "file"
    blocker.write_text("")
    busy(blocker)                                                            # the runs directory cannot exist


def test_one_lock_object_is_held_at_most_once(tmp_path):
    lock = runs.RunLock(tmp_path)
    with lock:
        with pytest.raises(runs.RunBusy):
            lock.__enter__()


# ---- separate processes ----

def test_concurrent_processes_cannot_both_enter(tmp_path):
    procs = [child(tmp_path, "brief") for _ in range(4)]
    outs = [p.communicate(timeout=60)[0].split() for p in procs]
    assert all(o and o[0] in ("entered", "busy") for o in outs)
    spans = sorted((float(o[1]), float(o[2])) for o in outs if o[0] == "entered")
    assert spans and all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))     # never two inside at once


def test_a_process_holding_the_lock_excludes_this_one_until_it_exits_or_is_killed(tmp_path):
    p = child(tmp_path, "hold")
    assert p.stdout.readline().strip() == "entered"
    busy(tmp_path)
    assert runs.RunLock(tmp_path).holder()["pid"] == p.pid
    p.stdin.write("\n")
    p.stdin.flush()
    p.wait(10)
    with runs.RunLock(tmp_path):                                             # released on exit
        pass
    p = child(tmp_path, "hold")
    assert p.stdout.readline().strip() == "entered"
    busy(tmp_path)
    p.kill()                                                                 # killed while holding it
    p.wait(10)
    with runs.RunLock(tmp_path):                                             # recovered without deleting anything
        pass
    assert (tmp_path / runs.LOCK_NAME).exists()


# ---- the paid path ----

def test_a_refused_concurrent_run_sends_no_model_request(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    root = synthetic["tmp"] / "live"
    p = child(root, "hold")
    try:
        assert p.stdout.readline().strip() == "entered"
        with pytest.raises(runs.RunBusy):
            runs.execute(runs.RunConfig(validation_run=runs.REFERENCE_RUN), synthetic["vi"],
                         confirm_budget_usd=0.5, authorization=lv1.PAID_RUN_AUTHORIZATION, root=root)
    finally:
        p.kill()
        p.wait(10)
    assert emu.requests == [] and saved_dirs(synthetic["tmp"]) == []


def test_sequential_authorized_runs_remain_possible(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    config = runs.RunConfig(validation_run=runs.REFERENCE_RUN)
    names = [runs.execute(config, synthetic["vi"], confirm_budget_usd=0.5,
                          authorization=lv1.PAID_RUN_AUTHORIZATION)[0] for _ in range(2)]
    assert len(set(names)) == 2 and sorted(saved_dirs(synthetic["tmp"])) == sorted(names)
    assert len(emu.requests) == 14
    assert all(runs.load_saved(synthetic["tmp"] / "live" / n).verified for n in names)
    assert runs.RunLock(synthetic["tmp"] / "live").holder() is None
