"""The Live Agent's one-investigation lock (investigation_v1_runs.RunLock: fcntl.flock, msvcrt.locking on Windows).
New file.

Threads and separate processes; no network, no model, no data. The paid-path tests use SYNTHETIC fixtures and the
in-process OpenRouter emulator only. The tests above "the Windows primitives" run on the platform's real lock; the
Windows primitives are also exercised on POSIX through a stand-in msvcrt backed by flock.
"""

import errno
import json
import os
import stat
import subprocess
import sys
import threading
import time
import types
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


def released(root):
    """Free again after its holder exited or was killed. POSIX releases at once; Windows releases a terminated
    process's byte-range locks asynchronously, so there it is polled for at most 10 s."""
    deadline = time.monotonic() + (10 if os.name == "nt" else 0)
    while runs.RunLock(root).holder() is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    return runs.RunLock(root).holder() is None


def symlink_or_skip(link, target):
    try:
        link.symlink_to(target)
    except OSError:                                  # Windows without the symlink privilege
        pytest.skip("symlinks cannot be created here")


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


MALFORMED = [b"", b"not json", b"{\"pid\": 1}", b"\x00\xff", b"[]"]


@pytest.mark.parametrize("content", MALFORMED)
def test_malformed_metadata_seeded_before_the_lock_never_bypasses_it(tmp_path, monkeypatch, content):
    (tmp_path / runs.LOCK_NAME).write_bytes(content)                         # seeded before anyone holds the lock
    monkeypatch.setattr(runs.RunLock, "_write_metadata", lambda self, fd: None)   # the holder never overwrites it
    with runs.RunLock(tmp_path):
        assert (tmp_path / runs.LOCK_NAME).read_bytes() == content           # an active lock, malformed metadata
        busy(tmp_path)
        assert runs.RunLock(tmp_path).holder() is not None


@pytest.mark.parametrize("content", MALFORMED)
def test_an_empty_or_malformed_file_never_bypasses_an_active_lock(tmp_path, content):
    with runs.RunLock(tmp_path):
        try:
            (tmp_path / runs.LOCK_NAME).write_bytes(content)                 # clobbered content
        except PermissionError:
            if os.name != "nt":                                              # Windows may refuse a second handle
                raise                                                        # truncating a locked file
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
    blocker = tmp_path / "file"
    blocker.write_text("")
    busy(blocker)                                                            # the runs directory cannot exist


def test_a_dangling_symlink_is_refused_without_creating_its_target(tmp_path, opener):
    target = tmp_path / "missing" / "target"                                 # its directory exists
    target.parent.mkdir()
    (tmp_path / "link").mkdir()
    symlink_or_skip(tmp_path / "link" / runs.LOCK_NAME, target)
    before = open_fds()
    busy(tmp_path / "link")
    assert runs.RunLock(tmp_path / "link").holder() is not None
    assert not target.exists() and list(target.parent.iterdir()) == []       # nothing was created behind it
    assert (tmp_path / "link" / runs.LOCK_NAME).is_symlink()                 # nor was the link replaced
    assert open_fds() == before


def test_an_existing_symlink_target_is_never_modified(tmp_path, opener):
    target = tmp_path / "elsewhere"
    target.write_bytes(b"precious")
    stamp = target.stat().st_mtime_ns
    (tmp_path / "link").mkdir()
    symlink_or_skip(tmp_path / "link" / runs.LOCK_NAME, target)
    busy(tmp_path / "link")
    assert runs.RunLock(tmp_path / "link").holder() is not None
    assert target.read_bytes() == b"precious" and target.stat().st_mtime_ns == stamp


def test_a_regular_lock_file_is_created_then_reused(tmp_path, opener):
    lock_file = tmp_path / runs.LOCK_NAME
    assert not lock_file.exists()
    with runs.RunLock(tmp_path):
        busy(tmp_path)
        identity = (lock_file.stat().st_dev, lock_file.stat().st_ino)
    with runs.RunLock(tmp_path):                                             # the same file, reused
        assert (lock_file.stat().st_dev, lock_file.stat().st_ino) == identity
        assert json.loads(lock_file.read_bytes())["pid"] == os.getpid()
    assert runs.RunLock(tmp_path).holder() is None


def test_a_directory_at_the_lock_path_fails_closed_without_leaking(tmp_path, opener):
    (tmp_path / runs.LOCK_NAME).mkdir()
    before = open_fds()
    busy(tmp_path)
    assert runs.RunLock(tmp_path).holder() is not None and open_fds() == before


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
    assert released(tmp_path)
    with runs.RunLock(tmp_path):                                             # released on exit
        pass
    p = child(tmp_path, "hold")
    assert p.stdout.readline().strip() == "entered"
    busy(tmp_path)
    p.kill()                                                                 # killed while holding it
    p.wait(10)
    assert released(tmp_path)
    with runs.RunLock(tmp_path):                                             # recovered without deleting anything
        pass
    assert (tmp_path / runs.LOCK_NAME).exists()


# ---- the Windows primitives (msvcrt.locking), also exercised on POSIX ----

def test_the_lock_is_the_platforms_os_lock():
    assert runs._LOCK.name == ("msvcrt.locking" if os.name == "nt" else "fcntl.flock")


class FlockMsvcrt:
    """SYNTHETIC stand-in for Windows' msvcrt on POSIX: locking() takes or releases an flock on the descriptor (one
    holder per descriptor, like a Windows byte-range lock per handle) and reports contention as Windows does
    (EACCES); scripted errnos are raised first. Records (mode, nbytes, position) of every call. open_osfhandle()
    returns the "handle" (a POSIX descriptor here) as the descriptor, or raises when told to."""
    LK_UNLCK, LK_LOCK, LK_NBLCK = 0, 1, 2

    def __init__(self, *errors, osfhandle_error=None):
        import fcntl
        self.fcntl, self.errors, self.calls, self.osfhandle_error = fcntl, list(errors), [], osfhandle_error

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR)))
        if self.errors:
            code = self.errors.pop(0)
            raise OSError(code, os.strerror(code))
        if mode == self.LK_UNLCK:
            self.fcntl.flock(fd, self.fcntl.LOCK_UN)
            return
        try:
            self.fcntl.flock(fd, self.fcntl.LOCK_EX | self.fcntl.LOCK_NB)
        except BlockingIOError:
            raise OSError(errno.EACCES, "Permission denied") from None

    def open_osfhandle(self, handle, flags):
        assert flags == 0                                                    # binary, read-write
        if self.osfhandle_error:
            raise self.osfhandle_error
        return handle


class FakeWinapi:
    """SYNTHETIC stand-in for _winapi.CreateFile on POSIX, with the semantics of OPEN_ALWAYS |
    FILE_FLAG_OPEN_REPARSE_POINT: a symlink at the path is opened as itself (O_SYMLINK on macOS, O_PATH | O_NOFOLLOW
    on Linux), never followed and never created through; a directory is refused (no FILE_FLAG_BACKUP_SEMANTICS);
    anything else is opened or created. `after_open` runs once the handle exists (to replace the path in between).
    It has no NULL attribute (only the two functions RunLock calls), so the code cannot depend on one."""

    def __init__(self, after_open=None):
        self.after_open, self.created, self.closed = after_open, [], []

    def CreateFile(self, name, access, share, security, disposition, flags, template):
        assert (access, disposition) == (runs.GENERIC_READ | runs.GENERIC_WRITE, runs.OPEN_ALWAYS)
        assert flags & runs.FILE_FLAG_OPEN_REPARSE_POINT and share == runs.FILE_SHARE_READ | runs.FILE_SHARE_WRITE
        assert security == 0 and template == 0                               # null pointers, passed as integers
        if os.path.islink(name):
            handle = os.open(name, getattr(os, "O_SYMLINK", 0) or (os.O_PATH | os.O_NOFOLLOW))
        elif os.path.isdir(name):
            raise PermissionError(errno.EACCES, "Access is denied")
        else:
            handle = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self.created.append(handle)
        if self.after_open:
            self.after_open()
        return handle

    def CloseHandle(self, handle):
        self.closed.append(handle)
        os.close(handle)


def simulated_attributes(st):
    """Windows' st_file_attributes, simulated from the POSIX mode (POSIX stat results have none)."""
    return (stat.FILE_ATTRIBUTE_REPARSE_POINT if stat.S_ISLNK(st.st_mode) else
            stat.FILE_ATTRIBUTE_DIRECTORY if stat.S_ISDIR(st.st_mode) else 0x80)


def simulate_windows(monkeypatch, msvcrt_module=None, winapi=None):
    msvcrt_module, winapi = msvcrt_module or FlockMsvcrt(), winapi or FakeWinapi()
    monkeypatch.setattr(runs, "_LOCK", runs._windows_primitives(msvcrt_module, winapi))
    monkeypatch.setattr(runs, "_file_attributes", simulated_attributes)
    return msvcrt_module, winapi


posix_only = pytest.mark.skipif(os.name == "nt", reason="the stand-ins use flock; Windows runs the real msvcrt above")


def open_fds():
    """This process's open descriptors (POSIX); None where they cannot be listed (then not compared)."""
    for d in ("/dev/fd", "/proc/self/fd"):
        if os.path.isdir(d):
            return len(os.listdir(d))
    return None


@pytest.fixture(params=["platform", pytest.param("simulated Windows", marks=posix_only)])
def opener(request, monkeypatch):
    """The platform's own lock primitives, or (on POSIX) the Windows ones over the stand-ins."""
    if request.param != "platform":
        simulate_windows(monkeypatch)
    return request.param


@pytest.fixture
def windows(monkeypatch):
    return simulate_windows(monkeypatch)[0]


@posix_only
def test_windows_lock_excludes_reads_metadata_and_releases(tmp_path, windows):
    lock = runs.RunLock(tmp_path)
    with lock:
        busy(tmp_path)                                                       # another descriptor is refused
        info = runs.RunLock(tmp_path).holder()                               # the metadata stays readable
        assert info["pid"] == os.getpid()
        size = (tmp_path / runs.LOCK_NAME).stat().st_size
        assert 0 < size < runs.METADATA_BYTES < runs.WINDOWS_LOCK_OFFSET    # the lock byte is far beyond it
        with pytest.raises(runs.RunBusy):
            lock.__enter__()
    assert runs.RunLock(tmp_path).holder() is None
    with runs.RunLock(tmp_path):
        pass
    assert {(m, n, pos) for m, n, pos in windows.calls} == {(windows.LK_NBLCK, 1, runs.WINDOWS_LOCK_OFFSET),
                                                             (windows.LK_UNLCK, 1, runs.WINDOWS_LOCK_OFFSET)}


@posix_only
def test_windows_lock_threads_never_both_enter(tmp_path, windows):
    inside, overlaps, refused = [], [], []

    def attempt():
        try:
            with runs.RunLock(tmp_path):
                inside.append(1)
                if len(inside) > 1:
                    overlaps.append(1)
                time.sleep(0.05)
                inside.pop()
        except runs.RunBusy:
            refused.append(1)
    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(10)
    assert overlaps == [] and refused and runs.RunLock(tmp_path).holder() is None


@posix_only
@pytest.mark.parametrize("code", sorted({errno.EACCES, getattr(errno, "EDEADLK", 36)}))
def test_windows_contention_is_busy(tmp_path, monkeypatch, code):
    simulate_windows(monkeypatch, FlockMsvcrt(code))
    with pytest.raises(runs.RunBusy, match="another investigation is running"):
        with runs.RunLock(tmp_path):
            pass


@posix_only
@pytest.mark.parametrize("code", [errno.EBADF, errno.EINVAL, errno.EIO])
def test_windows_other_lock_errors_fail_closed(tmp_path, monkeypatch, code):
    simulate_windows(monkeypatch, FlockMsvcrt(code))
    with pytest.raises(runs.RunBusy, match="cannot be taken"):
        with runs.RunLock(tmp_path):
            pytest.fail("entered without the lock")
    simulate_windows(monkeypatch, FlockMsvcrt(code))
    assert runs.RunLock(tmp_path).holder() is not None                      # a failed probe reads as held


@posix_only
def test_windows_a_path_replaced_after_the_open_fails_closed_and_is_untouched(tmp_path, monkeypatch):
    lock_file, other = tmp_path / runs.LOCK_NAME, tmp_path / "other"
    other.write_bytes(b"someone else's file")
    msvcrt_module, winapi = simulate_windows(monkeypatch, winapi=FakeWinapi(lambda: os.replace(other, lock_file)))
    before = open_fds()
    busy(tmp_path)                                                           # the handle is not the path's file
    assert lock_file.read_bytes() == b"someone else's file"                  # no metadata written, not truncated
    assert msvcrt_module.calls == [] and open_fds() == before                # never locked; nothing leaked


@posix_only
def test_windows_a_failed_descriptor_conversion_closes_the_handle(tmp_path, monkeypatch):
    msvcrt_module, winapi = simulate_windows(monkeypatch, FlockMsvcrt(osfhandle_error=OSError(errno.EMFILE, "x")))
    before = open_fds()
    busy(tmp_path)
    assert winapi.closed == winapi.created and len(winapi.created) == 1 and open_fds() == before


@posix_only
def test_windows_open_calls_create_file_with_plain_integer_nulls(tmp_path, monkeypatch):
    msvcrt_module, winapi = simulate_windows(monkeypatch)
    assert not hasattr(winapi, "NULL")
    with runs.RunLock(tmp_path):                                             # through CreateFile, then locked
        busy(tmp_path)
    assert len(winapi.created) == 2 and (tmp_path / runs.LOCK_NAME).is_file()     # the holder's and B's


def test_windows_verify_refuses_reparse_points_unknown_attributes_and_other_files(tmp_path, monkeypatch):
    path = tmp_path / "f"
    path.write_bytes(b"")
    real = os.lstat(path)

    def opened(attributes, ino=real.st_ino):
        return types.SimpleNamespace(st_file_attributes=attributes, st_dev=real.st_dev, st_ino=ino)
    if os.name != "nt":                                                      # POSIX lstat reports no attributes
        monkeypatch.setattr(runs, "_file_attributes", lambda st: getattr(st, "st_file_attributes", 0x80))
    runs._windows_verify(path, opened(0x80))                                 # the file at the path: accepted
    for bad in (opened(0x80 | stat.FILE_ATTRIBUTE_REPARSE_POINT), opened(stat.FILE_ATTRIBUTE_DIRECTORY),
                opened(0x80, ino=real.st_ino + 1)):                          # reparse point, directory, other file
        with pytest.raises(OSError):
            runs._windows_verify(path, bad)
    monkeypatch.setattr(runs, "_file_attributes", lambda st: None)          # attributes not reported: refused
    with pytest.raises(OSError):
        runs._windows_verify(path, opened(0x80))


def test_portable_metadata_io_ignores_the_position_and_finishes_short_writes(tmp_path, monkeypatch):
    fd = os.open(tmp_path / "m", os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
    try:
        data = json.dumps({"pid": 123, "started": 1.5}).encode()
        real = os.write
        monkeypatch.setattr(os, "write", lambda f, b: real(f, bytes(b[:3])))  # at most 3 bytes per call
        os.lseek(fd, runs.WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        runs._write_start(fd, data)
        monkeypatch.undo()
        os.lseek(fd, runs.WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        assert runs._read_start(fd) == data
        monkeypatch.setattr(os, "write", lambda f, b: 0)
        with pytest.raises(OSError):
            runs._write_start(fd, data)
    finally:
        os.close(fd)


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
