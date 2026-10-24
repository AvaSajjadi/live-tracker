"""Recorded REAL validation runs for the dashboard (new file).

A run is a directory written by scripts/validation_run_v1.py (Igor's ATTACK_RULE_V1 on the model B validation
scores, the monitor journal and the Contract v1 tools). The Replay and Investigation pages show such a run read
only: nothing is re-run, re-scored or re-classified, and the recorded order and ids are kept.

What load_run() can and cannot establish (VERIFICATION_LEVELS)
--------------------------------------------------------------
1. CONSISTENCY of the recorded execution (checked): the four files are re-derived against each other and against
   the tracked split manifest, the event_log validation, Contract v1's own rules and ATTACK_RULE_V1:
   - files are read through directory file descriptors with no-follow semantics (on Windows, handles opened
     relative to the held directory's handle, reparse points never traversed), each a regular file (fstat)
     within MAX_FILE_BYTES enforced while reading, read once and hashed from those bytes;
   - the summary is "verified" with every runner step ok, REAL labels, the test split not read, and its recorded
     digests equal the files;
   - the journal validates, is complete and verified_score_drop, freezes to exactly the stored snapshot, binds the
     score drop the summary names and the unchanged ATTACK_RULE_V1; every monitor count the summary reports
     (events, actions, end reasons, events per session, tracking closures, durations) is DERIVED from the journal
     and must agree;
   - the Contract v1 record: digest-checked attempts in the runner's fixed sequence; a query that is the rule's
     canonical query whose bouts are the journal's events; probe calls bound to the runner's probe rule (the first
     bout whose baseline starts at or after 0 s), its validation session, window, feature and quality policy, with
     a verdict re-decided from its recorded measurements by investigation_tools_v1.quality_reasons; comparisons
     with exactly Contract v1's result fields, whose execution-defining metadata (policies and their definitions,
     exclusion order, window definitions, baseline and gap, effect definition, wording, source) equals
     canonical_configuration() built from Contract v1's own registries, whose counts, contributing sessions and
     exclusions are reconstructed from the verified bouts and their exclusions, whose effect is the median of
     finite per-session paired differences, and whose gate status is the one those counts imply.
2. AUTHENTICATION (not available): no signature or independent registry binds these local files to the run that
   actually executed. A coordinated edit of the record and the summary with recomputed digests (for example other
   per-session paired differences with a matching median) cannot be detected without re-running the tools on the
   data. The pages therefore claim consistency only.
3. RECORDED ONLY (not re-derivable from the run's files): the event-level evaluation against the validation
   annotations, the annotated attack bouts per session, and the per-session tracking interruptions. They are
   checked for internal consistency and shown as recorded, unverified claims; incomplete annotation coverage is
   UNKNOWN, never zero, and no conclusion is built on them.
Anything inconsistent raises RunUnavailable with the reason; nothing is substituted (no synthetic fallback).
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import statistics
from dataclasses import dataclass
from pathlib import Path

from . import event_log as el
from . import investigation_tools_v1 as v1
from . import timebase as tb
from .evaluate import ATTACK_RULE_V1
from .investigation_data import load_manifest

__all__ = ["DISCLOSURES", "RUNS_DIR_ENV", "RunUnavailable", "VERIFICATION_LEVELS", "VerifiedRun", "comparison_rows",
           "event_rows", "exclusion_rows", "list_runs", "load_run", "per_session_rows", "provenance_rows", "runs_dir",
           "session_rows", "trace_rows"]

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR_ENV = "LIVE_TRACKER_VALIDATION_RUNS"
DEFAULT_RUNS_DIR = ROOT / "outputs" / "validation_run_v1"
DEFAULT_MANIFEST = ROOT / "splits" / "split_manifest.json"
SUMMARY, RECORD, JOURNAL, SNAPSHOT = ("run_summary.json", "contract_v1_record.json", "events.journal.jsonl",
                                      "journal_snapshot.json")
RUN_FILES = (SUMMARY, RECORD, JOURNAL, SNAPSHOT)
MAX_FILE_BYTES = 64 * 1024 * 1024
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
# the runner's identity (scripts/validation_run_v1.py)
STEPS = ("scores", "pose_cache", "approval", "monitor", "labels", "evaluation", "contract_v1")
RULE_ID, RUN_ID, INVESTIGATION_ID = "ATTACK_RULE_V1", "validation_run_v1", "validation_run_v1_attack"
PRE_S_ORDER = (2.0, 1.0, 5.0)
FEATURE, BASELINE_S, BASELINE_GAP_S, PROBE_PRE_S = "dist_center", 5.0, 1.0, 2.0
UNKNOWN = "unknown"

VERIFICATION_LEVELS = (
    ("Consistency of the recorded execution (checked)",
     "The four run files agree with each other, with the tracked split manifest, with the journal's own validation, "
     "with Contract v1's rules and with ATTACK_RULE_V1: file digests, journal and snapshot, the score drop, every "
     "monitor count (derived from the journal), the query and its bouts, the probe, and each comparison's counts, "
     "exclusions, contributing sessions, gate and effect aggregation."),
    ("Independent authentication (not available)",
     "No signature or independent registry binds these local files to the run that executed. A coordinated edit of "
     "the record and summary with recomputed digests (e.g. other per-session differences with a matching median) "
     "cannot be detected without re-running the tools on the data. Consistent is not authenticated."),
    ("Recorded only (unverified claims)",
     "The event-level evaluation against the annotations, the annotated attack bouts per session, the "
     "per-session tracking interruptions and the probe window's measurements (usable frames, flags, longest "
     "frozen / out-of-frame run) cannot be re-derived from the run's files. They are checked for internal "
     "consistency (the probe's verdict is re-decided from its measurements with Contract v1's own rule) and shown "
     "as recorded claims; missing annotation coverage is shown as unknown, never as zero."),
)

DISCLOSURES = (
    "REAL CalMS21 validation data. The held-out test split was not read by this run, and no page here can read it.",
    "Events are CLASSIFIER PREDICTIONS of attack (model B probabilities crossing the rule's thresholds), not "
    "annotated attacks. The annotations were used only to score the events (a recorded, unverified evaluation).",
    "ATTACK_RULE_V1 was chosen on these same 14 validation sessions, so the event metrics are optimistic.",
    "The model artifact is not identified: the score drop holds no model file, so model_artifact_sha256 is "
    "unavailable and the model label 'model_b' is unverified.",
    "'Consistent' means the recorded run's files agree with each other and with the rules they claim; it is not an "
    "independent authentication of the files and it does not mean the results are scientifically true.",
    "Event precision / recall would measure the classifier against annotations; the Contract v1 comparisons describe "
    "feature values before predicted bouts: associations only, no causal claim and no confidence interval.",
    "The Contract v1 calls were a FIXED, deterministic sequence chosen in the runner before any value was seen. "
    "No language model chose or interpreted them.",
)


class RunUnavailable(ValueError):
    """A run cannot be shown: missing, modified, inconsistent, unverified, synthetic or not a validation run."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _refuse(message):
    raise RunUnavailable(message)


def runs_dir(env=None):
    """The directory holding runs ($LIVE_TRACKER_VALIDATION_RUNS, else outputs/validation_run_v1)."""
    value = (os.environ if env is None else env).get(RUNS_DIR_ENV)
    return Path(value) if value else DEFAULT_RUNS_DIR


# ---- safe reading: directory descriptors, no-follow, fstat, bounded reads ----
# POSIX: openat(dir_fd) with O_NOFOLLOW / O_DIRECTORY (below). Windows has neither, so it has its own handle-based
# reader (further below) with the same guarantees; _read_run picks one, and a refusal never falls back to the other.

_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


def _platform_ok():
    if os.name == "nt":
        return _windows_opener() is not None
    return hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY") and os.open in os.supports_dir_fd


def _openat(name, flags, dir_fd=None):
    """os.open relative to a directory descriptor (one place, so tests can interpose a race)."""
    return os.open(name, flags, dir_fd=dir_fd)


def _open_dir(name, dir_fd=None):
    try:
        fd = _openat(name, _DIR_FLAGS, dir_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
            _refuse(f"{name!r} is not a real directory (symlinks are refused)")
        _refuse(f"{name!r} cannot be opened ({exc.strerror})")
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        _refuse(f"{name!r} is not a directory")
    return fd


def _read_bounded(fd, name):
    """Every byte of an open regular file, at most MAX_FILE_BYTES (enforced while reading)."""
    chunks, size = [], 0
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_FILE_BYTES:
            _refuse(f"{name} is larger than {MAX_FILE_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _read_at(dir_fd, name):
    """The bytes of regular file `name` directly inside dir_fd: no symlink, at most MAX_FILE_BYTES (enforced while
    reading)."""
    try:
        fd = _openat(name, _FILE_FLAGS, dir_fd)
    except FileNotFoundError:
        _refuse(f"{name} is missing")
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            _refuse(f"{name} is a symlink (refused)")
        _refuse(f"{name} cannot be opened ({exc.strerror})")
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            _refuse(f"{name} is not a regular file")
        return _read_bounded(fd, name)
    finally:
        os.close(fd)


def _read_run(root, name):
    if not _platform_ok():
        _refuse("this platform cannot open files without following symlinks; runs are not shown (fail closed)")
    if os.name == "nt":  # pragma: no cover - Windows
        return _read_run_windows(root, name, _windows_opener())
    root_fd = _open_dir(str(root))
    try:
        run_fd = _open_dir(name, root_fd)
        try:
            return {f: _read_at(run_fd, f) for f in RUN_FILES}
        finally:
            os.close(run_fd)
    finally:
        os.close(root_fd)


# ---- Windows: handle-based reading, the counterpart of openat(dir_fd) + O_NOFOLLOW ----
# The runs root is opened by its path with CreateFileW (_winapi.CreateFile); the run directory and each file are
# opened RELATIVE TO THE HANDLE of the directory already held, with NtCreateFile (OBJECT_ATTRIBUTES.RootDirectory),
# as openat(dir_fd) does on POSIX: once a directory is held, a later rename, replacement or link elsewhere in the
# path cannot redirect what is opened inside it. Every open uses FILE_OPEN_REPARSE_POINT (FILE_FLAG_OPEN_REPARSE_POINT
# for CreateFileW): a symlink, junction or any other reparse point is opened AS ITSELF, never traversed, and is then
# refused. No handle grants FILE_SHARE_DELETE (a held directory or file cannot be renamed, replaced or deleted while
# it is read) and a file does not grant FILE_SHARE_WRITE (it cannot change while it is read). Each handle is then
# checked before use: an ordinary directory / regular file by its attributes (a reparse point, or attributes that
# are not reported, are refused) and still the object at its path (device and file id). The bytes are read from that
# handle, bounded by MAX_FILE_BYTES, and hashed by load_run. Every handle is closed on every path. The opener is a
# parameter so this logic is testable anywhere (tests/test_validation_run_view.py uses a POSIX stand-in).

_GENERIC_READ, _FILE_GENERIC_READ = 0x80000000, 0x00120089
_FILE_SHARE_READ, _FILE_SHARE_WRITE = 0x1, 0x2
_OPEN_EXISTING = 3                                        # CreateFileW
_FILE_FLAG_BACKUP_SEMANTICS, _FILE_FLAG_OPEN_REPARSE_POINT = 0x02000000, 0x00200000
_FILE_OPEN = 1                                            # NtCreateFile disposition
_FILE_DIRECTORY_FILE, _FILE_NON_DIRECTORY_FILE = 0x1, 0x40
_FILE_SYNCHRONOUS_IO_NONALERT, _FILE_OPEN_REPARSE_POINT = 0x20, 0x00200000
_OBJ_CASE_INSENSITIVE = 0x40
_ERROR_DIRECTORY = 267                                    # "the directory name is invalid": not a directory


def _file_attributes(st):
    """Windows file attributes of a stat result (None when not reported: then the object is refused)."""
    return getattr(st, "st_file_attributes", None)


def _attributes(st):
    """The usable Windows attributes of a stat result: a 32-bit unsigned integer (0 included), else None."""
    a = _file_attributes(st)
    return a if isinstance(a, int) and not isinstance(a, bool) and 0 <= a <= 0xFFFFFFFF else None


def _file_identity(st):
    """(device, file id) of a stat result when both are usable, else None. A volume serial number or file id of 0
    is what Python reports when the filesystem gave none, so it never identifies an object."""
    pair = (getattr(st, "st_dev", None), getattr(st, "st_ino", None))
    if all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in pair):
        return pair
    return None


class _NtOpener:  # pragma: no cover - Windows (exercised by the Windows CI job)
    """root(path) and child(parent_fd, name, directory) -> a read-only CRT descriptor owning its handle."""

    def __init__(self):
        import _winapi
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class UnicodeString(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                        ("Buffer", wintypes.LPWSTR)]

        class ObjectAttributes(ctypes.Structure):
            _fields_ = [("Length", wintypes.ULONG), ("RootDirectory", wintypes.HANDLE),
                        ("ObjectName", ctypes.POINTER(UnicodeString)), ("Attributes", wintypes.ULONG),
                        ("SecurityDescriptor", ctypes.c_void_p), ("SecurityQualityOfService", ctypes.c_void_p)]

        class IoStatusBlock(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

        ntdll = ctypes.WinDLL("ntdll")
        create = ntdll.NtCreateFile
        create.restype = ctypes.c_long                      # NTSTATUS: negative is an error
        create.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.ULONG, ctypes.POINTER(ObjectAttributes),
                           ctypes.POINTER(IoStatusBlock), ctypes.c_void_p, wintypes.ULONG, wintypes.ULONG,
                           wintypes.ULONG, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG]
        to_dos = ntdll.RtlNtStatusToDosError
        to_dos.restype, to_dos.argtypes = wintypes.ULONG, [wintypes.ULONG]
        self._winapi, self._ctypes, self._msvcrt, self._wintypes = _winapi, ctypes, msvcrt, wintypes
        self._types, self._create, self._to_dos = (UnicodeString, ObjectAttributes, IoStatusBlock), create, to_dos

    def _fd(self, handle):
        try:
            return self._msvcrt.open_osfhandle(handle, os.O_RDONLY)     # the descriptor now owns the handle
        except BaseException:
            self._winapi.CloseHandle(handle)
            raise

    def root(self, path):
        handle = self._winapi.CreateFile(os.path.abspath(path), _GENERIC_READ, _FILE_SHARE_READ | _FILE_SHARE_WRITE,
                                         0, _OPEN_EXISTING, _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
                                         0)
        return self._fd(handle)

    def child(self, parent_fd, name, directory):
        ctypes, wintypes = self._ctypes, self._wintypes
        UnicodeString, ObjectAttributes, IoStatusBlock = self._types
        buffer = ctypes.create_unicode_buffer(name)
        size = 2 * len(name)
        object_name = UnicodeString(size, size + 2, ctypes.cast(buffer, wintypes.LPWSTR))
        attributes = ObjectAttributes(ctypes.sizeof(ObjectAttributes), self._msvcrt.get_osfhandle(parent_fd),
                                      ctypes.pointer(object_name), _OBJ_CASE_INSENSITIVE, None, None)
        handle, status_block = wintypes.HANDLE(), IoStatusBlock()
        share = _FILE_SHARE_READ | (_FILE_SHARE_WRITE if directory else 0)
        options = (_FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
                   | (_FILE_DIRECTORY_FILE if directory else _FILE_NON_DIRECTORY_FILE))
        status = self._create(ctypes.byref(handle), _FILE_GENERIC_READ, ctypes.byref(attributes),
                              ctypes.byref(status_block), None, 0, share, _FILE_OPEN, options, None, 0)
        if status < 0:
            raise ctypes.WinError(self._to_dos(status & 0xFFFFFFFF))
        return self._fd(handle.value)


_WINDOWS = []                                             # the one _NtOpener, once built (or None if unavailable)


def _windows_opener():
    if os.name != "nt":
        return None
    if not _WINDOWS:  # pragma: no cover - Windows
        try:
            _WINDOWS.append(_NtOpener())
        except (ImportError, OSError, AttributeError):
            _WINDOWS.append(None)                         # unavailable: _platform_ok() refuses every run
    return _WINDOWS[0]


def _windows_checked(path, label, directory, open_):
    """open_() -> fd, refused unless it is an ordinary directory / regular file, not a reparse point, and still the
    object at `path`. The descriptor is closed on any refusal."""
    try:
        fd = open_()
    except FileNotFoundError:
        _refuse(f"{label!r} cannot be opened (not found)" if directory else f"{label} is missing")
    except OSError as exc:
        if directory and getattr(exc, "winerror", None) == _ERROR_DIRECTORY:
            _refuse(f"{label!r} is not a real directory (symlinks are refused)")
        _refuse(f"{label!r} cannot be opened ({exc.strerror})" if directory else
                f"{label} cannot be opened ({exc.strerror})")
    try:
        # the opened handle: usable attributes and identity are required (never defaulted), then checked
        st = os.fstat(fd)
        attributes, identity = _attributes(st), _file_identity(st)
        if attributes is None or identity is None:
            _refuse(f"{label!r}: the opened object's Windows metadata (attributes, device and file id) is "
                    "incomplete (refused)")
        if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            _refuse(f"{label!r} is not a real directory (symlinks are refused)" if directory else
                    f"{label} is a symlink (refused)")
        is_dir_attribute = bool(attributes & stat.FILE_ATTRIBUTE_DIRECTORY)
        if directory and (not stat.S_ISDIR(st.st_mode) or not is_dir_attribute):
            _refuse(f"{label!r} is not a directory")
        if not directory and (not stat.S_ISREG(st.st_mode) or is_dir_attribute):
            _refuse(f"{label} is not a regular file")
        # the path, observed separately: the same requirements, then the same object
        try:
            at = os.lstat(path)
        except OSError:
            _refuse(f"{label!r} changed while it was being opened")
        path_attributes, path_identity = _attributes(at), _file_identity(at)
        if path_attributes is None or path_identity is None:
            _refuse(f"{label!r}: the path's Windows metadata (attributes, device and file id) is incomplete "
                    "(refused)")
        if stat.S_ISLNK(at.st_mode) or path_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT or \
                path_identity != identity:
            _refuse(f"{label!r} changed while it was being opened (symlinks are refused)")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_run_windows(root, name, opener):
    """The run's files read through `opener` (see above): every handle checked, bounded, and closed."""
    for part in (name, *RUN_FILES):
        if not part or part in (".", "..") or any(c in part for c in "\\/:"):
            _refuse(f"{part!r} is not a single plain name")
    root = Path(root)
    root_fd = _windows_checked(root, str(root), True, lambda: opener.root(str(root)))
    try:
        run_fd = _windows_checked(root / name, name, True, lambda: opener.child(root_fd, name, True))
        try:
            out = {}
            for f in RUN_FILES:
                fd = _windows_checked(root / name / f, f, False, lambda: opener.child(run_fd, f, False))
                try:
                    out[f] = _read_bounded(fd, f)
                finally:
                    os.close(fd)
            return out
        finally:
            os.close(run_fd)
    finally:
        os.close(root_fd)


def _link_like(path):
    """A symlink, or (Windows) any other reparse point such as a junction, or (Windows) an object whose attributes
    are not usable."""
    try:
        st = os.lstat(path)
    except OSError:
        return True
    if stat.S_ISLNK(st.st_mode):
        return True
    if os.name != "nt":
        return False
    attributes = _attributes(st)
    return attributes is None or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def list_runs(root=None):
    """Names of candidate run directories (plain names, real directories holding a run summary), sorted. Listing
    is not verification: load_run() decides."""
    root = Path(root) if root is not None else runs_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if NAME_RE.fullmatch(p.name) and p.is_dir() and not _link_like(p)
                  and (p / SUMMARY).is_file() and not _link_like(p / SUMMARY))


# ---- the run ----

@dataclass(frozen=True)
class VerifiedRun:
    name: str
    summary: dict
    record: dict
    events: list                       # §4 events from the journal, in firing order
    header: dict                       # the journal's run header (score-drop provenance and the approved rule)
    files: dict                        # file name -> SHA-256 of the bytes that were checked
    validation_sessions: tuple
    derived: dict                      # monitor counts DERIVED from the journal
    evaluation: dict                   # the recorded evaluation: {"status", "reason", "metrics", "per_session"}

    @property
    def query(self):
        return self.record["attempts"][0]["result"]

    @property
    def comparisons(self):
        return [a["result"] for a in self.record["attempts"] if a["tool"] == "compare_windows"]

    def probe(self, tool):
        return next((a["result"] for a in self.record["attempts"] if a["tool"] == tool), None)

    def annotated_bouts(self, session):
        """The recorded number of annotated attack bouts in a session, or UNKNOWN (never a default zero)."""
        per = self.evaluation.get("per_session")
        if per is None:
            return UNKNOWN
        return per.get(session, 0 if self.evaluation["status"] == "recorded_complete" else UNKNOWN)


def _json(name, data):
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        _refuse(f"{name} is not JSON")
    if not isinstance(obj, dict):
        _refuse(f"{name} is not a JSON object")
    return obj


def _journal_state(data):
    """The journal's records parsed from exactly the hashed bytes, through event_log's own validation."""
    if not data.endswith(b"\n"):
        _refuse("the journal ends with an incomplete line")
    records = []
    for n, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            _refuse(f"journal line {n} is not JSON")
        if not isinstance(rec, dict) or rec.get("record") not in el.RECORD_TYPES:
            _refuse(f"journal line {n} is not a journal record")
        records.append(rec)
    try:
        return el.JournalState.from_records(records)
    except el.JournalIntegrityError as exc:
        _refuse(f"the journal does not validate: {exc}")


def _identity(event_id, session, start_s, fired_s, end_s, end_reason):
    tail = str(event_id).rsplit("-", 1)[-1]
    if not tail.isdigit():
        _refuse(f"malformed event id {str(event_id)[:60]!r}")
    try:
        frames = tuple(tb.boundary_frame(t) for t in (start_s, fired_s, end_s))
    except (TypeError, tb.WindowError):
        _refuse(f"event {str(event_id)[:60]!r} has malformed times")
    return (int(tail), session, *frames, end_reason)


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _counter(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


# ---- the monitor: derived from the journal, the summary must agree ----

def _derive_monitor(events, mon, validation, frames_per_session):
    derived = {"n_events": len(events), "actions": _counter(e["action"] for e in events),
               "end_reasons": _counter(e["evidence"]["end_reason"] for e in events),
               "events_per_session": _counter(e["session"] for e in events)}
    derived["events_ended_by_tracking_loss"] = derived["end_reasons"].get("tracking_lost", 0)
    for k, v in derived.items():
        if mon.get(k) != v:
            _refuse(f"the summary's monitor {k} contradicts the journal")
    dur = [round(e["end_s"] - e["start_s"], 6) for e in events]
    rec = mon.get("duration_s")
    want = {"min": min(dur), "median": statistics.median(dur), "max": max(dur)} if dur else None
    if (rec is None) != (want is None) or (want and any(not _finite(rec.get(k)) or abs(rec[k] - want[k]) > 1e-6
                                                         for k in want)):
        _refuse("the summary's event durations contradict the journal")
    if mon.get("fps") != tb.FPS_INT:
        _refuse("the summary's frame rate is not the 30 fps timebase")
    # tracking interruptions are recorded only; they must at least be possible and agree with the journal's closures
    ti = mon.get("tracking_interruptions")
    if not isinstance(ti, dict) or set(ti) != validation:
        _refuse("the recorded tracking interruptions do not cover exactly the validation sessions")
    for sid, v in ti.items():
        lost, runs = (v or {}).get("tracking_lost_frames"), (v or {}).get("tracking_lost_runs")
        if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (lost, runs)) or \
                runs > lost or lost > frames_per_session[sid] or (lost > 0) != (runs > 0):
            _refuse(f"the recorded tracking interruptions of {sid} are impossible")
    for e in events:
        if e["evidence"]["end_reason"] == "tracking_lost" and ti[e["session"]]["tracking_lost_frames"] == 0:
            _refuse(f"event {e['event_id']} ended by tracking loss in a session recorded without any")
    return derived


# ---- the recorded evaluation (unverifiable here): internal consistency only ----

def _recorded_evaluation(ev, n_events, validation):
    """{"status": "recorded_complete" | "recorded_incomplete", "reason", "metrics", "per_session"}. Contradictory
    arithmetic refuses the run; incomplete annotation coverage leaves per-session values UNKNOWN."""
    ev = ev if isinstance(ev, dict) else {}
    m = ev.get("metrics")
    if not isinstance(m, dict):
        _refuse("the recorded evaluation has no metrics")
    n_ev, n_true, det = m.get("n_events"), m.get("n_true_bouts"), m.get("detected_true")
    p, r, f1 = m.get("event_precision"), m.get("event_recall"), m.get("event_f1")
    ints = all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (n_ev, n_true, det))
    if not ints or n_ev != n_events or det > n_true or not all(_finite(x) and 0 <= x <= 1 for x in (p, r, f1)):
        _refuse("the recorded evaluation does not match the journal's events or is malformed")
    matched = p * n_ev
    if abs(matched - round(matched)) > 1e-6 or (n_true and abs(r - det / n_true) > 1e-9) or \
            abs(f1 - (2 * p * r / (p + r) if p + r else 0.0)) > 1e-9:
        _refuse("the recorded evaluation is arithmetically inconsistent (precision, recall, F1 and counts)")
    rows = ev.get("true_bouts_per_session")
    per, reason, complete = None, None, False
    if not isinstance(rows, list) or not all(isinstance(x, dict) for x in rows):
        reason = "no per-session annotation counts were recorded"
    else:
        sids = [x.get("session") for x in rows]
        ok_rows = all(s in validation for s in sids) and len(set(sids)) == len(sids) and all(
            isinstance(x.get("true_bouts"), int) and isinstance(x.get("detected"), int) and
            0 <= x["detected"] <= x["true_bouts"] for x in rows)
        if not ok_rows:
            _refuse("the recorded per-session annotation counts are malformed")
        per = {x["session"]: x["true_bouts"] for x in rows}
        if sum(per.values()) == n_true and sum(x["detected"] for x in rows) == det:
            complete = True
        else:
            per, reason = None, ("the recorded per-session annotation counts do not account for every annotated "
                                 "bout; per-session annotation is unknown")
    return {"status": "recorded_complete" if complete else "recorded_incomplete", "reason": reason,
            "metrics": dict(m), "per_session": per}


# ---- the Contract v1 record ----

GATE = {"min_n_windows": v1.DEFAULT_MIN_N_WINDOWS, "min_n_sessions": v1.DEFAULT_MIN_N_SESSIONS,
        "overridden": False, "override_reason": None}
# the runner's fixed quality policy; compared against Contract v1's own registry entry
QUALITY = v1.QUALITY_POLICIES[v1.DEFAULT_QUALITY_POLICY]
# exact result schemas of the Contract v1 adapter (a missing or extra field is refused)
_BASE_KEYS = {"status", "data_source", "derived_from", "contract_version", "association_only"}
_COMPARE_KEYS = _BASE_KEYS | {"feature", "n_input", "n_used", "n_excluded", "n_sessions", "n_bouts", "n_windows",
                              "n_baseline_windows", "n_sessions_input", "excluded", "pre_s", "baseline_s",
                              "baseline_gap_s", "quality_policy", "quality_policy_definition", "overlap_policy",
                              "overlap_policy_definition", "exclusion_order", "evidence_gate", "windows"}
COMPARE_OK_KEYS = _COMPARE_KEYS | {"pre_mean", "baseline_mean", "mean_diff", "mean_diff_definition", "per_session",
                                   "wording"}
COMPARE_INSUFFICIENT_KEYS = _COMPARE_KEYS | {"reason", "n_available", "n_required", "failed_gate"}
QUALITY_KEYS = _BASE_KEYS | {"session", "t0_s", "t1_s", "usable_fraction", "n_frames", "n_usable_frames",
                             "flags_fired", "longest_veto_run_s", "verdict", "reason", "reasons", "quality_policy",
                             "quality_policy_definition"}
FEATURE_KEYS = _BASE_KEYS | {"session", "t0_s", "t1_s", "n_frames", "n_usable_frames", "summary",
                             "usable_value_rule", "causal"}


def canonical_configuration(data_source, derived_from):
    """Every execution-defining field a comparison of the runner must carry, from Contract v1's own registries
    and constants (never from the record)."""
    return {"data_source": data_source, "derived_from": derived_from, "contract_version": v1.CONTRACT_VERSION,
            "association_only": True, "feature": FEATURE, "baseline_s": BASELINE_S, "baseline_gap_s": BASELINE_GAP_S,
            "quality_policy": QUALITY.name, "quality_policy_definition": QUALITY.to_json(),
            "overlap_policy": v1.DEFAULT_OVERLAP_POLICY,
            "overlap_policy_definition": v1.OVERLAP_POLICIES[v1.DEFAULT_OVERLAP_POLICY].to_json(),
            "exclusion_order": list(v1.EXCLUSION_REASONS), "evidence_gate": GATE,
            "windows": dict(v1.WINDOW_DEFINITIONS)}


def _jsonable(x):
    return json.loads(json.dumps(x))


def check_quality_result(r, policy=QUALITY):
    """A recorded check_tracking_quality result must be the decision Contract v1 makes from its own recorded
    measurements (investigation_tools_v1.quality_reasons): the window's frame count, the usable-frame count and
    fraction, the longest veto run, and reasons / verdict / reason in the policy's order. The measurements
    themselves are recorded values (re-measuring them needs the pose data). Returns the reasons, or refuses."""
    if set(r) != QUALITY_KEYS:
        _refuse("the tracking-quality result does not have Contract v1's fields")
    if r["quality_policy"] != policy.name or r["quality_policy_definition"] != _jsonable(policy.to_json()):
        _refuse("the tracking-quality result's policy is not the approved policy definition")
    try:
        first, stop = tb.window_frames(r["t0_s"], r["t1_s"])
    except (tb.WindowError, TypeError):
        _refuse("the tracking-quality result has no valid window")
    n, n_ok = r["n_frames"], r["n_usable_frames"]
    if n != stop - first or not isinstance(n_ok, int) or isinstance(n_ok, bool) or not 0 <= n_ok <= n or \
            r["usable_fraction"] != n_ok / n:
        _refuse("the tracking-quality result's usable-frame counts are inconsistent")
    fired = r["flags_fired"]
    if not isinstance(fired, list) or len(set(fired)) != len(fired) or \
            fired != [f for f in v1.FLAG_NAMES if f in fired]:
        _refuse("the tracking-quality result's flags are malformed")
    longest = r["longest_veto_run_s"]
    if policy.veto_flags:
        if not _finite(longest) or longest < 0 or abs(longest * tb.FPS_INT - round(longest * tb.FPS_INT)) > 1e-9 or \
                round(longest * tb.FPS_INT) > n or (longest > 0 and not set(policy.veto_flags) & set(fired)):
            _refuse("the tracking-quality result's longest veto run is inconsistent")
        veto_frames = round(longest * tb.FPS_INT)
    else:
        if longest is not None:
            _refuse("the tracking-quality result reports a veto run under a policy without vetoes")
        veto_frames = 0
    reasons = v1.quality_reasons(policy, n, n_ok, veto_frames)
    if r["reasons"] != reasons or r["verdict"] != ("unreliable" if reasons else "reliable") or \
            r["reason"] != (reasons[0] if reasons else None):
        _refuse(f"the tracking-quality verdict contradicts its measurements under {policy.name} "
                f"(expected {'unreliable: ' + ', '.join(reasons) if reasons else 'reliable'})")
    return reasons


def _check_probe(attempts, bouts, validation, summary_c1, source, derived):
    """The probe calls must be the runner's probe: the first bout whose baseline starts at or after 0 s, its
    validation session, the window [start - 2 s, start), the feature and quality_v1; the summary must echo them."""
    probe = next((b for b in bouts if b["start_s"] - PROBE_PRE_S - BASELINE_GAP_S - BASELINE_S >= 0), None)
    calls = {a["tool"]: a for a in attempts if a["tool"] in ("check_tracking_quality", "get_window_features")}
    if probe is None:
        if calls or "probe_bout" in summary_c1:
            _refuse("probe calls were recorded although no bout qualifies as the probe")
        return
    win = {"session": probe["session"], "t0_s": probe["start_s"] - PROBE_PRE_S, "t1_s": probe["start_s"]}
    if probe["session"] not in validation or set(calls) != {"check_tracking_quality", "get_window_features"}:
        _refuse("the probe is not on a validation session or its calls are missing")
    q, f = calls["check_tracking_quality"], calls["get_window_features"]
    if q.get("request") != win or f.get("request") != {**win, "cols": [FEATURE]}:
        _refuse("the probe requests are not the runner's probe window")
    qr, fr = q.get("result") or {}, f.get("result") or {}
    for r in (qr, fr):
        if r.get("status") != "ok" or any(r.get(k) != v for k, v in win.items()) or r.get("association_only") \
                is not True or r.get("contract_version") != v1.CONTRACT_VERSION or \
                (r.get("data_source"), r.get("derived_from")) != (source, derived):
            _refuse("a probe result is not an executed result for the probe window")
    check_quality_result(qr)
    if set(fr) != FEATURE_KEYS or fr["usable_value_rule"] != v1.USABLE_VALUE_RULE or fr["causal"] != v1.CAUSAL_NOTE \
            or (fr["n_frames"], fr["n_usable_frames"]) != (qr["n_frames"], qr["n_usable_frames"]):
        _refuse("the probe's feature result does not have Contract v1's fields, rules or the window's counts")
    summ = (fr.get("summary") or {}).get(FEATURE)
    if set(fr.get("summary") or {}) != {FEATURE} or not isinstance(summ, dict) or \
            not isinstance(summ.get("n"), int) or summ["n"] < 0 or summ["n"] > fr.get("n_usable_frames", -1) or \
            (summ["n"] and not (all(_finite(summ.get(k)) for k in ("min", "mean", "max")) and
                                summ["min"] <= summ["mean"] <= summ["max"])) or \
            (not summ["n"] and any(summ.get(k) is not None for k in ("min", "mean", "max"))):
        _refuse("the probe's feature result is malformed or contradictory")
    if summary_c1.get("probe_bout") != {k: probe[k] for k in ("event_id", "session", "start_s")} or \
            summary_c1.get("check_tracking_quality") != qr or summary_c1.get("get_window_features") != fr:
        _refuse("the summary misreports the probe")


def _check_comparison(a, pre_s, q, bouts, summ, canonical):
    """One comparison, reconstructed from the verified bouts and its own exclusions (Contract v1's accounting)."""
    name = f"comparison {a.get('call')} (pre_s {pre_s:g})"
    r, req = a.get("result") or {}, a.get("request") or {}
    if req.get("query_run_id") != q["query_run_id"] or req.get("query") != q["query"] or req.get("bouts") != bouts \
            or req.get("pre_s") != pre_s or (req.get("feature"), req.get("baseline_s"), req.get("baseline_gap_s")) \
            != (FEATURE, BASELINE_S, BASELINE_GAP_S) or set(req) - {"query_run_id", "query", "bouts", "pre_s",
                                                                     "baseline_s", "baseline_gap_s", "feature"}:
        _refuse(f"{name} did not cite the verified query with the runner's settings")
    if r.get("status") not in ("ok", "insufficient_evidence") or \
            set(r) != (COMPARE_OK_KEYS if r["status"] == "ok" else COMPARE_INSUFFICIENT_KEYS):
        _refuse(f"{name} is not an executed comparison with Contract v1's fields")
    wrong = [k for k, v in {**canonical, "pre_s": pre_s}.items() if r.get(k) != v]
    if r["status"] == "ok":
        wrong += [k for k, v in (("mean_diff_definition", v1.MEAN_DIFF_DEFINITION), ("wording", v1.WORDING))
                  if r.get(k) != v]
    if wrong:
        _refuse(f"{name} contradicts the approved configuration in {', '.join(wrong)} (policies, exclusion order, "
                "windows and definitions must be Contract v1's own)")
    excluded = r.get("excluded")
    if not isinstance(excluded, list) or not all(isinstance(e, dict) for e in excluded):
        _refuse(f"{name} has malformed exclusions")
    keys = [(b["session"], b["start_s"]) for b in bouts]
    ex = [(e.get("session"), e.get("start_s")) for e in excluded]
    if len(set(ex)) != len(ex) or not set(ex) <= set(keys) or any(e.get("why") not in v1.EXCLUSION_REASONS
                                                                  for e in excluded):
        _refuse(f"{name} has exclusions that are not distinct bouts of the verified query with a known reason")
    used = _counter(sid for sid, st in keys if (sid, st) not in set(ex))
    counts = {"n_input": len(bouts), "n_used": sum(used.values()), "n_windows": sum(used.values()),
              "n_bouts": sum(used.values()), "n_baseline_windows": sum(used.values()), "n_excluded": len(ex),
              "n_sessions": len(used), "n_sessions_input": len({b["session"] for b in bouts})}
    wrong = [k for k, v in counts.items() if r.get(k) != v]
    if wrong:
        _refuse(f"{name}: {', '.join(wrong)} disagree with the verified bouts and exclusions")
    n_w, n_s = counts["n_windows"], counts["n_sessions"]
    met = n_w >= GATE["min_n_windows"] and n_s >= GATE["min_n_sessions"] and len(bouts) >= GATE["min_n_windows"]
    if r["status"] == "ok":
        if not met:
            _refuse(f"{name} is ok with {n_w} windows from {n_s} sessions, below the approved gate")
        per = r.get("per_session")
        if not isinstance(per, list) or not all(isinstance(p, dict) for p in per):
            _refuse(f"{name} is ok without per-session measurements")
        sids = [p.get("session") for p in per]
        if len(set(sids)) != len(sids) or {p.get("session"): p.get("n_bouts") for p in per} != used or \
                sids != sorted(sids):
            _refuse(f"{name}: the contributing sessions are not the used bouts of the verified query")
        diffs = [p.get("paired_diff") for p in per]
        if not all(_finite(d) for d in diffs) or r.get("mean_diff") != statistics.median(diffs) or \
                not _finite(r.get("pre_mean")) or not _finite(r.get("baseline_mean")):
            _refuse(f"{name}: the effect is not the median of finite per-session paired differences")
    else:
        if any(k in r for k in ("mean_diff", "per_session", "pre_mean", "baseline_mean")):
            _refuse(f"{name} carries an effect behind an unmet gate")
        unmet = ("too_few_bouts", "min_n_windows", len(bouts)) if len(bouts) < GATE["min_n_windows"] else \
            ("too_few_usable_windows", "min_n_windows", n_w) if n_w < GATE["min_n_windows"] else \
            ("too_few_usable_windows", "min_n_sessions", n_s) if n_s < GATE["min_n_sessions"] else None
        if unmet is None or (r.get("reason"), r.get("failed_gate"), r.get("n_available"), r.get("n_required")) != \
                (unmet[0], unmet[1], unmet[2], GATE[unmet[1]]):
            _refuse(f"{name}: its insufficient-evidence outcome is not the unmet condition")
    recount = _counter(e["why"] for e in excluded)
    if summ.get("exclusions_by_reason") != recount or any(summ.get(k) != r.get(k) for k in summ
                                                          if k != "exclusions_by_reason"):
        _refuse(f"the summary misreports {name}")


def _check_record(rec, c1, s, events, validation, synthetic):
    attempts = rec.get("attempts")
    if rec.get("investigation_id") != INVESTIGATION_ID or rec.get("contract_version") != v1.CONTRACT_VERSION or \
            rec.get("max_tool_calls") != v1.MAX_TOOL_CALLS or not isinstance(attempts, list) or not attempts or \
            not all(isinstance(a, dict) for a in attempts):
        _refuse("the Contract v1 record is not the runner's investigation")
    if rec.get("data_source") != ("synthetic" if synthetic else "classifier") or \
            c1.get("data_source") != rec.get("data_source"):
        _refuse("the Contract v1 record's data source contradicts the run's labelling")
    for n, a in enumerate(attempts, start=1):
        if a.get("call") != n or a.get("counted") is not True or a.get("investigation_id") != INVESTIGATION_ID or \
                a.get("result_sha256") != _sha(json.dumps(a.get("result"), sort_keys=True, default=str).encode()) \
                or not isinstance(a.get("result"), dict) or a.get("status") != a["result"].get("status"):
            _refuse(f"attempt {n} does not match its recorded digest, order or status")
    if rec.get("calls_used") != len(attempts) or rec["calls_used"] != c1.get("calls_used") or \
            c1.get("max_tool_calls") != v1.MAX_TOOL_CALLS:
        _refuse("the record's call count contradicts its attempts")
    q = attempts[0]["result"]
    query = {"session": sorted(validation), "behavior": ATTACK_RULE_V1["behavior"],
             "min_duration_s": float(ATTACK_RULE_V1["min_duration_s"]),
             "threshold": float(ATTACK_RULE_V1["ask_threshold"])}
    if (q.get("data_source"), q.get("derived_from")) != (("synthetic", "synthetic_scores") if synthetic else
                                                          ("classifier", "classifier_probabilities")) or \
            q.get("start_meaning") != v1.START_MEANING or q.get("latency_gap_s") is not None or \
            q.get("association_only") is not True or q.get("contract_version") != v1.CONTRACT_VERSION:
        _refuse("the recorded query's source or descriptors are not Contract v1's")
    if attempts[0].get("tool") != "query_events" or q.get("status") != "ok" or q.get("query") != query or \
            q.get("query_run_id") != (c1.get("query_events") or {}).get("query_run_id") or \
            (c1.get("query_events") or {}) != {k: v for k, v in q.items() if k != "bouts"}:
        _refuse("the recorded query is not the rule's canonical query on the validation split")
    bouts = q.get("bouts")
    if not isinstance(bouts, list) or q.get("n_bouts") != len(bouts) or \
            q.get("n_sessions") != len({b.get("session") for b in bouts}) or \
            q.get("n_sessions_searched") != len(validation):
        _refuse("the recorded query's counts contradict its bouts")
    if [_identity(b.get("event_id"), b.get("session"), b.get("start_s"), b.get("fired_s"), b.get("end_s"),
                  b.get("end_reason")) for b in bouts] != \
            [_identity(e["event_id"], e["session"], e["start_s"], e["evidence"]["fired_s"], e["end_s"],
                       e["evidence"]["end_reason"]) for e in events]:
        _refuse("the query's bouts are not the journal's events (identity or frame boundaries)")
    if c1.get("query_matches_monitor_bouts") is not True:
        _refuse("the run does not record that the query matched the monitor")
    probe = "probe_bout" in c1
    tools = [a.get("tool") for a in attempts]
    expected = ["query_events"] + (["check_tracking_quality", "get_window_features"] if probe else []) + \
        (["compare_windows"] * len(PRE_S_ORDER) if bouts else [])
    if tools != expected:
        _refuse(f"the call sequence {tools} is not the runner's fixed sequence")
    _check_probe(attempts, bouts, validation, c1, q.get("data_source"), q.get("derived_from"))
    comps = [a for a in attempts if a["tool"] == "compare_windows"]
    summaries = c1.get("compare_windows") or []
    if len(summaries) != len(comps) or not all(isinstance(x, dict) for x in summaries):
        _refuse("the summary and the record list different comparisons")
    canonical = _jsonable(canonical_configuration(q.get("data_source"), q.get("derived_from")))
    for a, summ, pre_s in zip(comps, summaries, PRE_S_ORDER):
        _check_comparison(a, pre_s, q, bouts, summ, canonical)
    outcome = [{"pre_s": a["result"]["pre_s"], "status": a["result"]["status"],
                "outcome": "evidence gate met (association only)" if a["result"]["status"] == "ok" else
                f"inconclusive: insufficient_evidence ({a['result']['reason']}, gate {a['result']['failed_gate']})"}
               for a in comps]
    if c1.get("scientific_outcome") != outcome:
        _refuse("the summary's scientific outcome contradicts the comparisons")


def load_run(name, root=None, manifest_path=DEFAULT_MANIFEST, allow_synthetic=False):
    """The run called `name` under the runs directory, checked for consistency (see VERIFICATION_LEVELS), or
    RunUnavailable. allow_synthetic is for tests of the checks themselves; the pages never pass it."""
    root = Path(root) if root is not None else runs_dir()
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        _refuse("a run is selected by its directory name only")
    raw = _read_run(root, name)
    files = {f: _sha(b) for f, b in raw.items()}
    s, rec = _json(SUMMARY, raw[SUMMARY]), _json(RECORD, raw[RECORD])

    # ---- the summary: a verified, real, validation-only run of the runner ----
    if s.get("status") != "verified":
        _refuse(f"the run's status is {s.get('status')!r}, not 'verified'")
    if s.get("steps") != [{"step": st, "status": "ok"} for st in STEPS]:
        _refuse("the run's steps are not the runner's steps, all ok")
    scores = s.get("scores") or {}
    synthetic = scores.get("synthetic")
    if synthetic is not False and not (allow_synthetic and synthetic is True):
        _refuse("the run is not labelled as real data (synthetic runs are never shown as real evidence)")
    labels = s.get("data_labels") or {}
    if not synthetic and (set(labels) != {"scores", "pose", "labels"} or
                          not all(str(v).startswith("REAL:") for v in labels.values())):
        _refuse("the run's data labels are not the runner's REAL labels")
    if not str(s.get("test_split", "")).startswith("not read"):
        _refuse("the run does not record that the test split was not read")
    if s.get("outputs") != {RECORD: files[RECORD], JOURNAL: files[JOURNAL], SNAPSHOT: files[SNAPSHOT]}:
        _refuse("the files are not the ones the run recorded (SHA-256 mismatch: modified, replaced or missing)")
    c1, mon = s.get("contract_v1") or {}, s.get("monitor") or {}
    if c1.get("record_sha256") != files[RECORD] or mon.get("journal_snapshot_sha256") != files[SNAPSHOT]:
        _refuse("the summary's record or snapshot digest does not match the file")

    # ---- split: the tracked manifest is the run's; validation sessions only, all of them ----
    manifest, manifest_sha = load_manifest(manifest_path)
    prov = scores.get("provenance") or {}
    validation, test = set(manifest["validation"]), set(manifest["test"])
    frames = scores.get("frames_per_session") or {}
    if prov.get("manifest_sha256") != manifest_sha or prov.get("split") != "validation":
        _refuse("the run used another split manifest or another split")
    if set(frames) != validation or validation & test or not all(
            isinstance(n, int) and n > 0 for n in frames.values()) or sum(frames.values()) != scores.get("n_frames"):
        _refuse("the run does not cover exactly the validation split")

    # ---- journal and snapshot (event_log's own validation) ----
    state = _journal_state(raw[JOURNAL])
    if state.run_ids() != [RUN_ID]:
        _refuse(f"the journal holds runs {state.run_ids()[:3]}, not exactly {RUN_ID!r}")
    jp = state.run_provenance(RUN_ID)
    if jp["status"] != "verified_score_drop" or jp["completion"] != "complete":
        _refuse(f"the journal run is {jp['status']} / {jp['completion']}, not verified_score_drop / complete")
    if mon.get("provenance_status") != jp["status"] or mon.get("completion") != jp["completion"]:
        _refuse("the summary misreports the journal's provenance status")
    try:
        frozen = el.freeze_run(state, RUN_ID)
        stored = el.load_snapshot(raw[SNAPSHOT], files[SNAPSHOT])
    except el.JournalIntegrityError as exc:
        _refuse(f"the journal snapshot does not verify: {exc}")
    if frozen.sha256 != stored.sha256 or frozen.canonical != raw[SNAPSHOT]:
        _refuse("the stored snapshot is not the frozen journal run")
    header = jp["header"]
    src = header["source"]
    for k in ("archive_sha256", "member_sha256", "rows_sha256", "manifest_sha256", "split"):
        if src.get(k) != prov.get(k):
            _refuse(f"the journal and the summary name different score drops ({k})")
    if (src.get("n_sessions"), src.get("n_frames")) != (scores.get("n_sessions"), scores.get("n_frames")) or \
            src.get("n_sessions") != len(validation):
        _refuse("the journal and the summary disagree on the sessions / frames scored")
    rule = {"rule_id": RULE_ID, "status": "approved", **ATTACK_RULE_V1}
    if header["rules"] != [rule] or {k: v for k, v in (s.get("rule") or {}).items() if k != "approval"} != \
            {"rule_id": RULE_ID, **ATTACK_RULE_V1, "source": "evaluate.ATTACK_RULE_V1"}:
        _refuse("the run's rule is not ATTACK_RULE_V1, unchanged and approved")
    events = state.events(RUN_ID)
    if len(events) != stored.n_events:
        _refuse("the journal and the snapshot disagree on the number of events")
    for e in events:
        if e["session"] not in validation or e["session"] in test:
            _refuse(f"event {e['event_id']} is not on a validation session")
    projected = [{"event_id": e["event_id"], "session": e["session"], "action": e["action"],
                  "start_s": e["start_s"], "fired_s": e["evidence"]["fired_s"], "end_s": e["end_s"],
                  "end_reason": e["evidence"]["end_reason"], "confidence": e["confidence"]} for e in events]
    recorded = mon.get("events")
    if not isinstance(recorded, list) or [{k: r.get(k) for k in p} for r, p in zip(recorded, projected)] != projected \
            or len(recorded) != len(projected):
        _refuse("the summary's events are not the journal's events (identity, order or values)")
    derived = _derive_monitor(events, mon, validation, frames)

    # ---- the Contract v1 record, then the recorded (unverifiable) evaluation ----
    _check_record(rec, c1, s, events, validation, synthetic)
    evaluation = _recorded_evaluation(s.get("evaluation"), len(events), validation)
    return VerifiedRun(name, s, rec, events, header, files, tuple(sorted(validation)), derived, evaluation)


# ---- rows for the pages (selection and renaming only) ----

def _short(session):
    return str(session).rsplit("/", 1)[-1]


def _annotation_label(run, session):
    n = run.annotated_bouts(session)
    return n if n == UNKNOWN else f"{n} (recorded, unverified)"


def session_rows(run):
    """One row per validation session: events, alerts / asks (derived from the journal), recorded tracking
    interruptions and recorded annotated attack bouts (or unknown)."""
    ti = run.summary["monitor"]["tracking_interruptions"]
    rows = []
    for sid in run.validation_sessions:
        evs = [e for e in run.events if e["session"] == sid]
        rows.append({"session": _short(sid), "events": len(evs), "alerts": sum(e["action"] == "alert" for e in evs),
                     "asks": sum(e["action"] == "ask" for e in evs),
                     "ended by tracking loss": sum(e["evidence"]["end_reason"] == "tracking_lost" for e in evs),
                     "tracking-lost frames (recorded)": ti[sid]["tracking_lost_frames"],
                     "tracking-lost runs (recorded)": ti[sid]["tracking_lost_runs"],
                     "annotated attack bouts": _annotation_label(run, sid)})
    return rows


def event_rows(run, session=None):
    """Recorded events (journal order, recorded ids), optionally of one session."""
    return [{"event id": e["event_id"], "session": _short(e["session"]), "action": e["action"],
             "start (s)": round(e["start_s"], 4), "fired (s)": round(e["evidence"]["fired_s"], 4),
             "end (s)": round(e["end_s"], 4), "duration (s)": round(e["end_s"] - e["start_s"], 4),
             "end reason": e["evidence"]["end_reason"], "confidence (mean score)": round(e["confidence"], 4)}
            for e in run.events if session is None or e["session"] == session]


def comparison_rows(run):
    out = []
    for i, c in enumerate(run.comparisons):
        out.append({"pre_s": c["pre_s"], "role": "primary" if i == 0 else f"follow-up {i}", "status": c["status"],
                    "used windows": c["n_used"], "of bouts": c["n_input"], "paired sessions": c["n_sessions"],
                    "gate": f">= {c['evidence_gate']['min_n_windows']} windows in >= "
                            f"{c['evidence_gate']['min_n_sessions']} sessions",
                    "median session-paired difference (pre - baseline)":
                        round(c["mean_diff"], 3) if c["status"] == "ok" else "none (insufficient evidence)"})
    return out


def exclusion_rows(run, pre_s):
    c = next(c for c in run.comparisons if c["pre_s"] == pre_s)
    counts = _counter(e["why"] for e in c["excluded"])
    return [{"reason": why, "excluded windows": n} for why, n in sorted(counts.items(), key=lambda x: -x[1])]


def per_session_rows(run, pre_s):
    c = next(c for c in run.comparisons if c["pre_s"] == pre_s)
    return [{"session": _short(p["session"]), "used windows": p["n_bouts"],
             "paired difference (pre - baseline)": round(p["paired_diff"], 3),
             "annotated attack bouts in session": _annotation_label(run, p["session"])}
            for p in c.get("per_session", [])]


def trace_rows(run):
    return [{"call": a["call"], "tool": a["tool"], "status": a["status"], "counted": a["counted"],
             "result sha256": a["result_sha256"][:16]} for a in run.record["attempts"]]


def provenance_rows(run):
    src, s = run.header["source"], run.summary
    return [{"item": k, "value": v} for k, v in (
        ("run directory", run.name),
        ("authentication", "NONE AVAILABLE: local files, consistency-checked only (not independently authenticated)"),
        ("data", "REAL CalMS21 validation split (test split not read)"),
        ("score archive", f"{src['archive_name']} · SHA-256 {src['archive_sha256']} (as recorded)"),
        ("score member", f"{src['member']} · SHA-256 {src['member_sha256']} (as recorded)"),
        ("verified score rows",
         f"SHA-256 {src['rows_sha256']} · {src['n_sessions']} sessions, {src['n_frames']} frames (as recorded)"),
        ("split manifest", f"SHA-256 {src['manifest_sha256']} (equals the tracked manifest)"),
        ("model", f"{src['model_label']} ({src['model_label_status']}); model artifact SHA-256: "
                  f"{src['model_artifact_sha256'] or 'UNAVAILABLE (unverified)'}"),
        ("pose cache", f"{s['pose_cache']['n_sessions']} sessions, {s['pose_cache']['n_frames']} frames, "
                       f"{s['pose_cache']['alignment']} (as recorded)"),
        ("journal", f"run {RUN_ID}: {s['monitor']['provenance_status']}, {s['monitor']['completion']} "
                    "(re-validated)"),
        ("journal snapshot", f"SHA-256 {run.files[SNAPSHOT]} (equals the frozen journal run)"),
        ("Contract v1 record", f"SHA-256 {run.files[RECORD]} · query {run.query['query_run_id']}"),
        ("evaluation", f"{run.evaluation['status']}" + (f": {run.evaluation['reason']}"
                                                         if run.evaluation["reason"] else "")),
        ("rule approval", s["rule"].get("approval", "") + " (as recorded)"))]
