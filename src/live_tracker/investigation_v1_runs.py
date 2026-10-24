"""Phase 3D: the researcher-facing backend of the Live Agent page (new file; no Streamlit here).

Everything runs through the reviewed code; nothing here is a new agent, tool, verifier or billing path:
- preflight and execution: scripts/investigation_v1_live.py (preflight_only, run), so the read-only preflight, the
  paid-run authorization (confirmed budget AND the exact authorization text), the key policy, budgets, the guarded
  OpenRouter adapter, the cost ledger, the output-directory protections and the saved-report checks are exactly
  the CLI's;
- reopening: load_saved() re-verifies every saved investigation before anything of it is shown, offline (no network
  and no model): verify_saved for Live Agent runs, verify_report_v1 plus the Markdown rendering and the recorded
  file digests for Phase 3B.1 offline runs. Historical outputs are only read;
- run5: reference_check() reports the fail-closed compatibility verdict of Phase 3B.1's check_query_against_run,
  unchanged, and separately explains which request parameters differ. A difference never weakens the verdict.

Two things are kept apart on purpose:
- autonomous investigation: the model chose its own valid Contract v1 parameters; a verified, finished report is a
  verified investigation of THOSE parameters, whether or not it reproduces run5;
- reference compatibility: equivalence with run5 needs the exact initial request and result. A mismatch says only
  that the investigation is not run5's analysis.

One investigation at a time: RunLock is an OS-level exclusive lock (fcntl.flock; msvcrt.locking on Windows) on a
stable file in the runs directory, held for the whole run by every browser session and process of the server alike,
and each preflight issues a one-use token that a paid run consumes.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import secrets
import stat
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import investigation_agent_v1 as av1
from . import investigation_live_v1 as lv1
from . import investigation_tools_v1 as v1

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR_ENV, OFFLINE_DIR_ENV = "LIVE_TRACKER_V1_LIVE_DIR", "LIVE_TRACKER_V1_OFFLINE_DIR"
DEFAULT_LIVE_DIR = ROOT / "outputs" / "investigation_v1_live"        # git-ignored
DEFAULT_OFFLINE_DIR = ROOT / "outputs" / "investigation_v1"          # git-ignored, Phase 3B.1
REFERENCE_RUN = "run5"
LOCK_NAME = ".investigation.lock"
OUTPUT_FILES = ("investigation_report.json", "investigation_report.md", "run.json")
MODELS = ("anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5", "anthropic/claude-opus-5.5")

_scripts = {}


def script(name):
    """A reviewed script module (scripts/<name>.py), loaded once, as the scripts load each other. Phase 3B.1's module
    is the one the live script loaded (one instance: its emulator, checker and RunRefused are the same objects)."""
    if name == "investigation_v1_validation":
        return script("investigation_v1_live").iv
    if name not in _scripts:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _scripts[name] = mod
    return _scripts[name]


def live_dir():
    return Path(os.environ.get(LIVE_DIR_ENV) or DEFAULT_LIVE_DIR)


def offline_dir():
    return Path(os.environ.get(OFFLINE_DIR_ENV) or DEFAULT_OFFLINE_DIR)


# ---- inputs and configuration ----

@dataclass(frozen=True)
class ValidationInputs:
    """The verified CalMS21 validation inputs of Phase 3B.1 (paths relative to the repository root, as run5's)."""
    label: str = "CalMS21 validation split: Igor's model B scores + verified pose cache (14 sessions, 94,305 frames)"
    archive: str = str(Path.home() / "Downloads" / "igor_scores_validation_model_b_repacked.zip")
    archive_sha256: str = "5c1a81e40ca12f00dea0ae511708b7d534d42e8b1feb43948af338b8b5c23427"
    member: str = "scores_validation_model_b.csv"
    cache_dir: str = "data/raw/calms21_validation"
    manifest: str = "splits/split_manifest.json"
    runs_dir: str | None = None                # recorded Phase 2 runs (default: validation_run_view's)
    allow_synthetic: bool = False              # tests only

    def problems(self):
        out = []
        if not Path(self.archive).expanduser().is_file():
            out.append(f"score archive not found: {Path(self.archive).name}")
        if not (Path(self.cache_dir) / "extraction_index.json").is_file():
            out.append(f"verified validation cache not found at {self.cache_dir}")
        if not Path(self.manifest).is_file():
            out.append(f"split manifest not found at {self.manifest}")
        if not self.allow_synthetic and Path.cwd().resolve() != ROOT:
            out.append("launch the app from the repository root (the data fingerprint, and so run5 compatibility, "
                       "depends on the relative cache path)")
        return out


def default_inputs():
    return ValidationInputs()


@dataclass(frozen=True)
class RunConfig:
    question: str = "Does the distance between the mice change before predicted attack bouts?"
    model: str = MODELS[0]
    max_model_turns: int = 8
    max_tool_calls: int = 8
    target_usd: float = lv1.MAX_TARGET_USD
    key_mode: str = "strict"                    # "strict" or "best-effort"
    acknowledged_key_limit_usd: float | None = None
    validation_run: str | None = REFERENCE_RUN  # the optional reference compatibility check

    def argv(self, inputs):
        a = ["--question", self.question, "--model", self.model, "--max-model-turns", str(self.max_model_turns),
             "--max-tool-calls", str(self.max_tool_calls), "--target-usd", repr(float(self.target_usd)),
             "--cache-dir", inputs.cache_dir, "--manifest", inputs.manifest, "--member", inputs.member,
             "--key-limit-mode", self.key_mode]
        if self.key_mode == "best-effort" and self.acknowledged_key_limit_usd is not None:
            a += ["--acknowledge-key-limit-usd", repr(float(self.acknowledged_key_limit_usd))]
        if inputs.allow_synthetic:
            a.append("--allow-synthetic")
        return a

    def digest(self, inputs):
        """Identifies the exact configuration a preflight approved (a changed setting needs a new preflight)."""
        return hashlib.sha256(json.dumps([asdict(self), asdict(inputs)], sort_keys=True).encode()).hexdigest()[:16]


def key_present(env=None):
    env = os.environ if env is None else env
    return bool((env.get("OPENROUTER_API_KEY") or "").strip())


def preflight(config, inputs, *, env=None, http_get=None):
    """(ready, report): the CLI's read-only --preflight-only (metadata GETs only; never a model request). A refused
    configuration gives (False, {"refused": message})."""
    lr = script("investigation_v1_live")
    try:
        args = lr.parser().parse_args(["--preflight-only", *config.argv(inputs)])
        code, text = lr.preflight_only(args, env=env, http_get=http_get)
    except ValueError as exc:
        return False, {"refused": str(exc)}
    return code == 0, json.loads(text)


# ---- one investigation at a time ----

class RunBusy(RuntimeError):
    """Another investigation holds the lock, or the lock cannot be taken safely (fail closed)."""


# ---- the lock primitives, per platform ----
# POSIX: fcntl.flock on the whole file (exclusive, non-blocking; contention is BlockingIOError), a shared lock as the
# holder probe, and pread/pwrite of the metadata.
# Windows (no fcntl): msvcrt.locking(LK_NBLCK) on ONE byte at WINDOWS_LOCK_OFFSET, far beyond the metadata. Windows
# byte-range locks are mandatory, so a lock on byte 0 would make the holder's pid unreadable to everyone else. Like
# flock, the lock belongs to one descriptor (two descriptors exclude each other, in one process or two) and is
# released when it is closed or its process ends. msvcrt has no shared lock, so the holder probe takes the same
# exclusive byte and releases it at once: a start racing a probe can be refused, never admitted twice. Contention
# (investigation_data.WINDOWS_LOCK_CONTENTION) is raised as BlockingIOError, like flock's; any other error
# propagates (fail closed). Windows has no pread/pwrite: the metadata is read and written after a seek, which is
# race-free because each descriptor belongs to one RunLock (or one probe) only.
# Opening: POSIX opens with O_NOFOLLOW (unchanged). Windows has no O_NOFOLLOW, and os.open would follow a symlink
# at the final component (creating a dangling link's target), so it opens with CreateFileW (_winapi.CreateFile,
# OPEN_ALWAYS | FILE_FLAG_OPEN_REPARSE_POINT): a reparse point (symlink, junction, ...) at the path is opened as
# itself, never traversed, so nothing is created or touched behind it. The handle then becomes a CRT descriptor
# (msvcrt.open_osfhandle; the handle is closed if that fails) and is checked before any lock or write: a regular
# file, not a reparse point, and still the very file at the path (device and file id), so a replaced path is
# refused. The msvcrt and _winapi modules are parameters so the Windows path is testable anywhere. There is no
# fallback: without fcntl or msvcrt, importing this module fails.

WINDOWS_LOCK_OFFSET = 1 << 20                       # 1 MiB: beyond any metadata (at most METADATA_BYTES are read)
METADATA_BYTES = 4096


class _Primitives:
    def __init__(self, name, lock, probe, unlock, read, write, open_, verify):
        self.name, self.lock, self.probe, self.unlock, self.read, self.write = name, lock, probe, unlock, read, write
        self.open, self.verify = open_, verify                 # verify(path, fstat) raises OSError when unacceptable


# CreateFileW arguments (Win32 ABI values)
GENERIC_READ, GENERIC_WRITE = 0x80000000, 0x40000000
FILE_SHARE_READ, FILE_SHARE_WRITE = 0x1, 0x2
OPEN_ALWAYS = 4
FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OPEN_REPARSE_POINT = 0x80, 0x00200000


def _file_attributes(st):
    """Windows file attributes of a stat result (None when not reported: then the file is refused)."""
    return getattr(st, "st_file_attributes", None)


def _windows_open(path, winapi, msvcrt_module):
    """A read-write descriptor of the file AT path: created if absent, and if path is a reparse point, that reparse
    point itself (refused by _windows_verify), never what it points to."""
    handle = winapi.CreateFile(str(path), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
                               0)                     # no security attributes (not inheritable), no template
    try:
        return msvcrt_module.open_osfhandle(handle, 0)      # binary; the descriptor now owns the handle
    except BaseException:
        winapi.CloseHandle(handle)
        raise


def _windows_verify(path, st):
    attributes = _file_attributes(st)
    if attributes is None or attributes & (stat.FILE_ATTRIBUTE_REPARSE_POINT | stat.FILE_ATTRIBUTE_DIRECTORY):
        raise OSError("the lock file is a reparse point or a directory (or its attributes are unknown)")
    at = os.lstat(path)
    if stat.S_ISLNK(at.st_mode) or (_file_attributes(at) or 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT or \
            (at.st_dev, at.st_ino) != (st.st_dev, st.st_ino):
        raise OSError("the lock path is a link, or not the file that was opened")


def _read_start(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    return os.read(fd, METADATA_BYTES)


def _write_start(fd, data):
    os.lseek(fd, 0, os.SEEK_SET)
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("the lock metadata could not be written")
        view = view[n:]


def _windows_primitives(msvcrt_module, winapi):
    from .investigation_data import WINDOWS_LOCK_CONTENTION

    def lock(fd):
        os.lseek(fd, WINDOWS_LOCK_OFFSET, os.SEEK_SET)          # locking() works from the current position
        try:
            msvcrt_module.locking(fd, msvcrt_module.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in WINDOWS_LOCK_CONTENTION:
                raise BlockingIOError(exc.errno, "the investigation lock is held") from None
            raise

    def unlock(fd):
        os.lseek(fd, WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt_module.locking(fd, msvcrt_module.LK_UNLCK, 1)
    return _Primitives("msvcrt.locking", lock, lock, unlock, _read_start, _write_start,
                       lambda path: _windows_open(path, winapi, msvcrt_module), _windows_verify)


if os.name == "nt":  # pragma: no cover - Windows
    import _winapi
    import msvcrt

    _LOCK = _windows_primitives(msvcrt, _winapi)
else:
    import fcntl

    _LOCK = _Primitives("fcntl.flock",
                        lambda fd: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB),
                        lambda fd: fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB),
                        lambda fd: fcntl.flock(fd, fcntl.LOCK_UN),
                        lambda fd: os.pread(fd, METADATA_BYTES, 0),
                        lambda fd, data: os.pwrite(fd, data, 0),
                        lambda path: os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600),
                        lambda path, st: None)


class RunLock:
    """One investigation at a time: an OS-level exclusive lock (fcntl.flock, LOCK_EX | LOCK_NB; on Windows
    msvcrt.locking of one byte, see the lock primitives above) on a stable lock file in the runs directory, held on
    one open file descriptor for the whole critical section. Ownership is the kernel's lock, never the file's
    content: the file is never unlinked (its inode is reused), its content (the holder's pid and start time) is
    informational only, and an empty, unreadable or malformed file neither proves nor disproves that a lock is held.
    The OS releases the lock when the descriptor is closed, including when the holding process dies, so a lock left
    on disk by a finished or killed process is simply free again. Any filesystem or locking error refuses the run
    (RunBusy). A RunLock object is held at most once at a time."""

    def __init__(self, root):
        self.path = Path(root) / LOCK_NAME
        self.fd = None

    def _open(self):
        """The lock file's descriptor: a regular file opened without following a symlink (on Windows, also not a
        reparse point and still the file at the path; see the lock primitives). Checked before any lock or write."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = _LOCK.open(self.path)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise OSError("the lock path is not a regular file")
            _LOCK.verify(self.path, st)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _write_metadata(self, fd):
        """Informational only (who holds it, since when); written after the lock is held."""
        os.ftruncate(fd, 0)
        _LOCK.write(fd, json.dumps({"pid": os.getpid(), "started": time.time()}).encode())

    def holder(self):
        """None when no one holds the lock; otherwise what is known of the holder ({} when its metadata is not
        readable). Probed with a non-blocking shared lock (on Windows, the exclusive byte, released at once), so it
        never takes the lock from a holder; any error is reported as held (fail closed)."""
        try:
            fd = self._open()
        except OSError:
            return {"error": "the lock file cannot be opened"}
        try:
            try:
                _LOCK.probe(fd)
            except OSError:                       # held exclusively elsewhere (or not lockable): busy
                try:
                    info = json.loads(_LOCK.read(fd) or b"{}")
                except (OSError, ValueError):
                    info = {}
                return info if isinstance(info, dict) else {}
            _LOCK.unlock(fd)
            return None
        finally:
            os.close(fd)

    def __enter__(self):
        if self.fd is not None:
            raise RunBusy("this lock is already held")
        try:
            fd = self._open()
        except OSError as exc:
            raise RunBusy(f"the investigation lock cannot be opened ({type(exc).__name__}); not started") from None
        try:
            _LOCK.lock(fd)
        except BlockingIOError:
            os.close(fd)
            raise RunBusy("another investigation is running; wait for it to finish") from None
        except OSError as exc:
            os.close(fd)
            raise RunBusy(f"the investigation lock cannot be taken ({type(exc).__name__}); not started") from None
        self.fd = fd                              # held from here: released only by __exit__ (or process exit)
        try:
            self._write_metadata(fd)
        except BaseException as exc:
            self._release()
            if isinstance(exc, OSError):
                raise RunBusy(f"the investigation lock cannot be recorded ({type(exc).__name__}); not started") \
                    from None
            raise
        return self

    def _release(self):
        fd, self.fd = self.fd, None
        if fd is None:
            return
        try:
            _LOCK.unlock(fd)
        finally:
            os.close(fd)

    def __exit__(self, *exc):
        self._release()
        return False


def new_run_name():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + secrets.token_hex(4)


# test seam: a factory of the httpx client the paid run uses (None: the SDK's own client)
HTTP_CLIENT_FACTORY = None
# test / demo seam: a factory of the emulated endpoint a FREE REHEARSAL uses (None: the CLI's own emulator). It is
# never used for a paid run (tests/live_agent_progress_demo.py sets a slowed emulator for watching live progress).
REHEARSAL_EMULATOR_FACTORY = None


def execute(config, inputs, *, confirm_budget_usd, authorization, paid=True, root=None, env=None, http_get=None,
            emulator=None, progress=None):
    """One investigation through the CLI's run() under the lock, into a new protected directory. paid=True needs the
    confirmed budget and the exact authorization text (refused by authorize_live before any request or file);
    paid=False is the free rehearsal over the emulated endpoint. Returns (run name, summary); raises the CLI's
    refusals (LiveRefused, RunRefused, ValueError, ...) or RunBusy. progress: an optional, presentation-only
    callback for the live view (the CLI's progress events, plus the lock); it never changes the run."""
    lr = script("investigation_v1_live")
    root = Path(root or live_dir())
    name = new_run_name()
    argv = [inputs.archive, "--archive-sha256", inputs.archive_sha256, "--out", str(root / name),
            *config.argv(inputs)]
    if config.validation_run:
        argv += ["--validation-run", config.validation_run] + (["--runs-dir", inputs.runs_dir] if inputs.runs_dir
                                                               else [])
    args = lr.parser().parse_args(argv + (["--live"] if paid else []))
    # passed as given: the CLI's authorize_live decides on them (a missing or wrong value is refused there)
    args.confirm_budget_usd, args.authorize_paid_run = (confirm_budget_usd, authorization) if paid else (None, None)
    if not paid and emulator is None and REHEARSAL_EMULATOR_FACTORY is not None:
        emulator = REHEARSAL_EMULATOR_FACTORY()
    sink = lr._Progress(progress)
    with RunLock(root):
        sink({"type": "stage", "stage": "lock", "state": "completed", "run": name})
        client = HTTP_CLIENT_FACTORY() if (paid and HTTP_CLIENT_FACTORY) else None
        summary = lr.run(args, env=env, http_get=http_get, http_client=client, emulator=emulator,
                         progress=sink if sink.sink is not None else None)
    return name, summary


# ---- reopening saved investigations (offline) ----

@dataclass
class SavedInvestigation:
    name: str
    path: Path
    kind: str                      # "live" (paid OpenRouter), "rehearsal" (emulated), "offline" (Phase 3B.1)
    summary: dict = field(default_factory=dict)
    doc: dict | None = None
    markdown: str | None = None
    problems: list = field(default_factory=list)

    @property
    def verified(self):
        return not self.problems and self.doc is not None


def list_saved():
    """[(label, path)] of saved investigation directories (Live Agent runs first, newest first)."""
    out = []
    for kind, root in (("Live Agent", live_dir()), ("Offline (Phase 3B.1)", offline_dir())):
        if root.is_dir():
            for d in sorted((p for p in root.iterdir() if p.is_dir() and (p / "run.json").is_file()), reverse=True):
                out.append((f"{kind}: {d.name}", d))
    return out


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_saved(path):
    """A saved investigation, re-verified offline before anything of it is used (problems empty: consistent; this is
    consistency, not authenticity). Nothing is written."""
    path = Path(path)
    saved = SavedInvestigation(path.name, path, "unknown")
    try:
        saved.summary = json.loads((path / "run.json").read_text(encoding="utf-8"))
        saved.doc = json.loads((path / "investigation_report.json").read_text(encoding="utf-8"))
        saved.markdown = (path / "investigation_report.md").read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        saved.doc = None
        saved.problems = [f"the saved investigation cannot be read ({type(exc).__name__}: "
                          f"{'a file is missing' if isinstance(exc, OSError) else 'malformed JSON'})"]
        return saved
    s = saved.summary
    if "key_policy" in s and "cost" in s:                                    # a Live Agent run (CLI run.json)
        lr = script("investigation_v1_live")
        saved.kind = "live" if s.get("mode") == lr.LIVE_MODE else "rehearsal"
        saved.problems = list(lr.verify_saved(path))
    elif (s.get("model") or {}).get("adapter") == script("investigation_v1_validation").OFFLINE_MODEL:
        saved.kind = "offline"
        problems = list(av1.verify_report_v1(saved.doc))
        if not problems and saved.markdown != av1.render_markdown_v1(saved.doc):
            problems.append("the Markdown is not the rendering of the report")
        for n, digest in (s.get("outputs") or {}).items():
            if not (path / n).is_file() or _sha(path / n) != digest:
                problems.append(f"{n} is not the file run.json recorded")
        if not s.get("outputs"):
            problems.append("run.json records no output digests")
        saved.problems = problems
    else:
        saved.problems = ["not a Stage 9 v1 investigation directory"]
    return saved


# ---- what a verified investigation shows ----

def autonomous_status(saved):
    """(kind, text) of the autonomous investigation itself, independent of run5."""
    if not saved.verified:
        return "unverified", "This saved investigation does not verify: nothing of it is shown as a finding."
    d = saved.doc
    if d["finished"]:
        who = {"offline": "Verified investigation by the offline deterministic investigator (no model)",
               "rehearsal": "Verified autonomous investigation (emulated rehearsal: no model, nothing spent)"}.get(
                   saved.kind, "Verified autonomous investigation")
        return "verified", f"{who}: finding `{d['finding']}` (consistency-checked; publication not authorized)."
    return "incomplete", (f"Verified but incomplete: the investigation stopped ({d['stopping_reason']}) before an "
                          "accepted finding; the evidence listed is partial and no finding was made.")


def attribution(saved):
    """Who chose the recorded actions, from the saved execution-mode metadata only (run.json's mode for Live Agent
    runs, its model adapter for Phase 3B.1 runs; never the directory name). A real model is named only for a paid
    run; anything unrecognised is attributed to no one in particular."""
    s, model = saved.summary, saved.summary.get("model") or {}
    if "key_policy" in s and "cost" in s:
        lr = script("investigation_v1_live")
        if s.get("mode") == lr.LIVE_MODE:
            name = model.get("model") or "the model"
            return {"mode": "paid_model", "label": f"paid model run (`{name}` via OpenRouter)",
                    "chooser": f"Chosen by the model (`{name}`)", "replies": "Model",
                    "intro": f"Each action was requested by the model `{name}` through paid OpenRouter requests; "
                             "results are the tools' recorded outputs. No reasoning beyond the model's recorded "
                             "text is shown."}
        if s.get("mode") == lr.DRY_MODE:
            return {"mode": "emulated", "label": "emulated rehearsal (no model, nothing spent)",
                    "chooser": "Scripted by the emulator (no model)", "replies": "Emulated",
                    "intro": "No model ran: the emulated endpoint answered each turn with the deterministic offline "
                             "investigator's next call, so the parameters are scripted, not a model's choices. "
                             "Results are the tools' recorded outputs."}
    if model.get("adapter") == script("investigation_v1_validation").OFFLINE_MODEL:
        return {"mode": "offline", "label": "offline deterministic investigation (no model)",
                "chooser": "Set by the deterministic procedure (no model)", "replies": "Investigator",
                "intro": "No model ran: Phase 3B.1's deterministic offline investigator chose each call by a fixed "
                         "procedure. Results are the tools' recorded outputs."}
    return {"mode": "unknown", "label": "execution mode not recorded", "chooser": "Recorded parameters",
            "replies": "Recorded", "intro": "The execution mode is not recorded, so no action is attributed to a "
                                            "model. Results are the tools' recorded outputs."}


def _compact(request):
    out = {}
    for k, v in request.items():
        if k in ("bouts", "query"):
            continue
        out[k] = f"{len(v)} sessions" if k == "session" and isinstance(v, list) else v
    return out


def _initial_query(evidence_or_attempts):
    return next((e for e in evidence_or_attempts if e.get("tool") == "query_events" and e.get("status") == "ok"),
                None)


def reference_check(saved, run_name=None, inputs=None):
    """The reference compatibility check against a recorded run (default run5), separately from the investigation:
    the verdict is Phase 3B.1's fail-closed check_query_against_run, unchanged; the rows explain which initial-query
    parameters differ (as sent, and as Contract v1 applied them)."""
    iv = script("investigation_v1_validation")
    vrv, run_name = iv.vrv, run_name or REFERENCE_RUN
    out = {"reference": run_name, "recorded": None, "available": False, "equivalent": False, "verdict": None,
           "differences": [], "rows": []}
    step = next((x for x in saved.summary.get("steps", []) if x.get("step") == "compatibility"), None)
    if step is not None:
        out["recorded"] = {"status": step["status"], "message": step.get("message")}
    if not saved.verified:
        out["verdict"] = "not checked: the investigation does not verify"
        return out
    inputs = inputs or default_inputs()
    try:                                       # loaded as check_validation_run loads it (manifest, synthetic label)
        run = vrv.load_run(run_name, root=inputs.runs_dir, manifest_path=inputs.manifest,
                           allow_synthetic=saved.doc["data"]["data_source"] == "synthetic")
    except (vrv.RunUnavailable, OSError, ValueError) as exc:
        out["verdict"] = f"reference run {run_name!r} unavailable ({type(exc).__name__})"
        return out
    out["available"] = True
    try:
        compat = iv.check_query_against_run(saved.doc, run)
        out.update(equivalent=True, verdict="equivalent: the initial request and result are exactly the reference's",
                   identical_results=len(compat["identical_results"]), unmatched=len(compat["unmatched"]))
    except iv.RunRefused as exc:
        out["verdict"] = f"not equivalent: {exc}"
    mine = _initial_query(saved.doc["evidence"])
    theirs = _initial_query(run.record["attempts"])
    if mine is None or theirs is None:
        out["explanation"] = "No executed initial query_events to compare."
        return out
    sent_a, sent_b = _compact(mine["request"]), _compact(theirs["request"])
    eff_a, eff_b = _compact(mine["result"].get("query", {})), _compact(theirs["result"].get("query", {}))
    for k in sorted(set(sent_a) | set(sent_b) | set(eff_a) | set(eff_b)):
        same_sent = sent_a.get(k, "not sent") == sent_b.get(k, "not sent")
        same_eff = eff_a.get(k) == eff_b.get(k)
        if k == "session":
            same_sent = same_eff = mine["request"].get("session") == theirs["request"].get("session")
        out["rows"].append({"parameter": k, "investigation (sent)": str(sent_a.get(k, "not sent")),
                            "investigation (applied)": str(eff_a.get(k, "")),
                            f"{run_name} (sent)": str(sent_b.get(k, "not sent")),
                            f"{run_name} (applied)": str(eff_b.get(k, "")),
                            "same": same_sent and same_eff})
        if not (same_sent and same_eff):
            out["differences"].append(k)
    if mine["result"].get("n_bouts") != theirs["result"].get("n_bouts"):
        out["rows"].append({"parameter": "n_bouts (result)", "investigation (sent)": "",
                            "investigation (applied)": str(mine["result"].get("n_bouts")),
                            f"{run_name} (sent)": "", f"{run_name} (applied)": str(theirs["result"].get("n_bouts")),
                            "same": False})
    if out["differences"]:
        out["explanation"] = (f"The model chose different query parameters ({', '.join(out['differences'])}), so its "
                              f"bouts and evidence are not {run_name}'s and no equivalence is claimed. Exact request "
                              "and result matching is required. This does not invalidate the autonomous "
                              "investigation, which is a verified analysis of its own parameters.")
    elif not out["equivalent"]:
        out["explanation"] = (f"The initial request matches {run_name}'s, but a result differs; no equivalence is "
                              "claimed.")
    else:
        out["explanation"] = f"The investigation's initial query reproduces {run_name} exactly."
    return out


def trace_rows(doc):
    rows = []
    for t in doc["trace"]:
        if t["kind"] == "model_turn":
            detail = ", ".join(c.get("name", "") for c in t.get("tool_calls") or []) or t.get("outcome", "")
        else:
            detail = json.dumps(t.get("arguments"), sort_keys=True)[:240] if "arguments" in t else ""
        rows.append({"step": t["step"], "kind": t["kind"], "tool": t.get("tool") or "",
                     "outcome": str(t.get("refused") or t.get("outcome") or t.get("stopping_reason")
                                    or t.get("reason") or ""),
                     "evidence_id": t.get("evidence_id") or "", "detail": detail,
                     "tokens in/out": "" if t["kind"] != "model_turn" else
                     f"{t.get('input_tokens')}/{t.get('output_tokens')}"})
    return rows


def evidence_rows(doc):
    return [{"evidence_id": e["evidence_id"], "tool": e["tool"], "status": e["status"],
             "call": e["provenance"]["call"], "request": json.dumps(_compact(e["request"]), sort_keys=True)[:300]}
            for e in doc["evidence"]]


def observation_rows(doc):
    return [{"evidence_id": o["evidence_id"], "field": o["field"], "value": json.dumps(o["value"])}
            for o in doc["observations"]]


KIND_LABELS = {"live": "paid OpenRouter investigation", "rehearsal": "emulated rehearsal: no model, no API spending",
               "offline": "offline deterministic investigator: no model"}


def ledger_rows(summary, simulated=False):
    """The recorded ledger, one row per request; simulated=True (a rehearsal) labels the costs as emulated."""
    reqs = ((summary.get("cost") or {}).get("ledger") or {}).get("requests") or []
    cost = "simulated cost (USD, not spent)" if simulated else "cost (USD)"
    return [{"request": n, "model requested": r.get("model_requested"), "model served": r.get("model_served"),
             "input tokens": r.get("input_tokens"), "output tokens": r.get("output_tokens"),
             cost: r.get("cost_usd"), "cost status": r.get("cost_status"),
             "reply used": bool(r.get("reply_used")), "withheld": r.get("withheld") or r.get("error") or ""}
            for n, r in enumerate(reqs, start=1)]


def provenance_rows(saved):
    d, s = saved.doc, saved.summary
    ev = d["evidence"][0]["provenance"] if d["evidence"] else {}
    scores = (s.get("inputs") or {}).get("scores") or {}
    prov = scores.get("provenance") or {}
    rows = [("kind", KIND_LABELS.get(saved.kind, saved.kind)),
            ("directory", saved.name), ("recorded status", s.get("status")),
            ("model", (s.get("model") or {}).get("model") or (s.get("model") or {}).get("adapter")),
            ("language model", (s.get("model") or {}).get("language_model")),
            ("data source", f"{d['data']['data_source']} ({d['data']['derived_from']})"),
            ("data fingerprint", d["data"]["fingerprint"]), ("sessions", len(d["data"]["sessions"])),
            ("split", ev.get("split")), ("contract", ev.get("contract_version")),
            ("investigation id", d["runner_record"]["investigation_id"]),
            ("findings format", d["findings_format"]), ("report fingerprint", d["fingerprint"]),
            ("finish fingerprint", d["finish_fingerprint"]), ("publication", d["publication"]),
            ("score archive SHA-256", prov.get("archive_sha256")), ("score rows SHA-256", scores.get("rows_sha256")),
            ("test split", s.get("test_split"))]
    if "key_policy" in s:
        rows.append(("key-limit mode", s["key_policy"]["mode"]))
    return [{"field": k, "value": "" if v is None else str(v)} for k, v in rows]


def cost_view(saved):
    """The recorded cost summary (Live Agent runs) or None; values as recorded and verified by verify_saved."""
    c = saved.summary.get("cost")
    return None if c is None else {k: copy.deepcopy(v) for k, v in c.items() if k != "ledger"}


# ---- UI-1: the scientific overview and the recorded agent activity (read from the verified report only) ----

FEATURE_MEANINGS = {                                  # features.py's own definitions of the pair features
    "dist_center": "center distance between the two mice",
    "dist_rate": "rate of change of the center distance (negative = approaching)",
    "rel_speed": "magnitude of the difference of the two mice's velocities",
    "close_time_s": "seconds of the trailing window with the mice closer than the close-distance threshold",
    "speed_corr_w": "trailing correlation of the two mice's speeds",
}
FINDING_HEADLINES = {
    "effect_described": "A descriptive association was measured",
    "insufficient_evidence": "Not enough usable evidence for an effect",
    "inconclusive": "Inconclusive",
    "incomplete": "Stopped before a finding",
}
TOOL_TITLES = {
    "query_events": ("Find predicted behavior bouts",
                     "Threshold crossings of the classifier's scores, kept when they last long enough."),
    "check_tracking_quality": ("Check tracking quality of one window",
                               "Contract v1's fixed quality policy (quality_v1) on one session window."),
    "get_window_features": ("Measure pair features in one window",
                            "Mean, min, max and count of pair features over one session window."),
    "compare_windows": ("Compare pre-bout windows with baselines",
                        "Session-paired comparison of a feature before each bout and in an earlier baseline, with "
                        "Contract v1's quality, overlap and evidence-gate policies."),
    av1.FINISH: ("Submit the finding", "A structured finding citing fields of this run's results; no prose."),
}


def _unit(feature):
    from .investigation_compare import UNITS
    return UNITS.get(feature, "")


def _evidence(doc):
    return {e["evidence_id"]: e for e in doc["evidence"]}


def _fmt(x, digits=4):
    if isinstance(x, float):
        return f"{x:.{digits}g}" if abs(x) < 1e4 else f"{x:.0f}"
    return str(x)


def overview(saved, reference=None):
    """The researcher-facing summary of a VERIFIED saved investigation: every value is read from its report or its
    accepted evidence (None when the report does not provide it); nothing is computed, interpreted or inferred."""
    if not saved.verified:
        return None
    d, ev = saved.doc, _evidence(saved.doc)
    finding = d["finding"]
    primary = ev.get(d["primary_evidence_id"]) if d["primary_evidence_id"] else None
    comp_ev = primary if primary and primary["tool"] == "compare_windows" else None
    if comp_ev is None:                                        # incomplete or inconclusive: the last comparison
        comp_ev = next((e for e in reversed(d["evidence"]) if e["tool"] == "compare_windows"), None)
    queries = [e for e in d["evidence"] if e["tool"] == "query_events"]
    query_ev = next((e for e in queries if comp_ev is not None and
                     e["result"].get("query_run_id") == comp_ev["request"].get("query_run_id")),
                    queries[0] if queries else None)               # the query the comparison used, else the first
    comp = comp_ev["result"] if comp_ev else None
    query = query_ev["result"] if query_ev else None
    out = {"question": d["question"], "verified": True, "finding": finding,
           "headline": FINDING_HEADLINES.get(finding, finding), "finding_text": av1.FINDING_TEXT.get(finding, ""),
           "stopping_reason": d["stopping_reason"], "finished": d["finished"], "effect": None, "comparison": None,
           "query": None, "spending": spending(saved), "primary_evidence_id": d["primary_evidence_id"]}
    if comp is not None:
        feature = comp.get("feature")
        out["comparison"] = {
            "status": comp.get("status"), "feature": feature, "unit": _unit(feature),
            "feature_meaning": FEATURE_MEANINGS.get(feature, ""), "pre_s": comp.get("pre_s"),
            "baseline_s": comp.get("baseline_s"), "baseline_gap_s": comp.get("baseline_gap_s"),
            "n_windows": comp.get("n_windows"), "n_sessions": comp.get("n_sessions"),
            "n_input": comp.get("n_input"), "n_used": comp.get("n_used"), "n_excluded": comp.get("n_excluded"),
            "reason": comp.get("reason"), "n_available": comp.get("n_available"),
            "n_required": comp.get("n_required"), "quality_policy": comp.get("quality_policy"),
            "overlap_policy": comp.get("overlap_policy"), "evidence_gate": comp.get("evidence_gate"),
            # the result's own definitions; Contract v1's when the result does not repeat them (e.g. insufficient)
            "windows": comp.get("windows") or dict(v1.WINDOW_DEFINITIONS),
            "mean_diff_definition": comp.get("mean_diff_definition") or v1.MEAN_DIFF_DEFINITION,
            "wording": comp.get("wording")}
        if finding == "effect_described" and comp.get("status") == "ok" and "mean_diff" in comp:
            out["effect"] = {"value": comp["mean_diff"], "unit": _unit(feature), "feature": feature,
                             "sign": "negative" if comp["mean_diff"] < 0 else "positive" if comp["mean_diff"] > 0
                             else "zero"}
    if query is not None:
        q = query.get("query") or {}
        out["query"] = {"behavior": query.get("behavior"), "threshold": query.get("threshold"),
                        "threshold_sent": "threshold" in query_ev["request"],
                        "min_duration_s": query.get("min_duration_s"), "n_bouts": query.get("n_bouts"),
                        "n_sessions": query.get("n_sessions"), "n_sessions_searched": query.get("n_sessions_searched"),
                        "data_source": query.get("data_source"), "start_meaning": query.get("start_meaning"),
                        "sessions_requested": len(q.get("session") or [])}
    if reference is not None:
        out["reference"] = reference_summary(reference)
    return out


def spending(saved):
    """Actual provider-reported spending (paid runs), $0 actual with a separate simulated cost (rehearsals), or $0
    with no model (offline)."""
    c = saved.summary.get("cost")
    if saved.kind == "live" and c is not None:
        return {"actual_usd": c["actual_spend_usd"], "complete": c["actual_spend_complete"], "simulated_usd": None,
                "basis": "provider-reported (OpenRouter usage.cost)"}
    if saved.kind == "rehearsal" and c is not None:
        return {"actual_usd": 0.0, "complete": True, "simulated_usd": c["actual_spend_usd"],
                "basis": "emulated endpoint: simulated cost, nothing spent"}
    return {"actual_usd": 0.0, "complete": True, "simulated_usd": None, "basis": "no model was called"}


def reference_summary(ref):
    """The reference check's verdict and the threshold each side applied (from its rows), for the status panel."""
    rows = {r["parameter"]: r for r in ref.get("rows", [])}
    th = rows.get("threshold")
    name = ref["reference"]
    return {"reference": name, "available": ref["available"], "equivalent": ref["equivalent"],
            "verdict": ref["verdict"], "differences": list(ref.get("differences", [])),
            "agent_threshold": th["investigation (applied)"] if th else None,
            "agent_threshold_sent": th["investigation (sent)"] != "not sent" if th else None,
            "reference_threshold": th[f"{name} (applied)"] if th else None}


def _arguments(name, args, step_of):
    """The model's arguments as (parameter, value) pairs; an evidence reference is shown as the step it came from."""
    if not isinstance(args, dict):
        return [("arguments", str(args)[:200])]
    out = []
    for k, v in args.items():
        if k == "query_from" and isinstance(v, dict):
            ref = v.get("evidence_id")
            out.append(("bouts from", f"step {step_of[ref]}'s query" if ref in step_of else f"{str(ref)[:40]}"))
        elif k == "observations" and isinstance(v, list):
            out.append(("cites", ", ".join(f"{o.get('field')} (step {step_of.get(o.get('evidence_id'), '?')})"
                                           for o in v if isinstance(o, dict))))
        elif k == "primary_evidence_id":
            out.append(("primary evidence", f"step {step_of[v]}" if v in step_of else str(v)))
        elif k == "sessions" and isinstance(v, list):
            out.append(("sessions", f"{len(v)} sessions"))
        elif k == "session" and isinstance(v, str):
            out.append(("session", v.rsplit("/", 1)[-1]))
        else:
            out.append((k, _fmt(v) if not isinstance(v, (list, dict)) else json.dumps(v)))
    if name == "query_events" and "threshold" not in args:
        out.append(("threshold", "not set (Contract v1's default applies)"))
    return out


def _results(tool, r):
    """The important recorded results of one accepted call, as (label, value) pairs."""
    if tool == "query_events":
        return [("bouts found", f"{r.get('n_bouts')} in {r.get('n_sessions')} of {r.get('n_sessions_searched')} "
                                f"sessions searched"), ("threshold applied", _fmt(r.get("threshold"))),
                ("minimum duration", f"{_fmt(r.get('min_duration_s'))} s")]
    if tool == "check_tracking_quality":
        return [("verdict", r.get("verdict")), ("usable fraction", _fmt(r.get("usable_fraction"))),
                ("frames", f"{r.get('n_usable_frames')} usable of {r.get('n_frames')}"),
                ("policy", r.get("quality_policy"))]
    if tool == "get_window_features":
        out = []
        for f, s in (r.get("summary") or {}).items():
            u = _unit(f)
            out.append((f, f"mean {_fmt(s.get('mean'))} {u}, min {_fmt(s.get('min'))}, max {_fmt(s.get('max'))}, "
                           f"n {s.get('n')}".replace("  ", " ")))
        return out
    if tool == "compare_windows":
        out = [("status", r.get("status"))]
        if r.get("status") == "ok":
            out += [("median session-paired difference", f"{_fmt(r.get('mean_diff'))} {_unit(r.get('feature'))}"),
                    ("windows used", f"{r.get('n_windows')} in {r.get('n_sessions')} sessions"),
                    ("excluded", f"{r.get('n_excluded')} of {r.get('n_input')} bouts")]
        else:
            out += [("reason", r.get("reason")), ("available / required",
                                                 f"{r.get('n_available')} / {r.get('n_required')}")]
        return out
    return []


def timeline(doc, replies="Model"):
    """The recorded investigation activity as visual steps, in trace order: every tool call and finish asked for
    (accepted with its evidence, or refused with the refusal), every reply error or budget stop, and the final stop.
    Built from the saved trace and accepted evidence only; no tool sequence is assumed. `replies` names what answered
    each turn (attribution()'s "replies": a model only for a paid run). A finish produces no measurement: its step is a
    finding accepted, never evidence."""
    ev = _evidence(doc)
    step_of, steps, note = {}, [], None
    for t in doc["trace"]:
        kind = t.get("kind")
        if kind == "model_turn":
            note = (t.get("text") or "").strip() or None
            if t.get("outcome") in ("malformed", "usage_unavailable", "over_budget"):
                steps.append({"kind": "event", "status": "problem", "title": f"{replies} reply not used",
                              "description": {"malformed": "The reply was not a usable tool call.",
                                              "usage_unavailable": "The reply reported no usable token usage.",
                                              "over_budget": "The reply arrived past a budget limit; none of it "
                                                             "was executed."}[t["outcome"]],
                              "details": {k: t.get(k) for k in ("outcome", "problem", "input_tokens",
                                                                "output_tokens")}, "trace_step": t["step"]})
            continue
        if kind in ("tool_call", "finish"):
            n = len(steps) + 1
            name = t.get("tool")
            title, description = TOOL_TITLES.get(name, (f"Unknown tool {str(name)[:40]}",
                                                        "Not a Contract v1 tool; refused."))
            step = {"kind": "action", "number": n, "tool": name, "title": title, "description": description,
                    "parameters": _arguments(name, t.get("arguments"), step_of), "trace_step": t["step"],
                    "model_note": note, "arguments": t.get("arguments"), "evidence_id": t.get("evidence_id")}
            if t.get("outcome") == "refused":
                step.update(status="refused", outcome=f"refused: {t.get('refused')}",
                            results=[("message", str(t.get("message"))[:400])])
            elif kind == "finish":
                step.update(status="finding_accepted", outcome="finding accepted (cites earlier evidence)",
                            results=[("finding", doc["finding"] if doc["finished"] else
                                      (t.get("arguments") or {}).get("finding"))])
            else:
                e = ev.get(t.get("evidence_id"))
                status = e["status"] if e else t.get("outcome")
                step.update(status="accepted" if status == "ok" else "accepted_insufficient"
                            if status == "insufficient_evidence" else "accepted_other",
                            outcome=f"accepted as evidence ({status})", results=_results(name, e["result"]) if e
                            else [])
                if e:
                    step_of[e["evidence_id"]] = n
            steps.append(step)
            note = None
        elif kind == "model_error":
            steps.append({"kind": "event", "status": "problem", "title": f"{replies} request stopped",
                          "description": f"The {replies.lower()} reply could not be used ({t.get('reason')}); the "
                                         "run stopped.",
                          "details": {"reason": t.get("reason")}, "trace_step": t["step"]})
        elif kind == "budget_exceeded":
            steps.append({"kind": "event", "status": "problem", "title": "Budget limit reached",
                          "description": f"Stopped by the budget ({t.get('stopping_reason')}); calls not executed: "
                                         f"{len(t.get('not_executed') or [])}.",
                          "details": {k: t.get(k) for k in ("stopping_reason", "not_executed")},
                          "trace_step": t["step"]})
        elif kind == "stop":
            reason = t.get("stopping_reason")
            finished = reason == "finished"
            steps.append({"kind": "end", "status": "finished" if finished else "incomplete",
                          "title": f"Finding: {doc['finding']}" if finished else "Ended without a finding",
                          "description": (av1.FINDING_TEXT.get(doc["finding"], "") if finished else
                                          f"The investigation stopped ({reason}) before an accepted finding; the "
                                          "evidence above is partial."),
                          "supporting": [{"step": step_of.get(o["evidence_id"]), "field": o["field"],
                                          "value": o["value"], "evidence_id": o["evidence_id"]}
                                         for o in doc["observations"]], "trace_step": t["step"]})
    return steps


# ---- UI-2: the recorded scientific evidence, as chart data (read from the report only) ----

EXCLUSION_TEXT = {
    "pre_window_overlaps_bout": "The pre-bout window overlaps another predicted bout of the same query",
    "window_before_session_start": "The baseline window would begin before the recording starts (before 0 s)",
    "pre_window_unreliable": "The pre-bout window failed the tracking-quality policy",
    "baseline_window_unreliable": "The baseline window failed the tracking-quality policy",
    "baseline_window_overlaps_bout": "The baseline window overlaps another predicted bout of the same query",
    "no_feature_values": "A window had no usable value of the feature",
}
EXCLUSION_LABELS = {                                    # short chart labels; EXCLUSION_TEXT is the meaning
    "pre_window_overlaps_bout": "Pre-bout window overlaps a bout",
    "window_before_session_start": "Baseline before recording start",
    "pre_window_unreliable": "Pre-bout window: poor tracking",
    "baseline_window_unreliable": "Baseline window: poor tracking",
    "baseline_window_overlaps_bout": "Baseline window overlaps a bout",
    "no_feature_values": "No feature values",
}
GATE_ORDER = (("min_n_windows", "n_windows", "Valid comparison windows"),
              ("min_n_sessions", "n_sessions", "Contributing sessions"))
GATE_REASONS = {"too_few_bouts": "fewer input bouts than the minimum number of windows (checked before measuring)",
                "too_few_usable_windows": "too few usable windows or sessions after the exclusions"}
FIRED_MEANING = ("fired_s is when the monitor fired: the bout had then lasted the minimum duration. It is recorded "
                 "but never used as the anchor.")


def _session_labels(sessions):
    """Short labels (the last path part) unless two sessions would share one."""
    short = {s: str(s).rsplit("/", 1)[-1] for s in sessions}
    return short if len(set(short.values())) == len(short) else {s: str(s) for s in sessions}


def comparison_choices(doc):
    """The accepted compare_windows evidence in recorded order: [{evidence_id, label, primary}]. Each is one call's
    result; none is merged with another."""
    comps = [e for e in doc["evidence"] if e["tool"] == "compare_windows"]
    out = []
    for i, e in enumerate(comps, 1):
        r, primary = e["result"], e["evidence_id"] == doc["primary_evidence_id"]
        out.append({"evidence_id": e["evidence_id"], "primary": primary,
                    "label": f"Comparison {i} of {len(comps)} (tool call {e['provenance'].get('call')}): "
                             f"{r.get('feature')}, pre_s {_fmt(r.get('pre_s'))} s, baseline_s "
                             f"{_fmt(r.get('baseline_s'))} s, gap {_fmt(r.get('baseline_gap_s'))} s: {r.get('status')}"
                             + (" (primary evidence)" if primary else "")})
    return out


def query_for(doc, comparison):
    """(query evidence, basis) of the bouts a comparison used: the query evidence its recorded call cited
    (query_from), accepted only if its query_run_id is the comparison's; otherwise the latest earlier query evidence
    with the comparison's query_run_id and query. (None, reason) when no recorded query matches."""
    ev, req = _evidence(doc), comparison["request"]
    call = next((t for t in doc["trace"] if t.get("kind") == "tool_call"
                 and t.get("evidence_id") == comparison["evidence_id"]), None)
    cited = ((call or {}).get("arguments") or {}).get("query_from") or {}
    q = ev.get(cited.get("evidence_id")) if isinstance(cited, dict) else None
    if q is not None and q["tool"] == "query_events" and q["result"].get("query_run_id") == req.get("query_run_id"):
        return q, "the query evidence this comparison's call cited"
    order = [e["evidence_id"] for e in doc["evidence"]]
    before = order[:order.index(comparison["evidence_id"])]
    matches = [ev[i] for i in before if ev[i]["tool"] == "query_events"
               and ev[i]["result"].get("query_run_id") == req.get("query_run_id")
               and ev[i]["result"].get("query") == req.get("query")]
    if matches:
        return matches[-1], "the earlier query evidence with this comparison's query_run_id and query"
    return None, "no recorded query evidence has this comparison's query_run_id"


def bout_view(query_ev):
    """The predicted bouts of one accepted query_events result, as recorded (duration = end_s - start_s, only where
    both are recorded numbers)."""
    r = query_ev["result"]
    bouts = r.get("bouts") or []
    searched = list((r.get("query") or {}).get("session") or [])
    labels = _session_labels(dict.fromkeys(searched + [b["session"] for b in bouts]))
    rows = []
    for b in bouts:
        s, e = b.get("start_s"), b.get("end_s")
        dur = e - s if isinstance(s, (int, float)) and isinstance(e, (int, float)) else None
        rows.append({"session": labels[b["session"]], "start_s": s, "fired_s": b.get("fired_s"), "end_s": e,
                     "duration_s": dur, "end_reason": b.get("end_reason"), "event_id": b.get("event_id")})
    per = {labels[s]: 0 for s in searched}
    for b in bouts:
        per[labels[b["session"]]] = per.get(labels[b["session"]], 0) + 1
    return {"evidence_id": query_ev["evidence_id"], "behavior": r.get("behavior"), "threshold": r.get("threshold"),
            "threshold_sent": "threshold" in query_ev["request"], "min_duration_s": r.get("min_duration_s"),
            "n_bouts": r.get("n_bouts"), "n_sessions": r.get("n_sessions"),
            "n_sessions_searched": r.get("n_sessions_searched"), "data_source": r.get("data_source"),
            "start_meaning": r.get("start_meaning"), "fired_meaning": FIRED_MEANING,
            "bouts": rows, "per_session": [{"session": s, "bouts": n} for s, n in per.items()],
            "durations_recorded": sum(x["duration_s"] is not None for x in rows)}


def evidence_view(doc, evidence_id):
    """Chart data for ONE accepted compare_windows result and the query evidence it used, read from the verified
    report only (nothing is recomputed: counts are tallies of the recorded lists, gate verdicts follow the recorded
    status and failed_gate in Contract v1's checking order)."""
    e = _evidence(doc)[evidence_id]
    if e["tool"] != "compare_windows":
        raise ValueError(f"{evidence_id} is not compare_windows evidence")
    r = e["result"]
    feature, ok = r.get("feature"), r.get("status") == "ok"
    per = (r.get("per_session") or []) if ok else []                # recorded only for an ok comparison
    labels = _session_labels([p["session"] for p in per])
    sessions = [{"session": labels[p["session"]], "full_session": p["session"], "n_bouts": p["n_bouts"],
                 "paired_diff": p["paired_diff"]} for p in per]
    order = list(r.get("exclusion_order") or v1.EXCLUSION_REASONS)
    excluded = r.get("excluded") or []
    tally = {}
    for x in excluded:
        tally[x["why"]] = tally.get(x["why"], 0) + 1
    order += [w for w in tally if w not in order]                  # (a reason outside the order: listed last)
    exclusions = [{"reason": w, "label": EXCLUSION_LABELS.get(w, w), "description": EXCLUSION_TEXT.get(w, w),
                   "bouts": tally.get(w, 0)} for w in order]
    gate, failed = r.get("evidence_gate") or {}, r.get("failed_gate")
    names = [g for g, _, _ in GATE_ORDER]
    gates = []
    for name, observed, label in GATE_ORDER:
        if ok:
            verdict = "met"
        elif r.get("status") == "insufficient_evidence" and failed in names:
            i, j = names.index(name), names.index(failed)
            verdict = "met" if i < j else "not met" if i == j else "not evaluated"
        else:
            verdict = "not recorded"
        row = {"gate": label, "field": name, "required": gate.get(name), "observed": r.get(observed),
               "verdict": verdict}
        if verdict == "not met":
            row.update(n_available=r.get("n_available"), n_required=r.get("n_required"))
        gates.append(row)
    query, basis = query_for(doc, e)
    return {"evidence_id": evidence_id, "status": r.get("status"), "primary": evidence_id == doc["primary_evidence_id"],
            "feature": feature, "unit": _unit(feature), "feature_meaning": FEATURE_MEANINGS.get(feature, ""),
            "pre_s": r.get("pre_s"), "baseline_s": r.get("baseline_s"), "baseline_gap_s": r.get("baseline_gap_s"),
            "mean_diff": r.get("mean_diff") if ok else None, "n_sessions": r.get("n_sessions"),
            "n_windows": r.get("n_windows"), "sessions": sessions,
            "n_input": r.get("n_input"), "n_used": r.get("n_used"), "n_excluded": r.get("n_excluded"),
            "exclusions": exclusions, "exclusions_consistent": sum(tally.values()) == r.get("n_excluded")
            and (r.get("n_used") or 0) + (r.get("n_excluded") or 0) == r.get("n_input"),
            "gate": {"min_n_windows": gate.get("min_n_windows"), "min_n_sessions": gate.get("min_n_sessions"),
                     "overridden": bool(gate.get("overridden")), "override_reason": gate.get("override_reason"),
                     "default_min_n_windows": v1.DEFAULT_MIN_N_WINDOWS,
                     "default_min_n_sessions": v1.DEFAULT_MIN_N_SESSIONS},
            "gates": gates, "reason": r.get("reason"), "reason_text": GATE_REASONS.get(r.get("reason")),
            "failed_gate": failed, "query": bout_view(query) if query else None, "query_basis": basis}
