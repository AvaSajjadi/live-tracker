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

One investigation at a time: RunLock is an OS-level exclusive lock (fcntl.flock on POSIX, msvcrt.locking on Windows) on a stable file in the runs
directory, held for the whole run by every browser session and process of the server alike, and each preflight
issues a one-use token that a paid run consumes.
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

# ---- the OS lock, per platform ----
# POSIX: fcntl.flock on the whole file (shared probe, exclusive hold); BlockingIOError means held elsewhere.
# Windows (no fcntl): msvcrt.locking on one byte far past the metadata, so the holder's pid record stays readable
# (Windows byte-range locks are mandatory); contention is one of investigation_data.WINDOWS_LOCK_CONTENTION, and a
# probe takes the same exclusive lock for an instant (it cannot be shared there) and releases it at once.

_WIN_LOCK_OFFSET = 1 << 20


def _lock(fd, exclusive):
    """True when the lock was taken without waiting, False when another holder has it; other errors propagate."""
    if os.name == "nt":  # pragma: no cover - Windows
        import msvcrt
        from .investigation_data import WINDOWS_LOCK_CONTENTION
        os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in WINDOWS_LOCK_CONTENTION:
                return False
            raise
    import fcntl
    try:
        fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(fd):
    if os.name == "nt":  # pragma: no cover - Windows
        import msvcrt
        os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_UN)


class RunBusy(RuntimeError):
    """Another investigation holds the lock, or the lock cannot be taken safely (fail closed)."""


class RunLock:
    """One investigation at a time: an OS-level exclusive non-blocking lock (see _lock) on a stable lock file
    in the runs directory, held on one open file descriptor for the whole critical section. Ownership is the kernel's
    lock, never the file's content: the file is never unlinked (its inode is reused), its content (the holder's pid
    and start time) is informational only, and an empty, unreadable or malformed file neither proves nor disproves
    that a lock is held. The kernel releases the lock when the descriptor is closed, including when the holding
    process dies, so a lock left on disk by a finished or killed process is simply free again. Any filesystem or
    locking error refuses the run (RunBusy). A RunLock object is held at most once at a time."""

    def __init__(self, root):
        self.path = Path(root) / LOCK_NAME
        self.fd = None

    def _open(self):
        """The lock file's descriptor: a regular file opened without following a symlink."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("the lock path is not a regular file")
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _write_metadata(self, fd):
        """Informational only (who holds it, since when); written after the lock is held."""
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps({"pid": os.getpid(), "started": time.time()}).encode())

    def holder(self):
        """None when no one holds the lock; otherwise what is known of the holder ({} when its metadata is not
        readable). Probed with a non-blocking shared lock, so it never takes the lock from anyone; any error is
        reported as held (fail closed)."""
        try:
            fd = self._open()
        except OSError:
            return {"error": "the lock file cannot be opened"}
        try:
            try:
                free = _lock(fd, exclusive=False)
            except OSError:                       # not lockable: busy
                free = False
            if not free:                          # held exclusively elsewhere
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    info = json.loads(os.read(fd, 4096) or b"{}")
                except (OSError, ValueError):
                    info = {}
                return info if isinstance(info, dict) else {}
            _unlock(fd)
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
            if not _lock(fd, exclusive=True):
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
            _unlock(fd)
        finally:
            os.close(fd)

    def __exit__(self, *exc):
        self._release()
        return False


def new_run_name():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + secrets.token_hex(4)


# test seam: a factory of the httpx client the paid run uses (None: the SDK's own client)
HTTP_CLIENT_FACTORY = None


def execute(config, inputs, *, confirm_budget_usd, authorization, paid=True, root=None, env=None, http_get=None,
            emulator=None):
    """One investigation through the CLI's run() under the lock, into a new protected directory. paid=True needs the
    confirmed budget and the exact authorization text (refused by authorize_live before any request or file);
    paid=False is the free rehearsal over the emulated endpoint. Returns (run name, summary); raises the CLI's
    refusals (LiveRefused, RunRefused, ValueError, ...) or RunBusy."""
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
    with RunLock(root):
        client = HTTP_CLIENT_FACTORY() if (paid and HTTP_CLIENT_FACTORY) else None
        summary = lr.run(args, env=env, http_get=http_get, http_client=client, emulator=emulator)
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
        return "verified", (f"Verified autonomous investigation: finding `{d['finding']}` (consistency-checked; "
                            "publication not authorized).")
    return "incomplete", (f"Verified but incomplete: the investigation stopped ({d['stopping_reason']}) before an "
                          "accepted finding; the evidence listed is partial and no finding was made.")


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
