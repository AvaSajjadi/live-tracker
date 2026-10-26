"""EXPERIMENTAL (Investigation mode, Stage 2): versioned tool contracts (new file).

Interfaces and validators only. No scientific computation lives here: the
four tools are implemented elsewhere (later stages, or Ava's own
implementations) behind the replaceable InvestigationTools interface, and
every request and result crosses this module's validators.

Contract version: CONTRACT_VERSION ("investigation-tools/0.1").

Tools
-----
query_events            behavioral episodes (session, frames, times, source, provenance)
get_window_features     causal measurements in pre-event and comparison windows
check_tracking_quality  window quality: derivative endpoints, history, smoothing
compare_windows         session-level comparison of qualified windows

Every request is a plain JSON object {"contract_version", ...parameters}.
Every result is a plain JSON object:

    {"contract_version", "tool", "status", "evidence_id",
     "measurements",   # measured facts (null when status == "error")
     "methodology",    # the choices that produced them, each marked provisional or approved
     "limitations",    # plain-language caveats
     "provenance",     # implementation, code and data identities
     "errors"}         # [{code, message}], required when status == "error"

status: "ok" | "no_data" (nothing matched) | "insufficient_evidence" (below the
caller's sufficiency criteria) | "error". Missing values are null, never NaN.

Safety rules enforced here, not left to implementations:
- Only JSON values (no NaN / Infinity, no tuples, string keys), exact key sets
  (unknown fields are errors), bounded sizes (MAX_* and MAX_REQUEST_BYTES).
- Event sources are explicit: "validation_annotation", "classifier_prediction",
  "monitor_event".
- Fail-closed evidence acceptance: validate_request / validate_result /
  call_tool REQUIRE a manifest-backed EvidencePolicy (built from
  splits/split_manifest.json). Annotation-derived evidence (annotation
  episodes and their provenance, label_context) is accepted only for
  sessions of the policy's annotation split (validation); every other
  session must be in the manifest; nothing annotation-derived may refer to a
  held-out test session. Synthetic fixtures are refused unless the policy
  explicitly allows them (tests). check_request_schema / check_result_schema
  check structure and internal consistency only; they never accept evidence.
- IDs are deterministic (episode_id / pre_window_id / baseline_window_id) and
  results may only mention IDs the request asked for; every requested ID is
  accounted for exactly once.
- tracking quality can never claim verified motion accuracy:
  "motion_accuracy_verified" is the constant false in this version, and
  smoothing contamination must be reported as assessed or not.
- Statistical sufficiency is a caller-supplied, provisional configuration
  (SufficiencyCriteria). A comparison carries every per-window count and
  value needed to re-check it: the validator recomputes window coverage,
  per-session medians and differences, the median of session differences,
  and whether every criterion is met (tolerance STAT_ABS_TOL / STAT_REL_TOL).
  "ok" exactly when all criteria are met; otherwise no effect values.
- Every tool has explicit status rules (see the _res_* checks).
- Integers must lie in the JSON-safe range (|n| <= 2**53).
- evidence_id = "ev:<tool>:<sha256 prefix>" of the canonical request + result,
  so a result cannot be altered without changing its ID. A matching ID is
  necessary, not sufficient: content is always re-checked.
- call_tool returns a private deep copy of the validated result, so the
  implementation cannot change evidence after validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .calms21 import FPS
from .features import DYADIC_COLS
from .rules import CLASSIFIER_BEHAVIORS

__all__ = [
    "CONTRACT_VERSION", "TOOLS", "SOURCES", "STATUSES", "FEATURES", "ContractError",
    "ForbiddenEvidence", "EvidencePolicy", "SufficiencyCriteria", "PROVISIONAL_SUFFICIENCY",
    "CONTRACT_V1_MAX_TOOL_CALLS", "CONTRACT_V1_MAX_FOLLOW_UPS", "CONTRACT_V1_PRE_S_GRID",
    "CONTRACT_V1_PRE_WINDOW_GRID_FRAMES",
    "InvestigationTools", "UnavailableTools", "ToolContractViolation", "episode_id",
    "pre_window_id", "baseline_window_id", "evidence_id", "make_result", "validate_request",
    "validate_result", "check_request_schema", "check_result_schema", "call_tool",
]

CONTRACT_VERSION = "investigation-tools/0.1"
# Contract v1 agent budget (docs/investigate_tool_contracts.md), shared by the Stage 9 agent loop
# (investigation_agent.Budget) and the v1 adapter (investigation_tools_v1.InvestigationRunner), so the two
# can never enforce different limits. A comparison counts once it ran (status other than "error").
CONTRACT_V1_MAX_TOOL_CALLS = 12
CONTRACT_V1_MAX_FOLLOW_UPS = 2                       # compare_windows analyses after the first
CONTRACT_V1_PRE_S_GRID = (1.0, 2.0, 5.0)             # preselected pre-window lengths, seconds
CONTRACT_V1_PRE_WINDOW_GRID_FRAMES = (30, 60, 150)   # the same grid at 30 fps
TOOLS = ("query_events", "get_window_features", "check_tracking_quality", "compare_windows")
SOURCES = ("validation_annotation", "classifier_prediction", "monitor_event")
STATUSES = ("ok", "no_data", "insufficient_evidence", "error")
BEHAVIORS = tuple(sorted(CLASSIFIER_BEHAVIORS))
FEATURES = tuple(DYADIC_COLS)  # Ava's causal pair features (features.py)
LABELS = ("attack", "investigation", "mount", "other")
ERROR_CODES = ("invalid_request", "unknown_id", "forbidden_split", "source_unavailable",
               "not_implemented", "internal_error")
DATA_KINDS = ("calms21_validation_cache", "classifier_scores", "monitor_journal",
              "synthetic_fixture")
EXCLUSION_REASONS = ("starts_before_session", "overlaps_event", "overlaps_pre_window",
                     "near_event", "insufficient_usable_frames", "missing_data")
QUALITY_VERDICTS = ("usable_with_caveats", "excluded", "insufficient_data")
CONTAMINATION_STATUS = ("not_assessed", "none_detected", "detected")
# the only reasons a comparison can be insufficient; reasons_not_met lists the
# unmet ones in this order
SUFFICIENCY_REASONS = ("min_sessions", "min_windows_per_group_pre", "min_windows_per_group_baseline")
TRACKING_OK_MEANING = ("every needed keypoint passed the score / frame / frozen flags; "
                       "not proof that the measured motion is accurate")

MAX_REQUEST_BYTES = 64_000
MAX_SESSIONS = 100
MAX_EPISODES = 500
MAX_WINDOWS = 2_000
MAX_HISTORY_VALUES = 5
MAX_FRAMES = 10_000_000
MAX_WINDOW_FRAMES = 900          # 30 s at 30 fps
MAX_TEXT = 500
MAX_LIST_TEXT = 20
MAX_SAFE_INT = 2 ** 53           # JSON-safe integer range
STAT_ABS_TOL = 1e-9              # tolerance for re-computed statistics
STAT_REL_TOL = 1e-9
_TIME_TOL = 1e-6

_SESSION = r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,127}"
_SESSION_RE = re.compile(_SESSION)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_EPISODE_RE = re.compile(rf"ep:({'|'.join(SOURCES)}):({'|'.join(BEHAVIORS)}):({_SESSION}):(0|[1-9]\d{{0,7}})")
_EVIDENCE_RE = re.compile(rf"ev:({'|'.join(TOOLS)}):[0-9a-f]{{24}}")


class ContractError(ValueError):
    """A request or result does not satisfy the contract. `path` locates it."""

    def __init__(self, path, message):
        super().__init__(f"{path}: {message}")
        self.path = path


class ForbiddenEvidence(ContractError):
    """Evidence would expose held-out test annotations or unauthorized data."""


# ---- identifiers ----

def _check_session_id(sid, path="session"):
    if not isinstance(sid, str) or not _SESSION_RE.fullmatch(sid) or ".." in sid:
        raise ContractError(path, f"invalid session id {sid!r}")
    return sid


# Identifier ranges: the constructors accept exactly what the parsers accept,
# and the parsers accept only the canonical text the constructors produce
# (no leading zeros), so build -> parse -> build is the identity.
_ID_FRAME_MAX = MAX_FRAMES                 # start / onset frames: 0 .. MAX_FRAMES
_ID_LENGTH_RANGE = (1, MAX_WINDOW_FRAMES)  # window length: 1 .. MAX_WINDOW_FRAMES
_ID_GAP_RANGE = (0, MAX_WINDOW_FRAMES)     # pre-event gap: 0 .. MAX_WINDOW_FRAMES


def _frames_arg(value, lo, hi, name):
    if type(value) is not int or not lo <= value <= hi:
        raise ContractError(name, f"must be an int in [{lo}, {hi}]")
    return value


def episode_id(source, behavior, session, start_frame):
    """ep:<source>:<behavior>:<session>:<start_frame> (deterministic)."""
    if source not in SOURCES or behavior not in BEHAVIORS:
        raise ContractError("episode_id", "unknown source or behavior")
    _check_session_id(session)
    _frames_arg(start_frame, 0, _ID_FRAME_MAX, "episode_id.start_frame")
    return f"ep:{source}:{behavior}:{session}:{start_frame}"


def pre_window_id(ep_id, length_frames, gap_frames):
    """win:pre:<length>:<gap>:<episode_id> (deterministic)."""
    parse_episode_id(ep_id)
    return (f"win:pre:{_frames_arg(length_frames, *_ID_LENGTH_RANGE, 'length_frames')}:"
            f"{_frames_arg(gap_frames, *_ID_GAP_RANGE, 'gap_frames')}:{ep_id}")


def baseline_window_id(source, behavior, session, start_frame, length_frames):
    """win:base:<length>:<source>:<behavior>:<session>:<start_frame> (deterministic)."""
    if source not in SOURCES or behavior not in BEHAVIORS:
        raise ContractError("baseline_window_id", "unknown source or behavior")
    _check_session_id(session)
    return (f"win:base:{_frames_arg(length_frames, *_ID_LENGTH_RANGE, 'length_frames')}:{source}:"
            f"{behavior}:{session}:{_frames_arg(start_frame, 0, _ID_FRAME_MAX, 'start_frame')}")


_NUM = r"(0|[1-9]\d{0,7})"   # canonical non-negative integer text (range checked by rebuilding)
_PRE_RE = re.compile(rf"win:pre:{_NUM}:{_NUM}:({_EPISODE_RE.pattern})")
_BASE_RE = re.compile(rf"win:base:{_NUM}:({'|'.join(SOURCES)}):({'|'.join(BEHAVIORS)}):"
                      rf"({_SESSION}):{_NUM}")


def parse_window_id(wid):
    """{"kind", "session", "source", "behavior", "length_frames", ...} or
    ContractError. Only canonical IDs the constructors could build are accepted."""
    if not isinstance(wid, str):
        raise ContractError("window_id", "must be a string")
    m = _PRE_RE.fullmatch(wid)
    if m:
        e = parse_episode_id(m.group(3))
        out = {"kind": "pre_event", "length_frames": int(m.group(1)), "gap_frames": int(m.group(2)),
               "episode_id": m.group(3), "source": e["source"], "behavior": e["behavior"],
               "session": e["session"], "onset_frame": e["start_frame"]}
        rebuilt = lambda: pre_window_id(out["episode_id"], out["length_frames"], out["gap_frames"])  # noqa: E731
    else:
        m = _BASE_RE.fullmatch(wid)
        if not m:
            raise ContractError("window_id", f"invalid window id {wid!r}")
        out = {"kind": "baseline", "length_frames": int(m.group(1)), "source": m.group(2),
               "behavior": m.group(3), "session": m.group(4), "start_frame": int(m.group(5)),
               "episode_id": None}
        rebuilt = lambda: baseline_window_id(out["source"], out["behavior"], out["session"],  # noqa: E731
                                             out["start_frame"], out["length_frames"])
    try:
        canonical = rebuilt()
    except ContractError:
        raise ContractError("window_id", f"window id out of range {wid!r}") from None
    if canonical != wid:
        raise ContractError("window_id", f"non-canonical window id {wid!r}")
    return out


def parse_episode_id(eid):
    m = _EPISODE_RE.fullmatch(eid) if isinstance(eid, str) else None
    if not m:
        raise ContractError("episode_id", f"invalid episode id {eid!r}")
    out = {"source": m.group(1), "behavior": m.group(2), "session": m.group(3),
           "start_frame": int(m.group(4))}
    try:
        canonical = episode_id(out["source"], out["behavior"], out["session"], out["start_frame"])
    except ContractError:
        raise ContractError("episode_id", f"episode id out of range {eid!r}") from None
    if canonical != eid:
        raise ContractError("episode_id", f"non-canonical episode id {eid!r}")
    return out


# ---- evidence policy (fail closed: required for every evidence acceptance) ----

class EvidencePolicy:
    """Manifest-backed rules for what may become agent evidence.

    Annotation-derived evidence may only use `annotation_split` sessions
    ("validation"; "train" only if the team decides so; "test" never).
    Predictions and monitor events may use any manifest session (they carry
    no labels). Sessions outside the manifest are refused. Synthetic fixtures
    (provenance kind "synthetic_fixture") are refused unless
    allow_synthetic=True (tests only)."""

    def __init__(self, manifest, *, annotation_split="validation", allow_synthetic=False):
        if annotation_split == "test":
            raise ForbiddenEvidence("policy", "test annotations are never agent evidence")
        if annotation_split not in ("validation", "train"):
            raise ContractError("policy", f"unknown split {annotation_split!r}")
        if not isinstance(manifest, dict):
            raise ContractError("policy", "the manifest must be an object")
        self.split_of = {}
        for split in ("train", "validation", "test"):
            ids = manifest.get(split)
            if not isinstance(ids, list):
                raise ContractError("policy", f"the manifest has no list {split!r}")
            for sid in ids:
                _check_session_id(sid, f"policy.{split}")
                if sid in self.split_of:
                    raise ContractError("policy", f"session {sid!r} is in two splits")
                self.split_of[sid] = split
        if not any(v == annotation_split for v in self.split_of.values()):
            raise ContractError("policy", f"the manifest has no {annotation_split!r} sessions")
        self.annotation_split = annotation_split
        self.allow_synthetic = bool(allow_synthetic)

    @classmethod
    def from_manifest(cls, path, **kwargs):
        from .investigation_data import load_manifest  # Stage 1 validator
        manifest, _ = load_manifest(path)
        return cls(manifest, **kwargs)

    def check_annotation_session(self, sid, path):
        split = self.split_of.get(sid)
        if split == "test":
            raise ForbiddenEvidence(path, f"{sid!r} is a held-out test session")
        if split != self.annotation_split:
            raise ForbiddenEvidence(path, f"{sid!r} is not in the {self.annotation_split!r} split")

    def check_known_session(self, sid, path):
        if sid not in self.split_of:
            raise ForbiddenEvidence(path, f"{sid!r} is not in the split manifest")


def _require_policy(policy):
    if not isinstance(policy, EvidencePolicy):
        raise ForbiddenEvidence("policy", "accepting evidence requires a manifest-backed "
                                "EvidencePolicy (use check_*_schema for structure-only checks)")


def _check_session_for_source(policy, sid, source, path):
    """policy None = schema-only checking (never evidence acceptance)."""
    if policy is None:
        return
    if source == "validation_annotation":
        policy.check_annotation_session(sid, path)
    else:
        policy.check_known_session(sid, path)


# ---- configurable sufficiency (provisional, never a hidden default) ----

@dataclass(frozen=True)
class SufficiencyCriteria:
    """Minimums a comparison must meet before it may report an effect. These
    are methodological choices supplied by the caller, not scientific truths."""
    min_sessions: int
    min_windows_per_group: int
    min_usable_fraction: float
    status: str = "provisional"     # "provisional" or "approved"
    decided_by: Optional[str] = None

    def to_json(self):
        return asdict(self)


# Placeholder values for development only; status "provisional" travels with
# them into every request and result that uses them.
PROVISIONAL_SUFFICIENCY = SufficiencyCriteria(min_sessions=4, min_windows_per_group=10,
                                              min_usable_fraction=0.9)


# ---- tiny validator kit ----

Validator = Callable[[Any, str], None]


def _fail(path, msg):
    raise ContractError(path, msg)


def S(max_len=MAX_TEXT, pattern=None, min_len=1):
    rx = re.compile(pattern) if isinstance(pattern, str) else pattern

    def v(x, path):
        if not isinstance(x, str) or not (min_len <= len(x) <= max_len):
            _fail(path, f"must be a string of {min_len}..{max_len} characters")
        if rx is not None and not rx.fullmatch(x):
            _fail(path, f"invalid value {x!r}")
    return v


def I(lo=0, hi=MAX_FRAMES):
    def v(x, path):
        if type(x) is not int or not lo <= x <= hi:
            _fail(path, f"must be an int in [{lo}, {hi}]")
    return v


def N(lo=-math.inf, hi=math.inf):
    def v(x, path):
        if isinstance(x, bool) or not isinstance(x, (int, float)) \
                or (isinstance(x, int) and abs(x) > MAX_SAFE_INT) \
                or not math.isfinite(x) or not lo <= x <= hi:
            _fail(path, f"must be a finite number in [{lo}, {hi}]")
    return v


def B(x, path):
    if type(x) is not bool:
        _fail(path, "must be true or false")


def E(*choices):
    def v(x, path):
        if not isinstance(x, (str, bool, int)) or isinstance(x, float) or x not in choices:
            _fail(path, f"must be one of {list(choices)}")
    return v


def C(value):
    def v(x, path):
        if type(x) is not type(value) or x != value:
            _fail(path, f"must be {value!r}")
    return v


def Null(inner):
    def v(x, path):
        if x is not None:
            inner(x, path)
    return v


def L(item, max_items, min_items=0, unique=False):
    def v(x, path):
        if not isinstance(x, list) or not min_items <= len(x) <= max_items:
            _fail(path, f"must be a list of {min_items}..{max_items} items")
        for i, el in enumerate(x):
            item(el, f"{path}[{i}]")
        if unique:
            keys = [json.dumps(el, sort_keys=True) for el in x]
            if len(set(keys)) != len(keys):
                _fail(path, "items must be unique")
    return v


def O(fields, optional=()):
    """Object with exactly `fields` (+ any of `optional`); unknown keys are errors."""
    def v(x, path):
        if not isinstance(x, dict) or not all(isinstance(k, str) for k in x):
            _fail(path, "must be an object with string keys")
        missing = sorted(set(fields) - set(optional) - set(x))
        extra = sorted(set(x) - set(fields))
        if missing:
            _fail(path, f"missing fields {missing}")
        if extra:
            _fail(path, f"unknown fields {extra}")
        for k, val in x.items():
            fields[k](val, f"{path}.{k}")
    return v


def M(key, value, max_items):
    """Map with validated keys and values."""
    def v(x, path):
        if not isinstance(x, dict) or len(x) > max_items:
            _fail(path, f"must be an object with at most {max_items} entries")
        for k, val in x.items():
            key(k, f"{path}<key>")
            value(val, f"{path}.{k}")
    return v


SESSION = S(128, _SESSION_RE)
HEX64 = S(64, _HEX64)
EPISODE = S(220, _EPISODE_RE)
WINDOW = S(260, re.compile(f"{_PRE_RE.pattern}|{_BASE_RE.pattern}"))
EVIDENCE = S(80, _EVIDENCE_RE)
TEXT = S(MAX_TEXT)
def JSON_SCALAR(x, path):
    """A methodology value: null, string, bool, finite number, or a short list of numbers."""
    if x is None or isinstance(x, (str, bool)):
        if isinstance(x, str) and len(x) > MAX_TEXT:
            _fail(path, "string too long")
        return
    if isinstance(x, list):
        L(N(), 20)(x, path)
        return
    N()(x, path)


def ensure_json(obj, path="$"):
    """Plain JSON only: dict (str keys) / list / str / bool / None / int /
    finite float. Tuples, NaN and Infinity are refused."""
    if obj is None or isinstance(obj, (str, bool)):
        return
    if isinstance(obj, int):
        if abs(obj) > MAX_SAFE_INT:
            _fail(path, "integer outside the JSON-safe range")
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            _fail(path, "non-finite number")
        return
    if isinstance(obj, list):
        for i, x in enumerate(obj):
            ensure_json(x, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, x in obj.items():
            if not isinstance(k, str):
                _fail(path, "object keys must be strings")
            ensure_json(x, f"{path}.{k}")
        return
    _fail(path, f"{type(obj).__name__} is not a JSON value")


# ---- shared result parts ----

METHOD_ITEM = O({"name": S(80), "value": JSON_SCALAR, "status": E("provisional", "approved"),
                 "decided_by": Null(S(120)), "note": Null(TEXT)})
DATA_ITEM = O({"kind": E(*DATA_KINDS), "path": Null(S(1000)), "sha256": Null(HEX64),
               "description": TEXT})
PROVENANCE = O({"implementation": S(120), "implementation_version": S(40),
                "contract_version": C(CONTRACT_VERSION), "code_sha256": Null(HEX64),
                "data": L(DATA_ITEM, 20)})
ERROR = O({"code": E(*ERROR_CODES), "message": TEXT})
SUFFICIENCY = O({"min_sessions": I(1, MAX_SESSIONS), "min_windows_per_group": I(1, MAX_WINDOWS),
                 "min_usable_fraction": N(0.0, 1.0), "status": E("provisional", "approved"),
                 "decided_by": Null(S(120))})

# ---- requests ----

REQUESTS = {
    "query_events": O({
        "contract_version": C(CONTRACT_VERSION),
        "source": E(*SOURCES),
        "behavior": E(*BEHAVIORS),
        "sessions": Null(L(SESSION, MAX_SESSIONS, 1, unique=True)),  # null: all the source allows
        "max_episodes": I(1, MAX_EPISODES),
        "merge_gap_frames": Null(I(0, MAX_WINDOW_FRAMES)),
        "min_duration_frames": Null(I(1, MAX_WINDOW_FRAMES)),
    }),
    "get_window_features": O({
        "contract_version": C(CONTRACT_VERSION),
        "episode_ids": L(EPISODE, MAX_EPISODES, 1, unique=True),
        "pre_event": O({"length_frames": I(1, MAX_WINDOW_FRAMES), "gap_frames": I(0, MAX_WINDOW_FRAMES)}),
        "baseline": Null(O({"strategy": E("same_session_tiles"),
                            "length_frames": I(1, MAX_WINDOW_FRAMES),
                            "clearance_frames": I(0, 10 * MAX_WINDOW_FRAMES),
                            "max_windows_per_session": I(1, MAX_WINDOWS)})),
        "features": L(E(*FEATURES), len(FEATURES), 1, unique=True),
    }),
    "check_tracking_quality": O({
        "contract_version": C(CONTRACT_VERSION),
        "window_ids": L(WINDOW, MAX_WINDOWS, 1, unique=True),
        "history_frames": L(I(1, MAX_WINDOW_FRAMES), MAX_HISTORY_VALUES, 1, unique=True),
    }),
    "compare_windows": O({
        "contract_version": C(CONTRACT_VERSION),
        "feature": E(*FEATURES),
        "statistic": E("median_difference"),
        "aggregation": E("session_level"),
        "pre_window_ids": L(WINDOW, MAX_WINDOWS, 1, unique=True),
        "baseline_window_ids": L(WINDOW, MAX_WINDOWS, 1, unique=True),
        "sufficiency": SUFFICIENCY,
        "uncertainty": O({"method": E("bootstrap_sessions", "none"), "level": Null(N(0.5, 0.999)),
                          "resamples": Null(I(100, 100_000)), "seed": Null(I(0, 2**31 - 1))}),
    }),
}


def check_request_schema(tool, request):
    """Structure and internal consistency only. NOT evidence acceptance:
    session splits and test-annotation rules are not applied."""
    _validate_request(tool, request, None)


def validate_request(tool, request, policy):
    """Trusted acceptance: the request is valid AND allowed by `policy`
    (a manifest-backed EvidencePolicy; required)."""
    _require_policy(policy)
    _validate_request(tool, request, policy)


def _validate_request(tool, request, policy):
    if tool not in TOOLS:
        raise ContractError("tool", f"unknown tool {tool!r}")
    ensure_json(request, "request")
    try:
        size = len(json.dumps(request, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ContractError("request", f"not serializable: {type(exc).__name__}") from None
    if size > MAX_REQUEST_BYTES:
        raise ContractError("request", f"{size} bytes exceeds {MAX_REQUEST_BYTES}")
    REQUESTS[tool](request, "request")
    _REQUEST_CHECKS[tool](request, policy)


def _req_query(r, policy):
    for i, sid in enumerate(r["sessions"] or []):
        _check_session_for_source(policy, sid, r["source"], f"request.sessions[{i}]")


def _one_source(items, path):
    kinds = {(x["source"], x["behavior"]) for x in items}
    if len(kinds) != 1:
        raise ContractError(path, f"mixes sources / behaviors {sorted(kinds)}")


def _req_features(r, policy):
    eps = [parse_episode_id(eid) for eid in r["episode_ids"]]
    _one_source(eps, "request.episode_ids")
    for i, e in enumerate(eps):
        _check_session_for_source(policy, e["session"], e["source"], f"request.episode_ids[{i}]")


def _req_quality(r, policy):
    for i, wid in enumerate(r["window_ids"]):
        w = parse_window_id(wid)
        _check_session_for_source(policy, w["session"], w["source"], f"request.window_ids[{i}]")


def _req_compare(r, policy):
    pre, base = r["pre_window_ids"], r["baseline_window_ids"]
    if set(pre) & set(base):
        raise ContractError("request", "a window cannot be in both groups")
    _one_source([parse_window_id(w) for w in pre + base], "request")
    for group, ids, kind in (("pre_window_ids", pre, "pre_event"), ("baseline_window_ids", base, "baseline")):
        for i, wid in enumerate(ids):
            w = parse_window_id(wid)
            if w["kind"] != kind:
                raise ContractError(f"request.{group}[{i}]", f"must be a {kind} window")
            _check_session_for_source(policy, w["session"], w["source"], f"request.{group}[{i}]")
    u = r["uncertainty"]
    if u["method"] == "none" and any(u[k] is not None for k in ("level", "resamples", "seed")):
        raise ContractError("request.uncertainty", "method 'none' takes no level / resamples / seed")
    if u["method"] == "bootstrap_sessions" and any(u[k] is None for k in ("level", "resamples", "seed")):
        raise ContractError("request.uncertainty", "bootstrap needs level, resamples and seed")
    s = r["sufficiency"]
    if s["status"] == "approved" and not s["decided_by"]:
        raise ContractError("request.sufficiency", "approved criteria must name who decided")


_REQUEST_CHECKS = {"query_events": _req_query, "get_window_features": _req_features,
                   "check_tracking_quality": _req_quality, "compare_windows": _req_compare}

# ---- result measurements ----

EPISODE_PROVENANCE = {
    "validation_annotation": O({"split": C("validation"), "annotator": S(40), "file_sha256": HEX64}),
    "classifier_prediction": O({"model_id": S(120), "score_file_sha256": HEX64,
                                "threshold": N(0.0, 1.0), "min_duration_frames": I(1, MAX_WINDOW_FRAMES)}),
    "monitor_event": O({"journal_sha256": HEX64, "run_id": S(80), "rule_id": S(80), "event_id": S(120)}),
}
# annotated_frames: frames inside [start_frame, end_frame) that are actually
# annotated with the behavior (merged gaps excluded). It exists only for
# human-annotation episodes: required for "validation_annotation", forbidden for
# predictions and monitor events (they have no annotations). Checked in _res_query.
ANNOTATION_COUNT_SOURCES = ("validation_annotation",)
EPISODE_ITEM = O({"episode_id": EPISODE, "session": SESSION, "source": E(*SOURCES),
                  "behavior": E(*BEHAVIORS), "start_frame": I(), "end_frame": I(),
                  "start_s": N(0), "end_s": N(0), "fps": C(FPS), "provenance": lambda x, p: None,
                  "annotated_frames": I(1, MAX_FRAMES)},
                 optional=("annotated_frames",))
FEATURE_SUMMARY = O({"n_frames": I(), "n_usable": I(), "median": Null(N()), "mean": Null(N())})
LABEL_CONTEXT = O({k: I() for k in LABELS})
WINDOW_ITEM = O({
    "window_id": WINDOW, "kind": E("pre_event", "baseline"), "session": SESSION,
    "episode_id": Null(EPISODE), "start_frame": I(), "end_frame": I(), "start_s": N(0), "end_s": N(0),
    "status": E("included", "excluded"), "exclusion_reasons": L(E(*EXCLUSION_REASONS), 6, unique=True),
    "features": lambda x, p: None,  # checked against the request below
    "label_context": Null(LABEL_CONTEXT),
    "preprocessing": O({"truncated_at_onset": B, "interpolation": C(False),
                        "smoothing": E("one_euro", "none")}),
})
QUALITY_ITEM = O({
    "window_id": WINDOW, "start_frame": I(), "end_frame": I(),
    "frames": I(1, MAX_WINDOW_FRAMES), "tracking_ok_frames": I(),
    "endpoint_valid_frames": I(), "usable_by_history": lambda x, p: None,
    "excluded_frames": O({"not_tracking_ok": L(I(), MAX_WINDOW_FRAMES),
                          "endpoint_invalid": L(I(), MAX_WINDOW_FRAMES)}),
    "smoothing_contamination": O({"status": E(*CONTAMINATION_STATUS),
                                  "frames": L(I(), MAX_WINDOW_FRAMES), "note": TEXT}),
    "tracking_ok_meaning": C(TRACKING_OK_MEANING),
    "motion_accuracy_verified": C(False),
    "verdict": E(*QUALITY_VERDICTS),
})
WINDOW_VALUE = O({"window_id": WINDOW, "n_frames": I(1, MAX_WINDOW_FRAMES),
                  "n_usable": I(0, MAX_WINDOW_FRAMES), "value": N()})
SESSION_ROW = O({"session": SESSION, "pre_windows": L(WINDOW_VALUE, MAX_WINDOWS),
                 "baseline_windows": L(WINDOW_VALUE, MAX_WINDOWS),
                 "pre_summary": Null(N()), "baseline_summary": Null(N()), "difference": Null(N())})
EXCLUDED_WINDOW = O({"window_id": WINDOW, "reason": E(*EXCLUSION_REASONS),
                     "n_frames": Null(I(0, MAX_WINDOW_FRAMES)), "n_usable": Null(I(0, MAX_WINDOW_FRAMES))})
MEASUREMENTS = {
    "query_events": O({"episodes": L(EPISODE_ITEM, MAX_EPISODES), "n_episodes_total": I(),
                       "truncated": B, "sessions_searched": L(SESSION, MAX_SESSIONS),
                       "sessions_without_episodes": L(SESSION, MAX_SESSIONS)}),
    "get_window_features": O({"windows": L(WINDOW_ITEM, MAX_WINDOWS)}),
    "check_tracking_quality": O({"windows": L(QUALITY_ITEM, MAX_WINDOWS)}),
    "compare_windows": O({
        "feature": E(*FEATURES), "aggregation": C("session_level"),
        "per_session": L(SESSION_ROW, MAX_SESSIONS),
        "excluded_windows": L(EXCLUDED_WINDOW, MAX_WINDOWS),
        "n_sessions_paired": I(0, MAX_SESSIONS), "n_windows_pre": I(0, MAX_WINDOWS),
        "n_windows_baseline": I(0, MAX_WINDOWS),
        "effect": O({"name": C("median_of_session_differences"), "value": Null(N()),
                     "ci_low": Null(N()), "ci_high": Null(N()), "ci_level": Null(N(0.5, 0.999))}),
        "sufficiency": O({"criteria": SUFFICIENCY, "met": B,
                          "reasons_not_met": L(E(*SUFFICIENCY_REASONS), len(SUFFICIENCY_REASONS))}),
    }),
}
RESULT = O({"contract_version": C(CONTRACT_VERSION), "tool": E(*TOOLS), "status": E(*STATUSES),
            "evidence_id": EVIDENCE, "measurements": lambda x, p: None,
            "methodology": L(METHOD_ITEM, 50), "limitations": L(TEXT, MAX_LIST_TEXT),
            "provenance": PROVENANCE, "errors": L(ERROR, MAX_LIST_TEXT)})


# ---- evidence ids ----

def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def evidence_id(tool, request, result_without_id):
    body = {k: v for k, v in result_without_id.items() if k != "evidence_id"}
    digest = hashlib.sha256(_canonical({"tool": tool, "request": request, "result": body}).encode())
    return f"ev:{tool}:{digest.hexdigest()[:24]}"


def make_result(tool, request, status, measurements, *, methodology=(), limitations=(),
                provenance, errors=()):
    """Assemble a result (implementations should use this) with its evidence_id."""
    result = {"contract_version": CONTRACT_VERSION, "tool": tool, "status": status,
              "measurements": measurements, "methodology": list(methodology),
              "limitations": list(limitations), "provenance": provenance, "errors": list(errors)}
    ensure_json(request, "request")
    ensure_json(result, "result")  # e.g. NaN -> ContractError, not a json ValueError
    result["evidence_id"] = evidence_id(tool, request, result)
    return result


# ---- result validation ----

def check_result_schema(tool, request, result):
    """Structure and internal consistency only. NOT evidence acceptance."""
    _validate_result(tool, request, result, None)


def validate_result(tool, request, result, policy):
    """Trusted acceptance: `result` is a valid, internally consistent answer
    to `request` and allowed by `policy` (manifest-backed; required)."""
    _require_policy(policy)
    _validate_result(tool, request, result, policy)


def _validate_result(tool, request, result, policy):
    _validate_request(tool, request, policy)
    ensure_json(result, "result")
    RESULT(result, "result")
    if result["tool"] != tool:
        raise ContractError("result.tool", f"is {result['tool']!r}, expected {tool!r}")
    if result["evidence_id"] != evidence_id(tool, request, result):
        raise ContractError("result.evidence_id", "does not match the request and result content")
    for i, m in enumerate(result["methodology"]):
        if m["status"] == "approved" and not m["decided_by"]:
            raise ContractError(f"result.methodology[{i}]", "approved choices must name who decided")
    synthetic = any(d["kind"] == "synthetic_fixture" for d in result["provenance"]["data"])
    if synthetic and policy is not None and not policy.allow_synthetic:
        raise ForbiddenEvidence("result.provenance", "synthetic fixtures are not evidence")
    if result["status"] == "error":
        if not result["errors"] or result["measurements"] is not None:
            raise ContractError("result", "an error result has errors and null measurements")
        return
    if result["errors"]:
        raise ContractError("result.errors", "only error results carry errors")
    if result["measurements"] is None:
        raise ContractError("result.measurements", "required unless status is 'error'")
    MEASUREMENTS[tool](result["measurements"], "result.measurements")
    _RESULT_CHECKS[tool](request, result, policy)


def _close(a, b):
    return math.isclose(a, b, rel_tol=STAT_REL_TOL, abs_tol=STAT_ABS_TOL)


def _times(item, path, allow_empty=False):
    if item["end_frame"] < item["start_frame"] or (
            item["end_frame"] == item["start_frame"] and not allow_empty):
        raise ContractError(path, "end_frame must be after start_frame")
    for f, t in (("start_frame", "start_s"), ("end_frame", "end_s")):
        if abs(item[t] - item[f] / FPS) > _TIME_TOL:
            raise ContractError(f"{path}.{t}", f"must equal {f} / {FPS}")


def _res_query(req, res, policy):
    m = res["measurements"]
    eps = m["episodes"]
    if len(eps) != min(m["n_episodes_total"], req["max_episodes"]):
        raise ContractError("result", "must return min(n_episodes_total, max_episodes) episodes")
    if m["truncated"] != (m["n_episodes_total"] > len(eps)):
        raise ContractError("result", "truncated exactly when n_episodes_total exceeds the episodes "
                            "returned (and then max_episodes were returned)")
    if (res["status"] == "no_data") != (m["n_episodes_total"] == 0):
        raise ContractError("result.status", "'no_data' exactly when no episode matched")
    if res["status"] == "insufficient_evidence":
        raise ContractError("result.status", "query_events does not judge sufficiency")
    for name in ("sessions_searched", "sessions_without_episodes"):
        if len(set(m[name])) != len(m[name]):
            raise ContractError(f"result.{name}", "sessions must be unique")
    searched = set(m["sessions_searched"])
    if req["sessions"] is not None and searched != set(req["sessions"]):
        raise ContractError("result.sessions_searched", "must be exactly the requested sessions "
                            "(a source that cannot search one must return an error, not skip it)")
    without = set(m["sessions_without_episodes"])
    if not without <= searched:
        raise ContractError("result.sessions_without_episodes", "must be searched sessions")
    returned = {e["session"] for e in eps}
    if not m["truncated"] and without != searched - returned:
        raise ContractError("result.sessions_without_episodes", "an untruncated result is complete: "
                            "exactly the searched sessions with no returned episode")
    # truncated: sessions absent from the page may have episodes beyond it, so only
    # "without" sessions with returned episodes are contradictions (checked below)
    seen = set()
    for i, e in enumerate(eps):
        path = f"result.measurements.episodes[{i}]"
        if e["source"] != req["source"] or e["behavior"] != req["behavior"]:
            raise ContractError(path, "source / behavior differ from the request")
        if e["episode_id"] != episode_id(e["source"], e["behavior"], e["session"], e["start_frame"]):
            raise ContractError(f"{path}.episode_id", "does not match its fields")
        if e["episode_id"] in seen:
            raise ContractError(path, "duplicate episode")
        seen.add(e["episode_id"])
        if e["session"] not in searched or e["session"] in m["sessions_without_episodes"]:
            raise ContractError(f"{path}.session", "not a searched session with episodes")
        _times(e, path)
        EPISODE_PROVENANCE[e["source"]](e["provenance"], f"{path}.provenance")
        duration = e["end_frame"] - e["start_frame"]
        if e["source"] in ANNOTATION_COUNT_SOURCES:
            if "annotated_frames" not in e:
                raise ContractError(f"{path}.annotated_frames", "required for annotation episodes")
            if e["annotated_frames"] > duration:
                raise ContractError(f"{path}.annotated_frames", f"{e['annotated_frames']} exceeds the "
                                    f"episode span of {duration} frames")
            # without merging (null, or a gap of 0 frames, which maximal runs never have)
            # an episode is one annotated run: every frame of the span is annotated
            if not req["merge_gap_frames"] and e["annotated_frames"] != duration:
                raise ContractError(f"{path}.annotated_frames", "an unmerged episode is annotated on "
                                    "every frame of its span")
        elif "annotated_frames" in e:
            raise ContractError(f"{path}.annotated_frames", f"{e['source']} episodes have no "
                                "annotations; annotated_frames is only for human annotations")
        minima = [("request.min_duration_frames", req["min_duration_frames"])]
        if e["source"] == "classifier_prediction":  # only this provenance declares a minimum
            minima.append(("provenance.min_duration_frames", e["provenance"]["min_duration_frames"]))
        for name, minimum in minima:  # every bound must hold, so the stricter one decides
            if minimum is not None and duration < minimum:
                raise ContractError(f"{path}.end_frame", f"duration {duration} frames is below "
                                    f"{name} = {minimum}")
        _check_session_for_source(policy, e["session"], e["source"], f"{path}.session")
    # episodes are maximal bouts: within one session and behavior they never overlap
    by_session = {}
    for e in eps:
        by_session.setdefault(e["session"], []).append((e["start_frame"], e["end_frame"], e["episode_id"]))
    for sid, spans in by_session.items():
        spans.sort()
        for (a0, a1, aid), (b0, b1, bid) in zip(spans, spans[1:]):
            if b0 < a1:
                raise ContractError("result.measurements.episodes", f"{aid} and {bid} overlap")
    for sid in searched:
        _check_session_for_source(policy, sid, req["source"], "result.sessions_searched")


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


_BASELINE_REASONS = {"overlaps_event", "overlaps_pre_window", "near_event",
                     "insufficient_usable_frames", "missing_data"}
_PRE_REASONS = {"starts_before_session", "overlaps_event", "insufficient_usable_frames", "missing_data"}
REASONS_BY_KIND = {"pre_event": _PRE_REASONS, "baseline": _BASELINE_REASONS}


def _near_onset(span, onset, clearance):
    """True if any frame of span [a, b) is within `clearance` frames of `onset`."""
    a, b = span
    return a <= onset + clearance and b - 1 >= onset - clearance


def _res_features(req, res, policy):
    windows = res["measurements"]["windows"]
    requested = set(req["episode_ids"])
    first = parse_episode_id(req["episode_ids"][0])
    pre, base = req["pre_event"], req["baseline"]
    pre_for, seen = {}, set()
    included_pre, included_base = {}, {}
    for i, w in enumerate(windows):
        path = f"result.measurements.windows[{i}]"
        if w["window_id"] in seen:
            raise ContractError(path, "duplicate window")
        seen.add(w["window_id"])
        parsed = parse_window_id(w["window_id"])
        if parsed["kind"] != w["kind"] or parsed["session"] != w["session"]:
            raise ContractError(f"{path}.window_id", "kind / session differ from the id")
        if (parsed["source"], parsed["behavior"]) != (first["source"], first["behavior"]):
            raise ContractError(f"{path}.window_id", "source / behavior differ from the episodes")
        if (w["status"] == "excluded") != bool(w["exclusion_reasons"]):
            raise ContractError(path, "excluded windows (and only they) list exclusion reasons")
        span = (w["start_frame"], w["end_frame"])
        if w["kind"] == "pre_event":
            if w["episode_id"] not in requested:
                raise ContractError(f"{path}.episode_id", "not a requested episode")
            if w["window_id"] != pre_window_id(w["episode_id"], pre["length_frames"], pre["gap_frames"]):
                raise ContractError(f"{path}.window_id", "does not match the episode and pre_event settings")
            end = parse_episode_id(w["episode_id"])["start_frame"] - pre["gap_frames"]
            start = end - pre["length_frames"]
            if start >= 0:   # the full window exists
                if span != (start, end) or "starts_before_session" in w["exclusion_reasons"]:
                    raise ContractError(path, "frames must be [onset - gap - length, onset - gap)")
                _times(w, path)
            else:            # it would start before frame 0: clipped, and must be excluded
                if span != (0, max(end, 0)) or w["status"] != "excluded" \
                        or "starts_before_session" not in w["exclusion_reasons"]:
                    raise ContractError(path, "a window starting before the session is clipped to "
                                        "[0, onset - gap) and excluded as starts_before_session")
                _times(w, path, allow_empty=True)
            if not w["preprocessing"]["truncated_at_onset"]:
                raise ContractError(f"{path}.preprocessing", "pre-event windows must be computed "
                                    "from data truncated at onset")
            if not set(w["exclusion_reasons"]) <= _PRE_REASONS:
                raise ContractError(f"{path}.exclusion_reasons", "not a pre-event exclusion reason")
            pre_for[w["episode_id"]] = pre_for.get(w["episode_id"], 0) + 1
            if w["status"] == "included":
                included_pre.setdefault(w["session"], []).append(span)
        else:
            if base is None:
                raise ContractError(path, "baseline windows were not requested")
            if w["episode_id"] is not None or parsed["length_frames"] != base["length_frames"] \
                    or span != (parsed["start_frame"], parsed["start_frame"] + base["length_frames"]) \
                    or w["start_frame"] % base["length_frames"]:
                raise ContractError(path, "baseline tiles are [k*length, (k+1)*length) of the request")
            if not set(w["exclusion_reasons"]) <= _BASELINE_REASONS:
                raise ContractError(f"{path}.exclusion_reasons", "not a baseline exclusion reason")
            _times(w, path)
            if w["session"] not in {parse_episode_id(e)["session"] for e in requested}:
                raise ContractError(f"{path}.session", "not a session of the requested episodes")
            if w["status"] == "included":
                included_base.setdefault(w["session"], []).append(span)
        n = w["end_frame"] - w["start_frame"]
        feats = w["features"]
        if not isinstance(feats, dict) or set(feats) != set(req["features"]):
            raise ContractError(f"{path}.features", "must hold exactly the requested features")
        for name, f in feats.items():
            FEATURE_SUMMARY(f, f"{path}.features.{name}")
            if f["n_frames"] != n or f["n_usable"] > f["n_frames"]:
                raise ContractError(f"{path}.features.{name}", "frame counts inconsistent")
            if (f["n_usable"] == 0) != (f["median"] is None) or (f["median"] is None) != (f["mean"] is None):
                raise ContractError(f"{path}.features.{name}", "median / mean are null exactly when "
                                    "no frame is usable")
            if w["status"] == "included" and f["n_usable"] == 0:
                raise ContractError(f"{path}.features.{name}", "an included window needs usable frames")
        if w["label_context"] is not None:
            if sum(w["label_context"].values()) != n:
                raise ContractError(f"{path}.label_context", "label counts must add up to the window length")
            if policy is not None:
                policy.check_annotation_session(w["session"], f"{path}.label_context")
        _check_session_for_source(policy, w["session"], parsed["source"], f"{path}.session")
    missing = sorted(requested - set(pre_for))
    if missing:
        raise ContractError("result.measurements.windows", f"no pre_event window for {missing[:3]}")
    if any(n != 1 for n in pre_for.values()):
        raise ContractError("result.measurements.windows", "one pre_event window per episode")
    # Temporal rules verifiable from the request alone: episode ids carry their onset
    # frame (not their end), and every pre-event window is listed. Clearance from the
    # full extent of each episode, and overlap with episodes that were not requested,
    # cannot be checked here and remain the implementation's responsibility.
    onsets = {}
    for e in requested:
        p = parse_episode_id(e)
        onsets.setdefault(p["session"], []).append(p["start_frame"])
    all_pre = {}
    for w in windows:
        if w["kind"] == "pre_event" and w["end_frame"] > w["start_frame"]:
            all_pre.setdefault(w["session"], []).append((w["start_frame"], w["end_frame"]))
    for sid, spans in included_pre.items():
        for span in spans:
            if any(span[0] <= o < span[1] for o in onsets.get(sid, [])):
                raise ContractError("result.measurements.windows", f"{sid}: an included pre-event "
                                    f"window {span} contains a requested onset (overlaps_event)")
    for sid, spans in included_base.items():
        if len(spans) > base["max_windows_per_session"]:
            raise ContractError("result.measurements.windows",
                                f"{sid}: more included baseline windows than max_windows_per_session")
        if any(_overlaps(b, p) for b in spans for p in all_pre.get(sid, [])):
            raise ContractError("result.measurements.windows", f"{sid}: an included baseline window "
                                "overlaps a pre-event window (included or excluded)")
        for b in spans:
            for o in onsets.get(sid, []):
                if _near_onset(b, o, base["clearance_frames"]):
                    raise ContractError("result.measurements.windows", f"{sid}: included baseline "
                                        f"window {b} is within clearance_frames of onset {o}")
    any_included = bool(included_pre) or bool(included_base)
    if res["status"] == "insufficient_evidence":
        raise ContractError("result.status", "get_window_features does not judge sufficiency")
    if (res["status"] == "no_data") == any_included:
        raise ContractError("result.status", "'no_data' exactly when no window is included")


def _res_quality(req, res, policy):
    items = res["measurements"]["windows"]
    if sorted(x["window_id"] for x in items) != sorted(req["window_ids"]):
        raise ContractError("result.measurements.windows", "must answer exactly the requested windows")
    hist = {str(h) for h in req["history_frames"]}
    for i, q in enumerate(items):
        path = f"result.measurements.windows[{i}]"
        if not q["endpoint_valid_frames"] <= q["tracking_ok_frames"] <= q["frames"]:
            raise ContractError(path, "need endpoint_valid <= tracking_ok <= frames")
        u = q["usable_by_history"]
        if not isinstance(u, dict) or set(u) != hist:
            raise ContractError(f"{path}.usable_by_history", "keys must be the requested history_frames")
        prev = q["endpoint_valid_frames"]
        for h in sorted(u, key=int):
            I(0, q["frames"])(u[h], f"{path}.usable_by_history.{h}")
            if u[h] > prev:
                raise ContractError(f"{path}.usable_by_history", "cannot grow with a longer history")
            prev = u[h]
        c = q["smoothing_contamination"]
        if (c["status"] == "detected") != bool(c["frames"]):
            raise ContractError(f"{path}.smoothing_contamination", "frames listed exactly when detected")
        if q["end_frame"] - q["start_frame"] != q["frames"]:
            raise ContractError(path, "frames must equal end_frame - start_frame")
        w = parse_window_id(q["window_id"])
        if w["length_frames"] != q["frames"] or (
                w["kind"] == "baseline" and w["start_frame"] != q["start_frame"]) or (
                w["kind"] == "pre_event" and q["end_frame"] != w["onset_frame"] - w["gap_frames"]):
            raise ContractError(path, "frame range does not match the window id")
        ex = q["excluded_frames"]
        for name in ("not_tracking_ok", "endpoint_invalid"):
            fr = ex[name]
            if fr != sorted(set(fr)) or any(not q["start_frame"] <= f < q["end_frame"] for f in fr):
                raise ContractError(f"{path}.excluded_frames.{name}", "sorted unique frames inside the window")
        for name in ("frames",):
            fr = q["smoothing_contamination"][name]
            if fr != sorted(set(fr)) or any(not q["start_frame"] <= f < q["end_frame"] for f in fr):
                raise ContractError(f"{path}.smoothing_contamination.frames",
                                    "sorted unique frames inside the window")
        if set(ex["not_tracking_ok"]) & set(ex["endpoint_invalid"]):
            raise ContractError(f"{path}.excluded_frames", "a frame has one exclusion reason")
        if len(ex["not_tracking_ok"]) != q["frames"] - q["tracking_ok_frames"]:
            raise ContractError(f"{path}.excluded_frames", "not_tracking_ok must list every such frame")
        if len(ex["endpoint_invalid"]) != q["tracking_ok_frames"] - q["endpoint_valid_frames"]:
            raise ContractError(f"{path}.excluded_frames", "endpoint_invalid must list every such frame")
        if q["verdict"] == "usable_with_caveats" and q["endpoint_valid_frames"] == 0:
            raise ContractError(f"{path}.verdict", "a window without valid frames is not usable")
        _check_session_for_source(policy, w["session"], w["source"], f"{path}.window_id")
    any_usable = any(q["endpoint_valid_frames"] > 0 for q in items)
    if res["status"] == "insufficient_evidence":
        raise ContractError("result.status", "check_tracking_quality does not judge sufficiency")
    if (res["status"] == "no_data") == any_usable:
        raise ContractError("result.status", "'no_data' exactly when no window has a valid frame")


def _median(values):
    """Middle value, or the mean of the two middle values."""
    return statistics.median(values)


def _res_compare(req, res, policy):
    m = res["measurements"]
    crit = req["sufficiency"]
    if m["feature"] != req["feature"] or m["sufficiency"]["criteria"] != crit:
        raise ContractError("result.measurements", "feature / sufficiency differ from the request")
    pre_req, base_req = set(req["pre_window_ids"]), set(req["baseline_window_ids"])
    used_pre, used_base, sessions, diffs = [], [], [], []
    for i, row in enumerate(m["per_session"]):
        path = f"result.measurements.per_session[{i}]"
        sessions.append(row["session"])
        groups = (("pre", row["pre_windows"], pre_req, used_pre),
                  ("baseline", row["baseline_windows"], base_req, used_base))
        if not row["pre_windows"] and not row["baseline_windows"]:
            raise ContractError(path, "a session row needs at least one window")
        for name, wins, allowed, used in groups:
            for j, wv in enumerate(wins):
                wpath = f"{path}.{name}_windows[{j}]"
                parsed = parse_window_id(wv["window_id"])
                if wv["window_id"] not in allowed:
                    raise ContractError(wpath, "not a requested window of this group")
                if parsed["session"] != row["session"]:
                    raise ContractError(wpath, "belongs to another session")
                if wv["n_frames"] != parsed["length_frames"] or wv["n_usable"] > wv["n_frames"]:
                    raise ContractError(wpath, "frame counts do not match the window")
                if wv["n_usable"] < crit["min_usable_fraction"] * wv["n_frames"] - STAT_ABS_TOL \
                        or wv["n_usable"] == 0:
                    raise ContractError(wpath, "a used window must meet min_usable_fraction "
                                        "(exclude it as insufficient_usable_frames)")
                used.append(wv["window_id"])
            summary = row[f"{name}_summary"]
            if wins:
                if summary is None or not _close(summary, _median([w["value"] for w in wins])):
                    raise ContractError(f"{path}.{name}_summary", "must be the median of its window values")
            elif summary is not None:
                raise ContractError(f"{path}.{name}_summary", "must be null without windows")
        paired = bool(row["pre_windows"]) and bool(row["baseline_windows"])
        if (row["difference"] is not None) != paired:
            raise ContractError(path, "a difference exists exactly when both groups have windows")
        if paired:
            if not _close(row["difference"], row["pre_summary"] - row["baseline_summary"]):
                raise ContractError(path, "difference must be pre_summary - baseline_summary")
            diffs.append(row["difference"])
        first = (row["pre_windows"] + row["baseline_windows"])[0]["window_id"]
        _check_session_for_source(policy, row["session"], parse_window_id(first)["source"], path)
    if len(set(sessions)) != len(sessions):
        raise ContractError("result.measurements.per_session", "one row per session (independence)")
    for i, x in enumerate(m["excluded_windows"]):
        path = f"result.measurements.excluded_windows[{i}]"
        n_f, n_u = x["n_frames"], x["n_usable"]
        if (n_f is None) != (n_u is None) or (n_f is not None and (
                n_u > n_f or n_f != parse_window_id(x["window_id"])["length_frames"])):
            raise ContractError(path, "frame counts inconsistent")
        if x["reason"] == "insufficient_usable_frames" and (
                n_f is None or n_u >= crit["min_usable_fraction"] * n_f - STAT_ABS_TOL and n_u > 0):
            raise ContractError(path, "insufficient_usable_frames needs counts below min_usable_fraction")
    for i, x in enumerate(m["excluded_windows"]):
        kind = parse_window_id(x["window_id"])["kind"]
        if x["reason"] not in REASONS_BY_KIND[kind]:
            raise ContractError(f"result.measurements.excluded_windows[{i}].reason",
                                f"{x['reason']!r} is impossible for a {kind} window")
    excluded = [x["window_id"] for x in m["excluded_windows"]]
    accounted = used_pre + used_base + excluded
    if len(accounted) != len(set(accounted)) or set(accounted) != pre_req | base_req:
        raise ContractError("result.measurements", "every requested window must be used or excluded "
                            "exactly once, and no other window may appear")
    if m["n_windows_pre"] != len(used_pre) or m["n_windows_baseline"] != len(used_base):
        raise ContractError("result.measurements", "window counts differ from per_session")
    if m["n_sessions_paired"] != len(diffs):
        raise ContractError("result.measurements.n_sessions_paired", "differs from per_session")

    # every sufficiency criterion, recomputed from the reported measurements
    unmet = []
    if len(diffs) < crit["min_sessions"]:
        unmet.append("min_sessions")
    if len(used_pre) < crit["min_windows_per_group"]:
        unmet.append("min_windows_per_group_pre")
    if len(used_base) < crit["min_windows_per_group"]:
        unmet.append("min_windows_per_group_baseline")
    met = not unmet
    eff, suf = m["effect"], m["sufficiency"]
    if suf["met"] != met:
        raise ContractError("result.measurements.sufficiency.met",
                            f"must be {met} (unmet: {unmet or 'none'})")
    if suf["reasons_not_met"] != unmet:
        raise ContractError("result.measurements.sufficiency.reasons_not_met",
                            f"must be exactly {unmet} (recomputed, in this order, no duplicates)")
    expected_status = "no_data" if not m["per_session"] else ("ok" if met else "insufficient_evidence")
    if res["status"] != expected_status:
        raise ContractError("result.status", f"must be {expected_status!r} for these measurements")
    values = (eff["value"], eff["ci_low"], eff["ci_high"], eff["ci_level"])
    if res["status"] != "ok":
        if any(v is not None for v in values):
            raise ContractError("result.measurements.effect", "no effect or interval unless 'ok'")
        return
    if eff["value"] is None or not _close(eff["value"], _median(diffs)):
        raise ContractError("result.measurements.effect.value",
                            "must be the median of the paired session differences")
    u = req["uncertainty"]
    if u["method"] == "none":
        if any(v is not None for v in values[1:]):
            raise ContractError("result.measurements.effect", "no interval without an uncertainty method")
        return
    if None in values or eff["ci_level"] != u["level"]:
        raise ContractError("result.measurements.effect", "bootstrap needs an interval at the requested level")
    lo, hi = min(diffs), max(diffs)
    if not (eff["ci_low"] <= eff["value"] <= eff["ci_high"]) or \
            eff["ci_low"] < lo - STAT_ABS_TOL or eff["ci_high"] > hi + STAT_ABS_TOL:
        raise ContractError("result.measurements.effect", "the interval must contain the value and lie "
                            "within the range of session differences")


_RESULT_CHECKS = {"query_events": _res_query, "get_window_features": _res_features,
                  "check_tracking_quality": _res_quality, "compare_windows": _res_compare}


# ---- the replaceable interface ----

@runtime_checkable
class InvestigationTools(Protocol):
    """What any implementation (ours, Ava's) provides. Each method receives a
    validated request dict (a private copy) and returns a result dict built
    with make_result(); call_tool() validates both sides."""

    implementation: str
    implementation_version: str

    def query_events(self, request: dict) -> dict: ...

    def get_window_features(self, request: dict) -> dict: ...

    def check_tracking_quality(self, request: dict) -> dict: ...

    def compare_windows(self, request: dict) -> dict: ...


class ToolContractViolation(RuntimeError):
    """An implementation returned a result that breaks the contract (a bug in
    the implementation, never shown to the agent as evidence)."""


def _provenance_of(tools):
    return {"implementation": str(getattr(tools, "implementation", "unknown"))[:120] or "unknown",
            "implementation_version": str(getattr(tools, "implementation_version", "unknown"))[:40] or "unknown",
            "contract_version": CONTRACT_VERSION, "code_sha256": None, "data": []}


def call_tool(tools, tool, request, policy):
    """Validate the request, call the implementation, validate the result.

    `policy` (a manifest-backed EvidencePolicy) is required. Raises
    ContractError for an invalid request (nothing is called). An exception
    inside the implementation becomes an "error" result with only the
    exception's class name. A result that breaks the contract raises
    ToolContractViolation. The returned result is a private deep copy made
    before validation, so the implementation cannot alter it afterwards."""
    validate_request(tool, request, policy)
    request = copy.deepcopy(request)  # the caller cannot change it under us either
    try:
        result = getattr(tools, tool)(copy.deepcopy(request))
    except Exception as exc:  # implementation bug or data problem; text may hold paths / secrets
        return make_result(tool, request, "error", None, provenance=_provenance_of(tools),
                           errors=[{"code": "internal_error",
                                    "message": f"{tool} failed ({type(exc).__name__})"}])
    try:
        snapshot = copy.deepcopy(result)
    except Exception as exc:
        raise ToolContractViolation(f"{tool} result cannot be copied ({type(exc).__name__})") from None
    try:
        validate_result(tool, request, snapshot, policy)
    except ContractError as exc:
        raise ToolContractViolation(f"{tool} result breaks the contract: {exc}") from exc
    return snapshot


class UnavailableTools:
    """Default implementation until real tools exist: every call answers with
    an explicit 'not_implemented' error result."""

    implementation = "unavailable"
    implementation_version = "0"

    def _unavailable(self, tool, request):
        return make_result(tool, request, "error", None, provenance=_provenance_of(self),
                           errors=[{"code": "not_implemented",
                                    "message": f"{tool} has no implementation yet"}])

    def query_events(self, request):
        return self._unavailable("query_events", request)

    def get_window_features(self, request):
        return self._unavailable("get_window_features", request)

    def check_tracking_quality(self, request):
        return self._unavailable("check_tracking_quality", request)

    def compare_windows(self, request):
        return self._unavailable("compare_windows", request)
