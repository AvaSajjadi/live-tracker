"""Contract v1 investigation tools: the four read-only tools of
docs/investigate_tool_contracts.md (Ava's PR #17), as thin adapters over the
existing Stage 9 backend (new file).

Option C (hybrid): the Stage 9 backend stays the base and is not rewritten.
This module only translates the v1 shared surface (seconds at the boundary,
ok / insufficient_evidence / error, association_only, separate counts) onto
code that already exists and is reviewed:

    query_events            monitor.run_monitor with a throwaway query rule
    get_window_features     investigation_windows (Ava's flags + dyadic features,
                            computed from the session truncated at the window end)
    check_tracking_quality  the same causal flags, judged by a named policy
    compare_windows         tools 2 + 3 composed, session-paired

`propose_rule` (tool 5) is deferred, as in the contract.

Shared surface (contract items 1-6)
-----------------------------------
1. The anchor is the bout's start_s (first frame at or above the threshold),
   never fired_s. start_s, fired_s and end_s stay separate fields.
2. Seconds outside, frames inside: every boundary goes through timebase.py
   (Ava's agreed rule: frame(t) = floor(t * 30 + 0.5), round half up).
3. Every response has status "ok" | "insufficient_evidence" | "error", with
   the contract's codes. Two codes are added and marked pending Ava:
   "invalid_request" (a malformed request: wrong type, unknown key, a value
   out of range) and "unverified_bouts" (compare_windows bouts that are not
   exactly the bouts of the cited query on this data source).
4. association_only is true on every response.
5. No tool touches researcher rules or the approval workflow (rules.RuleSet).
   query_events builds an internal query rule that is never added to a
   RuleSet, never persisted, and runs under run_id "inv_<hash>" with an
   in-memory journal; it is not a researcher rule and not an approval.
6. n_sessions, n_bouts and n_windows are reported separately. No confidence
   interval and no effect_size_dz (deferred until a session-aware method is
   agreed).

Data sources (data_source on every ok response): "synthetic" (any SYNTHETIC_
session), "ground_truth_labels" (CalMS21 validation annotations) or
"classifier" (an issued, verified validation ScoreDrop). derived_from says
what the per-frame scores are: classifier probabilities, 0/1 annotation
indicators, or synthetic numbers. Held-out test sessions are never loaded:
the real constructors go through the Stage 9 EvidencePolicy and the verified
cache, which only serve the validation split.

Finalized (approved by Ava): the overlap policy overlap_v1
------------------------------------------------------------
"Another bout" means another FIRED bout of the cited query: a monitor event
of the same session and behavior that lasted at least min_duration_s. A
threshold crossing that never fired (too short or sub-threshold) is not a
bout. The bouts are the complete, unchanged, provenance-verified result of
the cited query (_verified_bouts). Both windows are checked: a pre window
that intersects another fired bout is excluded (pre_window_overlaps_bout),
and so is a baseline window that intersects one
(baseline_window_overlaps_bout). The policy is named and versioned
(OVERLAP_POLICIES), and every compare_windows result carries its name and
definition, as with the quality policy.

Still open (see docs/contract_v1_integration.md)
------------------------------------------------
- event threshold, minimum duration and the score-to-event latency gap: the
  caller supplies threshold and min_duration_s; no latency gap is applied,
  so start_s is a classifier threshold crossing, not a behavior onset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import statistics
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from . import investigation_contracts as ic
from . import investigation_windows as iw
from . import score_drops
from . import timebase as tb
from .features import DYADIC_COLS, FEATURE_COLS
from .monitor import run_monitor
from .rules import CLASSIFIER_BEHAVIORS, Rule

__all__ = ["CONTRACT_VERSION", "ERROR_CODES", "evidence_id", "InvestigationData", "InvestigationRunner",
           "OVERLAP_POLICIES", "OverlapPolicy", "QUALITY_POLICIES", "QualityPolicy", "TOOLS", "check_tracking_quality",
           "compare_windows", "comparison_problems", "FollowUps", "get_window_features", "query_events"]

CONTRACT_VERSION = "investigate-tool-contracts/v1"
TOOLS = ("query_events", "get_window_features", "check_tracking_quality", "compare_windows")
DEFERRED_TOOLS = ("propose_rule",)
STATUSES = ("ok", "insufficient_evidence", "error")
DATA_SOURCES = ("synthetic", "ground_truth_labels", "classifier")
ERROR_CODES = ("unknown_session", "bad_window", "unknown_feature", "unknown_behavior", "no_scores",
               "budget_exhausted")
# pending Ava, not in contract v1: invalid_request (malformed request), unverified_bouts (compare_windows bouts
# that are not exactly the bouts of the cited query on this data source)
EXTENSION_ERROR_CODES = ("invalid_request", "unverified_bouts")
INSUFFICIENT_REASONS = ("too_few_bouts", "too_few_usable_windows", "window_unreliable")
# the order in which they are checked (the order of Ava's approved dry run, scripts/dryrun_contract_options.py):
# a bout is excluded for the FIRST reason that applies. The order changes which reason is reported, never which
# bouts are used
EXCLUSION_REASONS = ("pre_window_overlaps_bout", "window_before_session_start", "pre_window_unreliable",
                     "baseline_window_unreliable", "baseline_window_overlaps_bout", "no_feature_values")
SCORE_COLUMNS = score_drops.SCORE_COLUMNS
WINDOW_FEATURES = tuple(DYADIC_COLS)                  # Ava's pair features, as computed by the Stage 9 pipeline
FLAG_NAMES = ("missing", "out_of_frame", "jump", "frozen", "low_conf")
DEFAULT_MIN_N_WINDOWS = 10
DEFAULT_MIN_N_SESSIONS = 4
DEFAULT_THRESHOLD = 0.5
FOLLOW_UP_PRE_S = ic.CONTRACT_V1_PRE_S_GRID           # frozen before any effect value is seen (contract)
MAX_FOLLOW_UPS = ic.CONTRACT_V1_MAX_FOLLOW_UPS         # compare_windows analyses after the first
MAX_TOOL_CALLS = ic.CONTRACT_V1_MAX_TOOL_CALLS         # the same constants as investigation_agent.Budget
MAX_SESSIONS = 100
MAX_BOUTS = 500
QUERY_RULE_ID = "INV_QUERY"
START_MEANING = ("start_s is the first frame whose score is >= threshold with tracking_ok (a threshold crossing "
                 "of the source's scores); no score-to-event latency gap is applied, so for classifier scores it "
                 "is not the annotated behavior onset")
WORDING = "precedes / is associated with (never causes or leads to)"
USABLE_VALUE_RULE = "finite value with tracking_ok on frames t and t-1 (Stage 9 rule)"
CAUSAL_NOTE = "computed from the session truncated at t1_s"
# compare_windows' window semantics and effect definition, as reported in every result
WINDOW_DEFINITIONS = {
    "pre": "[start_s - pre_s, start_s)",
    "baseline": "[start_s - pre_s - baseline_gap_s - baseline_s, start_s - pre_s - baseline_gap_s)",
    "anchor": "start_s (never fired_s)", "truncation": "each window from the session truncated at its own end"}
MEAN_DIFF_DEFINITION = ("the MEDIAN across paired sessions of the per-session paired differences "
                        "(per_session[].paired_diff: each session's mean of its used bouts' pre - baseline); not a "
                        "mean over windows or sessions (the field name is historical and kept for compatibility)")


@dataclass(frozen=True)
class QualityPolicy:
    """A versioned window-quality rule (Ava owns the policy; the values are hers)."""
    name: str
    min_usable_fraction: float
    veto_flags: tuple = ()             # per-frame: any torso keypoint of either mouse has one of these flags
    veto_max_frames: int | None = None  # a veto run LONGER than this many frames makes the window unreliable

    def to_json(self):
        return {"name": self.name, "min_usable_fraction": self.min_usable_fraction,
                "veto_flags": list(self.veto_flags),
                "veto_max_s": None if self.veto_max_frames is None else self.veto_max_frames / tb.FPS_INT}


# quality_v1 = Ava's candidate A: usable_fraction >= 0.8 and no frozen / out_of_frame run > 0.5 s (15 frames);
# quality_v1_strict90 = candidate B, kept as the named alternative (never the default)
QUALITY_POLICIES = {
    "quality_v1": QualityPolicy("quality_v1", 0.8, ("frozen", "out_of_frame"), 15),
    "quality_v1_strict90": QualityPolicy("quality_v1_strict90", 0.9),
}
DEFAULT_QUALITY_POLICY = "quality_v1"


def quality_reasons(policy, n_frames, n_usable_frames, longest_veto_frames):
    """The reasons a window fails `policy`, in order, from its measurements (the ONE decision rule: _assess uses
    it, and anything that checks a recorded verdict can reuse it). An empty list means reliable."""
    reasons = []
    if Fraction(n_usable_frames) < Fraction(str(policy.min_usable_fraction)) * n_frames:   # exact: no float rounding
        reasons.append("usable_fraction_below_min")
    if policy.veto_flags and longest_veto_frames > policy.veto_max_frames:
        reasons.append("frozen_or_out_of_frame_run")
    return reasons


@dataclass(frozen=True)
class OverlapPolicy:
    """A versioned rule for which windows are excluded because another bout intersects them (Ava's decision)."""
    name: str
    other_bouts: str               # which bouts count
    windows: tuple                 # which windows are checked against them

    def to_json(self):
        return {"name": self.name, "other_bouts": self.other_bouts, "windows_checked": list(self.windows),
                "intersection": "half-open frame ranges [start, end) of the bout and the window share a frame",
                "not_bouts": "threshold crossings that never fired (shorter than min_duration_s or below the "
                             "threshold) are not bouts and exclude nothing"}


# overlap_v1 = Ava's approved V1 (docs/contract_v1_overlap_options.md): fired bouts; pre AND baseline windows
OVERLAP_POLICIES = {
    "overlap_v1": OverlapPolicy("overlap_v1", "the other fired bouts of the cited query (same session and behavior; "
                                              "the complete, verified query result)", ("pre", "baseline")),
}
DEFAULT_OVERLAP_POLICY = "overlap_v1"


class _Refused(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


# ======================================================================
# data
# ======================================================================

@dataclass(frozen=True)
class _Session:
    scores: pd.DataFrame        # §2 rows, frames 0..n-1
    pose: iw.SessionData        # Stage 9 pose pipeline (flags, tracking_ok, features), labels not used


class InvestigationData:
    """What the v1 tools read: per session, a §2 score table and the verified
    pose pipeline, aligned frame for frame. Built only by the constructors."""

    def __init__(self, data_source, derived_from, sessions, provenance):
        if data_source not in DATA_SOURCES:
            raise ValueError(f"data_source must be one of {DATA_SOURCES}")
        synthetic = [s for s in sessions if s.startswith("SYNTHETIC_") or "/SYNTHETIC_" in s]
        if data_source == "synthetic" and len(synthetic) != len(sessions):
            raise ValueError("a synthetic source may only hold SYNTHETIC_ sessions")
        if data_source != "synthetic" and synthetic:
            raise ValueError(f"SYNTHETIC_ sessions must be labelled data_source='synthetic', not {data_source!r}")
        if not sessions:
            raise ValueError("no sessions")
        for sid, s in sessions.items():
            _check_scores(sid, s.scores, s.pose.n_frames)
        self.data_source, self.derived_from = data_source, derived_from
        self._sessions = dict(sorted(sessions.items()))
        self.provenance = copy.deepcopy(list(provenance))
        rows = {sid: hashlib.sha256(pd.util.hash_pandas_object(s.scores, index=False).to_numpy().tobytes()).hexdigest()
                for sid, s in self._sessions.items()}
        self.fingerprint = _sha(json.dumps({"source": data_source, "derived_from": derived_from,
                                            "provenance": self.provenance, "rows": rows}, sort_keys=True, default=str))

    @property
    def sessions(self):
        return tuple(self._sessions)

    def n_frames(self, sid):
        return self._get(sid).pose.n_frames

    def _get(self, sid):
        if not isinstance(sid, str):
            raise _Refused("invalid_request", "a session id must be a string")
        if sid not in self._sessions:
            raise _Refused("unknown_session", f"unknown session {str(sid)[:128]!r}")
        return self._sessions[sid]

    # ---- constructors ----

    @classmethod
    def from_synthetic(cls, scores, poses):
        """SYNTHETIC test data: a §2 score table and a long pose table
        (calms21._session_to_long layout) over the same SYNTHETIC_ sessions."""
        sessions = {}
        for sid, g in scores.groupby("session", sort=True):
            p = poses[poses["session"] == sid]
            n = int(p["frame"].max()) + 1 if len(p) else 0
            sessions[sid] = _Session(g.reset_index(drop=True)[list(SCORE_COLUMNS)],
                                     iw._session_from_poses(sid, p, n, None))
        if set(poses["session"]) != set(sessions):
            raise ValueError("scores and poses must cover the same sessions")
        return cls("synthetic", "synthetic_scores", sessions,
                   [{"kind": "synthetic", "note": "SYNTHETIC test data, not CalMS21 and not model output"}])

    @classmethod
    def from_classifier(cls, drop, annotation_source, policy):
        """Classifier probabilities from an issued, verified validation ScoreDrop
        (score_drops.load_validation_scores), with pose from the verified
        validation cache (annotations are never read for this source)."""
        from .investigation_sources import _drop_sessions
        rows_sha = score_drops.trusted_rows_sha256(drop)
        if rows_sha is None:
            raise ValueError("the score drop is not an issued, verified validation drop")
        sids = _drop_sessions(drop, policy)                  # exactly the policy's validation split
        snap = annotation_source.snapshot(sids)
        table = drop.scores()
        sessions = {}
        for sid in sids:
            pose = iw.load_pose_session(sid, snap.sessions[sid])
            g = table[table["session"] == sid].sort_values("frame").reset_index(drop=True)
            if len(g) != pose.n_frames:
                raise ValueError(f"{sid}: the score drop has {len(g)} frames, the pose cache {pose.n_frames}")
            sessions[sid] = _Session(g[list(SCORE_COLUMNS)], pose)
        snap.finalize()
        if score_drops.trusted_rows_sha256(drop) != rows_sha:
            raise ValueError("the score drop changed while loading")
        synthetic = snap.synthetic or any("SYNTHETIC" in s for s in sids)
        prov = [{"kind": "classifier_scores", "score_drop": drop.provenance.to_json()}] + list(snap.data)
        return cls("synthetic" if synthetic else "classifier",
                   "synthetic_scores" if synthetic else "classifier_probabilities", sessions, prov)

    @classmethod
    def from_annotations(cls, annotation_source, sessions=None):
        """CalMS21 validation annotations (ground truth) as 0/1 per-frame scores,
        with the same verified pose. Label bouts are not split by tracking loss
        (tracking_ok is true in the score rows); tracking is judged by
        check_tracking_quality."""
        snap = annotation_source.snapshot(sessions)
        out = {}
        for sid in sorted(snap.sessions):
            pose = iw.load_session(sid, snap.sessions[sid])
            f = np.arange(pose.n_frames)
            g = pd.DataFrame({"session": sid, "frame": f, "time_s": f / tb.FPS_INT,
                              **{b: (pose.labels == b).astype(float) for b in ("attack", "mount", "investigation")},
                              "tracking_ok": True})
            out[sid] = _Session(g[list(SCORE_COLUMNS)], pose)
        snap.finalize()
        synthetic = snap.synthetic
        return cls("synthetic" if synthetic else "ground_truth_labels",
                   "synthetic_scores" if synthetic else "annotation_indicators", out, list(snap.data))


def _check_scores(sid, g, n):
    if list(g.columns) != list(SCORE_COLUMNS):
        raise ValueError(f"{sid}: score columns must be exactly {SCORE_COLUMNS}")
    frames = g["frame"].to_numpy()
    if len(g) != n or not np.array_equal(frames, np.arange(n)):
        raise ValueError(f"{sid}: score frames must be 0..{n - 1}, aligned with the pose")
    if not np.allclose(g["time_s"].to_numpy(float), frames / tb.FPS_INT, rtol=0, atol=score_drops.TIME_TOL):
        raise ValueError(f"{sid}: time_s is not frame / {tb.FPS_INT}")
    if set(g["session"]) != {sid}:
        raise ValueError(f"{sid}: mixed sessions")


# ======================================================================
# helpers
# ======================================================================

def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def evidence_id(tool, request, result):
    """The deterministic identity of one executed call: "ev:<tool>:<first 24 hex of the SHA-256>" of the canonical
    {tool, request, result}. Altering the request or the result changes it."""
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}")
    return f"ev:{tool}:{_sha(_canonical({'tool': tool, 'request': request, 'result': result}))[:24]}"


def _base(data):
    return {"data_source": data.data_source, "derived_from": data.derived_from,
            "contract_version": CONTRACT_VERSION, "association_only": True}


def _error(code, message):
    return {"status": "error", "code": code, "message": str(message)[:500], "association_only": True,
            "contract_version": CONTRACT_VERSION}


def _check_keys(request, required, optional=()):
    if not isinstance(request, dict):
        raise _Refused("invalid_request", "the request must be a JSON object")
    unknown = set(request) - set(required) - set(optional)
    if unknown:
        raise _Refused("invalid_request", f"unknown fields {sorted(unknown)}")
    missing = [k for k in required if k not in request]
    if missing:
        raise _Refused("invalid_request", f"missing fields {missing}")


def _finite(value):
    """float(value) for a real number (not bool) that is finite as a float; None otherwise
    (wrong type, NaN, infinity, or an integer too large for a float)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        v = float(value)
    except OverflowError:
        return None
    return v if math.isfinite(v) else None


def _number(value, name, lo=None, hi=None, lo_open=False):
    v = _finite(value)
    if v is None:
        raise _Refused("invalid_request", f"{name} must be a finite number")
    if lo is not None and (v < lo or (lo_open and v == lo)) or hi is not None and v > hi:
        raise _Refused("invalid_request", f"{name} {v!r} is out of range")
    return v


def _int(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not lo <= value <= hi:
        raise _Refused("invalid_request", f"{name} must be an integer in [{lo}, {hi}]")
    return int(value)


def _string(value, name):
    if not isinstance(value, str):
        raise _Refused("invalid_request", f"{name} must be a string")
    return value


def _behavior(value):
    if _string(value, "behavior") not in CLASSIFIER_BEHAVIORS:
        raise _Refused("unknown_behavior", f"behavior must be one of {sorted(CLASSIFIER_BEHAVIORS)}")
    return value


def _feature(value):
    if _string(value, "feature") in WINDOW_FEATURES:
        return value
    if value in FEATURE_COLS:
        raise _Refused("unknown_feature", f"{value!r} is a per-mouse feature; this adapter serves the pair "
                       f"features {list(WINDOW_FEATURES)} (per-mouse columns need a mouse argument: deferred)")
    raise _Refused("unknown_feature", f"unknown feature {str(value)[:80]!r}; available: {list(WINDOW_FEATURES)}")


def _policy(name):
    if _string(name, "quality_policy") not in QUALITY_POLICIES:
        raise _Refused("invalid_request", f"quality_policy must be one of {sorted(QUALITY_POLICIES)}")
    return QUALITY_POLICIES[name]


def _window(sess, t0_s, t1_s):
    try:
        return tb.window_frames(t0_s, t1_s, n_frames=sess.pose.n_frames)
    except tb.WindowError as exc:
        raise _Refused("bad_window", str(exc)) from None


def _guard(fn):
    def tool(data, request):
        try:
            return fn(data, request)
        except _Refused as exc:
            return _error(exc.code, exc.message)
    tool.__name__, tool.__doc__ = fn.__name__, fn.__doc__
    return tool


# ---- causal window measurements (reuse investigation_windows) ----

def _prefix_flags(pose, stop):
    """Torso flags of frames [0, stop) equal to flag_tracking on the session
    truncated at `stop` (the frozen flag looks ahead; same check as
    investigation_windows.prefix_arrays)."""
    if stop == pose.n_frames:
        return pose.flags
    m = iw.FROZEN_MIN_FRAMES
    frame = pose.torso["frame"]
    tail = iw._flags(pose.torso[(frame >= max(0, stop - 2 * m)) & (frame < stop)])
    lo = max(0, stop - m)
    full = pose.flags[(pose.flags["frame"] >= lo) & (pose.flags["frame"] < stop)]
    if iw._flag_key(tail[tail["frame"] >= lo]).equals(iw._flag_key(full)):
        return pose.flags[pose.flags["frame"] < stop]
    return iw._flags(pose.torso[frame < stop])


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _assess(pose, first, stop, policy):
    """Quality of frames [first, stop), from the session truncated at stop."""
    ok, _, _ = iw.prefix_arrays(pose, stop)
    n = stop - first
    n_ok = int(ok[first:stop].sum())
    flags = _prefix_flags(pose, stop)
    w = flags[(flags["frame"] >= first) & (flags["frame"] < stop)]
    fired = [f for f in FLAG_NAMES if f in w.columns and bool(w[f].any())]
    longest = 0
    if policy.veto_flags:
        bad = w.assign(_b=w[list(policy.veto_flags)].any(axis=1)).groupby("frame")["_b"].any()
        bad = bad.reindex(range(first, stop), fill_value=False).to_numpy(bool)
        longest = _longest_run(bad)
    reasons = quality_reasons(policy, n, n_ok, longest)
    return {"usable_fraction": n_ok / n, "n_frames": n, "n_usable_frames": n_ok, "flags_fired": fired,
            "longest_veto_run_s": longest / tb.FPS_INT if policy.veto_flags else None,
            "verdict": "unreliable" if reasons else "reliable", "reason": reasons[0] if reasons else None,
            "reasons": reasons, "quality_policy": policy.name}


def _values(pose, first, stop, feature):
    """Usable values of `feature` on [first, stop), from the session truncated
    at stop (finite, tracking_ok at t and t-1: investigation_windows' rule)."""
    ok, feats, _ = iw.prefix_arrays(pose, stop)
    usable = iw._endpoint_valid(ok, first, stop)
    v = feats[feature][first:stop]
    return v[usable & np.isfinite(v)]


# ======================================================================
# 1. query_events
# ======================================================================

def _query_args(data, request):
    """The canonical query (sorted sessions, floats) of a query_events request."""
    _check_keys(request, ("session", "behavior", "min_duration_s"), ("threshold",))
    sids = request["session"]
    sids = [sids] if isinstance(sids, str) else sids
    if not isinstance(sids, list) or not sids or len(sids) > MAX_SESSIONS or \
            not all(isinstance(x, str) for x in sids) or len(set(sids)) != len(sids):
        raise _Refused("invalid_request", f"session must be a session id or a list of 1-{MAX_SESSIONS} distinct ids")
    sessions = {sid: data._get(sid) for sid in sids}
    behavior = _behavior(request["behavior"])
    min_d = _number(request["min_duration_s"], "min_duration_s", 0.0, 3600.0, lo_open=True)
    threshold = _number(request.get("threshold", DEFAULT_THRESHOLD), "threshold", 0.0, 1.0, lo_open=True)
    for sid, s in sessions.items():
        if not np.isfinite(s.scores[behavior].to_numpy(float)).any():
            raise _Refused("no_scores", f"{sid}: no {behavior} scores")
    return {"session": sorted(sessions), "behavior": behavior, "min_duration_s": min_d, "threshold": threshold}


def _query_run_id(data, args):
    """inv_<hash> of the canonical query AND the data source's fingerprint (its
    provenance and every score row): another source, changed data or other
    parameters give another id."""
    return "inv_" + _sha(_canonical(args) + data.fingerprint)[:16]


def _run_query(data, args):
    run_id = _query_run_id(data, args)
    rule = Rule(QUERY_RULE_ID, args["behavior"], args["min_duration_s"], alert_threshold=args["threshold"],
                ask_threshold=args["threshold"], alert_mode="log",
                description="internal investigation query (not a researcher rule)", status="approved")
    rows = pd.concat([data._get(sid).scores for sid in args["session"]], ignore_index=True)
    events = run_monitor(rows, [rule], run_id=run_id)          # journal_path=None: in memory only
    bouts = [{"event_id": e["event_id"], "session": e["session"], "behavior": e["behavior"],
              "start_s": e["start_s"], "fired_s": e["evidence"]["fired_s"], "end_s": e["end_s"],
              "end_reason": e["evidence"]["end_reason"], "confidence": e["confidence"],
              "bout_mean_score": e["evidence"]["bout_mean_score"]} for e in events]
    return run_id, bouts


@_guard
def query_events(data, request):
    """Monitor bouts of one behavior under a throwaway query rule (read-only)."""
    args = _query_args(data, request)
    run_id, bouts = _run_query(data, args)
    return {"status": "ok", **_base(data), "bouts": bouts, "n_bouts": len(bouts),
            "n_sessions": len({b["session"] for b in bouts}), "n_sessions_searched": len(args["session"]),
            "behavior": args["behavior"], "min_duration_s": args["min_duration_s"], "threshold": args["threshold"],
            "query": copy.deepcopy(args), "query_run_id": run_id, "latency_gap_s": None,
            "start_meaning": START_MEANING}


# ======================================================================
# 2. get_window_features
# ======================================================================

@_guard
def get_window_features(data, request):
    """Summaries of pair features over [t0_s, t1_s) (no per-frame rows)."""
    _check_keys(request, ("session", "t0_s", "t1_s", "cols"))
    s = data._get(request["session"])
    cols = request["cols"]
    if not isinstance(cols, list) or not cols or len(set(map(str, cols))) != len(cols):
        raise _Refused("invalid_request", "cols must be a non-empty list of distinct feature names")
    cols = [_feature(c) for c in cols]
    first, stop = _window(s, request["t0_s"], request["t1_s"])
    ok, _, _ = iw.prefix_arrays(s.pose, stop)
    summary = {}
    for c in cols:
        v = _values(s.pose, first, stop, c)
        summary[c] = ({"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}
                      if v.size else {"mean": None, "min": None, "max": None, "n": 0})
    return {"status": "ok", **_base(data), "session": request["session"], "t0_s": float(request["t0_s"]),
            "t1_s": float(request["t1_s"]), "n_frames": stop - first, "n_usable_frames": int(ok[first:stop].sum()),
            "summary": summary, "usable_value_rule": USABLE_VALUE_RULE, "causal": CAUSAL_NOTE}


# ======================================================================
# 3. check_tracking_quality
# ======================================================================

@_guard
def check_tracking_quality(data, request):
    """Window verdict under a named, versioned quality policy (default quality_v1)."""
    _check_keys(request, ("session", "t0_s", "t1_s"), ("quality_policy",))
    s = data._get(request["session"])
    policy = _policy(request.get("quality_policy", DEFAULT_QUALITY_POLICY))
    first, stop = _window(s, request["t0_s"], request["t1_s"])
    a = _assess(s.pose, first, stop, policy)
    return {"status": "ok", **_base(data), "session": request["session"], "t0_s": float(request["t0_s"]),
            "t1_s": float(request["t1_s"]), **a, "quality_policy_definition": policy.to_json()}


# ======================================================================
# 4. compare_windows
# ======================================================================

def _verified_bouts(data, request):
    """The bouts of compare_windows, bound to authentic query evidence.

    The caller cites a query (query_run_id and its canonical parameters) and
    passes its bouts. Nothing the caller sends is trusted as proof: the run id
    is recomputed from the parameters and THIS data source's fingerprint, the
    query is re-run through the same monitor (deterministic, in memory), and
    the bouts must equal its result exactly, every field of every bout in
    order. A changed behavior, time, event id, session or parameter, a bout
    from another source or from changed data, a subset or an addition is
    refused (unverified_bouts). Only the recomputed bouts are used."""
    run_id, query, items = request["query_run_id"], request["query"], request["bouts"]
    if not isinstance(run_id, str) or not isinstance(query, dict):
        raise _Refused("invalid_request", "query_run_id must be a string and query the query object returned by "
                       "query_events")
    if not isinstance(items, list) or not items or len(items) > MAX_BOUTS:
        raise _Refused("invalid_request", f"bouts must be a list of 1-{MAX_BOUTS} query_events bouts")
    args = _query_args(data, query)
    if _query_run_id(data, args) != run_id:
        raise _Refused("unverified_bouts", "query_run_id does not belong to this query on this data source "
                       "(altered parameters, another source or changed data)")
    _, authentic = _run_query(data, args)
    if items != authentic:
        n = min(len(items), len(authentic))
        where = next((i for i in range(n) if items[i] != authentic[i]), n)
        raise _Refused("unverified_bouts", f"the bouts are not exactly the {len(authentic)} bouts of query "
                       f"{run_id} (first difference at bouts[{where}], {len(items)} given); pass them unchanged")
    out = []
    for b in authentic:
        out.append({"session": b["session"], "behavior": b["behavior"], "start_s": b["start_s"],
                    "start": tb.boundary_frame(b["start_s"]), "end": tb.boundary_frame(b["end_s"])})
    return out


def _duration_s(value, name, allow_zero=False):
    """A window length in seconds: finite, > 0 (>= 0 with allow_zero). It is
    never converted to frames on its own (see _bout_windows)."""
    v = _finite(value)
    if v is None:
        raise _Refused("bad_window", f"{name} must be a finite number of seconds")
    if v < 0 or (v == 0 and not allow_zero):
        raise _Refused("bad_window", f"{name} must be {'>= 0' if allow_zero else '> 0'} seconds")
    return v


def _bout_windows(i, bout, pre_s, base_s, gap_s):
    """(S, p0, b0, b1, before_start) for one bout. The absolute endpoints are
    formed in seconds first,

        pre      = [start_s - pre_s, start_s)
        baseline = [start_s - pre_s - gap_s - base_s, start_s - pre_s - gap_s)

    and EACH endpoint is converted with Ava's rule frame(t) = floor(t * 30 + 0.5)
    (timebase.boundary_frame); rounded duration counts are never subtracted.
    before_start: the baseline starts before 0 s. A window whose two converted
    endpoints are equal (or reversed) has no frame: bad_window."""
    start_s = bout["start_s"]
    p0_s = start_s - pre_s
    b1_s = p0_s - gap_s
    b0_s = b1_s - base_s
    try:
        S, p0, b1, b0 = (tb.boundary_frame(t) for t in (start_s, p0_s, b1_s, b0_s))
    except tb.WindowError as exc:
        raise _Refused("bad_window", f"bouts[{i}]: {exc}") from None
    for kind, a, z in (("pre", p0, S), ("baseline", b0, b1)):
        if z <= a:
            raise _Refused("bad_window", f"bouts[{i}] (start_s {start_s!r}): the {kind} window converts to the empty "
                           f"frame range [{a}, {z}) at {tb.FPS_INT} fps; use a longer window")
    return S, p0, b0, b1, b0_s < 0


def _gate(request):
    n_w = _int(request.get("min_n_windows", DEFAULT_MIN_N_WINDOWS), "min_n_windows", 1, MAX_BOUTS)
    n_s = _int(request.get("min_n_sessions", DEFAULT_MIN_N_SESSIONS), "min_n_sessions", 1, MAX_SESSIONS)
    why = request.get("gate_override_reason")
    lowered = n_w < DEFAULT_MIN_N_WINDOWS or n_s < DEFAULT_MIN_N_SESSIONS
    if lowered and (not isinstance(why, str) or not why.strip()):
        raise _Refused("invalid_request", f"a gate below the defaults ({DEFAULT_MIN_N_WINDOWS} windows, "
                       f"{DEFAULT_MIN_N_SESSIONS} sessions) needs a documented gate_override_reason")
    return {"min_n_windows": n_w, "min_n_sessions": n_s, "overridden": lowered,
            "override_reason": why.strip()[:300] if lowered else None}


def _gate_verdict(n_input, n_windows, n_sessions, gate):
    """The evidence gate's decision: None when it is met, else the insufficient_evidence fields."""
    if n_input < gate["min_n_windows"]:
        return {"reason": "too_few_bouts", "n_available": n_input, "n_required": gate["min_n_windows"],
                "failed_gate": "min_n_windows"}
    for name, have in (("min_n_windows", n_windows), ("min_n_sessions", n_sessions)):
        if have < gate[name]:
            return {"reason": "too_few_usable_windows", "n_available": have, "n_required": gate[name],
                    "failed_gate": name}
    return None


@_guard
def compare_windows(data, request):
    """Pre-bout vs baseline windows anchored on start_s, paired by session."""
    _check_keys(request, ("query_run_id", "query", "bouts", "pre_s", "baseline_s", "baseline_gap_s", "feature"),
                ("min_n_windows", "min_n_sessions", "gate_override_reason", "quality_policy"))
    feature = _feature(request["feature"])
    policy = _policy(request.get("quality_policy", DEFAULT_QUALITY_POLICY))
    pre_s = _duration_s(request["pre_s"], "pre_s")
    base_s = _duration_s(request["baseline_s"], "baseline_s")
    gap_s = _duration_s(request["baseline_gap_s"], "baseline_gap_s", allow_zero=True)
    gate = _gate(request)
    bouts = _verified_bouts(data, request)
    overlap = OVERLAP_POLICIES[DEFAULT_OVERLAP_POLICY]
    windows = [_bout_windows(i, b, pre_s, base_s, gap_s) for i, b in enumerate(bouts)]

    def hits_other_bout(k, a, z):          # [a, z) intersects another fired bout of the query (overlap policy)
        b = bouts[k]
        return any(j != k and o["session"] == b["session"] and o["behavior"] == b["behavior"]
                   and o["end"] > a and o["start"] < z for j, o in enumerate(bouts))

    used, excluded = [], []
    for k, (b, (S, p0, b0, b1, before_start)) in enumerate(zip(bouts, windows)):
        pose = data._get(b["session"]).pose
        why = None
        if "pre" in overlap.windows and hits_other_bout(k, p0, S):
            why = "pre_window_overlaps_bout"
        elif before_start:
            why = "window_before_session_start"
        elif _assess(pose, p0, S, policy)["verdict"] != "reliable":
            why = "pre_window_unreliable"
        elif _assess(pose, b0, b1, policy)["verdict"] != "reliable":
            why = "baseline_window_unreliable"
        elif "baseline" in overlap.windows and hits_other_bout(k, b0, b1):
            why = "baseline_window_overlaps_bout"
        else:
            pre, base = _values(pose, p0, S, feature), _values(pose, b0, b1, feature)
            if not pre.size or not base.size:
                why = "no_feature_values"
        if why:
            excluded.append({"session": b["session"], "start_s": b["start_s"], "why": why})
        else:
            used.append({"session": b["session"], "start_s": b["start_s"], "pre": float(pre.mean()),
                         "baseline": float(base.mean())})
    by_session = {}
    for u in used:
        by_session.setdefault(u["session"], []).append(u["pre"] - u["baseline"])
    counts = {"n_input": len(bouts), "n_used": len(used), "n_excluded": len(excluded),
              "n_sessions": len(by_session), "n_bouts": len(used), "n_windows": len(used),
              "n_baseline_windows": len(used), "n_sessions_input": len({b["session"] for b in bouts})}
    common = {**_base(data), "feature": feature, **counts, "excluded": excluded,
              "pre_s": float(request["pre_s"]), "baseline_s": float(request["baseline_s"]),
              "baseline_gap_s": float(request["baseline_gap_s"]), "quality_policy": policy.name,
              "quality_policy_definition": policy.to_json(),
              "overlap_policy": overlap.name, "overlap_policy_definition": overlap.to_json(),
              "exclusion_order": list(EXCLUSION_REASONS), "evidence_gate": gate,
              "windows": dict(WINDOW_DEFINITIONS)}
    verdict = _gate_verdict(len(bouts), counts["n_windows"], counts["n_sessions"], gate)
    if verdict:
        return {"status": "insufficient_evidence", **verdict, **common}
    per_session = [{"session": sid, "n_bouts": len(d), "paired_diff": statistics.fmean(d)}
                   for sid, d in sorted(by_session.items())]
    return {"status": "ok", **common,
            "pre_mean": statistics.fmean(u["pre"] for u in used),
            "baseline_mean": statistics.fmean(u["baseline"] for u in used),
            "mean_diff": statistics.median(p["paired_diff"] for p in per_session),
            "mean_diff_definition": MEAN_DIFF_DEFINITION,
            "per_session": per_session, "wording": WORDING}


_COMPARE_COMMON = frozenset({"data_source", "derived_from", "contract_version", "association_only", "status", "feature",
                             "n_input", "n_used", "n_excluded", "n_sessions", "n_bouts", "n_windows",
                             "n_baseline_windows", "n_sessions_input", "excluded", "pre_s", "baseline_s",
                             "baseline_gap_s", "quality_policy", "quality_policy_definition", "overlap_policy",
                             "overlap_policy_definition", "exclusion_order", "evidence_gate", "windows"})
_COMPARE_KEYS = {"ok": _COMPARE_COMMON | {"pre_mean", "baseline_mean", "mean_diff", "mean_diff_definition",
                                          "per_session", "wording"},
                 "insufficient_evidence": _COMPARE_COMMON | {"reason", "n_available", "n_required", "failed_gate"}}


def _used_sessions(bouts, excluded):
    """The sessions of the bouts a comparison used, in request order: the request's bouts without its excluded
    ones. compare_windows lists exclusions as {session, start_s, why} in bout order, so they must be a
    subsequence of the bouts with a reason from EXCLUSION_REASONS; None otherwise."""
    j, kept = 0, []
    for b in bouts:
        x = excluded[j] if j < len(excluded) else None
        if isinstance(x, dict) and set(x) == {"session", "start_s", "why"} and x["why"] in EXCLUSION_REASONS and \
                (x["session"], x["start_s"]) == (b["session"], b["start_s"]):
            j += 1
        else:
            kept.append(b["session"])
    return kept if j == len(excluded) else None


def comparison_problems(request, result):
    """The internal consistency of one saved compare_windows result with its request under Contract v1, without
    the data and without recomputing any measurement: its exact fields for its status; the request's parameters,
    the fixed policies and definitions echoed; the gate (_gate) of the request; counts that agree with each other
    and with the request's bouts; the status that the gate's own decision (_gate_verdict) gives for those counts;
    and, for "ok", exactly one per-session row for each session of the used bouts (the request's bouts without the
    documented exclusions), carrying that session's number of used bouts, and a mean_diff that is the median of
    their paired differences (its definition). Problems (empty: consistent). Consistency, not authenticity."""
    try:
        status = result.get("status")
        if status not in _COMPARE_KEYS or set(result) != _COMPARE_KEYS[status]:
            return [f"a compare_windows result with status {str(status)[:30]!r} must have exactly its fields"]
        gate = _gate(request)
        policy = request.get("quality_policy", DEFAULT_QUALITY_POLICY)
        bouts, r, problems = request["bouts"], result, []
        echoed = {"feature": request["feature"], "pre_s": _finite(request["pre_s"]),
                  "baseline_s": _finite(request["baseline_s"]), "baseline_gap_s": _finite(request["baseline_gap_s"]),
                  "quality_policy": policy, "quality_policy_definition": QUALITY_POLICIES[policy].to_json(),
                  "overlap_policy": DEFAULT_OVERLAP_POLICY,
                  "overlap_policy_definition": OVERLAP_POLICIES[DEFAULT_OVERLAP_POLICY].to_json(),
                  "exclusion_order": list(EXCLUSION_REASONS), "evidence_gate": gate,
                  "windows": dict(WINDOW_DEFINITIONS), "contract_version": CONTRACT_VERSION, "association_only": True}
        wrong = [k for k, v in echoed.items() if r[k] != v]
        if wrong:
            problems.append(f"the result does not echo its request and Contract v1's fixed method ({wrong})")
        counts = [r[k] for k in ("n_input", "n_used", "n_excluded", "n_sessions", "n_bouts", "n_windows",
                                 "n_baseline_windows", "n_sessions_input")]
        if not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in counts) or \
                not isinstance(r["excluded"], list):
            return problems + ["the result's counts are not counts"]
        used = r["n_used"]
        kept = _used_sessions(bouts, r["excluded"])
        if kept is None:
            problems.append("the excluded windows are not request bouts (in order) with a documented reason")
            kept = []
        if r["n_input"] != len(bouts) or r["n_sessions_input"] != len({b["session"] for b in bouts}) or \
                not used == r["n_bouts"] == r["n_windows"] == r["n_baseline_windows"] or \
                r["n_excluded"] != len(r["excluded"]) or used + r["n_excluded"] != r["n_input"] or \
                used != len(kept) or r["n_sessions"] != len(set(kept)):
            problems.append("the result's counts contradict each other or the request's bouts")
        verdict = _gate_verdict(r["n_input"], r["n_windows"], r["n_sessions"], gate)
        if (verdict is None) != (status == "ok"):
            problems.append(f"status {status!r} contradicts the evidence gate for {r['n_windows']} windows in "
                            f"{r['n_sessions']} sessions (at least {gate['min_n_windows']} in "
                            f"{gate['min_n_sessions']} required)")
        elif verdict is not None and {k: r[k] for k in verdict} != verdict:
            problems.append("the insufficient_evidence fields are not the gate's decision for its counts")
        elif status == "ok":
            rows = r["per_session"]
            if not isinstance(rows, list) or any(not isinstance(x, dict) or set(x) != {"session", "n_bouts",
                                                                                        "paired_diff"} or
                                                 _finite(x["paired_diff"]) is None or
                                                 isinstance(x["n_bouts"], bool) or
                                                 not isinstance(x["n_bouts"], int) or x["n_bouts"] < 1
                                                 for x in rows):
                return problems + ["per_session rows must be {session, n_bouts >= 1 (an integer), paired_diff}"]
            want = [(sid, kept.count(sid)) for sid in sorted(set(kept))]       # compare_windows' by_session
            if [(x["session"], x["n_bouts"]) for x in rows] != want:
                problems.append("the per_session rows are not the sessions and counts of the request's bouts "
                                "after the documented exclusions")
            if r["mean_diff_definition"] != MEAN_DIFF_DEFINITION or r["wording"] != WORDING or \
                    r["mean_diff"] != statistics.median(x["paired_diff"] for x in rows):
                problems.append("the effect is not the median of its per-session paired differences")
        return problems
    except (KeyError, TypeError, ValueError, AttributeError, statistics.StatisticsError, _Refused):
        return ["the compare_windows result or its request is malformed"]


# ======================================================================
# the budget (the agent's loop enforces; tools only report)
# ======================================================================

_TOOL_FUNCS = {"query_events": query_events, "get_window_features": get_window_features,
               "check_tracking_quality": check_tracking_quality, "compare_windows": compare_windows}
_FIXED_FOLLOW_UP = ("query_run_id", "query", "bouts", "baseline_s", "baseline_gap_s", "feature", "min_n_windows",
                    "min_n_sessions", "gate_override_reason", "quality_policy")


class FollowUps:
    """The fixed-follow-up rule of one investigation's compare_windows calls, data-independent: every pre_s from
    FOLLOW_UP_PRE_S; after the first analysis at most MAX_FOLLOW_UPS more, each changing only pre_s
    (_FIXED_FOLLOW_UP) to a value not yet analysed. InvestigationRunner applies it; investigation_agent_v1
    replays a saved report's comparisons through it."""

    def __init__(self):
        self.first = None
        self.calls = 0                       # compare_windows analyses that ran (status other than "error")
        self.analysed = set()                # their pre_s values

    def problem(self, req):
        """The refusal message for this request, or None."""
        if not isinstance(req, dict) or "pre_s" not in req:
            return None                                       # the tool itself reports the malformed request
        pre = _finite(req["pre_s"])
        if pre is None:
            return None                                       # ditto: invalid_request from the tool
        if pre not in FOLLOW_UP_PRE_S:
            return f"pre_s must be one of the preselected values {list(FOLLOW_UP_PRE_S)}"
        if self.first is None:
            return None
        if self.calls >= 1 + MAX_FOLLOW_UPS:
            return f"at most {MAX_FOLLOW_UPS} follow-up compare_windows analyses after the first"
        changed = [k for k in _FIXED_FOLLOW_UP if req.get(k) != self.first.get(k)]
        if changed:
            return f"a follow-up may change only pre_s (changed: {changed})"
        if pre in self.analysed:
            return f"pre_s {pre!r} was already analysed"
        return None

    def ran(self, req):
        """Record an analysis that ran (a result other than "error")."""
        if self.first is None:
            self.first = copy.deepcopy(req)
        self.calls += 1
        self.analysed.add(_finite(req["pre_s"]))


class InvestigationRunner:
    """One investigation's tool calls: at most MAX_TOOL_CALLS calls; at most
    MAX_FOLLOW_UPS compare_windows calls after the first, each changing only
    pre_s, and every compare_windows pre_s from the frozen grid
    FOLLOW_UP_PRE_S. Every call, including refused and failed ones, is
    recorded with the investigation id; results are deep copies."""

    def __init__(self, data, investigation_id, max_tool_calls=MAX_TOOL_CALLS):
        if not isinstance(investigation_id, str) or not investigation_id.strip():
            raise ValueError("investigation_id must be a non-empty string")
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or \
                not 1 <= max_tool_calls <= MAX_TOOL_CALLS:
            raise ValueError(f"max_tool_calls must be 1-{MAX_TOOL_CALLS}")
        self.data, self.investigation_id, self.max_tool_calls = data, investigation_id, max_tool_calls
        self.attempts = []
        self.follow_ups = FollowUps()

    @property
    def calls_used(self):
        return sum(1 for a in self.attempts if a["counted"])

    def call(self, tool, request):
        """Run one tool call. The attempt is recorded and counted BEFORE the tool
        runs, so a malformed or failing call always uses budget. A tool answers
        malformed requests with a structured error; an unexpected exception (a
        programming error) is recorded as status "exception" and re-raised."""
        req = copy.deepcopy(request)
        counted = self.calls_used < self.max_tool_calls
        attempt = {"investigation_id": self.investigation_id, "call": len(self.attempts) + 1,
                   "tool": str(tool)[:40], "request": req, "counted": counted, "status": "running"}
        self.attempts.append(attempt)
        if not counted:
            result = _error("budget_exhausted", f"the budget of {self.max_tool_calls} tool calls is spent")
        else:
            try:
                result = self._run(tool, req)
            except Exception as exc:
                attempt.update(status="exception", code=None, reason=type(exc).__name__)
                raise
        attempt.update(status=result["status"], code=result.get("code"), reason=result.get("reason"),
                       result_sha256=_sha(json.dumps(result, sort_keys=True, default=str)),
                       result=copy.deepcopy(result))
        return copy.deepcopy(result)

    def _run(self, tool, req):
        if not isinstance(tool, str):
            return _error("invalid_request", "the tool name must be a string")
        if tool in DEFERRED_TOOLS:
            return _error("invalid_request", f"{tool} is deferred (contract v1); rules are approved only by "
                                             "the researcher in the Rules page")
        if tool not in _TOOL_FUNCS:
            return _error("invalid_request", f"unknown tool {tool[:40]!r}; tools: {list(TOOLS)}")
        if tool == "compare_windows":
            problem = self.follow_ups.problem(req)
            if problem:
                return _error("invalid_request", problem)
        result = _TOOL_FUNCS[tool](self.data, req)
        if tool == "compare_windows" and result["status"] != "error":     # an analysis ran
            self.follow_ups.ran(req)
        return result

    def record(self):
        """The full attempt log (JSON-ready), for the investigation report."""
        return {"investigation_id": self.investigation_id, "contract_version": CONTRACT_VERSION,
                "max_tool_calls": self.max_tool_calls, "calls_used": self.calls_used,
                "data_source": self.data.data_source, "attempts": copy.deepcopy(self.attempts)}
