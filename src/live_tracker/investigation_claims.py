"""EXPERIMENTAL (Investigation mode, Findings v2 checkpoint M1): the closed,
evidence-bound claim registry `claims/2.0` (new file). Design:
docs/findings_v2_design.md sections 4-6.

A proposal (what the model will submit in M3) is structured data only:

    {"finding": "effect_described" | "insufficient_evidence" | "inconclusive",
     "primary_evidence_id": "ev:compare_windows:..." | null,
     "observations": [{"evidence_id": ..., "field": ...}],
     "claims": [{"type": ..., "evidence_ids": [...], "params": {...}}],
     "next_investigations": [{"action": ..., "params": {...}, "because": [claim indices]}]}

validate_findings() checks it against the run's evidence through the M0
shared context (L0 content consistency, L1 schema, L2 authorization,
execution, cross-snapshot identity / comparability / coverage) and returns a
frozen FindingsV2. Every claim type has a predicate that reads only the cited
snapshots and the context; its accepted result carries a structured
`grounding` (values read from the evidence) for the M2 renderer. Nothing here
writes report prose, computes a new statistic or accepts a model-supplied
number, direction or sentence.

Distinctions kept explicit in the result:
- content consistency (hash) is not authenticity; L2 `not_rechecked` (no
  policy) makes the result diagnostic, never publishable;
- authorization is not sufficiency: an authorized comparison can still be
  insufficient;
- complete coverage vs partial-but-scoped coverage (a descriptive effect
  under partial coverage carries its scope; `unknown` coverage refuses it);
- an inconclusive finding vs a supported descriptive effect.

Not in this registry version: measurement_context(mover_unidentified) (the
design grounds it in `dist_rate_meaning`, which only get_window_features
records, while the design's status table admits only compare_windows
evidence for measurement_context: withheld until the design is corrected),
alternative-feature / window-size sensitivity, grid-based actions.
"""

from __future__ import annotations

import json
import math
import re
import types
from dataclasses import dataclass
from typing import Any

from . import investigation_context as ctx
from . import investigation_contracts as ic

REGISTRY_VERSION = "claims/2.0"
GRID_VERSION = "none"
FINDINGS = ("effect_described", "insufficient_evidence", "inconclusive")
MAX_CLAIMS = 20
MAX_NEXT = 3
MAX_OBSERVATIONS = 20
MAX_LIST_VALUE = 20
PROPOSAL_KEYS = frozenset({"finding", "primary_evidence_id", "observations", "claims", "next_investigations"})
CLAIM_KEYS = frozenset({"type", "evidence_ids", "params"})
NEXT_KEYS = frozenset({"action", "params", "because"})

STATUS_CLAIMS = {"sufficiency_not_met": "insufficient_evidence", "comparison_no_data": "no_data",
                 "comparison_unavailable": "error"}
COVERAGE_ASPECTS = ("primary_comparison", "query_group", "truncated_query", "window_exclusions",
                    "sessions_without_baseline")
TRACKING_KINDS = {   # claim kind -> Stage 7 tracking-item sources (context.tracking)
    "smoothing_contamination": ("methodology.smoothing_contamination_windows_pre",
                                "methodology.smoothing_contamination_windows_baseline"),
    "usable_fraction": ("methodology.usable_frame_fraction_pre", "methodology.usable_frame_fraction_baseline"),
    "tracking_exclusions": ("measurements.windows[].exclusion_reasons", "measurements.excluded_windows"),
    "quality_verdict": ("measurements.windows[].verdict",),
    "quality_contamination_detected": ("measurements.windows[].smoothing_contamination",),
}
OPERATIONAL_KINDS = ("quality_never_called", "quality_check_refused", "quality_check_failed", "model_selected_plan")
CONFOUNDERS = ("investigation_label",)
MEASUREMENT_CONTEXT = ("session_timing", "pixel_units")
# exact registry markers: a tool_disclosure claim cites a snapshot whose own limitations contain one
DISCLOSURE_MARKERS = {
    "approach_unlabelled": {"query_events": "CalMS21 Task 1 has no 'approach' label; approach can only be measured",
                            "get_window_features": "it is not an 'approach' label (CalMS21 Task 1 has none)",
                            "compare_windows": "CalMS21 Task 1 has no 'approach' label."},
    "motion_accuracy_unverified": {"compare_windows": "Motion accuracy is not verified:",
                                   "check_tracking_quality": "not whether the measured motion is accurate"},
}
DISCLOSURE_SCOPE = {"approach_unlabelled": "dataset", "motion_accuracy_unverified": "tool"}
METHOD_LIMITATIONS = ("provisional_criteria", "provisional_parameters", "descriptive_only", "single_feature",
                      "baseline_clearance_not_rechecked")
PARAMS = {  # closed params per claim type: key -> allowed values (None: no params)
    "sufficiency_not_met": {}, "comparison_no_data": {}, "comparison_unavailable": {}, "no_comparison": {},
    "descriptive_effect": {},
    "coverage": {"aspect": COVERAGE_ASPECTS},
    "tracking_concern": {"kind": tuple(TRACKING_KINDS)},
    "operational_fact": {"kind": OPERATIONAL_KINDS},
    "potential_confounder": {"confounder": CONFOUNDERS},
    "sensitivity_excluding_flagged": {},
    "measurement_context": {"aspect": MEASUREMENT_CONTEXT},
    "tool_disclosure": {"disclosure": tuple(DISCLOSURE_MARKERS)},
    "methodological_limitation": {"kind": METHOD_LIMITATIONS},
}
CLAIM_TYPES = tuple(PARAMS)
NEXT_ACTIONS = {"check_tracking_quality": ("windows_from",), "exclude_flagged_windows": ("quality",),
                "include_more_sessions": ()}
# claims that may justify an inconclusive finding (design 5.3); measurement_context and tool_disclosure
# never do on their own
LIMITING = ("potential_confounder", "tracking_concern", "coverage", "sensitivity_excluding_flagged",
            "operational_fact", "methodological_limitation")
# effect values and window values: reportable only under effect_described (design 5.1)
EFFECT_FIELDS = re.compile(r"measurements\.(effect(\.|$)|per_session\[\d+\]\.(difference|pre_summary|"
                           r"baseline_summary|pre_windows|baseline_windows)($|[.\[])|"
                           r"windows\[\d+\]\.features\.[a-z_]+\.(median|mean)$)")
SENSITIVITY_COMPARE_ITEMS = ("window_value", "window_shape", "feature_lookback_frames", "min_usable_fraction",
                             "baseline_clearance_frames")
QUALITY_COMPATIBLE = ("event_source", "causal_computation", "tracking_flags", "score_validation", "tracking_ok",
                      "interpolation", "smoothing", "smoother_memory_frames", "fps")
_LABELS = re.compile(r"\s*(attack|investigation|mount|other) (\d+)\s*")
# Stage 5 `_spread`: "min / median / max" of the used windows' start times in seconds, each f"{v:.6g}" of a
# non-negative value (no sign; decimal or exponent form)
_NUMBER_6G = r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
_SPREAD = re.compile(rf"{_NUMBER_6G} / {_NUMBER_6G} / {_NUMBER_6G}")
# fixed and neutral whatever the values: no timing difference, drift or confounding is asserted
TIMING_NOTE = "Window-start summaries are recorded; drift is not measured; this is not evidence of confounding."


class ClaimProblem(Exception):
    """One claim cannot be accepted (the message says why)."""


# ======================================================================
# the result
# ======================================================================

@dataclass(frozen=True)
class FindingsV2:
    """A validated, frozen findings/2 representation (no prose)."""
    findings_format: str
    registry_version: str
    grid_version: str
    accepted: bool
    problems: tuple
    finding: str | None                # FINDINGS or "incomplete"
    primary_evidence_id: str | None
    accepted_direction: str | None     # the primary's effect sign, only under effect_described
    observations: tuple                # frozen {evidence_id, field, value}
    claims: tuple                      # frozen {type, evidence_ids, params, grounding}
    next_investigations: tuple         # frozen {action, params, because}
    authorization: str | None          # rechecked / not_rechecked / refused
    publishable: bool                  # accepted and authorization rechecked (Q4)
    fingerprint: str | None            # content fingerprint (M0 fp/1) of an accepted proposal

    def to_json(self):
        return {"findings_format": self.findings_format, "registry_version": self.registry_version,
                "grid_version": self.grid_version, "accepted": self.accepted, "problems": list(self.problems),
                "finding": self.finding, "primary_evidence_id": self.primary_evidence_id,
                "accepted_direction": self.accepted_direction,
                "observations": ctx.thaw(self.observations), "claims": ctx.thaw(self.claims),
                "next_investigations": ctx.thaw(self.next_investigations),
                "authorization": self.authorization, "publishable": self.publishable,
                "fingerprint": self.fingerprint}


def _refused(problems, authorization=None):
    return FindingsV2(findings_format=ctx.V2_FORMAT, registry_version=REGISTRY_VERSION, grid_version=GRID_VERSION,
                      accepted=False, problems=tuple(problems), finding=None, primary_evidence_id=None,
                      accepted_direction=None, observations=(), claims=(), next_investigations=(),
                      authorization=authorization, publishable=False, fingerprint=None)


# ======================================================================
# schema (types first, nothing interpreted until the shape is right)
# ======================================================================

def _is_str(x):
    return isinstance(x, str) and bool(x)


def _schema_problems(p):
    if not isinstance(p, (dict, types.MappingProxyType)):
        return ["the proposal is not an object"]
    out = []
    if set(p) != PROPOSAL_KEYS:
        out.append(f"the proposal has exactly the keys {sorted(PROPOSAL_KEYS)} "
                   f"(unknown {sorted(map(str, set(p) - PROPOSAL_KEYS))}, missing {sorted(PROPOSAL_KEYS - set(p))})")
        return out
    if p["finding"] not in FINDINGS or not isinstance(p["finding"], str):
        out.append(f"finding must be one of {list(FINDINGS)}")
    if p["primary_evidence_id"] is not None and not _is_str(p["primary_evidence_id"]):
        out.append("primary_evidence_id must be an evidence id or null")
    for name, keys, limit in (("observations", {"evidence_id", "field"}, MAX_OBSERVATIONS),
                              ("claims", CLAIM_KEYS, MAX_CLAIMS), ("next_investigations", NEXT_KEYS, MAX_NEXT)):
        items = p[name]
        if not isinstance(items, list):
            out.append(f"{name} must be a list")
            continue
        if len(items) > limit:
            out.append(f"{name}: at most {limit}")
        for i, it in enumerate(items):
            if not isinstance(it, dict) or set(it) != keys:
                out.append(f"{name}[{i}] has exactly the keys {sorted(keys)}")
    if out:
        return out
    for i, o in enumerate(p["observations"]):
        if not (_is_str(o["evidence_id"]) and _is_str(o["field"])):
            out.append(f"observations[{i}]: evidence_id and field are strings")
    for i, c in enumerate(p["claims"]):
        where = f"claims[{i}]"
        if c["type"] not in CLAIM_TYPES or not isinstance(c["type"], str):
            out.append(f"{where}: unknown claim type (claims/2.0 has {list(CLAIM_TYPES)})")
            continue
        if not isinstance(c["evidence_ids"], list) or not all(_is_str(e) for e in c["evidence_ids"]) or \
                len(set(c["evidence_ids"])) != len(c["evidence_ids"]):
            out.append(f"{where}: evidence_ids is a list of distinct evidence ids")
        allowed = PARAMS[c["type"]]
        if not isinstance(c["params"], dict) or set(c["params"]) != set(allowed):
            out.append(f"{where}: params has exactly the keys {sorted(allowed)}")
            continue
        for k, values in allowed.items():
            if not isinstance(c["params"][k], str) or c["params"][k] not in values:
                out.append(f"{where}: params.{k} must be one of {list(values)}")
    for i, n in enumerate(p["next_investigations"]):
        where = f"next_investigations[{i}]"
        if n["action"] not in NEXT_ACTIONS or not isinstance(n["action"], str):
            out.append(f"{where}: unknown action (claims/2.0 has {list(NEXT_ACTIONS)})")
            continue
        if not isinstance(n["params"], dict) or set(n["params"]) != set(NEXT_ACTIONS[n["action"]]) or \
                not all(_is_str(v) for v in n["params"].values()):
            out.append(f"{where}: params has exactly the string keys {list(NEXT_ACTIONS[n['action']])}")
        if not isinstance(n["because"], list) or not n["because"] or \
                not all(type(b) is int and 0 <= b < len(p["claims"]) for b in n["because"]):
            out.append(f"{where}: because lists claim indices of this proposal")
    return out


# ======================================================================
# reading the evidence
# ======================================================================

def _status(snap):
    return snap["result"]["status"]


def _measurements(snap):
    return snap["result"]["measurements"]


def _one(snaps, claim, tool, statuses, n=1):
    ids = claim["evidence_ids"]
    if len(ids) != n:
        raise ClaimProblem(f"cites exactly {n} evidence id(s)")
    out = []
    for eid in ids:
        snap = snaps.get(eid)
        if snap is None:
            raise ClaimProblem(f"{eid[:80]} is not evidence of this run")
        if snap["tool"] != tool:
            raise ClaimProblem(f"{eid[:80]} is a {snap['tool']} result, not {tool}")
        if _status(snap) not in statuses:
            raise ClaimProblem(f"{eid[:80]} has status {_status(snap)}, not {' / '.join(statuses)}")
        out.append(snap)
    return out[0] if n == 1 else out


def _sign(v):
    return ctx._sign(v)


def _label_counts(value):
    """{label: count} from a label_frames methodology value, or None when it is
    absent, null, not text, or not exactly one non-negative integer count for
    each CalMS21 label (no duplicates, signs, decimals or other labels)."""
    if not isinstance(value, str):
        return None
    out = {}
    for part in value.split(","):
        m = _LABELS.fullmatch(part)
        if not m or m.group(1) in out:
            return None
        out[m.group(1)] = int(m.group(2))
    return out if set(out) == set(ic.LABELS) else None


def _start_spread(value):
    """{"min", "median", "max"} start times (s) from a window_start_s_* value,
    or None unless it is exactly the producer's three finite, non-negative,
    ascending numbers. Arbitrary text is never measured."""
    if not isinstance(value, str):
        return None
    m = _SPREAD.fullmatch(value)
    if not m:
        return None
    lo, mid, hi = (float(x) for x in m.groups())
    if not all(math.isfinite(v) and v >= 0 for v in (lo, mid, hi)) or not lo <= mid <= hi:
        return None
    return {"min": lo, "median": mid, "max": hi}


def _used_frames(m, group):
    """Frames of the windows a comparison used in one group (sum of n_frames)."""
    return sum(w["n_frames"] for s in m["per_session"] for w in s[group])


def _used_windows(m, group):
    return {w["window_id"]: (s["session"], w["value"], w["n_frames"], w["n_usable"])
            for s in m["per_session"] for w in s[group]}


# ======================================================================
# claim predicates: return the grounding (structured, evidence-derived)
# ======================================================================

def _p_status(c, k, finding):
    status = STATUS_CLAIMS[c["type"]]
    snap = _one(k.snaps, c, "compare_windows", (status,))
    req, m = snap["request"], _measurements(snap)
    if c["type"] == "comparison_unavailable":
        return {"feature": req["feature"], "status": status,
                "error_codes": [e["code"] for e in snap["result"]["errors"]]}
    out = {"feature": m["feature"], "status": status, "n_sessions_paired": m["n_sessions_paired"],
           "n_windows_pre": m["n_windows_pre"], "n_windows_baseline": m["n_windows_baseline"]}
    if c["type"] == "sufficiency_not_met":
        if not m["sufficiency"]["reasons_not_met"]:
            raise ClaimProblem("an insufficient comparison records no unmet sufficiency reason")
        out.update(criteria=dict(m["sufficiency"]["criteria"]), reasons_not_met=list(m["sufficiency"]["reasons_not_met"]))
    return out


def _p_no_comparison(c, k, finding):
    if c["evidence_ids"]:
        raise ClaimProblem("cites no evidence")
    if any(s["tool"] == "compare_windows" for s in k.snaps.values()):
        raise ClaimProblem("a comparison exists in this run")
    return {"comparisons": 0}


def _p_descriptive(c, k, finding):
    snap = _one(k.snaps, c, "compare_windows", ("ok",))
    if snap["evidence_id"] != k.primary_id:
        raise ClaimProblem("descriptive_effect cites the primary comparison")
    if finding != "effect_described":
        raise ClaimProblem("descriptive_effect only under the finding effect_described")
    m, req = _measurements(snap), snap["request"]
    eff = m["effect"]
    if not m["sufficiency"]["met"] or eff["value"] is None:
        raise ClaimProblem("the comparison's sufficiency criteria are not met")
    if req["uncertainty"]["method"] != "none" or any(eff[x] is not None for x in ("ci_low", "ci_high", "ci_level")):
        raise ClaimProblem("outside the descriptive-only policy")
    cov = k.coverage["primary"]
    if cov is None or cov["group"] is None or cov["status"] not in ("complete", "partial"):
        raise ClaimProblem("primary coverage is not established "
                           f"({cov['reason'] if cov else 'no primary coverage'})")
    group = k.coverage["groups"][cov["group"]] if cov["group"] is not None else None
    out = {"feature": m["feature"], "value": eff["value"], "unit": ctx._unit(snap), "direction": _sign(eff["value"]),
           "n_sessions_paired": m["n_sessions_paired"], "contributing_sessions": list(cov["contributing"]),
           "coverage_status": cov["status"], "cross_snapshot_status": k.cross["status"],
           "scope": "the contributing sessions only, not the whole authorized split"}
    if cov["status"] == "partial":
        out["coverage"] = {"searched": len(group["searched"]) if group else None,
                           "eligible": group["eligible_count"] if group else None,
                           "not_searched": list(group["not_searched"] or []) if group else None,
                           "behavior": group["behavior"] if group else None, "reason": cov["reason"]}
    return out


def _p_coverage(c, k, finding):
    aspect = c["params"]["aspect"]
    if aspect == "primary_comparison":
        if c["evidence_ids"]:
            raise ClaimProblem("cites no evidence (the primary comparison's coverage comes from the context)")
        if k.coverage["primary"] is None:
            raise ClaimProblem("there is no primary comparison")
        return {"aspect": aspect, **k.coverage["primary"]}
    if aspect in ("query_group", "truncated_query"):
        snap = _one(k.snaps, c, "query_events", ("ok", "no_data"))
        group = next((g for g in k.coverage["groups"] if snap["evidence_id"] in g["query_evidence_ids"]), None)
        if aspect == "truncated_query":
            m = _measurements(snap)
            if not m["truncated"]:
                raise ClaimProblem("the query is not truncated")
            return {"aspect": aspect, "returned": len(m["episodes"]), "total": m["n_episodes_total"]}
        return {"aspect": aspect, **group}
    if aspect == "window_exclusions":
        snap = _one(k.snaps, c, "get_window_features", ("ok", "no_data"))
        windows = _measurements(snap)["windows"]
        distinct, reasons = {}, {}
        for w in windows:
            if w["status"] == "excluded":
                distinct[w["kind"]] = distinct.get(w["kind"], 0) + 1
                for r in w["exclusion_reasons"]:
                    reasons[f"{w['kind']}:{r}"] = reasons.get(f"{w['kind']}:{r}", 0) + 1
        if not distinct:
            raise ClaimProblem("no window is excluded")
        return {"aspect": aspect, "excluded_windows": dict(sorted(distinct.items())),
                "reason_counts": dict(sorted(reasons.items())), "reason_counts_can_overlap": True}
    snap = _one(k.snaps, c, "compare_windows", ("ok", "insufficient_evidence"))
    if snap["evidence_id"] != k.primary_id:
        raise ClaimProblem("sessions_without_baseline cites the primary comparison")
    sessions = [s["session"] for s in _measurements(snap)["per_session"] if s["pre_windows"] and
                not s["baseline_windows"]]
    if not sessions:
        raise ClaimProblem("every session with pre-event windows has a baseline window")
    return {"aspect": aspect, "sessions": sessions}


def _tracking_order(item, k):
    eids = list(k.snaps)
    return (not item["primary"], eids.index(item["evidence_id"]) if item["evidence_id"] in eids else len(eids),
            item["source"], tuple(item["window_ids"] or ()))


def _p_tracking(c, k, finding):
    if len(c["evidence_ids"]) > 1:
        raise ClaimProblem("cites at most one evidence id")
    sources = TRACKING_KINDS[c["params"]["kind"]]
    items = [t for t in k.tracking if t["source"] in sources
             and (not c["evidence_ids"] or t["evidence_id"] == c["evidence_ids"][0])]
    if c["evidence_ids"] and c["evidence_ids"][0] not in k.snaps:
        raise ClaimProblem(f"{c['evidence_ids'][0][:80]} is not evidence of this run")
    if not items:
        raise ClaimProblem("no recorded tracking fact of this kind")
    return {"kind": c["params"]["kind"], "category": "measured_fact",
            "items": [dict(t) for t in sorted(items, key=lambda t: _tracking_order(t, k))]}


def _p_operational(c, k, finding):
    kind = c["params"]["kind"]
    quality_calls = [t for t in k.trace if t.get("kind") == "tool_call" and t.get("tool") == "check_tracking_quality"]
    if kind == "quality_check_failed":
        snap = _one(k.snaps, c, "check_tracking_quality", ("error",))
        return {"kind": kind, "category": "operational_fact",
                "error_codes": [e["code"] for e in snap["result"]["errors"]]}
    if c["evidence_ids"]:
        raise ClaimProblem("cites no evidence (an execution-history fact)")
    if kind == "quality_never_called":
        if quality_calls:
            raise ClaimProblem("check_tracking_quality was called in this run")
        return {"kind": kind, "category": "operational_fact"}
    if kind == "quality_check_refused":
        if not quality_calls or any(t.get("outcome") != "refused" for t in quality_calls):
            raise ClaimProblem("check_tracking_quality was not attempted, or not every attempt was refused")
        return {"kind": kind, "category": "operational_fact",
                "refusals": sorted({str(t.get("refused")) for t in quality_calls})}
    turns = [t for t in k.trace if t.get("kind") == "model_turn" and t.get("model") and t.get("tool_calls")]
    if not turns:
        raise ClaimProblem("no tool call was chosen in a model turn with a recorded model id")
    return {"kind": kind, "category": "operational_fact", "model_ids": sorted({t["model"] for t in turns}),
            "tool_calls_chosen": sum(len(t["tool_calls"]) for t in turns)}


def _p_confounder(c, k, finding):
    snap = _one(k.snaps, c, "compare_windows", ("ok", "insufficient_evidence"))
    pre = _label_counts(ctx._method(snap["result"], "label_frames_pre"))
    base = _label_counts(ctx._method(snap["result"], "label_frames_baseline"))
    if pre is None or base is None or not sum(pre.values()) or not sum(base.values()):
        raise ClaimProblem("not assessed: label frames are absent, null, unparseable or empty")
    # each group's label frames must be exactly the frames of the windows it used (a re-hashed snapshot
    # is content-consistent, not authentic: impossible counts are refused here)
    m = _measurements(snap)
    for group, counts, windows in (("pre-event", pre, "pre_windows"), ("baseline", base, "baseline_windows")):
        if sum(counts.values()) != _used_frames(m, windows):
            raise ClaimProblem(f"impossible label counts: {group} label frames total {sum(counts.values())}, "
                               f"but the used {group} windows have {_used_frames(m, windows)} frames")
    # share pre > share baseline, compared exactly as integers (no rounding, no threshold)
    if pre["investigation"] * sum(base.values()) <= base["investigation"] * sum(pre.values()):
        raise ClaimProblem("contradicted: the measured investigation share before the event is not higher than "
                           "in the baseline")
    return {"confounder": "investigation_label", "category": "potential_confounder", "predicate": "measured",
            "label_frames_pre": pre, "label_frames_baseline": base,
            "rule": "investigation share pre-event > baseline (provisional, owner approval pending)"}


def _p_sensitivity(c, k, finding):
    ids = c["evidence_ids"]
    if len(ids) != 3:
        raise ClaimProblem("cites [original, variant, quality]")
    o, v = (_one(k.snaps, {"evidence_ids": [x]}, "compare_windows", ("ok", "insufficient_evidence", "no_data"))
            for x in ids[:2])
    q = _one(k.snaps, {"evidence_ids": [ids[2]]}, "check_tracking_quality", ("ok",))
    ro, rv = o["request"], v["request"]
    for key in ("feature", "statistic", "aggregation", "sufficiency", "uncertainty"):           # 1
        if ro[key] != rv[key]:
            raise ClaimProblem(f"requirement 1: {key} differs")
    if rv["baseline_window_ids"] != ro["baseline_window_ids"]:                                  # 2
        raise ClaimProblem("requirement 2: requested baselines differ")
    if not (set(rv["pre_window_ids"]) < set(ro["pre_window_ids"])):                             # 3
        raise ClaimProblem("requirement 3: the variant's pre-event windows are not a strict subset")
    mo, mv = _measurements(o), _measurements(v)
    if _used_windows(mo, "baseline_windows") != _used_windows(mv, "baseline_windows"):          # 4
        raise ClaimProblem("requirement 4: used baselines differ")
    po, pv = _used_windows(mo, "pre_windows"), _used_windows(mv, "pre_windows")
    if any(po[w] != pv[w] for w in set(po) & set(pv)):                                         # 5
        raise ClaimProblem("requirement 5: a shared pre-event window has different values")
    both = set(rv["pre_window_ids"]) | set(rv["baseline_window_ids"])                           # 6
    eo = {x["window_id"]: x["reason"] for x in mo["excluded_windows"] if x["window_id"] in both}
    ev = {x["window_id"]: x["reason"] for x in mv["excluded_windows"] if x["window_id"] in both}
    used_o = (set(po) | set(_used_windows(mo, "baseline_windows"))) & both
    used_v = (set(pv) | set(_used_windows(mv, "baseline_windows"))) & both
    if eo != ev or used_o != used_v:
        raise ClaimProblem("requirement 6: exclusion outcomes differ for windows requested by both")
    try:                                                                                        # 7
        for w in ro["pre_window_ids"] + ro["baseline_window_ids"]:
            ic.parse_window_id(w)
    except ic.ContractError:
        raise ClaimProblem("requirement 7: a window id does not parse to window bounds") from None
    for x, y in ((o, v), (o, q), (v, q)):                                                       # 8 identity
        problem = ctx._data_identity_problem(x, y)
        if problem:
            raise ClaimProblem(f"requirement 8: {problem} between {x['evidence_id']} and {y['evidence_id']}")

    def items(snap, names):
        rec = {m["name"]: (json.dumps(m["value"]), m["status"]) for m in snap["result"]["methodology"]}
        return {n: rec.get(n) for n in names}
    for names, req in ((ctx.PROCESSING_METHOD + SENSITIVITY_COMPARE_ITEMS, "requirement 8"),):
        io, iv = items(o, names), items(v, names)
        if any(io[n] is None for n in names) or io != iv:
            raise ClaimProblem(f"{req}: processing differs or is not recorded between original and variant")
    iq, io = items(q, QUALITY_COMPATIBLE), items(o, QUALITY_COMPATIBLE)                          # 9
    if any(iq[n] is None or io[n] is None for n in QUALITY_COMPATIBLE) or iq != io:
        raise ClaimProblem("requirement 9: the quality assessment configuration is not compatible")
    removed = sorted(set(ro["pre_window_ids"]) - set(rv["pre_window_ids"]))
    assessed = {w["window_id"]: w for w in _measurements(q)["windows"]}
    for w in removed:
        a = assessed.get(w)
        if a is None:
            raise ClaimProblem("requirement 9: a removed window was not assessed by the quality check")
        if not (a["smoothing_contamination"]["status"] == "detected" or a["verdict"] != "usable_with_caveats"):
            raise ClaimProblem("requirement 9: a removed window is not flagged by the quality check")
    contributing = [w for w in removed if w in po]
    excluded_o = {x["window_id"]: x["reason"] for x in mo["excluded_windows"]}
    paired = [{s["session"] for s in m["per_session"] if s["difference"] is not None} for m in (mo, mv)]
    with_pre = [{s["session"] for s in m["per_session"] if s["pre_windows"]} for m in (mo, mv)]
    out = {"variant": "excluding_flagged_windows", "original": o["evidence_id"], "variant_evidence": v["evidence_id"],
           "quality": q["evidence_id"], "statuses": [_status(o), _status(v)],
           "reasons_not_met": [list(mo["sufficiency"]["reasons_not_met"]), list(mv["sufficiency"]["reasons_not_met"])],
           "n_sessions_paired": [mo["n_sessions_paired"], mv["n_sessions_paired"]],
           "removed_contributing": [{"window_id": w, "flags": {"contamination": assessed[w]["smoothing_contamination"]
                                                               ["status"], "verdict": assessed[w]["verdict"]}}
                                    for w in contributing],
           "removed_previously_excluded": [{"window_id": w, "original_reason": excluded_o.get(w)}
                                           for w in removed if w not in po],
           "sessions_pairing_changed": sorted(paired[0] ^ paired[1]),
           "sessions_lost_all_pre": sorted(with_pre[0] - with_pre[1])}
    if finding == "effect_described" and _status(o) == "ok" and _status(v) == "ok":            # values only here
        uo, uv = ctx._unit(o), ctx._unit(v)
        out["effects"] = [{"value": mo["effect"]["value"], "unit": uo}, {"value": mv["effect"]["value"], "unit": uv}]
        signs = [_sign(mo["effect"]["value"]), _sign(mv["effect"]["value"])]
        out["arithmetic_sign_relation"] = (None if uo is None or uv is None or uo != uv else
                                           "a zero value" if "zero" in signs else
                                           "same arithmetic sign" if signs[0] == signs[1] else "opposite arithmetic signs")
        out["sign_relation_note"] = "an arithmetic comparison of two descriptive values, not scientific agreement"
    return out


def _p_measurement_context(c, k, finding):
    snap = _one(k.snaps, c, "compare_windows", ("ok", "insufficient_evidence", "no_data"))
    aspect = c["params"]["aspect"]
    if aspect == "session_timing":
        pre, base = (_start_spread(ctx._method(snap["result"], f"window_start_s_{g}")) for g in ("pre", "baseline"))
        if pre is None or base is None:
            raise ClaimProblem("not assessed: window start times are not recorded as min / median / max seconds")
        # grounded only in the parsed numbers; the producer's text is never passed on
        return {"aspect": aspect, "category": "measured_fact", "unit": "s",
                "window_start_s_pre": pre, "window_start_s_baseline": base,
                "scope": "start times of the windows this comparison used (min / median / max)",
                "note": TIMING_NOTE}
    unit = ctx._unit(snap)
    if unit is None:
        raise ClaimProblem("not assessed: the comparison records no unit")
    if "px" not in unit:
        raise ClaimProblem(f"contradicted: the comparison's unit is {unit!r}, not image pixels")
    return {"aspect": aspect, "category": "measured_fact", "unit": unit,
            "note": "image pixels, not calibrated between sessions; not a confounder"}


def _p_tool_disclosure(c, k, finding):
    disclosure = c["params"]["disclosure"]
    markers = DISCLOSURE_MARKERS[disclosure]
    if len(c["evidence_ids"]) != 1:
        raise ClaimProblem("cites exactly 1 evidence id(s)")
    snap = k.snaps.get(c["evidence_ids"][0])
    if snap is None:
        raise ClaimProblem(f"{c['evidence_ids'][0][:80]} is not evidence of this run")
    if snap["tool"] not in markers:
        raise ClaimProblem(f"a {snap['tool']} result cannot carry this disclosure (allowed: {sorted(markers)})")
    if _status(snap) == "error":
        raise ClaimProblem("an error result carries no disclosure")
    texts = [t for t in snap["result"]["limitations"] if markers[snap["tool"]] in t]
    if not texts:
        raise ClaimProblem("the cited result's limitations do not contain the registry marker")
    return {"disclosure": disclosure, "category": "tool_disclosure", "scope": DISCLOSURE_SCOPE[disclosure],
            "text": texts[0], "note": "a disclosure the tool makes for every result, not a measurement of this data"}


def _p_method_limitation(c, k, finding):
    kind = c["params"]["kind"]
    if len(c["evidence_ids"]) != 1:
        raise ClaimProblem("cites exactly 1 evidence id(s)")
    snap = k.snaps.get(c["evidence_ids"][0])
    if snap is None:
        raise ClaimProblem(f"{c['evidence_ids'][0][:80]} is not evidence of this run")
    if _status(snap) == "error":
        raise ClaimProblem("an error result carries no methodology")
    if kind == "provisional_parameters":
        names = sorted({m["name"] for m in snap["result"]["methodology"] if m["status"] == "provisional"})
        if not names:
            raise ClaimProblem("the cited result records no provisional methodology item")
        return {"kind": kind, "category": "measured_fact", "provisional_items": names}
    if snap["tool"] != "compare_windows":
        raise ClaimProblem(f"{kind} cites a compare_windows result")
    req = snap["request"]
    if kind == "provisional_criteria":
        crit = req["sufficiency"]
        if crit["status"] != "provisional":
            raise ClaimProblem("the comparison's sufficiency criteria are not provisional")
        return {"kind": kind, "category": "measured_fact", "criteria": dict(crit)}
    if kind == "descriptive_only":
        if req["uncertainty"]["method"] != "none" or ctx._method(snap["result"], "uncertainty_method") != "none":
            raise ClaimProblem("the comparison does not record the descriptive-only uncertainty method")
        return {"kind": kind, "category": "measured_fact", "uncertainty_method": "none"}
    if kind == "single_feature":
        features = sorted({s["request"]["feature"] for s in k.snaps.values() if s["tool"] == "compare_windows"})
        if len(features) != 1:
            raise ClaimProblem(f"comparisons of this run used {len(features)} features")
        return {"kind": kind, "category": "measured_fact", "feature": features[0]}
    item = next((m for m in snap["result"]["methodology"] if m["name"] == "baseline_clearance_frames"), None)
    if item is None or isinstance(item["value"], bool) or item["value"] != 0:
        raise ClaimProblem("the comparison does not record a re-checked baseline clearance of 0")
    return {"kind": kind, "category": "measured_fact", "rechecked_clearance_frames": 0, "tool_note": item["note"]}


PREDICATES = {"sufficiency_not_met": _p_status, "comparison_no_data": _p_status, "comparison_unavailable": _p_status,
              "no_comparison": _p_no_comparison, "descriptive_effect": _p_descriptive, "coverage": _p_coverage,
              "tracking_concern": _p_tracking, "operational_fact": _p_operational,
              "potential_confounder": _p_confounder, "sensitivity_excluding_flagged": _p_sensitivity,
              "measurement_context": _p_measurement_context, "tool_disclosure": _p_tool_disclosure,
              "methodological_limitation": _p_method_limitation}
assert set(PREDICATES) == set(CLAIM_TYPES)
ERROR_CITERS = {("comparison_unavailable", None), ("operational_fact", "quality_check_failed")}


@dataclass(frozen=True)
class _Knowledge:
    snaps: Any
    trace: Any
    primary_id: str | None
    coverage: Any
    tracking: Any
    cross: Any


# ======================================================================
# validation
# ======================================================================

def validate_findings(proposal, *, evidence, trace, policy, mode="prefix", stopping_reason=None, finished=None):
    """A frozen FindingsV2 for a proposal, accepted or refused with every
    problem found. `mode` is the M0 execution mode: "prefix" at the finish
    freeze point (Stage 6, M3), "complete" when re-validating a finished run."""
    try:
        return _validate(proposal, evidence, trace, policy, mode, stopping_reason, finished)
    except Exception as exc:                     # anything malformed is a refusal, never another exception
        return _refused([f"the proposal could not be validated ({type(exc).__name__})"])


def _validate(proposal, evidence, trace, policy, mode, stopping_reason, finished):
    problems = _schema_problems(proposal)
    if problems:
        return _refused(problems)
    if mode == "complete" and (finished is not True or stopping_reason != "finished"):
        # a run that stopped without an accepted finish is represented only by incomplete_findings()
        return _refused(["a completed validation needs a finished run (finished=True, stopping reason "
                         "'finished'); incomplete runs are represented only by incomplete_findings()"])
    p = proposal
    primary_id = p["primary_evidence_id"]
    context, context_problems = ctx.context_from_evidence(
        evidence, policy=policy, primary_evidence_id=primary_id, trace=trace, mode=mode,
        stopping_reason=stopping_reason, finished=finished)
    if context is None:
        return _refused(context_problems)
    if mode == "complete" and not context.execution.accepted_finish:
        return _refused(["a completed validation needs exactly one accepted finish in the trace"])
    snaps = context.snapshot_dicts()
    k = _Knowledge(snaps=snaps, trace=trace, primary_id=primary_id, coverage=ctx.thaw(context.coverage),
                   tracking=ctx.thaw(context.tracking), cross=ctx.thaw(context.cross_snapshot))
    finding = p["finding"]
    if primary_id is not None and snaps[primary_id]["tool"] != "compare_windows":
        problems.append("the primary evidence is a compare_windows result")

    # claims: each against its own predicate
    accepted = []
    for i, c in enumerate(p["claims"]):
        key = (c["type"], c["params"].get("kind"))
        for eid in c["evidence_ids"]:
            if eid in snaps and _status(snaps[eid]) == "error" and key not in ERROR_CITERS and \
                    (c["type"], None) not in ERROR_CITERS:
                problems.append(f"claims[{i}]: {eid[:80]} is an error result; only comparison_unavailable and "
                                "operational_fact(quality_check_failed) may cite one")
                break
        else:
            try:
                accepted.append({"type": c["type"], "evidence_ids": list(c["evidence_ids"]),
                                 "params": dict(c["params"]), "grounding": PREDICATES[c["type"]](c, k, finding)})
            except ClaimProblem as exc:
                problems.append(f"claims[{i}] ({c['type']}): {exc}")

    # observations: real fields, no effect values outside effect_described
    observations = []
    for i, o in enumerate(p["observations"]):
        snap = snaps.get(o["evidence_id"])
        if snap is None:
            problems.append(f"observations[{i}]: {o['evidence_id'][:80]} is not evidence of this run")
            continue
        if finding != "effect_described" and EFFECT_FIELDS.match(o["field"]):
            problems.append(f"observations[{i}]: effect values are not reported under {finding}")
            continue
        try:
            value = ctx.resolve_field(snap["result"], o["field"])
        except KeyError:
            problems.append(f"observations[{i}]: {o['field'][:80]!r} is not a field of its evidence")
            continue
        scalar = (str, int, float, bool, type(None))
        if not (isinstance(value, scalar) or isinstance(value, list) and len(value) <= MAX_LIST_VALUE
                and all(isinstance(x, scalar) for x in value)):
            problems.append(f"observations[{i}]: {o['field'][:80]!r} is not a value")
            continue
        observations.append({"evidence_id": o["evidence_id"], "field": o["field"], "value": value})

    problems += _coherence(p, k, snaps, accepted)
    problems += _next_problems(p, k, snaps)
    authorization = context.authorization.status
    if problems:
        return _refused(problems, authorization=authorization)
    accepted_direction = None
    if finding == "effect_described":
        accepted_direction = _sign(snaps[primary_id]["result"]["measurements"]["effect"]["value"])
    payload = ctx.fingerprint_payload(
        evidence=evidence, finding=finding, primary_evidence_id=primary_id,
        observations=[{"evidence_id": o["evidence_id"], "field": o["field"]} for o in p["observations"]],
        claims=[{"type": c["type"], "evidence_ids": list(c["evidence_ids"]), "params": dict(c["params"])}
                for c in p["claims"]],
        next_investigations=[{"action": n["action"], "params": dict(n["params"]), "because": list(n["because"])}
                             for n in p["next_investigations"]],
        registry_version=REGISTRY_VERSION, grid_version=GRID_VERSION)
    return FindingsV2(findings_format=ctx.V2_FORMAT, registry_version=REGISTRY_VERSION, grid_version=GRID_VERSION,
                      accepted=True, problems=(), finding=finding, primary_evidence_id=primary_id,
                      accepted_direction=accepted_direction, observations=ctx.freeze(observations),
                      claims=ctx.freeze(accepted),
                      next_investigations=ctx.freeze([dict(n) for n in p["next_investigations"]]),
                      authorization=authorization, publishable=authorization == "rechecked",
                      fingerprint=ctx.content_fingerprint(payload))


def _coherence(p, k, snaps, accepted):
    """Global rules (design 5.2): finding x claims, cardinality, duplicates."""
    out, finding, primary_id = [], p["finding"], p["primary_evidence_id"]
    claims = p["claims"]
    keys = [(c["type"], json.dumps(c["params"], sort_keys=True), tuple(sorted(c["evidence_ids"]))) for c in claims]
    if len(set(keys)) != len(keys):
        out.append("duplicate claims")
    status_claims = [c for c in claims if c["type"] in STATUS_CLAIMS]
    cited = [c["evidence_ids"][0] for c in status_claims if c["evidence_ids"]]
    if len(set(cited)) != len(cited):
        out.append("at most one comparison-status claim per comparison")
    effects = [c for c in claims if c["type"] == "descriptive_effect"]
    compares = [eid for eid, s in snaps.items() if s["tool"] == "compare_windows"]
    primary = snaps.get(primary_id) if primary_id else None
    if finding == "effect_described":
        if primary is None or _status(primary) != "ok":
            out.append("effect_described needs an ok primary comparison")
        if len(effects) != 1:
            out.append("effect_described needs exactly one descriptive_effect citing the primary")
        if any(primary_id in c["evidence_ids"] for c in status_claims):
            out.append("no comparison-status claim may cite the primary under effect_described")
    else:
        if effects:
            out.append(f"descriptive_effect is not allowed under {finding}")
    if finding == "insufficient_evidence":
        if compares:
            if primary is None or _status(primary) not in STATUS_CLAIMS.values():
                out.append("insufficient_evidence needs a primary comparison with status insufficient_evidence, "
                           "no_data or error")
            else:
                want = next(t for t, s in STATUS_CLAIMS.items() if s == _status(primary))
                if sum(1 for c in status_claims if c["type"] == want and c["evidence_ids"] == [primary_id]) != 1:
                    out.append(f"insufficient_evidence needs exactly one {want} claim citing the primary")
        else:
            if primary_id is not None or not any(c["type"] == "no_comparison" for c in claims):
                out.append("insufficient_evidence without any comparison needs a null primary and no_comparison")
    if finding == "inconclusive":
        if primary is not None and _status(primary) == "error":
            out.append("inconclusive needs a non-error primary comparison or a null primary")
        limiting = [a for a in accepted if a["type"] in LIMITING
                    and (a["type"] != "coverage" or a["params"]["aspect"] == "sessions_without_baseline"
                         or a["grounding"].get("status") in ("partial", "unknown"))
                    and (a["type"] != "operational_fact" or a["params"]["kind"] != "model_selected_plan")
                    and (a["type"] != "methodological_limitation" or a["params"]["kind"] == "provisional_criteria")]
        if not limiting:
            out.append("inconclusive needs at least one evidence-grounded limitation claim")
    return out


def _next_problems(p, k, snaps):
    out = []
    for i, n in enumerate(p["next_investigations"]):
        where = f"next_investigations[{i}]"
        if n["action"] == "include_more_sessions":
            cov = k.coverage["primary"]
            if cov is None or cov["status"] != "partial":
                out.append(f"{where}: include_more_sessions only when the primary coverage is partial")
        elif n["action"] == "check_tracking_quality":
            src = snaps.get(n["params"]["windows_from"])
            if src is None or src["tool"] != "get_window_features" or src["result"]["measurements"] is None:
                out.append(f"{where}: windows_from is a get_window_features result of this run")
                continue
            wanted = {w["window_id"] for w in src["result"]["measurements"]["windows"]}
            assessed = {w["window_id"] for s in snaps.values() if s["tool"] == "check_tracking_quality"
                        and s["result"]["measurements"] for w in s["result"]["measurements"]["windows"]}
            if wanted <= assessed:
                out.append(f"{where}: those windows were already assessed")
        else:
            q = snaps.get(n["params"]["quality"])
            if q is None or q["tool"] != "check_tracking_quality" or q["result"]["measurements"] is None or \
                    not any(w["smoothing_contamination"]["status"] == "detected" or w["verdict"] != "usable_with_caveats"
                            for w in q["result"]["measurements"]["windows"]):
                out.append(f"{where}: exclude_flagged_windows needs a quality result with flagged windows")
    return out


def incomplete_findings(*, evidence, trace, policy, stopping_reason):
    """The findings/2 representation of a run that stopped without an accepted
    finish: finding "incomplete", null primary, no claims, observations or
    next investigations. Nothing is invented; the completed trace is checked."""
    context, problems = ctx.context_from_evidence(evidence, policy=policy, primary_evidence_id=None, trace=trace,
                                                  mode="complete", stopping_reason=stopping_reason, finished=False)
    if context is None:
        return _refused(problems)
    payload = ctx.fingerprint_payload(evidence=evidence, finding="incomplete", primary_evidence_id=None,
                                      registry_version=REGISTRY_VERSION, grid_version=GRID_VERSION)
    status = context.authorization.status
    return FindingsV2(findings_format=ctx.V2_FORMAT, registry_version=REGISTRY_VERSION, grid_version=GRID_VERSION,
                      accepted=True, problems=(), finding="incomplete", primary_evidence_id=None,
                      accepted_direction=None, observations=(), claims=(), next_investigations=(),
                      authorization=status, publishable=status == "rechecked",
                      fingerprint=ctx.content_fingerprint(payload))


__all__ = ["CLAIM_TYPES", "FINDINGS", "FindingsV2", "GRID_VERSION", "NEXT_ACTIONS", "PARAMS", "REGISTRY_VERSION",
           "incomplete_findings", "validate_findings"]
