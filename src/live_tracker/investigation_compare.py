"""EXPERIMENTAL (Investigation mode, Stage 5): the real `compare_windows` tool (new file).

Read-only. Answers through the Stage 2 contract (investigation_contracts) and
reads CalMS21 validation sessions only through the Stage 3
AnnotationEventSource snapshot (manifest / policy binding, Stage 1
verification, provenance, final binding and index re-check). Windows must
come from "validation_annotation"; other sources answer "source_unavailable".

Question it serves: is a feature (e.g. dist_rate, negative = body centres
getting closer) different in the windows before an annotated behavior than
in baseline periods of the same sessions? It does NOT test whether approach
causes or predicts the behavior.

Measurements are Stage 4's, never an alternative
------------------------------------------------
Every window value is investigation_windows.measure_window: the median of
the feature over the window's usable frames (finite value and tracking_ok at
t-1 and t), computed from the session truncated at the window end (no frame
at or after the end influences it). The exclusion rules are Stage 4's,
applied exactly as get_window_features applies them with the evidence
determined by the request (see BASELINE_CLEARANCE). Smoothing contamination
counts come from investigation_windows.assess_window (check_tracking_quality).

Compatibility (refused as invalid_request, never mixed): all pre-event
windows share one length and gap, all baselines one length, equal to the
pre-event length, so every window value is a median over the same duration.

Statistic (contract: median_difference, session_level)
------------------------------------------------------
1. window value   median over usable frames (Stage 4)
2. per session    pre_summary = median of its used pre-event window values,
                  baseline_summary = median of its used baseline values,
                  difference = pre_summary - baseline_summary (paired: only
                  sessions with both groups have one)
3. effect         median of the session differences. Sessions, not windows
                  or frames, are the replicates: windows of one session share
                  the animals, the camera and the time course.
4. uncertainty    none. The effect is DESCRIPTIVE: the median of the
                  available paired session differences, with no interval and
                  no claim about a population. No p-value and no significance
                  statement is ever produced.

No population confidence interval until a method is approved
-------------------------------------------------------------
The contract also accepts uncertainty "bootstrap_sessions". This
implementation refuses it ("not_implemented", before anything is read) and
never substitutes another method. An earlier version reported a percentile
session bootstrap and treated 1 - 2 ** (1 - n) as the supported level. That
was wrong: 1 - 2 ** (1 - n) is the coverage of the full range [min, max] of n
differences (iid, continuous), only an upper bound for narrower intervals.
The percentile interval is narrower: for differences [0, 1, 2, 3] it is
[0.5, 2.5] at a nominal 0.875, whose coverage under iid sampling is about
0.75; four equal differences give a zero-width interval. Adding an approved
method later means one entry in APPROVED_UNCERTAINTY and its computation.

Status follows the contract: "no_data" when no requested window is usable,
"insufficient_evidence" (no effect, no interval) when any caller-supplied
sufficiency criterion is unmet, "ok" otherwise. Nothing is imputed: a
missing summary is null, a missing group gives no difference.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from . import investigation_contracts as ic
from . import investigation_windows as iw
from .calms21 import FPS
from .investigation_events import IMPLEMENTATION, IMPLEMENTATION_VERSION, SourceUnavailable, _method

__all__ = ["APPROVED_UNCERTAINTY", "BASELINE_CLEARANCE", "compare_windows"]

TOOL = "compare_windows"
# get_window_features takes clearance_frames; a compare request and a baseline
# window id carry none, so only clearance 0 (behavior frame inside the tile's
# feature history) can be re-checked here. Larger clearances are the caller's
# selection, made with get_window_features, and are not re-verified.
BASELINE_CLEARANCE = 0
CODE_FILES = iw.CODE_FILES + ("investigation_compare.py",)
# contract uncertainty methods this implementation may report; "none" only,
# until a session-level uncertainty method is approved (module docstring)
APPROVED_UNCERTAINTY = ("none",)
assert set(APPROVED_UNCERTAINTY) <= {"none", "bootstrap_sessions"}


# ---- shared result parts ----

def _code_identity():
    here = Path(__file__).parent
    files = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in CODE_FILES}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def _provenance(data=()):
    return {"implementation": IMPLEMENTATION, "implementation_version": IMPLEMENTATION_VERSION,
            "contract_version": ic.CONTRACT_VERSION, "code_sha256": _code_identity()[0], "data": list(data)}


UNITS = {"dist_center": "px", "dist_rate": "px/s", "rel_speed": "px/s", "close_time_s": "s",
         "speed_corr_w": "correlation (unitless)"}
assert set(UNITS) == set(ic.FEATURES), "a contract feature has no unit"


def _code_method():
    _, files = _code_identity()
    return _method("code_files", ", ".join(f"{k} {v[:12]}" for k, v in files.items()),
                   "provenance.code_sha256 = sha256 of the JSON {file: sha256} of these files")


def _error(request, code, message):
    return ic.make_result(TOOL, request, "error", None, provenance=_provenance(),
                          errors=[{"code": code, "message": message[:ic.MAX_TEXT]}])


def _criteria_method(name, value, crit, note):
    return {"name": name, "value": value, "status": crit["status"], "decided_by": crit["decided_by"],
            "note": note}


def _usability(summary, fraction):
    """Stage 4's usability rule with the request's min_usable_fraction."""
    if summary["n_frames"] and summary["n_usable"] == 0:
        return ["missing_data"]
    if summary["n_usable"] < fraction * summary["n_frames"] - ic.STAT_ABS_TOL:
        return ["insufficient_usable_frames"]
    return []


def _spread(values):
    if not values:
        return None
    return " / ".join(f"{v:.6g}" for v in (min(values), statistics.median(values), max(values)))


LIMITATIONS = [
    "Session-level comparison: windows and frames are not independent replicates; the effect is the "
    "median of per-session paired differences, so its precision is set by the number of paired sessions.",
    "No p-value, significance or confidence interval is reported: the effect is the median of the "
    "available paired session differences, a description of these sessions, not a population estimate. "
    "No session-level uncertainty method has been approved.",
    "Investigation confound (not resolved): windows before an attack are mostly annotated "
    "'investigation', baselines mostly 'other' (label_frames_* in methodology); a difference may reflect "
    "investigation versus other behavior, not 'before attack'.",
    "Session-timing confound (not resolved): pre-event windows and baseline tiles can lie at different "
    "times of the session (window_start_s_* in methodology); activity may drift within a session.",
    "Missing tracking: windows below min_usable_fraction are excluded and usable frames are not "
    "interpolated; if tracking loss is related to behavior, the remaining frames are a biased sample.",
    "Pixel scale: distances and rates are in image pixels, not calibrated between sessions; session "
    "differences from different camera geometries are pooled as they are.",
    "Motion accuracy is not verified: tracking_ok only means the keypoints passed the quality flags. "
    "One-frame keypoint collapses that pass every flag exist in these sessions (Stage 1 report, "
    "section 4); the smoother carries them for tens of frames. No jump limit has been chosen.",
    "dist_rate < 0 means the body-centre distance is decreasing; it does not say which mouse moved or "
    "that either approached. CalMS21 Task 1 has no 'approach' label. Association, not causation.",
    "Baseline clearance from the behavior is the caller's selection (get_window_features); this tool "
    "re-checks only clearance 0: no behavior frame in a baseline tile or its feature history.",
    "Window length, gap, min_usable_fraction, smoother-memory margin and sufficiency criteria are "
    "provisional choices, not validated settings.",
    iw.ANNOTATION_NOTE,
]


# ---- compare_windows ----

def compare_windows(tools, request):
    ic.validate_request(TOOL, request, tools.policy)       # nothing is read before this
    method = request["uncertainty"]["method"]
    if method not in APPROVED_UNCERTAINTY:
        return _error(request, "not_implemented",
                      f"uncertainty {method!r} is not available: no session-level uncertainty method has "
                      "been approved, so no confidence interval is reported. Request uncertainty 'none' "
                      "for the descriptive median of paired session differences")
    pre_ids, base_ids = request["pre_window_ids"], request["baseline_window_ids"]
    pre = [ic.parse_window_id(w) for w in pre_ids]
    base = [ic.parse_window_id(w) for w in base_ids]
    source, behavior = pre[0]["source"], pre[0]["behavior"]
    src = tools.sources.get(source)
    if source != "validation_annotation" or src is None:
        return _error(request, "source_unavailable", f"windows of source {source!r} cannot be verified by "
                      "this implementation; only validation_annotation windows are supported")
    shapes = {(w["length_frames"], w["gap_frames"]) for w in pre}
    if len(shapes) != 1:
        return _error(request, "invalid_request", "pre-event windows must share one length and gap; "
                      f"got {sorted(shapes)[:5]}")
    (L, gap), = shapes
    if {w["length_frames"] for w in base} != {L}:
        return _error(request, "invalid_request", f"baseline windows must have the pre-event length {L} "
                      "so every window value is a median over the same duration")

    sessions = sorted({w["session"] for w in pre + base})
    try:
        snap = src.snapshot(sessions)
        data = {sid: iw.load_session(sid, snap.sessions[sid]) for sid in sessions}
    except SourceUnavailable as exc:
        return _error(request, "source_unavailable", str(exc))
    for wid, w in zip(pre_ids, pre):
        if not iw._is_episode_start(data[w["session"]], behavior, w["onset_frame"]):
            return _error(request, "unknown_id", f"{wid}: {w['episode_id']} is not the first frame of an "
                          f"annotated {behavior!r} run in the verified session")
    for wid, w in zip(base_ids, base):
        if w["start_frame"] % L or w["start_frame"] + L > data[w["session"]].n_frames:
            return _error(request, "unknown_id", f"{wid}: not a whole tile of length {L} inside the session")

    feature = request["feature"]
    crit = request["sufficiency"]
    fraction = crit["min_usable_fraction"]
    lookback = iw.feature_lookback([feature])
    is_beh = {sid: data[sid].labels == behavior for sid in sessions}
    windows = {}                                  # wid -> measured window
    pre_spans = {sid: [] for sid in sessions}

    def measure(wid, kind, sid, start, end, reasons):
        s = iw.measure_window(data[sid], start, end, [feature])[feature] if end > start else None
        if s is not None:
            reasons += _usability(s, fraction)
        windows[wid] = {"kind": kind, "session": sid, "start": start, "end": end,
                        "reasons": [r for r in ic.EXCLUSION_REASONS if r in set(reasons)], "summary": s}

    # Stage 4 pre-event rules (get_window_features), per requested window
    for wid, w in zip(pre_ids, pre):
        sid, onset = w["session"], w["onset_frame"]
        end = onset - gap
        start = end - L
        reasons = []
        if start < 0:
            reasons.append("starts_before_session")
            start, end = 0, max(end, 0)
        hs = iw.history_start(iw.prefix_arrays(data[sid], end)[2], start - lookback) if end > 0 else 0
        if iw._any(is_beh[sid], hs, onset):
            reasons.append("overlaps_event")
        if end > start:
            pre_spans[sid].append((start, end))
        measure(wid, "pre_event", sid, start, end, reasons)

    # Stage 4 baseline rules at clearance BASELINE_CLEARANCE, against every requested pre-event window
    C = BASELINE_CLEARANCE
    for wid, w in zip(base_ids, base):
        sid, a = w["session"], w["start_frame"]
        b = a + L
        hs = iw.history_start(iw.prefix_arrays(data[sid], b)[2], a - lookback)
        reasons = []
        if iw._any(is_beh[sid], a, b):
            reasons.append("overlaps_event")
        if iw._any(is_beh[sid], hs - C, b + C):
            reasons.append("near_event")
        if any(p0 < b and hs < p1 for p0, p1 in pre_spans[sid]):
            reasons.append("overlaps_pre_window")
        measure(wid, "baseline", sid, a, b, reasons)

    used = {wid: x for wid, x in windows.items() if not x["reasons"]}
    excluded = []
    for wid in pre_ids + base_ids:
        x = windows[wid]
        if not x["reasons"]:
            continue
        clipped = x["end"] - x["start"] != L
        excluded.append({"window_id": wid, "reason": x["reasons"][0],
                         "n_frames": None if clipped else x["summary"]["n_frames"],
                         "n_usable": None if clipped else x["summary"]["n_usable"]})

    per_session, diffs = [], []
    for sid in sessions:
        groups = {}
        for kind in ("pre_event", "baseline"):
            ws = sorted(((x["start"], wid) for wid, x in used.items()
                         if x["session"] == sid and x["kind"] == kind))
            groups[kind] = [{"window_id": wid, "n_frames": used[wid]["summary"]["n_frames"],
                             "n_usable": used[wid]["summary"]["n_usable"],
                             "value": used[wid]["summary"]["median"]} for _, wid in ws]
        if not groups["pre_event"] and not groups["baseline"]:
            continue
        pre_s = statistics.median([v["value"] for v in groups["pre_event"]]) if groups["pre_event"] else None
        base_s = statistics.median([v["value"] for v in groups["baseline"]]) if groups["baseline"] else None
        diff = pre_s - base_s if pre_s is not None and base_s is not None else None
        if diff is not None:
            diffs.append(diff)
        per_session.append({"session": sid, "pre_windows": groups["pre_event"],
                            "baseline_windows": groups["baseline"], "pre_summary": pre_s,
                            "baseline_summary": base_s, "difference": diff})

    n_pre = sum(x["kind"] == "pre_event" for x in used.values())
    n_base = len(used) - n_pre
    unmet = []
    if len(diffs) < crit["min_sessions"]:
        unmet.append("min_sessions")
    if n_pre < crit["min_windows_per_group"]:
        unmet.append("min_windows_per_group_pre")
    if n_base < crit["min_windows_per_group"]:
        unmet.append("min_windows_per_group_baseline")
    status = "no_data" if not per_session else ("insufficient_evidence" if unmet else "ok")

    # descriptive only: no interval for any approved method yet (APPROVED_UNCERTAINTY)
    effect = {"name": "median_of_session_differences", "value": statistics.median(diffs) if status == "ok"
              else None, "ci_low": None, "ci_high": None, "ci_level": None}

    measurements = {"feature": feature, "aggregation": "session_level", "per_session": per_session,
                    "excluded_windows": excluded, "n_sessions_paired": len(diffs), "n_windows_pre": n_pre,
                    "n_windows_baseline": n_base, "effect": effect,
                    "sufficiency": {"criteria": dict(crit), "met": not unmet, "reasons_not_met": unmet}}
    methodology = _methodology(request, data, windows, used, diffs, L, gap, lookback)
    limitations = list(LIMITATIONS)
    if snap.synthetic:
        limitations.insert(0, "SYNTHETIC test fixture: not CalMS21 data, not evidence.")
    result = ic.make_result(TOOL, request, status, measurements, methodology=methodology,
                            limitations=limitations, provenance=_provenance(snap.data))
    ic.validate_result(TOOL, request, result, tools.policy)     # self-check
    try:
        snap.finalize()
    except SourceUnavailable as exc:
        return _error(request, "source_unavailable", str(exc))
    return result


def _methodology(request, data, windows, used, diffs, L, gap, lookback):
    crit, u, feature = request["sufficiency"], request["uncertainty"], request["feature"]

    def of(kind):
        return [x for x in used.values() if x["kind"] == kind]

    def labels(kind):
        tot = {k: 0 for k in ic.LABELS}
        for x in of(kind):
            for k, v in iw._label_context(data[x["session"]], x["start"], x["end"]).items():
                tot[k] += v
        return ", ".join(f"{k} {v}" for k, v in tot.items())

    def usable_fraction(kind):
        xs = of(kind)
        frames = sum(x["summary"]["n_frames"] for x in xs)
        return round(sum(x["summary"]["n_usable"] for x in xs) / frames, 6) if frames else None

    def contaminated(kind):
        return sum(iw.assess_window(data[x["session"]], x["start"], x["end"], [1])
                   ["smoothing_contamination"]["status"] == "detected" for x in of(kind))

    counts = {}
    for x in windows.values():
        for r in x["reasons"]:
            key = f"{'pre' if x['kind'] == 'pre_event' else 'baseline'} {r}"
            counts[key] = counts.get(key, 0) + 1
    n = len(diffs)
    sign = ("negative = in the session's pre-event windows the body-centre distance decreases faster "
            "(or grows slower) than in its baselines" if feature == "dist_rate"
            else "positive = higher in pre-event windows than in baselines")
    return iw._common_methodology() + [
        _method("window_value", "median of the feature over usable frames (Stage 4 measure_window)",
                "usable = finite value and tracking_ok at t-1 and t; nothing imputed"),
        _method("window_shape", f"pre-event [onset - {gap} - {L}, onset - {gap}); baseline tiles of {L} "
                "frames", "mixed lengths or gaps are refused (every value is a median over the same duration)"),
        _method("feature_lookback_frames", lookback, f"{feature} reads {lookback} earlier frames"),
        _method("exclusion_rules", "Stage 4 get_window_features rules",
                "pre: starts_before_session, overlaps_event (behavior in [history start, onset)); baseline: "
                "overlaps_event, near_event, overlaps_pre_window (against every requested pre-event "
                "window); both: missing_data, insufficient_usable_frames"),
        _method("baseline_clearance_frames", BASELINE_CLEARANCE,
                "not in the compare request or window id; only clearance 0 is re-checked (no behavior frame "
                "in the tile or its feature history); a larger clearance is the caller's selection"),
        _method("exclusion_reason_reported", "first reason in contract order",
                "a window can fail several rules; all are counted in exclusion_counts"),
        _method("exclusion_counts", "; ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none",
                "every rule each excluded window fails"),
        _criteria_method("min_usable_fraction", crit["min_usable_fraction"], crit,
                         "from request.sufficiency; below it: insufficient_usable_frames"),
        _method("session_summary", "median of the session's used window values, per group",
                "one row per session; sessions are the replicates, not windows or frames"),
        _method("session_difference", "pre_summary - baseline_summary",
                "only sessions with both groups (paired); others are listed without a difference"),
        _method("effect", "median of paired session differences",
                f"descriptive, over n = {n} paired sessions; {sign}"),
        _method("uncertainty_method", u["method"],
                "no confidence interval, p-value or significance: no session-level uncertainty method "
                "has been approved (bootstrap_sessions is refused)"),
        _criteria_method("min_sessions", crit["min_sessions"], crit, "paired sessions required for an effect"),
        _criteria_method("min_windows_per_group", crit["min_windows_per_group"], crit,
                         "used windows required in each group"),
        _method("label_frames_pre", labels("pre_event"), "annotated frames in used pre-event windows "
                "(investigation confound)"),
        _method("label_frames_baseline", labels("baseline"), "annotated frames in used baseline windows"),
        _method("window_start_s_pre", _spread([x["start"] / FPS for x in of("pre_event")]),
                "min / median / max start time of used pre-event windows (session-timing confound)"),
        _method("window_start_s_baseline", _spread([x["start"] / FPS for x in of("baseline")]),
                "min / median / max start time of used baselines"),
        _method("usable_frame_fraction_pre", usable_fraction("pre_event"), "over used pre-event windows"),
        _method("usable_frame_fraction_baseline", usable_fraction("baseline"), "over used baselines"),
        _method("smoothing_contamination_windows_pre", contaminated("pre_event"),
                "used pre-event windows with a frame failing tracking_ok in the smoother memory "
                "(check_tracking_quality); flag-passing collapses are not detected"),
        _method("smoothing_contamination_windows_baseline", contaminated("baseline"), "used baselines, same rule"),
        _method("units", UNITS[feature], "pixel quantities are image pixels, not calibrated between sessions"),
        _code_method(),
    ]
