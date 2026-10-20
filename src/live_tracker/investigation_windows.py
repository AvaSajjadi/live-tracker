"""EXPERIMENTAL (Investigation mode, Stage 4): real `get_window_features` and
`check_tracking_quality` (new file).

Read-only. Both tools answer through the Stage 2 contract
(investigation_contracts) and read CalMS21 validation sessions only through
the Stage 3 AnnotationEventSource snapshot (manifest binding, Stage 1
verification, provenance, final index re-check). Episodes must come from
"validation_annotation"; other sources answer "source_unavailable".

Reuse
-----
Ava's production code is used unchanged: quality.flag_tracking (flags),
features.frame_tracking_ok (tracking_ok), features.dyadic_features (body
centre, causal One-Euro smoothing, the five dyadic features). Igor's
calms21.load_calms21 reads the records. What is NOT reused, and why:
- quality.interpolate_short_gaps: two-sided (fills from frames after the gap).
  Never applied; the contract fixes interpolation = false.
- flag_tracking's frozen flag looks ahead (a frame is frozen when its run of
  identical positions is long enough, judged from the whole run, including
  later frames). It is made causal by truncation, not by a new flag: see below.
- quality.window_quality: fixed time bins of the session, not request windows.
- motion.py (Savitzky-Golay smoothing, two-sided interpolation): not causal.
- flag_tracking's low_conf flag lets a missing (null / NaN) score pass ("a
  missing score is not evidence of low confidence"), and any comparison with
  NaN is false. tracking_ok here REQUIRES a confidence of at least min_score,
  so the adapter (_flags) adds: a present keypoint whose score is not a finite
  number in SCORE_RANGE (CalMS21's documented [0, 1]) is low_conf, hence not
  valid and not tracking_ok. Only that keypoint-frame is affected, exactly
  like a low score: a confidence that cannot be shown to meet the minimum
  never satisfies it, and the other frames' evidence is kept. Ava's function
  is unchanged; legitimate scores give identical flags.

Causal computation (the smallest causal alternative, inside this module)
-----------------------------------------------------------------------
Every window [start, end) is measured from the session TRUNCATED AT `end`:
the values are exactly those of Ava's pipeline run on frames [0, end). For
a pre-event window end = onset - gap <= onset, so no frame at or after the
onset INFLUENCES any value; baseline tiles are truncated at their own end, so
both kinds are preprocessed identically (Stage 1 truncated only pre-event
windows).

No influence is not the same as not read. The pipeline runs once per
session, on all frames. Its features are causal given the flags (One-Euro
smoothing, backward differences, trailing windows), and the only flag that
looks ahead is `frozen`, which can differ from the truncated computation
only on the last FROZEN_MIN_FRAMES frames before `end` (a run of identical
positions crossing `end`). prefix_arrays recomputes those flags from frames
[end - 2 * FROZEN_MIN_FRAMES, end) and reuses the full-session values only if
they agree (then they provably equal the truncated values); otherwise it
reruns the pipeline on frames [0, end). Every frame therefore has one value,
equal to the truncated computation, whichever window asks. Annotations after
a window are read on purpose, by the exclusion rules (the gap, clearance).

The smoother is never restarted at a window or at its history start: every
value carries the filter state from frame 0 (Ava's filter restarts only after
a gap longer than its max_gap_s between valid samples of a mouse).

History (frames before a window that its values depend on, beyond a
negligible residual). The features' own lookback (0 dist_center; 1
dist_rate, rel_speed; 30 close_time_s, speed_corr_w = the 1-s trailing
window) plus the smoother's memory: all inputs older than
SMOOTHER_MEMORY_FRAMES (25 for Ava's defaults) newer valid samples of the
same mouse together weigh less than SMOOTHER_RESIDUAL (1 %) in its smoothed
position. Missing samples do not advance the filter, so the history start is
found per window by counting valid body-centre samples of both mice
backwards. Frames before the history start are read and keep this decaying
influence (< 1 % of each smoothed position; not a bound on the nonlinear
features computed from it). The history start is used only by the exclusion
rules (no behavior frame, no other window inside it); the 1 % level is a
provisional margin, not a validated setting.

Exclusions (provisional; listed in every result's methodology)
--------------------------------------------------------------
pre_event   starts_before_session  window would start before frame 0 (clipped)
            overlaps_event         a frame annotated with the behavior lies in
                                   [history start, onset): the window, its
                                   feature history or the gap
baseline    overlaps_event         a behavior frame inside the tile
            near_event             a behavior frame within clearance_frames of
                                   [history start, end) (tile plus history)
            overlaps_pre_window    [history start, end) overlaps a requested
                                   pre-event window (included or excluded)
both        missing_data           some requested feature has no usable frame
            insufficient_usable_frames  some requested feature has fewer than
                                   MIN_USABLE_FRACTION of its frames usable
Behavior frames are ALL frames of the session annotated with the behavior,
not only the requested episodes. A usable frame has a finite value and
tracking_ok on frames t-1 and t (both derivative endpoints).

Tracking quality is not motion accuracy
---------------------------------------
tracking_ok only says every torso keypoint of both mice passed the score /
image-frame / frozen flags. Stage 1 (docs/investigation_experiment_01.md,
section 4) found one-frame keypoint collapses that pass every flag and
whose effect the smoother carries for tens of frames; no jump limit has been
chosen, so they are NOT detected here. motion_accuracy_verified is always
false.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import investigation_contracts as ic
from .calms21 import FPS, IMAGE_SIZE, MICE, TORSO_KEYPOINTS, load_calms21
from .features import FeatureParams, body_center, dyadic_features, frame_tracking_ok
from .investigation_events import (IMPLEMENTATION, IMPLEMENTATION_VERSION, SourceUnavailable, _method,
                                   session_labels, coverage_metadata)
from .quality import SUGGESTED_MIN_SCORE, flag_tracking

__all__ = ["FEATURE_LOOKBACK", "SMOOTHER_MEMORY_FRAMES", "SessionData", "check_tracking_quality",
           "get_window_features", "history_start", "load_session", "measure_window", "prefix_arrays",
           "smoother_memory_frames"]

FROZEN_MIN_FRAMES = 15                        # flag_tracking's default, passed explicitly
FLAG_SETTINGS = {"frame_size": IMAGE_SIZE, "score_col": "score", "min_score": SUGGESTED_MIN_SCORE,
                 "frozen_min_frames": FROZEN_MIN_FRAMES, "max_speed_px_s": None}
FEATURE_PARAMS = FeatureParams()              # Ava's defaults: one_euro, min_cutoff 1.0, beta 0.0
SMOOTHER_RESIDUAL = 0.01
SCORE_RANGE = (0.0, 1.0)                      # CalMS21's documented score domain (calms21.py)
# the provisional per-window usable-fraction minimum: get_window_features has no
# request field for it, so the development placeholder is used and reported
MIN_USABLE_FRACTION = ic.PROVISIONAL_SUFFICIENCY.min_usable_fraction


class UnknownWindow(ValueError):
    """A requested episode or window does not exist in the verified data."""


class OutOfRange(ValueError):
    """A requested window cannot be measured (it lies outside the session)."""


def smoother_memory_frames(params=FEATURE_PARAMS, fps=FPS, residual=SMOOTHER_RESIDUAL):
    """Samples after which an input's weight in the One-Euro output is below
    `residual`. The cutoff is never below min_cutoff (beta >= 0), so the
    per-sample smoothing factor is never below alpha_min and an input k
    samples back weighs at most (1 - alpha_min) ** k."""
    if params.smooth is None:
        return 0
    if params.smooth != "one_euro":
        raise ValueError(f"no memory bound for smoother {params.smooth!r}")
    kw = params.smooth_kwargs
    if kw.get("beta", 0.0) < 0:
        raise ValueError("beta < 0 has no memory bound")
    tau = 1.0 / (2 * math.pi * kw.get("min_cutoff", 1.0))
    alpha = 1.0 / (1.0 + tau * fps)
    return math.ceil(math.log(residual) / math.log(1.0 - alpha))


SMOOTHER_MEMORY_FRAMES = smoother_memory_frames()
_TRAILING = int(round(FEATURE_PARAMS.window_s * FPS))
# frames before t that each feature's value at t reads (before smoothing memory)
FEATURE_LOOKBACK = {"dist_center": 0, "dist_rate": 1, "rel_speed": 1,
                    "close_time_s": _TRAILING, "speed_corr_w": _TRAILING}
assert set(FEATURE_LOOKBACK) == set(ic.FEATURES), "a contract feature has no lookback"


def feature_lookback(features):
    return max(FEATURE_LOOKBACK[f] for f in features)


# ---- verified session data ----

FLAG_COLS = ("missing", "out_of_frame", "jump", "frozen", "low_conf", "valid")


def _flags(torso):
    """flag_tracking, plus: a present keypoint whose score is missing, not
    finite or outside SCORE_RANGE is low_conf (its confidence cannot be shown
    to meet min_score), so it is not valid. Identical for legitimate scores."""
    flags = flag_tracking(torso, **FLAG_SETTINGS)
    score = flags["score"].to_numpy(float)
    lo, hi = SCORE_RANGE
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(score) & (score >= lo) & (score <= hi)) & ~flags["missing"].to_numpy(bool)
    flags["low_conf"] = flags["low_conf"].to_numpy(bool) | bad
    flags["valid"] = flags["valid"].to_numpy(bool) & ~bad
    return flags


def _pipeline(torso, features=ic.FEATURES):
    """Ava's pipeline on a torso long table: (flags, tracking_ok, {feature: array},
    centre_valid (2, n)), frame-indexed from 0 to the table's last frame."""
    flags = _flags(torso)
    if "imputed" in flags.columns:                 # nothing may be filled from other frames
        raise RuntimeError("imputed points found; temporal evidence must not be interpolated")
    n = int(torso["frame"].max()) + 1
    idx = range(n)
    ok = (frame_tracking_ok(flags, keypoints=TORSO_KEYPOINTS).set_index("frame")["tracking_ok"]
          .reindex(idx).astype("boolean").fillna(False).to_numpy(bool))
    d = dyadic_features(flags, FEATURE_PARAMS, TORSO_KEYPOINTS).set_index("frame").reindex(idx)
    feats = {f: d[f].to_numpy(float) for f in features}
    centre = body_center(flags, TORSO_KEYPOINTS)
    valid = np.stack([centre[centre["mouse"] == m].set_index("frame")["valid"].reindex(idx)
                      .astype("boolean").fillna(False).to_numpy(bool) for m in MICE])
    return flags, ok, feats, valid


@dataclass
class SessionData:
    session: str
    torso: object             # long table of torso keypoints (Igor's loader), sorted
    labels: np.ndarray        # per-frame label names
    n_frames: int
    flags: object = None      # full-session flags (frozen looks ahead: see prefix_arrays)
    ok: np.ndarray = None     # full-session tracking_ok
    feats: dict = None        # full-session features
    centre_valid: np.ndarray = None  # (2, n): body centre defined, per mouse
    _prefix: dict = field(default_factory=dict)  # end -> (ok, feats) recomputed on [0, end)


def load_session(sid, data):
    """One verified session: torso keypoints, labels and the full-session
    pipeline, refused unless every torso keypoint of both mice has exactly
    frames 0..n-1 at FPS."""
    labels = session_labels(sid, data)
    poses, _ = load_calms21(data)
    n = len(labels)
    torso = poses[poses["keypoint"].isin(TORSO_KEYPOINTS)]
    torso = torso.sort_values(["mouse", "keypoint", "frame"]).reset_index(drop=True)
    counts = torso.groupby(["mouse", "keypoint"])["frame"].agg(["count", "min", "max", "nunique"])
    if len(counts) != 2 * len(TORSO_KEYPOINTS) or not ((counts["count"] == n) & (counts["nunique"] == n)
                                                         & (counts["min"] == 0) & (counts["max"] == n - 1)).all():
        raise SourceUnavailable(f"{sid}: torso keypoints do not cover frames 0..n-1 for both mice")
    if not np.allclose(torso["time_s"].to_numpy(), torso["frame"].to_numpy() / FPS, rtol=0, atol=1e-9):
        raise SourceUnavailable(f"{sid}: time_s is not frame / {FPS}")
    flags, ok, feats, valid = _pipeline(torso)
    return SessionData(sid, torso, labels, n, flags, ok, feats, valid)


def _flag_key(flags):
    return flags.sort_values(["mouse", "keypoint", "frame"])[["mouse", "keypoint", "frame", *FLAG_COLS]] \
        .reset_index(drop=True)


def prefix_arrays(sess, end):
    """(tracking_ok, {feature: array}, centre_valid (2, n)) for frames [0, end),
    equal to Ava's pipeline run on the session truncated at `end`.

    The features are causal given the flags, and only the frozen flag looks
    ahead. For frames >= end - FROZEN_MIN_FRAMES, the flags of the slice
    [end - 2 * FROZEN_MIN_FRAMES, end) equal those of the truncated session (a
    run of identical positions that began before the slice still covers more
    than FROZEN_MIN_FRAMES frames inside it); earlier frames cannot differ (a
    run crossing `end` that covers them is at least FROZEN_MIN_FRAMES long
    before `end` too). If those flags agree with the full-session flags, the
    full-session values are the truncated values; otherwise the pipeline is
    rerun on [0, end)."""
    if not 0 < end <= sess.n_frames:
        raise ValueError(f"bad window end {end}")
    if end == sess.n_frames:
        return sess.ok, sess.feats, sess.centre_valid
    if end in sess._prefix:
        return sess._prefix[end]
    m = FROZEN_MIN_FRAMES
    frame = sess.torso["frame"]
    tail = _flags(sess.torso[(frame >= max(0, end - 2 * m)) & (frame < end)])
    lo = max(0, end - m)
    f_full = sess.flags[(sess.flags["frame"] >= lo) & (sess.flags["frame"] < end)]
    if _flag_key(tail[tail["frame"] >= lo]).equals(_flag_key(f_full)):
        out = sess.ok[:end], {k: v[:end] for k, v in sess.feats.items()}, sess.centre_valid[:, :end]
    else:
        _, ok, feats, valid = _pipeline(sess.torso[frame < end])
        out = ok, feats, valid
    sess._prefix[end] = out
    return out


def history_start(centre_valid, first_read):
    """Earliest frame whose body-centre sample still weighs >= SMOOTHER_RESIDUAL
    in a smoothed value at or after `first_read`: SMOOTHER_MEMORY_FRAMES + 1
    valid samples of each mouse back from first_read (conservative by one;
    smoother resets after long gaps are ignored, which only extends it).
    `centre_valid` must come from prefix_arrays of the window's own end, so
    the frozen flag's look-ahead cannot move it."""
    if first_read <= 0:
        return 0
    k = SMOOTHER_MEMORY_FRAMES + 1
    start = first_read
    for valid in centre_valid:
        pos = np.flatnonzero(valid[:first_read + 1])
        start = min(start, int(pos[-k]) if len(pos) >= k else 0)
    return start


def _endpoint_valid(ok, start, end):
    """tracking_ok at t and t-1 for t in [start, end) (frame 0 has no t-1)."""
    cur = ok[start:end]
    prev = np.zeros(end - start, dtype=bool)
    if start > 0:
        prev[:] = ok[start - 1:end - 1]
    else:
        prev[1:] = ok[start:end - 1]
    return cur & prev


def _summary(values, usable):
    v = values[usable & np.isfinite(values)]
    if not v.size:
        return {"n_frames": int(len(values)), "n_usable": 0, "median": None, "mean": None}
    return {"n_frames": int(len(values)), "n_usable": int(v.size),
            "median": float(np.median(v)), "mean": float(np.mean(v))}


def measure_window(sess, start, end, features):
    """Feature summaries for frames [start, end) from the session truncated at end."""
    if end <= start:
        return {f: {"n_frames": 0, "n_usable": 0, "median": None, "mean": None} for f in features}
    ok, feats, _ = prefix_arrays(sess, end)
    usable = _endpoint_valid(ok, start, end)
    return {f: _summary(feats[f][start:end], usable) for f in features}


def _usability_reasons(summaries):
    if any(s["n_frames"] and s["n_usable"] == 0 for s in summaries.values()):
        return ["missing_data"]
    if any(s["n_usable"] < MIN_USABLE_FRACTION * s["n_frames"] - ic.STAT_ABS_TOL
           for s in summaries.values() if s["n_frames"]):
        return ["insufficient_usable_frames"]
    return []


def _label_context(sess, start, end):
    lab = sess.labels[start:end]
    return {k: int((lab == k).sum()) for k in ic.LABELS}


def _any(mask, a, b):
    a, b = max(0, a), min(len(mask), b)
    return a < b and bool(mask[a:b].any())


def _ordered(reasons):
    return [r for r in ic.EXCLUSION_REASONS if r in set(reasons)]


def _is_episode_start(sess, behavior, frame):
    lab = sess.labels
    return 0 <= frame < sess.n_frames and lab[frame] == behavior and (frame == 0 or lab[frame - 1] != behavior)


def _even_selection(items, k):
    """k items spread evenly over `items` (deterministic, keeps the order)."""
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[(len(items) - 1) // 2]]
    step = (len(items) - 1) / (k - 1)
    return [items[int(math.floor(i * step + 0.5))] for i in range(k)]


# ---- shared result parts ----

CODE_FILES = ("investigation_windows.py", "investigation_events.py", "investigation_contracts.py",
              "investigation_data.py", "features.py", "quality.py", "filters.py", "calms21.py")


def _code_identity():
    here = Path(__file__).parent
    files = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in CODE_FILES}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return combined, files


def _provenance(data=()):
    combined, _ = _code_identity()
    return {"implementation": IMPLEMENTATION, "implementation_version": IMPLEMENTATION_VERSION,
            "contract_version": ic.CONTRACT_VERSION, "code_sha256": combined, "data": list(data)}


def _error(tool, request, code, message):
    return ic.make_result(tool, request, "error", None, provenance=_provenance(),
                          errors=[{"code": code, "message": message[:ic.MAX_TEXT]}])


def _code_method():
    _, files = _code_identity()
    return _method("code_files", ", ".join(f"{k} {v[:12]}" for k, v in files.items()),
                   "provenance.code_sha256 = sha256 of the JSON {file: sha256} of these files")


def _common_methodology():
    return [
        _method("event_source", "validation_annotation",
                "episodes and behavior frames are human annotations of validation sessions"),
        _method("causal_computation", "per-window truncation at the window end",
                "every value equals Ava's pipeline run on frames [0, end) of the session: frames at or "
                "after the window end have no influence on any value (a pre-event window: none at or "
                "after onset - gap). They are read: the pipeline runs once on the whole session and its "
                "values are reused only where they provably equal the truncated run (else it is rerun "
                "on [0, end)); annotations after the window are read by the exclusion rules"),
        _method("tracking_flags", f"flag_tracking: frame {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}, min_score "
                f"{SUGGESTED_MIN_SCORE}, frozen {FROZEN_MIN_FRAMES} frames, jump detection off",
                "jump limit not chosen (Ava's decision); single-frame keypoint collapses pass every flag"),
        _method("score_validation", f"finite score in [{SCORE_RANGE[0]}, {SCORE_RANGE[1]}] required",
                "a present keypoint with a missing (null / NaN), infinite or out-of-range score is "
                "low_conf for that frame (not tracking_ok); its confidence is never assumed"),
        _method("tracking_ok", "every torso keypoint of both mice passes the flags",
                "torso = " + ", ".join(TORSO_KEYPOINTS) + " (team decision); not motion accuracy"),
        _method("interpolation", "none", "interpolate_short_gaps is two-sided and is never applied"),
        _method("smoothing", "one_euro", f"features.FeatureParams default {FEATURE_PARAMS.smooth_kwargs}; "
                "causal; run from frame 0 and NOT restarted at the window or its history start (only "
                "after a gap longer than the filter's max_gap_s), so frames before the history start "
                "keep a decaying influence"),
        _method("smoother_memory_frames", SMOOTHER_MEMORY_FRAMES,
                f"valid samples after which all older inputs together weigh < {SMOOTHER_RESIDUAL} in a "
                "smoothed position (bound at the minimum cutoff); a provisional margin that sets the "
                "history start used by the exclusion rules, not a bound on the nonlinear features"),
        _method("fps", FPS, "calms21.FPS"),
    ]


ANNOTATION_NOTE = ("Behavior frames are human annotations (annotator 1) of CalMS21 Task 1 validation "
                   "sessions; boundaries are uncertain. Held-out test annotations are never read.")


# ---- get_window_features ----

FEATURE_LIMITATIONS = [
    "dist_rate < 0 means the body-centre distance is decreasing. It does not say which mouse moved "
    "or that either mouse approached the other; it is not an 'approach' label (CalMS21 Task 1 has none).",
    "tracking_ok is not motion accuracy. One-frame keypoint collapses that pass every quality flag "
    "exist in these sessions (Stage 1 report, section 4) and are not detected here; the median is the "
    "primary summary because it resists single-frame spikes. Use check_tracking_quality per window.",
    "Pre-event windows before an attack are mostly annotated 'investigation', baselines mostly "
    "'other' (see label_context): a difference may reflect investigation versus other, not 'before "
    "attack' (investigation confound, not resolved).",
    "Baseline tiles may lie at other times of the session than pre-event windows (time confound, "
    "not resolved).",
    "Windows of one session are not independent observations; sessions are the unit of analysis.",
    "Pixel distances are not calibrated between sessions (camera geometry may differ).",
    "Window inclusion requires every requested feature to meet the usable-frame minimum: request "
    "only the features that will be compared.",
    "The usable-frame minimum, history margin, smoother memory level and baseline thinning are "
    "provisional choices, not validated settings.",
    ANNOTATION_NOTE,
]


def get_window_features(tools, request):
    tool = "get_window_features"
    ic.validate_request(tool, request, tools.policy)      # nothing is read before this
    eps = [ic.parse_episode_id(e) for e in request["episode_ids"]]
    source, behavior = eps[0]["source"], eps[0]["behavior"]
    src = tools.sources.get(source)
    if source != "validation_annotation" or src is None:
        return _error(tool, request, "source_unavailable",
                      f"episodes of source {source!r} cannot be verified by this implementation; "
                      "only validation_annotation episodes are supported")
    sessions = sorted({e["session"] for e in eps})
    try:
        snap = src.snapshot(sessions)
        data = {sid: load_session(sid, snap.sessions[sid]) for sid in sessions}
    except SourceUnavailable as exc:
        return _error(tool, request, "source_unavailable", str(exc))
    for eid, e in zip(request["episode_ids"], eps):
        if not _is_episode_start(data[e["session"]], behavior, e["start_frame"]):
            return _error(tool, request, "unknown_id", f"{eid} is not the first frame of an annotated "
                          f"{behavior!r} run in the verified session")

    feats = request["features"]
    lookback = feature_lookback(feats)
    L, gap = request["pre_event"]["length_frames"], request["pre_event"]["gap_frames"]
    is_beh = {sid: data[sid].labels == behavior for sid in sessions}
    windows, pre_spans = [], {sid: [] for sid in sessions}

    def item(wid, kind, sid, eid, start, end, reasons, summaries):
        return {"window_id": wid, "kind": kind, "session": sid, "episode_id": eid,
                "start_frame": start, "end_frame": end, "start_s": start / FPS, "end_s": end / FPS,
                "status": "excluded" if reasons else "included", "exclusion_reasons": _ordered(reasons),
                "features": summaries, "label_context": _label_context(data[sid], start, end),
                "preprocessing": {"truncated_at_onset": kind == "pre_event", "interpolation": False,
                                  "smoothing": "one_euro"}}

    for eid, e in zip(request["episode_ids"], eps):
        sid, onset = e["session"], e["start_frame"]
        end = onset - gap
        start = end - L
        reasons = []
        if start < 0:
            reasons.append("starts_before_session")
            start, end = 0, max(end, 0)
        hs = history_start(prefix_arrays(data[sid], end)[2], start - lookback) if end > 0 else 0
        if _any(is_beh[sid], hs, onset):
            reasons.append("overlaps_event")
        summaries = measure_window(data[sid], start, end, feats)
        reasons += _usability_reasons(summaries)
        if end > start:
            pre_spans[sid].append((start, end))
        windows.append(item(ic.pre_window_id(eid, L, gap), "pre_event", sid, eid, start, end, reasons, summaries))

    base = request["baseline"]
    thinned = 0
    if base is not None:
        Lb, C = base["length_frames"], base["clearance_frames"]
        for sid in sessions:
            eligible, excluded = [], []
            for a in range(0, data[sid].n_frames - Lb + 1, Lb):
                b = a + Lb
                hs = history_start(prefix_arrays(data[sid], b)[2], a - lookback)
                reasons = []
                if _any(is_beh[sid], a, b):
                    reasons.append("overlaps_event")
                if _any(is_beh[sid], hs - C, b + C):
                    reasons.append("near_event")
                if any(p0 < b and hs < p1 for p0, p1 in pre_spans[sid]):
                    reasons.append("overlaps_pre_window")
                summaries = measure_window(data[sid], a, b, feats)
                reasons += _usability_reasons(summaries)
                w = item(ic.baseline_window_id(source, behavior, sid, a, Lb), "baseline", sid, None,
                         a, b, reasons, summaries)
                (excluded if reasons else eligible).append(w)
            kept = _even_selection(eligible, base["max_windows_per_session"])
            thinned += len(eligible) - len(kept)
            windows += sorted(kept + excluded, key=lambda w: w["start_frame"])
    if len(windows) > ic.MAX_WINDOWS:
        return _error(tool, request, "invalid_request", f"{len(windows)} windows exceed the contract "
                      f"maximum of {ic.MAX_WINDOWS}; use longer baseline tiles or fewer sessions")

    methodology = _common_methodology() + [
        _method("pre_event_window", f"[onset - {gap} - {L}, onset - {gap})",
                "onset = the episode's first annotated frame; clipped to frame 0 and excluded "
                "(starts_before_session) if it would start earlier"),
        _method("feature_lookback_frames", lookback, "largest lookback of the requested features "
                f"({', '.join(f'{f} {FEATURE_LOOKBACK[f]}' for f in feats)})"),
        _method("history_start", "start - lookback, then back over smoother_memory_frames + 1 valid "
                "body-centre samples of each mouse",
                "used only by the exclusion rules; earlier frames are still read and together weigh "
                "< 1% in each smoothed position of the window (missing samples extend it)"),
        _method("event_overlap_rule", "behavior frame in [history start, onset)",
                "pre-event windows whose window, feature history or gap contains any frame annotated "
                "with the behavior (any episode of the session) are excluded as overlaps_event"),
        _method("usable_frame", "finite value and tracking_ok at t-1 and t",
                "both derivative endpoints; the same rule as Stage 1 history H = 1"),
        _method("min_usable_fraction", MIN_USABLE_FRACTION,
                "PROVISIONAL_SUFFICIENCY placeholder; the request has no field for it. Below it: "
                "insufficient_usable_frames; no usable frame: missing_data"),
        _method("summary", "median (primary) and mean of usable frames", "nothing is imputed"),
        _method("dist_rate_meaning", "negative = body-centre distance decreasing",
                "pair-level; does not identify which mouse approached"),
    ]
    if base is not None:
        methodology += [
            _method("baseline_rules", "tiles [k*length, (k+1)*length) of each requested session",
                    "overlaps_event: behavior frame in the tile; near_event: behavior frame within "
                    "clearance_frames of [history start, end); overlaps_pre_window: [history start, "
                    "end) overlaps a requested pre-event window (included or excluded)"),
            _method("baseline_selection", "eligible tiles thinned evenly to max_windows_per_session",
                    "deterministic; omitted eligible tiles are not listed"),
            _method("baseline_tiles_omitted", thinned, "eligible tiles not returned because of "
                    "max_windows_per_session"),
        ]
    methodology.append(_code_method())
    cov_methods, cov_limits = coverage_metadata(getattr(snap, "coverage", None))    # metadata only
    methodology += cov_methods
    limitations = cov_limits + list(FEATURE_LIMITATIONS)
    if snap.synthetic:
        limitations.insert(0, "SYNTHETIC test fixture: not CalMS21 data, not evidence.")
    any_included = any(w["status"] == "included" for w in windows)
    result = ic.make_result(tool, request, "ok" if any_included else "no_data", {"windows": windows},
                            methodology=methodology, limitations=limitations, provenance=_provenance(snap.data))
    ic.validate_result(tool, request, result, tools.policy)     # self-check
    try:
        snap.finalize()
    except SourceUnavailable as exc:
        return _error(tool, request, "source_unavailable", str(exc))
    return result


# ---- check_tracking_quality ----

QUALITY_LIMITATIONS = [
    "tracking_ok and endpoint validity describe whether keypoints passed the quality flags, not "
    "whether the measured motion is accurate; no frame was checked against video.",
    "Jump detection is off (no limit chosen): one-frame keypoint collapses that pass every flag are "
    "counted as tracking_ok and are not reported as smoothing contamination (Stage 1 report, section 4).",
    "Smoothing contamination only covers frames whose smoother memory contains a frame that failed "
    "tracking_ok; the trailing 1-s features (close_time_s, speed_corr_w) reach 30 frames further back.",
    "Per-flag counts (low score, out of frame, frozen) are not part of the contract result; they are "
    "folded into tracking_ok.",
    "The verdict's usable-fraction minimum is a provisional placeholder.",
    ANNOTATION_NOTE,
]


def _window_span(w, sess):
    """[start, end) of a requested window, or UnknownWindow / OutOfRange."""
    L = w["length_frames"]
    if w["kind"] == "pre_event":
        if not _is_episode_start(sess, w["behavior"], w["onset_frame"]):
            raise UnknownWindow(f"{w['episode_id']} is not the first frame of an annotated "
                                f"{w['behavior']!r} run in the verified session")
        end = w["onset_frame"] - w["gap_frames"]
        start = end - L
        if start < 0:
            raise OutOfRange("a pre-event window that starts before frame 0 has no full window to "
                             "assess (get_window_features reports it as starts_before_session)")
        return start, end
    start = w["start_frame"]
    if start % L or start + L > sess.n_frames:
        raise UnknownWindow(f"baseline window at frame {start} is not a whole tile of length {L} "
                            "inside the session")
    return start, start + L


def assess_window(sess, start, end, history):
    """Quality of frames [start, end) from the session truncated at `end`
    (earlier frames are used for history; no frame at or after `end` influences it)."""
    n = end - start
    K = SMOOTHER_MEMORY_FRAMES
    ok, _, _ = prefix_arrays(sess, end)
    cur = ok[start:end]
    endpoint = _endpoint_valid(ok, start, end)
    run = np.zeros(end, dtype=int)                   # consecutive tracking_ok frames ending at t
    count = 0
    for i, v in enumerate(ok[:end]):
        count = count + 1 if v else 0
        run[i] = count
    usable = {str(h): int((endpoint & (run[start:end] >= h + 1)).sum()) for h in sorted(history)}
    bad = np.r_[0, np.cumsum(~ok[:end])]             # failed frames in ok[:i]
    idx = np.arange(start, end)
    recent_bad = bad[idx] - bad[np.maximum(idx - K, 0)]   # failed frames among the K before t
    contaminated = endpoint & (recent_bad > 0)
    frames = np.arange(start, end)
    valid = int(endpoint.sum())
    if valid == 0:
        verdict = "insufficient_data"
    elif valid < MIN_USABLE_FRACTION * n - ic.STAT_ABS_TOL:
        verdict = "excluded"
    else:
        verdict = "usable_with_caveats"
    return {
        "start_frame": start, "end_frame": end, "frames": n,
        "tracking_ok_frames": int(cur.sum()), "endpoint_valid_frames": valid,
        "usable_by_history": usable,
        "excluded_frames": {"not_tracking_ok": frames[~cur].tolist(),
                            "endpoint_invalid": frames[cur & ~endpoint].tolist()},
        "smoothing_contamination": {
            "status": "detected" if contaminated.any() else "none_detected",
            "frames": frames[contaminated].tolist(),
            "note": (f"endpoint-valid frames with a frame failing tracking_ok among the {K} frames before "
                     "them (One-Euro memory, < 1% weight beyond); flag-passing keypoint collapses are not "
                     "detected (no jump limit)")},
        "tracking_ok_meaning": ic.TRACKING_OK_MEANING,
        "motion_accuracy_verified": False,
        "verdict": verdict,
    }


def check_tracking_quality(tools, request):
    tool = "check_tracking_quality"
    ic.validate_request(tool, request, tools.policy)
    parsed = [ic.parse_window_id(w) for w in request["window_ids"]]
    src = tools.sources.get("validation_annotation")
    if src is None or any(w["source"] != "validation_annotation" for w in parsed):
        return _error(tool, request, "source_unavailable", "only windows of validation_annotation "
                      "episodes / sessions can be assessed by this implementation")
    sessions = sorted({w["session"] for w in parsed})
    try:
        snap = src.snapshot(sessions)
        data = {sid: load_session(sid, snap.sessions[sid]) for sid in sessions}
    except SourceUnavailable as exc:
        return _error(tool, request, "source_unavailable", str(exc))
    items = []
    for wid, w in zip(request["window_ids"], parsed):
        try:
            start, end = _window_span(w, data[w["session"]])
        except UnknownWindow as exc:
            return _error(tool, request, "unknown_id", f"{wid}: {exc}")
        except OutOfRange as exc:
            return _error(tool, request, "invalid_request", f"{wid}: {exc}")
        items.append({"window_id": wid, **assess_window(data[w["session"]], start, end,
                                                        request["history_frames"])})
    methodology = _common_methodology() + [
        _method("endpoint_valid", "tracking_ok at t-1 and t", "both endpoints of a frame-to-frame rate"),
        _method("usable_by_history", "endpoint-valid and tracking_ok on every frame t-h .. t",
                "frames before the window are read (past only); before frame 0 nothing is usable"),
        _method("smoothing_contamination", f"failed frame among the {SMOOTHER_MEMORY_FRAMES} frames before t",
                "only for endpoint-valid frames"),
        _method("verdict", f"insufficient_data: 0 endpoint-valid frames; excluded: below "
                f"{MIN_USABLE_FRACTION} of the window; otherwise usable_with_caveats",
                "provisional; never 'accurate'"),
        _code_method(),
    ]
    cov_methods, cov_limits = coverage_metadata(getattr(snap, "coverage", None))    # metadata only
    methodology += cov_methods
    limitations = cov_limits + list(QUALITY_LIMITATIONS)
    if snap.synthetic:
        limitations.insert(0, "SYNTHETIC test fixture: not CalMS21 data, not evidence.")
    status = "ok" if any(q["endpoint_valid_frames"] > 0 for q in items) else "no_data"
    result = ic.make_result(tool, request, status, {"windows": items}, methodology=methodology,
                            limitations=limitations, provenance=_provenance(snap.data))
    ic.validate_result(tool, request, result, tools.policy)
    try:
        snap.finalize()
    except SourceUnavailable as exc:
        return _error(tool, request, "source_unavailable", str(exc))
    return result
