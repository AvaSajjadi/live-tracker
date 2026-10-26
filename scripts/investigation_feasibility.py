"""EXPERIMENTAL (Investigation mode, Stage 1): feasibility measurements for
"Does approach tend to precede attack?" on verified CalMS21 validation
sessions (new file).

Descriptive counts only. Nothing here is a validated setting or a result:
every value that is varied (window length, baseline clearance, history
length, coverage bins) is a REPORTING parameter, listed in PROVISIONAL.

Input boundary: only sessions published by live_tracker.investigation_data
(index bound to the source + manifest, per-file sha256, session IDs read from
the file contents and checked against the manifest's validation split).

Preprocessing policy for temporal evidence (provisional, causality-safe):
- Every pre-attack window is computed from a copy of the session TRUNCATED
  at that episode's onset frame, so no frame at or after onset can influence
  its flags or features. This is needed because two steps of the default
  preprocessing look ahead: quality.interpolate_short_gaps fills a gap from
  valid frames on both sides, and quality.flag_tracking's "frozen" flag marks
  a frame frozen when it belongs to a run of identical positions that may
  continue after onset. Baseline tiles use the full session (they do not
  precede an event); the difference this makes inside the retained
  pre-attack windows is measured and reported ("causality_check").
- quality.flag_tracking with the CalMS21 settings used by calms21.flag_quality
  on igor/day2 (frame size, low_conf below SUGGESTED_MIN_SCORE), and WITHOUT
  quality.interpolate_short_gaps (no imputed coordinates at all).
- features.compute_features on those flags (it skips its own flagging and
  interpolation when "valid" is present), Ava's default FeatureParams
  (causal One-Euro smoothing; dist_rate = backward difference).
- A frame t counts as usable at history H when dist_rate[t] is finite and
  tracking_ok holds on every frame t-H .. t. H = 1 covers both derivative
  endpoints. H = 15 is only a provisional consecutive-quality sensitivity
  criterion. It is NOT a bound on the smoother's memory: the One-Euro filter
  is recursive, so a bad position decays geometrically (by about 1 - alpha =
  0.83 per frame at the default 1 Hz cutoff) but never fully disappears, and
  the filter only restarts after a gap longer than max_gap_s, not after a bad
  but valid-looking frame. The anomaly scan measures this directly.
- tracking_ok is NOT motion accuracy: score-based flags do not catch
  plausible single-frame identity / keypoint collapses (see the anomaly scan).

Usage:
    python scripts/investigation_feasibility.py data/raw/calms21_validation \
        --manifest splits/split_manifest.json --out results.json \
        [--write-doc docs/investigation_experiment_01.md | --check-doc docs/...]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from live_tracker.calms21 import FPS, IMAGE_SIZE, TORSO_KEYPOINTS, load_calms21
from live_tracker.features import FeatureParams, body_center, compute_features
from live_tracker.investigation_data import (peak_rss_mb, sha256_file, unlisted_files,
                                             verified_sessions)
from live_tracker.quality import SUGGESTED_MIN_SCORE, flag_tracking

ROOT = Path(__file__).resolve().parents[1]
WINDOW_S = (1, 2, 3)
CLEARANCE_S = (0, 5, 10)
HISTORY_FRAMES = (1, 15)
PRIMARY_HISTORY = 1
COVERAGE_BINS = (0.5, 0.9, 1.0)
TOP_ANOMALIES = 5
CONTAMINATION_LAGS = (2, 5, 10, 15, 16, 20, 30, 45)
LABELS = ("attack", "investigation", "mount", "other")
PROVISIONAL = {
    "window_s": "requested reporting values (1, 2, 3 s), not a chosen window",
    "clearance_s": "reporting values for baseline distance from attack frames, not a rule",
    "history_frames": "1 = derivative endpoints; 15 = consecutive-quality sensitivity criterion, "
                      "not a bound on smoother memory; neither validated",
    "primary_history": "H = 1 used for the main tables; H = 15 shown as sensitivity",
    "coverage_bins": "reporting bins, not an acceptance threshold",
    "episode": "maximal run of 'attack' frames, no merging, no minimum duration",
    "baseline": "non-overlapping tiles from frame 0, one possible construction",
    "summary": "per-window median and mean of usable dist_rate frames",
}
GEN_BEGIN = "<!-- BEGIN GENERATED: scripts/investigation_feasibility.py -->"
GEN_END = "<!-- END GENERATED -->"


class FPSMismatch(ValueError):
    """time_s does not equal frame / calms21.FPS."""


# ---- preprocessing and frame quality ----

def preprocess(poses):
    """Causality-safe flags, features and raw body centres (no interpolation)."""
    flags = flag_tracking(poses, frame_size=IMAGE_SIZE, score_col="score",
                          min_score=SUGGESTED_MIN_SCORE)
    if "imputed" in flags.columns:  # guard: nothing may have been filled from the future
        raise RuntimeError("imputed points found; temporal evidence must not be interpolated")
    feats = compute_features(flags, torso_keypoints=TORSO_KEYPOINTS)
    centres = body_center(flags, TORSO_KEYPOINTS)
    return flags, feats, centres


def frame_arrays(feats, n):
    """tracking_ok and dist_rate as arrays over frames 0 .. n-1."""
    f = feats.set_index("frame").reindex(range(n))
    return (f["tracking_ok"].fillna(False).to_numpy(dtype=bool),
            f["dist_rate"].to_numpy(dtype=float))


def causal_arrays(poses, onset):
    """tracking_ok and dist_rate for frames < onset, computed only from frames < onset."""
    _, feats, _ = preprocess(poses[poses["frame"] < onset])
    return frame_arrays(feats, onset)


def check_fps(frame, time_s, what):
    frame = np.asarray(frame, dtype=float)
    if not np.allclose(np.asarray(time_s, dtype=float), frame / FPS, rtol=0, atol=1e-9):
        raise FPSMismatch(f"{what}: time_s is not frame / {FPS} (calms21.FPS)")


def usable_frames(tracking_ok, dist_rate, history):
    """usable[t]: finite dist_rate[t] and tracking_ok on frames t-history .. t."""
    ok = np.asarray(tracking_ok, dtype=bool)
    dr = np.asarray(dist_rate, dtype=float)
    n = len(ok)
    run = np.zeros(n, dtype=int)  # consecutive tracking_ok frames ending at t
    count = 0
    for i, v in enumerate(ok):
        count = count + 1 if v else 0
        run[i] = count
    return np.isfinite(dr) & (run >= history + 1)


# ---- episodes and windows ----

def attack_runs(is_attack):
    """[(start, end_exclusive)] of maximal True runs."""
    a = np.asarray(is_attack, dtype=bool)
    if a.size == 0:
        return []
    edges = np.diff(np.r_[0, a.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def window_frames(seconds):
    return int(round(seconds * FPS))


def pre_window(onset, length):
    """Frames [onset - length, onset), or None if it starts before frame 0."""
    start = onset - length
    return None if start < 0 else (start, onset)


def baseline_tiles(n_frames, is_attack, length, clearance, blocked=()):
    """Non-overlapping [a, a+length) tiles from frame 0. Returns (kept,
    excluded) where excluded maps tile -> reason: 'near_attack' (a frame within
    `clearance` frames of an attack frame) or 'overlaps_pre_window' (any frame
    in a `blocked` interval)."""
    att = np.asarray(is_attack, dtype=bool)
    near = att.copy()
    if clearance > 0 and att.any():
        for i in np.flatnonzero(att):
            near[max(0, i - clearance):min(n_frames, i + clearance + 1)] = True
    pre = np.zeros(n_frames, dtype=bool)
    for a, b in blocked:
        pre[a:b] = True
    kept, excluded = [], {}
    for a in range(0, n_frames - length + 1, length):
        if near[a:a + length].any():
            excluded[(a, a + length)] = "near_attack"
        elif pre[a:a + length].any():
            excluded[(a, a + length)] = "overlaps_pre_window"
        else:
            kept.append((a, a + length))
    return kept, excluded


def _r(x, nd=2):
    """Rounded float, or None for missing / non-finite values."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def window_record(window_id, kind, session, a, b, ok, dr, usable, labels, **extra):
    """Everything about frames [a, b), with frame-level exclusions."""
    frames = np.arange(a, b)
    lab = np.asarray(labels[a:b])
    rec = {"window_id": window_id, "kind": kind, "session": session,
           "start_frame": int(a), "end_frame": int(b), "start_s": _r(a / FPS, 3),
           "frames": int(b - a), **extra,
           "tracking_ok_frac": _r(np.mean(ok[a:b]), 4),
           "labels": {k: int((lab == k).sum()) for k in LABELS},
           "excluded_frames": {
               "not_tracking_ok": frames[~ok[a:b]].tolist(),
               "dist_rate_not_finite": frames[~np.isfinite(dr[a:b])].tolist()},
           "by_history": {}}
    for h, use in usable.items():
        u = use[a:b]
        vals = dr[a:b][u]
        rec["by_history"][str(h)] = {
            "usable_frac": _r(u.mean(), 4),
            "not_usable_frames": frames[~u].tolist(),
            "dist_rate_median": _r(np.median(vals)) if vals.size else None,
            "dist_rate_mean": _r(vals.mean()) if vals.size else None}
    return rec


# ---- anomaly scan (no thresholds: the largest raw steps, described) ----

def smoother_contamination(poses, dist_rate, frame, mouse, lags=CONTAMINATION_LAGS):
    """How much one frame's positions of one mouse still change the smoothed
    dist_rate `lag` frames later: actual minus a counterfactual in which that
    mouse's points at `frame` are removed (everything else identical).
    Returns {lag: px/s or None}."""
    cf = poses.copy()
    cf.loc[(cf["frame"] == frame) & (cf["mouse"] == mouse), ["x", "y"]] = np.nan
    _, feats, _ = preprocess(cf)
    cf_dr = feats.set_index("frame")["dist_rate"].reindex(range(len(dist_rate))).to_numpy(float)
    out = {}
    for k in lags:
        t = frame + k
        out[str(k)] = _r(dist_rate[t] - cf_dr[t], 3) if t < len(dist_rate) else None
    return out


def anomaly_scan(centres, flags, ok, dr, labels, poses=None, top=TOP_ANOMALIES):
    """The `top` largest one-frame jumps of a raw (unsmoothed) body centre on
    frames with tracking_ok, with raw diagnostics. Not classified as errors.
    Values that do not exist (e.g. the frame after the last one) are None."""
    piv = centres.pivot(index="frame", columns="mouse", values=["x", "y"]).sort_index()
    mice = sorted(piv["x"].columns, key=str)
    n = len(ok)
    xy = {m: np.c_[piv["x"][m].reindex(range(n)).to_numpy(float),
                   piv["y"][m].reindex(range(n)).to_numpy(float)] for m in mice}
    scores = flags[flags["keypoint"].isin(TORSO_KEYPOINTS)].pivot_table(
        index="frame", columns=["mouse", "keypoint"], values="score")
    cand = []
    for m in mice:
        c = xy[m]
        step = np.r_[np.nan, np.hypot(*(c[1:] - c[:-1]).T)]
        for t in np.flatnonzero(np.isfinite(step) & ok):
            cand.append((float(step[t]), m, int(t)))
    cand.sort(reverse=True)
    out = []
    for step, m, t in cand[:top]:
        other = [o for o in mice if o != m][0]
        c, co = xy[m], xy[other]
        back = np.hypot(*(c[t + 1] - c[t - 1])) if t + 1 < n else None
        out.append({
            "frame": t, "mouse": m, "label": str(labels[t]),
            "step_px": _r(step, 1), "return_px": _r(back, 1),
            "dist_to_other_mouse_centre_px": _r(np.hypot(*(c[t] - co[t])), 1),
            "dist_to_other_mouse_centre_prev_px": _r(np.hypot(*(c[t - 1] - co[t - 1])), 1),
            "torso_scores": {k: _r(scores[(m, k)].get(t) if (m, k) in scores else None, 2)
                             for k in TORSO_KEYPOINTS},
            "torso_scores_prev": {k: _r(scores[(m, k)].get(t - 1) if (m, k) in scores else None, 2)
                                  for k in TORSO_KEYPOINTS},
            "tracking_ok": bool(ok[t]),
            "dist_rate_t_to_t+3": [_r(v, 1) for v in dr[t:t + 4]],
            "smoother_contamination_px_s": (smoother_contamination(poses, dr, t, m)
                                            if poses is not None else None),
            "verification": "unverified (no video available)"})
    return out


def centre_step_quantiles(centres, n):
    """Per mouse: distribution of one-frame raw body-centre steps (px), no threshold."""
    out = {}
    for m, part in centres.groupby("mouse", sort=True):
        part = part.set_index("frame").reindex(range(n))
        step = np.hypot(np.diff(part["x"].to_numpy(float)), np.diff(part["y"].to_numpy(float)))
        step = step[np.isfinite(step)]
        q = np.percentile(step, [50, 99, 99.9, 100]) if step.size else [np.nan] * 4
        out[str(m)] = {"n": int(step.size), "p50": _r(q[0], 1), "p99": _r(q[1], 1),
                       "p99_9": _r(q[2], 1), "max": _r(q[3], 1)}
    return out


def _quant(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [0, 25, 50, 75, 100])
    return {"n": int(v.size), "min": _r(q[0]), "q25": _r(q[1]), "median": _r(q[2]),
            "q75": _r(q[3]), "max": _r(q[4])}


def _bins(records, h=PRIMARY_HISTORY):
    return {f">={c}": sum(r["by_history"][str(h)]["usable_frac"] >= c for r in records)
            for c in COVERAGE_BINS}


# ---- one session ----

def analyse_session(sid, data):
    """All measurements for one verified session dict ({group: {sid: rec}})."""
    poses, labels = load_calms21(data)
    if sorted(labels["session"].unique()) != [sid]:
        raise ValueError(f"loaded sessions {sorted(labels['session'].unique())} != [{sid!r}]")
    labels = labels.sort_values("frame").reset_index(drop=True)
    check_fps(labels["frame"], labels["time_s"], f"{sid} labels")
    if not np.array_equal(labels["frame"].to_numpy(), np.arange(len(labels))):
        raise ValueError(f"{sid}: frames are not 0 .. n-1")
    flags, feats, centres = preprocess(poses)
    check_fps(feats["frame"], feats["time_s"], f"{sid} features")
    n = len(labels)
    lab = labels["label"].to_numpy()
    ok, dr = frame_arrays(feats, n)                 # full session: baseline + anomaly scan
    usable = {h: usable_frames(ok, dr, h) for h in HISTORY_FRAMES}
    is_att = lab == "attack"
    runs = attack_runs(is_att)
    short = sid.split("/")[-1]

    s = {"session": sid, "frames": n, "seconds": _r(n / FPS, 1),
         "tracking_ok_frac": _r(ok.mean(), 4),
         "dist_rate_finite_frac": _r(np.isfinite(dr).mean(), 4),
         "usable_frac": {str(h): _r(u.mean(), 4) for h, u in usable.items()},
         "attack_episodes": [{"episode_id": f"{short}:attack:{a}", "start_frame": a, "end_frame": b,
                              "onset_s": _r(a / FPS, 3), "duration_s": _r((b - a) / FPS, 3)}
                             for a, b in runs],
         "episode_durations_s": _quant([(b - a) / FPS for a, b in runs]),
         "gaps_between_episodes_s": _quant([(runs[i][0] - runs[i - 1][1]) / FPS
                                            for i in range(1, len(runs))]),
         "raw_centre_step_px": centre_step_quantiles(centres, n),
         "anomalies": anomaly_scan(centres, flags, ok, dr, lab, poses),
         "windows": {}, "baseline": {}}

    # causal (truncated-at-onset) arrays per episode, shared by all window lengths
    causal = {}
    for a_on, _ in runs:
        if a_on >= window_frames(min(WINDOW_S)):
            c_ok, c_dr = causal_arrays(poses, a_on)
            causal[a_on] = (c_ok, c_dr, {h: usable_frames(c_ok, c_dr, h) for h in HISTORY_FRAMES})
    check = {"tracking_ok_differs": 0, "dist_rate_differs": 0, "frames_compared": 0}

    for L in WINDOW_S:
        length = window_frames(L)
        pre, dropped = [], []
        for a_on, _ in runs:
            ep = f"{short}:attack:{a_on}"
            w = pre_window(a_on, length)
            if w is None:
                dropped.append({"episode_id": ep, "reason": "starts_before_session"})
                continue
            c_ok, c_dr, c_use = causal[a_on]
            rec = window_record(f"{short}:pre:{L}s:{a_on}", "pre_attack", sid, *w, c_ok, c_dr, c_use,
                                lab, episode_id=ep, onset_frame=a_on, preprocessing="truncated_at_onset")
            # how much the causal (truncated) computation differs from the full-session one
            a, b = w
            same_dr = (c_dr[a:b] == dr[a:b]) | (np.isnan(c_dr[a:b]) & np.isnan(dr[a:b]))
            check["frames_compared"] += b - a
            check["tracking_ok_differs"] += int((c_ok[a:b] != ok[a:b]).sum())
            check["dist_rate_differs"] += int((~same_dr).sum())
            if rec["labels"]["attack"]:
                dropped.append({"episode_id": ep, "reason": "overlaps_earlier_attack",
                                "window_id": rec["window_id"]})
                continue
            pre.append(rec)
        s["windows"][f"{L}s"] = {"retained": pre, "dropped": dropped}
        blocked = [(r["start_frame"], r["end_frame"]) for r in pre]
        s["baseline"][f"{L}s"] = {}
        for C in CLEARANCE_S:
            kept, excluded = baseline_tiles(n, is_att, length, window_frames(C), blocked)
            recs = [window_record(f"{short}:base:{L}s:C{C}:{a}", "baseline", sid, a, b, ok, dr,
                                  usable, lab) for a, b in kept]
            reasons = {}
            for r in excluded.values():
                reasons[r] = reasons.get(r, 0) + 1
            s["baseline"][f"{L}s"][f"C{C}"] = {"retained": recs, "excluded_counts": reasons}
    # which retained windows contain a listed jump or the 3 frames after it
    # (the smoothed dist_rate stays affected for several frames)
    for an in s["anomalies"]:
        span = set(range(an["frame"], an["frame"] + 4))
        an["in_pre_windows"] = sorted(r["window_id"] for w in s["windows"].values() for r in w["retained"]
                                      if span & set(range(r["start_frame"], r["end_frame"])))
        an["in_baseline_tiles"] = sum(1 for byc in s["baseline"].values() for b in byc.values()
                                      for r in b["retained"]
                                      if span & set(range(r["start_frame"], r["end_frame"])))
    s["causality_check"] = {
        "meaning": "frames of all candidate pre-attack windows (all window lengths, incl. later "
                   "dropped ones) where the truncated-at-onset computation differs from the "
                   "full-session one",
        **check}
    return s


# ---- provenance ----

def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def code_identity():
    files = ["scripts/investigation_feasibility.py", "src/live_tracker/investigation_data.py",
             "src/live_tracker/calms21.py", "src/live_tracker/quality.py",
             "src/live_tracker/features.py", "src/live_tracker/filters.py"]
    return {"git_head": _git("rev-parse", "HEAD"),
            "files_sha256": {f: sha256_file(ROOT / f) for f in files}}


def settings():
    return {"fps": FPS, "frame_size": list(IMAGE_SIZE), "torso_keypoints": list(TORSO_KEYPOINTS),
            "flag_tracking": {"score_col": "score", "min_score": SUGGESTED_MIN_SCORE,
                              "max_speed_px_s": None, "jump_detection": "off (flag_tracking default)"},
            "interpolate_short_gaps": "not used (two-sided; would use frames after onset)",
            "feature_params": dataclasses.asdict(FeatureParams()),
            "window_s": list(WINDOW_S), "clearance_s": list(CLEARANCE_S),
            "history_frames": list(HISTORY_FRAMES), "primary_history": PRIMARY_HISTORY,
            "coverage_bins": list(COVERAGE_BINS), "provisional": PROVISIONAL}


def run(cache_dir, manifest_path):
    t0 = time.perf_counter()
    verified, index = verified_sessions(cache_dir, manifest_path)
    sessions = [analyse_session(sid, data) for sid, data in verified]
    return {"experimental": True, "descriptive_only": True,
            "inputs": {"source": {k: index["source"][k] for k in ("path", "size", "sha256")},
                       "manifest_sha256": index["manifest"]["sha256"],
                       "files": {sid: {"file": e["file"], "sha256": e["sha256"]}
                                 for sid, e in index["sessions"].items()},
                       "ignored_unlisted_files": unlisted_files(cache_dir, index)},
            "split_label_summary": index["split_label_summary"],
            "settings": settings(), "code": code_identity(), "sessions": sessions,
            "run": {"seconds": _r(time.perf_counter() - t0, 1), "peak_rss_mb": peak_rss_mb(),
                    "extraction": index.get("extraction_run")}}


# ---- generated report section (deterministic: no timings) ----

def _fmt(v):
    """A number for the report, or 'n/a' when it does not exist."""
    return "n/a" if v is None else f"{v:g}"


def _q(d, keys=("min", "q25", "median", "q75", "max")):
    return " / ".join("–" if d.get(k) is None else f"{d[k]:g}" for k in keys) if d.get("n") else "–"


def render_markdown(res):
    L = [GEN_BEGIN, "",
         "*Generated by `scripts/investigation_feasibility.py`; check with `--check-doc`. "
         "Descriptive only.*", "",
         "#### Inputs and code", "",
         f"- Source: `{res['inputs']['source']['path']}` ({res['inputs']['source']['size']:,} bytes, "
         f"sha256 `{res['inputs']['source']['sha256']}`)",
         f"- Split manifest sha256 `{res['inputs']['manifest_sha256']}`"]
    for sid, f in res["inputs"]["files"].items():
        L.append(f"- `{sid}` → `{f['file']}` sha256 `{f['sha256']}`")
    L.append(f"- Ignored unlisted files in the cache directory: {res['inputs']['ignored_unlisted_files'] or 'none'}")
    # git HEAD is in the JSON output only: it changes with every commit (including the one
    # that adds this report), so it cannot be part of a section that must keep matching
    L.append("- Code (sha256 of the exact files used; git HEAD is recorded in the JSON output):")
    for f, h in res["code"]["files_sha256"].items():
        L.append(f"  - `{f}` sha256 `{h}`")
    st = res["settings"]
    L += ["", f"- Settings: FPS {st['fps']:g}; torso {', '.join(st['torso_keypoints'])}; "
          f"min_score {st['flag_tracking']['min_score']}; jump detection off; "
          f"interpolation not used; FeatureParams default "
          f"(smooth {st['feature_params']['smooth']}, {st['feature_params']['smooth_kwargs']}, "
          f"max_step_s {st['feature_params']['max_step_s']}); history H ∈ {st['history_frames']} "
          f"(primary {st['primary_history']})", ""]

    L += ["#### Validation split: annotation runs (all 14 sessions, annotations only)", "",
          "| Session | Frames | Attack runs | Investigation runs | Mount runs |", "|---|---:|---:|---:|---:|"]
    tot = 0
    for sid, v in res["split_label_summary"].items():
        r = v["label_runs"]
        tot += r["attack"]
        L.append(f"| {sid.split('/')[-1]} | {v['frames']:,} | {r['attack']} | {r['investigation']} | {r['mount']} |")
    n_att = sum(v["label_runs"]["attack"] > 0 for v in res["split_label_summary"].values())
    L += ["", f"Attack runs in validation: **{tot}**, in **{n_att}** of "
          f"{len(res['split_label_summary'])} sessions.", ""]

    L += ["#### Extracted sessions", "",
          "| Session | Seconds | tracking_ok | finite dist_rate | usable H=1 | usable H=15 | Attack episodes | Episode duration s (min/q25/med/q75/max) | Gap between episodes s (min/q25/med/q75/max) |",
          "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for s in res["sessions"]:
        L.append(f"| {s['session'].split('/')[-1]} | {s['seconds']:g} | {s['tracking_ok_frac']:.4f} | "
                 f"{s['dist_rate_finite_frac']:.4f} | {s['usable_frac']['1']:.4f} | {s['usable_frac']['15']:.4f} | "
                 f"{len(s['attack_episodes'])} | {_q(s['episode_durations_s'])} | {_q(s['gaps_between_episodes_s'])} |")

    for s in res["sessions"]:
        if not s["attack_episodes"]:
            continue
        short = s["session"].split("/")[-1]
        L += ["", f"#### Pre-attack windows: {short}", "",
              "| Window | Episodes | Dropped: before session start | Dropped: overlaps earlier attack | Retained | usable frac H=1 (min/median) | ≥0.9 usable H=1 | ≥0.9 usable H=15 | Per-window median dist_rate H=1 (min/q25/med/q75/max) | Per-window mean dist_rate H=1 (min/q25/med/q75/max) | Frames: investigation / other / mount |",
              "|---|---:|---:|---:|---:|---|---:|---:|---|---|---|"]
        for Lw, w in s["windows"].items():
            ret = w["retained"]
            dr_ = [d["reason"] for d in w["dropped"]]
            u = _quant([r["by_history"]["1"]["usable_frac"] for r in ret])
            lab = {k: sum(r["labels"][k] for r in ret) for k in LABELS}
            L.append(f"| {Lw} | {len(s['attack_episodes'])} | {dr_.count('starts_before_session')} | "
                     f"{dr_.count('overlaps_earlier_attack')} | {len(ret)} | "
                     f"{_q(u, ('min', 'median'))} | {_bins(ret, 1)['>=0.9']} | {_bins(ret, 15)['>=0.9']} | "
                     f"{_q(_quant([r['by_history']['1']['dist_rate_median'] for r in ret]))} | "
                     f"{_q(_quant([r['by_history']['1']['dist_rate_mean'] for r in ret]))} | "
                     f"{lab['investigation']} / {lab['other']} / {lab['mount']} |")
        L += ["", f"Retained pre-attack window IDs and onsets (s): " + "; ".join(
            f"`{r['window_id']}` ({r['start_s'] + int(Lw[:-1]):g})"
            for Lw, w in s["windows"].items() if Lw == "1s" for r in w["retained"])]

        L += ["", f"#### Same-session baseline tiles: {short}", "",
              "| Window | Clearance | Retained | Excluded near attack | Excluded overlapping pre-window | ≥0.9 usable H=1 | Frames: investigation / other | Tile start s (q25/med/q75) | Per-window median dist_rate H=1 (q25/med/q75) |",
              "|---|---:|---:|---:|---:|---:|---|---|---|"]
        for Lw, byc in s["baseline"].items():
            for C, b in byc.items():
                ret = b["retained"]
                lab = {k: sum(r["labels"][k] for r in ret) for k in LABELS}
                ex = b["excluded_counts"]
                L.append(f"| {Lw} | {C[1:]} s | {len(ret)} | {ex.get('near_attack', 0)} | "
                         f"{ex.get('overlaps_pre_window', 0)} | {_bins(ret, 1)['>=0.9']} | "
                         f"{lab['investigation']} / {lab['other']} | "
                         f"{_q(_quant([r['start_s'] for r in ret]), ('q25', 'median', 'q75'))} | "
                         f"{_q(_quant([r['by_history']['1']['dist_rate_median'] for r in ret]), ('q25', 'median', 'q75'))} |")

    L += ["", "#### One-frame raw body-centre steps (px), all frames with both endpoints present", "",
          "| Session | Mouse | n | p50 | p99 | p99.9 | max |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in res["sessions"]:
        for m, q in s["raw_centre_step_px"].items():
            L.append(f"| {s['session'].split('/')[-1]} | {m} | {q['n']} | {_fmt(q['p50'])} | "
                     f"{_fmt(q['p99'])} | {_fmt(q['p99_9'])} | {_fmt(q['max'])} |")
    L += ["", f"#### Largest {TOP_ANOMALIES} one-frame raw body-centre steps per session on tracking_ok frames (unverified)", "",
          "| Session | Frame | Mouse | Step px | Return px (t−1→t+1) | Distance to other mouse px (t−1 → t) | Torso scores t−1 → t (neck, left_hip, right_hip) | dist_rate t..t+3 (smoothed) | Smoother contamination px/s at t+15 / t+16 / t+30 / t+45 | Label | In retained pre-windows (t..t+3) | In retained baseline tiles (t..t+3) |",
          "|---|---:|---|---:|---:|---|---|---|---|---|---|---:|"]
    for s in res["sessions"]:
        for a in s["anomalies"]:
            sc = lambda d: ", ".join(_fmt(d[k]) for k in TORSO_KEYPOINTS)
            cont = a.get("smoother_contamination_px_s") or {}
            L.append(f"| {s['session'].split('/')[-1]} | {a['frame']} | {a['mouse']} | {_fmt(a['step_px'])} | "
                     f"{_fmt(a['return_px'])} | {_fmt(a['dist_to_other_mouse_centre_prev_px'])} → "
                     f"{_fmt(a['dist_to_other_mouse_centre_px'])} | {sc(a['torso_scores_prev'])} → "
                     f"{sc(a['torso_scores'])} | {', '.join(_fmt(v) for v in a['dist_rate_t_to_t+3'])} | "
                     f"{' / '.join(_fmt(cont.get(k)) for k in ('15', '16', '30', '45'))} | {a['label']} | "
                     f"{', '.join(a['in_pre_windows']) or 'none'} | {a['in_baseline_tiles']} |")
    L += ["", GEN_END]
    return "\n".join(L)


def splice(doc_text, block):
    a, b = doc_text.find(GEN_BEGIN), doc_text.find(GEN_END)
    if a < 0 or b < a:
        raise ValueError("document has no generated section markers")
    return doc_text[:a] + block + doc_text[b + len(GEN_END):]


def main(argv=None):
    ap = argparse.ArgumentParser(description="EXPERIMENTAL feasibility counts (descriptive only)")
    ap.add_argument("cache_dir")
    ap.add_argument("--manifest", default=str(ROOT / "splits" / "split_manifest.json"))
    ap.add_argument("--out")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write-doc")
    g.add_argument("--check-doc")
    a = ap.parse_args(argv)
    res = run(a.cache_dir, a.manifest)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    block = render_markdown(res)
    if a.write_doc:
        p = Path(a.write_doc)
        p.write_text(splice(p.read_text(encoding="utf-8"), block), encoding="utf-8")
    if a.check_doc:
        text = Path(a.check_doc).read_text(encoding="utf-8")
        if splice(text, block) != text:
            print("generated section differs from the document", file=sys.stderr)
            return 1
        print("generated section matches")
    print(json.dumps({"sessions": len(res["sessions"]), "run": res["run"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
