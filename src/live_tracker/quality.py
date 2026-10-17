"""Tracking-quality checks for pose trajectories.

New file for Live Tracker (written by us, not part of the starter).

The main rule: missing tracking is NOT the same as an animal standing still.
So instead of silently fixing bad data, we mark it with flag columns:

  missing       x or y is NaN
  out_of_frame  point is outside the image (only if frame size is given)
  jump          speed between two frames is physically implausible
  frozen        exactly the same coordinates for many frames in a row
                (real trackers always jitter a little, so this usually means
                the tracker got stuck, not that the mouse stopped)
  low_conf      tracker score below min_score (only if a score column is given)
  imputed       value was filled in by interpolate_short_gaps()

All thresholds are placeholders. They must be chosen on training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Chosen on the 14 CalMS21 validation sessions (torso points): torso points scoring
# below this jump a median 8+ px/frame (3 px above 0.7) and it costs ~1% of frames
# (~3% of attack frames). Opt-in: pass it as min_score.
SUGGESTED_MIN_SCORE = 0.15


def _groups(df, group_cols):
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return [df.index]
    return list(df.groupby(group_cols, sort=False).groups.values())


def _runs(mask):
    """Return (start, end) index pairs (end exclusive) of True runs."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))


def flag_tracking(df, x_col="x", y_col="y", time_col="time_s",
                  group_cols=("session", "mouse", "keypoint"),
                  frame_size=None, frozen_min_frames=15, frozen_tol_px=1e-6,
                  max_speed_px_s=None, score_col=None, min_score=None):
    """Add quality flag columns. Returns a copy; raw x/y are not changed.

    low_conf is only checked when both score_col and min_score are given.
    """
    out = df.copy()
    for col in ("missing", "out_of_frame", "jump", "frozen", "low_conf"):
        out[col] = False

    out["missing"] = out[x_col].isna() | out[y_col].isna()

    if score_col is not None and min_score is not None:
        # a missing score is not evidence of low confidence; "missing" covers lost points
        out["low_conf"] = ~out["missing"] & (out[score_col] < min_score)

    if frame_size is not None:
        w, h = frame_size
        x, y = out[x_col], out[y_col]
        out["out_of_frame"] = ~out["missing"] & ((x < 0) | (x > w) | (y < 0) | (y > h))

    for idx in _groups(out, group_cols):
        part = out.loc[idx].sort_values(time_col)
        x = part[x_col].to_numpy(float)
        y = part[y_col].to_numpy(float)
        t = part[time_col].to_numpy(float)

        step = np.hypot(np.diff(x), np.diff(y))  # NaN if either frame missing
        dt = np.diff(t)

        if max_speed_px_s is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = np.where(dt > 0, step / dt, np.nan)
            jump = np.concatenate([[False], np.nan_to_num(speed, nan=0.0) > max_speed_px_s])
            out.loc[part.index, "jump"] = jump

        # frozen: a run of "no movement at all" steps long enough
        still = np.concatenate([[False], np.nan_to_num(step, nan=np.inf) <= frozen_tol_px])
        frozen = np.zeros(len(part), dtype=bool)
        for s, e in _runs(still):
            # run of still steps s..e-1 means frames s-1..e-1 have the same position
            if (e - s + 1) >= frozen_min_frames:
                frozen[max(s - 1, 0):e] = True
        out.loc[part.index, "frozen"] = frozen

    out["valid"] = ~(out["missing"] | out["out_of_frame"] | out["jump"] | out["frozen"]
                     | out["low_conf"])
    return out


def interpolate_short_gaps(df, cols=("x", "y"), time_col="time_s",
                           group_cols=("session", "mouse", "keypoint"),
                           max_gap_frames=3):
    """Linearly fill gaps of at most max_gap_frames missing frames.

    Only gaps with valid data on BOTH sides are filled (no extrapolation).
    Longer gaps stay NaN. Filled rows get imputed=True so later steps
    (and the UI) know these were not real measurements.
    """
    out = df.copy()
    out["imputed"] = False
    for idx in _groups(out, group_cols):
        part = out.loc[idx].sort_values(time_col)
        missing = part[list(cols)].isna().any(axis=1).to_numpy()
        n = len(part)
        fill = np.zeros(n, dtype=bool)
        for s, e in _runs(missing):
            if s > 0 and e < n and (e - s) <= max_gap_frames:
                fill[s:e] = True
        if not fill.any():
            continue
        t = part[time_col].to_numpy(float)
        for col in cols:
            v = part[col].to_numpy(float)
            ok = ~np.isnan(v)
            filled = v.copy()
            filled[fill] = np.interp(t[fill], t[ok], v[ok])
            out.loc[part.index, col] = filled
        out.loc[part.index[fill], "imputed"] = True
    return out


def window_quality(df, time_col="time_s", window_s=1.0,
                   group_cols=("session", "mouse"), min_quality=0.8):
    """Summarise flags in fixed time windows.

    quality = fraction of rows that are valid and not imputed.
    warning = quality < min_quality.
    Needs the columns added by flag_tracking (and optionally imputed).
    """
    needed = {"missing", "frozen", "jump", "out_of_frame", "valid"}
    if not needed <= set(df.columns):
        raise ValueError("run flag_tracking() first")
    d = df.copy()
    if "imputed" not in d.columns:
        d["imputed"] = False
    d["window"] = np.floor(d[time_col] / window_s).astype(int)
    d["clean"] = d["valid"] & ~d["imputed"]
    keys = [c for c in group_cols if c in d.columns] + ["window"]
    agg = d.groupby(keys, sort=True).agg(
        start_s=(time_col, "min"),
        frac_missing=("missing", "mean"),
        frac_frozen=("frozen", "mean"),
        frac_jump=("jump", "mean"),
        frac_out_of_frame=("out_of_frame", "mean"),
        frac_imputed=("imputed", "mean"),
        quality=("clean", "mean"),
    ).reset_index()
    agg["warning"] = agg["quality"] < min_quality
    return agg


def tracking_loss_events(df, time_col="time_s", group_cols=("session", "mouse"),
                         min_duration_s=0.5, flag="missing"):
    """List stretches where a flag (default: missing) lasts >= min_duration_s.

    Used for the "tracking lost" warning in the dashboard. Rows are checked
    per frame: if several keypoints share a frame, the frame counts as lost
    when ANY of them has the flag.
    """
    events = []
    gcols = [c for c in group_cols if c in df.columns]
    frame = df.groupby(gcols + [time_col], sort=True)[flag].any().reset_index()
    for key, part in (frame.groupby(gcols, sort=False) if gcols else [((), frame)]):
        t = part[time_col].to_numpy(float)
        for s, e in _runs(part[flag].to_numpy()):
            # duration = from first flagged frame to the next good frame
            end_t = t[e] if e < len(t) else t[-1]
            if end_t - t[s] >= min_duration_s:
                ev = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
                ev.update(type=f"tracking_{flag}", start_s=float(t[s]), end_s=float(end_t))
                events.append(ev)
    return events
