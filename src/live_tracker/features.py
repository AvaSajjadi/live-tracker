"""Causal motion features per mouse (ScopeScore-inspired, adapted to mice).

New file for Live Tracker (written by us, not part of the starter).

These are motion MEASUREMENTS (how fast, how straight, how still), not
behavior labels. Speed or turning alone cannot tell us a mouse is attacking;
the classifier decides that.

Everything here is causal: the value at time t only uses frames at or before t,
so the same code works for live replay. Trailing windows look back window_s
seconds; at the start of a session (or after a gap) they stay NaN until enough
past data exists.

Missing tracking is never treated as standing still: a missing or invalid
point gives NaN features, not 0.

Units: pixels, seconds, radians. All thresholds are placeholders and must be
tuned on train/val sessions (never test). Column meanings: INTERFACES.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .filters import smooth_dataframe

FEATURE_COLS = ("point_ok", "uses_imputed", "speed", "accel", "heading", "turn_rate",
                "path_len_w", "straightness_w", "idle_frac_w", "in_pause")

_TOL = 1e-9


@dataclass
class FeatureParams:
    """Feature settings. Every number is a placeholder, tune on train/val.

    smooth            "one_euro" (default), "ema", or None for raw positions
    smooth_kwargs     passed to filters.smooth_dataframe
    max_step_s        a step longer than this is not one frame step (dropped
                      frames / gap), so no speed is computed across it
    window_s          trailing window length for the *_w features
    min_coverage      fraction of the window that must have good steps
    heading_min_speed_px_s  below this speed the heading is noise -> NaN
    idle_speed_px_s   speed below this counts as idle / pausing
    pause_min_duration_s  how long the mouse must be idle to be in a pause
    min_path_px       straightness is NaN when the path is shorter than this
    """
    smooth: str | None = "one_euro"
    smooth_kwargs: dict = field(default_factory=lambda: {"min_cutoff": 1.0, "beta": 0.0})
    max_step_s: float = 0.1
    window_s: float = 1.0
    min_coverage: float = 0.9
    heading_min_speed_px_s: float = 5.0
    idle_speed_px_s: float = 10.0
    pause_min_duration_s: float = 0.5
    min_path_px: float = 1.0


def _usable(df, x_col="x", y_col="y"):
    """Points we trust: valid by quality flags, or filled by a short-gap interpolation.

    Imputed rows were 'missing' when flag_tracking ran, so they are not 'valid';
    we still use them, but mark the features with uses_imputed.
    """
    ok = df[x_col].notna() & df[y_col].notna()
    if "valid" in df.columns:
        good = df["valid"].astype(bool)
        if "imputed" in df.columns:
            good = good | df["imputed"].astype(bool)
        ok = ok & good
    return ok.to_numpy()


def _window_start(t, window_s):
    """Index of the first frame inside the trailing window (t_i - window_s, t_i]."""
    return np.searchsorted(t, t - window_s - _TOL, side="left")


def _group_features(t, x, y, ok, imputed, p):
    n = len(t)
    nan = np.full(n, np.nan)

    dt = nan.copy()
    dt[1:] = np.diff(t)
    step_ok = np.zeros(n, dtype=bool)
    step_ok[1:] = ok[1:] & ok[:-1] & (dt[1:] > 0) & (dt[1:] <= p.max_step_s)

    dx, dy = nan.copy(), nan.copy()
    dx[1:], dy[1:] = np.diff(x), np.diff(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = np.where(step_ok, dx / dt, np.nan)
        vy = np.where(step_ok, dy / dt, np.nan)
    speed = np.hypot(vx, vy)
    step_len = np.where(step_ok, np.hypot(dx, dy), 0.0)

    # velocities sit at step midpoints, so the time between two of them
    # is the mean of the two dts (matters for irregular timestamps)
    pair_ok = np.zeros(n, dtype=bool)
    pair_ok[1:] = step_ok[1:] & step_ok[:-1]
    dt_mid = nan.copy()
    dt_mid[1:] = (dt[1:] + dt[:-1]) / 2
    accel = nan.copy()
    accel[1:] = np.hypot(np.diff(vx), np.diff(vy)) / dt_mid[1:]
    accel[~pair_ok] = np.nan

    heading = np.where(speed >= p.heading_min_speed_px_s, np.arctan2(vy, vx), np.nan)
    turn_rate = nan.copy()
    # wrap the difference to (-pi, pi], so crossing +-pi is a small turn
    dh = (np.diff(heading) + np.pi) % (2 * np.pi) - np.pi
    turn_rate[1:] = dh / dt_mid[1:]
    turn_rate[~pair_ok] = np.nan

    # trailing windows via cumulative sums: steps j+1..i are inside the window
    j = _window_start(t, p.window_s)
    idx = np.arange(n)

    def win_sum(v):
        c = np.cumsum(v)
        return c - c[j]

    covered = win_sum(np.where(step_ok, dt, 0.0))
    full = (covered >= p.min_coverage * p.window_s - _TOL) & ok
    path = win_sum(step_len)
    path_len_w = np.where(full, path, np.nan)
    net = np.hypot(x - x[j], y - y[j])
    with np.errstate(divide="ignore", invalid="ignore"):
        straight = np.minimum(net / path, 1.0)  # <1 by geometry; clip float noise
    straightness_w = np.where(full & ok[j] & (path >= p.min_path_px), straight, np.nan)
    n_steps = win_sum(step_ok.astype(float))
    low = step_ok & (speed < p.idle_speed_px_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        idle_frac_w = np.where(full, win_sum(low.astype(float)) / n_steps, np.nan)

    # causal pause: idle since run start s, and at least pause_min_duration_s
    # has passed since the last moving frame (t[s-1]); missing breaks the run
    run_start = np.maximum.accumulate(np.where(low & ~np.r_[False, low[:-1]], idx, 0))
    elapsed = t - t[np.maximum(run_start - 1, 0)]
    paused = low & (elapsed >= p.pause_min_duration_s - _TOL)
    in_pause = pd.array(np.where(step_ok, paused, False), dtype="boolean")
    in_pause[~step_ok] = pd.NA  # unknown, not "moving"

    # any imputed point among the inputs: window frames j..i, or i-2..i for accel
    c_imp = np.r_[0, np.cumsum(imputed.astype(int))]
    first = np.maximum(np.minimum(j, idx - 2), 0)
    uses_imputed = (c_imp[idx + 1] - c_imp[first]) > 0

    return {
        "point_ok": ok, "uses_imputed": uses_imputed, "speed": speed, "accel": accel,
        "heading": heading, "turn_rate": turn_rate, "path_len_w": path_len_w,
        "straightness_w": straightness_w, "idle_frac_w": idle_frac_w, "in_pause": in_pause,
    }


def individual_features(df, params=None, x_col="x", y_col="y", time_col="time_s"):
    """Causal per-mouse motion features from the long pose table.

    Input: long table (session, mouse, frame, time_s, x, y), optionally with the
    flag columns from quality.flag_tracking / interpolate_short_gaps. Invalid
    points are dropped (set to NaN) before smoothing.

    For now each mouse must have one point per frame. With several keypoints we
    need a body center first, which waits for the real CalMS21 keypoint list.

    Returns one row per session x mouse x frame with the columns in FEATURE_COLS.
    """
    p = params or FeatureParams()
    if "keypoint" in df.columns:
        per_mouse = df.groupby([c for c in ("session", "mouse") if c in df.columns])["keypoint"]
        if (per_mouse.nunique() > 1).any():
            raise ValueError("several keypoints per mouse: body center not defined yet "
                             "(waits for the CalMS21 keypoint schema)")

    d = df.copy()
    ok = _usable(d, x_col, y_col)
    d.loc[~ok, [x_col, y_col]] = np.nan
    d["_ok"] = ok
    if p.smooth is not None:
        d = smooth_dataframe(d, value_cols=(x_col, y_col), time_col=time_col,
                             method=p.smooth, **p.smooth_kwargs)
        xs, ys = f"{x_col}_smooth", f"{y_col}_smooth"
    else:
        xs, ys = x_col, y_col
    if "imputed" not in d.columns:
        d["imputed"] = False

    group_cols = [c for c in ("session", "mouse") if c in d.columns]
    parts = []
    groups = d.groupby(group_cols, sort=False) if group_cols else [((), d)]
    for _, part in groups:
        part = part.sort_values(time_col)
        feats = _group_features(part[time_col].to_numpy(float), part[xs].to_numpy(float),
                                part[ys].to_numpy(float), part["_ok"].to_numpy(bool),
                                part["imputed"].to_numpy(bool), p)
        keep = [c for c in ("session", "mouse", "frame") if c in part.columns] + [time_col]
        out = part[keep].reset_index(drop=True)
        for name, values in feats.items():
            out[name] = values
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def to_wide(feat, time_col="time_s"):
    """One row per session x frame, per-mouse columns as <feature>_<mouse>."""
    keys = [c for c in ("session", "frame") if c in feat.columns]
    wide = None
    for mouse, part in feat.groupby("mouse", sort=True):
        part = part[keys + [time_col] + list(FEATURE_COLS)].rename(
            columns={c: f"{c}_{mouse}" for c in FEATURE_COLS})
        wide = part if wide is None else wide.merge(part, on=keys + [time_col], how="outer")
    return wide.sort_values(keys).reset_index(drop=True)
