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
from .quality import flag_tracking, interpolate_short_gaps, tracking_loss_events

DYADIC_COLS = ("dist_center", "dist_rate", "rel_speed", "close_time_s", "speed_corr_w")

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
    close_dist_px     center distance below this counts as "close"
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
    close_dist_px: float = 50.0


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


def _win_sum(v, j):
    """Sum of v[j+1..i] for every i (the steps inside each trailing window)."""
    c = np.cumsum(v)
    return c - c[j]


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
        return _win_sum(v, j)

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


def body_center(df, torso_keypoints=None, x_col="x", y_col="y"):
    """One point per mouse per frame: the mean of the usable torso keypoints.

    Without a keypoint column (or with one keypoint per mouse) that point is the
    center. With several keypoints you must name the torso ones; we do not guess
    them, they come from the CalMS21 readme (H1a). Never use the nose alone: it
    moves with head turns, not with the body.

    If some torso keypoints are unusable in a frame we average the rest
    (n_keypoints says how many); if none are usable the center is NaN. The
    center can shift a little when the set of keypoints changes.
    Returns session, mouse, frame, time_s, x, y, valid, imputed, n_keypoints.
    """
    if "keypoint" not in df.columns:
        return df.copy()
    keys = [c for c in ("session", "mouse") if c in df.columns]
    if torso_keypoints is None:
        if (df.groupby(keys)["keypoint"].nunique() > 1).any():
            raise ValueError("several keypoints per mouse: pass torso_keypoints to define "
                             "the body center (names from the CalMS21 keypoint schema)")
        return df.drop(columns="keypoint")
    unknown = set(torso_keypoints) - set(df["keypoint"].unique())
    if unknown:
        raise ValueError(f"torso keypoints not in data: {sorted(unknown)}")

    d = df[df["keypoint"].isin(torso_keypoints)].copy()
    ok = _usable(d, x_col, y_col)
    d.loc[~ok, [x_col, y_col]] = np.nan
    d["_ok"] = ok
    d["_imp"] = ok & (d["imputed"].astype(bool) if "imputed" in d.columns else False)
    out = d.groupby(keys + ["frame"], sort=True).agg(
        time_s=("time_s", "first"), _cx=(x_col, "mean"), _cy=(y_col, "mean"),
        n_keypoints=("_ok", "sum"), imputed=("_imp", "any")).reset_index()
    out = out.rename(columns={"_cx": x_col, "_cy": y_col})
    out["valid"] = out["n_keypoints"] > 0
    return out


def _prepare(df, p, x_col="x", y_col="y", time_col="time_s"):
    """Drop unusable points, then smooth causally. Adds _x, _y, _ok, imputed."""
    d = df.copy()
    ok = _usable(d, x_col, y_col)
    d.loc[~ok, [x_col, y_col]] = np.nan
    d["_ok"] = ok
    if p.smooth is not None:
        d = smooth_dataframe(d, value_cols=(x_col, y_col), time_col=time_col,
                             method=p.smooth, **p.smooth_kwargs)
        d["_x"], d["_y"] = d[f"{x_col}_smooth"], d[f"{y_col}_smooth"]
    else:
        d["_x"], d["_y"] = d[x_col], d[y_col]
    if "imputed" not in d.columns:
        d["imputed"] = False
    return d


def _individual(d, p, time_col):
    group_cols = [c for c in ("session", "mouse") if c in d.columns]
    parts = []
    groups = d.groupby(group_cols, sort=False) if group_cols else [((), d)]
    for _, part in groups:
        part = part.sort_values(time_col)
        feats = _group_features(part[time_col].to_numpy(float), part["_x"].to_numpy(float),
                                part["_y"].to_numpy(float), part["_ok"].to_numpy(bool),
                                part["imputed"].to_numpy(bool), p)
        keep = [c for c in ("session", "mouse", "frame") if c in part.columns] + [time_col]
        out = part[keep].reset_index(drop=True)
        for name, values in feats.items():
            out[name] = values
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def individual_features(df, params=None, torso_keypoints=None,
                        x_col="x", y_col="y", time_col="time_s"):
    """Causal per-mouse motion features from the long pose table.

    Input: long table (session, mouse, frame, time_s, x, y), optionally with
    keypoint and the flag columns from quality.flag_tracking /
    interpolate_short_gaps. Invalid points are dropped (set to NaN) before
    smoothing. Several keypoints per mouse are reduced with body_center().

    Returns one row per session x mouse x frame with the columns in FEATURE_COLS.
    """
    p = params or FeatureParams()
    center = body_center(df, torso_keypoints, x_col, y_col)
    return _individual(_prepare(center, p, x_col, y_col, time_col), p, time_col)


def _pair_arrays(t, xa, ya, oka, xb, yb, okb, p):
    n = len(t)
    nan = np.full(n, np.nan)
    ok = oka & okb
    dt = nan.copy()
    dt[1:] = np.diff(t)
    step_ok = np.zeros(n, dtype=bool)
    step_ok[1:] = ok[1:] & ok[:-1] & (dt[1:] > 0) & (dt[1:] <= p.max_step_s)

    def rate(v):
        out = nan.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.diff(v) / dt[1:]
        return np.where(step_ok, out, np.nan)

    dist = np.where(ok, np.hypot(xa - xb, ya - yb), np.nan)
    dist_rate = rate(dist)
    vax, vay, vbx, vby = rate(xa), rate(ya), rate(xb), rate(yb)
    rel_speed = np.hypot(vax - vbx, vay - vby)

    j = _window_start(t, p.window_s)
    covered = _win_sum(np.where(step_ok, dt, 0.0), j)
    full = (covered >= p.min_coverage * p.window_s - _TOL) & ok
    close = step_ok & (dist < p.close_dist_px)
    close_time_s = np.where(full, _win_sum(np.where(close, dt, 0.0), j), np.nan)

    # Pearson correlation of the two speeds over the window, from running sums
    sa = np.where(step_ok, np.hypot(vax, vay), 0.0)
    sb = np.where(step_ok, np.hypot(vbx, vby), 0.0)
    k = _win_sum(step_ok.astype(float), j)
    sum_a, sum_b = _win_sum(sa, j), _win_sum(sb, j)
    sq_a, sq_b = k * _win_sum(sa * sa, j), k * _win_sum(sb * sb, j)
    var_a, var_b = sq_a - sum_a ** 2, sq_b - sum_b ** 2
    cov = k * _win_sum(sa * sb, j) - sum_a * sum_b
    # a constant speed has no correlation; the relative floor absorbs float noise
    varies = (var_a > 1e-9 * (sq_a + 1)) & (var_b > 1e-9 * (sq_b + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.clip(cov / np.sqrt(var_a * var_b), -1.0, 1.0)
    speed_corr_w = np.where(full & varies, corr, np.nan)

    return {"dist_center": dist, "dist_rate": dist_rate, "rel_speed": rel_speed,
            "close_time_s": close_time_s, "speed_corr_w": speed_corr_w}


def _dyadic(d, p, time_col):
    parts = []
    sessions = d.groupby("session", sort=False) if "session" in d.columns else [(None, d)]
    for session, part in sessions:
        mice = sorted(part["mouse"].unique(), key=str)
        if len(mice) != 2:
            raise ValueError(f"session {session!r}: dyadic features need exactly 2 mice, got {mice}")
        a, b = (part[part["mouse"] == m].set_index("frame") for m in mice)
        frames = a.index.union(b.index).sort_values()
        a, b = a.reindex(frames), b.reindex(frames)
        t = a[time_col].fillna(b[time_col]).to_numpy(float)
        oka = a["_ok"].astype("boolean").fillna(False).to_numpy(bool)
        okb = b["_ok"].astype("boolean").fillna(False).to_numpy(bool)
        feats = _pair_arrays(t, a["_x"].to_numpy(float), a["_y"].to_numpy(float), oka,
                             b["_x"].to_numpy(float), b["_y"].to_numpy(float), okb, p)
        out = pd.DataFrame({"frame": frames.to_numpy(), time_col: t, **feats})
        if session is not None:
            out.insert(0, "session", session)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def dyadic_features(df, params=None, torso_keypoints=None,
                    x_col="x", y_col="y", time_col="time_s"):
    """Causal two-mouse features from body centers (columns in DYADIC_COLS).

    dist_center   center distance, px
    dist_rate     its rate, px/s (negative = approaching)
    rel_speed     |velocity A - velocity B|, px/s
    close_time_s  seconds of the trailing window with dist_center < close_dist_px
    speed_corr_w  trailing correlation of the two speeds (NaN if one is constant)

    Measurements only: being close and fast is not the same as fighting.
    Keypoint features (nose-to-nose, facing angle) wait for the CalMS21 schema.
    """
    p = params or FeatureParams()
    center = body_center(df, torso_keypoints, x_col, y_col)
    return _dyadic(_prepare(center, p, x_col, y_col, time_col), p, time_col)


def to_wide(feat, time_col="time_s"):
    """One row per session x frame, per-mouse columns as <feature>_<mouse>."""
    keys = [c for c in ("session", "frame") if c in feat.columns]
    wide = None
    for mouse, part in feat.groupby("mouse", sort=True):
        part = part[keys + [time_col] + list(FEATURE_COLS)].rename(
            columns={c: f"{c}_{mouse}" for c in FEATURE_COLS})
        wide = part if wide is None else wide.merge(part, on=keys + [time_col], how="outer")
    return wide.sort_values(keys).reset_index(drop=True)


def frame_tracking_ok(flags_df, keypoints=None, allow_imputed=True):
    """One bool per session x frame: is every needed point of every mouse usable?

    keypoints: which keypoints are needed (default: all in the table). A frame
    where any mouse has a needed point missing / frozen / jumping / out of frame,
    or no row at all, is not ok. Short-gap imputed points count as ok unless
    allow_imputed=False. Returns session, frame, time_s, tracking_ok.
    """
    d = flags_df
    if keypoints is not None and "keypoint" in d.columns:
        d = d[d["keypoint"].isin(keypoints)]
    ok = d["x"].notna() & d["y"].notna()
    if "valid" in d.columns:
        good = d["valid"].astype(bool)
        if allow_imputed and "imputed" in d.columns:
            good = good | d["imputed"].astype(bool)
        ok = ok & good
    point_cols = [c for c in ("mouse", "keypoint") if c in d.columns]
    d = d.assign(_ok=ok.to_numpy(), _pid=d[point_cols].astype(str).agg("|".join, axis=1))
    keys = [c for c in ("session",) if c in d.columns]
    per_frame = d.groupby(keys + ["frame"], sort=True).agg(
        time_s=("time_s", "first"), n_ok=("_ok", "sum"), n_points=("_pid", "nunique")
    ).reset_index()
    # every point seen anywhere in the session must be present and usable
    if keys:
        need = per_frame["session"].map(d.groupby("session")["_pid"].nunique())
    else:
        need = d["_pid"].nunique()
    per_frame["tracking_ok"] = (per_frame["n_ok"] == need) & (per_frame["n_points"] == need)
    return per_frame[keys + ["frame", "time_s", "tracking_ok"]]


WARNING_TEXT = {
    "missing": "Tracking lost for mouse {mouse}",
    "frozen": "Tracker may be stuck for mouse {mouse} (identical coordinates; "
              "not the same as the mouse being still)",
    "out_of_frame": "Mouse {mouse} tracked outside the image",
}


def tracking_warning_events(flags_df, flags=("missing", "frozen", "out_of_frame"),
                            keypoints=None, min_duration_s=0.5):
    """Tracking problems as event-log entries (INTERFACES.md §4, action "tracking_warning").

    Wraps quality.tracking_loss_events for each flag. keypoints limits which
    keypoints count (e.g. the torso ones), so a briefly hidden nose does not
    raise a warning. min_duration_s is a placeholder. Event IDs are "T1", "T2",
    ... so they never clash with the monitor's "E" IDs. Sorted by start time.
    """
    d = flags_df
    if keypoints is not None and "keypoint" in d.columns:
        d = d[d["keypoint"].isin(keypoints)]
    found = []
    for flag in flags:
        if flag in d.columns and d[flag].any():
            found += [(flag, ev) for ev in
                      tracking_loss_events(d, min_duration_s=min_duration_s, flag=flag)]
    found.sort(key=lambda fe: (str(fe[1].get("session")), fe[1]["start_s"], str(fe[1].get("mouse"))))
    events = []
    for i, (flag, ev) in enumerate(found, start=1):
        mouse = ev.get("mouse")
        text = WARNING_TEXT.get(flag, "Tracking problem ({flag}) for mouse {mouse}")
        events.append({
            "event_id": f"T{i}", "session": ev.get("session"), "rule_id": None,
            "behavior": None, "start_s": ev["start_s"], "end_s": ev["end_s"],
            "confidence": None, "action": "tracking_warning", "researcher_answer": None,
            "evidence": {"mouse": None if mouse is None else str(mouse), "flag": flag,
                         "duration_s": ev["end_s"] - ev["start_s"],
                         "message": text.format(mouse=mouse, flag=flag)},
        })
    return events


def compute_features(long_df, params=None, torso_keypoints=None, frame_size=None):
    """The whole feature table (INTERFACES.md §5) in one call.

    If long_df has no quality flags yet, runs quality.flag_tracking and
    interpolate_short_gaps with their (placeholder) defaults first.
    Returns one row per session x frame: session, frame, time_s, tracking_ok,
    <feature>_<mouse> for both mice, then the dyadic columns.
    """
    p = params or FeatureParams()
    flags = long_df
    if "valid" not in flags.columns:
        flags = interpolate_short_gaps(flag_tracking(flags, frame_size=frame_size))
    prep = _prepare(body_center(flags, torso_keypoints), p)
    keys = [c for c in ("session", "frame", "time_s") if c in prep.columns]
    ok = frame_tracking_ok(flags, keypoints=torso_keypoints)
    out = (ok.merge(to_wide(_individual(prep, p, "time_s")), on=keys, how="outer")
             .merge(_dyadic(prep, p, "time_s"), on=keys, how="left"))
    out["tracking_ok"] = out["tracking_ok"].astype("boolean").fillna(False).astype(bool)
    return out.sort_values(keys).reset_index(drop=True)
