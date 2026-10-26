"""Motion features from already-tracked 2D landmarks.

Coordinates are in pixels; features are in pixels or pixels/second.
Missing coordinates must be NaN. Do not interpret tracking failure as inactivity.
"""
import numpy as np
from scipy.signal import savgol_filter


def clean_trajectory(xy, fps=30.0, max_gap_s=0.10, smooth_s=0.17):
    """Interpolate only short internal gaps; return cleaned positions and validity mask.

    No extrapolation at edges. Gaps longer than max_gap_s remain NaN.
    Smoothing is applied only to contiguous valid segments.
    """
    a = np.asarray(xy, dtype=float).copy()
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('Expected array of shape (frames, 2)')
    if fps <= 0:
        raise ValueError('fps must be positive')
    a[~np.isfinite(a).all(axis=1)] = np.nan
    max_gap = int(round(max_gap_s * fps))
    for dim in range(2):
        v = a[:, dim]
        valid = np.isfinite(v)
        ix = np.flatnonzero(valid)
        for left, right in zip(ix[:-1], ix[1:]):
            missing = right - left - 1
            if 0 < missing <= max_gap:
                v[left + 1:right] = np.interp(
                    np.arange(left + 1, right), [left, right], [v[left], v[right]]
                )
    valid = np.isfinite(a).all(axis=1)
    width = max(3, int(round(smooth_s * fps)) | 1)
    starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
    ends = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
    for start, end in zip(starts, ends):
        length = end - start
        w = min(width, length if length % 2 else length - 1)
        if w >= 3:
            a[start:end] = savgol_filter(a[start:end], window_length=w,
                                           polyorder=min(2, w - 1), axis=0)
    return a, valid


def kinematics(xy, fps=30.0):
    """Frame-aligned velocity/speed/acceleration, never crossing missing gaps."""
    p = np.asarray(xy, dtype=float)
    v = np.full_like(p, np.nan)
    a = np.full_like(p, np.nan)
    good = np.isfinite(p).all(axis=1)
    adjacent = good[1:] & good[:-1]
    v[1:][adjacent] = (p[1:][adjacent] - p[:-1][adjacent]) * fps
    vg = np.isfinite(v).all(axis=1)
    adjacent_v = vg[1:] & vg[:-1]
    a[1:][adjacent_v] = (v[1:][adjacent_v] - v[:-1][adjacent_v]) * fps
    return {'velocity': v, 'speed': np.linalg.norm(v, axis=1),
            'acceleration': np.linalg.norm(a, axis=1), 'valid': good}


def pair_features(mouse_a, mouse_b, fps=30.0):
    """Relative position and closing speed; negative closing speed means approaching."""
    a, b = np.asarray(mouse_a, float), np.asarray(mouse_b, float)
    if a.shape != b.shape:
        raise ValueError('Mouse trajectories must have the same shape')
    dist = np.linalg.norm(a - b, axis=1)
    dd = np.full(len(dist), np.nan)
    valid = np.isfinite(dist)
    both = valid[1:] & valid[:-1]
    dd[1:][both] = (dist[1:][both] - dist[:-1][both]) * fps
    return {'distance_px': dist, 'distance_rate_px_s': dd}


def movement_bouts(speed, fps=30.0, threshold_px_s=10.0, min_duration_s=0.25):
    """Return start/end indices (end exclusive) for runs above threshold.

    Threshold is a placeholder for software testing, not a biological cutoff.
    """
    s = np.asarray(speed, float)
    active = np.isfinite(s) & (s >= threshold_px_s)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    minimum = int(np.ceil(min_duration_s * fps))
    return [(int(x), int(y)) for x, y in zip(starts, ends) if y - x >= minimum]


def frame_features(mouse_a, mouse_b, fps=30.0):
    """Return features and quality indicators; each mouse represented by one center point."""
    a, qa = clean_trajectory(mouse_a, fps)
    b, qb = clean_trajectory(mouse_b, fps)
    ka, kb = kinematics(a, fps), kinematics(b, fps)
    pair = pair_features(a, b, fps)
    import pandas as pd
    return pd.DataFrame({
        'speed_a_px_s': ka['speed'], 'speed_b_px_s': kb['speed'],
        'accel_a_px_s2': ka['acceleration'], 'accel_b_px_s2': kb['acceleration'],
        'distance_px': pair['distance_px'],
        'distance_rate_px_s': pair['distance_rate_px_s'],
        'tracking_ok': qa & qb,
    })
