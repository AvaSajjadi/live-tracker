"""Causal smoothing filters for pose trajectories.

New file for Live Tracker (written by us, not part of the starter).

Why this exists: the starter uses Savitzky-Golay smoothing, which looks at
future frames. That is fine for offline analysis but not for "live" replay,
because a live system cannot see the future. The filters here only use the
current frame and frames before it (causal).

Both filters:
  - reset when the time gap between samples is too long (tracking was lost),
  - reset at session/animal boundaries (we filter each group separately),
  - leave missing values (NaN) as NaN instead of making up a position.

Units: x/y in pixels, time in seconds. Default parameters are placeholders;
they still need to be picked on training/validation data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _check_inputs(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.shape != times.shape:
        raise ValueError("values and times must have the same length")
    if len(times) > 1 and np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted (non-decreasing)")
    return values, times


def ema_filter(values, times, tau_s=0.1, max_gap_s=0.5):
    """Exponential moving average with a time constant tau_s (seconds).

    We use alpha = 1 - exp(-dt / tau) so irregular frame spacing is handled:
    a longer dt means we trust the new sample more.
    If dt > max_gap_s, or the previous sample was missing for too long,
    the filter restarts from the new sample.
    """
    values, times = _check_inputs(values, times)
    if tau_s <= 0:
        raise ValueError("tau_s must be positive")

    out = np.full_like(values, np.nan)
    prev_val = None
    prev_t = None
    for i, (v, t) in enumerate(zip(values, times)):
        if np.isnan(v):
            # missing sample: output stays NaN, keep old state for now
            continue
        if prev_val is None or (t - prev_t) > max_gap_s:
            prev_val = v  # (re)start the filter
        else:
            dt = t - prev_t
            alpha = 1.0 - math.exp(-dt / tau_s) if dt > 0 else 0.0
            prev_val = prev_val + alpha * (v - prev_val)
        prev_t = t
        out[i] = prev_val
    return out


class OneEuroFilter:
    """Streaming One-Euro filter (Casiez et al., 2012).

    Idea: smooth a lot when the animal moves slowly (removes jitter),
    smooth less when it moves fast (less lag). The cutoff frequency grows
    with the estimated speed:  cutoff = min_cutoff + beta * |speed|.
    Call update(value, t) once per frame.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, max_gap_s=0.5):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_gap_s = max_gap_s
        self.reset()

    def reset(self):
        self._x = None
        self._dx = 0.0
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        # standard low-pass smoothing factor for a given cutoff (Hz)
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, value, t):
        if value is None or np.isnan(value):
            return np.nan
        if self._x is None or (t - self._t) > self.max_gap_s:
            self._x, self._dx, self._t = float(value), 0.0, t
            return self._x
        dt = t - self._t
        if dt <= 0:
            # duplicate timestamp: nothing new to learn, return current state
            return self._x
        raw_dx = (value - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = self._dx + a_d * (raw_dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = self._x + a * (value - self._x)
        self._t = t
        return self._x


def one_euro_filter(values, times, **kwargs):
    """Run OneEuroFilter over a whole array (still causal, frame by frame)."""
    values, times = _check_inputs(values, times)
    f = OneEuroFilter(**kwargs)
    return np.array([f.update(v, t) for v, t in zip(values, times)])


def smooth_dataframe(df, value_cols=("x", "y"), time_col="time_s",
                     group_cols=("session", "mouse"), method="one_euro", **kwargs):
    """Smooth each animal in each session separately.

    Grouping is what makes the filter reset at session boundaries: a new
    group always starts a fresh filter. Adds <col>_smooth columns and
    returns a copy (the raw columns are kept so we can compare).
    """
    func = {"one_euro": one_euro_filter, "ema": ema_filter}.get(method)
    if func is None:
        raise ValueError(f"unknown method {method!r}")
    group_cols = [c for c in group_cols if c in df.columns]
    out = df.copy()
    groups = out.groupby(group_cols, sort=False).groups if group_cols else {None: out.index}
    for idx in groups.values():
        part = out.loc[idx].sort_values(time_col)
        for col in value_cols:
            out.loc[part.index, f"{col}_smooth"] = func(part[col].to_numpy(),
                                                       part[time_col].to_numpy(), **kwargs)
    return out
