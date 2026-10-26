"""Steadier body points for the live camera path (new file). Off unless pose_model.json has a "stabilize" block.

The steps come from the ScopeScore tracker (Ava's surgical video paper), applied to the 14 mouse points of every picture:
  1. flow mix   the point of the model is mixed with the point followed from the last picture by optical flow
                (Lucas-Kanade): p = alpha * model + (1 - alpha) * flow. ScopeScore used alpha = 0.2.
  2. smoothing  a One-Euro filter on every coordinate of every point (past pictures only). ScopeScore: 0.6 Hz, beta 0.04.
  3. tolerance  a point that moves less than about sigma pixels stays still, a larger move is followed almost at once:
                s = s_prev + (1 - exp(-|d|^2 / (2 sigma^2))) * d. ScopeScore: sigma = 1% of the picture height.
Nothing is made up: when the model finds no mouse, the answer is None, and a point that is missing for longer than max_gap_s starts again.
Settings (all optional): {"stabilize": {"alpha": 0.2, "min_cutoff": 0.6, "beta": 0.04, "sigma_px": 5.7, "flow": true, "max_gap_s": 0.5}}.
Set "flow": false (or "alpha": 1) to switch the flow mix off, "sigma_px": 0 to switch the tolerance off, "min_cutoff": 0 to switch smoothing off.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from .calms21 import IMAGE_SIZE
from .filters import OneEuroFilter

DEFAULTS = {"alpha": 0.2, "min_cutoff": 0.6, "beta": 0.04, "d_cutoff": 1.0, "sigma_px": 0.01 * IMAGE_SIZE[1], "flow": True,
            "max_gap_s": 0.5, "max_flow_error": 12.0}
WIN = (21, 21)                                              # Lucas-Kanade window


def _gray(image):
    if image.ndim == 2:
        return image
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class PointStabilizer:
    """One per camera. stabilize(image, table) -> the table with steadier x, y (same columns, same rows)."""

    def __init__(self, settings=None, now_fn=time.monotonic):
        s = {**DEFAULTS, **(settings or {})}
        self.alpha, self.sigma, self.use_flow = float(s["alpha"]), float(s["sigma_px"]), bool(s["flow"]) and float(s["alpha"]) < 1.0
        self.min_cutoff, self.beta, self.d_cutoff = float(s["min_cutoff"]), float(s["beta"]), float(s["d_cutoff"])
        self.max_gap_s, self.max_flow_error, self.now_fn = float(s["max_gap_s"]), float(s["max_flow_error"]), now_fn
        self.reset()

    def reset(self):
        self._gray, self._fused, self._filters, self._still, self._last_t = None, {}, {}, {}, None

    def _flow(self, gray, keys):
        """Where the last fused points went, from the last picture to this one: {key: (x, y)} for the points that could be followed."""
        if not (self.use_flow and self._gray is not None and keys):
            return {}
        import cv2
        old = [k for k in keys if k in self._fused]
        if not old:
            return {}
        p0 = np.array([self._fused[k] for k in old], np.float32).reshape(-1, 1, 2)
        p1, ok, err = cv2.calcOpticalFlowPyrLK(self._gray, gray, p0, None, winSize=WIN, maxLevel=2)
        out = {}
        for k, q, good, e in zip(old, p1.reshape(-1, 2), ok.reshape(-1), err.reshape(-1)):
            if good and e <= self.max_flow_error and np.isfinite(q).all():
                out[k] = (float(q[0]), float(q[1]))
        return out

    def _smooth(self, key, axis, value, t):
        if self.min_cutoff <= 0:
            return value
        f = self._filters.get((key, axis))
        if f is None:
            f = self._filters[(key, axis)] = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff, self.max_gap_s)
        return float(f.update(value, t))

    def _tolerate(self, key, x, y):
        if self.sigma <= 0:
            return x, y
        s = self._still.get(key)
        if s is None:
            self._still[key] = (x, y)
            return x, y
        dx, dy = x - s[0], y - s[1]
        gain = 1.0 - math.exp(-(dx * dx + dy * dy) / (2.0 * self.sigma ** 2))
        self._still[key] = (s[0] + gain * dx, s[1] + gain * dy)
        return self._still[key]

    def stabilize(self, image, table):
        t = self.now_fn()
        if table is None or len(table) == 0:
            return table
        if self._last_t is not None and t - self._last_t > self.max_gap_s:
            self.reset()
        gray = _gray(image) if self.use_flow else None
        keys = list(zip(table["mouse"], table["keypoint"]))
        flow = self._flow(gray, keys) if self.use_flow else {}
        out = table.copy()
        new_fused = {}
        for i, key in enumerate(keys):
            x, y = float(table["x"].iloc[i]), float(table["y"].iloc[i])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if key in flow:
                x = self.alpha * x + (1.0 - self.alpha) * flow[key][0]
                y = self.alpha * y + (1.0 - self.alpha) * flow[key][1]
            new_fused[key] = (x, y)
            x, y = self._smooth(key, "x", x, t), self._smooth(key, "y", y, t)
            x, y = self._tolerate(key, x, y)
            out.iloc[i, out.columns.get_loc("x")], out.iloc[i, out.columns.get_loc("y")] = x, y
        # the next flow starts from what was fused (before smoothing and tolerance), so the filters do not add lag to the flow
        self._fused, self._gray, self._last_t = new_fused, gray, t
        return out


def wrap(pose_function, settings, now_fn=time.monotonic):
    """The pose function with steadier points. A picture in, a table (or None) out, like the function it wraps."""
    stabilizer = PointStabilizer(settings, now_fn)

    def pose(image):
        return stabilizer.stabilize(image, pose_function(image))
    pose.stabilizer = stabilizer
    return pose
