"""
Causal per-track tip+endpoint smoother for the YOLO11m-seg + RobustTipDetector
output. Single class, identical math whether driven by live YOLO (per-frame as
inference happens) or by replaying an existing CSV one row at a time.

Pipeline per detection (always causal — only past frames):
  1. Tip/start swap correction       — uses prev frame's tip/start
  2. Speed-cap rejection             — bounded by MAX_JUMP_PX vs last accepted
  3. Causal median filter (window N) — last N accepted tips, no future peek
  4. 1-euro filter                   — adaptive low-pass

State is keyed by (tool_class, side); per-track gap > gap_threshold resets
the per-track filters so a re-entering tool doesn't smear across the gap.

Strict scope: writes only inside cholec20/post_processing/.
Imports nothing from outside cholec20/.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================================
# 1-euro filter — vector flavor, causal by construction.
# =============================================================================

class OneEuroVec2:
    """Casiez et al. (2012) — adaptive low-pass; uses only past samples."""
    def __init__(self, freq: float = 25.0,
                 mincutoff: float = 1.0, beta: float = 0.007,
                 dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        if self._x_prev is None:
            self._x_prev = xy.copy()
            self._dx_prev = np.zeros_like(xy)
            return xy
        dx = (xy - self._x_prev) * self.freq
        a_d = self._alpha(self.dcutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, self.freq)
        x_hat = a * xy + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


# =============================================================================
# Per-track state.
# =============================================================================

@dataclass
class TrackState:
    last_frame: int = -1
    # last *accepted* (post speed-cap) raw positions, used as continuity anchor
    last_tip: Optional[np.ndarray] = None
    last_start: Optional[np.ndarray] = None
    # ringbuffer of last N accepted tips for causal median
    tip_hist: deque = field(default_factory=lambda: deque(maxlen=5))
    start_hist: deque = field(default_factory=lambda: deque(maxlen=5))
    consecutive_rejects: int = 0
    tip_filter: Optional[OneEuroVec2] = None
    start_filter: Optional[OneEuroVec2] = None
    # last *output* smoothed positions (what the renderer uses)
    out_tip: Optional[np.ndarray] = None
    out_start: Optional[np.ndarray] = None


# =============================================================================
# Smoother — single class, drive from live or CSV replay.
# =============================================================================

class CausalTipSmoother:
    """
    Stateful, causal smoothing for (tip, start) trajectories.

    Usage (live or replay, same call):
        s = CausalTipSmoother(fps=25.0)
        smoothed = s.step(frame_idx=N, key=(tool_class, side),
                          raw_tip=(x,y), raw_start=(x,y))

    Returns dict:
        {"tip": (x,y) | None, "start": (x,y) | None,
         "rejected": bool, "swapped": bool}
    """

    def __init__(self,
                 fps: float = 25.0,
                 max_jump_px: float = 80.0,
                 max_consecutive_rejects: int = 8,
                 median_window: int = 5,
                 gap_threshold_frames: int = 12,
                 oneeuro_mincutoff: float = 1.0,
                 oneeuro_beta: float = 0.007):
        self.fps = float(fps)
        self.max_jump_px = float(max_jump_px)
        self.max_consecutive_rejects = int(max_consecutive_rejects)
        self.median_window = int(median_window)
        self.gap_threshold_frames = int(gap_threshold_frames)
        self.mincutoff = float(oneeuro_mincutoff)
        self.beta = float(oneeuro_beta)
        self._tracks: dict = {}

    # ----- track plumbing -----
    def _new_filters(self) -> tuple[OneEuroVec2, OneEuroVec2]:
        return (
            OneEuroVec2(freq=self.fps, mincutoff=self.mincutoff, beta=self.beta),
            OneEuroVec2(freq=self.fps, mincutoff=self.mincutoff, beta=self.beta),
        )

    def _get_track(self, key, frame_idx: int) -> TrackState:
        tr = self._tracks.get(key)
        if tr is None:
            tr = TrackState(
                tip_hist=deque(maxlen=self.median_window),
                start_hist=deque(maxlen=self.median_window),
            )
            tr.tip_filter, tr.start_filter = self._new_filters()
            self._tracks[key] = tr
            return tr
        # Long gap -> reset per-track filters (re-entry treated as fresh trajectory).
        if frame_idx - tr.last_frame > self.gap_threshold_frames:
            tr.tip_hist.clear(); tr.start_hist.clear()
            tr.last_tip = None; tr.last_start = None
            tr.consecutive_rejects = 0
            tr.tip_filter, tr.start_filter = self._new_filters()
            tr.out_tip = None; tr.out_start = None
        return tr

    # ----- the four steps, all causal -----
    @staticmethod
    def _maybe_swap(raw_tip: np.ndarray, raw_start: np.ndarray,
                    prev_tip: Optional[np.ndarray],
                    prev_start: Optional[np.ndarray]) -> tuple:
        """If new tip is closer to prev_start AND new start is closer to prev_tip,
        the labels flipped — swap them.  (No-op when no previous frame yet.)"""
        if prev_tip is None or prev_start is None:
            return raw_tip, raw_start, False
        d_tip_to_prevtip   = np.linalg.norm(raw_tip   - prev_tip)
        d_tip_to_prevstart = np.linalg.norm(raw_tip   - prev_start)
        d_start_to_prevtip   = np.linalg.norm(raw_start - prev_tip)
        d_start_to_prevstart = np.linalg.norm(raw_start - prev_start)
        if (d_tip_to_prevstart < d_tip_to_prevtip
                and d_start_to_prevtip < d_start_to_prevstart):
            return raw_start.copy(), raw_tip.copy(), True
        return raw_tip, raw_start, False

    def _speed_cap(self, raw_tip: np.ndarray, raw_start: np.ndarray,
                   tr: TrackState) -> tuple:
        """If new tip jumps > max_jump_px from last accepted tip, reject and
        keep last accepted (for both tip and start). After max_consecutive_rejects
        in a row, accept the new position (genuine re-entry / fast move)."""
        if tr.last_tip is None:
            return raw_tip, raw_start, False  # first frame, accept

        jump = float(np.linalg.norm(raw_tip - tr.last_tip))
        if jump <= self.max_jump_px:
            tr.consecutive_rejects = 0
            return raw_tip, raw_start, False
        # over the cap
        tr.consecutive_rejects += 1
        if tr.consecutive_rejects >= self.max_consecutive_rejects:
            # forced acceptance (tool re-entered or moved fast)
            tr.consecutive_rejects = 0
            return raw_tip, raw_start, False
        # rejected — return previous accepted positions
        return tr.last_tip.copy(), tr.last_start.copy(), True

    @staticmethod
    def _causal_median(buf: deque) -> np.ndarray:
        """Median of (last N accepted) values per coord, causal (no future peek)."""
        arr = np.asarray(buf, dtype=np.float64)  # shape (k,2)
        return np.array([float(np.median(arr[:, 0])),
                         float(np.median(arr[:, 1]))], dtype=np.float64)

    # ----- main entry -----
    def step(self, frame_idx: int, key,
             raw_tip: tuple[float, float] | np.ndarray,
             raw_start: tuple[float, float] | np.ndarray) -> dict:
        tr = self._get_track(key, frame_idx)
        rt = np.asarray(raw_tip, dtype=np.float64)
        rs = np.asarray(raw_start, dtype=np.float64)

        # 1. tip/start swap correction
        rt, rs, swapped = self._maybe_swap(rt, rs, tr.last_tip, tr.last_start)

        # 2. speed-cap rejection
        accepted_tip, accepted_start, rejected = self._speed_cap(rt, rs, tr)

        # 3. update history with accepted values, take causal median
        tr.tip_hist.append(accepted_tip.copy())
        tr.start_hist.append(accepted_start.copy())
        med_tip = self._causal_median(tr.tip_hist)
        med_start = self._causal_median(tr.start_hist)

        # 4. 1-euro filter on the median output
        smooth_tip = tr.tip_filter(med_tip)
        smooth_start = tr.start_filter(med_start)

        # update state
        tr.last_frame = frame_idx
        tr.last_tip = accepted_tip.copy()
        tr.last_start = accepted_start.copy()
        tr.out_tip = smooth_tip.copy()
        tr.out_start = smooth_start.copy()

        return {
            "tip": (float(smooth_tip[0]), float(smooth_tip[1])),
            "start": (float(smooth_start[0]), float(smooth_start[1])),
            "rejected": bool(rejected),
            "swapped": bool(swapped),
        }

    def last_output(self, key) -> Optional[dict]:
        tr = self._tracks.get(key)
        if tr is None or tr.out_tip is None:
            return None
        return {
            "tip": (float(tr.out_tip[0]), float(tr.out_tip[1])),
            "start": (float(tr.out_start[0]), float(tr.out_start[1])),
        }

    def purge_stale(self, current_frame: int) -> None:
        """Drop tracks that haven't been updated for >gap_threshold_frames."""
        stale = [k for k, tr in self._tracks.items()
                 if current_frame - tr.last_frame > self.gap_threshold_frames]
        for k in stale:
            del self._tracks[k]
