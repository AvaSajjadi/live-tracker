"""The live numbers of a camera run, written by the run program and charted by the Real session page (new file).

A camera has no keypoints file, so the page cannot compute anything itself. The run program appends, for each frame:
  - live_positions.csv: the body center of each mouse (mean of the torso keypoints), empty when the mouse is lost;
  - live_scores.csv: the classifier's scores (attack, investigation, mount) and tracking_ok, as the monitor gets them.
These are what the monitor already sees; nothing new is decided here. Lines are flushed about twice a second.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd

from .calms21 import TORSO_KEYPOINTS

POSITIONS_FILE = "live_positions.csv"
SCORES_FILE = "live_scores.csv"
BEHAVIORS = ("attack", "investigation", "mount")
FLUSH_S = 0.5


def _num(value):
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.4g}"


class LiveLog:
    def __init__(self, run_dir, now_fn=time.monotonic):
        run_dir = Path(run_dir)
        self.now_fn, self._flushed = now_fn, now_fn()
        self.positions = open(run_dir / POSITIONS_FILE, "w", encoding="utf-8")
        self.scores = open(run_dir / SCORES_FILE, "w", encoding="utf-8")
        self.positions.write("frame,time_s,x_a,y_a,x_b,y_b\n")
        self.scores.write("frame,time_s," + ",".join(BEHAVIORS) + ",tracking_ok\n")

    def position(self, frame):
        pose = frame.pose
        torso = pose[pose["keypoint"].isin(TORSO_KEYPOINTS)]
        centers = torso.groupby("mouse")[["x", "y"]].mean()
        cells = []
        for mouse in ("a", "b"):
            x, y = (centers.loc[mouse, "x"], centers.loc[mouse, "y"]) if mouse in centers.index else (None, None)
            cells += [_num(x), _num(y)]
        time_s = float(pose["time_s"].iloc[0]) if len(pose) else ""
        self.positions.write(f"{frame.frame},{_num(time_s)}," + ",".join(cells) + "\n")
        self._maybe_flush()

    def score(self, row):
        self.scores.write(f"{row['frame']},{_num(row['time_s'])}," + ",".join(_num(row.get(b)) for b in BEHAVIORS)
                          + f",{int(bool(row.get('tracking_ok')))}\n")
        self._maybe_flush()

    def _maybe_flush(self):
        if self.now_fn() - self._flushed >= FLUSH_S:
            self.flush()

    def flush(self):
        self._flushed = self.now_fn()
        self.positions.flush()
        self.scores.flush()

    def close(self):
        for f in (self.positions, self.scores):
            try:
                f.close()
            except OSError:
                pass


def _read(path):
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def read_scores(run_dir, last_s):
    """The scores of the last `last_s` seconds (time_s and the three behaviors), or None before the first line."""
    df = _read(Path(run_dir) / SCORES_FILE)
    if df is None or df.empty:
        return None
    return df[df["time_s"] >= df["time_s"].iloc[-1] - last_s]


def read_motion(run_dir, last_s, fps):
    """Speed of each mouse, their distance and tracking_ok for the last `last_s` seconds, or None before the first line."""
    from .live_session import _series_from_centers
    df = _read(Path(run_dir) / POSITIONS_FILE)
    if df is None or df.empty:
        return None
    df = df[df["frame"] >= df["frame"].iloc[-1] - int(last_s * fps)].drop_duplicates("frame", keep="last")
    frames = df["frame"].to_numpy()
    cx = pd.DataFrame({"a": df["x_a"].to_numpy(float), "b": df["x_b"].to_numpy(float)}, index=frames)
    cy = pd.DataFrame({"a": df["y_a"].to_numpy(float), "b": df["y_b"].to_numpy(float)}, index=frames)
    return _series_from_centers(cx, cy, frames, fps)
