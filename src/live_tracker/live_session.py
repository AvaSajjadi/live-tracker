"""The live session behind the Real session page: start the run for a chosen source, read its progress (new file).

A session source (uploads.SessionSource) with keypoints is played through the real run program at real speed:
keypoints, live scores from Model B, the rules the researcher approved, the monitor, the event log, the report.
If the source has a video, the run waits for a GO signal so that the page can start the video at the same moment.

It is a simulated live stream from a recording. It needs no camera and is not a camera test.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import monitor_runner as mr
from . import run_control as rc
from . import run_schedule as rs
from .calms21 import FPS, TORSO_KEYPOINTS
from .pose_preview import PREVIEW_FILE

MODEL = "models/model_b"
EXTRA_S = 120.0                         # the schedule lasts this much longer than the recording, so it never ends first
SCHEDULE_ID = "LIVE"
CAMERA_DEFAULT_S = 1800.0               # a camera run with no time chosen on the Rules page stops after 30 minutes


def length_s(poses_path, fps=FPS):
    """Length of the recording in seconds (from its frames)."""
    frames = pd.read_parquet(poses_path, columns=["frame"])["frame"]
    return (int(frames.max()) + 1) / fps


def run_spec(source):
    """The run_sources description of a source: the demo recording is checked against the CalMS21 split,
    anything uploaded is checked by name."""
    if not source.can_score:
        raise mr.RunRefusedError("this source has no keypoints, so there is nothing to score")
    return {"kind": "poses_file" if source.kind == "demo" else "upload_poses", "path": str(source.poses),
            "session": source.session}


def live_schedule(seconds, now=None, tz="America/New_York", limit_s=None):
    """An approved schedule that starts now. It lasts the recording plus a margin, or exactly `limit_s` seconds when
    the researcher chose a shorter time on the Rules page. Starting the session is the approval."""
    now = now or datetime.now(timezone.utc)
    total = limit_s if limit_s is not None and limit_s < seconds else seconds + EXTRA_S
    return rs.for_hours(total / 3600.0, now=now, tz=tz, schedule_id=SCHEDULE_ID,
                        description="Live session").approve()


def start_live(source, rules, *, wait_for_go, now=None, launcher=None, limit_s=None):
    """Start the run for this source with the researcher's approved rules. Returns the run folder."""
    now = now or datetime.now(timezone.utc)
    run_dir = rc.new_run_dir(SCHEDULE_ID, now)
    config = rc.build_config(run_dir, live_schedule(length_s(source.poses, source.fps), now, limit_s=limit_s), rules, run_spec(source),
                             model=MODEL, speed=1.0, fps=source.fps)
    config.wait_for_go = bool(wait_for_go)
    return rc.start_run(config, launcher=launcher)


def camera_limit_s(limit_s=None):
    """How long a camera run lasts: the time confirmed on the Rules page, else CAMERA_DEFAULT_S. A camera has no end."""
    return float(limit_s) if limit_s else CAMERA_DEFAULT_S


def camera_spec(source):
    """The run_sources description of a real camera. No GO signal: the camera is already live."""
    if source.kind != "camera":
        raise mr.RunRefusedError("this source is not a camera")
    spec = {"kind": "camera", "device": (source.device or "0").strip(), "session": "camera"}
    if source.screen is not None:
        spec["screen"] = [list(p) for p in source.screen]
    return spec


def start_camera(source, rules, *, now=None, launcher=None, limit_s=None):
    """Start a run on a real camera with the researcher's approved rules. Returns the run folder."""
    now = now or datetime.now(timezone.utc)
    seconds = camera_limit_s(limit_s)
    run_dir = rc.new_run_dir("CAMERA", now)
    schedule = rs.for_hours(seconds / 3600.0, now=now, tz="America/New_York", schedule_id="CAMERA",
                            description="Camera session").approve()
    spec = {**camera_spec(source), "preview": str(Path(run_dir) / PREVIEW_FILE)}   # the newest drawn picture, for the page
    config = rc.build_config(run_dir, schedule, rules, spec, model=MODEL, speed=1.0, fps=source.fps,
                            start_when_ready=True)                       # loading the model and the camera is not part of the chosen time
    return rc.start_run(config, launcher=launcher)


def go_if_ready(run_dir):
    """True when the run is ready and has not been told to start yet."""
    status = rc.read_status(run_dir)
    return bool(status and status.get("state") == "ready" and not (Path(run_dir) / mr.GO_FILE).exists())


def latest_alert(events):
    """The most recent alert or question, or None."""
    shown = [e for e in events if e.get("action") in ("alert", "ask")]
    return max(shown, key=lambda e: e["start_s"]) if shown else None


def failed_message(status):
    """Why the run is not going to start, or None."""
    if status and status.get("state") == "failed":
        return status.get("message") or "the run program failed"
    return None


# ---- the motion numbers shown next to the video ----

def _series_from_centers(cx, cy, frames, fps):
    dt = pd.Series(frames, index=frames).diff() / fps
    out = pd.DataFrame({"time_s": np.asarray(frames) / fps}, index=frames)
    for mouse in ("a", "b"):
        raw = np.hypot(cx[mouse].diff(), cy[mouse].diff()) / dt
        smooth = raw.rolling(5, min_periods=1).mean()                              # causal: only the past 5 frames
        out[f"speed_{mouse}"] = smooth.where(raw.notna())                          # missing stays missing, never "still"
    out["distance"] = np.hypot(cx["a"] - cx["b"], cy["a"] - cy["b"])
    out["tracking_ok"] = cx["a"].notna() & cx["b"].notna()
    return out.reset_index(drop=True)


def motion_series(poses_path, fps=FPS):
    """Speed of each mouse (px/s), the distance between them (px) and whether both were tracked, per frame.

    The body center is the mean of the torso keypoints (neck and both hips), as in the motion engine. These are
    measurements of movement, not behavior labels."""
    poses = pd.read_parquet(poses_path, columns=["mouse", "keypoint", "frame", "x", "y"])
    torso = poses[poses["keypoint"].isin(TORSO_KEYPOINTS)]
    centers = torso.groupby(["frame", "mouse"])[["x", "y"]].mean()
    cx, cy = centers["x"].unstack("mouse"), centers["y"].unstack("mouse")
    return _series_from_centers(cx, cy, cx.index.to_numpy(), fps)


def centers_series(df, fps=FPS):
    """The same numbers from a centers table (x_a, y_a, x_b, y_b), one row per frame."""
    frames = np.arange(len(df))
    cx = pd.DataFrame({"a": df["x_a"].to_numpy(float), "b": df["x_b"].to_numpy(float)}, index=frames)
    cy = pd.DataFrame({"a": df["y_a"].to_numpy(float), "b": df["y_b"].to_numpy(float)}, index=frames)
    return _series_from_centers(cx, cy, frames, fps)
