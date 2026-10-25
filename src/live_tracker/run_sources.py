"""Where a run gets its frames from (new file).

A source gives the runner one frame at a time:
  - ScoreFileSource      ready-made score rows (INTERFACES.md section 2), played back like a stream.
                         The synthetic demo and saved validation scores use it.
  - KeypointFileSource   recorded pose keypoints of one session, played back like the output of a pose
                         model on a camera. The runner turns them into scores with live_scoring.
  - CameraSource         a real camera, read with OpenCV, plus a POSE MODEL that turns each picture into
                         keypoints. There is no mouse pose model in this repository, so a camera cannot
                         be started until one is supplied (NoPoseModelError says so).

Frame numbers are regular (frame = round(seconds * fps) since the first frame); dropped frames just
skip numbers, which the monitor treats as a gap. Sources never read the CalMS21 test split: the factory
refuses a test session.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calms21 import FPS, KEYPOINTS, MICE, check_manifest

ROOT = Path(__file__).resolve().parents[2]
SCORE_COLUMNS = ("session", "frame", "time_s", "attack", "mount", "investigation", "tracking_ok")


class SourceInterrupted(RuntimeError):
    """The source stopped giving frames (camera unplugged, stream dropped). The runner may reconnect."""


class NoPoseModelError(RuntimeError):
    """A camera needs a pose model that turns a picture into keypoints, and none is configured."""


class CameraUnavailableError(RuntimeError):
    """OpenCV is missing or the camera could not be opened."""


class SourceRefusedError(ValueError):
    """The source is not allowed (for example a test session) or its description is wrong."""


@dataclass(frozen=True)
class Frame:
    frame: int
    scores: dict | None = None        # a ready score row, or
    pose: object = None               # pose rows (DataFrame) that still have to be scored


class ScoreFileSource:
    kind = "scores"
    paced = True

    def __init__(self, scores, session=None, fps=FPS, description="score file"):
        df = pd.read_csv(scores) if isinstance(scores, (str, Path)) else scores.copy()
        missing = [c for c in SCORE_COLUMNS if c not in df.columns]
        if missing:
            raise SourceRefusedError(f"the score table lacks columns {missing}")
        if session is None:
            if df["session"].nunique() != 1:
                raise SourceRefusedError("the score table has several sessions; name one")
            session = df["session"].iloc[0]
        df = df[df["session"] == session].sort_values("frame")
        if df.empty:
            raise SourceRefusedError(f"no rows for session {session!r}")
        self.session, self.fps, self.description = session, float(fps), description
        self._rows = df[list(SCORE_COLUMNS)].to_dict("records")
        self._i = 0

    def read(self):
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        row = {**row, "frame": int(row["frame"]), "time_s": float(row["time_s"]), "tracking_ok": bool(row["tracking_ok"])}
        return Frame(frame=row["frame"], scores=row)

    def reconnect(self):
        return True

    def close(self):
        pass


def synthetic_source(fps=FPS):
    from .fake_scores import demo_scores
    df = demo_scores(fps=fps)
    return ScoreFileSource(df, fps=fps, description="SYNTHETIC demo scores (not CalMS21, not a model)")


class KeypointFileSource:
    kind = "pose"
    paced = True

    def __init__(self, poses, session, fps=FPS, description="recorded keypoints"):
        df = poses[poses["session"] == session]
        if df.empty:
            raise SourceRefusedError(f"no pose rows for session {session!r}")
        self.session, self.fps, self.description = session, float(fps), description
        self._groups = [(int(f), g.reset_index(drop=True)) for f, g in df.groupby("frame", sort=True)]
        self._i = 0

    def read(self):
        if self._i >= len(self._groups):
            return None
        frame, rows = self._groups[self._i]
        self._i += 1
        return Frame(frame=frame, pose=rows)

    def reconnect(self):
        return True

    def close(self):
        pass


def lost_pose_rows(session, frame, time_s):
    """Every keypoint missing: the pose model found no mouse. Features see it as lost tracking."""
    rows = [{"session": session, "mouse": m, "keypoint": k, "frame": frame, "time_s": time_s,
             "x": np.nan, "y": np.nan, "score": 0.0} for m in MICE for k in KEYPOINTS]
    return pd.DataFrame(rows)


class CameraSource:
    """A camera plus a pose model.

    capture        object with read() -> (ok, image) and release() (cv2.VideoCapture fits)
    pose_estimator callable(image) -> DataFrame with columns mouse, keypoint, x, y, score in the CalMS21
                   image frame, or None when no mouse is found
    reopen         optional callable() -> a new capture (or None) used by reconnect()
    now_fn         seconds clock (monotonic), replaced in tests
    """
    kind = "pose"
    paced = False                # reading the camera blocks until the next picture

    def __init__(self, capture, pose_estimator, *, session, fps=FPS, reopen=None, now_fn=time.monotonic,
                 description="camera"):
        if pose_estimator is None:
            raise NoPoseModelError(
                "A camera needs a pose model that turns each picture into mouse keypoints, and none is "
                "configured. This repository has no mouse pose model; supply one (a callable that takes an "
                "image and returns the keypoints) before starting a camera run.")
        if not callable(pose_estimator):
            raise NoPoseModelError("the pose model must be a callable that takes an image")
        self.capture, self.pose_estimator, self.reopen = capture, pose_estimator, reopen
        self.session, self.fps, self.now_fn, self.description = session, float(fps), now_fn, description
        self._t0 = None
        self._last = -1

    def read(self):
        while True:
            try:
                ok, image = self.capture.read()
            except Exception as exc:                          # a camera driver can fail in many ways
                raise SourceInterrupted(f"camera read failed: {exc}") from exc
            if not ok:
                raise SourceInterrupted("the camera gave no picture")
            t = self.now_fn()
            if self._t0 is None:
                self._t0 = t
            frame = int(round((t - self._t0) * self.fps))
            if frame <= self._last:                           # two pictures in one frame slot: keep the first
                continue
            self._last = frame
            found = self.pose_estimator(image)
            time_s = frame / self.fps
            if found is None or len(found) == 0:
                return Frame(frame=frame, pose=lost_pose_rows(self.session, frame, time_s))
            rows = pd.DataFrame(found).copy()
            rows.insert(0, "session", self.session)
            rows["frame"], rows["time_s"] = frame, time_s
            return Frame(frame=frame, pose=rows[["session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score"]])

    def reconnect(self):
        if self.reopen is None:
            return False
        try:
            capture = self.reopen()
        except Exception:
            return False
        if capture is None:
            return False
        self.close()
        self.capture = capture
        return True

    def close(self):
        try:
            self.capture.release()
        except Exception:
            pass


def camera_device(value):
    """A camera number (0, 1, ...) or a stream address. Digits become a number."""
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def open_camera(device=0):
    """An OpenCV capture for a camera index or a stream address. OpenCV is imported only here."""
    try:
        import cv2
    except ImportError as exc:
        raise CameraUnavailableError("OpenCV is not installed (pip install opencv-python)") from exc
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        raise CameraUnavailableError(f"could not open the camera {device!r}")
    return capture


def make_source(spec, root=ROOT):
    """Build a source from a plain description (it travels as JSON to the runner program).

    {"kind": "synthetic"}
    {"kind": "scores", "path": "outputs/scores_validation_model_b.csv", "session": "..."}
    {"kind": "keypoints", "json": "data/raw/calms21/calms21_task1_train.json", "session": "..."}
    {"kind": "poses_file", "path": "data/raw/calms21_video/mouse064_poses.parquet", "session": "..."}
    {"kind": "upload_poses", "path": "outputs/uploads/<time>/mice_poses.parquet", "session": "UPLOAD_mice"}
    {"kind": "camera", "device": 0}
    """
    root = Path(root)
    kind = spec.get("kind")
    fps = float(spec.get("fps", FPS))
    if kind == "synthetic":
        return synthetic_source(fps)
    if kind == "scores":
        return ScoreFileSource(root / spec["path"] if not Path(spec["path"]).is_absolute() else spec["path"],
                               spec.get("session"), fps, description=f"score file {Path(spec['path']).name}")
    if kind in ("keypoints", "poses_file"):
        manifest = check_manifest(json.loads((root / "splits/split_manifest.json").read_text(encoding="utf-8")))
        session = spec.get("session")
        path = Path(spec["json"] if kind == "keypoints" else spec["path"])
        if "test" in path.name.lower() or session in manifest["test"]:
            raise SourceRefusedError("test-split sessions are never used by the agent")
        if session not in manifest["train"] + manifest["validation"]:
            raise SourceRefusedError(f"session {session!r} is not in the train or validation split")
        path = path if path.is_absolute() else root / path
        if kind == "keypoints":
            from .calms21 import load_calms21
            poses, _ = load_calms21(path, sessions=[session])
        else:                                   # one session saved by scripts/prepare_demo_session.py: starts at once
            poses = pd.read_parquet(path)
            if set(poses["session"]) != {session}:
                raise SourceRefusedError(f"{path.name} does not hold exactly the session {session!r}")
        return KeypointFileSource(poses, session, fps, description=f"recorded keypoints of {session}")
    if kind == "upload_poses":                  # a keypoints file the researcher uploaded (already checked by uploads.py)
        import re
        session = spec.get("session")
        path = Path(spec["path"])
        path = path if path.is_absolute() else root / path
        if not session or re.search(r"(^|/)test(/|$)", session):
            raise SourceRefusedError("test-split sessions are never used by the agent")
        try:
            if session in check_manifest(json.loads((root / "splits/split_manifest.json").read_text(encoding="utf-8")))["test"]:
                raise SourceRefusedError("test-split sessions are never used by the agent")
        except (OSError, ValueError):
            pass                                # no manifest here: the name check above is all there is
        poses = pd.read_parquet(path)
        if set(poses["session"]) != {session}:
            raise SourceRefusedError(f"{path.name} does not hold exactly the session {session!r}")
        return KeypointFileSource(poses, session, fps, description=f"uploaded keypoints ({path.name})")
    if kind == "camera":
        # the pose model is checked first: no camera is opened when there is no model to read its pictures
        from . import pose_model
        estimator = spec.get("pose_estimator")
        if estimator is None:
            estimator, reason = pose_model.load(root)
            if estimator is None:
                raise NoPoseModelError(reason)
        device = camera_device(spec.get("device", 0))
        return CameraSource(open_camera(device), estimator, session=spec.get("session", "camera"), fps=fps,
                            reopen=lambda: open_camera(device), description=f"camera {device}")
    raise SourceRefusedError(f"unknown source kind {kind!r}")
