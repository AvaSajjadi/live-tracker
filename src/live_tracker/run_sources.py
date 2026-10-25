"""Where a run gets its frames from (new file).

A source gives the runner one frame at a time:
  - ScoreFileSource      ready-made score rows (INTERFACES.md section 2), played back like a stream.
                         The synthetic demo and saved validation scores use it.
  - KeypointFileSource   recorded pose keypoints of one session, played back like the output of a pose
                         model on a camera. The runner turns them into scores with live_scoring.
  - CameraSource         a real camera, read with OpenCV, plus a POSE MODEL that turns each picture into
                         keypoints. A camera cannot be started until a pose model is set up
                         (NoPoseModelError says so; see pose_model.py and pose_backends.py).
  - video_camera         a video file played like a camera (PacedVideoCapture) into the same CameraSource: for rehearsals and
                         acceptance tests of the live camera path without any camera hardware.

Frame numbers are regular (frame = round(seconds * fps) since the first frame); dropped frames just
skip numbers, which the monitor treats as a gap. Sources never read the CalMS21 test split: the factory
refuses a test session.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calms21 import FPS, KEYPOINTS, MICE, check_manifest

ROOT = Path(__file__).resolve().parents[2]
MAX_FILL = 3                        # a gap of up to this many missing frame numbers is filled with repeats; a longer one stays a gap
BUFFER_FRAMES = 15                  # a camera run keeps the last half second of pictures (see _ReaderThread)
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


class _ReaderThread:
    """Reads a capture all the time and keeps the last few pictures, each with the moment it was taken.

    Without it, the pictures are read only when the consumer asks, so a slow moment of the consumer (the scoring takes 150 ms every
    10th frame) makes the camera drive drop pictures. With it, the pictures wait in a short queue; only when the consumer is slower
    than the camera for longer than the queue holds, the OLDEST pictures are dropped (so the keypoints are never older than the queue)."""

    def __init__(self, capture, now_fn, maxlen):
        self.capture, self.now_fn = capture, now_fn
        self._queue = collections.deque(maxlen=int(maxlen))
        self._cv = threading.Condition()
        self._error = None
        self._stop = threading.Event()
        self.dropped = 0
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera-reader")
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                ok, image = self.capture.read()
            except Exception as exc:                          # a camera driver can fail in many ways
                return self._fail(f"camera read failed: {exc}")
            if not ok:
                return self._fail("the camera gave no picture")
            t = self.now_fn()
            with self._cv:
                if len(self._queue) == self._queue.maxlen:
                    self.dropped += 1
                self._queue.append((image, t))
                self._cv.notify()
            time.sleep(0.002)                                 # a real camera blocks in read(); this stops a read that does not from spinning

    def _fail(self, message):
        with self._cv:
            self._error = message
            self._cv.notify_all()

    def get(self, timeout=10.0):
        """(picture, the time it was taken). Queued pictures come first; when the camera failed and none are left, SourceInterrupted."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._queue or self._error, timeout=timeout):
                raise SourceInterrupted("the camera gave no picture (timeout)")
            if self._queue:
                return self._queue.popleft()
            raise SourceInterrupted(self._error)

    def stop(self):
        self._stop.set()


class CameraSource:
    """A camera plus a pose model.

    capture        object with read() -> (ok, image) and release() (cv2.VideoCapture fits)
    pose_estimator callable(image) -> DataFrame with columns mouse, keypoint, x, y, score in the CalMS21
                   image frame, or None when no mouse is found
    reopen         optional callable() -> a new capture (or None) used by reconnect()
    now_fn         seconds clock (monotonic), replaced in tests
    buffer_frames  0 reads the camera only when a frame is asked for. A number (for example 15) reads it all the time in a thread
                   and keeps that many pictures, so a slow moment of the program does not make the camera drop pictures
    """
    kind = "pose"
    paced = False                # reading the camera blocks until the next picture
    live = True                  # a live picture stream: the run program scores it in larger chunks (monitor_runner.build_scorer)

    def __init__(self, capture, pose_estimator, *, session, fps=FPS, reopen=None, now_fn=time.monotonic,
                 description="camera", buffer_frames=0):
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
        self._previous = None
        self._pending = []
        self.buffer_frames = int(buffer_frames)
        self._reader = _ReaderThread(capture, now_fn, self.buffer_frames) if self.buffer_frames else None

    @property
    def dropped(self):
        """Pictures dropped because the program was slower than the camera for longer than the queue holds (0 without a reader)."""
        return self._reader.dropped if self._reader else 0

    def _next_picture(self):
        if self._reader is not None:
            return self._reader.get()
        try:
            ok, image = self.capture.read()
        except Exception as exc:                              # a camera driver can fail in many ways
            raise SourceInterrupted(f"camera read failed: {exc}") from exc
        if not ok:
            raise SourceInterrupted("the camera gave no picture")
        return image, self.now_fn()

    def read(self):
        if self._pending:                                     # repeats of the last picture that fill a small gap come first
            return self._pending.pop(0)
        while True:
            image, t = self._next_picture()
            if self._t0 is None:
                self._t0 = t
            frame = int(round((t - self._t0) * self.fps))     # the frame number is the clock's slot, so it never drifts
            if frame <= self._last:                           # two pictures in one frame slot: keep the first
                continue
            found = self.pose_estimator(image)
            skipped = frame - self._last - 1 if self._last >= 0 else 0
            if 0 < skipped <= MAX_FILL and self._previous is not None:
                # A camera slower than 30 fps (a webcam gives 15), or a late picture: repeat the previous keypoints for the missing
                # numbers, so the numbers stay consecutive. A gap would end every alert in progress (the monitor treats it as a stall).
                self._pending = [self._make_frame(self._previous, n) for n in range(self._last + 1, frame)]
            self._last = frame
            self._previous = found
            current = self._make_frame(found, frame)
            if self._pending:
                self._pending.append(current)
                return self._pending.pop(0)
            return current

    def _make_frame(self, found, frame):
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
        self._reader = _ReaderThread(capture, self.now_fn, self.buffer_frames) if self.buffer_frames else None
        return True

    def close(self):
        if self._reader is not None:
            self._reader.stop()
        try:
            self.capture.release()
        except Exception:
            pass


class PacedVideoCapture:
    """A video file that behaves like a camera: read() gives the picture that is due NOW at `fps`.

    It waits when the next picture is not due yet, and when the reader is slow it skips ahead to the newest picture, like a live camera
    that drops what nobody read. At the end the video starts again from `start`. Same read() / release() as cv2.VideoCapture."""

    def __init__(self, path, *, start=0, fps=FPS, now_fn=time.monotonic, sleep_fn=time.sleep):
        try:
            import cv2
        except ImportError as exc:
            raise CameraUnavailableError("OpenCV is not installed (pip install opencv-python)") from exc
        self.path, self.start, self.fps = str(path), int(start), float(fps)
        self.now_fn, self.sleep_fn = now_fn, sleep_fn
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise CameraUnavailableError(f"could not open the video {self.path!r}")
        self._cv2 = cv2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        self._t0, self._pos = None, 0

    def read(self):
        now = self.now_fn()
        if self._t0 is None:
            self._t0 = now
        due = int((now - self._t0) * self.fps)
        if due < self._pos:                                    # the next picture is not due yet: wait for it
            self.sleep_fn(max(0.0, self._pos / self.fps - (now - self._t0)))
            due = self._pos
        while self._pos < due:                                 # a slow reader: drop the pictures nobody read
            if not self.cap.grab():
                self._restart()
                break
            self._pos += 1
        ok, image = self.cap.read()
        if not ok:                                             # the end of the video: start again
            self._restart()
            ok, image = self.cap.read()
        self._pos += 1
        return ok, image

    def _restart(self):
        self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, self.start)

    def release(self):
        self.cap.release()


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
    {"kind": "video_camera", "path": "data/raw/calms21_video/x.mp4", "start": 1800}      (a video played like a camera, for rehearsals)
    {"kind": "camera", "device": 0, "screen": [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]]}
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
    if kind in ("camera", "video_camera"):
        # the pose model is checked first: no camera is opened when there is no model to read its pictures
        from . import pose_model
        estimator = spec.get("pose_estimator")
        if estimator is None:
            estimator, reason = pose_model.load(root)
            if estimator is None:
                raise NoPoseModelError(reason)
        if spec.get("screen") is not None:      # the camera films a screen: stretch the video part to the CalMS21 frame first
            from .screen_map import ScreenError, with_screen
            try:
                estimator = with_screen(estimator, spec["screen"])
            except ScreenError as exc:
                raise SourceRefusedError(f"the screen corners cannot be used: {exc}") from exc
        if kind == "video_camera":
            path = Path(spec["path"])
            path = path if path.is_absolute() else root / path
            start = int(spec.get("start", 0))
            return CameraSource(PacedVideoCapture(path, start=start, fps=fps), estimator, session=spec.get("session", "camera"), fps=fps,
                                reopen=lambda: PacedVideoCapture(path, start=start, fps=fps), description=f"video as camera ({path.name})",
                                buffer_frames=BUFFER_FRAMES)
        device = camera_device(spec.get("device", 0))
        return CameraSource(open_camera(device), estimator, session=spec.get("session", "camera"), fps=fps,
                            reopen=lambda: open_camera(device), description=f"camera {device}", buffer_frames=BUFFER_FRAMES)
    raise SourceRefusedError(f"unknown source kind {kind!r}")
