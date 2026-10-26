"""Checks for the camera step on the Upload page (new file).

Three questions, answered in plain words: is OpenCV installed, can the camera be opened and give a picture, and is
there a pose model. A camera can only be analysed when all three are yes. OpenCV is imported only here and in
run_sources.open_camera.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import pose_model
from . import run_sources as rsrc

TRY_NUMBERS = range(4)                      # camera numbers tried by find_cameras


@dataclass(frozen=True)
class CameraCheck:
    opencv: bool
    camera_ok: bool
    camera_message: str
    pose_model_ok: bool
    pose_model_message: str

    @property
    def can_analyse(self):
        return self.opencv and self.camera_ok and self.pose_model_ok


def opencv_available():
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def snapshot(device, opener=None):
    """(PNG bytes of one picture, None), or (None, the reason). The camera is released right away."""
    opener = opener or rsrc.open_camera
    try:
        capture = opener(rsrc.camera_device(device))
    except rsrc.CameraUnavailableError as exc:
        return None, str(exc)
    try:
        ok, image = capture.read()
    except Exception as exc:                                  # a driver can fail in many ways
        return None, f"the camera gave an error ({type(exc).__name__})"
    finally:
        capture.release()
    if not ok or image is None:
        return None, "the camera is open but gave no picture"
    import cv2
    done, png = cv2.imencode(".png", image)
    return (png.tobytes(), None) if done else (None, "the picture could not be saved")


def find_cameras(opener=None, numbers=TRY_NUMBERS):
    """The camera numbers that open and give a picture."""
    return [n for n in numbers if snapshot(n, opener)[0] is not None]


def _pose(root, env):
    ok, reason = pose_model.check(root, env)                  # not load(): loading a trained model takes seconds on every click
    return ok, "A pose model is set up." if ok else reason


def check(device, opener=None, root=None, env=None):
    if not opencv_available():
        return CameraCheck(False, False, "OpenCV is not installed (pip install opencv-python-headless).",
                           *_pose(root, env))
    png, why = snapshot(device, opener)
    message = "The camera gives a picture." if png else f"The camera cannot be used: {why}."
    return CameraCheck(True, png is not None, message, *_pose(root, env))
