"""One open camera shared by the pages that show live video, with a recorder (new file).

On Windows only one program can use a camera at a time. The Upload step (camera test, corners, recording) and the Real
session page (staged demo) both show the live picture, so they share ONE open camera kept in the Streamlit session. A small
background thread (Grabber) reads the camera all the time and keeps the newest picture; the pages only look at that picture.
The same thread writes the recording, so the video has every picture the camera gives, not only the ones a page asked for.

release() stops the thread and gives the camera back: by the Stop button, when the session is reset, and before a real camera
run starts (that run opens the camera in its own program).

Recordings are written at a fixed 30 frames per second (the rate Model B and MARS expect). If the camera is slower (a webcam
often gives 15), each picture is written twice, so the video keeps the right length.
"""

from __future__ import annotations

import threading
import time

import streamlit as st

from . import run_sources as rsrc

KEY_GRABBER = "camera_live_grabber"
RECORD_FPS = 30.0
IDLE_S = 20.0                       # no page looked at the camera for this long (and no recording): give the camera back


class Recorder:
    """Writes pictures to an mp4 at a fixed frame rate. With corners, the screen is stretched to 1024 x 570 first."""

    def __init__(self, path, corners=None, fps=RECORD_FPS):
        self.path, self.corners, self.fps = str(path), corners, float(fps)
        self.writer, self.t0, self.written, self.size = None, None, 0, None

    def _output(self, image):
        if self.corners is None:
            return image
        from . import screen_map as sm
        return sm.warp(image, self.corners)

    def add(self, image, t):
        import cv2
        frame = self._output(image)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.size = (w, h)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.writer.isOpened():
                raise OSError(f"the recording file {self.path} cannot be written")
            self.t0 = t
        target = int(round((t - self.t0) * self.fps)) + 1
        while self.written < target:                    # a slower camera: the same picture again, so the length stays right
            self.writer.write(frame)
            self.written += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return {"path": self.path, "frames": self.written, "seconds": self.written / self.fps, "fps": self.fps,
                "width": self.size[0] if self.size else None, "height": self.size[1] if self.size else None}


class Grabber:
    def __init__(self, device, capture):
        self.device, self.capture = str(device).strip(), capture
        self.frame, self.counter, self.error = None, 0, None
        self._last_seen = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._recorder = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"camera-{self.device}")
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            if self._recorder is None and time.monotonic() - self._last_seen > IDLE_S:
                self.error = "the camera was given back because no page was looking at it"
                break
            try:
                ok, image = self.capture.read()
            except Exception as exc:                    # a camera driver can fail in many ways
                self.error = f"the camera gave an error ({type(exc).__name__})"
                return
            if not ok or image is None:
                self.error = "the camera gave no picture"
                time.sleep(0.05)
                continue
            self.error = None
            with self._lock:
                self.frame, self.counter = image, self.counter + 1
            recorder = self._recorder
            if recorder is not None:
                try:
                    recorder.add(image, time.monotonic())
                except Exception as exc:                # the recording stops; the live picture goes on
                    self.error = f"the recording stopped: {exc}"
                    self._recorder = None
            time.sleep(0.002)
        try:
            self.capture.release()
        except Exception:
            pass

    def latest(self):
        self._last_seen = time.monotonic()
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def start_recording(self, path, corners=None):
        if self._recorder is not None:
            raise RuntimeError("a recording is already running")
        self._recorder = Recorder(path, corners)

    @property
    def recording(self):
        return self._recorder is not None

    def recording_status(self):
        r = self._recorder
        return None if r is None else {"frames": r.written, "seconds": r.written / r.fps, "path": r.path}

    def stop_recording(self):
        recorder, self._recorder = self._recorder, None
        return recorder.close() if recorder is not None else None

    def stop(self):
        self.stop_recording()
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.capture.release()
        except Exception:
            pass


def release():
    grabber = st.session_state.pop(KEY_GRABBER, None)
    if grabber is not None:
        grabber.stop()


def grabber(device):
    """The running Grabber for this camera number or address. Opens the camera on first use; another camera replaces it."""
    device = str(device).strip()
    g = st.session_state.get(KEY_GRABBER)
    if g is not None and g.device == device and g._thread.is_alive():
        return g
    release()
    g = Grabber(device, rsrc.open_camera(rsrc.camera_device(device)))    # raises CameraUnavailableError
    st.session_state[KEY_GRABBER] = g
    return g


def read(device):
    """(the newest picture, None), or (None, the reason in plain words)."""
    try:
        g = grabber(device)
    except rsrc.CameraUnavailableError as exc:
        return None, str(exc)
    image = g.latest()
    if image is None:
        return None, g.error or "the camera has not given a picture yet"
    if g.error and not g._thread.is_alive():
        release()
        return None, g.error
    return image, None


def current():
    """The running Grabber, or None (does not open the camera)."""
    return st.session_state.get(KEY_GRABBER)
