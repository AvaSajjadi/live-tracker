"""Tests for camera_live: the background camera reader, the recorder, and giving the camera back (fake cameras)."""

import time

import cv2
import numpy as np
import pytest

from live_tracker import camera_live as cl
from live_tracker import screen_map as sm


class FakeCapture:
    def __init__(self, interval=0.02, fail_after=None):
        self.interval, self.fail_after, self.reads, self.released = interval, fail_after, 0, False

    def read(self):
        time.sleep(self.interval)
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise RuntimeError("unplugged")
        return True, np.full((60, 80, 3), self.reads % 255, dtype=np.uint8)

    def release(self):
        self.released = True


def wait_for(cond, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_the_grabber_keeps_the_newest_picture_and_stop_gives_the_camera_back():
    cap = FakeCapture()
    g = cl.Grabber("0", cap)
    assert wait_for(lambda: g.latest() is not None) and g.latest().shape == (60, 80, 3)
    first = int(g.latest()[0, 0, 0])
    assert wait_for(lambda: int(g.latest()[0, 0, 0]) != first)             # the picture keeps changing
    g.stop()
    assert cap.released and not g._thread.is_alive()


def test_a_camera_error_is_reported_and_stops_the_reader():
    g = cl.Grabber("0", FakeCapture(fail_after=3))
    assert wait_for(lambda: g.error is not None and not g._thread.is_alive())
    assert "unplugged" not in g.error and "camera gave an error" in g.error
    g.stop()


def test_the_camera_is_given_back_when_nobody_looks_at_it(monkeypatch):
    monkeypatch.setattr(cl, "IDLE_S", 0.2)
    cap = FakeCapture()
    g = cl.Grabber("0", cap)
    assert wait_for(lambda: cap.released and not g._thread.is_alive())
    assert "given back" in g.error


def test_a_recording_has_30_frames_per_second_even_when_the_camera_is_slower(tmp_path):
    cap = FakeCapture(interval=1 / 15)                                    # a webcam that gives 15 pictures a second
    g = cl.Grabber("0", cap)
    assert wait_for(lambda: g.latest() is not None)
    path = tmp_path / "rec.mp4"
    g.start_recording(path)
    assert g.recording
    time.sleep(1.5)
    assert g.recording_status()["frames"] > 20
    info = g.stop_recording()
    g.stop()
    video = cv2.VideoCapture(str(path))
    n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n == info["frames"] and 35 <= n <= 60                          # about 1.5 s at 30 frames per second
    assert abs(video.get(cv2.CAP_PROP_FPS) - 30.0) < 0.5 and info["seconds"] == pytest.approx(n / 30.0)


def test_a_recording_with_corners_is_the_stretched_screen(tmp_path):
    g = cl.Grabber("0", FakeCapture())
    assert wait_for(lambda: g.latest() is not None)
    path = tmp_path / "stretched.mp4"
    g.start_recording(path, sm.check_corners(sm.DEFAULT_CORNERS))
    time.sleep(0.5)
    info = g.stop_recording()
    g.stop()
    assert (info["width"], info["height"]) == (1024, 570)
    assert cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_WIDTH) == 1024


def test_a_second_recording_cannot_start_and_stopping_twice_is_harmless(tmp_path):
    g = cl.Grabber("0", FakeCapture())
    assert wait_for(lambda: g.latest() is not None)
    g.start_recording(tmp_path / "a.mp4")
    with pytest.raises(RuntimeError, match="already running"):
        g.start_recording(tmp_path / "b.mp4")
    assert g.stop_recording() is not None and g.stop_recording() is None
    g.stop()


def test_stopping_a_recording_while_the_camera_thread_is_writing_never_crashes(tmp_path):
    """The camera thread writes pictures while the page stops the recording: used to be an 'access violation' that killed Python."""
    cap = FakeCapture(interval=0.001)                                    # a very fast camera, so the thread is nearly always inside add()
    g = cl.Grabber("0", cap)
    assert wait_for(lambda: g.latest() is not None)
    for i in range(25):
        g.start_recording(tmp_path / f"r{i}.mp4")
        time.sleep(0.02)
        info = g.stop_recording()                                        # closes the file at a random moment of the thread's writing
        assert info is not None and info["frames"] >= 1
    g.stop()
    assert cap.released


def test_a_closed_recorder_ignores_more_pictures(tmp_path):
    rec = cl.Recorder(tmp_path / "x.mp4", None)
    img = np.full((60, 80, 3), 90, np.uint8)
    rec.add(img, 1.0)
    info = rec.close()
    rec.add(img, 1.1)                                                    # after the close: nothing happens, nothing breaks
    assert rec.close()["frames"] == info["frames"]
