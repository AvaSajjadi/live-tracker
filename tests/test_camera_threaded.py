"""Tests for the camera reader thread of CameraSource (buffer_frames): slow moments of the program must not make the camera drop pictures."""

import threading
import time

import pandas as pd
import pytest

from live_tracker import run_sources as rsrc
from live_tracker.calms21 import KEYPOINTS, MICE


def good_pose(image):
    return pd.DataFrame([{"mouse": m, "keypoint": k, "x": 100.0, "y": 200.0, "score": 0.9} for m in MICE for k in KEYPOINTS])


class SteadyCamera:
    """30 pictures per second (real time), forever; fails after `fail_after` pictures if told to."""

    def __init__(self, fps=30.0, fail_after=None):
        self.fps, self.fail_after, self.n, self.released = fps, fail_after, 0, False
        self.t0 = None

    def read(self):
        now = time.monotonic()
        self.t0 = self.t0 or now
        due = self.t0 + self.n / self.fps
        if due > now:
            time.sleep(due - now)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise OSError("usb reset")
        return True, f"picture {self.n}"

    def release(self):
        self.released = True


def test_slow_moments_do_not_drop_pictures_when_the_reader_thread_is_on():
    cam = rsrc.CameraSource(SteadyCamera(), good_pose, session="CAM", fps=30.0, buffer_frames=15)
    frames = []
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        frames.append(cam.read().frame)
        if len(frames) % 8 == 0:
            time.sleep(0.12)                                        # a slow moment of the program, like the scoring every few frames
    cam.close()
    gaps = [b - a for a, b in zip(frames, frames[1:]) if b - a != 1]
    assert len(frames) >= 65 and cam.dropped == 0 and not gaps      # every picture, in order, none skipped (a late one may be repeated)


def test_without_the_thread_the_same_slow_moments_lose_real_pictures():
    class DropsWhenNobodyReads(SteadyCamera):
        """Like a driver that keeps only the newest picture: the one read after a pause is the one that is due NOW."""
        def read(self):
            now = time.monotonic()
            self.t0 = self.t0 or now
            self.n = max(self.n, int((now - self.t0) * self.fps))
            return super().read()

    def run(buffer_frames, camera):
        calls = []
        cam = rsrc.CameraSource(camera, lambda image: (calls.append(image), good_pose(image))[1], session="CAM", fps=30.0,
                                buffer_frames=buffer_frames)
        n, t_end = 0, time.monotonic() + 2.5
        while time.monotonic() < t_end:
            cam.read()
            n += 1
            if n % 8 == 0:
                time.sleep(0.12)
        cam.close()
        return len(set(calls))                                      # how many DIFFERENT real pictures reached the pose model

    with_thread = run(15, SteadyCamera())
    without_thread = run(0, DropsWhenNobodyReads())
    assert with_thread > without_thread + 8                          # the thread keeps the pictures that the driver would have dropped


def test_a_consumer_that_is_always_too_slow_loses_the_oldest_pictures_and_stays_close_to_live():
    cam = rsrc.CameraSource(SteadyCamera(), good_pose, session="CAM", fps=30.0, buffer_frames=6)
    last = None
    for _ in range(6):
        time.sleep(0.25)                                            # the consumer needs 0.25 s per picture, the camera gives one per 0.033 s
        last = cam.read()
    cam.close()
    assert cam.dropped > 0
    lag_frames = int(round((time.monotonic() - cam._t0) * 30.0)) - last.frame        # how many pictures behind the camera it is
    assert 0 <= lag_frames <= 6 + 4                                  # the queue (6 pictures) plus a little: never seconds behind


def test_pictures_already_queued_are_delivered_before_a_failure_is_raised_and_reconnect_restarts_the_thread():
    cam = rsrc.CameraSource(SteadyCamera(fail_after=5), good_pose, session="CAM", fps=30.0, buffer_frames=15,
                            reopen=lambda: SteadyCamera())
    time.sleep(0.4)                                                 # the first 5 pictures are queued, then the camera fails
    got = [cam.read().frame for _ in range(5)]
    assert got == sorted(got) and len(got) == 5
    with pytest.raises(rsrc.SourceInterrupted, match="usb reset"):
        cam.read()
    assert cam.reconnect() is True
    assert cam.read().frame > got[-1]                               # the numbers go on after the gap
    cam.close()


def test_close_stops_the_reader_thread_and_releases_the_camera():
    camera = SteadyCamera()
    cam = rsrc.CameraSource(camera, good_pose, session="CAM", buffer_frames=5)
    cam.read()
    cam.close()
    time.sleep(0.2)
    assert camera.released and not any(t.name == "camera-reader" and t.is_alive() for t in threading.enumerate())
