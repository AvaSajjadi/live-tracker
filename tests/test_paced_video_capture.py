"""Tests for PacedVideoCapture (a video played like a camera) and the video_camera source."""

import cv2
import numpy as np
import pandas as pd
import pytest

from live_tracker import run_sources as rs
from live_tracker.calms21 import KEYPOINTS


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))               # frame i has the grey level 10 * i
    w.release()
    return path


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def level(img):
    return int(round(img.mean() / 10.0))


def test_it_waits_for_the_next_picture_and_gives_every_picture_to_a_fast_reader(video):
    c = Clock()
    cap = rs.PacedVideoCapture(video, fps=30.0, now_fn=c.now, sleep_fn=c.sleep)
    seen = [level(cap.read()[1]) for _ in range(5)]
    assert seen == [0, 1, 2, 3, 4]
    assert len(c.slept) == 4 and all(abs(s - 1 / 30.0) < 1e-6 for s in c.slept)       # it waited one picture time each time


def test_a_slow_reader_skips_to_the_newest_picture_like_a_live_camera(video):
    c = Clock()
    cap = rs.PacedVideoCapture(video, fps=30.0, now_fn=c.now, sleep_fn=c.sleep)
    assert level(cap.read()[1]) == 0
    c.t = 0.30                                                       # 9 picture times pass while the reader is busy
    assert level(cap.read()[1]) == 9


def test_it_starts_where_it_is_told_and_starts_again_at_the_end(video):
    c = Clock()
    cap = rs.PacedVideoCapture(video, start=17, fps=30.0, now_fn=c.now, sleep_fn=c.sleep)
    seen = [level(cap.read()[1]) for _ in range(5)]
    assert seen[:3] == [17, 18, 19] and seen[3] == 17                # after the last picture it goes back to `start`


def test_a_missing_video_is_a_clear_error(tmp_path):
    with pytest.raises(rs.CameraUnavailableError):
        rs.PacedVideoCapture(tmp_path / "nope.mp4")


def test_the_video_camera_source_gives_frames_with_keypoints_from_the_pose_function(video, tmp_path):
    rows = pd.DataFrame([{"mouse": m, "keypoint": k, "x": 100.0 + i, "y": 200.0, "score": 0.9} for m in "ab" for i, k in enumerate(KEYPOINTS)])
    src = rs.make_source({"kind": "video_camera", "path": str(video), "start": 0, "pose_estimator": lambda image: rows}, root=tmp_path)
    frames = [src.read() for _ in range(6)]
    assert [f.frame for f in frames] == sorted(f.frame for f in frames) and len(set(f.frame for f in frames)) == 6
    assert set(frames[0].pose["mouse"]) == {"a", "b"} and len(frames[0].pose) == 14
    assert "video as camera" in src.description
    src.close()
