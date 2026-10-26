"""Tests for screen_map: the video inside a camera picture is stretched to the CalMS21 frame (no hardware)."""

import numpy as np
import pytest

from live_tracker import run_sources as rsrc
from live_tracker import screen_map as sm
from live_tracker import uploads as up
from live_tracker.calms21 import IMAGE_SIZE

W, H = 640, 480
BOX = ((160, 120), (480, 120), (480, 360), (160, 360))                 # a bright box on a dark picture (pixels)
CORNERS = tuple((x / W, y / H) for x, y in BOX)


def picture():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[120:360, 160:480] = (200, 200, 200)
    return img


def test_the_screen_fills_the_calms21_frame():
    out = sm.warp(picture(), CORNERS)
    assert out.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    assert out[5:-5, 5:-5].min() > 150                                  # no dark border inside: the box fills the frame


def test_a_corner_marked_in_the_picture_lands_in_the_matching_corner():
    img = picture()
    img[125:135, 165:175] = (0, 0, 255)                                 # red dot near the top-left of the box
    out = sm.warp(img, CORNERS)
    ys, xs = np.where((out[..., 2] > 150) & (out[..., 1] < 50))
    assert xs.mean() < IMAGE_SIZE[0] * 0.1 and ys.mean() < IMAGE_SIZE[1] * 0.1


def test_bad_corners_say_what_is_wrong():
    with pytest.raises(sm.ScreenError, match="exactly 4"):
        sm.check_corners(CORNERS[:3])
    with pytest.raises(sm.ScreenError, match="inside the picture"):
        sm.check_corners(((0, 0), (1.5, 0), (1, 1), (0, 1)))
    with pytest.raises(sm.ScreenError, match="in order"):
        sm.check_corners((CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]))      # crossed
    with pytest.raises(sm.ScreenError, match="too small"):
        sm.check_corners(((0.5, 0.5), (0.52, 0.5), (0.52, 0.52), (0.5, 0.52)))


def test_the_outline_keeps_the_picture_size():
    assert sm.outline(picture(), CORNERS).shape == (H, W, 3)


def test_the_camera_source_gives_the_pose_model_the_stretched_picture(monkeypatch):
    seen = []

    class Capture:
        def read(self):
            return True, picture()

        def release(self):
            pass

    monkeypatch.setattr(rsrc, "open_camera", lambda device: Capture())
    spec = {"kind": "camera", "device": 0, "screen": [list(p) for p in CORNERS],
            "pose_estimator": lambda image: seen.append(image.shape) or None}
    source = rsrc.make_source(spec)
    source.read()
    source.close()
    assert seen == [(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)]


def test_unusable_corners_refuse_the_run(monkeypatch):
    monkeypatch.setattr(rsrc, "open_camera", lambda device: None)
    with pytest.raises(rsrc.SourceRefusedError, match="screen corners"):
        rsrc.make_source({"kind": "camera", "screen": [[0, 0], [1, 0], [1, 1]], "pose_estimator": lambda i: None})


def test_a_source_keeps_its_corners_through_to_dict_and_from_dict():
    source = up.SessionSource("camera", "c", device="0", screen=CORNERS)
    again = up.SessionSource.from_dict(__import__("json").loads(__import__("json").dumps(source.to_dict())))
    assert again.screen == source.screen
    with pytest.raises(up.UploadError, match="screen corners"):
        up.SessionSource("camera", "c", screen=((0, 0), (1, 0), (1, 1)))
