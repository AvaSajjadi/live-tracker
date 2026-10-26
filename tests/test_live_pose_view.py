import numpy as np
import pandas as pd

from live_tracker import pose_preview as pp
from live_tracker.calms21 import IMAGE_SIZE


def found():
    return pd.DataFrame({"mouse": ["a", "a"], "keypoint": ["nose", "left_ear"], "x": [100.0, 140.0],
                         "y": [100.0, 120.0], "score": [0.9, 0.05]})


def test_to_frame_brings_the_picture_to_the_calms21_frame():
    out = pp.to_frame(np.zeros((768, 1024, 3), np.uint8), None)
    assert out.shape[:2] == (IMAGE_SIZE[1], IMAGE_SIZE[0])


def test_draw_marks_the_found_points_and_leaves_the_input_alone():
    image = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    out = pp.draw(image, found())
    assert image.sum() == 0
    assert out[100, 100].any() and out[110, 120].any()            # the nose point and the nose - left ear line


def test_draw_without_mice_is_the_plain_picture():
    image = np.full((10, 10, 3), 7, np.uint8)
    assert (pp.draw(image, None) == image).all()


def test_with_preview_saves_the_drawn_picture_at_most_every_period(tmp_path):
    seen, clock = [], iter([0.0, 0.1, 0.25])
    estimate = pp.with_preview(lambda frame: seen.append(frame.shape) or found(), tmp_path / pp.PREVIEW_FILE,
                               every_s=0.2, now_fn=lambda: next(clock))
    path = tmp_path / pp.PREVIEW_FILE
    estimate(np.zeros((768, 1024, 3), np.uint8))
    first = path.stat().st_mtime_ns
    path.write_bytes(b"x")
    estimate(np.zeros((768, 1024, 3), np.uint8))                  # 0.1 s later: not saved again
    assert path.read_bytes() == b"x"
    estimate(np.zeros((768, 1024, 3), np.uint8))                  # 0.25 s: saved again
    assert path.read_bytes()[:2] == b"\xff\xd8" and first
    assert seen == [(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)] * 3          # the model gets the CalMS21 frame


def test_a_preview_never_stops_the_run(tmp_path):
    estimate = pp.with_preview(lambda frame: found(), tmp_path / "missing_folder" / pp.PREVIEW_FILE)
    assert estimate(np.zeros((570, 1024, 3), np.uint8)) is not None
