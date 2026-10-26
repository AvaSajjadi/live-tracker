"""Tests for scripts/mars_pose_offline.py: the parts that need no TensorFlow (corners, boxes, keypoints, the notebook file)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from live_tracker import screen_map as sm

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mars_pose_offline", ROOT / "scripts" / "mars_pose_offline.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


def test_the_corner_text_from_the_app_is_what_the_script_reads():
    corners = ((0.23, 0.22), (0.81, 0.22), (0.81, 0.65), (0.23, 0.65))
    text = sm.corners_text(corners)
    assert text == "0.230,0.220,0.810,0.220,0.810,0.650,0.230,0.650"
    assert mp.parse_corners(text) == pytest.approx(list(corners))
    assert mp.parse_corners("") is None
    with pytest.raises(SystemExit):
        mp.parse_corners("0.1,0.2,0.3")
    with pytest.raises(SystemExit):
        mp.parse_corners("0,0,2,0,1,1,0,1")


def test_the_script_and_the_app_stretch_the_screen_the_same_way():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[120:360, 160:480] = (200, 200, 200)
    corners = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))
    a, b = mp.to_frame(image, corners), sm.warp(image, corners)
    assert a.shape == b.shape == (570, 1024, 3) and int(np.abs(a.astype(int) - b.astype(int)).max()) <= 1
    assert mp.to_frame(image, None).shape == (570, 1024, 3)


def test_the_best_box_is_stretched_and_a_missing_mouse_keeps_the_last_good_box():
    locs = np.zeros((5, 4)); confs = np.zeros((5, 1))
    locs[2] = [0.2, 0.3, 0.6, 0.7]; confs[2] = 0.9
    box, conf = mp.post_det(locs, confs, ([0.0, 0.0, 0.1, 0.1], 0.1))
    assert conf == pytest.approx(0.9) and box[0] < 0.2 and box[2] > 0.6            # 10% wider on each side
    prev = ([0.4, 0.4, 0.5, 0.5], 0.8)
    assert mp.post_det(locs, np.zeros((5, 1)), prev) == prev                       # nothing found: the last good box


def test_the_brightest_heatmap_point_becomes_the_keypoint_in_frame_pixels():
    hm = np.zeros((64, 64, 7), dtype=np.float32)
    for j in range(7):
        hm[32, 16 + j, j] = 1.0                                                    # row 32 = middle of the box, column 16 + j
    box = [0.25, 0.25, 0.75, 0.75]                                                 # half of the frame, centered
    kps = mp.decode(hm, box)
    assert len(kps) == 7 and all(0.5 < k[2] <= 1.0 for k in kps)         # a one-pixel peak gets softer when it is enlarged
    assert kps[0][1] == pytest.approx(0.5 * 570, abs=8) and kps[0][0] < kps[6][0]    # y in the middle, x grows with the keypoint number


def test_the_colab_notebook_is_valid_and_points_at_the_script():
    nb = json.loads((ROOT / "colab" / "mars_pose_colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert nb["nbformat"] == 4 and "scripts/mars_pose_offline.py" in code and "files.download" in code
    assert "CORNERS" in code and "--max-frames 60" in code                          # a speed test comes before the full run
