"""Tests for scripts/realtime_pose_test.py: the live-feed simulation (fake models, no TensorFlow, no video)."""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("realtime_pose_test", ROOT / "scripts" / "realtime_pose_test.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def frames(n):
    return [np.zeros((4, 4, 3), np.uint8) for _ in range(n)]


def test_a_fast_model_keeps_up_with_the_live_feed():
    done, wall = rt.simulate(frames(31), 30.0, lambda f: np.zeros((2, 7, 2)), seconds=3.0)
    assert len(done) >= 28                                             # nearly every frame (the clock decides the exact count)
    assert wall < 1.6
    assert max(d[2] for d in done) < 0.1                                # keypoints are ready within 0.1 s


def test_a_slow_model_drops_frames_and_always_works_on_the_newest_one():
    done, wall = rt.simulate(frames(61), 30.0, lambda f: (time.sleep(0.25), np.zeros((2, 7, 2)))[1], seconds=2.0)
    assert 4 <= len(done) <= 10                                         # about 2 s / 0.25 s, nowhere near the 61 frames
    indexes = [d[0] for d in done]
    assert indexes == sorted(indexes) and len(set(indexes)) == len(indexes)
    assert all(b - a >= 3 for a, b in zip(indexes, indexes[1:]))       # frames in between were dropped
    assert min(d[2] for d in done) >= 0.2                               # a slow model is always late


def test_the_saved_keypoints_are_read_as_two_mice_with_seven_points():
    rows = []
    for f in (5, 6):
        for m, off in (("a", 0.0), ("b", 100.0)):
            for j, k in enumerate(rt.KEYPOINTS):
                rows.append({"frame": f, "mouse": m, "keypoint": k, "x": off + j, "y": off + 2 * j})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "poses.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)
        truth = rt.saved_keypoints(p, [5])
    assert set(truth) == {5} and truth[5].shape == (2, 7, 2)
    assert truth[5][0, 3].tolist() == [3.0, 6.0] and truth[5][1, 0].tolist() == [100.0, 100.0]
