"""Tests for point_stabilizer: steadier points from noisy ones, with no invented points."""

import json

import numpy as np
import pandas as pd

from live_tracker import point_stabilizer as ps
from live_tracker import pose_model as pm
from live_tracker.calms21 import KEYPOINTS, MICE


class Clock:
    def __init__(self):
        self.t = 0.0

    def tick(self):
        self.t += 1 / 30.0

    def __call__(self):
        return self.t


def table(x, y):
    return pd.DataFrame([{"mouse": m, "keypoint": k, "x": float(x), "y": float(y), "score": 0.9} for m in MICE for k in KEYPOINTS])


def blank():
    return np.zeros((570, 1024, 3), np.uint8)


def run(settings, xs, ys=None):
    clock, st, out = Clock(), None, []
    st = ps.PointStabilizer(settings, clock)
    for i, x in enumerate(xs):
        clock.tick()
        res = st.stabilize(blank(), table(x, 200.0 if ys is None else ys[i]))
        out.append(float(res["x"].iloc[0]))
    return np.array(out)


def test_a_still_point_with_noise_is_much_steadier_and_the_rows_do_not_change():
    rng = np.random.default_rng(1)
    noisy = 300.0 + rng.normal(0, 2.0, 150)
    steady = run({"flow": False}, noisy)
    assert steady[30:].std() < 0.3 * noisy[30:].std()
    clock = Clock()
    clock.tick()
    raw = table(300.0, 200.0)
    out = ps.PointStabilizer({"flow": False}, clock).stabilize(blank(), raw)
    assert list(out.columns) == list(raw.columns) and len(out) == 14 and (out["score"] == raw["score"]).all()
    assert (raw["x"] == 300.0).all()                                # the table that came in is not changed


def test_a_real_move_is_followed_without_a_big_delay():
    xs = np.r_[np.full(60, 100.0), np.linspace(100.0, 400.0, 30), np.full(60, 400.0)]       # a move of 300 px in one second
    out = run({"flow": False}, xs)
    assert abs(out[-1] - 400.0) < 3.0                              # it arrives
    assert np.max(np.abs(out[60:90] - xs[60:90])) < 80.0           # and does not lag far behind while moving


def test_the_tolerance_keeps_a_still_mouse_still_but_follows_a_big_step_at_once():
    xs = np.r_[np.full(10, 100.0), 100.0 + np.array([1, -1, 2, -2, 1, -1, 2, -2.0]), np.full(5, 100.0)]
    out = run({"flow": False, "min_cutoff": 0, "sigma_px": 5.0}, xs)
    assert np.ptp(out[10:18]) < 0.5                                 # shakes of 2 px or less do not show
    step = run({"flow": False, "min_cutoff": 0, "sigma_px": 5.0}, np.r_[np.full(5, 100.0), np.full(5, 200.0)])
    assert step[5] > 190.0                                          # a jump of 100 px is followed almost at once


def test_no_mouse_means_none_and_a_long_gap_starts_again():
    clock = Clock()
    st = ps.PointStabilizer({"flow": False}, clock)
    assert st.stabilize(blank(), None) is None
    clock.tick()
    st.stabilize(blank(), table(100, 100))
    clock.t += 5.0                                                  # 5 seconds with nothing
    clock.tick()
    out = st.stabilize(blank(), table(500, 100))
    assert abs(float(out["x"].iloc[0]) - 500.0) < 1e-6              # no pull towards the old place


def test_optical_flow_pulls_a_wrong_jump_of_the_model_back_towards_the_followed_point():
    clock = Clock()
    st = ps.PointStabilizer({"alpha": 0.2, "min_cutoff": 0, "sigma_px": 0}, clock)
    img = np.random.default_rng(0).integers(0, 255, (570, 1024, 3)).astype(np.uint8)        # texture, so that flow can follow a point
    for _ in range(3):
        clock.tick()
        st.stabilize(img, table(400, 300))
    clock.tick()
    out = st.stabilize(img, table(440, 300))                       # the model jumps 40 px, the picture did not move
    assert float(out["x"].iloc[0]) < 420.0                          # flow says "still": 0.2 * 440 + 0.8 * 400 = 408


def test_a_missing_keypoint_stays_missing():
    clock = Clock()
    st = ps.PointStabilizer({"flow": False}, clock)
    part = table(100, 100).iloc[:10]
    clock.tick()
    out = st.stabilize(blank(), part)
    assert len(out) == 10


def test_it_is_off_unless_the_settings_ask_for_it(monkeypatch, tmp_path):
    from live_tracker import pose_backends as pb

    class Fake:
        pass

    (tmp_path / "models/pose").mkdir(parents=True)
    pd.DataFrame([{"mouse": m, "keypoint": k, "frame": 0, "x": 1.0, "y": 1.0, "score": 0.9} for m in MICE for k in KEYPOINTS]).to_parquet(tmp_path / "p.parquet")
    base = {"backend": "stub", "poses": "p.parquet"}
    (tmp_path / pm.DEFAULT_CONFIG).write_text(json.dumps(base))
    fn, _ = pm.load(tmp_path, env={})
    assert not hasattr(fn, "stabilizer")
    (tmp_path / pm.DEFAULT_CONFIG).write_text(json.dumps({**base, "stabilize": {"flow": False}}))
    out, reason = pm.load(tmp_path, env={})
    assert reason is None and out(blank()) is not None
