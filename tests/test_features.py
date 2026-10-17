"""Tests for the causal per-mouse features (new file)."""

import numpy as np
import pandas as pd
import pytest

from live_tracker.features import FEATURE_COLS, FeatureParams, individual_features, to_wide
from live_tracker.quality import flag_tracking, interpolate_short_gaps

FPS = 30.0
RAW = FeatureParams(smooth=None)  # exact numbers, no filter lag


def _track(x, y, session="s1", mouse="a", times=None):
    n = len(x)
    t = np.arange(n) / FPS if times is None else np.asarray(times, float)
    return pd.DataFrame({"session": session, "mouse": mouse, "frame": np.arange(n),
                         "time_s": t, "x": np.asarray(x, float), "y": np.asarray(y, float)})


def test_constant_position_is_still_not_missing():
    f = individual_features(_track(np.full(90, 50.0), np.full(90, 20.0)), RAW)
    assert np.allclose(f["speed"].iloc[1:], 0.0)
    assert np.isnan(f["speed"].iloc[0])  # no previous frame yet
    assert f["heading"].isna().all()  # direction of a still mouse is noise
    # straightness is undefined (NaN) for a still mouse: no path to be straight along
    assert f["straightness_w"].isna().all()
    assert np.allclose(f["idle_frac_w"].iloc[30:], 1.0)
    assert f["in_pause"].iloc[30:].all()


def test_straight_line_known_speed():
    n = 120
    x = 10 + 60.0 * np.arange(n) / FPS  # 60 px/s along +x
    f = individual_features(_track(x, np.full(n, 5.0)), RAW)
    assert np.allclose(f["speed"].iloc[1:], 60.0)
    assert np.allclose(f["accel"].iloc[2:], 0.0, atol=1e-6)
    assert np.allclose(f["heading"].iloc[1:], 0.0)
    assert np.allclose(f["turn_rate"].iloc[2:], 0.0, atol=1e-6)
    late = f.iloc[40:]
    assert np.allclose(late["straightness_w"], 1.0)
    assert np.allclose(late["path_len_w"], 60.0)  # 1 s window
    assert np.allclose(late["idle_frac_w"], 0.0)
    assert not late["in_pause"].any()


def test_straight_line_with_default_smoothing_converges():
    n = 150
    x = 60.0 * np.arange(n) / FPS
    f = individual_features(_track(x, np.zeros(n)))  # one_euro by default
    assert np.allclose(f["speed"].iloc[-10:], 60.0, rtol=0.02)


def test_turn_rate_across_pi_is_small():
    # moving in -x with a tiny zigzag in y: heading flips between about +pi and -pi
    n = 60
    x = -60.0 * np.arange(n) / FPS
    y = np.where(np.arange(n) % 2 == 0, 0.0, 0.01)
    f = individual_features(_track(x, y), RAW)
    assert np.nanmax(np.abs(f["turn_rate"])) < 1.0  # not ~2*pi*30


def test_circle_turn_rate():
    n, r, omega = 150, 50.0, 1.0  # 1 rad/s
    t = np.arange(n) / FPS
    f = individual_features(_track(r * np.cos(omega * t), r * np.sin(omega * t)), RAW)
    assert np.allclose(f["turn_rate"].iloc[2:], omega, rtol=1e-3)
    assert np.allclose(f["speed"].iloc[1:], r * omega, rtol=1e-3)


def test_missing_points_give_nan_not_zero():
    x = np.full(90, 50.0)
    x[40:50] = np.nan  # still mouse, then tracking lost
    f = individual_features(_track(x, np.full(90, 50.0)), RAW)
    gap = f.iloc[40:51]  # frame 50 has no valid previous frame
    assert gap["speed"].isna().all()
    assert gap["in_pause"].isna().all()
    assert not f["point_ok"].iloc[40:50].any()
    # window features wait until the window is mostly good tracking again
    assert f["idle_frac_w"].iloc[40:77].isna().all()
    assert np.allclose(f["idle_frac_w"].iloc[80:], 1.0)


def test_frozen_tracker_is_not_a_pause():
    x = 60.0 * np.arange(90) / FPS
    x[30:60] = x[30]  # tracker stuck for 1 s
    flags = flag_tracking(_track(x, np.zeros(90)), frozen_min_frames=15)
    f = individual_features(flags, RAW)
    assert f["speed"].iloc[30:61].isna().all()
    assert not f["in_pause"].iloc[30:61].any()


def test_short_gap_fill_is_marked_imputed():
    n = 90
    x = 60.0 * np.arange(n) / FPS
    x[50:52] = np.nan
    d = interpolate_short_gaps(flag_tracking(_track(x, np.zeros(n))), max_gap_frames=3)
    f = individual_features(d, RAW)
    assert np.allclose(f["speed"].iloc[50:53], 60.0)  # filled, so speed exists
    assert f["uses_imputed"].iloc[50:53].all()
    assert not f["uses_imputed"].iloc[:50].any()
    assert f["uses_imputed"].iloc[60:80].all()  # imputed point still in the 1 s window
    assert not f["uses_imputed"].iloc[85:].any()


def test_irregular_timestamps_use_real_dt():
    t = np.cumsum(np.r_[0, np.where(np.arange(59) % 3 == 0, 2 / FPS, 1 / FPS)])
    f = individual_features(_track(40.0 * t, np.zeros(60), times=t), RAW)
    assert np.allclose(f["speed"].iloc[1:], 40.0)


def test_long_time_jump_breaks_speed():
    t = np.r_[np.arange(30) / FPS, 5.0 + np.arange(30) / FPS]  # 4 s with no frames
    f = individual_features(_track(10.0 * t, np.zeros(60), times=t), RAW)
    assert np.isnan(f["speed"].iloc[30])


def test_sessions_do_not_carry_over():
    n = 90
    a = _track(60.0 * np.arange(n) / FPS, np.zeros(n), session="s1")
    b = _track(1000 + np.zeros(n), np.full(n, 500.0), session="s2")
    both = individual_features(pd.concat([a, b], ignore_index=True))
    alone = individual_features(b)
    s2 = both[both["session"] == "s2"].reset_index(drop=True)
    pd.testing.assert_frame_equal(s2, alone)
    assert np.isnan(s2["speed"].iloc[0])


@pytest.mark.parametrize("params", [FeatureParams(), RAW, FeatureParams(smooth="ema", smooth_kwargs={})])
def test_features_are_causal(params):
    rng = np.random.default_rng(0)
    n, cut = 200, 120
    x = np.cumsum(rng.normal(0, 3, n))
    y = np.cumsum(rng.normal(0, 3, n))
    x2, y2 = x.copy(), y.copy()
    x2[cut + 1:] += rng.normal(0, 50, n - cut - 1)  # change the future only
    y2[cut + 1:] = np.nan
    f1 = individual_features(_track(x, y), params)
    f2 = individual_features(_track(x2, y2), params)
    pd.testing.assert_frame_equal(f1.iloc[:cut + 1], f2.iloc[:cut + 1])
    assert not f1.iloc[cut + 1:].equals(f2.iloc[cut + 1:])


def test_pause_known_only_after_min_duration():
    n = 120
    x = np.r_[60.0 * np.arange(60) / FPS, np.full(n - 60, 60.0 * 59 / FPS)]
    f = individual_features(_track(x, np.zeros(n)), RAW)  # still from frame 59 on
    # 0.5 s after the last moving frame (59) is frame 74
    assert not f["in_pause"].iloc[:74].any()
    assert f["in_pause"].iloc[74:].all()


def test_several_keypoints_rejected_for_now():
    d = pd.concat([_track(np.zeros(5), np.zeros(5)).assign(keypoint=k) for k in ("nose", "neck")])
    with pytest.raises(ValueError, match="body center"):
        individual_features(d)


def test_to_wide_layout():
    n = 40
    d = pd.concat([_track(np.arange(n), np.zeros(n), mouse="a"),
                   _track(np.zeros(n), np.arange(n), mouse="b")], ignore_index=True)
    w = to_wide(individual_features(d, RAW))
    assert len(w) == n
    for c in FEATURE_COLS:
        assert f"{c}_a" in w.columns and f"{c}_b" in w.columns
    assert np.allclose(w["speed_b"].iloc[1:], 30.0)
