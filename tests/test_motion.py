import numpy as np
from live_tracker.motion import clean_trajectory, kinematics, pair_features, movement_bouts
from live_tracker.events import threshold_events


def test_constant_velocity():
    t = np.arange(100) / 30
    p = np.column_stack([3 * t, 0 * t])
    speed = kinematics(p, fps=30)['speed']
    assert np.allclose(speed[1:], 3.0)


def test_long_gap_stays_missing():
    p = np.column_stack([np.arange(20, dtype=float)] * 2)
    p[5:12] = np.nan
    cleaned, valid = clean_trajectory(p, fps=30)
    assert not valid[8] and np.isnan(cleaned[8]).all()


def test_approach():
    a = np.zeros((10, 2))
    b = np.column_stack([np.arange(10, 0, -1), np.zeros(10)])
    d = pair_features(a, b, fps=30)
    assert np.all(d['distance_rate_px_s'][1:] < 0)


def test_bouts_and_alerts():
    speed = np.r_[np.zeros(10), np.full(12, 20.), np.zeros(10)]
    assert movement_bouts(speed, fps=10, threshold_px_s=10, min_duration_s=1) == [(10, 22)]
    p = np.r_[np.zeros(10), np.full(25, .9), np.zeros(10)]
    e = threshold_events(p, fps=10, min_duration_s=2)
    assert len(e) == 1 and e[0]['duration_s'] == 2.5
