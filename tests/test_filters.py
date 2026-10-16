"""Tests for the causal filters (new file)."""

import numpy as np
import pandas as pd
import pytest

from live_tracker.filters import ema_filter, one_euro_filter, OneEuroFilter, smooth_dataframe

FPS = 30.0


def _times(n):
    return np.arange(n) / FPS


def test_constant_position_stays_constant():
    t = _times(60)
    x = np.full(60, 100.0)
    assert np.allclose(ema_filter(x, t), 100.0)
    assert np.allclose(one_euro_filter(x, t, min_cutoff=1.0, beta=0.01), 100.0)


def test_filter_is_causal():
    # changing a future value must not change earlier outputs
    t = _times(50)
    x = np.linspace(0, 50, 50)
    y = x.copy()
    y[40:] += 500
    for f in (ema_filter, one_euro_filter):
        a, b = f(x, t), f(y, t)
        assert np.allclose(a[:40], b[:40])


def test_linear_motion_followed_with_small_lag():
    t = _times(300)
    x = 60.0 * t  # 60 px/s
    out = one_euro_filter(x, t, min_cutoff=1.0, beta=0.5)
    # after warm-up the error should be small compared to how far it moved
    assert np.max(np.abs(out[100:] - x[100:])) < 5.0


def test_noise_is_reduced():
    rng = np.random.default_rng(0)
    t = _times(600)
    clean = np.full(600, 200.0)
    noisy = clean + rng.normal(0, 3, 600)
    out = ema_filter(noisy, t, tau_s=0.2)
    assert np.std(out[30:] - clean[30:]) < np.std(noisy - clean) / 2


def test_missing_values_stay_missing():
    t = _times(10)
    x = np.arange(10, dtype=float)
    x[4] = np.nan
    for f in (ema_filter, one_euro_filter):
        out = f(x, t)
        assert np.isnan(out[4])
        assert not np.isnan(out[5])


def test_resets_after_long_gap():
    t = np.concatenate([_times(30), 5.0 + _times(30)])  # 4+ s gap
    x = np.concatenate([np.zeros(30), np.full(30, 300.0)])
    out = ema_filter(x, t, tau_s=1.0, max_gap_s=0.5)
    assert out[30] == 300.0  # restarted, did not drag the old value in


def test_session_boundary_resets():
    n = 20
    df = pd.DataFrame({
        "session": ["a"] * n + ["b"] * n,
        "mouse": [0] * (2 * n),
        "time_s": np.concatenate([_times(n), _times(n)]),
        "x": np.concatenate([np.zeros(n), np.full(n, 400.0)]),
        "y": np.zeros(2 * n),
    })
    out = smooth_dataframe(df, method="ema", tau_s=1.0)
    first_b = out[out.session == "b"].iloc[0]
    assert first_b["x_smooth"] == 400.0
    assert "x" in out.columns  # raw column kept


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        ema_filter([1, 2], [0.0])
    with pytest.raises(ValueError):
        ema_filter([1, 2], [1.0, 0.0])
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0)
