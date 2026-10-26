"""Tests for tracking-quality flags (new file)."""

import numpy as np
import pandas as pd
import pytest

from live_tracker.quality import (flag_tracking, interpolate_short_gaps,
                                  window_quality, tracking_loss_events)

FPS = 30.0


def make_track(x, y=None, session="s1", mouse=0):
    n = len(x)
    if y is None:
        y = np.full(n, 50.0)
    return pd.DataFrame({"session": session, "mouse": mouse,
                         "time_s": np.arange(n) / FPS,
                         "x": np.asarray(x, float), "y": np.asarray(y, float)})


def jitter(n, seed=0):
    return 100 + np.random.default_rng(seed).normal(0, 0.5, n)


def test_clean_track_is_all_valid():
    out = flag_tracking(make_track(jitter(60)))
    assert out["valid"].all()


def test_missing_is_flagged_not_called_still():
    x = jitter(30)
    x[10:15] = np.nan
    out = flag_tracking(make_track(x))
    assert out["missing"].sum() == 5
    assert not out["frozen"].any()  # missing must not look like immobility


def test_frozen_coordinates_flagged():
    x = jitter(60)
    x[20:40] = 123.0  # tracker stuck for 20 frames
    out = flag_tracking(make_track(x), frozen_min_frames=15)
    assert out.loc[20:39, "frozen"].all()
    assert not out.loc[:19, "frozen"].any()


def test_short_freeze_not_flagged():
    x = jitter(60)
    x[20:25] = 123.0
    out = flag_tracking(make_track(x), frozen_min_frames=15)
    assert not out["frozen"].any()


def test_jump_flagged():
    x = jitter(30)
    x[15] = 900.0  # teleport
    out = flag_tracking(make_track(x), max_speed_px_s=2000)
    assert out.loc[15, "jump"]


def test_out_of_frame_flagged():
    x = jitter(10)
    x[3] = -5
    out = flag_tracking(make_track(x), frame_size=(1024, 570))
    assert out.loc[3, "out_of_frame"] and not out.loc[3, "valid"]


def test_low_confidence_flagged_only_when_configured():
    t = make_track(jitter(10))
    t["score"] = 0.9
    t.loc[4, "score"] = 0.05
    off = flag_tracking(t)
    assert not off["low_conf"].any() and off["valid"].all()
    on = flag_tracking(t, score_col="score", min_score=0.2)
    assert on["low_conf"].sum() == 1 and on.loc[4, "low_conf"] and not on.loc[4, "valid"]


def test_low_confidence_frames_are_not_interpolated():
    t = make_track(jitter(10))
    t["score"] = 0.9
    t.loc[4, "score"] = 0.05
    out = interpolate_short_gaps(flag_tracking(t, score_col="score", min_score=0.2))
    assert not out["imputed"].any()


def test_short_gap_interpolated_and_marked():
    x = np.arange(20, dtype=float)
    x[5:7] = np.nan
    out = interpolate_short_gaps(make_track(x), max_gap_frames=3)
    assert np.allclose(out.loc[5:6, "x"], [5.0, 6.0])
    assert out.loc[5:6, "imputed"].all()
    assert out["imputed"].sum() == 2


def test_long_gap_not_interpolated():
    x = np.arange(30, dtype=float)
    x[5:15] = np.nan
    out = interpolate_short_gaps(make_track(x), max_gap_frames=3)
    assert out.loc[5:14, "x"].isna().all()
    assert not out["imputed"].any()


def test_gap_at_edge_not_extrapolated():
    x = np.arange(10, dtype=float)
    x[:2] = np.nan
    out = interpolate_short_gaps(make_track(x), max_gap_frames=3)
    assert out.loc[:1, "x"].isna().all()


def test_gaps_do_not_cross_sessions():
    a = make_track(np.r_[np.arange(5.0), np.nan], session="a")
    b = make_track(np.r_[np.nan, np.arange(5.0)], session="b")
    out = interpolate_short_gaps(pd.concat([a, b], ignore_index=True))
    assert not out["imputed"].any()  # both gaps sit at a session edge


def test_window_quality_warns():
    x = jitter(60)
    x[30:60] = np.nan  # second window fully missing
    w = window_quality(flag_tracking(make_track(x)), window_s=1.0)
    assert list(w["warning"]) == [False, True]
    assert w.loc[1, "frac_missing"] == 1.0


def test_window_quality_requires_flags():
    with pytest.raises(ValueError):
        window_quality(make_track(jitter(10)))


def test_tracking_loss_event_reported():
    x = jitter(90)
    x[30:60] = np.nan  # 1 s lost
    ev = tracking_loss_events(flag_tracking(make_track(x)), min_duration_s=0.5)
    assert len(ev) == 1
    assert ev[0]["start_s"] == pytest.approx(1.0)
    assert ev[0]["end_s"] == pytest.approx(2.0)
