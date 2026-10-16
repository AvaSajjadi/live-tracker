"""Tests for wide/long conversion and the failure injector (new file)."""

import numpy as np
import pandas as pd
import pytest

from live_tracker.formats import inject_tracking_failures, wide_to_long
from live_tracker.quality import flag_tracking, tracking_loss_events


def wide(n=60):
    t = np.arange(n, dtype=float)
    return pd.DataFrame({"x_a": t, "y_a": t * 0 + 10, "x_b": 100 - t, "y_b": t * 0 + 20})


def test_wide_to_long_shape_and_time():
    long = wide_to_long(wide(10), fps=10)
    assert len(long) == 20
    assert set(long["mouse"]) == {"a", "b"}
    b = long[long.mouse == "b"]
    assert b["time_s"].iloc[3] == pytest.approx(0.3)
    assert b["x"].iloc[3] == 97


def test_wide_to_long_missing_column():
    with pytest.raises(ValueError):
        wide_to_long(wide().drop(columns="y_b"))


def test_injected_failures_are_detected():
    bad = inject_tracking_failures(wide(120), fps=30, gap_s=1.0, freeze_s=1.0)
    flags = flag_tracking(wide_to_long(bad, fps=30), frozen_min_frames=15)
    assert flags[flags.mouse == "b"]["missing"].sum() == 30
    assert flags[flags.mouse == "a"]["frozen"].sum() == 30
    ev = tracking_loss_events(flags, min_duration_s=0.5)
    assert len(ev) == 1 and ev[0]["mouse"] == "b"


def test_injector_does_not_change_input():
    w = wide()
    inject_tracking_failures(w)
    assert not w.isna().any().any()
