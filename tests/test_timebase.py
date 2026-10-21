"""Tests for the 30 fps seconds <-> frames rule (timebase.py, Contract v1
shared surface item 2): Ava's final rule frame(t) = floor(t * 30 + 0.5),
round half up at both window boundaries. New file. SYNTHETIC data only."""

import math
import random

import numpy as np
import pytest

from live_tracker import timebase as tb
from live_tracker.fake_scores import Segment, TrackingLoss, build_scores
from live_tracker.monitor import run_monitor
from live_tracker.rules import Rule


def ava(t):
    """The rule exactly as written, in float64 (what Ava's side computes)."""
    return int(np.floor(np.float64(t) * 30 + 0.5))


# ---- the rule ----

def test_exact_frame_multiples():
    assert tb.boundary_frame(0.0) == 0 and tb.boundary_frame(-0.0) == 0
    assert tb.window_frames(0.0, 1.0) == (0, 30)
    assert tb.window_frames(1.0, 2.0) == (30, 60)
    assert tb.window_frames(0.1, 0.2) == (3, 6)          # decimal literals that are frame times
    assert tb.window_frames(0.7, 1.3) == (21, 39)
    assert tb.window_frames(5.0, 10.0) == (150, 300)


@pytest.mark.parametrize("t, frame", [(0.5 / 30, 1), (1.5 / 30, 2), (0.05, 2), (2.5 / 30, 3), (1.05, 32),
                                      (60.5 / 30, 61), (10.05, 302), (1e6 + 0.05, 30_000_002)])
def test_exact_half_frame_times_round_up(t, frame):
    assert tb.boundary_frame(t) == frame == ava(t)


@pytest.mark.parametrize("t, frame", [(0.01, 0), (0.0166, 0), (0.017, 1), (0.04, 1), (1.02, 31), (1.0166, 30),
                                      (2.98, 89), (3.3333, 100)])
def test_nonaligned_times_round_to_the_nearest_frame(t, frame):
    assert tb.boundary_frame(t) == frame == ava(t)


def test_the_rule_is_exactly_floor_30t_plus_half_in_float64():
    rng = np.random.default_rng(0)
    for t in np.concatenate([rng.uniform(0, 100, 20000), rng.uniform(0, 1e6, 5000),
                             (np.arange(0, 3000) + 0.5) / 30, np.arange(0, 3000) / 30]):
        assert tb.boundary_frame(float(t)) == ava(t) == math.floor(float(t) * 30 + 0.5)


def test_round_trip_of_every_frame_timestamp():
    for f in range(0, 200_000):
        assert tb.boundary_frame(f / 30) == f
    rng = random.Random(0)
    for f in rng.sample(range(200_000, 10_000_000), 5000):
        assert tb.boundary_frame(f / 30) == f
        assert tb.boundary_frame(tb.frame_time_s(f)) == f


def test_timestamps_with_representation_error_map_to_their_frame():
    period = 1.0 / 30
    for k in range(1, 100_000):
        assert tb.boundary_frame(0.0 + k * period) == k           # the monitor's end_s form
        assert tb.boundary_frame(float(f"{k / 30:.6f}")) == k     # text with only 6 decimals
    t = 7 / 30
    assert tb.boundary_frame(math.nextafter(t, 1)) == 7 and tb.boundary_frame(math.nextafter(t, 0)) == 7


def test_boundary_frame_is_monotone():
    xs = sorted({float(x) for x in np.linspace(0, 20, 50_001)} |
                {(f + d) / 30 for f in range(0, 600) for d in (0.0, 0.5, 0.4999999, 0.5000001)})
    values = [tb.boundary_frame(x) for x in xs]
    assert values == sorted(values)


def test_adjacent_windows_partition_without_duplicates_or_gaps():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(5, 5000))
        cuts = [0.0, n / 30] + list(np.round(rng.uniform(0, n / 30, 6), int(rng.integers(1, 8))))
        cuts += [(int(rng.integers(0, n)) + 0.5) / 30 for _ in range(3)]          # exact half-frame cuts
        cuts = sorted(set(cuts))
        covered = []
        for a, b in zip(cuts, cuts[1:]):
            try:
                first, stop = tb.window_frames(a, b, n_frames=n)
            except tb.WindowError:                          # a sliver: both ends round to the same frame
                assert tb.boundary_frame(a) == tb.boundary_frame(b)
                continue
            covered.extend(range(first, stop))
        assert covered == list(range(n))


def test_windows_at_half_frame_boundaries():
    assert tb.window_frames(0.05, 0.1) == (2, 3)            # 1.5 -> 2 at the start, 3.0 -> 3 at the end
    assert tb.window_frames(0.0, 0.05) == (0, 2)            # frames 0 and 1: 1.5 rounds up at the end too
    assert tb.window_frames(1.5 / 30, 2.5 / 30) == (2, 3)
    assert tb.window_frames(0.0, 0.5 / 30) == (0, 1)        # half a frame is enough to include frame 0


def test_zero_frame_and_very_short_windows():
    assert tb.window_frames(2.0, 2.0 + 1 / 30) == (60, 61)  # exactly one frame
    assert tb.window_frames(0.0, 0.017) == (0, 1)           # just over half a frame
    assert tb.window_frames(0.01, 0.02) == (0, 1)           # 0.3 -> 0 and 0.6 -> 1: frame 0
    for t0, t1 in [(0.0, 1e-6), (0.0, 0.01), (0.02, 0.03), (2.0, 2.0 + 0.4 / 30), (0.6 / 30, 1.4 / 30)]:
        with pytest.raises(tb.WindowError, match="contains no frame") as err:
            tb.window_frames(t0, t1)                        # both ends round to the same frame
        assert err.value.code == "bad_window"


@pytest.mark.parametrize("t0, t1, msg", [
    (1.0, 1.0, "must be after"), (2.0, 1.0, "must be after"), (-0.1, 1.0, "before the session start"),
    (float("nan"), 1.0, "finite"), (0.0, float("inf"), "finite"), (True, 1.0, "number"), ("0", 1.0, "number"),
    (None, 1.0, "number"), (0.0, 1e307, "too large")])
def test_invalid_or_reversed_windows(t0, t1, msg):
    with pytest.raises(tb.WindowError, match=msg) as err:
        tb.window_frames(t0, t1)
    assert err.value.code == "bad_window"


def test_session_boundaries():
    n = 600                                                  # 20 s
    assert tb.window_frames(19.0, 20.0, n_frames=n) == (570, 600)   # the session end itself is allowed
    assert tb.window_frames(19.0, 20.01, n_frames=n) == (570, 600)  # 600.3 rounds to 600: still inside
    with pytest.raises(tb.WindowError, match="past the session end"):
        tb.window_frames(19.0, 20.0 + 0.5 / 30, n_frames=n)         # 600.5 rounds up to 601
    with pytest.raises(tb.WindowError, match="past the session end"):
        tb.window_frames(19.0, 20.0 + 1 / 30, n_frames=n)


@pytest.mark.parametrize("n", [1, 2, 29, 30, 31, 9000, 54_000, 1_234_567])
def test_arbitrary_valid_session_lengths(n):
    end = n / 30
    assert tb.window_frames(0.0, end, n_frames=n) == (0, n)
    assert tb.window_s(0, n) == (0.0, end)
    assert tb.window_frames(*tb.window_s(n - 1, n), n_frames=n) == (n - 1, n)


def test_durations_use_the_same_rule():
    assert [tb.duration_frames(x) for x in (1, 2.0, 5.0, 1 / 30, 0.1)] == [30, 60, 150, 1, 3]
    assert tb.duration_frames(1.5 / 30) == 2 and tb.duration_frames(0.5 / 30) == 1    # halves round up
    assert tb.duration_frames(1.4 / 30) == 1 and tb.duration_frames(0.017) == 1        # 0.51 frames -> 1
    assert tb.duration_frames(0.0, allow_zero=True) == 0
    assert tb.duration_frames(0.4 / 30, allow_zero=True) == 0
    for bad in (0.0, -1.0, float("nan"), 0.4 / 30, 0.01, 1e-12):           # 0.01 s = 0.3 frames -> 0
        with pytest.raises(tb.WindowError):
            tb.duration_frames(bad)


def test_frame_time_s_matches_time_s_and_rejects_non_frames():
    scores = build_scores("SYNTHETIC_tb", 10.0)
    assert [tb.frame_time_s(int(f)) for f in scores["frame"]] == list(scores["time_s"])
    assert tb.frame_time_s(np.int64(45)) == 1.5
    for bad in (-1, 1.0, True):
        with pytest.raises(tb.WindowError):
            tb.frame_time_s(bad)


# ---- consistency with the monitor's onset / firing / exclusive end ----

def test_monitor_start_fired_and_end_map_to_its_frames():
    scores = build_scores("SYNTHETIC_tb_monitor", 30.0, [
        Segment("attack", 1.0, 3.5, 0.9, 1.0), Segment("attack", 7.1, 9.9, 0.9, 1.0),
        Segment("attack", 12.0, 18.0, 0.9, 1.0)], [TrackingLoss(15.0, 15.5)], seed=3)
    events = run_monitor(scores, [Rule("R1", "attack", 1.0, status="approved")])
    assert len(events) == 4                                   # the tracking loss splits the third bout
    for e in events:
        ev = e["evidence"]
        assert tb.boundary_frame(e["start_s"]) == ev["first_frame"]
        assert tb.boundary_frame(ev["fired_s"]) == ev["fired_frame"]
        assert tb.boundary_frame(e["end_s"]) == ev["last_frame"] + 1   # exclusive end
        first, stop = tb.window_frames(e["start_s"], e["end_s"], n_frames=len(scores))
        assert stop - first == ev["bout_frames"]
        assert first == ev["first_frame"]
        pre = tb.window_frames(e["start_s"] - 1.0, e["start_s"]) if e["start_s"] >= 1.0 else None
        if pre:                                               # a pre-bout window ends just before the onset
            assert pre[1] == ev["first_frame"] and ev["first_frame"] not in range(*pre)
