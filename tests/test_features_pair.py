"""Tests for body center, two-mouse features, tracking_ok and warnings (new file).

Keypoint names here (SYN_*) are made up for a synthetic fixture; they are not
the CalMS21 keypoints, which we take from the dataset readme once we have it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from live_tracker.features import (DYADIC_COLS, FEATURE_COLS, FeatureParams, body_center,
                                   compute_features, dyadic_features, frame_tracking_ok,
                                   individual_features, tracking_warning_events)
from live_tracker.formats import inject_tracking_failures, wide_to_long
from live_tracker.quality import flag_tracking, interpolate_short_gaps

FPS = 30.0
RAW = FeatureParams(smooth=None)
TORSO = ("SYN_neck", "SYN_hip_l", "SYN_hip_r")
# offsets from the true body center, so the torso mean is exactly the center
OFFSETS = {"SYN_nose": (30, 0), "SYN_neck": (10, 0), "SYN_hip_l": (-5, 6), "SYN_hip_r": (-5, -6)}
DEMO = Path(__file__).resolve().parents[1] / "example_pose_demo.csv"


def _mouse(cx, cy, mouse, session="SYNTHETIC_s1", keypoints=None):
    """Long table for one mouse; one point per frame unless keypoints are given."""
    n = len(cx)
    base = {"session": session, "mouse": mouse, "frame": np.arange(n), "time_s": np.arange(n) / FPS}
    if keypoints is None:
        return pd.DataFrame({**base, "x": np.asarray(cx, float), "y": np.asarray(cy, float)})
    return pd.concat([pd.DataFrame({**base, "keypoint": k, "x": np.asarray(cx, float) + OFFSETS[k][0],
                                    "y": np.asarray(cy, float) + OFFSETS[k][1]})
                      for k in keypoints], ignore_index=True)


def _pair(ax, ay, bx, by, **kw):
    return pd.concat([_mouse(ax, ay, "a", **kw), _mouse(bx, by, "b", **kw)], ignore_index=True)


# --- body center -------------------------------------------------------------

def test_body_center_is_torso_mean_not_nose():
    n = 10
    d = _mouse(np.full(n, 100.0), np.full(n, 50.0), "a", keypoints=list(OFFSETS))
    c = body_center(d, TORSO)
    assert np.allclose(c["x"], 100.0) and np.allclose(c["y"], 50.0)
    assert (c["n_keypoints"] == 3).all()


def test_body_center_uses_remaining_keypoints():
    n = 10
    d = _mouse(np.full(n, 100.0), np.full(n, 50.0), "a", keypoints=list(OFFSETS))
    d.loc[(d.keypoint == "SYN_neck") & (d.frame == 3), "x"] = np.nan
    d.loc[d.frame == 7, ["x", "y"]] = np.nan  # every keypoint gone
    c = body_center(d, TORSO).set_index("frame")
    assert c.loc[3, "n_keypoints"] == 2
    assert np.isclose(c.loc[3, "x"], 95.0)  # mean of the two hips
    assert np.isnan(c.loc[7, "x"]) and not c.loc[7, "valid"]


def test_body_center_needs_named_keypoints():
    d = _mouse(np.zeros(3), np.zeros(3), "a", keypoints=list(OFFSETS))
    with pytest.raises(ValueError, match="torso_keypoints"):
        body_center(d)
    with pytest.raises(ValueError, match="not in data"):
        body_center(d, ["SYN_tail"])


def test_individual_features_with_keypoints():
    n = 60
    cx = 60.0 * np.arange(n) / FPS
    d = _mouse(cx, np.zeros(n), "a", keypoints=list(OFFSETS))
    f = individual_features(d, RAW, torso_keypoints=TORSO)
    assert np.allclose(f["speed"].iloc[1:], 60.0)


# --- dyadic --------------------------------------------------------------------

def test_approaching_and_separating():
    n = 60
    t = np.arange(n) / FPS
    toward = dyadic_features(_pair(np.zeros(n), np.zeros(n), 300 - 30 * t, np.zeros(n)), RAW)
    away = dyadic_features(_pair(np.zeros(n), np.zeros(n), 300 + 30 * t, np.zeros(n)), RAW)
    assert np.allclose(toward["dist_rate"].iloc[1:], -30.0)
    assert np.allclose(away["dist_rate"].iloc[1:], 30.0)
    assert np.allclose(toward["rel_speed"].iloc[1:], 30.0)
    assert np.allclose(toward["dist_center"], 300 - 30 * t)


def test_close_time():
    n = 90
    near = dyadic_features(_pair(np.zeros(n), np.zeros(n), np.full(n, 20.0), np.zeros(n)), RAW)
    far = dyadic_features(_pair(np.zeros(n), np.zeros(n), np.full(n, 200.0), np.zeros(n)), RAW)
    # window counts as full once 90 % (min_coverage) is covered: frame 27
    assert near["close_time_s"].iloc[:27].isna().all()
    assert np.isclose(near["close_time_s"].iloc[27], 0.9)
    assert np.allclose(near["close_time_s"].iloc[30:], 1.0)
    assert np.allclose(far["close_time_s"].iloc[30:], 0.0)


def test_speed_correlation():
    n = 120
    t = np.arange(n) / FPS
    wobble = 20 * np.sin(2 * np.pi * 2 * t)  # speed varies, direction +x
    xa = np.cumsum(50 + wobble) / FPS
    same = dyadic_features(_pair(xa, np.zeros(n), xa, np.full(n, 100.0)), RAW)
    xb = np.cumsum(50 - wobble) / FPS
    opposite = dyadic_features(_pair(xa, np.zeros(n), xb, np.full(n, 100.0)), RAW)
    steady = dyadic_features(_pair(xa, np.zeros(n), 40 * t, np.full(n, 100.0)), RAW)
    assert np.allclose(same["speed_corr_w"].iloc[40:], 1.0)
    assert np.allclose(opposite["speed_corr_w"].iloc[40:], -1.0)
    assert steady["speed_corr_w"].isna().all()  # constant speed: no correlation


def test_dyadic_missing_mouse_gives_nan():
    n = 60
    bx = np.full(n, 100.0)
    bx[20:30] = np.nan
    f = dyadic_features(_pair(np.zeros(n), np.zeros(n), bx, np.zeros(n)), RAW)
    assert f["dist_center"].iloc[20:30].isna().all()
    assert f["dist_rate"].iloc[20:31].isna().all()
    assert np.isfinite(f["dist_center"].iloc[:20]).all()


def test_dyadic_needs_two_mice():
    with pytest.raises(ValueError, match="exactly 2 mice"):
        dyadic_features(_mouse(np.zeros(5), np.zeros(5), "a"))


def test_dyadic_is_causal():
    rng = np.random.default_rng(1)
    n, cut = 200, 110
    walk = lambda: np.cumsum(rng.normal(0, 3, n))
    ax, ay, bx, by = walk(), walk(), walk(), walk()
    bx2 = bx.copy()
    bx2[cut + 1:] += 80
    f1 = dyadic_features(_pair(ax, ay, bx, by))
    f2 = dyadic_features(_pair(ax, ay, bx2, by))
    pd.testing.assert_frame_equal(f1.iloc[:cut + 1], f2.iloc[:cut + 1])
    assert not f1.iloc[cut + 1:].equals(f2.iloc[cut + 1:])


# --- tracking_ok and warnings ----------------------------------------------------

def _flagged_keypoint_pair(n=60):
    d = _pair(np.zeros(n) + np.arange(n), np.zeros(n), 200 + np.arange(n), np.zeros(n),
              keypoints=list(OFFSETS))
    return d


def test_frame_tracking_ok_needed_keypoints_only():
    d = _flagged_keypoint_pair()
    d.loc[(d.mouse == "a") & (d.keypoint == "SYN_nose") & d.frame.between(10, 19), "x"] = np.nan
    flags = flag_tracking(d)
    torso_ok = frame_tracking_ok(flags, keypoints=TORSO)
    all_ok = frame_tracking_ok(flags)
    assert torso_ok["tracking_ok"].all()  # nose is not needed
    assert not all_ok["tracking_ok"].iloc[10:20].any()
    assert all_ok["tracking_ok"].iloc[20:].all()
    assert len(torso_ok) == 60


def test_frame_tracking_ok_missing_row_and_imputed():
    d = _pair(np.arange(60.0), np.zeros(60), 200 + np.arange(60.0), np.zeros(60))
    d.loc[(d.mouse == "b") & (d.frame == 30), "x"] = np.nan  # 1-frame gap, will be filled
    d = d[~((d.mouse == "a") & (d.frame == 40))]  # row for mouse a missing entirely
    flags = interpolate_short_gaps(flag_tracking(d))
    ok = frame_tracking_ok(flags).set_index("frame")["tracking_ok"]
    strict = frame_tracking_ok(flags, allow_imputed=False).set_index("frame")["tracking_ok"]
    assert ok[30] and not strict[30]
    assert not ok[40]
    assert ok.drop([40]).all()


def test_warnings_on_injected_failures():
    demo = pd.read_csv(DEMO)
    flags = flag_tracking(wide_to_long(inject_tracking_failures(demo), session="SYNTHETIC_demo"))
    events = tracking_warning_events(flags)
    found = {(e["evidence"]["flag"], e["evidence"]["mouse"]): (e["start_s"], e["end_s"])
             for e in events}
    assert found[("missing", "b")] == pytest.approx((5.0, 6.5))
    assert found[("frozen", "a")] == pytest.approx((2.5, 3.5))
    assert [e["event_id"] for e in events] == [f"T{i}" for i in range(1, len(events) + 1)]
    assert [e["start_s"] for e in events] == sorted(e["start_s"] for e in events)
    for e in events:
        assert e["action"] == "tracking_warning" and e["session"] == "SYNTHETIC_demo"
        assert set(e) == {"event_id", "session", "rule_id", "behavior", "start_s", "end_s",
                          "confidence", "action", "researcher_answer", "evidence"}
        json.dumps(e)  # must be writable to the JSONL event log


def test_warnings_ignore_unneeded_keypoints():
    d = _flagged_keypoint_pair()
    d.loc[(d.mouse == "a") & (d.keypoint == "SYN_nose") & d.frame.between(10, 49), "x"] = np.nan
    flags = flag_tracking(d)
    assert tracking_warning_events(flags, keypoints=TORSO) == []
    assert len(tracking_warning_events(flags)) == 1


def test_no_warnings_on_clean_data():
    flags = flag_tracking(wide_to_long(pd.read_csv(DEMO), session="SYNTHETIC_demo"))
    assert tracking_warning_events(flags) == []


# --- whole table -------------------------------------------------------------------

def test_compute_features_table():
    demo = pd.read_csv(DEMO)
    long = wide_to_long(inject_tracking_failures(demo), session="SYNTHETIC_demo")
    table = compute_features(long)
    assert len(table) == len(demo)
    expected = (["session", "frame", "time_s", "tracking_ok"]
                + [f"{c}_{m}" for m in ("a", "b") for c in FEATURE_COLS] + list(DYADIC_COLS))
    assert list(table.columns) == expected
    gap = table[table["time_s"].between(5.0, 6.49)]
    assert not gap["tracking_ok"].any()
    assert gap["speed_b"].isna().all() and gap["dist_center"].isna().all()
    assert table["tracking_ok"].iloc[:60].all()


def test_compute_features_with_keypoints():
    d = _flagged_keypoint_pair()
    table = compute_features(d, RAW, torso_keypoints=TORSO)
    assert len(table) == 60
    assert np.allclose(table["speed_a"].iloc[1:], 30.0)
    assert np.allclose(table["dist_center"], 200.0)
