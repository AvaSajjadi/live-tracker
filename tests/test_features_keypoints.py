"""Tests for the keypoint features: nose/tail distances and facing angle (new file).

Mice are built from the real CalMS21 keypoint names with exact geometry, so every
expected value can be worked out by hand. Synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from live_tracker.calms21 import TORSO_KEYPOINTS
from live_tracker.features import (KEYPOINT_COLS, FeatureParams, compute_features, keypoint_features)

FPS = 30.0
RAW = FeatureParams(smooth=None)  # exact numbers, no filter lag
# body-frame offsets (x forward); the torso mean (neck, hips) is exactly the body center
BODY = {"nose": (30.0, 0.0), "neck": (10.0, 0.0), "left_hip": (-5.0, 6.0),
        "right_hip": (-5.0, -6.0), "tail_base": (-25.0, 0.0)}


def _mouse(cx, cy, theta, mouse, session="SYNTHETIC_k1", drop=(), frames=None):
    """Long table for one mouse whose body points along angle theta (rad)."""
    cx, cy = np.broadcast_to(np.asarray(cx, float), np.shape(cx)), np.asarray(cy, float)
    n = len(cx)
    theta = np.broadcast_to(np.asarray(theta, float), (n,))
    c, s = np.cos(theta), np.sin(theta)
    rows = []
    for k, (ox, oy) in BODY.items():
        if k in drop:
            continue
        rows.append(pd.DataFrame({"session": session, "mouse": mouse, "keypoint": k, "frame": np.arange(n),
                                  "time_s": np.arange(n) / FPS,
                                  "x": cx + c * ox - s * oy, "y": cy + s * ox + c * oy}))
    return pd.concat(rows, ignore_index=True)


def _pair(a, b, **kw):
    """a, b: dicts with cx, cy, theta (and optional drop)."""
    return pd.concat([_mouse(a["cx"], a["cy"], a["theta"], "a", drop=a.get("drop", ()), **kw),
                      _mouse(b["cx"], b["cy"], b["theta"], "b", drop=b.get("drop", ()), **kw)],
                     ignore_index=True)


def _features(long_df, **kw):
    return keypoint_features(long_df, RAW, TORSO_KEYPOINTS, **kw)


N = 10
Z = np.zeros(N)


def test_columns_and_one_row_per_frame():
    f = _features(_pair({"cx": Z + 100, "cy": Z + 100, "theta": 0.0}, {"cx": Z + 300, "cy": Z + 100, "theta": np.pi}))
    assert list(f.columns) == ["session", "frame", "time_s", *KEYPOINT_COLS]
    assert len(f) == N and f["frame"].tolist() == list(range(N))


def test_known_distances():
    # A at x=100 facing +x, B at x=300 facing -x: noses are 200 - 30 - 30 apart
    f = _features(_pair({"cx": Z + 100, "cy": Z, "theta": 0.0}, {"cx": Z + 300, "cy": Z, "theta": np.pi}))
    assert f["dist_nose_nose"].tolist() == pytest.approx([140.0] * N)
    # nose of A (130) to tail base of B (300 + 25 = 325)
    assert f["dist_nose_tail_ab"].tolist() == pytest.approx([195.0] * N)
    # nose of B (270) to tail base of A (100 - 25 = 75)
    assert f["dist_nose_tail_ba"].tolist() == pytest.approx([195.0] * N)


def test_facing_each_other_is_zero():
    f = _features(_pair({"cx": Z + 100, "cy": Z, "theta": 0.0}, {"cx": Z + 300, "cy": Z, "theta": np.pi}))
    assert f["facing_ab"].tolist() == pytest.approx([0.0] * N, abs=1e-9)
    assert f["facing_ba"].tolist() == pytest.approx([0.0] * N, abs=1e-9)


def test_facing_away_is_pi():
    f = _features(_pair({"cx": Z + 100, "cy": Z, "theta": np.pi}, {"cx": Z + 300, "cy": Z, "theta": 0.0}))
    assert f["facing_ab"].tolist() == pytest.approx([np.pi] * N, abs=1e-9)
    assert f["facing_ba"].tolist() == pytest.approx([np.pi] * N, abs=1e-9)


def test_perpendicular_is_half_pi_and_one_sided():
    # A points up (+y) while B is to its right; B points at A
    f = _features(_pair({"cx": Z + 100, "cy": Z, "theta": np.pi / 2}, {"cx": Z + 300, "cy": Z, "theta": np.pi}))
    assert f["facing_ab"].tolist() == pytest.approx([np.pi / 2] * N)
    assert f["facing_ba"].tolist() == pytest.approx([0.0] * N, abs=1e-9)


def test_facing_has_no_sign_left_or_right():
    up = _features(_pair({"cx": Z, "cy": Z, "theta": np.pi / 4}, {"cx": Z + 200, "cy": Z, "theta": 0.0}))
    down = _features(_pair({"cx": Z, "cy": Z, "theta": -np.pi / 4}, {"cx": Z + 200, "cy": Z, "theta": 0.0}))
    assert up["facing_ab"].tolist() == pytest.approx(down["facing_ab"].tolist())
    assert up["facing_ab"].tolist() == pytest.approx([np.pi / 4] * N)


def test_missing_nose_gives_nan_not_zero_and_only_where_needed():
    a = {"cx": Z + 100, "cy": Z, "theta": 0.0, "drop": ("nose",)}
    f = _features(_pair(a, {"cx": Z + 300, "cy": Z, "theta": np.pi}))
    for col in ("dist_nose_nose", "dist_nose_tail_ab", "facing_ab"):
        assert f[col].isna().all(), col
    # B's nose and A's tail base are still there
    assert f["dist_nose_tail_ba"].tolist() == pytest.approx([195.0] * N)
    assert f["facing_ba"].tolist() == pytest.approx([0.0] * N, abs=1e-9)


def test_unusable_frame_is_nan_in_that_frame_only():
    long_df = _pair({"cx": Z + 100, "cy": Z, "theta": 0.0}, {"cx": Z + 300, "cy": Z, "theta": np.pi})
    # 4 missing frames: longer than the short-gap fill (3 frames), so they stay NaN
    hit = (long_df["mouse"] == "a") & (long_df["keypoint"] == "nose") & long_df["frame"].between(3, 6)
    long_df.loc[hit, ["x", "y"]] = np.nan
    f = _features(long_df)
    bad = [3, 4, 5, 6]
    assert f.loc[bad, "dist_nose_nose"].isna().all() and f.loc[bad, "facing_ab"].isna().all()
    assert f["dist_nose_nose"].drop(index=bad).notna().all()
    assert f["facing_ba"].notna().all()          # B's own nose is fine


def test_one_frame_hole_is_filled_like_the_other_features():
    long_df = _pair({"cx": Z + 100, "cy": Z, "theta": 0.0}, {"cx": Z + 300, "cy": Z, "theta": np.pi})
    hit = (long_df["mouse"] == "a") & (long_df["keypoint"] == "nose") & (long_df["frame"] == 4)
    long_df.loc[hit, ["x", "y"]] = np.nan
    assert _features(long_df)["dist_nose_nose"].notna().all()   # same short-gap rule as compute_features


def test_nose_on_tail_base_has_no_direction():
    long_df = _pair({"cx": Z + 100, "cy": Z, "theta": 0.0}, {"cx": Z + 300, "cy": Z, "theta": np.pi})
    both = (long_df["mouse"] == "a") & long_df["keypoint"].isin(["nose", "tail_base"])
    long_df.loc[both, ["x", "y"]] = [100.0, 0.0]
    f = _features(long_df)
    assert f["facing_ab"].isna().all()          # axis is 0 px long: no direction, not "0 rad"
    assert f["facing_ba"].notna().all()


def test_causal_change_after_t_leaves_t_unchanged():
    rng = np.random.default_rng(0)
    n = 60
    ax, ay = 100 + np.cumsum(rng.normal(0, 2, n)), 100 + np.cumsum(rng.normal(0, 2, n))
    bx, by = 300 + np.cumsum(rng.normal(0, 2, n)), 100 + np.cumsum(rng.normal(0, 2, n))
    th_a, th_b = np.cumsum(rng.normal(0, 0.1, n)), np.pi + np.cumsum(rng.normal(0, 0.1, n))
    base = _pair({"cx": ax, "cy": ay, "theta": th_a}, {"cx": bx, "cy": by, "theta": th_b})
    changed = base.copy()
    late = changed["frame"] > 30
    changed.loc[late, ["x", "y"]] = changed.loc[late, ["x", "y"]] + rng.normal(0, 50, (int(late.sum()), 2))
    for params in (RAW, FeatureParams()):       # also with the causal One-Euro smoother
        f0 = keypoint_features(base, params, TORSO_KEYPOINTS).iloc[:31]
        f1 = keypoint_features(changed, params, TORSO_KEYPOINTS).iloc[:31]
        pd.testing.assert_frame_equal(f0, f1)


def test_two_sessions_do_not_carry_over():
    s1 = _pair({"cx": Z + 100, "cy": Z, "theta": 0.0}, {"cx": Z + 300, "cy": Z, "theta": np.pi}, session="SYNTHETIC_s1")
    s2 = _pair({"cx": Z + 100, "cy": Z, "theta": np.pi}, {"cx": Z + 140, "cy": Z, "theta": 0.0}, session="SYNTHETIC_s2")
    both = _features(pd.concat([s1, s2], ignore_index=True))
    alone = _features(s2)
    got = both[both["session"] == "SYNTHETIC_s2"].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, alone)
    assert both["session"].tolist() == ["SYNTHETIC_s1"] * N + ["SYNTHETIC_s2"] * N


def test_torso_keypoints_are_required():
    with pytest.raises(ValueError, match="torso_keypoints"):
        keypoint_features(_pair({"cx": Z, "cy": Z, "theta": 0.0}, {"cx": Z + 100, "cy": Z, "theta": np.pi}), RAW)


def test_missing_keypoint_column_is_an_error():
    long_df = _pair({"cx": Z, "cy": Z, "theta": 0.0}, {"cx": Z + 100, "cy": Z, "theta": np.pi})
    with pytest.raises(ValueError, match="not in data"):
        keypoint_features(long_df[long_df["keypoint"] != "tail_base"], RAW, TORSO_KEYPOINTS)


def test_compute_features_table_is_unchanged():
    long_df = _pair({"cx": Z + 100, "cy": Z, "theta": 0.0}, {"cx": Z + 300, "cy": Z, "theta": np.pi})
    table = compute_features(long_df, RAW, torso_keypoints=TORSO_KEYPOINTS)
    assert not set(KEYPOINT_COLS) & set(table.columns), "the §5 table (and model B's inputs) must not change"
