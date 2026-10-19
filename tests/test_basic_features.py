"""Tests for the basic per-frame geometry features (model A baseline)."""

import numpy as np
import pandas as pd
import pytest

from live_tracker.basic_features import basic_feature_names, basic_features
from live_tracker.calms21 import (FPS, IMAGE_SIZE, KEYPOINTS, MICE, TORSO_KEYPOINTS, flag_quality,
                                  load_calms21)
from live_tracker.quality import SUGGESTED_MIN_SCORE

T = 6


def _calms21(seed=0, n=T):
    rng = np.random.default_rng(seed)
    rec = {"keypoints": rng.uniform(0, 500, size=(n, 2, 2, 7)).tolist(),
           "scores": rng.uniform(0, 1, size=(n, 2, 7)).tolist(),
           "annotations": [3] * n}
    return {"g": {"task1/train/mouse001_task1_annotator1": rec}}


@pytest.fixture
def poses():
    return load_calms21(_calms21())[0]


def _row(feat, frame):
    return feat[feat.frame == frame].iloc[0]


def test_one_row_per_frame_and_columns(poses):
    f = basic_features(poses)
    assert len(f) == T and not f.duplicated(["session", "frame"]).any()
    assert list(f.columns) == ["session", "frame", "time_s"] + basic_feature_names()
    assert len(basic_feature_names()) == 2 * 7 * 2 + 7 * 7
    assert f.time_s.iloc[-1] == pytest.approx((T - 1) / FPS)


def test_values_by_hand(poses):
    raw = np.asarray(_calms21()["g"]["task1/train/mouse001_task1_annotator1"]["keypoints"])
    f = _row(basic_features(poses), 3)
    torso = [KEYPOINTS.index(k) for k in TORSO_KEYPOINTS]
    # mouse b (index 1), nose (index 0): nose minus mean of the torso points
    center_b = raw[3, 1, :, torso].mean(axis=0)              # (xy,)
    assert f["rel_nose_x_b"] == pytest.approx(raw[3, 1, 0, 0] - center_b[0])
    assert f["rel_nose_y_b"] == pytest.approx(raw[3, 1, 1, 0] - center_b[1])
    # a's nose (index 0) to b's tail_base (index 6)
    d = np.hypot(raw[3, 0, 0, 0] - raw[3, 1, 0, 6], raw[3, 0, 1, 0] - raw[3, 1, 1, 6])
    assert f["pd_nose__tail_base"] == pytest.approx(d)


def test_torso_points_average_to_zero(poses):
    f = basic_features(poses)
    for m in MICE:
        for c in ("x", "y"):
            s = sum(f[f"rel_{k}_{c}_{m}"] for k in TORSO_KEYPOINTS)
            assert np.allclose(s, 0)


def test_causal_each_row_uses_only_its_own_frame(poses):
    before = basic_features(poses)
    changed = poses.copy()
    later = changed.frame > 2
    changed.loc[later, ["x", "y"]] = changed.loc[later, ["x", "y"]] * 3 + 100
    after = basic_features(changed)
    pd.testing.assert_frame_equal(before[before.frame <= 2], after[after.frame <= 2])


def test_missing_point_gives_nan_not_zero(poses):
    p = poses.copy()
    hit = (p.frame == 2) & (p.mouse == "a") & (p.keypoint == "nose")
    p.loc[hit, ["x", "y"]] = np.nan
    f = _row(basic_features(p), 2)
    assert np.isnan(f["rel_nose_x_a"]) and np.isnan(f["pd_nose__nose"])
    assert not np.isnan(f["rel_neck_x_a"])          # other points still fine


def test_center_uses_remaining_torso_points(poses):
    p = poses.copy()
    p.loc[(p.frame == 1) & (p.mouse == "a") & (p.keypoint == "neck"), ["x", "y"]] = np.nan
    f = _row(basic_features(p), 1)
    assert f["rel_left_hip_x_a"] + f["rel_right_hip_x_a"] == pytest.approx(0)


def test_all_torso_missing_gives_nan_relative_but_keeps_distances(poses):
    p = poses.copy()
    p.loc[(p.frame == 4) & (p.mouse == "b") & p.keypoint.isin(TORSO_KEYPOINTS), ["x", "y"]] = np.nan
    f = _row(basic_features(p), 4)
    assert np.isnan(f["rel_nose_x_b"])
    assert not np.isnan(f["pd_nose__nose"])


def test_invalid_flag_masks_point(poses):
    p = poses.assign(valid=True)
    p.loc[(p.frame == 0) & (p.mouse == "b") & (p.keypoint == "nose"), "valid"] = False
    f = _row(basic_features(p), 0)
    assert np.isnan(f["rel_nose_x_b"]) and np.isnan(f["pd_tail_base__nose"])


def test_missing_rows_become_nan(poses):
    p = poses[~((poses.frame == 5) & (poses.mouse == "a") & (poses.keypoint == "nose"))]
    f = basic_features(p)
    assert len(f) == T and np.isnan(_row(f, 5)["rel_nose_x_a"])


def test_several_sessions_kept_apart():
    data = _calms21(0)
    data["g"]["task1/train/mouse002_task1_annotator1"] = _calms21(1)["g"][
        "task1/train/mouse001_task1_annotator1"]
    poses, _ = load_calms21(data)
    f = basic_features(poses)
    assert len(f) == 2 * T
    one = basic_features(poses[poses.session.str.contains("mouse002")])
    both = f[f.session.str.contains("mouse002")].reset_index(drop=True)
    pd.testing.assert_frame_equal(one, both)


def test_imputed_point_is_used(poses):
    p = poses.assign(valid=True, imputed=False)
    hit = (p.frame == 0) & (p.mouse == "a") & (p.keypoint == "nose")
    p.loc[hit, ["valid", "imputed"]] = [False, True]
    assert not np.isnan(_row(basic_features(p), 0)["rel_nose_x_a"])


def test_low_confidence_point_is_dropped(poses):
    p = poses.copy()
    p["score"] = 0.9
    hit = (p.frame == 3) & (p.mouse == "b") & (p.keypoint == "nose")
    p.loc[hit, "score"] = SUGGESTED_MIN_SCORE / 2
    flags = flag_quality(p)
    assert flags.loc[hit, "low_conf"].all()
    f = _row(basic_features(flags), 3)
    assert np.isnan(f["rel_nose_x_b"]) and not np.isnan(f["rel_neck_x_b"])


def test_rows_match_compute_features(poses):
    from live_tracker.features import compute_features
    flags = flag_quality(poses)
    feat = compute_features(flags, torso_keypoints=TORSO_KEYPOINTS)
    same = compute_features(poses, torso_keypoints=TORSO_KEYPOINTS, frame_size=IMAGE_SIZE,
                            score_col="score", min_score=SUGGESTED_MIN_SCORE)
    pd.testing.assert_frame_equal(feat, same)        # flag_quality == Ava's recommended call
    basic = basic_features(flags)
    assert basic[["session", "frame"]].equals(feat[["session", "frame"]])
    assert not set(basic_feature_names()) & set(feat.columns)   # no name clash for model B


def test_rejects_unknown_keypoint(poses):
    p = poses.copy()
    p.loc[p.index[0], "keypoint"] = "paw"
    with pytest.raises(ValueError, match="paw"):
        basic_features(p)
