"""Basic pose-geometry features: the baseline (model A) for the ablation.

Plain geometry from ONE frame at a time, nothing else:
- each keypoint relative to its own mouse's body center (rel_<kp>_x_<mouse>, rel_<kp>_y_<mouse>)
- distance between every keypoint of mouse a and every keypoint of mouse b
  (pd_<kp of a>__<kp of b>)

No time windows, no smoothing, no speed, so it is causal by construction: the
row for frame t only uses frame t. The motion features (speed, heading,
windows, ...) are Ava's, in features.py; model B = these + hers.

The body center is the mean of the usable torso keypoints, the same definition
as features.body_center(). Missing or invalid points give NaN, never 0.

Units: pixels. Input: the long pose table from calms21.load_calms21(), ideally
after calms21.flag_quality() so a point counts as usable by the same rule as in
features.py (valid, or filled by a short-gap interpolation). Flag once and pass
the same table to both, so model A and model B see the same points:

    flags = flag_quality(poses)
    basic = basic_features(flags)
    motion = compute_features(flags, torso_keypoints=TORSO_KEYPOINTS)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .calms21 import KEYPOINTS, MICE, TORSO_KEYPOINTS

KEYS = ["session", "frame"]


def basic_feature_names(keypoints=KEYPOINTS, mice=MICE):
    """Column names in output order (without session, frame, time_s)."""
    rel = [f"rel_{k}_{c}_{m}" for m in mice for k in keypoints for c in ("x", "y")]
    pair = [f"pd_{ka}__{kb}" for ka in keypoints for kb in keypoints]
    return rel + pair


def _to_array(df, keypoints, mice, x_col, y_col):
    """Long table -> (index of session/frame, array (frames, mouse, keypoint, xy))."""
    d = df
    if "valid" in d.columns:
        usable = d["valid"].astype(bool)
        if "imputed" in d.columns:
            usable = usable | d["imputed"].astype(bool)
        d = d.copy()
        d.loc[~usable, [x_col, y_col]] = np.nan
    unknown = set(d["keypoint"].unique()) - set(keypoints)
    if unknown:
        raise ValueError(f"keypoints not in the schema: {sorted(unknown)}")
    if d.duplicated(KEYS + ["mouse", "keypoint"]).any():
        raise ValueError("duplicate session/frame/mouse/keypoint rows")
    wide = d.set_index(KEYS + ["mouse", "keypoint"])[[x_col, y_col]].unstack(["mouse", "keypoint"])
    cols = pd.MultiIndex.from_product([[x_col, y_col], mice, keypoints])
    wide = wide.reindex(columns=cols).sort_index()          # missing mouse/keypoint -> NaN
    arr = wide.to_numpy(float).reshape(len(wide), 2, len(mice), len(keypoints))
    return wide.index, arr.transpose(0, 2, 3, 1)             # (T, mouse, kp, xy)


def basic_features(long_df, torso_keypoints=TORSO_KEYPOINTS, keypoints=KEYPOINTS, mice=MICE,
                   x_col="x", y_col="y"):
    """Per-frame geometry features, one row per session x frame.

    Returns session, frame, time_s, then the columns of basic_feature_names().
    """
    if len(mice) != 2:
        raise ValueError("expected exactly two mice")
    missing = set(torso_keypoints) - set(keypoints)
    if missing:
        raise ValueError(f"torso keypoints not in keypoints: {sorted(missing)}")
    index, arr = _to_array(long_df, list(keypoints), list(mice), x_col, y_col)

    torso = [list(keypoints).index(k) for k in torso_keypoints]
    with warnings.catch_warnings():
        # all torso points missing -> NaN center, which is what we want
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmean(arr[:, :, torso, :], axis=2, keepdims=True)  # (T, mouse, 1, xy)
    rel = arr - center                                                  # (T, mouse, kp, xy)
    diff = arr[:, 0, :, None, :] - arr[:, 1, None, :, :]                # (T, kp_a, kp_b, xy)
    pair = np.hypot(diff[..., 0], diff[..., 1])

    values = np.concatenate([rel.reshape(len(rel), -1), pair.reshape(len(pair), -1)], axis=1)
    out = pd.DataFrame(values, columns=basic_feature_names(keypoints, mice))
    times = long_df.groupby(KEYS, sort=True)["time_s"].first()
    head = index.to_frame(index=False)
    head["time_s"] = times.reindex(index).to_numpy()
    return pd.concat([head, out], axis=1)

