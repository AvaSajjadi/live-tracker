"""Made-up pose data for the live-scoring and runner tests (SYNTHETIC, not CalMS21)."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from live_tracker.calms21 import FPS, KEYPOINTS, MICE
from live_tracker.classify import feature_columns, session_features

SESSION = "SYNTHETIC_live_01"


def make_poses(n, seed=0, session=SESSION):
    """Two mice taking random walks in the CalMS21 image frame, 7 keypoints each."""
    rng = np.random.default_rng(seed)
    rows = []
    centers = rng.uniform(300, 700, size=(2, 2))
    for f in range(n):
        centers = np.clip(centers + rng.normal(0, 6, size=(2, 2)), 50, 950)
        for m, mouse in enumerate(MICE):
            for k, kp in enumerate(KEYPOINTS):
                x, y = centers[m] + np.array([k * 4.0, -k * 2.0]) + rng.normal(0, 0.5, 2)
                rows.append({"session": session, "mouse": mouse, "keypoint": kp, "frame": f, "time_s": f / FPS,
                             "x": float(x), "y": float(y), "score": float(rng.uniform(0.6, 1.0))})
    return pd.DataFrame(rows)


def features_of(poses, motion=True):
    basic, mot = session_features(poses)
    if not motion:
        return basic
    extra = [c for c in mot.columns if c not in basic.columns or c in ("session", "frame")]
    return basic.merge(mot[extra], on=["session", "frame"], how="inner")


def tiny_model(feats):
    """A quick classifier on made-up labels; only the equality of scores matters here."""
    cols = feature_columns(feats)
    x = feats[cols].astype("float32").to_numpy()
    rng = np.random.default_rng(1)
    y = np.array(["attack", "investigation", "mount", "other"])[rng.integers(0, 4, len(x))]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=200)).fit(x, y)
    return model, cols
