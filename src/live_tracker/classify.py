"""Behavior classifier (model A = basic features; model B = basic + features.py).

Class-weighted logistic regression on per-frame features:
- train on manifest["train"] sessions only
- choose C on manifest["validation"] only (frame macro F1 of attack,
  investigation, mount)
- optional calibration (FitParams.calibrate="sigmoid", OFF by default): fits a sigmoid on
  manifest["validation"] with the classifier frozen. Scores are then closer to real
  probabilities, but attack scores drop and the demo attack rule must be chosen again
  (docs/model_settings.md). Validation numbers are then slightly optimistic.
- output probabilities for attack, investigation, mount (INTERFACES.md §2)

Consecutive frames are nearly identical, so training uses every
`train_stride`-th frame (default 3) to keep memory low on a laptop. Scores are
always produced for every frame.

Usage (from the repo root):
    # 1. features for all sessions, one session at a time -> data/processed/ (git-ignored)
    python -m live_tracker.classify features \\
        --train data/raw/calms21/calms21_task1_train.json \\
        --test  data/raw/calms21/calms21_task1_test.json
    # 2. train + tune model A, report validation; saves models/model_a.joblib + .json
    python -m live_tracker.classify train --name model_a --features basic
    # model B (Day 3): --features basic,motion
    # 3. per-frame scores for the agent (INTERFACES.md §2) -> outputs/ (git-ignored)
    python -m live_tracker.classify scores --name model_a --split-name validation
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .basic_features import basic_features
from .calms21 import (FPS, TORSO_KEYPOINTS, _read_json, _sessions, check_manifest, flag_quality,
                      load_calms21)
from .evaluate import BEHAVIORS, CLASSES, frame_metrics, predict_labels, summary_metrics
from .features import compute_features

KEYS = ["session", "frame"]
SCORE_COLUMNS = ["session", "frame", "time_s", "attack", "mount", "investigation", "tracking_ok"]
NON_FEATURE = {"session", "frame", "time_s", "tracking_ok", "mouse", "label"}


@dataclass
class FitParams:
    """Classifier settings. C_grid is searched on validation."""
    C_grid: tuple = (0.01, 0.1, 1.0)
    train_stride: int = 3
    max_iter: int = 300
    calibrate: str | None = None   # "sigmoid" = fit on validation after C is chosen. OFF by default: see docs/model_settings.md
    extra: dict = field(default_factory=dict)


def feature_columns(features):
    return [c for c in features.columns if c not in NON_FEATURE]


def _matrix(d, features):
    # nullable booleans (pd.NA) from features.py become NaN, then the imputer fills them
    return d[feature_columns(features)].astype("float32").to_numpy()


def _xy(features, labels, sessions, stride=1):
    d = features[features["session"].isin(sessions)]
    if stride > 1:
        d = d[d["frame"] % stride == 0]
    d = d.merge(labels[KEYS + ["label"]], on=KEYS, how="left", validate="one_to_one")
    if d["label"].isna().any():
        raise ValueError("some feature rows have no label")
    return _matrix(d, features), d["label"].to_numpy()


def _model(C, seed, max_iter):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter,
                                            random_state=seed))


def score_frames(model, features, sessions):
    """INTERFACES.md §2 table for these sessions: no label columns."""
    d = features[features["session"].isin(sessions)].sort_values(KEYS)
    proba = model.predict_proba(_matrix(d, features)).astype(float)
    out = d[[c for c in ("session", "frame", "time_s", "tracking_ok") if c in d.columns]]
    out = out.reset_index(drop=True)
    classes = list(model.classes_)
    for b in BEHAVIORS:
        out[b] = proba[:, classes.index(b)] if b in classes else 0.0
    return out


def fit_model(features, labels, manifest, seed=0, params=None, log=print):
    """Train on train, pick C on validation. Returns (model, info dict)."""
    p = params or FitParams()
    manifest = check_manifest(manifest)
    x_tr, y_tr = _xy(features, labels, manifest["train"], p.train_stride)
    x_va, y_va = _xy(features, labels, manifest["validation"])
    if not set(y_tr) >= set(CLASSES):
        log(f"warning: train labels miss {sorted(set(CLASSES) - set(y_tr))}")
    best, tried = None, []
    for C in p.C_grid:
        m = _model(C, seed, p.max_iter).fit(x_tr, y_tr)
        proba = pd.DataFrame(m.predict_proba(x_va), columns=m.classes_)
        f1 = frame_metrics(y_va, predict_labels(proba.reindex(columns=BEHAVIORS, fill_value=0)))
        macro = float(f1.set_index("class").loc[list(BEHAVIORS), "f1"].mean())
        tried.append({"C": C, "val_macro_f1": macro})
        log(f"  C={C:g}: validation frame macro F1 = {macro:.3f}")
        if best is None or macro > best[1]:
            best = (m, macro, C)
    model = best[0]
    if p.calibrate:
        # the classifier stays frozen; only a per-class sigmoid is fitted, on validation sessions
        model = CalibratedClassifierCV(FrozenEstimator(model), method=p.calibrate).fit(x_va, y_va)
    info = {"seed": seed, "C": best[2], "val_macro_f1": best[1], "search": tried,
            "calibration": p.calibrate and f"{p.calibrate}, fitted on validation sessions",
            "n_train_rows": int(len(y_tr)), "n_features": int(x_tr.shape[1]),
            "feature_columns": feature_columns(features), "fit_params": asdict(p),
            "train_sessions": len(manifest["train"]),
            "validation_sessions": len(manifest["validation"])}
    return model, info


def fit_scores(features, labels, manifest, split, seed=0):
    """fit_fn for scripts/ablation.py: train on train, tune on validation, score `split`."""
    model, _ = fit_model(features, labels, manifest, seed, log=lambda *_: None)
    return score_frames(model, features, check_manifest(manifest)[split])


# --- feature files ------------------------------------------------------------------------

def session_features(poses):
    """(basic, motion) feature tables for one or more sessions, from one flagging pass."""
    flags = flag_quality(poses)
    motion = compute_features(flags, torso_keypoints=TORSO_KEYPOINTS)
    return basic_features(flags), motion


def build_feature_files(json_paths, out_dir, log=print):
    """Write basic.parquet, motion.parquet, labels.parquet for every session.

    Reads each JSON once and processes one session at a time to keep memory low.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = {"basic": [], "motion": [], "labels": []}
    for path in json_paths:
        recs = _sessions(_read_json(path))
        for i, sid in enumerate(sorted(recs), 1):
            poses, labels = load_calms21({"g": {sid: recs[sid]}})
            basic, motion = session_features(poses)
            parts["basic"].append(basic.astype({c: np.float32 for c in basic.columns[3:]}))
            parts["motion"].append(motion)
            parts["labels"].append(labels)
            log(f"  {path.name if hasattr(path, 'name') else path} [{i}/{len(recs)}] {sid}: "
                f"{len(basic)} frames")
            del poses, basic, motion
        del recs
        gc.collect()
    for name, frames in parts.items():
        pd.concat(frames, ignore_index=True).to_parquet(out / f"{name}.parquet", index=False)
    return {name: out / f"{name}.parquet" for name in parts}


def load_feature_set(processed_dir, names):
    """'basic' -> basic table; 'basic,motion' -> both joined on session, frame."""
    d = Path(processed_dir)
    tables = [pd.read_parquet(d / f"{n}.parquet") for n in names]
    out = tables[0]
    for t in tables[1:]:
        extra = [c for c in t.columns if c not in out.columns or c in KEYS]
        out = out.merge(t[extra], on=KEYS, how="inner", validate="one_to_one")
    return out


def check_scores(scores, sessions, fps=FPS):
    """Raise unless `scores` is a clean INTERFACES.md §2 table for exactly these sessions."""
    if list(scores.columns) != SCORE_COLUMNS:
        raise ValueError(f"columns {list(scores.columns)}, expected {SCORE_COLUMNS}")
    if set(scores["session"]) != set(sessions):
        raise ValueError("scores must cover exactly the requested sessions")
    if scores[list(BEHAVIORS)].isna().any().any():
        raise ValueError("NaN scores")
    for sid, g in scores.groupby("session"):
        if not np.array_equal(g["frame"].to_numpy(), np.arange(len(g))):
            raise ValueError(f"{sid}: frames are not 0..T-1 without gaps")
        if not np.allclose(g["time_s"].to_numpy(), g["frame"].to_numpy() / fps, rtol=0, atol=1e-9):
            raise ValueError(f"{sid}: time_s is not frame / {fps:g}")
    return scores


def scores_for_split(model, info, features, tracking, manifest, split):
    """§2 score table for one split. tracking: session, frame, tracking_ok (from features.py)."""
    if feature_columns(features) != info["feature_columns"]:
        raise ValueError("feature columns differ from the ones the model was trained on")
    sessions = check_manifest(manifest)[split]
    s = score_frames(model, features, sessions).drop(columns="tracking_ok", errors="ignore")
    s = s.merge(tracking[KEYS + ["tracking_ok"]], on=KEYS, how="left", validate="one_to_one")
    s["tracking_ok"] = s["tracking_ok"].fillna(False).astype(bool)
    return check_scores(s[SCORE_COLUMNS], sessions)


# --- CLI ----------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="CalMS21 behavior classifier")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="write per-frame feature files to data/processed/")
    f.add_argument("--train", required=True)
    f.add_argument("--test", required=True)
    f.add_argument("--out", default="data/processed")
    t = sub.add_parser("train", help="train + tune on train/validation, save to models/")
    t.add_argument("--name", default="model_a")
    t.add_argument("--features", default="basic", help="comma list: basic, motion")
    t.add_argument("--processed", default="data/processed")
    t.add_argument("--split", default="splits/split_manifest.json")
    t.add_argument("--models", default="models")
    t.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("scores", help="write INTERFACES.md §2 scores for one split")
    sc.add_argument("--name", default="model_a")
    sc.add_argument("--split-name", required=True, choices=["validation", "test"])
    sc.add_argument("--processed", default="data/processed")
    sc.add_argument("--split", default="splits/split_manifest.json")
    sc.add_argument("--models", default="models")
    sc.add_argument("--out", default=None, help="default outputs/scores_<split>_<name>.csv")
    a = ap.parse_args(argv)

    if a.cmd == "features":
        files = build_feature_files([Path(a.train), Path(a.test)], a.out)
        print("wrote " + ", ".join(str(p) for p in files.values()))
        return 0

    manifest = check_manifest(json.loads(Path(a.split).read_text()))
    if a.cmd == "scores":
        info = json.loads((Path(a.models) / f"{a.name}.json").read_text())
        model = joblib.load(Path(a.models) / f"{a.name}.joblib")
        feats = load_feature_set(a.processed, info["features"].split(","))
        tracking = pd.read_parquet(Path(a.processed) / "motion.parquet",
                                   columns=KEYS + ["tracking_ok"])
        s = scores_for_split(model, info, feats, tracking, manifest, a.split_name)
        out = Path(a.out or f"outputs/scores_{a.split_name}_{a.name}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(out, index=False)
        print(f"wrote {out}: {len(s)} frames, {s.session.nunique()} {a.split_name} sessions, "
              f"tracking_ok False on {int((~s.tracking_ok).sum())} frames")
        return 0
    feats = load_feature_set(a.processed, a.features.split(","))
    labels = pd.read_parquet(Path(a.processed) / "labels.parquet")
    print(f"{a.name}: {len(feature_columns(feats))} features ({a.features})")
    model, info = fit_model(feats, labels, manifest, a.seed)
    val = summary_metrics(score_frames(model, feats, manifest["validation"]), labels)
    import sklearn
    info.update({"sklearn_version": sklearn.__version__, "name": a.name, "features": a.features, "split_manifest": a.split,
                 "validation_metrics": val})
    out = Path(a.models)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"{a.name}.joblib")
    (out / f"{a.name}.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    print(f"saved {out / a.name}.joblib (C={info['C']:g})")
    print("validation:")
    for k, v in val.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
