"""Tests for classify.py on small synthetic data (no real CalMS21 data)."""

import json

import numpy as np
import pandas as pd
import pytest

from live_tracker.calms21 import VOCAB
from live_tracker.classify import (SCORE_COLUMNS, FitParams, check_scores, feature_columns, fit_model, fit_scores, main,
                                   score_frames)
from live_tracker.evaluate import BEHAVIORS, CLASSES

N = 200     # frames per session
MANIFEST = {"seed": 0, "train": ["tr1", "tr2", "tr3"], "validation": ["va1"], "test": ["te1"]}


def _session(sid, seed):
    """Labels in blocks of 50 frames; feature f0 tells the class apart, f1 is noise."""
    rng = np.random.default_rng(seed)
    lab = np.array(CLASSES)[(np.arange(N) // 50) % 4]
    code = np.array([CLASSES.index(c) for c in lab], float)
    feats = pd.DataFrame({"session": sid, "frame": np.arange(N), "time_s": np.arange(N) / 30,
                          "tracking_ok": True,
                          "f0": code + rng.normal(0, 0.1, N), "f1": rng.normal(0, 1, N)})
    feats["f2"] = (code == 0) + rng.normal(0, 0.1, N)     # attack vs rest
    labels = pd.DataFrame({"session": sid, "frame": np.arange(N), "time_s": np.arange(N) / 30,
                           "label": lab})
    return feats, labels


@pytest.fixture
def data():
    parts = [_session(s, i) for i, s in enumerate(["tr1", "tr2", "tr3", "va1", "te1"])]
    return (pd.concat([p[0] for p in parts], ignore_index=True),
            pd.concat([p[1] for p in parts], ignore_index=True))


def test_feature_columns_skip_keys(data):
    assert feature_columns(data[0]) == ["f0", "f1", "f2"]


def test_fit_scores_format_and_split(data):
    feats, labels = data
    s = fit_scores(feats, labels, MANIFEST, "validation", seed=0)
    assert set(s.session) == {"va1"} and len(s) == N
    assert list(s.columns) == ["session", "frame", "time_s", "tracking_ok", *BEHAVIORS]
    assert not {"label", "labels"} & set(s.columns)
    p = s[list(BEHAVIORS)].to_numpy()
    assert ((p >= 0) & (p <= 1)).all() and (p.sum(axis=1) <= 1 + 1e-5).all()


def test_learns_the_separable_signal(data):
    feats, labels = data
    s = fit_scores(feats, labels, MANIFEST, "validation")
    truth = labels[labels.session == "va1"].label.to_numpy()
    attack = s.attack.to_numpy()
    assert attack[truth == "attack"].mean() > 0.7 > attack[truth == "other"].mean()


def test_never_needs_test_labels(data):
    feats, labels = data
    no_test = labels[labels.session != "te1"]
    s = fit_scores(feats, no_test, MANIFEST, "test")
    assert set(s.session) == {"te1"} and len(s) == N


def test_tuning_picks_best_C_and_records_it(data):
    feats, labels = data
    model, info = fit_model(feats, labels, MANIFEST, params=FitParams(C_grid=(1e-4, 1.0)),
                            log=lambda *_: None)
    assert info["C"] == max(info["search"], key=lambda r: r["val_macro_f1"])["C"]
    assert info["n_train_rows"] == 3 * len(range(0, N, 3))   # every 3rd frame, 3 sessions
    assert info["feature_columns"] == ["f0", "f1", "f2"]


def test_calibration_is_on_by_default_and_recorded(data):
    feats, labels = data
    model, info = fit_model(feats, labels, MANIFEST, log=lambda *_: None)
    assert "sigmoid" in info["calibration"] and "validation" in info["calibration"]
    assert type(model).__name__ == "CalibratedClassifierCV"
    p = score_frames(model, feats, ["va1"])[list(BEHAVIORS)].to_numpy()
    assert ((p >= 0) & (p <= 1)).all()


def test_calibration_can_be_switched_off(data):
    feats, labels = data
    model, info = fit_model(feats, labels, MANIFEST, params=FitParams(calibrate=None),
                            log=lambda *_: None)
    assert not info["calibration"]
    assert type(model).__name__ == "Pipeline"


def test_calibration_keeps_the_chosen_C(data):
    feats, labels = data
    grid = FitParams(C_grid=(1e-4, 1.0))
    _, raw = fit_model(feats, labels, MANIFEST, params=FitParams(C_grid=grid.C_grid, calibrate=None),
                       log=lambda *_: None)
    _, cal = fit_model(feats, labels, MANIFEST, params=grid, log=lambda *_: None)
    assert raw["C"] == cal["C"] and raw["search"] == cal["search"]   # C is chosen before calibrating


def test_same_seed_same_scores(data):
    feats, labels = data
    a = fit_scores(feats, labels, MANIFEST, "validation", seed=1)
    b = fit_scores(feats, labels, MANIFEST, "validation", seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_nullable_boolean_features_work(data):
    feats, labels = data
    feats = feats.assign(in_pause=pd.array([True, False, pd.NA] * (len(feats) // 3)
                                           + [True] * (len(feats) % 3), dtype="boolean"))
    s = fit_scores(feats, labels, MANIFEST, "validation")
    assert s[list(BEHAVIORS)].notna().all().all()


def test_cli_features_and_train(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(0)

    def rec(n=40):
        return {"keypoints": rng.uniform(50, 500, size=(n, 2, 2, 7)).tolist(),
                "scores": rng.uniform(0.5, 1, size=(n, 2, 7)).tolist(),
                "annotations": (np.arange(n) // 10 % 4).tolist(),
                "metadata": {"vocab": dict(VOCAB)}}

    def file(split, ids):
        return {"g": {f"task1/{split}/mouse{i:03d}_task1_annotator1": rec() for i in ids}}

    tr, te = tmp_path / "train.json", tmp_path / "test.json"
    tr.write_text(json.dumps(file("train", range(1, 6))))
    te.write_text(json.dumps(file("test", [71])))
    proc, models = tmp_path / "processed", tmp_path / "models"
    assert main(["features", "--train", str(tr), "--test", str(te), "--out", str(proc)]) == 0
    basic = pd.read_parquet(proc / "basic.parquet")
    assert basic.session.nunique() == 6 and len(basic) == 6 * 40
    split = tmp_path / "split.json"
    ids = sorted(basic.session.unique())
    split.write_text(json.dumps({"seed": 0, "train": ids[:4], "validation": [ids[4]],
                                 "test": [ids[5]]}))
    for feats, name in (("basic", "a"), ("basic,motion", "b")):
        assert main(["train", "--name", name, "--features", feats, "--processed", str(proc),
                     "--split", str(split), "--models", str(models)]) == 0
        info = json.loads((models / f"{name}.json").read_text())
        assert "macro_f1" in info["validation_metrics"] and (models / f"{name}.joblib").exists()
    assert json.loads((models / "b.json").read_text())["n_features"] > \
        json.loads((models / "a.json").read_text())["n_features"]
    out = tmp_path / "scores.csv"
    assert main(["scores", "--name", "a", "--split-name", "validation", "--processed", str(proc),
                 "--split", str(split), "--models", str(models), "--out", str(out)]) == 0
    s = pd.read_csv(out)
    assert list(s.columns) == SCORE_COLUMNS and set(s.session) == {ids[4]} and len(s) == 40


def test_check_scores_rejects_bad_tables():
    s = pd.DataFrame({"session": "v", "frame": range(4), "time_s": [i / 30 for i in range(4)],
                      "attack": .1, "mount": .1, "investigation": .1, "tracking_ok": True})
    check_scores(s, ["v"])
    with pytest.raises(ValueError, match="gaps"):
        check_scores(s.drop(index=2), ["v"])
    with pytest.raises(ValueError, match="time_s"):
        check_scores(s.assign(time_s=s.time_s * 1.01), ["v"])
    with pytest.raises(ValueError, match="exactly"):
        check_scores(s, ["v", "w"])
    with pytest.raises(ValueError, match="columns"):
        check_scores(s.assign(label="other"), ["v"])
