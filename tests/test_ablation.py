"""Tests for scripts/ablation.py (new file).

Uses a toy fit/evaluate pair on SYNTHETIC data, only to prove the plumbing and
the fairness checks. The numbers say nothing about mice or about ScopeScore.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation", ROOT / "scripts" / "ablation.py")
abl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abl)

MANIFEST = {"seed": 0, "train": [f"SYN_t{i}" for i in range(6)],
            "validation": [f"SYN_v{i}" for i in range(6)], "test": [f"SYN_x{i}" for i in range(4)]}


def make_data(informative, n=200, seed=0):
    """basic column is noise; the extra column tracks the label iff informative."""
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("train", "validation", "test"):
        for s in MANIFEST[split]:
            y = (rng.random(n) < 0.3).astype(int)
            rows.append(pd.DataFrame({
                "session": s, "frame": np.arange(n), "time_s": np.arange(n) / 30, "label": np.where(y, "attack", "other"),
                "basic1": rng.normal(size=n) + 1.5 * y,
                "scope1": (y + rng.normal(scale=0.7, size=n)) if informative else rng.normal(size=n)}))
    all_ = pd.concat(rows, ignore_index=True)
    labels = all_[["session", "frame", "time_s", "label"]]
    basic = all_[["session", "frame", "time_s", "basic1"]]
    scope = all_[["session", "frame", "time_s", "scope1"]]
    return basic, scope, labels


def fit_fn(features, labels, manifest, split, seed):
    cols = [c for c in features.columns if c not in abl.NON_FEATURE]
    d = features.merge(labels[["session", "frame", "label"]], on=abl.KEYS)
    tr = d[d.session.isin(manifest["train"])]
    model = LogisticRegression(max_iter=200, random_state=seed).fit(tr[cols], tr["label"] == "attack")
    te = d[d.session.isin(manifest[split])]
    return pd.DataFrame({"session": te.session, "frame": te.frame,
                         "attack": model.predict_proba(te[cols])[:, 1]}).reset_index(drop=True)


def evaluate_fn(scores, labels):
    m = scores.merge(labels, on=abl.KEYS)
    pred, true = m["attack"] >= 0.5, m["label"] == "attack"
    tp = (pred & true).sum()
    denom = 2 * tp + (pred & ~true).sum() + (~pred & true).sum()
    return {"attack_f1": float(2 * tp / denom) if denom else float("nan")}


def run(informative, **kw):
    basic, scope, labels = make_data(informative)
    sets = abl.build_feature_sets(basic, scope)
    return sets, labels, abl.run_ablation(sets, labels, MANIFEST, fit_fn, evaluate_fn, **kw)


def test_informative_feature_is_detected():
    sets, labels, (overall, per) = run(informative=True)
    a, b = list(sets)
    assert overall.set_index("feature_set").loc[b, "attack_f1"] > overall.set_index("feature_set").loc[a, "attack_f1"]
    assert abl.paired_difference(per, a, b, "attack_f1")["verdict"] == "b better"
    assert len(per) == 2 * len(MANIFEST["validation"])


def test_useless_feature_gives_no_clear_difference():
    sets, _, (_, per) = run(informative=False)
    d = abl.paired_difference(per, *list(sets), "attack_f1")
    assert d["verdict"] == "no clear difference"
    assert d["ci_low"] < 0 < d["ci_high"]


def test_too_few_sessions_is_not_judged():
    per = pd.DataFrame({"feature_set": ["A", "B"] * 3, "session": ["s1", "s1", "s2", "s2", "s3", "s3"],
                        "f1": [0.1, 0.9] * 3})
    assert "too few" in abl.paired_difference(per, "A", "B", "f1")["verdict"]


def test_feature_sets_must_cover_same_frames():
    basic, scope, _ = make_data(True)
    with pytest.raises(ValueError, match="different session/frame rows"):
        abl.build_feature_sets(basic, scope.iloc[:-10])
    with pytest.raises(ValueError, match="adds no columns"):
        abl.build_feature_sets(basic, basic)


def test_manifest_overlap_is_rejected():
    bad = {**MANIFEST, "test": MANIFEST["test"] + [MANIFEST["train"][0]]}
    with pytest.raises(ValueError, match="in both"):
        abl.load_manifest(bad)
    with pytest.raises(ValueError, match="missing split"):
        abl.load_manifest({"train": [], "validation": []})


def test_scores_with_labels_or_wrong_sessions_are_rejected():
    sets, labels, _ = run(informative=True)

    def leaky(features, labels, manifest, split, seed):
        return fit_fn(features, labels, manifest, split, seed).assign(label="attack")

    def wrong_split(features, labels, manifest, split, seed):
        return fit_fn(features, labels, manifest, "test", seed)

    with pytest.raises(ValueError, match="label column"):
        abl.run_ablation(sets, labels, MANIFEST, leaky, evaluate_fn)
    with pytest.raises(ValueError, match="exactly the sessions"):
        abl.run_ablation(sets, labels, MANIFEST, wrong_split, evaluate_fn)


def test_test_set_can_only_be_evaluated_once(tmp_path):
    basic, scope, labels = make_data(True)
    sets = abl.build_feature_sets(basic, scope)
    lock = tmp_path / "test.lock"
    overall, _ = abl.run_final_test(sets, labels, MANIFEST, fit_fn, evaluate_fn, lock)
    assert set(overall.split) == {"test"} and lock.exists()
    with pytest.raises(RuntimeError, match="already evaluated"):
        abl.run_final_test(sets, labels, MANIFEST, fit_fn, evaluate_fn, lock)


def test_failed_test_run_does_not_burn_the_lock(tmp_path):
    basic, scope, labels = make_data(True)
    sets = abl.build_feature_sets(basic, scope)

    def broken(*a, **k):
        raise RuntimeError("boom")

    lock = tmp_path / "test.lock"
    with pytest.raises(RuntimeError, match="boom"):
        abl.run_final_test(sets, labels, MANIFEST, broken, evaluate_fn, lock)
    assert not lock.exists()


def test_markdown_table():
    md = abl.to_markdown(pd.DataFrame({"feature_set": ["A"], "f1": [0.12345]}))
    assert md.splitlines()[0] == "| feature_set | f1 |" and "0.123" in md
