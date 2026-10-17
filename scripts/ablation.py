"""Ablation: basic features vs basic + ScopeScore-inspired features.

Question: do the motion-engine features (features.py) help a classifier,
compared with plain pose geometry alone? An honest "no clear difference" is a
valid answer and gets reported as such.

Fair comparison rules, enforced here where they can be checked:
  - both feature sets cover exactly the same session x frame rows
  - the SAME fit function, split and seed are used for both (the fit function
    owns the classifier type and the tuning budget, so give both the same)
  - sessions come from the split manifest and must not overlap between splits
  - the model is compared on VALIDATION; the test set is evaluated once, ever
    (a lock file stops a second run)
  - the score table returned by the fit function must cover exactly the
    sessions of the requested split and contain no label columns

This script does not train anything itself. It plugs into Igor's code:
  fit_fn(features, labels, manifest, split, seed) -> scores
      Train on manifest["train"] only, choose hyperparameters on
      manifest["validation"] only, return INTERFACES.md section 2 scores
      (session, frame, attack, ...) for the sessions of `split`.
  evaluate_fn(scores, labels) -> dict of float metrics
      e.g. {"macro_f1": ..., "attack_f1": ..., "attack_event_recall": ...}.
      Called on all frames of the split, then on each session alone.

ASSUMED manifest layout (change load_manifest when Igor's file lands):
  {"seed": 0, "train": [...], "validation": [...], "test": [...]}   (session IDs)

Usage:
  python scripts/ablation.py --basic basic.csv --scope scope.csv --labels labels.csv \\
      --manifest splits/split_manifest.json --fit live_tracker.classify:fit_scores \\
      --evaluate live_tracker.evaluate:summary_metrics --primary macro_f1
  add --final once, at the very end, to also evaluate the test set.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KEYS = ["session", "frame"]
NON_FEATURE = {"session", "frame", "time_s", "tracking_ok", "mouse", "label"}
SPLITS = ("train", "validation", "test")
LABEL_LIKE = {"label", "labels", "annotation", "annotator"}


# --- inputs ---------------------------------------------------------------------------

def load_manifest(path_or_dict):
    m = path_or_dict if isinstance(path_or_dict, dict) else json.loads(Path(path_or_dict).read_text())
    missing = [s for s in SPLITS if s not in m]
    if missing:
        raise ValueError(f"manifest missing split(s): {missing}")
    seen = {}
    for split in SPLITS:
        for s in m[split]:
            if s in seen:
                raise ValueError(f"session {s!r} is in both {seen[s]} and {split}")
            seen[s] = split
    return m


def build_feature_sets(basic, scope, basic_name="A basic", scope_name="B basic+ScopeScore"):
    """Return {name: table}: A = basic columns, B = basic + the motion-engine columns.

    Both inputs are frame-level tables keyed by session, frame. Columns that
    appear in both are taken from `basic`.
    """
    for name, t in (("basic", basic), ("scope", scope)):
        if t.duplicated(KEYS).any():
            raise ValueError(f"{name} table has duplicate session/frame rows")
    extra = [c for c in scope.columns if c not in NON_FEATURE and c not in basic.columns]
    if not extra:
        raise ValueError("scope table adds no columns beyond the basic features")
    merged = basic.merge(scope[KEYS + extra], on=KEYS, how="inner")
    if len(merged) != len(basic) or len(merged) != len(scope):
        raise ValueError("basic and scope tables cover different session/frame rows")
    return {basic_name: basic.reset_index(drop=True), scope_name: merged.reset_index(drop=True)}


def _check_same_frames(feature_sets):
    ref_name, ref = next(iter(feature_sets.items()))
    ref_keys = ref[KEYS].sort_values(KEYS).reset_index(drop=True)
    for name, t in feature_sets.items():
        if not t[KEYS].sort_values(KEYS).reset_index(drop=True).equals(ref_keys):
            raise ValueError(f"feature set {name!r} covers different frames than {ref_name!r}")


def _check_scores(scores, sessions, name):
    if set(scores["session"].unique()) != set(sessions):
        raise ValueError(f"{name}: scores must cover exactly the sessions of the split")
    leaked = LABEL_LIKE & set(scores.columns)
    if leaked:
        raise ValueError(f"{name}: score table contains label column(s) {sorted(leaked)}")


# --- the comparison -----------------------------------------------------------------------

def run_ablation(feature_sets, labels, manifest, fit_fn, evaluate_fn, split="validation", seed=0):
    """Fit and evaluate every feature set the same way.

    Returns (overall, by_session): one row per feature set, and one row per
    feature set x session.
    """
    manifest = load_manifest(manifest)
    _check_same_frames(feature_sets)
    sessions = list(manifest[split])
    lab = labels[labels["session"].isin(sessions)]
    overall, by_session = [], []
    for name, feats in feature_sets.items():
        scores = fit_fn(feats, labels, manifest, split, seed)
        _check_scores(scores, sessions, name)
        overall.append({"feature_set": name, "split": split, **evaluate_fn(scores, lab)})
        for s in sessions:
            m = evaluate_fn(scores[scores["session"] == s], lab[lab["session"] == s])
            by_session.append({"feature_set": name, "split": split, "session": s, **m})
    return pd.DataFrame(overall), pd.DataFrame(by_session)


def run_final_test(feature_sets, labels, manifest, fit_fn, evaluate_fn, lock_path, seed=0):
    """Evaluate the held-out test sessions. Works once; a lock file blocks reruns."""
    lock = Path(lock_path)
    if lock.exists():
        raise RuntimeError(f"test set already evaluated ({lock.read_text().strip()[:120]}); "
                           "it is reported once. Delete the lock only if you start over, "
                           "and say so in the report.")
    result = run_ablation(feature_sets, labels, manifest, fit_fn, evaluate_fn, "test", seed)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"evaluated_at": datetime.now(timezone.utc).isoformat(),
                                "feature_sets": list(feature_sets), "seed": seed}))
    return result


def paired_difference(by_session, set_a, set_b, metric, n_boot=5000, seed=0, min_sessions=5):
    """Mean per-session difference (b - a) with a bootstrap 95 % interval over sessions.

    Verdict is "b better" / "b worse" only if the interval excludes 0;
    otherwise "no clear difference". Few sessions give wide intervals, so
    read the verdict together with n_sessions.
    """
    wide = by_session.pivot(index="session", columns="feature_set", values=metric)
    d = (wide[set_b] - wide[set_a]).dropna().to_numpy(float)
    out = {"metric": metric, "a": set_a, "b": set_b, "n_sessions": len(d)}
    if len(d) < min_sessions:
        return {**out, "mean_diff": float(d.mean()) if len(d) else np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "verdict": f"too few sessions (< {min_sessions}) to judge"}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    verdict = "b better" if lo > 0 else "b worse" if hi < 0 else "no clear difference"
    return {**out, "mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "verdict": verdict}


def to_markdown(df, digits=3):
    """Small markdown table (no extra dependency) to paste into the report."""
    cols = list(df.columns)
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# --- command line -----------------------------------------------------------------------

def _resolve(spec):
    module, _, func = spec.partition(":")
    if not func:
        raise SystemExit(f"expected module:function, got {spec!r}")
    return getattr(importlib.import_module(module), func)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--basic", required=True, help="CSV: session, frame + basic feature columns")
    ap.add_argument("--scope", required=True, help="CSV: compute_features() output")
    ap.add_argument("--labels", required=True, help="CSV: session, frame, time_s, label")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fit", required=True, help="module:function (see module docstring)")
    ap.add_argument("--evaluate", required=True, help="module:function (see module docstring)")
    ap.add_argument("--primary", default="macro_f1", help="metric for the paired comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="also evaluate TEST, once")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = ap.parse_args(argv)

    sets = build_feature_sets(pd.read_csv(args.basic), pd.read_csv(args.scope))
    labels = pd.read_csv(args.labels)
    fit_fn, eval_fn = _resolve(args.fit), _resolve(args.evaluate)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    overall, per = run_ablation(sets, labels, args.manifest, fit_fn, eval_fn, "validation", args.seed)
    overall.to_csv(out / "ablation_validation.csv", index=False)
    per.to_csv(out / "ablation_validation_by_session.csv", index=False)
    print("VALIDATION\n" + to_markdown(overall))
    a, b = list(sets)
    diff = paired_difference(per, a, b, args.primary, seed=args.seed)
    print(f"\n{args.primary}: {b} minus {a} = {diff['mean_diff']:.3f} "
          f"(95% CI {diff['ci_low']:.3f} to {diff['ci_high']:.3f}, "
          f"{diff['n_sessions']} sessions): {diff['verdict']}")

    if args.final:
        test, _ = run_final_test(sets, labels, args.manifest, fit_fn, eval_fn,
                                 out / "ablation_test.lock", args.seed)
        test.to_csv(out / "ablation_test.csv", index=False)
        print("\nTEST (reported once)\n" + to_markdown(test))


if __name__ == "__main__":
    main()
