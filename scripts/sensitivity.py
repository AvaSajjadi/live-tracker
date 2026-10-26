"""Feature sensitivity: how much do the motion features change when the input
or the settings are disturbed a little?

This measures FEATURE STABILITY, not classification accuracy. Keep the two
apart in the report: a stable feature can still be useless for telling
behaviors apart, and an unstable one can still help a classifier.

Disturbances (each compared with the baseline run on the same data):
  jitter      add Gaussian noise (1, 2, 5 px) to every coordinate
  gaps        blank short runs of frames (2 frames: gets filled; 10: stays NaN)
  smoothing   other filter settings (none, EMA, One-Euro with other cutoffs)
  threshold   each placeholder threshold in FeatureParams x0.8 and x1.2

Per feature column the table reports (random disturbances are averaged over
--repeats seeds):
  n_compared         frames where baseline and disturbed value both exist
  median_abs_diff    median |disturbed - baseline| (heading: wrapped angle;
                     booleans: fraction of frames that disagree)
  p95_abs_diff       95th percentile of the same
  norm_median_diff   median_abs_diff / std of the baseline column
  frac_nan_changed   fraction of frames that became NaN or stopped being NaN

IMPORTANT: run this on train/validation sessions only, never on test sessions.
Usage:
  python scripts/sensitivity.py                       # synthetic demo data
  python scripts/sensitivity.py --long-csv sessions.csv --sessions s01 s02
  python scripts/sensitivity.py --calms21 data/raw/calms21/calms21_task1_train.json \
      --sessions task1/train/mouse005_task1_annotator1 ...      # real validation sessions
--calms21 reads the official train file, keeps only the named sessions, and refuses any
session that is not in the manifest's train or validation split (test sessions can never
be named). It uses the team's real settings: torso keypoints, image frame size, low_conf
at SUGGESTED_MIN_SCORE.
Output: outputs/sensitivity.csv (git-ignored; copy the table into the report).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker.calms21 import IMAGE_SIZE, TORSO_KEYPOINTS, check_manifest, load_calms21  # noqa: E402
from live_tracker.features import FeatureParams, compute_features  # noqa: E402
from live_tracker.formats import wide_to_long  # noqa: E402
from live_tracker.quality import SUGGESTED_MIN_SCORE  # noqa: E402

KEY_COLS = ("session", "frame", "time_s", "tracking_ok")
THRESHOLD_FIELDS = ("max_step_s", "window_s", "heading_min_speed_px_s", "idle_speed_px_s",
                    "pause_min_duration_s", "min_path_px", "close_dist_px")
JITTER_PX = (1.0, 2.0, 5.0)
GAP_FRAMES = (2, 10)
GAP_SHARE = 0.05  # about this share of each track is blanked


# --- disturbances of the input ---------------------------------------------------

def add_jitter(long_df, sigma_px, rng):
    out = long_df.copy()
    out["x"] = out["x"] + rng.normal(0, sigma_px, len(out))
    out["y"] = out["y"] + rng.normal(0, sigma_px, len(out))
    return out


def blank_gaps(long_df, gap_frames, rng, share=GAP_SHARE):
    """Set x/y to NaN for random runs of gap_frames frames, per session x mouse."""
    out = long_df.copy()
    keys = [c for c in ("session", "mouse", "keypoint") if c in out.columns]
    for _, part in out.groupby(keys, sort=False):
        idx = part.sort_values("time_s").index.to_numpy()
        n = len(idx)
        n_runs = max(1, int(round(share * n / gap_frames)))
        for start in rng.integers(0, max(n - gap_frames, 1), n_runs):
            out.loc[idx[start:start + gap_frames], ["x", "y"]] = np.nan
    return out


# --- comparison ------------------------------------------------------------------

def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def _compare(base, pert, col):
    b, p = base[col], pert[col]
    both = (b.notna() & p.notna()).to_numpy()
    row = {"feature": col, "n_compared": int(both.sum()),
           "frac_nan_changed": float((b.isna() != p.isna()).mean())}
    if pd.api.types.is_bool_dtype(b):
        diff = (b[both].astype(bool).to_numpy() != p[both].astype(bool).to_numpy()).astype(float)
        scale = np.nan  # no meaningful std for a flag
    else:
        bv, pv = b.to_numpy(float)[both], p.to_numpy(float)[both]
        diff = np.abs(_wrap(pv - bv) if col.startswith("heading") else pv - bv)
        scale = float(np.std(bv)) if len(bv) else np.nan
    if len(diff):
        row["median_abs_diff"] = float(np.median(diff))
        row["p95_abs_diff"] = float(np.percentile(diff, 95))
    else:
        row["median_abs_diff"] = row["p95_abs_diff"] = np.nan
    row["norm_median_diff"] = row["median_abs_diff"] / scale if scale and scale > 0 else np.nan
    return row


def compare_tables(base, pert):
    """One row per feature column. Both tables must cover the same frames."""
    keys = [c for c in ("session", "frame") if c in base.columns]
    if not base[keys].reset_index(drop=True).equals(pert[keys].reset_index(drop=True)):
        raise ValueError("baseline and disturbed tables cover different frames")
    cols = [c for c in base.columns if c not in KEY_COLS]
    return pd.DataFrame([_compare(base, pert, c) for c in cols])


# --- the experiment ----------------------------------------------------------------

def smoothing_variants():
    return {
        "none": dict(smooth=None),
        "ema tau=0.05": dict(smooth="ema", smooth_kwargs={"tau_s": 0.05}),
        "ema tau=0.2": dict(smooth="ema", smooth_kwargs={"tau_s": 0.2}),
        "one_euro min_cutoff=0.5": dict(smooth_kwargs={"min_cutoff": 0.5, "beta": 0.0}),
        "one_euro min_cutoff=2": dict(smooth_kwargs={"min_cutoff": 2.0, "beta": 0.0}),
        "one_euro beta=0.05": dict(smooth_kwargs={"min_cutoff": 1.0, "beta": 0.05}),
    }


def run_sensitivity(long_df, params=None, repeats=5, seed=0, **feature_kwargs):
    """Return the sensitivity table (one row per disturbance x feature column)."""
    params = params or FeatureParams()
    valid_fields = {f.name for f in fields(FeatureParams)}
    missing = [f for f in THRESHOLD_FIELDS if f not in valid_fields]
    if missing:
        raise ValueError(f"FeatureParams has no field(s) {missing}")
    base = compute_features(long_df, params, **feature_kwargs)

    jobs = []  # (kind, setting, input table builder, params, random?)
    for s in JITTER_PX:
        jobs.append(("jitter", f"{s:g} px", lambda r, s=s: add_jitter(long_df, s, r), params, True))
    for g in GAP_FRAMES:
        jobs.append(("gaps", f"{g} frames", lambda r, g=g: blank_gaps(long_df, g, r), params, True))
    for name, change in smoothing_variants().items():
        jobs.append(("smoothing", name, lambda r: long_df, replace(params, **change), False))
    for f in THRESHOLD_FIELDS:
        for factor in (0.8, 1.2):
            jobs.append(("threshold", f"{f} x{factor}", lambda r: long_df,
                         replace(params, **{f: getattr(params, f) * factor}), False))

    rows = []
    for kind, setting, build, p, random in jobs:
        rng = np.random.default_rng(seed)
        parts = []
        for _ in range(repeats if random else 1):
            pert = compute_features(build(rng), p, **feature_kwargs)
            parts.append(compare_tables(base, pert))
        table = pd.concat(parts).groupby("feature", sort=False).mean(numeric_only=True).reset_index()
        table.insert(0, "setting", setting)
        table.insert(0, "disturbance", kind)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def load_real_sessions(json_path, sessions, manifest_path):
    """(long pose table, compute_features keyword arguments) for named train/validation sessions.

    Refuses test sessions and anything not in the manifest before the JSON is opened."""
    import json
    manifest = check_manifest(json.loads(Path(manifest_path).read_text()))
    allowed = set(manifest["train"]) | set(manifest["validation"])
    refused = sorted(set(sessions) - allowed)
    if refused:
        raise SystemExit(f"refused: not in the train/validation split (test sessions are never used here): {refused}")
    poses, _ = load_calms21(json_path, sessions=sessions)
    kwargs = dict(torso_keypoints=TORSO_KEYPOINTS, frame_size=IMAGE_SIZE, score_col="score",
                  min_score=SUGGESTED_MIN_SCORE)
    return poses, kwargs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calms21", help="official CalMS21 Task 1 train JSON; needs --sessions")
    ap.add_argument("--manifest", default=str(ROOT / "splits" / "split_manifest.json"))
    ap.add_argument("--long-csv", help="long pose table (session, mouse, frame, time_s, x, y); "
                                       "default: the synthetic demo")
    ap.add_argument("--sessions", nargs="*", help="only these sessions (train/val only!)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sensitivity.csv"))
    args = ap.parse_args(argv)

    feature_kwargs = {}
    if args.calms21:
        if not args.sessions:
            raise SystemExit("--calms21 needs --sessions (train/validation only)")
        long_df, feature_kwargs = load_real_sessions(args.calms21, args.sessions, args.manifest)
        print(f"Using REAL CalMS21 sessions {args.sessions} (not the test split).")
    elif args.long_csv:
        long_df = pd.read_csv(args.long_csv)
    else:
        long_df = wide_to_long(pd.read_csv(ROOT / "example_pose_demo.csv"), session="SYNTHETIC_demo")
        print("Using SYNTHETIC demo data (not CalMS21): numbers only show the script works.")
    if args.sessions:
        long_df = long_df[long_df["session"].isin(args.sessions)]
        if long_df.empty:
            raise SystemExit(f"no rows for sessions {args.sessions}")

    result = run_sensitivity(long_df, repeats=args.repeats, seed=args.seed, **feature_kwargs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Wrote {len(result)} rows to {out}")
    print(summarize(result).to_string(index=False))


def summarize(result):
    """Per disturbance setting: the feature that changed most, and the largest
    share of frames whose NaN status changed. Full detail is in the CSV."""
    rows = []
    for (kind, setting), part in result.groupby(["disturbance", "setting"], sort=False):
        top = part.loc[part["norm_median_diff"].idxmax()] if part["norm_median_diff"].notna().any() else None
        rows.append({"disturbance": kind, "setting": setting,
                     "most_changed_feature": None if top is None else top["feature"],
                     "its_norm_median_diff": np.nan if top is None else round(top["norm_median_diff"], 3),
                     "max_frac_nan_changed": round(part["frac_nan_changed"].max(), 3)})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    main()
