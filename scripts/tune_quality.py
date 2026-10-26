"""Tune tracking-quality and smoothing settings on real CalMS21 data (task 8).

Tunes on TRAIN sessions, then re-checks on VALIDATION. Never reads test sessions.
Torso keypoints only (neck, left_hip, right_hip), as in the other dry runs.

  frozen   how long can identical coordinates run in real data?
  jump     where does the step-size tail turn into tracker errors (low score)?
  smooth   One-Euro (min_cutoff, beta): jitter removed vs lag added

    python scripts/tune_quality.py data/raw/calms21/calms21_task1_train.json
"""
import argparse
import json

import numpy as np
import pandas as pd

from live_tracker.calms21 import FPS, load_calms21
from live_tracker.filters import one_euro_filter

TORSO = ("neck", "left_hip", "right_hip")
JUMP_PX_FRAME = 100.0          # chosen rule, see jump_table()
GRID = [(mc, b) for mc in (0.5, 1.0, 2.0, 4.0) for b in (0.0, 0.01, 0.05)]


def tracks(poses):
    t = poses[poses.keypoint.isin(TORSO)].sort_values(["session", "mouse", "keypoint", "frame"])
    for _, g in t.groupby(["session", "mouse", "keypoint"], sort=False):
        yield g


def frozen_table(poses):
    runs = []
    for g in tracks(poses):
        d = np.hypot(np.diff(g.x.to_numpy(float)), np.diff(g.y.to_numpy(float)))
        z = np.r_[0, (d == 0).astype(int), 0]
        e = np.flatnonzero(np.diff(z))
        runs += list(e[1::2] - e[::2])
    r = np.array(runs)
    return {f">={L}": int((r >= L).sum()) for L in (3, 5, 10, 15, 30)}


def jump_table(poses):
    S, SC = [], []
    for g in tracks(poses):
        sc = g.score.to_numpy(float)
        S.append(np.hypot(np.diff(g.x.to_numpy(float)), np.diff(g.y.to_numpy(float))))
        SC.append(np.minimum(sc[1:], sc[:-1]))
    s, sc = np.concatenate(S), np.concatenate(SC)
    above = s >= JUMP_PX_FRAME
    return {"frac_steps_flagged": float(above.mean()),
            "median_score_flagged": float(np.median(sc[above])),
            "median_score_rest": float(np.median(sc[~above]))}


def smooth_table(poses, n_sessions=10):
    keep = sorted(poses.session.unique())[:n_sessions]
    rows = {g: [] for g in GRID}
    for g in tracks(poses[poses.session.isin(keep)]):
        if g.keypoint.iloc[0] != "neck":
            continue
        t = g.time_s.to_numpy(float)
        raw = np.c_[g.x, g.y].astype(float)
        speed = np.r_[0, np.hypot(*np.diff(raw, axis=0).T)]
        slow = speed < 3.0                 # px/frame: animal essentially still
        for (mc, b) in GRID:
            sm = np.c_[one_euro_filter(raw[:, 0], t, min_cutoff=mc, beta=b),
                       one_euro_filter(raw[:, 1], t, min_cutoff=mc, beta=b)]
            sstep = np.r_[0, np.hypot(*np.diff(sm, axis=0).T)]
            jit = np.median(sstep[slow]) / max(np.median(speed[slow]), 1e-9)
            errs = [np.nanmean(np.hypot(*(raw[:len(raw) - k] - sm[k:]).T)) for k in range(0, 8)]
            rows[(mc, b)].append((jit, int(np.argmin(errs))))
    out = [{"min_cutoff": mc, "beta": b,
            "jitter_ratio": float(np.mean([r[0] for r in v])),
            "lag_frames": float(np.mean([r[1] for r in v]))} for (mc, b), v in rows.items()]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--manifest", default="splits/split_manifest.json")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    for split in ("train", "validation"):
        poses, _ = load_calms21(a.path, sessions=man[split])
        print(f"== {split} ({len(man[split])} sessions)")
        print("identical-coordinate runs (frames):", frozen_table(poses))
        print(f"jump >= {JUMP_PX_FRAME} px/frame:", jump_table(poses))
        if split == "train":
            sm = smooth_table(poses)
            print(sm.round(3).to_string(index=False))
            sm.to_csv("outputs/tuning_smoothing.csv", index=False)


if __name__ == "__main__":
    main()
