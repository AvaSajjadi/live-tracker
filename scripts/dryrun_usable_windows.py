"""Dry run: how many attack bouts have a usable pre-bout window?

Uses CalMS21 *ground-truth labels* on the validation sessions (no classifier yet),
so the numbers are an upper bound on what the agent will see. Counts only; no
effect is estimated. Rules follow docs/investigate_tool_contracts.md.

    python scripts/dryrun_usable_windows.py data/raw/calms21/calms21_task1_train.json
"""

import argparse
import json

import numpy as np

from live_tracker.calms21 import FPS, load_calms21
from live_tracker.features import compute_features

TORSO = ("neck", "left_hip", "right_hip")
MIN_USABLE = 0.8     # contract: usable_fraction for a "reliable" window
MIN_BOUT_S = 1.0     # min attack bout length for this dry run
PRE_GRID = (2.0, 5.0, 10.0)
BASELINE_S, GAP_S = 5.0, 1.0


def label_bouts(labels, behavior="attack"):
    out = {}
    for sid, d in labels.groupby("session"):
        m = (d.sort_values("frame")["label"] == behavior).to_numpy()
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        out[sid] = [(a / FPS, b / FPS) for a, b in zip(edges[::2], edges[1::2])]
    return out


def usable(ok, t0, t1):
    i0, i1 = max(int(round(t0 * FPS)), 0), int(round(t1 * FPS))
    if t0 < 0 or i1 > len(ok) or i1 <= i0:
        return None
    return float(ok[i0:i1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--manifest", default="splits/split_manifest.json")
    a = ap.parse_args()
    val = json.load(open(a.manifest))["validation"]
    poses, labels = load_calms21(a.path, sessions=val)
    feats = compute_features(poses, torso_keypoints=TORSO)
    bouts = label_bouts(labels)
    print(f"{len(val)} validation sessions")
    for pre_s in PRE_GRID:
        n_all = n_short = n_overlap = n_edge = n_bad_pre = n_bad_base = n_used = 0
        for sid, bl in bouts.items():
            ok = feats[feats.session == sid].sort_values("frame")["tracking_ok"].to_numpy()
            for k, (s, e) in enumerate(bl):
                if e - s < MIN_BOUT_S:
                    n_short += 1
                    continue
                n_all += 1
                p0, p1 = s - pre_s, s
                b1 = p0 - GAP_S
                b0 = b1 - BASELINE_S
                if any(e2 > p0 and s2 < p1 for j, (s2, e2) in enumerate(bl) if j != k):
                    n_overlap += 1
                    continue
                up, ub = usable(ok, p0, p1), usable(ok, b0, b1)
                if up is None or ub is None:
                    n_edge += 1
                elif up < MIN_USABLE:
                    n_bad_pre += 1
                elif ub < MIN_USABLE:
                    n_bad_base += 1
                else:
                    n_used += 1
        print(f"pre_s={pre_s:>4}: bouts>={MIN_BOUT_S}s {n_all} (short skipped {n_short}) | "
              f"overlap {n_overlap}, session edge {n_edge}, bad pre {n_bad_pre}, "
              f"bad baseline {n_bad_base} -> usable {n_used}")


if __name__ == "__main__":
    main()
