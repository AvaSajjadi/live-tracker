"""Compare window-quality policies A and B on ground-truth attack bouts (counts only).

A: usable_fraction >= 0.8 and no frozen/out_of_frame run > 0.5 s in the window.
B: usable_fraction >= 0.9.
Both use the same frame-level tracking_ok (torso points). No effect is estimated.
Frozen/out_of_frame is computed here per frame (any torso point, either mouse).

    python scripts/compare_quality_policies.py data/raw/calms21/calms21_task1_train.json \
        [--splits validation train]
"""
import argparse
import json

import numpy as np

from live_tracker.calms21 import FPS, IMAGE_SIZE, load_calms21
from live_tracker.features import frame_tracking_ok
from live_tracker.quality import flag_tracking, interpolate_short_gaps
from dryrun_usable_windows import BASELINE_S, GAP_S, MIN_BOUT_S, TORSO, label_bouts, usable

PRE_GRID = (1.0, 2.0, 5.0)
POLICIES = {"A(0.8+veto)": (0.8, True), "B(0.9)": (0.9, False)}
VETO_S = 0.5


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def window_ok(ok, bad, t0, t1, min_frac, veto):
    f = usable(ok, t0, t1)
    if f is None:
        return None
    if f < min_frac:
        return False
    if veto:
        i0, i1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        if longest_run(bad[i0:i1]) / FPS > VETO_S:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--manifest", default="splits/split_manifest.json")
    ap.add_argument("--splits", nargs="+", default=["validation"])
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    sessions = [s for k in a.splits for s in man[k]]
    poses, labels = load_calms21(a.path, sessions=sessions)
    flags = interpolate_short_gaps(flag_tracking(poses, frame_size=IMAGE_SIZE))
    tor = flags[flags["keypoint"].isin(TORSO)]
    ok_df = frame_tracking_ok(flags, keypoints=TORSO)
    fz = (tor.assign(b=tor["frozen"] | tor["out_of_frame"])
             .groupby(["session", "frame"])["b"].any().rename("bad").reset_index())
    bouts = label_bouts(labels)
    print(f"{len(sessions)} sessions ({'+'.join(a.splits)}), ground-truth attack bouts >= {MIN_BOUT_S}s")
    for pre_s in PRE_GRID:
        for name, (frac, veto) in POLICIES.items():
            n_all = used = overlap = edge = bad_pre = bad_base = 0
            sess_used = set()
            for sid, bl in bouts.items():
                o = ok_df[ok_df.session == sid].sort_values("frame")
                ok = o["tracking_ok"].to_numpy()
                b = fz[fz.session == sid].set_index("frame").reindex(o["frame"])["bad"].fillna(False).to_numpy(bool)
                for k, (s, e) in enumerate(bl):
                    if e - s < MIN_BOUT_S:
                        continue
                    n_all += 1
                    p0, b1 = s - pre_s, s - pre_s - GAP_S
                    if any(e2 > p0 and s2 < s for j, (s2, e2) in enumerate(bl) if j != k):
                        overlap += 1; continue
                    rp = window_ok(ok, b, p0, s, frac, veto)
                    rb = window_ok(ok, b, b1 - BASELINE_S, b1, frac, veto)
                    if rp is None or rb is None: edge += 1
                    elif not rp: bad_pre += 1
                    elif not rb: bad_base += 1
                    else: used += 1; sess_used.add(sid)
            print(f"pre_s={pre_s:>3} {name:<12} bouts {n_all} | overlap {overlap}, edge {edge}, "
                  f"bad pre {bad_pre}, bad base {bad_base} -> usable {used} in {len(sess_used)} sessions")


if __name__ == "__main__":
    main()
