"""Deterministic offline demo of the Contract v1 investigation tools on
SYNTHETIC data (new file).

    python scripts/contract_v1_demo.py            # JSON record to stdout
    python scripts/contract_v1_demo.py --out d.json   # refuses to overwrite

Five made-up sessions (fake_scores plus jittered pose in the CalMS21 layout),
three 2-s attack bouts each. One InvestigationRunner (12-call budget) runs
query_events -> check_tracking_quality -> get_window_features ->
compare_windows (pre_s 2) -> the two allowed follow-ups (pre_s 1 and 5) ->
one refused extra follow-up. Every attempt is in the record.

SYNTHETIC DATA ONLY: no CalMS21, no model output, no network, no API key.
The numbers say nothing about mouse behavior; they show the tool surface.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from live_tracker import investigation_tools_v1 as v1  # noqa: E402
from live_tracker.calms21 import KEYPOINTS, _session_to_long  # noqa: E402
from live_tracker.fake_scores import Segment, build_scores  # noqa: E402

N_FRAMES = 1800
STARTS = (12.0, 25.0, 40.0)
SESSIONS = [f"SYNTHETIC_demo_v1_{i}" for i in range(5)]


def synthetic_data():
    scores, poses = [], []
    for i, sid in enumerate(SESSIONS):
        scores.append(build_scores(sid, N_FRAMES / 30, [Segment("attack", s, s + 2.0, 0.85, 0.95) for s in STARTS],
                                   seed=i))
        rng = np.random.default_rng(100 + i)
        kp = np.zeros((N_FRAMES, 2, 2, len(KEYPOINTS)))
        for m, cx in enumerate((300.0, 650.0)):
            walk = np.cumsum(rng.normal(0, 1.5, (N_FRAMES, 2)), axis=0) + [cx, 280.0]
            for k in range(len(KEYPOINTS)):
                kp[:, m, 0, k] = walk[:, 0] + 8 * k + rng.normal(0, 0.8, N_FRAMES)
                kp[:, m, 1, k] = walk[:, 1] + rng.normal(0, 0.8, N_FRAMES)
        poses.append(_session_to_long(sid, {"keypoints": kp, "scores": np.full((N_FRAMES, 2, len(KEYPOINTS)), 0.9)}))
    return v1.InvestigationData.from_synthetic(pd.concat(scores, ignore_index=True),
                                               pd.concat(poses, ignore_index=True))


def run():
    data = synthetic_data()
    r = v1.InvestigationRunner(data, "demo-contract-v1")
    q = r.call("query_events", {"session": list(data.sessions), "behavior": "attack", "min_duration_s": 1.0,
                                "threshold": 0.5})
    first = q["bouts"][0]
    r.call("check_tracking_quality", {"session": first["session"], "t0_s": first["start_s"] - 2.0,
                                      "t1_s": first["start_s"]})
    r.call("get_window_features", {"session": first["session"], "t0_s": first["start_s"] - 2.0,
                                   "t1_s": first["start_s"], "cols": ["dist_center", "rel_speed"]})
    base = {"query_run_id": q["query_run_id"], "query": q["query"], "bouts": q["bouts"],   # the query, unchanged
            "baseline_s": 5.0, "baseline_gap_s": 1.0, "feature": "dist_center"}
    for pre_s in (2.0, 1.0, 5.0, 1.0):                  # the last one is a refused third follow-up
        r.call("compare_windows", {**base, "pre_s": pre_s})
    return {"note": "SYNTHETIC DATA ONLY: the tool surface, not a finding", **r.record()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write the JSON record here (never overwritten)")
    a = ap.parse_args(argv)
    text = json.dumps(run(), indent=2, sort_keys=True)
    if a.out:
        out = Path(a.out)
        try:
            with open(out, "x", encoding="utf-8") as f:
                f.write(text + "\n")
        except FileExistsError:
            print(f"{out} exists; it is never overwritten", file=sys.stderr)
            return 2
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
