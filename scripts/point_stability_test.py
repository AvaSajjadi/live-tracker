"""Noise test of the point stabilizer (src/live_tracker/point_stabilizer.py) on real mouse points.

Truth: the CalMS21 keypoints of VALIDATION sessions (data/raw/calms21_validation; the test split is not used).
Input to the stabilizer: the truth plus Gaussian noise of 1 px and 3 px (a stand-in for the jitter of a pose model).
Flow is OFF here: there are no pictures. The flow mix is checked in tests/test_point_stabilizer.py only, and must be measured
later with the trained model on real video.

PASS RULES (written before the first run, in the style of the ScopeScore acceptance rule):
  1. Accuracy:  median error against the truth is not worse than the noisy input, at both noise levels.
  2. Precision: the median error of the frame-to-frame STEP is at most 70% of the noisy input's, at 3 px noise.
  3. Lag:       in frames where the mouse moves fast (truth step of at least 3 px per frame), the median error is at most 3 px
                larger than the noisy input's, at 1 px noise.
  4. Still:     in frames where the truth moves less than 0.5 px per frame, the median step of the output is at most 50%
                of the noisy input's, at 3 px noise.
Each rule is judged on all sessions together. Nothing about these rules was changed after seeing results.

Run: python scripts/point_stability_test.py [--sessions 3] [--frames 1800] [--out outputs/point_stability.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker import calms21                      # noqa: E402
from live_tracker.point_stabilizer import PointStabilizer   # noqa: E402

BLANK = np.zeros((4, 4, 3), np.uint8)


def stabilize_session(table, settings):
    """table: long table of one session (mouse, keypoint, frame, x, y, score). Returns x, y arrays in the same row order."""
    frames = sorted(table["frame"].unique())
    clock = {"t": 0.0}
    st = PointStabilizer({**settings, "flow": False}, lambda: clock["t"])
    by_frame = {f: part.drop(columns="frame").reset_index(drop=True) for f, part in table.groupby("frame")}
    out_x, out_y = [], []
    for f in frames:
        clock["t"] = f / calms21.FPS
        res = st.stabilize(BLANK, by_frame[f])
        out_x.append(res["x"].to_numpy())
        out_y.append(res["y"].to_numpy())
    return np.array(out_x), np.array(out_y)            # frames x 14 points


def grid(table):
    """(frames, 14 points) arrays of x and y in the order of the table's first frame."""
    frames = sorted(table["frame"].unique())
    part = {f: p for f, p in table.groupby("frame")}
    return (np.array([part[f]["x"].to_numpy() for f in frames]), np.array([part[f]["y"].to_numpy() for f in frames]))


def measure(sessions, frames, settings, seed=0):
    rng = np.random.default_rng(seed)
    pooled = {s: {"err_in": [], "err_out": [], "step_in": [], "step_out": [], "fast_in": [], "fast_out": [], "still_in": [], "still_out": []}
              for s in (1.0, 3.0)}
    for path in sessions:
        poses, _ = calms21.load_calms21(path)
        poses = poses[poses["frame"] < frames].sort_values(["frame", "mouse", "keypoint"]).reset_index(drop=True)
        tx, ty = grid(poses)
        tstep = np.hypot(np.diff(tx, axis=0), np.diff(ty, axis=0))
        for sigma, acc in pooled.items():
            noisy = poses.copy()
            noisy["x"] = poses["x"] + rng.normal(0, sigma, len(poses))
            noisy["y"] = poses["y"] + rng.normal(0, sigma, len(poses))
            nx, ny = grid(noisy)
            ox, oy = stabilize_session(noisy, settings)
            e_in, e_out = np.hypot(nx - tx, ny - ty), np.hypot(ox - tx, oy - ty)
            s_in = np.abs(np.hypot(np.diff(nx, axis=0), np.diff(ny, axis=0)) - tstep)
            s_out = np.abs(np.hypot(np.diff(ox, axis=0), np.diff(oy, axis=0)) - tstep)
            raw_step_in, raw_step_out = np.hypot(np.diff(nx, axis=0), np.diff(ny, axis=0)), np.hypot(np.diff(ox, axis=0), np.diff(oy, axis=0))
            fast, still = tstep >= 3.0, tstep < 0.5
            acc["err_in"].append(e_in.ravel()); acc["err_out"].append(e_out.ravel())
            acc["step_in"].append(s_in.ravel()); acc["step_out"].append(s_out.ravel())
            acc["fast_in"].append(e_in[1:][fast]); acc["fast_out"].append(e_out[1:][fast])
            acc["still_in"].append(raw_step_in[still]); acc["still_out"].append(raw_step_out[still])
    med = lambda parts: float(np.median(np.concatenate(parts)))
    res = {}
    for sigma, a in pooled.items():
        res[f"noise_{sigma:g}px"] = {k: med(v) for k, v in a.items()}
    return res


def judge(res):
    n1, n3 = res["noise_1px"], res["noise_3px"]
    rules = [
        ("1 Accuracy: not worse than the noisy input (1 px and 3 px noise)",
         n1["err_out"] <= n1["err_in"] and n3["err_out"] <= n3["err_in"], f"{n1['err_in']:.2f} to {n1['err_out']:.2f} px, {n3['err_in']:.2f} to {n3['err_out']:.2f} px"),
        ("2 Precision: step error at most 70% of the input (3 px noise)",
         n3["step_out"] <= 0.7 * n3["step_in"], f"{n3['step_in']:.2f} to {n3['step_out']:.2f} px"),
        ("3 Lag: fast frames at most 3 px worse than the input (1 px noise)",
         n1["fast_out"] <= n1["fast_in"] + 3.0, f"{n1['fast_in']:.2f} to {n1['fast_out']:.2f} px"),
        ("4 Still: a still mouse moves at most 50% as much (3 px noise)",
         n3["still_out"] <= 0.5 * n3["still_in"], f"{n3['still_in']:.2f} to {n3['still_out']:.2f} px per frame"),
    ]
    return rules


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "point_stability.json"))
    a = ap.parse_args(argv)
    files = sorted((ROOT / "data" / "raw" / "calms21_validation").glob("*.json"))
    files = [f for f in files if f.name != "extraction_index.json"][: a.sessions]
    if not files:
        print("No validation sessions found in data/raw/calms21_validation.")
        return 2
    out = {"sessions": [f.name for f in files], "frames": a.frames, "settings": {}}
    for name, settings in (("defaults", {}), ("no tolerance", {"sigma_px": 0}), ("tolerance 2 px", {"sigma_px": 2.0})):
        res = measure(files, a.frames, settings)
        rules = judge(res)
        out[name] = {"medians": res, "rules": [{"rule": r, "pass": bool(p), "detail": d} for r, p, d in rules]}
        print(f"\n== {name} ==")
        for r, p, d in rules:
            print(f"  {'PASS' if p else 'FAIL'}  {r}   [{d}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
