"""How often are BOTH mice found, and how good are the points, for other confidence settings and for keeping the last points a short time?

One pass of the trained model over 8 parts of a CalMS21 video at a very low confidence (0.05). Every setting is then simulated from the same
detections: a detection counts only when its confidence is at least the setting. The reference points are the MARS points of the same video.

SELECTION RULE (written before the first run): choose the setting with the most pictures where both mice are found, but only if
  1. the median error against the MARS points is not more than 1 px worse than the current setting (conf 0.25, no hold), and
  2. wrong detections (mean error of the mouse above 40 px) do not grow by more than 2 percentage points.
Everything is from ONE video (mouse064): this is a development check, not a validation.

python scripts/mouse_found_experiment.py --video data/raw/calms21_video/mouse064_task1_annotator1.mp4 \
    --poses data/raw/calms21_video/mouse064_task1_annotator1_poses.parquet --weights models/pose/mouse_pose_best_openvino_model
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2                      # noqa: E402
import pandas as pd             # noqa: E402

from live_tracker.calms21 import IMAGE_SIZE, KEYPOINTS, MICE   # noqa: E402

CONFS = (0.25, 0.20, 0.15, 0.10, 0.05)
HOLDS = (0, 3)
WRONG_PX = 40.0


def detect(model, image, device):
    r = model.predict(image, imgsz=(384, 640), device=device, conf=0.05, verbose=False)[0]
    out = []
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
        return out
    cls, conf, xy = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.keypoints.xy.cpu().numpy()
    for i in range(len(cls)):
        out.append((int(cls[i]), float(conf[i]), xy[i]))
    return out


def pick(dets, conf_min):
    """{class: (conf, keypoints 7x2)}: the most confident detection of each class that reaches conf_min."""
    best = {}
    for c, p, xy in dets:
        if p >= conf_min and (c not in best or p > best[c][0]):
            best[c] = (p, xy)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="intel:gpu")
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "mouse_found_experiment.json"))
    a = ap.parse_args(argv)
    from ultralytics import YOLO
    model = YOLO(a.weights, task="pose")
    poses = pd.read_parquet(a.poses, columns=["mouse", "keypoint", "frame", "x", "y"])
    truth = {}
    for (f, m), part in poses.groupby(["frame", "mouse"]):
        part = part.set_index("keypoint").reindex(list(KEYPOINTS))
        truth[(int(f), m)] = part[["x", "y"]].to_numpy(float)
    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = np.linspace(300, total - a.length - 300, a.parts).astype(int)
    model.predict(np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8), imgsz=(384, 640), device=a.device, verbose=False)
    results = {(c, h): {"both": 0, "one": 0, "none": 0, "errs": [], "wrong": 0, "points": 0} for c in CONFS for h in HOLDS}
    n = 0
    for s in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s))
        dets_seq = []
        for k in range(a.length):
            ok, img = cap.read()
            if not ok:
                break
            dets_seq.append((int(s) + k, detect(model, cv2.resize(img, IMAGE_SIZE), a.device)))
        for (c, h), acc in results.items():
            last, age = {}, {}
            for f, dets in dets_seq:
                found = pick(dets, c)
                for cl in (0, 1):
                    if cl in found:
                        last[cl], age[cl] = found[cl], 0
                    elif cl in last and h and age[cl] < h:
                        age[cl] += 1
                        found[cl] = last[cl]
                    else:
                        last.pop(cl, None)
                acc["both"] += len(found) == 2
                acc["one"] += len(found) == 1
                acc["none"] += len(found) == 0
                for cl, (p, xy) in found.items():
                    ref = truth.get((f, MICE[cl]))
                    if ref is None or np.isnan(ref).any():
                        continue
                    e = np.hypot(*(xy - ref).T)
                    acc["errs"].append(float(np.median(e)))
                    acc["wrong"] += float(np.mean(e)) > WRONG_PX
                    acc["points"] += 1
        n += len(dets_seq)
    cap.release()
    rows = []
    base = None
    for (c, h), acc in results.items():
        r = {"conf": c, "hold_frames": h, "both_pct": 100 * acc["both"] / n, "one_pct": 100 * acc["one"] / n, "none_pct": 100 * acc["none"] / n,
             "median_error_px": float(np.median(acc["errs"])) if acc["errs"] else None,
             "wrong_pct": 100 * acc["wrong"] / max(1, acc["points"])}
        rows.append(r)
        if c == 0.25 and h == 0:
            base = r
    for r in rows:
        r["allowed"] = bool(r["median_error_px"] is not None and r["median_error_px"] <= base["median_error_px"] + 1.0
                            and r["wrong_pct"] <= base["wrong_pct"] + 2.0)
    print(f"{n} pictures from {a.parts} parts of the video")
    print(f"{'conf':>5} {'hold':>4} {'both%':>6} {'one%':>6} {'none%':>6} {'median px':>9} {'wrong%':>7}  allowed")
    for r in rows:
        print(f"{r['conf']:>5} {r['hold_frames']:>4} {r['both_pct']:6.1f} {r['one_pct']:6.1f} {r['none_pct']:6.1f} {r['median_error_px']:9.2f} {r['wrong_pct']:7.1f}  {'yes' if r['allowed'] else 'NO'}")
    allowed = [r for r in rows if r["allowed"]]
    best = max(allowed, key=lambda r: r["both_pct"])
    print("chosen by the rule:", best["conf"], "hold", best["hold_frames"], "->", round(best["both_pct"], 1), "% both")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"pictures": n, "rows": rows, "chosen": best}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
