"""Measure how accurate a trained mouse pose model is, in pixels, with a separate look at touching mice (new file).

For every test picture the model must find the black mouse (class 0) and the white mouse (class 1). For each mouse we take the most
confident detection of its class and compare its 7 keypoints with the hand labels (pixels of the 1024 x 570 picture).
"Touching" means the centers of the two mice are closer than TOUCH_PX (a body is about 195 px long).

    python scripts/eval_yolo_pose.py --model runs/mouse_pose/weights/best.pt --data dataset --imgsz 640 --max 600 --out eval.json

--model can also be an exported folder (for example an OpenVINO folder); use --device intel:gpu for the Iris Xe chip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEYPOINTS = ("nose", "right_ear", "left_ear", "neck", "right_side", "left_side", "tail_base")
CLASSES = ("black", "white")
TOUCH_PX = 120.0
W, H = 1024, 570


def read_labels(path):
    """{class: array (7, 2) in pixels} from a YOLO pose label file."""
    import numpy as np
    out = {}
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        v = line.split()
        if len(v) < 5 + 21:
            continue
        kp = np.array(v[5:5 + 21], dtype=float).reshape(7, 3)[:, :2] * np.array([W, H])
        out[int(v[0])] = kp
    return out


def summarize(errors, found, total):
    import numpy as np
    e = np.array(errors) if errors else np.zeros((0, 7))
    row = {"pictures": total, "found": found, "found_share": round(found / total, 4) if total else None}
    if len(e):
        row["median_error_px"] = {k: round(float(np.median(e[:, i])), 1) for i, k in enumerate(KEYPOINTS)}
        row["median_error_all_px"] = round(float(np.median(e)), 2)
        row["share_within_5px"] = round(float((e <= 5).mean()), 4)
        row["share_within_10px"] = round(float((e <= 10).mean()), 4)
    return row


def evaluate(model_path, data_dir, imgsz, max_pictures, device, conf=0.25):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    data_dir = Path(data_dir)
    names = [p for p in (data_dir / "val.txt").read_text(encoding="utf-8").split("\n") if p.strip()]
    step = max(1, len(names) // max_pictures) if max_pictures else 1
    names = names[::step][:max_pictures] if max_pictures else names
    model = YOLO(model_path, task="pose")
    kw = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if device:
        kw["device"] = device
    subsets = {"all": ([], [0, 0]), "touching": ([], [0, 0]), "apart": ([], [0, 0])}   # errors, [found, total]
    per_class = {c: ([], [0, 0]) for c in CLASSES}
    for p in names:
        img = cv2.imread(p)
        gt = read_labels(data_dir / "labels" / (Path(p).stem + ".txt"))
        if img is None or len(gt) < 2:
            continue
        centers = [gt[c].mean(axis=0) for c in (0, 1)]
        touching = float(np.hypot(*(centers[0] - centers[1]))) < TOUCH_PX
        r = model.predict(img, **kw)[0]
        kps = r.keypoints.xy.cpu().numpy() if r.keypoints is not None and len(r.boxes) else np.zeros((0, 7, 2))
        cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
        cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        for c in (0, 1):
            keys = ["all", "touching" if touching else "apart"]
            for k in keys:
                subsets[k][1][1] += 1
            per_class[CLASSES[c]][1][1] += 1
            idx = [i for i in range(len(cls)) if cls[i] == c]
            if not idx:
                continue
            best = max(idx, key=lambda i: cf[i])
            err = np.hypot(*(kps[best] - gt[c]).T)
            for k in keys:
                subsets[k][0].append(err); subsets[k][1][0] += 1
            per_class[CLASSES[c]][0].append(err); per_class[CLASSES[c]][1][0] += 1
    out = {"model": str(model_path), "imgsz": imgsz, "pictures": len(names)}
    for k, (errs, (f, t)) in subsets.items():
        out[k] = summarize(errs, f, t)
    for k, (errs, (f, t)) in per_class.items():
        out["mouse_" + k] = summarize(errs, f, t)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="the dataset folder made by mars_pose_to_yolo.py")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max", type=int, default=600, help="how many test pictures to use (spread over the whole test set)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = evaluate(a.model, a.data, a.imgsz, a.max, a.device)
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
