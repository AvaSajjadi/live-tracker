"""Accuracy and precision of MARS keypoints against the MARS hand-labeled pictures (new file).

Run it with the Python of the MARS environment (it needs TensorFlow), after scripts/mars_pose_to_yolo.py made the dataset:

    %USERPROFILE%\\mars_local\\env\\Scripts\\python.exe scripts\\accuracy_mars_handlabels.py ^
        --dataset C:\\Users\\AnTiC\\fastpose_test\\dataset --models-dir %USERPROFILE%\\mars_local\\models --n 60 --out mars_accuracy.json

Truth: the hand labels (the median of 5 workers) of test pictures spread over the whole test set. Each picture is run on its own (MARS keeps the
last box when it finds no mouse; that memory is reset before every picture). It measures:
  accuracy                 median pixel error, share of keypoints within 5 and 10 px, per keypoint, per mouse, touching versus apart;
  precision (repeatability) the same pictures run twice (largest difference), and how far each keypoint moves when the picture is made
                           smaller, blurry, compressed and dimmer (the median shift), plus the accuracy on those degraded pictures.
The numbers go into scripts/accuracy_precision_report.py (--mars-json).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def degrade(img):
    """Like a webcam that films a screen: smaller, soft, compressed, dimmer."""
    import cv2
    small = cv2.resize(img, (580, 323), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (1024, 570), interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    img = cv2.convertScaleAbs(img, alpha=0.8, beta=10)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])[1], cv2.IMREAD_COLOR)


def summarize(errors, touching, with_parts):
    """errors: array (pictures, 2 mice, 7 keypoints) in px."""
    e = np.asarray(errors)
    out = {"median_all": round(float(np.median(e)), 2), "within_5": round(float((e <= 5).mean()), 4),
           "within_10": round(float((e <= 10).mean()), 4)}
    t = np.asarray(touching, bool)
    out["median_touching"] = round(float(np.median(e[t])), 2) if t.any() else None
    out["median_apart"] = round(float(np.median(e[~t])), 2) if (~t).any() else None
    if with_parts:
        names = ("nose", "right_ear", "left_ear", "neck", "right_side", "left_side", "tail_base")
        out["per_keypoint"] = {n: round(float(np.median(e[:, :, i])), 1) for i, n in enumerate(names)}
        out["per_mouse"] = {"black": round(float(np.median(e[:, 0])), 2), "white": round(float(np.median(e[:, 1])), 2)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--n", type=int, default=60, help="how many test pictures (spread over the test set)")
    ap.add_argument("--repeat", type=int, default=10, help="how many of them are run twice")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import cv2
    mp, evp = load_module("mars_pose_offline"), load_module("eval_yolo_pose")
    data = Path(a.dataset)
    names = [p for p in (data / "val.txt").read_text(encoding="utf-8").split("\n") if p.strip()]
    step = max(1, len(names) // a.n)
    names = names[::step][:a.n]
    mars = mp.Mars(a.models_dir)

    def run(img):
        mars.prev = [(list(mp.PREV_BOX), 1e-4), (list(mp.PREV_BOX), 1e-4)]       # each picture on its own
        res = mars.run([img])[0]
        return np.array([[(x, y) for x, y, _ in res[k][0]] for k in (0, 1)])    # (2 mice, 7, 2)

    clean_err, degr_err, shifts, touching, first_runs, repeat_diffs = [], [], [], [], [], []
    for i, p in enumerate(names):
        img = cv2.imread(p)
        gt = evp.read_labels(data / "labels" / (Path(p).stem + ".txt"))
        if img is None or len(gt) < 2:
            continue
        truth = np.stack([gt[0], gt[1]])
        k_clean = run(img)
        k_degr = run(degrade(img))
        clean_err.append(np.hypot(*(k_clean - truth).transpose(2, 0, 1)))
        degr_err.append(np.hypot(*(k_degr - truth).transpose(2, 0, 1)))
        shifts.append(np.hypot(*(k_degr - k_clean).transpose(2, 0, 1)))
        touching.append(float(np.hypot(*(truth[0].mean(axis=0) - truth[1].mean(axis=0)))) < evp.TOUCH_PX)
        if i < a.repeat:
            repeat_diffs.append(float(np.abs(run(img) - k_clean).max()))
        print(f"{i + 1}/{len(names)}", flush=True)
    res = {"pictures": len(clean_err), "clean": summarize(clean_err, touching, True), "degraded": summarize(degr_err, touching, False),
           "shift_median_px": round(float(np.median(shifts)), 2), "shift_p95_px": round(float(np.percentile(shifts, 95)), 2),
           "repeat_max_px": round(max(repeat_diffs), 4) if repeat_diffs else None}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
