"""Speed test: small YOLO pose models on this laptop, PyTorch CPU versus OpenVINO (CPU and Intel Iris Xe).
The pretrained weights know PEOPLE (17 keypoints), not mice, so this measures SPEED only, not accuracy.
usage: python bench.py <video> <start_frame> <n_frames>
"""
import statistics
import sys
import time

import cv2
import numpy as np

video, start, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
frames = []
while len(frames) < n:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
print("frames:", len(frames), frames[0].shape, flush=True)

import ultralytics
from ultralytics import YOLO
print("ultralytics", ultralytics.__version__, flush=True)

WEIGHTS = None
for name in ("yolo26n-pose.pt", "yolo11n-pose.pt"):
    try:
        model = YOLO(name)
        WEIGHTS = name
        break
    except Exception as exc:
        print(f"could not load {name}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
if WEIGHTS is None:
    raise SystemExit("no pose weights could be loaded")
print("using", WEIGHTS, flush=True)


def measure(m, imgsz, label, device=None, warm=10):
    kw = {"imgsz": imgsz, "verbose": False}
    if device:
        kw["device"] = device
    try:
        for f in frames[:warm]:
            m.predict(f, **kw)
        pre, inf, post, wall = [], [], [], []
        for f in frames[warm:]:
            t = time.perf_counter()
            r = m.predict(f, **kw)[0]
            wall.append((time.perf_counter() - t) * 1000)
            pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
        print(f"{label:46s} pre {statistics.median(pre):5.1f} | model {statistics.median(inf):6.1f} | post {statistics.median(post):5.1f} "
              f"| TOTAL median {statistics.median(wall):6.1f} ms  (worst 10%: {sorted(wall)[int(len(wall) * 0.9)]:.1f} ms)", flush=True)
    except Exception as exc:
        print(f"{label:46s} FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)


print("\n--- speed per frame (1024x570 picture), median over", len(frames) - 10, "frames ---", flush=True)
measure(model, (384, 640), f"{WEIGHTS} PyTorch CPU, 384x640")
measure(model, 640, f"{WEIGHTS} PyTorch CPU, 640x640")

for size, tag in (((384, 640), "384x640"), (640, "640x640"), ((288, 480), "288x480")):
    try:
        path = model.export(format="openvino", imgsz=size, half=False, verbose=False)
    except Exception as exc:
        print(f"export {tag} FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        continue
    ov = YOLO(path, task="pose")
    measure(ov, size, f"{WEIGHTS} OpenVINO CPU, {tag}", device="intel:cpu")
    if tag == "384x640":
        measure(ov, size, f"{WEIGHTS} OpenVINO Iris Xe GPU, {tag}", device="intel:gpu")
print("DONE", flush=True)
