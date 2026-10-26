"""Real-time test of a video-to-keypoints model: feed a video at exactly 30 frames per second, as a live camera would (new file).

A producer makes frame i available at start + i / 30 s. The model takes the NEWEST available frame, runs, and takes the newest one again;
frames that arrived while it was busy are dropped (a live system cannot wait). We measure:
  keep-up      the share of frames the model processed, and the frames per second it reached (live needs about 30);
  latency      how long after a frame was available its keypoints were ready (median, 95th percentile);
  accuracy     the pixel error against saved keypoints of the same frames (--poses, the CalMS21 file of the demo video), if given.

    # MARS (run with the Python of the MARS environment)
    %USERPROFILE%\\mars_local\\env\\Scripts\\python.exe scripts\\realtime_pose_test.py data\\raw\\calms21_video\\mouse064_task1_annotator1.mp4 ^
        --backend mars --models-dir %USERPROFILE%\\mars_local\\models --start 1800 --seconds 20 --poses data\\raw\\calms21_video\\mouse064_task1_annotator1_poses.parquet
    # fast YOLO model (run with the Python that has ultralytics and openvino)
    python scripts\\realtime_pose_test.py <video> --backend yolo --weights <best.pt or an OpenVINO folder> --device intel:gpu ...

For a camera that is slower than 30 fps use --fps (for example 15).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEYPOINTS = ("nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base")   # CalMS21 names, MARS slot order


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_mars(models_dir):
    mp = load_module("mars_pose_offline")
    mars = mp.Mars(models_dir)

    def run(frame):
        res = mars.run([frame])[0]
        return np.array([[(x, y) for x, y, _ in res[k][0]] for k in (0, 1)], float)       # (2 mice, 7, 2)
    return run


def make_yolo(weights, device, imgsz):
    from ultralytics import YOLO
    model = YOLO(weights, task="pose")

    def run(frame):
        kw = {"imgsz": imgsz, "conf": 0.25, "verbose": False}
        if device:
            kw["device"] = device
        r = model.predict(frame, **kw)[0]
        out = np.full((2, 7, 2), np.nan)
        if r.keypoints is not None and len(r.boxes):
            kps, cls, cf = r.keypoints.xy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
            for c in (0, 1):
                idx = [i for i in range(len(cls)) if cls[i] == c]
                if idx:
                    out[c] = kps[max(idx, key=lambda i: cf[i])][:7]
        return out
    return run


def saved_keypoints(poses_path, frames):
    """{frame: (2, 7, 2)} from the CalMS21 keypoints file (mouse a = black, b = white)."""
    import pandas as pd
    d = pd.read_parquet(poses_path)
    d = d[d["frame"].isin(set(frames))]
    out = {}
    for f, part in d.groupby("frame"):
        arr = np.full((2, 7, 2), np.nan)
        for i, m in enumerate(("a", "b")):
            for j, k in enumerate(KEYPOINTS):
                row = part[(part["mouse"] == m) & (part["keypoint"] == k)]
                if len(row):
                    arr[i, j] = (row["x"].iloc[0], row["y"].iloc[0])
        out[int(f)] = arr
    return out


def simulate(frames, fps, run, seconds):
    """Real-time feed. Returns the list of (frame index, finished at, latency, keypoints)."""
    n = len(frames)
    t0 = time.perf_counter()
    done, last = [], -1
    while True:
        now = time.perf_counter() - t0
        newest = min(n - 1, int(now * fps))
        if newest <= last:
            if now >= seconds or last >= n - 1:
                break
            time.sleep(max(0.0, ((last + 1) / fps) - now))
            continue
        i = newest
        kps = run(frames[i])
        finished = time.perf_counter() - t0
        done.append((i, finished, finished - i / fps, kps))
        last = i
        if i >= n - 1:
            break
    return done, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--backend", choices=("mars", "yolo"), required=True)
    ap.add_argument("--models-dir", help="MARS models folder")
    ap.add_argument("--weights", help="YOLO weights (.pt) or an exported folder")
    ap.add_argument("--device", default=None, help="for YOLO, for example intel:gpu or cpu")
    ap.add_argument("--imgsz", default="384,640", help="YOLO input: height,width")
    ap.add_argument("--start", type=int, default=0, help="first frame of the video")
    ap.add_argument("--seconds", type=float, default=20.0, help="how long the live feed lasts")
    ap.add_argument("--fps", type=float, default=30.0, help="frames per second of the feed")
    ap.add_argument("--poses", help="CalMS21 keypoints (parquet) of the same video, for accuracy")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    import cv2
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    frames = []
    for _ in range(int(a.seconds * a.fps) + 1):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if a.backend == "mars":
        run = make_mars(a.models_dir)
    else:
        h, w = (int(v) for v in a.imgsz.split(","))
        run = make_yolo(a.weights, a.device, (h, w))
    for f in frames[:3]:                                    # warm-up, not counted
        run(f)
    done, wall = simulate(frames, a.fps, run, a.seconds)

    lat = np.array([d[2] for d in done])
    res = {"backend": a.backend, "feed_fps": a.fps, "seconds_fed": round(min(wall, a.seconds), 1), "frames_available": len(frames),
           "frames_processed": len(done), "share_processed": round(len(done) / max(1, len(frames)), 4),
           "reached_fps": round(len(done) / wall, 2), "latency_median_s": round(float(np.median(lat)), 3),
           "latency_p95_s": round(float(np.percentile(lat, 95)), 3), "keeps_up_with_live": bool(len(done) / max(1, len(frames)) >= 0.95)}
    if a.poses:
        truth = saved_keypoints(a.poses, [a.start + d[0] for d in done])
        errs, found = [], 0
        for i, _, _, kps in done:
            t = truth.get(a.start + i)
            if t is None:
                continue
            e = np.hypot(*(kps - t).transpose(2, 0, 1))
            errs.append(e)
            found += int(np.isfinite(kps).all())
        if errs:
            e = np.array(errs)
            res["accuracy_vs_saved_keypoints"] = {"pictures": len(errs), "both_mice_found_share": round(found / len(errs), 4),
                                                  "median_error_px": round(float(np.nanmedian(e)), 2),
                                                  "within_5px": round(float(np.nanmean(e <= 5)), 4),
                                                  "within_10px": round(float(np.nanmean(e <= 10)), 4)}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
