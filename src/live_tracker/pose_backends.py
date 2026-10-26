"""Pose model backends for the live camera path (new file): a picture in, mouse keypoints out. See pose_model.py and docs/pose_model.md.

yolo  A small YOLO pose model trained on the MARS hand-labeled set (scripts/train_yolo_pose.py): class 0 = the black mouse (mouse "a"),
      class 1 = the white mouse (mouse "b"), 7 keypoints each, in the MARS order that CalMS21 also uses. Weights: a .pt file or an exported
      OpenVINO / CoreML folder. Settings (models/pose/pose_model.json):
          {"backend": "yolo", "weights": "models/pose/mouse_pose_openvino_model", "device": "intel:gpu", "imgsz": [384, 640], "conf": 0.25}
      hold: keep the last points of a mouse that is missed for up to this many pictures (default 0 = off; 3 is 0.1 s).
      device: "intel:gpu" (Intel Iris Xe, OpenVINO), "intel:cpu", "cpu", "mps" (Apple), "0" (NVIDIA). Omit it to let Ultralytics choose.

stub  NOT a pose model. It returns the saved CalMS21 keypoints of a recording, following the wall clock, as if a perfect and instant model
      were looking at the camera. It lets scripts/live_camera_acceptance.py test everything around the model (camera speed, scores,
      monitor, alerts) before a trained model exists. Settings: {"backend": "stub", "poses": "<poses.parquet>", "start_frame": 1800}.

Both give the table pose_model.py expects: columns mouse, keypoint, x, y, score, in pixels of the 1024 x 570 CalMS21 frame, or None when
no mouse is found. A picture of another size is resized to 1024 x 570 first (a camera that films a screen is stretched by
screen_map.py before it gets here).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .calms21 import IMAGE_SIZE, KEYPOINTS, MICE

DEFAULT_IMGSZ = (384, 640)                    # height, width: wide and short, like the 1024 x 570 picture
DEFAULT_CONF = 0.25


def _load_yolo(path):
    """The Ultralytics model (the place tests replace; ultralytics is imported only here)."""
    from ultralytics import YOLO
    return YOLO(str(path), task="pose")


def _to_frame(image):
    if image.shape[1] == IMAGE_SIZE[0] and image.shape[0] == IMAGE_SIZE[1]:
        return image
    import cv2
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def yolo_factory(settings, root):
    weights = Path(settings["weights"])
    weights = weights if weights.is_absolute() else Path(root) / weights
    if not weights.exists():
        raise FileNotFoundError(str(weights))
    imgsz = tuple(settings.get("imgsz", DEFAULT_IMGSZ))
    conf = float(settings.get("conf", DEFAULT_CONF))
    hold = int(settings.get("hold", 0))                     # keep the last points of a mouse that is missed for up to this many pictures
    memory = {}                                               # mouse -> (rows, pictures since it was seen)
    device = settings.get("device")
    model = _load_yolo(weights)
    kw = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if device:
        kw["device"] = device
    # The first picture makes OpenVINO prepare the model for the Intel GPU, and that takes several seconds (a stall of about 10 s was
    # seen in the first 10 s of a run, 21 instead of 29 pictures per second). Do it here, once, before the first real picture.
    model.predict(np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8), **kw)

    def pose(image):
        r = model.predict(_to_frame(image), **kw)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return _held({}) if hold else None
        cls = r.boxes.cls.cpu().numpy().astype(int)
        box_conf = r.boxes.conf.cpu().numpy()
        xy = r.keypoints.xy.cpu().numpy()
        kp_conf = r.keypoints.conf.cpu().numpy() if getattr(r.keypoints, "conf", None) is not None else None
        rows = []
        for c, mouse in enumerate(MICE):
            idx = [i for i in range(len(cls)) if cls[i] == c]
            if not idx:
                continue
            best = max(idx, key=lambda i: box_conf[i])
            for j, name in enumerate(KEYPOINTS):
                score = float(kp_conf[best][j]) if kp_conf is not None else float(box_conf[best])
                rows.append({"mouse": mouse, "keypoint": name, "x": float(xy[best][j][0]), "y": float(xy[best][j][1]),
                             "score": min(1.0, max(0.0, score))})
        seen = {}
        for row in rows:
            seen.setdefault(row["mouse"], []).append(row)
        return _held(seen) if hold else (pd.DataFrame(rows) if rows else None)

    def _held(seen):
        """The mice seen now, plus the last points of a mouse that was missed for at most `hold` pictures (a short miss is not a loss)."""
        out = []
        for mouse in MICE:
            if mouse in seen:
                memory[mouse] = (seen[mouse], 0)
                out += seen[mouse]
            elif mouse in memory and memory[mouse][1] < hold:
                memory[mouse] = (memory[mouse][0], memory[mouse][1] + 1)
                out += memory[mouse][0]
            else:
                memory.pop(mouse, None)
        return pd.DataFrame(out) if out else None
    return pose


def stub_factory(settings, root):
    path = Path(settings["poses"])
    path = path if path.is_absolute() else Path(root) / path
    poses = pd.read_parquet(path, columns=["mouse", "keypoint", "frame", "x", "y", "score"])
    start = int(settings.get("start_frame", 0))
    by_frame = {int(f): part[["mouse", "keypoint", "x", "y", "score"]].reset_index(drop=True) for f, part in poses.groupby("frame")}
    last = max(by_frame)
    clock = {"t0": None}

    def pose(image):
        now = time.perf_counter()
        if clock["t0"] is None:
            clock["t0"] = now
        frame = min(last, start + int(round((now - clock["t0"]) * 30.0)))
        return by_frame.get(frame)
    return pose
