"""Offline pose estimation with the MARS top-view models: a video in, a CalMS21-layout keypoints file out (new file).

This runs on a computer with TensorFlow (for example Google Colab with a GPU). It is NOT part of the live program:
the project has no live pose model, because MARS needs about 0.9 s per frame on a laptop CPU. See docs/camera.md.

What it does (the same steps as MARS_pose_machinery.py, written again for modern TensorFlow):
  1. (optional) stretch the screen part of each picture to 1024 x 570 with the 4 corners you give
  2. a detector for the black mouse and one for the white mouse give a box each
  3. each box is cut out and resized to 256 x 256
  4. the pose network gives heatmaps; the brightest point of each heatmap is a keypoint
Mouse "a" is the black (resident) mouse, mouse "b" the white (intruder) mouse, like in CalMS21.

The MARS models are CC BY-NC 4.0 (Segalin et al. 2021, eLife). Only the .pb graph files are used; the .pkl files in the
same zip are never opened, because pickle files can run code.

    python scripts/mars_pose_offline.py --download-only --models-dir mars_models
    python scripts/mars_pose_offline.py clip.mp4 --out keypoints.parquet --models-dir mars_models --batch 8 \
        --corners 0.1,0.2,0.9,0.2,0.9,0.8,0.1,0.8 --save-video stretched.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import zipfile

KEYPOINTS = ("nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base")   # MARS order = CalMS21 order
MICE = ("a", "b")
IMAGE_SIZE = (1024, 570)                                   # width, height of the CalMS21 frame
DET_IM_SIZE, POSE_IM_SIZE = 299, 256
MODELS_URL = "https://data.caltech.edu/records/801be-66057/files/models.zip?download=1"
MODELS_MD5 = "58efd60b0b3ca2cb27e40cef7e9d0da1"
PB_FILES = ("detection/MARS_top_detection_black.pb", "detection/MARS_top_detection_white.pb", "pose/MARS_top_pose.pb")
PREV_BOX = [1e-2, 1e-2, 2e-2, 2e-2]                         # MARS starts with this tiny box until a mouse is found


def download_models(models_dir):
    """Download models.zip (543 MB), check its md5, and extract only the three .pb files into models_dir."""
    os.makedirs(models_dir, exist_ok=True)
    if all(os.path.isfile(os.path.join(models_dir, p)) for p in PB_FILES):
        print("models already there")
        return
    zip_path = os.path.join(models_dir, "models.zip")
    print("downloading models.zip (543 MB) ...", flush=True)
    urllib.request.urlretrieve(MODELS_URL, zip_path)
    md5 = hashlib.md5()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    if md5.hexdigest() != MODELS_MD5:
        raise SystemExit(f"the download is damaged or changed (md5 {md5.hexdigest()}, expected {MODELS_MD5})")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("models/") and name.endswith(".pb") and ".." not in name:
                target = os.path.join(models_dir, name[len("models/"):])
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    os.remove(zip_path)
    print("models ready in", models_dir)


def parse_corners(text):
    """'x1,y1,x2,y2,x3,y3,x4,y4' as 4 (x, y) fractions of the picture: top-left, top-right, bottom-right, bottom-left."""
    if not text:
        return None
    v = [float(x) for x in text.split(",")]
    if len(v) != 8 or any(not 0.0 <= x <= 1.0 for x in v):
        raise SystemExit("--corners needs 8 numbers between 0 and 1")
    return [(v[i], v[i + 1]) for i in range(0, 8, 2)]


def to_frame(image, corners):
    """The picture as a 1024 x 570 frame: the screen stretched with the corners, or the whole picture resized."""
    import cv2
    import numpy as np
    w, h = IMAGE_SIZE
    if corners is None:
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    ih, iw = image.shape[:2]
    src = np.float32([(x * iw, y * ih) for x, y in corners])
    dst = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h))


def prep_det(frame):
    import cv2
    import numpy as np
    x = cv2.resize(frame.astype(np.float32), (DET_IM_SIZE, DET_IM_SIZE), interpolation=cv2.INTER_NEAREST)
    return ((x - 128) / 128).ravel()


def post_det(locations, confidences, prev):
    """The best box (stretched by 10%) and its confidence, or the last good box when nothing is found (as in MARS)."""
    import numpy as np
    locs = np.clip(locations, 0.0, 1.0)
    conf = np.squeeze(confidences, axis=-1) if confidences.ndim == 2 else np.asarray(confidences).reshape(len(locs), -1)[:, 0]
    best = int(np.argmax(conf))
    if float(conf[best]) <= 0.005:
        return prev[0], prev[1]
    xmin, ymin, xmax, ymax = [float(v) for v in locs[best]]
    if abs(xmin - xmax) < 0.005:
        xmin, xmax = (xmin - 0.005, xmax) if xmin > 0.006 else (xmin, xmax + 0.005)
    if abs(ymin - ymax) < 0.005:
        ymin, ymax = (ymin - 0.005, ymax) if ymin > 0.006 else (ymin, ymax + 0.005)
    sx = min(min(xmin, abs(1 - xmax)), (xmax - xmin) * 0.10)
    sy = min(min(ymin, abs(1 - ymax)), (ymax - ymin) * 0.10)
    return [xmin - sx, ymin - sy, xmax + sx, ymax + sy], float(conf[best])


def crop(frame, box):
    import cv2
    import numpy as np
    w, h = IMAGE_SIZE
    x1, y1, x2, y2 = np.round(np.array(box) * np.array([w, h, w, h])).astype(int)
    c = frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)].astype(np.float32)
    return (cv2.resize(c, (POSE_IM_SIZE, POSE_IM_SIZE), interpolation=cv2.INTER_NEAREST) - 128) / 128


def decode(heatmaps, box):
    """7 keypoints (x, y, score) in frame pixels from the 64 x 64 x 7 heatmaps of one mouse."""
    import cv2
    import numpy as np
    w, h = IMAGE_SIZE
    hm = cv2.resize(np.clip(heatmaps, 0.0, 1.0), (POSE_IM_SIZE, POSE_IM_SIZE), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = box
    bw, bh = max(1, int(np.round((x2 - x1) * w))), max(1, int(np.round((y2 - y1) * h)))
    hm = cv2.resize(hm, (bw, bh), interpolation=cv2.INTER_LINEAR)
    out = []
    for j in range(hm.shape[2]):
        y, x = np.unravel_index(np.argmax(hm[:, :, j]), hm.shape[:2])
        out.append((float(x + x1 * w), float(y + y1 * h), float(hm[:, :, j].max())))
    return out


class Mars:
    def __init__(self, models_dir):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        self.tf = tf
        self.black = self._load(os.path.join(models_dir, PB_FILES[0]))
        self.white = self._load(os.path.join(models_dir, PB_FILES[1]))
        self.pose = self._load(os.path.join(models_dir, PB_FILES[2]))
        self.prev = [(list(PREV_BOX), 1e-4), (list(PREV_BOX), 1e-4)]

    def _load(self, pb):
        tf = self.tf
        gd = tf.compat.v1.GraphDef()
        with open(pb, "rb") as f:
            gd.ParseFromString(f.read())
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(gd, name="")
        return g, tf.compat.v1.Session(graph=g)

    def _detect(self, model, x):
        g, s = model
        return s.run([g.get_operation_by_name("predicted_locations").outputs[0], g.get_operation_by_name("Multibox/Sigmoid").outputs[0]],
                     {g.get_operation_by_name("images").outputs[0]: x})

    def run(self, frames):
        """frames: list of 1024 x 570 BGR pictures. Returns one list per frame: [(black keypoints, box, conf), (white ...)]."""
        import numpy as np
        x = np.stack([prep_det(f) for f in frames])
        locs_b, conf_b = self._detect(self.black, x)
        locs_w, conf_w = self._detect(self.white, x)
        boxes = []
        for i in range(len(frames)):
            row = []
            for k, (locs, conf) in enumerate(((locs_b, conf_b), (locs_w, conf_w))):
                box, c = post_det(locs[i], conf[i], self.prev[k])
                self.prev[k] = (box, c)
                row.append((box, c))
            boxes.append(row)
        crops = np.stack([crop(f, b[0]) for f, row in zip(frames, boxes) for b in row])
        g, s = self.pose
        hms = s.run(g.get_operation_by_name("output_heatmaps").outputs[0], {g.get_operation_by_name("images").outputs[0]: crops})
        results = []
        for i, row in enumerate(boxes):
            results.append([(decode(hms[2 * i + k], row[k][0]), row[k][0], row[k][1]) for k in range(2)])
        return results


def process(video, out, models_dir, corners=None, start=0, max_frames=None, batch=8, session="camera", fps=None, save_video=None):
    import cv2
    import pandas as pd
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    todo = min(max_frames, total - start) if (max_frames and total) else (max_frames or (total - start if total else None))
    writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, IMAGE_SIZE) if save_video else None
    mars = Mars(models_dir)
    rows, n, t0, box_conf = [], 0, time.time(), []
    while True:
        frames = []
        while len(frames) < batch and (todo is None or n + len(frames) < todo):
            ok, img = cap.read()
            if not ok:
                break
            frames.append(to_frame(img, corners))
        if not frames:
            break
        for i, per_mouse in enumerate(mars.run(frames)):
            frame_no = n + i
            for m, (kps, box, conf) in zip(MICE, per_mouse):
                box_conf.append(conf)
                for name, (x, y, score) in zip(KEYPOINTS, kps):
                    rows.append((session, m, name, frame_no, frame_no / fps, x, y, score))
            if writer is not None:
                writer.write(frames[i])
        n += len(frames)
        el = time.time() - t0
        print(f"{n}/{todo or '?'} frames, {n / el:.1f} frames per second" + (f", about {(todo - n) / (n / el) / 60:.1f} min left" if todo else ""), flush=True)
    if writer is not None:
        writer.release()
    df = pd.DataFrame(rows, columns=["session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score"])
    df.to_parquet(out, index=False)
    low = sum(c < 0.05 for c in box_conf)
    print(f"saved {len(df)} rows ({n} frames) to {out}; boxes with confidence under 0.05: {low} of {len(box_conf)}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", nargs="?")
    ap.add_argument("--out", default="keypoints.parquet")
    ap.add_argument("--models-dir", default="mars_models")
    ap.add_argument("--download-only", action="store_true", help="only download the MARS models")
    ap.add_argument("--corners", help="8 numbers (fractions): the 4 corners of the screen in the picture")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--batch", type=int, default=8, help="pictures processed together (more is faster on a GPU)")
    ap.add_argument("--session", default="camera")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--save-video", help="also save the 1024 x 570 pictures the model saw, as an mp4")
    a = ap.parse_args(argv)
    download_models(a.models_dir)
    if a.download_only:
        return 0
    if not a.video:
        raise SystemExit("give a video file")
    process(a.video, a.out, a.models_dir, parse_corners(a.corners), a.start_frame, a.max_frames, a.batch, a.session, a.fps, a.save_video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
