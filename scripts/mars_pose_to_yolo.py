"""Turn the MARS hand-labeled top-view pose set into a YOLO pose dataset (new file).

Input (Caltech Data record 2011, CC BY-NC 4.0, NOT stored in Git): raw_images_top.zip (15,000 pictures, 1024 x 570) and
MARS_keypoints_top.json (the median of 5 workers per keypoint; 7 keypoints per mouse for a "black" and a "white" mouse).

Output (a folder, outside the project): images/*.jpg, labels/*.txt, train.txt, val.txt, dataset.yaml.
  - Two classes: 0 = black mouse (the resident, mouse "a" in CalMS21), 1 = white mouse (the intruder, mouse "b"). The class
    gives the identity, so the model does not need to guess who is who from the position.
  - 7 keypoints in the MARS order (nose, right ear, left ear, neck, right side, left side, tail base). The project's CalMS21 files
    call these slots nose, left_ear, right_ear, neck, left_hip, right_hip, tail_base (the same positions).
  - flip_idx swaps the two ears and the two sides, so a flipped picture keeps the right labels.
  - The box of a mouse is made from its keypoints plus a margin (the labels have no boxes).
  - Split: the last 1,500 pictures are the test set (the paper used 13,500 / 1,500). The names are not known to separate videos,
    so near-identical frames can sit on both sides: the test accuracy can be a little too good. Use new videos for the final check.

    python scripts/mars_pose_to_yolo.py --images raw_images_top.zip --labels MARS_keypoints_top.json --out dataset
    python scripts/mars_pose_to_yolo.py --images dataset/images --labels MARS_keypoints_top.json --out dataset --preview 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path

CLASSES = ("black", "white")
KEYPOINT_NAMES = ("nose", "right_ear", "left_ear", "neck", "right_side", "left_side", "tail_base")
FLIP_IDX = [0, 2, 1, 3, 5, 4, 6]
VAL_COUNT = 1500
MARGIN_FRACTION, MARGIN_MIN_PX = 0.20, 12.0
MD5 = {"raw_images_top.zip": "77aa6739caf01c592674dff1e09e8968", "MARS_keypoints_top.json": "c42cc0a9f342e07ac9ed7e269cb590dd"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_md5(path):
    want = MD5.get(Path(path).name)
    if want and md5(path) != want:
        raise SystemExit(f"{path} is damaged or changed (md5 does not match {want})")


def extract_images(zip_path, images_dir):
    """Extract only .jpg files, flat, with safe names."""
    images_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name.lower().endswith(".jpg") or ".." in info.filename.split("/"):
                continue
            target = images_dir / name
            if not target.exists():
                with z.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            n += 1
    return n


def label_line(cls, xs, ys, w, h):
    """One YOLO pose line: class, box (center x, center y, width, height), then x y visible for every keypoint (all 0..1)."""
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    m = max(MARGIN_MIN_PX, MARGIN_FRACTION * max(x1 - x0, y1 - y0))
    x0, x1, y0, y1 = max(0.0, x0 - m), min(float(w), x1 + m), max(0.0, y0 - m), min(float(h), y1 + m)
    parts = [str(cls), f"{(x0 + x1) / 2 / w:.6f}", f"{(y0 + y1) / 2 / h:.6f}", f"{(x1 - x0) / w:.6f}", f"{(y1 - y0) / h:.6f}"]
    for x, y in zip(xs, ys):
        parts += [f"{min(max(x / w, 0.0), 1.0):.6f}", f"{min(max(y / h, 0.0), 1.0):.6f}", "2"]
    return " ".join(parts)


def convert(records, images_dir, out):
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for rec in records:
        img = images_dir / rec["filename"]
        if not img.is_file():
            continue
        w, h = rec["width"], rec["height"]
        lines = [label_line(i, rec["coords"][c]["x"], rec["coords"][c]["y"], w, h) for i, c in enumerate(CLASSES)]
        (labels_dir / (Path(rec["filename"]).stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        kept.append(rec["filename"])
    kept.sort()
    val = kept[-VAL_COUNT:] if len(kept) > 2 * VAL_COUNT else kept[-max(1, len(kept) // 10):]
    train = [k for k in kept if k not in set(val)]
    (out / "train.txt").write_text("\n".join(str((images_dir / k).resolve()).replace("\\", "/") for k in train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(str((images_dir / k).resolve()).replace("\\", "/") for k in val) + "\n", encoding="utf-8")
    (out / "dataset.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: train.txt\nval: val.txt\nkpt_shape: [7, 3]\nflip_idx: {FLIP_IDX}\n"
        f"names:\n  0: black\n  1: white\n", encoding="utf-8")
    return len(train), len(val)


def preview(records, images_dir, out, n):
    import cv2
    colors = {0: (40, 40, 255), 1: (255, 200, 0)}
    random.Random(1).shuffle(records)
    (out / "preview").mkdir(exist_ok=True)
    for rec in records[:n]:
        img = cv2.imread(str(images_dir / rec["filename"]))
        for i, c in enumerate(CLASSES):
            xs, ys = rec["coords"][c]["x"], rec["coords"][c]["y"]
            for j, (x, y) in enumerate(zip(xs, ys)):
                cv2.circle(img, (int(x), int(y)), 5, colors[i], -1)
                cv2.putText(img, str(j), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.polylines(img, [__import__("numpy").int32(list(zip(xs[:4], ys[:4])))], False, colors[i], 1)
        cv2.imwrite(str(out / "preview" / rec["filename"]), img)
    print("preview pictures in", out / "preview")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="raw_images_top.zip or a folder of the extracted jpg files")
    ap.add_argument("--labels", required=True, help="MARS_keypoints_top.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", type=int, default=0, help="also draw this many labeled pictures into out/preview")
    a = ap.parse_args(argv)
    out = Path(a.out)
    check_md5(a.labels)
    records = json.loads(Path(a.labels).read_text(encoding="utf-8"))
    src = Path(a.images)
    if src.is_file():
        check_md5(src)
        images_dir = out / "images"
        print("extracted", extract_images(src, images_dir), "pictures")
    else:
        images_dir = src
    n_train, n_val = convert(records, images_dir, out)
    print(f"labels written: {n_train} train pictures, {n_val} test pictures, in {out}")
    if a.preview:
        preview(records, images_dir, out, a.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
