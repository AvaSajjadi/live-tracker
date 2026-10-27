"""
Stitch two annotated videos side by side for visual comparison.

Typical use: compare the pure per-frame geometry (no temporal filters)
against the filtered pipeline output.

    python build_grid.py \
        --left  vid07_pure_0_120s.mp4 \
        --right vid07_0_120s.mp4 \
        --out   vid07_grid_0_120s.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def _label(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--left-label", default="pure geometry (no filters)")
    ap.add_argument("--right-label", default="filtered")
    ap.add_argument("--gap", type=int, default=6,
                    help="black gap (px) between the two panels")
    args = ap.parse_args()

    capL = cv2.VideoCapture(str(args.left))
    capR = cv2.VideoCapture(str(args.right))
    if not capL.isOpened():
        raise SystemExit(f"can't open {args.left}")
    if not capR.isOpened():
        raise SystemExit(f"can't open {args.right}")

    fps = capL.get(cv2.CAP_PROP_FPS)
    W = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WR = int(capR.get(cv2.CAP_PROP_FRAME_WIDTH))
    HR = int(capR.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != (WR, HR):
        print(f"[warn] panel sizes differ: left {W}x{H}, right {WR}x{HR}; "
              f"right will be resized to the left's size", flush=True)

    nL = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    nR = int(capR.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(nL, nR)
    gap = max(0, args.gap)
    out_w = W * 2 + gap

    writer = cv2.VideoWriter(str(args.out),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, H))
    if not writer.isOpened():
        raise SystemExit(f"can't open writer {args.out}")
    print(f"[grid] {args.left.name} | {args.right.name} -> {args.out.name} "
          f"({out_w}x{H}@{fps:.1f}, {n} frames)", flush=True)

    i = 0
    while i < n:
        okL, fL = capL.read()
        okR, fR = capR.read()
        if not okL or not okR:
            break
        if (fR.shape[1], fR.shape[0]) != (W, H):
            fR = cv2.resize(fR, (W, H))
        canvas = np.zeros((H, out_w, 3), dtype=np.uint8)
        canvas[:, :W] = fL
        canvas[:, W + gap:] = fR
        _label(canvas, args.left_label, 12, 28)
        _label(canvas, args.right_label, W + gap + 12, 28)
        writer.write(canvas)
        i += 1
        if i % 200 == 0:
            print(f"  [grid] {i}/{n}", flush=True)

    writer.release()
    capL.release()
    capR.release()
    print(f"[done] {i} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
