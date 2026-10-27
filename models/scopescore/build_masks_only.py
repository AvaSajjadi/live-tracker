"""
Tiny mask-only renderer for retrain comparison.

For each frame, runs YOLO11m-seg on the input video and overlays ONLY
the predicted masks (semi-transparent, class-coloured). No skeleton,
no polyline, no S/T/C/T2 dots, no HUD. Pair two runs (one per weights
file) and stitch with build_grid.py to show "PhaKIR mask vs Cholec mask"
side by side.

Usage:
    python build_masks_only.py \\
        --video /path/vid07.mp4 \\
        --weights /path/to/best.pt \\
        --out   /path/out.mp4 \\
        --start-sec 0 --end-sec 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_COLORS_BGR = {
    0: (40,  40, 255), 1: (0,   200, 255), 2: (60,  220, 60),
    3: (255, 200, 40), 4: (255, 60,  255), 5: (255, 255, 40),
    6: (200, 200, 200),
}
TOOL_CLASSES = (0, 1, 2, 3, 4, 5, 6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=120.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms-iou", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="mask overlay alpha (higher = more opaque)")
    ap.add_argument("--label", default="",
                    help="optional small label drawn top-left "
                         "(e.g. 'PhaKIR-trained YOLO')")
    args = ap.parse_args()

    print(f"[init] loading YOLO-seg: {args.weights}", flush=True)
    yolo = YOLO(str(args.weights))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"can't open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sf = max(0, int(round(args.start_sec * fps)))
    ef = min(total, int(round(args.end_sec * fps)))

    writer = cv2.VideoWriter(str(args.out),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"can't open writer {args.out}")
    print(f"[run] {args.video.name} -> {args.out.name} "
          f"({W}x{H}@{fps:.1f}, frames {sf}..{ef})", flush=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    n = 0
    for fi in range(sf, ef):
        ok, frame = cap.read()
        if not ok:
            break
        res = yolo.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                           iou=args.nms_iou, agnostic_nms=True,
                           device=args.device, verbose=False)[0]

        canvas = frame.copy()
        if (res.boxes is not None and res.masks is not None
                and len(res.boxes) > 0):
            classes = res.boxes.cls.cpu().numpy().astype(int)
            masks = res.masks.data.cpu().numpy()
            overlay = canvas.copy()
            for k in range(len(classes)):
                cls = int(classes[k])
                if cls not in TOOL_CLASSES:
                    continue
                color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))
                m = masks[k]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_NEAREST)
                mb = m > 0.5
                overlay[mb] = color
            cv2.addWeighted(overlay, args.alpha,
                            canvas, 1.0 - args.alpha, 0, dst=canvas)

        if args.label:
            cv2.putText(canvas, args.label, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                        cv2.LINE_AA)
            cv2.putText(canvas, args.label, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)

        writer.write(canvas)
        n += 1
        if n % 200 == 0 or n == 1:
            print(f"  [run] {n}/{ef - sf}", flush=True)

    writer.release()
    cap.release()
    print(f"[done] {n} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
