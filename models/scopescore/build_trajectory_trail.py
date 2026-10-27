"""
Rhizomatiks-style fading-trail renderer.

Reads a tool-geometry CSV + the source video and writes a clean
trajectory-trail video: thin glowing lines that trace each tool's tip
path over the last N frames, alpha fading from full at the latest
position to transparent at the oldest. One colour per track.

Run:
    python build_trajectory_trail.py \\
        --video /path/to/vid06.mp4 \\
        --csv   /path/to/vid06_1200_1320s.csv \\
        --out   /path/to/vid06_trails_1200_1320s.mp4 \\
        --start-sec 1200 --end-sec 1320
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

# One bright colour per track_id (cycles).
TRACK_COLORS_BGR = [
    (255, 200,  40),    # cyan-ish
    ( 40,  80, 255),    # red-orange
    ( 80, 255,  80),    # green
    (255,  80, 255),    # magenta
    ( 40, 220, 255),    # yellow
    (255,  80,  80),    # blue
]


def color_for_track(tid: int) -> tuple[int, int, int]:
    return TRACK_COLORS_BGR[tid % len(TRACK_COLORS_BGR)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True,
                    help="source video the CSV was generated from")
    ap.add_argument("--csv", type=Path, required=True,
                    help="tool-geometry per-frame CSV")
    ap.add_argument("--out", type=Path, required=True,
                    help="output trail video (mp4)")
    ap.add_argument("--start-sec", type=float, default=None,
                    help="if set, only render this time window from "
                         "the source video (must match the CSV)")
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--trail-len", type=int, default=100,
                    help="trail length in frames (default 100 = 4 s @25fps)")
    ap.add_argument("--line-thickness", type=int, default=3)
    ap.add_argument("--dot-radius", type=int, default=7)
    ap.add_argument("--dim-background", type=float, default=0.55,
                    help="multiply source frame by this to darken the "
                         "underlying footage (0 = black, 1 = no dim)")
    ap.add_argument("--show-label", action="store_true",
                    help="render the tool label next to the current tip")
    args = ap.parse_args()

    # --- read CSV: frame -> list of (track_id, tool_label, tip_x, tip_y) ---
    per_frame: dict[int, list[tuple[int, str, float, float]]] = defaultdict(list)
    with args.csv.open(newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                fi = int(float(r["frame"]))
                tid = int(float(r["track_id"]))
                lab = r.get("tool_label", "")
                tx = float(r["tip_x"]); ty = float(r["tip_y"])
            except (KeyError, ValueError, TypeError):
                continue
            per_frame[fi].append((tid, lab, tx, ty))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"can't open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sf = (max(0, int(round(args.start_sec * fps)))
          if args.start_sec is not None else 0)
    ef = (min(total, int(round(args.end_sec * fps)))
          if args.end_sec is not None else total)

    writer = cv2.VideoWriter(str(args.out),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"can't open writer {args.out}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.trail_len))

    print(f"[trails] {args.video.name} -> {args.out.name} "
          f"({W}x{H}@{fps:.1f}, frames {sf}..{ef}, "
          f"trail_len={args.trail_len})", flush=True)

    n = 0
    for fi in range(sf, ef):
        ok, frame = cap.read()
        if not ok:
            break

        # 1. dim the source frame so the trails pop
        canvas = (frame.astype(np.float32) * args.dim_background).astype(np.uint8)

        # 2. append this frame's tip positions to per-track history
        for (tid, lab, tx, ty) in per_frame.get(fi, []):
            history[tid].append((tx, ty, lab))

        # 3. draw each track's trail with alpha that fades to zero at the
        #    oldest point. We compose onto an overlay then add it weighted.
        overlay = np.zeros_like(canvas)
        for tid, hist in history.items():
            if len(hist) < 2:
                continue
            color = color_for_track(tid)
            n_pts = len(hist)
            for i in range(1, n_pts):
                x0, y0, _ = hist[i - 1]
                x1, y1, _ = hist[i]
                # alpha 0 (oldest end) -> 1 (newest end)
                a = i / float(n_pts - 1)
                # scale segment color by alpha — then we'll addWeighted
                seg_color = tuple(int(c * a) for c in color)
                cv2.line(overlay,
                         (int(round(x0)), int(round(y0))),
                         (int(round(x1)), int(round(y1))),
                         seg_color, args.line_thickness, cv2.LINE_AA)
        # add overlay onto the dimmed background — additive blend gives a glow
        canvas = cv2.add(canvas, overlay)

        # 4. for every track that has a position this frame, draw a bright dot
        for (tid, lab, tx, ty) in per_frame.get(fi, []):
            color = color_for_track(tid)
            ix, iy = int(round(tx)), int(round(ty))
            cv2.circle(canvas, (ix, iy), args.dot_radius + 2,
                       (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (ix, iy), args.dot_radius,
                       color, -1, cv2.LINE_AA)
            if args.show_label and lab:
                cv2.putText(canvas, lab, (ix + 12, iy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, lab, (ix + 12, iy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            color, 1, cv2.LINE_AA)

        writer.write(canvas)
        n += 1
        if n % 200 == 0 or n == 1:
            print(f"  [trails] {n}/{ef - sf}", flush=True)

    writer.release()
    cap.release()
    print(f"[done] {n} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
