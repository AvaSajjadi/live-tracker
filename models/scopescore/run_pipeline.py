"""
Tool-geometry pipeline — YOLO-seg + tip_detector + KLT + One-Euro.

The tip / start / center decisions come from a single geometric rule
(distance to the FOV-out region) implemented in tip_detector.py. No
multi-step cascade, no prev_state, no width fallback.

Per-frame steps:
  1. YOLO-seg                  -> bbox + class + mask
  2. tip_detector.detect_tip_start(mask, fov_dist)
                               -> tip, start, center, jaw_state, main_path
  3. KLT optical flow + alpha-blend + One-Euro on tip / start / center
  4. Re-snap smoothed dots to mask boundary along the local tangent
  5. Render polyline along skeleton path; secondary jaw branches when
     jaw_state == 'open'

Output: <stem>_<S>_<E>s.mp4 + .csv per-frame log, written to
cholec20/tool_geometry/ by default.
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from causal_smoother import OneEuroVec2  # noqa: E402
from fov_utils import compute_fov_mask, compute_fov_distance  # noqa: E402
from tip_detector import detect_tip_start  # noqa: E402

YOLO_NAMES = ["grasper", "bipolar", "hook", "scissors", "clipper",
              "irrigator", "specimen_bag"]
TOOL_CLASSES = [0, 1, 2, 3, 4, 5]
CLASS_COLORS_BGR = {
    0: (40,  40, 255), 1: (0,   200, 255), 2: (60,  220, 60),
    3: (255, 200, 40), 4: (255, 60,  255), 5: (255, 255, 40),
}
TIP_BGR = (40, 40, 255)
START_BGR = (255, 80, 0)
CENTER_BGR = (0, 220, 255)
LINE_BGR = (255, 255, 255)


@dataclass
class Track:
    track_id: int
    tool_class: int
    last_frame: int
    last_bbox: Tuple[float, float, float, float]
    last_mask: np.ndarray
    last_yolo_conf: float
    last_tip: Tuple[float, float]
    last_start: Tuple[float, float]
    last_center: Tuple[float, float]
    last_gray: np.ndarray
    f_tip: Optional[OneEuroVec2] = None
    f_start: Optional[OneEuroVec2] = None
    f_center: Optional[OneEuroVec2] = None
    # temporal memory of the last tip seen fully inside the frame, so a
    # tip that exits the screen is "remembered" rather than lost.
    last_in_frame_tip: Optional[Tuple[float, float]] = None


def mask_iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def dedupe_bbox_only(raw, bbox_iou_thresh):
    kept = []
    for det in raw:
        drop = False
        for prev in kept:
            if bbox_iou(det["bbox"], prev["bbox"]) > bbox_iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(det)
    return kept


def klt_track_one(prev_gray, cur_gray, prev_xy, win=31, max_level=3):
    pts = np.array([[prev_xy]], dtype=np.float32)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, pts, None,
        winSize=(win, win), maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if status is None or status[0, 0] == 0:
        return None, False
    nx = float(next_pts[0, 0, 0])
    ny = float(next_pts[0, 0, 1])
    H, W = cur_gray.shape[:2]
    if not (0 <= nx < W and 0 <= ny < H):
        return None, False
    return (nx, ny), True


def _resnap_to_boundary(pt_xy, outward_xy, mask, max_walk=60):
    """Force a smoothed point onto the mask boundary by walking along the
    local outward tangent. If pt is inside the mask, walk OUTWARD until
    leaving; the last in-mask pixel is on the boundary at the tool's end.
    If pt is outside the mask, walk INWARD until entering; the first
    in-mask pixel is on the boundary."""
    H, W = mask.shape
    x, y = float(pt_xy[0]), float(pt_xy[1])
    dx, dy = outward_xy
    ix, iy = int(round(x)), int(round(y))
    inside = (0 <= iy < H and 0 <= ix < W
              and bool(mask[iy, ix]))
    if inside:
        last = (x, y)
        for s in range(1, max_walk + 1):
            nx = x + dx * s
            ny = y + dy * s
            iny, inx = int(round(ny)), int(round(nx))
            if iny < 0 or iny >= H or inx < 0 or inx >= W:
                break
            if not mask[iny, inx]:
                break
            last = (nx, ny)
        return last
    # outside the mask — walk inward until we enter it
    for s in range(1, max_walk + 1):
        nx = x - dx * s
        ny = y - dy * s
        iny, inx = int(round(ny)), int(round(nx))
        if iny < 0 or iny >= H or inx < 0 or inx >= W:
            continue
        if mask[iny, inx]:
            return (nx, ny)
    return (x, y)


def _outward_tangent_xy(raw_path, end='start'):
    """Outward unit tangent (dx, dy) at one end of a (row, col) skeleton path."""
    if raw_path is None or len(raw_path) < 2:
        return None
    if end == 'start':
        p0_r, p0_c = raw_path[0]
        p1_r, p1_c = raw_path[1]
    else:
        p0_r, p0_c = raw_path[-1]
        p1_r, p1_c = raw_path[-2]
    dy = float(p0_r - p1_r)
    dx = float(p0_c - p1_c)
    L = (dy * dy + dx * dx) ** 0.5
    if L < 1e-6:
        return None
    return (dx / L, dy / L)


def alpha_blend(p_klt, p_raw, alpha_raw):
    if p_klt is None and p_raw is None:
        return None
    if p_klt is None:
        return p_raw
    if p_raw is None:
        return p_klt
    return (alpha_raw * p_raw[0] + (1 - alpha_raw) * p_klt[0],
            alpha_raw * p_raw[1] + (1 - alpha_raw) * p_klt[1])


def _simplify_polyline(pts, epsilon):
    """Douglas-Peucker simplification of an (x, y) polyline. Near-collinear
    runs collapse to straight segments; real bends (deviation > epsilon px)
    are kept. A straight tool renders as a straight line; a bent tool keeps
    its bend. Endpoints are always preserved."""
    if pts is None or len(pts) < 3 or epsilon <= 0:
        return pts
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    simp = cv2.approxPolyDP(arr, float(epsilon), False)
    out = [(int(p[0][0]), int(p[0][1])) for p in simp]
    # approxPolyDP keeps both endpoints, but guard anyway
    if out[0] != tuple(pts[0]):
        out[0] = tuple(pts[0])
    if out[-1] != tuple(pts[-1]):
        out[-1] = tuple(pts[-1])
    return out


def _tip_state(tip_xy, W, H, margin=8):
    """'edge' when the tip dot sits within `margin` px of the image
    border (the tool tip is exiting / has exited the screen), else
    'in_frame'."""
    if tip_xy is None:
        return 'in_frame'
    x, y = tip_xy
    if x <= margin or x >= W - 1 - margin or y <= margin or y >= H - 1 - margin:
        return 'edge'
    return 'in_frame'


def _build_and_append(out_dets, cls, mb, bbox, tid,
                      tip_final, start_final, center_final, klt_ok,
                      tip_state, raw_path, r, simplify_epsilon):
    """Build the primary + secondary-jaw polylines, optionally simplify
    them (Douglas-Peucker), and append the detection dict to out_dets.
    simplify_epsilon <= 0 disables simplification. The per-frame CSV
    line is written later (after left/right labels are assigned)."""
    # Primary polyline: smoothed/re-snapped S and T as the two ends,
    # raw skeleton interior in between.
    path_xy = None
    if raw_path is not None and len(raw_path) >= 2:
        pts = [(int(round(start_final[0])), int(round(start_final[1])))]
        for (rr, cc) in raw_path[1:-1]:
            pts.append((int(cc), int(rr)))
        pts.append((int(round(tip_final[0])), int(round(tip_final[1]))))
        path_xy = _simplify_polyline(pts, simplify_epsilon)

    # Secondary jaw branches (when the jaws are open).
    aux_path_xy = []
    jaw_paths = r.debug.get("jaw_paths") or []
    for jp in jaw_paths[1:]:
        if len(jp) < 2:
            continue
        a_pts = [(int(round(start_final[0])), int(round(start_final[1])))]
        for (rr, cc) in jp[1:-1]:
            a_pts.append((int(cc), int(rr)))
        a_pts.append((int(jp[-1][1]), int(jp[-1][0])))
        aux_path_xy.append(_simplify_polyline(a_pts, simplify_epsilon))
    jaw_state = r.debug.get("jaw_state", "unknown")

    out_dets.append({
        "cls": cls, "mask": mb, "bbox": bbox, "track_id": tid,
        "tip_xy": tip_final, "start_xy": start_final,
        "center_xy": center_final, "klt_ok": klt_ok,
        "path_xy": path_xy,
        "aux_path_xy": aux_path_xy,
        "jaw_state": jaw_state,
        "tip_state": tip_state,
        "tool_label": YOLO_NAMES[cls] if 0 <= cls < len(YOLO_NAMES) else str(cls),
    })


def assign_left_right_labels(out_dets):
    """For any tool class with 2+ instances in this frame, sort those
    instances by their S (start) x-coordinate and label them left / right
    (positional index for 3+). A class with a single instance keeps its
    plain YOLO name. Mutates out_dets in place."""
    by_class = {}
    for det in out_dets:
        by_class.setdefault(det["cls"], []).append(det)
    for cls, group in by_class.items():
        name = YOLO_NAMES[cls] if 0 <= cls < len(YOLO_NAMES) else str(cls)
        if len(group) < 2:
            continue
        group.sort(key=lambda d: d["start_xy"][0])  # left -> right by S.x
        if len(group) == 2:
            sides = ["left", "right"]
        elif len(group) == 3:
            sides = ["left", "mid", "right"]
        else:
            sides = [f"#{i + 1}" for i in range(len(group))]
        for det, side in zip(group, sides):
            det["tool_label"] = f"{side} {name}"


def render(frame, dets, fi, time_sec, H, W, n_tracks, n_klt_ok, fps_inst,
           mode_label="filtered"):
    vis = frame.copy()
    if dets:
        ov = vis.copy()
        for det in dets:
            color = CLASS_COLORS_BGR[det["cls"]]
            ov[det["mask"] > 0] = color
        cv2.addWeighted(ov, 0.30, vis, 0.70, 0, dst=vis)

    for det in dets:
        if det["tip_xy"] is None or det["start_xy"] is None:
            continue
        sx = int(round(det["start_xy"][0])); sy = int(round(det["start_xy"][1]))
        tx = int(round(det["tip_xy"][0])); ty = int(round(det["tip_xy"][1]))
        path_xy = det.get("path_xy")
        if path_xy is not None and len(path_xy) >= 2:
            poly = np.array(path_xy, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly], isClosed=False,
                          color=LINE_BGR, thickness=3, lineType=cv2.LINE_AA)
        else:
            cv2.line(vis, (sx, sy), (tx, ty), LINE_BGR, 3, cv2.LINE_AA)
        # secondary jaw branches (when jaws are open) — share shaft, fork at junction
        for aux_path in (det.get("aux_path_xy") or []):
            if len(aux_path) >= 2:
                poly2 = np.array(aux_path, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [poly2], isClosed=False,
                              color=LINE_BGR, thickness=3, lineType=cv2.LINE_AA)
                jx, jy = aux_path[-1]
                cv2.circle(vis, (jx, jy), 7, LINE_BGR, 2, cv2.LINE_AA)
                cv2.circle(vis, (jx, jy), 5, TIP_BGR, -1, cv2.LINE_AA)
                cv2.putText(vis, "T2", (jx + 9, jy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            TIP_BGR, 2, cv2.LINE_AA)
        if det.get("center_xy") is not None:
            cx = int(round(det["center_xy"][0]))
            cy = int(round(det["center_xy"][1]))
            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, CENTER_BGR, -1, cv2.LINE_AA)
            cv2.putText(vis, "C", (cx + 9, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_BGR, 2, cv2.LINE_AA)
            # tool label (e.g. "left grasper") under C
            label = det.get("tool_label")
            if label:
                cv2.putText(vis, label, (cx - 18, cy + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(vis, label, (cx - 18, cy + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
        # T dot: hollow yellow ring when the tip is at/beyond the frame
        # edge ("remembered" but exiting the screen), solid red otherwise.
        if det.get("tip_state") == "edge":
            cv2.circle(vis, (tx, ty), 10, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(vis, (tx, ty), 9, (40, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, "T(edge)", (tx + 12, ty + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 255), 2,
                        cv2.LINE_AA)
        else:
            cv2.circle(vis, (tx, ty), 9, LINE_BGR, 2, cv2.LINE_AA)
            cv2.circle(vis, (tx, ty), 7, TIP_BGR, -1, cv2.LINE_AA)
            cv2.putText(vis, "T", (tx + 11, ty + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TIP_BGR, 2, cv2.LINE_AA)
        cv2.circle(vis, (sx, sy), 9, LINE_BGR, 2, cv2.LINE_AA)
        cv2.circle(vis, (sx, sy), 7, START_BGR, -1, cv2.LINE_AA)
        cv2.putText(vis, "S", (sx + 11, sy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, START_BGR, 2, cv2.LINE_AA)

    cv2.putText(vis,
                f"tool-geometry [{mode_label}]  "
                f"f={fi}  t={time_sec:5.2f}s  dets={len(dets)}  "
                f"tracks={n_tracks}  klt_ok={n_klt_ok}  "
                f"fps={fps_inst:5.1f}",
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True,
                    help="path to the laparoscopic .mp4")
    ap.add_argument("--weights-seg", type=Path,
                    default=HERE / "weights" / "best.pt",
                    help="YOLO11m-seg checkpoint; defaults to "
                         "tool_geometry/weights/best.pt")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=120.0)
    ap.add_argument("--out-dir", type=Path, default=HERE,
                    help="where to write the .mp4 + .csv; defaults to "
                         "the tool_geometry folder")
    ap.add_argument("--imgsz-seg", type=int, default=1280)
    ap.add_argument("--conf-seg", type=float, default=0.30)
    ap.add_argument("--max-area-frac", type=float, default=0.30)
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--match-iou", type=float, default=0.20)
    ap.add_argument("--track-death-frames", type=int, default=90,
                    help="keep a track alive this many idle frames before "
                         "dropping it — large enough to survive a brief "
                         "full exit of the tool from the frame")
    ap.add_argument("--tip-edge-margin", type=int, default=8,
                    help="a tip dot within this many px of the image "
                         "border is flagged tip_state=edge (exiting screen)")
    ap.add_argument("--nms-iou", type=float, default=0.15)
    ap.add_argument("--bbox-dedup-iou", type=float, default=0.30)
    ap.add_argument("--alpha-tip", type=float, default=0.20)
    ap.add_argument("--alpha-start", type=float, default=0.10)
    ap.add_argument("--alpha-center", type=float, default=0.15)
    ap.add_argument("--klt-win", type=int, default=31)
    ap.add_argument("--klt-levels", type=int, default=3)
    # tip: the published setting (ScopeScore paper, Supplementary Table: 0.6 Hz, 0.04)
    ap.add_argument("--tip-mincutoff", type=float, default=0.6)
    ap.add_argument("--tip-beta", type=float, default=0.04)
    # S sits at the near-static trocar end, so it can be smoothed hard.
    # Low mincutoff = heavy baseline smoothing; tiny beta = barely
    # relaxes during motion. Isolated to f_start — T and C unaffected.
    ap.add_argument("--start-mincutoff", type=float, default=0.1)
    ap.add_argument("--start-beta", type=float, default=0.001)
    ap.add_argument("--center-mincutoff", type=float, default=0.6)
    ap.add_argument("--center-beta", type=float, default=0.01)
    ap.add_argument("--no-oneeuro", action="store_true")
    ap.add_argument("--no-snap", action="store_true",
                    help="render at skeleton endpoints (no boundary snap)")
    ap.add_argument("--snap-max-walk", type=int, default=30)
    ap.add_argument("--tangent-k", type=int, default=5)
    ap.add_argument("--s-max-walk", type=int, default=80,
                    help="max pixels to walk the midline outward when "
                         "locating S (the tool's starting point on the "
                         "mask edge)")
    ap.add_argument("--s-straight-len", type=float, default=35.0,
                    help="arc length (px) of the trocar-end segment that "
                         "is replaced by one straight S->anchor line; the "
                         "skeleton polyline resumes past this point")
    ap.add_argument("--poly-epsilon", type=float, default=4.0,
                    help="Douglas-Peucker tolerance (px) for simplifying "
                         "the rendered polyline; near-collinear wiggles "
                         "below this collapse to straight segments")
    ap.add_argument("--pure-geometry", action="store_true",
                    help="render the raw per-frame geometry: no KLT, no "
                         "alpha-blend, no One-Euro, no re-snap, no "
                         "polyline simplification")
    ap.add_argument("--jaw-branch-ratio", type=float, default=0.7,
                    help="min branch path length as fraction of longest"
                         " branch for a secondary jaw to qualify")
    ap.add_argument("--jaw-min-branch-px", type=int, default=10,
                    help="absolute min skeleton-path length (px) from S"
                         " for a branch to qualify")
    ap.add_argument("--jaw-min-tip-sep", type=float, default=20.0,
                    help="min image-space separation (px) between the"
                         " secondary jaw tip and the primary tip for"
                         " jaw_state to be flagged as 'open'")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    s = int(round(args.start_sec))
    e = int(round(args.end_sec))
    stem = args.video.stem
    suffix = "_pure" if args.pure_geometry else ""
    out_path = args.out_dir / f"{stem}{suffix}_{s}_{e}s.mp4"
    csv_path = args.out_dir / f"{stem}{suffix}_{s}_{e}s.csv"

    print(f"[init] loading YOLO-seg from {args.weights_seg}", flush=True)
    seg = YOLO(str(args.weights_seg))

    print(f"[init] computing FOV mask + distance from {args.video.name}",
          flush=True)
    fov_mask = compute_fov_mask(args.video, n_sample=20, dark_thresh=15)
    fov_dist = compute_fov_distance(fov_mask)
    print(f"[init] FOV mask: {int(fov_mask.sum())} / {fov_mask.size} px "
          f"({100 * fov_mask.sum() / fov_mask.size:.1f}%); "
          f"max FOV-out distance = {float(fov_dist.max()):.1f} px",
          flush=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"can't open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sf = max(0, int(round(args.start_sec * fps)))
    ef = min(total, int(round(args.end_sec * fps)))
    print(f"[vid] {args.video.name} {W}x{H}@{fps:.1f}  frames {sf}..{ef} "
          f"({(ef - sf) / fps:.1f}s)", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"can't open writer {out_path}")
    print(f"[out] {out_path.name}", flush=True)

    csv_lines = ["frame,time_sec,track_id,tool_class,tool_label,"
                 "tip_x,tip_y,start_x,start_y,center_x,center_y,"
                 "jaw_state,tip_state,klt_ok"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    tracks: dict = {}
    next_id = 1
    n = 0
    tot_seg = 0
    tot_klt_ok = 0
    t0 = time.time()
    last_t = t0

    for fi in range(sf, ef):
        ok, frame = cap.read()
        if not ok:
            break
        time_sec = fi / fps
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        seg_res = seg.predict(source=frame, imgsz=args.imgsz_seg,
                              conf=args.conf_seg, iou=args.nms_iou,
                              agnostic_nms=True, device=args.device,
                              verbose=False)[0]

        seg_raw = []
        if (seg_res.boxes is not None and seg_res.masks is not None
                and len(seg_res.boxes) > 0):
            confs = seg_res.boxes.conf.cpu().numpy()
            classes = seg_res.boxes.cls.cpu().numpy().astype(int)
            xyxy = seg_res.boxes.xyxy.cpu().numpy()
            masks = seg_res.masks.data.cpu().numpy()
            order = np.argsort(-confs)
            for idx in order:
                cls = int(classes[idx])
                if cls not in TOOL_CLASSES:
                    continue
                if confs[idx] < args.conf_seg:
                    continue
                bbox = tuple(xyxy[idx].tolist())
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > args.max_area_frac * H * W:
                    continue
                m = masks[idx]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_NEAREST)
                mb = (m > 0.5).astype(np.uint8)
                if int(mb.sum()) < args.min_area:
                    continue
                seg_raw.append({"cls": cls, "conf": float(confs[idx]),
                                "bbox": bbox, "mask": mb})
        seg_kept = dedupe_bbox_only(seg_raw, args.bbox_dedup_iou)
        tot_seg += len(seg_kept)

        # match seg -> tracks by mask IoU
        for tid in list(tracks.keys()):
            if fi - tracks[tid].last_frame > args.track_death_frames:
                del tracks[tid]
        used_tracks = set()
        for sd in seg_kept:
            best_iou = args.match_iou
            best_tid = None
            for tid, tr in tracks.items():
                if tid in used_tracks:
                    continue
                if tr.tool_class != sd["cls"]:
                    continue
                iou = mask_iou(tr.last_mask, sd["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            sd["track_id"] = best_tid
            if best_tid is not None:
                used_tracks.add(best_tid)

        out_dets = []
        n_klt_ok_f = 0

        for sd in seg_kept:
            cls = sd["cls"]
            mb = sd["mask"]
            bbox = sd["bbox"]
            tid = sd["track_id"]

            r = detect_tip_start(
                mask=mb, fov_dist=fov_dist, frame_idx=fi,
                image_shape=(H, W), tool_class=cls,
                min_area=args.min_area,
                snap_to_boundary=not args.no_snap,
                tangent_k=args.tangent_k,
                snap_max_walk=args.snap_max_walk,
                s_max_walk=args.s_max_walk,
                s_straight_len=args.s_straight_len,
                jaw_branch_ratio=args.jaw_branch_ratio,
                jaw_min_branch_px=args.jaw_min_branch_px,
                jaw_min_tip_separation_px=args.jaw_min_tip_sep,
            )
            if r.tip_xy is None or r.start_xy is None or r.conf <= 0:
                continue
            tip_geom = (float(r.tip_xy[0]), float(r.tip_xy[1]))
            start_geom = (float(r.start_xy[0]), float(r.start_xy[1]))
            center_geom = ((float(r.center_xy[0]), float(r.center_xy[1]))
                           if r.center_xy is not None else None)

            if args.pure_geometry:
                # Raw per-frame geometry: no KLT, no blend, no One-Euro,
                # no re-snap. The dots are exactly what the detector
                # produced this frame.
                tip_final = tip_geom
                start_final = start_geom
                center_final = center_geom
                klt_ok = False
                tip_state = _tip_state(tip_final, W, H,
                                       margin=args.tip_edge_margin)
                if tid is None:
                    tid = next_id
                    next_id += 1
                    tracks[tid] = Track(
                        track_id=tid, tool_class=cls, last_frame=fi,
                        last_bbox=bbox, last_mask=mb,
                        last_yolo_conf=sd["conf"],
                        last_tip=tip_final, last_start=start_final,
                        last_center=(center_final if center_final
                                     else tip_final),
                        last_gray=cur_gray,
                        last_in_frame_tip=(tip_final
                                           if tip_state == 'in_frame'
                                           else None),
                    )
                else:
                    tr = tracks[tid]
                    tr.last_frame = fi
                    tr.last_bbox = bbox
                    tr.last_mask = mb
                    tr.last_yolo_conf = sd["conf"]
                    tr.last_gray = cur_gray
                    tr.last_tip = tip_final
                    tr.last_start = start_final
                    if center_final is not None:
                        tr.last_center = center_final
                    if tip_state == 'in_frame':
                        tr.last_in_frame_tip = tip_final
                raw_path = r.debug.get("main_path") if r is not None else None
                _build_and_append(out_dets, cls, mb, bbox, tid,
                                  tip_final, start_final, center_final,
                                  klt_ok, tip_state, raw_path, r,
                                  simplify_epsilon=0.0)
                continue

            # KLT
            klt_tip = klt_start = klt_center = None
            klt_ok = False
            if tid is not None and tracks[tid].last_gray is not None:
                klt_tip, ok_t = klt_track_one(
                    tracks[tid].last_gray, cur_gray,
                    tracks[tid].last_tip,
                    args.klt_win, args.klt_levels)
                klt_start, ok_s = klt_track_one(
                    tracks[tid].last_gray, cur_gray,
                    tracks[tid].last_start,
                    args.klt_win, args.klt_levels)
                klt_center, ok_c = klt_track_one(
                    tracks[tid].last_gray, cur_gray,
                    tracks[tid].last_center,
                    args.klt_win, args.klt_levels)
                klt_ok = ok_t and ok_s
                if klt_ok:
                    tot_klt_ok += 1
                    n_klt_ok_f += 1

            # blend
            #   TIP / CENTER: KLT-dominated (they move and have texture).
            #   START: NO KLT. The trocar exit sits next to the textureless
            #   black void, where KLT drifts inward up the shaft. The raw
            #   per-frame s_border (min FOV-distance mask pixel) is already
            #   a clean, stable measurement, so we pass it straight to
            #   One-Euro below.
            if tid is None:
                tip_smooth = tip_geom
                start_smooth = start_geom
                center_smooth = center_geom
            else:
                tip_smooth = alpha_blend(klt_tip if klt_ok else None,
                                         tip_geom, args.alpha_tip) \
                             or tracks[tid].last_tip
                start_smooth = start_geom
                if center_geom is not None:
                    center_smooth = alpha_blend(klt_center,
                                                center_geom,
                                                args.alpha_center) \
                                    or tracks[tid].last_center
                else:
                    center_smooth = tracks[tid].last_center

            if tid is None:
                if args.no_oneeuro:
                    f_tip = f_start = f_center = None
                    smoothed_tip = tip_smooth
                    smoothed_start = start_smooth
                    smoothed_center = center_smooth
                else:
                    f_tip = OneEuroVec2(freq=fps,
                                        mincutoff=args.tip_mincutoff,
                                        beta=args.tip_beta)
                    f_start = OneEuroVec2(freq=fps,
                                          mincutoff=args.start_mincutoff,
                                          beta=args.start_beta)
                    f_center = OneEuroVec2(freq=fps,
                                           mincutoff=args.center_mincutoff,
                                           beta=args.center_beta)
                    smoothed_tip = tuple(float(v) for v in
                                         f_tip(np.array(tip_smooth,
                                                        dtype=np.float64)))
                    smoothed_start = tuple(float(v) for v in
                                           f_start(np.array(start_smooth,
                                                            dtype=np.float64)))
                    smoothed_center = (tuple(float(v) for v in
                                             f_center(np.array(center_smooth,
                                                               dtype=np.float64)))
                                       if center_smooth is not None else None)
                tid = next_id
                next_id += 1
                tracks[tid] = Track(
                    track_id=tid, tool_class=cls, last_frame=fi,
                    last_bbox=bbox, last_mask=mb, last_yolo_conf=sd["conf"],
                    last_tip=smoothed_tip, last_start=smoothed_start,
                    last_center=smoothed_center if smoothed_center else smoothed_tip,
                    last_gray=cur_gray,
                    f_tip=f_tip, f_start=f_start, f_center=f_center,
                )
                tip_final = smoothed_tip
                start_final = smoothed_start
                center_final = smoothed_center
            else:
                tr = tracks[tid]
                tr.last_frame = fi
                tr.last_bbox = bbox
                tr.last_mask = mb
                tr.last_yolo_conf = sd["conf"]
                tr.last_gray = cur_gray
                if tr.f_tip is not None:
                    tip_final = tuple(float(v) for v in
                                      tr.f_tip(np.array(tip_smooth,
                                                        dtype=np.float64)))
                else:
                    tip_final = tip_smooth
                if tr.f_start is not None:
                    start_final = tuple(float(v) for v in
                                        tr.f_start(np.array(start_smooth,
                                                            dtype=np.float64)))
                else:
                    start_final = start_smooth
                if tr.f_center is not None and center_smooth is not None:
                    center_final = tuple(float(v) for v in
                                         tr.f_center(np.array(center_smooth,
                                                              dtype=np.float64)))
                elif center_smooth is not None:
                    center_final = center_smooth
                else:
                    center_final = tr.last_center
                tr.last_tip = tip_final
                tr.last_start = start_final
                if center_final is not None:
                    tr.last_center = center_final

            # TIP re-snap: walk along the polyline's local outward tangent
            # to the last in-mask pixel, undoing KLT / One-Euro drift.
            # START is NOT re-snapped: the detector already returns S as
            # the midline ∩ FOV-boundary point, and One-Euro only lightly
            # smooths it. A mask-pixel re-snap would pull S off the
            # midline and re-introduce the start branch.
            raw_path = r.debug.get("main_path") if r is not None else None
            out_tan_tip = _outward_tangent_xy(raw_path, end='tip')
            if out_tan_tip is not None:
                tip_final = _resnap_to_boundary(tip_final, out_tan_tip,
                                                mb, max_walk=60)
            # propagate the values back into the track so KLT starts from
            # a valid pixel next frame.
            tracks[tid].last_tip = tip_final
            tracks[tid].last_start = start_final

            # Tip-out-of-frame memory: flag when the tip dot reaches the
            # image border, and remember the last tip seen fully inside.
            tip_state = _tip_state(tip_final, W, H,
                                   margin=args.tip_edge_margin)
            if tip_state == 'in_frame':
                tracks[tid].last_in_frame_tip = tip_final

            # Build + simplify the polyline(s) and append the detection.
            _build_and_append(out_dets, cls, mb, bbox, tid,
                              tip_final, start_final, center_final,
                              klt_ok, tip_state, raw_path, r,
                              simplify_epsilon=args.poly_epsilon)

        # Assign left/right labels to any same-class tool group, then
        # write the per-frame CSV lines.
        assign_left_right_labels(out_dets)
        for det in out_dets:
            tf = det["tip_xy"]
            sfp = det["start_xy"]
            cfp = det.get("center_xy")
            cx_csv = (f"{cfp[0]:.2f},{cfp[1]:.2f}"
                      if cfp is not None else ",")
            csv_lines.append(
                f"{fi},{time_sec:.4f},{det['track_id']},{det['cls']},"
                f"{det['tool_label']},"
                f"{tf[0]:.2f},{tf[1]:.2f},{sfp[0]:.2f},{sfp[1]:.2f},"
                f"{cx_csv},{det['jaw_state']},{det['tip_state']},"
                f"{int(det['klt_ok'])}"
            )

        now = time.time()
        fps_inst = 1.0 / max(1e-6, now - last_t)
        last_t = now
        vis = render(frame, out_dets, fi, time_sec, H, W,
                     len(tracks), n_klt_ok_f, fps_inst,
                     mode_label=("pure-geometry" if args.pure_geometry
                                 else "filtered"))
        writer.write(vis)

        n += 1
        if n % 100 == 0 or n == 1:
            ips = n / max(1e-6, time.time() - t0)
            eta = (ef - sf - n) / max(1e-6, ips)
            print(f"  [run] {n}/{ef - sf}  {ips:.1f} fps  "
                  f"seg={tot_seg}  klt_ok={tot_klt_ok}  "
                  f"tracks={len(tracks)}  eta={eta:.0f}s", flush=True)

    writer.release()
    cap.release()
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"\n[done] {n} frames in {time.time() - t0:.1f}s -> {out_path}")
    print(f"[csv]  {csv_path}")
    print(f"[stats] seg={tot_seg}  klt_ok={tot_klt_ok}  "
          f"end_fps={n / max(1e-6, time.time() - t0):.1f}")


if __name__ == "__main__":
    main()
