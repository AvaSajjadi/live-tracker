# Methods — tool-geometry tip tracking pipeline

The current pipeline lives in `tool_geometry/`. Three modules:

- `tool_geometry/tip_detector.py` — per-frame geometric detector
  (tip, start, center, secondary jaw, polyline).
- `tool_geometry/fov_utils.py` — FOV mask + distance transform, computed
  once per video.
- `tool_geometry/run_pipeline.py` — driver: YOLO-seg + the detector +
  KLT + alpha-blend + One-Euro + boundary re-snap + render.

It replaces every earlier "which end is the tip?" heuristic with a
single geometric rule grounded in the endoscope's circular field of
view (FOV).

## Key insight

Every surgical tool enters through a trocar that sits **outside** the
endoscope's circular FOV; the working end is **deep inside** the FOV.
So the skeleton endpoint farthest from the FOV-out region is, by
construction, the tip, and the trocar end is the endpoint geodesically
farthest from the tip (= end of the longest skeleton run).

No pose model, no border heuristic, no temporal carry-forward inside
the detector.

## Why "tip first, then start" (and not "start first")

Earlier versions picked S first (skeleton endpoint nearest FOV-out) and
ran BFS from S to find T. That fails on a blobby mask near the trocar:
a noisy protrusion off the shaft becomes an endpoint very close to the
FOV-out region and wins the "nearest FOV-out" vote, dragging S onto a
corner of the mask. The line from that S then makes a sharp V-bend at
the very start.

The current rule inverts the order:

1. **Tip first.** `tip_guess` = endpoint with the **largest** distance
   to FOV-out. The working end is deep in the visible field, so this is
   robust to junk endpoints near the trocar.
2. **Start from tip.** `start_ep` = endpoint geodesically farthest
   from `tip_guess` (BFS through the whole skeleton). That is the end
   of the longest skeleton run, which is the main shaft's trocar end.
   A blobby protrusion near the trocar is a SHORT side-spur, so it
   cannot win "farthest from tip" against the long main shaft — it
   becomes an unused branch instead of an endpoint that drags S.

## Precomputed once per video — `fov_utils.py`

```
fov_mask  = compute_fov_mask(video, n_sample=20, dark_thresh=15)
            sample 20 frames spread across the video
            per pixel: bright if not "dark in >= 50% of samples"
            keep largest bright connected component
            fill internal holes that don't touch the image border
fov_dist  = cv2.distanceTransform(fov_mask, cv2.DIST_L2, 3)
            -> per-pixel float distance to the nearest FOV-out pixel
```

## End-to-end per-frame algorithm (`run_pipeline.py` driving `detect_tip_start`)

```
read frame -> grayscale (for KLT)
│
├── YOLO11m-seg (cholec-trained, weights/best.pt)
│     bbox + class + polygon mask, conf >= --conf-seg (default 0.30)
│     class-agnostic NMS @ --nms-iou (0.15)
│     bbox dedup @ --bbox-dedup-iou (0.30) — drops near-duplicate boxes
│     reject tiny masks (< --min-area = 300 px) and huge boxes
│       (> --max-area-frac * H*W = 30 % of frame)
│
├── match seg -> existing tracks by mask IoU >= --match-iou (0.20),
│     same class. Track dies after --track-death-frames (30) idle frames
│     — long enough to survive a brief full exit of the tool from frame.
│
├── tip_detector.detect_tip_start(mask, fov_dist, tool_class, ...)
│     1. clean_mask  -> largest CC, close holes (radius 3), area >= 300 px
│     2. skeletonize -> 1-px medial axis
│     3. build_skeleton_graph -> graph + all endpoints
│     4. tip_guess  = argmax over endpoints of fov_dist[ep]
│     5. start_ep   = geodesically farthest endpoint from tip_guess (BFS)
│     6. BFS from start_ep to ALL endpoints -> {ep: (depth, path)}
│        primary path = longest reachable path (start_ep -> primary_tip)
│        secondary jaw candidates = endpoints with
│           depth >= jaw_branch_ratio * max_depth   (default 0.7)
│           AND depth >= jaw_min_branch_px           (default 10 px)
│           AND image-space dist >= jaw_min_tip_sep  (default 20 px)
│        T2 / jaw_state ONLY for jawed classes
│        (grasper, bipolar, scissors, clipper);
│        hook / irrigator -> jaw_state = 'n/a' (a hook's curved tip
│        makes a branchy skeleton but that branch is not a jaw).
│     7. T placement:
│           - take real skeleton tip
│           - walk outward along local tangent (from BFS k=5 steps inward)
│             to the last in-mask pixel (snap_max_walk = 30 px default)
│           - that's T (snap_to_boundary default ON)
│     8. S placement — _straight_axis_S:
│           - PCA-fit ONE straight axis to the whole skeleton main path
│           - orient axis so +axis points toward the tip
│           - take trocar-side extreme projected onto the axis line
│           - walk OUTWARD along the axis to the mask edge
│             (s_max_walk = 80 px default)
│           - that's S — on the tool's true centreline, not on a
│             local mask wiggle near the trocar
│     9. rebuild polyline — straight trocar segment + real skeleton:
│           - anchor = skeleton point ~s_straight_len px (default 35)
│             of arc length past the trocar end (capped at 40 % of total
│             path length so short tools keep most of their skeleton)
│           - rendered path = [S, anchor, ...real skeleton..., T]
│           - jaw branches share the [S, anchor] prefix and fork at the
│             real junction near the tip (NOT at S)
│    10. C = geodesic midpoint of the rebuilt polyline
│           (a point ON the line, not the bounding-box centre).
│
├── KLT optical flow from prev-frame
│     ON: tip and center (texture-rich, real motion)
│     OFF: start. KLT drifts S inward up the shaft — the trocar exit
│     sits next to the textureless black FOV-out void with no usable
│     features. The detector's per-frame S is already a clean,
│     stable measurement, so it is passed straight to One-Euro.
│
├── alpha-blend (KLT prediction, raw detector position)
│     tip:    alpha_raw = 0.20  (KLT dominates motion)
│     center: alpha_raw = 0.15
│     start:  -- not applicable (no KLT) --
│
├── One-Euro filter per track per endpoint
│     tip:    freq=fps, mincutoff=1.0,  beta=0.05    (responsive)
│     start:  freq=fps, mincutoff=0.1,  beta=0.001   (heavy, trocar near-static)
│     center: freq=fps, mincutoff=0.6,  beta=0.01    (medium)
│
├── tip re-snap: walk the smoothed tip along the local outward tangent
│     back onto the mask boundary (max 60 px). Undoes any drift that
│     KLT + One-Euro introduced. S is NOT re-snapped — pulling S onto a
│     mask-edge pixel would re-introduce the V-bend the PCA axis was
│     designed to avoid.
│
├── Douglas-Peucker polyline simplification (--poly-epsilon = 4.0 px)
│     Near-collinear runs collapse to straight segments; real bends
│     (deviation > epsilon) are kept; endpoints always preserved.
│     A straight tool renders as one straight line; a bent tool keeps
│     its bend.
│
├── tool-label disambiguation
│     For any class with 2+ instances in this frame, sort by S.x and
│     label "left <class>" / "right <class>" (or "left/mid/right" for
│     3 instances, "#1..#N" for 4+).
│
├── tip-edge memory
│     If T is within --tip-edge-margin (8 px) of the image border ->
│     tip_state = 'edge' (rendered as a hollow yellow ring "T(edge)").
│     Track stores last_in_frame_tip so the tool isn't lost during a
│     brief exit.
│
└── render
    - mask overlay (class colour, alpha 0.30)
    - white anti-aliased polyline (thickness 3) along the simplified
      [S, anchor, ...skeleton..., T] path
    - secondary jaw polylines when jaw_state == 'open': straight prefix
      to anchor + skeleton from anchor to T2
    - T: red filled circle ø 7 + white outline ø 9, label "T"
         (yellow hollow "T(edge)" when tip is at the frame border)
    - S: blue filled circle ø 7 + white outline ø 9, label "S"
    - C: yellow filled circle ø 4 + white outline ø 6, label "C"
    - T2: small red filled circle + white outline, label "T2"
    - tool label (e.g. "left grasper") below C
    - HUD bottom-left: pipeline name [filtered | pure-geometry],
      frame, time, dets, tracks, klt_ok, instantaneous fps
```

## CSV log

14 columns per detection per frame:

`frame, time_sec, track_id, tool_class, tool_label, tip_x, tip_y,
start_x, start_y, center_x, center_y, jaw_state, tip_state, klt_ok`

## Why this architecture

Each piece replaces a specific failure mode of an earlier version
(see `TODO.md` for the chronological history).

- **FOV-distance + tip-first ordering.** The original
  `RobustTipDetector` measured distance to the rectangular image edge,
  which is wrong on Cholec80 (the FOV is a circle inscribed in a black
  rectangle — the trocar end can sit 150+ px from any rectangular
  edge). Using FOV-distance fixes that. Picking the **tip** first
  (largest FOV-distance) fixes the "blobby mask near trocar wins"
  failure of the older "start first (smallest FOV-distance)" rule.
- **PCA axis for S.** S placed as a raw skeleton endpoint always landed
  on a mask corner / protrusion → V-bend right after S. PCA-fitting one
  straight axis to the whole skeleton averages out the trocar-end
  wiggle, so S lands on the tool's true centreline.
- **Straight S → anchor + real skeleton.** Earlier we tried a fully
  straight S→T chord; it broke open graspers (both jaws started at S
  and the line cut across the tissue). The settled design keeps the
  skeleton polyline — so jaws fork at the real junction near the tip —
  and only replaces the noisy first ~35 px of arc length with one
  straight S→anchor segment.
- **JAW_CLASSES gate for T2.** Hook and irrigator have no jaws, but
  their masks fork in the skeleton (a hook's curved tip; an irrigator's
  splash region). Restricting T2 to {grasper, bipolar, scissors,
  clipper} eliminates spurious "open jaw" flags on those tools.
- **No KLT on S.** The textureless black void next to the trocar exit
  has no features; KLT drifts S inward up the shaft each frame. The
  raw per-frame S is already stable (PCA over the whole shaft is
  insensitive to per-frame wiggle), so we pass it straight to One-Euro.
- **One-Euro per endpoint.** Per-frame mask wobble produces 5–15 px of
  jitter at the endpoints. One-Euro with asymmetric params (responsive
  tip, heavily damped start, medium center) gives smooth motion without
  visible lag at the tip when the tool moves fast.
- **Tip re-snap each frame.** KLT + One-Euro can drift the tip dot
  inside the mask or just past the mask edge during fast motion. A
  per-frame outward-tangent walk to the boundary undoes that drift so
  the dot stays glued to the visible metal.
- **Douglas-Peucker on the polyline.** Sub-pixel wiggle in the
  skeleton interior creates a visible "ladder" between frames. DP
  smooths a straight shaft to one line segment while keeping real
  curvature, so the rendered line is steady between frames.

## Data — CholecInstanceSeg segmenter retrain

`cholecinstanceseg_yolo/` holds the YOLO-seg dataset derived from
CholecInstanceSeg (Nature Scientific Data 2025, arXiv 2406.16039).
The trained checkpoint is vendored at
`tool_geometry/weights/best.pt`.

### Source

- **CholecInstanceSeg JSON annotations** in LabelMe polygon format
  (downloaded ZIP, extracted to
  `cholecinstanceseg/cholecinstanceseg/{train,val,test}/VID{NN}_{source}/ann_dir/*.json`).
- **Frames extracted** from Cholec80 source mp4s at
  `<JEDeye>/modules/phase_detection/3D-ResNet/data/videos/videoNN.mp4`.

### Filename → frame mapping (empirically verified by `--probe`)

The JSON filename encodes `(source, video_num, frame_idx)`:

- `seg8k_video37_000772.json` → source = CholecSeg8k. **Frame idx is
  clip-relative; no source-frame mapping exists in the filename.** Skipped.
- `t50_VID32_001320.json` → source = CholecT50. **Frame idx is a 1-fps
  sample number; source mp4 frame = idx × 25.**
- `t80_VID34_000720.json` → source = Cholec80 sparse. **Frame idx is a
  1-fps sample number; source mp4 frame = idx × 25.**

### Holdouts

- `train/VID06_t50_sparse/` is in `EXCLUDE_TRAIN_DIRS` (held out because
  evaluation uses `vid06.mp4`).
- `test/VID07_*` is in the dataset's own test split (never seen).
- Snare instances are dropped (`SKIP_LABELS`) to keep the 6-class
  taxonomy matching the older `yolo_phakir` checkpoint.

### Result

| stat | value |
|---|---|
| Total annotations | 41,933 |
| seg8k (skipped) | 8,080 |
| t50 + t80 collected after VID06 holdout | 31,021 |
| Usable polygons → samples written | **22,043** (train 19,347, val 2,696) |
| Disk | 2.0 GB |
| Classes | grasper, bipolar, hook, scissors, clipper, irrigator (nc = 6) |

## Training

Hyperparameters (matching `yolo_phakir` where sensible):

- model: `yolo11m-seg.pt` (Ultralytics auto-download)
- epochs: 100, patience: 20
- imgsz: 1280
- batch: 8 requested → **OOM-fallback to batch=2** because
  `llama-server` was holding 22 GB of the 32 GB VRAM (Ultralytics auto-retry)
- AMP on, optimizer auto (AdamW), workers 6, device 0
- LR / loss weights: Ultralytics defaults

### Run summary

- Phase 1: 22 epochs, then died because SSH disconnect propagated
  SIGHUP through the background job.
  **Lesson: wrap long jobs with `setsid nohup … &` and `disown`.**
- Phase 2 (resumed via `resume=True`): started epoch 23 with
  `setsid nohup`, ran to epoch 73 where early-stop fired (20 epochs
  without improvement since epoch 52).
- Best metrics (val on 2,696 images, 4,931 instances):
  - Box mAP50      = **0.639** (epoch 32)
  - Box mAP50-95   = **0.374** (epoch 71)
  - Mask mAP50     = **0.469** (epoch 52, best.pt saved here)
  - Mask mAP50-95  = **0.216** (epoch 71)
- best.pt: `cholecinstanceseg_yolo/runs/yolo11m-seg/weights/best.pt`
  (129 MB, 254 layers, 22.4 M params).
- Vendored copy for portability: `tool_geometry/weights/best.pt`.

## Empirical comparison: PhaKIR weights vs cholec weights (vid06 1200–1320 s)

Same pipeline, only the YOLO-seg weights swapped:

| | PhaKIR-trained | cholec-trained (ep 52) |
|---|---|---|
| seg detections | 5132 | **6387** (+24 %) |
| tracks locked | 60 | **28** (much fewer track births = stabler) |
| KLT success | 78 % | **89 %** |
| visual quality | jittery, irrigator often missed | irrigator caught, masks tighter, line steadier |

Conclusion: cholec-domain training is the right unlock.
