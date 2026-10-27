# ScopeScore — standalone tip / start / center / jaw tracker for laparoscopic videos

ScopeScore (Sajjadi, 2026, manuscript submitted) tracks surgical instruments in laparoscopic video. Live Tracker does
**not** run this model: it tracks mice, not instruments. Live Tracker re-uses ScopeScore's **methods**, re-written for
mice (see [How Live Tracker uses ScopeScore](#how-live-tracker-uses-scopescore)). The model, code and settings are kept
here so the source of those methods can be checked.

This folder is fully self-contained. Move it anywhere and run it on any
laparoscopic .mp4 — no external code, no project-specific paths.

## What it does

For every frame of an endoscopic video, for every tool the YOLO
segmenter finds, it produces:

- **T** (tip) — the free working end, snapped to the visible tool
  silhouette
- **S** (start) — the trocar end, placed on the tool's centreline via a
  PCA-fit straight axis
- **C** (center) — the geodesic midpoint along the tool's rendered axis
- **T2** (secondary jaw tip) — only when a grasper / scissors / clipper
  / bipolar is visibly open
- **jaw_state** — `'open'` / `'closed'` for jawed classes; `'n/a'`
  for hook / irrigator
- A polyline that follows the tool's actual curve: one straight S →
  anchor segment that kills the noisy trocar-end medial axis, followed
  by the real skeleton out to T (so open jaws fork at the real junction
  near the tip, not at S)
- A per-frame CSV log of every tracked point (14 columns)

The pipeline is two layers:

1. **Per-frame geometry** (`tip_detector.py`) — picks **T first** as
   the skeleton endpoint farthest from the FOV-out region (the working
   end is deep in the visible field), then **S** as the endpoint
   geodesically farthest from T (end of the longest skeleton run = the
   main shaft's trocar end), then rebuilds the polyline as
   `[S, anchor, …real skeleton…, T]`. No learned keypoints, no pose
   model, no temporal carry-forward inside the detector. See
   `methods.md` for the full algorithm.
2. **Temporal smoothing** (`run_pipeline.py`) — KLT optical flow on T
   and C (NOT S), alpha-blend with the raw geometry, One-Euro per
   endpoint, per-frame tip re-snap to the mask boundary, and
   Douglas-Peucker simplification of the rendered polyline.

## Folder layout

```
models/scopescore/
├── run_pipeline.py          driver — what you actually run
├── tip_detector.py          per-frame geometric detector
├── fov_utils.py             FOV mask + distance transform helpers
├── skeleton_utils.py        clean_mask, build_skeleton_graph, …
├── causal_smoother.py       OneEuroVec2 filter (used for T, S, C smoothing)
├── build_grid.py            side-by-side pure-geometry vs filtered stitcher
├── build_masks_only.py      masks-only rendering helper
├── build_trajectory_trail.py  trajectory-trail rendering helper
├── methods.md               the full per-frame algorithm and design rationale
├── configs/                 training settings (train_args.yaml), training curve (train_results.csv), data.yaml
├── weights/
│   └── best.pt              YOLO11m-seg weights (CholecInstanceSeg, epoch 52)
├── requirements.txt         pip dependencies
└── README.md                this file
```

## Setup (once)

```bash
# any Python >= 3.10 venv
pip install -r requirements.txt
```

## Run on any laparoscopic video

```bash
python run_pipeline.py --video /path/to/your_video.mp4
```

Defaults: `--start-sec 0`, `--end-sec 120`, `--imgsz-seg 1280`,
`--conf-seg 0.30`, `--device cuda:0`. Output is written next to
`run_pipeline.py`. Pick a range and an output folder:

```bash
python run_pipeline.py \
    --video /path/to/your_video.mp4 \
    --start-sec 30 --end-sec 60 \
    --out-dir /path/to/results
```

### Outputs

- `<video-stem>_<start>_<end>s.mp4` — annotated video (mask overlay,
  polyline, S / T / C / T2 dots, tool labels, HUD)
- `<video-stem>_<start>_<end>s.csv` — per-frame log with columns:
  `frame, time_sec, track_id, tool_class, tool_label, tip_x, tip_y,
  start_x, start_y, center_x, center_y, jaw_state, tip_state, klt_ok`

CSV semantics:

- `tool_label` — plain class name; or `left <class>` / `right <class>`
  (or `left/mid/right`, or `#1..#N` for 4+) when several instances of
  the same class are in the frame, sorted left → right by S.x.
- `jaw_state` — `open` / `closed` for grasper, bipolar, scissors,
  clipper; `n/a` for hook / irrigator; `unknown` if the detector can't
  decide.
- `tip_state` — `in_frame` / `edge`. `edge` means T is within
  `--tip-edge-margin` (8 px) of the image border — the tool is exiting
  / has exited the screen. The track stores `last_in_frame_tip` so a
  brief exit doesn't lose the tool.
- `klt_ok` — 1 if KLT optical flow tracked both T and S into this frame.

## Useful flags

- `--weights-seg /other/path/best.pt` — drop in a different YOLO seg checkpoint.
- `--device cuda:0` / `cuda:1` / `cpu`.
- `--imgsz-seg 1280` (or `640` for faster inference).
- `--conf-seg 0.30` — mask confidence threshold. Lower it (e.g. 0.15)
  if a particular tool class is under-detected on a new domain.
- `--no-snap` — render at raw skeleton endpoints (skip the tip boundary snap).
- `--no-oneeuro` — disable temporal smoothing.
- `--s-straight-len 35` — arc length (px) of the straight S → anchor
  segment that replaces the noisy trocar-end medial axis.
- `--poly-epsilon 4.0` — Douglas-Peucker tolerance (px) for simplifying
  the rendered polyline. 0 disables simplification.
- `--jaw-min-tip-sep 20` — minimum image-space px between T and T2
  for `jaw_state == 'open'`. Raise to be stricter; lower to flag tiny openings.
- `--track-death-frames 90` — keep a track alive this many idle frames
  (≈ 3.75 s @ 24 fps) before dropping it. Survives brief tool exits.
- `--pure-geometry` — bypass ALL filters (no KLT, no alpha-blend, no
  One-Euro, no re-snap, no polyline simplification). Useful for
  debugging the raw per-frame geometry.

## YOLO checkpoint

`weights/best.pt` was trained on the **CholecInstanceSeg** dataset
(Cholec80 domain) with the **YOLO11m-seg** architecture, 22.4 M params,
43 MB here (the same weights without the training state; the training checkpoint was 129 MB). It outputs 6 tool classes: grasper, bipolar, hook,
scissors, clipper, irrigator.

Best val metrics (epoch 52, 2 696 images, 4 931 instances):
Box mAP50 = 0.639, Mask mAP50 = 0.469.

If you have a different lap-surgery domain (robotic, other body
cavities) drop in any Ultralytics `yolo*-seg` checkpoint with a
compatible class set via `--weights-seg`.

## Smoothing defaults

The One-Euro params are asymmetric — T is responsive (it actually
moves), S is heavily damped (the trocar end is near-static), C is
medium:

| endpoint | mincutoff | beta  | KLT on? |
|----------|-----------|-------|---------|
| T (tip)    | 0.6       | 0.04  | yes     |
| S (start)  | 0.1       | 0.001 | **no**  |
| C (center) | 0.6       | 0.01  | yes     |

S has no KLT because the trocar exit sits next to the textureless
black void; KLT drifts S inward up the shaft. The raw per-frame S is
already stable (the PCA axis is fit over the whole shaft, so per-frame
wiggle averages out), and One-Euro alone is enough.

## What's NOT in this folder

- A Python interpreter (use any 3.10+ venv).
- Input videos (you provide them).

That's it. No external repo paths, no shared state.

## Settings: the published values

The defaults in `run_pipeline.py` are the settings of the paper (Supplementary Table): segmentation confidence 0.30,
non-maximum suppression 0.15, track overlap 0.20, a track ends after 90 idle frames, Lucas-Kanade window 31 x 31 with
3 levels, flow weight 0.20 for the tip and 0.15 for the centre, One-Euro tip 0.6 Hz / 0.04, start 0.1 / 0.001,
centre 0.6 / 0.01.

The paper's **tolerance** step (the tip is followed by a point that ignores changes well below sigma = 5 px, about 1% of
the picture height) and the camera-motion compensation belong to the scoring stage, which is not in this folder.

## How Live Tracker uses ScopeScore

| ScopeScore method | In Live Tracker (adapted to mice) | Setting there |
|---|---|---|
| Flag points that are missing, jumping, frozen or low confidence; treat missing as missing | `src/live_tracker/quality.py` | its own thresholds for mice (`FeatureParams`) |
| One-Euro smoothing, past frames only | `src/live_tracker/filters.py`, `features.py` | causal, reset per session, mouse and long gap |
| Motion measures of a tracked point (speed, turning, pauses) | `src/live_tracker/features.py` | per mouse and per pair of mice |
| Flow mix, One-Euro and tolerance on every keypoint | `src/live_tracker/point_stabilizer.py` (off by default) | flow weight 0.20, One-Euro 0.6 Hz / 0.04, sigma = 1% of the picture height: the published tip settings |

What was not carried over: the surgical thresholds and any clinical meaning. Measured effect of the ScopeScore-inspired
motion features on the mouse behavior classifier: `docs/motion_engine_results.md` (Model A vs Model B). The ScopeScore
settings on mouse keypoints: `docs/pose_model.md` (mice move faster at 30 frames per second than instruments at 24, so
the filter lags).

## See also

- `methods.md` — full per-frame algorithm and design rationale.
- `configs/train_args.yaml`, `configs/train_results.csv` — how the weights were trained.
- Sajjadi, A. (2026). ScopeScore: A video processing model for surgery [Manuscript submitted].
