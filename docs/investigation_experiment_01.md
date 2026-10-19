# Investigation experiment 01: feasibility on CalMS21 validation data

> **EXPERIMENTAL. Not reviewed by the team. Descriptive counts only.**
> Branch `syed/experimental-investigation`, Stage 1 of the Investigation-mode plan
> (revised after two independent reviews), 2026-10-19. Nothing here is a scientific
> result, and no value below is a validated setting: everything that is varied
> (window length, baseline clearance, history length, coverage bins) is a reporting
> parameter, listed as provisional in the generated settings.

Question under study (later): *Does approach tend to precede attack?*
This stage only asks: **can the validation data we have support that question,
and what has to be decided first?**

## 1. Data boundary

- **Source**: the official CalMS21 Task 1 train file on the external drive (path,
  size and sha256 in the generated section). The test file is never opened.
- **Split**: `splits/split_manifest.json` (56 train, 14 validation, 19 test sessions).
  Validation sessions live inside the official train file.
- **Extraction** (`python -m live_tracker.investigation_data`): requests are checked
  against the manifest before the source is touched (validation only; test, train and
  unlisted IDs refused). The source is read once, read-only, hashed from the same
  bytes that are parsed, and refused if it contains any test session. Every record is
  validated before anything is published: Igor's `load_calms21` must accept it, and in
  addition `metadata.vocab` must be present and equal `calms21.VOCAB`, and every
  annotation must be a plain int label id (the loader's `dtype=int` cast would turn
  0.5 into 0). The source must contain every validation session, so the label summary
  covers the whole split. Writers and readers of a cache directory are serialized by an
  OS lock (`fcntl.flock` / `msvcrt.locking`). Each payload is serialized once, hashed from
  those exact bytes and published atomically through a unique temporary file; files only
  count once listed in a versioned index bound to the source sha256, the manifest sha256
  and each payload sha256. A cache is reused only after the source content is hashed
  again in full (path, size and mtime are not proof of identity; cost in §9); otherwise
  it is rebuilt.
- **Analysis** (`scripts/investigation_feasibility.py`) accepts only sessions in that
  index. Under the same lock it re-checks the manifest hash; reads each session only from
  its canonical file name directly inside the cache directory (absolute paths, traversal,
  other names and symlinks are refused before opening); checks each file hash, the frame
  count and the session IDs **inside** each file against the manifest's validation list
  before Igor's loader reads the annotations. The per-session label summary in the index
  is not trusted: it must cover exactly the manifest's validation sessions, be
  well-formed (positive frame counts, non-negative integer run counts for exactly the four
  labels, no more runs than frames), and equal the counts recomputed from every extracted
  record. Summary entries of sessions that were not extracted can only be checked for
  structure (§7). Unlisted files in the directory are never read (they are listed).
- **Sessions**: mouse005, mouse009, mouse021, chosen before looking at labels (the first
  three validation IDs in sorted order).
- **Labels**: validation annotations (annotator 1) only. Igor's classifier
  (`igor/day2`, PR #15) is not used in this stage.

## 2. Preprocessing policy for temporal evidence (provisional)

1. `quality.flag_tracking` with the CalMS21 settings of `calms21.flag_quality` on
   `igor/day2` (frame size 1024×570, `low_conf` below `SUGGESTED_MIN_SCORE` = 0.15).
   Jump detection stays off (`max_speed_px_s=None`, the default); choosing a speed
   limit is Ava's decision and has not been made.
2. **No `quality.interpolate_short_gaps`.** It fills a gap from valid frames on both
   sides, so a frame just before onset could take coordinates from the attack itself.
3. **Each pre-attack window is computed from the session truncated at that episode's
   onset.** Besides interpolation, `flag_tracking`'s *frozen* flag looks ahead: a frame
   is frozen if its run of identical positions is long enough, and that run may
   continue after onset. Truncation makes every pre-attack value independent of frames
   at or after onset (regression tests perturb future coordinates, future gaps and
   future frozen runs). Baseline tiles use the full session; the generated
   `causality_check` counts how often the two computations differ inside candidate
   pre-attack windows (0 frames in these sessions).
4. `features.compute_features` with Ava's default `FeatureParams` (causal One-Euro
   smoothing; `dist_rate` = backward difference of the smoothed centre distance, px/s,
   negative = centres getting closer).
5. A frame t is **usable at history H** if `dist_rate[t]` is finite and `tracking_ok`
   holds on every frame t−H … t. H = 1 covers both derivative endpoints (the rate at t
   uses t−1). H = 15 is only a provisional consecutive-quality sensitivity criterion; it
   is **not** a bound on the smoother's memory. The One-Euro filter is recursive: a bad
   position decays geometrically (by about 1 − α = 0.83 per frame at the default 1 Hz
   cutoff) but never fully disappears, and the filter only restarts after a gap longer
   than `max_gap_s` = 0.5 s, not after a bad frame that passes the flags. §4 measures the
   remaining contamination beyond 15 frames. H = 1 is used for the main tables, H = 15 as
   sensitivity; neither is validated.
6. Window summaries report the **median** and the mean of usable frames. The median is
   the primary summary because single-frame tracking errors are not removed by any
   flag (§4).

**`tracking_ok` is not motion accuracy.** It only says every torso keypoint passed
the score / frame / frozen flags. §4 shows frames that pass all of them while one
mouse's keypoints are almost certainly misplaced.

## 3. Definitions used for counting (provisional)

- **Attack episode**: a maximal run of consecutive frames annotated `attack`; no
  merging, no minimum duration.
- **Pre-attack window of L s**: frames `[onset − 30·L, onset)`; dropped if it would start
  before frame 0 or contains an `attack` frame (overlaps an earlier episode).
- **Baseline tiles**: non-overlapping L-second tiles from frame 0 of the same session,
  dropped if any frame is within C seconds of an attack frame ("near attack") or
  overlaps a retained pre-attack window of the same length. Tiling is deterministic;
  this is one possible construction, not a proposal.
- IDs: episodes `<session>:attack:<onset frame>`, windows
  `<session>:pre:<L>s:<onset frame>` and `<session>:base:<L>s:C<C>:<start frame>`. The
  JSON output lists every window with its frame range, the frames excluded and why,
  label counts and per-history summaries.

## 4. Tracking anomalies (raw diagnostics; unverified)

The two frames flagged by the reviewers show the same pattern. At **mouse005 frame
2812** and **mouse021 frame 6916**, all torso keypoints of mouse b (intruder) move about
690–711 px in one frame and land on mouse a: the distance between the two body centres
drops from 675.7 to 31.2 px (mouse005) and from 684.9 to 14.4 px (mouse021). On the
next frame they return (the centre is 3.5 / 3.7 px from where it was two frames
earlier). Mouse b's torso scores fall from 0.57–0.86 to 0.18–0.31, still above the 0.15
cut-off, so every point stays valid and `tracking_ok` is true. The raw `dist_rate` spike
(−3,662 / −3,567 px/s) is followed by +713 / +679, +585 / +559 and +456 / +464 px/s on
the next three frames, whose raw positions are normal again: the One-Euro filter carries
the error forward.

How long it carries it was measured with a counterfactual: the same session with only
mouse b's points at the anomalous frame removed, everything else identical. The
difference in smoothed `dist_rate` is still 44.7 / 43.3 px/s 15 frames later, 36.9 /
35.7 at 16 frames, 2.6 / 2.5 at 30 and 0.15 / 0.14 at 45 (mouse005 / mouse021). Every
one of those frames is `tracking_ok`, so the H = 15 criterion counts frames 16 and more
after the error as clean while they still carry tens of px/s from it (for scale: in the
generated baseline table, the per-tile median `dist_rate` of 1-s tiles with C = 5 s has an
interquartile range of −31.2 to 17.74 px/s). The generated table shows this measurement for every listed step; a
regression test reproduces it on synthetic data.

**Assessment: plausible tracking errors** (a one-frame collapse of mouse b's keypoints
onto mouse a; ~700 px in 1/30 s and back is not a plausible movement). **Not verified**:
no video is available, so no frame here is a verified error.

The generated table lists the five largest one-frame centre steps per session on
`tracking_ok` frames. They are not limited to these two frames: every session has two
or more one-frame steps above 590 px, some returning within one frame and some where the
centre stays displaced (e.g. mouse b moving from 41–42 px to ~700 px from mouse a, which
suggests an earlier collapse lasting several frames). On one side of every listed step
at least one torso score is ≤ 0.2 (0.08–0.49 across the three keypoints). Where a score
is below the 0.15 cut-off, that frame is not `tracking_ok` and the endpoint rule (H = 1)
removes the derivative across it; where all scores are ≥ 0.15 (e.g. frames 2812 and
6916) nothing removes it. Raising the cut-off is not a free fix: the note on
`calms21.flag_quality` (`igor/day2`) says it costs attack and mount frames first.
Three listed steps (mouse005 frames 1058, 1079, 1080) fall inside the retained 3-s
pre-attack window before onset 1147; that window's mean `dist_rate` is 20.7 px/s and
its median 8.67 px/s (H = 1). How much of that difference those frames cause has not
been isolated. Whether to add a jump limit, robust summaries, or both, is open.

## 5. Results

The tables below are generated from the cached data by the script and can be checked
with `--check-doc` (§8).

<!-- BEGIN GENERATED: scripts/investigation_feasibility.py -->

*Generated by `scripts/investigation_feasibility.py`; check with `--check-doc`. Descriptive only.*

#### Inputs and code

- Source: `/Volumes/T7 Hackathon/Datasets/CalMS21/extracted/task1_classic_classification/calms21_task1_train.json` (1,224,936,223 bytes, sha256 `c66f1caeed7bd2f66a468fef735376398652844736cbc0ac59a5fdcc4376ecab`)
- Split manifest sha256 `501fed578b6ec9aa8d699723c5d4dfa9bf7264b7ec3161947748d700a187ac09`
- `task1/train/mouse005_task1_annotator1` → `task1__train__mouse005_task1_annotator1.json` sha256 `0233682ffce18e2f7bf094a384e844c9a1d59cd709fc6e511d72e2145be331eb`
- `task1/train/mouse009_task1_annotator1` → `task1__train__mouse009_task1_annotator1.json` sha256 `aedd539acfdb2e71c48860ddc937950f0ba7afc23f0c12725d342873bf67acb6`
- `task1/train/mouse021_task1_annotator1` → `task1__train__mouse021_task1_annotator1.json` sha256 `04cd9ac68a0b7709d664a18293a323129a7bab3867faa4ae4a22858f68249c38`
- Ignored unlisted files in the cache directory: none
- Code (sha256 of the exact files used; git HEAD is recorded in the JSON output):
  - `scripts/investigation_feasibility.py` sha256 `78f355ff2afaaa39cc936795529e54f0bac4dd7e972f6c1a512dbc88cb056220`
  - `src/live_tracker/investigation_data.py` sha256 `dd76565ac3c69f601c016ffbbbe566680a9674a3ed2af661eb0852bc1383b8ca`
  - `src/live_tracker/calms21.py` sha256 `158c30cdf51eec6556dbe3853757e23fa16e14c8f665b4605e9f12d90673fa46`
  - `src/live_tracker/quality.py` sha256 `338e653a427f806a4e85bd26962fa723bcf66f467163271a1e7a16515305804f`
  - `src/live_tracker/features.py` sha256 `2fdc3d9c193fc67a4d0f43fab4c3bb769ec52af7643ff94d1802fc0095f5e3b1`
  - `src/live_tracker/filters.py` sha256 `2375c2b67b0fe6061df1498d670557bc118bb55d34b99abd38a730aef96de583`

- Settings: FPS 30; torso neck, left_hip, right_hip; min_score 0.15; jump detection off; interpolation not used; FeatureParams default (smooth one_euro, {'min_cutoff': 1.0, 'beta': 0.0}, max_step_s 0.1); history H ∈ [1, 15] (primary 1)

#### Validation split: annotation runs (all 14 sessions, annotations only)

| Session | Frames | Attack runs | Investigation runs | Mount runs |
|---|---:|---:|---:|---:|
| mouse005_task1_annotator1 | 5,641 | 7 | 15 | 0 |
| mouse009_task1_annotator1 | 8,332 | 0 | 23 | 0 |
| mouse021_task1_annotator1 | 11,322 | 0 | 19 | 3 |
| mouse024_task1_annotator1 | 8,899 | 0 | 14 | 0 |
| mouse028_task1_annotator1 | 1,036 | 6 | 9 | 0 |
| mouse035_task1_annotator1 | 4,879 | 0 | 14 | 3 |
| mouse037_task1_annotator1 | 5,133 | 0 | 5 | 2 |
| mouse045_task1_annotator1 | 2,713 | 20 | 21 | 2 |
| mouse047_task1_annotator1 | 2,751 | 0 | 12 | 8 |
| mouse051_task1_annotator1 | 17,326 | 0 | 71 | 0 |
| mouse061_task1_annotator1 | 3,996 | 0 | 24 | 0 |
| mouse064_task1_annotator1 | 7,150 | 23 | 17 | 0 |
| mouse065_task1_annotator1 | 8,103 | 0 | 26 | 0 |
| mouse067_task1_annotator1 | 7,024 | 0 | 33 | 0 |

Attack runs in validation: **56**, in **4** of 14 sessions.

#### Extracted sessions

| Session | Seconds | tracking_ok | finite dist_rate | usable H=1 | usable H=15 | Attack episodes | Episode duration s (min/q25/med/q75/max) | Gap between episodes s (min/q25/med/q75/max) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| mouse005_task1_annotator1 | 188 | 0.9917 | 0.9995 | 0.9853 | 0.9227 | 7 | 0.33 / 0.53 / 0.73 / 3.25 / 5.5 | 1.1 / 3.27 / 4.48 / 11.02 / 26.9 |
| mouse009_task1_annotator1 | 277.7 | 0.9214 | 0.9986 | 0.9059 | 0.8649 | 0 | – | – |
| mouse021_task1_annotator1 | 377.4 | 0.9945 | 0.9999 | 0.9900 | 0.9395 | 0 | – | – |

#### Pre-attack windows: mouse005_task1_annotator1

| Window | Episodes | Dropped: before session start | Dropped: overlaps earlier attack | Retained | usable frac H=1 (min/median) | ≥0.9 usable H=1 | ≥0.9 usable H=15 | Per-window median dist_rate H=1 (min/q25/med/q75/max) | Per-window mean dist_rate H=1 (min/q25/med/q75/max) | Frames: investigation / other / mount |
|---|---:|---:|---:|---:|---|---:|---:|---|---|---|
| 1s | 7 | 0 | 0 | 7 | 0.87 / 1 | 6 | 4 | -211.47 / -41.55 / -3.65 / 2.09 / 9.09 | -230.47 / -52.23 / -0.24 / 1.41 / 2.85 | 161 / 49 / 0 |
| 2s | 7 | 0 | 1 | 6 | 0.93 / 1 | 6 | 4 | -94.16 / -70.93 / -57.51 / -20.65 / -3.05 | -102.72 / -50.39 / -42.25 / -24.8 / -2.05 | 215 / 145 / 0 |
| 3s | 7 | 1 | 1 | 5 | 0.91 / 0.98 | 5 | 2 | -72.18 / -38.77 / -21.92 / -1.41 / 8.67 | -98 / -40.52 / -31.91 / -2.22 / 20.7 | 244 / 206 / 0 |

Retained pre-attack window IDs and onsets (s): `mouse005_task1_annotator1:pre:1s:79` (2.633); `mouse005_task1_annotator1:pre:1s:251` (8.367); `mouse005_task1_annotator1:pre:1s:736` (24.533); `mouse005_task1_annotator1:pre:1s:934` (31.133); `mouse005_task1_annotator1:pre:1s:1147` (38.233); `mouse005_task1_annotator1:pre:1s:1969` (65.633); `mouse005_task1_annotator1:pre:1s:2072` (69.067)

#### Same-session baseline tiles: mouse005_task1_annotator1

| Window | Clearance | Retained | Excluded near attack | Excluded overlapping pre-window | ≥0.9 usable H=1 | Frames: investigation / other | Tile start s (q25/med/q75) | Per-window median dist_rate H=1 (q25/med/q75) |
|---|---:|---:|---:|---:|---:|---|---|---|
| 1s | 0 s | 162 | 20 | 6 | 161 | 567 / 4293 | 62.25 / 106.5 / 146.75 | -38.65 / -3.19 / 19.27 |
| 1s | 5 s | 131 | 57 | 0 | 131 | 304 / 3626 | 89.5 / 122 / 154.5 | -31.2 / -3 / 17.74 |
| 1s | 10 s | 114 | 74 | 0 | 114 | 229 / 3191 | 102.25 / 130.5 / 158.75 | -33.43 / -3.93 / 17.15 |
| 2s | 0 s | 76 | 12 | 6 | 76 | 459 / 4101 | 73.5 / 111 / 148.5 | -39.79 / -5.26 / 12.3 |
| 2s | 5 s | 64 | 30 | 0 | 64 | 304 / 3536 | 91.5 / 123 / 154.5 | -28.55 / -5.56 / 10.82 |
| 2s | 10 s | 56 | 38 | 0 | 56 | 229 / 3131 | 103.5 / 131 / 158.5 | -20.96 / -5.56 / 11.61 |
| 3s | 0 s | 48 | 11 | 3 | 48 | 399 / 3921 | 77.25 / 112.5 / 147.75 | -21.18 / -3.72 / 17.31 |
| 3s | 5 s | 42 | 20 | 0 | 42 | 274 / 3506 | 90.75 / 121.5 / 152.25 | -18.51 / -7.47 / 12.07 |
| 3s | 10 s | 36 | 26 | 0 | 36 | 229 / 3011 | 104.25 / 130.5 / 156.75 | -17.04 / -4.21 / 13.41 |

#### One-frame raw body-centre steps (px), all frames with both endpoints present

| Session | Mouse | n | p50 | p99 | p99.9 | max |
|---|---|---:|---:|---:|---:|---:|
| mouse005_task1_annotator1 | a | 5638 | 3.6 | 28.3 | 49.4 | 75 |
| mouse005_task1_annotator1 | b | 5640 | 1.8 | 35.9 | 340.9 | 711.4 |
| mouse009_task1_annotator1 | a | 8322 | 3.8 | 27.3 | 201.7 | 507.1 |
| mouse009_task1_annotator1 | b | 8329 | 2.8 | 26.5 | 193.7 | 768.6 |
| mouse021_task1_annotator1 | a | 11321 | 2.5 | 23 | 30.2 | 46.1 |
| mouse021_task1_annotator1 | b | 11321 | 1.8 | 38 | 675.7 | 820.6 |

#### Largest 5 one-frame raw body-centre steps per session on tracking_ok frames (unverified)

| Session | Frame | Mouse | Step px | Return px (t−1→t+1) | Distance to other mouse px (t−1 → t) | Torso scores t−1 → t (neck, left_hip, right_hip) | dist_rate t..t+3 (smoothed) | Smoother contamination px/s at t+15 / t+16 / t+30 / t+45 | Label | In retained pre-windows (t..t+3) | In retained baseline tiles (t..t+3) |
|---|---:|---|---:|---:|---|---|---|---|---|---|---:|
| mouse005_task1_annotator1 | 2813 | b | 711.4 | 707.4 | 31.2 → 689.3 | 0.3, 0.18, 0.2 → 0.86, 0.7, 0.65 | 712.8, 584.5, 456.4, 396.4 | -1.376 / -1.137 / -0.079 / -0.005 | other | none | 9 |
| mouse005_task1_annotator1 | 2812 | b | 711 | 3.5 | 675.7 → 31.2 | 0.86, 0.69, 0.65 → 0.3, 0.18, 0.2 | -3662, 712.8, 584.5, 456.4 | 44.66 / 36.906 / 2.566 / 0.147 | other | none | 9 |
| mouse005_task1_annotator1 | 1058 | b | 611.4 | 609.1 | 501.2 → 144 | 0.49, 0.15, 0.17 → 0.73, 0.61, 0.33 | 65.1, 121.4, 99, 177 | -0.95 / -0.863 / -0.082 / -0.005 | investigation | mouse005_task1_annotator1:pre:3s:1147 | 1 |
| mouse005_task1_annotator1 | 1080 | b | 420.6 | 432.3 | 251.3 → 246.3 | 0.34, 0.28, 0.15 → 0.71, 0.47, 0.35 | 411.6, 243.1, 186.3, 109.5 | -1.159 / -0.939 / -0.064 / -0.004 | other | mouse005_task1_annotator1:pre:3s:1147 | 1 |
| mouse005_task1_annotator1 | 1079 | b | 415.3 | 6.6 | 257.7 → 251.3 | 0.7, 0.54, 0.26 → 0.34, 0.28, 0.15 | -1347.1, 411.6, 243.1, 186.3 | 20.209 / 17.066 / 1.23 / 0.061 | other | mouse005_task1_annotator1:pre:3s:1147 | 2 |
| mouse009_task1_annotator1 | 4896 | b | 766.6 | 766 | 42.4 → 718.2 | 0.14, 0.35, 0.32 → 0.83, 0.73, 0.75 | 718.3, 681.5, 596.5, 494.9 | -1.276 / -1.055 / -0.072 / -0.004 | other | none | 9 |
| mouse009_task1_annotator1 | 4859 | b | 718.3 | 718.4 | 41.1 → 696.1 | 0.19, 0.11, 0.1 → 0.87, 0.67, 0.72 | 749.5, 636.7, 500, 398.7 | -1.345 / -1.112 / -0.077 / -0.004 | other | none | 18 |
| mouse009_task1_annotator1 | 6599 | b | 594.7 | 594.5 | 552.4 → 242 | 0.16, 0.14, 0.14 → 0.86, 0.61, 0.74 | -9.5, 104.8, 167.9, 207.3 | -0.431 / -0.354 / -0.041 / -0.002 | other | none | 15 |
| mouse009_task1_annotator1 | 3184 | a | 312.4 | 308.9 | 730.9 → 428.4 | 0.2, 0.08, 0.12 → 0.7, 0.52, 0.59 | -587.6, -463.9, -368.6, -288.3 | 1.287 / 1.063 / 0.074 / 0.004 | other | none | 9 |
| mouse009_task1_annotator1 | 3156 | a | 213.5 | 154 | 559 → 398.3 | 0.29, 0.14, 0.18 → 0.33, 0.32, 0.36 | -1179.2, -973.9, -597.9, 648.3 | 3.808 / 3.165 / 0.246 / 0.012 | other | none | 9 |
| mouse021_task1_annotator1 | 1040 | b | 816.1 | 815.2 | 747.5 → 128.3 | 0.21, 0.1, 0.14 → 0.85, 0.7, 0.67 | -334.9, -142.9, -13.3, 76.8 | -0.824 / -0.682 / -0.046 / -0.003 | investigation | none | 9 |
| mouse021_task1_annotator1 | 776 | b | 744.3 | 744.1 | 485.2 → 322.5 | 0.32, 0.13, 0.15 → 0.86, 0.72, 0.75 | 488, 312.7, 140.5, 1.1 | -1.632 / -1.345 / -0.099 / -0.006 | other | none | 9 |
| mouse021_task1_annotator1 | 1262 | b | 723.8 | 724.3 | 668.5 → 118 | 0.34, 0.1, 0.15 → 0.85, 0.74, 0.7 | -162.2, -74.6, -3.8, 31.7 | -0.444 / -0.359 / -0.04 / -0.003 | investigation | none | 9 |
| mouse021_task1_annotator1 | 985 | b | 703.5 | 703.3 | 625.8 → 119.3 | 0.31, 0.19, 0.1 → 0.83, 0.7, 0.61 | -958.8, -681.8, -445.3, -293.6 | -1.173 / -1.062 / -0.016 / -0.003 | investigation | none | 9 |
| mouse021_task1_annotator1 | 6916 | b | 691.7 | 3.7 | 684.9 → 14.4 | 0.84, 0.67, 0.57 → 0.23, 0.31, 0.2 | -3567, 679.4, 559.2, 463.8 | 43.264 / 35.748 / 2.494 / 0.144 | other | none | 9 |

<!-- END GENERATED -->

## 6. Changes since the first version of this report, and why

| What | First version | Now | Why |
|---|---|---|---|
| Interpolation | `interpolate_short_gaps` applied | not applied | two-sided; could use frames after onset |
| Pre-attack computation | full session | truncated at each onset | interpolation and the *frozen* flag look ahead. Measured effect here: 0 frames changed, so all pre-window counts are unaffected |
| Usable frame | `tracking_ok` and finite `dist_rate` at t | also `tracking_ok` at t−1 (H = 1); H = 15 reported as a sensitivity criterion | the rate at t uses frame t−1. (An earlier revision described H = 15 as covering the filter's memory; that was wrong, see §2.5 and §4) |
| Window summary | mean | median (primary) and mean | single-frame errors pass every flag |
| 1-s pre-window usable fraction, min | 0.93 | 0.87 (window before onset 1147) | frames 1124 and 1127 are not `tracking_ok`; the endpoint rule also removes 1125 and 1128, whose rates cross them: two more frames, 28/30 → 26/30. (An earlier revision said "one more frame"; that was wrong.) ≥ 0.9 usable windows: 7 → 6 |
| 3-s pre-window mean `dist_rate`, min | −96.51 | −98.0 | frames removed by the endpoint rule; no interpolated frames |
| Baseline tiles, C = 0 s | 168 / 82 / 51 (1/2/3 s) | 162 / 76 / 48 | tiles overlapping retained pre-attack windows are now excluded; C = 5 and 10 s unchanged (those tiles were already within the clearance) |
| Session `tracking_ok` | 99.2 / 92.1 / 99.5 % | 99.17 / 92.14 / 99.45 % | no interpolated points; differences below 0.1 point |
| −230 px/s 1-s window | "may be an outlier (not verified)" | its per-frame median is −211.47 px/s | the robust summary shows it is not driven by a single spike |
| Tracking anomalies | "~120 px per frame, not checked" | §4: identified as plausible single-frame keypoint collapses, unverified; more listed | raw diagnostics |
| Terminology | "CalMS21 has no approach label" | "CalMS21 **Task 1** has no approach label" | the claim was only checked for Task 1 |
| Extraction run | 36.0 s, 2.60 GB peak | see §9 | the raw bytes are now kept in memory while they are hashed and parsed; reuse now re-hashes the whole source |
| Smoother memory | not measured | contamination measured per listed step (§4) | H = 15 does not bound it |
| Label summary | trusted from the index | validated against the manifest and the extracted records | the index is not a trusted source |

Unchanged: attack episodes (7 in mouse005, none in mouse009 / mouse021), episode
durations and gaps, dropped pre-windows (2 s: 1 overlap; 3 s: 1 before session start and
1 overlap), label composition of the pre-attack windows, and the validation-split
counts (56 attack runs in 4 of 14 sessions).

## 7. Limitations

1. **Too few attack sessions in validation**: 4 of 14 sessions, 56 runs, spread 6 / 7 /
   20 / 23. With sessions as the unit (episodes within a session are not independent),
   validation alone allows n ≤ 4. The window measurements here rest on one session
   (mouse005, 7 episodes).
2. **Investigation confound** (not resolved): pre-attack windows are mostly annotated
   *investigation*; baseline tiles are mostly *other* (generated tables). A `dist_rate`
   difference could reflect investigation versus other, not "before attack". A baseline
   matched on annotation (e.g. investigation not followed by attack) is one option.
3. **Time confound** (not resolved): in mouse005 every retained pre-attack window ends
   by 69.1 s of a 188 s session (onsets listed in the generated section), while baseline
   tiles start at a median of 106.5–130.5 s (1-s tiles; larger clearances push them later).
   Activity may drift over a session. Time-matched baselines are one option.
4. **"Approach" is a measurement, not a label**: CalMS21 Task 1 has no approach label.
   `dist_rate` is a pair-level rate of change of body-centre distance; it cannot say which
   mouse approaches (no heading-to-other feature yet).
5. **Single-frame tracking errors pass every quality flag** (§4); they reach windows used
   for both pre-attack and baseline summaries.
6. **Short, close episodes**: median attack run 0.73 s, runs as close as 1.1 s; merging
   rules change both the number of episodes and the overlaps.
7. **Asymmetric preprocessing**: pre-attack windows are truncated at onset, baseline tiles
   are not. It made no difference here (`causality_check`), but it is not guaranteed in
   general.
8. **Smoother memory is not bounded by any criterion used here**: a valid-looking bad frame
   still changes the smoothed `dist_rate` by tens of px/s more than 15 frames later
   (§4). Neither H = 1 nor H = 15 removes this; a jump limit (not chosen), raw instead of
   smoothed rates, or excluding a span after each detected anomaly are options, none
   decided.
9. **Label summary for sessions that were not extracted** is checked for structure and
   manifest membership only; it was computed from the source at extraction time and
   cannot be cross-checked without re-reading the source. The four attack sessions'
   counts that drive §8 are therefore only verified for mouse005 (extracted).
10. mouse028 (1,036 frames, 34.5 s, 6 attack runs) will lose pre-windows to the session
    start and to overlaps.

## 8. Are more sessions needed, and what must be decided

**More sessions: yes.** Next would be the other three validation attack sessions
(mouse028, mouse045, mouse064): 4 sessions, 56 runs, still few sessions. Whether to also
use training sessions (Igor's classifier training data, not held-out test data) is a team
decision. Test annotations stay excluded either way.

**Decisions needed (none made here)**: episode merging / minimum duration; window length
and gap; baseline construction (plain, annotation-matched, time-matched; clearance);
the approach measure; jump limit and / or robust summaries; the history requirement H;
unit of analysis and sufficiency; whether train sessions may be used.

## 9. Reproduce

```bash
.venv/bin/python -m live_tracker.investigation_data \
  --source "/Volumes/T7 Hackathon/Datasets/CalMS21/extracted/task1_classic_classification/calms21_task1_train.json" \
  --manifest splits/split_manifest.json --out data/raw/calms21_validation \
  --sessions task1/train/mouse005_task1_annotator1 task1/train/mouse009_task1_annotator1 \
             task1/train/mouse021_task1_annotator1
.venv/bin/python scripts/investigation_feasibility.py data/raw/calms21_validation \
  --out data/raw/investigation_experiment_01_results.json \
  --check-doc docs/investigation_experiment_01.md     # or --write-doc to regenerate §5
```

Run figures for this report (machine dependent, not in the generated section; the source
is on an external drive):

- Extraction (cache rebuilt): 37.2 s and 3.03 GB peak when the source was not in the OS
  file cache; 14.2 s and 2.77 GB when it was (this report's final rebuild).
- Cache reuse: the whole source is re-hashed before the cache is trusted: 31.3 s
  (0.66 s user + 0.46 s system CPU, i.e. almost all I/O), 0.1 GB peak. Verified reuse therefore
  saves memory but little time compared with a cold rebuild.
- Analysis of the three sessions, including the counterfactual smoother runs: a few
  seconds, about 0.3 GB peak.
- Memory figures are reported as unavailable on platforms without the `resource` module
  (Windows).
