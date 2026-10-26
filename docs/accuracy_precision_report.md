# Accuracy and precision report

Made by `scripts/accuracy_precision_report.py`. **Accuracy** = how close to the truth. **Precision** has two meanings, named each time: **(a) right alerts out of all alerts** and **(b) repeatability** (how much a result moves when the input is the same or slightly noisy). Every number below was measured; numbers that could not be measured are marked.

## 1. Keypoints with behavior labels (demo camera, video + keypoints, staged demo)

Scores: Igor's delivered scores (scores_validation_model_b.csv, checksum matched). Truth: the CalMS21 labels of the same 14 VALIDATION sessions (94,305 frames). The test split was not used. These are the same scores and the same rule the app uses.

| Measure | Value | 95% interval over sessions |
|---|---|---|
| Frame accuracy (all 4 classes) | 84.3% | 78.7% to 88.5% |
| Macro F1 (attack, investigation, mount) | 0.651 | |
| **Alert precision** (right alerts / all alerts) | 82.5% (80 alerts) | 18.2% to 92.5% |
| **Alert recall** (attack bouts found / all) | 64.3% (36 of 56) | 21.1% to 87.2% |
| False alerts per hour | 16.03 | |
| Alert start versus true start, median | -0.15 s (positive = late) | |
| Alert fires versus true start, median | +0.35 s | |
| Repeatability: the same input twice gives the same alerts | yes | |

Rule used: attack, ask above 0.6, alert at 0.8, at least 0.5 s.

Per class (frame level):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| attack | 45.1% | 77.6% | 0.570 |
| investigation | 73.0% | 83.8% | 0.780 |
| mount | 48.6% | 79.6% | 0.604 |
| other | 97.5% | 85.1% | 0.909 |

The demo session (`mouse064`): frame accuracy 75.8%; 21 of 23 attack bouts found; 43 of 44 alerts right.

Only 4 of the 14 validation sessions have attack bouts, and two sessions hold most of them, so the intervals are wide. The rule was chosen on these same sessions, so the numbers are a little optimistic.

## 2. Motion-only centers CSV

Real centers files have no labels, so the truth is made: mouse a walks at exactly 90 px/s, mouse b stands still, the true distance is known. Noise is Gaussian jitter added to the positions.

| Jitter (px) | Speed error, median (px/s) | Speed error (%) | **Precision (b): spread of the speed (px/s)** | A still mouse reads (px/s) | Distance error (px) |
|---|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.5 | 3.14 | 3.49 | 4.43 | 25.85 | 0.44 |
| 1.0 | 10.14 | 11.26 | 9.93 | 51.71 | 0.88 |
| 2.0 | 39.87 | 44.29 | 24.14 | 103.42 | 1.78 |

The same input twice gives the same numbers: yes. A gap in tracking is shown as missing and not as a still mouse: yes. **Warning:** with jitter, a mouse that does not move still reads as moving (the table shows how fast), so noisy centers files show motion that is not real. There are no behavior scores or alerts for this data type.

## 3. Video to keypoints with MARS (a recording, or the demo video)

Truth: the MARS hand-labeled pictures (median of 5 workers), 60 test pictures, in pixels of the 1024 x 570 picture (a mouse body is about 195 px long).

| Measure | Original picture | Degraded picture (small, blurry, JPEG, dim) |
|---|---|---|
| **Accuracy**: median error, all keypoints (px) | 6.52 | 7.64 |
| Share of keypoints within 5 px | 34.9% | 29.8% |
| Share within 10 px | 69.9% | 62.0% |
| Touching mice: median error (px) | 6.78 | 7.92 |
| Apart: median error (px) | 6.48 | 7.55 |
| **Precision (b)**: how far a keypoint moves when the picture is degraded, median (px) | | 3.55 |
| Repeatability: the same pictures twice, largest difference (px) | 0.0 | |

Median error per keypoint, original picture (px): nose 5.1, right_ear 5.6, left_ear 5.4, neck 5.7, right_side 11.9, left_side 12.2, tail_base 5.7.

Per mouse (original): black 6.56 px, white 6.5 px.

## 3b. Real-time test: a live feed at 30 frames per second

The video is fed at exactly 30 frames per second (or the stated rate). The model always takes the newest frame; frames that arrived while it was busy are dropped, as in a live camera. **Keeping up** needs at least 95% of the frames processed.

| Model | Feed | Frames processed | Reached (frames/s) | Keypoints late, median (s) | late, 95th pct (s) | Keeps up? | Error vs saved keypoints, median (px) |
|---|---|---|---|---|---|---|---|
| mars | 30 fps, 20.0 s | 22 of 601 (3.7%) | 1.03 | 0.987 | 1.144 | **NO** | 3.31 |

## 4. Camera recording and the screen stretch

Truth is synthetic: bright squares placed at known points of a screen in a 640 x 480 picture, and a camera that gives 15 pictures per second with timing jitter.

- **Accuracy of the stretch:** 5 test squares (4 near the corners and 1 in the middle of the frame) land within 1.49 px of their right place in the 1024 x 570 frame (errors: 1.21, 1.49, 0.43, 1.13, 0.46 px; the squares are placed to the nearest whole pixel of the 640 x 480 picture, which already costs up to about 1 px).
- **Repeatability:** the same picture stretched twice is identical: yes.
- **Recording length accuracy:** a 5 s recording has 150 frames (expected 150), an error of 0.0 s, and the video says 30.0 frames per second.

A real camera adds blur, glare and color, which are not measured here.

## 5. Fast mouse model (YOLO pose)

**Not trained yet, so not measured.** When `best.pt` exists, run `scripts/eval_yolo_pose.py` and pass its JSON with `--yolo-json`.

## Not measured (and why)

- Behavior accuracy on **your own recordings**: there are no labels for them. Only keypoint accuracy on labeled pictures is measurable.
- The **live** chain (camera, fast model, scores, rules) with everything running at once: not built end to end yet.
- The **staged demo's** sync with the iPad (GO and play pressed by hand): a human timing, not measurable here. The alert timing against the labels is in section 1.
