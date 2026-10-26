# The pose model slot (camera step)

A camera gives pictures. The scorer needs mouse keypoints. A **pose model** turns one into the other.
This repository has **no pose model yet**. `src/live_tracker/pose_model.py` is the place to plug one in.

## What a pose model must do

It is a function. It takes one camera picture (a NumPy image from OpenCV) and returns a table with these columns:

| column | meaning |
|---|---|
| `mouse` | `a` or `b` |
| `keypoint` | one of `nose, left_ear, right_ear, neck, left_hip, right_hip, tail_base` |
| `x`, `y` | pixels in the **CalMS21 frame, 1024 x 570**. The model must map the camera picture to this frame |
| `score` | confidence from 0 to 1 |

It returns `None` when it finds no mouse. The program then marks that frame as lost tracking (never as a still mouse).
The answer is checked on every frame (`pose_model.check_output`). A wrong layout stops the run with a clear message.

Warning: Model B was trained on a top-down view like CalMS21. A camera with another view or another cage gives
scores that mean little, even if the keypoints are right.

## A camera that films a screen (the presentation setup)

If the camera is on a holder and films an iPad that plays a CalMS21 video, choose "The camera films a screen" on the
Upload step and set the 4 corners of the video (`src/live_tracker/screen_map.py`). Every picture is then stretched so the
video fills a 1024 x 570 frame **before** the pose model sees it. In that case the model does not have to map the picture
itself: it only has to find the mice and give keypoints in the picture it receives. Without a screen, the model still must
do the mapping itself.

Tips: iPad brightness high, no light shining on the screen, camera still and facing the screen straight on, and the whole
video visible. The screen setup page shows the stretched picture so the corners can be checked.

## Backends that ship with the program

`src/live_tracker/pose_backends.py` registers two backends. The settings go in `models/pose/pose_model.json` (not in Git).

**`yolo`** is the small fast model trained with `scripts/train_yolo_pose.py` on the MARS hand-labeled set (see
`colab/train_yolo_pose_colab.ipynb`). Class 0 is the black mouse (mouse a), class 1 the white mouse (mouse b), 7 keypoints each.

    {"backend": "yolo", "weights": "models/pose/mouse_pose_openvino_model", "device": "intel:gpu", "imgsz": [384, 640], "conf": 0.25}

Export the trained `best.pt` first: `yolo export model=best.pt format=openvino imgsz=384,640` (Intel laptop; the folder is the weights path).
On a Mac use `format=coreml` and `"device": "mps"` or leave the device out. A picture that is not 1024 x 570 is resized to it.

**`stub`** is NOT a model. It returns the saved keypoints of a recording, following the clock. Only for rehearsals:

    {"backend": "stub", "poses": "data/raw/calms21_video/mouse064_task1_annotator1_poses.parquet", "start_frame": 1800}

Test the whole live path with `scripts/live_camera_acceptance.py` (see `docs/camera.md`, section 9).

## How to plug one in

1. Write a backend: `factory(settings, root)` returns the function above.
2. Call `pose_model.register_backend("name", factory)` when the program starts.
3. Put the settings in `models/pose/pose_model.json`, for example `{"backend": "name", "path": "models/pose/mouse.onnx"}`.
   (Another file can be named with the environment variable `LIVE_TRACKER_POSE_MODEL`.)

The settings file only names a backend. It never runs code. Model files are big and are not committed to Git.

## Install OpenCV

`python -m pip install opencv-python-headless` (also listed as the `camera` option in `pyproject.toml`).

## Steadier points (optional, off by default)

`src/live_tracker/point_stabilizer.py` makes the 14 points of each picture steadier. It uses the same steps as the ScopeScore tracker (Ava's surgical video paper):
1. **Flow mix.** The model's point is mixed with the point followed from the last picture by optical flow: `p = 0.2 * model + 0.8 * flow`.
2. **Smoothing.** A One-Euro filter on every coordinate (past pictures only; 0.6 Hz, beta 0.04).
3. **Tolerance.** A point that moves less than about 1% of the picture height (5.7 px) stays still. A bigger move is followed almost at once.

It invents nothing. If the model finds no mouse, the answer is still "no mouse". After a gap of more than 0.5 s it starts again.

Turn it on in `models/pose/pose_model.json` by adding a block, for example `"stabilize": {}` for the defaults, or
`"stabilize": {"alpha": 0.2, "min_cutoff": 0.6, "beta": 0.04, "sigma_px": 5.7, "flow": true}`. Set `"flow": false`, `"sigma_px": 0` or `"min_cutoff": 0` to switch one step off.

**Not proven yet.** The tests show that it steadies noisy points and follows real moves. It has not been measured on the trained mouse model or on real camera video. Measure first (noise test and unseen videos), then decide if it stays on.

### Test of the steadier points (noise test, validation sessions, measured)

Script: `scripts/point_stability_test.py` (pass rules are written at the top of the script, before the first run). Input: the CalMS21 keypoints of 3 validation sessions (1800 frames each) plus 1 px or 3 px of noise. Flow is off in this test (there are no pictures).

| Settings | Accuracy | Precision of steps | Lag | Still mouse |
|---|---|---|---|---|
| ScopeScore settings (defaults) | FAIL (1.18 to 5.55 px at 1 px noise) | FAIL | FAIL | PASS |
| No tolerance | FAIL | PASS | FAIL | PASS |
| Search of 18 other settings (changed after seeing results) | best settings pass 2 of 4 rules | | | |

**What it means.** The ScopeScore settings were made for slow surgical instruments at 24 frames per second. Mice move fast at 30 frames per second, so the filter lags behind. The settings that avoid the lag also remove almost none of the noise (3 px noise goes down to about 3.4 px). **No setting passed all 4 rules.** So the stabilizer stays OFF.

The search of other settings (min_cutoff 2, 5, 10; beta 0.5, 2, 8; sigma 0 and 1.5) was done after the first result was seen. The pass rules were not changed. The flow mix is still untested on real video.
