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

## How to plug one in

1. Write a backend: `factory(settings, root)` returns the function above.
2. Call `pose_model.register_backend("name", factory)` when the program starts.
3. Put the settings in `models/pose/pose_model.json`, for example `{"backend": "name", "path": "models/pose/mouse.onnx"}`.
   (Another file can be named with the environment variable `LIVE_TRACKER_POSE_MODEL`.)

The settings file only names a backend. It never runs code. Model files are big and are not committed to Git.

## Install OpenCV

`python -m pip install opencv-python-headless` (also listed as the `camera` option in `pyproject.toml`).
