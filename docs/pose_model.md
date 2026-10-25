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

## How to plug one in

1. Write a backend: `factory(settings, root)` returns the function above.
2. Call `pose_model.register_backend("name", factory)` when the program starts.
3. Put the settings in `models/pose/pose_model.json`, for example `{"backend": "name", "path": "models/pose/mouse.onnx"}`.
   (Another file can be named with the environment variable `LIVE_TRACKER_POSE_MODEL`.)

The settings file only names a backend. It never runs code. Model files are big and are not committed to Git.

## Install OpenCV

`python -m pip install opencv-python-headless` (also listed as the `camera` option in `pyproject.toml`).
