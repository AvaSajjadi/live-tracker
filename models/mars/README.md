# MARS pretrained models

MARS (the Mouse Action Recognition System) finds two mice, one black and one white, in a top-view video and gives the same
7 keypoints per mouse as CalMS21. Live Tracker uses it on the **offline route**: a recorded camera video is analyzed
afterwards ("Record, then analyze with MARS" on the Upload step, or `scripts/mars_pose_offline.py`). The live camera uses
the faster YOLO pose model instead (`models/pose/`).

## The files

Two of the three files are over GitHub's 100 MB limit, so the files are attached to the release
[`mars-models-v1`](https://github.com/AvaSajjadi/live-tracker/releases/tag/mars-models-v1) instead of the repository. The
two 230 MB detection files are stored there in five 50 MB pieces each (`.part-a` to `.part-e`).

| File | Size | Place | md5 |
|---|---|---|---|
| `MARS_top_detection_black.pb` | 230 MB (5 pieces) | `models/mars/detection/` | `79f9b55a8eff1192b899e55cc0ea570c` |
| `MARS_top_detection_white.pb` | 230 MB (5 pieces) | `models/mars/detection/` | `4469996d5d7003c8c05d6279ed1506af` |
| `MARS_top_pose.pb` | 98 MB | `models/mars/pose/` | `d380063a5177545721103a34fa650b56` |

One command, from the project folder (any system), downloads them, joins the pieces and checks every md5:

    python scripts/download_mars_models.py

By hand (macOS / Linux): download the pieces from the release page, then join them in order and check the md5:

    cat MARS_top_detection_black.pb.part-? > models/mars/detection/MARS_top_detection_black.pb
    cat MARS_top_detection_white.pb.part-? > models/mars/detection/MARS_top_detection_white.pb
    md5 -r models/mars/*/*.pb          # Linux: md5sum models/mars/*/*.pb

When the three files are here, the app uses this folder (`src/live_tracker/mars_job.py`); otherwise it looks in
`~/mars_local/models` (or `$MARS_HOME/models`). The `.pb` files are ignored by Git.

MARS also needs a Python environment with TensorFlow (`~/mars_local/env`); see `docs/camera.md`, section 8.

## Source and license

The files are unchanged copies of the `.pb` graph files in `models.zip` (543 MB, md5 `58efd60b0b3ca2cb27e40cef7e9d0da1`)
from the Caltech Data record https://data.caltech.edu/records/801be-66057. The `.pkl` files in that zip are not used
(pickle files can run code).

License: **CC BY-NC 4.0** (non-commercial use, with credit).

Segalin, C., Williams, J., Karigo, T., Hui, M., Zelikowsky, M., Sun, J. J., Perona, P., Anderson, D. J., & Kennedy, A.
(2021). The Mouse Action Recognition System (MARS) software pipeline for automated analysis of social behaviors in mice.
eLife, 10, e63720. https://doi.org/10.7554/eLife.63720
