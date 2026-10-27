# MARS pretrained models

MARS (the Mouse Action Recognition System) finds two mice, one black and one white, in a top-view video and gives the same
7 keypoints per mouse as CalMS21. Live Tracker uses it on the **offline route**: a recorded camera video is analyzed
afterwards ("Record, then analyze with MARS" on the Upload step, or `scripts/mars_pose_offline.py`). The live camera uses
the faster YOLO pose model instead (`models/pose/`).

## The files

Two of the three files are over GitHub's 100 MB limit, so they are attached to the release
[`mars-models-v1`](https://github.com/AvaSajjadi/live-tracker/releases/tag/mars-models-v1) instead of the repository.
Put them here:

| File | Size | Place | md5 |
|---|---|---|---|
| `MARS_top_detection_black.pb` | 230 MB | `models/mars/detection/` | `79f9b55a8eff1192b899e55cc0ea570c` |
| `MARS_top_detection_white.pb` | 230 MB | `models/mars/detection/` | `4469996d5d7003c8c05d6279ed1506af` |
| `MARS_top_pose.pb` | 98 MB | `models/mars/pose/` | `d380063a5177545721103a34fa650b56` |

From the project folder (macOS / Linux):

    R=https://github.com/AvaSajjadi/live-tracker/releases/download/mars-models-v1
    mkdir -p models/mars/detection models/mars/pose
    curl -L -o models/mars/detection/MARS_top_detection_black.pb $R/MARS_top_detection_black.pb
    curl -L -o models/mars/detection/MARS_top_detection_white.pb $R/MARS_top_detection_white.pb
    curl -L -o models/mars/pose/MARS_top_pose.pb $R/MARS_top_pose.pb
    md5 -r models/mars/*/*.pb          # Linux: md5sum models/mars/*/*.pb

Windows PowerShell:

    $R = "https://github.com/AvaSajjadi/live-tracker/releases/download/mars-models-v1"
    New-Item -ItemType Directory -Force models\mars\detection, models\mars\pose | Out-Null
    Invoke-WebRequest "$R/MARS_top_detection_black.pb" -OutFile models\mars\detection\MARS_top_detection_black.pb
    Invoke-WebRequest "$R/MARS_top_detection_white.pb" -OutFile models\mars\detection\MARS_top_detection_white.pb
    Invoke-WebRequest "$R/MARS_top_pose.pb" -OutFile models\mars\pose\MARS_top_pose.pb
    Get-FileHash -Algorithm MD5 models\mars\*\*.pb

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
