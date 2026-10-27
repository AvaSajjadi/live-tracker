# How the camera is used in this project

A camera is the live "eyes" of the project. A trained pose model finds the two mice in every camera picture, so the
alerts **do** come from what the camera sees. Tested with a Nikon D7500 (through the Nikon Webcam Utility) filming a
phone that plays a CalMS21 video, on a MacBook Air.

## 1. The full chain

1. **Camera** takes pictures, about 30 per second.
2. **Screen stretch** (`src/live_tracker/screen_map.py`, optional) finds the screen video in the picture and stretches it to
   1024 x 570, the CalMS21 size. Without corners, the whole picture is resized to 1024 x 570.
3. **Pose model** (`src/live_tracker/pose_model.py`, `pose_backends.py`) turns each picture into 7 body points per mouse:
   nose, two ears, neck, two hips, tail base. It is a small YOLO pose model trained on the MARS hand-labeled set
   (`models/pose/`). About 10 ms per picture on a MacBook Air (Apple GPU). See `docs/pose_model.md`.
4. **Live scorer** (`src/live_tracker/live_scoring.py`, with Model B) reads the last 6 seconds of points and gives a score
   for each behavior: attack, mount, investigation.
5. **Monitor** (`src/live_tracker/monitor.py`) compares the scores with the rule the researcher approved, for example
   "attack for at least 2 s". It makes an **alert** or a **question**.
6. **Event log** (`journal.jsonl`), **live log** (`live_scores.csv`, `live_positions.csv`) and **report**: every event and
   every frame's scores are saved. At the end the program writes a report.
7. **Page** (`src/live_tracker/live_view.py`) shows the camera with the skeletons drawn on it, the counts, live charts of
   the behavior scores and the motion, the alerts and the report.

## 2. The guided flow with a camera

1. **Upload:** "Connect a camera", camera number (`0` is the first camera; on the test Mac 0 is the Nikon), **Test the
   camera**. Optional: tick "The camera films a screen" and drag the 4 corners. Choose **Live real pose**, then
   **Use this camera**.
2. **Rules:** for example `Alert me if the mice fight for more than 2 seconds`, **Propose rule**, **Confirm**.
3. **Replay:** the live camera with the skeletons the pose model finds (red: mouse a, the black mouse; blue: mouse b,
   the white mouse). Nothing is scored here; it is the check that the camera and the model work.
4. **Real session:** **Start the camera session**. The run program (`python -m live_tracker.monitor_runner`) opens the
   camera itself, so closing the browser does not stop it. The page shows the camera as the model sees it, the alert
   counts and the live charts, then the report.

A second route, **Record, then analyze with MARS**, is in section 8.

## 3. Which file does what

| Job | File |
|---|---|
| List cameras and check one works | `camera_check.py`, `scripts/find_cameras.py` |
| Keep one camera open and share it between pages | `camera_live.py` |
| Live video, corner dragging, record and analyze (Upload step, and the Camera page under More) | `upload_ui.py`, `camera_ui.py`, `corner_picker.py` |
| Stretch the screen video to 1024 x 570 | `screen_map.py` |
| Read pictures, reconnect | `run_sources.py` (`CameraSource`) |
| The pose model and its settings | `pose_model.py`, `pose_backends.py`, `models/pose/` |
| Draw the skeletons on a picture | `pose_preview.py` (Replay: `live_pose_view.py`) |
| Start a camera run, show the live panel and charts | `live_session.py`, `live_view.py`, `live_log.py` |
| Turn points into scores, then alerts | `live_scoring.py`, `monitor.py` |

On Windows only one program can use a camera at a time. The pages share one open camera; it is released by the Stop
button, by a new camera test, by a reset, and before a camera run starts.

## 4. Setup for the presentation

1. Put the camera on a holder, 30 to 50 cm from the phone or iPad, straight in front of the screen.
2. Play a CalMS21 top-view video on the screen (for example `mouse064_task1_annotator1_web.mp4`). Brightness high,
   auto-dim off, no lamp or window reflecting in the glass.
3. Focus the camera on the screen (half-press the shutter button). A blurry picture gives no mice.
4. Follow the guided flow in section 2. Check Replay first: both skeletons should sit on the mice.

**Nikon D7500 on a Mac:** install the Nikon Webcam Utility. If the app shows the Utility's "connect your camera" screen
instead of video, quit the Utility from its menu bar icon and open it again. If the picture is black, the camera is
asleep: half-press the shutter. In the camera menu set Custom settings > c3 Power off delay > Live view to the longest
time, and keep the battery charged (or use the AC adapter).

## 5. What video quality the pose model needs

- CalMS21 frames are 1024 x 570, 30 frames per second, grey. A mouse body is about 190 px long. A head is about 58 px wide.
- The screen video should fill **at least 1024 pixels of width** in the camera picture. A 1920 x 1080 camera gives about
  1300. A 1280 x 720 camera gives about 950. A 640 x 480 webcam gives about 600, which is too blurry.
- Sharp focus, fast shutter (1/60 s or faster), no glare, steady light, 30 frames per second.

## 6. What is real and what is not

- **Real:** camera opening, live video, corners and stretched view, finding the mice in each camera picture (the YOLO
  pose model), live scores, alerts, questions, the event log, the live charts, the report, tests.
- **Limits:** the pose model and Model B were trained on CalMS21 top-view videos (one black and one white mouse). A camera
  filming a screen that plays such a video works; another cage, view or pair of mice gives scores that mean little.
  Classifier scores are model output, not ground truth.
- **Tried and failed:** the public DeepLabCut SuperAnimal-TopViewMouse model. It found a mouse in 31 of 150 frames of the demo
  video, never the second mouse, and it needs about 5.8 seconds per frame on a laptop (a live camera needs 0.03).
- **Tried and works, but slowly:** MARS (Segalin et al. 2021, license CC BY-NC 4.0), whose pretrained models make the same 7
  keypoints in the CalMS21 order. On 20 frames of the demo video it found both mice in every frame, with a median error of
  2 to 7 px against the saved CalMS21 keypoints (a body is about 190 px). On a laptop CPU it needs about 0.9 s per frame,
  so it is the offline route (section 8).

## 7. How to say it in the presentation

"A camera films a screen that plays a mouse video. A small pose model finds both mice in every picture, about 30 times a
second. The behavior model scores attack, investigation and mount from those points, and the alert fires only for the
rule the researcher confirmed. Everything is logged, and a report is written at the end."

## 8. Offline route with MARS, on this computer (or on Google Colab)

Live pose on this laptop is too slow, so a recorded video is analyzed afterwards and the result is loaded into the app.

**Where things are:** the three MARS model files (584 MB of `.pb` files; CC BY-NC 4.0, from Caltech Data record 801be-66057)
go in `models/mars/` in the project, downloaded from the GitHub release `mars-models-v1` (`models/mars/README.md`), or in
`%USERPROFILE%\mars_local\models\`. The Python environment with TensorFlow is `%USERPROFILE%\mars_local\env\` (not in Git).

**Speed on this laptop (CPU, 8 cores):** about 1.2 frames per second after a start-up of about 45 s. So 300 frames (10 s of video) take
about 5 minutes, and 1 minute of video takes about 25 minutes. Start with a short clip.

**In the app (recommended):**

1. Open the **Camera** page (left menu) and press **Test the camera**. The live video starts. (The Upload page is for recorded data only.)
2. Drag the 4 red corners onto the 4 corners of the video on the screen. Check the stretched view.
3. Mode **"Record, then analyze with MARS"** (the default). Press **Start recording**, play the video on the screen, press
   **Stop recording** (10 to 20 seconds first). The recording is saved as 1024 x 570 at 30 frames per second in
   `outputs/recordings/<time>/camera.mp4` (not in Git).
4. Press **Analyze with MARS (about N min)**. It runs in the background with a progress bar; you can leave the page.
5. When it says finished, press **Use this recording**. Then Rules and Real session as usual.

About 45 s of the time is start-up, then about 1.2 frames per second.

**On the command line (same result):**

1. Record the camera video (for example with the Windows Camera app). Keep the screen steady in the picture.
2. In the app, Camera page: drag the 4 corners on the live video. Copy the line "Corners for the Colab notebook"
   (it works for the local command too).
3. Run, from the project folder (keypoints.parquet and stretched.mp4 appear in the folder you run it from):

       scripts\mars_local.cmd my_video.mp4 --corners 0.23,0.22,0.81,0.22,0.81,0.65,0.23,0.65 --max-frames 300

4. In the app: Upload page, "Upload a video and its keypoints", choose `stretched.mp4` and `keypoints.parquet`. Then Rules and
   Real session as usual.

Note: OpenCV on this laptop cannot write H.264, so the recording is an older mp4 format. It is fine for the analysis. To see it in
the browser, the project's optional converter (`imageio-ffmpeg`) must be installed (the page then shows a "Prepare the video" button).

The script (`scripts/mars_pose_offline.py`) does the same steps as MARS: stretch the screen, two detectors for the black and the
white mouse, crop, hourglass heatmaps, brightest point. It only reads the `.pb` model files. It never opens the `.pkl` files in the
MARS zip (pickle can run code). The models are not stored in Git (they are big; the team keeps big files on the shared Drive).

**Google Colab (not used so far):** `colab/mars_pose_colab.ipynb` does the same on a free GPU. Its speed is not measured.

**If the environment is missing** (new computer): create it with `python -m venv %USERPROFILE%\mars_local\env`, install
`tensorflow numpy opencv-python-headless pandas pyarrow` into it, then run
download the model files as in `models/mars/README.md` (or run
`python scripts\mars_pose_offline.py --download-only --models-dir %USERPROFILE%\mars_local\models`: 543 MB from Caltech, md5 checked).

## 9. Acceptance test of the live camera path

`scripts/live_camera_acceptance.py` starts the real run program (the one the app starts) on a camera, lets it run, and checks every step with
PASS, WARN or FAIL. Run it when the camera is connected and the trained model is set up in `models/pose/pose_model.json` (docs/pose_model.md):

    python scripts/live_camera_acceptance.py --camera 0 --seconds 30 --corners 0.23,0.22,0.81,0.22,0.81,0.65,0.23,0.65

It checks: the pose model loads; the pose model needs 33 ms or less per picture (measured on 60 pictures); the camera gives pictures (25 or more per
second is a PASS, a 15 fps webcam is a WARN); the run program starts, does not crash and has no camera interruption; it keeps up with 30 frames per
second (25 or more is a PASS); the scores reach the monitor; alerts or questions appear (with `--expect-events`). Add `--expect-mice` when the
camera sees the mice. The exit code is 0 only when nothing FAILED.

**Rehearsal without a trained model** (a stub gives perfect keypoints, so the pose speed check is SKIPPED and the result is NOT a test of the model):

    python scripts/live_camera_acceptance.py --camera 0 --stub-poses data/raw/calms21_video/mouse064_task1_annotator1_poses.parquet --start 1800 --seconds 40 --expect-events
    python scripts/live_camera_acceptance.py --video data/raw/calms21_video/mouse064_task1_annotator1.mp4 --start 1800 --stub-poses <same poses file> --seconds 40 --expect-events

**What the rehearsals found and fixed (2026-10-25):** even with an instant stub the chain reached only 17 frames per second. (1) The camera was read
only when the program asked, so every slow moment dropped pictures: a reader thread with a half-second queue now reads it all the time. (2) Frame
numbers came from the clock, so timing wobble made gaps that end alerts, and a 15 fps webcam made a gap on every picture: small gaps (up to 3
numbers) are now filled with repeated keypoints. (3) The scorer recomputes the features of the last 6 s for every chunk (about 130 ms): live cameras
are scored in chunks of 15 frames (up to half a second more delay). (4) A crash when "Stop recording" closed the video while the camera thread was
writing: a lock now prevents it. Result with the real USB webcam and the stub: camera 25.3 pictures per second, run program 29.5 frames per second
over 40 s (slowest 10 s: 29.9), all scores reached the monitor, 8 alerts and 1 question.

## 10. Next steps

1. Measure how well the live pose model matches the saved CalMS21 keypoints on more videos and lighting setups.
2. Try a real cage instead of a screen; the models may need retraining for a new view.
