# How the camera is used in this project

A camera is meant to be the live "eyes" of the project. Today it is real in one part only: the live video on the screen.
The part that finds the mice in a camera picture (a pose model) does not exist yet. So the alerts do **not** come from the
camera picture.

## 1. The full chain

1. **Camera** takes pictures, about 30 per second.
2. **Screen stretch** (`src/live_tracker/screen_map.py`) finds the iPad video in the picture and stretches it to
   1024 x 570, the CalMS21 size.
3. **Pose model** (`src/live_tracker/pose_model.py`) turns each picture into 7 body points per mouse: nose, two ears, neck,
   two hips, tail base. **This part is missing.** See `docs/pose_model.md`.
4. **Live scorer** (`src/live_tracker/live_scoring.py`, with Model B) reads the last 6 seconds of points and gives a score
   for each behavior: attack, mount, investigation.
5. **Monitor** (`src/live_tracker/monitor.py`) compares the scores with the rule the researcher approved, for example
   "attack for at least 0.5 s". It makes an **alert** or a **question**.
6. **Event log** (`journal.jsonl`) and **report**: every event is saved. At the end the program writes a report.
7. **Page** (`src/live_tracker/live_view.py`) shows the counts, the motion chart, the alerts and the report.

Steps 4 to 7 are built and tested. Step 3 is the gap.

## 2. The three ways the camera appears

**A. Demo camera** (Upload page, "Demo camera").
No real camera. A recorded video plays, and the program reads the saved CalMS21 keypoints.

**B. Staged demo** (Camera page, "Staged demo").
- A USB camera films the iPad. The page shows the live camera video (`camera_live.py`). This part is real.
- Behind it, the program reads the **saved keypoints** of the same video at real speed.
- The alerts come from those saved keypoints, **not from the camera picture**.
- The researcher presses GO and presses play on the iPad at the same moment. The error is about 0.3 seconds.
- The page shows a yellow "STAGED DEMO" box, so nobody is misled.

**C. Real camera run** (Camera page, "Live real pose").
- `CameraSource` in `run_sources.py` opens the camera, reads pictures, and sends each one to the pose model.
- It numbers the frames by time, and it tries to reconnect if the camera drops out.
- If no pose model is set up, the run is refused and the page says why.
- No pose model exists, so this path has never run on a real camera.

## 3. Which file does what

| Job | File |
|---|---|
| List cameras and check one works | `camera_check.py`, `scripts/find_cameras.py` |
| Keep one camera open and share it between pages | `camera_live.py` |
| Live video, corner dragging, record and analyze, staged or real (own page: Camera) | `camera_ui.py`, `corner_picker.py`, `pages/7_Camera.py` |
| Stretch the iPad video to 1024 x 570 | `screen_map.py` |
| Read pictures, reconnect | `run_sources.py` (`CameraSource`) |
| The empty slot for the pose model | `pose_model.py` |
| Start a camera run or a staged run, show the live panel | `live_session.py`, `live_view.py` |
| Turn points into scores, then alerts | `live_scoring.py`, `monitor.py` |

On Windows only one program can use a camera at a time. The Upload page and the Real session page share one open camera.
It is released by the Stop button, by a new camera test, by a reset, and before a real camera run starts.

## 4. Setup for the presentation (staged demo)

1. Put the camera on a holder, 30 to 50 cm from the iPad, straight in front of the screen. All four corners of the video
   must be inside the picture.
2. Put the browser-friendly copy of the demo video (`mouse064_task1_annotator1_web.mp4`, H.264) on the iPad. Do not press play yet.
3. iPad brightness high, auto-dim off, no lamp or window reflecting in the glass.
4. Camera page (left menu): press "Test the camera".
5. Move the 4 corner sliders until the green shape sits on the edge of the iPad video. The stretched view should be filled
   by the video, with no border.
6. Choose "Staged demo", then "Use the staged demo".
7. Rules step: confirm the rule. Real session step: press "Start the staged session". When it says "Ready", press GO and press
   play on the iPad at the same moment.

## 5. What video quality a pose model would need

These are targets from the demo data, not tested limits (there is no working pose model).

- CalMS21 frames are 1024 x 570, 30 frames per second, grey. A mouse body is about 190 px long. A head is about 58 px wide.
- The iPad video should fill **at least 1024 pixels of width** in the camera picture. A 1920 x 1080 camera gives about 1300.
  A 1280 x 720 camera gives about 950. The 640 x 480 webcam gives about 600, which is too blurry for a pose model.
- Sharp focus, fast shutter (1/60 s or faster), no glare, steady light, 30 frames per second.
- Today's webcam gives about 15 pictures per second at 640 x 480. That is fine for showing the video, not for analysis.

## 6. What is real and what is not

- **Real:** camera opening, live video, corners and stretched view, the staged run, alerts, log, report, tests.
- **Not real:** finding the mice in the camera picture. The staged demo uses a saved file instead.
- **Tried and failed:** the public DeepLabCut SuperAnimal-TopViewMouse model. It found a mouse in 31 of 150 frames of the demo
  video, never the second mouse, and it needs about 5.8 seconds per frame on this laptop (a live camera needs 0.03).
- **Tried and works, but slowly:** MARS (Segalin et al. 2021, license CC BY-NC 4.0), whose pretrained models make the same 7
  keypoints in the CalMS21 order. On 20 frames of the demo video it found both mice in every frame, with a median error of
  2 to 7 px against the saved CalMS21 keypoints (a body is about 190 px). On a degraded copy (small, blurry, compressed,
  dimmer, like a webcam filming a screen) it still found both mice, but the white mouse's hips and tail base were off by
  17 to 37 px. On a laptop CPU it needs about 0.9 s per frame (live needs 0.03 s). Caveats: the saved keypoints were made by
  MARS itself and may come from its training data; MARS needs a top view with one black and one white mouse.
- **The offline route (Google Colab):** `scripts/mars_pose_offline.py` and `colab/mars_pose_colab.ipynb` turn a recorded video
  into a keypoints file in the layout the app reads. See section 8. The speed on a Colab GPU is not measured yet.

## 7. How to say it in the presentation

"A camera films a screen, and the page shows it live. The behavior alerts are computed from the saved keypoints of that
video. A pose model that finds the mice in the camera picture is the next step, and the program already has a place for it."

## 8. Offline route with MARS, on this computer (or on Google Colab)

Live pose on this laptop is too slow, so a recorded video is analyzed afterwards and the result is loaded into the app.

**Where things are (outside the project, not in Git):** `%USERPROFILE%\mars_local` holds the MARS models (`models\`, 328 MB of `.pb`
files) and a Python environment with TensorFlow (`env\`). MARS models: CC BY-NC 4.0, from Caltech Data record 801be-66057.

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
`python scripts\mars_pose_offline.py --download-only --models-dir %USERPROFILE%\mars_local\models` (543 MB, md5 checked).

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

1. Train the fast model (Colab notebook), put it in `models/pose/`, and run the acceptance test above WITHOUT `--stub-poses`. This is the real test.
2. Run it with the Nikon (camera number of the Nikon Webcam Utility) when it is charged.
3. Test the staged demo and the recording in the browser with the real camera and the iPad.
