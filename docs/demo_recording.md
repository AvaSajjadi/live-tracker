# The guided flow, the live session, and the screen recording

The dashboard now has a **guided flow** of four steps. Each page has a bar at the top that shows where you are, and
**Back** and **Next** buttons.

| Step | Page | What you do |
|---|---|---|
| 1 | **Upload** | three choices: **Simulated camera (recorded video)**, **Upload a video** (with its keypoints), **Upload a CSV** of mouse positions. The video shows on the right as soon as there is one. No real camera can be analysed yet, so the camera choice is a clearly labeled **demo camera**: a recorded video stands in for the camera feed |
| 2 | **Rules** | tell the agent what to look for **in your own words**; the smart reading copes with typos and everyday wording and shows what it understood; then Confirm the rule |
| 3 | **Replay** | the replay page (a recorded run, played back) |
| 4 | **Real session** | the recording plays on one side while the analysis runs on the other: alerts, counts, motion charts, the event table, and the report at the end |

More pages are under "More" in the menu: Investigation, Live Agent, Run (start and stop runs, answer questions, read
reports) and Motion tool (the old home page for a positions CSV).

## What the Real session page shows

- **Left:** the analysis. **Right:** the video preview and compact counts; analysis from the real run program, working on the keypoints of the same
  recording at real speed: the latest alert, the counts, the speed of each mouse and the distance between them
  (last 30 s), a table of events, and a warning if the laptop falls behind. When the session ends, the report is shown.
- It is a **simulated live stream from a recording.** No camera is used and no camera has been tested (there is no mouse
  pose model yet; see `docs/lab_run.md`). The video shows the mice, but the system works from the saved keypoints, not
  from the picture.
- A video alone cannot be analysed: the behavior scores come from keypoints. A CSV of positions (x_a, y_a, x_b, y_b) gives
  motion and tracking quality only, with no behavior scores and no alerts.

## The demo recording

| | |
|---|---|
| Session | `task1/train/mouse064_task1_annotator1` (a validation session) |
| Why this one | most events: 44 (40 alerts, 4 questions) with the models trained in the project's Python (scikit-learn 1.9.1), 43 of them inside a human-labeled attack; the next best session has 19. With the earlier models (scikit-learn 1.7.2) it was 46. The count depends a little on the scikit-learn version. It is the best case, not a typical one |
| Length | 238.3 s |
| Source | CalMS21, Caltech Data, CC BY 4.0: https://data.caltech.edu/records/s0vdx-0k302 , file `task1_videos_mp4/train/mouse064_task1_annotator1.mp4` inside `task1_videos_mp4.zip` (28 GB; the video is about 200 MB inside it) |
| Credit | Sun et al. 2021, "The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions", arXiv:2104.02710 |
| In the project | `data/raw/calms21_video/mouse064_task1_annotator1.mp4` and `..._poses.parquet` (git-ignored, never in Git) |

The poses file is made from the CalMS21 train file with
`python scripts/prepare_demo_session.py --session task1/train/mouse064_task1_annotator1 --out data/raw/calms21_video/mouse064_task1_annotator1_poses.parquet`.
The script refuses a test session. Do not use test videos for the demo.

### The video format

This video uses an older format (`mp4v`) that **browsers cannot play**. The page says so and offers **"Prepare the
video for the browser"**, which writes an H.264 copy next to the original (`..._web.mp4`; the original is not changed).
That button needs a converter, the optional package `imageio-ffmpeg`, installed in the project's Python:
`.\.venv\Scripts\python.exe -m pip install imageio-ffmpeg`. Without it, the analysis still runs, but without the picture.

## Before recording

1. Plug the charger in. Set the power mode to Best performance. Turn off sleep.
2. Close other programs and browser tabs. Turn on Focus assist so no pop-ups appear.
3. Start the dashboard from the project folder (the last option keeps it closed to other computers):
   ```
   Set-Location "C:\Users\AnTiC\Downloads\live-tracker\live-tracker"
   $env:PYTHONPATH = "src"
   & ".\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8766 --server.address localhost
   ```
4. Open http://localhost:8766. Zoom the browser (Ctrl and +) until it is easy to read.
5. If you use a TV, connect with the HDMI cable and press Windows + P, then choose Duplicate.

## Recording (Windows 11)

- **Snipping Tool:** press Windows + Shift + R, drag over the browser window, click Start.
- **or Xbox Game Bar:** click the browser, press Windows + Alt + R to start and again to stop. The file goes to `Videos\Captures`.

Steps on the pages:
1. **Upload:** choose **Simulated camera (recorded video)**. The main success message says
   **"Simulated camera ready — prerecorded video, no physical camera connected."** The video appears on the right.
   Describe it as a recording with saved keypoints, then choose **Next**.
2. **Rules:** write the rule in your own words. For the validated demo rule, include the thresholds, for example
   `let me know if the mice fight for more than 0.5 seconds, ask threshold 0.6, alert threshold 0.8`. Click **Propose rule**,
   read "Your sentence was read as ..." and the assumptions, click **Confirm**, then **Next**.
3. **Replay:** click **Next** (or look at it, if you want).
4. **Real session:** read the recording and the rule, click **Start recorded session**. It says "Getting ready" for a few
   seconds, then the video and the analysis start together.
5. Record enough of the run to show alerts, asks and the event table, then answer a question on **Run** and inspect
   the completion report. Event timing and counts depend on the model artifact, approved rule and machine; rehearse
   locally instead of promising a specific alert time. Stop the recording and watch it once.

If the video does not start by itself, click its play button once. If the alerts come **before** the fight in the
video, raise **Sync** (under the rule text) a little before starting again. If they come late, lower it.

## What the page tells you

- **"Keeping up with real time (slowest delay ..)"** means the laptop is fast enough. If it says the laptop **fell
  behind**, close other programs and try again, or the alerts will be late by that much.
- Questions (asks) are answered on the **Run** page, not here, so that the video is never redrawn.
- When the session ends, the full report is on the **Run** page. It states whether the run was complete.

## Before uploading

- Check the file size against the Canvas limit.
- Watch the whole recording once. Check that no private information is on the screen (tabs, notifications).
- In the text, call it a **simulated live stream from a recording** and give the dataset credit.

## Offline backup and handoff

Use [demo_readiness.md](demo_readiness.md) to inventory local files and prepare an offline rehearsal. This document is
an instruction sheet, not evidence that a backup recording or slides have been produced. H5 remains incomplete until
an approved held-out run is completed by the team.
