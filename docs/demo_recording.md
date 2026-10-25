# The live demo and the screen recording

The **Live demo** page plays a CalMS21 video and, at the same moment, runs the real run program on that
video's pose keypoints: live scores from Model B, the validated demo rule, the monitor, the event log. The
audience sees the mice and the alerts together.

**What to say, and what not to say.** This is a **simulated live stream from a recording.** No camera is used,
and no camera has been tested (there is no mouse pose model yet; see `docs/lab_run.md`). The video shows the
mice, but the system works from the saved keypoints, not from the picture. The session was chosen because it has
the most events among the validation sessions, so it is the best case, not a typical one.

## The video

| | |
|---|---|
| Session | `task1/train/mouse064_task1_annotator1` (a validation session) |
| Why this one | most events: 46 (41 alerts, 5 questions), 45 of them inside a human-labeled attack; the next best session has 19 |
| Length | 238.3 s |
| Source | CalMS21, Caltech Data, CC BY 4.0: https://data.caltech.edu/records/s0vdx-0k302 , file `task1_videos_mp4/train/mouse064_task1_annotator1.mp4` inside `task1_videos_mp4.zip` (28 GB; the video is about 200 MB inside it) |
| Credit | Sun et al. 2021, "The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions", arXiv:2104.02710 |
| In the project | `data/raw/calms21_video/mouse064_task1_annotator1.mp4` and `..._poses.parquet` (git-ignored, never in Git) |

The poses file is made from the CalMS21 train file with
`python scripts/prepare_demo_session.py --session task1/train/mouse064_task1_annotator1 --out data/raw/calms21_video/mouse064_task1_annotator1_poses.parquet`.
The script refuses a test session. Do not use test videos for the demo.

## Before recording

1. Plug the charger in. Set the power mode to Best performance. Turn off sleep.
2. Close other programs and browser tabs. Turn on Focus assist so no pop-ups appear.
3. Start the dashboard from the project folder:
   ```
   $env:PYTHONPATH = "src"
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
4. Open **Live demo** in the menu. Zoom the browser (Ctrl and +) until it is easy to read.
5. If you use a TV, connect with the HDMI cable and press Windows + P, then choose Duplicate.

## Recording (Windows 11)

- **Snipping Tool:** press Windows + Shift + R, drag over the browser window, click Start.
- **or Xbox Game Bar:** click the browser, press Windows + Alt + R to start and again to stop. The file goes to `Videos\Captures`.

Steps on the page:
1. Read the rule and the length on the page, then click **Confirm the rule and start the demo.** This click is the
   researcher's approval of the rule and the run.
2. The page says "Getting ready" for a few seconds, then the video starts and the monitor starts with it.
3. The first alert comes at about **0:11** into the video. Record about **60 to 90 seconds**, so the viewer sees
   several alerts, a question, the counters and the table. Stop the recording.
4. Watch the recording once. The alerts should appear just after the fighting starts.

If the video does not start by itself, click its play button once. If the alerts come **before** the fight in the video,
raise **Sync** (under the rule text) a little before starting again. If they come late, lower it.

## What the page tells you

- **"Keeping up with real time (slowest delay ..)"** means the laptop is fast enough. If it says the laptop **fell behind**,
  close other programs and try again, or the alerts will be late by that much.
- Questions (asks) are answered on the **Run** page, not here, so that the video is never redrawn.
- When the demo ends, the report is on the **Run** page. It states whether the run was complete.

## Before uploading

- Check the file size against the Canvas limit.
- Watch the whole recording once. Check that no private information is on the screen (tabs, notifications).
- In the text, call it a **simulated live stream from a recording** and give the dataset credit.
