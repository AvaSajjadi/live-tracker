# Running the monitor on a lab computer

This page says what works today, what does not, and how to try it. It is written from the code and from
measurements made on 2026-10-24. Nothing here claims that a real camera works: **no camera has been tested.**

## The idea

```
camera ──cable──▶ lab computer (always on) ──▶ messages to the researcher's phone
                  │
                  ├─ run program   reads frames, scores them, runs the monitor, writes the event log and the report
                  └─ dashboard     sets the rule and the schedule, starts and stops the run, shows results
```

The run is a **separate program** (`monitor_runner`). Closing the browser or the dashboard does not stop it.
The dashboard is the control panel and the viewer. It does not do the night's work.

## What exists and what does not

| Part | State |
|---|---|
| Run schedule: "for N hours" or "from - to", Confirm / Reject, saved as a file | Built and tested (`run_schedule.py`, `schedule_workflow.py`, Rules page) |
| Run program: waits for the start, runs to the end, event log, answers, report, messages | Built and tested (`monitor_runner.py`) |
| Live scoring: pose keypoints, one frame at a time, to Model B scores | Built and tested (`live_scoring.py`) |
| Sources: score file, recorded keypoints, camera reader | Score file and recorded keypoints work. The camera reader is built and tested with a fake camera only |
| Messages: log file, and phone through ntfy | Built and tested. Phone messages are off unless you set a topic |
| Run page: start, stop, watch, answer, report | Built and tested (`pages/6_Run.py`) |
| **A mouse pose model** (picture to keypoints) | **Does not exist in this repository.** A camera run is refused until one is supplied |
| A real camera on a real cage | **Not tested** |

## Try it now (no camera)

1. Open **Rules**. Type a rule, click **Propose rule**, then **Confirm**.
2. On the same page, under **Run schedule**, choose a length (at least one minute) and **Confirm** it.
3. Open **Run**. Pick the schedule. Choose "Demo: made-up scores" or "Recorded validation session".
   The recorded session needs the CalMS21 train file and `models/model_b` on this computer.
4. Click **Start run**. Watch the counts. Answer a question with Confirm or Reject.
5. When the run ends, read the **Report** and download it.

From a terminal, a run folder can also be started by hand:

```
PYTHONPATH=src python -m live_tracker.monitor_runner --config outputs/runs/<run folder>/config.json
```

Create a file named `STOP` in the run folder to stop it early. The program still writes the report.

## Phone messages (optional)

Messages always go to `notifications.log` in the run folder. To also get them on a phone, install the ntfy app,
pick a long random topic name, and set it before the run program starts:

```
LIVE_TRACKER_NTFY_TOPIC=<long random topic>
LIVE_TRACKER_NTFY_SERVER=https://ntfy.sh        (optional; the default)
```

Anyone who knows the topic can read the messages, so treat it like a password. A message names the behavior,
the confidence and the time. It never contains video or positions. If the phone link fails, the run goes on and the
failure is only noted.

## What a run writes (in `outputs/runs/<run>/`, not in Git)

| File | What |
|---|---|
| `config.json` | what was run: the approved schedule, the approved rules, the source |
| `journal.jsonl` | the event log (append-only; the run program is its only writer while it runs) |
| `status.json` | a heartbeat every few seconds: state, counts, time left |
| `answers.jsonl` | answers given in the dashboard while the run was going; the program applies them |
| `STOP` | create it to stop the run |
| `report.md`, `report.json` | the end-of-run report: outcome, planned and recorded time, gaps, events, answers |
| `notifications.log` | every message that was sent |
| `runner.log`, `error.txt` | the program's output, and the error if it failed |

A question that is still open when the run ends can be answered the next morning: the answer goes straight
into the log and the report is written again. If the program died (a power cut), the page says so.

## Rules the run program keeps

- It refuses a schedule or a rule that is not approved.
- The monitor only sees frames inside the schedule window.
- It never reads the CalMS21 test split.
- A lost camera is a **gap** in the frame numbers, never "the mouse is still". Every gap is listed in the report.
- A run that starts late or ends early is reported as **partial**, with the missing time. One that is stopped by
  the researcher is reported as **stopped early**.
- A restart inside the window starts a **new run folder and a new report**. A run is not resumed.

## What was measured (2026-10-24, one machine, validation data only)

- **Live scores equal whole-session scores.** On one validation session (7,150 frames, Model B), a 6 s moving buffer
  scored in chunks of 5 frames gave the same scores as scoring the whole session: largest difference 5e-13.
  This is one session, not a guarantee for every setup. Repeat it with
  `LIVE_TRACKER_SLOW=1 python -m pytest tests/test_live_scoring.py -k real`.
- **Speed.** About 95 ms per 5-frame chunk (95th percentile 153 ms) against a 167 ms budget at 30 fps. On this
  laptop the live scorer used a bit more than half of real time. A slower computer may not keep up.
- **End to end.** The run program, started as its own process, played that session in real time (255 s for a 238 s
  recording). It produced **46 events, and the whole-session calculation produced the same 46**, with identical
  start times, confidences and alert / ask decisions. The validated demo rule was used (ask 0.6, 0.5 s, alert 0.8).
  (Measured with the models trained under scikit-learn 1.7.2. The models were retrained under 1.9.1 on 2026-10-25; the
  demo session then has 44 events, and the live scores still equal the whole-session scores in the real-data test.)

## Limits you should know

- **No pose model, no camera.** The camera reader needs a model that turns each picture into the 7 CalMS21 keypoints
  of both mice, in the CalMS21 image frame (1024 x 570 pixels), with a confidence value. Model B was trained on
  top-down views of two mice in that setup. A different view or box gives meaningless scores.
- **Scores are not calibrated.** Model B gives raw scores. A score of 0.8 is not an 80 % chance (see
  `docs/model_settings.md`).
- **Demo speed.** A speed above 1 plays a recording faster. At a very high speed the time the computer needs counts
  as part of the night, so the window can end early.
- **Scores come in chunks,** so a score appears up to 0.17 s after its frame. Alerts are that much later.
- **Report time tolerance.** A run counts as complete if it starts and ends within one minute of the plan.
- **The schedule must be at least one minute and at most one week.**

## Checklist for a real overnight setup

1. A lab computer that stays on. Turn off sleep and automatic restarts for the night.
2. A camera on a cable (USB or Ethernet), top-down, with infrared light if the room is dark.
3. A pose model that fits the setup, and a check that its keypoints match the CalMS21 layout.
4. A dedicated network, or a cable, for the camera. Wi-Fi can drop for minutes.
5. Enough disk space for the run folder, and a place to copy the report.
6. A private way to open the dashboard from outside (for example a VPN). Do not open a port to the internet.
7. Ask the animal facility whether the video may leave the room. Our run program keeps the video on the lab computer
   and sends only short text messages.
