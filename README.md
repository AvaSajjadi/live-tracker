# Live Tracker

**Turning mouse pose data into researcher-approved behavior alerts, with every step kept honest about what the data
can and cannot show.**

Behavioral neuroscience labs record hours of video of mice interacting. Finding the moments that matter (an attack, a
mount, a close investigation) still mostly means watching footage or labeling it frame by frame. Live Tracker is a
hackathon prototype that watches a **live camera** (or a recording) and:
- finds both mice in every camera picture with a small trained pose model (7 body points per mouse, about 30 times a
  second);
- checks how trustworthy the tracking is;
- computes motion features with causal filters and formulas;
- scores attack, investigation and mount with a behavior classifier trained on the public CalMS21 dataset;
- lets a researcher describe, in plain English, what they want to be alerted to.

A proposed rule does nothing until the researcher confirms it. A deterministic monitor, with no language model in the
loop, then turns per-frame scores into timestamped, journaled alerts and questions, and writes a report at the end.

> **Status:** hackathon project (October 2026). Nothing here is a validated scientific instrument; see
> [Scientific integrity and limitations](#scientific-integrity-and-limitations).

## Latest demo

A recording of the full live run (camera → pose model → rule → alerts → report) is in
[`latest_demo/`](latest_demo/). The quickest way to try it yourself is [Quick start](#quick-start).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[camera]" -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open http://localhost:8501 and follow the guided flow:
1. **Upload:** Connect a camera → camera number `0` → **Test the camera** → **Live real pose** → **Use this camera**.
2. **Rules:** type `Alert me if the mice fight for more than 2 seconds` → **Propose rule** → **Confirm**.
3. **Replay:** check that both mice have a skeleton drawn on them (red: the black mouse, blue: the white mouse).
4. **Real session:** **Start the camera session**. Alerts, live charts and the report appear on the page.

The models (`models/`) are in the repository, so no training is needed. Point the camera at a top-view CalMS21 video
(for example on a phone or tablet), in focus. Camera setup and troubleshooting: [`docs/camera.md`](docs/camera.md).

## Why this matters

Social behavior in mice is a core readout in neuroscience: aggression, mounting and investigation are studied in
models of stress, disease and drug effects. Automatic detection can save annotation time, but only when it is
transparent about three things:
- which frames it could actually see (tracking quality);
- what its thresholds are;
- that a researcher, not the software, decides what counts as an event.

Live Tracker is built around those three constraints.

## What is on `main`

| Area | What it does | Where |
|---|---|---|
| **Tracking quality** | Flags missing, out-of-frame, jumping, frozen and low-confidence keypoints; fills only short interior gaps (and marks them as imputed); reports per-window quality and tracking-loss events. Retrospective, not causal: gap filling uses the frame after the gap, and a frozen run is flagged from its first frame once it is long enough | `src/live_tracker/quality.py` |
| **Causal smoothing** | EMA and One-Euro filters that use only the current and past frames, reset after long gaps and keep missing data missing | `src/live_tracker/filters.py` |
| **Motion features** | Per-mouse features (speed, acceleration, heading, turn rate, trailing-window path measures, pauses) and pair features (body-centre distance and its rate, relative speed, close time, speed correlation), plus a per-frame `tracking_ok` flag. The feature formulas use only current and past frames, but `compute_features()` first runs the retrospective tracking-quality step on unflagged input | `src/live_tracker/features.py`, `INTERFACES.md` §5 |
| **CalMS21 loader and split** | Loads CalMS21 Task 1 JSON into a long table and makes a session-level train / validation / test split (the committed split has 56 / 14 / 19 sessions) | `src/live_tracker/calms21.py`, `splits/split_manifest.json`, `docs/calms21_schema.md` |
| **Behavior classifier** | Class-weighted logistic regression: model A on basic pose-geometry features, model B on basic + motion features. Trains on train sessions, tunes on validation, writes per-frame attack / mount / investigation probabilities | `src/live_tracker/classify.py`, `src/live_tracker/basic_features.py` |
| **Evaluation** | Frame metrics, confusion matrix, predicted bouts, event and bout metrics, onset / offset timing against CalMS21 labels | `src/live_tracker/evaluate.py` |
| **Rules from plain English** | An offline, rule-based parser turns a sentence into a proposed rule, or asks for clarification. An optional Claude cross-check only accepts what the offline parser already accepts | `src/live_tracker/agent.py`, `src/live_tracker/claude_agent.py` |
| **Researcher approval** | Every rule starts as `proposed`; only an explicit Confirm approves it. The workflow is UI-independent and tested without a browser | `src/live_tracker/rules.py`, `src/live_tracker/rule_workflow.py` |
| **Streaming monitor** | Deterministic: approved rules + per-frame scores → alert / ask / log events. Decisions use only past and current frames; tracking loss ends a bout | `src/live_tracker/monitor.py` |
| **Event journal** | Append-only lifecycle journal with crash recovery, plus the `INTERFACES.md` §4 event-log export | `src/live_tracker/event_log.py` |
| **Live camera** | Opens a USB camera (tested with a Nikon D7500 through the Nikon Webcam Utility), optionally stretches a filmed screen to the CalMS21 frame, and reconnects if the camera drops out | `camera_live.py`, `run_sources.py`, `screen_map.py` |
| **Pose model** | A small YOLO pose model trained on the MARS hand-labeled set: 7 keypoints for each of the two mice, about 10 ms per picture on a MacBook Air | `models/pose/`, `pose_model.py`, `pose_backends.py`, `docs/pose_model.md` |
| **Streamlit dashboard** | A guided flow (Upload, Rules, Replay, Real session) with the live camera, the skeletons, live charts of the scores and motion, alerts and the report; plus Investigation, Live Agent, Run, Camera and Motion tool pages | `app.py`, `pages/` |
| **Synthetic test data** | A labeled synthetic pose CSV and a synthetic per-frame score generator. Every synthetic session id starts with `SYNTHETIC_` | `example_pose_demo.csv`, `src/live_tracker/fake_scores.py` |

## How the parts fit together

```
 live camera ── pose_model.py (YOLO) ── 7 keypoints per mouse, every picture
        │            or
        │      CalMS21 Task 1 JSON / an uploaded recording with keypoints (calms21.py, uploads.py)
        ▼
 quality.py ── flags, short-gap filling, tracking_ok     (retrospective: see limitations)
        │
        ▼
 features.py (+ filters.py), basic_features.py ── feature table (INTERFACES.md §5)
        │
        ▼
 classify.py (model A / B)                       researcher instruction (plain English)
        │                                               │
        │ per-frame probabilities                       ▼
        │ (INTERFACES.md §2)                     agent.py ── proposed rule
        │                                               │
        │                                               ▼
        │                                        researcher Confirm (rules.py)
        │                                               │ approved rule (INTERFACES.md §3)
        ▼                                               │
 monitor.py ◄───────────────────────────────────────────┘
        │  alert / ask / log events
        ▼
 event_log.py ── journal, §4 event-log export, live charts, report
        │
        ▼
 Real session page; replay and evidence-grounded investigation (offline, association-only)
```

The monitor has two separate inputs. Classifier probabilities say how likely each behavior is in each frame. They
never create or approve a rule; only the researcher's Confirm does that.

The data formats passed between these parts are defined in `INTERFACES.md`:
- §1: pose data;
- §2: per-frame scores;
- §3: rules;
- §4: the event log;
- §5: the feature table.

## Installation

Requires **Python 3.10 or newer** (`pyproject.toml`). CI runs the tests on Python 3.11, on Ubuntu and Windows.

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e . -r requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . -r requirements.txt
```

`requirements.txt` lists version ranges for `numpy`, `pandas`, `scipy`, `scikit-learn`, `streamlit`, `joblib`, `pyarrow` (the
classifier's feature files are Parquet) and `pytest`.

For the live camera, install the `camera` extra as well. It adds OpenCV and `ultralytics` (the pose model, with PyTorch):

```bash
.venv/bin/python -m pip install -e ".[camera]"                 # Windows: .\.venv\Scripts\python.exe -m pip install -e ".[camera]"
```

The pose model settings are picked automatically: `models/pose/pose_model_mac.json` on a Mac (the Apple GPU), else
`models/pose/pose_model.json` (OpenVINO on an Intel GPU; also `pip install openvino`). See `docs/pose_model.md`.

Installing the dependencies may need internet access. After installation, the default application (all pages, with
the offline rule parser) runs without an API key and does not need a network connection.

The optional Claude cross-check on the Rules page needs the extra `claude` dependency, installed into the same
virtual environment:

```bash
.venv/bin/python -m pip install -e ".[claude]"                 # Windows: .\.venv\Scripts\python.exe -m pip install -e ".[claude]"
```

It also needs two environment variables:
- `ANTHROPIC_API_KEY`;
- `LIVE_TRACKER_CLAUDE_MODEL`, the model id.

The code reads them from the process environment. It does not load a `.env` file by itself, so export them in the shell
that starts Streamlit. Each cross-check is a paid API call.

## Running the dashboard

```bash
.venv/bin/python -m streamlit run app.py          # Windows: .\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints the local URL, normally http://localhost:8501. To keep the server on this machine only, add
`--server.address 127.0.0.1`.

Always start from the repository root (the Live Agent page finds its data by relative path). If `live_tracker` is not
installed with `pip install -e .`, put `src` on the Python path first. In Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Using the application

### Guided flow (front page, `app.py`)

Upload → Rules → Replay → Real session.

- **Upload:** a camera (live, see [Quick start](#quick-start)), a recorded video with its matching keypoints, or a
  positions CSV (motion only, no behavior alerts). A camera can also be recorded and analyzed afterwards with MARS
  (`docs/camera.md`, section 8).
- **Rules:** rules are proposed, reviewed and explicitly confirmed (below).
- **Replay:** for a camera, the live picture with the skeletons the pose model finds; for a recording, its video with
  the matching keypoints drawn in sync. Replay never scores or monitors.
- **Real session:** runs the approved rules on the chosen source (below).

### Motion tool (`pages/8_Motion_Tool.py`)

Explores tracked positions and tracking quality. It makes **no behavior claims**.

**Inputs:**
- **Built-in demo:** tick **"Try synthetic demo data"** to load `example_pose_demo.csv`. It is
  made-up test data, and the page labels it as such.
- **Your own data:** a CSV with columns `x_a, y_a, x_b, y_b` (two mouse-centre positions in pixels), one row per frame.
  Missing points are left blank. No physical scale is assumed.
- **Advanced settings:** warning window quality, loss duration and optional failure injection; the engine uses the
  shared fixed settings. **"Inject fake tracking failures"** shows how the warnings react.

**Outputs:**
- **Tracking quality warnings**, e.g. tracking lost for a mouse, or a tracker that looks stuck.
- **Causal smoothing (live-safe):** One-Euro, current and past frames only.
- **"Motion features (OFFLINE ONLY)":** the original Savitzky-Golay section. It uses future frames, so it is labeled
  offline only.

### Rules (`pages/1_Rules.py`)

This page turns a sentence into a monitoring rule, and nothing is approved without you.

1. Pick **Interpretation**. The default is **"Smart reading (default, no API key, no cost)"**; the Claude cross-check
   is optional.
2. Type an instruction, for example:
   ```
   Alert me when the mice attack for at least 2 seconds
   ```
3. Click **Propose rule**. This sentence gives rule R1: attack, at least 2.0 s, alert mode `alert`. Thresholds you
   don't state use untuned defaults (alert 0.8, ask 0.5); the (?) next to the box lists every assumption.
4. Click **Confirm** or **Reject**. Only confirmed rules are approved, and the page shows whether monitoring can start.

**What the parser supports:**
- Behaviors: `attack`, `mount` and `investigation`, the classifier's three behaviors.
- Actions: alert me, ask me (each event waits for your confirm / reject), or log.
- Anything it cannot turn into a valid rule gives a clarification, never a guessed rule.

### Real Session (`pages/4_Real_Session.py`)

**With a camera:** press **Start the camera session**. A separate run program opens the camera, finds the mice in every
picture, scores them live with Model B and runs the approved rules; closing the browser does not stop it. The page shows:
- the camera as the pose model sees it, with both skeletons;
- the alert and question counts, and the latest alert;
- live charts of the last 30 s: the behavior scores (attack, investigation, mount) and the motion (speed of each mouse,
  distance between them), with a warning when tracking is lost;
- at the end, the report and a plain-language summary that can be downloaded.

**With a recording:** the same analysis runs on the saved keypoints while the video plays next to it. Questions are
answered on the Run page.

Select **Explore validation motion measurements** for the motion engine on one real CalMS21 **validation** session:
tracking quality and warnings, the causal
features, the keypoint features, and the dataset's own annotation as context (never used by a feature). It reads only
the hash-verified validation cache. Build it once from the official train file, then pick a session:

```bash
.venv/bin/python -m live_tracker.investigation_data --source data/raw/calms21/calms21_task1_train.json --all
```

Without the cache, the page says so and shows this command. Test sessions cannot be opened from it.

### Monitor (Python API)

The monitor powers the synthetic replay sessions (Investigation) and recorded-keypoint runs; saved real validation runs can also be browsed without re-running it. It can also be run from Python on synthetic scores:

```python
from live_tracker.fake_scores import demo_scores
from live_tracker.monitor import run_monitor
from live_tracker.rules import RuleSet

scores = demo_scores()                      # SYNTHETIC_demo_01, INTERFACES.md §2 layout
rules = RuleSet()
rules.add({"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0})
rules.approve("R1")                         # the researcher's confirmation
for event in run_monitor(scores, rules):    # in-memory journal unless journal_path is given
    print(event["session"], event["action"], event["start_s"], event["end_s"])
```

On this synthetic session it prints one `alert` (2.0 s to 5.0 s) and one `ask` (7.0 s to 10.0 s). These numbers come
from made-up scores and say nothing about mice.

## Working with CalMS21 (external data)

Real data is not in this repository. Data, videos and keys never go into Git; the trained models in `models/` are. Download CalMS21 yourself from the
official Caltech record (link under [Attribution](#attribution)), then:
- **Download only Task 1.** Get `task1_classic_classification.zip` (no videos, no MARS features). Unzip it and put
  `calms21_task1_train.json` (1.2 GB) and `calms21_task1_test.json` (632 MB) in `data/raw/calms21/` (git-ignored). The
  `taskprog_features_*.json` files in the same zip are not used. Building the features loads the train file once and
  needs about 4 GB of RAM.
- **Check the files first.** Verify the JSON layout before relying on it (`docs/calms21_schema.md`). This prints the
  structure of a file without assuming it:
  `.venv/bin/python -m live_tracker.calms21_inspect data/raw/calms21/<file>.json`
  (Windows: `.\.venv\Scripts\python.exe -m live_tracker.calms21_inspect data\raw\calms21\<file>.json`).
- **Keep the split.** `splits/split_manifest.json` is session-level and already committed. The official test set is
  held out: never train, tune or replay-tune on it.
- **The classifier pipeline** reads from `data/raw/calms21/` and writes to the git-ignored `data/processed/` and `outputs/`, and to
  `models/` (full usage in the `classify.py` docstring):
  macOS / Linux:
  ```bash
  .venv/bin/python -m live_tracker.classify features --train data/raw/calms21/calms21_task1_train.json \
                                                     --test  data/raw/calms21/calms21_task1_test.json
  .venv/bin/python -m live_tracker.classify train --name model_a --features basic
  .venv/bin/python -m live_tracker.classify train --name model_b --features basic,motion   # the model the demo uses
  .venv/bin/python -m live_tracker.classify scores --name model_b --split-name validation
  .venv/bin/python scripts/tune_attack_rule.py     # demo attack rule, chosen on validation only
  .venv/bin/python scripts/check_leakage.py        # test sessions never reach training, tuning or demo scores
  ```
  Windows PowerShell (the backtick continues a line):
  ```powershell
  .\.venv\Scripts\python.exe -m live_tracker.classify features `
      --train data\raw\calms21\calms21_task1_train.json `
      --test  data\raw\calms21\calms21_task1_test.json
  .\.venv\Scripts\python.exe -m live_tracker.classify train --name model_a --features basic
  .\.venv\Scripts\python.exe -m live_tracker.classify train --name model_b --features basic,motion
  .\.venv\Scripts\python.exe -m live_tracker.classify scores --name model_b --split-name validation
  .\.venv\Scripts\python.exe scripts\tune_attack_rule.py
  .\.venv\Scripts\python.exe scripts\check_leakage.py
  ```
  - Model A and model B both tune `C` on validation. Model B (basic + motion features) is the one the demo uses.
  - `--split-name test` writes test-session scores. Run it only once, with the rule frozen, under the team's test
    policy. It is never an input for tuning or for the demo.
  - The chosen demo rule is `evaluate.ATTACK_RULE_V1`. Method, grid and caveats are in `docs/attack_rule_v1.md`.

The scripts in `scripts/` (`ablation.py`, `tune_quality.py`, `tune_attack_rule.py`, `check_leakage.py`,
`compare_quality_policies.py`, `dryrun_*.py`) need the real data. `scripts/sensitivity.py` runs on the synthetic demo by default
(`.venv/bin/python scripts/sensitivity.py`; Windows: `.\.venv\Scripts\python.exe scripts\sensitivity.py`).

## Tests

```bash
.venv/bin/python -m pytest -q                     # Windows: .\.venv\Scripts\python.exe -m pytest -q
```

- About 5,000 tests. The last full run of this version (MacBook Air, Python 3.12): 5060 passed, 127 skipped, 0 failed.
- The tests use synthetic fixtures only. They need no dataset, no network and no API key; tests that need the CalMS21
  data or a camera are skipped without them.
- The Streamlit pages are tested headlessly with Streamlit's `AppTest`.

## Scientific integrity and limitations

- **Synthetic is labeled synthetic.** Demo pose data and generated scores are made up for software testing. Their
  session ids start with `SYNTHETIC_`, and the pages say so. Nothing computed on them is evidence about mouse
  behavior.
- **No performance claims in this README.** The classifier and evaluation code exist, but this README reports no
  accuracy, F1 or latency numbers. Any such result belongs with the run that produced it and the split it used.
- **Thresholds are not tuned unless stated.** Quality and feature thresholds in `FeatureParams` and the default rule
  thresholds (0.8 / 0.5) are placeholders. `src/live_tracker/params.py` records the values Ava chose on train and
  validation sessions; they are not yet wired into the defaults. The demo attack rule (`evaluate.ATTACK_RULE_V1`) was
  chosen on the 14 validation sessions only (`docs/attack_rule_v1.md`); `Rule` defaults are unchanged.
- **Not every step is causal.** The complete default pose-to-feature pipeline is **not guaranteed to be causal**.
  - **Causal:** the smoothing filters, the feature arithmetic given its input flags, and the monitor's decisions use
    only past and current frames. The tests check this.
  - **Retrospective:** tracking-quality preprocessing.
    - `quality.interpolate_short_gaps` fills a short gap from the frames on both sides.
    - `quality.flag_tracking` flags a frozen run from its first frame once the run is long enough, which is known
      only later.
  - `features.compute_features()` runs both on unflagged input. The classifier's feature pipeline uses them through
    `calms21.flag_quality()`, and the dashboard's tracking warnings are computed over the whole loaded file.
  - The Savitzky-Golay starter section is also non-causal, and is labeled offline only.
- **A score is not a behavior.** Classifier probabilities are model output, not ground truth. A threshold crossing
  marks where a score passed a threshold, not the onset of the behavior. Tracking loss ends a bout; it is never read as
  the mouse standing still.
- **The researcher decides.** No rule is active until a person confirms it, and the language model (when used) can
  only propose.
- **Held-out data stays held out.** Test sessions are never used for training or tuning.
- **Pixels, not millimetres.** No physical scale is claimed.

## Investigation, replay and session summary

Investigation offers synthetic demonstrations and saved real **validation** runs. The documented real
validation run is described in [validation_run_v1.md](docs/validation_run_v1.md); browsing checks consistency of local
records, not independent authenticity, and does not re-score data. This does not complete H5 on the held-out test set:
that work remains subject to team agreement and Ava's approved run. See [demo readiness](docs/demo_readiness.md) for
required local artifacts and an offline backup.

| Work | Status |
|---|---|
| **Investigation foundation:** verified score-drop loading with provenance, journal provenance binding, and a written 30 fps seconds-to-frame rule (`frame(t) = floor(30·t + 0.5)`) | merged, [PR #19](https://github.com/AvaSajjadi/live-tracker/pull/19) |
| **Contract v1 investigation tools:** `query_events`, `get_window_features`, `check_tracking_quality` and `compare_windows`, with versioned tracking-quality policies, session-aware counts and evidence gates | merged, [PR #20](https://github.com/AvaSajjadi/live-tracker/pull/20) |
| **Evidence-grounded claims and reports:** structured claims checked against evidence, association-only wording, and a bounded agent loop (12 tool calls, at most 2 follow-up comparisons) | merged, [PR #21](https://github.com/AvaSajjadi/live-tracker/pull/21) |
| **Offline integration and docs:** a deterministic, synthetic, offline integration run and a Contract v1 demo | merged, [PR #22](https://github.com/AvaSajjadi/live-tracker/pull/22) |
| **Replay page and session summary** for recorded sessions (initial synthetic implementation; the page is now recorded video/keypoint inspection, and saved validation runs are browsed on Investigation), with a Contract v1 evidence panel | merged, [PR #24](https://github.com/AvaSajjadi/live-tracker/pull/24) |
| **Investigation page:** the bouts, windows, tracking status and comparison outcome behind a replayed session | merged, [PR #25](https://github.com/AvaSajjadi/live-tracker/pull/25) |

Investigation results are designed to be **association-only** ("precedes / is associated with"), never causal.
They report insufficient evidence when the data cannot support a comparison. The tool contract is
`docs/investigate_tool_contracts.md`.

## Live Agent

Page `pages/5_Live_Agent.py` is included in the application. Full launch and demo steps, including Windows PowerShell: `docs/live_agent_demo.md`.

- A researcher asks a question. A language model chooses the Contract v1 tool calls; every number in the answer comes
  from those tools, not from the model.
- It works on the verified CalMS21 **validation** split only. The test split is not used.
- The saved report is reloaded and re-verified offline before anything is shown. Verification checks consistency, not
  authenticity: the files are local and unsigned.
- **Reopening saved investigations and the free rehearsal need no API key and no network.** A new paid run needs an
  OpenRouter key, one ready preflight per run, a confirmation, and typed authorization. It is guarded on the client
  side only, and only one investigation can run at a time.
- Results are **association only**. The autonomous agent's run is not necessarily equivalent to the fixed reference
  protocol (run5); the page shows that check separately.
- The progress view, evidence charts and cost ledger show what was asked, which tools ran, and what was spent.

## Lab run (overnight)

A run is set in the Rules page (**Run schedule**: for N hours, or from one time to another), then started on the
**Run** page. The work is done by a separate program, `python -m live_tracker.monitor_runner`, so closing the browser
does not stop it. It waits for the start time, reads frames, scores them live with Model B, runs the approved rules,
writes the event log, sends short messages (a log file, and a phone through ntfy if you set a topic) and writes a report
at the end. A question left open can be answered the next morning.

It works with a live camera and with recorded data. What was measured, the limits and a setup checklist:
[`docs/lab_run.md`](docs/lab_run.md) and [`docs/camera.md`](docs/camera.md).

## Project structure

```
app.py                     Streamlit front door: the menu (guided flow first)
pages/0_Upload.py          guided flow 1: choose a camera, a recording with keypoints, or a positions CSV
pages/1_Rules.py           guided flow 2: plain-English rules with researcher approval
pages/2_Replay.py          guided flow 3: live camera with skeletons, or a recording with its keypoints
pages/4_Real_Session.py    guided flow 4: the live run, alerts, live charts and report
pages/3_, 5_ to 8_*.py     Investigation, Live Agent, Run, Camera, Motion tool
src/live_tracker/          the package (camera, pose model, quality, features, classify, rules, monitor, event log, ...)
models/                    every model in the report: YOLO pose, Model A and B, ScopeScore (see models/README.md)
tests/                     pytest suite (synthetic fixtures only)
scripts/                   analysis, training and camera test scripts
colab/                     notebooks: train the pose model, run MARS on a GPU
splits/split_manifest.json session-level train / validation / test split
docs/                      camera, pose model, CalMS21 schema, investigation and run notes
INTERFACES.md              data formats between components
latest_demo/               the latest demo recording
example_pose_demo.csv      synthetic pose demo (not CalMS21)
data/                      local only (CalMS21 and videos are not in Git)
```

## Team and contributing

| Person | Area |
|---|---|
| Ava | Motion engine: tracking quality, causal features, tuning, sensitivity and ablation |
| Igor | CalMS21 data and model: loader, split, classifier, evaluation, per-frame scores |
| Syed | Agent and interface: plain-English rules, approval, monitor, event log, replay, investigation |

**How to contribute:**
- Branch from `main` and open a pull request; never push to `main`.
- Get one teammate's review, and keep the tests green.
- Keep data, videos and API keys out of Git.
- Code owned by another area is changed only after asking its owner. For example, `motion.py`, `filters.py`,
  `quality.py` and `features.py` belong to Ava.

## Origins: ScopeScore adaptations

The motion engine started as a generic 2D motion starter inspired by ScopeScore and adapted to mice. What was built
and tested (`features.py`, `quality.py`, `filters.py`):
- tracking validity flags (missing, out of frame, jump, frozen, low confidence) and tracking-loss warnings;
- short-gap interpolation (at most 3 frames), no long-gap interpolation; filled points are marked `imputed`;
- causal One-Euro and EMA smoothing, reset per session, mouse and long gap;
- per-mouse speed, acceleration, heading, turn rate, trailing-window path length, straightness and idle fraction,
  and causal pauses;
- two-mouse body-center distance and its rate (approach / separation), relative speed, time spent close and
  speed correlation;
- keypoint features (nose-to-nose, nose-to-tail-base, facing angle), opt-in via `keypoint_features()`; they are
  not in the trained models;
- causality tests (changing later frames leaves earlier values unchanged), and a sensitivity script for feature
  stability.

Measured on the 14 validation sessions (`docs/motion_engine_results.md`): adding these features to plain pose
geometry improved investigation and attack detection and did not improve mount. The attack result rests on four
validation sessions, and validation shares animals with train, so the numbers are optimistic.

Deliberately not transferred or not done:
- ScopeScore's surgical thresholds, and any clinical meaning; attack is not claimed from speed alone;
- Savitzky-Golay smoothing uses future frames, so it stays in the Motion tool, labeled offline only;
  the starter's `movement_bouts` finds whole runs after the fact and no live code calls it (the causal `in_pause`
  feature replaces it);
- SPARC / log dimensionless jerk: never built;
- the tuned quality values in `params.py` are not wired into the defaults, and the models were not retrained with them.

## Attribution

CalMS21: Sun, J. J., et al. "The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions." NeurIPS Datasets and
Benchmarks Track, 2021. arXiv:2104.02710. Dataset record: https://data.caltech.edu/records/s0vdx-0k302 (CC BY 4.0).
