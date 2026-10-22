# Live Tracker

**Turning mouse pose data into researcher-approved behavior alerts, with every step kept honest about what the data
can and cannot show.**

Behavioral neuroscience labs record hours of video of mice interacting. Finding the moments that matter (an attack, a
mount, a close investigation) still mostly means watching footage or labeling it frame by frame. Live Tracker is a
hackathon prototype that works on **already-tracked pose data** (2D body keypoints per frame). It:
- checks how trustworthy the tracking is;
- computes motion features with causal filters and formulas (the tracking-quality preprocessing in front of them is
  not causal; see [limitations](#scientific-integrity-and-limitations));
- trains a per-frame behavior classifier on the public CalMS21 dataset;
- lets a researcher describe, in plain English, what they want to be alerted to.

A proposed rule does nothing until the researcher confirms it. A deterministic monitor, with no language model in the
loop, then turns per-frame scores into timestamped, journaled events.

> **Status:** active hackathon project (October 2026). This README describes what is on `main`. Work that is still in
> open pull requests is listed separately under [In progress](#in-progress-not-on-main-yet). Nothing here is a
> validated scientific instrument; see [Scientific integrity and limitations](#scientific-integrity-and-limitations).

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
| **Streamlit dashboard** | Two pages: the motion engine (tracking quality and smoothing) and the Rules page (propose → Confirm / Reject) | `app.py`, `pages/1_Rules.py` |
| **Synthetic test data** | A labeled synthetic pose CSV and a synthetic per-frame score generator. Every synthetic session id starts with `SYNTHETIC_` | `example_pose_demo.csv`, `src/live_tracker/fake_scores.py` |

## How the parts fit together

```
 CalMS21 Task 1 JSON (pose + labels)
        │
        ▼
 calms21.py ── long pose table
        │
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
 event_log.py ── journal, §4 event-log export
        │
        ▼
 in progress: replay page, session summary, evidence-grounded investigation
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

`requirements.txt` lists version ranges for `numpy`, `pandas`, `scipy`, `scikit-learn`, `streamlit`, `joblib` and `pytest`.

Installing the dependencies may need internet access. After installation, the default application (both pages, with
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

## Using the application

### Motion engine (main page, `app.py`)

Explores tracked positions and tracking quality. It makes **no behavior claims**.

**Inputs:**
- **Built-in demo:** tick **"Use built-in synthetic demo data (not CalMS21)"** to load `example_pose_demo.csv`. It is
  made-up test data, and the page labels it as such.
- **Your own data:** a CSV with columns `x_a, y_a, x_b, y_b` (two mouse-centre positions in pixels), one row per frame.
  Missing points are left blank. No physical scale is assumed.
- **Sidebar:** quality thresholds and the causal smoothing settings. These are placeholder values, and the page says
  so. **"Inject fake tracking failures"** shows how the warnings react.

**Outputs:**
- **Tracking quality warnings**, e.g. tracking lost for a mouse, or a tracker that looks stuck.
- **Causal smoothing (live-safe):** One-Euro, current and past frames only.
- **"Motion features (OFFLINE ONLY)":** the original Savitzky-Golay section. It uses future frames, so it is labeled
  offline only.

### Rules (`pages/1_Rules.py`)

This page turns a sentence into a monitoring rule, and nothing is approved without you.

1. Pick **Interpretation**. The default is **"Offline parser (default, no API key, no cost)"**; the Claude cross-check
   is optional.
2. Type an instruction, for example:
   ```
   Alert me when the mice attack for at least 2 seconds
   ```
3. Click **Propose rule**. This sentence gives rule R1: attack, at least 2.0 s, alert mode `alert`. Thresholds you
   don't state use untuned defaults (alert 0.8, ask 0.5), and the page warns about them.
4. Click **Confirm** or **Reject**. Only confirmed rules are approved, and the page shows whether monitoring can start.

**What the parser supports:**
- Behaviors: `attack`, `mount` and `investigation`, the classifier's three behaviors.
- Actions: alert me, ask me (each event waits for your confirm / reject), or log.
- Anything it cannot turn into a valid rule gives a clarification, never a guessed rule.

### Monitor (Python API)

On `main` the monitor has no page yet; the replay page is [in progress](#in-progress-not-on-main-yet). It can be run
from Python on synthetic scores:

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

Real data is not in this repository. Data, videos and keys never go into Git. Download CalMS21 yourself from the
official Caltech record (link under [Attribution](#attribution)), then:
- **Check the files first.** Verify the JSON layout before relying on it (`docs/calms21_schema.md`). This prints the
  structure of a file without assuming it:
  `.venv/bin/python -m live_tracker.calms21_inspect data/raw/calms21/<file>.json`
  (Windows: `.\.venv\Scripts\python.exe -m live_tracker.calms21_inspect data\raw\calms21\<file>.json`).
- **Keep the split.** `splits/split_manifest.json` is session-level and already committed. The official test set is
  held out: never train, tune or replay-tune on it.
- **The classifier pipeline** reads from `data/raw/calms21/` and writes to the git-ignored `data/processed/`,
  `models/` and `outputs/` (full usage in the `classify.py` docstring):
  macOS / Linux:
  ```bash
  .venv/bin/python -m live_tracker.classify features --train data/raw/calms21/calms21_task1_train.json \
                                                     --test  data/raw/calms21/calms21_task1_test.json
  .venv/bin/python -m live_tracker.classify train --name model_a --features basic   # model B: --features basic,motion
  .venv/bin/python -m live_tracker.classify scores --name model_a --split-name validation
  ```
  Windows PowerShell (the backtick continues a line):
  ```powershell
  .\.venv\Scripts\python.exe -m live_tracker.classify features `
      --train data\raw\calms21\calms21_task1_train.json `
      --test  data\raw\calms21\calms21_task1_test.json
  .\.venv\Scripts\python.exe -m live_tracker.classify train --name model_a --features basic
  .\.venv\Scripts\python.exe -m live_tracker.classify scores --name model_a --split-name validation
  ```

The scripts in `scripts/` (`ablation.py`, `tune_quality.py`, `compare_quality_policies.py`, `dryrun_*.py`) need the
real data. `scripts/sensitivity.py` runs on the synthetic demo by default
(`.venv/bin/python scripts/sensitivity.py`; Windows: `.\.venv\Scripts\python.exe scripts\sensitivity.py`).

## Tests

```bash
.venv/bin/python -m pytest -q                     # Windows: .\.venv\Scripts\python.exe -m pytest -q
```

- Every pull request and every push to `main` runs the full suite in GitHub Actions on `ubuntu-latest` and
  `windows-latest` (`.github/workflows/tests.yml`).
- The tests use synthetic fixtures only. They need no dataset, no network and no API key.
- The Streamlit pages are tested headlessly with Streamlit's `AppTest`.

## Scientific integrity and limitations

- **Synthetic is labeled synthetic.** Demo pose data and generated scores are made up for software testing. Their
  session ids start with `SYNTHETIC_`, and the pages say so. Nothing computed on them is evidence about mouse
  behavior.
- **No performance claims in this README.** The classifier and evaluation code exist, but this README reports no
  accuracy, F1 or latency numbers. Any such result belongs with the run that produced it and the split it used.
- **Thresholds are not tuned unless stated.** Quality and feature thresholds in `FeatureParams` and the default rule
  thresholds (0.8 / 0.5) are placeholders. `src/live_tracker/params.py` records the values Ava chose on train and
  validation sessions; they are not yet wired into the defaults.
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

## In progress (not on `main` yet)

These are open pull requests and branches. They are not part of `main` until they merge, and their scope may change in
review.

| Work | Status |
|---|---|
| **Investigation foundation:** verified score-drop loading with provenance, journal provenance binding, and a written 30 fps seconds-to-frame rule (`frame(t) = floor(30·t + 0.5)`) | [PR #19](https://github.com/AvaSajjadi/live-tracker/pull/19) |
| **Contract v1 investigation tools:** `query_events`, `get_window_features`, `check_tracking_quality` and `compare_windows`, with versioned tracking-quality policies, session-aware counts and evidence gates | [PR #20](https://github.com/AvaSajjadi/live-tracker/pull/20) (stacked on #19) |
| **Evidence-grounded claims and reports:** structured claims checked against evidence, association-only wording, and a bounded agent loop (12 tool calls, at most 2 follow-up comparisons) | [PR #21](https://github.com/AvaSajjadi/live-tracker/pull/21) (stacked on #20) |
| **Offline integration and docs:** a deterministic, synthetic, offline integration run and a Contract v1 demo | [PR #22](https://github.com/AvaSajjadi/live-tracker/pull/22) (stacked on #21) |
| **Replay page and session summary** for recorded sessions (synthetic sessions only so far; not a live camera) | branch `syed/day34-replay` |

Investigation results are designed to be **association-only** ("precedes / is associated with"), never causal.
They report insufficient evidence when the data cannot support a comparison. The tool contract is
`docs/investigate_tool_contracts.md`.

## Project structure

```
app.py                     Streamlit main page: motion engine and tracking quality
pages/1_Rules.py           Streamlit page: plain-English rules with researcher approval
src/live_tracker/          the package (quality, filters, features, calms21, classify, evaluate,
                           agent, claude_agent, rules, rule_workflow, monitor, event_log, ...)
tests/                     pytest suite (synthetic fixtures only)
scripts/                   analysis scripts (sensitivity, ablation, quality tuning, dry runs)
splits/split_manifest.json session-level train / validation / test split
docs/                      CalMS21 schema notes, investigation tool contract
INTERFACES.md              data formats between components
IMPLEMENTATION_STATUS.md   progress log
plan/                      team plan and per-person task lists
example_pose_demo.csv      synthetic pose demo (not CalMS21)
data/, models/             local only, git-ignored contents
```

## Team and contributing

| Person | Area |
|---|---|
| Ava | Motion engine: tracking quality, causal features, tuning, sensitivity and ablation |
| Igor | CalMS21 data and model: loader, split, classifier, evaluation, per-frame scores |
| Syed | Agent and interface: plain-English rules, approval, monitor, event log, replay, investigation |

The plan and the per-person task lists are in `plan/`.

**How to contribute** (details in `CONTRIBUTING.md`):
- Branch from `main` and open a pull request; never push to `main`.
- Get one teammate's review, and keep the tests green.
- Keep data, videos and API keys out of Git.
- Code owned by another area is changed only after asking its owner. For example, `motion.py`, `filters.py`,
  `quality.py` and `features.py` belong to Ava.

## Origins: ScopeScore adaptations

The motion engine started as a generic 2D motion starter inspired by ScopeScore and adapted to mice:
- short-gap interpolation, with no long-gap interpolation;
- segment-aware smoothing;
- frame-to-frame velocity and acceleration;
- relative mouse distance and approach / separation speed;
- movement bout detection with configurable thresholds;
- tracking validity flags;
- deterministic, duration-gated alert extraction.

ScopeScore's surgical thresholds are not transferred, and attack is not claimed from speed alone.

## Attribution

CalMS21: Sun, J. J., et al. "The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions." NeurIPS Datasets and
Benchmarks Track, 2021. arXiv:2104.02710. Dataset record: https://data.caltech.edu/records/s0vdx-0k302 (CC BY 4.0).
