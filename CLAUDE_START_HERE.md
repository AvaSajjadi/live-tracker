# Live Tracker - Claude Code implementation handoff

## Your role and immediate objective

You are the lead Python/ML engineer for **Live Tracker**, a hackathon prototype for monitoring mouse social behavior from CalMS21 pose trajectories. You are working inside the existing `live_tracker_starter` repository in VS Code. **Begin implementation now, not just planning.** Read the repository first, run the tests, then implement the highest-priority verified next milestone. Work in small, tested increments and report what actually works.

This project adapts general **ScopeScore-inspired motion-analysis principles** (trajectory quality, smoothing, speed, movement segmentation, pauses, robustness checks) to mouse pose data. Do **not** copy surgical instrument models, thresholds, patient data, proprietary research code, or clinical interpretations. This project uses public CalMS21 data, not hospital data.

## Product objective

A researcher enters a natural-language instruction such as: "Alert me if the mice attack for at least 2 seconds." The system translates it into a structured rule, obtains researcher approval, monitors a recorded CalMS21 session replayed at real-time speed, detects qualifying events, produces timestamped alerts with evidence and uncertainty, asks for confirmation on uncertain events, warns on tracking failure, and writes a summary strictly from the event log.

The hackathon proposal specifies an end-to-end working prototype by **October 20, 2026**, and submission by **October 25, 2026**. Prioritize a real, reproducible minimum viable prototype over advanced features.

## Nonnegotiable scientific and engineering rules

1. Read `README.md`, `app.py`, `pyproject.toml`, `requirements.txt`, `src/live_tracker/*.py`, and tests before making changes.
2. Do not claim CalMS21 integration, classifier accuracy, real-time behavior detection, agent autonomy, or video alert clips until they are implemented and verified.
3. `example_pose_demo.csv` is **synthetic test data**, not CalMS21. Label it clearly in the UI and outputs.
4. Verify the **actual CalMS21 Task 1 dataset schema and landmark ordering** from the official files/readme before implementing an adapter. Do not guess array dimensions, landmark ordering, behavior label mappings, or license terms. If the dataset is not available locally, implement a schema-inspection utility and a loader interface, provide exact download/setup instructions, and stop short of claiming integration.
5. Split CalMS21 by **recording/session**, not randomly by frame. Keep test labels out of model training, threshold tuning, and agent inputs. Establish train/validation/test splits before fitting or tuning.
6. Distinguish *behavior classification* from *motion feature measurement*. Speed or proximity alone cannot prove an attack. Build a classifier and compare against expert-labeled bouts.
7. Use mouse-specific parameters selected on training/validation data. Never transfer ScopeScore surgical thresholds (e.g. 5 mm, 0.5 s) as defaults justified for mouse behavior.
8. Ensure streaming mode uses **causal** filters/features only. Existing Savitzky-Golay smoothing is offline/noncausal; do not use it to claim live monitoring without replacing it or clearly isolating offline analysis.
9. Track missing, frozen, and implausible coordinates; do not mislabel missing tracking as immobility. Interpolate only short gaps and mark imputed data.
10. Log each alert with session ID, time, event type, rule ID, classifier confidence or calibrated score, and supporting evidence. Do not invent events in the LLM-generated summary.
11. Keep the researcher in control: confirm rules before monitoring; ambiguous events may request confirmation; never actuate or intervene in experiments.
12. Prefer deterministic Python functions for numerical calculations, rule enforcement, event aggregation, evaluation, and summaries. The LLM handles language interpretation and explanations, not the source of numeric truth.
13. Keep secrets out of Git; use environment variables and `.env.example` for API keys. Do not require an API key to run the motion engine or offline tests.
14. Do not install dependencies, download large datasets, or make paid API calls without explaining what will happen and obtaining user approval if needed. Do not fabricate successful tests.
15. Preserve existing functioning code; add focused tests for every new feature. If tests fail, diagnose and fix them before moving on.

## Existing starter (inspect; do not assume functionality beyond these files)

- `app.py`: Streamlit dashboard for uploaded CSV trajectories.
- `src/live_tracker/motion.py`: generic 2D trajectory processing and motion features.
- `src/live_tracker/events.py`: simple duration-gated event extraction.
- `tests/test_motion.py`: starter automated tests.
- `example_pose_demo.csv`: synthetic demonstration data.
- `data/raw/`, `models/`: placeholders.

Run on Windows PowerShell from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If the environment already exists, reuse it. If commands fail, report the actual error and fix it; do not report success prematurely.

## Motion-analysis engine: precise adaptation plan

Use pose coordinates p_i(t) for mouse i at timestamp t, with 30 Hz as a **proposal-reported** nominal frame rate; verify dataset metadata and timestamps. Units initially pixels and seconds, not millimeters.

A. **Pose and trajectory validation**: inspect frame index, session ID, coordinate system, keypoint confidence if present, NaNs, out-of-frame values, frozen tracks, abrupt jumps, and animal identity continuity. Mark validity; avoid silently repairing long gaps.

B. **Representative mouse position**: derive a robust body-center estimate from suitable torso landmarks only after confirming CalMS21's actual landmark schema. Preserve individual keypoints for relational features. Document the center definition.

C. **Causal smoothing**: add a configurable One-Euro or exponential low-pass filter, reset across long missing intervals and session boundaries. Compare with raw trajectories. Avoid leaking future frames into live features.

D. **Individual movement**: calculate displacement, speed in pixels/s, acceleration in pixels/s^2, heading, angular velocity, path length, straightness, idle fraction, and contiguous movement/pause bouts. Handle irregular timestamps, missing samples, and zero-duration windows. Use configurable thresholds chosen on training/validation data.

E. **Dyadic movement**: calculate mouse-to-mouse body-center distance, distance derivative (approach/separation rate), relative velocity, nose-to-nose distance, nose-to-body/tail distance where supported, body orientation, facing/approach indicators, proximity duration, and relative motion synchrony. Do not interpret these as attacks without labels.

F. **Optional advanced features**: SPARC and log dimensionless jerk only after the MVP works, with documented definitions, units/normalization, valid-window requirements, and a held-out ablation showing whether they help.

G. **Tracking quality**: provide a per-window quality score or explicit flags and a warning for tracking loss. Test with missing points, frozen coordinates, noise, and identity swaps where feasible.

H. **Scientific validation**: measure sensitivity to small coordinate jitter, short gaps, smoothing settings, and threshold changes. Separate motion-feature stability from behavior-classification accuracy.

## Build stages and acceptance criteria

### Stage 0: Repository audit and baseline
- Inspect files, describe current implemented vs missing features.
- Run tests and launch app if possible.
- Create a brief `IMPLEMENTATION_STATUS.md` showing commands/results and blockers.
- Fix obvious setup/runtime bugs without rewriting working modules.

### Stage 1: Robust motion features (first coding priority)
- Add unit-tested causal filtering and tracking validity.
- Add windowed individual and dyadic features to `src/live_tracker/`.
- Retain backwards compatibility for the starter CSV demo.
- Show new features and quality warnings in Streamlit.
- Unit tests must cover constant position, linear motion, approaching/separating mice, missing points, short gaps, frozen tracks, and session boundaries.

### Stage 2: Real CalMS21 adapter
- Inspect official downloaded dataset metadata/readme and one real session.
- Implement schema-verified data loader, keypoint mapping, labels, and timestamps.
- Write a script to display the real session as two landmark skeletons; video is optional.
- Create a recording-level split manifest with fixed random seed.
- Do not download the 28+ GB video archive unless needed and explicitly approved; pose/label files suffice for initial classification.

### Stage 3: Behavior classifier
- Train a transparent baseline (e.g. class-weighted logistic regression or random forest) using train sessions, with validation-selected hyperparameters/thresholds.
- Train a second model with the ScopeScore-inspired motion feature additions.
- Evaluate on held-out sessions: per-class precision/recall/F1, confusion matrix, event-level attack precision/recall, onset-to-alert latency, and false alerts per hour.
- Convert frame predictions into bouts using explicit duration/merge rules. No leakage across session boundaries.
- Save reproducible model artifacts, split manifest, parameters, and evaluation outputs.

### Stage 4: Rules and agent
- First implement deterministic JSON rule schema and validator (behavior, minimum duration, confidence threshold, alert mode).
- Build a user-confirmation step; unapproved rules cannot activate.
- Add optional Claude API tool-use to parse natural language into proposed rules and explain logged events. Require explicit API configuration; offer a deterministic no-key fallback for the demo.
- Never let the model invent classifier outputs, counts, durations, or evidence.

### Stage 5: Replay, alerting, and dashboard
- Replay one held-out session at nominal recorded speed (clearly label as replay, not a live camera).
- Show mouse poses/video if available, current classification, active rules, alerts, and tracking warnings.
- Attach timestamps and an optional clip or highlighted pose replay for events.
- Build an end-of-session summary from the structured event log only.
- Add adaptive sampling only after a fixed-rate baseline, ensuring it does not skip events and measuring compute/latency tradeoffs.

### Stage 6: End-to-end reproducibility
- One command starts the dashboard; one command runs all tests.
- README documents setup, CalMS21 attribution, actual dataset placement, reproducible training/evaluation, API-key configuration, and limitations.
- Produce a test report with expected/actual outcomes and measured results (not invented numbers).
- Keep a working demo path that does not depend on live network access.

## How to work with me

- **Start immediately with Stage 0 and Stage 1.** Do not respond only with a plan.
- Before each substantive change, say which files you will edit and why.
- Make small changes, run tests, and give a compact progress summary.
- Ask me a question only if an essential decision, missing dataset, permission, or credential blocks progress.
- Never assert the full hackathon prototype is complete until its acceptance tests pass.
- At the end of the first session report: files changed, commands run, tests passed/failed, what is verified, remaining blockers, and the next exact command I should run in VS Code.

## First action now

Inspect the existing repository and run its tests. Then implement causal trajectory filtering and robust tracking-quality flags, add focused unit tests, and update the Streamlit dashboard to show tracking warnings. If real CalMS21 data is absent, keep the synthetic CSV demonstration working and clearly label it synthetic. Do not wait for another planning prompt.
