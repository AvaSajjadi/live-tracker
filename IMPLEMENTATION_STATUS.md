# Implementation status

Last updated: 2026-10-22

## Stage 0: starter audit

Starter files: `app.py` (Streamlit, CSV upload), `motion.py` (short-gap interpolation,
Savitzky-Golay smoothing, speed/acceleration, mouse distance, movement bouts),
`events.py` (duration-gated threshold events), `tests/test_motion.py` (4 tests),
`example_pose_demo.csv` (synthetic, 300 frames, no missing values).

- Starter tests: 4 passed.
- Starter smoothing is offline (uses future frames), so it can't be used for live replay.
- No CalMS21 loader, classifier, agent, or replay yet.

## Stage 1: added in this branch

| File | What it does | Tests |
|---|---|---|
| `src/live_tracker/filters.py` | Causal smoothing (EMA, One-Euro). Resets after long gaps and per session/mouse. Missing stays missing. | `test_filters.py` (8) |
| `src/live_tracker/quality.py` | Flags missing / out-of-frame / jump / frozen points; fills only short interior gaps and marks them `imputed`; per-window quality + warning; tracking-loss events | `test_quality.py` (13) |
| `src/live_tracker/formats.py` | Wide starter CSV to long table; fake failure injector for testing | `test_formats.py` (4) |
| `src/live_tracker/rules.py` | Rule format, strict validation, approval gate (Stage 4 groundwork) | `test_rules.py` (16) |
| `src/live_tracker/calms21_inspect.py` | Prints the structure of a JSON file without assuming its layout. Loader is a placeholder on purpose. | `test_calms21_inspect.py` (5) |
| `app.py` | Added synthetic-demo option (labeled), tracking-quality section with warnings, causal smoothing chart, failure injection. Starter section kept. | `test_app.py` (3) |

Command: `python -m pytest -q` → **53 passed** (Linux, Python 3.13, pandas 2.3.3, numpy 2.5.4, streamlit 1.66).
Also checked by running the dashboard and looking at it: injected failures show
"Tracking lost for mouse b: 5.00 s to 6.50 s" and "stuck for mouse a: 2.50 s to 3.50 s".
Also passes on Windows: 53 passed locally (Python 3.12.10) and in GitHub Actions (windows-latest, ubuntu-latest).

## Stage 2: motion engine features (Ava, 2026-10-17)

| File | What it does | Tests |
|---|---|---|
| `src/live_tracker/features.py` | Causal per-mouse features (speed, accel, heading, turn rate, trailing-window path length / straightness / idle fraction, causal pauses); `body_center()` from explicitly named torso keypoints; dyadic features (center distance and rate, relative speed, close time, speed correlation); `frame_tracking_ok()`; `tracking_warning_events()` in the event-log format; `compute_features()` for the whole table | `test_features.py` (17), `test_features_pair.py` (17) |
| `INTERFACES.md` §5 | Feature table columns, units, causality rules, usage example | |

Missing or invalid tracking gives NaN, never 0. Short-gap fills are marked `uses_imputed`.
Causality is tested: changing frames after t leaves features at or before t unchanged.
All thresholds in `FeatureParams` are placeholders, not tuned on any data.
Only tested on synthetic data (a fixture with made-up `SYN_*` keypoint names).

Merged to `main` (PRs #4, #5): `python -m pytest -q` -> **87 passed** (Windows, Python 3.12).

Open PRs (not on `main` yet):

| PR | What | Tests |
|---|---|---|
| #6 | `scripts/sensitivity.py` (feature stability under jitter, gaps, smoothing, thresholds +-20%); starter section of `app.py` labeled OFFLINE ONLY; test that no live code uses Savitzky-Golay | `test_sensitivity.py` (10), `test_app.py` (+1) |
| #7 | `scripts/ablation.py` (basic vs basic + ScopeScore features, same split and seed, test set evaluated once). Uses toy fit/evaluate on synthetic data until Igor's `classify.py` / `evaluate.py` exist | `test_ablation.py` (9) |

With #6 and #7 both merged: 107 tests (measured with both applied on one tree).
The sensitivity script has only been run on the synthetic demo; those numbers show it
works, not anything about mice. The ablation has no real results.

Commands:

```
python -m pytest -q
python scripts/sensitivity.py            # synthetic demo -> outputs/sensitivity.csv
python scripts/ablation.py --help        # needs Igor's fit / evaluate functions
```

## Stage 3: motion engine on real data (Ava, 2026-10-22, branch `ava/finish-motion-engine`)

Measured on the CalMS21 train / validation sessions only. The test split was not read and `--final` of the
ablation was not run. Full numbers, caveats and commands: `docs/motion_engine_results.md`.

| What | Result |
|---|---|
| Ablation, basic vs basic + ScopeScore features (14 validation sessions) | macro F1 0.499 -> 0.651 (paired +0.060, 95 % interval +0.040 to +0.082); better in 14 of 14 sessions. Helps investigation and attack; does **not** help mount (0.620 -> 0.605). Attack rests on 4 sessions; validation shares animals with train, so numbers are optimistic. |
| Sensitivity on 4 real validation sessions | Position features are stable; acceleration is the fragile one (5 px jitter: 0.57 of its spread; no smoother: 2.8). Threshold changes show up in the tails, not the median. |
| `features.keypoint_features()` | nose-to-nose, nose-to-tail, facing angle; causal, opt-in, not in the section 5 table or the models; 15 tests. Not yet tested as classifier inputs. |
| `pages/4_Real_Session.py` + `session_view.py` | one real validation session: tracking quality, features, keypoint features, annotation as context. Reads only the hash-verified validation cache. 11 tests; run headless on real data. |
| `scripts/ablation.py` | reads Parquet as well as CSV. |
| `scripts/sensitivity.py` | `--calms21` runs on real train/validation sessions and refuses any session outside the manifest's train/validation lists. |

Full suite on this branch (Windows, Python 3.12.10, Streamlit 1.66.1, pyarrow installed):
`python -m pytest -q` -> **3419 passed, 46 skipped, 11 xfailed, 0 failed** (37 min).

Commands:

```
python scripts/ablation.py --basic data/processed/basic.parquet --scope data/processed/motion.parquet     --labels data/processed/labels.parquet --manifest splits/split_manifest.json     --fit live_tracker.classify:fit_scores --evaluate live_tracker.evaluate:summary_metrics
python scripts/sensitivity.py --calms21 data/raw/calms21/calms21_task1_train.json --sessions <validation ids>
python -m live_tracker.investigation_data --source data/raw/calms21/calms21_task1_train.json --all
python -m streamlit run app.py
```

## Stage 4: integration with Syed's platform and UI (2026-10-24, branch `ava/pr31-windows-compat`, PR #35)

Merged into this branch: Syed's cross-platform run locking and safe Windows run loading (#31), and his UI stack
(#32 verified overview and timeline, #33 evidence visualizations, #34 live agent progress). Docs: README Live Agent
section and Windows steps in `docs/live_agent_demo.md`.

Full suite (Windows, Python 3.12, `PYTHONPATH=src python -m pytest -q`) on the merged code:
**4450 passed, 129 skipped, 11 xfailed, 0 failed** (39 min). GitHub Actions on PR #35: see the PR.

Command: `PYTHONPATH=src python -m pytest -q`. Without `PYTHONPATH=src` (and without `pip install -e .`) the tests
fail with "No module named live_tracker".

Still open: test-split evaluation (not run; needs `calms21_task1_test.json` and the team's go), demo rehearsal, backup
video, slides and poster for the submission.

## Stage 5: lab run program (Ava, 2026-10-24, branch `ava/monitor-runner`)

| File | What it does | Tests |
|---|---|---|
| `src/live_tracker/run_schedule.py`, `schedule_workflow.py`, `schedule_ui.py` | Run schedule: for N hours or from A to B, approval, clock changes, completeness; the form on the Rules page | `test_run_schedule.py` (39), `test_schedule_workflow.py` (12), `test_schedule_page.py` (8) |
| `src/live_tracker/live_scoring.py` | Pose frames to Model B scores from a short moving buffer | `test_live_scoring.py` (9, the last one opt-in on real data) |
| `src/live_tracker/run_sources.py` | Score file, recorded keypoints, camera reader (fake-camera tests only), pose-model slot | `test_run_sources.py` (17) |
| `src/live_tracker/notify.py` | Log file and optional ntfy phone messages | `test_notify.py` (13) |
| `src/live_tracker/monitor_runner.py` | The program: wait, run, log, answers, interruptions, report, stop | `test_monitor_runner.py` (21) |
| `src/live_tracker/run_control.py`, `run_ui.py`, `pages/6_Run.py` | Start, stop, watch, answer (also after the run), report | `test_run_control.py` (22), `test_run_page.py` (12) |
| `src/live_tracker/demo.py`, `demo_ui.py`, `pages/7_Demo.py`, `scripts/prepare_demo_session.py` | Live demo page: a recorded video and its run start together; GO signal, delay meter; see `docs/demo_recording.md` | `test_demo.py` (12), `test_demo_page.py` (11) |

Measured on one validation session: live scores equal whole-session scores (largest difference 5e-13); the program,
run as its own process, produced the same 46 events as the whole-session calculation. Details and limits:
`docs/lab_run.md`. **No camera and no mouse pose model exist yet; nothing here was tested on a real camera.**

## Still missing / open

- Test-split evaluation: not run. It happens once, at the end, under a policy the team agrees first.
- Whether the keypoint features help a classifier: untested (needs regenerating features and retraining).
- Tuned quality values in `params.py` (jump 3000 px/s, One-Euro beta 0.05) are not wired into the defaults, and
  the models were not retrained with them. Stage 9 investigation runs with the jump flag off.
- `pyarrow` is needed by `classify.py` and the ablation but is not declared in `pyproject.toml` or
  `requirements.txt`.
- Slides / report, and the demo rehearsal.
- Contract v1 decisions still to confirm with Syed: see `plan/AVA.md` and the open items in
  `docs/contract_v1_integration.md` section 6.
- The motion-engine models were refit here from Igor's Oct 19 feature files; his saved model artifacts are not in
  the repository, so Model B's exact settings and validation results still need to be committed or written down.
