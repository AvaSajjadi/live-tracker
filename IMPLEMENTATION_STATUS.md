# Implementation status

Last updated: 2026-10-14

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

## Still missing / blockers

- CalMS21 not downloaded or inspected: no loader, keypoint mapping, labels, or fps check.
- All thresholds are placeholders, not tuned on training data.
- Dyadic features beyond distance (nose-to-nose, facing angle) need real keypoints.
- No classifier, agent, replay, or evaluation yet.

## Next

1. Run `python -m live_tracker.calms21_inspect <task1 json>` and read the official `readme.md`.
2. Write the CalMS21 loader and a session-level train/val/test split with a fixed seed.
3. Baseline classifier (attack / mount / investigation) on train sessions only.
