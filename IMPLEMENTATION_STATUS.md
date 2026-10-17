# Implementation status

Last updated: 2026-10-17

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
Not yet run on Windows (the GitHub Actions workflow tests Windows and Ubuntu).

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

## Still missing / blockers

- CalMS21 not downloaded or inspected: no loader, keypoint mapping, labels, or fps check.
  Waiting on Igor (schema notes, then loader and split manifest).
- Body center is implemented but untested on real data: the torso keypoint names must come
  from the CalMS21 readme, not from memory.
- Not built yet (need the real keypoints): nose-to-nose, nose-to-tail, facing angle.
- All thresholds are placeholders. Tuning on train/validation sessions needs the split manifest.
- Ablation and sensitivity have no real results. The expected split manifest layout
  (`train` / `validation` / `test` lists of session IDs) is an assumption.
- No classifier, agent, replay, or evaluation yet.

## Next

1. Igor: schema notes, loader, split manifest.
2. Ava: nose / tail / facing features, tune parameters on train/val, sensitivity on real
   sessions, motion-engine page with a real session, ablation once `evaluate()` exists.
3. Baseline classifier (attack / mount / investigation) on train sessions only.
