# Ava: ScopeScore-inspired motion engine

You own everything ScopeScore-related: causal motion features, tracking quality,
parameter tuning, sensitivity tests, and the "basic vs ScopeScore features" comparison.
Overview and handoff table: `plan/00_TEAM_PLAN.md`.

**You deliver:** H2 (feature table → Igor), H2b (tracking warnings → Syed)
**You wait for:** H1a / H1 (CalMS21 schema + loader ← Igor), H4 (`evaluate()` + model A results ← Igor)

Your files: `src/live_tracker/motion.py`, `filters.py`, `quality.py`, `features.py`,
`app.py` (motion-engine page), `scripts/sensitivity.py`, `scripts/ablation.py`, and their tests.

---

## Day 1: Sat Oct 17 (no dependencies)

### 1. Team setup (H0, with everyone, 1 h)
- [ ] Clone, venv, `python -m pytest -q` passes (53 tests expected)
- [ ] Review and merge PR #2 (`ignore-egg-info`)
- [ ] Agree on the new `INTERFACES.md` sections with Igor and Syed

### 2. Write the "Feature table" interface in `INTERFACES.md`
One row per `session × frame`. Columns (pixels, seconds):
`session, frame, time_s, tracking_ok`, then per mouse (`_a`, `_b`, or the real mouse IDs from H1a):
`speed, accel, heading, turn_rate, path_len_w, straightness_w, idle_frac_w, in_pause`,
then dyadic: `dist_center, dist_rate, rel_speed, dist_nose_nose, dist_nose_tail_ab, dist_nose_tail_ba, facing_ab, facing_ba, close_time_s, speed_corr_w`.
`_w` = computed over a **trailing** window (past frames only). Write down the window length as a parameter.
Igor builds his classifier against this, so publish it early (by noon).

### 3. `src/live_tracker/features.py`: causal individual features
Input: long table (`session, mouse, keypoint, frame, time_s, x, y` + quality flags).
For now use one center point per mouse (synthetic data via `formats.wide_to_long`).
- [ ] Smooth first with `filters.smooth_dataframe(method="one_euro")` (causal)
- [ ] Displacement, speed (px/s), acceleration (px/s²): use real `dt` (irregular timestamps), never across a missing gap or a session boundary
- [ ] Heading (rad) and turn rate (rad/s), with angles unwrapped; NaN when the mouse barely moves (heading is noise then)
- [ ] Trailing-window path length, straightness (net displacement / path length), idle fraction
- [ ] Pause / movement bouts: reuse the logic of `motion.movement_bouts`, but causal (a pause is known only once it has lasted `min_duration_s`)
- [ ] All thresholds are function arguments with "placeholder, tune on train/val" in the docstring

### 4. Tests (`tests/test_features.py`)
- [ ] Constant position → speed 0, straightness NaN or 1 by your definition (document it)
- [ ] Straight line at known speed → correct speed, straightness ≈ 1, turn rate ≈ 0
- [ ] Missing points → NaN features, not 0 (missing ≠ still)
- [ ] Short gap filled by `interpolate_short_gaps` → features marked as coming from imputed data
- [ ] Two sessions back to back → nothing carries over from session 1 to session 2
- [ ] **Causality test:** change frames after t; features at or before t must not change

**End of Day 1:** PR "Causal individual features" opened and reviewed.

---

## Day 2: Sun Oct 18

### ⏳ Wait for H1a (schema notes) and H1 (loader) from Igor
**While you wait** (morning): read the CalMS21 readme together with Igor, help verify
keypoint order and fps; add a synthetic multi-keypoint fixture to your tests.

### 5. Body center (needs H1a)
- [ ] Center = mean of torso keypoints (e.g. neck + left hip + right hip, **only after the readme confirms these exist**). Never the nose alone.
- [ ] If some torso keypoints are missing in a frame, use the rest; if all are missing, NaN
- [ ] Write the definition in the docstring and in `INTERFACES.md`

### 6. Dyadic features (needs H1)
- [ ] Center distance and its rate (negative = approaching)
- [ ] Relative speed
- [ ] Nose-to-nose distance; nose of A to tail base of B, and of B to A
- [ ] Facing angle: angle between A's body axis (tail base → nose or neck) and the direction to B, for both mice
- [ ] Time spent close (trailing count of seconds with distance below a parameter)
- [ ] Speed synchrony: trailing-window correlation of the two speeds
- [ ] Tests: approaching / separating mice, A facing B vs facing away, missing keypoints

These are motion **measurements**, not behavior labels. Write that in the docstring.

### 7. Quality per frame + warnings for the agent
- [ ] `frame_tracking_ok(flags_df, ...)` → one bool per `session × frame` (false if a needed keypoint of either mouse is invalid)
- [ ] `tracking_warning_events(flags_df, ...)` → list of dicts in the event-log format from `INTERFACES.md` §4 with `action: "tracking_warning"` (wrap the existing `quality.tracking_loss_events`)
- [ ] `compute_features(long_df, params)` → the whole feature table in one call

**→ Deliver H2 to Igor and H2b to Syed (merged by end of Day 2).** Post in the chat with a
3-line usage example.

---

## Day 3: Mon Oct 19

### 8. Tune quality + filter parameters on real data (train/val sessions only)
Use Igor's `splits/split_manifest.json`. **Never look at test sessions for tuning.**
- [ ] Look at the real jitter: distribution of frame-to-frame step size, so `frozen_tol_px` and `frozen_min_frames` make sense
- [ ] Jump threshold from the real speed distribution (e.g. far above the 99.9th percentile; write down how you chose it)
- [ ] One-Euro `min_cutoff` / `beta`: compare raw vs smoothed noise and the added lag
- [ ] Save chosen values in one place (e.g. `src/live_tracker/params.py` or a JSON), with a short note on how each was chosen

### 9. Sensitivity tests (`scripts/sensitivity.py`)
For a few validation sessions, measure how much the features change when you:
- [ ] add small coordinate jitter (e.g. 1, 2, 5 px)
- [ ] remove short runs of frames (gaps)
- [ ] change smoothing settings
- [ ] change thresholds by ±20%
Save a small results table to `outputs/sensitivity.csv` (outputs/ is git-ignored; copy the table into the report).
This measures **feature stability**, not classification accuracy. Keep them separate.

### 10. Live paths are causal only
- [ ] Make sure nothing Igor or Syed uses for replay calls `motion.clean_trajectory` (Savitzky-Golay)
- [ ] Keep the old starter section in `app.py` but labeled "offline only"

**While you wait for H4:** prepare `scripts/ablation.py` so it only needs `evaluate()` plugged in.

---

## Day 4: Tue Oct 20 (prototype day)

### ⏳ Wait for H4 (`evaluate()` + model A results) from Igor

### 11. Ablation: basic features vs basic + ScopeScore features
- [ ] Same classifier type, same split, same tuning budget, only the features differ
- [ ] Compare on **validation**; report the test set **once**, at the end, for both models
- [ ] Metrics: per-class F1 (attack, mount, investigation), event-level attack precision/recall, onset-to-alert latency, false alerts/hour
- [ ] Write an honest conclusion: if ScopeScore features don't help, say so
- [ ] Optional, **only if time is left**: SPARC / log dimensionless jerk, with written definitions, and kept only if they help on validation

### 12. Motion-engine page in `app.py`
- [ ] Load one real CalMS21 session (via Igor's loader), show features + tracking warnings
- [ ] Synthetic demo option stays, still clearly labeled

### 13. Help integration (H5)
- [ ] Review Igor's and Syed's PRs; fix feature bugs they find

---

## Day 5: Wed Oct 21

- [ ] README: rewrite "ScopeScore adaptations" to list **only what was built and tested**, plus what was deliberately *not* transferred (no surgical thresholds, no clinical meaning)
- [ ] Report / slides: motion engine section, sensitivity table, ablation table (measured numbers only)
- [ ] Final full test run on Windows; update `IMPLEMENTATION_STATUS.md` with commands and results
- [ ] Rehearse demo with the team

## Done when
- `compute_features()` works on real CalMS21 sessions and is causal (test proves it)
- All parameters chosen on train/val and written down
- Sensitivity table and ablation table exist with real numbers
- README ScopeScore section matches the code
