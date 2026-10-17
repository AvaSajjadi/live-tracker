# Igor: CalMS21 data and behavior model

You own the real data and the classifier: download, schema, loader, session splits,
baseline model, evaluation, and the per-frame score file the agent replays.
Overview and handoff table: `plan/00_TEAM_PLAN.md`.

**You deliver:** H1a (schema notes, Day 1 midday), H1 (loader + splits, Day 1 end),
H3 (test-session scores → Syed, Day 3 end), H4 (`evaluate()` + model A results → Ava, Day 3 end)
**You wait for:** H2 (Ava's `compute_features()`, Day 2 end)

**You are first on the critical path.** Ava's Day 2 and Syed's skeleton view wait for H1,
so the loader comes before anything else.

Your files: `src/live_tracker/calms21.py`, `calms21_inspect.py`, `basic_features.py`,
`classify.py`, `evaluate.py`, `splits/`, `docs/calms21_schema.md`, and their tests.
Do **not** edit Ava's motion-engine files (`motion.py`, `filters.py`, `quality.py`, `features.py`);
ask her if something there needs to change.

---

## Day 1: Sat Oct 17

### 1. Team setup (H0, with everyone, 1 h)
- [ ] Clone, venv, `python -m pytest -q` passes (53 tests expected)
- [ ] Merge PR #2; agree on interfaces

### 2. Get the data
- [ ] From https://data.caltech.edu/records/s0vdx-0k302 download **Task 1 pose + annotation files only**. **Do not** download the large video archive.
- [ ] Put them in `data/raw/calms21/` (git-ignored) and upload a copy to the shared Drive for Ava and Syed
- [ ] Read the official readme. Note the license (CC BY 4.0) and how to cite it

### 3. Inspect and write `docs/calms21_schema.md` → **H1a by midday**
- [ ] `python -m live_tracker.calms21_inspect data/raw/calms21/<task1 file>.json`
- [ ] Write down, **from the files and readme, not from memory**:
  - file layout (how sessions are stored, the session ID key)
  - pose array shape and what each axis means
  - keypoint names **in order**, and which mouse is which (resident / intruder)
  - x/y order, pixel units, image size if given
  - frame rate
  - label numbers → names (attack, investigation, mount, other?)
  - official train / test division
- [ ] Post in the chat: Ava needs the keypoint order for the body center, Syed for the skeleton drawing

### 4. Loader: `src/live_tracker/calms21.py`
- [ ] Replace the `load_calms21` placeholder in `calms21_inspect.py` (or move it)
- [ ] `load_calms21(path) -> (poses_long, labels)`:
  - `poses_long`: the long table from `INTERFACES.md` §1 (`session, mouse, keypoint, frame, time_s, x, y`)
  - `labels`: `session, frame, time_s, label` (string names)
- [ ] Option to load only some sessions (the full JSON is large)
- [ ] Tests with a **tiny fixture JSON** that copies the real structure (2 sessions, ~20 frames). Don't commit real data.

### 5. Session split: `splits/split_manifest.json`
- [ ] Official test sessions = **held-out test**. Never used for training, tuning, or by the agent
- [ ] Split official train sessions into train / validation **by session**, fixed seed (write the seed in the file)
- [ ] Test: no session appears in two splits

**→ Deliver H1 (loader + manifest merged) by end of Day 1.**

---

## Day 2: Sun Oct 18

### 6. Basic features: `basic_features.py`
Plain pose geometry, **no ScopeScore-inspired additions** (those are Ava's; this is the baseline for her ablation):
- [ ] Keypoint coordinates relative to each mouse's own center, and pairwise distances between the two mice's keypoints, for each frame
- [ ] Only the current frame (no time windows, so it is causal by construction)

### 7. Model A (baseline)
- [ ] `classify.py`: class-weighted logistic regression or random forest (scikit-learn is already in `requirements.txt`)
- [ ] Train on **train** sessions only; choose hyperparameters on **validation**
- [ ] Output probabilities for `attack`, `mount`, `investigation`
- [ ] Save the model with joblib to `models/` (git-ignored) + parameters JSON

### 8. Evaluation: `evaluate.py`
- [ ] Frame level: per-class precision / recall / F1, confusion matrix
- [ ] Turn frame scores into bouts with `events.threshold_events` (explicit threshold + minimum duration + merge rule; never across sessions)
- [ ] Event level for attack: precision, recall (match rule written down, e.g. overlap ≥ 1 frame)
- [ ] Onset-to-alert latency (s) and false alerts per hour
- [ ] Tests with tiny hand-made label/score arrays where you know the right answer

**While you wait for H2:** run model A fully on validation so H4 is ready early.

---

## Day 3: Mon Oct 19

### ⏳ Wait for H2 (`compute_features()` from Ava)

### 9. Model B = basic + Ava's features
- [ ] Same classifier type and tuning budget as model A (fair comparison for Ava's ablation)
- [ ] Choose model A or B on **validation** only
- [ ] Calibrate scores on validation (e.g. `CalibratedClassifierCV`) so 0.8 means something

### 10. Score file for the agent → **H3 by end of Day 3**
- [ ] For each **test** session: one row per frame in the format of `INTERFACES.md` §2
  (`session, frame, time_s, attack, mount, investigation, tracking_ok`)
- [ ] `tracking_ok` comes from Ava's `frame_tracking_ok()`
- [ ] **No label columns** in this file
- [ ] Features must be causal (Ava's are; yours are per-frame), so replaying the file is honest
- [ ] Put it on the Drive (and in `data/processed/`, git-ignored) and tell Syed

### 11. Send H4 to Ava
- [ ] `evaluate()` merged + model A validation results table → Ava runs the ablation on Day 4

If H2 is late: **ship H3 with model A** and replace it with model B later. Syed must not wait.

---

## Day 4: Tue Oct 20 (prototype day)

- [ ] Choose default `alert_threshold` / `ask_threshold` for the demo attack rule on **validation**; send them to Syed
- [ ] One command to retrain and re-evaluate, e.g. `python -m live_tracker.classify --split splits/split_manifest.json`
- [ ] Save artifacts: model, parameters, split manifest, evaluation outputs (small result tables)
- [ ] Help Syed with H5 (full pipeline run on one real test session); fix bugs

---

## Day 5: Wed Oct 21

- [ ] README: data download + placement, CalMS21 citation, training and evaluation commands
- [ ] Report: results section, **measured numbers only**, both models, test set reported once
- [ ] Final leakage check: grep that test sessions never appear in training, tuning, or agent inputs
- [ ] Rehearse demo with the team

## Done when
- Loader works on the real files and its tests pass
- Split is by session and is saved with its seed
- Model A and model B exist, chosen on validation, test reported once
- Test-session score file delivered to Syed without labels
