# Motion engine results (Ava, 2026-10-22)

Measured numbers only. Every number below is on the **train / validation** sessions of the CalMS21
Task 1 train file. The test split was never read: no test file is in `data/raw/`, the feature files
used here contain no test session (56 train + 14 validation sessions), the sensitivity script
refuses any session outside the manifest's train/validation lists, and `--final` of the ablation was
not run (`outputs/ablation_test.lock` does not exist). Test is reported once, at the very end, under
a policy the team agrees first.

Commands to reproduce are at the bottom.

## 1. Ablation: basic features vs basic + ScopeScore-inspired features

Question: do the motion-engine features (`features.py`) help a classifier compared with plain pose
geometry (`basic_features.py`)?

Setup (identical for both): class-weighted logistic regression, `C` chosen from {0.01, 0.1, 1}
on the validation sessions, training on every 3rd frame of the 56 train sessions, seed 0.
A = basic features. B = basic + the motion-engine columns (Igor's "model B").
Compared on the 14 validation sessions (94,305 frames).

| | A basic | B basic + ScopeScore |
|---|---|---|
| macro F1 (attack, investigation, mount) | 0.499 | **0.651** |
| attack F1 | 0.226 | **0.569** |
| investigation F1 | 0.649 | **0.780** |
| mount F1 | **0.620** | 0.605 |
| attack event precision | 0.161 | **0.702** |
| attack event recall | 0.179 | **0.714** |
| attack onset-to-alert latency (median) | 1.43 s | **0.38 s** |
| attack false alerts per hour | 107.7 | **32.1** |
| true / predicted attack bouts | 56 / 112 | 56 / 94 |

Paired difference in macro F1, B minus A, over the 14 sessions: **+0.060** (95 % bootstrap interval
+0.040 to +0.082). B is better in 14 of 14 sessions.

By class:
- **Investigation:** B is better in 14 of 14 sessions (mean F1 difference +0.135).
- **Attack:** B is better in all 4 sessions that contain attack frames and equal in the other 10
  (no attack frames there). Mean F1 difference +0.056 over all 14.
- **Mount:** the motion-engine features do **not** help. Mount F1 is 0.620 vs 0.605, and A is
  better in 4 sessions. Say this plainly in the report.

### What this does and does not show

- It supports: adding the causal motion features to plain pose geometry improved investigation and
  attack detection on these validation sessions, and did not improve mount.
- Only 4 of the 14 validation sessions contain any attack. The attack numbers rest on 56 true bouts
  in 4 sessions, so they are the least certain.
- Validation numbers are optimistic. `C` was chosen on the same validation sessions. The validation
  sessions were split from the official train file **by session, not by animal**, so the same
  animals appear in train and validation (only the official test split is by animal identity).
- The logistic regression hit its 300-iteration limit in some fits (scikit-learn convergence
  warnings). Both feature sets got the same budget, but the models are not fully converged and B has
  more features.
- Event metrics use the placeholder bout rule (threshold 0.5, minimum 0.5 s, no merge gap). It was
  not tuned, and the latency of 0.38 s for B is not a recommended operating point.
- Models were refit here from Igor's Oct 19 feature files (`basic`, `motion`, `labels` parquet;
  56 train + 14 validation sessions) with the Oct 17 `features.py`. Igor's saved models are not in
  the repository, so this is a refit with his code, not his artifact.
- Keypoint features (section 3) are **not** in either model.

## 2. Sensitivity: how stable are the features?

This measures feature stability, not classification accuracy. Four real validation sessions
(mouse005, mouse028, mouse045, mouse047; 12,135 frames; two with attack, one with mounting),
the team's real quality settings (torso keypoints, image frame size, `low_conf` below 0.15),
3 repeats of each random disturbance. Table: median |change| divided by the standard deviation
of the feature (0 = unchanged, 1 = a change as big as the feature's own spread).

| disturbance | dist_center | dist_rate | speed_a | accel_a | turn_rate_a | straightness_w_a | speed_corr_w |
|---|---|---|---|---|---|---|---|
| jitter 1 px | 0.001 | 0.019 | 0.013 | 0.103 | 0.058 | 0.006 | 0.037 |
| jitter 2 px | 0.002 | 0.038 | 0.026 | 0.207 | 0.116 | 0.015 | 0.080 |
| jitter 5 px | 0.004 | 0.094 | 0.066 | 0.567 | 0.289 | 0.058 | 0.212 |
| gaps of 2 frames | 0.000 | 0.004 | 0.002 | 0.002 | 0.002 | 0.005 | 0.029 |
| gaps of 10 frames | 0.003 | 0.024 | 0.000 | 0.001 | 0.001 | 0.021 | 0.186 |
| no smoothing | 0.039 | 0.437 | 0.349 | 2.813 | 0.983 | 0.834 | 0.579 |
| EMA tau 0.05 | 0.028 | 0.227 | 0.197 | 0.985 | 0.325 | 0.457 | 0.389 |
| EMA tau 0.2 | 0.005 | 0.024 | 0.024 | 0.065 | 0.022 | 0.045 | 0.058 |
| One-Euro min_cutoff 0.5 | 0.030 | 0.120 | 0.118 | 0.262 | 0.092 | 0.212 | 0.304 |
| One-Euro min_cutoff 2 | 0.017 | 0.113 | 0.104 | 0.392 | 0.127 | 0.226 | 0.229 |
| One-Euro beta 0.05 | 0.032 | 0.247 | 0.211 | 1.037 | 0.355 | 0.451 | 0.394 |

Findings:
- **Position-level features are stable.** `dist_center` barely moves under any disturbance.
- **Acceleration is the fragile feature.** 5 px of jitter changes it by about 0.57 of its spread, and
  removing the smoother changes it by 2.8. Rely on it least.
- **The smoother matters more than the noise.** The One-Euro setting changes the derived features
  (speed, acceleration, turn rate) by more than 1 px of jitter does. In particular, the tuned
  `beta = 0.05` in `params.py` would change `accel_a` by about 1.0 of its spread compared with the
  `beta = 0` the models were trained on. That is why `params.py` is not wired into the defaults and
  Model B was not retrained with it.
- **Short gaps are harmless.** Filling gaps of 2 frames changes almost nothing; 10-frame gaps mostly
  hurt `speed_corr_w`, and up to 5 % of frames change NaN status.
- **Thresholds, +-20 %:** the median change is 0 for almost every feature because most frames are
  unaffected. The effect is in the tails, so read the 95th percentile in
  `outputs/sensitivity_real_validation.csv`: for example `close_time_s` changes by up to 0.07 s
  (`close_dist_px` x0.8) and 0.17 s (x1.2), and `straightness_w_a` by up to 0.26 (`window_s` x0.8).
  Median-based summaries understate threshold sensitivity.
- Real validation tracking is almost clean (see the real-session page): 0 % missing, frozen, jump or
  out-of-frame torso frames in the sessions inspected, 0.2 % `low_conf`. The disturbances above are
  therefore synthetic stress on clean data, not a replay of real tracker failures.

## 3. Keypoint features (built, tested, not in the models)

`features.keypoint_features()` adds nose-to-nose distance, nose-to-tail-base distances (both
directions) and the facing angle of each mouse (angle between its tail-base-to-nose axis and the
direction to the other mouse's body center, 0 = pointing at it). It is causal (tested), gives NaN
for missing keypoints, and is **opt-in**: it is not part of `compute_features()`, so the section 5
table and the trained models are unchanged. On the two real sessions inspected (mouse005, mouse045)
the five columns are present on 97 to 99.7 % of frames.

They have not been evaluated as classifier inputs. Whether they help is an open question; testing it
means regenerating features and retraining, which we decided against before the demo deadline.

## 4. Real-session page

`pages/4_Real_Session.py` shows one real validation session: tracking quality and warnings, the
causal features, the keypoint features, and the ground-truth annotation as context. It reads only
the hash-verified validation cache (`investigation_data`), which refuses any session outside the
manifest's validation split. About 4 s per session.

## Reproduce

```
# feature files: python -m live_tracker.classify features --train ... --test ...   (Igor; needs pyarrow)
python scripts/ablation.py --basic data/processed/basic.parquet --scope data/processed/motion.parquet \
    --labels data/processed/labels.parquet --manifest splits/split_manifest.json \
    --fit live_tracker.classify:fit_scores --evaluate live_tracker.evaluate:summary_metrics --primary macro_f1
python scripts/sensitivity.py --calms21 data/raw/calms21/calms21_task1_train.json --repeats 3 \
    --sessions task1/train/mouse005_task1_annotator1 task1/train/mouse028_task1_annotator1 \
               task1/train/mouse045_task1_annotator1 task1/train/mouse047_task1_annotator1 \
    --out outputs/sensitivity_real_validation.csv
python -m live_tracker.investigation_data --source data/raw/calms21/calms21_task1_train.json --all
python -m streamlit run app.py        # then open "Real Session" in the sidebar
```

Outputs go to `outputs/` (git-ignored). Do **not** pass `--final` to `ablation.py` until the team
has agreed when the single test evaluation happens.
