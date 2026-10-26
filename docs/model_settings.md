# Model A and model B: exact settings (reproduced 2026-10-24)

The saved model files are not in Git (`models/` is ignored). This page records what they were, so anyone can rebuild
them. Both models were rebuilt from the CalMS21 Task 1 **train** file only. The test file was not read.

| | Model A | Model B |
|---|---|---|
| Features | basic pose geometry, 77 | basic + motion engine, 102 |
| Classifier | class-weighted logistic regression | same |
| `C` searched | 0.01, 0.1, 1 | same |
| `C` chosen on validation | **1.0** | **1.0** |
| Training frames | every 3rd frame of the 56 train sessions (137,827 rows) | same |
| `max_iter`, seed | 300, 0 | same |
| Validation macro F1 | 0.499 | **0.651** |

Model B validation F1 by `C`: 0.01 gives 0.634, 0.1 gives 0.649, 1 gives 0.651.

Rebuilt validation numbers match `docs/motion_engine_results.md` exactly: model A attack event precision 0.161,
recall 0.179, onset-to-alert 1.433 s, 107.7 false alerts per hour; model B 0.702, 0.714, 0.383 s, 32.1.
The 137,827 training rows match `scripts/check_leakage.py`.

The full list of Model B's 102 feature names is in `models/model_b.json` after training (`feature_columns`).

## Rebuild

```powershell
# features from the TRAIN file only (no test file needed)
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -c "from live_tracker.classify import build_feature_files; build_feature_files(['data/raw/calms21/calms21_task1_train.json'], 'data/processed')"
.\.venv\Scripts\python.exe -m live_tracker.classify train --name model_a --features basic
.\.venv\Scripts\python.exe -m live_tracker.classify train --name model_b --features basic,motion
```

The normal `classify features` command also needs the test file and writes the test sessions into the same files.
The numbers above do not use them.

## Calibration: NOT applied (still to do)

`plan/IGOR.md` item 9 asked for scores calibrated on validation. It was not done, so the saved models give **raw
class-weighted logistic regression scores**. A score of 0.8 is not an 80 % chance. Say this in the report.

Calibration code is in `classify.py` and is **off** by default. Turn it on with `FitParams(calibrate="sigmoid")`.
It fits a sigmoid on the validation sessions with the classifier frozen (2026-10-24 test, not kept):

| | Raw (used) | Calibrated |
|---|---|---|
| Model B validation macro F1 | 0.651 | 0.592 |
| Model A validation macro F1 | 0.499 | 0.497 |
| Demo rule ask 0.6 / 0.5 s | event F1 0.728 | detects 0 of 56 attack bouts |
| Best rule after calibration | | ask 0.3 / 0.5 s, event F1 0.734, 11.5 false alerts per hour |

Calibration lowers the attack scores, because the balanced class weights had pushed them up. If we turn it on, the
demo rule (`evaluate.ATTACK_RULE_V1`), the alert threshold of 0.8, the Stage 9 runs, the tests and these docs must be
redone. Validation numbers would also be slightly optimistic, because validation fits the calibration.

## Igor's original models (received 2026-10-24)

Igor's Oct 19 files (`model_a`, `model_b`, `.joblib` and `.json`) were checked by reading the `.json` files only; the
`.joblib` files were not loaded. They are not in Git.

| | Igor | Rebuilt here |
|---|---|---|
| Machine | macOS arm64 | Windows 11 |
| Python | 3.13.5 | 3.12.10 |
| scikit-learn | 1.6.1 | 1.7.2 |
| numpy / pandas / scipy | 2.1.3 / 2.2.3 / 1.15.3 | 2.3.5 / 2.3.3 / 1.16.3 |

Same in both: `C` 1.0, seed 0, training rows (137,827), feature count (77 and 102) and feature names.
Validation macro F1 differs only in the third decimal: model A 0.4978 (Igor) vs 0.4986 (rebuilt), model B 0.65143 vs
0.65148. The score differences of up to 0.09 are most likely from the different library versions and platform.
That cause was not tested.
