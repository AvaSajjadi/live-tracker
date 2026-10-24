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
