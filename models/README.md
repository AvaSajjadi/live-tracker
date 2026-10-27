# Models

Every model named in the project report, what it does here, and where it is.

| Model | What it does in Live Tracker | Files | License |
|---|---|---|---|
| **YOLO pose model** (Ultralytics YOLO26n-pose, fine-tuned) | Finds both mice in every live camera picture: 7 keypoints per mouse, about 10 ms per picture on a MacBook Air | `pose/mouse_pose_best.pt`, `pose/mouse_pose_best_openvino_model/` (Intel laptops), settings `pose/pose_model_mac.json` and `pose/pose_model.json` | AGPL-3.0 (Ultralytics) |
| **Model A** (logistic regression, 77 basic pose-geometry features) | Baseline behavior classifier, used to measure what the motion engine adds | `model_a.joblib`, `model_a.json` | this project |
| **Model B** (logistic regression, 102 features: basic + motion engine) | Scores attack, investigation and mount on every frame; the alerts use it | `model_b.joblib`, `model_b.json` | this project |
| **ScopeScore** (Sajjadi, 2026; YOLO11m-seg fine-tuned on CholecInstanceSeg) | The surgical-instrument tracker whose methods were adapted for mice: tracking checks, causal One-Euro smoothing, motion features, point stabilizer (`src/live_tracker/quality.py`, `filters.py`, `features.py`, `point_stabilizer.py`). Not run by Live Tracker; its defaults are the published settings, the same values `point_stabilizer.py` uses (see `scopescore/README.md`) | `scopescore/weights/best.pt`, its code, configs and `methods.md` | weights: AGPL-3.0 (Ultralytics) and non-commercial share-alike (Cholec80 data, CC BY-NC-SA 4.0); code: the author's |
| **MARS** (Segalin et al., 2021) | The offline route: finds the mice in a recorded video when live pose is not needed (`scripts/mars_pose_offline.py`) | not stored here, see below | CC BY-NC 4.0 |

## Training

- **YOLO pose model:** `scripts/mars_pose_to_yolo.py` (MARS hand-labeled set to YOLO format), `scripts/train_yolo_pose.py` or
  `colab/train_yolo_pose_colab.ipynb`, then `scripts/eval_yolo_pose.py` and `scripts/bench_yolo_pose_*.py`. See `docs/pose_model.md`.
- **Model A and Model B:** `python -m live_tracker.classify features / train / scores` on CalMS21 Task 1 (commands in the main
  README, "Working with CalMS21"); `scripts/ablation.py` compares the two (`docs/motion_engine_results.md`).
- **ScopeScore:** `scopescore/configs/train_args.yaml` (the training settings), `scopescore/configs/train_results.csv` (the
  training curve), `scopescore/methods.md`. Training data: CholecInstanceSeg, built on the public Cholec80 videos; no hospital
  data. The paths in these files start with `<ScopeScore>/`, the ScopeScore project folder.

## MARS

The MARS pretrained models are published by Caltech (CC BY-NC 4.0, record 801be-66057). They are not stored in this
repository. To use the offline route, download them once:

    python scripts/mars_pose_offline.py --download-only --models-dir ~/mars_local/models

(543 MB zip; the md5 is checked and only the three `.pb` graph files are kept.) If the download is refused, open
https://data.caltech.edu/records/801be-66057 in a browser, download `models.zip`, and unzip its `models/` folder there.

## References

- Jocher, G., et al. (2026). Ultralytics YOLO26. arXiv:2606.03748.
- Sajjadi, A. (2026). ScopeScore: A video processing model for surgery [Manuscript submitted].
- Segalin, C., et al. (2021). The Mouse Action Recognition System (MARS). eLife, 10, e63720.
- Twinanda, A. P., et al. (2017). EndoNet (the Cholec80 dataset). IEEE Transactions on Medical Imaging, 36(1).
