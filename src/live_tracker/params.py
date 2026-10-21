"""Quality and smoothing values chosen on real CalMS21 data (plan/AVA.md task 8).

Tuned on the 56 TRAIN sessions and re-checked on the 14 VALIDATION sessions
(torso keypoints). Test sessions were never read. Reproduce with
scripts/tune_quality.py. These are NOT wired into the function defaults yet;
pass them explicitly.
"""

# Identical coordinates for 15 frames (0.5 s). In real data such runs are rare:
# >=15 frames: 3 on train, 0 on validation; >=10 frames: 11 on train, 0 on validation.
# Kept at 15 so it matches the 0.5 s frozen veto in quality_v1. Because it almost
# never fires on CalMS21, it is a safety net, not an active filter.
FROZEN_MIN_FRAMES = 15

# Jump: a torso step of >= 100 px/frame (3000 px/s). Chosen where the step-size tail
# turns into tracker errors: such steps are 0.19% of train (0.20% of validation) and
# have median keypoint score 0.20 (0.21), against 0.72 (0.71) for all other steps.
MAX_SPEED_PX_S = 3000.0

# One-Euro: min_cutoff 1.0, beta 0.05. Cuts jitter on still frames by about 47%
# (median step 0.53 x raw) with 0 frames of added lag. A lower min_cutoff with
# beta 0 removes a little more jitter but lags 4-7 frames. Lag is measured in whole
# frames, so it is coarse. Values in outputs/tuning_smoothing.csv (git-ignored).
ONE_EURO = {"min_cutoff": 1.0, "beta": 0.05}
