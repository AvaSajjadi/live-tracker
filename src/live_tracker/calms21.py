"""CalMS21 Task 1 loader and session split.

Facts used here come from the official readme and the real files
(see docs/calms21_schema.md):
- JSON layout: {group: {session_id: {keypoints, scores, annotations, metadata}}}
- keypoints: (frames, mouse, xy, bodypart) = (T, 2, 2, 7), pixels, image 1024 x 570
- scores:    (frames, mouse, bodypart)     = (T, 2, 7), 0..1
- mouse 0 = resident (black), mouse 1 = intruder (white)
- vocab: attack 0, investigation 1, mount 2, other 3
- 30 fps (CalMS21 paper, Section 4.1)

Mouse IDs in our tables are "a" (resident) and "b" (intruder), so the
feature code's `_a` / `_b` columns and its sorted-mouse order (a first)
stay the same as on the synthetic data.

Usage:
    poses, labels = load_calms21("data/raw/calms21/calms21_task1_train.json",
                                 sessions=["task1/train/mouse001_task1_annotator1"])
    python -m live_tracker.calms21 split \
        --train data/raw/calms21/calms21_task1_train.json \
        --test  data/raw/calms21/calms21_task1_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .quality import SUGGESTED_MIN_SCORE, flag_tracking, interpolate_short_gaps

FPS = 30.0
IMAGE_SIZE = (1024, 570)  # width, height in pixels
KEYPOINTS = ("nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base")
MICE = ("a", "b")                       # index 0, index 1 in the JSON
MOUSE_ROLE = {"a": "resident", "b": "intruder"}
VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
# Team decision, NOT from the readme (it defines no torso). See docs/calms21_schema.md.
TORSO_KEYPOINTS = ("neck", "left_hip", "right_hip")

SPLITS = ("train", "validation", "test")


def _read_json(path_or_dict):
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict) as f:
        return json.load(f)


def _sessions(data):
    """Flatten {group: {session_id: record}} to {session_id: record}."""
    out = {}
    for group, sessions in data.items():
        for sid, rec in sessions.items():
            if sid in out:
                raise ValueError(f"session {sid!r} appears in more than one group")
            out[sid] = rec
    return out


def session_ids(path_or_dict):
    """All session IDs in a file, sorted."""
    return sorted(_sessions(_read_json(path_or_dict)))


def _session_to_long(sid, rec):
    kp = np.asarray(rec["keypoints"], dtype=float)
    sc = np.asarray(rec["scores"], dtype=float)
    t = kp.shape[0]
    if kp.shape[1:] != (len(MICE), 2, len(KEYPOINTS)):
        raise ValueError(f"{sid}: keypoints shape {kp.shape}, expected (T, 2, 2, 7)")
    if sc.shape != (t, len(MICE), len(KEYPOINTS)):
        raise ValueError(f"{sid}: scores shape {sc.shape}, expected ({t}, 2, 7)")
    # (T, mouse, xy, kp) -> (mouse, kp, T) so the rows run frame-fastest
    x = kp[:, :, 0, :].transpose(1, 2, 0).ravel()
    y = kp[:, :, 1, :].transpose(1, 2, 0).ravel()
    s = sc.transpose(1, 2, 0).ravel()
    n_m, n_k = len(MICE), len(KEYPOINTS)
    frames = np.tile(np.arange(t), n_m * n_k)
    return pd.DataFrame({
        "session": sid,
        "mouse": np.repeat(MICE, n_k * t),
        "keypoint": np.tile(np.repeat(KEYPOINTS, t), n_m),
        "frame": frames,
        "time_s": frames / FPS,
        "x": x, "y": y, "score": s,
    })


def _session_labels(sid, rec):
    vocab = rec.get("metadata", {}).get("vocab")
    if vocab is not None and vocab != VOCAB:
        raise ValueError(f"{sid}: vocab {vocab} differs from expected {VOCAB}")
    ids = np.asarray(rec["annotations"], dtype=int)
    if len(ids) != len(rec["keypoints"]):
        raise ValueError(f"{sid}: {len(ids)} annotations but {len(rec['keypoints'])} frames")
    names = np.array(sorted(VOCAB, key=VOCAB.get))
    if ids.size and (ids.min() < 0 or ids.max() >= len(names)):
        raise ValueError(f"{sid}: label id outside {VOCAB}")
    frames = np.arange(len(ids))
    return pd.DataFrame({"session": sid, "frame": frames, "time_s": frames / FPS,
                         "label": names[ids]})


def load_calms21(path_or_dict, sessions=None):
    """Load a CalMS21 Task 1 file.

    Returns (poses_long, labels):
      poses_long: session, mouse, keypoint, frame, time_s, x, y, score  (INTERFACES.md §1 + score)
      labels:     session, frame, time_s, label (string name)
    `sessions`: optional list of session IDs to keep (the whole JSON is still read).
    """
    recs = _sessions(_read_json(path_or_dict))
    if sessions is not None:
        unknown = sorted(set(sessions) - set(recs))
        if unknown:
            raise KeyError(f"sessions not in file: {unknown}")
        keep = [s for s in sorted(recs) if s in set(sessions)]
    else:
        keep = sorted(recs)
    if not keep:
        raise ValueError("no sessions selected")
    poses = pd.concat([_session_to_long(s, recs[s]) for s in keep], ignore_index=True)
    labels = pd.concat([_session_labels(s, recs[s]) for s in keep], ignore_index=True)
    # Plain strings, not categoricals: groupby on categoricals (observed=False)
    # would invent rows for session/frame combinations that do not exist.
    return poses, labels


def flag_quality(poses):
    """Quality flags with the team settings for CalMS21 (quality.py, owner: Ava).

    Frame size = IMAGE_SIZE, low_conf below quality.SUGGESTED_MIN_SCORE (chosen
    by Ava on the validation sessions; do not raise it without checking, it costs
    attack and mount frames first), then short gaps filled. The result can go
    straight into features.compute_features() and basic_features().
    """
    return interpolate_short_gaps(flag_tracking(
        poses, frame_size=IMAGE_SIZE, score_col="score", min_score=SUGGESTED_MIN_SCORE))


def make_split(train_ids, test_ids, val_frac=0.2, seed=0):
    """Session-level split. Official test stays test; official train -> train / validation."""
    train_ids, test_ids = sorted(set(train_ids)), sorted(set(test_ids))
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"sessions in both official train and test: {sorted(overlap)}")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    n_val = max(1, int(round(len(train_ids) * val_frac)))
    if n_val >= len(train_ids):
        raise ValueError("not enough train sessions for a validation split")
    order = np.random.default_rng(seed).permutation(len(train_ids))
    val = sorted(train_ids[i] for i in order[:n_val])
    tr = sorted(train_ids[i] for i in order[n_val:])
    return {"seed": seed, "val_frac": val_frac,
            "note": "Split by session. test = official CalMS21 Task 1 test; never train, tune or replay-tune on it.",
            "train": tr, "validation": val, "test": test_ids}


def check_manifest(m):
    """Raise if a split is missing or a session is in two splits."""
    seen = {}
    for split in SPLITS:
        if split not in m:
            raise ValueError(f"manifest missing split {split!r}")
        for s in m[split]:
            if s in seen:
                raise ValueError(f"session {s!r} in both {seen[s]} and {split}")
            seen[s] = split
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description="CalMS21 tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="write splits/split_manifest.json")
    sp.add_argument("--train", required=True)
    sp.add_argument("--test", required=True)
    sp.add_argument("--out", default="splits/split_manifest.json")
    sp.add_argument("--val-frac", type=float, default=0.2)
    sp.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    m = check_manifest(make_split(session_ids(a.train), session_ids(a.test), a.val_frac, a.seed))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2) + "\n")
    print(f"wrote {out}: train {len(m['train'])}, validation {len(m['validation'])}, "
          f"test {len(m['test'])} sessions (seed {a.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
