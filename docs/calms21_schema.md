# CalMS21 Task 1 schema (H1a)

Code constants: `live_tracker.calms21` (`KEYPOINTS`, `TORSO_KEYPOINTS`, `FPS`, `VOCAB`, `MICE`).

Source: official `readme.md` and `calms21_convert_to_npy.py` from
https://data.caltech.edu/records/s0vdx-0k302, checked against the real files.
fps is from the CalMS21 paper. The torso choice is a team decision.

## Files

- Download: `task1_classic_classification.zip` only (no videos, no MARS features).
- Inside: `calms21_*.json` = keypoints + scores + annotations (we use these).
  `taskprog_features_*.json` = pretrained TREBA features (we do **not** use these).
- Place in `data/raw/calms21/` (git-ignored).

## JSON layout

```
{ "annotator-id_0": {                                  # group key
    "task1/train/mouse001_task1_annotator1": {         # session ID
        "keypoints":   [...],   # frames x 2 x 2 x 7
        "scores":      [...],   # frames x 2 x 7
        "annotations": [...],   # frames
        "metadata":    {...}    # annotator_id (int), vocab (name -> id)
    }, ... } }
```

- Session ID = the inner key, e.g. `task1/train/mouse001_task1_annotator1`.
- Official train / test division: **separate files**, `calms21_task1_train.json`
  (1.2 GB) and `calms21_task1_test.json` (632 MB). The session ID also contains `train` / `test`.
- Test file (checked): 1 group `annotator-id_0`, **19 sessions**,
  `mouse071` ... `mouse089`, about 18k–24k frames each.
- Train file (checked): 1 group `annotator-id_0`, **70 sessions**, `mouse001` ... `mouse070`.
- Checked on `mouse001`: keypoints `(21364, 2, 2, 7)`, scores `(21364, 2, 7)`,
  x 6–1006, y 7–563, so axis 2 is (x, y) as the readme says.
- `metadata` (checked) = `{'annotator-id': 0, 'vocab': {...}}`. No fps in it.
- The official split is **by animal identity**.

## Keypoints

- Shape: `(frames, mouse, xy, bodypart)` = `(T, 2, 2, 7)`.
- Axis 2: index 0 = x, index 1 = y.
- Units: pixels, relative to the full image. Image size **1024 x 570**.
- `scores`: `(T, 2, 7)`, confidence 0 to 1.

### Mouse IDs

| JSON index | our `mouse` ID | animal |
|---|---|---|
| 0 | `a` | resident (black) |
| 1 | `b` | intruder (white) |

We use `a` / `b` so feature columns stay `_a` / `_b` and the resident sorts first.

Labels describe behavior **initiated by the resident** (mouse 0).

### Body parts, in order

| index | name |
|---|---|
| 0 | nose |
| 1 | left_ear |
| 2 | right_ear |
| 3 | neck |
| 4 | left_hip |
| 5 | right_hip |
| 6 | tail_base |

### Torso keypoints (team decision, not from the readme)

The readme does not define a torso. Proposal: `neck, left_hip, right_hip`.
Tail base is left out because the tail moves on its own. **Needs team agreement.**

## Labels (Task 1)

| id | name |
|---|---|
| 0 | attack |
| 1 | investigation |
| 2 | mount |
| 3 | other |

Each session's `metadata.vocab` also holds this mapping. The loader should read it
from there and check it matches.

## Frame rate

**30 fps.** Not in the readme or metadata. Source: CalMS21 paper, Section 4.1:
"Videos are acquired at 30 fps and 1024x570 pixel resolution."
So `time_s = frame / 30`.

## License and citation

License: **Creative Commons Attribution 4.0 International** (record page).
Cite: Sun et al., "The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions",
arXiv:2104.02710 (2021). The record page asks for this paper and gives BibTeX key `calms21`.

## Readme guidance we follow

- Develop on train only; hold out validation from train.
- Report test once, across several random seeds.
