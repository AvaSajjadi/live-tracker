# Interfaces between team parts

These are the formats each pair hands to the next one. If you need to change
one, tell the team first, because the other pairs build against it.

Times are in seconds from the start of the session. Positions are in pixels.

## 1. Pose data (input to everything)

Right now the dashboard reads the starter's **wide** CSV: one row per frame,
columns `x_a, y_a, x_b, y_b` (one center point per mouse, pixels, NaN if missing).
`formats.wide_to_long()` turns it into the **long** table below, which
`quality.py` and `filters.py` use. The CalMS21 loader will also output the long table.

Long table, one row per session × mouse × keypoint × frame
(`keypoint` is optional while we only have one center point per mouse):

| column | type | notes |
|---|---|---|
| session | str | recording ID |
| mouse | str or int | which animal (e.g. resident / intruder, still to be confirmed from the CalMS21 readme) |
| keypoint | str | keypoint name (names and order still to be confirmed from the readme) |
| frame | int | frame index |
| time_s | float | frame / fps (fps still to be confirmed) |
| x, y | float | pixels; NaN if missing |

After `quality.flag_tracking()` it also has: `missing, out_of_frame, jump, frozen, valid`
and after `quality.interpolate_short_gaps()`: `imputed`.

## 2. Data & model → Agent: per-frame scores

One row per session × frame (CSV or DataFrame):

```json
{"session": "s01", "frame": 120, "time_s": 4.0,
 "attack": 0.82, "mount": 0.05, "investigation": 0.10,
 "tracking_ok": true}
```

- Scores are classifier probabilities between 0 and 1.
- `tracking_ok` is false if the frame's tracking quality is too low.
- Test-set labels are never in this table.

## 3. Researcher → Agent: rules

See `src/live_tracker/rules.py`. Example:

```json
{"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0,
 "alert_threshold": 0.8, "ask_threshold": 0.5, "alert_mode": "alert",
 "description": "Alert me if the mice attack for at least 2 seconds"}
```

Every rule starts as `"status": "proposed"` and only becomes active after the
researcher clicks Confirm.

## 4. Agent → Interface: event log

One entry per event (JSON lines file, one object per line):

```json
{"event_id": "E7", "session": "s01", "rule_id": "R1", "behavior": "attack",
 "start_s": 4.0, "end_s": 6.5, "confidence": 0.82,
 "action": "alert", "researcher_answer": null,
 "evidence": {"mean_score": 0.82, "min_score": 0.71, "frames": 75}}
```

- `action` is one of `alert`, `ask`, `log`, `tracking_warning`.
- `researcher_answer` is filled in only for `ask` events (`"confirmed"` / `"rejected"`).
- The end-of-session summary is written **only** from this log.

## 5. Motion engine → Classifier: feature table

Built by `features.py` (owner: Ava). These are motion **measurements**, not
behavior labels: speed or closeness alone does not mean "attack".

Everything is **causal**: the value at time t uses only frames at or before t
(a test checks this), so the same table is valid for live replay. Columns
ending in `_w` are computed over a **trailing** window of `window_s` seconds
(`FeatureParams.window_s`, placeholder 1.0 s). They are NaN until the window has
at least `min_coverage` (placeholder 90 %) of good tracking, e.g. at the start
of a session or right after a gap.

Missing or invalid tracking gives **NaN**, never 0 (missing is not standing still).

One row per session × frame. Per mouse, with suffix `_<mouse>` (`_a`, `_b` for
the synthetic data; real CalMS21 mouse IDs once H1a confirms them):

| column | unit | meaning | status |
|---|---|---|---|
| `point_ok` | bool | this mouse's point is usable (valid, or short-gap imputed) | Day 1 |
| `uses_imputed` | bool | value was computed from at least one imputed point | Day 1 |
| `speed` | px/s | backward difference / real dt; NaN across gaps, session starts | Day 1 |
| `accel` | px/s² | change of velocity vector / dt | Day 1 |
| `heading` | rad | direction of movement, (-π, π]; NaN when nearly still (noise) | Day 1 |
| `turn_rate` | rad/s | heading change / dt, correct across ±π | Day 1 |
| `path_len_w` | px | distance travelled in the window | Day 1 |
| `straightness_w` | 0–1 | net displacement / path length; NaN if path < `min_path_px` | Day 1 |
| `idle_frac_w` | 0–1 | fraction of the window with speed < `idle_speed_px_s` | Day 1 |
| `in_pause` | bool/NA | still for ≥ `pause_min_duration_s`; True only once that time has passed | Day 1 |

Dyadic (one column per pair; `_ab` means A relative to B):

| column | unit | meaning | status |
|---|---|---|---|
| `dist_center`, `dist_rate` | px, px/s | body-center distance and its rate (negative = approaching) | done |
| `rel_speed` | px/s | speed of A relative to B | done |
| `dist_nose_nose`, `dist_nose_tail_ab`, `dist_nose_tail_ba` | px | keypoint distances (nose of A to nose of B; nose of A to tail base of B; nose of B to tail base of A) | opt-in: `keypoint_features()`, not in this table |
| `facing_ab`, `facing_ba` | rad, 0 to π | angle between the mouse's body axis (tail base → nose) and the direction from its body center to the other's; 0 = pointing at it, π = away. NaN if the axis is shorter than `min_axis_px` (default 5 px) | opt-in: `keypoint_features()`, not in this table |
| `close_time_s` | s | seconds of the trailing window with `dist_center` < `close_dist_px` | done |
| `speed_corr_w` | -1–1 | trailing correlation of the two speeds (NaN if one is constant) | done |

The keypoint rows are produced by `features.keypoint_features(long_df, torso_keypoints=...)`, a separate causal
function with the columns `dist_nose_nose, dist_nose_tail_ab, dist_nose_tail_ba, facing_ab, facing_ba` (one row per
session × frame). They are **not** in the `compute_features()` table, so the classifier's input (model B) is unchanged;
nothing has yet tested whether they help.

Plus `session, frame, time_s, tracking_ok`. `tracking_ok` comes from
`frame_tracking_ok()`: false if any needed point of either mouse is missing,
frozen, jumping or out of frame in that frame (short-gap imputed points count as ok).

Body center (`body_center()`): mean of the usable torso keypoints, which must be
named explicitly (`torso_keypoints=`, from the CalMS21 readme, not guessed); the
rest are used if some are missing, NaN if all are. Never the nose alone.

All thresholds are placeholders until tuned on train/val sessions (Day 3).

Usage:

```python
from live_tracker.features import compute_features, tracking_warning_events
table = compute_features(long_df, torso_keypoints=TORSO)   # §5 table, one row per session × frame
warnings = tracking_warning_events(flags_df, keypoints=TORSO)  # §4 event-log dicts
```

`tracking_warning_events()` returns event-log entries (§4) with
`action: "tracking_warning"`, IDs `T1, T2, ...`, `rule_id`/`behavior`/`confidence`
set to null, and `evidence = {mouse, flag, duration_s, message}`.
