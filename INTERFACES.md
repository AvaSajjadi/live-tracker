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
