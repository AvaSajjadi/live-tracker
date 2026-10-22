# Demo attack rule v1 (chosen on validation)

**Rule:** `ask_threshold` 0.6, `min_duration_s` 0.5, `alert_threshold` 0.8, `alert_mode` "alert".
**Code constant:** `evaluate.ATTACK_RULE_V1`. **Reproduce:** `python scripts/tune_attack_rule.py`.

## Data and rule semantics
- **Scores:** model B validation scores, `outputs/scores_validation_model_b.csv` (sha256 `ff00f6fe31d9348d…`).
  - Model B is not retrained. It uses default `FeatureParams` and `SUGGESTED_MIN_SCORE` 0.15.
- **Sessions:** the 14 validation sessions (94,305 frames, 52.4 min). Attack is labeled in 4 of them (56 bouts).
- **Test split:** not read.
- **Events:** produced by `monitor.run_monitor`, unchanged.
  - A frame is evidence when `tracking_ok` is true and score ≥ `ask_threshold`.
  - An event fires once the bout has lasted `min_duration_s`.
  - A `tracking_ok` = false frame ends the bout.
- **Metrics:** `evaluate.monitor_event_metrics`.
  - An event and a true bout match if they share a frame.
  - Precision = matched events / events.
  - Recall = detected true bouts / true bouts.
  - Start offset = event `first_frame` (= `start_s`) − true onset. Positive = late.
  - Fire offset = `fired_frame` − true onset (when the alert appears).

## How it was chosen
- **Grid:** `ask_threshold` in {0.3, 0.4, 0.5, 0.6, 0.7} × `min_duration_s` in {0.2, 0.5, 1.0, 1.5, 2.0}.
- **Criterion:** highest attack event F1.
- **Tie-break:** the top four settings have the same F1 (0.722 to 0.724, noise for 56 bouts). Among settings within 0.01 F1 of the best, the one with the fewest false alerts per hour wins.
  - This tie-break was added after seeing the grid. The plain F1 maximum (0.5 / 0.5, F1 0.724) is an equally valid choice with more alerts.

| ask | min s | events | precision | recall | F1 | false alerts/h | fire offset median |
|---|---|---|---|---|---|---|---|
| 0.4 | 0.5 | 132 | 0.71 | 0.73 | 0.722 | 43.5 | +0.33 |
| 0.5 | 0.5 | 103 | 0.78 | 0.68 | 0.724 | 26.3 | +0.40 |
| **0.6** | **0.5** | **80** | **0.83** | **0.64** | **0.723** | **16.0** | **+0.35** |
| 0.7 | 0.2 | 146 | 0.75 | 0.70 | 0.724 | 41.2 | +0.20 |
| 0.5 | 2.0 | 7 | 1.00 | 0.18 | 0.303 | 0.0 | +2.10 |

The full grid is written to `outputs/attack_rule_grid.csv` (git-ignored).

**Why not 2.0 s (the INTERFACES.md example):** with `tracking_ok` ending bouts, only 10 of 56 bouts fire (7 events, 2 of 4 sessions).
- `tracking_ok` is false on 3.1% of attack frames, against 1.1% of all frames.
- So long bouts are often split before they reach 2 s.

## Result for the chosen rule (validation)
- **Events:** 80 events, 66 on a true attack bout (precision 0.83). 16.0 false alerts per hour.
- **Recall:** 36 of 56 true bouts detected (0.64), in all 4 attack sessions.
  - Most missed bouts are short: 13 of the 20 are 1.07 s or shorter.
- **Start offset:** median −0.15 s (P5 −1.77, P95 +1.27, max +2.27).
  - 33 of 36 detected bouts start within 1.0 s after the true onset (or earlier).
- **Fire offset:** median +0.35 s (P95 +1.77, max +2.77).

| session (validation) | true bouts | detected | median start offset | median fire offset |
|---|---|---|---|---|
| mouse005 | 7 | 2 | +0.03 | +0.53 |
| mouse028 | 6 | 1 | +2.27 | +2.77 |
| mouse045 | 20 | 12 | −0.20 | +0.30 |
| mouse064 | 23 | 21 | −0.17 | +0.33 |

**`alert_threshold` 0.8 (the `rules.Rule` default) is a weak choice.** Event confidence (mean score at firing) barely separates true from false events:
- events with confidence ≥ 0.8: 69, precision 0.84
- events below 0.8 (asks): 11, precision 0.73
- The scores are not calibrated.

## Caveats
- **Chosen and measured on the same 14 sessions,** so these numbers are optimistic. Only 4 sessions contain attack, and 2 of them hold 43 of the 56 bouts.
- **The test split is evaluated once,** with this rule frozen, under the team's test policy.
- **Ava's contract dry run** (`scripts/dryrun_contract_options.py`) depends on this threshold and minimum duration and should be re-run with them.
