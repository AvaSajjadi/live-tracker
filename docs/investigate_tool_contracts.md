# Investigation tools: draft contracts (v0, for Syed to review)

Status: PROPOSAL. Nothing here is implemented. Once agreed, freeze it; changes need both of us.

## Conventions (all tools)

- `session`: string, exactly the `session` column of INTERFACES.md §2/§5 (CalMS21 ids; synthetic ones start `SYNTHETIC_`).
- Times: float seconds since session start (observation time, same as `time_s`). Windows are half-open `[t0_s, t1_s)`. Field names always end `_s`.
- Every call returns a JSON object with `status` in `"ok" | "insufficient_evidence" | "error"`. The agent branches on `status` and `reason`, never on prose.
- `data_source`: `"synthetic" | "ground_truth_labels" | "classifier"` is echoed on every ok response so the report can label it.
- Errors: `{"status": "error", "code": <below>, "message": str}`. Codes: `unknown_session`, `bad_window` (t1 <= t0 or t0 < 0 or past session end), `unknown_feature`, `unknown_behavior`, `no_scores`, `budget_exhausted`.
- Insufficient evidence: `{"status": "insufficient_evidence", "reason": "too_few_bouts" | "too_few_usable_windows" | "window_unreliable", "n_available": int, "n_required": int}`.
- Tools are read-only and deterministic: same input, same output. No tool touches the researcher's active rules or journal run_ids.

## 1. query_events (Syed's monitor, wrapped)

In: `session` (str or list of str), `behavior` (attack|mount|investigation), `min_duration_s` (float), `threshold` (float, default 0.5).
Runs a throwaway rule under run_id `inv_<hash of args>`.
Out ok: `bouts`: list of
`{event_id, session, behavior, start_s, fired_s, end_s, end_reason, confidence, bout_mean_score}`.
`start_s` is the bout start and is the ONLY time anchor for pre-bout analysis (not `fired_s`, which already includes `min_duration_s` of the bout).
Plus `n_bouts`, `data_source`.

## 2. get_window_features (my features table, sliced)

In: `session`, `t0_s`, `t1_s`, `cols` (subset of FEATURE_COLS + DYADIC_COLS, e.g. `dist_center`, `dist_rate`, `close_time_s`, `rel_speed`).
Out ok: `{session, t0_s, t1_s, n_frames, n_usable_frames, summary: {<col>: {mean, min, max, n}}}`. Summaries use only `tracking_ok` frames. Raw per-frame rows are NOT returned (keeps the agent's context small); add `include_rows` later if needed.

## 3. check_tracking_quality (my flags)

In: `session`, `t0_s`, `t1_s`.
Out ok: `{session, t0_s, t1_s, usable_fraction, flags_fired: [str], verdict: "reliable" | "unreliable", reason}`.
Verdict rule (fixed, not tuned per run): `reliable` iff `usable_fraction >= 0.8` and no `frozen`/`out_of_frame` run longer than 0.5 s inside the window. Thresholds live in one constant, logged in the report.

## 4. compare_windows (my tools 2+3 composed)

In: `bouts` (list of `{session, start_s}` from tool 1), `pre_s` (float, window length before `start_s`), `baseline_s` (float), `baseline_gap_s` (float, baseline ends this long before the pre window), `feature` (str), `min_n` (int, default 10).
Windows: pre = `[start_s - pre_s, start_s)`, baseline = `[start_s - pre_s - baseline_gap_s - baseline_s, start_s - pre_s - baseline_gap_s)`. A bout is used only if BOTH windows are `reliable`, and the pre window does not overlap another bout of the same behavior.
Out ok: `{feature, n_input, n_used, n_excluded, excluded: [{session, start_s, why}], pre_mean, baseline_mean, mean_diff, effect_size_dz, ci95: [lo, hi], association_only: true}`.
Out insufficient: if `n_used < min_n`, `status: "insufficient_evidence"`, `reason: "too_few_usable_windows"`, with the same counts.
`association_only` is always true. The report template says "precedes / is associated with", never "causes" or "leads to".

## 5. propose_rule (rules.py gate)

In: a rule dict as in rules.py.
Out ok: `{rule_id, status: "proposed"}`. Does nothing until the researcher approves through the existing approve step. The agent can never approve its own rule.

## Agent budget (Syed's loop enforces; tools only report)

- Max 12 tool calls per investigation.
- Max 2 follow-up analyses after the first `compare_windows` (e.g. widen `pre_s`), and the follow-up parameters are fixed in advance: `pre_s` in {2, 5, 10}. Every attempt is reported, including the failed ones, so nothing is cherry-picked.
- Every call and result is journaled with the investigation id.

## Data hygiene

- Development: synthetic scores (labelled) and ground-truth labels (labelled).
- Strategy choices (pre_s grid, min_n, quality thresholds) are fixed on the train/validation split only.
- The test split is touched once, for the final run, with Igor's real classifier scores. The agent is never given test labels.
