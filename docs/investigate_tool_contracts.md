# Investigation tools: contract v1 (shared surface, for freeze)

Status: v1 DRAFT after the option-C (hybrid) decision. Syed's backend (provenance, frozen journals, claims, bounded loop) is the base and is NOT rewritten to match this doc. Only the **Shared surface** below is frozen; everything else is advisory or deferred. Changes to the shared surface need both of us.

## Shared surface (frozen once both approve)

1. Time anchor is bout `start_s`, never `fired_s`. `start_s`, `fired_s`, `end_s` are separate fields.
2. Seconds at the tool boundary, half-open `[t0_s, t1_s)`. Frames stay canonical inside; the 30 fps seconds-to-frame rounding rule is written down and unit-tested (open item: Syed to state the rule, Ava to review).
3. `status` in `ok | insufficient_evidence | error`, with fixed reason/error codes (below).
4. `association_only` is always `true`.
5. One researcher approval gate for rules; the agent never approves its own rule. `alert_mode="ask"` means each event waits for confirm/reject.
6. Report `n_sessions`, `n_bouts`, `n_windows` separately everywhere. No CI until a session-aware method is agreed.

## Scope

- First: four read-only tools (1-4 below). `propose_rule` (5) is a later proposal-only adapter. `ci95`, `effect_size_dz` and per-bout baselines are deferred/optional.
- S9-7 (merge-gap and window choices made after seeing counts) is exploratory: kept on record, labelled exploratory, excluded from the first PRs.

## Conventions (all tools)

- `session`: string, exactly the `session` column of INTERFACES.md §2/§5 (CalMS21 ids; synthetic ones start `SYNTHETIC_`).
- Times: float seconds since session start (observation time, same as `time_s`). Windows are half-open `[t0_s, t1_s)`. Field names always end `_s`. Frame conversion at 30 fps: see Shared surface item 2.
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
Verdict rule (fixed, not tuned per run, versioned by name): Ava owns the quality policy. Candidate A: `usable_fraction >= 0.8` and no `frozen`/`out_of_frame` run > 0.5 s. Candidate B (Syed's): `usable_fraction >= 0.9`. **Decision: `quality_v1` = candidate A**; B stays as the named alternative `quality_v1_strict90`. Evidence (`scripts/compare_quality_policies.py`, ground-truth attack bouts >= 1 s): A and B give identical counts on the 14 validation sessions and on pooled train+validation (pre_s 1/2/5 s: 33/21/14 usable on validation, 97/71/41 pooled); quality excludes 0 bouts, all exclusions are overlap or session edge. So the data cannot separate them; A is chosen because the frozen/out-of-frame veto guards a failure mode the fraction alone misses. Caveat: usable bouts sit in only 4 validation sessions (16 pooled), so the 4-paired-session gate is barely met on validation alone. Thresholds live in one constant and the version name is logged in the report.

## 4. compare_windows (my tools 2+3 composed)

In: `bouts` (list of `{session, start_s}` from tool 1), `pre_s` (float, window length before `start_s`), `baseline_s` (float), `baseline_gap_s` (float, baseline ends this long before the pre window), `feature` (str), `min_n_windows` (int, default 10), `min_n_sessions` (int, default 4).
Windows: pre = `[start_s - pre_s, start_s)`, baseline = `[start_s - pre_s - baseline_gap_s - baseline_s, start_s - pre_s - baseline_gap_s)`. A bout is used only if BOTH windows are `reliable`, and the pre window does not overlap another bout of the same behavior. **Finalized (overlap policy `overlap_v1`):** "another bout" means another fired bout of the cited query (sub-threshold or too-short runs are not bouts), and the baseline window is excluded too when it overlaps one (reason `baseline_window_overlaps_bout`); see `docs/contract_v1_integration.md`.
Out ok: `{feature, n_input, n_used, n_excluded, n_sessions, n_bouts, n_windows, excluded: [{session, start_s, why}], pre_mean, baseline_mean, mean_diff (median of per-session paired differences), association_only: true}`. Windows truncate at the window end. `effect_size_dz` and `ci95` are omitted until a session-aware method is approved.
Out insufficient: if `n_windows < min_n_windows` OR paired `n_sessions < min_n_sessions`, `status: "insufficient_evidence"`, `reason: "too_few_usable_windows"`, with the same counts.
`association_only` is always true. The report template says "precedes / is associated with", never "causes" or "leads to".

## 5. propose_rule (rules.py gate)

In: a rule dict as in rules.py.
Out ok: `{rule_id, status: "proposed"}`. Does nothing until the researcher approves through the existing approve step. The agent can never approve its own rule.

## Agent budget (Syed's loop enforces; tools only report)

- Max 12 tool calls per investigation.
- Latency gap between score and event is open: needs a justified value from Igor's score lag.
- Max 2 follow-up analyses after the first `compare_windows` (e.g. widen `pre_s`), and the follow-up parameters are fixed in advance: `pre_s` in {1, 2, 5} (frozen before anyone looks at effect values). Every attempt is reported, including the failed ones, so nothing is cherry-picked.
- Every call and result is journaled with the investigation id.

## Data hygiene

- Development: synthetic scores (labelled) and ground-truth labels (labelled).
- Strategy choices (pre_s grid, min_n, quality thresholds) are fixed on train + validation sessions only (pooled, because validation alone is small).
- Dry run on the 14 validation sessions with ground-truth labels (`scripts/dryrun_usable_windows.py`): of 40 attack bouts >= 1 s, usable pre-windows are 21 (2 s), 14 (5 s), 5 (10 s). Almost all exclusions are another attack inside the window, none are tracking quality. Hence the grid above has no 10 s. Real classifier bouts will differ, so re-run on Drop A.
- The test split is touched once, for the final run, with Igor's real classifier scores. The agent is never given test labels.


## PR order (Syed, small stacked PRs)

(a) score_drops + journal provenance, (b) sources/windows/compare, (c) claims/report, (d) integration script + docs. Base on main once #16 merges; until then stack on #16. Merge order of the others: #15 -> #18; #16 independent.

## Open items before freeze

- 30 fps rounding rule text + test (Syed).
- Latency gap value (Igor).
- Syed: declare `httpx2` in requirements/pyproject; skip POSIX-only tests (0o700, symlinks) on Windows with a marker.
- Demo uses a deterministic emulator, not a live LLM; reports are insufficient_evidence / not_authorized. Say so to the judges.
- Deadline: unmerged by Oct 23 is cut from the demo; Oct 25 submission is the hard stop.
