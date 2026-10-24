# Contract v1 integration (Stage 9 backend + Ava's investigation tool contract)

**Status:** implementation for review. Synthetic data only; nothing in this work has been run on CalMS21 data, a
cache or Igor's score archives. Contract: `docs/investigate_tool_contracts.md` on Ava's PR #17 (contract v1, option
C).

Option C means the existing Stage 9 backend is the base and is not rewritten:
- provenance and score-drop verification;
- frozen journals;
- evidence ids and claims;
- the bounded agent loop;
- the deterministic emulator.

The contract's shared surface is added as narrow adapters (`src/live_tracker/investigation_tools_v1.py`) and one
conversion module (`src/live_tracker/timebase.py`). The S9-7 exploratory extension (merge-gap and window choices made
after seeing counts) is **not** included.

## 1. The six shared-surface items

| # | Contract v1 | Where | Notes |
|---|---|---|---|
| 1 | Anchor is bout `start_s`, never `fired_s`; `start_s`, `fired_s`, `end_s` are separate | `query_events` returns all three; `compare_windows` builds windows from `start_s` only | Test: moving every `fired_s` leaves `compare_windows` unchanged. |
| 2 | Seconds at the boundary, half-open `[t0_s, t1_s)`, frames inside, written 30 fps rule | `timebase.py`, `docs/timebase_30fps.md` | Ava's final rule: `frame(t) = floor(30·t + 0.5)` (round half up) at both boundaries. |
| 3 | `status` ∈ `ok`, `insufficient_evidence`, `error`, fixed codes | every v1 tool | Contract codes exactly, plus two marked additions (pending Ava): `invalid_request` and `unverified_bouts`. |
| 4 | `association_only` always `true` | every v1 response; findings/2 report documents (`association_only: true`, rendered, tamper-refused) | The legacy/1 renderer (`investigation_report.py`) is not changed. |
| 5 | One researcher approval gate; the agent never approves its own rule | unchanged Rules page (PR #16) | v1 tools never touch a `RuleSet`. `propose_rule` is deferred, and the runner refuses it. |
| 6 | `n_sessions`, `n_bouts`, `n_windows` separate; no CI until a session-aware method is agreed | `query_events`, `compare_windows` | No `ci95` and no `effect_size_dz` anywhere in v1 output. |

## 2. The four tools

Each tool is called as `tool(data, request)` and returns one JSON-ready dict. Every response carries:
- `status`, `association_only: true` and `contract_version`;
- `data_source` and `derived_from` on `ok` and `insufficient_evidence` responses.

Errors have the form `{"status": "error", "code", "message"}`.

### `query_events`: the monitor, wrapped

**Input:** `session` (one id or a list), `behavior` (attack, mount, investigation), `min_duration_s` (> 0) and
`threshold` (default 0.5).

**How it runs:** `monitor.run_monitor` with an internal query rule. The rule:
- has mode `log` and `ask_threshold = alert_threshold = threshold`;
- runs under `run_id = "inv_<sha256 of args and data>"` with an **in-memory** journal;
- is never added to a `RuleSet` and never written to disk.

**Output:**
- `bouts`: `{event_id, session, behavior, start_s, fired_s, end_s, end_reason, confidence, bout_mean_score}`;
- `n_bouts`, `n_sessions` (sessions with a bout) and `n_sessions_searched`;
- `latency_gap_s: null` and `start_meaning`.

**What `start_s` means:** the first frame whose score is at or above the threshold with `tracking_ok`. This follows
the monitor's evidence rule. For classifier scores it is a **threshold crossing, not the annotated onset**: no
score-to-event latency gap is applied (the gap is open, see §6).

**Errors:** `unknown_session`, `unknown_behavior`, `no_scores` (no finite score for the behavior) and
`invalid_request`.

### `get_window_features`: Ava's pair features, sliced

**Input:** `session`, `t0_s`, `t1_s`, `cols`.

**Output:** `n_frames`, `n_usable_frames` (`tracking_ok` frames) and `summary: {col: {mean, min, max, n}}`. No
per-frame rows.

**Columns:** the five pair features, `DYADIC_COLS`, computed by Ava's unchanged `dyadic_features` through the Stage 9
pipeline. Per-mouse `FEATURE_COLS` need a mouse argument the contract does not have, so they return
`unknown_feature` with that explanation (deferred).

**Causal computation:** values come from the session **truncated at `t1_s`** (`investigation_windows.prefix_arrays`).
Frames after the window cannot change them; this is tested.

**Which values count:** a value is used when it is finite and `tracking_ok` holds on frames t and t−1. This is the
existing Stage 9 rule, because the rate features use frame t−1. The contract says "only tracking_ok frames"; this is
slightly stricter (see §6).

### `check_tracking_quality`: Ava's flags, judged by a named policy

**Input:** `session`, `t0_s`, `t1_s`, `quality_policy` (default `quality_v1`).

**Output:**
- `usable_fraction`, `flags_fired` and `verdict` (`reliable` or `unreliable`);
- `reason` (`usable_fraction_below_min` or `frozen_or_out_of_frame_run`) and `reasons`;
- `longest_veto_run_s`, `quality_policy` and `quality_policy_definition`.

**The two policies:**

| policy | rule |
|---|---|
| `quality_v1` (default, Ava's candidate A) | `usable_fraction >= 0.8` (compared exactly) and no run of frames > 0.5 s (more than 15 frames) in which any torso keypoint of either mouse is `frozen` or `out_of_frame`. A run of exactly 0.5 s passes, as in `compare_quality_policies.py` (`longest_run / 30 > 0.5`). |
| `quality_v1_strict90` (candidate B, the named alternative, never the default) | `usable_fraction >= 0.9`. |

**Causal flags:** the flags come from the session truncated at the window end. Ava's frozen flag looks ahead; a
frozen run that continues past `t1_s` is judged only on the frames before `t1_s`. This is tested.

### `compare_windows`: tools 2 and 3 composed

**Input:**
- `query_run_id`, `query` and `bouts`: the `query_events` result's id, its canonical parameters and its bouts,
  **unchanged**;
- `pre_s`, `baseline_s`, `baseline_gap_s` and `feature`;
- optional `min_n_windows` (10), `min_n_sessions` (4), `gate_override_reason` and `quality_policy`.

**Bout provenance.** Nothing the caller sends is trusted as proof, including event ids:
- the run id is recomputed from `query` and this data source's fingerprint (its provenance and every score row), so
  altered parameters, another source or changed data give another id;
- the query is re-run through the same monitor code (deterministic, in memory, no second detector);
- `bouts` must equal that result exactly: every field of every bout, in order, with nothing dropped, added or
  duplicated.

Anything else is `error` with code `unverified_bouts`. Only the recomputed bouts are used, and the overlap policy
checks against all of them.

**Windows.** The absolute endpoints are formed in seconds first, and each one is converted with
`frame(t) = floor(30·t + 0.5)`:
- pre = `[frame(start_s − pre_s), frame(start_s))`;
- baseline = `[frame(start_s − pre_s − gap − baseline), frame(start_s − pre_s − gap))`.

Rounded duration counts are never subtracted; see `docs/timebase_30fps.md`. A window whose converted endpoints are
equal is `bad_window`. Each window is measured from the session truncated at its own end.

**Overlap policy `overlap_v1` (finalized, approved by Ava).** "Another bout" is another **fired** bout of the cited
query: a monitor event of the same session and behavior that lasted at least `min_duration_s`, taken from the
complete, unchanged, provenance-verified query result. A threshold crossing that never fired (shorter than
`min_duration_s`, or below the threshold) is not a bout and excludes nothing. **Both** windows are checked: a pre
window or a baseline window that shares a frame with another fired bout (half-open frame ranges) is excluded. This is
variant V1 of `docs/contract_v1_overlap_options.md`. The policy is named and versioned like the quality policy
(`OVERLAP_POLICIES`); the caller cannot choose another one (an `overlap_policy` field is `invalid_request`).

**Exclusions.** A bout is excluded for the first reason that applies, in this order (the order of Ava's dry run,
`scripts/dryrun_contract_options.py`; the order decides which reason is reported, never which bouts are used). Each
exclusion is listed as `{session, start_s, why}`, and every result carries the order as `exclusion_order`.

| why | when |
|---|---|
| `pre_window_overlaps_bout` | another fired bout of the cited query (same session and behavior) intersects the pre window |
| `window_before_session_start` | the baseline would start before frame 0 |
| `pre_window_unreliable` | the pre window fails the quality policy |
| `baseline_window_unreliable` | the baseline window fails the quality policy |
| `baseline_window_overlaps_bout` | another fired bout of the cited query intersects the baseline window (`overlap_v1`) |
| `no_feature_values` | either window has no usable value of the feature |

**Traceability.** Every `compare_windows` result (`ok` and `insufficient_evidence`) carries `overlap_policy`,
`overlap_policy_definition` and `exclusion_order`. The Replay bridge records the policy and its definition in its
configuration snapshot (`inputs.config`, part of the exported JSON and of the configuration SHA-256 that keys its
cache), and the Investigation page shows the policy and the exclusion order next to the quality policy. The bridge
refuses (`PolicyMismatch`) a configuration snapshot that is not exactly the configuration the tools execute now for the
quality policy it selects: preset, `threshold_from`, `contract_version`, quality policy name and definition (any
registered policy, checked against its own definition), overlap policy name, definition and exclusion order, evidence
gate and tool-call budget, with nothing missing or extra. This is checked before any tool call and again before the
result is returned, and every executed response must report the same contract version, threshold, preset, quality and
overlap policies and gate (`compare_windows` echoes `quality_policy_definition` for this). The executed query is bound to
the approved rule: its canonical `query` (session, behavior, `min_duration_s`, threshold = the rule's ask threshold)
and every duplicate the response carries, its run id and its complete bout set are recomputed from the rule and the
data source and must match exactly, before the query is used; the comparison must cite exactly that query and those
bouts. Nothing is returned, cached
or exported on a mismatch; the Replay and Investigation pages show the refusal. A tool refusal stays a refusal and is
given no policy metadata.

**Counts:**
- `n_input`, `n_used` and `n_excluded`;
- `n_sessions`: sessions with at least one used bout, i.e. paired sessions;
- `n_bouts` (used bouts) and `n_windows` (used pre windows: one per used bout in v1);
- `n_baseline_windows` and `n_sessions_input`.

**Aggregation:**
- `mean_diff` is the **median across paired sessions of the per-session paired differences**, as the contract says.
  It is not a mean over windows or over sessions: the field name is historical and kept for compatibility, and
  `mean_diff_definition` in every `ok` result says so;
- a session's paired difference (`per_session[].paired_diff`) is the mean of its used bouts' `pre − baseline`;
- `per_session` lists each session's `n_bouts` and `paired_diff`;
- `pre_mean` and `baseline_mean` are plain means over used windows, reported for orientation only.

**Evidence gate:**

| condition | result |
|---|---|
| `n_input < min_n_windows` | `insufficient_evidence`, reason `too_few_bouts` |
| `n_windows < min_n_windows`, or `n_sessions < min_n_sessions` | `insufficient_evidence`, reason `too_few_usable_windows`, with `failed_gate` |

An insufficient result carries every count and every exclusion, but **no effect value**. Lowering the gate below
10 windows / 4 sessions needs a written `gate_override_reason`, which is echoed in `evidence_gate`.

## 3. Budget and attempts (`InvestigationRunner`)

The limits are single constants in `investigation_contracts.py` (`CONTRACT_V1_MAX_TOOL_CALLS = 12`,
`CONTRACT_V1_MAX_FOLLOW_UPS = 2`, `CONTRACT_V1_PRE_S_GRID = (1, 2, 5)` s). The runner and the agent loop both import
them, so they cannot drift apart. In both, a comparison counts toward the follow-up limit once it ran (status other
than `error`).

The runner enforces the contract's budget:
- **At most 12 tool calls, every attempt counted.** An attempt is recorded and counted *before* the tool runs, so
  malformed and failing calls use budget too. A 13th call returns `error` with `budget_exhausted`; it is recorded but
  not counted. Malformed values (wrong types, NaN, infinity, integers too large for a float) get a structured error,
  never an exception. An unexpected exception (a programming error) is recorded as status `exception` and re-raised,
  not hidden.
- **Follow-ups are limited.** `compare_windows` takes `pre_s` only from the preselected grid {1, 2, 5} s. At most 2
  follow-ups are allowed after the first. A follow-up may change only `pre_s` and never repeats a value already
  analysed.
- **Every attempt is recorded.** Refused and failed calls are included, each with the investigation id, the request,
  the full result and the result's SHA-256.

**The Stage 9 agent loop** (`investigation_agent`) enforces the same policy:
- At most 12 executed tool calls. The default changed from 16 to 12, and a larger value is refused.
- At most 2 `compare_windows` analyses after the first, in every run. A further one is refused before execution
  (`follow_up_limit`), and the refusal and its reason are traced.
- The pre-window grid of 30, 60 or 150 frames (1, 2 or 5 s) applies with `CONTRACT_V1_BUDGET` (refusal kind
  `pre_window_not_in_grid`). It is not on by default: the legacy and synthetic workflows use other window lengths, and
  every existing scripted run already stayed within three comparisons.

One accounting difference is deliberate and visible. The loop counts executed calls toward 12, and it bounds refused
calls separately (3 consecutive refusals, 2 identical repeats stop the run). The runner counts every attempt. Both keep
executed calls at 12 or fewer.

The LLM loop still calls the Stage 9 internal tools (episode and window ids, the same backend). Exposing the v1
seconds-based JSON surface to the model is **not wired yet** (§7).

## 4. Data sources

`InvestigationData` holds, per session, a §2 score table and the pose pipeline, aligned frame for frame. There are
three constructors:

| constructor | data_source | derived_from |
|---|---|---|
| `from_synthetic(scores, poses)` | `synthetic` (only `SYNTHETIC_` sessions accepted) | `synthetic_scores` |
| `from_classifier(drop, annotation_source, policy)` | `classifier`; `synthetic` if any session is synthetic | `classifier_probabilities` |
| `from_annotations(annotation_source)` | `ground_truth_labels`; `synthetic` if synthetic | `annotation_indicators` (0/1 per frame) |

**Classifier source:**
- It takes an **issued, verified** ScoreDrop (`score_drops.trusted_rows_sha256`); a hand-built drop is refused.
- Its sessions must be exactly the policy's validation split, so held-out test sessions are never loaded.
- Pose comes from the verified validation cache without reading annotations.

**Annotation source:** label bouts are not split by tracking loss. This follows Ava's `label_bouts`. Tracking is
judged by `check_tracking_quality`.

**Labelling is enforced:** `SYNTHETIC_` sessions can only be labelled `synthetic`. Both real constructors are tested
only on the existing Stage 9 synthetic cache and an issued synthetic score drop.

## 5. Dependencies and Windows (Ava's second request)

**`httpx2` is a real distribution** (pydantic/httpx2, 2.14.0 installed here). The deterministic emulator imports it
directly (`httpx2.MockTransport`), and `anthropic` 1.x itself requires `httpx2>=2.0,<3` (checked in the installed
metadata).

**Packaging:** `pyproject.toml` declares a new extra, `investigation = ["anthropic>=1.15,<2", "httpx2>=2.0,<3"]`. 1.15
is the lowest anthropic version verified here. CI installs `.[investigation]`. The base install (`requirements.txt`)
is unchanged, so the Streamlit rules and replay pages still install without the SDK.

**Without the extra:** the 6 tests in `test_stage9_integration.py` that run the emulator to completion are marked
`needs_sdk` and skip instead of failing. The other 4 in that file, and the other SDK tests, already passed or skipped
without it. Checked by blocking `anthropic`, `httpx2`, `httpx` and `sniffio` at import time: 191 passed, 6 skipped.
A real fresh install was not possible offline.

**Windows:** 7 POSIX-only tests are marked `posix_only` (skipped only on `win32`): 5 in `test_score_drops.py` and 2 in
`test_investigation_data.py`. They cover filesystem symlinks and the `0o700`/`0o600` permission bits; macOS and Linux
coverage is unchanged. Zip "symlink member" tests are portable and stay.

One test gate in `test_investigation_data.py` now checks the local cache before the `/Volumes/...` source drive. This
way the external drive path is never looked up unless the local cache exists.

## 6. Decisions and open items

### Finalized (approved by Ava)

- **Pre-window overlap:** only the other **fired** bouts of the cited query count; sub-threshold or too-short runs are
  not bouts (former item 3).
- **Baseline overlap:** a baseline window that contains another fired bout is excluded, with its own reason
  `baseline_window_overlaps_bout`, under the named policy `overlap_v1` (former item 4).
- The evidence gate (10 windows, 4 sessions), the pre_s grid and `quality_v1` are unchanged; nothing relaxes the gate
  when baseline overlap removes windows (an honest `insufficient_evidence` instead).

### Historical: the questions as they were asked (now answered above)

Items 3 and 4 below are kept as they were written before the decision. The dry run that informed it is
`docs/contract_v1_overlap_options.md`.

### For Ava (still open unless marked answered)

1. **The 30 fps rule** `floor(30·t + 0.5)` is applied to every absolute window endpoint, including endpoints formed
   as `start_s − pre_s − …` in seconds. Durations are not rounded on their own. Nothing is open here unless you want
   durations restricted to whole frames.
2. **Two added error codes:** `invalid_request` (malformed requests) and `unverified_bouts` (`compare_windows` bouts
   that are not exactly the bouts of the cited query on this data source).
3. *(Answered: fired bouts only.)* **"Overlaps another bout"** is implemented as overlap with another bout of the
   same behavior *of the cited query*: fired events, so runs shorter than `min_duration_s` do not count. Your label
   dry run counts every annotated bout, including short ones. Which do you want?
4. *(Answered: baseline windows are excluded too, `overlap_v1`.)* **Baseline overlap:** following the contract's
   wording, only the pre window was checked for overlap, and a baseline window that contained a bout was not
   excluded; the internal Stage 9 tool does exclude it. Confirm or extend.
5. **`compare_windows` input:** it needs `query_run_id`, `query` and the complete, unchanged bouts of that query (to
   bind them to authentic evidence). The contract lists only `bouts: [{session, start_s}]`. A subset of bouts is
   refused: to compare fewer sessions, query fewer sessions.
6. **Usable values:** "tracking_ok on frames t and t−1" is used for feature values, rather than the contract's
   "tracking_ok frames".
7. **Differences from your dry-run scripts that will change counts.**
   - Stage 9 never interpolates. `compare_quality_policies.py` applies `interpolate_short_gaps`, which is two-sided,
     before computing `tracking_ok`.
   - The `jump` flag is inactive (`max_speed_px_s=None`). Your tuned `MAX_SPEED_PX_S = 3000` and One-Euro
     `beta = 0.05` (`params.py`, PR #17) are "not wired into defaults yet", and this work does not change your
     defaults.
8. **`n_windows`:** it counts used pre windows (one per used bout). Should baselines be counted into it?
9. **Per-mouse features** (`FEATURE_COLS`): need a `mouse` argument.

### For Igor

1. **Final event threshold and minimum duration.** The tools take them as arguments; nothing is frozen.
2. **The score-to-event latency gap.** It is not applied. The proposed 1.0 s applies only to threshold 0.5 with no
   minimum duration, so it is not used as a general value.
3. **Whether the final feature configuration changes.**
4. **Event-level evaluation through `monitor.run_monitor`.** This is Igor-owned evaluation code; `query_events` shows
   the call.
5. **Authorization of the one-time held-out evaluation.** No code here reads test sessions.

### Real data needed later (not touched now)

- The verified validation cache: `data/raw/calms21_validation/`, built from the CalMS21 train JSON by
  `investigation_data.py`.
- Igor's validation score member: `scores_validation_model_b.csv` in his archive.

They are needed for the first real `query_events` / `compare_windows` run and for the real-data tests that currently
skip.

## 7. Deferred

- `propose_rule`.
- CI, `effect_size_dz` and per-bout baselines.
- Per-mouse features.
- Exposing the v1 JSON surface to the LLM loop. That needs tool schemas, claim grounding and report templates for v1
  results, and is a separate reviewed step.
- The S9-7 exploratory extension.

## 8. Demo

`python scripts/contract_v1_demo.py` runs the four tools through one `InvestigationRunner` on five synthetic sessions:
- `query_events`, `check_tracking_quality` and `get_window_features`;
- `compare_windows` at `pre_s` 2 s, then the two allowed follow-ups (1 s and 5 s);
- one refused third follow-up.

It is deterministic and offline, and it uses no API key. The numbers are synthetic and say nothing about behavior.
