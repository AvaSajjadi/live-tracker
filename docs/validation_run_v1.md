# Phase 2: real validation run of `ATTACK_RULE_V1` (2026-10-22)

**Real data.** Igor's model B classifier probabilities on the 14 CalMS21 validation sessions, the verified
validation pose cache and the validation annotations (evaluation only). The test split was not read. Offline: no
network and no model call. Association only: nothing here is a causal claim.

Reproduce (the outputs are written to a new, git-ignored directory; nothing is overwritten):

    python scripts/validation_run_v1.py ~/Downloads/igor_scores_validation_model_b_repacked.zip \
        --archive-sha256 5c1a81e40ca12f00dea0ae511708b7d534d42e8b1feb43948af338b8b5c23427 \
        --confirm ATTACK_RULE_V1 --out outputs/validation_run_v1/<new>

Two runs gave byte-identical journals, snapshots, Contract v1 records and summaries.

**Acceptance.** A run is `verified` (exit 0) only if every step executed and every required Contract v1 call
(`query_events`; `check_tracking_quality` and `get_window_features` on the probe window; `compare_windows` at each
preselected pre_s) returned a well-formed result that agrees with its request and the fixed policies, and only after
the monitor's and the query's fired bouts were found identical (firing order, session and 30 fps frame boundaries of
start, firing and end, and end reason). A disagreement is refused (exit 2) before any window is measured; a tool
error or a malformed result fails the run (exit 1); neither writes the Contract v1 record. A comparison that returns
`insufficient_evidence` executed correctly: it is accepted and reported as inconclusive (`scientific_outcome`).
A comparison is never accepted on its declared status alone. An `ok` must meet the approved gate (at least 10 used
windows from at least 4 distinct sessions); its counts, exclusions and per-session records must be exactly the used
bouts of the verified query (no duplicated or invented session), and its effect the median of finite per-session
differences. An `insufficient_evidence` must name the condition that is genuinely unmet and carry no effect. Finally
the result must equal what the Contract v1 adapter itself computes for the same request (recomputed directly, not
through the tool dispatch). On a failed or refused run the summary states that no investigation evidence was accepted.

## Inputs and identity

| item | value |
|---|---|
| archive | `igor_scores_validation_model_b_repacked.zip`, SHA-256 `5c1a81e4…c23427` (matched the expected digest) |
| member | `scores_validation_model_b.csv`, SHA-256 `ff00f6fe…a008676` (matched `SHA256SUMS.txt`, in-band and unsigned) |
| verified rows | SHA-256 `b22df250…0b31e9`; 14 sessions, 94,305 frames; validation split |
| split manifest | `splits/split_manifest.json`, SHA-256 `501fed57…a187ac09` |
| pose cache | `data/raw/calms21_validation` (git-ignored), 14 / 14 sessions, 94,305 frames, every session aligned frame for frame with the scores; built from the CalMS21 Task 1 train file (SHA-256 `c66f1cae…76ecab`) |
| rule | `evaluate.ATTACK_RULE_V1`: attack, ask 0.6, min 0.5 s, alert 0.8, mode alert; approved for the run by `--confirm` |
| journal snapshot | SHA-256 `e75a808a…dca619`, `verified_score_drop`, complete |
| Contract v1 record | SHA-256 `f2873c27…64ebe`; query run id `inv_82f19185a7f3ebb3` |

The model artifact itself is not identified by the drop (`model_artifact_sha256` is null): the scores are bound to
the archive, not to a model file.

## Monitor (the unchanged `monitor.Monitor`)

- **80 events** in 11 of the 14 sessions: 69 alerts, 11 asks (confidence below 0.8).
- Events per session: mouse064 44, mouse045 19, mouse005 6, mouse009 / 021 / 035 2 each, mouse024 / 028 / 051 / 065 /
  067 1 each.
- Each event fires 0.5 s after its start (the minimum duration). Durations 0.53 to 4.13 s, median 0.87 s.
- **Tracking:** 30 events ended because `tracking_ok` became false, 50 because the score fell below 0.6. Across the
  14 sessions 1,067 frames (457 runs) have `tracking_ok` false; mouse009 alone has 655.
- Every event (id, session, start, fired, end, action, end reason, confidence) is in `run_summary.json` and the
  journal.

## Evaluation (`evaluate.monitor_event_metrics`, validation annotations)

| events | on a true bout | precision | true bouts detected | recall | F1 | false alerts / h | start offset (median) | fire offset (median) |
|---|---|---|---|---|---|---|---|---|
| 80 | 66 | 0.825 | 36 of 56 | 0.643 | 0.723 | 16.0 | −0.15 s | +0.35 s |

Detected per attack session: mouse005 2 / 7, mouse028 1 / 6, mouse045 12 / 20, mouse064 21 / 23. This reproduces
`docs/attack_rule_v1.md` exactly, computed independently here (labels from the verified cache, not Igor's parquet).

## Contract v1 tools (one `InvestigationRunner`, 6 of 12 calls)

- **`query_events`** (classifier probabilities, threshold 0.6, min 0.5 s, all 14 sessions): 80 bouts in 11 sessions,
  exactly the monitor's events (same session, start and end).
- **`check_tracking_quality`** on the pre window [22.3, 24.3) s of the first bout whose baseline starts at or after
  0 s (mouse005): reliable under `quality_v1`, usable fraction 1.0, no flag.
- **`get_window_features`** on the same window: `dist_center` mean 272.8 px over 60 usable frames.
- **`compare_windows`** (`dist_center`, baseline 5 s, gap 1 s, `quality_v1`, `overlap_v1`; pre_s fixed in advance as
  2 s, then the follow-ups 1 s and 5 s):

| pre_s | status | used / input | paired sessions | excluded: pre overlap / baseline overlap / edge / pre unreliable | median of session-paired differences (pre − baseline) |
|---|---|---|---|---|---|
| 2 s | ok (gate met) | 24 / 80 | 11 | 38 / 15 / 2 / 1 | −85.9 px |
| 1 s | ok (gate met) | 24 / 80 | 11 | 30 / 24 / 1 / 1 | −98.6 px |
| 5 s | ok (gate met) | 18 / 80 | 11 | 52 / 8 / 2 / 0 | −34.2 px |

## What these numbers do and do not support

- **The bouts are classifier predictions, not attacks.** 7 of the 11 paired sessions contain **no annotated attack**:
  all their bouts are false alerts. At pre_s 2 s they contribute 10 of the 24 used windows, and the per-session
  median mixes them with the 4 attack sessions (14 windows). The gate is met by session count, not by the number of
  real attacks.
- In the 4 annotated-attack sessions the pre − baseline difference of `dist_center` is negative at pre_s 1 and 2 s
  (the mice are closer just before a predicted attack than in the baseline); at 5 s it is mixed (two positive, two
  negative). It is an association with predicted onsets, in 4 sessions, not a causal or general result.
- The rule was chosen on these same 14 sessions, so the event metrics are optimistic (see `docs/attack_rule_v1.md`).
- One feature, one baseline and one gap were analysed; the pre_s grid and follow-ups were fixed before the run. No
  confidence interval is reported (none is approved for Contract v1).

## In the app (read only)

The Replay and Investigation pages have a **Data** choice: the existing SYNTHETIC demonstration (unchanged, the
default) or a **recorded real validation run**. A run is chosen by its directory name under `outputs/validation_run_v1/`
(or `$LIVE_TRACKER_VALIDATION_RUNS`) and checked again on every page run by
`live_tracker.validation_run_view.load_run`; anything that does not check out is shown as a refusal with its reason,
and nothing is substituted. The files are read through directory descriptors without following symlinks, each a
regular file within a size limit enforced while reading.

What the pages claim, in three levels (`VERIFICATION_LEVELS`):

1. **Consistency of the recorded execution (checked).** The four files agree with each other, with the tracked split
   manifest, with the journal's own validation (complete, `verified_score_drop`, frozen to exactly the stored
   snapshot), with Contract v1's rules and with `ATTACK_RULE_V1`. Every monitor count (events, alerts / asks, end
   reasons, events per session, tracking closures, durations) is derived from the journal. The query's bouts are the
   journal's events. The probe is the runner's probe (session, window, feature, `quality_v1`). Each comparison's
   counts, exclusions and contributing sessions are reconstructed from the verified bouts and its exclusions, its
   gate status is the one those counts imply, and its effect is the median of finite per-session paired differences.
2. **Independent authentication (not available).** No signature or independent registry binds these local files to
   the run that executed. A coordinated edit of the record and summary with recomputed digests (other per-session
   differences with a matching median) cannot be detected without re-running the tools on the data, so the pages say
   "consistent", never "authenticated".
3. **Recorded only (unverified claims).** The event-level evaluation, the annotated attack bouts per session and the
   per-session tracking interruptions cannot be re-derived from the run's files. They are checked for internal
   consistency and labelled as recorded, unverified claims; incomplete annotation coverage is shown as unknown, never
   as zero, and the pages draw no conclusion about false alerts or attack prevalence from them.

- **Replay (real):** the recorded rule and approval; journal-derived counts; per session the predicted events
  (journal order and ids) and the recorded tracking interruptions; the recorded evaluation, labelled unverified; the
  provenance. Browsing re-runs nothing and is not an approval.
- **Investigation (real):** the query and its bouts, the probe, the three comparisons (2 s primary, 1 s and 5 s
  follow-ups) with the gate, contributing sessions, exclusions by reason and both policies; the recorded trace with
  each result's digest; what the evidence supports; provenance and a download of the recorded record. No language
  model is involved.

The `replay_investigation` bridge stays synthetic-only (the synthetic Replay sessions are its only input).
