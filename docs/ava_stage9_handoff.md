# Ava: experimental investigation backend and Day 3–4 handoff

From Syed · Prepared 2026-10-20 · **For discussion, not automatic merge approval**

## 1. Executive summary

**TL;DR:** I built an experimental scientific-investigation backend alongside our original Day 3–4 work. The pushed baseline is [`0d3062f`](https://github.com/AvaSajjadi/live-tracker/commit/0d3062f5208552b15b0348825748f9de4bc2f296) on [`syed/experimental-investigation`](https://github.com/AvaSajjadi/live-tracker/tree/0d3062f5208552b15b0348825748f9de4bc2f296). It connects verified validation-score ingestion, monitoring, frozen journals, four scientific tools, a bounded investigator, and evidence-bound reports. The offline integration works; its scientific settings remain provisional, and prediction reports are **not authorized for publication**.

You asked me to review and mark up [PR #17](https://github.com/AvaSajjadi/live-tracker/pull/17) before freezing the contract. I developed this alternative in parallel. **You have not approved it, and I have not implemented your proposed contract as written.** I have not opened a PR for the experimental branch because I want us to align first. This document is my proposed comparison and handoff, not a request to merge the entire branch automatically.

The baseline demonstration is a deterministic offline emulator over real validation predictions—not another paid LLM investigation. It produced two separate `claims/2.1` reports, both `insufficient_evidence` and `not_authorized`. “Release-ready” in the final review means ready for this documented offline demonstration and your inspection, not scientifically validated or a completed hackathon dashboard.

**Scope boundary:** Git HEAD and the local remote-tracking ref both identify `0d3062f`. Local S9-7a edits now exist beyond that commit. Every implementation link below is pinned to the pushed baseline unless explicitly marked otherwise. Local S9-7 work is excluded from the baseline and described separately in section 9.

## 2. Original project timeline and ownership

The authoritative assignments are the [team plan](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/plan/00_TEAM_PLAN.md), [my plan](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/plan/SYED.md), [your plan](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/plan/AVA.md), and [Igor’s plan](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/plan/IGOR.md). They set a Day 4 prototype, Day 5 demo freeze, and October 25 submission. Their dated status tables are historical, not reliable evidence of everyone’s current progress.

| Day | Original coordination | My assigned work |
|---|---|---|
| 1 — Sat Oct 17 | Igor: schema/loader/split (H1a/H1). Ava: causal motion features and tracking groundwork. | Synthetic score generator, streaming monitor, event journal, approved-rule-only behavior. |
| 2 — Sun Oct 18 | Ava: torso/dyadic features and tracking warnings (H2/H2b). Igor: basic features, model A, evaluation. | Offline natural-language parser, optional Claude path, Confirm/Reject Rules page; real-session skeleton drawing after H1. |
| 3 — Mon Oct 19 | Ava: train/validation tuning and sensitivity. Igor: model B, score export (H3), evaluation handoff (H4). | Recorded-speed replay, skeletons/scores/rules/events, researcher answers, tracking-warning display, then real-score replay. |
| 4 — Tue Oct 20 | Ava: ablation, real motion-engine page, integration review. Igor: validation-selected thresholds and saved model artifacts. | Deterministic log-only session summary; H5 end-to-end sentence → Confirm → replay → events/warnings → summary on an authorized held-out session. |
| 5 — Wed Oct 21 | Documentation, Windows checks, report/slides, offline backup demo and rehearsal. | Dashboard instructions, offline demo, screen recording, rehearsal. |

My earlier [PR #16](https://github.com/AvaSajjadi/live-tracker/pull/16) implements Day 2 rule proposals and approval. It remains separate from scientific investigation. It includes `agent.py`, `claude_agent.py`, `rule_workflow.py`, `pages/1_Rules.py`, and tests. Proposals do not activate monitoring; Claude only cross-checks offline-valid instructions. PR #16 reports 1,185 passing tests and independent reviews; those are historical reported results, not tests rerun for this handoff.

At `0d3062f`, `pages/1_Rules.py` is the only file under `pages/`; there is no `pages/2_Replay.py` or `src/live_tracker/summary.py`. The starter `app.py` is a motion/CSV dashboard, not the planned full rule-to-replay flow. Therefore my Day 3 replay and Day 4 session-summary UI remain pending **on this branch**. I cannot infer work on teammates’ unpublished branches.

Skeleton drawing, event timelines, recorded-speed playback (1×/2×/5×), ask-event answers, and tracking warnings are explicit assignments. A specific time/frame navigation widget, event-overlay layout, and summary-page design are sensible implementation proposals, not separately frozen requirements. The Stage 9 scientific report does **not** replace the original event-log-only session summary.

### Dependency PR status at this handoff

Authenticated GitHub PR metadata was read on 2026-10-20; all four PRs were **OPEN**, with no merge timestamp:

| PR | Current purpose/status | Dependency implications |
|---|---|---|
| [#15](https://github.com/AvaSajjadi/live-tracker/pull/15), head `bbaaf074` | Igor’s basic features, model A, evaluation. | Needed for model/evaluation integration; reported 153 tests and validation results are PR-author results, not rerun here. |
| [#16](https://github.com/AvaSajjadi/live-tracker/pull/16), head `c85ebd04` | My Day 2 parser/API adapter/approval UI. | Experimental history builds on this work; agree on PR base/order before creating a large downstream diff. |
| [#17](https://github.com/AvaSajjadi/live-tracker/pull/17), head `90453bbe` | Your investigation-contract proposal and dry-run scripts. | Still awaiting agreement; no claim that this branch implements or supersedes it. |
| [#18](https://github.com/AvaSajjadi/live-tracker/pull/18), head `77cd1280` | Igor’s score export and model B results; explicitly stacked on #15. | Export checks §2 schema, exact split sessions, finite scores, contiguous frames, and `time_s = frame / 30`. Merge #15 first, per PR description. |

PR #18 reports validation model-B scores for 14 sessions/94,305 frames and test scores for 19 sessions/262,107 frames. It says model B was selected on validation and test evaluation has not been performed. These are Igor’s reported artifacts/results; this handoff does not evaluate the test split or independently authenticate training provenance.

## 3. Your PR #17: what it actually proposes

Reviewed all three changed files at head [`90453bbe77f7d18459efec6540087afd8f3810fd`](https://github.com/AvaSajjadi/live-tracker/tree/90453bbe77f7d18459efec6540087afd8f3810fd):

- [Draft contracts](https://github.com/AvaSajjadi/live-tracker/blob/90453bbe77f7d18459efec6540087afd8f3810fd/docs/investigate_tool_contracts.md).
- [Annotation usable-window dry run](https://github.com/AvaSajjadi/live-tracker/blob/90453bbe77f7d18459efec6540087afd8f3810fd/scripts/dryrun_usable_windows.py).
- [Classifier-bout dry run](https://github.com/AvaSajjadi/live-tracker/blob/90453bbe77f7d18459efec6540087afd8f3810fd/scripts/dryrun_classifier_bouts.py).

The description says “Docs only, no code,” but the current diff also contains two executable feasibility scripts. The contract itself explicitly says **PROPOSAL; nothing implemented**. The scripts are counts-only experiments, not implementations of the complete five-tool API.

The proposal uses observation-time seconds, half-open `[t0_s, t1_s)` windows, named error/insufficiency reasons, and explicit `data_source` labels (`synthetic`, `ground_truth_labels`, `classifier`). Tools do not modify active researcher rules or existing journal run IDs.

| Proposed tool | Proposed behavior |
|---|---|
| `query_events` | Wrap the monitor using a throwaway rule and `inv_<hash of args>` run ID. Inputs include sessions, behavior, threshold (default 0.5), minimum seconds. Return bout start, fired/end times, end reason, confidence and bout mean. Anchor analysis at **start**, never alert firing. |
| `get_window_features` | Slice your feature table by explicit seconds and requested columns. Summarize tracking-valid frames with mean/min/max/count; do not send raw rows. |
| `check_tracking_quality` | `reliable` iff usable fraction ≥0.8 and no frozen/out-of-frame run longer than 0.5 seconds. Fixed, logged criteria. |
| `compare_windows` | Pair each bout’s pre-window with an earlier baseline. Pre: `[start−pre_s, start)`; baseline: `[start−pre_s−gap−baseline_s, start−pre_s−gap)`. Both must be reliable; pre must not overlap another same-behavior bout. Default `min_n=10`. Return counts/exclusions, pre/baseline means, mean difference, paired `effect_size_dz`, and `ci95`; always `association_only=true`. |
| `propose_rule` | Return a proposed rule through `rules.py`; the researcher must approve it. The agent cannot approve its own rule. |

The agent budget is **12 tool calls**, at most **two follow-up analyses after the first comparison**, with pre-window choices fixed in advance to `{1,2,5}` seconds. Every attempt is recorded. Strategies are chosen on train + validation, with one final test-score run and no test labels exposed to the agent. That access policy differs from my validation-only scientific backend.

The contract reports 40 annotated attack bouts ≥1 second; usable pre-windows were 21 at 2 seconds, 14 at 5 seconds, and 5 at 10 seconds, predominantly excluded for overlapping attacks. Those are **reported dry-run counts**, not rerun here. Both scripts still use `PRE_GRID=(2,5,10)`, 5-second baselines, and a 1-second gap; they are evidence motivating the proposed `{1,2,5}` grid, not tests of that exact final grid. The scripts check usable fraction and window boundaries; they do not implement the separate >0.5-second frozen/out-of-frame veto from the proposed quality contract. No classifier dry-run output was present in the three-file diff, so I do not assert classifier counts from it.

## 4. My experimental implementation and sequence

This is an additional backend, not a replacement for your motion engine or Igor’s classifier. The [shared interfaces](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/INTERFACES.md) still define score rows, approved rules, event exports and feature tables. Your `features.py`, `quality.py`, `filters.py` and `motion.py` remain your ownership boundary.

### Stages 1–8 and Findings v2

| Checkpoint | Implemented responsibility and source |
|---|---|
| Day 1–2 foundation | [Monitor](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/monitor.py), [event journal](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/event_log.py), parser/approval work in #16. Monitoring uses approved rules only; investigation does not activate rules. |
| Stage 1, `bbb3d10` | Validation extraction and feasibility measurements; [experiment report](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/docs/investigation_experiment_01.md). Explicit tracking-artifact, filter-memory, timing and label-confound limitations. |
| Stage 2, `978efcc` plus hardening `97c71b9`/`a75bf42` | [Contracts and EvidencePolicy](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_contracts.py): validate requests/results, policy and content hashes before accepting evidence. |
| Stage 3, `5d7b764` | [Annotation event retrieval](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_events.py) over verified validation cache snapshots. |
| Stage 4, `bac6310` | [Feature windows and tracking quality](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_windows.py): integer frames, half-open windows, endpoint validity and computation equivalent to truncation at the window end. |
| Stage 5, `4cb1b4f` | [Comparisons](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_compare.py): window medians, session-level pairing, median paired difference, provisional sufficiency, no approved uncertainty method. |
| Stage 6, `c84e3d8` | [Bounded plan–act–observe loop](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_agent.py): model chooses tools/parameters from previous validated results; policy, sufficiency and uncertainty are outside model control. |
| Stage 7, `858701c` | [Evidence-verified reporting](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_report.py), retained as legacy rendering. |
| Stage 8, `f179bd3` | [Provider/runner](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_live.py): OpenRouter transport, explicit live authorization, budget ledger, no hidden retries, offline emulator, full validation coverage. Anthropic live fallback remains disabled, not automatic. |
| Findings v2 M0–M4 | `db3fc3a`, `a64f614`, `0d11b0d`, `66a650f`, `ea9ff25`: shared context, typed claims, deterministic templates, structured finish, legacy lexical isolation. [Design](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/docs/findings_v2_design.md), [context](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_context.py), [registry](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_claims.py), [v2 renderer](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_report_v2.py). |

The loop supports an actual model choosing its actions; the Stage 9 demo deliberately substitutes an offline emulator. A prior authorized Stage 8 real-model run is a separate historical artifact, not evidence that the Stage 9 demo used an LLM. Neither is proof of a biological association.

Findings v2 replaces unrestricted scientific prose with closed claim types and evidence references. The renderer supplies validated numbers, directions and fixed wording. Evidence identity, execution consistency, authorization, measurement comparability and scientific sufficiency are distinct assessments. Saved JSON is rebuilt from its canonical payload; editing a sentence, publication status or source disclosure cannot promote it. Incomplete runs acquire no final claims. Fingerprints provide consistency, **not signatures or producer authentication**.

### Stage 9: the pushed pipeline

| Commit | Addition | Main files |
|---|---|---|
| [`aa69428`](https://github.com/AvaSajjadi/live-tracker/commit/aa69428) — S9-1 | Validation score ZIP ingestion; checksums, split/schema/frame checks, loader-issued score objects; monitor smoke runner. | [`score_drops.py`](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/score_drops.py), [`stage9_smoke.py`](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/scripts/stage9_smoke.py). |
| [`7745c95`](https://github.com/AvaSajjadi/live-tracker/commit/7745c95) — S9-2 | Run headers bind rules and score provenance; completion binds processed-row digest/count and event lifecycle digest/count. Frozen snapshots exclude later researcher answers and unrelated runs. | `monitor.py`, `event_log.py`, `score_drops.py`. |
| [`d5e9f6c`](https://github.com/AvaSajjadi/live-tracker/commit/d5e9f6c) — S9-3 | Verified classifier and completed-monitor evidence adapters; no annotation fallback. | [`investigation_sources.py`](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/src/live_tracker/investigation_sources.py), `investigation_events.py`. |
| [`97589f6`](https://github.com/AvaSajjadi/live-tracker/commit/97589f6) — S9-4 | Prediction-aware windows, quality and comparisons; source-specific exclusions; annotation composition uncomputed. | `investigation_windows.py`, `investigation_compare.py`, [provisional-methods document](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/docs/stage9_prediction_windows.md). |
| [`1087c89`](https://github.com/AvaSajjadi/live-tracker/commit/1087c89) — S9-5 | `claims/2.1`: one prediction source per completed run, source-derived disclosures, cross-snapshot identity, publication disabled. `claims/2.0` remains annotation-only. | Claims/context/renderer/agent/source modules above. |
| [`0d3062f`](https://github.com/AvaSajjadi/live-tracker/commit/0d3062f) — S9-6 | Complete offline chain, separate classifier/monitor reports, saved-report verification and failure summary. | [`stage9_integration.py`](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/scripts/stage9_integration.py), [integration guide](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/docs/stage9_integration.md). |

Three event sources remain distinct: **validation annotations** describe labeled frames; **classifier predictions** are threshold runs in scores; **monitor detections** are fired bouts from an approved rule’s completed journal. The last two are not verified behavioral events. Monitor onset, firing and closure are distinct timestamps.

The score archive’s checksum verifies the received bytes; the in-archive checksum verifies internal consistency. Neither establishes training provenance. Model labels are inferred from filenames and remain explicitly unverified without the actual model artifact, configuration and producing commit.

## 5. Shared intent

Both approaches ground statements in tool evidence, inspect tracking quality, query events, compare windows, bound investigation and preserve researcher control. Both distinguish association from causality and aim for deterministic, reviewable tools.

These are shared goals, not API equivalence. Your proposal integrates rule proposals, per-bout baselines, a specified reliability veto and uncertainty outputs. Mine already implements provenance chains, session-level descriptive aggregation and a fixed claim renderer, but lacks the proposal tool and does not implement your statistical or window contract. Your scripts provide useful feasibility evidence; my integration provides different, complementary backend evidence.

## 6. Differences and proposed reconciliation

**Every recommendation in this table is mine and provisional—not an agreed team decision.** Sources are the pinned [PR #17 contract](https://github.com/AvaSajjadi/live-tracker/blob/90453bbe77f7d18459efec6540087afd8f3810fd/docs/investigate_tool_contracts.md), [S9-4 methods](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/docs/stage9_prediction_windows.md), and implementation links above.

| Area | Your PR #17 | My pushed implementation | Practical implication | My provisional recommendation |
|---|---|---|---|---|
| Tools | Five, including `propose_rule`. | Four scientific tools plus finish protocol. | Existing rule UI is separate, not a fifth investigation tool. | Freeze four read-only tools first; add a proposal-only adapter if needed. |
| Rule approval | Agent may propose, never approve. | Day 2 Confirm/Reject gate; integration explicitly approves a labeled local smoke rule. | Local demonstration approval is not researcher approval. | Keep one researcher gate and visibly separate disposable demo rules. |
| Time interface | Explicit seconds, `[t0_s,t1_s)`. | Frames and episode/window IDs, with seconds reported. | Conversion/rounding must be specified at 30 fps. | Offer seconds at UI boundary, canonical frames internally, tested rounding. |
| Pre-window choices | Fixed `{1,2,5}` seconds; two follow-ups. | Model can select lengths/gaps under contract limits; no equivalent approved grid (`grid_version=none`). Emulator uses its own bounded sequence. | Different search freedom and selection risks. | Freeze a small reviewed grid before comparative scientific use. |
| Baseline | One earlier baseline per bout, fixed relation to pre-window. | Same-session, nonoverlapping tiles, clearance and deterministic thinning; excludes retained pre-windows. | Different estimands, timing distributions and counts. | Name/version both strategies; agree on one initial method rather than silently translating. |
| Event exclusions | Pre-window cannot overlap another same-behavior bout. | Also considers feature history/gap; masks come from the declared source. Monitor mask includes fired bouts only. | Short unfired monitor activity remains eligible; classifier misses remain invisible. | Review masks explicitly; never switch source silently. |
| Reliability | ≥0.8 usable plus frozen/out-of-frame veto >0.5 s. | Default 0.9 endpoint-valid usability; `insufficient_data`, `excluded`, `usable_with_caveats`; no equivalent dedicated veto. | Neither setting is an interchangeable assurance of accuracy. | Let you approve the quality policy and fixtures; preserve caveated wording. |
| Tracking/interpolation | Slice your tracking-valid feature rows. | Score tracking may incorporate Igor’s short-gap interpolation; scientific pose windows use non-imputed validity, both derivative endpoints and window-end truncation. | Same frame may have different scientific usability from score eligibility. | Record both policies; obtain exact feature/export settings and agree how imputation is exposed. |
| Onset | Monitor bout start, not fired time. | Classifier threshold crossing or monitor detection onset; firing happens later. | Neither prediction onset guarantees pre-behavior observations. | Keep distinct fields and “pre-threshold/pre-detection” language. |
| Duration | Seconds through monitor. | Classifier ≥N frames; monitor elapsed observed time (0.5 s requires 16 samples at 30 fps, versus 15-frame classifier minimum). | Equal-looking settings select different events. | Document both rather than force accidental equivalence. |
| Provenance | Echo source and journal attempts. | Verified score member, run/rule/journal binding, evidence hashes, shared context and saved reconstruction. | Stronger consistency checks, still not authentication. | Retain these checks behind whichever public API we agree. |
| Authorization | Researcher-controlled rule proposals. | Separate evidence policy and `claims/2.1` publication prohibition. | An approved monitoring rule does not authorize scientific publication. | Keep these separate approvals. |
| Statistics | Means, paired `dz`, 95% CI, `min_n=10`. | Median window values; median of session-level paired differences; ≥4 paired sessions and ≥10 windows/group by default; no CI/p-value. | Repeated bouts are not independent sessions; uncertainty needs an explicit method. | Defer CI until you approve session-aware estimation and sampling. |
| Budget | 12 calls, two follow-ups, all attempts logged. | Generic loop defaults 12 turns/16 calls; S9-6 uses 10 turns/9 calls, repeat/rejection/time/token guards. | Bounds exist but do not match your exact protocol. | Adopt one documented profile; keep failure/repeat accounting. |
| Annotation context | Labeled development and train/validation strategy selection. | Prediction runs never compute annotated composition or annotation confounding. | Cannot claim investigation-label confounding was assessed. | Decide whether a separate explicit contextual-label tool is useful; defer until approved. |
| Latency gap | Pre ends at bout start. | Gap configurable; demo uses zero. | Late detection can include behavior inside “pre” windows. | Ask you and Igor to select a justified gap/sensitivity, without claiming latency correction now. |
| Thresholds | Query threshold/minimum inputs; fixed quality criteria. | Source-level classifier threshold/minimum and frozen monitor rule; demo values provisional. | Model must not silently retune a source. | Freeze approved configurations with explicit versions. |
| Your actual outputs | Slice your feature table and flags. | Recompute selected pose-derived measurements from verified cache using your functions under stricter temporal handling. | Not a drop-in consumer of an arbitrary motion parquet. | Agree on artifact schema, causality, torso mapping and validation fixtures before replacing either path. |
| Data access | Train+validation development; final test predictions once. | Scientific tools and score loader are validation-only. | Original held-out replay deliverable needs a separate authorized path. | Keep validation investigation isolated; plan final replay/test evaluation with Igor explicitly. |

## 7. Verification and individualized testing

The review history in our working conversation records individual checks of Stages 1–9 and Findings v2 M0–M4. Git commits and regression tests preserve implementations; they do not independently prove every historical reviewer ran every claimed test. No suites or real archive were rerun while writing this document.

Representative corrected blockers include future-coordinate leakage, invalid duration evidence, cross-snapshot contradictions, missing/unknown units, invalid provider-cost responses executing actions, duplicate provenance paths, incomplete traces promoted to findings, unsupported timing prose, fabricated score provenance, unfinished/deleted journal lifecycles, merged episode starts, and contradictory monitor journal metadata/data entries. Their presence in regression tests is evidence of coverage, not proof that every possible defect is excluded.

Key test sources at the baseline:

- [Score ingestion](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/tests/test_score_drops.py) and [journal provenance](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/tests/test_journal_provenance.py): held-out member refusals, loader issuance, processed rows, lifecycle/completion binding, answer exclusion.
- [Evidence adapters](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/tests/test_investigation_sources.py) and [prediction windows](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/tests/test_prediction_windows.py): source isolation, threshold/duration boundaries, merged starts, no annotation use and honest timing.
- [Source findings](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/tests/test_source_findings.py): `claims/2.0` compatibility, cross-source/configuration refusal, empty queries, source disclosures, saved-document promotion refusal, rehashed journal mutations across all four tools and first/only snapshots.
- [Integration tests](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/tests/test_stage9_integration.py): synthetic end-to-end run, stable fingerprints, incomplete-journal rejection, forbidden test member, report reconstruction, output overwrite refusal, Stage 8 emulator defaults.
- [v2 report tests](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/tests/test_investigation_report_v2.py): legacy output and historical-artifact checks, deterministic reconstruction and trust boundaries.

### Recorded review evidence, with limits

| Evidence | Recorded outcome | What this establishes |
|---|---|---|
| Final S9-4 issuance follow-up in this conversation | 23 parametrized cases passed in 2.82 s. | Fabricated/altered issuance records rejected; genuine merged starts retained. Historical execution, not rerun for this document. |
| S9-5 correction run in this conversation | 42 passed, 15 deselected, 26.65 s. | Targeted mutations plus directly affected compatibility/empty-query tests. |
| Final narrow S9-5 verification in this conversation | 37 passed, 20 deselected, 26.01 s. | 32 mutation cases plus five relevant chain regressions; APPROVE. |
| Final cross-stage review in this conversation | RELEASE-READY for the documented offline demonstration. No pytest rerun. | Two saved reports reconstructed with matching fingerprints/Markdown in 2.147 s; journal/source reconciliation took 0.013 s. |
| Other checkpoint totals quoted during development | E.g. 421/666/935 broader focused results were reported by Claude. | Not independently recovered as test logs here; not additive, not a current full-suite count. |

“Independent” here describes the recorded review workflow, not scientific peer review. Some corrections and later verification occurred within this same Codex conversation; this document does not claim a fully separate clean-room reviewer for every fix.

### Existing real-validation demonstration

The [integration guide](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/docs/stage9_integration.md) reports a 63-second run. The final audit inspected the existing `stage9_integration_real3` artifacts in a local scratch directory, rather than repeating the archive run:

| Measurement | Saved result |
|---|---|
| Input | 94,305 frames, all 14 authorized validation sessions. |
| Monitor | 7 events; completed `verified_score_drop` snapshot; `LOCAL_SMOKE_TEST` rule, 2 s / ask 0.5 / alert 0.8. |
| Classifier report | 4 paired sessions; 7 pre-event windows; 263 baseline windows in the final comparison. |
| Monitor report | 2 paired sessions; 4 pre-event windows; 50 baseline windows in the final comparison. |
| Conclusions | Both `insufficient_evidence`; both `not_authorized`; no aggregate-effect conclusion promoted. |
| Execution | Separate source chains; eight emulated requests each; six unique scientific evidence snapshots each; $0 provider-reported cost. |

Each run attempted a second tracking-quality call after remeasuring baselines; it was refused as an identical already-answered request. This is logged, not an unexecuted call misrepresented as fresh evidence. Cross-snapshot agreement is only `partially_verified` because feature snapshots do not explicitly record units, although compared values agree. Reports disclose that limitation. Their summary says split “not recorded,” while policy rechecking establishes validation authorization; that display limitation is not permission to access another split.

The local artifact bundle contains journal JSONL, frozen snapshot, per-source investigation JSON, report JSON/Markdown, and `integration_summary.json`. It is not committed or available through the source permalinks. I need to give you a durable approved shared location for it. Historical test timings and local checks above are transcript evidence, not newly executed measurements for this handoff.

## 8. What works, and what remains unfinished

**Working backend at the pushed baseline:** verified validation score ingestion; approved-rule monitor; durable completed journal; frozen snapshots; four source-aware tools; bounded model-capable loop; typed claims; deterministic saved reports; offline integration and explicit failure summaries. The CLI refuses existing output names, writes only to allowed output locations, and distinguishes verified, failed and refused-input outcomes.

The documented command is:

```bash
python scripts/stage9_integration.py <validation-score-drop.zip> \
  --out <new-ignored-or-external-directory> \
  --archive-sha256 <separately-received-digest> \
  --member scores_validation_model_b.csv \
  --cache-dir data/raw/calms21_validation \
  --manifest splits/split_manifest.json
```

Use a checkout pinned to `0d3062f` for this baseline; my current working tree has S9-7 changes. This command is documentation, not something run for this handoff. The expected archive digest is optional in the CLI but recommended. Output reconstruction does not require another paid model call.

**Not established:** model accuracy, causal biology, an approved association, calibrated thresholds, publication authorization, or real-model autonomy in the Stage 9 demo. The existing model-capable loop and earlier Stage 8 run are separate from the deterministic emulator demonstration.

**Still pending here:** full Streamlit replay; skeleton playback/navigation; unified event/warning overlays and timeline; replay ask answers; deterministic event-log-only summary module/UI; H5 dashboard integration; final held-out demonstration under a separately agreed policy. Ava/Igor’s latest unpublished progress is unknown. The [implementation-status document](https://github.com/AvaSajjadi/live-tracker/blob/0d3062f/IMPLEMENTATION_STATUS.md) is dated October 17 and cannot establish today’s blockers.

## 9. Exploratory S9-7 — not part of the release-ready baseline

**At pushed `0d3062f`, S9-7 was proposed/planned only.** During preparation, Git status showed local changes in integration/source/context/claims/renderer/emulator files, an untracked `tests/test_strict_episodes.py`, and an untracked `docs/stage9_s97_exploration.json`; later status also showed integration-test edits. These indicate S9-7a implementation has begun locally. They are not in the baseline linked above, were not validated for this document, and must not be included implicitly in its readiness claim.

The local exploration record reports:

- Unmerged `prediction-episodes/1` classifier predictions: **109 episodes**.
- One-second merging (`merge_gap_frames=30` at 30 fps): **117 episodes**.
- This increase is possible because merging happens **before** minimum-span filtering. Two individually too-short runs plus the intervening gap can meet a 15-frame span minimum; merging is not simply combining an already-filtered list.
- **43 of 117** merged episodes contain fewer than 15 genuinely qualifying frames (minimum reported: 2). The count therefore should not be described as 117 sustained ≥15-frame predictions.
- Proposed opt-in `prediction-episodes/2`: require both sufficient span and at least 15 qualifying frames. The record’s **counts-only estimate is 74 episodes**; usable windows for that stricter rule had not been measured in that recorded estimate.

The record describes six feasibility attempts varying merge gap and window length, including contract-limit failures. Selected settings were chosen **after** examining episode/window counts and sufficiency, not prospectively preregistered. It explicitly says no movement-effect values or directions were inspected in that scan. These are local recorded claims, not independently recomputed counts here. A later local effect run, if one exists, must be disclosed separately rather than retroactively turning this into a prospective protocol.

The unmerged Stage 9 demonstration’s `insufficient_evidence` finding remains valid and unchanged. The optional stricter rule and exploration need your and Igor’s methodology review; **S9-7 is not a prerequisite for considering the existing Stage 9 integration**. My preference is to pause expansion until we agree whether it is worth hackathon time.

## 10. Decisions needed from you

### Essential alignment before a team-facing contract/merge

- **Tool contract:** five tools or staged four-tool API; seconds/frame mapping; source names; whether to retain IDs and canonical evidence underneath; exact status/error translation.
- **Scientific methodology:** pre-window grid, baseline strategy, eligibility masks, minimum independent sessions/windows, tracking vetoes, imputation policy, latency gap, and which settings remain explicitly provisional. These block scientific reliance, not inspection of the existing software.
- **Integration dependencies:** your stable pose/features/warnings interface and Igor’s stable score/model outputs; which commits to build against while #15/#16/#18 remain open.
- **PR strategy:** base branch and stacking order; one experimental PR or small reviewable checkpoints; which components belong in the hackathon deliverable.
- **Original timeline:** whether replay/summary/H5 should now take priority over further investigation work.

### Optional or post-hackathon

- Session-aware uncertainty intervals and effect-size methodology.
- Explicit annotation-context analysis without contaminating prediction evidence.
- Stronger model-artifact authentication and signed evidence.
- S9-7 stricter episode definitions and exploratory runs.
- Richer investigation UI, grid-based follow-ups, and approved publication policy.

I do not regard those optional additions as reasons to discard working code or delay your review of the current backend.

## 11. Proposed collaboration plan

**A. Keep PR #17’s public contract and adapt my backend.** This prioritizes your interface proposal. We can reuse provenance, monitor, policy and reporting internals, but need explicit translation for window definitions, statistics, reliability and `propose_rule`; merely renaming fields would be unsafe.

**B. Use my current implementation as the base, incorporating selected PR #17 requirements.** This preserves a tested end-to-end path and evidence handling. It asks you to review a larger existing API and leaves some of your intended semantics unimplemented until we agree them.

**C. Hybrid: approve a small contract first.** My provisional recommendation is to agree on event-source/onset semantics, time conversion, quality policy, one baseline strategy, sufficiency and approval boundaries; keep provenance/claims internals; implement only the agreed adapter/UI surface. Keep unapproved statistical outputs and S9-7 out of that first integration.

C seems the lowest-risk hackathon path: preserve functioning code, avoid a wholesale rewrite, and make every scientific change explicit. You may prefer A or B. I can mark up #17 with concrete differences once you choose the direction, then open a scoped PR against the base you specify. Nothing in this document authorizes me to merge, change your motion files or run another paid investigation.

## 12. Original Day 3–4 tasks for tomorrow

Would you prefer I stop agentic extensions and finish the original Day 3/4 deliverables tomorrow? Relative to preparation on October 20, tomorrow is October 21—the original Day 5—so this would be catching up with the prototype tasks, not claiming we are still on schedule.

Proposed priority order, subject to your direction:

1. **Streamlit replay:** recorded-session banner, synthetic/real choice, recorded-time playback, current frame/time and speed controls. Time/frame scrubbing is my proposed UI detail.
2. **Pose/skeleton view:** both mice and verified keypoint connections, synchronized with scores and active rules. The skeleton task was originally Day 2 and remains a dependency for replay.
3. **Events/warnings:** overlay and timeline, detection/firing/closure distinctions, tracking-loss display and export, ask-event Confirm/Reject.
4. **Deterministic session summary/report UI:** counts, durations, times and confidences from the event log only; keep this separate from scientific findings. Optional model wording should not delay the offline path.
5. **H5 integration:** sentence → proposal → explicit approval → replay → events/warnings → summary, with a stable run command and handoff acceptance checklist.

To implement that safely, I need from you:

- Current status of H2/H2b, real-session feature work, threshold tuning and the motion-engine UI—not just the October 17 checklist.
- Stable branch/commit or merged PR for pose, features, quality flags and `tracking_warning_events`; which outputs are final versus exploratory.
- Exact DataFrame/file schemas, dtypes, units, paths, keypoint ordering/skeleton edges, mouse identities, torso selection and coordinate conventions.
- Session/frame/time mapping, fps and gap conventions; alignment expectations between score rows, pose rows and warning spans.
- Tracking thresholds, interpolation/`uses_imputed` behavior, frozen/out-of-frame conventions, and which rows are safe for playback versus scientific measurement.
- Warning/event fields and ID namespaces; expected lifecycle updates; how warnings coexist with monitoring events and researcher answers.
- Expected replay/summary UI behavior, integration acceptance criteria, handoff deadline and any changed priority or blocker.

From Igor I need the agreed H3 score files and checksums, split manifest, exact exporter/model commit, model artifact/configuration and validation-selected thresholds. PR #18 supplies export semantics and reported score drops, but not authenticated model identity inside the Stage 9 archive. The planned held-out replay must be coordinated explicitly: the validation-only investigation loader should not be weakened merely to make the UI demo consume test predictions. Test labels remain unavailable to the agent.

## 13. Final questions for Ava

1. Do we agree with the general direction and architecture of my experimental branch, within its current exploratory limits?
2. Which parts of your PR #17 should I incorporate, preserve as the public interface, or replace in my implementation?
3. Should I adapt my branch to your contract, or should we evolve and freeze the contract together first?
4. Once aligned, should I open a PR from `syed/experimental-investigation`, and against which base while PR #16 is still open? Would you prefer smaller stacked PRs?
5. Would you prefer I stop agentic extensions, including local S9-7 work, for now?
6. Should I finish my original Day 3 and Day 4 deliverables tomorrow to return to our hackathon timeline?
7. What exactly is your current progress on the pose/tracking/features deliverables that my replay and summary integration needs?
8. Which interfaces, files, artifact paths and branches/commits are stable for me to build against, and which dependencies from Igor are ready?
9. What integration sequence, acceptance criteria and deadline would you like us to follow?
