# Findings v2: evidence-bound structured claims (design, revision 3)

Status: DESIGN ONLY, not implemented. Revision 3 resolves the five blockers of
Codex 2's review of revision 2 (sensitivity matching, shared-context
fingerprint, global claim rules, confound / tracking categories, version
dispatch) and narrows the first release (registry `claims/2.0`) to the claims
needed for one reliable v2 investigation.

## 1. Goal, scope and non-goals

Goal: no reportable scientific sentence is written by the model. The model
chooses claims from a closed registry; each claim cites evidence; every
sentence comes from a fixed template filled only with values read from
content-consistent snapshots and enum labels.

Unchanged: autonomous tool selection and planning, all Stage 6 budgets and
termination rules, evidence snapshots, Stage 2 contracts, Stage 3-5 science,
Stage 7 deterministic JSON -> Markdown rendering, Stage 8 provider / budget /
guard code, behavior-scoped coverage and data-identity rules.

Registry `claims/2.0` contains only: comparison status claims, the
descriptive effect, one sensitivity variant (excluding flagged windows),
coverage facts, one measured potential confounder, measurement context,
tool disclosures, tracking concerns, operational facts, methodological
limitations, and three next-investigation actions. Everything else is
deferred (section 13).

## 2. Verification layers (distinct, never inferred from one another)

| Layer | Meaning | Mechanism | Report field |
|---|---|---|---|
| L0 content consistency | each snapshot's content matches its recorded evidence_id | sha256 of (tool, request, result) | `verification.content_consistency` |
| L1 schema | each snapshot is a well-formed contract result of its request | `check_result_schema` | `verification.schema` |
| L2 authorization | the evidence may be used under the policy | `validate_result(.., policy)` | `verification.authorization`: `rechecked` / `not_rechecked` / `refused` |
| L3 cross-snapshot | snapshots read the same data and agree | data identity, measurement comparability, value agreement, coverage provenance | `verification.cross_snapshot`, `coverage` |
| L4 claim grounding | each claim's predicate holds in the cited evidence | registry predicates | `claims[].grounding` |
| L5 interpretation | biological meaning | not verified by software | always `descriptive`, never `proven` |

L0 is content consistency, not authentic tool origin: an evidence_id is an
unkeyed hash, so a consistent rewrite of snapshots and references is not
detected (no signatures). Templates never say "proves", "shows that",
"demonstrates", "establishes", "causes", "significant", "predicts" or
magnitude adjectives; a template test enforces it. "Verified" always names
its layer.

## 3. Shared validation: fingerprint, assessments, immutability

### 3.1 Content fingerprint (what Stage 6 and Stage 7 must agree on)

`fingerprint = sha256(canonical_bytes(F))`, where F is:

```json
{"fingerprint_version": "fp/1",
 "findings_format": "findings/2", "registry_version": "claims/2.0", "grid_version": "none",
 "evidence": [["<evidence_id>", "<tool>", {request}, {result}], ...],
 "finding": "effect_described | insufficient_evidence | inconclusive | incomplete",
 "primary_evidence_id": "... | null",
 "observations": [{"evidence_id": "...", "field": "..."}],
 "claims": [{"type": "...", "evidence_ids": ["..."], "params": {...}}],
 "next_investigations": [{"action": "...", "params": {...}, "because": [0]}]}
```

- Serialization: `json.dumps(F, sort_keys=True, separators=(",", ":"),
  ensure_ascii=False, allow_nan=False).encode("utf-8")`. Numbers are the
  values parsed from the snapshots (Python `json` float repr); no rounding.
- Ordering: `evidence` sorted by evidence_id; `observations`, `claims`,
  `next_investigations` and every `evidence_ids` list in submitted order
  (indices are referenced by `because`); object keys sorted by the
  serializer.
- Excluded: trace, execution history, stopping reason, usage, timings,
  model ids, policy and authorization outcomes, rendered text, coverage and
  any other derived value (derived values are recomputed, never trusted).
- Freeze point, finished runs: Stage 6 computes it once, immediately after
  `verify_final` accepts the finish, from deep-copied canonical bytes, and
  records it in the report as `fingerprint`.
- Freeze point, incomplete runs (no finish accepted: budget, refusal,
  unavailable model, ...): at deterministic final-report construction
  (`stop()`), from the evidence collected so far with `finding:
  "incomplete"`, `primary_evidence_id: null` and empty `observations`,
  `claims` and `next_investigations`. No finish is invented; nothing the
  model proposed in a refused finish enters F.

### 3.2 Separately validated assessments (not in the fingerprint)

- `ExecutionAssessment`: from the trace; every executed tool call matches a
  snapshot (id, tool, request, order), refusals and their kinds, model ids,
  the stopping reason, `finished`. Checked twice:
  - at the freeze point (Stage 6): the execution prefix, i.e. the trace up
    to and including the accepted finish, or up to report construction for
    an incomplete run (no `stop` event yet);
  - at render time (Stage 7): the final completed trace, which must extend
    that prefix and end with exactly one `stop` event whose stopping reason
    equals the report's (`finished` with an accepted finish, otherwise a
    non-`finished` reason and `finding: "incomplete"`).
  Status `consistent` / `inconsistent` (refuse).
- `AuthorizationAssessment`: per snapshot `rechecked` / `not_rechecked` /
  `refused`, plus the policy's manifest digest when a policy is given.

### 3.3 Order of checks at render time

1. Format dispatch (section 9).
2. L0, L1, L2 (if a policy is given), ExecutionAssessment.
3. Rebuild the context (L3 facts, coverage, tracking, predicates).
4. L4: every claim and observation re-validated against the rebuilt context.
5. Only then compute the fingerprint from the report's content and compare
   it with the recorded one; a mismatch refuses the report.

A fingerprint is never compared for content that failed revalidation.

### 3.4 Immutability

The context is built from a deep copy of the canonical JSON and recursively
frozen (dict -> `types.MappingProxyType`, list -> `tuple`); any mutation
raises. Accessors that return evidence for rendering return fresh deep
copies. Tests mutate every nested level and expect `TypeError`.

### 3.5 Context contents

`ValidatedContext` (frozen): format and versions; snapshots with L0/L1/L2
results; error snapshot ids; primary facts (status, effect value, recorded
effect sign, sufficiency, per-session rows, own unit); accepted direction
(section 5.1); identity and comparability per pair (the shared
`_data_identity_problem` / `_comparability` helpers); behavior-scoped
coverage (query groups; primary coverage via compare -> features -> episodes
-> query group with identity on every link); typed facts (section 6);
the two assessments.

### 3.6 Finish-time versus render-time tests

- Same report: Stage 6 fingerprint == Stage 7 fingerprint; also after a JSON
  round trip and after re-loading from disk.
- Changing any evidence, claim, observation, finding or primary id changes
  the fingerprint and refuses.
- Changing trace, usage or stopping reason leaves the fingerprint unchanged
  but fails the ExecutionAssessment where inconsistent.
- Rendering with a different policy changes only the AuthorizationAssessment.
- Incomplete runs: the fingerprint built at `stop()` equals the one Stage 7
  rebuilds (finding `incomplete`, null primary, empty lists); a refused
  finish's arguments never change it; adding a claim or observation to an
  incomplete report changes it and refuses; a trace whose final `stop` event
  is missing, duplicated or has a different stopping reason fails the
  render-time ExecutionAssessment while the fingerprint still matches.

## 4. v2 finish schema

```json
{"finding": "effect_described | insufficient_evidence | inconclusive",
 "primary_evidence_id": "ev:compare_windows:… | null",
 "observations": [{"evidence_id": "…", "field": "measurements.…"}],
 "claims": [{"type": "<TYPE>", "evidence_ids": ["…"], "params": {"<key>": "<enum>"}}],
 "next_investigations": [{"action": "<ACTION>", "params": {}, "because": [0]}]}
```
`additionalProperties: false` everywhere; no free strings. `direction`,
`interpretations`, `additional_limitations`, `next_steps`, `statement`,
`text` are argument errors.

## 5. Claim registry `claims/2.0`

### 5.1 Effect values, signs and direction

- `recorded_effect_sign`: sign of a compare snapshot's `effect.value` when
  not null. A property of evidence: shown in the evidence appendix, and
  outside it only as the accepted direction of `descriptive_effect` or the
  arithmetic relation of a valid sensitivity claim, both under
  `effect_described` (below).
- `accepted_direction`: equals the primary's recorded sign only when the
  accepted finding is `effect_described`; otherwise null.
- Executive summary: an aggregate effect value or direction appears only
  through the `descriptive_effect` sentence, and only under
  `effect_described`. Sensitivity sentences never appear in the executive
  summary.
- Findings section: under `effect_described` it may additionally show the
  two effects of a valid `sensitivity_excluding_flagged` claim (5.4).
- Under `insufficient_evidence` / `inconclusive`: no effect value or sign
  anywhere outside the evidence appendix. Observations of
  `measurements.effect.*`, `per_session[*].difference|pre_summary|baseline_summary`
  and window `value` fields are refused; sensitivity claims render status
  only (5.4). Raw snapshots stay in the appendix, labelled audit evidence.
- A sign relation between two recorded values is arithmetic only; it is
  never described as scientific or biological agreement.

### 5.2 Global coherence, cardinality, statuses

| finding | required | forbidden |
|---|---|---|
| `effect_described` | primary `compare_windows` status `ok`, sufficiency met (independent paired sessions), uncertainty `none`, every evidence layer passing; exactly one `descriptive_effect` citing it; primary coverage `complete` or `partial` (decision Q2; `unknown` refuses: the provenance chain is an evidence prerequisite); with `partial`, the result is scoped to the contributing sessions and the coverage is stated beside it (5.6) | status claims citing the primary; any wording implying the result represents the whole authorized split |
| `insufficient_evidence` | with any compare snapshot: a primary whose status is `insufficient_evidence`, `no_data` or `error`, cited by exactly one matching status claim (`sufficiency_not_met`, `comparison_no_data`, `comparison_unavailable` respectively); with no compare snapshot: null primary and `no_comparison` | an `ok` primary (use `effect_described` or `inconclusive`); `descriptive_effect`; effect values (5.1) |
| `inconclusive` | at least one evidence-grounded limitation (5.3); the primary may have any non-error status or be null | `descriptive_effect`; effect values (5.1) |
| incomplete (Stage 6 stop, no accepted finish) | null primary, no claims, observations or recommendations; Stage 7 renders system facts only (fingerprint, 3.1) | all claims |

Cardinality: <= 20 claims; <= 1 `descriptive_effect`; at most one
comparison-status claim (`sufficiency_not_met`, `comparison_no_data`,
`comparison_unavailable`) per cited comparison, and exactly one for the
primary under `insufficient_evidence`; an `ok` comparison has no status
claim type (its outcome is stated only by `descriptive_effect` when it is the
primary under `effect_described`); duplicates (type, sorted params, sorted
evidence_ids) refused; <= 3 `next_investigations`, each `because` resolving.

Exact statuses (anything else refuses):

| claim | evidence role: status |
|---|---|
| `sufficiency_not_met` | compare: `insufficient_evidence` |
| `comparison_no_data` | compare: `no_data` |
| `comparison_unavailable` | compare: `error` |
| `no_comparison` | none (context has no compare snapshot) |
| `descriptive_effect` | primary compare: `ok` |
| `sensitivity_excluding_flagged` | original, variant compare: `ok` / `insufficient_evidence` / `no_data`; quality: `ok` |
| `coverage` | per aspect (5.5) |
| `potential_confounder` | compare: `ok` / `insufficient_evidence` |
| `measurement_context` | compare: `ok` / `insufficient_evidence` / `no_data`; exception for `aspect = mover_unidentified` only: exactly one `get_window_features` result, status `ok` / `no_data` (below) |
| `tool_disclosure` | any non-error snapshot carrying the marker |
| `tracking_concern` | per kind (section 6) |
| `operational_fact` | history (no evidence ids) or quality `error` |
| `methodological_limitation` | any non-error snapshot carrying the item |

Error snapshots are citable only by `comparison_unavailable` and
`operational_fact(quality_check_failed)`; they never supply values.

### 5.3 Evidence-grounded limitations (for `inconclusive`)

Qualifying claims: `potential_confounder` (measured), `tracking_concern`
(measured or operational), `coverage` with status `partial` / `unknown`
or `sessions_without_baseline`, `sensitivity_excluding_flagged`,
`methodological_limitation(provisional_criteria)`. `tool_disclosure` and
`measurement_context` alone do not qualify.

### 5.4 `sensitivity_excluding_flagged` (only sensitivity variant in 2.0)

Roles, in order: `original`, `variant`, `quality`.

Request-level requirements:
1. Same feature, statistic, aggregation, sufficiency criteria, uncertainty.
2. `variant.baseline_window_ids == original.baseline_window_ids`.
3. `variant.pre_window_ids` is a strict subset of `original.pre_window_ids`.

Measurement-level requirements (from the compare measurements):
4. Used baselines identical: the same baseline window ids are used in both,
   per session, with identical `value`, `n_frames`, `n_usable`.
5. Every pre-event window used by both has identical `value`, `n_frames`,
   `n_usable`.
6. Exclusion outcomes unchanged, compared only for windows requested by
   both (`variant.pre_req ∪ baselines`): each such window excluded in one is
   excluded in the other with the same reason, and each used in one is used
   in the other. Windows requested only by the original are the removed
   windows (derived sets below), not compared here.
7. Window bounds: every window id parses (`ic.parse_window_id`) to the same
   episode, onset, length, gap (pre) or start, length (baseline); ids are the
   canonical bound encoding.
8. Source and processing: data identity original <-> variant,
   original <-> quality and variant <-> quality; between the original and
   variant comparisons only, complete equality of every `PROCESSING_METHOD`
   item and of the compare items `window_value`, `window_shape`,
   `feature_lookback_frames`, `min_usable_fraction`,
   `baseline_clearance_frames`. The quality snapshot is held to
   requirement 9, not to this list.
9. Quality configuration compatible: exactly these quality methodology
   items equal the original comparison's: `event_source`,
   `causal_computation`, `tracking_flags`, `score_validation`, `tracking_ok`,
   `interpolation`, `smoothing`, `smoother_memory_frames`, `fps` (all must
   be recorded on both); it assessed every removed requested window;
   each removed requested window is flagged there (contamination
   `detected` or verdict not `usable_with_caveats`).

Derived sets: `removed_requested = original.pre_req - variant.pre_req`;
`removed_contributing = removed_requested ∩ windows used by original`;
`removed_previously_excluded = removed_requested - removed_contributing`
(already excluded in the original, with their original exclusion reasons);
`sessions_pairing_changed` (paired in one only); `sessions_lost_all_pre`.

Rendering (findings section only; never the executive summary):
- finding `effect_described` and both `ok`: both effects with their own
  units; an arithmetic sign relation ("the two recorded values have the same
  arithmetic sign" / "opposite arithmetic signs" / "a zero value") only when
  both units are recorded and equal, else "sign relation not stated (units
  not both recorded or differ)"; always followed by "this is an arithmetic
  comparison of two descriptive values, not scientific agreement".
- `insufficient_evidence` / `inconclusive`, or either side not `ok`:
  statuses, unmet reasons and paired-session counts only; no effect values
  and no signs.
- always: removed contributing windows (ids, quality flags), removed
  previously excluded windows (ids, original exclusion reasons), changed
  sessions.

### 5.5 `coverage` (behavior-scoped; status always from the context)
Aspects: `primary_comparison` (no ids), `query_group` (one query),
`truncated_query` (query must be truncated), `window_exclusions` (features:
distinct excluded windows and overlapping reason counts),
`sessions_without_baseline` (primary per-session rows).

### 5.6 Status claims and `descriptive_effect` templates
- `sufficiency_not_met`: "The {feature} comparison ({eid}) did not meet the
  {criteria.status} sufficiency criteria (unmet: {reasons}; minimum
  {min_sessions} sessions, {min_windows_per_group} windows per group). It
  paired {n_sessions_paired} sessions from {n_windows_pre} pre-event and
  {n_windows_baseline} baseline windows. No aggregate effect is reported."
- `comparison_no_data`: "The {feature} comparison ({eid}) returned no data:
  {n_windows_pre} pre-event and {n_windows_baseline} baseline windows were
  usable. No aggregate effect is reported."
- `comparison_unavailable` / `no_comparison`: error codes / "No comparison
  was run."
- `descriptive_effect`: "Across the {n} contributing paired sessions
  ({contributing session ids}), the median of per-session differences
  (pre-event minus baseline) in {feature} was {value} {unit} ({direction}).
  Descriptive only: no interval or test; not a causal reading; it describes
  these sessions, not the whole authorized split." plus the cross-snapshot
  status when not `verified`. With primary coverage `partial`, the same
  sentence continues, inside the claim and never in a separate section:
  "Coverage: partial: {searched} of {eligible} eligible sessions were
  searched for {behavior} ({query group definition}; not searched:
  {session ids}); the unsearched sessions are not described." The executive
  summary uses the identical rendered sentence, so the scope and the
  coverage statement cannot be separated from the value.

### 5.7 Next investigations (2.0)
`check_tracking_quality {windows_from}` (refused if those windows were
already assessed), `exclude_flagged_windows {quality}` (requires flagged
windows in that quality result), `include_more_sessions {}` (only when
primary coverage is `partial`). Template: "Suggested by the model (not
evidence): …". Grid-based actions are deferred; `grid_version` is `"none"`
in 2.0 and reserved.

## 6. Facts: measured, disclosed, potential confounders, operational

Fact categories, each rendered with its category label:
- `measured_fact`: a value recorded in a snapshot.
- `tool_disclosure`: a limitation string the tool emits for every result;
  never evidence that the condition holds in this data.
- `potential_confounder`: a measured imbalance that could confound; never
  "confounding" as established.
- `operational_fact`: from execution history or coverage.

Value rules: an absent item, a `null` value, an unparseable string or a zero
denominator gives `not_assessed` for that predicate; nothing is guessed.
A measured value that contradicts a condition overrides the generic
disclosure: the disclosure is then rendered with "not supported by the
measured values of {eid}" and a claim of that condition is refused.

| claim | params | grounded in | outcome |
|---|---|---|---|
| `potential_confounder` | `investigation_label` | compare `label_frames_pre` / `label_frames_baseline` parsed to four counts | measured when investigation share pre > baseline; contradicted when <=; else not_assessed |
| `measurement_context` | `session_timing` | `window_start_s_pre` and `window_start_s_baseline` each parse as the producer's "min / median / max" seconds: exactly three finite, non-negative numbers, min <= median <= max; anything else (missing, reordered, negative, NaN, infinite, prose) is not_assessed | a measured fact grounded only in the parsed numbers (unit s, scope: the comparison's used windows), never the producer text; never a confounder (drift is not measured) |
| `measurement_context` | `pixel_units` | the compare's own `units` contains `px` | a measured fact about scale; not a confounder |
| `measurement_context` | `mover_unidentified` | aspect-specific exception: cites exactly one `get_window_features` result (status `ok` / `no_data`); `dist_rate` among its requested features; its methodology records `dist_rate_meaning` with exactly the marker "negative = body-centre distance decreasing"; its limitations carry the pair-level qualification that the measurement does not identify which mouse approached; a missing or altered marker or qualification refuses the claim | definitional fact rendered from fixed registry text (never the methodology prose); never justifies `inconclusive` on its own. Implemented at M2 (`investigation_claims.MOVER_MARKERS`; category `definitional_fact`; rendered from fixed registry text) |
| `tool_disclosure` | `approach_unlabelled` | query, features or compare limitation with the registry marker; dataset-scoped (query evidence allowed) | disclosure |
| `tool_disclosure` | `motion_accuracy_unverified` | compare / quality limitation marker | disclosure |
| `tracking_concern` | `smoothing_contamination`, `usable_fraction`, `tracking_exclusions`, `quality_verdict`, `quality_contamination_detected` | existing typed Stage 7 tracking items | measured; several items: with `evidence_id` those of that snapshot, else all, ordered primary chain, evidence order, source field, sorted window ids |
| `operational_fact` | `quality_never_called` | history: no `check_tracking_quality` call attempted (executed or refused) | operational |
| `operational_fact` | `quality_check_refused` | history: attempted, every attempt refused | operational |
| `operational_fact` | `quality_check_failed` | a quality snapshot with status `error` | operational |
| `operational_fact` | `model_selected_plan` | history: >= 1 tool call in a `model_turn` with a recorded model id | operational (model ids, call count) |
| `methodological_limitation` | `provisional_criteria`, `provisional_parameters`, `descriptive_only`, `single_feature`, `baseline_clearance_not_rechecked` | named methodology items / request fields | measured fact about method |

Stage 7 still lists every tracking item and coverage fact whether or not a
claim cites it.

## 7. Examples (from this branch's offline full-validation evidence)

Insufficient evidence (clearance 150, finding `insufficient_evidence`):
- `sufficiency_not_met` -> "The dist_rate comparison (ev:compare_windows:4116…) did not meet the provisional sufficiency criteria (unmet: min_sessions; minimum 4 sessions, 10 windows per group). It paired 2 sessions from 30 pre-event and 164 baseline windows. No aggregate effect is reported."
- `coverage(sessions_without_baseline)` -> "Sessions with pre-event windows but no eligible baseline window: mouse028_task1_annotator1, mouse045_task1_annotator1."

Descriptive comparison (clearance 30, finding `effect_described`, primary coverage complete):
- `descriptive_effect` -> "Across the 4 contributing paired sessions (mouse005_task1_annotator1, mouse028_task1_annotator1, mouse045_task1_annotator1, mouse064_task1_annotator1), the median of per-session differences (pre-event minus baseline) in dist_rate was -7.8209 px/s (negative). Descriptive only: no interval or test; not a causal reading; it describes these sessions, not the whole authorized split. Cross-snapshot agreement: not_verified." (The run's features <-> compare pair is `unit_unverified`, so the cross-snapshot clause is present; coverage is complete, so no coverage clause.)

Potential confounder:
- `potential_confounder(investigation_label)` -> "Potential confounder (measured in ev:compare_windows:…): pre-event label frames attack 0, investigation 101, mount 0, other 49; baseline attack 0, investigation 274, mount 0, other 3566. The difference may reflect investigation versus other behaviour rather than time before attack; this is not established."

Tracking:
- `tracking_concern(smoothing_contamination)` -> "Measured: 1 used pre-event window with smoothing contamination (ev:compare_windows:…, methodology.smoothing_contamination_windows_pre). Tool note: …"
- `operational_fact(quality_never_called)` -> "Operational: check_tracking_quality was never called in this investigation."

## 8. Model reasoning
`model_turn.text` stays in the JSON trace as `model_reasoning_unverified`;
never rendered in Markdown, never a finding, claim, coverage statement or
limitation. The v2 system prompt is generated from the registry.

## 9. Version dispatch

- `legacy/1`: `findings_format` absent AND none of `claims`,
  `next_investigations`, `fingerprint`, `registry_version`, `grid_version`
  present. Absent format with any of them: refuse ("mixed payload").
- `findings/2`: `findings_format == "findings/2"`; `registry_version`,
  `grid_version` and the report schema must each be in the known set
  (`KNOWN_REGISTRIES = {"claims/2.0"}`, `KNOWN_GRIDS = {"none"}`); any
  non-empty legacy prose field (`interpretations`, `next_steps`, `model`
  limitations, `direction` supplied by the model) refuses ("mixed payload").
- Unknown `findings_format`: refuse.
- Verification layers are reported separately in both formats, including
  `authorization: not_rechecked` without a policy.
- Historical rendering: a report is rendered with the registry and
  templates of its recorded `registry_version`; registries are kept
  versioned (`investigation_claims_v2_0`). Rendering under a newer registry
  needs an explicit migration artifact (old claims -> new claims, its own
  fingerprint, stored beside the original); the original is never
  re-rendered implicitly or rewritten.

## 10. Compatibility

| artifact / suite | handling |
|---|---|
| Original live Sonnet run (`outputs/investigation_live/20261019T220059Z_openrouter/`) | never rewritten; the four SHA-256 values pinned by a test; dispatched as `legacy/1`; `legacy_verified` (L0-L3 enforced, authorization reported); coverage `partial` (attack group, 3 of 14); prose quarantined as "legacy model-authored prose (unverified)", never in the summary; re-renders go to a new directory. |
| Stage 6 tests | fake investigators and scripted finishes move to v2; budget / termination / refusal / usage tests unchanged; lexical tests and the 11 xfail leaks move to a legacy-warning module. |
| Stage 7 tests | v2 fixtures; legacy fixtures and the live run assert `legacy_verified`; identity, coverage, tamper tests for both formats. |
| Stage 8 | unchanged except the dry-run emulator's finish (M3); `run.json` records format, registry, grid, schema, fingerprint. |

### 10.1 M2 rendering decisions (implemented; owner review pending)

Recorded so a reviewer can check them; none adds a scientific conclusion or
widens a claim's semantics.

- **Saved documents are untrusted.** A findings/2 document (schema 0.2)
  carries its `canonical` payload (evidence, trace, execution state,
  structured proposal, recorded fingerprint). `render_markdown_v2` /
  `verify_saved_v2` rebuild the whole document from that payload through the
  M0 context and the M1 registry (order 3.3, under the policy given at render
  time) and refuse any saved document that is not exactly the rebuild. No
  grounding, number, direction, sentence, summary, publication status,
  authorization label, fingerprint status, verification or policy statement
  is read from the saved document. The fingerprint is a consistency check
  over evidence and claims, not proof of origin; the trace, question and
  usage are not in F (3.1), so they are bound only by L0 (evidence) and the
  execution assessment, and are never rendered as findings.
- **Publication status.** `publishable`: authorization `rechecked` under a
  policy given at render time and a finished run with a validated finding.
  `diagnostic`: no policy (Q4), with a banner. `record_only`: an incomplete
  run under a policy; it never has a finding. A saved `publishable` document
  rendered without the policy is refused, never downgraded silently.
- **`definitional_fact`.** The category of
  `measurement_context(mover_unidentified)`: a definition the tool records
  (what `dist_rate` measures), rendered from fixed registry text; neither a
  measurement of this data nor a limitation that justifies `inconclusive`.
- **Legacy quarantine wrapper.** Legacy/1 reports render through the
  unchanged legacy renderer (JSON and Markdown byte-identical to the pre-M0
  baseline). `render_payload` returns them in a `RenderedReport` labelled
  `legacy_verified`, whose `quarantine` metadata and `notice` mark the
  interpretations, next steps and model limitations as "legacy model-authored
  prose (unverified)". The quarantine lives outside the legacy bytes: a
  consumer that exports only the inner legacy document or Markdown drops the
  label (the legacy Markdown still heads that prose "Model interpretations
  (not evidence; model-authored, unverified)" and "Suggested next steps
  (model suggestions, not evidence; unverified)", but does not use the
  quarantine wording). Exports should carry the wrapper's notice.
- **Untrusted text is not rendered.** Tool notes, methodology notes,
  limitation texts, `decided_by`, model reasoning, finish arguments and
  observation values that are free text appear in the JSON (canonical
  evidence / trace) but never in the Markdown; a textual observation is shown
  as "(text value; see the evidence snapshot)". Identifiers must match a
  closed grammar; context-generated texts (coverage reasons, eligibility
  bases, tracking facts) must match closed patterns; anything else refuses
  the report (`TemplateError`). The research question is operator text,
  escaped and labelled "not evidence".
- **Behaviour-specific confound wording.** `potential_confounder` says
  "rather than time before {behavior}" using the compared windows' single
  event behaviour (from the window ids), or "the event" when not single; the
  section 7 example ("attack") is the attack case of this template.
- **Partial-coverage reason.** The `descriptive_effect` coverage clause adds
  "; reason: {reason}" inside its parenthesis when the context records one
  (for example a truncated originating query), so a partial status is never
  shown without its cause.
- **Unit allowlist.** Units are rendered only from the known set
  (`investigation_compare.UNITS`: px, px/s, s, correlation (unitless)); a
  missing unit renders "(unit not recorded)". Any other unit text refuses the
  report through a controlled `ReportIntegrityError`, even when the registry
  accepted the claim (the registry checks evidence, the renderer the output
  grammar).
- **Controlled failures.** Every public entry point (`render_payload`,
  `build_report_v2`, `render_v2`, `verify_saved_v2`, `render_markdown_v2`)
  raises `ReportIntegrityError` for malformed input, never another exception.
- **Legacy byte-identity cases.** `scripts/legacy_golden.py` and
  `tests/test_legacy_golden.py` render 14 fixed cases against SHA-256 values
  computed with the pre-M0 source (c62711d), pinned in
  `tests/fixtures/legacy_golden/manifest.json`. Ten use tracked synthetic
  inputs; four (`real_full`, the live run, each with and without a policy)
  need git-ignored local inputs and are skipped without them.

### 10.2 M3 structured finish (implemented)

- `investigate(..., findings_format="findings/2")` is the default; legacy
  needs `findings_format="legacy/1"` (and `LivePlan.findings_format` /
  `--findings-format` in Stage 8). Each run sends only its own finish schema
  and system prompt; the findings/2 ones (`FINISH_SPEC_V2`,
  `SYSTEM_PROMPT_V2`) are generated from the registry (claim types, closed
  params, actions). The four scientific tools, their schemas, budgets,
  refusals and repeated-call rules are unchanged; nothing fixes a sequence.
- Freeze point: the accepted finish's proposal is validated by
  `validate_findings(mode="prefix")` on the trace up to and including that
  finish; the proposal, evidence bytes and prefix bytes are frozen with the
  fingerprint. A refused finish is returned to the model and never enters F.
- Stop: `finished_v2_problems` requires the completed trace to be the frozen
  prefix plus one stop event, the evidence byte-identical, the accepted
  finish's arguments equal to the proposal, and the complete-mode validation
  to reproduce the fingerprint; otherwise `FinishVerificationError` and no
  report (an inconsistent finish never becomes an accepted report).
  Incomplete runs: `incomplete_findings` only.
- Stage 7 (M2 renderer) now also binds a finished payload's accepted finish
  arguments to its proposal (`finish_arguments_problems`), and checks only
  the accepted finish for legacy prose (a refused attempt is recorded history
  and never rendered).
- Stage 8: the offline emulator chooses from the evidence (no episodes ->
  `no_comparison`; `ok` comparison -> `effect_described`; insufficient ->
  one re-measurement with a narrower baseline clearance, then whatever that
  evidence supports) and finishes with findings/2 claims.

## 11. Decisions on the open questions

Codex's answers, adopted:

- Q1: a named scientific owner approves versioned provisional methodology
  (grids, the confound share criterion, and any later method choices),
  recorded as `decided_by` with status `provisional`. Proposed owner: the
  supervising biology professor, pending the team's confirmation. The model
  can never alter the grid or the confound criteria; they are fixed by the
  registry / grid version the investigation runs under. (`claims/2.0`
  ships no grid actions, so this does not block M0-M4.)
- Q2: partial primary coverage does not by itself forbid `effect_described`.
  A descriptive effect is permitted when the independent-session,
  sufficiency, evidence and uncertainty prerequisites all pass (5.2). The
  claim is scoped to the contributing sessions, states the partial coverage
  directly beside the result (5.6), and never implies representativeness of
  the whole authorized split. `unknown` primary coverage still refuses.
- Q3: `inconclusive` requires an evidence-grounded limitation (5.3).
- Q4: reports rendered without a policy are diagnostic only, with
  `authorization: not_rechecked`; they are never presented as verified
  findings.
- Q5: historical registry versions are preserved; rendering under a newer
  registry only through a migration that creates a new artifact (section 9).

## 12. Implementation plan and checkpoints

- **M0 shared validation and pinned history.** Pin the four live artifacts
  (test). Add format dispatch (legacy detection only). Extract
  `investigation_context` (assessments, identity, comparability, coverage,
  typed facts, freezing, fingerprint function) from Stage 7 with no
  rendering change. Gate: legacy reports' JSON byte-identical before and
  after; freeze tests (3.4) on legacy fixtures; fingerprint-function tests
  (serialization, ordering, round trip, finished and incomplete inputs) on
  hand-built canonical F inputs (legacy reports carry no fingerprint).
- **M1 minimal registry and versioned representation.** `claims/2.0`
  types, predicates and templates; `InvestigationReport` v2 fields. Not
  wired. Gate: per-claim unit tests (section 12 list below).
- **M2 legacy quarantine and v2 rendering.** Stage 7 dispatch, legacy
  renderer, v2 renderer (schema 0.2), revalidate-then-fingerprint. Gate:
  hand-built v2 reports render deterministically; mixed / unknown payloads
  refused; legacy live run `legacy_verified`.
- **M3 atomic Stage 6 and Stage 8 switch.** v2 finish schema, registry-built
  prompt, `verify_final` via the shared context, fingerprint at the freeze
  point, Stage 8 dry-run emulator emitting v2, fake investigators migrated,
  in one change. Gate: offline dry run on the full validation cache ends
  `finished` with a v2 report whose Stage 6 and Stage 7 fingerprints match.
- **M4 remove lexical checks from v2.** Only after M3 passes; lexical code
  kept for legacy warnings. Gate: no v2 path imports it; template
  vocabulary test.

Tests per claim: valid case with exact sentence; missing role; wrong status;
error snapshot; contradicted predicate; not_assessed (absent, null,
unparseable); forged re-hashed evidence refused at render; unknown params.
Sensitivity: each of requirements 1-9 violated once; requirement 6 ignores
windows requested only by the original; requirement 8 not applied to the
quality snapshot; status-only rendering (no values, no signs) under
non-effect findings; units missing or different; removed contributing vs
removed previously excluded; sensitivity sentences absent from the executive
summary; the only occurrence of "agreement" in sign wording is the fixed
negated disclaimer. Global: every coherence row,
including `insufficient_evidence` with an `ok` primary refused, `ok`
primary under `effect_described` accepted without any status claim, two
status claims on one comparison refused, `insufficient_evidence` without the
primary's status claim refused; cardinality; leakage of effect values
through observations or sensitivity under non-effect findings refused;
incomplete-run fingerprint and execution prefix / final trace (3.6).

Q2 acceptance tests: `effect_described` with complete primary coverage is
accepted and names the contributing sessions; with partial coverage (a
restricted query, or an explicit partial cache) it is accepted only when
every prerequisite passes, and the rendered claim contains the contributing
session ids, "Coverage: partial", searched / eligible counts and the
not-searched ids in the same sentence, identically in the executive summary
and the findings; with `unknown` primary coverage it is refused; a partial
case that fails sufficiency is refused as `effect_described`; no template
under partial coverage contains wording about the whole split (template
vocabulary test).

## 13. Deferred

Alternative-feature sensitivity; window-size and clearance sensitivity;
grid-based and multi-step next-investigation planning (`vary_window`,
`vary_baseline_clearance`, `compare_feature`, `control_confound`); intervals,
tests, population inference; cross-behavior contrasts and confound
adjustment; claims over classifier or monitor sources; signed provenance;
training-split evidence; LLM-written summaries.

## 14. Genuinely unresolved

1. The team's confirmation of the proposed scientific owner (Q1).
2. That owner's approval of the provisional methodology this registry fixes,
   including `potential_confounder(investigation_label)`'s strict share rule
   (pre > baseline).
