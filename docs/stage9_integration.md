# Stage 9 S9-6: full offline integration

One command runs the whole Stage 9 chain offline (no network, no API key, no
paid call):

```
verified validation score ZIP -> score ingestion (S9-1a) -> monitor journal with
source binding (S9-2) -> frozen, verified snapshot -> classifier_prediction and
monitor_event tools (S9-3, S9-4) -> bounded Stage 6 investigation driven by the
Stage 8 offline emulator -> Findings v2.1 (S9-5) -> saved JSON + Markdown,
re-verified from the saved JSON
```

## Command

```
python scripts/stage9_integration.py <score-drop.zip> --out <new-dir> \
    [--archive-sha256 <hex>] [--member scores_validation_model_b.csv] \
    [--cache-dir data/raw/calms21_validation] [--manifest splits/split_manifest.json]
```

- `<score-drop.zip>`: Igor's validation score drop (read in place, never
  extracted; the test member is refused unopened).
- `--out`: a new or empty directory outside the repository or git-ignored
  (e.g. `outputs/stage9_integration_<date>`); nothing is overwritten.
- `--archive-sha256`: the archive digest received separately (recommended).
- The verified validation pose cache (`python -m live_tracker.investigation_data
  --all`) is needed for the windows; its annotations are not read for these runs.

Exit status: 0 every step verified; 1 a step failed (recorded in the summary);
2 refused inputs (missing ZIP, cache or manifest; output directory not allowed).

## Artifacts (in `--out`)

| File | Content |
|---|---|
| `events.journal.jsonl` | monitor journal: run header (verified_load), events, run_finished |
| `journal_snapshot.json` | canonical frozen snapshot (its SHA-256 is in the summary) |
| `<source>_investigation.json` | the Stage 6 InvestigationReportV2 (evidence, trace, fingerprint) |
| `<source>_report.json` / `.md` | Findings v2.1 document and Markdown, re-verified from the saved JSON |
| `integration_summary.json` | inputs' provenance, every step's status, per-investigation outcome |

`<source>` is `classifier_prediction` and `monitor_event`: two separate
investigations, never one mixed findings chain.

## Result on Igor's validation drop (model B, 2026-10-20)

All steps verified (63 s). Monitor: 7 events (LOCAL_SMOKE_TEST), snapshot
`verified_score_drop` / `complete`. Both investigations ran query_events,
get_window_features, check_tracking_quality and compare_windows twice (the
emulator re-measures with a narrower baseline clearance) and finished
`insufficient_evidence` under the provisional sufficiency criteria
(classifier: 4 paired sessions but 7 pre-event windows < 10; monitor: 2
sessions), status `not_authorized`. Provider-reported cost $0.00.

## Limitations

- **LOCAL_SMOKE_TEST** (attack, 2.0 s, ask 0.5, alert 0.8) is a labelled
  placeholder rule, approved locally for the demonstration only; it is not a
  validated monitoring rule. The classifier source threshold (0.5) and minimum
  (15 frames) are provisional too.
- **The investigator is the Stage 8 offline emulator**, a deterministic script
  that adapts to tool results (re-measures, retries oversized window requests
  with longer windows, caps baselines at 25 per session so compare_windows stays
  under the contract's 64 kB request size). It is not a scientific agent.
- **Not publishable**: claims/2.1 is not publication-authorized; reports are
  `not_authorized` (finished) or `record_only` (incomplete).
- Model identity is unverified (no model artifact or commit was supplied);
  validation scores are optimistic (the model was tuned on these sessions);
  annotated label composition and the investigation confound are not assessed.
- Saved-document verification is consistency checking, not authentication
  (no signatures).
