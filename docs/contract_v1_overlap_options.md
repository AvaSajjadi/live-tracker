# Contract v1: what should "overlaps another bout" mean? (dry run, 2026-10-22)

Counts only. No effect value was computed, and none should be looked at before the rules below are
chosen. Validation sessions only; the test split was not read.

Reproduce: `python scripts/dryrun_contract_options.py` (ground-truth labels) and
`python scripts/dryrun_contract_options.py --score-drop <zip> --member scores_validation_model_b.csv`
(classifier scores). Source: Syed's `investigation_tools_v1` adapter, `quality_v1`, no interpolation,
baseline 5 s, gap 1 s, attack bouts of at least 1.0 s at threshold 0.5, `dist_center`.
Gate: at least 10 used windows in at least 4 sessions.

## The two open choices (docs/contract_v1_integration.md section 6, items 3 and 4)

1. **Pre-window overlap.** Which "other bouts" exclude a pre-event window?
   `fired` = other bouts of the cited query (a run above threshold for at least `min_duration_s`).
   `any_run` = any run of frames at or above the threshold, however short.
2. **Baseline overlap.** Is a baseline window that contains a bout excluded? Today: no.

## Check that the dry run is right

Variant V2 (`any_run`, baseline not checked) on ground-truth labels gives **33 / 21 / 14** usable
windows at pre_s 1 / 2 / 5 s, with overlap 5 / 16 / 24 and edge 2 / 3 / 2. That is exactly what
`scripts/compare_quality_policies.py` reports, so the definitions are implemented as intended.
Tracking quality excludes no window in either source ("bad" = 0 everywhere).

## Result A: ground-truth labels (an upper bound; 40 attack bouts in 4 sessions)

| variant | pre_s | used | sessions | gate |
|---|---|---|---|---|
| V0 fired, baseline not checked (**as built**) | 1 / 2 / 5 | 35 / 27 / 18 | 4 / 4 / 4 | pass / pass / pass |
| V1 fired, baseline fired | 1 / 2 / 5 | 16 / 14 / 7 | 4 / 4 / 2 | pass / pass **FAIL** |
| V2 any_run, baseline not checked | 1 / 2 / 5 | 33 / 21 / 14 | 4 / 4 / 4 | pass / pass / pass |
| V3 any_run, baseline any_run | 1 / 2 / 5 | 11 / 9 / 6 | 3 / 3 / 2 | **FAIL** / **FAIL** / **FAIL** |

## Result B: classifier scores (what the demo uses; model B refit, 40 predicted bouts in 6 sessions)

| variant | pre_s | used | sessions | gate |
|---|---|---|---|---|
| V0 fired, baseline not checked (**as built**) | 1 / 2 / 5 | 29 / 25 / 20 | 6 / 6 / 6 | pass / pass / pass |
| V1 fired, baseline fired | 1 / 2 / 5 | 19 / 19 / 14 | 6 / 6 / 6 | pass / pass / pass |
| V2 any_run, baseline not checked | 1 / 2 / 5 | 3 / 2 / 0 | 3 / 2 / 0 | **FAIL** (all) |
| V3 any_run, baseline any_run | 1 / 2 / 5 | 0 / 0 / 0 | 0 / 0 / 0 | **FAIL** (all) |

The classifier scores are my refit of Igor's model B (his code, seed 0, validation macro F1 0.651, the same as
the ablation), not his saved artifact. The 0.5 threshold and 1.0 s minimum are placeholders; his final
values will change these counts.

## What this shows

- **Item 3 (pre-window overlap).**
  - On labels the choice moves 2, 6 and 4 windows (35 vs 33, 27 vs 21, 18 vs 14). Small, and the gate
    passes either way. `fired` keeps windows that contain a short (under 1 s) burst of the behavior.
  - On classifier scores `any_run` is **unusable**: noisy probabilities cross 0.5 in scattered frames next
    to nearly every bout, so 37 to 40 of 40 windows are excluded. A single-frame blip is not a bout.
  - So `fired` is the only definition that works for prediction sources. Keep it as built, and say in the
    contract that short blips do not count as bouts.
- **Item 4 (baseline overlap).** This is the choice that matters. Excluding baselines that contain a bout
  removes 6 to 19 windows with `fired` (labels: 19, 13, 11 at pre_s 1 / 2 / 5; classifier: 10, 6, 6).
  - With `fired` it still passes the gate at pre_s 1 and 2 in both sources. The pre_s 2 s preset
    gives 14 windows in 4 sessions on labels (a narrow pass) and 19 in 6 sessions on classifier scores.
  - At pre_s 5 s it fails on labels (7 windows in 2 sessions). That is an honest
    `insufficient_evidence`, and the report lists every attempt anyway.
  - The strictest variant (V3) fails everywhere on labels, so a rule built on it would make the
    question unanswerable. It is not needed: V1 already excludes baselines containing a detected bout.
- **Why windows are lost.** The look-back (pre window, 1 s gap and 5 s baseline) is 7 to 11 s, so a bout
  that follows another detected bout inside that span loses its window. That is consistent with attacks
  coming in clusters; this dry run did not measure the clustering itself. The 4-session gate is tight
  because only four validation sessions contain attack.

## Recommendation (to be confirmed by Ava before it goes into the contract)

- **Item 3:** keep `fired` for every source. State that sub-threshold or too-short runs are not bouts.
- **Item 4:** exclude baselines that contain a `fired` bout (V1), as an explicit, named setting in the
  report, not a silent change. It is the scientifically defensible rule and it still leaves the gate
  passable at the default preset.
- **Do not** shorten or move the baseline, or change the gate, to rescue pre_s 5 s or the labels result.
  Per the contract, the grid, gate and quality policy are fixed before effect values are examined, and
  that is what keeps the analysis honest.
- Re-run this script once Igor's final threshold and minimum duration are known; the counts depend on them.

## Limits of this dry run

- One behavior (attack), one feature (`dist_center`, only to confirm values exist), one threshold.
- Ground-truth labels are an upper bound; predicted bouts include false alerts, so `used` windows are not
  all real attacks.
- The counts are windows of the adapter's own rules and may shift if Syed changes the adapter.
