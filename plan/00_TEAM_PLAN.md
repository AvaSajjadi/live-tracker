# Team plan: 5 days (Oct 17 to Oct 21, 2026)

Three people, full time. The goal is a working end-to-end prototype on **Day 4 (Oct 20)**,
which is the proposal's prototype date, and a clean, documented, demo-ready repo on **Day 5 (Oct 21)**.
Submission is Oct 25, so Oct 22 to 25 are spare time for problems.

| Person | Area | Personal file |
|---|---|---|
| **Ava** | ScopeScore-inspired motion engine: causal features, tracking quality, tuning, sensitivity tests, feature ablation | `plan/AVA.md` |
| **Igor** | CalMS21 data and model: loader, splits, classifier, evaluation, per-frame scores | `plan/IGOR.md` |
| **Syed** | Agent and interface: rules from plain English, monitor, replay dashboard, event log, summary | `plan/SYED.md` |

**All ScopeScore-related work belongs to Ava.** Igor and Syed use her functions but
do not change `motion.py`, `filters.py`, `quality.py` or `features.py` without asking her.

## How the parts connect

```
           Igor                       Ava                          Syed
  CalMS21 files ─► loader (long table) ─► features + tracking_ok ─┐
                         │                                       │
                         └─► classifier ◄── feature table ────────┘
                               │
                               └─► per-frame scores (test sessions) ─► monitor ─► event log ─► replay + summary
                                                                       ▲
                                         researcher sentence ─► agent ─┘ (rule, after Confirm)
```

## Handoffs (who waits for whom)

These are the points where one person cannot continue until another person delivers.
"Delivered" means **merged into `main`** (or a file on the shared Drive, if marked).

| ID | When | From → To | What is handed over | Who is blocked without it |
|---|---|---|---|---|
| **H0** | Day 1, first hour | everyone | Setup done, PR #2 merged, new interface sections agreed in `INTERFACES.md` | everyone |
| **H1a** | Day 1, midday | Igor → Ava, Syed | `docs/calms21_schema.md`: keypoint names and order, mouse IDs, fps, label numbers, file layout (from the real files and the official readme) | Ava (body center, dyadic features), Syed (skeleton drawing) |
| **H1** | Day 1, end | Igor → Ava, Syed | `load_calms21()` returns the long table + labels table; `splits/split_manifest.json` (by session, fixed seed) | Ava's Day 2, Syed's skeleton replay |
| **H2** | Day 2, end | Ava → Igor | `compute_features(long_df)` returns the feature table (one row per session × frame) + `frame_tracking_ok()` | Igor's model B (Day 3) |
| **H2b** | Day 2, end | Ava → Syed | `tracking_warning_events()` returns tracking warnings in the event-log format (`action: tracking_warning`) | Syed's tracking warnings in the replay |
| **H3** | Day 3, end | Igor → Syed | Per-frame score file for the held-out test sessions (`INTERFACES.md` §2), produced with causal features only | Syed's real-data replay |
| **H4** | Day 3, end | Igor → Ava | `evaluate()` function + model A (basic features) results on validation | Ava's ablation (Day 4) |
| **H5** | Day 4, end | Syed (with all) | Full pipeline runs on one real held-out session: sentence → rule → Confirm → replay → alerts → summary | Day 5 demo and report |

### Handoff status (updated Sat Oct 17, Ava's side)

| ID | Status |
|---|---|
| H0 | Partly: Ava's tests pass (53 then, 87 now). PR #2 and #3 still open. |
| H1a | Not seen on GitHub yet. **Ava is waiting** (keypoint names and order, torso keypoints, mouse IDs, fps). |
| H1 | Not seen yet. **Ava is waiting** (loader and split manifest). |
| H2 | **Delivered, in `main`** (#4, #5): `compute_features()`, `frame_tracking_ok()`. Synthetic data only. |
| H2b | **Delivered, in `main`** (#5): `tracking_warning_events()`. |
| H3 | Igor to Syed: pending. |
| H4 | Igor to Ava: pending. Ava's `scripts/ablation.py` is ready (#7) and needs `fit_fn` / `evaluate_fn`. |
| H5 | Pending. |

**Ava, done so far:** feature table spec (`INTERFACES.md` §5), causal individual and dyadic features, body center (needs the real names), tracking warnings, sensitivity script (#6), ablation script (#7), status doc (#8). Details and checkboxes in `plan/AVA.md`.

**Ava needs:**
- **Igor:** H1a, then H1, then H4 (details in `plan/AVA.md`).
- **Syed:** feedback on `INTERFACES.md` §5 and the warning format; replay needs from `tracking_warning_events()`.
- **Everyone:** reviews of #4–#8, and merge #2 and #3.

**Rule:** if your handoff is late, tell the group chat **before** the deadline, not after.
Every blocked person has a "while you wait" task in their own file, so nobody sits idle.

## Day-by-day order

### Day 1: Sat Oct 17 (foundations; nobody is blocked yet)
| Order | Ava | Igor | Syed |
|---|---|---|---|
| 1 | H0 together (1 h): clone, venv, tests pass, merge PR #2, agree on interfaces | H0 | H0 |
| 2 | Write the "Feature table" section in `INTERFACES.md` | Download CalMS21 Task 1 **pose + labels only** (no videos), read the readme, run `calms21_inspect` | Write the "Score file" usage + event log notes; build `fake_scores.py` (labeled synthetic) |
| 3 | `features.py`: causal individual features (speed, heading, turning, straightness, pauses) + tests, using synthetic data | **H1a** schema notes by midday | `monitor.py`: streaming rule checking on per-frame scores → events + JSONL event log + tests |
| 4 | Causality test (changing future frames must not change past features) | Loader + labels + split manifest + tests → **H1** | Continue monitor (alert / ask / log, event timing) |

### Day 2: Sun Oct 18 (real data enters)
| Order | Ava | Igor | Syed |
|---|---|---|---|
| 1 | ⏳ needs **H1**: body center from torso keypoints (not the nose) | "Basic" features (plain geometry, no ScopeScore additions) | `agent.py`: Claude rule proposal + no-key fallback parser |
| 2 | Dyadic features: distance, approach rate, nose-to-nose, nose-to-tail, facing angle, proximity time, synchrony | Model A (baseline) trained on train sessions, tuned on validation | Rule confirmation page (Confirm / Reject; unapproved rules never run) |
| 3 | `frame_tracking_ok()` + `tracking_warning_events()` → **H2, H2b** | `evaluate()`: per-class P/R/F1, confusion matrix, event-level metrics, latency, false alerts/hour | ⏳ needs **H1**: skeleton drawing of a real session |

### Day 3: Mon Oct 19 (models and live replay)
| Order | Ava | Igor | Syed |
|---|---|---|---|
| 1 | Tune quality thresholds and filter settings on **train/val only** | ⏳ needs **H2**: model B = basic + Ava's features | Replay page at real recorded speed, using fake scores |
| 2 | Sensitivity tests: jitter, short gaps, smoothing, thresholds | Choose model on validation; calibrate scores | Researcher "ask" buttons fill `researcher_answer`; ⏳ needs **H2b**: tracking warnings shown |
| 3 | Remove Savitzky-Golay from every live code path | Export test-session scores → **H3**; send `evaluate()` + model A results → **H4** | ⏳ needs **H3**: switch replay to real scores |

### Day 4: Tue Oct 20 (end-to-end prototype; proposal deadline)
| Order | Ava | Igor | Syed |
|---|---|---|---|
| 1 | ⏳ needs **H4**: ablation, model A vs model B, chosen on validation, test reported once | Pick default alert/ask thresholds on validation for the demo rule | End-of-session summary written only from the event log |
| 2 | Motion-engine page in `app.py` shows real CalMS21 features + warnings | Save model artifacts, parameters, split manifest; one training command | **H5**: full pipeline on one real held-out session |
| 3 | Review Syed's and Igor's PRs | Bug fixes from integration | Bug fixes from integration |

### Day 5: Wed Oct 21 (freeze, document, demo)
| Order | Ava | Igor | Syed |
|---|---|---|---|
| 1 | README "ScopeScore adaptations" section, true to what was built | README data + training + evaluation section | README dashboard + API-key section |
| 2 | Motion-engine part of the test report and slides | Results part of the report (measured numbers only) | Offline demo path (no network, no key) + backup screen recording |
| 3 | Final full test run on Windows; update `IMPLEMENTATION_STATUS.md` | Final check: no test labels used anywhere in training/tuning/agent | Rehearse the demo with everyone |

## Who owns which files (to avoid merge conflicts)

| Owner | Files |
|---|---|
| Ava | `src/live_tracker/motion.py`, `filters.py`, `quality.py`, `features.py`, `app.py` (motion-engine page), `scripts/sensitivity.py`, `scripts/ablation.py`, tests for these |
| Igor | `src/live_tracker/calms21.py`, `calms21_inspect.py`, `basic_features.py`, `classify.py`, `evaluate.py`, `splits/`, `docs/calms21_schema.md`, tests for these |
| Syed | `src/live_tracker/monitor.py`, `agent.py`, `summary.py`, `fake_scores.py`, `pages/` (Streamlit replay/monitor pages), `rules.py` (shared, see below), tests for these |
| Shared (tell the group before editing) | `INTERFACES.md`, `README.md`, `IMPLEMENTATION_STATUS.md`, `requirements.txt`, `events.py`, `rules.py` |

## Rules for everyone (from `CLAUDE_START_HERE.md`)

- Branch + PR for every change, one teammate reviews, tests must be green. Never push to `main`.
- Split CalMS21 **by session**. Test-session labels are never used for training, tuning, or by the agent.
- Live replay uses **causal** processing only (no future frames).
- Never claim something works until a test or a measured result shows it. No made-up numbers.
- Missing tracking is not "the mouse is still". Flag it, don't hide it.
- No ScopeScore surgical thresholds (e.g. 5 mm, 0.5 s) used as mouse defaults.
- No data, videos, or API keys in Git.

## Daily check-ins
- **Morning (10 min):** what I finished, what I do today, which handoff I am waiting for or delivering.
- **Evening (10 min):** were today's handoffs merged? If not, what is the fallback for tomorrow?

## Fallbacks if something is late
| Late | Fallback |
|---|---|
| H1 (loader) | Ava keeps building on synthetic data and helps Igor with the readme; Syed keeps using fake scores |
| H2 (features) | Igor ships H3 with model A only; model B scores replace them when ready |
| H3 (scores) | Syed demos with fake scores (clearly labeled) while Igor finishes |
| Claude API problems | The no-key fallback parser is the demo path |
