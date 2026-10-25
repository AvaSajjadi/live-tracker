# Syed: agent, monitor, replay dashboard

You own what the researcher sees and talks to: turning a sentence into a rule,
the Confirm step, the monitor that turns scores into alerts, the replay page,
the event log, and the end-of-session summary.
Overview and handoff table: `plan/00_TEAM_PLAN.md`.

**You deliver:** H5 (full end-to-end run, Day 4 end)
**You wait for:** H1 (loader, for skeleton drawing ← Igor), H2b (tracking warnings ← Ava),
H3 (real test-session scores ← Igor)

**You can start right away.** Build everything against fake scores first; real data plugs in later.

Your files: `src/live_tracker/monitor.py`, `agent.py`, `summary.py`, `fake_scores.py`,
`pages/` (Streamlit pages), and their tests. `rules.py` already exists; you are its main user.
Tell the group before changing it.
Do **not** edit Ava's motion-engine files (`motion.py`, `filters.py`, `quality.py`, `features.py`).

---

## Day 1: Sat Oct 17 (no dependencies)

### 1. Team setup (H0, with everyone, 1 h)
- [ ] Clone, venv, `python -m pytest -q` passes (53 tests expected)
- [ ] Merge PR #2; agree on interfaces

### 2. `fake_scores.py`: synthetic score file
- [x] Makes a table in the `INTERFACES.md` §2 format (`session, frame, time_s, attack, mount, investigation, tracking_ok`)
- [x] Includes a clear attack bout (> 2 s, high score), an uncertain one (scores in the "ask" range), a short one (< 2 s, must not alert), and a `tracking_ok = false` stretch
- [x] `session` name starts with `SYNTHETIC_` so it can never be mistaken for real data

### 3. `monitor.py`: rules + scores → events
- [ ] Input: approved rules (`RuleSet.active()` from `rules.py`) + scores, **one frame at a time** (streaming, like live)
- [ ] An event starts when the score stays above `ask_threshold` for `min_duration_s`; it fires at **that moment**, not at the end of the bout (alert latency matters)
- [ ] Confidence = mean score over the bout so far; ≥ `alert_threshold` → `alert`, else `ask`; `alert_mode: "log"` → just log
- [ ] Frames with `tracking_ok = false` don't count as behavior; they end or pause a bout (write down which)
- [ ] Writes the event log as JSONL in the `INTERFACES.md` §4 format, with evidence (`mean_score`, `min_score`, `frames`)
- [x] Only approved rules are ever used (test this!)
- [x] Tests with `fake_scores`: right number of events, right start times, the short bout ignored, the uncertain one is `ask`

---

## Day 2: Sun Oct 18

### 4. `agent.py`: sentence → proposed rule
- [x] **No-key fallback first** (the demo must work offline): a simple parser for sentences like
  "alert me if the mice attack for at least 2 seconds" → behavior word + duration + mode
- [ ] Then the Claude API path: tool use with a tool whose input schema matches the `Rule` fields;
  the result **always** goes through `rules.validate_rule()`; key from `ANTHROPIC_API_KEY` in `.env`, model name from an env var
- [ ] If the sentence is unclear, return a question to the researcher instead of guessing
- [ ] The LLM never produces scores, counts, durations, or events, only the proposed rule and explanations of logged events
- [ ] Tests run **without** a key (fallback path; mock the API call)

### 5. Rule confirmation page: `pages/1_Rules.py`
- [x] Text box → proposed rule shown as readable text + JSON
- [x] Confirm / Reject buttons → `RuleSet.approve()` / `reject()`
- [x] Monitoring can't start while no rule is approved

### ⏳ Needs H1 (loader from Igor, end of Day 1)
### 6. Skeleton drawing of a real session
- [ ] Draw both mice's keypoints + connecting lines for one frame, using the keypoint order in `docs/calms21_schema.md`
- [ ] Fast enough to redraw many times per second (try matplotlib vs plotly; measure it)

---

## Day 3: Mon Oct 19

### 7. Replay page: `pages/2_Replay.py`
- [ ] Banner: **"Replay of a recorded session, not a live camera"**
- [ ] Plays the score file at the recorded speed (use `time_s`; allow 1×, 2×, 5× for the demo)
- [ ] Shows: skeletons, current scores, active rules, alerts as they fire, timeline of events
- [ ] "Ask" events show Confirm / Reject → fills `researcher_answer` in the log
- [ ] Researcher stays in control: the system only alerts and logs, it never acts on the experiment

### ⏳ Needs H2b (tracking warnings from Ava, end of Day 2)
- [ ] Show tracking warnings in the replay and write them to the event log (`action: "tracking_warning"`)

### ⏳ Needs H3 (real test-session scores from Igor, end of Day 3)
- [ ] Switch replay from fake scores to real scores; keep the fake option, labeled synthetic

---

## Day 4: Tue Oct 20 (prototype day)

### 8. `summary.py`: end-of-session summary
- [x] Deterministic part: counts, total durations, times, confidences, all computed **from the event log only**
- [ ] Optional Claude wording on top, given only the log; then check that every number in the text appears in the log (test this). If the check fails, show the deterministic summary instead
- [x] Works without a key

### 9. → **H5: full pipeline on one real held-out session**
Sentence → proposed rule → Confirm → replay → alerts / asks / tracking warnings → event log → summary.
- [x] One command starts the dashboard: `python -m streamlit run app.py`
- [ ] Use Igor's validated thresholds for the demo rule
- [ ] Write down every bug found and who fixes it

---

## Day 5: Wed Oct 21

- [x] Offline demo path: no network, no API key (fallback parser + saved score file)
- [ ] Record a backup screen video of the full demo
- [x] README: how to run the dashboard, `.env` / API key setup, what "replay" means
- [ ] Slides: demo section; rehearse with the team

## Done when
- A researcher can type a sentence, confirm the rule, watch a replay, get alerts with time + confidence + evidence, answer "ask" events, and read a summary that only uses logged facts
- Everything works without an API key
- Monitor and summary have tests

## UI-5 application polish (2026-10-25)

- [x] Retain Ava's terracotta/sage theme; share semantic colors and mobile layout rules (`ui_style.py`).
- [x] Label recorded-camera simulation in Upload's main success message; preserve source choice on return.
- [x] Propose → review assumptions/thresholds → Confirm / Reject; only approved rules enable Next.
- [x] Shared red ALERT / amber ASK notices and styled event tables on Real session and Run.
- [x] Group optional schedules, validation exploration and technical details without hiding errors or scientific limits.
- [x] Correct synthetic-only README wording and document local artifacts and offline rehearsal (`docs/demo_readiness.md`).
- [ ] Complete H5 on one real held-out session (team agreement; Ava runs the frozen test protocol).
- [ ] Transfer and verify demo artifacts on the selected presentation computer.
- [ ] Produce and review a backup screen recording (artifact not established here).
- [ ] Produce slides and rehearse with the team (artifacts not established here).

Evidence: synthetic AppTest coverage in `test_flow.py`, `test_rules_page.py`, `test_live_view.py`, `test_run_page.py`,
`test_app.py` and shared styling tests; monitor/workflow/summary tests verify the checked implementation milestones.
The checked startup and offline items under H5/Day 5 mean the software path exists, not that H5 or a rehearsal on the
presentation computer has been completed. Earlier unverified team milestones remain unchecked.
