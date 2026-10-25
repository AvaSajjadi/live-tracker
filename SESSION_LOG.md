# Session Log

## 2026-10-24

### 1. What we did
1. Opened the repo folder. There was no log yet.
2. Fetched from GitHub. Syed had 2 new commits on `syed/final-validation-integration` (Windows lock and safe run loading).
3. Tried to merge them. There were 2 conflicts (`investigation_v1_runs.py`, `test_live_agent_page.py`). Stopped and undid the merge.
4. Tested Syed's code in a separate copy. All tests passed.
5. Merged again. Used Syed's version of both conflict files. Committed and pushed (`3a18947`).
6. Ran the full test suite: 4,287 passed, 0 failed.
7. Fetched again. Syed had new UI branches (`ui2-review`, `ui3-review`). `ui3-review` includes `ui2-review`.
8. Merged `ui3-review`. No conflicts. Committed and pushed (`5e40e58`).
9. The new tests passed (41 passed, 3 skipped). A full test run on the UI merge was started, and it was still running at 21% with no failures when we committed.

### 2. Commands we ran
- `git fetch`: gets new commits from GitHub. Changes no files.
- `git merge --no-commit --no-ff <branch>`: tries a merge but stops before saving it.
- `git merge --abort`: undoes a merge that is not finished.
- `git checkout --theirs <file>`: takes Syed's version of a file in a conflict.
- `git worktree add` / `git worktree prune`: makes or cleans a separate copy of the code for testing.
- `PYTHONPATH=src python -m pytest -q`: runs the tests. `PYTHONPATH=src` tells Python where the code is.
- `git commit`, `git push`: save the merge and send it to GitHub.

### 3. Files we changed, and why
- `src/live_tracker/investigation_v1_runs.py` and `tests/test_live_agent_page.py`: took Syed's version. His lock code and test guard do the same job as ours, and are stricter.
- The UI merge added: `pages/5_Live_Agent.py`, `investigation_v1_ui.py`, `investigation_agent_v1.py`, `investigation_v1_runs.py`, `scripts/investigation_v1_live.py` and new tests.
- `SESSION_LOG.md`: this file.

### 4. Problems and fixes
- Tests failed with "No module named live_tracker". Fix: run with `PYTHONPATH=src`.
- One full run (without `PYTHONPATH=src`, in the copy) had 304 failures and 851 errors. We did not find the exact cause. The same code passed when run correctly.
- Windows would not delete the temp folder at first. It was deleted later.

### 5. Not finished
- The full test run on the UI merge (`5e40e58`) was not finished when we committed. Check its result.
- Untracked files are not committed on purpose: `.whl` files (big binaries) and `reply_to_syed*.txt`.
- Branches `syed/day34-replay` and `syed/experimental-investigation` show new commits. We did not look at them.

### 6. Next time
1. Check the result of the full test run on `5e40e58`. If tests fail, find out why.
2. Look at the two branches in section 5.
3. Decide what to do with the untracked files.
4. Tell Syed that his work is merged.

## 2026-10-24 (part 2: project check and calibration)

### 1. What we did
1. Read this log. Found that newer work was already on GitHub (PR #35 merged, 4450 tests passed).
2. Checked if all parts of the project are built. Read the team plan, the three plan files and `IMPLEMENTATION_STATUS.md`.
3. Looked at the two Syed branches (`syed/day34-replay`, `syed/experimental-investigation`). They hold no work that is missing from our code.
4. Decided: we will NOT wire the tuned quality values into the defaults. No retraining.
5. Checked pyarrow. It is already in `requirements.txt`. Removed the old note.
6. Checked Igor's work item by item against `plan/IGOR.md`. His 58 tests pass.
7. Found that calibration (Igor's item 9) was never done.
8. Added calibration code to `classify.py` and tried it. Result: model B macro F1 fell from 0.651 to 0.592, and the demo attack rule found 0 of 56 attack bouts.
9. Decided: NO calibration for the submission. Switched it off by default. Retrained both models. The old numbers came back (0.499 and 0.651).
10. Checked Igor's two zips (validation scores and model files). Compared them with our rebuilt models.
11. Wrote messages for Igor and Syed.

### 2. Commands we ran
- `git log`, `git branch`, `git diff --stat`: show history and differences. They change nothing.
- `python scripts/check_leakage.py`: checks that test sessions are not used in training. 7 passed, 1 failed (see problems).
- `python -m pytest -q <test files>`: runs the tests. Igor's 58 passed. Classify tests passed.
- `python -m live_tracker.classify train --name model_a/model_b ...`: trains and saves a model.
- `python -m live_tracker.classify scores ...`: writes per-frame scores for the validation sessions.
- `python scripts/tune_attack_rule.py`: chooses the attack rule on validation scores.
- `sha256sum`: makes a file fingerprint, to check Igor's score file.
- Unzipped Igor's files into a new empty folder. We did not load the `.joblib` files.

### 3. Files we changed, and why
- `IMPLEMENTATION_STATUS.md`: removed the old pyarrow note; recorded the quality-values decision and that calibration was not applied.
- `src/live_tracker/classify.py` and `tests/test_classify.py`: added optional calibration (`FitParams(calibrate="sigmoid")`). It is OFF by default. 3 new tests.
- `docs/model_settings.md`: added the calibration test result and Igor's original model versions.
- `SESSION_LOG.md`: this part.
- Commits: `1e54787`, `1900f09`, `2e981df`, `f86c29d`, `b5e5c34`, `fff9251`, `cad3ba1`. All pushed to `ava/model-settings`.

### 4. Problems and fixes
- `check_leakage.py` has 1 FAIL ("test = all official test sessions, 19 in manifest, 0 official"). It is not a leak. `labels.parquet` was rebuilt from the train file only, so it has no test sessions. Not fixed on purpose: fixing it means reading the test file.
- Calibration broke the demo attack rule. Fix: we turned calibration off.
- Our rebuilt scores differ from Igor's by up to 0.09. Igor's zip shows why: he used a Mac with scikit-learn 1.6.1; we use Windows with 1.7.2. Cause not tested.

### 5. Not finished
- Calibration (Igor's item 9): not applied. The report must say the scores are NOT calibrated.
- Check of Syed's work against `plan/SYED.md`: not done.
- The final test run: not done on purpose (test split can be used once).
- Slides, report, demo rehearsal, backup video.
- Igor's and Syed's plan checkboxes are old and wrong.
- Untracked files not committed on purpose: `.whl` files and `reply_to_syed*.txt`.

### 6. Next time
1. Check Syed's work item by item against `plan/SYED.md`.
2. Run the full test suite once more (about 40 minutes).
3. Agree the test-split policy with Igor, then run the final test once.
4. Slides, report, demo rehearsal, backup video.
5. Decide if the `check_leakage.py` FAIL needs a fix.

## 2026-10-24 (part 3: UI polish, branch `ava/ui-polish`)

### 1. What we did
1. Checked Syed's work against `plan/SYED.md` (1,292 tests passed). Opened the dashboard and went through Rules, Replay and the summary by hand.
2. Found: the summary shows only after the "End of session" button; the real-data replay needs a saved run; skeleton lines are provisional; tracking warnings were "pending Ava".
3. Decided: tracking warnings stay stored apart from the monitor event log. Removed the "pending Ava" text for this.
4. Made the UI branch `ava/ui-polish`: theme file, shared style (`ui_style.py`), short notices with "More info", colored alert and ask rows.
5. Applied the project manager's list: less space at the top, system font, new colors (`#F1F5F4`, `#DDE6E3`, `#FFFFFF`, `#B76A54`), home page cleanup, "Advanced settings" folded under the upload box.
6. Full test suite on the branch (before the last look changes): 4459 passed, 0 failed. Page tests after the last changes: 273 passed, 91 skipped.

### 2. Commands we ran
- `git checkout -b ava/ui-polish`: new branch for the UI work.
- `python -m streamlit run app.py --server.port 8765`: starts the dashboard.
- `python -m pytest -q <files>`: runs tests. The full run takes about 39 minutes.
- `gh pr view 36`: shows the state of a pull request.

### 3. Files we changed, and why
- `.streamlit/config.toml` (new): theme colors and font settings.
- `src/live_tracker/ui_style.py` (new): shared CSS, badges, `note()`, colored action cells.
- `app.py`: dropped two intro lines, CSV format moved into the upload box, engine settings fixed from `params.py`, sidebar replaced by "Advanced settings".
- `pages/*.py`, `validation_run_ui.py`, `investigation_v1_ui.py`: long notices shortened, full text under "More info". Spending warnings left as they are.
- `pages/2_Replay.py`, `summary.py`: tracking-warning text, End-of-session hint.
- Tests: `test_app.py`, `test_rules_page.py`, `test_replay_page.py`, `test_ui_style.py` (new).

### 4. Problems and fixes
- Some tests check exact words in notices. Two safety statements ("no language model chose or interpreted them", "browsing is not one") must stay in the short line. Fixed.
- 84 to 91 tests are skipped here because the real validation run is not on this computer.
- The Chrome extension was not connected, so we could not take our own screenshots. The project manager sent screenshots.

### 5. Not finished
- Not merged. Syed must look at the branch first, then merge to `main`.
- Items not applied: Rules page (fold the Claude cross-check) and renaming "app" to "Home".
- Camera idea: undecided (simulated live stream, real camera with a pose estimator, or keep Replay).
- Real-data Replay needs a run from `scripts/validation_run_v1.py`.
- Skeleton lines are still "PROVISIONAL".
- Backup video, slides, report, demo rehearsal. Final test run (once only).
- The last look changes (space, font, colors) have not been seen by the project manager yet.

### 6. Next time
1. Look at the new design with the project manager and fix what still looks bad.
2. Decide the camera direction and the Rules page option.
3. Open a PR from `ava/ui-polish` for Syed.
4. Full test run, then the final test run once.

## Part 4 - 2026-10-25 (guided flow, smart reading, fixes, share with Syed)

### 1. What we did
1. Made a guided flow: Upload, Rules, Replay, Real session, with a bar and Back/Next buttons. `app.py` is now a router.
2. First page has three choices: camera, video, CSV. The camera choice shows "Camera connected successfully!" in green and starts the video (demo mode, a small line says so).
3. Smart rule reading (works offline, handles typos and everyday words) and an optional paid Claude reader. The Claude reader was tested with a fake client only.
4. Removed four notices on request (Rules "proposed" warning, Replay banner, Replay synthetic note, Real session caption).
5. Fixed "SimpleImputer ... _fill_dtype": the models were trained with another scikit-learn. We trained both again with the project Python (1.9.1). The run program now says clearly when a model has the wrong version.
6. Added `scripts/fetch_demo_video.py` and `docs/run_on_another_computer.md` so Syed can get the same video and run the app.
7. Merged everything into `ava/demo-integration` and opened PR #38 to `main` (not merged).

### 2. Commands
- `git merge --ff-only ava/flex-rules-upload`: move the main folder to the new commits.
- `git push origin ava/demo-integration`: send the branch to GitHub.
- `python -m live_tracker.classify train ...`: train a model again.
- `gh pr create`: open the pull request.
- `pytest` on live scoring, run program and downloader: 47 passed.

### 3. Files changed
`pages/1_Rules.py`, `pages/2_Replay.py`, `src/live_tracker/live_view.py` (notices removed); `monitor_runner.py`, `classify.py` (version check); `scripts/fetch_demo_video.py`, `docs/run_on_another_computer.md` (new); docs and status numbers.

### 4. Problems
- Wrong scikit-learn version in the models: trained them again. Old models are in `models/backup_sklearn_1.7.2/`.
- The demo video is not in Git (200 MB and 89 MB; GitHub limit is 100 MB, and our rule says no videos in Git). Syed gets it with the downloader, which checks size and checksum.

### 5. Not finished
- Full test run on the final code (last full run was before the guided flow).
- PR #38 needs Syed's review. Not merged.
- Camera is a demo only. No pose model exists.
- Real key test of the Claude reader.

### 6. Next time
1. Run the full tests, then fix anything that fails.
2. Ask Syed to review PR #38 and try the steps in `docs/run_on_another_computer.md`.
3. Record the screen video and screenshots for Canvas.

### Part 4, later in the day: one-by-one removals (2026-10-25)

The project manager sends screenshots. Each one is removed, tested, and committed alone. All are on `ava/demo-integration` and in PR #38.

| # | What was removed or moved | Commit message |
|---|---|---|
| 1 | Green "Chosen: Demo camera ..." box at the bottom of the Upload page | Upload page: remove the green 'Chosen' box |
| 2 | Grey "Keypoints present ..." notice at the bottom of the Upload page | Upload page: remove the grey 'Keypoints present' notice |
| 3 | Assumptions bullets, "Untuned default thresholds" warning and the refusal reasons on the Rules page. They now show in a hover "?" next to the instruction box (`help=` on the text box). The page reruns after "Propose rule" so the "?" is fresh | Rules page: reasons and assumptions move into a hover (?) |
| 4 | Whole "Rules in this session" section (table, "No rules decided yet", "Monitoring cannot start" / "Monitoring can start" box). Next stays locked until a rule is confirmed | Rules page: remove the 'Rules in this session' section |
| 5 | "Demo mode: no physical camera is attached ..." line under "Camera connected successfully!" | Upload page: remove the 'Demo mode' line |

Notes:
- The green "Camera connected successfully!" message stays. The project manager asked for it, and knows there is no camera.
- Tests were changed to match: they now check the hover text (`at.text_area[0].help`) and the rule state, not the removed text.
- Test runs after each change: 377 passed (rule and flow tests), then 82, then 55. **No full-suite run on the final code yet.**
- The project manager mentioned `photo_...465_y.jpg`. That file did not exist. We used `...467` for the "Rules in this session" removal.
- The tool call for item 5 was interrupted by the user; the edit was already on disk, tests were run afterwards (55 passed), then committed.

### How to start the next chat
1. Read `SESSION_LOG.md` (parts 1 to 4), then tell the user in 3 to 5 short lines where we stopped.
2. Dashboard: `cd C:\Users\AnTiC\Downloads\live-tracker\live-tracker`, `$env:PYTHONPATH="src"`, `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8766 --server.address localhost`. Restart it to see code changes.
3. Branch `ava/demo-integration`, PR #38 to `main` (not merged; Syed must review). Dev copy: `C:\Users\AnTiC\Downloads\live-tracker\work-guided-flow` (branch `ava/flex-rules-upload`, older).
4. User rules: very simple English; do what is asked directly; one commit per small change; never push to `main`; never commit videos, data or keys; no `git push --force`.

### Still to do
1. More removals from the user (they send them one by one).
2. Full test run (about 45 minutes) on the final code; play a beep when done.
3. Ask Syed to review PR #38 and try `docs/run_on_another_computer.md`.
4. The demo video is NOT in Git (200 MB and 89 MB; GitHub limit 100 MB; our rule). Syed uses `python scripts/fetch_demo_video.py`. The user has not answered if they still want it committed.
5. Docs still mention the removed notices in places (`docs/demo_recording.md` talks about the demo-mode line, the "Chosen" box and the assumptions list). Update them.
6. Open from before: camera needs a mouse pose model (none exists); real-key test of the Claude reader; real-data Replay run; backup video, slides, report, rehearsal; record the screen video and screenshots for Canvas.
7. Demo models: both retrained with the project Python (scikit-learn 1.9.1). Old ones are in `models/backup_sklearn_1.7.2/`. Demo session now gives 44 events.

## Part 5 - 2026-10-25 (Real session layout, new Replay page)

### 1. What we did
1. Real session page: run details on the left, small live video (1/4 width) on the right (`live_view.py`). Commit `88cdb61`.
2. Replay page: removed all run details. It now shows only the real video and a skeleton drawing of the mice that follows the video time. Purpose: check that the model finds the mice.
3. New file `src/live_tracker/skeleton_player.py` (reads the poses, builds a small canvas page). `pages/2_Replay.py` rewritten.

### 2. Commands
- `pytest` on the live view, flow, replay, investigation and run view tests.

### 3. Files changed
`pages/2_Replay.py`, `src/live_tracker/skeleton_player.py` (new), `tests/test_replay_page.py` (rewritten, keeps helpers `DEMO`, `TRACK`, `workflow`, `start`), and old Replay page tests removed in `test_replay_investigation.py`, `test_validation_run_view.py`, `test_investigation_page.py`, `test_demo_smoke.py`.

### 4. Problems
- Two `test_session_view.py` tests fail only when run together with other test files. They pass alone. Cause not found.
- The skeleton canvas finds the video through the parent page. I could not check it in a browser (no browser tool).

### 5. Not finished
- Look at the Replay page in the browser. The skeletons must move with the video.
- Skeleton lines are still PROVISIONAL. Full test run on the final code.
- The video is not drawn under the skeletons (side by side only).

## Part 5, later (2026-10-25): layout, time limit, report for the biologist
- Real session: big video right (45%), small Alerts/Questions boxes above it, chart moved up (`0fc87a5`).
- The run now stops at the schedule time confirmed on the Rules page, and the page says the report is ready (`851b365`).
- New `src/live_tracker/biologist_report.py`: a button on the Real session page asks Claude (one request, facts only, no video) to write a plain report for the biologist. The program adds an exact facts table. The report is saved in the run folder, can be downloaded, and a short phone message goes out when `LIVE_TRACKER_NTFY_TOPIC` is set.
- Tests: `tests/test_biologist_report.py` (fake client, no network) and two page tests.
- Not tried with a real Claude key yet. The model is `$LIVE_TRACKER_CLAUDE_MODEL`, or `claude-sonnet-5-5`.
- Lesson: patching `urllib.request.urlopen` did not stop the ntfy class (it keeps its own copy). Patch `NtfyNotifier.send` in tests. One test may have sent one harmless message to ntfy.sh topic "longrandomtopic" before the fix.
