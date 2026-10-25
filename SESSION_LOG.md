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

## Part 6 - 2026-10-25 (Review of the demo, no code changed)
1. Read the log, the branch commits, `docs/demo_recording.md`, `skeleton_player.py`, `pages/2_Replay.py`, `biologist_report.py`, `live_view.py`.
2. Ran `pytest` on `test_biologist_report.py`, `test_replay_page.py`, `test_session_view.py`, `test_demo_smoke.py`: 23 passed. (Runs the tests; no files changed.)
3. Found: `docs/demo_recording.md` is out of date (Demo mode line, Confirm button, assumptions list, video side). `save_and_notify` has no error handling for the file write. Skeleton canvas is not checked in a browser. Full test run still not done.
4. Next: fix the doc, check Replay in a browser, full test run, Syed reviews PR #38.

## Part 7 - 2026-10-25 (Real backend, step 1 and 2: the camera)
1. Plan from the user: make the demo real, starting with the camera. Choices: build the pose-model plug-in only (no model yet), USB webcam first.
2. Installed OpenCV in the project Python: `.\.venv\Scripts\python.exe -m pip install opencv-python-headless` (lets the program open a camera).
3. New `src/live_tracker/pose_model.py`: the slot for a pose model (picture to keypoints), checks the model output on every frame. `docs/pose_model.md` explains it. No backend ships yet.
4. `run_sources.py`: `make_source({"kind": "camera"})` now checks the pose model first, then opens the real camera (number or address) and can reconnect. New `camera_device`.
5. New `src/live_tracker/camera_check.py`: is OpenCV there, does the camera give a picture, is there a pose model.
6. Upload page: "Connect a camera" now has two types: Demo camera (recorded video, as before) and USB camera (number box, "Test the camera" shows one picture, "Use this camera"). `SessionSource` has a new `device` field.
7. Tests: `tests/test_pose_model.py` (20), `tests/test_upload_camera.py` (4). 344 passed in the camera, flow, upload, replay, live view and run tests. Full suite not run.
8. Bug found by a test and fixed: choosing USB camera kept the demo source selected in the background.
9. Commits: d934f6e, 248fbb3 (pushed on `ava/demo-integration`).

Not finished: the Real session page still says a camera cannot be analysed (`live_view.py`). A camera run needs: a run spec for the camera, a live panel that does not need a poses file or video, and the Rules page schedule as the time limit. No pose model exists. No real camera was tried (no hardware test).
Next time: step 3 (start a camera run from the Real session page), then the pose model backend, then a real camera test.

### Part 7, later: connecting the Nikon D7500 (2026-10-25)
1. What we did, step by step
   1. Tested camera numbers with OpenCV. Camera 0 = laptop webcam (640x480), works. Camera 1 (1024x768) gives a flat dark picture (brightness 7.6 every time, no change). It is NOT the Nikon: Windows lists "USB2.0 IR UVC WebCam" (the laptop infrared camera for face login), most likely camera 1.
   2. Learned: the D7500 is a Nikon (not Canon). It has Wi-Fi and Bluetooth (SnapBridge), but they send photos, not a live video stream our program can read. So we need a cable: USB with Nikon Webcam Utility, or HDMI with a capture card (better for overnight runs).
   3. A phone can also be a camera (IP Webcam on Android, or DroidCam/Camo): type its address in the camera box. Good for a test, weak for all-night runs.
   4. The user installed Nikon "Webcam Utility" 1.1.1.30000 (today). Windows sees the D7500 only as a file device (WPD), not as a webcam. Windows Settings > Cameras shows only "USB2.0 HD UVC WebCam". Teams was running (it can hold the camera).
   5. Started a second dashboard with the new code on port 8767 (the old one on 8766 was left alone).
2. Commands we ran
   - OpenCV test scripts: open camera numbers 0 to 5, read 20 to 30 pictures, print only size and brightness (no picture saved the second time).
   - `Get-PnpDevice`: list camera and Nikon devices Windows sees (read only).
   - Registry and folder look: check that Nikon Webcam Utility is installed (read only).
   - `streamlit run app.py --server.port 8767` (background): the new dashboard.
3. Files changed: none in this part (only this log).
4. Problems
   - Nikon does not show up as a webcam yet. No light on the camera at one point (power or battery). Lens cap was removed by the user.
   - One test picture of the user (webcam, camera 0) is still in the scratchpad folder `...\scratchpad\cam0.png`. The user said no to deleting it. It is not in Git. Delete it by hand when you want.
   - Two photos of the user's screen are in `Downloads` (`photo_5109786690603126395_y.jpg`, `photo_5109786690603126400_y.jpg`). Not in Git.
5. Not finished
   - Make the Nikon work as a webcam.
   - Step 3: the Real session page for a camera. A pose model (none exists).
6. What to do next time
   1. Restart the computer, turn the D7500 on, plug in USB, wait 10 seconds.
   2. Open Zoom or Teams and look for a Nikon camera name. Tell the assistant the names.
   3. Camera on video mode with Live View (LV) on. Check the D7500 is on Nikon's supported list.
   4. If it still fails: HDMI cable and a capture card (clean HDMI out in the camera menu).
   5. Then run the camera test again (camera numbers 0 to 5, brightness must change between pictures).
   6. Then step 3 and the pose model.

## Part 8 - 2026-10-25 (Web page back to demo only)
1. What we did
   1. The user asked to take the web version back to the demo only.
   2. We undid commit 248fbb3 (the USB camera choice on the Upload page) with `git revert`. The Upload page is now Demo camera only, as before.
2. Commands
   - `git revert --no-edit 248fbb3`: makes a new commit that undoes that one commit. History is kept.
   - `pytest tests -k "upload or session_view or demo_smoke or replay"`: runs the web tests.
3. Files changed: `src/live_tracker/upload_ui.py`, `src/live_tracker/uploads.py`, and `tests/test_upload_camera.py` (deleted) - all by the revert.
4. Problems: 2 tests in `test_session_view.py` fail when run together with the others. They fail the same way before our change. They pass when run alone. So it is a test order problem, not new.
5. Not finished: the camera code is still in the project but no page uses it: `pose_model.py`, `camera_check.py`, `make_source` camera, `scripts/find_cameras.py`. Nikon is still not a webcam.
6. Next time: find out why the 2 tests fail in a group. Camera work waits until the user wants it back (`git revert` the revert to get the USB page back).

### Part 8, later: all camera work removed
1. The user said the camera option was still there. We also reverted `d934f6e` (pose model slot, camera check, camera in `make_source`) and `427a437` (camera scan script).
2. Code, pages, tests and docs now match commit `53278e2` (the demo before the real camera work). Only the session log is different.
3. The old dashboard on port 8766 is the demo version. The test dashboard on port 8767 is gone. Restart `streamlit run app.py` and press Ctrl+F5 in the browser.
4. To get camera work back: `git revert` the three revert commits.

### Part 8, later: Real session "Access is denied" (2026-10-25)
1. Problem: Real session said "The run could not start: PermissionError [WinError 5] ... status.json.tmp -> status.json". Windows blocks the rename when the page has `status.json` open at the same moment.
2. Fix: `_write_json` in `monitor_runner.py` now tries the rename up to 40 times (50 ms apart). New test in `tests/test_monitor_runner.py`.
3. Not changed: the same rename pattern is in `event_log.write_jsonl`, `investigation_data.py` and `schedule_workflow.py`. They could fail the same way. Not seen yet.

### Part 8, later: report button (2026-10-25)
1. Problem: the "Write the report for the biologist" button did nothing useful. Two causes: (a) without a Claude key the button only showed an error; (b) the new `write_plain` had real line breaks inside a normal string, a syntax error, so the file could not load.
2. Fix: `write_plain` in `biologist_report.py` now uses `\n`. If Claude cannot write, `live_view.py` shows a plain report made by the program, with a note saying why. New test in `tests/test_live_view.py`.
3. Commands: `pytest tests -k "live_view or biologist"` runs the related tests (29 passed).
4. Not tried: the real Claude request with a real key.

### Part 8, later: report looks too big (2026-10-25)
1. Problem: on the Real session page the report headings were huge (screenshot from the user).
2. Fix: `_small_headings` in `live_view.py` makes `#` into `####` and `##` into `#####` for the page only. The saved and downloaded file is not changed.
3. Not finished: the user may want more changes to the look (spacing, table, shorter text).

Later: the smaller headings still looked big (the theme in `ui_style.py` sizes all headings). Now `_small_headings` turns heading lines into bold lines. Dashboard on 8766 restarted.

Later: the report table was wider than the box (screenshot). New `_show_report` in `live_view.py` shows the Markdown table as a small data table (`st.dataframe`, full width). The text parts stay Markdown.

## Part 9 - 2026-10-25 (Real camera, step 3)
1. What we did
   1. The user said the demo is basically fake and wants the real camera. Answers: they have a trained pose model (file not given yet), the camera is the Nikon D7500, and we restart from the old camera code.
   2. Reapplied the old camera work with `git revert` on the three revert commits (`e675814`, `7415198`, `bdcddc2`). Pose model slot, camera check, scan script and USB camera choice are back.
   3. Installed OpenCV (`pip install opencv-python-headless`) and ran `scripts/find_cameras.py`. Camera 0 (USB webcam) works. Camera 1 is the infrared camera (flat picture). No Nikon found.
   4. Step 3 done: the Real session page can start a camera run. `live_session.py` has `camera_spec`, `camera_limit_s`, `start_camera`. `live_view.py` has `render_camera`. It shows the pose model error when there is none.
2. Commands
   - `git revert --no-edit <commit>`: undoes an earlier undo.
   - `python scripts/find_cameras.py`: lists cameras and checks if the Nikon is a webcam.
   - `pytest tests/test_live_view.py tests/test_live_session.py ...`: 82 passed.
3. Files changed: `src/live_tracker/live_session.py`, `src/live_tracker/live_view.py`, `tests/test_live_view.py`, `tests/test_live_session.py`, this log.
4. Problems: I again wrote a line break inside an f-string (syntax error). Fixed with `\n`.
5. Not finished
   - The pose model backend: needs the user's model file and its type. Then `register_backend` + `models/pose/pose_model.json`.
   - The Nikon is not a webcam. Needs HDMI + capture card, or Nikon Webcam Utility working.
   - No real camera run has been tried. A camera run time is the Rules time, else 30 minutes.
   - Model B was trained on CalMS21 top-down view: a different view gives weak scores.
6. Next time: get the pose model file, write the backend, test with the USB webcam (camera 0), then the Nikon.

### Part 9, later: camera films an iPad (2026-10-25)
1. The real presentation: a camera on a holder films an iPad that plays the demo video (CalMS21 `mouse064_task1_annotator1`, 1024 x 570, validation split).
2. New `screen_map.py`: the researcher sets the 4 corners of the video; each picture is stretched to 1024 x 570 before the pose model. `SessionSource.screen` keeps the corners. `run_sources.make_source` and `live_session.camera_spec` pass them on. The Upload page has the corner sliders and shows the stretched picture.
3. Tests: `tests/test_screen_map.py` and one more in `tests/test_upload_camera.py` (123 passed with the related tests). Docs: `docs/pose_model.md`.
4. Not finished: the pose model file and backend (still missing). No real picture of an iPad was tried yet. The USB webcam (camera 0) is enough for this setup; the Nikon is not needed.
5. Next time: get the pose model, test the corners with the webcam pointed at the iPad, check glare and brightness.

### Part 9, later: pose model search (2026-10-25)
1. The repo has NO mouse pose model (searched all 34 branches, history, ignored files, Downloads, zips). `pose_model.py` is only the empty slot. The demo read saved CalMS21 keypoints, never pictures.
2. Tested the public DeepLabCut SuperAnimal-TopViewMouse (hrnet_w32 + fasterrcnn_resnet50_fpn_v2) on 5 s of the demo video, CPU only, in a separate env in the scratchpad (1.76 GB, DELETED after the test at the user's request).
   - Speed: about 5.8 s per frame (detector about 5.7 s, pose network about 40 ms). Real time needs 0.033 s.
   - Quality: a mouse found in only 31 of 150 frames, the second mouse never. With a lower detector cut-off (0.2) on 12 frames: no mouse found at all. Not usable on CalMS21-style footage.
3. Found MARS (Segalin et al. 2021, the tool behind the CalMS21 keypoints): pretrained top-view pose models, v1.8, Caltech record 801be-66057 (models.zip 543 MB, license CC BY-NC 4.0, needs black and white mice). Not tried yet. Needs an older TensorFlow setup; GPU recommended; speed not reported.
4. Not finished: no working pose model. Next: decide between trying MARS (old TensorFlow, may not run on this laptop) and the staged demo (camera films the iPad, saved keypoints are read, said openly).
5. Small test files left in the scratchpad (about 6 MB). pip cache may hold some GB (not deleted).

### Part 9, later: staged demo (2026-10-25)
1. Built the staged demo, the safe plan for the presentation: a real USB camera films the iPad that plays the demo video; the page shows the camera live; the alerts come from the SAVED keypoints of that video (not from the picture). The page says this in a yellow box on the Upload step and on the Real session page.
2. Upload page, USB camera: new choice "Staged demo" (the other choice is "Real pose model"). It creates a `demo` source with a camera number (`SessionSource.staged_camera`) and the screen corners.
3. Real session page: `render_staged_intro`, `camera_preview` (live picture, stretched to the video part when corners are set), `render_staged_running` with a GO button. GO is pressed together with play on the iPad (about 0.3 s human delay).
4. Tests: 106 passed (live view, upload camera, uploads, screen map, flow, live session, demo smoke). Not tried with a real camera and a real iPad yet.
5. Not finished: real test with the webcam on the iPad; MARS try (543 MB models, old TensorFlow); the Nikon.

### Part 9, later: live video on the Upload step (2026-10-25)
1. The user: the camera works, but it should be video, not a picture. We asked what they meant; they chose "Live video on the Upload step too" only.
2. New `camera_live.py`: ONE open camera shared by the Upload step and the Real session page (Windows lets only one program use a camera). `release()` is called by the Stop button, by the Test button before a new test, by the session reset, and before a real camera run starts.
3. Upload page: after "Test the camera" it shows live video (`live_preview`, a fragment that runs about 10 times a second) with the screen outline and the stretched screen. "Stop the camera" releases it. Real camera speed measured: 15 pictures per second at 640 x 480.
4. Tests: 93 passed (upload camera, live view, screen map, uploads, flow), with a fake camera. Real webcam checked only for speed, not through the page.
5. Not done (not chosen): saving the camera as a video file in the run folder; smoother live video on the Real session page (it still refreshes about twice a second).

### Part 9, later: camera doc (2026-10-25)
1. Wrote `docs/camera.md` (the whole camera chain, the three ways the camera is used, which file does what, presentation setup, video quality needed, what is real). A copy is on the Desktop as `camera.md`.
2. Restarted the dashboard on port 8766 after the live video work (it was running old code). Fresh page: press Ctrl+F5.
3. Not finished: real test of the staged demo with the camera on the iPad; MARS test; optional camera video file and smoother live video on the Real session page.

### Part 9, later: drag the corners (2026-10-25)
1. The user wanted to set the grid (4 screen corners) by dragging. New `corner_picker.py`: an inline custom component (st.components.v2) draws the live camera picture on a canvas with 4 red handles; dragging works with mouse or touch; the corners go back to Python when a handle is let go. The 8 sliders are gone. A Reset button makes a new picker (new key) at the default corners.
2. `upload_ui.py`: `live_preview` shows the picker and the stretched screen under it; the corners are read from the picker state (`_corners_now`).
3. Tested in a real Chrome tab on the dashboard (port 8766, restarted): live video works, dragging corner 1 moved the outline and the stretched view changed. Unit tests: 41 passed (corner picker, upload camera, screen map, live view). The browser test used the real webcam and was released with Stop the camera.
4. Known small thing: the browser console shows an occasional "Image source error" for the fast refreshing stretched picture (st.image 10 times a second). Harmless so far.
5. Not finished: staged demo end-to-end with the real camera and the iPad; MARS; saving a camera video file.

### Part 9, later: MARS pose model test (2026-10-25, user approved the download)
1. Downloaded MARS models v1.8 (Caltech record 801be-66057, models.zip 543 MB, md5 verified, license CC BY-NC 4.0) and cloned neuroethology/MARS into the scratchpad (not in the project). Extracted only the `.pb` graphs (2 detectors for the black and the white mouse, 1 top-view pose model). Did NOT load the `.pkl` classifier files (pickle can run code).
2. The models are frozen TensorFlow graphs, so they run with modern TensorFlow 2.21 on Python 3.12 (separate env `mars_env`, no old TensorFlow needed). Wrote `scratchpad/mars/mars_run.py`, a port of MARS_pose_machinery.py (resize to 299, detectors, crop to 256, hourglass heatmaps, argmax).
3. Result on 20 frames (60 s of the demo video): both mice found in 20 of 20 frames; median error against the saved CalMS21 keypoints (made by MARS) 2 to 7 px (body about 190 px). Keypoint order matches the CalMS21 order directly. Degraded copy (small, blur, JPEG, dimmer, like a webcam filming a screen): both mice still found 20 of 20; most keypoints 3 to 8 px, but the white mouse's hips and tail base 17 to 37 px.
4. Speed on this CPU after warm-up: about 0.9 s per frame (detectors about 160 ms each, pose about 540 ms). Needs 0.033 s for 30 fps live. DeepLabCut was 5.8 s per frame.
5. Caveats: the comparison is MARS against MARS-made keypoints and these videos may be in its training data, so it does not prove it works on a phone screen or other mice. MARS needs one black and one white mouse, top view.
6. Not finished: not wired into the project (`pose_model.py` backend); not tested on the live camera picture; speed ideas not tried (skip detectors by tracking boxes, batch, ONNX/OpenVINO, GPU).

### Part 9, later: Colab route (2026-10-25)
1. The user chose Google Colab. Wrote `scripts/mars_pose_offline.py` (video in, CalMS21 keypoints parquet out; optional screen corners; batches for a GPU; downloads the MARS models with an md5 check and only extracts the .pb files) and `colab/mars_pose_colab.ipynb` (GPU check, get script and models, choose video, corners and a 60-frame speed test, full run, quick look, download).
2. Tested the script on this laptop (frames 1800+): median error 3.2 px against the saved keypoints; the project own upload check (`normalize_keypoints`) accepts the output file. 0.7 frames per second on CPU with batches. Not run on Colab yet.
3. Upload page: the live preview now shows "Corners for the Colab notebook", one line of 8 numbers to copy.
4. Tests: 81 passed (mars script helpers, notebook file, screen map, upload camera, corner picker, live view, uploads).
5. Not finished: run the notebook on Colab and measure the GPU speed; a record-camera-to-file button in the app (today: record with the Windows Camera app); wiring MARS into pose_model.py for a live GPU machine.

### Part 9, later: MARS runs locally (2026-10-25)
1. The user decided to skip Colab and run MARS on this laptop.
2. Set up a stable folder `C:\Users\AnTiC\mars_local` (outside Git): `models\` (the 3 .pb files, 328 MB) and `env\` (Python 3.12 venv with TensorFlow 2.21, numpy, opencv-python-headless, pandas, pyarrow). Deleted the temporary copies in the scratchpad (env, extracted models, models.zip, MARS code clone).
3. New `scripts/mars_local.cmd` runs the offline script with those paths (defaults: keypoints.parquet and stretched.mp4 in the folder you run it from). Tested on 32 frames: works; steady speed about 1.2 frames per second after a 20 s warm-up (1 minute of video is about 25 minutes).
4. Docs: `docs/camera.md` section 8 is now the local route (Colab kept as an option).
5. Not finished: a record-to-file button in the app; speed-ups (skip detectors by tracking boxes, other model formats); live use (needs about 27 times more speed); wiring MARS into pose_model.py.

### Part 9, later: record and analyze inside the app (2026-10-25)
1. The user needs the camera to record and convert to keypoints. True real time is not possible on this laptop (Intel i7-1165G7, 4 cores, Intel Iris Xe, no NVIDIA; MARS about 0.85 s per frame, live needs 0.033). The user chose: staged demo for the live part, real pose offline.
2. `camera_live.py` rewritten: a background thread (Grabber) reads the camera all the time, keeps the newest picture, and writes the recording (Recorder: fixed 30 frames per second, a 15 fps webcam gets each picture twice; with corners the screen is stretched to 1024 x 570). The camera is given back after 20 s if no page looks at it.
3. New `mars_job.py`: starts scripts/mars_pose_offline.py in the background with the mars_local env, reads progress from mars.log and a .done.json file (the script now writes it, also on failure).
4. Upload page, USB camera: three modes (Record then analyze with MARS (default), Staged demo, Live real pose). Record: Start recording / Stop recording, Analyze with MARS (time estimate), progress bar, Use this recording (builds a video_keypoints source, 30 fps).
5. Checked in a real Chrome tab with the real webcam: modes and buttons show, recording works (14 s, 427 frames, 1024 x 570, valid mp4). Real background job on a 40-frame clip of the demo video: progress, finished, result file (start-up about 45 s, then 1.2 fps). The test recording showed the user face (camera was not on the iPad), so it was deleted at once; MARS was not run on it.
6. Tests: 120 passed in the related files before the last small change (camera live, mars job, mars script, upload camera, live view, screen map, corner picker, uploads, flow, replay, demo smoke).
7. Not finished: OpenCV cannot write H.264 here, so the recording is old-format mp4 and the browser cannot play it without the optional imageio-ffmpeg (not installed; asked the user). Not tried: a full run on a real recording of the iPad. Live (30 fps) pose is not possible on this laptop.

### Part 9, later: Live Agent page blocked (2026-10-25)
1. The Live Agent page (`/Live_Agent`) says "score archive not found: igor_scores_validation_model_b_repacked.zip". It needs that exact file in Downloads (SHA-256 `5c1a81e40ca12f00dea0ae511708b7d534d42e8b1feb43948af338b8b5c23427`, set in `investigation_v1_runs.py`).
2. Searched the repo (all branches and history), the disk and the other zips: the file is not here. The user has `igor_scores_validation_model_b.zip` (SHA-256 `e45e63c1...`, 24 Oct 17:17): it is a different zip, but its `scores_validation_model_b.csv` has the expected inner checksum `ff00f6fe...a008676`, so the data is the same, only the packing differs. `igor_models (1).zip` is an exact copy of `igor_models.zip` (models only).
3. Decision: do NOT change Syed guard code. The user asks Igor for the exact repacked zip (message written in the chat). Option kept in reserve: accept the second archive checksum with a test and tell Syed.
4. When the file arrives: put it in Downloads with the same name, check its SHA-256, reload the Live Agent page, run the free rehearsal (no key, no cost). A paid run needs a separate decision.
5. Camera work status: Upload page has Record then analyze with MARS (default), Staged demo, Live real pose; live 30 fps pose is not possible on this laptop. Next: a real recording of the iPad; optional `imageio-ffmpeg` (not installed, asked).

### Part 9, later: deep research on a much faster pose model (2026-10-25)
1. Four researchers (mouse toolkits, fast generic frameworks, speeding up MARS, training data and Apple M5) and a report writer produced `docs/fast_pose_options.md` (copy of the report in the scratchpad `reports/`; notes in `research_notes/`). All speeds for this laptop and the M5 in it are GUESSES from other tests; nobody published a test on the i7-1165G7 or on an M5.
2. Recommendation: do not speed up MARS; train a small, fast model that copies it. First choice YOLO26n-pose (AGPL-3.0, exports to OpenVINO for Intel and CoreML for the Mac); second choice RTMPose-t/s (Apache-2.0, top-down). Best data: MARS hand-labeled pose set, Caltech record 2011 (15,000 top frames, 1024x570, CC BY-NC 4.0, keep out of the public repo); CalMS21 points are MARS output (extra pseudo-labels, license unclear: use the stricter). Faster MARS (skip detectors, fewer stacks, OpenVINO) is only a fallback, about 5 to 10 fps by estimate.
3. Next tests (need the user OK for each download): Test 1 laptop speed floor with a pretrained YOLO pose model (pip install ultralytics openvino); then download the MARS pose set (1.5 GB images + 21.6 MB json), convert labels to YOLO format, train on a GPU (Colab or Mac), measure accuracy (touching mice!) and speed on the laptop and the M5.

### Part 9, later: Test 1, speed of a small YOLO pose model on this laptop (2026-10-25, user approved the install)
1. Made `C:\Users\AnTiCastpose_test` (outside the project): a venv with ultralytics 8.4.189, openvino 2026.4.1, torch 2.14.1 CPU, opencv. Downloaded the small pretrained `yolo26n-pose.pt` (7.5 MB, human pose) by ultralytics. Speed only, no mouse accuracy.
2. Result (1024x570 frames, 384x640 input): PyTorch CPU 64 ms; OpenVINO CPU 61 ms; **OpenVINO on the Intel Iris Xe graphics chip 16.5 ms (about 60 fps)**. Sustained 4 minutes: 14,258 frames, average 59 fps, median 14.8 to 18.4 ms, 99% of frames under 20 to 36 ms. The 33 ms target is met for this step on this laptop, with the integrated GPU.
3. Scripts saved: `scripts/bench_yolo_pose_speed.py`, `scripts/bench_yolo_pose_sustained.py`. Results are in the appendix of `docs/fast_pose_options.md`.
4. Not finished: train a mouse model (needs the MARS pose set, Caltech record 2011, 1.5 GB images + 21.6 MB json, CC BY-NC 4.0, user OK needed for the download; and a GPU for training: Colab T4 or the M5 MacBook on Tuesday); check accuracy (touching mice, mouse ID); measure the whole live chain; plug into the app.
