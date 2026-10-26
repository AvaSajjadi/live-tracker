# Demo readiness and offline backup

The current app supports three distinct demonstrations:

| Demonstration | Inputs and meaning |
|---|---|
| Synthetic rehearsal | Built-in generated scores/poses, clearly marked SYNTHETIC; proves software behavior, not scientific findings |
| Saved real validation run | Local recorded journals, summaries and evidence; Investigation checks consistency and browses without running a model |
| Recorded-video simulation | Local video plus matching saved keypoints and Model B; the runner scores keypoints at playback speed, without camera-image inference |

Real validation work is documented in [validation_run_v1.md](validation_run_v1.md). Validation shares animals with
train and the demo rule was chosen on validation; it is not a held-out test result. Local consistency verification
is not independent authentication. H5 remains open until the team agrees to a frozen held-out test run with Ava as
runner. Do not use test data for rehearsal or tuning.

## Files to prepare on the demo computer

These artifacts are git-ignored; cloning the repository does not supply them. Arrange a team handoff before demo day.

| Path / artifact | Needed for |
|---|---|
| Installed Python environment and requirements | Every demo; install and test before disconnecting |
| `example_pose_demo.csv` (tracked) | Motion tool synthetic backup, no model needed |
| `models/model_b.joblib` and `models/model_b.json` | New recorded-keypoint scoring; use compatible scikit-learn versions |
| `data/raw/calms21_video/<session>.mp4` and `<session>_poses.parquet` | Simulated camera; matching train/validation session only |
| Browser-playable `<session>_web.mp4`, if original codec cannot play | Video preview; convert before rehearsal |
| `data/raw/calms21/calms21_task1_train.json` | Preparing keypoints, models or validation cache; not needed merely to open a completed run |
| `data/raw/calms21_validation/` verified cache | Validation motion measurements and relevant evidence verification; build from official train JSON |
| Igor's validation Model B scores zip and expected digest | Reproducing a new validation run under the existing explicit approval requirements |
| `outputs/validation_run_v1/<run>/` | Saved Investigation run: complete journal, snapshot, summary and Contract v1 evidence files |
| Saved Live Agent run (for example `live_run1`) with its required evidence files | Offline reopening on Live Agent; follow [live_agent_demo.md](live_agent_demo.md) |

Do not commit datasets, model files, videos, credentials or local run outputs. Do not fetch the 28 GB video archive
for a rehearsal; arrange one authorized recording or use the explicit single-video fetch instructions in
[run_on_another_computer.md](run_on_another_computer.md). Nothing in this checklist downloads data automatically.

## Prepare a safe offline backup

1. Install requirements beforehand. Use a fresh output folder for each rehearsal; saved answers persist.
2. Start from the repository root with the environment's Python:
   `python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false`.
3. Keep Rules on **Smart reading (default, no API key, no cost)**. Propose an instruction, review thresholds and
   assumptions, then Confirm. Proposal alone does not approve it. Omitted thresholds are untuned examples.
4. With no local dataset/model/video, open **Motion tool**, choose **Try synthetic demo data**, then optionally
   inject tracking failures. On **Run**, confirm a short schedule on Rules and choose **Demo: made-up scores
   (SYNTHETIC)**. Start, inspect alert/ask colors, answer an ask, stop or finish, and download the deterministic report.
   This is the dependable artifact-free fallback; never describe its output as real-data results.
5. If local artifacts are present, rehearse the recorded-video path: Upload simulated camera → Rules (explicit
   validation demo thresholds only if agreed) → Replay → Real session → Start recorded session. Replay shows the
   Upload recording with its keypoints (inspection only); use Next to reach its scoring view. A positions CSV has no behavior alerts.
6. For a saved validation or Live Agent run, open it offline and check that verification succeeds before the talk;
   a missing or invalid run must remain an error, never a silent synthetic replacement.
7. Clear paid-provider credentials and notification configuration (`LIVE_TRACKER_NTFY_TOPIC`) from the demo shell if you want the rehearsal
   entirely offline. Do not click paid interpretation or **Write report with Claude (paid call)**. Do not start a
   new paid Live Agent run. Saved records and deterministic reports need no paid provider.
8. Disconnect networking and repeat the intended path. Check the target display, video codec, lag warnings and
   readable alert times/confidences. Record a backup screen video on that computer, watch it and store its path
   for the team. Prepare and rehearse the slides separately.

## Team actions still required

Issue [#37](https://github.com/AvaSajjadi/live-tracker/issues/37) asks for a demo-machine decision, transfer of saved
runs and validation scores, a backup recording and the approved H5 run. These instructions do not establish that
those tasks are complete. Closing old PRs and confirming old branch retirement require the owner's separate GitHub
action; this UI change does neither. No backup video or slides are claimed here.
