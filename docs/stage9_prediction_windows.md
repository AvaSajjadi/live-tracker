# Stage 9 S9-4: windows over prediction episodes (provisional, for review)

Status: implemented as provisional choices in `investigation_windows.evidence_context`,
`investigation_sources` and `investigation_compare`. Nothing here is approved;
each item below needs Ava's review (PR #17) before results are relied on.

## What changed

`get_window_features`, `check_tracking_quality` and `compare_windows` accept
windows of `classifier_prediction` and `monitor_event` episodes (S9-3 sources)
as well as `validation_annotation` episodes (unchanged).

For a prediction source:

| Aspect | Annotation source (unchanged) | Prediction sources |
|---|---|---|
| Pose | verified validation cache | the same cache; each record reduced to keypoints + scores before parsing (annotations never read) |
| Episode onset | first annotated frame | classifier: first frame with tracking_ok and score >= threshold; monitor: detection onset (first frame of a fired bout) |
| Pre-event window | pre-behavior | pre-threshold-crossing / pre-detection: not necessarily pre-behavior |
| Behavior frames for exclusions | annotated frames | classifier: qualifying frames; monitor: bouts [first_frame, last_frame + 1) of fired events |
| `label_context`, `label_frames_*` | computed | null / not computed: the investigation confound is not assessed |
| Frame alignment | n/a | cache frame count must equal the score drop's, per session |
| Mixed sources | refused (contract) | refused (contract; quality: `invalid_request`) |

Window shapes, causal truncation, smoothing, tracking flags, usable-frame rule,
baseline tiling, thinning, sufficiency criteria and the descriptive effect are
the Stage 4 / Stage 5 code, unchanged.

## Decisions to review (all provisional)

1. **Exclusion frames for predictions.** Exclusions use the declared source's
   frames only. Classifier: qualifying frames (missed behavior is not excluded;
   false detections are). Monitor: only bouts that fired (shorter qualifying runs
   are invisible to the journal and are not excluded). Alternative: exclude with
   the classifier's qualifying frames for monitor runs as well.
2. **No label context.** Prediction-only investigations never read annotations,
   so the investigation-versus-other confound is not assessed. Reading validation
   labels as *context only* (not as episodes or exclusions) is possible but was
   not done.
3. **Onset latency.** No latency gap is added: a pre-threshold-crossing window can
   contain the start of the behavior when the classifier detects late (Igor's
   validation median attack latency 0.23 s). A gap (`gap_frames`) is the model's
   choice per request; a minimum may be warranted.
4. **Two duration rules.** The classifier source keeps runs of at least
   `min_duration_frames` frames; the monitor fires when elapsed observed time
   reaches `min_duration_s` (16 frames for 0.5 s at 30 fps). They are reported as
   different and are not interchangeable (blocker B7).
5. **Tracking.** The classifier's qualifying frames use the score file's
   `tracking_ok` (Igor's flags, with two-sided short-gap interpolation); window
   usability uses the investigator's own non-interpolated flags. They agreed on
   every validation frame checked (S9 reconnaissance), but are computed differently.
6. **Thresholds.** The classifier source's threshold and minimum duration are set
   when the source is created, not per request, and are reported in every result.
   No values have been tuned or approved.
