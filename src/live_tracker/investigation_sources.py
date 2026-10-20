"""EXPERIMENTAL (Stage 9, checkpoint S9-3): classifier-prediction and
monitor-event evidence sources for `query_events` (new file).

Both implement investigation_events.EventSource and plug into
CalMS21InvestigationTools next to (or instead of) the annotation source:

    tools = CalMS21InvestigationTools(policy, [
        AnnotationEventSource(cache_dir, manifest_path, policy),
        ClassifierEventSource(drop, policy, threshold=0.5, min_duration_frames=15),
        MonitorEventSource(snapshot, drop, policy, rule_id="R1")])

Neither ever reads annotations: a request for a source that is not plugged in
is still refused by the tool, and nothing is substituted. Every episode goes
through the unchanged Stage 2 contract validator (EPISODE_PROVENANCE for its
source) before the result is returned.

classifier_prediction (ClassifierEventSource)
---------------------------------------------
Input: a ScoreDrop issued by score_drops.load_validation_scores whose rows
still hash to its verified rows digest (score_drops.trusted_rows_sha256),
checked at construction and again on every query. Its sessions must be
exactly the policy's validation split; a held-out test session is never
searched (and a request naming one is refused by the policy).

Episode rule (PREDICTION_EPISODES_VERSION, fixed at construction, never
chosen per request): a frame qualifies when tracking_ok is true and the
behavior's score is >= `threshold` (">=", tracking loss and every
non-qualifying frame end a run: the same evidence rule as the monitor); an
episode is a maximal run of qualifying frames, half-open [start_frame,
end_frame), optionally merged by the request's merge_gap_frames (the merged
span then contains non-qualifying frames), kept when its span is at least
max(source min_duration_frames, request min_duration_frames). start_frame is
the first frame at or above threshold: a CLASSIFIER THRESHOLD CROSSING, not
the onset of the annotated behavior.

Provenance per episode (contract classifier_prediction): model_id is
"unverified-label:<label>" (the label inferred from the member file name; no
model artifact or identity is verified), score_file_sha256 the verified
member SHA-256, threshold and min_duration_frames the source's.

monitor_event (MonitorEventSource)
----------------------------------
Input: a RunSnapshot (event_log.freeze_run) that re-validates with
event_log.load_snapshot, is complete and has provenance status
"verified_score_drop", plus the issued ScoreDrop it was monitored from (its
verified rows digest and provenance must equal the snapshot header's: that
is how the searched sessions are known), and one rule_id of the run (one
source answers for one rule: episodes of one rule never overlap).

Each event_fired / event_closed lifecycle of that rule becomes one episode:
start_frame = first_frame (DETECTION ONSET: the first frame of the bout at or
above the rule's ask_threshold), end_frame = last_frame + 1 (CLOSURE,
exclusive). The ALERT FIRING frame (fired_frame, after min_duration_s) is not
an episode boundary; it stays in the frozen snapshot, retrievable with
lifecycle(event_id), and the episode cites the snapshot by journal_sha256 (the
frozen snapshot's SHA-256), run_id, rule_id and event_id. Monitor events are
never merged (a merged span would no longer be one event).
"""

from __future__ import annotations

import math

import numpy as np

from . import event_log
from . import investigation_contracts as ic
from . import score_drops
from .calms21 import FPS
from .investigation_events import (EventQuery, SourceUnavailable, _method, behavior_runs, coverage_metadata,
                                   merge_runs)

__all__ = ["ClassifierEventSource", "MonitorEventSource", "PREDICTION_EPISODES_VERSION"]

PREDICTION_EPISODES_VERSION = "prediction-episodes/1"
MONITOR_EPISODES_VERSION = "monitor-episodes/1"
_TIME_TOL = 1e-9

COMMON_LIMITATIONS = [
    "Episodes are derived from classifier predictions, not human annotations: a predicted episode is not an "
    "observed behavior, and findings about predicted episodes do not transfer to annotated behavior.",
    "The classifier's identity is not verified: the model label comes from the score file name only; no model "
    "artifact, configuration or producing commit was supplied.",
    "Scores on validation sessions were produced by a model whose settings were chosen on these sessions, so "
    "they are optimistic.",
    "Episodes within one session are not independent observations.",
]


def _validation_sessions(policy):
    return {sid for sid, split in policy.split_of.items() if split == "validation"}


def _check_policy(policy):
    ic._require_policy(policy)
    if policy.annotation_split != "validation":
        raise ic.ForbiddenEvidence("policy", "prediction evidence is read for validation sessions only")


def _trusted(drop):
    digest = score_drops.trusted_rows_sha256(drop)
    if digest is None:
        raise ic.ForbiddenEvidence("source", "not an issued, verified ScoreDrop (score_drops.load_validation_scores)")
    return digest


def _drop_sessions(drop, policy):
    """The drop's sessions, which must be exactly the policy's validation split."""
    table = drop._table
    sessions = sorted(set(table["session"]))
    for sid in sessions:
        if policy.split_of.get(sid) == "test":
            raise ic.ForbiddenEvidence("source", f"{sid!r} is a held-out test session")
    if set(sessions) != _validation_sessions(policy):
        raise ic.ForbiddenEvidence("source", "the score drop's sessions are not exactly the policy's validation "
                                   "split")
    return sessions


def _requested(request, sessions, policy):
    if request["sessions"] is None:
        return list(sessions)
    wanted = sorted(set(request["sessions"]))
    for sid in wanted:
        if policy.split_of.get(sid) == "test":
            raise ic.ForbiddenEvidence("request.sessions", f"{sid!r} is a held-out test session")
        if sid not in sessions:
            raise SourceUnavailable(f"{sid!r} is not a session of this source")
    return wanted


def _data_items(prov, synthetic):
    kind = "synthetic_fixture" if synthetic else "classifier_scores"
    label = "SYNTHETIC test fixture, not CalMS21 data: " if synthetic else ""
    return [{"kind": kind, "path": None, "sha256": prov.archive_sha256,
             "description": f"{label}score-drop archive {prov.archive_name} (read once, in memory)"[:ic.MAX_TEXT]},
            {"kind": kind, "path": None, "sha256": prov.member_sha256,
             "description": f"{label}score member {prov.member} (verified against SHA256SUMS.txt; in-band, "
                            f"unsigned); rows sha256 {prov.rows_sha256}"[:ic.MAX_TEXT]}]


def _session_table(drop, sid):
    t = drop._table
    return t[t["session"] == sid]


def _drop_n_frames(drop, sid, sessions):
    if sid not in sessions:
        raise SourceUnavailable(f"{sid!r} is not a session of this source")
    return int(len(_session_table(drop, sid)))


def _complete_coverage(n):
    return {"mode": "complete", "eligible": n, "available": n, "missing": [], "complete": True}


class ClassifierEventSource:
    """classifier_prediction episodes from an issued, verified validation ScoreDrop."""

    source = "classifier_prediction"

    def __init__(self, drop, policy, *, threshold, min_duration_frames):
        _check_policy(policy)
        self._rows_sha = _trusted(drop)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) \
                or not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be a number in (0, 1]")
        if isinstance(min_duration_frames, bool) or not isinstance(min_duration_frames, int) or \
                not 1 <= min_duration_frames <= ic.MAX_WINDOW_FRAMES:
            raise ValueError(f"min_duration_frames must be an integer in [1, {ic.MAX_WINDOW_FRAMES}]")
        self.drop, self.policy = drop, policy
        self.threshold, self.min_duration_frames = float(threshold), min_duration_frames
        self.sessions = _drop_sessions(drop, policy)
        self.model_id = f"unverified-label:{drop.provenance.model_label}"[:120]
        self.synthetic = any("SYNTHETIC" in sid for sid in self.sessions)
        # episodes this source has returned: episode_id -> (end_frame, merge_gap_frames, min_duration_frames).
        # A merged episode can start at a run shorter than the minimum; it is a valid start only because
        # this source returned it under those exact settings.
        self._issued = {}

    def _check(self):
        if score_drops.trusted_rows_sha256(self.drop) != self._rows_sha:
            raise SourceUnavailable("the score drop changed since this source was created")

    # ---- the window interface (S9-4: investigation_windows / investigation_compare) ----

    def n_frames(self, sid):
        return _drop_n_frames(self.drop, sid, self.sessions)

    def behavior_frames(self, sid, behavior):
        """Per-frame bool: the frame qualifies for `behavior` under this source's
        rule (tracking_ok and score >= threshold). These, not annotations, are
        the behavior frames the window exclusion rules use."""
        if behavior not in score_drops.SCORE_CLASSES:
            raise SourceUnavailable(f"the classifier has no score for {behavior!r}")
        g = _session_table(self.drop, sid)
        return g["tracking_ok"].to_numpy(bool) & (g[behavior].to_numpy(float) >= self.threshold)

    def is_episode_start(self, sid, behavior, frame):
        """True when `frame` starts an episode this source produces: an
        unmerged qualifying run at least min_duration_frames long, or an
        episode this source returned (with its merge settings). Any other
        frame, including one inside a run, is refused."""
        q = self.behavior_frames(sid, behavior)
        if not (0 <= frame < len(q)) or not q[frame] or (frame > 0 and q[frame - 1]):
            return False
        record = self._issued.get(ic.episode_id(self.source, behavior, sid, frame))
        if record is not None and self._reproduces(q, frame, record):
            return True
        end = frame
        while end < len(q) and q[end]:
            end += 1
        return end - frame >= self.min_duration_frames

    def _reproduces(self, q, start, record):
        """True only when the issuance record (end_frame, merge_gap_frames,
        min_duration_frames) is re-derived exactly from the verified score
        rows: merging the qualifying runs with that gap yields [start, end_frame),
        whose span meets that minimum, which is never below the source's own.
        A record is never trusted for its membership alone."""
        if not (isinstance(record, tuple) and len(record) == 3):
            return False
        end, gap, minimum = record
        ints = all(type(x) is int for x in (end, minimum)) and (gap is None or type(gap) is int)
        if not ints or (gap is not None and gap < 0) or minimum < self.min_duration_frames:
            return False
        return any(s == start and e == end and e - s >= minimum
                   for s, e, _ in merge_runs(behavior_runs(q), gap))

    def episode_methods(self, episode_ids):
        """Methodology for requested episodes that were merged when returned."""
        merged = [(eid, self._issued[eid]) for eid in episode_ids if eid in self._issued and self._issued[eid][1]]
        if not merged:
            return []
        gaps = sorted({g for _, (_, g, _) in merged})
        return [_method("merged_episodes", len(merged),
                        f"returned with merge_gap_frames {', '.join(map(str, gaps))}: their spans contain frames "
                        "below threshold or without tracking; windows use the episode start only, and exclusions "
                        "use qualifying frames only (merged gap frames are not behavior frames)")]

    def window_methodology(self):
        return [
            _method("event_source", "classifier_prediction",
                    f"episodes and behavior frames are classifier threshold runs ({PREDICTION_EPISODES_VERSION}: "
                    f"tracking_ok and score >= {self.threshold}); not human annotations"),
            _method("onset_meaning", "classifier threshold crossing",
                    "a pre-event window is pre-threshold-crossing, not necessarily pre-behavior"),
            _method("score_threshold", self.threshold, "fixed by the source (provisional)"),
            _method("source_min_duration_frames", self.min_duration_frames, "fixed by the source (provisional)"),
            _method("model_identity", self.model_id, "UNVERIFIED: inferred from the score file name"),
        ]

    def window_limitations(self):
        return [
            "Onset is a classifier threshold crossing: a pre-event window ends before the score first reached "
            "the threshold, which can be before or after the annotated behavior began; it is not necessarily "
            "before the behavior.",
            "Exclusion rules use the classifier's qualifying frames (tracking_ok and score >= threshold) as the "
            "behavior frames; frames the classifier missed are not excluded, and false detections are.",
            *COMMON_LIMITATIONS,
        ]

    def data_items(self):
        return _data_items(self.drop.provenance, self.synthetic)

    def query(self, request):
        self._check()
        wanted = _requested(request, self.sessions, self.policy)
        behavior = request["behavior"]
        if behavior not in score_drops.SCORE_CLASSES:
            raise SourceUnavailable(f"the classifier has no score for {behavior!r}")
        minimum = max(self.min_duration_frames, request["min_duration_frames"] or 0)
        table = self.drop._table
        prov = self.drop.provenance
        episodes = []
        for sid in wanted:
            g = table[table["session"] == sid]
            frames = g["frame"].to_numpy()
            if not np.array_equal(frames, np.arange(len(g))) or \
                    not np.allclose(g["time_s"].to_numpy(), frames / FPS, rtol=0, atol=_TIME_TOL):
                raise SourceUnavailable(f"{sid}: frames / times are not 0..n-1 at {FPS} fps")
            qualifying = g["tracking_ok"].to_numpy(bool) & (g[behavior].to_numpy(float) >= self.threshold)
            for start, end, _ in merge_runs(behavior_runs(qualifying), request["merge_gap_frames"]):
                if end - start < minimum:
                    continue
                self._issued[ic.episode_id(self.source, behavior, sid, start)] = (
                    int(end), request["merge_gap_frames"], minimum)
                episodes.append({
                    "episode_id": ic.episode_id(self.source, behavior, sid, start), "session": sid,
                    "source": self.source, "behavior": behavior, "start_frame": int(start), "end_frame": int(end),
                    "start_s": start / FPS, "end_s": end / FPS, "fps": FPS,
                    "provenance": {"model_id": self.model_id, "score_file_sha256": prov.member_sha256,
                                   "threshold": self.threshold, "min_duration_frames": self.min_duration_frames}})
        methodology = [
            _method("event_source", "classifier_prediction",
                    "per-frame classifier scores (Igor's score drop); not human annotations"),
            _method("episode_rules_version", PREDICTION_EPISODES_VERSION, "fixed by the source, not by the request"),
            _method("episode_definition", "maximal run of frames with tracking_ok and score >= threshold",
                    "start_frame is a classifier threshold crossing, not annotated behavioral onset; tracking loss "
                    "and every frame below threshold end a run; end_frame is exclusive"),
            _method("score_threshold", self.threshold, "fixed by the source (provisional)"),
            _method("source_min_duration_frames", self.min_duration_frames,
                    "fixed by the source; the stricter of it and the request's min_duration_frames applies"),
            _method("merge_gap_frames", request["merge_gap_frames"],
                    "null: runs are never merged; otherwise merged spans contain frames below threshold or "
                    "without tracking"),
            _method("min_duration_frames", request["min_duration_frames"], "measured on the episode span"),
            _method("model_identity", self.model_id, "UNVERIFIED: inferred from the score file name"),
            _method("score_member_sha256", prov.member_sha256, "verified against the archive's SHA256SUMS.txt"),
            _method("ordering_and_truncation", "sorted by (session, start_frame); first max_episodes kept"),
            _method("fps", FPS, "calms21.FPS"),
            *coverage_metadata(_complete_coverage(len(self.sessions)))[0],
        ]
        limitations = list(COMMON_LIMITATIONS)
        limitations.insert(1, "An episode starts where the score first reaches the threshold: a threshold "
                              "crossing, which can precede or follow the annotated behavior's onset.")
        if self.synthetic:
            limitations.insert(0, "SYNTHETIC test fixture: not CalMS21 data, not evidence.")
        return EventQuery(episodes=episodes, sessions_searched=wanted, data=_data_items(prov, self.synthetic),
                          methodology=methodology, limitations=limitations, finalize=self._check)


class MonitorEventSource:
    """monitor_event episodes from a complete, verified frozen monitor run (one rule)."""

    source = "monitor_event"

    def __init__(self, snapshot, drop, policy, *, rule_id):
        _check_policy(policy)
        if not isinstance(snapshot, event_log.RunSnapshot):
            raise ic.ForbiddenEvidence("source", "a frozen event_log.RunSnapshot is required")
        try:
            snap = event_log.load_snapshot(snapshot.canonical, snapshot.sha256)   # re-validated in full
        except event_log.JournalIntegrityError as exc:
            raise ic.ForbiddenEvidence("source", f"the journal snapshot does not verify: {exc}") from None
        if snap.completion != "complete":
            raise ic.ForbiddenEvidence("source", f"the monitor run is not complete ({snap.completion})")
        if snap.provenance_status != "verified_score_drop":
            raise ic.ForbiddenEvidence("source", f"the monitor run's source provenance is {snap.provenance_status}, "
                                       "not verified_score_drop")
        payload = snap.payload()
        header = payload["header"]
        self._rows_sha = _trusted(drop)
        if header["source"] != drop.provenance.to_json() or header["source"]["rows_sha256"] != self._rows_sha:
            raise ic.ForbiddenEvidence("source", "the snapshot was not monitored from this score drop")
        if abs(header["frame_period_s"] - 1.0 / FPS) > _TIME_TOL:
            raise ic.ForbiddenEvidence("source", f"the monitor ran at another frame rate than {FPS} fps")
        rule = next((r for r in header["rules"] if r["rule_id"] == rule_id), None)
        if rule is None:
            raise ValueError(f"rule {rule_id!r} is not a rule of this monitor run")
        self.policy, self.drop, self.snapshot, self.rule = policy, drop, snap, dict(rule)
        self.sessions = _drop_sessions(drop, policy)
        self.synthetic = any("SYNTHETIC" in sid for sid in self.sessions)
        closed = {r["event_id"]: r for r in payload["records"] if r["record"] == "event_closed"}
        self._lifecycles = []
        for fired in (r for r in payload["records"] if r["record"] == "event_fired"):
            if fired["rule_id"] != rule_id:
                continue
            c = closed[fired["event_id"]]                    # load_snapshot guarantees every event closed
            life = {"event_id": fired["event_id"], "session": fired["session"], "run_id": fired["run_id"],
                    "rule_id": rule_id, "behavior": fired["behavior"],
                    "detection_onset_frame": fired["first_frame"], "alert_fired_frame": fired["fired_frame"],
                    "last_frame": c["last_frame"], "closure_end_frame": c["last_frame"] + 1,
                    "start_s": fired["start_s"], "fired_s": fired["fired_s"], "end_s": c["end_s"],
                    "end_reason": c["end_reason"], "action": fired["action"]}
            if fired["session"] not in self.sessions or \
                    abs(life["start_s"] - life["detection_onset_frame"] / FPS) > _TIME_TOL or \
                    abs(life["end_s"] - life["closure_end_frame"] / FPS) > 1e-6:
                raise ic.ForbiddenEvidence("source", f"event {fired['event_id']!r}: frames and times disagree, or "
                                           "its session is not a validation session of the drop")
            self._lifecycles.append(life)
        starts = [(x["session"], x["detection_onset_frame"]) for x in self._lifecycles]
        if len(set(starts)) != len(starts):
            raise ic.ForbiddenEvidence("source", "two events of one rule start on the same frame")

    # ---- the window interface (S9-4) ----

    def n_frames(self, sid):
        return _drop_n_frames(self.drop, sid, self.sessions)

    def behavior_frames(self, sid, behavior):
        """Per-frame bool: inside a bout [first_frame, last_frame + 1) of one of
        this rule's monitor events for `behavior`. Bouts that never fired (shorter
        than the rule's min_duration_s) are not in the journal and are not
        behavior frames here."""
        mask = np.zeros(self.n_frames(sid), dtype=bool)
        for x in self._lifecycles:
            if x["session"] == sid and x["behavior"] == behavior:
                mask[x["detection_onset_frame"]:x["closure_end_frame"]] = True
        return mask

    def is_episode_start(self, sid, behavior, frame):
        return any(x["session"] == sid and x["behavior"] == behavior and x["detection_onset_frame"] == frame
                   for x in self._lifecycles)

    def window_methodology(self):
        r = self.rule
        return [
            _method("event_source", "monitor_event",
                    f"episodes and behavior frames are bouts of fired monitor events ({MONITOR_EPISODES_VERSION}, "
                    f"rule {r['rule_id']}); not human annotations"),
            _method("onset_meaning", "monitor detection onset",
                    "a pre-event window is pre-detection (before the first frame at or above the ask threshold); "
                    "the alert fires later and is not the onset"),
            _method("monitor_rule", f"{r['rule_id']}: {r['behavior']}, ask {r['ask_threshold']}, alert "
                    f"{r['alert_threshold']}, min {r['min_duration_s']} s, mode {r['alert_mode']}"),
            _method("journal_snapshot_sha256", self.snapshot.sha256, "journal-snapshot/1"),
        ]

    def window_limitations(self):
        return [
            "Onset is monitor detection onset (the first frame of the bout at or above the rule's ask "
            "threshold). Detection onset precedes alert firing (the alert fires once the bout has lasted the "
            "rule's minimum duration), so a pre-detection window precedes both; it is not necessarily before "
            "the behavior.",
            "Exclusion rules use the bouts of fired monitor events as the behavior frames; qualifying runs that "
            "never fired (shorter than the rule's minimum duration) are not excluded. Monitor bouts and "
            "classifier threshold runs use different duration rules and are not interchangeable.",
            *COMMON_LIMITATIONS,
        ]

    def data_items(self):
        return _data_items(self.drop.provenance, self.synthetic) + [{
            "kind": "synthetic_fixture" if self.synthetic else "monitor_journal", "path": None,
            "sha256": self.snapshot.sha256,
            "description": f"frozen monitor run {self.snapshot.run_id} (journal-snapshot/1, verified_score_drop, "
                           "complete)"}]

    def lifecycle(self, event_id):
        """Detection onset, alert firing and closure of one event (a copy)."""
        life = next((x for x in self._lifecycles if x["event_id"] == event_id), None)
        if life is None:
            raise KeyError(event_id)
        return dict(life)

    def _check(self):
        if score_drops.trusted_rows_sha256(self.drop) != self._rows_sha:
            raise SourceUnavailable("the score drop changed since this source was created")

    def query(self, request):
        self._check()
        wanted = _requested(request, self.sessions, self.policy)
        if request["merge_gap_frames"]:
            raise SourceUnavailable("monitor events are never merged: one episode is one monitor event")
        behavior = request["behavior"]
        minimum = request["min_duration_frames"] or 0
        episodes = []
        for life in self._lifecycles:
            if life["behavior"] != behavior or life["session"] not in wanted:
                continue
            start, end = life["detection_onset_frame"], life["closure_end_frame"]
            if end - start < minimum:
                continue
            episodes.append({
                "episode_id": ic.episode_id(self.source, behavior, life["session"], start),
                "session": life["session"], "source": self.source, "behavior": behavior,
                "start_frame": start, "end_frame": end, "start_s": start / FPS, "end_s": end / FPS, "fps": FPS,
                "provenance": {"journal_sha256": self.snapshot.sha256, "run_id": life["run_id"],
                               "rule_id": life["rule_id"], "event_id": life["event_id"]}})
        r = self.rule
        methodology = [
            _method("event_source", "monitor_event", "events of one completed, verified monitor run (frozen "
                    "journal snapshot); not human annotations"),
            _method("episode_rules_version", MONITOR_EPISODES_VERSION, "fixed by the source"),
            _method("episode_definition", "monitor bout: [first_frame, last_frame + 1)",
                    "start = detection onset (first frame at or above ask_threshold with tracking); the alert "
                    "fires later (fired_frame, after min_duration_s) and is not a boundary; end = closure, exclusive"),
            _method("monitor_rule", f"{r['rule_id']}: {r['behavior']}, ask {r['ask_threshold']}, alert "
                    f"{r['alert_threshold']}, min {r['min_duration_s']} s, mode {r['alert_mode']}"),
            _method("journal_snapshot_sha256", self.snapshot.sha256, "journal-snapshot/1 (event_log.freeze_run)"),
            _method("monitor_run_id", self.snapshot.run_id),
            _method("merge_gap_frames", request["merge_gap_frames"], "monitor events are never merged"),
            _method("min_duration_frames", request["min_duration_frames"], "measured on the episode span"),
            _method("ordering_and_truncation", "sorted by (session, start_frame); first max_episodes kept"),
            _method("fps", FPS, "calms21.FPS"),
            *coverage_metadata(_complete_coverage(len(self.sessions)))[0],
        ]
        limitations = list(COMMON_LIMITATIONS)
        limitations.insert(1, "An episode is a monitor bout: it starts at detection onset (score at or above "
                              "the rule's ask threshold), not at the annotated behavior's onset, and not at the "
                              "alert, which fires only after the rule's minimum duration.")
        if self.synthetic:
            limitations.insert(0, "SYNTHETIC test fixture: not CalMS21 data, not evidence.")
        data = self.data_items()
        return EventQuery(episodes=episodes, sessions_searched=wanted, data=data, methodology=methodology,
                          limitations=limitations, finalize=self._check)
