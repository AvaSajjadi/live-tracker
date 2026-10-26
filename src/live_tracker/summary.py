"""End-of-session summary from the event log only (new file, D4-1).

A deterministic session ACTIVITY report: what the monitor recorded under the
approved rules and what the researcher answered. It is not a behavior
detector or a statistical analysis: nothing here rescans score frames,
infers behavior or welfare, or computes effects, significance or causes.

Evidence, and only this:
- the run's monitor journal records (event_log: event_fired / event_closed /
  event_answered, optional run_started / run_finished), validated through
  event_log.JournalState, so malformed or contradictory histories are
  refused; only the given run_id is read, and every event must belong to the
  given session. Journal records are what the journal CLAIMS: a header saying
  "verified_load" is reported as a claim, never as verification;
- rule metadata: supplied rules must be approved, unique and agree with the
  journal (the run header's full rule set when there is one, else the rules
  that fired); descriptions are supplied metadata, never journal facts;
- optional, separately identified inputs, each validated and labelled with
  its source: the recording extent (score rows replayed), answer positions
  (replay layer), tracking warnings (a warning source, kept apart from rule
  events), and independent verification of the score source (a ScoreDrop
  issued by score_drops.load_validation_scores, checked with
  score_drops.trusted_rows_sha256 against the journal's header and completion).

Wording discipline: confidence is the classifier's mean score at firing, a
model score, not certainty; events are classifier-score detections under the
rules, not annotations or ground truth; an unanswered ask stays unanswered; a
bout duration is "complete" only when the bout ended below threshold; other
endings (tracking lost, missing score, frame gap, end of recording) give an
observed duration, and an unrecorded closure gives none.

No LLM call. A future wording layer may only rephrase this validated data,
with the deterministic Markdown as the fallback.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

from . import event_log as el
from . import score_drops
from .fake_scores import SYNTHETIC_PREFIX
from .rules import Rule

__all__ = ["SCHEMA", "SessionSummary", "SummaryError", "build_summary", "summary_from_replay"]

SCHEMA = "session-summary/1"
COMPLETE_ENDING = "below_threshold"        # the only ending that leaves no doubt about the bout's end
ENDINGS = {
    "below_threshold": "the score fell below the ask threshold",
    "tracking_lost": "tracking was lost (the behavior may have continued)",
    "missing_score": "a score was missing (the behavior may have continued)",
    "frame_gap": "frames were missing from the recording (the behavior may have continued)",
    "session_end": "the session ended (still going)",
    "stream_end": "the recording ended (still going)",
}
RULE_SETTINGS = ("min_duration_s", "ask_threshold", "alert_threshold", "alert_mode")
CLAIMED_VERIFIED = "journal-reported verified loading — not independently verified"
CONFIDENCE_NOTE = ("Confidence is the classifier's mean score over the bout up to firing: a model score, "
                   "not a probability that the behavior happened and not scientific certainty.")
DETECTION_NOTE = ("Events are detections by the approved rules on classifier scores. They are not human "
                  "annotations or ground truth; no annotation was used for this summary.")
WARNING_KEYS = {"event_id", "session", "rule_id", "behavior", "start_s", "end_s", "confidence", "action",
                "researcher_answer", "evidence"}
WARNING_EVIDENCE = {"mouse": (str, type(None)), "flag": str, "kind": str, "coordinates_present": bool,
                    "ongoing_at_session_end": bool, "message": str}
_TOL = 1e-6


class SummaryError(ValueError):
    """The summary cannot be built honestly from the given evidence (message says why)."""


@dataclass(frozen=True)
class SessionSummary:
    data: dict                      # the structured summary (SCHEMA); JSON-safe

    def as_dict(self):
        return copy.deepcopy(self.data)

    def to_json(self):
        """Stable serialization: sorted keys, fixed separators, no NaN."""
        return json.dumps(self.data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    def to_markdown(self):
        return _markdown(self.data)


# ---- small checks ----

def _int(v):
    return type(v) is int


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ---- provenance ----

def _provenance(session, state, run_id, declared, verified_drop):
    """Labels, from the most conservative reading of the evidence:
    SYNTHETIC for synthetic sessions; "verified score drop" only with an
    independently checked loader-issued drop; CLAIMED_VERIFIED when only the
    journal says so; otherwise "unverified source"."""
    synthetic = str(session).startswith(SYNTHETIC_PREFIX)
    try:
        run = state.run_provenance(run_id)
    except KeyError:
        run = {"status": "no_source_provenance", "completion": "legacy_unrecorded", "header": None, "finished": None}
    claimed = run["status"] == "verified_score_drop"
    verification = "not checked (no independent verification evidence supplied)"
    if verified_drop is not None:
        if not isinstance(verified_drop, score_drops.ScoreDrop):
            raise SummaryError("unsupported verification evidence: only a ScoreDrop from "
                               "score_drops.load_validation_scores can verify a score source")
        trusted = score_drops.trusted_rows_sha256(verified_drop)
        if trusted is None:
            raise SummaryError("the verification ScoreDrop was not issued by the loader, or its rows were changed")
        header, finished = run["header"], run["finished"]
        if header is None or finished is None:
            raise SummaryError("independent verification needs a journal run with a source header and a completion "
                               "record; this run has none")
        if header["source"] != verified_drop.provenance.to_json() or header["source"]["rows_sha256"] != trusted \
                or finished["input_rows_sha256"] != trusted or finished["n_rows"] != verified_drop.provenance.n_frames:
            raise SummaryError("the journal's source and processed rows do not match the independently loaded score "
                               "drop; refused")
        verification = "checked against a loader-issued score drop (score_drops.trusted_rows_sha256)"
    independently_verified = verified_drop is not None
    if synthetic:
        label = "SYNTHETIC"
    elif independently_verified:
        label = "verified score drop"
    elif claimed:
        label = CLAIMED_VERIFIED
    else:
        label = "unverified source"
    return {"label": label, "synthetic": synthetic, "independently_verified": independently_verified,
            "verification": verification, "journal_claim": run["status"], "journal_completion": run["completion"],
            "declared": declared}


# ---- rules ----

def _settings(r):
    return {k: r[k] for k in RULE_SETTINGS}


def _rules(supplied, state, run_id, fired):
    """(rules used, source text): supplied rules checked against the journal."""
    recorded = {}
    for r in fired:
        rule = {"rule_id": r["rule_id"], "behavior": r["behavior"], **_settings(r["rule"])}
        if recorded.setdefault(r["rule_id"], rule) != rule:
            raise SummaryError(f"rule {r['rule_id']!r} has different settings in two events of the run")
    header = state.get("run_started", run_id)
    header_rules = None
    if header is not None:
        header_rules = {h["rule_id"]: {"rule_id": h["rule_id"], "behavior": h["behavior"], **_settings(h)}
                        for h in header["rules"]}
        for rid, rule in recorded.items():
            if header_rules.get(rid) != rule:
                raise SummaryError(f"event rule {rid!r} does not match the run header's rules")
    if supplied is None:
        if header_rules is not None:
            return ([{**header_rules[k], "description": None, "description_source": None,
                      "journal_check": "run header"} for k in sorted(header_rules)],
                    "the run header's complete approved-rule set (no descriptions recorded)")
        return ([{**recorded[k], "description": None, "description_source": None, "journal_check": "fired events"}
                 for k in sorted(recorded)],
                "rules recorded in the journal's events (only rules that fired are known)")
    rules = list(supplied)
    for r in rules:
        if not isinstance(r, Rule):
            raise SummaryError("supplied rules must be rules.Rule objects")
        if r.status != "approved":
            raise SummaryError(f"rule {r.rule_id!r} is {r.status!r}, not approved: only approved rules are monitored")
    ids = [r.rule_id for r in rules]
    if len(set(ids)) != len(ids):
        raise SummaryError("duplicate rule_id in the supplied rules")
    given = {r.rule_id: {"rule_id": r.rule_id, "behavior": r.behavior, "min_duration_s": float(r.min_duration_s),
                         "ask_threshold": float(r.ask_threshold), "alert_threshold": float(r.alert_threshold),
                         "alert_mode": r.alert_mode} for r in rules}
    if header_rules is not None and given != header_rules:
        raise SummaryError("the supplied rules are not the run header's complete approved-rule set "
                           f"(supplied {sorted(given)}, header {sorted(header_rules)}); refused")
    for rid, rule in recorded.items():
        if given.get(rid) != rule:
            raise SummaryError(f"the journal's rule {rid!r} does not match the approved rules supplied")
    out = []
    for r in sorted(rules, key=lambda r: r.rule_id):
        check = "run header" if header_rules is not None else \
            "fired events" if r.rule_id in recorded else "not in the journal (never fired; no header)"
        out.append({**given[r.rule_id], "description": r.description or None,
                    "description_source": "supplied metadata" if r.description else None, "journal_check": check})
    source = ("supplied approved rules, matched exactly with the run header's complete rule set"
              if header_rules is not None else
              "supplied approved rules; only rules that fired could be checked against the journal (it records "
              "no complete rule set)")
    return out, source


# ---- recording, positions, warnings ----

def _recording(recording, lifecycle, header, recorded_frames):
    """The checked recording extent. Every journal frame/time pair must fit
    the recording's sampling under the monitor's time contract: an observed
    row (onset, firing, a closure's last frame) has time_s = first_time_s +
    (frame - first_frame) / fps within the monitor's tolerance; a closure's
    end_s is the nominal, exclusive slot time after last_frame (not an
    observed row). The frame period recorded by every event (and by the run
    header) must be 1 / fps. recorded_frames: the exact recorded frame IDs,
    when known."""
    need = {"n_rows", "first_frame", "last_frame", "first_time_s", "last_time_s", "fps", "source"}
    if not isinstance(recording, dict) or set(recording) != need:
        raise SummaryError(f"recording must have exactly {sorted(need)}")
    n, f0, f1, t0, t1, fps = (recording[k] for k in ("n_rows", "first_frame", "last_frame", "first_time_s",
                                                     "last_time_s", "fps"))
    if not (_int(n) and _int(f0) and _int(f1) and n >= 1 and 0 <= f0 <= f1):
        raise SummaryError("recording rows and frames must be integers: n_rows >= 1, 0 <= first_frame <= last_frame")
    if n > f1 - f0 + 1:
        raise SummaryError(f"recording claims {n} rows over frames {f0}-{f1} ({f1 - f0 + 1} frame slots)")
    if not (_num(t0) and _num(t1) and _num(fps) and fps > 0 and 0 <= t0 <= t1):
        raise SummaryError("recording times must be finite with 0 <= first_time_s <= last_time_s and fps > 0")
    period = 1 / fps
    tol = 2 * el.SAMPLING_TOL * period + 1e-9                 # the monitor's sampling tolerance (two stamps)
    if abs((t1 - t0) - (f1 - f0) * period) > tol:
        raise SummaryError("recording times do not match its frames at its fps")
    if not (isinstance(recording["source"], str) and recording["source"].strip()):
        raise SummaryError("recording needs a source label")
    if header is not None and abs(header["frame_period_s"] - period) > 1e-12 * max(1.0, period):
        raise SummaryError(f"the run header samples at {1 / header['frame_period_s']:g} fps, the recording claims "
                           f"{fps:g} fps")
    domain = None
    if recorded_frames is not None:
        frames = list(recorded_frames)
        if not all(_int(f) for f in frames) or any(b <= a for a, b in zip(frames, frames[1:])):
            raise SummaryError("recorded_frames must be strictly increasing whole frame numbers")
        if len(frames) != n or not frames or frames[0] != f0 or frames[-1] != f1:
            raise SummaryError("recorded_frames do not match the recording's rows and frame range")
        domain = frozenset(frames)

    def slot(frame):
        return t0 + (frame - f0) * period

    for fired, closed in lifecycle:
        eid = fired["event_id"]
        if abs(fired["frame_period_s"] - period) > 1e-12 * max(1.0, period):
            raise SummaryError(f"event {eid!r} was monitored at {1 / fired['frame_period_s']:g} fps, the recording "
                               f"claims {fps:g} fps")
        observed = [(fired["first_frame"], fired["start_s"], "onset"), (fired["fired_frame"], fired["fired_s"], "firing")]
        if closed is not None:
            observed.append((closed["last_frame"], None, "last bout frame"))
        for frame, t, what in observed:
            if not f0 <= frame <= f1 or (domain is not None and frame not in domain):
                raise SummaryError(f"event {eid!r}: its {what} frame {frame} is not a recorded frame")
            if t is not None and abs(t - slot(frame)) > tol:
                raise SummaryError(f"event {eid!r}: {what} at frame {frame} is at {t:g} s, but the recording places "
                                   f"that frame at {slot(frame):g} s")
        if closed is not None and abs(closed["end_s"] - slot(closed["last_frame"] + 1)) > tol:
            raise SummaryError(f"event {eid!r}: its end {closed['end_s']:g} s is not the slot after its last frame "
                               f"({slot(closed['last_frame'] + 1):g} s)")
    return {**recording, "duration_s": t1 - t0 + period, "missing_frames": (f1 - f0 + 1) - n,
            "frame_domain": "exact recorded frames" if domain is not None else
            "extent only (exact frame presence not known)"}, domain


def _positions(answer_positions, events, recording, domain):
    """{event_id: frame} for answered asks only; None means timing unknown.
    With an exact frame domain a position must be a recorded frame."""
    if answer_positions is None:
        return {}
    if not isinstance(answer_positions, dict):
        raise SummaryError("answer_positions must be a mapping of event_id to replay frame")
    by_id = {e["event_id"]: e for e in events}
    out = {}
    for eid, frame in sorted(answer_positions.items(), key=lambda kv: str(kv[0])):
        if frame is None:
            continue                                         # timing unknown: kept unknown
        e = by_id.get(eid)
        if e is None or e["action"] != "ask" or e["_answered"] is None:
            raise SummaryError(f"answer position for {str(eid)[:60]!r}, which is not an answered ask event of this run")
        if not _int(frame) or frame < 0:
            raise SummaryError(f"answer position of {eid!r} must be a whole replay frame >= 0, got {frame!r}")
        if frame < e["fired_frame"]:
            raise SummaryError(f"answer position of {eid!r} (frame {frame}) is before the event fired "
                               f"(frame {e['fired_frame']})")
        if recording is not None and not recording["first_frame"] <= frame <= recording["last_frame"]:
            raise SummaryError(f"answer position of {eid!r} (frame {frame}) is outside the recording")
        if domain is not None and frame not in domain:
            raise SummaryError(f"answer position of {eid!r} (frame {frame}) is not a recorded frame (a gap in the "
                               "recording)")
        out[eid] = frame
    return out


def _warnings(warnings, warning_source, session, recording):
    if warnings is None:
        return {"supplied": False, "source": None, "total": None, "by_kind": None, "warnings": []}
    if not (isinstance(warning_source, str) and warning_source.strip()):
        raise SummaryError("tracking warnings need their named source (warning_source); they are not monitor events")
    rows, seen = [], {}
    for w in warnings:
        if not isinstance(w, dict) or set(w) != WARNING_KEYS or w["action"] != "tracking_warning":
            raise SummaryError("a tracking warning is malformed (§4 tracking_warning entries only)")
        wid = w["event_id"]
        if not (isinstance(wid, str) and wid.strip()):
            raise SummaryError("a tracking warning has no identity")
        if w["session"] != session:
            raise SummaryError(f"warning {wid!r} belongs to another session")
        if wid in seen:
            raise SummaryError(f"warning {wid!r} is given twice" + ("" if seen[wid] == w else " with different content"))
        seen[wid] = w
        if any(w[k] is not None for k in ("rule_id", "behavior", "confidence", "researcher_answer")):
            raise SummaryError(f"warning {wid!r} carries monitor-event fields")
        if w["end_s"] is None:
            raise SummaryError(f"warning {wid!r} has no end: summarize the end-of-session state only")
        if not (_num(w["start_s"]) and _num(w["end_s"])) or w["start_s"] < 0:
            raise SummaryError(f"warning {wid!r} times must be finite and >= 0")
        if w["end_s"] < w["start_s"]:
            raise SummaryError(f"warning {wid!r} ends before it starts")
        ev = w["evidence"]
        if not isinstance(ev, dict) or any(k not in ev or not isinstance(ev[k], t) for k, t in WARNING_EVIDENCE.items()):
            raise SummaryError(f"warning {wid!r} has malformed evidence")
        if recording is not None:
            t0, t1 = recording["first_time_s"], recording["last_time_s"]
            if not (t0 - _TOL <= w["start_s"] and w["end_s"] <= t1 + 1 / recording["fps"] + _TOL):
                raise SummaryError(f"warning {wid!r} lies outside the recording")
        rows.append({"warning_id": wid, "kind": ev["kind"], "mouse": ev["mouse"],
                     "coordinates_present": ev["coordinates_present"], "start_s": w["start_s"], "end_s": w["end_s"],
                     "ongoing_at_session_end": ev["ongoing_at_session_end"], "message": ev["message"]})
    rows.sort(key=lambda r: (r["start_s"], r["warning_id"]))
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {"supplied": True, "source": warning_source, "total": len(rows), "by_kind": dict(sorted(kinds.items())),
            "warnings": rows}


def _closure(fired, closed):
    if closed is None:
        return {"status": "not_recorded", "end_s": None, "last_frame": None, "end_reason": None,
                "ending": "no closure in the journal (the run may have stopped early)",
                "duration_s": None, "duration_kind": "unknown"}
    return {"status": "closed", "end_s": closed["end_s"], "last_frame": closed["last_frame"],
            "end_reason": closed["end_reason"], "ending": ENDINGS[closed["end_reason"]],
            "duration_s": closed["end_s"] - fired["start_s"],
            "duration_kind": "complete" if closed["end_reason"] == COMPLETE_ENDING else "observed"}


# ---- building ----

def build_summary(records, *, run_id, session, rules=None, recording=None, answer_positions=None,
                  warnings=None, warning_source=None, declared_provenance=None, verified_drop=None,
                  recorded_frames=None):
    """A SessionSummary of one run from its journal records.

    records: the journal's records (other runs may be present; only run_id is
    read). rules: the approved rules.Rule objects monitored (None: the
    journal's own record of them). recording: {"n_rows", "first_frame",
    "last_frame", "first_time_s", "last_time_s", "fps", "source"} of the
    replayed rows. answer_positions: {event_id: replay frame, or None when
    unknown}, answered asks only. warnings: §4 tracking_warning entries of the
    end state, which need a warning_source label. declared_provenance: text,
    shown as declared. verified_drop: a loader-issued score_drops.ScoreDrop
    that independently verifies the journal's score source. recorded_frames:
    the exact recorded frame IDs (needs recording); without them, positions
    are only checked against the recording's extent and labelled so."""
    if not isinstance(run_id, str) or not run_id:
        raise SummaryError("an explicit run_id is required")
    if not isinstance(session, str) or not session:
        raise SummaryError("an explicit session is required")
    try:
        state = el.JournalState.from_records(records)
    except el.JournalIntegrityError as exc:
        raise SummaryError(f"the journal is malformed or contradictory: {exc}") from None
    except (TypeError, KeyError, ValueError, AttributeError) as exc:
        raise SummaryError(f"the journal is malformed ({type(exc).__name__}: {exc})") from None
    fired = [r for r in state.records if r["record"] == "event_fired" and r["run_id"] == run_id]
    other = sorted({r["session"] for r in fired} - {session})
    if other:
        raise SummaryError(f"run {run_id!r} holds events of another session ({other[0]!r}); refused")
    used, rules_source = _rules(rules, state, run_id, fired)

    events = []
    for f in fired:
        eid = f["event_id"]
        closed, answered = state.get("event_closed", eid), state.get("event_answered", eid)
        if answered is not None and f["action"] != "ask":
            raise SummaryError(f"{eid!r} is a {f['action']!r} event but has a researcher answer")
        events.append({"event_id": eid, "rule_id": f["rule_id"], "behavior": f["behavior"], "action": f["action"],
                       "onset_s": f["start_s"], "onset_frame": f["first_frame"], "fired_s": f["fired_s"],
                       "fired_frame": f["fired_frame"], "confidence_model_score": f["confidence"],
                       "decision_frames": f["decision_frames"], "closure": _closure(f, closed), "_answered": answered})
    events.sort(key=lambda e: (e["fired_frame"], e["event_id"]))
    if recorded_frames is not None and recording is None:
        raise SummaryError("recorded_frames need the recording they belong to")
    rec, domain = (None, None) if recording is None else _recording(
        recording, [(f, state.get("event_closed", f["event_id"])) for f in fired],
        state.get("run_started", run_id), recorded_frames)
    positions = _positions(answer_positions, events, rec, domain)
    timed = ("recorded replay frame" if domain is not None else
             "within the recording extent; exact frame presence not verified" if rec is not None else
             "position supplied; not checked against a recording")
    for e in events:
        answered = e.pop("_answered")
        if e["action"] == "ask":
            frame = positions.get(e["event_id"])
            e["answer"] = {"researcher_answer": answered["researcher_answer"] if answered else "unanswered",
                           "answered_frame": frame,
                           "timing": ("not answered" if answered is None else
                                      timed if frame is not None else "unknown")}

    counts = {a: sum(e["action"] == a for e in events) for a in el.ACTIONS}
    asks = [e for e in events if e["action"] == "ask"]
    answers = {k: sum(e["answer"]["researcher_answer"] == k for e in asks) for k in (*el.ANSWERS, "unanswered")}
    answers["timing_unknown"] = sum(e["answer"]["timing"] == "unknown" for e in asks)
    per_rule = [{"rule_id": r["rule_id"], **{a: sum(e["action"] == a and e["rule_id"] == r["rule_id"]
                                                  for e in events) for a in el.ACTIONS}} for r in used]
    tracking = _warnings(warnings, warning_source, session, rec)
    prov = _provenance(session, state, run_id, declared_provenance, verified_drop)

    caveats = [DETECTION_NOTE, CONFIDENCE_NOTE,
               "Counts describe this replay of this recording under these rules; they are not rates, effects or "
               "evidence of a behavioral or welfare finding."]
    if prov["synthetic"]:
        caveats.insert(0, "SYNTHETIC test data: not CalMS21 and not predictions of any model.")
    if prov["label"] == CLAIMED_VERIFIED:
        caveats.append("The journal reports a verified score-drop load, but no independent verification was "
                       "supplied: that claim is not checked here.")
    if any(r["description_source"] == "supplied metadata" for r in used):
        caveats.append("Rule descriptions are supplied metadata; the journal does not record them.")
    if any(e["closure"]["duration_kind"] != "complete" for e in events):
        caveats.append("Durations marked observed end because tracking, scores or frames were lost or the recording "
                       "ended: the behavior may have lasted longer. Durations marked unknown have no recorded end.")
    if answers["unanswered"]:
        caveats.append(f"{answers['unanswered']} ask event(s) were not answered; they are neither confirmed nor "
                       "rejected.")
    if answers["timing_unknown"]:
        caveats.append(f"{answers['timing_unknown']} answer(s) have no recorded replay time.")
    if tracking["supplied"]:
        caveats.append("Tracking warnings come from a separate warning source and are not monitor events; "
                       "they are stored apart from the event log (decided with Ava, 2026-10-24). "
                       "Their thresholds are untuned placeholders.")
    else:
        caveats.append("No tracking-warning source was supplied: tracking quality is not summarized here "
                       "(this is not the same as no tracking problems).")
    if rec is not None and rec["missing_frames"]:
        caveats.append(f"{rec['missing_frames']} frame(s) are missing from the recorded rows.")
    if prov["journal_completion"] == "unfinished":
        caveats.append("The journal has no completion record for this run: it may have stopped early.")

    data = {"schema": SCHEMA, "session": session, "run_id": run_id, "provenance": prov, "recording": rec,
            "rules": {"source": rules_source, "rules": used}, "counts": counts, "answers": answers,
            "per_rule": per_rule, "events": events, "tracking_warnings": tracking, "caveats": caveats,
            "evidence": "monitor journal records of this run only" + (
                "; recording extent from the replayed score rows" if rec else "") + (
                "; answer positions from the replay layer" if positions else "") + (
                "; tracking warnings from " + warning_source if tracking["supplied"] else "")}
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"the summary is not JSON-safe ({exc})") from None
    return SessionSummary(data)


def summary_from_replay(replay, *, warning_source=None):
    """The end-of-session summary of a ReplaySession (the end state only:
    never the state of an earlier frame)."""
    end = replay.end_state()
    positions = {}
    for e in end.events:
        if e["action"] == "ask":
            st = replay.answer_status(e["event_id"])
            if st["researcher_answer"] is not None:
                positions[e["event_id"]] = st["answered_frame"]           # None: timing unknown
    recording = {"n_rows": replay.n_frames, "first_frame": replay.frames[0], "last_frame": replay.frames[-1],
                 "first_time_s": replay.times[0], "last_time_s": replay.times[-1], "fps": replay.fps,
                 "source": "replayed score rows"}
    warnings = list(end.warnings) if warning_source else None
    return build_summary(replay.journal.records, run_id=replay.run_id, session=replay.session, rules=replay.rules,
                         recording=recording, answer_positions=positions, warnings=warnings,
                         warning_source=warning_source, recorded_frames=replay.frames)


# ---- Markdown ----

def _s(t):
    return "—" if t is None else f"{t:.2f} s"


def _markdown(d):
    p, c, a = d["provenance"], d["counts"], d["answers"]
    out = [f"# Session summary: {d['session']}", ""]
    if p["synthetic"]:
        out += ["**SYNTHETIC test data: not CalMS21 and not predictions of any model.**", ""]
    out += [f"- Run: `{d['run_id']}`", f"- Source: {p['label']}"
            + (f" (declared: {p['declared']})" if p["declared"] else ""),
            f"- Source verification: {p['verification']}", f"- Evidence: {d['evidence']}", ""]
    r = d["recording"]
    out += ["## Recording", ""]
    if r:
        out += [f"- {r['n_rows']} {r['source']}, frames {r['first_frame']}–{r['last_frame']}, "
                f"{r['duration_s']:.2f} s at {r['fps']:g} fps"
                + (f"; {r['missing_frames']} frame(s) missing" if r["missing_frames"] else ""),
                f"- Frame domain: {r['frame_domain']}", ""]
    else:
        out += ["- Not supplied.", ""]
    out += ["## Rules used", "", f"_{d['rules']['source']}_", ""]
    if d["rules"]["rules"]:
        out += ["| rule | behavior | min duration | ask threshold | alert threshold | mode | checked against | "
                "description (supplied metadata) |", "|---|---|---|---|---|---|---|---|"]
        for x in d["rules"]["rules"]:
            out.append(f"| {x['rule_id']} | {x['behavior']} | {x['min_duration_s']:g} s | {x['ask_threshold']:g} | "
                       f"{x['alert_threshold']:g} | {x['alert_mode']} | {x['journal_check']} | "
                       f"{x['description'] or '—'} |")
    else:
        out.append("- None recorded.")
    out += ["", "## Monitored events", "",
            f"- Alerts: {c['alert']}, asks: {c['ask']}, logs: {c['log']} (total {sum(c.values())})",
            f"- Asks: {a['confirmed']} confirmed, {a['rejected']} rejected, {a['unanswered']} unanswered"
            + (f"; {a['timing_unknown']} answer(s) with unknown replay time" if a["timing_unknown"] else ""), ""]
    if d["events"]:
        out += ["| event | rule | action | onset | fired | ended | duration | confidence (model score) | answer |",
                "|---|---|---|---|---|---|---|---|---|"]
        for e in d["events"]:
            cl = e["closure"]
            ended = "not recorded" if cl["status"] != "closed" else f"{_s(cl['end_s'])}: {cl['ending']}"
            dur = ("—" if cl["duration_s"] is None else
                   f"{cl['duration_s']:.2f} s" + ("" if cl["duration_kind"] == "complete" else " (observed)"))
            ans = "—"
            if "answer" in e:
                x = e["answer"]
                ans = x["researcher_answer"] + (
                    (f" (frame {x['answered_frame']})" if x["timing"] == "recorded replay frame" else
                     f" (frame {x['answered_frame']}, presence unverified)") if x["answered_frame"] is not None
                    else " (time unknown)" if x["timing"] == "unknown" else "")
            out.append(f"| {e['event_id'].rsplit('-', 1)[-1]} | {e['rule_id']} | {e['action']} | "
                       f"{_s(e['onset_s'])} (frame {e['onset_frame']}) | {_s(e['fired_s'])} (frame "
                       f"{e['fired_frame']}) | {ended} | {dur} | {e['confidence_model_score']:.2f} | {ans} |")
    else:
        out.append("No event fired in this run.")
    t = d["tracking_warnings"]
    out += ["", "## Tracking warnings (separate from monitored events)", ""]
    if not t["supplied"]:
        out.append("- No tracking-warning source supplied.")
    else:
        out.append(f"- Source: {t['source']}")
        out.append(f"- Total: {t['total']}" + (" (" + ", ".join(f"{k}: {n}" for k, n in t["by_kind"].items()) + ")"
                                                if t["by_kind"] else ""))
        for w in t["warnings"]:
            end = "still going when the session ended" if w["ongoing_at_session_end"] else f"until {_s(w['end_s'])}"
            coords = "coordinates present but unreliable" if w["coordinates_present"] else "coordinates missing"
            out.append(f"  - {_s(w['start_s'])}, {end}: {w['kind']} (mouse {w['mouse']}, {coords}): {w['message']}")
    out += ["", "## Caveats", ""] + [f"- {x}" for x in d["caveats"]] + [""]
    return "\n".join(out)
