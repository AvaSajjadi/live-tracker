"""Tests for Findings v2 checkpoint M2: deterministic findings/2 rendering and
legacy quarantine (new file).

Evidence comes from real Stage 6 runs over the Stage 4 SYNTHETIC fixtures
(and re-hashed forgeries of them); the original live run is read, never
written. No model or network is used.
"""

import copy
import hashlib
import json
import re

import pytest

import live_tracker.investigation_claims as cl
import live_tracker.investigation_context as ctx
import live_tracker.investigation_contracts as ic
import live_tracker.investigation_report as ir
import live_tracker.investigation_report_v2 as v2
from live_tracker.investigation_agent import Budget
from test_investigation_agent import investigate  # legacy/1, explicitly (M3)
from live_tracker.investigation_events import CalMS21InvestigationTools
from live_tracker.investigation_report import ReportIntegrityError
import live_tracker.investigation_agent as ia
from test_investigation_agent import (GOOD, LOOSE, QUESTION, Investigator, Scripted, _until_compare, call,  # noqa: F401
                                      finish, last_result, no_network, query)
from test_investigation_claims import (C, dirty_run, effect_run, env, ev, insufficient_run,  # noqa: F401
                                       scripted_compare, sensitivity_ids)
from test_investigation_context import ARTIFACTS, LEGACY_RENDER, LIVE_RUN, REAL_MANIFEST, needs_live, sha
from test_investigation_report import forge
from test_investigation_windows import VAL_A, VAL_B, VAL_C

BANNED = re.compile(r"\b(proves?|proven|shows that|demonstrates?|establishes|causes|caused by|significant(ly)?|"
                    r"predicts?|confirms?|large|small|strong(ly)?|weak(ly)?|substantial(ly)?|dramatic(ally)?|"
                    r"marked(ly)?|modest(ly)?|slight(ly)?|huge|clearly|definitely|robust(ly|ness)?|"
                    r"generali[sz]es?|representative|population)\b", re.IGNORECASE)


# ---- helpers ----

def payload(report, policy, finding, primary=None, claims=(), observations=(), next_=()):
    """A findings/2 payload for a finished Stage 6 run: its evidence and trace
    (the finish carries the structured proposal, never legacy prose) and the
    fingerprint the registry computes when the findings are frozen."""
    proposal = {"finding": finding, "primary_evidence_id": primary, "observations": list(observations),
                "claims": list(claims), "next_investigations": list(next_)}
    trace = copy.deepcopy(report.trace)
    for t in trace:
        if t["kind"] == "finish":
            t["arguments"] = copy.deepcopy(proposal)
    r = cl.validate_findings(proposal, evidence=report.evidence, trace=trace, policy=policy, mode="complete",
                             stopping_reason=report.stopping_reason, finished=report.finished)
    assert r.accepted, r.problems
    return {"findings_format": "findings/2", "registry_version": "claims/2.0", "grid_version": "none",
            "question": report.question, "finished": report.finished, "stopping_reason": report.stopping_reason,
            **copy.deepcopy(proposal), "evidence": copy.deepcopy(report.evidence), "trace": trace,
            "usage": copy.deepcopy(report.usage), "fingerprint": r.fingerprint}


def incomplete_payload(report, policy):
    r = cl.incomplete_findings(evidence=report.evidence, trace=report.trace, policy=policy,
                               stopping_reason=report.stopping_reason)
    assert r.accepted, r.problems
    return {"findings_format": "findings/2", "registry_version": "claims/2.0", "grid_version": "none",
            "question": report.question, "finished": False, "stopping_reason": report.stopping_reason,
            "finding": "incomplete", "primary_evidence_id": None, "observations": [], "claims": [],
            "next_investigations": [], "evidence": copy.deepcopy(report.evidence),
            "trace": copy.deepcopy(report.trace), "usage": copy.deepcopy(report.usage), "fingerprint": r.fingerprint}


def refused(fn, text):
    with pytest.raises(ReportIntegrityError) as e:
        fn()
    assert any(text in p for p in e.value.problems), e.value.problems
    return e.value


def section(md, title):
    m = re.search(rf"^#+ {re.escape(title)}\n(.*?)(?=^#+ |\Z)", md, re.S | re.M)
    assert m, title
    return m.group(1)


def snap(report, eid):
    return next(e for e in report.evidence if e["evidence_id"] == eid)


def sentence(doc, claim_type, **params):
    return next(c["sentence"] for c in doc["claims"] if c["type"] == claim_type
                and all(c["params"].get(k) == v for k, v in params.items()))


@pytest.fixture
def effect_doc(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    p = payload(report, policy, "effect_described", pid,
                claims=[C("descriptive_effect", pid), C("coverage", aspect="primary_comparison"),
                        C("potential_confounder", pid, confounder="investigation_label"),
                        C("measurement_context", pid, aspect="session_timing"),
                        C("measurement_context", pid, aspect="pixel_units"),
                        C("measurement_context", ev(report, "get_window_features"), aspect="mover_unidentified"),
                        C("tool_disclosure", ev(report, "query_events"), disclosure="approach_unlabelled"),
                        C("methodological_limitation", pid, kind="descriptive_only"),
                        C("operational_fact", kind="model_selected_plan")],
                observations=[{"evidence_id": pid, "field": "measurements.effect.value"},
                              {"evidence_id": pid, "field": "measurements.n_sessions_paired"}])
    doc, md = v2.render_v2(p, policy=policy)
    return report, policy, p, doc, md


# ======================================================================
# 1. exact templates and placement, end to end
# ======================================================================

def test_descriptive_effect_exact_sentence_in_summary_and_finding(effect_doc):
    report, policy, p, doc, md = effect_doc
    pid = p["primary_evidence_id"]
    m = snap(report, pid)["result"]["measurements"]
    paired = sorted(s["session"] for s in m["per_session"] if s["difference"] is not None)
    value = m["effect"]["value"]
    expected = (f"Across the 2 contributing paired sessions ({', '.join(paired)}), the median of per-session "
                f"differences (pre-event minus baseline) in dist_rate was {ctx._fmt(value)} px/s "
                f"({ctx._sign(value)}). Descriptive only: no interval or test; not a causal reading; it describes these "
                "sessions, not the whole authorized split. Cross-snapshot agreement: not_verified.")
    assert sentence(doc, "descriptive_effect") == expected
    assert doc["executive_summary"][2] == expected                           # the identical sentence
    assert f"- {expected}" in section(md, "Executive summary")
    assert expected in section(md, "Finding")
    assert doc["status"]["accepted_direction"] == ctx._sign(value) and doc["status"]["publication"] == "publishable"


def test_placement_of_every_family(effect_doc):
    _, _, _, doc, md = effect_doc
    placed = {(c["type"], c["section"]) for c in doc["claims"]}
    assert placed == {("descriptive_effect", "finding"), ("coverage", "coverage"),
                      ("potential_confounder", "potential_confounders"),
                      ("measurement_context", "measurement_context"), ("tool_disclosure", "tool_disclosures"),
                      ("methodological_limitation", "measured_limitations"), ("operational_fact", "operational_facts")}
    assert sentence(doc, "potential_confounder") in section(md, "Potential confounders (measured; not established)")
    assert sentence(doc, "tool_disclosure") in section(md, "Tool disclosures (not measurements of this data)")
    assert sentence(doc, "operational_fact") in section(md, "Operational facts")
    assert sentence(doc, "methodological_limitation") in section(md, "Measured limitations")
    for title in ("Potential confounders", "Tool disclosures", "Operational facts", "Measured limitations"):
        assert not any(c["sentence"] in section(md, "Executive summary") for c in doc["claims"]
                       if c["type"] != "descriptive_effect")


def finished_partial(env):
    turns = [[query(sessions=[VAL_A, VAL_B])]] + _until_compare(LOOSE)[1:] + [finish(GOOD)]
    report = investigate(QUESTION, env[0], env[1], Scripted(*turns), sufficiency=LOOSE)
    assert report.finished, report.stopping_reason
    return report


def test_partial_coverage_is_stated_inside_the_claim_and_the_summary(env):
    report = finished_partial(env)
    pid = ev(report, "compare_windows")
    doc, md = v2.render_v2(payload(report, env[1], "effect_described", pid, claims=[C("descriptive_effect", pid)],
                                   next_=[{"action": "include_more_sessions", "params": {}, "because": [0]}]),
                           policy=env[1])
    s = sentence(doc, "descriptive_effect")
    assert s.endswith(" Coverage: partial: 2 of 3 eligible sessions were searched for attack (source "
                      "validation_annotation, behavior attack, no merge gap, no minimum duration; not searched: "
                      f"{VAL_C}); the unsearched sessions are not described.")
    assert VAL_A in s and VAL_B in s and s in doc["executive_summary"]
    assert not re.search(r"\b(all|whole|every) (sessions|mice|animals|split)\b", s.replace("not the whole authorized "
                                                                                           "split", ""))
    assert doc["next_investigations"][0]["sentence"] == ("Suggested by the model (not evidence): search the eligible "
                                                         "sessions that were not searched (motivated by claims 0).")
    assert doc["next_investigations"][0]["sentence"] in section(md, "Suggested by the model (not evidence)")


def test_insufficient_evidence_exact_sentence_and_no_values(insufficient_run):
    report, policy = insufficient_run
    pid = ev(report, "compare_windows")
    doc, md = v2.render_v2(payload(report, policy, "insufficient_evidence", pid,
                                   claims=[C("sufficiency_not_met", pid)]), policy=policy)
    m = snap(report, pid)["result"]["measurements"]
    expected = (f"The dist_rate comparison ({pid}) did not meet the provisional sufficiency criteria (unmet: "
                f"{', '.join(m['sufficiency']['reasons_not_met'])}; minimum 4 sessions, 10 windows per group). It "
                f"paired {m['n_sessions_paired']} sessions from {m['n_windows_pre']} pre-event and "
                f"{m['n_windows_baseline']} baseline windows. No aggregate effect is reported.")
    assert sentence(doc, "sufficiency_not_met") == expected and expected in doc["executive_summary"]
    for s in m["per_session"]:                       # no per-session value reaches the Markdown
        for v in (s["difference"], s["pre_summary"], s["baseline_summary"]):
            if v is not None:
                assert ctx._fmt(v) not in md
    assert doc["status"]["accepted_direction"] is None and "Accepted direction | none" in md


def test_no_comparison_no_data_and_unavailable_sentences(env, monkeypatch):
    stop = lambda m: [call(ia.FINISH, finding="insufficient_evidence", primary_evidence_id=None,  # noqa: E731
                           observations=[{"evidence_id": last_result(m)["evidence_id"], "field": "status"}])]
    report = investigate(QUESTION, env[0], env[1], Scripted([query()], stop), budget=Budget(max_model_turns=2))
    assert report.finished, (report.stopping_reason, report.trace[-2:])
    doc, _ = v2.render_v2(payload(report, env[1], "insufficient_evidence", None, claims=[C("no_comparison")]),
                          policy=env[1])
    assert sentence(doc, "no_comparison") == "No comparison was run." and "No comparison was run." in \
        doc["executive_summary"]
    tpl = v2.TEMPLATES
    eid = "ev:compare_windows:" + "a" * 24
    assert tpl["comparison_no_data"]({"evidence_ids": [eid], "params": {}, "grounding": {
        "feature": "dist_rate", "n_windows_pre": 0, "n_windows_baseline": 3}}) == (
        f"The dist_rate comparison ({eid}) returned no data: 0 pre-event and 3 baseline windows were usable. No "
        "aggregate effect is reported.")
    assert tpl["comparison_unavailable"]({"evidence_ids": [eid], "params": {}, "grounding": {
        "feature": "dist_rate", "error_codes": ["internal_error"]}}) == (
        f"The dist_rate comparison ({eid}) returned an error (error codes: internal_error). No aggregate effect is "
        "reported.")


# ======================================================================
# 2. exact wording of every family (hand-built grounding)
# ======================================================================

E1, E2, E3 = ("ev:compare_windows:" + "1" * 24, "ev:compare_windows:" + "2" * 24,
              "ev:check_tracking_quality:" + "3" * 24)
EF, EQ = "ev:get_window_features:" + "4" * 24, "ev:query_events:" + "5" * 24
GROUP = {"source": "validation_annotation", "behavior": "attack", "merge_gap_frames": 5, "min_duration_frames": None,
         "query_evidence_ids": [EQ], "status": "partial", "basis": "split manifest via the evidence policy "
         "(validation split)", "eligible_count": 3, "eligible": ["a", "b", "c"], "available_count": 3,
         "searched": ["a", "b"], "not_searched": ["c"], "with_behavior": ["a", "b"], "truncated": False}


def T(claim_type, ids, grounding, **params):
    return v2.TEMPLATES[claim_type]({"type": claim_type, "evidence_ids": list(ids), "params": params,
                                     "grounding": grounding})


@pytest.mark.parametrize("claim_type, ids, params, grounding, expected", [
    ("coverage", [], {"aspect": "primary_comparison"},
     {"status": "partial", "reason": None, "group": 0, "contributing": ["a"], "excluded": [
         {"session": "b", "reason": "no eligible baseline window"}], "query_group": GROUP},
     "Coverage of the primary comparison: partial: 2 of 3 eligible sessions were searched for attack (source "
     "validation_annotation, behavior attack, merge gap 5 frames, no minimum duration; eligibility: split manifest "
     "via the evidence policy (validation split); not searched: c); 1 contributing sessions (a); searched sessions "
     "with attack not contributing: b (no eligible baseline window)."),
    ("coverage", [], {"aspect": "primary_comparison"},
     {"status": "unknown", "reason": "the compared windows come from more than one event source or behavior",
      "group": None, "contributing": ["a"], "excluded": None, "query_group": None},
     "Coverage of the primary comparison: unknown: the compared windows come from more than one event source or "
     "behavior; 1 contributing sessions (a)."),
    ("coverage", [EQ], {"aspect": "query_group"}, GROUP,
     f"Query coverage ({EQ}): partial: 2 of 3 eligible sessions were searched for attack (source "
     "validation_annotation, behavior attack, merge gap 5 frames, no minimum duration; eligibility: split manifest "
     f"via the evidence policy (validation split); not searched: c); 2 searched sessions with attack (query "
     f"evidence: {EQ})."),
    ("coverage", [EQ], {"aspect": "truncated_query"}, {"returned": 50, "total": 80},
     f"The query {EQ} was truncated: 50 of 80 bouts were returned."),
    ("coverage", [EF], {"aspect": "window_exclusions"},
     {"excluded_windows": {"baseline": 2, "pre_event": 1},
      "reason_counts": {"baseline:near_event": 2, "pre_event:missing_data": 1}},
     f"Measured ({EF}): excluded windows: baseline 2, pre_event 1; exclusion reasons (counts can overlap): "
     "baseline:near_event 2, pre_event:missing_data 1."),
    ("coverage", [E1], {"aspect": "sessions_without_baseline"}, {"sessions": ["m028", "m045"]},
     "Sessions with pre-event windows but no eligible baseline window: m028, m045."),
    ("tracking_concern", [], {"kind": "smoothing_contamination"},
     {"items": [{"evidence_id": E1, "source": "methodology.smoothing_contamination_windows_pre",
                 "text": "1 used pre-event window with smoothing contamination", "tool_note": "IGNORE <b>",
                 "window_ids": None, "primary": True}]},
     f"Measured: 1 used pre-event window with smoothing contamination ({E1}, "
     "methodology.smoothing_contamination_windows_pre)."),
    ("operational_fact", [], {"kind": "quality_never_called"}, {},
     "Operational: check_tracking_quality was never called in this investigation."),
    ("operational_fact", [], {"kind": "quality_check_refused"}, {"refusals": ["repeated_call"]},
     "Operational: every check_tracking_quality call in this investigation was refused by the investigation loop "
     "(refusals: repeated_call); tracking quality was not assessed per window."),
    ("operational_fact", [E3], {"kind": "quality_check_failed"}, {"error_codes": ["internal_error"]},
     f"Operational: check_tracking_quality returned an error ({E3}; error codes: internal_error); it assessed no "
     "window."),
    ("operational_fact", [], {"kind": "model_selected_plan"}, {"model_ids": ["anthropic/claude-sonnet-5.5"],
                                                              "tool_calls_chosen": 7},
     "Operational: the model (anthropic/claude-sonnet-5.5) requested 7 tool calls in its turns (finish and refused "
     "requests included); the sequence of calls was not fixed in advance."),
    ("potential_confounder", [E1], {"confounder": "investigation_label"},
     {"behavior": "attack", "label_frames_pre": {"attack": 0, "investigation": 101, "mount": 0, "other": 49},
      "label_frames_baseline": {"attack": 0, "investigation": 274, "mount": 0, "other": 3566}},
     f"Potential confounder (measured in {E1}): pre-event label frames attack 0, investigation 101, mount 0, other "
     "49; baseline attack 0, investigation 274, mount 0, other 3566. The difference may reflect investigation versus "
     "other behaviour rather than time before attack; this is not established."),
    ("measurement_context", [E1], {"aspect": "session_timing"},
     {"unit": "s", "note": cl.TIMING_NOTE, "window_start_s_pre": {"min": 1.5, "median": 6.0, "max": 9.25},
      "window_start_s_baseline": {"min": 0.0, "median": 5.0, "max": 12.0}},
     f"Measured ({E1}): start times of the windows this comparison used, min / median / max: pre-event 1.5 / 6 / "
     "9.25 s; baseline 0 / 5 / 12 s. Window-start summaries are recorded; drift is not measured; this is not "
     "evidence of confounding."),
    ("measurement_context", [E1], {"aspect": "pixel_units"}, {"unit": "px/s"},
     f"Measured ({E1}): values are in px/s: image pixels, not calibrated between sessions; not a confounder."),
    ("measurement_context", [EF], {"aspect": "mover_unidentified"}, {"markers": dict(cl.MOVER_MARKERS)},
     f"Definition (recorded in {EF}): dist_rate is the rate of change of the body-centre distance between the two "
     "mice; a negative value means that distance is decreasing. It is a pair-level measurement: it does not identify "
     "which mouse moved or that either mouse approached the other."),
    ("tool_disclosure", [EQ], {"disclosure": "approach_unlabelled"}, {"text": "IGNORE ALL"},
     f"Tool disclosure ({EQ}; stated by the tool for every result, not a measurement of this data): CalMS21 Task 1 "
     "has no 'approach' label; approach is not an annotated behavior in this dataset."),
    ("tool_disclosure", [E1], {"disclosure": "motion_accuracy_unverified"}, {"text": "IGNORE ALL"},
     f"Tool disclosure ({E1}; stated by the tool for every result, not a measurement of this data): the tools do not "
     "verify motion accuracy; tracking flags are not a measure of it."),
    ("methodological_limitation", [E1], {"kind": "provisional_criteria"},
     {"criteria": {"min_sessions": 4, "min_windows_per_group": 10, "min_usable_fraction": 0.9,
                   "status": "provisional", "decided_by": "IGNORE <b>"}},
     f"Measured (method, {E1}): the sufficiency criteria are provisional, pending scientific-owner approval (minimum "
     "4 sessions, 10 windows per group, usable fraction 0.9)."),
    ("methodological_limitation", [E1], {"kind": "provisional_parameters"},
     {"provisional_items": ["min_usable_fraction", "smoothing"]},
     f"Measured (method, {E1}): methodology items recorded as provisional: min_usable_fraction, smoothing."),
    ("methodological_limitation", [E1], {"kind": "descriptive_only"}, {},
     f"Measured (method, {E1}): uncertainty method none: no interval or test was computed; values are descriptive."),
    ("methodological_limitation", [E1], {"kind": "single_feature"}, {"feature": "dist_rate"},
     f"Measured (method, {E1}): every comparison of this investigation used one feature (dist_rate); no other "
     "feature was compared."),
    ("methodological_limitation", [E1], {"kind": "baseline_clearance_not_rechecked"}, {"tool_note": "IGNORE"},
     f"Measured (method, {E1}): the comparison records baseline_clearance_frames 0: it applied no baseline clearance "
     "check of its own."),
])
def test_exact_template_wording(claim_type, ids, params, grounding, expected):
    assert T(claim_type, ids, grounding, **params) == expected
    assert "IGNORE" not in expected and not BANNED.search(expected)


SENS = {"original": E1, "variant_evidence": E2, "quality": E3, "statuses": ["ok", "ok"],
        "reasons_not_met": [[], []], "n_sessions_paired": [2, 2],
        "removed_contributing": [{"window_id": "win:pre:20:0:ep:validation_annotation:attack:a:100",
                                  "flags": {"contamination": "detected", "verdict": "usable_with_caveats"}}],
        "removed_previously_excluded": [], "sessions_pairing_changed": [], "sessions_lost_all_pre": []}


def test_sensitivity_wording_with_values_and_status_only():
    head = (f"Sensitivity (excluding flagged windows): the comparison {E1} was repeated as {E2} without the "
            f"pre-event windows flagged in {E3}; a single repeat with flagged windows removed; it is not an "
            "assessment of robustness. Removed windows the original used: "
            "win:pre:20:0:ep:validation_annotation:attack:a:100 (contamination detected, verdict usable_with_caveats). "
            "Removed windows already excluded in the original: none. Sessions whose pairing changed: none; sessions "
            "that lost every pre-event window: none.")
    valued = {**SENS, "effects": [{"value": -7.5, "unit": "px/s"}, {"value": -6.25, "unit": "px/s"}],
              "arithmetic_sign_relation": "same arithmetic sign", "sign_relation_note": "x"}
    assert T("sensitivity_excluding_flagged", [E1, E2, E3], valued) == head + (
        " Recorded effects: original -7.5 px/s, variant -6.25 px/s; the two recorded values have the same arithmetic "
        "sign; this is an arithmetic comparison of two descriptive values, not scientific agreement.")
    unstated = {**valued, "effects": [{"value": -7.5, "unit": None}, {"value": 1.0, "unit": "px/s"}],
                "arithmetic_sign_relation": None}
    assert "original -7.5 (unit not recorded), variant 1 px/s; sign relation not stated (units not both recorded or " \
           "differ); this is an arithmetic comparison" in T("sensitivity_excluding_flagged", [E1, E2, E3], unstated)
    status_only = {**SENS, "statuses": ["ok", "insufficient_evidence"], "reasons_not_met": [[], ["min_sessions"]],
                   "n_sessions_paired": [2, 1]}
    text = T("sensitivity_excluding_flagged", [E1, E2, E3], status_only)
    assert text == head + (" Statuses: original ok, variant insufficient_evidence; unmet sufficiency reasons: "
                           "original none, variant min_sessions; paired sessions: original 2, variant 1. No effect "
                           "value or sign is reported for this repeat.")
    assert not re.search(r"-?\d+\.\d+|positive|negative|sign\b(?! is reported)", text.split("Statuses:")[1])


def test_an_exclusion_of_no_used_window_is_not_a_sensitivity_intervention():
    g = {**SENS, "removed_contributing": [], "removed_previously_excluded": [
        {"window_id": "win:pre:20:0:ep:validation_annotation:attack:a:100", "original_reason": "missing_data"}]}
    text = T("sensitivity_excluding_flagged", [E1, E2, E3], g)
    assert ("Removed windows the original used: none; no window the original comparison used was removed, so the "
            "variant used the same windows: this repeat is not a sensitivity intervention.") in text
    assert "(original reason missing_data)" in text


def test_sensitivity_end_to_end_is_in_the_findings_section_only(dirty_run):
    report, policy = dirty_run
    ids = sensitivity_ids(report)
    doc, md = v2.render_v2(payload(report, policy, "effect_described", ids[0],
                                   claims=[C("descriptive_effect", ids[0]), C("sensitivity_excluding_flagged", *ids)]),
                           policy=policy)
    s = sentence(doc, "sensitivity_excluding_flagged")
    assert s in section(md, "Sensitivity (findings section only)")
    assert not any("Sensitivity" in x or ids[1] in x for x in doc["executive_summary"])
    assert s not in section(md, "Executive summary")
    assert re.findall(r"agreement", s) == ["agreement"] and s.count("not scientific agreement") == 1
    assert re.findall(r"robust\w*", s) == ["robustness"] and "not an assessment of robustness" in s
    inc, md2 = v2.render_v2(payload(report, policy, "inconclusive", ids[0],
                                    claims=[C("sensitivity_excluding_flagged", *ids)]), policy=policy)
    s2 = sentence(inc, "sensitivity_excluding_flagged")
    assert "Recorded effects" not in s2 and "No effect value or sign is reported" in s2
    for k in (0, 1):
        v = snap(report, ids[k])["result"]["measurements"]["effect"]["value"]
        assert v is None or ctx._fmt(v) not in md2


# ======================================================================
# 3. mover_unidentified: the approved exception, implemented
# ======================================================================

def test_mover_unidentified_is_grounded_in_features_evidence_only(effect_run):
    report, policy = effect_run
    pid, fid = ev(report, "compare_windows"), ev(report, "get_window_features")
    limit = C("methodological_limitation", pid, kind="provisional_criteria")
    from test_investigation_claims import check, refused as claim_refused
    ok = check(report, policy, "effect_described", pid,
               claims=[C("descriptive_effect", pid), C("measurement_context", fid, aspect="mover_unidentified")])
    assert ok.accepted and ok.claims[1]["grounding"]["category"] == "definitional_fact", ok.problems
    claim_refused(check(report, policy, "inconclusive", pid,
                        claims=[C("measurement_context", pid, aspect="mover_unidentified"), limit]),
                  "is a compare_windows result, not get_window_features")
    claim_refused(check(report, policy, "inconclusive", pid,
                        claims=[C("measurement_context", fid, aspect="mover_unidentified")]),
                  "evidence-grounded limitation")                      # never justifies inconclusive
    for mutate, text in [
        (lambda req, res: [m.update(value="negative = mouse approaching") for m in res["methodology"]
                           if m["name"] == "dist_rate_meaning"], "dist_rate_meaning marker"),
        (lambda req, res: res.update(methodology=[m for m in res["methodology"] if m["name"] != "dist_rate_meaning"]),
         "dist_rate_meaning marker"),
        (lambda req, res: res.update(limitations=[t for t in res["limitations"] if "which mouse" not in t]),
         "pair-level qualification"),
    ]:
        forged = forge(report, "get_window_features", mutate, check=False)
        claim_refused(check(forged, None, "inconclusive", None,
                            claims=[C("measurement_context", ev(forged, "get_window_features"),
                                      aspect="mover_unidentified"),
                                    C("operational_fact", kind="quality_check_refused")]), text)


def test_mover_requires_dist_rate_among_the_requested_features(env):
    feats = call("get_window_features", episode_ids=["ep:validation_annotation:attack:" + VAL_A + ":100"],
                 pre_event={"length_frames": 20, "gap_frames": 0}, features=["dist_center"])
    report = investigate(QUESTION, env[0], env[1], Scripted([feats]), budget=Budget(max_model_turns=1))
    from test_investigation_claims import check, refused as claim_refused
    fid = ev(report, "get_window_features")
    claim_refused(check(report, env[1], "insufficient_evidence", None,
                        claims=[C("no_comparison"), C("measurement_context", fid, aspect="mover_unidentified")]),
                  "dist_rate is not among")


def test_no_mover_identity_causal_or_generalised_wording(effect_doc, dirty_run):
    _, _, _, doc, md = effect_doc
    report, policy = dirty_run
    ids = sensitivity_ids(report)
    _, md2 = v2.render_v2(payload(report, policy, "inconclusive", ids[0],
                                  claims=[C("sensitivity_excluding_flagged", *ids),
                                          C("tracking_concern", kind="quality_contamination_detected")]),
                          policy=policy)
    for text in (md, md2):
        body = text.replace(v2.MOVER_TEXT, "").replace(v2.NOT_ROBUSTNESS, "").replace(QUESTION, "")
        assert not re.search(r"\b(approach(ed|es|ing)?|chas\w+|pursu\w+|the (resident|intruder) mouse)\b",
                             body.replace("'approach' label; approach is not an annotated behavior", ""))
        assert not BANNED.search(body.replace(v2.DESCRIPTIVE_POLICY, "")), BANNED.search(body)


# ======================================================================
# 4. determinism and JSON round trip
# ======================================================================

def test_rerender_and_json_round_trip_are_byte_identical(effect_doc):
    report, policy, p, doc, md = effect_doc
    doc2, md2 = v2.render_v2(copy.deepcopy(p), policy=policy)
    assert ir.to_json_text(doc) == ir.to_json_text(doc2) and md == md2
    loaded = json.loads(ir.to_json_text(doc))
    assert v2.render_markdown_v2(loaded, policy=policy) == md
    assert ir.to_json_text(v2.verify_saved_v2(loaded, policy=policy)) == ir.to_json_text(doc)
    again, md3 = v2.render_v2(json.loads(json.dumps(p)), policy=policy)
    assert ir.to_json_text(again) == ir.to_json_text(doc) and md3 == md
    assert doc["format"] == {"findings_format": "findings/2", "registry_version": "claims/2.0", "grid_version": "none",
                             "report_schema": "live-tracker-investigation-report/0.2", "fingerprint_version": "fp/1"}
    assert md.startswith("# Investigation report (findings/2)\n\nFormat: `findings/2` · registry `claims/2.0` · grid "
                         "`none` · report schema `live-tracker-investigation-report/0.2`")


# ---- saved documents are untrusted: coordinated edits (Codex 1 M2 finding) ----

def retemplate(doc):
    """A careful forger: after editing, regenerate every sentence and the summary
    with the renderer's own templates so the document is internally consistent."""
    for c in doc["claims"]:
        c["sentence"] = v2.TEMPLATES[c["type"]](c)
    doc["executive_summary"] = v2._executive_summary(doc)
    return doc


def diagnostic_doc(effect_run):
    report, _ = effect_run
    pid = ev(report, "compare_windows")
    return v2.build_report_v2(payload(report, None, "effect_described", pid, claims=[C("descriptive_effect", pid)]))


def test_codex_1_effect_value_with_sentence_and_summary_is_refused(effect_doc):
    *_, policy, p, doc, md = effect_doc
    assert doc["claims"][0]["grounding"]["value"] < 0
    bad = copy.deepcopy(doc)
    bad["claims"][0]["grounding"].update(value=123456.0, direction="positive")
    bad["status"]["accepted_direction"] = "positive"
    retemplate(bad)
    assert "123456 px/s (positive)" in bad["executive_summary"][2]           # the forgery is self-consistent
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), "differs from its reconstruction")
    # also editing the canonical evidence value: L0 content consistency refuses it
    bad2 = copy.deepcopy(bad)
    snap2 = next(e for e in bad2["canonical"]["evidence"] if e["evidence_id"] == p["primary_evidence_id"])
    snap2["result"]["measurements"]["effect"]["value"] = 123456.0
    refused(lambda: v2.render_markdown_v2(bad2, policy=policy), "claims:")


def test_codex_2_causal_statistical_policy_is_refused(effect_doc):
    *_, policy, _, doc, _ = effect_doc
    bad = copy.deepcopy(doc)
    bad["statistical_policy"] = "Approach causes attack (p < 0.001); the effect is significant."
    bad["executive_summary"][-1] = bad["statistical_policy"]
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), "statistical_policy")


def test_codex_3_diagnostic_cannot_be_promoted_to_publishable(effect_run):
    _, policy = effect_run
    doc = diagnostic_doc(effect_run)
    assert doc["status"]["publication"] == "diagnostic"
    assert v2.render_markdown_v2(copy.deepcopy(doc)).startswith("# Investigation report")    # untouched: renders
    bad = copy.deepcopy(doc)
    bad["status"]["publication"] = "publishable"
    bad["verification"]["authorization"] = "rechecked"
    bad["executive_summary"] = v2._executive_summary(bad)
    assert bad["executive_summary"][0].startswith("Publishable")
    refused(lambda: v2.render_markdown_v2(bad), "reconstructed under the policy given now")
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), "differs from its reconstruction")


def test_publishable_document_needs_the_policy_now(effect_doc):
    *_, policy, _, doc, md = effect_doc
    assert v2.render_markdown_v2(copy.deepcopy(doc), policy=policy) == md
    refused(lambda: v2.render_markdown_v2(copy.deepcopy(doc)), "reconstructed under the policy given now")


def test_record_only_cannot_become_publishable(env):
    report = scripted_compare(env)
    doc = v2.build_report_v2(incomplete_payload(report, env[1]), policy=env[1])
    assert doc["status"]["publication"] == "record_only"
    bad = copy.deepcopy(doc)
    bad["status"].update(publication="publishable", finding="effect_described",
                         finding_label=v2.FINDING_LABEL["effect_described"])
    bad["executive_summary"] = v2._executive_summary(bad)
    refused(lambda: v2.render_markdown_v2(bad, policy=env[1]), "differs from its reconstruction")
    bad2 = copy.deepcopy(doc)
    bad2["canonical"].update(finished=True, stopping_reason="finished")
    refused(lambda: v2.render_markdown_v2(bad2, policy=env[1]), "only an unfinished run")


@pytest.mark.parametrize("change, text", [
    (lambda d: d["canonical"].update(evidence=[]) or d.update(evidence_appendix=[]), "claims:"),
    (lambda d: d["canonical"]["evidence"].pop(0), "claims:"),
    (lambda d: d.pop("canonical"), "missing ['canonical']"),
    (lambda d: d.update(canonical=None), "canonical payload is missing"),
])
def test_codex_4_missing_evidence_is_refused(effect_doc, change, text):
    *_, policy, _, doc, _ = effect_doc
    bad = copy.deepcopy(doc)
    change(bad)
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), text)


def test_codex_5_unknown_registry_is_refused(effect_doc):
    *_, policy, _, doc, _ = effect_doc
    bad = copy.deepcopy(doc)
    bad["format"]["registry_version"] = "claims/999"
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), "differs from its reconstruction")
    bad["canonical"]["registry_version"] = "claims/999"
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), "unknown registry_version")
    for key, value in (("findings_format", "findings/3"), ("grid_version", "grid/9")):
        worse = copy.deepcopy(doc)
        worse["canonical"][key] = value
        refused(lambda: v2.render_markdown_v2(worse, policy=policy), f"unknown {key}")
    mixed = copy.deepcopy(doc)
    mixed["canonical"]["interpretations"] = [{"statement": "approach causes attack"}]
    refused(lambda: v2.render_markdown_v2(mixed, policy=policy), "mixed payload")


def test_codex_6_fingerprint_status_and_verification_text_are_refused(effect_doc):
    *_, policy, _, doc, _ = effect_doc
    bad = copy.deepcopy(doc)
    bad["verification"]["fingerprint"]["match"] = False
    bad["verification"]["claim_grounding"] = "re-validated: everything is fine"
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), "verification")
    fake = "f" * 64
    bad2 = copy.deepcopy(doc)
    bad2["verification"]["fingerprint"].update(recorded=fake, recomputed=fake)
    bad2["canonical"]["fingerprint"] = fake
    refused(lambda: v2.render_markdown_v2(bad2, policy=policy), "fingerprint mismatch")


@pytest.mark.parametrize("change", [
    lambda d: d["claims"][2].update(section="finding"),
    lambda d: d["claims"][1].update(limiting=True),
    lambda d: d["observations"][0].update(value=-1.0, value_rendered="-1"),
    lambda d: d["recorded"]["tracking"].clear() or d["recorded"]["tracking_sentences"].clear(),
    lambda d: d["data_provenance"].update(synthetic=False),
    lambda d: d.update(unit_of_analysis="Each window is an independent replicate."),
    lambda d: d["evidence_appendix"][0].update(status="ok-verified"),
    lambda d: d["trace"]["steps"][0].update(model_reasoning_unverified="tampered"),
    lambda d: d.update(extra="the effect is significant"),
])
def test_any_other_saved_edit_is_refused(effect_doc, change):
    *_, policy, _, doc, _ = effect_doc
    bad = copy.deepcopy(doc)
    change(bad)
    retemplate(bad)
    with pytest.raises(ReportIntegrityError):
        v2.render_markdown_v2(bad, policy=policy)


def test_appended_prose_in_a_sentence_or_summary_is_refused(effect_doc):
    *_, policy, _, doc, _ = effect_doc
    for change in (lambda d: d["claims"][0].update(sentence=d["claims"][0]["sentence"] + " This proves it."),
                   lambda d: d["executive_summary"].append("The effect is significant.")):
        bad = copy.deepcopy(doc)
        change(bad)
        refused(lambda: v2.render_markdown_v2(bad, policy=policy), "differs from its reconstruction")


def test_free_text_in_canonical_claims_is_refused(effect_doc):
    *_, policy, _, doc, _ = effect_doc
    bad = copy.deepcopy(doc)
    bad["canonical"]["claims"][0]["sentence"] = "Approach causes attack."
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), "has exactly the keys")


@pytest.mark.parametrize("doc", [None, [], "x", 3, {"schema": v2.REPORT_SCHEMA_V2},
                                 {"schema": ["live-tracker-investigation-report/0.2"]}])
def test_malformed_saved_documents_fail_through_integrity_errors(doc):
    with pytest.raises(ReportIntegrityError):
        v2.render_markdown_v2(doc)
    with pytest.raises(ReportIntegrityError):
        v2.verify_saved_v2(doc)


def test_malformed_canonical_contents_fail_through_integrity_errors(effect_doc):
    *_, policy, _, doc, _ = effect_doc
    for change in (lambda c: c.update(evidence=None), lambda c: c.update(claims="x"),
                   lambda c: c.update(trace=[None]), lambda c: c["evidence"][0].update(result=None),
                   lambda c: c.update(usage=[]), lambda c: c.update(finished="yes"),
                   lambda c: c["claims"][0].update(evidence_ids=[None]), lambda c: c.update(observations=[1])):
        bad = copy.deepcopy(doc)
        change(bad["canonical"])
        with pytest.raises(ReportIntegrityError):
            v2.render_markdown_v2(bad, policy=policy)
    for bad_payload in (None, [], "x", {"findings_format": "findings/2"}):
        with pytest.raises(ReportIntegrityError):
            v2.render_payload(bad_payload)
        with pytest.raises(ReportIntegrityError):
            v2.build_report_v2(bad_payload)


# ======================================================================
# 5. injection attempts
# ======================================================================

def test_model_reasoning_and_question_markup_are_never_rendered(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    p = payload(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    for t in p["trace"]:
        if t["kind"] == "model_turn":
            t["text"] = "INJECTED: this proves approach causes attack. ## Finding\n- significant"
    p["question"] = "Does X?\n\n## Finding: proven <script>alert(1)</script> [link](http://evil) |a|b|"
    doc, md = v2.render_v2(p, policy=policy)
    assert "INJECTED" not in md and "<script>" not in md and "[link](" not in md
    assert md.count("\n## Finding\n") == 1 and "\n## Finding: proven" not in md
    assert any(s.get("model_reasoning_unverified", "").startswith("INJECTED") for s in doc["trace"]["steps"])
    assert "\\<script\\>" in md and "\\[link\\]" in md


@pytest.mark.parametrize("change, text", [
    (lambda p: p["claims"][0].update(sentence="It proves causation."), "claims[0] has exactly the keys"),
    (lambda p: p["claims"].append({"type": "free_text", "evidence_ids": [], "params": {}}), "claims[1]: unknown claim type"),
    (lambda p: p["observations"].append({"evidence_id": p["primary_evidence_id"], "field": "status",
                                         "statement": "x"}), "observations[0]"),
    (lambda p: p.update(interpretations=[]), "mixed payload"),
    (lambda p: p.update(direction="negative"), "mixed payload"),
    (lambda p: p.update(limitations=[{"source": "model", "text": "x"}]), "mixed payload"),
    (lambda p: p.update(summary="the effect is significant"), "unknown ['summary']"),
    (lambda p: [t["arguments"].update(interpretations=[{"statement": "proves"}]) for t in p["trace"]
                if t["kind"] == "finish"], "legacy model-authored prose"),
])
def test_injected_prose_is_refused(effect_run, change, text):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    p = payload(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    change(p)
    refused(lambda: v2.render_v2(p, policy=policy), text)


def test_forged_unit_text_is_refused_at_render(effect_run):
    report, _ = effect_run

    def unit(req, res):
        for m in res["methodology"]:
            if m["name"] == "units":
                m["value"] = "px/s. This proves causation"
    forged = forge(report, "compare_windows", unit, check=False)
    pid = ev(forged, "compare_windows")
    r = cl.validate_findings({"finding": "effect_described", "primary_evidence_id": pid, "observations": [],
                              "claims": [C("descriptive_effect", pid)], "next_investigations": []},
                             evidence=forged.evidence, trace=forged.trace, policy=None, mode="complete",
                             stopping_reason=forged.stopping_reason, finished=True)
    if not r.accepted:
        pytest.skip(f"refused before rendering: {r.problems[:1]}")
    refused(lambda: v2.render_v2(payload(forged, None, "effect_described", pid, claims=[C("descriptive_effect", pid)])),
            "is not a known unit")


def test_observation_text_values_are_not_interpolated():
    assert v2._observation_value("ok") == "ok"
    assert v2._observation_value("<b>proves</b> [x](y)") == "(text value; see the evidence snapshot)"
    assert v2._observation_value([1, "a b c", None]) == "[1, (text value; see the evidence snapshot), null]"


# ======================================================================
# 6. revalidate, then fingerprint; tampering
# ======================================================================

def test_fingerprint_mismatch_is_refused_after_revalidation(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    p = payload(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    refused(lambda: v2.render_v2({**p, "fingerprint": "0" * 64}, policy=policy), "fingerprint mismatch")
    # content that fails re-validation is refused for that reason; its fingerprint is never compared
    bad = {**p, "fingerprint": "0" * 64, "claims": [C("descriptive_effect", pid), C("no_comparison")]}
    e = refused(lambda: v2.render_v2(bad, policy=policy), "claims[1] (no_comparison)")
    assert not any("fingerprint" in x for x in e.problems)
    # the fingerprint binds the claims: same evidence, other valid claims, old fingerprint
    other = copy.deepcopy({**p, "claims": [C("descriptive_effect", pid), C("coverage", aspect="primary_comparison")]})
    next(t for t in other["trace"] if t["kind"] == "finish")["arguments"]["claims"] = copy.deepcopy(other["claims"])
    refused(lambda: v2.render_v2(other, policy=policy), "fingerprint mismatch")      # trace and claims agree; F not
    unbound = {**p, "claims": other["claims"]}                                       # M3: the finish binds the claims
    refused(lambda: v2.render_v2(unbound, policy=policy), "execution: the accepted finish's arguments")


def test_tampered_evidence_is_refused(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    p = payload(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    p["evidence"][-1]["result"]["measurements"]["effect"]["value"] = 1.0       # not re-hashed
    refused(lambda: v2.render_v2(p, policy=policy), "claims:")


# ======================================================================
# 7. publishable / diagnostic / incomplete
# ======================================================================

def test_without_a_policy_the_report_is_diagnostic(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    doc, md = v2.render_v2(payload(report, None, "effect_described", pid, claims=[C("descriptive_effect", pid)]))
    assert doc["status"]["publication"] == "diagnostic" and doc["verification"]["authorization"] == "not_rechecked"
    assert doc["executive_summary"][0] == v2.DIAGNOSTIC_BANNER and f"> **{v2.DIAGNOSTIC_BANNER}**" in md
    assert "L2 authorization | not_rechecked" in md and "Publishable" not in md
    pub, md2 = v2.render_v2(payload(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)]),
                            policy=policy)
    assert pub["status"]["publication"] == "publishable" and v2.DIAGNOSTIC_BANNER not in md2
    r = v2.render_payload(payload(report, None, "effect_described", pid, claims=[C("descriptive_effect", pid)]))
    assert (r.format, r.verification, r.authorization, r.quarantine) == ("findings/2", "diagnostic", "not_rechecked",
                                                                         None)


def test_incomplete_run_renders_without_any_finding(env):
    report = scripted_compare(env)
    assert report.stopping_reason == "max_model_turns" and not report.finished
    p = incomplete_payload(report, env[1])
    doc, md = v2.render_v2(p, policy=env[1])
    assert doc["status"]["finding"] == "incomplete" and doc["claims"] == [] and doc["observations"] == []
    assert doc["status"]["accepted_direction"] is None and doc["status"]["primary_evidence_id"] is None
    assert doc["executive_summary"][1] == (
        f"Incomplete: the investigation stopped before an accepted finish (max_model_turns). No finding was made; "
        f"{len(report.evidence)} validated results were collected and are listed in the evidence appendix, none "
        "presented as a finding.")
    assert "- No finding: the run is incomplete." in md and "Accepted direction | none" in md
    assert doc["status"]["publication"] == "record_only" and doc["executive_summary"][0].startswith("Record only:")
    assert "Publishable" not in md and v2.DIAGNOSTIC_BANNER not in md
    assert v2.build_report_v2(incomplete_payload(report, None))["status"]["publication"] == "diagnostic"
    pid = ev(report, "compare_windows")
    for change, text in [
        (lambda q: q.update(claims=[C("descriptive_effect", pid)]), "never acquires"),
        (lambda q: q.update(primary_evidence_id=pid), "never acquires"),
        (lambda q: q.update(finding="effect_described"), "only an unfinished run"),
        (lambda q: q.update(finished=True), "only an unfinished run"),
    ]:
        bad = copy.deepcopy(p)
        change(bad)
        refused(lambda: v2.render_v2(bad, policy=env[1]), text)


# ======================================================================
# 8. legacy quarantine and format isolation
# ======================================================================

def test_legacy_reports_render_byte_identically_and_are_quarantined(effect_run):
    report, policy = effect_run
    for p in (policy, None):
        doc, md = ir.render(report, policy=p)
        r = v2.render_payload(report, policy=p)
        assert (r.format, r.verification) == ("legacy/1", "legacy_verified")
        assert ir.to_json_text(r.doc) == ir.to_json_text(doc) and r.markdown == md          # legacy bytes
        r2 = v2.render_payload(json.loads(json.dumps(report.to_json())), policy=p)
        assert ir.to_json_text(r2.doc) == ir.to_json_text(doc) and r2.markdown == md
        assert r.authorization == ("rechecked" if p else "not_rechecked")
    q = r.quarantine
    assert q["label"] == "legacy model-authored prose (unverified)" and q["upgraded_to_findings_v2"] is False
    assert q["fields"]["interpretations"] == len(report.interpretations) >= 1
    assert r.notice not in r.markdown and "legacy model-authored prose (unverified)" in r.notice
    summary = " ".join(r.doc["executive_summary"])
    assert not any(i["statement"] in summary for i in report.interpretations)
    assert not any(s in summary for s in report.next_steps)


def test_legacy_and_v2_never_cross(effect_run, effect_doc):
    report, policy = effect_run
    *_, v2doc, _ = effect_doc
    legacy_doc = ir.build_report(report, policy=policy)
    refused(lambda: v2.build_report_v2(report, policy=policy), "never rendered as findings/2")
    refused(lambda: v2.build_report_v2(report.to_json(), policy=policy), "never rendered as findings/2")
    refused(lambda: v2.render_markdown_v2(legacy_doc), "not a findings/2 report document")
    refused(lambda: ir.render_markdown(v2doc), "unknown report schema")
    with pytest.raises(ReportIntegrityError, match="not rendered by the legacy renderer"):
        ir.load_report({**effect_doc[2]})


@pytest.mark.parametrize("change, text", [
    (lambda p: p.update(findings_format="findings/3"), "unknown findings_format"),
    (lambda p: p.update(registry_version="claims/9.9"), "unknown registry_version"),
    (lambda p: p.update(grid_version="grid/1"), "unknown grid_version"),
    (lambda p: p.update(findings_format=["findings/2"]), "unknown findings_format"),
    (lambda p: p.pop("findings_format"), "mixed payload"),
    (lambda p: p.pop("usage"), "missing ['usage']"),
    (lambda p: p.update(next_steps=["do more"]), "mixed payload"),
    (lambda p: p.update(primary_evidence={"evidence_id": "x"}), "mixed payload"),
])
def test_mixed_and_unknown_payloads_are_refused(effect_doc, change, text):
    *_, p, _, _ = effect_doc
    bad = copy.deepcopy(p)
    change(bad)
    refused(lambda: v2.render_payload(bad, policy=effect_doc[1]), text)


def test_legacy_payload_with_v2_fields_is_mixed(effect_run):
    report, policy = effect_run
    data = report.to_json()
    for extra in ({"claims": []}, {"fingerprint": "x"}, {"registry_version": "claims/2.0"}):
        refused(lambda: v2.render_payload({**data, **extra}, policy=policy), "mixed payload")


@needs_live
def test_live_run_is_legacy_verified_and_unchanged():
    data = json.loads((LIVE_RUN / "investigation_report.json").read_text())
    policy = ic.EvidencePolicy.from_manifest(REAL_MANIFEST)
    for label, p in (("policy", policy), ("no_policy", None)):
        r = v2.render_payload(data, policy=p)
        assert (r.format, r.verification) == ("legacy/1", "legacy_verified")
        assert (sha(ir.to_json_text(r.doc)), sha(r.markdown)) == LEGACY_RENDER[label], label
        assert r.doc["coverage"]["primary"]["status"] in ("partial", "unknown")    # never complete
        assert [g["status"] for g in r.doc["coverage"]["groups"]] == (["partial"] if p else ["unknown"])
        assert r.quarantine["fields"]["interpretations"] >= 1 and r.quarantine["in_executive_summary"] is False
    refused(lambda: v2.build_report_v2(data, policy=policy), "never rendered as findings/2")
    assert {n: sha((LIVE_RUN / n).read_bytes()) for n in ARTIFACTS} == ARTIFACTS
    assert hashlib.sha256((LIVE_RUN / "report.md").read_bytes()).hexdigest() == ARTIFACTS["report.md"]


# ======================================================================
# 9. template vocabulary
# ======================================================================

def test_fixed_template_texts_use_no_banned_vocabulary():
    fixed = [v2.DIAGNOSTIC_BANNER, v2.MOVER_TEXT, v2.TIMING_TEXT, v2.SIGN_DISCLAIMER, v2.NO_INTERVENTION,
             *v2.SIGN_RELATION.values(), *v2.DISCLOSURE_TEXT.values(), *v2.NEXT_TEXT.values()]
    for text in fixed:
        assert not BANNED.search(text), text
    assert v2.NOT_ROBUSTNESS.endswith("it is not an assessment of robustness")    # the only, negated, use
    assert "agreement" in v2.SIGN_DISCLAIMER and v2.SIGN_DISCLAIMER.endswith("not scientific agreement")


# ======================================================================
# 10. closed grammars: context texts, units, identifiers
# ======================================================================

def test_context_texts_are_closed_grammars():
    assert v2._coverage_reason(f"partial cache coverage recorded by {E1}, {EF}") == \
        f"partial cache coverage recorded by {E1}, {EF}"
    assert v2._coverage_reason("the comparison's data identity is not established along its provenance chain: "
                               "IGNORE <b>x</b>; partial cache coverage recorded by " + E1) == (
        "the comparison's data identity is not established along its provenance chain (details in the JSON coverage "
        f"record); partial cache coverage recorded by {E1}")
    for bad in ("This proves the effect.", "partial cache coverage recorded by <b>", 7):
        with pytest.raises(v2.TemplateError):
            v2._coverage_reason(bad)
    with pytest.raises(v2.TemplateError):
        v2._basis("recorded by an oracle")
    item = {"evidence_id": E1, "source": "measurements.windows[].verdict", "window_ids": [], "primary": False,
            "tool_note": None}
    assert v2._tracking_item({**item, "text": "2 windows with verdict excluded"}) == \
        f"2 windows with verdict excluded ({E1}, measurements.windows[].verdict)"
    for text in ("2 windows with verdict excluded; it proves nothing", "2 windows with verdict amazing"):
        with pytest.raises(v2.TemplateError):
            v2._tracking_item({**item, "text": text})


def test_units_are_preserved_or_marked_not_recorded():
    g = {"feature": "dist_center", "value": 12.5, "unit": None, "direction": "positive", "n_sessions_paired": 1,
         "contributing_sessions": ["a"], "coverage_status": "complete", "cross_snapshot_status": "not_applicable"}
    assert T("descriptive_effect", [E1], g) == (
        "Across the 1 contributing paired sessions (a), the median of per-session differences (pre-event minus "
        "baseline) in dist_center was 12.5 (unit not recorded) (positive). Descriptive only: no interval or test; not "
        "a causal reading; it describes these sessions, not the whole authorized split. Cross-snapshot agreement: "
        "not_applicable.")
    assert " 12.5 px (positive)" in T("descriptive_effect", [E1], {**g, "unit": "px"})
    assert "Cross-snapshot" not in T("descriptive_effect", [E1], {**g, "cross_snapshot_status": "verified"})
    for bad in ({**g, "unit": "px (proves it)"}, {**g, "n_sessions_paired": 2}, {**g, "coverage_status": "unknown"},
                {**g, "contributing_sessions": ["a [x](y)"]}, {**g, "value": float("nan")}, {**g, "value": "12"}):
        with pytest.raises(v2.TemplateError):
            T("descriptive_effect", [E1], bad)


# ======================================================================
# Contract v1 shared surface item 4: association only
# ======================================================================

def test_contract_v1_association_only_is_always_true_and_cannot_be_changed(effect_doc):
    *_, policy, _, doc, md = effect_doc
    assert doc["association_only"] is True
    assert "Association only: the comparisons describe what precedes or is associated with" in md
    bad = copy.deepcopy(doc)
    bad["association_only"] = False
    refused(lambda: v2.render_markdown_v2(bad, policy=policy), "association_only")
    missing = copy.deepcopy(doc)
    del missing["association_only"]
    refused(lambda: v2.render_markdown_v2(missing, policy=policy), "association_only")
