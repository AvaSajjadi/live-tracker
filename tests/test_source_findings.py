"""Tests for Stage 9 S9-5: source-aware Findings v2 (registry claims/2.1) for
classifier_prediction and monitor_event evidence. New file.

Runs go through the real Stage 6 loop (a scripted model chooses the tools)
over the S9-4 synthetic world: pose cache with annotated attacks, and a
classifier that predicts attacks elsewhere. No model, network or paid call.
"""

import copy
import json

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_claims as cl
import live_tracker.investigation_report as ir
import live_tracker.investigation_report_v2 as v2
from live_tracker.investigation_agent import Budget
from live_tracker.investigation_contracts import PROVISIONAL_SUFFICIENCY
from test_investigation_agent import LOOSE, QUESTION, Scripted, call, last_result, last_result_of, obs, windows_of  # noqa: F401
from test_investigation_findings_v2 import C, finish
from test_prediction_windows import VAL_A, world  # noqa: F401


def query(source):
    return [call("query_events", source=source, behavior="attack")]


def features_turn(length=20):
    return lambda m: [call("get_window_features", episodes_from={"evidence_id": last_result(m)["evidence_id"]},
                           pre_event={"length_frames": length, "gap_frames": 0},
                           baseline={"length_frames": length, "clearance_frames": 10, "max_windows_per_session": 500},
                           features=["dist_rate"])]


def compare_turn(m):
    return [call("compare_windows", feature="dist_rate", windows_from={"evidence_id": last_result(m)["evidence_id"]})]


def effect_finish(m):
    c = last_result_of(m, "compare_windows")
    return [finish(finding="effect_described", primary_evidence_id=c["evidence_id"],
                   observations=[obs(c, "measurements.effect.value")],
                   claims=[C("descriptive_effect", c["evidence_id"]), C("coverage", aspect="primary_comparison")])]


def insufficient_finish(m):
    c = last_result_of(m, "compare_windows")
    return [finish(finding="insufficient_evidence", primary_evidence_id=c["evidence_id"],
                   observations=[obs(c, "status")], claims=[C("sufficiency_not_met", c["evidence_id"])])]


def run(world, key, source, *turns, sufficiency=LOOSE, turns_budget=6):
    return ia.investigate(QUESTION, world[key], world["policy"], Scripted(query(source), *turns),
                          sufficiency=sufficiency, budget=Budget(max_model_turns=turns_budget))


def sentence(doc, claim_type):
    return next(c["sentence"] for c in doc["claims"] if c["type"] == claim_type)


# ======================================================================
# 1. exact source-aware sentences and disclosures
# ======================================================================

def test_classifier_effect_is_reported_as_predicted_episodes(world):
    report = run(world, "clf", "classifier_prediction", features_turn(), compare_turn, effect_finish)
    assert report.finished and report.registry_version == "claims/2.1", report.trace[-3:]
    doc, md = v2.render_v2(report, policy=world["policy"])
    g = next(c for c in doc["claims"] if c["type"] == "descriptive_effect")["grounding"]
    src = g["source"]
    assert src == {"source": "classifier_prediction", "behavior": "attack", "n_episodes": 4, "merged": False,
                   "model_label": "model_b", "model_identity": "unverified", "threshold": 0.5,
                   "min_duration_frames": 15, "score_file_sha256": world["drop"].provenance.member_sha256}
    phrase = "attack episodes predicted by the classifier (score >= 0.5, at least 15 frames; model label model_b, unverified)"
    s = sentence(doc, "descriptive_effect")
    assert s.startswith(f"Across the {g['n_sessions_paired']} contributing paired sessions (")
    assert f"for {phrase}, the median of per-session differences (pre-threshold-crossing minus baseline)" in s
    assert "The windows precede classifier threshold crossings, not necessarily the behavior." in s
    assert "not a causal reading; not a measure of model accuracy" in s
    assert doc["source"] == {"source": "classifier_prediction", "phrase": phrase, "disclosures": [
        v2.SOURCE_DISCLOSURES["prediction"].format(what="classifier predictions"),
        v2.SOURCE_DISCLOSURES["onset_classifier"], v2.SOURCE_DISCLOSURES["model"],
        v2.SOURCE_DISCLOSURES["thresholds"], v2.SOURCE_DISCLOSURES["tuning"], v2.SOURCE_DISCLOSURES["labels"],
        v2.SOURCE_DISCLOSURES["no_inference"]]}
    assert doc["executive_summary"][1] == f"Evidence source: {phrase}; not annotated behavior."
    assert s in doc["executive_summary"]
    assert "## Evidence source and its limitations" in md and v2.SOURCE_DISCLOSURES["labels"] in md
    for banned in ("significant", "causes", "proves", "accuracy of"):
        assert banned not in md.replace("no statistical significance", "").replace("not a measure of model accuracy",
                                                                                     "")


def test_monitor_effect_is_reported_as_detected_events(world):
    report = run(world, "mon", "monitor_event", features_turn(), compare_turn, effect_finish)
    assert report.finished and report.registry_version == "claims/2.1"
    doc, md = v2.render_v2(report, policy=world["policy"])
    src = doc["claims"][0]["grounding"]["source"]
    assert (src["source"], src["rule_id"], src["run_id"], src["journal_sha256"]) == (
        "monitor_event", "R1", "mon", world["snap"].sha256)
    s = sentence(doc, "descriptive_effect")
    assert "for attack events detected by monitor rule R1 (run mon), the median of per-session differences " \
           "(pre-detection minus baseline)" in s
    assert "The windows precede monitor detection onsets (which precede the alerts), not necessarily the behavior." in s
    assert v2.SOURCE_DISCLOSURES["onset_monitor"] in doc["source"]["disclosures"]
    assert "model_label" not in src


def test_insufficient_evidence_and_no_data_sentences(world):
    report = run(world, "clf", "classifier_prediction", features_turn(), compare_turn, insufficient_finish,
                 sufficiency=PROVISIONAL_SUFFICIENCY)
    assert report.finished, report.trace[-3:]
    doc, _ = v2.render_v2(report, policy=world["policy"])
    s = sentence(doc, "sufficiency_not_met")
    assert s.startswith("The dist_rate comparison (ev:compare_windows:") and \
        ") over attack episodes predicted by the classifier (score >= 0.5, at least 15 frames; model label " \
        "model_b, unverified) did not meet the provisional sufficiency criteria" in s
    assert s in doc["executive_summary"]
    t = v2.TEMPLATES["comparison_no_data"]({"evidence_ids": ["ev:compare_windows:" + "a" * 24], "params": {},
                                            "grounding": {"feature": "dist_rate", "n_windows_pre": 0,
                                                          "n_windows_baseline": 2,
                                                          "source": doc["claims"][0]["grounding"]["source"]}})
    assert t == ("The dist_rate comparison (ev:compare_windows:" + "a" * 24 + ") over attack episodes predicted by "
                 "the classifier (score >= 0.5, at least 15 frames; model label model_b, unverified) returned no "
                 "data: 0 pre-event and 2 baseline windows were usable. No aggregate effect is reported.")


# ======================================================================
# 2. publication authorization
# ======================================================================

def test_prediction_findings_are_never_publishable(world):
    report = run(world, "clf", "classifier_prediction", features_turn(), compare_turn, effect_finish)
    doc, md = v2.render_v2(report, policy=world["policy"])
    assert doc["status"]["publication"] == "not_authorized" and doc["verification"]["authorization"] == "rechecked"
    assert doc["executive_summary"][0] == v2.NOT_AUTHORIZED_BANNER and f"> **{v2.NOT_AUTHORIZED_BANNER}**" in md
    assert "Publishable" not in md
    no_policy, _ = v2.render_v2(report)
    assert no_policy["status"]["publication"] == "diagnostic"
    assert v2.render_payload(report, policy=world["policy"]).verification == "not_authorized"
    assert cl.PUBLICATION_AUTHORIZED == {"claims/2.0": True, "claims/2.1": False}


def test_incomplete_prediction_runs_are_record_only(world):
    report = run(world, "clf", "classifier_prediction", turns_budget=2)
    assert not report.finished and report.registry_version == "claims/2.1"
    doc, md = v2.render_v2(report, policy=world["policy"])
    assert doc["status"]["publication"] == "record_only" and doc["claims"] == []
    assert doc["source"]["source"] == "classifier_prediction"


@pytest.mark.parametrize("edit", [
    lambda d: d["status"].update(publication="publishable"),
    lambda d: d["status"].update(publication="publishable") or d.update(
        executive_summary=["Publishable: ..."] + d["executive_summary"][1:]),
    lambda d: d.pop("source"),
    lambda d: d["source"]["disclosures"].pop(),
    lambda d: d["source"].update(phrase="attack episodes (annotated)"),
    lambda d: d["claims"][0].update(sentence=d["claims"][0]["sentence"].replace("predicted by the classifier",
                                                                                 "annotated")),
    lambda d: d["canonical"].update(registry_version="claims/2.0"),
    lambda d: d["format"].update(registry_version="claims/2.0"),
])
def test_saved_prediction_documents_cannot_be_promoted_or_relabelled(world, edit):
    report = run(world, "clf", "classifier_prediction", features_turn(), compare_turn, effect_finish)
    doc, _ = v2.render_v2(report, policy=world["policy"])
    saved = json.loads(ir.to_json_text(doc))
    assert v2.render_markdown_v2(copy.deepcopy(saved), policy=world["policy"])           # untouched: renders
    edit(saved)
    with pytest.raises(ir.ReportIntegrityError):
        v2.render_markdown_v2(saved, policy=world["policy"])


# ======================================================================
# 3. unsupported claims, invented quantities, cross-source and registry mismatch
# ======================================================================

def test_unsupported_claims_and_invented_quantities_are_refused(world):
    def confounder(m):
        c = last_result_of(m, "compare_windows")
        return [finish(finding="inconclusive", primary_evidence_id=c["evidence_id"], observations=[],
                       claims=[C("potential_confounder", c["evidence_id"], confounder="investigation_label")])]

    def invented(m):
        c = last_result_of(m, "compare_windows")
        return [call(ia.FINISH, finding="effect_described", primary_evidence_id=c["evidence_id"], observations=[],
                     claims=[{"type": "descriptive_effect", "evidence_ids": [c["evidence_id"]],
                              "params": {"value": -3.2, "threshold": 0.9}}], next_investigations=[])]
    for bad, text in ((confounder, "unknown claim type (claims/2.1 has"), (invented, "unknown fields")):
        report = run(world, "clf", "classifier_prediction", features_turn(), compare_turn, bad, turns_budget=4)
        refused = [t for t in report.trace if t["kind"] == "finish"]
        assert not report.finished and refused and text in refused[0]["message"], refused[0]["message"]


def test_mixed_source_runs_are_refused(world):
    def both(m):
        return [call("query_events", source="validation_annotation", behavior="attack")]
    report = ia.investigate(QUESTION, world["clf"], world["policy"],
                            Scripted(query("classifier_prediction"), both, features_turn(), compare_turn, effect_finish),
                            sufficiency=LOOSE, budget=Budget(max_model_turns=6))
    refused = [t for t in report.trace if t["kind"] == "finish"]
    assert not report.finished and "mixed-source evidence" in refused[0]["message"]
    doc, _ = v2.render_v2(report, policy=world["policy"])     # still recorded: no claim, no finding, no source label
    assert doc["status"]["publication"] == "record_only" and doc["claims"] == [] and "source" not in doc


def test_registry_must_match_the_evidence(world):
    report = run(world, "clf", "classifier_prediction", features_turn(), compare_turn, effect_finish)
    proposal = {k: getattr(report, k) for k in cl.PROPOSAL_KEYS}
    as_20 = cl.validate_findings(proposal, evidence=report.evidence, trace=report.trace, policy=world["policy"],
                                 mode="complete", stopping_reason="finished", finished=True)
    assert not as_20.accepted and "is reported by claims/2.1, not claims/2.0" in as_20.problems[0]
    data = report.to_json()
    with pytest.raises(ir.ReportIntegrityError, match="claims/2.1, not claims/2.0"):
        v2.render_v2({**data, "registry_version": "claims/2.0"}, policy=world["policy"])
    ann = ia.investigate(QUESTION, world["ann_only"], world["policy"],
                         Scripted(query("validation_annotation")), budget=Budget(max_model_turns=1))
    as_21 = cl.incomplete_findings(evidence=ann.evidence, trace=ann.trace, policy=world["policy"],
                                   stopping_reason=ann.stopping_reason, registry_version="claims/2.1")
    assert not as_21.accepted and "is reported by claims/2.0, not claims/2.1" in as_21.problems[0]
    assert cl.validate_findings(proposal, evidence=report.evidence, trace=report.trace, policy=world["policy"],
                                registry_version="claims/3.0").problems == ("unknown registry 'claims/3.0'",)


def test_saved_round_trip_and_fingerprint(world):
    report = run(world, "mon", "monitor_event", features_turn(), compare_turn, effect_finish)
    doc, md = v2.render_v2(json.loads(json.dumps(report.to_json())), policy=world["policy"])
    assert doc["verification"]["fingerprint"]["recomputed"] == report.fingerprint
    saved = json.loads(ir.to_json_text(doc))
    assert v2.render_markdown_v2(saved, policy=world["policy"]) == md


# ======================================================================
# 4. the annotation workflow is unchanged
# ======================================================================

def test_annotation_runs_stay_on_claims_2_0(world):
    from test_investigation_findings_v2 import InvestigatorV2
    report = ia.investigate(QUESTION, world["ann_only"], world["policy"], InvestigatorV2(), sufficiency=LOOSE)
    assert report.finished and report.registry_version == "claims/2.0"
    doc, md = v2.render_v2(report, policy=world["policy"])
    assert doc["status"]["publication"] == "publishable" and "source" not in doc
    assert all("source" not in c["grounding"] for c in doc["claims"])
    assert "## Evidence source" not in md and v2.NOT_AUTHORIZED_BANNER not in md
    assert sentence(doc, "descriptive_effect").startswith("Across the ") and \
        "median of per-session differences (pre-event minus baseline)" in sentence(doc, "descriptive_effect")


# ======================================================================
# Codex S9-5 follow-up: one source identity per chain; empty prediction queries
# ======================================================================

import live_tracker.event_log as el  # noqa: E402
import live_tracker.investigation_context as ctx  # noqa: E402
import live_tracker.investigation_contracts as ic  # noqa: E402
from live_tracker.investigation_events import CalMS21InvestigationTools  # noqa: E402
from live_tracker.investigation_sources import ClassifierEventSource, MonitorEventSource  # noqa: E402
from live_tracker.monitor import Monitor  # noqa: E402
from live_tracker.rules import RuleSet  # noqa: E402


def monitor_tools(world, run_id, ask=0.5, alert=0.8):
    rules = RuleSet()
    rules.add({"rule_id": "R1", "behavior": "attack", "min_duration_s": 0.5, "alert_threshold": alert,
               "ask_threshold": ask})
    rules.approve("R1")
    m = Monitor(rules.active(), run_id=run_id, provenance=world["drop"])
    for row in world["drop"].rows():
        m.step(row)
    m.finish()
    src = MonitorEventSource(el.freeze_run(m.journal, run_id), world["drop"], world["policy"], rule_id="R1")
    return CalMS21InvestigationTools(world["policy"], [world["ann_only"].sources["validation_annotation"], src])


def tool(tools, name, request, policy):
    return ic.call_tool(tools, name, {"contract_version": ic.CONTRACT_VERSION, **request}, policy)


def qreq(source):
    return {"source": source, "behavior": "attack", "sessions": None, "max_episodes": 500, "merge_gap_frames": None,
            "min_duration_frames": None}


def freq(episode_ids):
    return {"episode_ids": episode_ids, "pre_event": {"length_frames": 20, "gap_frames": 0},
            "baseline": {"strategy": "same_session_tiles", "length_frames": 20, "clearance_frames": 10,
                         "max_windows_per_session": 500}, "features": ["dist_rate"]}


def snapshot(result, request, name):
    return {"evidence_id": result["evidence_id"], "tool": name, "status": result["status"],
            "request": {"contract_version": ic.CONTRACT_VERSION, **request}, "result": result}


def trace_of(evidence):
    return [{"step": i, "kind": "tool_call", "tool": e["tool"], "request": e["request"],
             "evidence_id": e["evidence_id"], "outcome": e["status"]} for i, e in enumerate(evidence)]


def chain_refusal(evidence, policy):
    proposal = {"finding": "insufficient_evidence", "primary_evidence_id": None, "claims": [C("no_comparison")],
                "observations": [{"evidence_id": evidence[0]["evidence_id"], "field": "status"}],
                "next_investigations": []}
    return cl.validate_findings(proposal, evidence=evidence, trace=trace_of(evidence), policy=policy,
                                mode="prefix", registry_version="claims/2.1")


def test_codex_cross_journal_chain_is_refused(world):
    p = world["policy"]
    run_a, run_b = monitor_tools(world, "run_a"), monitor_tools(world, "run_b")      # same drop, same rule
    qa = tool(run_a, "query_events", qreq("monitor_event"), p)
    ids = [e["episode_id"] for e in qa["measurements"]["episodes"]]
    fb = tool(run_b, "get_window_features", freq(ids), p)                          # run B accepts run A's ids
    assert fb["status"] == "ok"
    sq, sf = snapshot(qa, qreq("monitor_event"), "query_events"), snapshot(fb, freq(ids), "get_window_features")
    assert ctx._data_identity_problem(sq, sf) == ("prediction source identity differs (source, configuration, "
                                                  "journal, run, rule or score member)")
    refused = chain_refusal([sq, sf], p)
    assert not refused.accepted and "different monitor_event identities" in refused.problems[0]
    fa = tool(run_a, "get_window_features", freq(ids), p)                          # the same run: one identity
    assert ctx._data_identity_problem(sq, snapshot(fa, freq(ids), "get_window_features")) is None
    assert chain_refusal([sq, snapshot(fa, freq(ids), "get_window_features")], p).accepted


def test_classifier_configuration_conflicts_are_refused_despite_one_score_file(world):
    p = world["policy"]
    ann = world["ann_only"].sources["validation_annotation"]
    a = CalMS21InvestigationTools(p, [ann, ClassifierEventSource(world["drop"], p, threshold=0.5,
                                                                 min_duration_frames=15)])
    b = CalMS21InvestigationTools(p, [ann, ClassifierEventSource(world["drop"], p, threshold=0.6,
                                                                 min_duration_frames=15)])
    qa = tool(a, "query_events", qreq("classifier_prediction"), p)
    ids = [e["episode_id"] for e in qa["measurements"]["episodes"]]
    fb = tool(b, "get_window_features", freq(ids), p)
    sq, sf = snapshot(qa, qreq("classifier_prediction"), "query_events"), snapshot(fb, freq(ids), "get_window_features")
    assert "prediction source identity differs" in ctx._data_identity_problem(sq, sf)
    assert "different classifier_prediction identities" in chain_refusal([sq, sf], p).problems[0]


def empty_run(world, source):
    p = world["policy"]
    ann = world["ann_only"].sources["validation_annotation"]
    tools = (CalMS21InvestigationTools(p, [ann, ClassifierEventSource(world["drop"], p, threshold=0.99,
                                                                      min_duration_frames=15)])
             if source == "classifier_prediction" else monitor_tools(world, "empty_run", ask=0.95, alert=0.99))
    return tools


@pytest.mark.parametrize("source", ["classifier_prediction", "monitor_event"])
def test_empty_prediction_query_renders_as_an_incomplete_record(world, source):
    tools = empty_run(world, source)
    report = ia.investigate(QUESTION, tools, world["policy"], Scripted(query(source)), budget=Budget(max_model_turns=1))
    assert not report.finished and report.evidence[0]["status"] == "no_data"
    assert report.registry_version == "claims/2.1"
    doc, md = v2.render_v2(json.loads(json.dumps(report.to_json())), policy=world["policy"])
    assert doc["status"]["publication"] == "record_only" and doc["claims"] == []
    src = ctx.thaw(doc["source"])
    assert src["source"] == source and src["disclosures"][0].startswith("The episodes are")
    phrase = ("attack episodes predicted by the classifier (score >= 0.99, at least 15 frames; model label model_b, "
              "unverified)" if source == "classifier_prediction" else
              "attack events detected by monitor rule R1 (run empty_run)")
    assert src["phrase"] == phrase                                          # from the tool's own metadata
    saved = json.loads(ir.to_json_text(doc))
    assert v2.render_markdown_v2(saved, policy=world["policy"]) == md        # saved round trip
    again, md2 = v2.render_v2(report, policy=world["policy"])
    assert ir.to_json_text(again) == ir.to_json_text(doc) and md2 == md     # deterministic
    promoted = copy.deepcopy(saved)
    promoted["status"]["publication"] = "publishable"
    with pytest.raises(ir.ReportIntegrityError):
        v2.render_markdown_v2(promoted, policy=world["policy"])


@pytest.mark.parametrize("source", ["classifier_prediction", "monitor_event"])
def test_empty_prediction_query_can_finish_insufficient(world, source):
    tools = empty_run(world, source)
    stop = lambda m: [finish(finding="insufficient_evidence", observations=[obs(last_result(m), "status")],  # noqa: E731
                             claims=[C("no_comparison")])]
    report = ia.investigate(QUESTION, tools, world["policy"], Scripted(query(source), stop),
                            budget=Budget(max_model_turns=2))
    assert report.finished and report.registry_version == "claims/2.1", report.trace[-2:]
    doc, md = v2.render_v2(report, policy=world["policy"])
    assert doc["status"]["publication"] == "not_authorized" and doc["source"]["source"] == source
    assert doc["claims"][0]["grounding"]["source"]["n_episodes"] == 0
    assert v2.render_markdown_v2(json.loads(ir.to_json_text(doc)), policy=world["policy"]) == md


def test_unrecorded_source_fields_are_explicit_never_invented():
    src = {"source": "classifier_prediction", "behavior": "attack", "threshold": None, "min_duration_frames": None,
           "model_label": None}
    assert v2._source_phrase(src) == ("attack episodes predicted by the classifier (score threshold not recorded, "
                                      "minimum duration not recorded; model label not recorded, unverified)")
    assert v2._source_phrase({"source": "monitor_event", "behavior": "attack", "rule_id": None, "run_id": None}) == \
        "attack events detected by an unrecorded monitor rule (run not recorded)"


@pytest.mark.parametrize("tool_name", ["query_events", "get_window_features", "check_tracking_quality", "compare_windows"])
@pytest.mark.parametrize("mutation", ["both", "path", "hash", "missing", "duplicate", "malformed", "rule", "member"])
def test_monitor_internal_journal_provenance(world, tool_name, mutation):
    report = run(world, "mon", "monitor_event", features_turn(), compare_turn, effect_finish)
    snaps = {e["tool"]: e for e in report.evidence}
    feature = snaps["get_window_features"]
    wid = feature["result"]["measurements"]["windows"][0]["window_id"]
    req = {"window_ids": [wid], "history_frames": [1]}
    result = tool(world["mon"], "check_tracking_quality", req, world["policy"])
    snaps["check_tracking_quality"] = snapshot(result, req, "check_tracking_quality")
    original = snaps[tool_name]
    assert ctx._data_identity_problem(snaps["query_events"], original) is None
    assert not cl._source_binding({original["evidence_id"]: original}, cl.REGISTRY_V21)[0]
    assert report.finished
    altered = copy.deepcopy(original)
    data = altered["result"]["provenance"]["data"]
    journal = next(d for d in data if d["path"].startswith("journal-snapshot:"))
    if mutation in ("both", "path"):
        journal["path"] = "journal-snapshot:other-run"
    if mutation in ("both", "hash"):
        journal["sha256"] = "f" * 64
    if mutation == "missing":
        data.remove(journal)
    if mutation == "duplicate":
        data.append(copy.deepcopy(journal))
    if mutation == "malformed":
        journal["sha256"] = "not-a-hash"
    if mutation == "rule":
        next(m for m in altered["result"]["methodology"] if m["name"] == "monitor_rule_id")["value"] = ""
    if mutation == "member":
        member = next(d for d in data if d["path"].startswith(ctx.SCORE_MEMBER_PREFIX))
        data.append({**member, "path": member["path"] + ".other", "sha256": "e" * 64})
    # Rehash: semantic validation must reject, not merely stale evidence hashes.
    eid = ic.evidence_id(tool_name, altered["request"], altered["result"])
    altered["evidence_id"] = altered["result"]["evidence_id"] = eid
    assert ctx._data_identity_problem(snaps["query_events"], altered) is not None
    assert ctx._data_identity_problem(altered, snaps["query_events"]) is not None
    assert cl._source_binding({eid: altered}, cl.REGISTRY_V21)[0]
    assert not chain_refusal([altered], world["policy"]).accepted
    assert not chain_refusal([altered, original], world["policy"]).accepted
