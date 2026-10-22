"""Tests for the Findings v2 M1 claim registry `claims/2.0` (new file).

Evidence comes from real Stage 6 runs over the Stage 4 SYNTHETIC fixtures
(and re-hashed forgeries of them); no model or network is used.
"""

import copy
import dataclasses
import json

import pytest

import live_tracker.investigation_claims as cl
import live_tracker.investigation_context as ctx
import live_tracker.investigation_contracts as ic
from live_tracker.investigation_agent import Budget
from test_investigation_agent import investigate  # legacy/1, explicitly (M3)
from live_tracker.investigation_contracts import PROVISIONAL_SUFFICIENCY
from live_tracker.investigation_events import CalMS21InvestigationTools
from test_investigation_agent import (LOOSE, QUESTION, Investigator, Scripted, _until_compare, call,  # noqa: F401
                                      contaminated, no_network, query)
from test_investigation_report import forge
from test_investigation_windows import MANIFEST, VAL_A, VAL_B, VAL_C, build, ep


# ---- helpers ----

def ev(report, tool, k=0):
    return [e["evidence_id"] for e in report.evidence if e["tool"] == tool][k]


def C(claim_type, *ids, **params):
    return {"type": claim_type, "evidence_ids": list(ids), "params": params}


def check(report, policy, finding, primary=None, claims=(), observations=(), next_=(), mode="prefix"):
    proposal = {"finding": finding, "primary_evidence_id": primary, "observations": list(observations),
                "claims": list(claims), "next_investigations": list(next_)}
    trace = report.trace[:-1] if mode == "prefix" else report.trace       # the freeze-point prefix: no stop event
    return provider_schema_holds(proposal, cl.validate_findings(
        proposal, evidence=report.evidence, trace=trace, policy=policy, mode=mode,
        stopping_reason=report.stopping_reason, finished=report.finished))


def provider_schema_holds(proposal, result):
    """Every proposal M1 accepts also satisfies the provider-facing finish
    schema (Stage 6 FINISH_SPEC_V2), by the local validator and by jsonschema."""
    if result.accepted:
        import live_tracker.investigation_agent as ia
        spec = ia.FINISH_SPEC_V2["input_schema"]
        problems = []
        ia._schema_problems(proposal, spec, "arguments", problems)
        assert problems == [], problems
        try:                                           # optional reference check: never skips the M1 test
            import jsonschema
        except ImportError:
            return result
        jsonschema.validate(proposal, spec)
    return result


def refused(result, text):
    assert not result.accepted and any(text in p for p in result.problems), result.problems
    return result


def last_eid(m):
    return next(json.loads(b["content"])["evidence_id"] for x in reversed(m) if isinstance(x["content"], list)
                for b in reversed(x["content"]) if b.get("type") == "tool_result" and "evidence_id" in b["content"])


@pytest.fixture
def env(tmp_path):
    return build(tmp_path)


@pytest.fixture
def effect_run(env):                                          # ok comparison, 2 sessions, complete coverage
    return investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE), env[1]


@pytest.fixture
def insufficient_run(env):                                    # provisional criteria: insufficient comparison
    return investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=PROVISIONAL_SUFFICIENCY), env[1]


@pytest.fixture
def dirty_run(tmp_path):                                      # contamination -> original, variant and quality
    e = build(tmp_path, contaminated, name="dirty")
    return investigate(QUESTION, e[0], e[1], Investigator(), sufficiency=LOOSE), e[1]


def scripted_compare(env, sessions=None, sufficiency=LOOSE):
    turns = [[query(sessions=sessions)] if sessions else [query()]] + _until_compare(LOOSE)[1:]
    return investigate(QUESTION, env[0], env[1], Scripted(*turns), sufficiency=sufficiency,
                       budget=Budget(max_model_turns=3))


# ======================================================================
# accepted: descriptive effects where the design permits them
# ======================================================================

def test_accepted_descriptive_effect_with_complete_coverage(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    r = check(report, policy, "effect_described", pid,
              claims=[C("descriptive_effect", pid), C("coverage", aspect="primary_comparison")],
              observations=[{"evidence_id": pid, "field": "measurements.effect.value"}])
    assert r.accepted and r.publishable and r.authorization == "rechecked", r.problems
    snap = next(e for e in report.evidence if e["evidence_id"] == pid)
    g = r.claims[0]["grounding"]
    assert g["value"] == snap["result"]["measurements"]["effect"]["value"]               # read, not supplied
    assert g["direction"] == r.accepted_direction == ctx._sign(g["value"]) and g["coverage_status"] == "complete"
    assert g["unit"] == "px/s" and len(g["contributing_sessions"]) == 2 and "not the whole" in g["scope"]
    assert r.observations[0]["value"] == g["value"]
    payload = ctx.fingerprint_payload(evidence=report.evidence, finding="effect_described", primary_evidence_id=pid,
                                      observations=[{"evidence_id": pid, "field": "measurements.effect.value"}],
                                      claims=[C("descriptive_effect", pid), C("coverage", aspect="primary_comparison")])
    assert r.fingerprint == ctx.content_fingerprint(payload)


def test_partial_coverage_effect_is_accepted_and_scoped(env):
    report = scripted_compare(env, sessions=[VAL_A, VAL_B])
    pid = ev(report, "compare_windows")
    r = check(report, env[1], "effect_described", pid, claims=[C("descriptive_effect", pid)])
    assert r.accepted, r.problems
    g = r.claims[0]["grounding"]
    assert g["coverage_status"] == "partial" and list(g["coverage"]["not_searched"]) == [VAL_C]
    assert g["coverage"]["searched"] == 2 and g["coverage"]["eligible"] == 3


def test_unknown_coverage_refuses_a_descriptive_effect(env):
    feats = call("get_window_features", episode_ids=[ep(VAL_A, 100), ep(VAL_B, 200)],
                 pre_event={"length_frames": 20, "gap_frames": 0},
                 baseline={"length_frames": 20, "clearance_frames": 10, "max_windows_per_session": 500},
                 features=["dist_rate"])
    compare = lambda m: [call("compare_windows", feature="dist_rate", windows_from={"evidence_id": last_eid(m)})]  # noqa
    report = investigate(QUESTION, env[0], env[1], Scripted([feats], compare), sufficiency=LOOSE,
                         budget=Budget(max_model_turns=2))
    pid = ev(report, "compare_windows")
    refused(check(report, env[1], "effect_described", pid, claims=[C("descriptive_effect", pid)]),
            "primary coverage is not established")


# ======================================================================
# insufficient / no data / unavailable / no comparison
# ======================================================================

def test_insufficient_evidence_with_its_status_claim(insufficient_run):
    report, policy = insufficient_run
    pid = ev(report, "compare_windows")
    r = check(report, policy, "insufficient_evidence", pid, claims=[C("sufficiency_not_met", pid)])
    assert r.accepted and r.accepted_direction is None, r.problems
    g = r.claims[0]["grounding"]
    assert g["reasons_not_met"] and g["criteria"]["min_sessions"] == 4 and "value" not in g


@pytest.mark.parametrize("claims, text", [
    ([], "exactly one sufficiency_not_met claim citing the primary"),
    ("twice", "at most one comparison-status claim"),
    ("effect", "descriptive_effect"),
])
def test_insufficient_evidence_rules(insufficient_run, claims, text):
    report, policy = insufficient_run
    pid = ev(report, "compare_windows")
    claims = {"twice": [C("sufficiency_not_met", pid), C("sufficiency_not_met", pid)],
              "effect": [C("sufficiency_not_met", pid), C("descriptive_effect", pid)]}[claims] \
        if isinstance(claims, str) else claims
    refused(check(report, policy, "insufficient_evidence", pid, claims=claims), text)


def test_insufficient_evidence_never_reports_effect_values(insufficient_run):
    report, policy = insufficient_run
    pid = ev(report, "compare_windows")
    for field in ("measurements.effect.value", "measurements.per_session[0].difference",
                  "measurements.per_session[0].pre_windows[0].value"):
        refused(check(report, policy, "insufficient_evidence", pid, claims=[C("sufficiency_not_met", pid)],
                      observations=[{"evidence_id": pid, "field": field}]), "effect values are not reported")
    features = ev(report, "get_window_features")
    refused(check(report, policy, "insufficient_evidence", pid, claims=[C("sufficiency_not_met", pid)],
                  observations=[{"evidence_id": features, "field": "measurements.windows[0].features.dist_rate.median"}]),
            "effect values are not reported")
    ok = check(report, policy, "insufficient_evidence", pid, claims=[C("sufficiency_not_met", pid)],
               observations=[{"evidence_id": pid, "field": "measurements.n_sessions_paired"}])
    assert ok.accepted, ok.problems


def test_ok_primary_cannot_be_called_insufficient(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    refused(check(report, policy, "insufficient_evidence", pid, claims=[C("sufficiency_not_met", pid)]),
            "has status ok")


def test_no_comparison(env):
    report = investigate(QUESTION, env[0], env[1], Scripted([query()]), budget=Budget(max_model_turns=1))
    r = check(report, env[1], "insufficient_evidence", None, claims=[C("no_comparison")])
    assert r.accepted and r.claims[0]["grounding"] == {"comparisons": 0}, r.problems
    refused(check(report, env[1], "insufficient_evidence", None, claims=[]), "no_comparison")


def test_no_comparison_is_refused_when_one_exists(effect_run):
    report, policy = effect_run
    refused(check(report, policy, "inconclusive", None, claims=[C("no_comparison")]), "a comparison exists")


def test_comparison_unavailable_and_error_citing_rules(env, monkeypatch):
    def broken(self, request):
        raise RuntimeError("backend down")
    monkeypatch.setattr(CalMS21InvestigationTools, "compare_windows", broken)
    report = scripted_compare(env)
    pid = ev(report, "compare_windows")
    r = check(report, env[1], "insufficient_evidence", pid, claims=[C("comparison_unavailable", pid)])
    assert r.accepted and list(r.claims[0]["grounding"]["error_codes"]) == ["internal_error"], r.problems
    refused(check(report, env[1], "inconclusive", None, claims=[C("potential_confounder", pid,
                                                                   confounder="investigation_label")]),
            "is an error result")


def test_comparison_no_data(env):
    report = scripted_compare(env)
    feats = next(e for e in report.evidence if e["tool"] == "get_window_features")
    # windows whose exclusion Stage 5 re-applies (it re-checks clearance 0 only, so near_event baselines return)
    windows = feats["result"]["measurements"]["windows"]
    pre = [w["window_id"] for w in windows if w["kind"] == "pre_event" and w["status"] == "excluded"]
    base = [w["window_id"] for w in windows if w["kind"] == "baseline" and "overlaps_event" in w["exclusion_reasons"]]
    if not pre or not base:
        pytest.skip("fixture has no excluded pre-event and baseline windows")
    model = Scripted([call("compare_windows", feature="dist_rate", pre_window_ids=pre, baseline_window_ids=base)])
    run2 = investigate(QUESTION, env[0], env[1], model, budget=Budget(max_model_turns=1))
    pid = ev(run2, "compare_windows")
    assert next(e for e in run2.evidence if e["evidence_id"] == pid)["status"] == "no_data"
    r = check(run2, env[1], "insufficient_evidence", pid, claims=[C("comparison_no_data", pid)])
    assert r.accepted, r.problems
    refused(check(run2, env[1], "insufficient_evidence", pid, claims=[C("sufficiency_not_met", pid)]),
            "has status no_data")


# ======================================================================
# inconclusive, confounders, tracking, operational facts
# ======================================================================

def test_inconclusive_needs_an_evidence_grounded_limitation(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    refused(check(report, policy, "inconclusive", pid, claims=[]), "evidence-grounded limitation")
    refused(check(report, policy, "inconclusive", pid, claims=[C("operational_fact", kind="model_selected_plan")]),
            "evidence-grounded limitation")
    r = check(report, policy, "inconclusive", pid, claims=[C("potential_confounder", pid,
                                                              confounder="investigation_label")])
    assert r.accepted and r.accepted_direction is None, r.problems
    g = r.claims[0]["grounding"]
    assert g["predicate"] == "measured" and g["category"] == "potential_confounder"
    assert g["label_frames_pre"]["investigation"] * sum(g["label_frames_baseline"].values()) > \
        g["label_frames_baseline"]["investigation"] * sum(g["label_frames_pre"].values())


@pytest.mark.parametrize("value, text", [
    ("attack 0, investigation 0, mount 0, other 60", "contradicted"),              # consistent total, no excess
    ("n/a", "not assessed"), (None, "not assessed"), ("attack 0, investigation 0, mount 0, other 0", "not assessed"),
    ("attack 1, attack 2, mount 0, other 3", "not assessed"),
])
def test_confounder_contradicted_or_not_assessed(effect_run, value, text):
    report, policy = effect_run

    def relabel(req, res):
        for m in res["methodology"]:
            if m["name"] == "label_frames_pre":
                m["value"] = value
    forged = forge(report, "compare_windows", relabel)
    pid = ev(forged, "compare_windows")
    refused(check(forged, policy, "inconclusive", pid, claims=[C("potential_confounder", pid,
                                                                  confounder="investigation_label")]), text)


def test_tracking_concern_and_quality_operational_facts(dirty_run):
    report, policy = dirty_run
    pid = ev(report, "compare_windows")
    r = check(report, policy, "inconclusive", pid, claims=[C("tracking_concern", kind="quality_contamination_detected")])
    assert r.accepted, r.problems
    items = r.claims[0]["grounding"]["items"]
    assert items and all(i["source"] == "measurements.windows[].smoothing_contamination" for i in items)
    refused(check(report, policy, "inconclusive", pid, claims=[C("tracking_concern", kind="quality_verdict")]),
            "no recorded tracking fact")
    refused(check(report, policy, "inconclusive", pid, claims=[C("operational_fact", kind="quality_never_called")]),
            "was called in this run")
    m = check(report, policy, "inconclusive", pid, claims=[C("tracking_concern", kind="quality_contamination_detected"),
                                                           C("operational_fact", kind="model_selected_plan")])
    assert m.accepted and list(m.claims[1]["grounding"]["model_ids"]) == ["fake-investigator"]


def test_quality_never_called_and_failed(env, monkeypatch):
    report = scripted_compare(env)
    pid = ev(report, "compare_windows")
    r = check(report, env[1], "inconclusive", pid, claims=[C("operational_fact", kind="quality_never_called")])
    assert r.accepted, r.problems

    def broken(self, request):
        raise RuntimeError("quality down")
    monkeypatch.setattr(CalMS21InvestigationTools, "check_tracking_quality", broken)
    run2 = investigate(QUESTION, env[0], env[1], Scripted([call("check_tracking_quality", window_ids=[
        "win:pre:20:0:" + ep(VAL_A, 100)], history_frames=[1])]), budget=Budget(max_model_turns=1))
    qid = ev(run2, "check_tracking_quality")
    ok = check(run2, env[1], "insufficient_evidence", None,
               claims=[C("no_comparison"), C("operational_fact", qid, kind="quality_check_failed")])
    assert ok.accepted and list(ok.claims[1]["grounding"]["error_codes"]) == ["internal_error"], ok.problems


# ======================================================================
# flagged-window sensitivity
# ======================================================================

def sensitivity_ids(report):
    return [ev(report, "compare_windows", 0), ev(report, "compare_windows", 1), ev(report, "check_tracking_quality")]


def test_sensitivity_excluding_flagged_windows(dirty_run):
    report, policy = dirty_run
    ids = sensitivity_ids(report)
    r = check(report, policy, "effect_described", ids[0],
              claims=[C("descriptive_effect", ids[0]), C("sensitivity_excluding_flagged", *ids)])
    assert r.accepted, r.problems
    g = r.claims[1]["grounding"]
    assert [x["window_id"] for x in g["removed_contributing"]] == ["win:pre:20:0:" + ep(VAL_A, 100)]
    assert g["removed_contributing"][0]["flags"]["contamination"] == "detected"
    assert len(g["effects"]) == 2 and g["arithmetic_sign_relation"] in ("same arithmetic sign",
                                                                         "opposite arithmetic signs", "a zero value")
    assert "not scientific agreement" in g["sign_relation_note"]
    inc = check(report, policy, "inconclusive", ids[0], claims=[C("sensitivity_excluding_flagged", *ids)])
    assert inc.accepted and "effects" not in inc.claims[0]["grounding"]                  # status only
    assert "arithmetic_sign_relation" not in inc.claims[0]["grounding"]


@pytest.mark.parametrize("order, text", [((1, 0, 2), "requirement 3"), ((0, 0, 2), "distinct evidence ids")])
def test_sensitivity_roles_are_enforced(dirty_run, order, text):
    report, policy = dirty_run
    ids = sensitivity_ids(report)
    refused(check(report, policy, "inconclusive", ids[0],
                  claims=[C("sensitivity_excluding_flagged", *[ids[i] for i in order])]), text)


def test_sensitivity_refuses_changed_processing_or_conflicting_identity(dirty_run):
    report, policy = dirty_run

    def smoothing(req, res):
        for m in res["methodology"]:
            if m["name"] == "smoother_memory_frames":
                m["value"] = 5
    forged = forge(report, "compare_windows", smoothing, k=1)
    refused(check(forged, policy, "inconclusive", ev(forged, "compare_windows"),
                  claims=[C("sensitivity_excluding_flagged", *sensitivity_ids(forged))]), "requirement 8")

    def other_index(req, res):
        for d in res["provenance"]["data"]:
            if d["path"].endswith("extraction_index.json"):
                d["sha256"] = "3" * 64
    forged = forge(report, "check_tracking_quality", other_index)
    refused(check(forged, policy, "inconclusive", ev(forged, "compare_windows"),
                  claims=[C("sensitivity_excluding_flagged", *sensitivity_ids(forged))]), "data provenance differs")


def test_sensitivity_refuses_an_unflagged_removed_window(dirty_run):
    report, policy = dirty_run

    def unflag(req, res):
        for w in res["measurements"]["windows"]:
            w["smoothing_contamination"]["status"] = "none_detected"
            w["smoothing_contamination"]["frames"] = []
    forged = forge(report, "check_tracking_quality", unflag, check=False)
    r = check(forged, policy, "inconclusive", ev(forged, "compare_windows"),
              claims=[C("sensitivity_excluding_flagged", *sensitivity_ids(forged))])
    assert not r.accepted          # refused by the contract check of the quality snapshot or by requirement 9


# ======================================================================
# malformed, fabricated, forged
# ======================================================================

GOOD_SHAPE = {"finding": "inconclusive", "primary_evidence_id": None, "observations": [], "claims": [],
              "next_investigations": []}


@pytest.mark.parametrize("change, text", [
    (None, "not an object"), ([], "not an object"),
    ({"direction": "positive"}, "exactly the keys"),                                    # forged direction
    ({"interpretations": ["Approach causes attack."]}, "exactly the keys"),            # prose
    ({"finding": "proven"}, "finding must be one of"), ({"finding": ["inconclusive"]}, "finding must be one of"),
    ({"finding": True}, "finding must be one of"),
    ({"primary_evidence_id": 7}, "evidence id or null"), ({"primary_evidence_id": ""}, "evidence id or null"),
    ({"claims": "x"}, "claims must be a list"), ({"claims": [None]}, "exactly the keys"),
    ({"claims": [{"type": "x", "evidence_ids": []}]}, "exactly the keys"),
    ({"claims": [C("causes")]}, "unknown claim type"),
    ({"claims": [{"type": ["coverage"], "evidence_ids": [], "params": {}}]}, "unknown claim type"),
    ({"claims": [C("coverage", aspect="everything")]}, "params.aspect must be one of"),
    ({"claims": [C("coverage", aspect=["primary_comparison"])]}, "params.aspect must be one of"),
    ({"claims": [C("descriptive_effect", direction="positive")]}, "params has exactly the keys"),
    ({"claims": [C("descriptive_effect", value=-7.8)]}, "params has exactly the keys"),
    ({"claims": [C("potential_confounder", statement="approach causes attack")]}, "params has exactly the keys"),
    ({"claims": [{"type": "no_comparison", "evidence_ids": "ev:x", "params": {}}]}, "distinct evidence ids"),
    ({"claims": [{"type": "no_comparison", "evidence_ids": [3], "params": {}}]}, "distinct evidence ids"),
    ({"observations": [{"evidence_id": "x", "field": 3}]}, "are strings"),
    ({"observations": [{"evidence_id": "x", "field": "status", "value": 9}]}, "exactly the keys"),
    ({"next_investigations": [{"action": "vary_window", "params": {}, "because": [0]}]}, "unknown action"),
    ({"next_investigations": [{"action": "include_more_sessions", "params": {}, "because": [5]}]}, "claim indices"),
    ({"next_investigations": [{"action": "include_more_sessions", "params": {}, "because": [True]}]},
     "claim indices"),
    ({"claims": [C("no_comparison")] * 21}, "at most 20"),
])
def test_malformed_and_forged_proposals_are_refused_without_raising(effect_run, change, text):
    report, policy = effect_run
    proposal = change if not isinstance(change, dict) else {**GOOD_SHAPE, **change}
    r = cl.validate_findings(proposal, evidence=report.evidence, trace=report.trace[:-1], policy=policy)
    refused(r, text)


def test_fabricated_references_are_refused(effect_run):
    report, policy = effect_run
    fake = "ev:compare_windows:" + "f" * 24
    refused(check(report, policy, "effect_described", fake, claims=[C("descriptive_effect", fake)]),
            "primary evidence is not one of the snapshots")
    pid = ev(report, "compare_windows")
    refused(check(report, policy, "inconclusive", pid, claims=[C("potential_confounder", fake,
                                                                  confounder="investigation_label")]),
            "is not evidence of this run")
    refused(check(report, policy, "inconclusive", pid, claims=[C("tracking_concern", fake, kind="usable_fraction")]),
            "is not evidence of this run")
    refused(check(report, policy, "inconclusive", pid, claims=[C("coverage", aspect="primary_comparison")],
                  observations=[{"evidence_id": fake, "field": "status"}]), "is not evidence of this run")


def test_wrong_effect_direction_cannot_be_asserted(effect_run):
    """The direction is never an input: it is derived from the effect's sign,
    and only under effect_described; any attempt to supply one is refused."""
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    snap = next(e for e in report.evidence if e["evidence_id"] == pid)
    r = check(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    assert r.accepted_direction == ctx._sign(snap["result"]["measurements"]["effect"]["value"])
    wrong = "positive" if r.accepted_direction == "negative" else "negative"
    proposal = {**GOOD_SHAPE, "finding": "effect_described", "primary_evidence_id": pid,
                "claims": [C("descriptive_effect", pid)], "direction": wrong}
    refused(cl.validate_findings(proposal, evidence=report.evidence, trace=report.trace[:-1], policy=policy),
            "exactly the keys")


def test_descriptive_effect_needs_effect_described_and_the_primary(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    refused(check(report, policy, "inconclusive", pid, claims=[C("descriptive_effect", pid)]),
            "only under the finding effect_described")
    refused(check(report, policy, "effect_described", pid, claims=[]), "exactly one descriptive_effect")
    refused(check(report, policy, "effect_described", pid,
                  claims=[C("descriptive_effect", pid), C("sufficiency_not_met", pid)]), "has status ok")


def test_tampered_or_conflicting_evidence_is_refused(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    tampered = copy.deepcopy(report.evidence)
    tampered[-1]["result"]["measurements"]["effect"]["value"] = 999.0
    r = cl.validate_findings({**GOOD_SHAPE, "finding": "effect_described", "primary_evidence_id": pid,
                              "claims": [C("descriptive_effect", pid)]},
                             evidence=tampered, trace=report.trace[:-1], policy=policy)
    refused(r, "failed verification")

    def other_index(req, res):
        for d in res["provenance"]["data"]:
            if d["path"].endswith("extraction_index.json"):
                d["sha256"] = "4" * 64
    forged = forge(report, "get_window_features", other_index)
    pid = ev(forged, "compare_windows")
    refused(check(forged, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)]),
            "primary coverage is not established")


def test_authorization_is_not_sufficiency(effect_run, insufficient_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    diag = check(report, None, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    assert diag.accepted and diag.authorization == "not_rechecked" and not diag.publishable    # diagnostic only
    refused(check(report, ic.EvidencePolicy(MANIFEST), "effect_described", pid,
                  claims=[C("descriptive_effect", pid)]), "failed verification")         # authorization refused
    ins, pol = insufficient_run
    pid2 = ev(ins, "compare_windows")
    r = check(ins, pol, "effect_described", pid2, claims=[C("descriptive_effect", pid2)])
    assert r.authorization == "rechecked" or not r.accepted
    refused(r, "has status insufficient_evidence")                                       # authorized, insufficient


def test_execution_is_checked(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    with_stop = cl.validate_findings({**GOOD_SHAPE, "finding": "effect_described", "primary_evidence_id": pid,
                                      "claims": [C("descriptive_effect", pid)]},
                                     evidence=report.evidence, trace=report.trace, policy=policy)   # prefix mode
    refused(with_stop, "an execution prefix has no stop event")
    refused(cl.validate_findings({**GOOD_SHAPE, "finding": "effect_described", "primary_evidence_id": pid,
                                  "claims": [C("descriptive_effect", pid)]},
                                 evidence=report.evidence, trace=report.trace[:-1] + [None], policy=policy),
            "malformed trace entries")


# ======================================================================
# incomplete runs, next investigations, representation
# ======================================================================

def test_incomplete_findings_invent_nothing(env):
    report = scripted_compare(env)
    assert report.stopping_reason == "max_model_turns"
    r = cl.incomplete_findings(evidence=report.evidence, trace=report.trace, policy=env[1],
                               stopping_reason=report.stopping_reason)
    assert r.accepted and r.finding == "incomplete" and r.claims == () and r.primary_evidence_id is None
    assert r.fingerprint == ctx.content_fingerprint(ctx.fingerprint_payload(
        evidence=report.evidence, finding="incomplete", primary_evidence_id=None))
    bad = cl.incomplete_findings(evidence=report.evidence, trace=report.trace[:-1], policy=env[1],
                                 stopping_reason=report.stopping_reason)
    assert not bad.accepted                                                        # the completed trace is required
    refused(cl.validate_findings({**GOOD_SHAPE, "finding": "incomplete"}, evidence=report.evidence,
                                 trace=report.trace[:-1], policy=env[1]), "finding must be one of")


def test_next_investigations(env, dirty_run):
    partial = scripted_compare(env, sessions=[VAL_A, VAL_B])
    pid = ev(partial, "compare_windows")
    nxt = [{"action": "include_more_sessions", "params": {}, "because": [0]}]
    assert check(partial, env[1], "effect_described", pid, claims=[C("descriptive_effect", pid)], next_=nxt).accepted
    complete, policy = dirty_run
    cid = ev(complete, "compare_windows")
    base = [C("tracking_concern", kind="quality_contamination_detected")]
    refused(check(complete, policy, "inconclusive", cid, claims=base, next_=nxt), "only when the primary coverage")
    q = ev(complete, "check_tracking_quality")
    ok = check(complete, policy, "inconclusive", cid, claims=base,
               next_=[{"action": "exclude_flagged_windows", "params": {"quality": q}, "because": [0]}])
    assert ok.accepted, ok.problems
    feats = ev(complete, "get_window_features")
    r = check(complete, policy, "inconclusive", cid, claims=base,
              next_=[{"action": "check_tracking_quality", "params": {"windows_from": feats}, "because": [0]}])
    assert r.accepted or any("already assessed" in p for p in r.problems)


def test_representation_is_frozen_deterministic_and_versioned(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    a = check(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    b = check(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    assert a.to_json() == b.to_json() and json.loads(json.dumps(a.to_json())) == a.to_json()
    assert (a.findings_format, a.registry_version, a.grid_version) == ("findings/2", "claims/2.0", "none")
    with pytest.raises(TypeError):
        a.claims[0]["grounding"]["value"] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.accepted = False
    assert ctx.detect_format({"findings_format": a.findings_format, "registry_version": a.registry_version,
                              "grid_version": a.grid_version}) == ctx.V2_FORMAT


# ======================================================================
# Codex 2 M1 follow-up
# ======================================================================

from live_tracker.investigation_contracts import SufficiencyCriteria  # noqa: E402


def complete(report, policy, finding, primary=None, claims=(), finished=None, reason=None, trace=None):
    proposal = {"finding": finding, "primary_evidence_id": primary, "observations": [], "claims": list(claims),
                "next_investigations": []}
    return provider_schema_holds(proposal, cl.validate_findings(
        proposal, evidence=report.evidence, trace=report.trace if trace is None else trace, policy=policy,
        mode="complete", stopping_reason=report.stopping_reason if reason is None else reason,
        finished=report.finished if finished is None else finished))


# ---- 1. complete mode: only an actually finished run ----

def test_complete_mode_accepts_a_finished_run(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    r = complete(report, policy, "effect_described", pid, claims=[C("descriptive_effect", pid)])
    assert r.accepted and r.publishable, r.problems


def test_complete_mode_refuses_a_budget_stopped_run(env):
    report = scripted_compare(env)                                    # max_model_turns, finished False
    pid = ev(report, "compare_windows")
    r = refused(complete(report, env[1], "effect_described", pid, claims=[C("descriptive_effect", pid)]),
                "incomplete runs are represented only by incomplete_findings()")
    assert not r.publishable
    assert cl.incomplete_findings(evidence=report.evidence, trace=report.trace, policy=env[1],
                                  stopping_reason=report.stopping_reason).finding == "incomplete"


@pytest.mark.parametrize("finished, reason", [(True, "timeout"), (False, "finished"), (False, "max_model_turns"),
                                              (None, "finished"), ("yes", "finished")])
def test_complete_mode_refuses_contradictory_finish_states(effect_run, finished, reason):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    proposal = {"finding": "effect_described", "primary_evidence_id": pid, "observations": [],
                "claims": [C("descriptive_effect", pid)], "next_investigations": []}
    r = cl.validate_findings(proposal, evidence=report.evidence, trace=report.trace, policy=policy, mode="complete",
                             stopping_reason=reason, finished=finished)
    refused(r, "incomplete runs are represented only by incomplete_findings()")


def test_complete_mode_refuses_a_trace_without_an_accepted_finish(env):
    """A budget-stopped run's trace relabelled as finished: the execution
    assessment refuses it (no accepted finish exists)."""
    report = scripted_compare(env)
    pid = ev(report, "compare_windows")
    trace = report.trace[:-1] + [{**report.trace[-1], "stopping_reason": "finished"}]
    refused(complete(report, env[1], "effect_described", pid, claims=[C("descriptive_effect", pid)],
                     finished=True, reason="finished", trace=trace),
            "an accepted finish exists exactly when the investigation finished")


def test_complete_mode_refuses_a_run_whose_finishes_were_all_refused(env):
    """Failed (refused) finishes are not an accepted finish."""
    report = scripted_compare(env)
    pid = ev(report, "compare_windows")
    failed = {"step": 99, "kind": "finish", "tool": "finish_investigation", "outcome": "refused",
              "refused": "final_answer_rejected"}
    trace = report.trace[:-1] + [failed, {**report.trace[-1], "stopping_reason": "finished"}]
    refused(complete(report, env[1], "effect_described", pid, claims=[C("descriptive_effect", pid)],
                     finished=True, reason="finished", trace=trace),
            "an accepted finish exists exactly when the investigation finished")


# ---- 2. impossible label counts ----

def relabel(name, value):
    def mutate(req, res):
        for m in res["methodology"]:
            if m["name"] == name:
                m["value"] = value
    return mutate


@pytest.mark.parametrize("name, value, text", [
    ("label_frames_pre", "attack 0, investigation 999999, mount 0, other 0", "pre-event label frames total 999999"),
    ("label_frames_pre", "attack 0, investigation 21, mount 0, other 40", "pre-event label frames total 61"),
    ("label_frames_baseline", "attack 0, investigation 0, mount 0, other 519", "baseline label frames total 519"),
    ("label_frames_baseline", "attack 0, investigation 5, mount 0, other 520", "baseline label frames total 525"),
])
def test_impossible_label_counts_are_refused_despite_consistent_hashes(effect_run, name, value, text):
    report, policy = effect_run
    forged = forge(report, "compare_windows", relabel(name, value))      # contract-valid, re-hashed everywhere
    assert forged.verify_evidence() == []
    pid = ev(forged, "compare_windows")
    refused(check(forged, policy, "inconclusive", pid,
                  claims=[C("potential_confounder", pid, confounder="investigation_label")]),
            f"impossible label counts: {text}")


@pytest.mark.parametrize("value", ["attack 0, investigation -20, mount 0, other 80",
                                   "attack 0, investigation 20.0, mount 0, other 40",
                                   "attack 0, investigation 20, investigation 0, other 40",
                                   "attack 0, investigation 20, mount 0", "attack 0, investigation 20, mount 0, "
                                   "other 40, rearing 0", "attack 0, investigation +20, mount 0, other 40"])
def test_malformed_label_counts_are_not_assessed(effect_run, value):
    report, policy = effect_run
    forged = forge(report, "compare_windows", relabel("label_frames_pre", value))
    pid = ev(forged, "compare_windows")
    refused(check(forged, policy, "inconclusive", pid,
                  claims=[C("potential_confounder", pid, confounder="investigation_label")]), "not assessed")


def test_consistent_label_counts_remain_measured(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    g = check(report, policy, "inconclusive", pid,
              claims=[C("potential_confounder", pid, confounder="investigation_label")]).claims[0]["grounding"]
    assert g["predicate"] == "measured" and sum(g["label_frames_pre"].values()) == 60


# ---- 3. inconclusive primary status ----

def test_inconclusive_refuses_an_error_primary_whatever_the_limitation(env, monkeypatch):
    def broken(self, request):
        raise RuntimeError("backend down")
    monkeypatch.setattr(CalMS21InvestigationTools, "compare_windows", broken)
    report = scripted_compare(env)
    pid = ev(report, "compare_windows")
    for claims in ([C("operational_fact", kind="quality_never_called")],
                   [C("operational_fact", kind="quality_never_called"), C("comparison_unavailable", pid)]):
        refused(check(report, env[1], "inconclusive", pid, claims=claims),
                "inconclusive needs a non-error primary comparison or a null primary")
    ok = check(report, env[1], "inconclusive", None, claims=[C("operational_fact", kind="quality_never_called")])
    assert ok.accepted, ok.problems                                      # a null primary is allowed


def test_inconclusive_with_a_non_error_primary_is_allowed(insufficient_run):
    report, policy = insufficient_run
    pid = ev(report, "compare_windows")
    r = check(report, policy, "inconclusive", pid, claims=[C("methodological_limitation", pid,
                                                              kind="provisional_criteria")])
    assert r.accepted, r.problems


# ---- 4. the three restored families ----

def test_measurement_context(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    snap = next(e for e in report.evidence if e["evidence_id"] == pid)
    r = check(report, policy, "inconclusive", pid, claims=[
        C("measurement_context", pid, aspect="session_timing"), C("measurement_context", pid, aspect="pixel_units"),
        C("methodological_limitation", pid, kind="provisional_criteria")])
    assert r.accepted, r.problems
    timing, units = r.claims[0]["grounding"], r.claims[1]["grounding"]
    raw = ctx._method(snap["result"], "window_start_s_pre")
    assert tuple(timing["window_start_s_pre"].values()) == tuple(float(x) for x in raw.split(" / "))   # parsed
    assert timing["category"] == units["category"] == "measured_fact" and timing["note"] == cl.TIMING_NOTE
    assert units["unit"] == "px/s"
    refused(check(report, policy, "inconclusive", pid, claims=[C("measurement_context", pid, aspect="session_timing")]),
            "evidence-grounded limitation")                       # context alone never justifies inconclusive


@pytest.mark.parametrize("mutate, text", [
    (relabel("units", "mm/s"), "contradicted"),
    (lambda req, res: res["methodology"].__setitem__(slice(None), [m for m in res["methodology"]
                                                                   if m["name"] != "units"]), "not assessed"),
])
def test_pixel_units_contradicted_or_not_assessed(effect_run, mutate, text):
    report, policy = effect_run
    forged = forge(report, "compare_windows", mutate)
    pid = ev(forged, "compare_windows")
    refused(check(forged, policy, "inconclusive", pid, claims=[C("measurement_context", pid, aspect="pixel_units"),
                                                               C("methodological_limitation", pid,
                                                                 kind="provisional_criteria")]), text)


def test_measurement_context_citations(effect_run):
    report, policy = effect_run
    pid, fid = ev(report, "compare_windows"), ev(report, "get_window_features")
    limit = C("methodological_limitation", pid, kind="provisional_criteria")
    refused(check(report, policy, "inconclusive", pid, claims=[C("measurement_context", fid, aspect="session_timing"),
                                                               limit]), "is a get_window_features result")
    refused(check(report, policy, "inconclusive", pid, claims=[C("measurement_context", pid,
                                                                 aspect="mover_unidentified"), limit]),
            "is a compare_windows result, not get_window_features")   # M2: features evidence only


@pytest.mark.parametrize("tool, disclosure, accepted", [
    ("query_events", "approach_unlabelled", True), ("get_window_features", "approach_unlabelled", True),
    ("compare_windows", "approach_unlabelled", True), ("check_tracking_quality", "approach_unlabelled", False),
    ("compare_windows", "motion_accuracy_unverified", True), ("check_tracking_quality", "motion_accuracy_unverified",
                                                               True),
    ("query_events", "motion_accuracy_unverified", False),
])
def test_tool_disclosure_citations(effect_run, tool, disclosure, accepted):
    report, policy = effect_run
    pid, eid = ev(report, "compare_windows"), ev(report, tool)
    r = check(report, policy, "inconclusive", pid, claims=[C("tool_disclosure", eid, disclosure=disclosure),
                                                           C("methodological_limitation", pid,
                                                             kind="provisional_criteria")])
    if accepted:
        assert r.accepted, r.problems
        g = r.claims[0]["grounding"]
        assert g["category"] == "tool_disclosure" and g["text"] in next(
            e for e in report.evidence if e["evidence_id"] == eid)["result"]["limitations"]
        assert g["scope"] == ("dataset" if disclosure == "approach_unlabelled" else "tool")
    else:
        refused(r, "cannot carry this disclosure")


def test_tool_disclosure_needs_the_marker_and_never_justifies_inconclusive(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    refused(check(report, policy, "inconclusive", pid, claims=[C("tool_disclosure", pid,
                                                                 disclosure="approach_unlabelled")]),
            "evidence-grounded limitation")
    forged = forge(report, "compare_windows", lambda req, res: res["limitations"].__setitem__(
        slice(None), [t for t in res["limitations"] if "has no 'approach' label" not in t]))
    pid2 = ev(forged, "compare_windows")
    refused(check(forged, policy, "inconclusive", pid2, claims=[C("tool_disclosure", pid2,
                                                                  disclosure="approach_unlabelled"),
                                                                C("methodological_limitation", pid2,
                                                                  kind="provisional_criteria")]),
            "do not contain the registry marker")


def test_methodological_limitations(effect_run):
    report, policy = effect_run
    pid, qid = ev(report, "compare_windows"), ev(report, "query_events")
    r = check(report, policy, "inconclusive", pid, claims=[
        C("methodological_limitation", pid, kind="provisional_criteria"),
        C("methodological_limitation", qid, kind="provisional_parameters"),
        C("methodological_limitation", pid, kind="descriptive_only"),
        C("methodological_limitation", pid, kind="single_feature"),
        C("methodological_limitation", pid, kind="baseline_clearance_not_rechecked")])
    assert r.accepted, r.problems
    g = [c["grounding"] for c in r.claims]
    assert g[0]["criteria"]["status"] == "provisional" and g[1]["provisional_items"]
    assert g[2]["uncertainty_method"] == "none" and g[3]["feature"] == "dist_rate"
    assert g[4]["rechecked_clearance_frames"] == 0
    only = check(report, policy, "inconclusive", pid, claims=[C("methodological_limitation", pid,
                                                                kind="provisional_parameters")])
    refused(only, "evidence-grounded limitation")            # only provisional_criteria justifies inconclusive


@pytest.mark.parametrize("claim, text", [
    (lambda pid, qid: C("methodological_limitation", qid, kind="provisional_criteria"), "cites a compare_windows"),
    (lambda pid, qid: C("methodological_limitation", pid, qid, kind="descriptive_only"), "cites exactly 1"),
    (lambda pid, qid: C("methodological_limitation", pid, kind="everything"), "params.kind must be one of"),
    (lambda pid, qid: C("methodological_limitation", pid, kind=["provisional_criteria"]), "params.kind must be one"),
    (lambda pid, qid: C("methodological_limitation", pid, kind="provisional_criteria", note="x"),
     "params has exactly the keys"),
    (lambda pid, qid: C("tool_disclosure", pid, disclosure="causes_attack"), "params.disclosure must be one of"),
    (lambda pid, qid: C("measurement_context", pid, aspect="pixel_units", unit="mm"), "params has exactly the keys"),
])
def test_malformed_parameters_and_unsupported_citations(effect_run, claim, text):
    report, policy = effect_run
    pid, qid = ev(report, "compare_windows"), ev(report, "query_events")
    refused(check(report, policy, "inconclusive", pid, claims=[claim(pid, qid)]), text)


def test_approved_criteria_are_not_a_provisional_limitation(env):
    approved = SufficiencyCriteria(min_sessions=1, min_windows_per_group=1, min_usable_fraction=0.9,
                                   status="approved", decided_by="SYNTHETIC test owner")
    report = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=approved)
    pid = ev(report, "compare_windows")
    refused(check(report, env[1], "inconclusive", pid, claims=[C("methodological_limitation", pid,
                                                                 kind="provisional_criteria")]), "not provisional")


def test_clearance_other_than_zero_is_refused(effect_run):
    report, policy = effect_run
    forged = forge(report, "compare_windows", relabel("baseline_clearance_frames", 30))
    pid = ev(forged, "compare_windows")
    refused(check(forged, policy, "inconclusive", pid, claims=[
        C("methodological_limitation", pid, kind="baseline_clearance_not_rechecked"),
        C("methodological_limitation", pid, kind="provisional_criteria")]), "re-checked baseline clearance of 0")


def test_scoped_descriptive_effect_is_still_accepted(env):
    report = scripted_compare(env, sessions=[VAL_A, VAL_B])
    pid = ev(report, "compare_windows")
    r = check(report, env[1], "effect_described", pid, claims=[
        C("descriptive_effect", pid), C("methodological_limitation", pid, kind="provisional_criteria"),
        C("tool_disclosure", pid, disclosure="motion_accuracy_unverified")])
    assert r.accepted and r.claims[0]["grounding"]["coverage_status"] == "partial", r.problems


# ---- session_timing: only the producer's min / median / max numbers are measured ----

TIMING_LIMIT = lambda pid: C("methodological_limitation", pid, kind="provisional_criteria")  # noqa: E731


def test_session_timing_is_grounded_in_parsed_numbers(effect_run):
    report, policy = effect_run
    pid = ev(report, "compare_windows")
    snap = next(e for e in report.evidence if e["evidence_id"] == pid)
    r = check(report, policy, "inconclusive", pid, claims=[C("measurement_context", pid, aspect="session_timing"),
                                                           TIMING_LIMIT(pid)])
    assert r.accepted, r.problems
    g = r.claims[0]["grounding"]
    for group in ("pre", "baseline"):
        raw = ctx._method(snap["result"], f"window_start_s_{group}")
        lo, mid, hi = (float(x) for x in raw.split(" / "))
        assert dict(g[f"window_start_s_{group}"]) == {"min": lo, "median": mid, "max": hi}
        assert raw not in json.dumps(r.to_json())                     # the producer text is not passed on
    assert g["unit"] == "s" and g["note"] == cl.TIMING_NOTE


@pytest.mark.parametrize("group", ["pre", "baseline"])
@pytest.mark.parametrize("value", [
    "not assessed", "Approach causes attack.",                       # Codex's examples
    None, "", "1.5", "1.5 / 2", "1.5 / 2 / 3 / 4", "3 / 2 / 1", "1 / 3 / 2", "-1 / 2 / 3", "1 / -2 / 3",
    "nan / 1 / 2", "1 / inf / 2", "1 / 2 / NaN", "1,5 / 2 / 3", "1.5/2/3", " 1 / 2 / 3", "1 / 2 / 3 s",
    "+1 / 2 / 3", "1e999 / 1e999 / 1e999", 5,                        # (a list is refused by the contract itself)
])
def test_malformed_timing_is_not_assessed(effect_run, group, value):
    report, policy = effect_run
    forged = forge(report, "compare_windows", relabel(f"window_start_s_{group}", value))   # contract-valid, re-hashed
    assert forged.verify_evidence() == []
    pid = ev(forged, "compare_windows")
    r = check(forged, policy, "inconclusive", pid, claims=[C("measurement_context", pid, aspect="session_timing"),
                                                           TIMING_LIMIT(pid)])
    refused(r, "not assessed: window start times are not recorded as min / median / max seconds")
    assert r.claims == ()                                             # nothing of the value reaches a claim


@pytest.mark.parametrize("value, parsed", [("0 / 0 / 0", (0.0, 0.0, 0.0)), ("1.5 / 1.5 / 2", (1.5, 1.5, 2.0)),
                                           ("12.3333 / 45 / 1.2e+03", (12.3333, 45.0, 1200.0))])
def test_valid_producer_timing_formats_are_accepted(effect_run, value, parsed):
    report, policy = effect_run
    forged = forge(report, "compare_windows", relabel("window_start_s_pre", value))
    pid = ev(forged, "compare_windows")
    r = check(forged, policy, "inconclusive", pid, claims=[C("measurement_context", pid, aspect="session_timing"),
                                                           TIMING_LIMIT(pid)])
    assert r.accepted, r.problems
    assert tuple(r.claims[0]["grounding"]["window_start_s_pre"].values()) == parsed


NO_DIFFERENCE_WORDS = ("difference", "differ", "earlier", "later", "drift is present", "drifted", "confounded",
                       "confounder", "is evidence of")


def _timing_claim(report, policy, pre, base):
    forged = forge(forge(report, "compare_windows", relabel("window_start_s_pre", pre)),
                   "compare_windows", relabel("window_start_s_baseline", base))
    pid = ev(forged, "compare_windows")
    return check(forged, policy, "inconclusive", pid, claims=[C("measurement_context", pid, aspect="session_timing"),
                                                              TIMING_LIMIT(pid)])


def test_identical_timing_summaries_assert_no_difference(effect_run):
    report, policy = effect_run
    r = _timing_claim(report, policy, "0 / 0 / 0", "0 / 0 / 0")
    assert r.accepted, r.problems
    g = r.claims[0]["grounding"]
    assert dict(g["window_start_s_pre"]) == dict(g["window_start_s_baseline"]) == {"min": 0.0, "median": 0.0,
                                                                                    "max": 0.0}
    assert g["unit"] == "s" and g["note"] == cl.TIMING_NOTE
    text = json.dumps({k: v for k, v in ctx.thaw(g).items() if isinstance(v, str)}).lower()
    assert not any(w in text.replace("not evidence of confounding", "") for w in NO_DIFFERENCE_WORDS), text


def test_unequal_timing_summaries_use_the_same_neutral_text(effect_run):
    report, policy = effect_run
    r = _timing_claim(report, policy, "1.5 / 2 / 3", "10 / 20.5 / 1.2e+02")
    assert r.accepted, r.problems
    g = r.claims[0]["grounding"]
    assert dict(g["window_start_s_pre"]) == {"min": 1.5, "median": 2.0, "max": 3.0}
    assert dict(g["window_start_s_baseline"]) == {"min": 10.0, "median": 20.5, "max": 120.0}
    assert g["note"] == cl.TIMING_NOTE
    text = json.dumps({k: v for k, v in ctx.thaw(g).items() if isinstance(v, str)}).lower()
    assert not any(w in text.replace("not evidence of confounding", "") for w in NO_DIFFERENCE_WORDS), text
