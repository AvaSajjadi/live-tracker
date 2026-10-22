"""Tests for Findings v2 checkpoint M3: the structured findings/2 finish in
the Stage 6 loop and the Stage 8 offline emulator (new file).

Evidence comes from real Stage 6 runs over the Stage 4 SYNTHETIC fixtures;
no model, network or paid provider is used.
"""

import copy
import json

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_claims as cl
import live_tracker.investigation_context as ctx
import live_tracker.investigation_live as live
import live_tracker.investigation_report as ir
import live_tracker.investigation_report_v2 as v2
from live_tracker.investigation_agent import Budget, ModelReply
from live_tracker.investigation_contracts import PROVISIONAL_SUFFICIENCY
from test_investigation_agent import (LOOSE, QUESTION, Investigator, Scripted, _until_compare, call, contaminated,  # noqa: F401
                                      history, last_result, last_result_of, no_network, obs, query, text)
from test_investigation_windows import build

V2_KEYS = {"findings_format", "registry_version", "grid_version", "question", "finished", "stopping_reason", "finding",
           "primary_evidence_id", "observations", "claims", "next_investigations", "evidence", "trace", "usage",
           "fingerprint"}


def run(env, model, **kw):
    """A findings/2 investigation (the default format)."""
    return ia.investigate(QUESTION, env[0], env[1], model, **kw)


def C(claim_type, *ids, **params):
    return {"type": claim_type, "evidence_ids": list(ids), "params": params}


def finish(**args):
    base = {"finding": "inconclusive", "primary_evidence_id": None, "observations": [], "claims": [],
            "next_investigations": []}
    return call(ia.FINISH, **{**base, **args})


class InvestigatorV2(Investigator):
    """The Stage 6 fake investigator (it decides only from the tool results in
    its transcript), finishing with structured findings/2 claims chosen from
    the evidence it has seen."""

    def respond(self, system, tools, messages):
        assert system == ia.SYSTEM_PROMPT_V2 and tools == ia.TOOL_SPECS_V2       # the v2 contract is sent
        self.n += 1
        done = [x for x in history(messages) if "evidence_id" in x[2]]
        why, name, args = self.decide(done)
        if name == ia.FINISH:
            args = self.structured(done)
        return ModelReply(content=[{"type": "text", "text": why},
                                   {"type": "tool_use", "id": f"tu_{self.n}", "name": name, "input": args}],
                          stop_reason="tool_use", input_tokens=1000, output_tokens=200, model="fake-investigator-v2")

    @staticmethod
    def structured(done):
        comps = [x[2] for x in done if x[0] == "compare_windows"]
        quals = [x[2] for x in done if x[0] == "check_tracking_quality"]
        if not comps:
            q = next(x[2] for x in done if x[0] == "query_events")
            return {"finding": "insufficient_evidence", "primary_evidence_id": None, "observations": [obs(q, "status")],
                    "claims": [C("no_comparison")], "next_investigations": []}
        main = comps[0]
        pid = main["evidence_id"]
        observations = [obs(main, "status"), obs(main, "measurements.n_sessions_paired")]
        if main["status"] == "ok":
            claims = [C("descriptive_effect", pid), C("coverage", aspect="primary_comparison")]
            if len(comps) > 1 and quals:
                claims.append(C("sensitivity_excluding_flagged", pid, comps[1]["evidence_id"], quals[0]["evidence_id"]))
            return {"finding": "effect_described", "primary_evidence_id": pid,
                    "observations": observations + [obs(main, "measurements.effect.value")], "claims": claims,
                    "next_investigations": []}
        return {"finding": "insufficient_evidence", "primary_evidence_id": pid, "observations": observations,
                "claims": [C("sufficiency_not_met", pid), C("methodological_limitation", pid,
                                                            kind="provisional_criteria")],
                "next_investigations": []}


@pytest.fixture
def env(tmp_path):
    return build(tmp_path)


def finish_steps(report):
    return [t for t in report.trace if t["kind"] == "finish"]


# ======================================================================
# 1. structured finish end to end
# ======================================================================

def test_structured_finish_is_accepted_frozen_and_rendered(env):
    report = run(env, InvestigatorV2(), sufficiency=LOOSE)
    assert isinstance(report, ia.InvestigationReportV2) and set(report.to_json()) == V2_KEYS
    assert report.finished and report.stopping_reason == "finished" and report.finding == "effect_described"
    assert (report.findings_format, report.registry_version, report.grid_version) == ("findings/2", "claims/2.0",
                                                                                      "none")
    [fin] = finish_steps(report)
    proposal = {k: getattr(report, k) for k in cl.PROPOSAL_KEYS}
    assert fin["outcome"] == "accepted" and fin["arguments"] == proposal                  # the trace binds the claims
    assert [t["kind"] for t in report.trace[-2:]] == ["finish", "stop"]
    # Stage 6 fingerprint == the registry's complete-mode fingerprint == Stage 7's
    again = cl.validate_findings(proposal, evidence=report.evidence, trace=report.trace, policy=env[1],
                                 mode="complete", stopping_reason="finished", finished=True)
    assert again.accepted and again.fingerprint == report.fingerprint
    doc, md = v2.render_v2(report, policy=env[1])
    assert doc["verification"]["fingerprint"]["recomputed"] == report.fingerprint
    assert doc["status"]["publication"] == "publishable"
    loaded = json.loads(json.dumps(report.to_json()))                                     # JSON round trip
    doc2, md2 = v2.render_v2(loaded, policy=env[1])
    assert ir.to_json_text(doc2) == ir.to_json_text(doc) and md2 == md
    assert not ({"direction", "interpretations", "next_steps", "limitations"} & set(report.to_json()))


def test_no_model_prose_reaches_the_report(env):
    report = run(env, InvestigatorV2(), sufficiency=LOOSE)
    doc, md = v2.render_v2(report, policy=env[1])
    reasoning = [t["text"] for t in report.trace if t["kind"] == "model_turn" and t.get("text")]
    assert reasoning and not any(r in md for r in reasoning)                             # recorded, never rendered


# ======================================================================
# 2. rejected finishes: prose, numbers, directions, fabricated references
# ======================================================================

def good_after_compare(m):
    c = last_result_of(m, "compare_windows")
    return [finish(finding="effect_described", primary_evidence_id=c["evidence_id"],
                   observations=[obs(c, "measurements.effect.value")], claims=[C("descriptive_effect", c["evidence_id"])])]


def bad(**override):
    def turn(m):
        c = last_result(m)
        args = {"finding": "effect_described", "primary_evidence_id": c["evidence_id"],
                "observations": [obs(c, "status")], "claims": [C("descriptive_effect", c["evidence_id"])],
                "next_investigations": []}
        for k, v in override.items():
            args[k] = v(c) if callable(v) else v
        return [call(ia.FINISH, **args)]
    return turn


@pytest.mark.parametrize("turn, text_", [
    (bad(interpretations=[{"statement": "Approach precedes attack.", "evidence_ids": []}]), "not part of the findings/2"),
    (bad(direction="negative"), "not part of the findings/2"),
    (bad(next_steps=["do more"]), "not part of the findings/2"),
    (bad(summary="the effect is large"), "unknown fields ['summary']"),
    (bad(claims=lambda c: [C("descriptive_effect", c["evidence_id"], value="-75.4")]), "unknown fields ['value']"),
    (bad(claims=lambda c: [{"type": "descriptive_effect", "evidence_ids": [c["evidence_id"]], "params": {},
                            "statement": "faster approach"}]), "unknown fields ['statement']"),
    (bad(claims=lambda c: [C("free_text", c["evidence_id"])]), "must match one of the alternatives"),
    (bad(observations=lambda c: [{"evidence_id": c["evidence_id"], "field": "status", "value": "ok"}]),
     "unknown fields ['value']"),
    (bad(claims=[C("descriptive_effect", "ev:compare_windows:" + "f" * 24)]), "not evidence of this run"),
    (bad(primary_evidence_id="ev:compare_windows:" + "e" * 24), "not one of the snapshots"),
    (bad(finding="inconclusive"), "descriptive_effect only under"),
    (bad(claims=lambda c: [C("descriptive_effect", c["evidence_id"]), C("descriptive_effect", c["evidence_id"])]),
     "duplicate claims"),
])
def test_unsupported_finishes_are_refused_and_may_be_revised(env, turn, text_):
    report = run(env, Scripted(*_until_compare(LOOSE), turn, good_after_compare), sufficiency=LOOSE,
                 budget=Budget(max_model_turns=6))
    refused_, accepted = [t for t in finish_steps(report) if t["outcome"] == "refused"], \
        [t for t in finish_steps(report) if t["outcome"] == "accepted"]
    assert len(refused_) == 1 and refused_[0]["refused"] == "final_answer_rejected"
    assert text_ in refused_[0]["message"], refused_[0]["message"]
    assert len(accepted) == 1 and report.finished                                          # the revision is accepted
    assert accepted[0]["arguments"]["claims"] == report.claims
    doc, md = v2.render_v2(report, policy=env[1])         # a refused attempt never blocks or enters the report
    assert doc["status"]["publication"] == "publishable" and refused_[0]["message"] not in md
    # only the accepted proposal is in F
    assert report.fingerprint == ctx.content_fingerprint(ctx.fingerprint_payload(
        evidence=report.evidence, finding=report.finding, primary_evidence_id=report.primary_evidence_id,
        observations=report.observations, claims=report.claims, next_investigations=report.next_investigations))


def test_a_refused_finish_never_becomes_a_report(env):
    report = run(env, Scripted(*_until_compare(LOOSE), bad(direction="negative")), sufficiency=LOOSE,
                 budget=Budget(max_model_turns=4))
    assert not report.finished and report.finding == "incomplete" and report.claims == []
    assert report.stopping_reason == "max_model_turns"
    assert report.fingerprint == ctx.content_fingerprint(ctx.fingerprint_payload(
        evidence=report.evidence, finding="incomplete", primary_evidence_id=None))     # nothing proposed enters F
    doc, _ = v2.render_v2(report, policy=env[1])
    assert doc["status"]["publication"] == "record_only" and doc["claims"] == []


def test_the_legacy_lexical_validator_is_never_used_for_findings_v2(env, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("the legacy prose validator ran in findings/2 mode")
    import live_tracker.investigation_lexical as lx         # M4: the lexical checks live here (legacy/1 only)
    monkeypatch.setattr(lx, "_prose_problems", forbidden)
    monkeypatch.setattr(lx, "prose_problems", forbidden)
    monkeypatch.setattr(ia._Run, "verify_final", forbidden)
    report = run(env, InvestigatorV2(), sufficiency=LOOSE)
    assert report.finished and report.finding == "effect_described"


# ======================================================================
# 3. execution prefix, completed trace, frozen content
# ======================================================================

def frozen_of(report):
    return {"proposal": {k: copy.deepcopy(getattr(report, k)) for k in cl.PROPOSAL_KEYS},
            "evidence": ctx.canonical_bytes(report.evidence), "prefix": ctx.canonical_bytes(report.trace[:-1]),
            "fingerprint": report.fingerprint}


@pytest.fixture
def finished_v2(env):
    return run(env, InvestigatorV2(), sufficiency=LOOSE), env[1]


def test_completed_trace_agrees_with_the_frozen_finish(finished_v2):
    report, policy = finished_v2
    assert ia.finished_v2_problems(frozen_of(report), report.evidence, report.trace, policy, "finished") == []


def test_post_finish_activity_and_changes_are_detected(finished_v2):
    report, policy = finished_v2
    frozen = frozen_of(report)
    check = lambda ev, tr, reason="finished": ia.finished_v2_problems(frozen, ev, tr, policy, reason)  # noqa: E731
    first_call = next(t for t in report.trace if t["kind"] == "tool_call")
    post = report.trace[:-1] + [{**copy.deepcopy(first_call), "step": len(report.trace) - 1}] + report.trace[-1:]
    assert any("frozen execution prefix" in p for p in check(report.evidence, post))
    changed = copy.deepcopy(report.evidence)
    changed[0]["result"]["measurements"]["n_episodes_total"] += 1
    assert any("evidence changed" in p for p in check(changed, report.trace))
    args = copy.deepcopy(report.trace)
    next(t for t in args if t["kind"] == "finish")["arguments"]["claims"].append(C("no_comparison"))
    assert any("arguments are not the report's structured findings" in p for p in check(report.evidence, args))
    assert any("stops with reason 'finished'" in p for p in check(report.evidence, report.trace, "max_model_turns"))
    twice = report.trace + [copy.deepcopy(report.trace[-1])]
    assert check(report.evidence, twice)
    assert ia.finished_v2_problems(None, report.evidence, report.trace, policy, "finished") == ["no finish was frozen"]


def test_an_unverifiable_finish_raises_instead_of_reporting(env, monkeypatch):
    real = cl.validate_findings

    def drifting(proposal, **kw):
        r = real(proposal, **kw)
        return r if kw["mode"] == "prefix" else type(r)(**{**r.__dict__, "fingerprint": "0" * 64})
    monkeypatch.setattr(cl, "validate_findings", drifting)
    with pytest.raises(ia.FinishVerificationError, match="fingerprint differs"):
        run(env, InvestigatorV2(), sufficiency=LOOSE)


def test_tool_calls_after_an_accepted_finish_are_not_executed(env):
    def finish_then_query(m):
        c = last_result(m)
        return [call(ia.FINISH, finding="effect_described", primary_evidence_id=c["evidence_id"], observations=[],
                     claims=[C("descriptive_effect", c["evidence_id"])], next_investigations=[]),
                query(max_episodes=3)]
    report = run(env, Scripted(*_until_compare(LOOSE), finish_then_query), sufficiency=LOOSE)
    assert report.finished and report.usage["tool_calls"] == 3 and len(report.evidence) == 3
    i = next(i for i, t in enumerate(report.trace) if t["kind"] == "finish")
    assert [t["kind"] for t in report.trace[i:]] == ["finish", "stop"]


def test_fingerprint_and_finish_tampering_refuse_at_render(finished_v2):
    report, policy = finished_v2
    data = report.to_json()
    with pytest.raises(ir.ReportIntegrityError, match="fingerprint mismatch"):
        v2.render_v2({**data, "fingerprint": "1" * 64}, policy=policy)
    swapped = copy.deepcopy(data)
    next(t for t in swapped["trace"] if t["kind"] == "finish")["arguments"]["observations"] = []
    e = pytest.raises(ir.ReportIntegrityError, v2.render_v2, swapped, policy=policy)
    assert any("execution: the accepted finish's arguments" in p for p in e.value.problems)
    unfinished = {**copy.deepcopy(data), "stopping_reason": "max_model_turns"}
    pytest.raises(ir.ReportIntegrityError, v2.render_v2, unfinished, policy=policy)


# ======================================================================
# 4. incomplete runs
# ======================================================================

@pytest.mark.parametrize("budget", [Budget(max_model_turns=2), Budget(max_tool_calls=1, max_model_turns=5)])
def test_budget_stopped_runs_are_incomplete_findings(env, budget):
    report = run(env, InvestigatorV2(), sufficiency=LOOSE, budget=budget)
    assert not report.finished and report.finding == "incomplete" and report.stopping_reason != "finished"
    assert (report.primary_evidence_id, report.observations, report.claims, report.next_investigations) == \
        (None, [], [], [])
    expected = cl.incomplete_findings(evidence=report.evidence, trace=report.trace, policy=env[1],
                                      stopping_reason=report.stopping_reason)
    assert expected.accepted and expected.fingerprint == report.fingerprint
    doc, md = v2.render_v2(report, policy=env[1])
    assert doc["status"]["finding"] == "incomplete" and "No finding: the run is incomplete." in md


# ======================================================================
# 5. legacy / findings/2 isolation
# ======================================================================

class Spy(InvestigatorV2):
    def respond(self, system, tools, messages):
        self.sent = (system, tools)
        return ModelReply(content=[text("stop")], stop_reason="end_turn", input_tokens=1, output_tokens=1)


def test_each_format_sends_only_its_own_contract(env):
    spec = next(t for t in ia.TOOL_SPECS_V2 if t["name"] == ia.FINISH)["input_schema"]
    assert set(spec["properties"]) == set(cl.PROPOSAL_KEYS) and spec["additionalProperties"] is False
    assert set(spec["required"]) == set(cl.PROPOSAL_KEYS)
    for t in cl.CLAIM_TYPES:                                          # generated from the registry
        assert t in spec["properties"]["claims"]["description"]
    assert [t for t in ia.TOOL_SPECS_V2 if t["name"] != ia.FINISH] == [t for t in ia.TOOL_SPECS if t["name"] != ia.FINISH]
    s = Spy()
    run(env, s, budget=Budget(max_model_turns=1))
    assert s.sent == (ia.SYSTEM_PROMPT_V2, ia.TOOL_SPECS_V2)
    legacy = ia.investigate(QUESTION, env[0], env[1], s, budget=Budget(max_model_turns=1), findings_format="legacy/1")
    assert s.sent == (ia.SYSTEM_PROMPT, ia.TOOL_SPECS) and isinstance(legacy, ia.InvestigationReport)
    with pytest.raises(ValueError, match="findings_format"):
        ia.investigate(QUESTION, env[0], env[1], s, findings_format="findings/3")


def test_v2_arguments_are_refused_in_legacy_mode_and_vice_versa(env):
    legacy_finish = lambda m: [call(ia.FINISH, finding="inconclusive", observations=[obs(last_result(m), "status")],  # noqa: E731
                                    interpretations=[])]
    r = run(env, Scripted([query()], legacy_finish), budget=Budget(max_model_turns=2))
    assert not r.finished and "not part of the findings/2" in finish_steps(r)[0]["message"]
    v2_finish = lambda m: [finish(finding="insufficient_evidence", observations=[obs(last_result(m), "status")],  # noqa: E731
                                  claims=[C("no_comparison")])]
    old = ia.investigate(QUESTION, env[0], env[1], Scripted([query()], v2_finish), budget=Budget(max_model_turns=2),
                         findings_format="legacy/1")
    assert not old.finished and "unknown fields" in finish_steps(old)[0]["message"]


def test_renderers_never_cross_formats(finished_v2):
    report, policy = finished_v2
    with pytest.raises(ir.ReportIntegrityError):
        ir.build_report(report, policy=policy)
    with pytest.raises(ir.ReportIntegrityError, match="not rendered by the legacy renderer"):
        ir.load_report(report.to_json())
    r = v2.render_payload(report, policy=policy)
    assert (r.format, r.verification, r.quarantine) == ("findings/2", "publishable", None)


# ======================================================================
# 6. adaptive tool selection (autonomy preserved)
# ======================================================================

def tools_used(report):
    return [t["tool"] for t in report.trace if t["kind"] in ("tool_call", "finish")]


def test_the_same_investigator_takes_different_paths_on_different_evidence(tmp_path):
    clean = build(tmp_path)
    dirty = build(tmp_path, contaminated, name="dirty")
    a = ia.investigate(QUESTION, clean[0], clean[1], InvestigatorV2(), sufficiency=LOOSE)
    b = ia.investigate(QUESTION, dirty[0], dirty[1], InvestigatorV2(), sufficiency=LOOSE)
    c = ia.investigate(QUESTION, clean[0], clean[1], InvestigatorV2(), sufficiency=PROVISIONAL_SUFFICIENCY)
    assert tools_used(a) != tools_used(b)                     # contamination -> a repeat without flagged windows
    assert tools_used(b).count("compare_windows") == 2 and any(x["type"] == "sensitivity_excluding_flagged"
                                                                for x in b.claims)
    assert (a.finding, b.finding, c.finding) == ("effect_described", "effect_described", "insufficient_evidence")
    assert all(r.finished for r in (a, b, c))


def test_the_model_still_chooses_behaviours_windows_and_features(env):
    turns = [[call("query_events", source="validation_annotation", behavior="attack", max_episodes=2)],
             lambda m: [call("get_window_features", episodes_from={"evidence_id": last_result(m)["evidence_id"]},
                             pre_event={"length_frames": 10, "gap_frames": 5}, features=["dist_center", "rel_speed"])],
             lambda m: [finish(finding="insufficient_evidence", observations=[obs(last_result(m), "status")],
                               claims=[C("no_comparison"),
                                       C("measurement_context", last_result(m)["evidence_id"],
                                         aspect="mover_unidentified")])]]
    r = run(env, Scripted(*turns), budget=Budget(max_model_turns=4))
    req = r.evidence[1]["request"]
    assert req["pre_event"] == {"length_frames": 10, "gap_frames": 5} and req["features"] == ["dist_center",
                                                                                              "rel_speed"]
    fin = finish_steps(r)[0]
    assert fin["outcome"] == "refused" and "dist_rate is not among" in fin["message"]   # the registry, not a script


# ======================================================================
# 7. Stage 8 offline emulator
# ======================================================================

def test_emulator_paths_follow_the_evidence():
    e = live._EmulatedOpenRouter()
    q = {"tool": "query_events", "evidence_id": "ev:query_events:" + "1" * 24, "status": "ok",
         "measurements": {"n_episodes_total": 0}}
    assert e.decide([]) [1] == "query_events"
    _, name, args = e.decide([q])
    assert name == ia.FINISH and args["claims"] == [C("no_comparison")]                 # path: nothing to compare
    q["measurements"]["n_episodes_total"] = 5
    f1 = {"tool": "get_window_features", "evidence_id": "ev:get_window_features:" + "2" * 24, "status": "ok"}
    _, name, args = e.decide([q])
    assert name == "get_window_features" and args["baseline"]["clearance_frames"] == 150
    assert e.decide([q, f1])[1] == "compare_windows"
    ok = {"tool": "compare_windows", "evidence_id": "ev:compare_windows:" + "3" * 24, "status": "ok"}
    _, name, args = e.decide([q, f1, ok])                                               # path: ok at once
    assert name == ia.FINISH and args["finding"] == "effect_described"
    low = {**ok, "status": "insufficient_evidence"}
    _, name, args = e.decide([q, f1, low])                                              # path: adapt the window
    assert name == "get_window_features" and args["baseline"]["clearance_frames"] == 30
    f2 = {**f1, "evidence_id": "ev:get_window_features:" + "4" * 24}
    low2 = {**low, "evidence_id": "ev:compare_windows:" + "5" * 24}
    _, name, args = e.decide([q, f1, low, f2, low2])
    assert name == ia.FINISH and args["finding"] == "insufficient_evidence" and \
        args["claims"] == [C("sufficiency_not_met", low2["evidence_id"])]


def test_offline_dry_run_on_synthetic_data_ends_with_a_verified_v2_report(tmp_path, monkeypatch):
    pytest.importorskip("anthropic")             # live.dry_run drives the SDK: the investigation extra
    pytest.importorskip("httpx2")
    e = build(tmp_path)
    monkeypatch.setattr(live, "_tools_and_policy", lambda plan: (e[0], e[1]))
    out = live.dry_run()
    assert out["finished"] and out["report_verified"], out["report_problems"]
    assert out["finding"] == "insufficient_evidence"          # 3 synthetic sessions < the provisional minimum of 4
    assert out["tools_used"] == ["query_events", "get_window_features", "compare_windows", "get_window_features",
                                 "compare_windows", ia.FINISH]                         # it adapted once, then stopped
    assert out["format"] == {**out["format"], "findings_format": "findings/2", "registry_version": "claims/2.0",
                             "publication": "publishable"}
    assert out["provider_reported_cost_usd"] == 0.0 and all(r["url"] == "https://openrouter.ai/api/v1/messages"
                                                             for r in out["emulated_requests"])
    legacy = live.dry_run(live.LivePlan(findings_format="legacy/1"))
    assert legacy["format"] == {"findings_format": "legacy/1"} and legacy["report_verified"]
    assert legacy["tools_used"] == ["query_events", ia.FINISH]


def test_stage8_budget_uses_the_format_it_sends():
    assert live._base_tokens("findings/2") != live._base_tokens("legacy/1")
    with pytest.raises(ValueError, match="findings_format"):
        live.LivePlan(findings_format="findings/3")


# ======================================================================
# 8. the provider-facing finish schema is closed (Codex 1 M3 follow-up)
# ======================================================================

SPEC = ia.FINISH_SPEC_V2["input_schema"]
# each differential check runs once per validator: "local" (Stage 6 _schema_problems, always) and "jsonschema"
# (the reference implementation, optional: only those variants skip when it is not installed)
VALIDATORS = ("local", "jsonschema")
EV = {"compare_windows": "ev:compare_windows:" + "a" * 24, "get_window_features": "ev:get_window_features:" + "b" * 24,
      "check_tracking_quality": "ev:check_tracking_quality:" + "c" * 24, "query_events": "ev:query_events:" + "d" * 24}


def proposal(claims=(), next_=()):
    return {"finding": "inconclusive", "primary_evidence_id": None, "observations": [], "claims": list(claims),
            "next_investigations": list(next_)}


def _jsonschema():
    return pytest.importorskip("jsonschema")          # skips only the calling test


def local_problems(p):
    """The local validator's problems for a finish proposal."""
    out = []
    ia._schema_problems(p, SPEC, "arguments", out)
    return out


def jsonschema_accepts(p):
    js = _jsonschema()
    try:
        js.validate(p, SPEC)
        return True
    except js.ValidationError:
        return False


def needs(validator):
    if validator == "jsonschema":
        _jsonschema()


def test_the_schema_itself_is_valid_json_schema():
    _jsonschema().Draft202012Validator.check_schema(SPEC)


def test_the_schema_is_generated_from_the_registry():
    branches = SPEC["properties"]["claims"]["items"]["anyOf"]
    assert [b["properties"]["type"]["const"] for b in branches] == list(cl.CLAIM_TYPES)
    for b in branches:                                 # every params object is closed, generated from cl.PARAMS
        params = b["properties"]["params"]
        t = b["properties"]["type"]["const"]
        assert params["additionalProperties"] is False and set(params["required"]) == set(cl.PARAMS[t])
        assert {k: v["enum"] for k, v in params["properties"].items()} == {k: list(v) for k, v in cl.PARAMS[t].items()}
    nexts = SPEC["properties"]["next_investigations"]["items"]["anyOf"]
    assert [b["properties"]["action"]["const"] for b in nexts] == list(cl.NEXT_ACTIONS)
    for b in nexts:
        assert b["properties"]["params"]["additionalProperties"] is False


@pytest.mark.parametrize("validator", VALIDATORS)
def test_codex_reproduction_is_rejected_by_both_validators(validator):
    needs(validator)
    p = proposal([{"type": "no_comparison", "evidence_ids": [],
                   "params": {"statement": "approach causes attack", "value": -7.82, "direction": "negative"}}])
    if validator == "jsonschema":
        assert jsonschema_accepts(p) is False
    else:
        assert local_problems(p) == ["arguments.claims[0].params: unknown fields ['direction', 'statement', 'value']"]


@pytest.mark.parametrize("claim, text", [
    ({"type": "coverage", "evidence_ids": [], "params": {"aspect": "primary_comparison", "note": "all sessions"}},
     "unknown fields ['note']"),                                                          # unknown parameter
    ({"type": "coverage", "evidence_ids": [], "params": {}}, "missing fields ['aspect']"),  # missing parameter
    ({"type": "coverage", "evidence_ids": [], "params": {"aspect": 3}}, "must be one of"),  # wrong type
    ({"type": "tracking_concern", "evidence_ids": [], "params": {"kind": "severe"}}, "must be one of"),  # bad enum
    ({"type": "descriptive_effect", "evidence_ids": [EV["compare_windows"]], "params": {"value": 1.5}},
     "unknown fields ['value']"),                                                         # params must be {}
    ({"type": "descriptive_effect", "evidence_ids": [EV["compare_windows"]], "params": {"direction": "negative"}},
     "unknown fields ['direction']"),
    ({"type": "descriptive_effect", "evidence_ids": [EV["compare_windows"]], "params": None}, "must be object"),
    ({"type": "descriptive_effect", "evidence_ids": ["the attack comparison"], "params": {}}, "required form"),
    ({"type": "narrative", "evidence_ids": [], "params": {}}, "must match one of the alternatives"),
    ({"type": "no_comparison", "evidence_ids": [], "params": {}, "text": "x"}, "unknown fields ['text']"),
])
@pytest.mark.parametrize("validator", VALIDATORS)
def test_claim_params_are_closed(claim, text, validator):
    needs(validator)
    p = proposal([claim])
    if validator == "jsonschema":
        assert jsonschema_accepts(p) is False
    else:
        local = local_problems(p)
        assert any(text in x for x in local), local


@pytest.mark.parametrize("nxt, text", [
    ({"action": "include_more_sessions", "params": {"sessions": ["x"]}, "because": [0]}, "unknown fields ['sessions']"),
    ({"action": "check_tracking_quality", "params": {}, "because": [0]}, "missing fields ['windows_from']"),
    ({"action": "exclude_flagged_windows", "params": {"quality": "the flagged windows"}, "because": [0]},
     "required form"),                                                                    # prose instead of an id
    ({"action": "exclude_flagged_windows", "params": {"quality": EV["compare_windows"]}, "because": [0]},
     "required form"),                                                                    # the wrong tool's id
    ({"action": "exclude_flagged_windows", "params": {"quality": EV["check_tracking_quality"], "direction": "up"},
      "because": [0]}, "unknown fields ['direction']"),
    ({"action": "check_tracking_quality", "params": {"windows_from": 4}, "because": [0]}, "must be string"),
    ({"action": "include_more_sessions", "params": {}, "because": []}, "at least 1"),
    ({"action": "include_more_sessions", "params": {}, "because": [0], "reason": "more data"},
     "unknown fields ['reason']"),
    ({"action": "vary_window", "params": {}, "because": [0]}, "must match one of the alternatives"),
])
@pytest.mark.parametrize("validator", VALIDATORS)
def test_next_investigation_params_are_closed(nxt, text, validator):
    needs(validator)
    p = proposal([{"type": "no_comparison", "evidence_ids": [], "params": {}}], [nxt])
    if validator == "jsonschema":
        assert jsonschema_accepts(p) is False
    else:
        local = local_problems(p)
        assert any(text in x for x in local), local


def every_valid_claim():
    for t, params in cl.PARAMS.items():
        if not params:
            yield {"type": t, "evidence_ids": [EV["compare_windows"]], "params": {}}
        for k, values in params.items():
            for v in values:
                yield {"type": t, "evidence_ids": [EV["compare_windows"]], "params": {k: v}}


def accepted_by(validator, p):
    return jsonschema_accepts(p) is True if validator == "jsonschema" else local_problems(p) == []


@pytest.mark.parametrize("validator", VALIDATORS)
def test_every_claim_family_and_action_has_an_accepted_shape(validator):
    needs(validator)
    claims = list(every_valid_claim())
    assert {c["type"] for c in claims} == set(cl.CLAIM_TYPES)
    nexts = [{"action": a, "params": {k: EV[ia.NEXT_PARAM_TOOLS[k]] for k in keys}, "because": [0]}
             for a, keys in cl.NEXT_ACTIONS.items()]
    for c in claims:
        assert accepted_by(validator, proposal([c])), c
    for n in nexts:
        assert accepted_by(validator, proposal(claims[:1], [n])), n
    assert accepted_by(validator, proposal([{"type": "sensitivity_excluding_flagged", "params": {}, "evidence_ids": [
        EV["compare_windows"], "ev:compare_windows:" + "e" * 24, EV["check_tracking_quality"]]}]))


def test_schema_rejection_happens_before_m1_and_m1_stays_authoritative(env, monkeypatch):
    calls = []
    real = cl.validate_findings
    monkeypatch.setattr(cl, "validate_findings", lambda *a, **k: calls.append(1) or real(*a, **k))
    codex = lambda m: [finish(finding="insufficient_evidence", observations=[obs(last_result(m), "status")],  # noqa: E731
                              claims=[{"type": "no_comparison", "evidence_ids": [],
                                       "params": {"statement": "x", "value": 1, "direction": "negative"}}])]
    r = run(env, Scripted([query()], codex), budget=Budget(max_model_turns=2))
    assert not r.finished and "unknown fields" in finish_steps(r)[0]["message"]
    assert calls == []                                 # refused by the schema before the registry was consulted
    # schema-valid but evidence-unsupported: still refused, by M1
    unsupported = lambda m: [finish(finding="insufficient_evidence", observations=[obs(last_result(m), "status")],  # noqa: E731
                                    claims=[C("no_comparison"), C("coverage", aspect="primary_comparison")])]
    r2 = run(env, Scripted([query()], unsupported), budget=Budget(max_model_turns=2))
    assert not r2.finished and "there is no primary comparison" in finish_steps(r2)[0]["message"]


@pytest.mark.parametrize("validator", VALIDATORS)
def test_accepted_v2_runs_satisfy_the_provider_schema(tmp_path, validator):
    needs(validator)
    clean, dirty = build(tmp_path), build(tmp_path, contaminated, name="dirty")
    for e, suff in ((clean, LOOSE), (dirty, LOOSE), (clean, PROVISIONAL_SUFFICIENCY)):
        r = ia.investigate(QUESTION, e[0], e[1], InvestigatorV2(), sufficiency=suff)
        assert r.finished and accepted_by(validator, {k: getattr(r, k) for k in cl.PROPOSAL_KEYS})
