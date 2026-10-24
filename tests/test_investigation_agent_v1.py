"""The Stage 9 agent on the approved Contract v1 tools (investigation_agent_v1). New file.

Offline only: SYNTHETIC Contract v1 data (test_investigation_tools_v1.make), deterministic fake models and, for
the OpenRouter path, the real Anthropic SDK over an in-process transport (no network, no key, no cost). The fake
investigator decides each action ONLY from the tool results in the transcript it is sent; production uses the
real model's replies, never a scripted sequence.
"""

import copy
import json
from types import SimpleNamespace

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_agent_v1 as av1
import live_tracker.investigation_tools_v1 as v1
import test_investigation_tools_v1 as T
from live_tracker.investigation_agent import Budget, ModelReply, ModelUnavailable


@pytest.fixture(scope="module")
def data5():
    """5 sessions x 3 attack bouts: compare_windows meets the gate."""
    return T.make()


@pytest.fixture(scope="module")
def data3():
    """3 sessions x 3 bouts: the 4-session gate cannot be met."""
    return T.make(T.SESSIONS[:3])


def run(data, model, question="Does the distance between the mice change before attack?", **kw):
    return av1.investigate_v1(question, data, model, allow_synthetic=True, **kw)


def history(messages):
    """[(tool, input, parsed result)] of the transcript (results the model was actually sent)."""
    calls, out = {}, []
    for m in messages:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                calls[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and b["tool_use_id"] in calls:
                out.append((*calls[b["tool_use_id"]], json.loads(b["content"]), bool(b.get("is_error"))))
    return out


class Investigator:
    """Deterministic fake model that chooses its next action from what it observed (no fixed sequence)."""

    def __init__(self):
        self.n, self.seen = 0, []

    def respond(self, system, tools, messages):
        self.n += 1
        self.seen.append({"tools": [t["name"] for t in tools], "system": system})
        question = messages[0]["content"].lower()
        why, name, args = self.decide(question, [h for h in history(messages) if not h[3]])
        return ModelReply(content=[{"type": "text", "text": why},
                                   {"type": "tool_use", "id": f"tu_{self.n}", "name": name, "input": args}],
                          stop_reason="tool_use", input_tokens=900, output_tokens=150, model="fake-investigator")

    def decide(self, question, done):
        def of(tool):
            return [r for t, a, r, _ in done if t == tool]
        behavior = "mount" if "mount" in question else "attack"
        queries, comps, quals = of("query_events"), of("compare_windows"), of("check_tracking_quality")
        if not queries:
            return f"Find the predicted {behavior} bouts.", "query_events", {"behavior": behavior,
                                                                             "min_duration_s": 1.0}
        q = queries[-1]
        if q["n_bouts"] == 0:
            return "No bout: nothing to compare.", av1.FINISH, {
                "finding": "inconclusive", "observations": [{"evidence_id": q["evidence_id"], "field": "n_bouts"}]}
        if "tracking" in question and not quals:
            b = next(b for b in q["bouts"] if b["start_s"] >= 2.0)
            return "Check tracking before the first bout.", "check_tracking_quality", {
                "session": b["session"], "t0_s": b["start_s"] - 2.0, "t1_s": b["start_s"]}
        if not comps:
            return "Compare 2 s before each bout with a 5 s baseline.", "compare_windows", {
                "query_from": {"evidence_id": q["evidence_id"]}, "pre_s": 2, "baseline_s": 5.0,
                "baseline_gap_s": 1.0, "feature": "dist_center"}
        c = comps[-1]
        if c["status"] == "insufficient_evidence" and len(comps) == 1:
            return "Insufficient at 2 s: try the shorter preselected window.", "compare_windows", {
                "query_from": {"evidence_id": q["evidence_id"]}, "pre_s": 1, "baseline_s": 5.0,
                "baseline_gap_s": 1.0, "feature": "dist_center"}
        if c["status"] == "ok":
            return "The gate is met.", av1.FINISH, {
                "finding": "effect_described", "primary_evidence_id": c["evidence_id"],
                "observations": [{"evidence_id": c["evidence_id"], "field": f}
                                 for f in ("mean_diff", "n_windows", "n_sessions")]}
        return "The evidence is insufficient.", av1.FINISH, {
            "finding": "insufficient_evidence", "primary_evidence_id": c["evidence_id"],
            "observations": [{"evidence_id": c["evidence_id"], "field": f} for f in ("reason", "n_windows")]}


class Scripted:
    """Replies in order; each item is a list of (name, input) calls or a ModelReply / exception."""

    def __init__(self, *turns):
        self.turns, self.n, self.seen = list(turns), 0, []

    def respond(self, system, tools, messages):
        self.seen.append(copy.deepcopy(messages))
        if not self.turns:
            raise ModelUnavailable("script_exhausted")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, ModelReply):
            return turn
        self.n += 1
        return ModelReply(content=[{"type": "tool_use", "id": f"s{self.n}_{i}", "name": n, "input": a}
                                   for i, (n, a) in enumerate(turn)], stop_reason="tool_use",
                          input_tokens=500, output_tokens=50)


def results(report):
    return [(t["tool"], t["outcome"], t.get("refused")) for t in report.trace if t["kind"] in ("tool_call", "finish")]


def last_eid(model, tool):
    return [r["evidence_id"] for t, a, r, err in history(model.seen[-1]) if t == tool and not err][-1]


QUERY = ("query_events", {"behavior": "attack", "min_duration_s": 1.0})


# ======================================================================
# schemas and arguments
# ======================================================================

def test_the_model_sees_only_the_four_contract_tools_and_the_finish():
    assert [t["name"] for t in av1.TOOL_SPECS_V1] == ["query_events", "check_tracking_quality",
                                                      "get_window_features", "compare_windows", av1.FINISH]
    blob = json.dumps(av1.TOOL_SPECS_V1)
    for forbidden in ("min_n_windows", "min_n_sessions", "gate_override_reason", "quality_policy\"", "uncertainty",
                      "\"bouts\"", "query_run_id", "sufficiency"):
        assert forbidden not in blob
    cmp = next(t for t in av1.TOOL_SPECS_V1 if t["name"] == "compare_windows")["input_schema"]
    assert cmp["properties"]["pre_s"]["enum"] == [1, 2, 5] and cmp["additionalProperties"] is False


@pytest.mark.parametrize("args, why", [
    ({"min_n_windows": 1}, "unknown fields"),                                 # the gate is not the model's
    ({"gate_override_reason": "small data"}, "unknown fields"),
    ({"quality_policy": "quality_v1_strict90"}, "unknown fields"),
    ({"overlap_policy": "none"}, "unknown fields"),
    ({"uncertainty": {"method": "bootstrap"}}, "unknown fields"),
    ({"pre_s": 3}, "must be one of"),                                         # not in the grid
    ({"pre_s": True}, "must be one of"),
])
def test_scientific_controls_are_not_the_models(data5, args, why):
    model = Scripted([QUERY], [])
    first = run(data5, model)
    eid = first.evidence[0]["evidence_id"]
    req = {"query_from": {"evidence_id": eid}, "pre_s": 2, "baseline_s": 5.0, "baseline_gap_s": 1.0,
           "feature": "dist_center", **args}
    model = Scripted([QUERY], [("compare_windows", req)])
    report = run(data5, model, budget=Budget(max_model_turns=2))
    refused = [t for t in report.trace if t.get("refused")]
    assert refused and refused[0]["refused"] == "invalid_arguments" and why in refused[0]["message"]
    assert [e["tool"] for e in report.evidence] == ["query_events"]          # nothing was executed


@pytest.mark.parametrize("call, kind", [
    (("query_events", {"behavior": "attack", "min_duration_s": 1.0, "sessions": ["task1/test/mouse999"]}),
     "tool_refused:unknown_session"),                                         # held-out / unknown sessions refused
    (("check_tracking_quality", {"session": "task1/test/mouse999", "t0_s": 0, "t1_s": 1}),
     "tool_refused:unknown_session"),
    (("get_window_features", {"session": T.SESSIONS[0], "t0_s": 5, "t1_s": 4, "cols": ["dist_center"]}),
     "tool_refused:bad_window"),
    (("compare_windows", {"query_from": {"evidence_id": "ev:query_events:000000000000000000000000"}, "pre_s": 2,
                          "baseline_s": 5.0, "baseline_gap_s": 1.0, "feature": "dist_center"}), "unknown_evidence"),
    (("delete_everything", {}), "unknown_tool"),
])
def test_refused_calls_never_become_evidence(data5, call, kind):
    report = run(data5, Scripted([call]), budget=Budget(max_model_turns=1))
    assert report.evidence == [] and [t["refused"] for t in report.trace if t.get("refused")] == [kind]


def test_whole_number_pre_s_given_as_a_float_is_the_grid_value(data5):
    model = Investigator()
    first = run(data5, model, budget=Budget(max_model_turns=1))
    eid = first.evidence[0]["evidence_id"]
    req = {"query_from": {"evidence_id": eid}, "pre_s": 2.0, "baseline_s": 5.0, "baseline_gap_s": 1.0,
           "feature": "dist_center"}
    report = run(data5, Scripted([QUERY], [("compare_windows", req)]), budget=Budget(max_model_turns=2))
    assert [e["tool"] for e in report.evidence] == ["query_events", "compare_windows"]
    assert report.evidence[1]["result"]["pre_s"] == 2.0


# ======================================================================
# model-selected execution
# ======================================================================

def test_the_model_chooses_tools_from_results_and_finishes_with_an_effect(data5):
    model = Investigator()
    report = run(data5, model)
    assert report.finished and report.finding == "effect_described" and report.stopping_reason == "finished"
    assert results(report) == [("query_events", "ok", None), ("compare_windows", "ok", None),
                               (av1.FINISH, "accepted", None)]
    compare = report.evidence[1]
    assert report.primary_evidence_id == compare["evidence_id"]
    assert {o["field"]: o["value"] for o in report.observations} == {
        "mean_diff": compare["result"]["mean_diff"], "n_windows": compare["result"]["n_windows"],
        "n_sessions": compare["result"]["n_sessions"]}                         # read from the result, not the model
    assert model.seen[0]["tools"] == list(av1.TOOL_NAMES_V1)
    assert report.publication == "not_authorized" and report.findings_format == "contract-v1/1"


def test_compare_windows_runs_on_the_complete_verified_bouts_of_the_cited_query(data5):
    report = run(data5, Investigator())
    q, c = report.evidence[0], report.evidence[1]
    assert c["request"]["bouts"] == q["result"]["bouts"] and c["request"]["query_run_id"] == q["result"]["query_run_id"]
    assert c["result"]["n_input"] == q["result"]["n_bouts"]
    assert c["result"]["quality_policy"] == "quality_v1" and c["result"]["overlap_policy"] == "overlap_v1"
    assert c["result"]["evidence_gate"]["overridden"] is False


def test_different_questions_lead_to_different_tool_sequences(data5):
    attack = run(data5, Investigator(), "Does distance change before attack?")
    tracking = run(data5, Investigator(), "Is tracking reliable before attack, and does distance change?")
    mount = run(data5, Investigator(), "Does distance change before mount?")
    seq = {k: [t for t, _, _ in results(r)] for k, r in (("attack", attack), ("tracking", tracking),
                                                          ("mount", mount))}
    assert seq["attack"] == ["query_events", "compare_windows", av1.FINISH]
    assert seq["tracking"] == ["query_events", "check_tracking_quality", "compare_windows", av1.FINISH]
    assert seq["mount"] == ["query_events", av1.FINISH] and mount.finding == "inconclusive"   # no mount bout


def test_the_model_changes_pre_s_for_a_follow_up_and_finishes_with_insufficient_evidence(data3):
    report = run(data3, Investigator())
    comps = [e for e in report.evidence if e["tool"] == "compare_windows"]
    assert [c["request"]["pre_s"] for c in comps] == [2, 1]
    assert all(c["status"] == "insufficient_evidence" for c in comps)
    assert report.finding == "insufficient_evidence" and report.primary_evidence_id == comps[-1]["evidence_id"]
    assert {o["field"] for o in report.observations} == {"reason", "n_windows"}
    assert "mean_diff" not in comps[-1]["result"]


# ======================================================================
# budgets, repeats and follow-ups
# ======================================================================

def cmp(eid, pre, **kw):
    return ("compare_windows", {"query_from": {"evidence_id": eid}, "pre_s": pre, "baseline_s": 5.0,
                                "baseline_gap_s": 1.0, "feature": "dist_center", **kw})


def eid_of_query(data):
    return run(data, Scripted([QUERY]), budget=Budget(max_model_turns=1)).evidence[0]["evidence_id"]


def test_repeated_requests_and_a_third_follow_up_are_refused(data5):
    q = eid_of_query(data5)
    model = Scripted([QUERY], [cmp(q, 2)], [cmp(q, 2)], [cmp(q, 1)], [cmp(q, 5)], [cmp(q, 2)])
    report = run(data5, model, budget=Budget(max_model_turns=6, max_repeated_calls=3,
                                             max_consecutive_rejections=5))
    refused = [t["refused"] for t in report.trace if t.get("refused")]
    assert refused[0] == "repeated_call"                                        # identical request, not executed
    assert refused[-1] == "follow_up_limit"                                     # a 3rd follow-up, never executed
    assert [e["request"]["pre_s"] for e in report.evidence if e["tool"] == "compare_windows"] == [2, 1, 5]


def test_a_follow_up_may_change_only_pre_s(data5):
    q = eid_of_query(data5)
    report = run(data5, Scripted([QUERY], [cmp(q, 2)], [cmp(q, 1, feature="rel_speed")]),
                 budget=Budget(max_model_turns=3))
    assert [t["refused"] for t in report.trace if t.get("refused")] == ["tool_refused:invalid_request"]
    assert "may change only pre_s" in json.dumps(report.trace)


def test_a_fourth_comparison_is_refused_before_the_runner(data5):
    q = eid_of_query(data5)
    report = run(data5, Scripted([QUERY], [cmp(q, 2)], [cmp(q, 1)], [cmp(q, 5)],
                                 [cmp(q, 5, baseline_s=4.0)]), budget=Budget(max_model_turns=5))
    assert report.trace[-2]["refused"] == "follow_up_limit" and "follow-up" in report.trace[-2]["message"]
    assert [a["tool"] for a in report.runner_record["attempts"]].count("compare_windows") == 3


def test_the_tool_call_budget_is_enforced(data5):
    sessions = [("check_tracking_quality", {"session": T.SESSIONS[i % 5], "t0_s": float(i), "t1_s": i + 1.0})
                for i in range(5)]
    model = Scripted(*[[c] for c in sessions])
    report = run(data5, model, budget=Budget(max_model_turns=6, max_tool_calls=3))
    assert report.usage["tool_calls"] == 3 and len(report.evidence) == 3
    assert report.stopping_reason == "max_tool_calls" and report.finding == "incomplete"
    assert report.runner_record["calls_used"] == 3


def test_budget_above_contract_v1_is_impossible():
    with pytest.raises(ValueError):
        Budget(max_tool_calls=13)


# ======================================================================
# the finish: evidence-grounded, structured, verified
# ======================================================================

def finish(**kw):
    return (av1.FINISH, {"finding": "inconclusive", "observations": [], **kw})


@pytest.mark.parametrize("extra", [{"statement": "approach causes attack"}, {"causal": True}, {"p_value": 0.01},
                                   {"confidence_interval": [1, 2]}, {"significant": True},
                                   {"generalizes_to_test_split": True}, {"ground_truth_attacks": True},
                                   {"direction": "negative"}, {"effect_size": 3.0}])
def test_a_finish_with_any_claim_beyond_references_is_refused(data5, extra):
    q = eid_of_query(data5)
    model = Scripted([QUERY], [finish(observations=[{"evidence_id": q, "field": "n_bouts"}], **extra)])
    report = run(data5, model, budget=Budget(max_model_turns=2))
    assert report.finding == "incomplete" and report.trace[-2]["refused"] == "final_answer_rejected"
    assert "unknown fields" in report.trace[-2]["message"]


@pytest.mark.parametrize("make, why", [
    (lambda q, c: finish(finding="effect_described", primary_evidence_id=q,
                         observations=[{"evidence_id": q, "field": "n_bouts"}]), "status 'ok'"),
    (lambda q, c: finish(finding="effect_described", primary_evidence_id=c,
                         observations=[{"evidence_id": c, "field": "n_windows"}]), "observe the primary"),
    (lambda q, c: finish(finding="insufficient_evidence", primary_evidence_id=c,
                         observations=[{"evidence_id": c, "field": "status"}]), "insufficient_evidence needs"),
    (lambda q, c: finish(observations=[{"evidence_id": "ev:compare_windows:ffffffffffffffffffffffff",
                                        "field": "mean_diff"}]), "not a result of this run"),
    (lambda q, c: finish(observations=[{"evidence_id": c, "field": "wording"}]), "no observable field"),
    (lambda q, c: finish(observations=[{"evidence_id": c, "field": "per_session"}]), "not a value"),
    (lambda q, c: finish(observations=[{"evidence_id": c, "field": "mean_diff; 42"}]), "no observable field"),
    (lambda q, c: finish(observations=[]), "1 to 20 observations"),
])
def test_unsupported_findings_are_refused_and_never_reach_the_report(data5, make, why):
    model = Investigator()
    base = run(data5, model, budget=Budget(max_model_turns=2))          # query + compare (ok)
    q, c = base.evidence[0]["evidence_id"], base.evidence[1]["evidence_id"]
    first = Scripted([QUERY], [cmp(q, 2)], [make(q, c)])
    report = run(data5, first, budget=Budget(max_model_turns=3))
    assert report.finding == "incomplete" and not report.observations
    msg = report.trace[-2]["message"]
    assert report.trace[-2]["refused"] == "final_answer_rejected" and why in msg, msg


def test_a_refused_finish_can_be_corrected(data5):
    q = eid_of_query(data5)
    bad = finish(finding="effect_described", primary_evidence_id=q, observations=[{"evidence_id": q,
                                                                                   "field": "n_bouts"}])
    good = finish(observations=[{"evidence_id": q, "field": "n_bouts"}])
    report = run(data5, Scripted([QUERY], [bad], [good]))
    assert report.finished and report.finding == "inconclusive"
    assert [t.get("refused") or t["outcome"] for t in report.trace if t["kind"] == "finish"] == [
        "final_answer_rejected", "accepted"]


def test_finishing_before_any_tool_is_refused(data5):
    report = run(data5, Scripted([finish(observations=[{"evidence_id": "x", "field": "status"}])]),
                 budget=Budget(max_model_turns=1))
    assert report.finding == "incomplete" and "call a tool before finishing" in report.trace[-2]["message"]


# ======================================================================
# outages, incomplete runs, data guards
# ======================================================================

def test_a_model_outage_gives_a_clearly_incomplete_report(data5):
    report = run(data5, Scripted([QUERY], ModelUnavailable("rate_limited")))
    assert report.finding == "incomplete" and not report.finished and report.stopping_reason == "model_unavailable"
    assert len(report.evidence) == 1 and av1.verify_report_v1(report.to_json()) == []
    md = av1.render_markdown_v1(report.to_json())
    assert "stopped before an accepted final answer" in md and "model_unavailable" in md


def test_synthetic_data_is_refused_for_real_runs(data5):
    with pytest.raises(ValueError, match="synthetic data is refused"):
        av1.investigate_v1("q?", data5, Investigator())
    with pytest.raises(TypeError):
        av1.investigate_v1("q?", {"not": "data"}, Investigator())


# ======================================================================
# evidence identity, trace integrity and report reconstruction
# ======================================================================

def test_evidence_ids_are_deterministic_and_bind_request_and_result(data5):
    a, b = run(data5, Investigator()), run(data5, Investigator())
    assert [e["evidence_id"] for e in a.evidence] == [e["evidence_id"] for e in b.evidence]
    assert a.fingerprint != b.fingerprint or a.trace == b.trace           # the fingerprint binds the timed trace
    fixed = [run(data5, Investigator(), clock=lambda: 0.0) for _ in range(2)]
    assert fixed[0].fingerprint == fixed[1].fingerprint                    # same choices, same clock: same fingerprint
    for e in a.evidence:
        assert e["evidence_id"] == v1.evidence_id(e["tool"], e["request"], e["result"])
        assert e["provenance"]["contract_version"] == v1.CONTRACT_VERSION and e["provenance"]["split"] == "synthetic"
        assert e["provenance"]["data_fingerprint"] == data5.fingerprint
    assert v1.evidence_id("compare_windows", a.evidence[1]["request"], {**a.evidence[1]["result"], "mean_diff": 0}) \
        != a.evidence[1]["evidence_id"]


def test_a_saved_report_is_reconstructed_deterministically(data5):
    report = run(data5, Investigator())
    doc = json.loads(json.dumps(report.to_json()))                     # what would be saved
    assert av1.verify_report_v1(doc) == []
    assert av1.render_markdown_v1(doc) == av1.render_markdown_v1(json.loads(json.dumps(doc)))
    md = av1.render_markdown_v1(doc)
    assert "Publication:** not_authorized" in md and "classifier predictions" in md and "Association only" in md


def tampered(doc, change):
    d = copy.deepcopy(doc)
    change(d)
    return av1.verify_report_v1(d)


@pytest.mark.parametrize("change, why", [
    (lambda d: d["observations"][0].update(value=999.0), "observed values"),
    (lambda d: d["evidence"][1]["result"].update(mean_diff=999.0), "does not match its content"),
    (lambda d: d["evidence"].pop(0), "executed calls are not the evidence"),
    (lambda d: d["trace"].pop(1), "out of order"),
    (lambda d: d.update(finding="inconclusive"), "fingerprint"),
    (lambda d: d.update(fingerprint="fpv1:" + "0" * 64), "fingerprint"),
    (lambda d: d.update(publication="authorized"), "not_authorized"),
    (lambda d: d.update(primary_evidence_id=d["evidence"][0]["evidence_id"]), "status 'ok'"),
])
def test_a_tampered_saved_report_does_not_verify_or_render(data5, change, why):
    doc = json.loads(json.dumps(run(data5, Investigator()).to_json()))
    problems = tampered(doc, change)
    assert problems and any(why in p for p in problems), problems
    bad = copy.deepcopy(doc)
    change(bad)
    with pytest.raises(av1.ReportVerificationError):
        av1.render_markdown_v1(bad)


def test_the_runner_record_holds_every_attempt_including_refused_ones(data5):
    q = eid_of_query(data5)
    report = run(data5, Scripted([QUERY], [cmp(q, 2)], [cmp(q, 1, feature="rel_speed")]),
                 budget=Budget(max_model_turns=3))
    rec = report.runner_record
    assert [a["tool"] for a in rec["attempts"]] == ["query_events", "compare_windows", "compare_windows"]
    assert rec["attempts"][-1]["status"] == "error" and rec["calls_used"] == 3


# ======================================================================
# OpenRouter compatibility (offline): the v1 schemas through the existing model abstraction
# ======================================================================

class FakeClient:
    """An anthropic.Anthropic-like client whose replies come from the adaptive investigator."""

    def __init__(self):
        self.calls, self.brain, self.messages, self.beta = [], Investigator(), self, self

    def with_options(self, **kw):
        return self

    def create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        reply = self.brain.respond(kw["system"][0]["text"], kw["tools"], kw["messages"])
        usage = SimpleNamespace(input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                                cache_creation_input_tokens=0, cache_read_input_tokens=0)
        return SimpleNamespace(content=reply.content, stop_reason=reply.stop_reason, usage=usage, model="fake")


def test_the_v1_schemas_go_through_the_existing_claude_adapter(data5):
    client = FakeClient()
    model = ia.ClaudeInvestigationModel(client, model="anthropic/claude-test", fallbacks=False, max_retries=0)
    report = run(data5, model)
    assert report.finding == "effect_described"
    first = client.calls[0]
    assert first["tools"] == av1.TOOL_SPECS_V1 and first["tool_choice"] == {"type": "auto"}
    assert first["system"][0]["text"] == av1.SYSTEM_PROMPT_V1


def test_the_guarded_openrouter_path_runs_the_v1_agent_offline(data5, monkeypatch):
    """The real Anthropic SDK and the Stage 8 guarded OpenRouter client (cost ledger) over an in-process
    transport: the Contract v1 schemas serialize, the provider pin is sent, every request is costed."""
    httpx2 = pytest.importorskip("httpx2")
    pytest.importorskip("anthropic")
    import live_tracker.investigation_live as live
    brain, seen = Investigator(), []

    def endpoint(request):
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "tools": [t["name"] for t in body["tools"]],
                     "provider": body.get("provider"), "auth": "authorization" in {k.lower() for k in request.headers}})
        reply = brain.respond(body["system"][0]["text"] if isinstance(body["system"], list) else body["system"],
                              body["tools"], body["messages"])
        return httpx2.Response(200, json={"id": f"msg_{len(seen)}", "type": "message", "role": "assistant",
                                           "model": body["model"], "content": reply.content,
                                           "stop_reason": reply.stop_reason, "stop_sequence": None,
                                           "usage": {"input_tokens": 900, "output_tokens": 150, "cost": 0.0}})
    for var in live.UNSAFE_AMBIENT:
        monkeypatch.delenv(var, raising=False)
    plan = live.LivePlan(model="anthropic/claude-test")
    model, ledger = live.build_model(plan, env={"OPENROUTER_API_KEY": "offline-placeholder-not-a-key"},
                                     http_client=httpx2.Client(transport=httpx2.MockTransport(endpoint)))
    report = run(data5, model)
    assert report.finding == "effect_described" and report.stopping_reason == "finished"
    assert all(s["tools"] == list(av1.TOOL_NAMES_V1) and s["auth"] for s in seen)
    assert seen[0]["url"].endswith("/v1/messages") and seen[0]["provider"] == {"order": ["anthropic"],
                                                                                "allow_fallbacks": False}
    assert ledger.complete and len(ledger.entries) == len(seen)
    assert "offline-placeholder-not-a-key" not in json.dumps(report.to_json())


# ======================================================================
# Codex review: saved-report integrity, completion semantics, follow-up budget
# ======================================================================

def saved(report):
    return json.loads(json.dumps(report.to_json()))


def resign(doc):
    """What a careful forger does after editing: recompute both fingerprints (nobody needs a secret for that)."""
    d = copy.deepcopy(doc)
    if d["finished"]:
        d["finish_fingerprint"] = av1._fingerprint(d["question"], d["finding"], d["primary_evidence_id"],
                                                   d["observations"], d["evidence"], d["trace"][:-1])
    d["fingerprint"] = av1._content_fingerprint(d)
    return d


@pytest.fixture(scope="module")
def finished_doc(data5):
    return saved(run(data5, Investigator()))


@pytest.fixture(scope="module")
def incomplete_doc(data5):
    return saved(run(data5, Scripted([QUERY]), budget=Budget(max_model_turns=1)))


def test_genuine_finished_and_incomplete_reports_verify_and_render(finished_doc, incomplete_doc):
    for doc in (finished_doc, incomplete_doc):
        assert av1.verify_report_v1(doc) == [] and av1.verify_report_v1(resign(doc)) == []
        md = av1.render_markdown_v1(doc)
        assert all(d in md for d in av1.DISCLOSURES)
    assert finished_doc["finish_fingerprint"] and incomplete_doc["finish_fingerprint"] is None


def forged(doc, change, sign=True):
    d = copy.deepcopy(doc)
    change(d)
    return resign(d) if sign else d


def rejected(doc, why):
    problems = av1.verify_report_v1(doc)
    assert problems and any(why in p for p in problems), problems
    with pytest.raises(av1.ReportVerificationError):
        av1.render_markdown_v1(doc)


@pytest.mark.parametrize("sign", [False, True])
def test_codex_injected_disclosure_is_rejected_and_never_rendered(finished_doc, sign):
    claim = "Approach causes attacks; this finding is statistically significant."
    doc = forged(finished_doc, lambda d: d.update(disclosures=[claim]), sign)
    rejected(doc, "canonical disclosures")
    doc = forged(finished_doc, lambda d: d["disclosures"].append(claim), sign)
    rejected(doc, "canonical disclosures")


@pytest.mark.parametrize("change, why", [
    (lambda d: d["data"].update(data_source="ground_truth_labels"), "another data source"),          # Codex
    (lambda d: d["data"].update(derived_from="annotation_indicators"), "another data source"),
    (lambda d: d["data"].update(fingerprint="0" * 64), "not run on the recorded data"),
    (lambda d: d["data"].update(sessions=d["data"]["sessions"][1:]), "outside the recorded data"),
    (lambda d: d["evidence"][0]["provenance"].update(split="validation"), "another provenance"),
    (lambda d: d["evidence"][0]["provenance"].update(investigation_id="other"), "another provenance"),
])
def test_tampered_data_provenance_is_rejected(finished_doc, change, why):
    rejected(forged(finished_doc, change), why)


@pytest.mark.parametrize("change, why", [
    (lambda d: d["runner_record"]["attempts"].pop(), "runner record does not match"),                  # Codex
    (lambda d: d["runner_record"].update(attempts=[]), "runner record does not match"),
    (lambda d: d["runner_record"].update(calls_used=1), "runner record does not match"),
    (lambda d: d["runner_record"]["attempts"][1]["result"].update(mean_diff=0.0), "not the evidence it produced"),
    (lambda d: d["runner_record"]["attempts"][1].update(tool="get_window_features"), "not the trace's call"),
    (lambda d: d["runner_record"].update(max_tool_calls=12) if d["budget"]["max_tool_calls"] != 12 else
     d["budget"].update(max_tool_calls=11), "runner record does not match"),
])
def test_deleted_or_altered_runner_attempts_are_rejected(finished_doc, change, why):
    rejected(forged(finished_doc, change), why)


@pytest.mark.parametrize("change, why", [
    (lambda d: d["usage"].update({k: 0 for k in ("model_turns", "tool_calls", "input_tokens", "output_tokens")}),
     "usage contradicts"),                                                                             # Codex
    (lambda d: d["usage"].update(tool_calls=1), "usage contradicts"),
    (lambda d: d["usage"].update(repeated_calls=2), "usage contradicts"),
    (lambda d: d["usage"].update(elapsed_s=-1.0), "elapsed time"),
    (lambda d: d["usage"].pop("cost_usd"), "usage is malformed"),
    (lambda d: d["budget"].update(max_model_turns=1), "exceeds the recorded budget"),
    (lambda d: d["budget"].update(max_tool_calls=99), "budget is invalid"),
])
def test_falsified_usage_counts_and_budget_are_rejected(finished_doc, change, why):
    rejected(forged(finished_doc, change), why)


def test_unknown_report_fields_are_rejected(finished_doc):
    rejected(forged(finished_doc, lambda d: d.update(interpretation="approach causes attack")), "fields")
    doc = copy.deepcopy(finished_doc)
    doc.pop("budget")
    assert av1.verify_report_v1(doc)


@pytest.mark.parametrize("sign", [False, True])
def test_codex_complete_to_timeout_forgery_is_rejected(finished_doc, sign):
    def to_timeout(d):
        d["stopping_reason"] = "timeout"
        d["trace"][-1]["stopping_reason"] = "timeout"
    rejected(forged(finished_doc, to_timeout, sign), "the report's stop is not the trace's stop")

    def to_timeout_unfinished(d):
        to_timeout(d)
        d["finished"] = False
    rejected(forged(finished_doc, to_timeout_unfinished, sign), "must be 'incomplete'")


def test_codex_incomplete_to_finished_forgery_is_rejected_even_with_new_fingerprints(incomplete_doc):
    q = incomplete_doc["evidence"][0]

    def finish_it(d):
        d.update(finished=True, stopping_reason="finished", finding="inconclusive",
                 observations=[{"evidence_id": q["evidence_id"], "field": "n_bouts", "value": q["result"]["n_bouts"]}])
        d["trace"][-1]["stopping_reason"] = "finished"
    doc = forged(incomplete_doc, finish_it)                     # both fingerprints recomputed
    rejected(doc, "exactly one accepted finish")


def test_a_relabelled_budget_stop_cannot_become_complete(data5):
    doc = saved(run(data5, Scripted([QUERY], [QUERY]), budget=Budget(max_model_turns=2, max_repeated_calls=5)))
    assert doc["stopping_reason"] in ("max_model_turns", "rejected_actions", "repeated_actions")

    def relabel(d):
        d.update(finished=True, stopping_reason="finished")
        d["trace"][-1]["stopping_reason"] = "finished"
    rejected(forged(doc, relabel), "exactly one accepted finish")


def test_finish_arguments_inconsistent_with_the_accepted_finish_are_rejected(finished_doc):
    def fewer(d):
        d["observations"] = d["observations"][:1]
    rejected(forged(finished_doc, fewer), "not the accepted finish's arguments")

    def other_args(d):
        d["trace"][-2]["arguments"]["observations"] = d["trace"][-2]["arguments"]["observations"][:1]
    rejected(forged(finished_doc, other_args), "not the accepted finish's arguments")

    def other_finding(d):
        d["finding"] = "inconclusive"
    rejected(forged(finished_doc, other_finding), "not the accepted finish's arguments")


def test_extra_tool_execution_after_the_accepted_finish_is_rejected(finished_doc):
    def extra(d):                                    # a careful forger keeps steps and times in order
        call = copy.deepcopy(next(t for t in d["trace"] if t["kind"] == "tool_call"))
        stop = d["trace"].pop()
        call["elapsed_s"] = stop["elapsed_s"]
        d["trace"] += [call, stop]
        for n, t in enumerate(d["trace"]):
            t["step"] = n
    doc = forged(finished_doc, extra)
    problems = av1.verify_report_v1(doc)
    assert any("no model turn asked" in p for p in problems), problems
    with pytest.raises(av1.ReportVerificationError):
        av1.render_markdown_v1(doc)

    def extra_with_turn(d):                          # ... and even invents the model turn that asked for it
        call = copy.deepcopy(next(t for t in d["trace"] if t["kind"] == "tool_call"))
        stop = d["trace"].pop()
        turn = {"kind": "model_turn", "elapsed_s": stop["elapsed_s"], "model": "x", "stop_reason": "tool_use",
                "text": "", "tool_calls": [{"id": call["tool_use_id"], "name": call["tool"]}], "input_tokens": 0,
                "output_tokens": 0}
        call["elapsed_s"] = stop["elapsed_s"]
        d["trace"] += [turn, call, stop]
        for n, t in enumerate(d["trace"]):
            t["step"] = n
    rejected(forged(finished_doc, extra_with_turn), "immediately before its stop")


def test_a_finish_event_without_a_model_turn_asking_for_it_is_rejected(finished_doc):
    def orphan(d):
        d["trace"][-2]["tool_use_id"] = "tu_never_asked"
    rejected(forged(finished_doc, orphan), "no model turn asked")


# ---- the caller's follow-up budget: the stricter of it and Contract v1's two ----

def test_codex_zero_follow_up_budget_is_enforced(data5):
    q = eid_of_query(data5)
    report = run(data5, Scripted([QUERY], [cmp(q, 2)], [cmp(q, 1)]),
                 budget=Budget(max_compare_follow_ups=0, max_model_turns=3))
    assert [e["request"]["pre_s"] for e in report.evidence if e["tool"] == "compare_windows"] == [2]
    assert report.trace[-2]["refused"] == "follow_up_limit"
    assert av1.verify_report_v1(saved(report)) == []


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_the_effective_follow_up_limit_is_the_stricter_one(data5, limit):
    q = eid_of_query(data5)
    report = run(data5, Scripted([QUERY], [cmp(q, 2)], [cmp(q, 1)], [cmp(q, 5)], [cmp(q, 5, baseline_s=4.0)]),
                 budget=Budget(max_compare_follow_ups=limit, max_model_turns=5, max_consecutive_rejections=5))
    executed = [e["request"]["pre_s"] for e in report.evidence if e["tool"] == "compare_windows"]
    assert executed == [2, 1, 5][:1 + limit]
    refused = [t["refused"] for t in report.trace if t.get("refused")]
    assert refused == ["follow_up_limit"] * (3 - limit)                       # traced, never evidence
    assert report.usage["tool_calls"] == 2 + limit                             # refused before the runner
    assert av1.verify_report_v1(saved(report)) == []


def test_a_budget_cannot_raise_the_contract_follow_up_limit():
    with pytest.raises(ValueError, match="0-2"):
        Budget(max_compare_follow_ups=5)
    assert av1.follow_up_limit(Budget(max_compare_follow_ups=2)) == 2


# ======================================================================
# Codex re-review: saved arguments are bound to execution, the saved finish to the live finish schema
# ======================================================================
# Every forgery below recomputes both fingerprints (resign): the verifier must refuse it on its content alone.

def step(d, tool, n=0):
    """The n-th accepted (or refused) trace event for one tool."""
    return [t for t in d["trace"] if t["kind"] in ("tool_call", "finish") and t["tool"] == tool][n]


@pytest.fixture(scope="module")
def query_eid(data5):
    return eid_of_query(data5)


@pytest.fixture(scope="module")
def tracking_doc(data5):
    """query -> check_tracking_quality -> compare -> effect_described."""
    return saved(run(data5, Investigator(), "Is tracking reliable before attack, and does distance change?"))


@pytest.fixture(scope="module")
def features_doc(data5, query_eid):
    """query -> get_window_features -> compare -> a second query -> inconclusive."""
    feats = ("get_window_features", {"session": T.SESSIONS[0], "t0_s": 1.0, "t1_s": 3.0,
                                     "cols": ["dist_center", "rel_speed"]})
    second = ("query_events", {"behavior": "attack", "min_duration_s": 0.5})
    model = Scripted([QUERY], [feats], [cmp(query_eid, 2)], [second],
                     [finish(observations=[{"evidence_id": query_eid, "field": "n_bouts"}])])
    doc = saved(run(data5, model))
    assert doc["finished"] and [e["tool"] for e in doc["evidence"]] == [
        "query_events", "get_window_features", "compare_windows", "query_events"]
    return doc


@pytest.fixture(scope="module")
def refused_doc(data5, query_eid):
    """A finished run with every kind of refusal: loop, schema, unknown tool, repeat, runner, finish."""
    q = query_eid
    bad_finish = finish(finding="effect_described", primary_evidence_id=q,
                        observations=[{"evidence_id": q, "field": "n_bouts"}])
    model = Scripted([QUERY], [cmp(q, 2, min_n_windows=1)], [("delete_everything", {})], [cmp(q, 2)],
                     [cmp(q, 2)], [cmp(q, 1, feature="rel_speed")], [bad_finish],
                     [finish(observations=[{"evidence_id": q, "field": "n_bouts"}])])
    doc = saved(run(data5, model, budget=Budget(max_model_turns=8, max_repeated_calls=3,
                                                max_consecutive_rejections=5)))
    assert doc["finished"] and [t.get("refused") for t in doc["trace"] if t.get("refused")] == [
        "invalid_arguments", "unknown_tool", "repeated_call", "tool_refused:invalid_request",
        "final_answer_rejected"]
    return doc


def test_genuine_reports_of_every_tool_and_refusal_verify_and_render(finished_doc, incomplete_doc, tracking_doc,
                                                                     features_doc, refused_doc):
    for doc in (finished_doc, incomplete_doc, tracking_doc, features_doc, refused_doc):
        assert av1.verify_report_v1(doc) == [] and av1.verify_report_v1(resign(doc)) == []
        assert "## Disclosures" in av1.render_markdown_v1(doc)
    assert incomplete_doc["finding"] == "incomplete" and not incomplete_doc["finished"]


# ---- Blocker 1: saved tool arguments ----

def test_codex_altered_compare_arguments_are_rejected(finished_doc):
    def codex(d):
        call = next(t for t in d["trace"] if t["kind"] == "tool_call" and t["tool"] == "compare_windows")
        call["arguments"].update(pre_s=5, query_from={"evidence_id": "unknown"})
    rejected(forged(finished_doc, codex), "refused by the live checks (unknown_evidence:")


@pytest.mark.parametrize("ref", ["unknown", "ev:query_events:000000000000000000000000", None, 7])
def test_an_unknown_query_reference_is_rejected(finished_doc, ref):
    doc = forged(finished_doc, lambda d: step(d, "compare_windows")["arguments"].update(query_from={"evidence_id": ref}))
    rejected(doc, "refused by the live checks")


def test_a_query_reference_to_later_evidence_is_rejected(features_doc):
    later = features_doc["evidence"][3]
    assert later["tool"] == "query_events" and later["result"]["bouts"]
    doc = forged(features_doc, lambda d: step(d, "compare_windows")["arguments"].update(
        query_from={"evidence_id": later["evidence_id"]}))
    rejected(doc, "refused by the live checks (unknown_evidence:")


def test_a_query_reference_to_a_non_query_result_is_rejected(features_doc):
    feats = features_doc["evidence"][1]["evidence_id"]
    doc = forged(features_doc, lambda d: step(d, "compare_windows")["arguments"].update(
        query_from={"evidence_id": feats}))
    rejected(doc, "refused by the live checks (unknown_evidence:")


@pytest.mark.parametrize("change", [{"pre_s": 5}, {"pre_s": 1}, {"pre_s": 5.0}, {"baseline_s": 4.0},
                                    {"baseline_gap_s": 2.0}, {"feature": "rel_speed"}])
def test_altered_comparison_parameters_are_rejected(finished_doc, change):
    doc = forged(finished_doc, lambda d: step(d, "compare_windows")["arguments"].update(change))
    rejected(doc, "not the request its arguments produce")


@pytest.mark.parametrize("change", [{"pre_s": 3}, {"pre_s": True}, {"pre_s": "2"}, {"baseline_s": -1.0},
                                    {"feature": "annotation"}, {"query_from": "ev:query_events:x"}])
def test_comparison_arguments_outside_the_schema_are_rejected(finished_doc, change):
    doc = forged(finished_doc, lambda d: step(d, "compare_windows")["arguments"].update(change))
    rejected(doc, "refused by the live checks (invalid_arguments:")


@pytest.mark.parametrize("change", [{"sessions": [T.SESSIONS[0]]}, {"sessions": T.SESSIONS[:4]}, {"behavior": "mount"},
                                    {"min_duration_s": 2.0}, {"threshold": 0.6}])
def test_altered_query_arguments_are_rejected(finished_doc, change):
    doc = forged(finished_doc, lambda d: step(d, "query_events")["arguments"].update(change))
    rejected(doc, "not the request its arguments produce")


def test_query_arguments_with_unknown_sessions_or_behaviors_are_rejected(finished_doc):
    for change in ({"behavior": "ground_truth_attack"}, {"threshold": 1.5}, {"min_duration_s": -1}):
        doc = forged(finished_doc, lambda d: step(d, "query_events")["arguments"].update(change))
        rejected(doc, "refused by the live checks (invalid_arguments:")


@pytest.mark.parametrize("tool, extra", [
    ("compare_windows", {"min_n_windows": 1}), ("compare_windows", {"min_n_sessions": 1}),
    ("compare_windows", {"gate_override_reason": "small data"}), ("compare_windows", {"quality_policy": "none"}),
    ("compare_windows", {"overlap_policy": "none"}), ("compare_windows", {"uncertainty": {"method": "bootstrap"}}),
    ("compare_windows", {"bouts": []}), ("query_events", {"quality_policy": "none"}),
])
def test_added_policy_or_gate_override_fields_are_rejected(finished_doc, tool, extra):
    doc = forged(finished_doc, lambda d: step(d, tool)["arguments"].update(extra))
    rejected(doc, "refused by the live checks (invalid_arguments:")


@pytest.mark.parametrize("tool, change", [
    ("check_tracking_quality", {"t0_s": 0.0}), ("check_tracking_quality", {"session": T.SESSIONS[4]}),
])
def test_altered_tracking_quality_arguments_are_rejected(tracking_doc, tool, change):
    doc = forged(tracking_doc, lambda d: step(d, tool)["arguments"].update(change))
    rejected(doc, "not the request its arguments produce")


@pytest.mark.parametrize("change, why", [
    ({"t1_s": 4.0}, "not the request"), ({"session": T.SESSIONS[1]}, "not the request"),
    ({"cols": ["dist_center"]}, "not the request"), ({"cols": []}, "(invalid_arguments:"),
    ({"cols": ["annotation"]}, "(invalid_arguments:"),
])
def test_altered_feature_window_arguments_are_rejected(features_doc, change, why):
    doc = forged(features_doc, lambda d: step(d, "get_window_features")["arguments"].update(change))
    rejected(doc, why)


def renamed(doc, n, change):
    """Alter evidence n's request (and its runner attempt) consistently, give it its recomputed evidence id and
    rename that id everywhere: only the model's recorded arguments still say what was asked."""
    d = copy.deepcopy(doc)
    e = d["evidence"][n]
    change(e["request"])
    old, new = e["evidence_id"], v1.evidence_id(e["tool"], e["request"], e["result"])
    call = next(t for t in d["trace"] if t.get("evidence_id") == old)["call"]
    d["runner_record"]["attempts"][call - 1]["request"] = copy.deepcopy(e["request"])
    return resign(json.loads(json.dumps(d).replace(old, new)))


@pytest.mark.parametrize("n, change", [
    (1, lambda r: r.update(t0_s=0.5)),                                                    # get_window_features
    (2, lambda r: r.update(pre_s=5)),                                                     # compare_windows
    (2, lambda r: r.update(bouts=r["bouts"][1:])),                                        # a bout dropped
    (0, lambda r: r.update(session=r["session"][:2])),                                    # query_events
])
def test_evidence_and_runner_attempt_that_disagree_with_the_recorded_request_are_rejected(features_doc, n, change):
    rejected(renamed(features_doc, n, change), "not the request its arguments produce")


def test_a_runner_attempt_that_disagrees_with_the_recorded_request_is_rejected(finished_doc):
    doc = forged(finished_doc, lambda d: d["runner_record"]["attempts"][1]["request"].update(pre_s=5))
    rejected(doc, "runner attempt is not the request its arguments produce")


# ---- rejected calls stay rejected ----

def test_a_refused_call_cannot_become_admissible(refused_doc):
    def fix_schema(d):                               # the refused call's arguments edited into valid ones
        step(d, "compare_windows")["arguments"].pop("min_n_windows")
    rejected(forged(refused_doc, fix_schema), "recorded as refused before execution")

    def relabel(d):                                  # the refusal's kind changed
        step(d, "compare_windows")["refused"] = "follow_up_limit"
    rejected(forged(refused_doc, relabel), "does not record that refusal")

    def reword(d):                                   # ... or its message
        step(d, "delete_everything")["message"] = "accepted"
    rejected(forged(refused_doc, reword), "does not record that refusal")


def test_a_repeat_or_a_refused_finish_cannot_be_rewritten(refused_doc):
    def unrepeat(d):
        t = next(t for t in d["trace"] if t.get("refused") == "repeated_call")
        t["refused"] = "invalid_arguments"
        d["usage"]["repeated_calls"] = 0
    rejected(forged(refused_doc, unrepeat), "does not record that refusal")

    def finish_args(d):                              # the refused finish now looks like the accepted one
        bad = step(d, av1.FINISH, 0)
        bad["arguments"] = copy.deepcopy(step(d, av1.FINISH, 1)["arguments"])
    rejected(forged(refused_doc, finish_args), "recorded as refused")


def test_a_refused_call_cannot_be_promoted_into_evidence(refused_doc):
    def promote(d):                                  # the refused schema-violating call claims the evidence
        bad, good = step(d, "compare_windows", 0), step(d, "compare_windows", 1)
        for t in (bad, good):
            t.pop("refused", None), t.pop("message", None), t.pop("evidence_id", None), t.pop("call", None)
        e = next(e for e in d["evidence"] if e["tool"] == "compare_windows")
        bad.update(outcome=e["status"], evidence_id=e["evidence_id"], call=2)
        good.update(outcome="refused", refused="invalid_arguments", message="x")
    rejected(forged(refused_doc, promote), "refused by the live checks (invalid_arguments:")


# ---- Blocker 2: the saved finish ----

def test_codex_causal_proof_finding_is_rejected_and_never_rendered(finished_doc):
    def codex(d):
        d["finding"] = "causal_proof"
        d["trace"][-2]["arguments"]["finding"] = "causal_proof"
    doc = forged(finished_doc, codex)
    rejected(doc, "unknown finding")
    rejected(doc, "refused by the live checks (final_answer_rejected:")


def test_an_extra_causal_finish_field_is_rejected(finished_doc):
    doc = forged(finished_doc, lambda d: d["trace"][-2]["arguments"].update(causal=True))
    rejected(doc, "refused by the live checks (final_answer_rejected:")
    problems = av1.verify_report_v1(doc)
    assert any("unknown fields ['causal']" in p for p in problems), problems


@pytest.mark.parametrize("change", [
    lambda a: a.pop("finding"), lambda a: a.pop("observations"), lambda a: a.update(statement="it causes it"),
    lambda a: a.update(p_value=0.01), lambda a: a.update(finding=7), lambda a: a.update(finding=None),
    lambda a: a.update(primary_evidence_id=123), lambda a: a.update(observations="mean_diff"),
    lambda a: a["observations"][0].update(field=5), lambda a: a["observations"][0].update(value=1.0),
    lambda a: a["observations"][0].pop("field"), lambda a: a["observations"].append("n_windows"),
])
def test_missing_extra_and_mistyped_finish_fields_are_rejected(finished_doc, change):
    rejected(forged(finished_doc, lambda d: change(d["trace"][-2]["arguments"])), "final_answer_rejected")


def test_the_saved_finish_must_cite_evidence_of_the_run(finished_doc):
    ghost = "ev:compare_windows:ffffffffffffffffffffffff"

    def cite(d):
        d["trace"][-2]["arguments"]["observations"].append({"evidence_id": ghost, "field": "mean_diff"})
        d["observations"].append({"evidence_id": ghost, "field": "mean_diff", "value": 0.0})
    rejected(forged(finished_doc, cite), "not a result of this run")


def test_live_and_saved_finish_checks_are_one_function(data5, finished_doc):
    """The live run refuses exactly what the verifier refuses (both go through _arguments / _Ledger.admit)."""
    c = finished_doc["evidence"][1]["evidence_id"]
    args = copy.deepcopy(finished_doc["trace"][-2]["arguments"])
    model = Scripted([QUERY], [cmp(eid_of_query(data5), 2)], [(av1.FINISH, {**args, "causal": True})],
                     [(av1.FINISH, {**args, "finding": "causal_proof"})])
    live = saved(run(data5, model, budget=Budget(max_model_turns=4)))
    refusals = [t for t in live["trace"] if t.get("refused")]
    assert [t["refused"] for t in refusals] == ["final_answer_rejected"] * 2 and live["finding"] == "incomplete"
    assert av1.verify_report_v1(live) == []                     # the refusals replay as refusals
    assert c == live["evidence"][1]["evidence_id"]


# ======================================================================
# Codex re-review 2: the runner's follow-up rule and the evidence gate hold for saved reports
# ======================================================================

def coordinated(doc, n, request=None, result=None):
    """Codex's coordinated tamper of evidence n: its request and/or result changed, then its evidence id, its
    runner attempt (request, result, status, hash) and its trace outcome recomputed, and the id renamed
    everywhere. The caller edits the model's arguments / finish and re-signs."""
    d = copy.deepcopy(doc)
    e = d["evidence"][n]
    (request or (lambda r: None))(e["request"])
    (result or (lambda r: None))(e["result"])
    e["status"] = e["result"]["status"]
    old, new = e["evidence_id"], v1.evidence_id(e["tool"], e["request"], e["result"])
    t = next(t for t in d["trace"] if t.get("evidence_id") == old)
    t["outcome"] = e["status"]
    d["runner_record"]["attempts"][t["call"] - 1].update(
        request=copy.deepcopy(e["request"]), result=copy.deepcopy(e["result"]), status=e["status"],
        reason=e["result"].get("reason"), result_sha256=v1._sha(json.dumps(e["result"], sort_keys=True, default=str)))
    return json.loads(json.dumps(d).replace(old, new)), new


def comparisons(doc):
    return [n for n, e in enumerate(doc["evidence"]) if e["tool"] == "compare_windows"]


@pytest.fixture(scope="module")
def follow_up_doc(data3):
    """3 sessions: compare at pre_s 2, a follow-up at pre_s 1, both insufficient_evidence."""
    doc = saved(run(data3, Investigator()))
    assert [doc["evidence"][n]["request"]["pre_s"] for n in comparisons(doc)] == [2, 1]
    return doc


@pytest.fixture(scope="module")
def three_follow_ups_doc(data3):
    q = eid_of_query(data3)
    doc = saved(run(data3, Scripted([QUERY], [cmp(q, 2)], [cmp(q, 1)], [cmp(q, 5)],
                                    [finish(observations=[{"evidence_id": q, "field": "n_bouts"}])])))
    assert doc["finished"] and [doc["evidence"][n]["request"]["pre_s"] for n in comparisons(doc)] == [2, 1, 5]
    return doc


@pytest.fixture(scope="module")
def refused_follow_up_doc(data5):
    q = eid_of_query(data5)
    doc = saved(run(data5, Scripted([QUERY], [cmp(q, 2)], [cmp(q, 1, feature="rel_speed")]),
                    budget=Budget(max_model_turns=3)))
    assert [t["refused"] for t in doc["trace"] if t.get("refused")] == ["tool_refused:invalid_request"]
    return doc


def test_legitimate_follow_ups_and_insufficient_evidence_verify_and_render(follow_up_doc, three_follow_ups_doc,
                                                                           refused_follow_up_doc, finished_doc,
                                                                           incomplete_doc):
    for doc in (follow_up_doc, three_follow_ups_doc, refused_follow_up_doc, finished_doc, incomplete_doc):
        assert av1.verify_report_v1(doc) == [] and av1.verify_report_v1(resign(doc)) == []
        assert "## Disclosures" in av1.render_markdown_v1(doc)
    assert follow_up_doc["finding"] == "insufficient_evidence"
    assert all(follow_up_doc["evidence"][n]["status"] == "insufficient_evidence" for n in comparisons(follow_up_doc))
    for doc in (follow_up_doc, finished_doc):                     # genuine results satisfy every invariant
        for n in comparisons(doc):
            e = doc["evidence"][n]
            assert v1.comparison_problems(e["request"], e["result"]) == []


# ---- P1: the runner's fixed-follow-up rule ----

@pytest.mark.parametrize("field, value", [("feature", "rel_speed"), ("baseline_s", 4.0), ("baseline_gap_s", 2.0)])
def test_codex_follow_up_with_a_changed_fixed_parameter_is_rejected(follow_up_doc, field, value):
    n = comparisons(follow_up_doc)[1]
    doc, _ = coordinated(follow_up_doc, n, request=lambda r: r.update({field: value}),
                         result=lambda r: r.update({field: value}))
    step(doc, "compare_windows", 1)["arguments"][field] = value
    rejected(resign(doc), f"follow-up may change only pre_s (changed: ['{field}'])")


def test_codex_follow_up_script_is_rejected(follow_up_doc):
    """Codex's /tmp/live_tracker_phase3a_followup_review.py, step for step."""
    n = comparisons(follow_up_doc)[1]
    doc, _ = coordinated(follow_up_doc, n, request=lambda r: r.update(feature="rel_speed"),
                         result=lambda r: r.update(feature="rel_speed"))
    step(doc, "compare_windows", 1)["arguments"]["feature"] = "rel_speed"
    doc = resign(doc)
    rejected(doc, "the runner's follow-up rule refuses this comparison")
    live = v1.InvestigationRunner(T.make(T.SESSIONS[:3]), "independent-audit")
    first, follow = (doc["evidence"][m]["request"] for m in comparisons(doc))
    assert live.call("compare_windows", first)["status"] != "error"
    assert live.call("compare_windows", follow)["code"] == "invalid_request"    # the saved verdict is the live one


def test_a_follow_up_to_an_already_analysed_pre_s_is_rejected(three_follow_ups_doc):
    n = comparisons(three_follow_ups_doc)[2]
    doc, _ = coordinated(three_follow_ups_doc, n, request=lambda r: r.update(pre_s=1),
                         result=lambda r: r.update(pre_s=1.0))
    step(doc, "compare_windows", 2)["arguments"]["pre_s"] = 1
    rejected(resign(doc), "refused by the live checks (repeated_call:")


def test_a_refused_follow_up_stays_refused(refused_follow_up_doc):
    def reword(d):                                  # the runner's refusal message altered
        d["runner_record"]["attempts"][-1]["result"]["message"] = "accepted"
    rejected(forged(refused_follow_up_doc, reword), "does not record that refusal")

    def relabel(d):
        step(d, "compare_windows", 1)["refused"] = "tool_refused:bad_window"
    rejected(forged(refused_follow_up_doc, relabel), "follow-up rule refuses this comparison")


# ---- P1: the evidence gate ----

def effect_finish(doc, eid, value):
    doc["finding"], doc["primary_evidence_id"] = "effect_described", eid
    doc["trace"][-2]["arguments"].update(finding="effect_described", primary_evidence_id=eid,
                                         observations=[{"evidence_id": eid, "field": "mean_diff"}])
    doc["observations"] = [{"evidence_id": eid, "field": "mean_diff", "value": value}]
    return resign(doc)


def test_codex_gate_forgery_is_rejected(follow_up_doc):
    """Codex's /tmp/live_tracker_phase3a_gate_review.py: 9 windows in 3 sessions relabelled 'ok', mean_diff 123."""
    n = comparisons(follow_up_doc)[-1]
    r = follow_up_doc["evidence"][n]["result"]
    assert (r["n_windows"], r["n_sessions"]) == (9, 3)
    doc, eid = coordinated(follow_up_doc, n, result=lambda r: r.update(status="ok", mean_diff=123.0))
    doc = effect_finish(doc, eid, 123.0)
    rejected(doc, "status 'ok' must have exactly its fields")
    rejected(doc, "effect_described needs primary_evidence_id")              # it cannot support the finding


def as_ok(r, diff=123.0):
    """A complete, self-consistent 'ok' result made from an insufficient one (counts unchanged)."""
    for k in ("reason", "n_available", "n_required", "failed_gate"):
        r.pop(k)
    sessions = sorted({b["session"] for b in r.pop("_bouts")})[:r["n_sessions"]]
    rows = [{"session": s, "n_bouts": r["n_used"] if i == 0 else 0, "paired_diff": diff}
            for i, s in enumerate(sessions)]
    r.update(status="ok", pre_mean=1.0, baseline_mean=0.0, mean_diff=diff, per_session=rows,
             mean_diff_definition=v1.MEAN_DIFF_DEFINITION, wording=v1.WORDING)


def forge_ok(doc, n, also=lambda r: None):
    bouts = doc["evidence"][n]["request"]["bouts"]

    def change(r):
        r["_bouts"] = bouts
        as_ok(r)
        also(r)
    d, eid = coordinated(doc, n, result=change)
    return effect_finish(d, eid, d["evidence"][n]["result"]["mean_diff"])


def test_a_complete_ok_forgery_below_the_gate_is_rejected(follow_up_doc):
    rejected(forge_ok(follow_up_doc, comparisons(follow_up_doc)[-1]), "contradicts the evidence gate for 9 windows "
                                                                      "in 3 sessions")


def test_inflated_counts_cannot_meet_the_gate(follow_up_doc):
    def inflate(r):
        r.update(n_used=10, n_bouts=10, n_windows=10, n_baseline_windows=10, n_sessions=4)
        r["per_session"] = [{"session": s, "n_bouts": k, "paired_diff": 123.0}
                            for s, k in zip(T.SESSIONS[:4], (4, 2, 2, 2))]
    rejected(forge_ok(follow_up_doc, comparisons(follow_up_doc)[-1], inflate), "counts contradict")


@pytest.mark.parametrize("change, why", [
    (lambda r: r.update(mean_diff=r["mean_diff"] + 1.0), "median of its per-session"),
    (lambda r: r["per_session"].pop(), "median of its per-session"),
    (lambda r: r.update(evidence_gate={**r["evidence_gate"], "min_n_windows": 1, "overridden": True}), "echo"),
    (lambda r: r.update(quality_policy="none"), "echo"),
    (lambda r: r.update(overlap_policy="none"), "echo"),
    (lambda r: r.update(n_sessions=3), "contradicts the evidence gate"),
    (lambda r: r.update(n_excluded=r["n_excluded"] + 1), "counts contradict"),
    (lambda r: r.update(causal=True), "exactly its fields"),
])
def test_an_ok_comparison_that_contradicts_itself_is_rejected(finished_doc, change, why):
    n = comparisons(finished_doc)[0]
    doc, eid = coordinated(finished_doc, n, result=change)
    for o in doc["observations"]:                                   # keep the finish's values consistent
        o["value"] = doc["evidence"][n]["result"].get(o["field"])
    rejected(resign(doc), why)


@pytest.mark.parametrize("change, why", [
    (lambda r: r.update(failed_gate="min_n_sessions", n_available=3, n_required=4), "not the gate's decision"),
    (lambda r: r.update(reason="too_few_usable_windows"), "not the gate's decision"),
    (lambda r: r.update(n_required=1), "not the gate's decision"),
    (lambda r: r.update(n_windows=10, n_used=10, n_bouts=10, n_baseline_windows=10, n_sessions=4),
     "counts contradict"),
])
def test_insufficient_evidence_that_contradicts_the_gate_is_rejected(follow_up_doc, change, why):
    n = comparisons(follow_up_doc)[-1]
    doc, eid = coordinated(follow_up_doc, n, result=change)
    for o in doc["observations"]:
        o["value"] = doc["evidence"][n]["result"].get(o["field"])
    rejected(resign(doc), why)


def test_an_ok_comparison_relabelled_insufficient_is_rejected(finished_doc):
    n = comparisons(finished_doc)[0]

    def demote(r):
        for k in ("pre_mean", "baseline_mean", "mean_diff", "mean_diff_definition", "per_session", "wording"):
            r.pop(k)
        r.update(status="insufficient_evidence", reason="too_few_usable_windows", n_available=r["n_sessions"],
                 n_required=4, failed_gate="min_n_sessions")
    doc, eid = coordinated(finished_doc, n, result=demote)
    doc["finding"], doc["observations"] = "insufficient_evidence", [
        {"evidence_id": eid, "field": "reason", "value": "too_few_usable_windows"}]
    doc["trace"][-2]["arguments"].update(finding="insufficient_evidence",
                                         observations=[{"evidence_id": eid, "field": "reason"}])
    rejected(resign(doc), "contradicts the evidence gate")


# ======================================================================
# Codex re-review 3: per-session rows are the sessions of the requested bouts
# ======================================================================

def ok_forgery(doc, n, change):
    """Coordinated change of an 'ok' comparison's result, with the finish's observed values kept consistent."""
    d, _ = coordinated(doc, n, result=change)
    r = d["evidence"][n]["result"]
    for o in d["observations"]:
        o["value"] = r.get(o["field"])
    return resign(d)


@pytest.fixture(scope="module")
def excluded_doc(data5, query_eid):
    """A legitimate 'ok' comparison over 5 sessions with 5 documented exclusions (2 used bouts per session)."""
    model = Scripted([QUERY], [cmp(query_eid, 2, baseline_s=9.0)],
                     [finish(observations=[{"evidence_id": query_eid, "field": "n_bouts"}])])
    doc = saved(run(data5, model))
    r = doc["evidence"][1]["result"]
    assert r["status"] == "ok" and r["n_excluded"] == 5 and [p["n_bouts"] for p in r["per_session"]] == [2] * 5
    return doc


def test_legitimate_multi_session_comparisons_verify(finished_doc, excluded_doc):
    for doc in (finished_doc, excluded_doc):
        assert av1.verify_report_v1(doc) == [] and "## Disclosures" in av1.render_markdown_v1(doc)
        e = doc["evidence"][comparisons(doc)[0]]
        assert v1.comparison_problems(e["request"], e["result"]) == []
        assert len(e["result"]["per_session"]) == e["result"]["n_sessions"] == 5


def test_codex_foreign_session_rows_are_rejected(finished_doc):
    def foreign(r):
        for i, row in enumerate(r["per_session"]):
            row["session"] = f"foreign_{i}"
    rejected(ok_forgery(finished_doc, comparisons(finished_doc)[0], foreign),
             "not the sessions and counts of the request's bouts")


def test_codex_negative_count_with_preserved_sum_is_rejected(finished_doc):
    def skew(r):
        rows = r["per_session"]
        assert [rows[0]["n_bouts"], rows[1]["n_bouts"]] == [3, 3]
        rows[0]["n_bouts"], rows[1]["n_bouts"] = -1, 7
    rejected(ok_forgery(finished_doc, comparisons(finished_doc)[0], skew), "n_bouts >= 1 (an integer)")


def test_a_shifted_count_with_preserved_sum_is_rejected(finished_doc):
    def shift(r):                                    # positive counts, right total, wrong sessions
        r["per_session"][0]["n_bouts"], r["per_session"][1]["n_bouts"] = 2, 4
    rejected(ok_forgery(finished_doc, comparisons(finished_doc)[0], shift), "not the sessions and counts")


@pytest.mark.parametrize("count", [0, True, 3.0, "3", None, [3]])
def test_per_session_counts_must_be_positive_integers(finished_doc, count):
    rejected(ok_forgery(finished_doc, comparisons(finished_doc)[0],
                        lambda r: r["per_session"][0].update(n_bouts=count)), "n_bouts >= 1 (an integer)")


@pytest.mark.parametrize("change, why", [
    (lambda r: r["per_session"][0].pop("n_bouts"), "per_session rows must be"),
    (lambda r: r["per_session"][0].update(note="x"), "per_session rows must be"),
    (lambda r: r["per_session"][0].update(paired_diff="big"), "per_session rows must be"),
    (lambda r: r["per_session"].__setitem__(0, "row"), "per_session rows must be"),
    (lambda r: r.update(per_session={}), "per_session rows must be"),
    (lambda r: r["per_session"][1].update(session=r["per_session"][0]["session"]), "not the sessions"),   # duplicate
    (lambda r: r["per_session"].reverse(), "not the sessions"),                                         # order
    (lambda r: r["per_session"].pop(), "not the sessions"),                                             # missing
    (lambda r: r["per_session"].append({"session": T.SESSIONS[0], "n_bouts": 1, "paired_diff": 0.0}),
     "not the sessions"),                                                                                # extra
    (lambda r: r["per_session"][0].update(session="task1/test/mouse999"), "not the sessions"),           # held out
])
def test_missing_duplicated_or_foreign_session_rows_are_rejected(finished_doc, change, why):
    rejected(ok_forgery(finished_doc, comparisons(finished_doc)[0], change), why)


@pytest.mark.parametrize("change, why", [
    # an exclusion dropped and its bout counted as used in that session
    (lambda r: (r["excluded"].pop(0), r.update(n_used=11, n_bouts=11, n_windows=11, n_baseline_windows=11,
                                               n_excluded=4)), "not the sessions and counts"),
    # an exclusion moved to a session the bout is not in
    (lambda r: r["excluded"][0].update(session=T.SESSIONS[4] if r["excluded"][0]["session"] != T.SESSIONS[4]
                                       else T.SESSIONS[0]), "excluded windows are not request bouts"),
    (lambda r: r["excluded"][0].update(start_s=-1.0), "excluded windows are not request bouts"),
    (lambda r: r["excluded"][0].update(why="looked_wrong"), "excluded windows are not request bouts"),
    (lambda r: r["excluded"].reverse(), "excluded windows are not request bouts"),
    # the used bouts moved between sessions (total kept): the rows no longer match the exclusions
    (lambda r: [row.update(n_bouts=k) for row, k in zip(r["per_session"], (3, 1, 2, 2, 2))],
     "not the sessions and counts"),
])
def test_session_coverage_inconsistent_with_the_exclusions_is_rejected(excluded_doc, change, why):
    rejected(ok_forgery(excluded_doc, comparisons(excluded_doc)[0], change), why)


def test_insufficient_comparisons_must_cover_the_requested_sessions(follow_up_doc):
    n = comparisons(follow_up_doc)[-1]
    rejected(ok_forgery(follow_up_doc, n, lambda r: r.update(n_sessions=2)), "counts contradict")
