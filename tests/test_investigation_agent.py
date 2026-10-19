"""Tests for the experimental autonomous investigation loop (new file).

No test calls the Anthropic API: models are deterministic fakes (or the real
SDK over an in-process mock transport) and sockets are blocked. Synthetic
parts reuse the Stage 4 SYNTHETIC fixtures (published through the real Stage
1 extraction code); the real-data tests read the verified validation cache
and are skipped when it is not present.
"""

import ast
import json
import socket
from pathlib import Path

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_contracts as ic
import live_tracker.investigation_events as ev
from live_tracker.investigation_agent import Budget, ModelReply, ModelUnavailable, investigate
from live_tracker.investigation_contracts import PROVISIONAL_SUFFICIENCY, SufficiencyCriteria
from live_tracker.investigation_data import INDEX_NAME
from live_tracker.investigation_events import CalMS21InvestigationTools
from test_investigation_windows import REAL_CACHE, REAL_MANIFEST, TEST, TRAIN, VAL_A, VAL_B, build, ep

QUESTION = "Does approach tend to precede attack?"
LOOSE = SufficiencyCriteria(min_sessions=1, min_windows_per_group=1, min_usable_fraction=0.9)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv(ia.API_KEY_ENV, raising=False)
    monkeypatch.delenv(ia.MODEL_ENV, raising=False)


@pytest.fixture
def env(tmp_path):
    return build(tmp_path)


def contaminated(sid, kp, sc):
    if sid == VAL_A:
        sc[68, 0, 3] = 0.01        # one failed frame in the smoother memory of window [80, 100)


# ---- fake models ----

def history(messages):
    """[(tool, input, parsed tool result)] in order, from the transcript only."""
    calls, out = {}, []
    for m in messages:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                calls[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and b["tool_use_id"] in calls:
                out.append((*calls[b["tool_use_id"]], json.loads(b["content"])))
    return out


def windows_of(result):
    """Window items of a result as shown to the model (a shortened list ends
    with an {"_omitted_items": n} marker)."""
    return [w for w in result["measurements"]["windows"] if "window_id" in w]


class Investigator:
    """Deterministic fake model. It decides its next action ONLY from the tool
    results in the transcript it is given (no other input, no hidden state)."""

    def __init__(self, length=20, clearance=10):
        self.length, self.clearance, self.n = length, clearance, 0

    def respond(self, system, tools, messages):
        self.n += 1
        why, name, args = self.decide([x for x in history(messages) if "evidence_id" in x[2]])
        return ModelReply(content=[{"type": "text", "text": why},
                                   {"type": "tool_use", "id": f"tu_{self.n}", "name": name, "input": args}],
                          stop_reason="tool_use", input_tokens=1000, output_tokens=200, model="fake-investigator")

    def decide(self, done):
        def last(tool):
            return next((x[2] for x in reversed(done) if x[0] == tool), None)
        q, f = last("query_events"), last("get_window_features")
        quals = [x[2] for x in done if x[0] == "check_tracking_quality"]
        comps = [x[2] for x in done if x[0] == "compare_windows"]
        if q is None:
            return "Find the attack episodes first.", "query_events", {"source": "validation_annotation",
                                                                       "behavior": "attack"}
        if q["measurements"]["n_episodes_total"] == 0:
            return "No episodes.", ia.FINISH, {"finding": "inconclusive", "observations": [
                obs(q, "measurements.n_episodes_total")]}
        if f is None:
            return "Measure dist_rate before each attack and in baselines.", "get_window_features", {
                "episodes_from": {"evidence_id": q["evidence_id"]},
                "pre_event": {"length_frames": self.length, "gap_frames": 0},
                "baseline": {"length_frames": self.length, "clearance_frames": self.clearance,
                             "max_windows_per_session": 500}, "features": ["dist_rate"]}
        if not quals:
            return "Check the tracking quality of the pre-event windows.", "check_tracking_quality", {
                "windows_from": {"evidence_id": f["evidence_id"], "status": "included", "kind": "pre_event"},
                "history_frames": [1]}
        concerns = [w["window_id"] for w in windows_of(quals[0])
                    if w["smoothing_contamination"]["status"] == "detected" or w["verdict"] != "usable_with_caveats"]
        if not comps:
            return "Compare pre-event windows with baselines, session by session.", "compare_windows", {
                "feature": "dist_rate", "windows_from": {"evidence_id": f["evidence_id"]}}
        main = comps[0]
        if concerns and len(comps) == 1:
            clean = [w["window_id"] for w in windows_of(quals[0]) if w["window_id"] not in concerns]
            if clean:
                return ("Tracking concerns in some pre-event windows: repeat the comparison without them.",
                        "compare_windows", {"feature": "dist_rate", "pre_window_ids": clean,
                                            "windows_from": {"evidence_id": f["evidence_id"], "kind": "baseline"}})
        m = main["measurements"]
        observations = [obs(main, f) for f in ("status", "measurements.n_sessions_paired",
                                               "measurements.n_windows_pre", "measurements.n_windows_baseline")]
        interpretations = []
        if len(comps) > 1:
            s = comps[1]
            observations += [obs(s, "status"), obs(s, "measurements.n_windows_pre")]
            interpretations.append({"statement": "The comparison was repeated without the pre-event windows flagged "
                                                 "by check_tracking_quality.",
                                    "evidence_ids": [s["evidence_id"], quals[0]["evidence_id"]]})
        if main["status"] == "ok":
            v = m["effect"]["value"]
            observations.append(obs(main, "measurements.effect.value"))
            return "Enough evidence for a descriptive answer.", ia.FINISH, {
                "finding": "effect_described", "primary_evidence_id": main["evidence_id"],
                "direction": "negative" if v < 0 else "positive" if v > 0 else "zero", "observations": observations,
                "interpretations": interpretations + [
                    {"statement": "In these sessions the body centres approach each other faster before attacks "
                                  "than in baselines; investigation behavior and session timing are not controlled.",
                     "evidence_ids": [main["evidence_id"]]}],
                "additional_limitations": ["Only the locally cached validation sessions were searched."]}
        excluded_pre = [w for w in windows_of(f) if w["kind"] == "pre_event" and w["status"] == "excluded"]
        if excluded_pre and len(quals) == 1:
            return ("Insufficient: check whether tracking explains the excluded pre-event windows.",
                    "check_tracking_quality", {"windows_from": {"evidence_id": f["evidence_id"], "status": "excluded",
                                                                "kind": "pre_event"}, "history_frames": [1]})
        observations.append(obs(main, "measurements.sufficiency.reasons_not_met"))
        if len(quals) > 1:
            observations.append(obs(quals[1], "status"))
            interpretations.append({"statement": "check_tracking_quality was run on the excluded pre-event windows.",
                                    "evidence_ids": [quals[1]["evidence_id"]]})
        return "The evidence is insufficient.", ia.FINISH, {
            "finding": "insufficient_evidence", "primary_evidence_id": main["evidence_id"], "direction": None,
            "observations": observations, "interpretations": interpretations + [
                {"statement": "Too few paired sessions to describe an effect.", "evidence_ids": [main["evidence_id"]]}],
            "next_steps": ["Extract the remaining validation sessions that contain attacks."]}


class Scripted:
    """Fake model replaying turns; a turn is a list of blocks, a ModelReply, an
    exception, or a callable(messages) returning one of those."""

    def __init__(self, *turns, tokens=(10, 5)):
        self.turns, self.tokens, self.seen = list(turns), tokens, []

    def respond(self, system, tools, messages):
        self.seen.append(messages)
        turn = self.turns.pop(0) if self.turns else [text("out of script")]
        if callable(turn):
            turn = turn(messages)
        if isinstance(turn, BaseException):
            raise turn
        if not isinstance(turn, list):
            return turn
        stop = "tool_use" if any(b.get("type") == "tool_use" for b in turn) else "end_turn"
        return ModelReply(content=turn, stop_reason=stop, input_tokens=self.tokens[0], output_tokens=self.tokens[1])


_ids = iter(range(10**6))


def call(name, **args):
    return {"type": "tool_use", "id": f"call_{next(_ids)}", "name": name, "input": args}


def text(t):
    return {"type": "text", "text": t}


def obs(result, field):
    """An observation: a reference to a field of a result (the loop reads the value)."""
    return {"evidence_id": result["evidence_id"], "field": field}


def query(**kw):
    return call("query_events", source="validation_annotation", behavior="attack", **kw)


def last_result(messages):
    return history(messages)[-1][2]


def tools_used(report):
    return [t["tool"] for t in report.trace if t["kind"] in ("tool_call", "finish")]


def counting(env, monkeypatch):
    """Requests that reached a tool implementation (after contract validation)."""
    calls = []
    for tool in ic.TOOLS:
        real = getattr(CalMS21InvestigationTools, tool)

        def spy(self, request, _tool=tool, _real=real):
            calls.append((_tool, request))
            return _real(self, request)
        monkeypatch.setattr(CalMS21InvestigationTools, tool, spy)
    return calls


# ---- autonomy: the next action depends on what was observed ----

def test_successful_multistep_investigation(env):
    report = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    assert report.stopping_reason == "finished" and report.finding == "effect_described"
    assert tools_used(report) == ["query_events", "get_window_features", "check_tracking_quality",
                                  "compare_windows", ia.FINISH]
    p = report.primary_evidence
    assert p["tool"] == "compare_windows" and p["status"] == "ok" and p["n_sessions_paired"] == 2
    assert report.direction == ("negative" if p["effect"]["value"] < 0 else "positive")
    assert p["sufficiency"]["criteria"] == LOOSE.to_json()           # injected by the loop, not the model
    assert {e["evidence_id"] for e in report.evidence} >= {p["evidence_id"]}
    assert all(o["kind"] == "observation" for o in report.observations)
    assert all(i["kind"] == "interpretation" for i in report.interpretations)
    tool_lims = [l["text"] for l in report.limitations if l["source"] == "tool"]
    assert any("Investigation confound" in t for t in tool_lims)
    assert any("No p-value, significance or confidence interval" in t for t in tool_lims)
    assert {"source": "model", "text": "Only the locally cached validation sessions were searched."} in report.limitations


def test_insufficient_evidence_changes_the_next_action(env):
    """Same model, same data, stricter criteria: the compare result is
    insufficient, so the model investigates the excluded windows instead of
    answering, then reports insufficient evidence."""
    loose = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    strict = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=PROVISIONAL_SUFFICIENCY)
    assert tools_used(strict) == ["query_events", "get_window_features", "check_tracking_quality",
                                  "compare_windows", "check_tracking_quality", ia.FINISH]
    assert tools_used(loose)[4] == ia.FINISH and tools_used(strict)[4] == "check_tracking_quality"
    follow = [t for t in strict.trace if t["kind"] == "tool_call"][4]
    assert follow["arguments"]["windows_from"]["status"] == "excluded"
    assert follow["request"]["window_ids"] == ["win:pre:20:0:" + ep(VAL_A, 130)]
    assert strict.finding == "insufficient_evidence" and strict.direction is None
    assert strict.primary_evidence["effect"]["value"] is None


def test_tracking_quality_concern_alters_the_investigation(tmp_path):
    """A tracking failure in the smoother memory of one pre-event window makes
    the model repeat the comparison without that window."""
    clean = investigate(QUESTION, *build(tmp_path, name="clean")[:2], Investigator(), sufficiency=LOOSE)
    env = build(tmp_path, contaminated, name="dirty")
    dirty = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    assert tools_used(clean).count("compare_windows") == 1
    assert tools_used(dirty) == ["query_events", "get_window_features", "check_tracking_quality",
                                 "compare_windows", "compare_windows", ia.FINISH]
    q = next(t for t in dirty.trace if t["kind"] == "tool_call" and t["tool"] == "check_tracking_quality")
    flagged = "win:pre:20:0:" + ep(VAL_A, 100)
    second = [t for t in dirty.trace if t["kind"] == "tool_call" and t["tool"] == "compare_windows"][1]
    assert flagged in q["request"]["window_ids"] and flagged not in second["request"]["pre_window_ids"]
    assert second["request"]["baseline_window_ids"]                      # baselines kept via windows_from
    assert any("flagged by check_tracking_quality" in i["statement"] for i in dirty.interpretations)


def test_the_model_only_sees_validated_results(env):
    model = Scripted([query()], [text("done")])
    report = investigate(QUESTION, env[0], env[1], model, budget=Budget(max_model_turns=2))
    shown = last_result(model.seen[1])
    (call_entry,) = [t for t in report.trace if t["kind"] == "tool_call"]
    ic.validate_result("query_events", call_entry["request"], shown, env[1])      # exactly a validated result
    assert shown["evidence_id"] == call_entry["evidence_id"] == report.evidence[0]["evidence_id"]


# ---- invalid model output ----

@pytest.mark.parametrize("turn, problem", [
    (ModelReply(content="not blocks", stop_reason="end_turn"), "content must be a list"),
    (ModelReply(content=[{"text": "no type"}], stop_reason="end_turn"), "typed blocks"),
    (ModelReply(content=[{"type": "tool_use", "name": "query_events", "input": {}}], stop_reason="tool_use"),
     "id and a name"),
    ("not a reply", "not a ModelReply"),
])
def test_malformed_replies_are_rejected_and_bounded(env, turn, problem):
    model = Scripted(turn, turn, turn, turn)
    unmetered = Budget(max_consecutive_rejections=3, max_input_tokens=None, max_output_tokens=None)
    report = investigate(QUESTION, env[0], env[1], model, budget=unmetered)
    assert report.stopping_reason == "rejected_actions" and report.finding == "incomplete"
    bad = [t for t in report.trace if t["kind"] == "model_turn"]
    assert len(bad) == 3 and all(t["outcome"] == "malformed" and problem in t["problem"] for t in bad)
    assert report.evidence == [] and report.usage["tool_calls"] == 0


# ---- usage of malformed replies is accounted (Codex 1, budget accounting) ----

def malformed(tokens_in=100, tokens_out=100):
    return ModelReply(content="not blocks", stop_reason="end_turn", input_tokens=tokens_in, output_tokens=tokens_out)


@pytest.mark.parametrize("budget, reason", [
    (Budget(max_output_tokens=1), "token_budget"),
    (Budget(max_input_tokens=50), "token_budget"),
    (Budget(max_cost_usd=0.0001, price_per_mtok=(1.0, 1.0)), "cost_budget"),
])
def test_a_malformed_reply_over_budget_is_not_retried(env, monkeypatch, budget, reason):
    calls = counting(env, monkeypatch)
    model = Scripted(malformed(), [call(ia.FINISH, finding="inconclusive", observations=[])])
    report = investigate(QUESTION, env[0], env[1], model, budget=budget)
    assert report.stopping_reason == reason and not report.finished and report.finding == "incomplete"
    assert len(model.seen) == 1 and calls == []                               # no retry, no finish
    assert (report.usage["input_tokens"], report.usage["output_tokens"]) == (100, 100)
    if reason == "cost_budget":
        assert report.usage["cost_usd"] == pytest.approx(0.0002)
    assert [t["outcome"] for t in report.trace if t["kind"] == "model_turn"] == ["over_budget"]


def test_repeated_malformed_replies_accumulate_usage_until_the_budget_stops_them(env):
    model = Scripted(*[malformed(10, 40)] * 6, [call(ia.FINISH, finding="inconclusive", observations=[])])
    report = investigate(QUESTION, env[0], env[1], model,
                         budget=Budget(max_output_tokens=100, max_consecutive_rejections=10))
    # 40, 80 are within the limit (two malformed retries); 120 > 100 stops before any further request
    assert report.stopping_reason == "token_budget" and len(model.seen) == 3 and not report.finished
    assert (report.usage["input_tokens"], report.usage["output_tokens"], report.usage["model_turns"]) == (30, 120, 3)
    outcomes = [t["outcome"] for t in report.trace if t["kind"] == "model_turn"]
    assert outcomes == ["malformed", "malformed", "over_budget"]
    assert [t["output_tokens"] for t in report.trace if t["kind"] == "model_turn"] == [40, 40, 40]


def test_a_malformed_reply_past_the_deadline_is_not_retried(env):
    clock = Clock()
    model = Scripted(at(clock, 10.0, malformed(1, 1)), [call(ia.FINISH, finding="inconclusive", observations=[])])
    report = investigate(QUESTION, env[0], env[1], model, budget=Budget(timeout_s=1), clock=clock)
    assert report.stopping_reason == "timeout" and len(model.seen) == 1 and report.usage["output_tokens"] == 1


@pytest.mark.parametrize("tokens_in, tokens_out", [
    (None, 5), (5, None), (-1, 5), (5, -3), (True, 5), (5, 2.5), ("100", 5), (5, float("nan")),
])
def test_invalid_usage_metadata_stops_a_metered_run(env, monkeypatch, tokens_in, tokens_out):
    calls = counting(env, monkeypatch)
    finish_now = [call(ia.FINISH, finding="inconclusive", observations=[])]
    reply = ModelReply(content=finish_now, stop_reason="tool_use", input_tokens=tokens_in, output_tokens=tokens_out)
    model = Scripted(reply, finish_now)
    report = investigate(QUESTION, env[0], env[1], model)                    # default budget: token limits
    assert report.stopping_reason == "usage_unavailable" and not report.finished and len(model.seen) == 1
    assert calls == [] and not [t for t in report.trace if t["kind"] == "finish"]
    valid = [n for n in (tokens_in, tokens_out) if isinstance(n, int) and not isinstance(n, bool) and n >= 0]
    assert report.usage["input_tokens"] + report.usage["output_tokens"] == sum(valid)   # valid part recorded
    assert any("no valid token usage" in l["text"] for l in report.limitations)


def test_usage_without_any_limit_is_not_required(env):
    finish_now = [call(ia.FINISH, finding="inconclusive", observations=[])]
    reply = ModelReply(content=finish_now, stop_reason="tool_use", input_tokens=None, output_tokens=None)
    report = investigate(QUESTION, env[0], env[1], Scripted(reply),
                         budget=Budget(max_input_tokens=None, max_output_tokens=None))
    assert report.finished and report.usage["input_tokens"] == report.usage["output_tokens"] == 0
    assert report.usage["turns_without_usage"] == 1


def test_object_raising_on_usage_access_does_not_crash(env):
    class Hostile:
        def __getattr__(self, name):
            raise RuntimeError("no")
    report = investigate(QUESTION, env[0], env[1], Scripted(Hostile()))
    assert report.stopping_reason == "usage_unavailable"


def test_well_formed_replies_are_counted_exactly_once(env):
    model = Scripted([query()], malformed(7, 3), [query(max_episodes=2)], answer_inconclusive, tokens=(100, 20))
    report = investigate(QUESTION, env[0], env[1], model, budget=Budget(price_per_mtok=(1.0, 10.0)))
    assert report.finished and report.usage["model_turns"] == 4
    assert (report.usage["input_tokens"], report.usage["output_tokens"]) == (3 * 100 + 7, 3 * 20 + 3)
    assert report.usage["cost_usd"] == pytest.approx((307 * 1.0 + 63 * 10.0) / 1e6)
    turns = [t for t in report.trace if t["kind"] == "model_turn"]
    assert sum(t["input_tokens"] for t in turns) == 307 and sum(t["output_tokens"] for t in turns) == 63


def test_claude_adapter_reports_missing_usage_as_unknown():
    resp = Resp([Block(type="text", text="x")], "end_turn")
    resp.usage = type("U", (), {"input_tokens": None, "output_tokens": 4})()
    reply = ia.ClaudeInvestigationModel(FakeClient([resp])).respond("s", [], [])
    assert reply.input_tokens is None and reply.output_tokens == 4
    resp = Resp([Block(type="text", text="x")], "end_turn")
    resp.usage = type("U", (), {"input_tokens": 9, "output_tokens": 4})()        # no cache fields: counted as 0
    assert ia.ClaudeInvestigationModel(FakeClient([resp])).respond("s", [], []).input_tokens == 9
    resp.usage = None
    assert ia.ClaudeInvestigationModel(FakeClient([resp])).respond("s", [], []).output_tokens is None


def test_unknown_tool_text_only_and_truncated_replies_are_refused(env, monkeypatch):
    calls = counting(env, monkeypatch)
    model = Scripted([call("delete_everything")], [text("I think the answer is yes.")],
                     ModelReply(content=[query()], stop_reason="max_tokens"))
    report = investigate(QUESTION, env[0], env[1], model)
    assert report.stopping_reason == "rejected_actions" and calls == []
    refused = [t for t in report.trace if t.get("outcome") == "refused"]
    assert refused[0]["refused"] == "unknown_tool"
    assert "No tool was called" in json.dumps(model.seen[2])
    assert [t["stop_reason"] for t in report.trace if t["kind"] == "model_turn"] == ["tool_use", "end_turn",
                                                                                    "max_tokens"]


def test_refusal_and_unavailable_model_stop_safely(env):
    r = investigate(QUESTION, env[0], env[1], Scripted(ModelReply(content=[], stop_reason="refusal")))
    assert r.stopping_reason == "model_refusal" and r.finding == "incomplete"
    r = investigate(QUESTION, env[0], env[1], Scripted([query()], ModelUnavailable("timeout")))
    assert r.stopping_reason == "model_unavailable" and len(r.evidence) == 1
    assert r.trace[-2] == {**r.trace[-2], "kind": "model_error", "reason": "timeout"}
    r = investigate(QUESTION, env[0], env[1], Scripted(RuntimeError("secret sk-ant-123 in text")))
    assert r.stopping_reason == "model_unavailable" and "sk-ant" not in json.dumps(r.to_json())


# ---- invalid / unauthorized tool arguments ----

@pytest.mark.parametrize("bad, kind", [
    (call("query_events", source="validation_annotation"), "invalid_arguments"),                   # missing
    (call("query_events", source="validation_annotation", behavior="attack", policy="open"), "invalid_arguments"),
    (call("query_events", source="validation_annotation", behavior="flying"), "invalid_arguments"),  # contract
    (call("compare_windows", feature="dist_rate", windows_from={"evidence_id": "ev:x"},
          sufficiency={"min_sessions": 1}), "invalid_arguments"),              # the model may not set criteria
    (call("compare_windows", feature="dist_rate", windows_from={"evidence_id": "ev:get_window_features:"
                                                                          "0123456789abcdef01234567"}),
     "unknown_evidence"),                                                     # fabricated evidence id
    (call("get_window_features", episode_ids=[ep(VAL_A, 100)], pre_event={"length_frames": 0, "gap_frames": 0},
          features=["dist_rate"]), "invalid_arguments"),
    ({"type": "tool_use", "id": "x1", "name": "query_events", "input": ["not", "an", "object"]}, "invalid_arguments"),
])
def test_invalid_arguments_are_refused_before_any_tool_runs(env, monkeypatch, bad, kind):
    calls = counting(env, monkeypatch)
    report = investigate(QUESTION, env[0], env[1], Scripted([bad]), budget=Budget(max_consecutive_rejections=1))
    assert calls == [] and report.evidence == []
    (t,) = [t for t in report.trace if t["kind"] == "tool_call"]
    assert t["outcome"] == "refused" and t["refused"] == kind
    assert report.stopping_reason == "rejected_actions"


@pytest.mark.parametrize("bad", [query(sessions=[TEST]), query(sessions=[TRAIN]),
                                 call("get_window_features", episode_ids=[ep(TEST, 100)],
                                      pre_event={"length_frames": 20, "gap_frames": 0}, features=["dist_rate"])])
def test_unauthorized_requests_never_read_data(env, monkeypatch, bad):
    def no_read(*a, **k):
        raise AssertionError("the cache was read")
    monkeypatch.setattr(ev, "verified_sessions", no_read)
    report = investigate(QUESTION, env[0], env[1], Scripted([bad]), budget=Budget(max_consecutive_rejections=1))
    (t,) = [t for t in report.trace if t["kind"] == "tool_call"]
    assert t["refused"] == "unauthorized" and report.evidence == []


# ---- Codex finding 2: malformed arguments are refused, never crash the loop ----

@pytest.mark.parametrize("bad, problem", [
    (call("get_window_features", episode_ids=None, pre_event={"length_frames": 20, "gap_frames": 0},
          features=["dist_rate"]), "episode_ids must be array, not null (leave the field out"),
    (call("check_tracking_quality", window_ids=None, history_frames=[1]), "window_ids must be array, not null"),
    (call("check_tracking_quality", windows_from=None, history_frames=[1]), "windows_from must be object"),
    (call("compare_windows", feature="dist_rate", pre_window_ids=None, baseline_window_ids=["x"]),
     "pre_window_ids must be array"),
    (call("get_window_features", episodes_from={"evidence_id": 7}, pre_event={"length_frames": 20, "gap_frames": 0},
          features=["dist_rate"]), "evidence_id must be string, not integer"),
    (call("get_window_features", episode_ids=[ep(VAL_A, 100)], pre_event={"length_frames": True, "gap_frames": 0},
          features=["dist_rate"]), "length_frames must be integer, not boolean"),
    (call("check_tracking_quality", window_ids=[["nested"]], history_frames=[1]), "window_ids[0] must be string"),
    (call("query_events", source="validation_annotation", behavior="attack", max_episodes=2.5),
     "max_episodes must be integer, not number"),
    (call(ia.FINISH, finding="inconclusive", primary_evidence_id=[], observations=[]),
     "primary_evidence_id must be string or null, not array"),
    (call(ia.FINISH, finding="inconclusive", observations=None), "observations must be array, not null"),
    (call(ia.FINISH, finding="inconclusive", observations=[{"evidence_id": None, "field": "status"}]),
     "evidence_id must be string, not null"),
    (call(ia.FINISH, finding=["inconclusive"], observations=[]), "finding must be one of"),
    (call(ia.FINISH, finding="inconclusive", observations=[], next_steps="one string"), "next_steps must be array"),
    (call(ia.FINISH, finding="inconclusive", observations=[], interpretations=[{"statement": 3, "evidence_ids": []}]),
     "statement must be string"),
])
def test_malformed_arguments_are_refused_and_the_model_can_revise(env, monkeypatch, bad, problem):
    calls = counting(env, monkeypatch)
    report = investigate(QUESTION, env[0], env[1], Scripted([query()], [bad], answer_inconclusive))
    assert report.stopping_reason == "finished" and report.finding == "inconclusive"     # revised and accepted
    (refused,) = [t for t in report.trace if t.get("outcome") == "refused"]
    assert problem in refused["message"] and len(calls) == 1                           # only the query ran


def test_malformed_arguments_end_in_an_incomplete_report_when_never_revised(env):
    bad = call("get_window_features", episode_ids=None, pre_event={"length_frames": 20, "gap_frames": 0},
               features=["dist_rate"])
    report = investigate(QUESTION, env[0], env[1], Scripted([query()], [bad], [bad], [bad]),
                         budget=Budget(max_consecutive_rejections=3, max_repeated_calls=5))
    assert report.stopping_reason == "rejected_actions" and report.finding == "incomplete" and len(report.evidence) == 1


def test_null_is_accepted_where_the_schema_allows_it(env):
    report = investigate(QUESTION, env[0], env[1], Scripted(
        [query(sessions=None, merge_gap_frames=None)], answer_inconclusive))
    assert report.finished and report.evidence[0]["request"]["sessions"] is None


def test_an_unexpected_failure_while_expanding_arguments_is_a_refusal(env, monkeypatch):
    def broken(self, *a):
        raise KeyError("boom")
    monkeypatch.setattr(ia._Run, "build_request", broken)
    report = investigate(QUESTION, env[0], env[1], Scripted([query()]), budget=Budget(max_consecutive_rejections=1))
    (t,) = [t for t in report.trace if t["kind"] == "tool_call"]
    assert t["refused"] == "invalid_arguments" and "KeyError" in t["message"]
    assert report.stopping_reason == "rejected_actions"


def test_unavailable_source_is_a_validated_answer_not_a_substitute(env):
    model = Scripted([call("query_events", source="classifier_prediction", behavior="attack")],
                     lambda m: [text(json.dumps(last_result(m)))])
    report = investigate(QUESTION, env[0], env[1], model, budget=Budget(max_model_turns=2))
    (e,) = report.evidence
    assert e["status"] == "error" and last_result(model.seen[1])["errors"][0]["code"] == "source_unavailable"


# ---- the final answer cannot fabricate ----

def _until_compare(status_crit):
    """Turns: query, features, compare (by reference); returns the scripted turns."""
    return [
        [query()],
        lambda m: [call("get_window_features", episodes_from={"evidence_id": last_result(m)["evidence_id"]},
                        pre_event={"length_frames": 20, "gap_frames": 0},
                        baseline={"length_frames": 20, "clearance_frames": 10, "max_windows_per_session": 500},
                        features=["dist_rate"])],
        lambda m: [call("compare_windows", feature="dist_rate",
                        windows_from={"evidence_id": last_result(m)["evidence_id"]})],
    ]


def finish(build_args):
    return lambda m: [call(ia.FINISH, **build_args(last_result_of(m, "compare_windows")))]


def last_result_of(messages, tool):
    return next(r for n, _, r in reversed(history(messages)) if n == tool and "evidence_id" in r)


def GOOD(c):
    v = c["measurements"]["effect"]["value"]
    v = 0.0 if v is None else v              # an insufficient comparison has no effect to cite
    return {"finding": "effect_described", "primary_evidence_id": c["evidence_id"],
            "direction": "negative" if v < 0 else "positive",
            "observations": [obs(c, "measurements.effect.value"), obs(c, "measurements.n_sessions_paired")]}


def interp(c, statement):
    return [{"statement": statement, "evidence_ids": [c["evidence_id"]]}]


@pytest.mark.parametrize("mutate, problem", [
    (lambda a, c: a["observations"][0].update(evidence_id="ev:compare_windows:" + "f" * 24),
     "not a result of this run"),
    (lambda a, c: a.update(primary_evidence_id="ev:compare_windows:" + "e" * 24), "not a result of this run"),
    (lambda a, c: a["observations"][0].update(field="measurements.no_such_field"), "has no field"),
    (lambda a, c: a["observations"][0].update(field="measurements.effect"), "is not a value"),
    (lambda a, c: a["observations"][0].update(field="provenance.cache_path"), "is not a path"),
    (lambda a, c: a["observations"][0].update(statement="The effect was -123.4 px/s."), "unknown fields"),
    (lambda a, c: a.update(interpretations=interp(c, "The effect was -123.4 px/s.")), "-123.4"),
    (lambda a, c: a.update(interpretations=interp(c, "An effect of nine hundred px/s.")), "number words"),
    (lambda a, c: a.update(interpretations=interp(c, "The difference is statistically significant.")),
     "significance"),
    (lambda a, c: a.update(interpretations=interp(c, "Approach causes attack.")), "causal claim"),
    (lambda a, c: a.update(next_steps=["Show that approach causes attack (p < 0.05)."]), "causal claim"),
    (lambda a, c: a.update(additional_limitations=["The p-value could not be computed."]), "p-value"),
    (lambda a, c: a.update(interpretations=[{"statement": "x", "evidence_ids": []}]), "evidence_ids"),
    (lambda a, c: a.update(direction="positive" if a["direction"] == "negative" else "negative"), "direction must be"),
    (lambda a, c: a.update(finding="insufficient_evidence"), "carries no direction"),
    (lambda a, c: a.update(observations=[]), "at least one observation"),
    (lambda a, c: a.update(confidence=0.99), "unknown fields"),
])
def test_unsupported_final_answers_are_rejected_then_revised(env, mutate, problem):
    def bad(c):
        a = GOOD(c)
        mutate(a, c)
        return a
    turns = _until_compare(LOOSE) + [finish(bad), finish(GOOD)]
    report = investigate(QUESTION, env[0], env[1], Scripted(*turns), sufficiency=LOOSE)
    rejected, accepted = [t for t in report.trace if t["kind"] == "finish"]
    assert rejected["outcome"] == "refused" and problem in rejected["message"]
    assert accepted["outcome"] == "accepted" and report.stopping_reason == "finished"
    assert report.finding == "effect_described"


def test_effect_cannot_be_claimed_from_an_insufficient_comparison(env):
    turns = _until_compare(None) + [finish(GOOD), finish(GOOD), finish(GOOD)]
    report = investigate(QUESTION, env[0], env[1], Scripted(*turns))     # provisional criteria
    assert report.stopping_reason == "rejected_actions" and report.finding == "incomplete"
    assert all("status 'ok'" in t["message"] for t in report.trace if t["kind"] == "finish")


def test_observations_are_read_from_the_stored_result(env):
    """The model names fields; values and statements come from the validated
    result, including list indices and session IDs."""
    def cites(c):
        return {**GOOD(c), "observations": [obs(c, f) for f in (
            "measurements.effect.value", "measurements.n_windows_pre", "measurements.per_session[0].session",
            "measurements.per_session[1].difference", "measurements.sufficiency.reasons_not_met", "status")]}
    report = investigate(QUESTION, env[0], env[1], Scripted(*_until_compare(LOOSE), finish(cites)), sufficiency=LOOSE)
    assert report.finished
    (stored,) = [e["result"] for e in report.evidence if e["tool"] == "compare_windows"]
    m = stored["measurements"]
    expected = [m["effect"]["value"], m["n_windows_pre"], m["per_session"][0]["session"],
                m["per_session"][1]["difference"], m["sufficiency"]["reasons_not_met"], "ok"]
    assert [o["value"] for o in report.observations] == expected
    assert all(o["kind"] == "observation" and o["tool"] == "compare_windows" and o["evidence_id"] in o["statement"]
               and json.dumps(o["value"]) in o["statement"] for o in report.observations)


def test_fabricated_findings_on_an_insufficient_comparison_are_refused(env):
    """Codex finding 1: an insufficient comparison (null effect) cannot carry
    a claimed measurement, number words, effect language or prohibited claims
    in any prose field; after revision the null effect is reported as null."""
    def free_text(c):          # the old prose observations: no longer accepted at all
        return {"finding": "insufficient_evidence", "primary_evidence_id": c["evidence_id"], "observations": [
            {"statement": "The median paired-session effect was 10 px/s.", "evidence_ids": [c["evidence_id"]]}]}

    def prose(c):
        return {"finding": "insufficient_evidence", "primary_evidence_id": c["evidence_id"],
                "observations": [obs(c, "measurements.effect.value")],
                "interpretations": interp(c, "The effect was nine hundred px/s; distance decreased before attack."),
                "next_steps": ["Approach causes attack with p-value 0.00001."]}

    def honest(c):
        return {"finding": "insufficient_evidence", "primary_evidence_id": c["evidence_id"],
                "observations": [obs(c, "measurements.effect.value"), obs(c, "measurements.sufficiency.reasons_not_met")],
                "interpretations": interp(c, "Too few paired sessions to describe an effect."),
                "next_steps": ["Extract more validation sessions."]}
    turns = _until_compare(None) + [finish(free_text), finish(prose), finish(honest)]
    report = investigate(QUESTION, env[0], env[1], Scripted(*turns))        # provisional: insufficient
    first, second, accepted = [t for t in report.trace if t["kind"] == "finish"]
    assert first["outcome"] == "refused" and "unknown fields ['evidence_ids', 'statement']" in first["message"]
    msg = second["message"]
    for problem in ("number words ['nine', 'hundred']", "effect language ['decreased']",
                    "next_steps[0]: causal claim", "next_steps[0]: p-value", "numbers or number words ['0.00001']"):
        assert problem in msg
    assert accepted["outcome"] == "accepted" and report.finding == "insufficient_evidence"
    assert report.primary_evidence["effect"]["value"] is None
    assert report.observations[0]["value"] is None and report.observations[0]["statement"].endswith("= null")
    assert report.direction is None


# ---- repeated calls, budgets, partial evidence ----

def test_repeated_identical_requests_are_not_reexecuted(env, monkeypatch):
    calls = counting(env, monkeypatch)
    report = investigate(QUESTION, env[0], env[1], Scripted([query()], [query()], [query()]),
                         budget=Budget(max_repeated_calls=2))
    assert len(calls) == 1 and report.stopping_reason == "repeated_actions"
    refused = [t for t in report.trace if t.get("refused") == "repeated_call"]
    assert len(refused) == 2 and report.evidence[0]["evidence_id"] in refused[0]["message"]


def test_tool_budget_leaves_one_turn_to_finish(env):
    def answer(m):
        q = last_result_of(m, "query_events")
        return [call(ia.FINISH, finding="inconclusive", observations=[obs(q, "measurements.n_episodes_total")])]
    model = Scripted([query()], [query(max_episodes=3)], answer)
    report = investigate(QUESTION, env[0], env[1], model, budget=Budget(max_tool_calls=2))
    assert report.stopping_reason == "finished" and report.usage["tool_calls"] == 2
    assert "tool-call budget is spent" in json.dumps(model.seen[2])


def test_budget_exhaustion_stops_with_partial_evidence(env):
    report = investigate(QUESTION, env[0], env[1], Scripted([query()], [query(max_episodes=3)], [query(max_episodes=4)]),
                         budget=Budget(max_tool_calls=2))
    assert report.stopping_reason == "max_tool_calls" and report.finding == "incomplete"
    assert len(report.evidence) == 2 and report.usage["tool_calls"] == 2
    assert [t["refused"] for t in report.trace if t.get("outcome") == "refused"] == ["tool_budget_exhausted"]
    texts = [l["text"] for l in report.limitations]
    assert any("stopped before a final answer (max_tool_calls)" in t for t in texts)
    assert any("Episodes come from human annotations" in t for t in texts)      # tool limitations kept


@pytest.mark.parametrize("budget, tokens, reason", [
    (Budget(max_model_turns=2), (10, 5), "max_model_turns"),
    (Budget(max_input_tokens=1500), (1000, 5), "token_budget"),
    (Budget(max_output_tokens=900), (10, 500), "token_budget"),
    (Budget(max_cost_usd=0.01, price_per_mtok=(4.0, 20.0)), (2000, 100), "cost_budget"),
])
def test_turn_token_and_cost_budgets(env, budget, tokens, reason):
    model = Scripted(*[[query(max_episodes=k)] for k in range(1, 10)], tokens=tokens)
    report = investigate(QUESTION, env[0], env[1], model, budget=budget)
    assert report.stopping_reason == reason and report.finding == "incomplete" and report.evidence
    if reason == "cost_budget":
        assert report.usage["cost_usd"] >= 0.01


def test_cost_limit_without_prices_is_refused():
    with pytest.raises(ValueError):
        Budget(max_cost_usd=1.0)
    with pytest.raises(ValueError):
        Budget(max_result_chars=500)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def at(clock, t, turn):
    """A scripted turn that arrives at simulated time t."""
    def arrive(messages):
        clock.t = t
        return turn(messages) if callable(turn) else turn
    return arrive


def test_timeout_is_enforced_between_steps(env):
    clock = Clock()
    report = investigate(QUESTION, env[0], env[1], Scripted(at(clock, 5.0, [query()]),
                                                            at(clock, 1000.0, [query(max_episodes=2)])),
                         budget=Budget(timeout_s=100), clock=clock)
    assert report.stopping_reason == "timeout" and len(report.evidence) == 1


# ---- Codex finding 3: a reply that overran a limit executes nothing, not even a final answer ----

def answer_inconclusive(m):
    done = [r for _, _, r in history(m) if "evidence_id" in r]           # the latest validated result
    return [call(ia.FINISH, finding="inconclusive", observations=[obs(done[-1], "status")] if done else [])]


@pytest.mark.parametrize("budget, tokens, reason", [
    (Budget(max_output_tokens=1), (100, 100), "token_budget"),
    (Budget(max_input_tokens=50), (100, 1), "token_budget"),
    (Budget(max_cost_usd=0.000001, price_per_mtok=(1.0, 1.0)), (100, 100), "cost_budget"),
])
def test_finishing_after_overrunning_a_budget_is_refused(env, monkeypatch, budget, tokens, reason):
    calls = counting(env, monkeypatch)
    for first in ([call(ia.FINISH, finding="inconclusive", observations=[])], [query()]):
        report = investigate(QUESTION, env[0], env[1], Scripted(first, answer_inconclusive, tokens=tokens),
                             budget=budget)
        assert report.stopping_reason == reason and not report.finished and report.finding == "incomplete"
        assert report.usage["model_turns"] == 1 and calls == []           # neither the finish nor the query ran
        (over,) = [t for t in report.trace if t["kind"] == "budget_exceeded"]
        assert over["stopping_reason"] == reason and over["not_executed"] == [first[0]["id"]]
        assert any("went past the" in l["text"] for l in report.limitations)


def test_a_reply_arriving_after_the_deadline_is_not_executed(env, monkeypatch):
    calls = counting(env, monkeypatch)
    clock = Clock()
    finish_now = [call(ia.FINISH, finding="inconclusive", observations=[])]
    report = investigate(QUESTION, env[0], env[1], Scripted(at(clock, 10.0, finish_now)),
                         budget=Budget(timeout_s=1), clock=clock)
    assert report.stopping_reason == "timeout" and not report.finished and report.usage["elapsed_s"] == 10.0
    assert not [t for t in report.trace if t["kind"] == "finish"] and calls == []


def test_deadline_is_rechecked_between_tool_actions(env, monkeypatch):
    """Two calls in one turn: the first runs past the deadline, so the second
    (a final answer) is not executed."""
    clock = Clock()
    real = CalMS21InvestigationTools.query_events

    def slow(self, request):
        clock.t = 50.0
        return real(self, request)
    monkeypatch.setattr(CalMS21InvestigationTools, "query_events", slow)
    turn = [query(), call(ia.FINISH, finding="inconclusive", observations=[])]
    report = investigate(QUESTION, env[0], env[1], Scripted(turn), budget=Budget(timeout_s=10), clock=clock)
    assert report.stopping_reason == "timeout" and not report.finished and len(report.evidence) == 1
    assert not [t for t in report.trace if t["kind"] == "finish"]


def test_requests_are_bounded_by_the_remaining_budget(env):
    seen = []

    class Bounded(Scripted):
        def respond(self, system, tools, messages, limits=None):
            seen.append(limits)
            return super().respond(system, tools, messages)
    clock = Clock()
    model = Bounded(at(clock, 30.0, [query()]), at(clock, 40.0, [query(max_episodes=2)]), tokens=(10, 300))
    investigate(QUESTION, env[0], env[1], model, clock=clock,
                budget=Budget(max_output_tokens=1000, max_cost_usd=1.0, price_per_mtok=(0.0, 1000.0),
                              timeout_s=100, max_model_turns=2))
    assert seen[0] == ia.RequestLimits(max_output_tokens=1000, timeout_s=100.0)
    # 700 tokens left; cost: (1.0 - 0.3) USD / 1000 USD per Mtok = 700 tokens; 70 s left
    assert seen[1].max_output_tokens == 700 and seen[1].timeout_s == pytest.approx(70.0)


def test_setup_is_checked(env):
    with pytest.raises(TypeError):
        investigate(QUESTION, object(), env[1], Scripted())
    with pytest.raises(ic.ContractError):
        investigate(QUESTION, env[0], None, Scripted())
    with pytest.raises(ValueError):
        investigate("", env[0], env[1], Scripted())


# ---- Codex finding 4: every report keeps the validated evidence whole ----

def test_incomplete_report_keeps_validated_snapshots(env):
    turns = _until_compare(None)
    report = investigate(QUESTION, env[0], env[1], Scripted(*turns), budget=Budget(max_model_turns=3))
    assert report.stopping_reason == "max_model_turns" and report.finding == "incomplete"
    assert [e["tool"] for e in report.evidence] == ["query_events", "get_window_features", "compare_windows"]
    for e, t in zip(report.evidence, [t for t in report.trace if t["kind"] == "tool_call"]):
        assert e["request"] == t["request"] and e["result"]["evidence_id"] == e["evidence_id"]
        ic.validate_result(e["tool"], e["request"], e["result"], env[1])           # still a valid result
        assert e["result"]["provenance"] and e["result"]["methodology"] and e["result"]["limitations"]
    compare = report.evidence[2]["result"]
    assert compare["measurements"]["sufficiency"]["reasons_not_met"] and compare["measurements"]["per_session"]
    assert len(report.evidence[2]["request"]["baseline_window_ids"]) > 20           # never shortened
    assert report.verify_evidence() == []
    json.loads(json.dumps(report.to_json()))


def test_report_snapshots_are_copies_and_tampering_is_detectable(env):
    report = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    with pytest.raises(AttributeError):
        report.finding = "effect_described"                                        # frozen
    dumped = report.to_json()
    dumped["evidence"][0]["result"]["status"] = "error"
    assert report.evidence[0]["result"]["status"] == "ok"                           # to_json is a deep copy
    primary = next(e for e in report.evidence if e["evidence_id"] == report.primary_evidence["evidence_id"])
    assert report.primary_evidence["effect"] is not primary["result"]["measurements"]["effect"]
    assert report.observations[-1]["value"] == primary["result"]["measurements"]["effect"]["value"]
    report.evidence[0]["result"]["measurements"]["n_episodes_total"] += 1
    assert report.verify_evidence() == [report.evidence[0]["evidence_id"]]


def test_stored_evidence_is_isolated_from_what_the_model_receives(env):
    class Vandal(Investigator):
        def respond(self, system, tools, messages):
            reply = super().respond(system, tools, messages)
            for m in messages:                 # the model mutates everything it was given
                if isinstance(m["content"], list):
                    m["content"].clear()
            return reply
    report = investigate(QUESTION, env[0], env[1], Vandal(), sufficiency=LOOSE)
    assert report.finished and report.verify_evidence() == []


# ---- audit trace ----

def test_trace_is_complete_and_serializable(env):
    report = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    data = json.loads(json.dumps(report.to_json()))
    turns = [t for t in data["trace"] if t["kind"] == "model_turn"]
    calls = [t for t in data["trace"] if t["kind"] == "tool_call"]
    assert all(t["text"] and t["model"] == "fake-investigator" for t in turns)
    assert all({"arguments", "request", "outcome", "evidence_id"} <= set(t) for t in calls)
    assert [c["evidence_id"] for c in calls] == [e["evidence_id"] for e in data["evidence"]]
    assert data["trace"][-1] == {**data["trace"][-1], "kind": "stop", "stopping_reason": "finished"}
    assert data["usage"]["model_turns"] == 5 and data["usage"]["input_tokens"] == 5000


def test_long_results_are_shortened_for_the_model_but_passed_whole_by_reference(env):
    seen = []

    class Recording(Investigator):
        def respond(self, system, tools, messages):
            seen.append(messages)
            return super().respond(system, tools, messages)
    report = investigate(QUESTION, env[0], env[1], Recording(), sufficiency=LOOSE,
                         budget=Budget(max_result_chars=8000))
    assert report.finished
    features = next(t for t in report.trace if t["kind"] == "tool_call" and t["tool"] == "get_window_features")
    compare = next(t for t in report.trace if t["kind"] == "tool_call" and t["tool"] == "compare_windows")
    shown = next(b["content"] for m in seen[-1] if isinstance(m["content"], list) for b in m["content"]
                 if b.get("type") == "tool_result" and b["tool_use_id"] == features["tool_use_id"])
    assert len(shown) <= 8000 and "_rendering" in shown and features["evidence_id"] in shown
    json.loads(shown)                                    # always valid JSON, never a cut string
    shown_baselines = shown.count('"kind":"baseline"')
    assert len(compare["request"]["baseline_window_ids"]) > shown_baselines


def test_a_result_too_long_for_any_shortening_becomes_a_valid_summary():
    result = {"contract_version": ic.CONTRACT_VERSION, "tool": "query_events", "status": "ok",
              "evidence_id": "ev:query_events:" + "a" * 24, "errors": [],
              "measurements": {"n_episodes_total": 3, "episodes": [{"x": "y" * 50}] * 50, "big": "z" * 5000},
              "limitations": ["keep me"]}
    text = ia._render(result, 1000)
    assert len(text) <= 1000
    shown = json.loads(text)
    assert shown["evidence_id"] == result["evidence_id"] and shown["status"] == "ok"
    assert shown["measurements_scalars"]["n_episodes_total"] == 3 and "_rendering" in shown


# ---- separation from the Day 2 parser ----

def test_independent_of_the_day2_rule_parser():
    tree = ast.parse(Path(ia.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {"agent", "claude_agent", "live_tracker.agent", "live_tracker.claude_agent", None} - {None}
    assert ".agent" not in str(imported) and "claude_agent" not in str(imported)


# ---- the Claude adapter (no API calls) ----

class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls, self.options = list(responses), [], []
        self.beta = self
        self.messages = self

    def with_options(self, **kw):
        self.options.append(kw)
        return self

    def create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw)))
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


class Block:
    def __init__(self, **d):
        self.d = d

    def model_dump(self, mode=None, exclude_none=False):
        return dict(self.d)


class Resp:
    def __init__(self, blocks, stop="tool_use", model="claude-opus-5-5"):
        self.content, self.stop_reason, self.model = blocks, stop, model
        self.usage = type("U", (), {"input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": 900,
                                    "cache_creation_input_tokens": 0})()


def test_claude_adapter_runs_the_loop_and_echoes_blocks_unchanged(env):
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    q = {"type": "tool_use", "id": "toolu_1", "name": "query_events",
         "input": {"source": "validation_annotation", "behavior": "attack"}}
    client = FakeClient([Resp([Block(**thinking), Block(**q)]), Resp([Block(type="text", text="stop")], "end_turn"),
                         Resp([Block(type="text", text="stop")], "end_turn")])
    model = ia.ClaudeInvestigationModel(client, max_retries=1, timeout_s=30)
    report = investigate(QUESTION, env[0], env[1], model, budget=Budget(max_model_turns=3))
    assert report.usage["input_tokens"] == 950 * 3 and len(report.evidence) == 1
    first, second = client.calls[0], client.calls[1]
    assert first["model"] == "claude-opus-5-5" and first["tool_choice"] == {"type": "auto"}
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in first["tools"]] == list(ia.TOOL_NAMES) and "thinking" not in first
    assert second["messages"][1] == {"role": "assistant", "content": [thinking, q]}       # unchanged, append-only
    assert second["messages"][2]["content"][0]["tool_use_id"] == "toolu_1"
    assert client.options[0] == {"timeout": 30, "max_retries": 1}
    assert report.trace[0]["model"] == "claude-opus-5-5"


def test_claude_adapter_bounds_tokens_and_timeout_by_the_remaining_budget():
    client = FakeClient([Resp([Block(type="text", text="x")], "end_turn")])
    model = ia.ClaudeInvestigationModel(client, max_tokens=16_000, timeout_s=180, max_retries=2)
    model.respond("s", [], [], limits=ia.RequestLimits(max_output_tokens=250, timeout_s=30.0))
    assert client.calls[0]["max_tokens"] == 250
    assert client.options[0] == {"timeout": 10.0, "max_retries": 2}         # 30 s over 3 attempts
    assert model.request("s", [], [], ia.RequestLimits(None, 900.0))["max_tokens"] == 16_000


def test_claude_adapter_errors_and_configuration(monkeypatch):
    class APITimeoutError(Exception):
        pass
    with pytest.raises(ModelUnavailable) as e:
        ia.ClaudeInvestigationModel(FakeClient([APITimeoutError("secret")])).respond("s", [], [])
    assert e.value.reason == "timeout" and "secret" not in str(e.value)
    with pytest.raises(ModelUnavailable) as e:
        ia.ClaudeInvestigationModel().respond("s", [], [])
    assert e.value.reason == "missing_api_key"
    monkeypatch.setenv(ia.MODEL_ENV, "claude-sonnet-5-5")
    assert ia.ClaudeInvestigationModel(FakeClient([])).model == "claude-sonnet-5-5"
    no_fallback = ia.ClaudeInvestigationModel(FakeClient([]), fallbacks=False).request("s", [], [])
    assert "fallbacks" not in no_fallback and "betas" not in no_fallback


def test_real_sdk_request_over_a_mock_transport(env):
    httpx = pytest.importorskip("httpx2")
    anthropic = pytest.importorskip("anthropic")
    seen = []

    def handler(request):
        seen.append((request, json.loads(request.content)))
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "query_events",
                         "input": {"source": "validation_annotation", "behavior": "attack"}}],
            "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 7, "output_tokens": 3}})
    client = anthropic.Anthropic(api_key="sk-ant-test-NOT-REAL", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    report = investigate(QUESTION, env[0], env[1], ia.ClaudeInvestigationModel(client, max_retries=0),
                         budget=Budget(max_model_turns=1))
    assert report.stopping_reason == "max_model_turns" and len(report.evidence) == 1
    request, body = seen[0]
    assert "server-side-fallback-2026-07-01" in request.headers.get("anthropic-beta", "")
    assert body["fallbacks"] == "default" and body["output_config"] == {"effort": "high"}
    assert body["tools"][0]["name"] == "query_events" and body["messages"][0]["role"] == "user"


def test_real_sdk_server_error_is_model_unavailable(env):
    httpx = pytest.importorskip("httpx2")
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="sk-ant-test-NOT-REAL", http_client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "x"}}))))
    report = investigate(QUESTION, env[0], env[1], ia.ClaudeInvestigationModel(client, max_retries=0))
    assert report.stopping_reason == "model_unavailable" and report.trace[0]["reason"] == "api_error"


# ---- real validation data (verified cache; skipped when absent) ----

@pytest.fixture(scope="module")
def real_env():
    if not (REAL_CACHE / INDEX_NAME).exists() or not REAL_MANIFEST.exists():
        pytest.skip("verified CalMS21 validation cache not present")
    policy = ic.EvidencePolicy.from_manifest(REAL_MANIFEST)
    return CalMS21InvestigationTools.for_validation_cache(REAL_CACHE, REAL_MANIFEST, policy), policy


def test_real_investigation_follows_up_and_reports_insufficient_evidence(real_env):
    """On the three cached validation sessions (one with attacks), the model
    finds tracking concerns, repeats the comparison without them, finds the
    evidence insufficient, checks the excluded windows and stops."""
    report = investigate(QUESTION, *real_env, Investigator(length=30, clearance=150))
    assert tools_used(report) == ["query_events", "get_window_features", "check_tracking_quality",
                                  "compare_windows", "compare_windows", "check_tracking_quality", ia.FINISH]
    assert report.stopping_reason == "finished" and report.finding == "insufficient_evidence"
    p = report.primary_evidence
    assert (p["status"], p["n_sessions_paired"], p["n_windows_pre"], p["n_windows_baseline"]) == \
        ("insufficient_evidence", 1, 5, 128)
    assert p["effect"]["value"] is None
    assert p["sufficiency"]["reasons_not_met"] == ["min_sessions", "min_windows_per_group_pre"]
    compare = next(t for t in report.trace if t["kind"] == "tool_call" and t["tool"] == "compare_windows")
    assert len(compare["request"]["baseline_window_ids"]) == 128              # passed whole by reference
    assert all(e["status"] != "error" for e in report.evidence)
    lims = [l["text"] for l in report.limitations if l["source"] == "tool"]
    for phrase in ("Investigation confound", "Session-timing confound", "Pixel scale", "Motion accuracy is not verified"):
        assert any(phrase in t for t in lims)
    assert any("Extract the remaining validation sessions" in n for n in report.next_steps)


def test_real_investigation_cannot_be_talked_into_an_effect(real_env):
    """With one paired session the model's claim of an effect is refused,
    whatever it writes."""
    turns = [[query()],
             lambda m: [call("get_window_features", episodes_from={"evidence_id": last_result(m)["evidence_id"]},
                             pre_event={"length_frames": 30, "gap_frames": 0},
                             baseline={"length_frames": 30, "clearance_frames": 150, "max_windows_per_session": 500},
                             features=["dist_rate"])],
             lambda m: [call("compare_windows", feature="dist_rate",
                             windows_from={"evidence_id": last_result(m)["evidence_id"]})],
             finish(lambda c: {"finding": "effect_described", "primary_evidence_id": c["evidence_id"],
                               "direction": "negative",
                               "observations": [obs(c, "measurements.per_session[0].difference")],
                               "interpretations": interp(c, "Distance decreased before attack.")})]
    report = investigate(QUESTION, *real_env, Scripted(*turns), budget=Budget(max_consecutive_rejections=1))
    assert report.stopping_reason == "rejected_actions" and report.finding == "incomplete"
    msg = next(t for t in report.trace if t["kind"] == "finish")["message"]
    assert "status 'ok'" in msg
    (compare,) = [e for e in report.evidence if e["tool"] == "compare_windows"]      # kept whole although incomplete
    assert compare["result"]["measurements"]["n_sessions_paired"] == 1 and report.verify_evidence() == []
