"""Phase 3B.2 (investigation_live_v1, scripts/investigation_v1_live.py): the Stage 9 v1 agent through the guarded
Stage 8 model infrastructure. New file.

No test makes a network request (sockets and Stage 8's metadata GET are blocked) or spends credit: the OpenRouter
endpoint is emulated in-process (the real Anthropic SDK over httpx2.MockTransport, answered by Phase 3B.1's
deterministic offline investigator), metadata GETs are injected and keys are fake strings. Data are SYNTHETIC
fixtures only (labelled synthetic everywhere, never real validation results); no real data are read.
"""

import copy
import hashlib
import importlib.util
import json
import socket
from pathlib import Path

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_agent_v1 as av1
import live_tracker.investigation_live as live
import live_tracker.investigation_live_v1 as lv1
from test_investigation_v1_validation import inputs, recorded_run  # noqa: F401 (fixture)

httpx2 = pytest.importorskip("httpx2")
pytest.importorskip("anthropic")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("investigation_v1_live", ROOT / "scripts" / "investigation_v1_live.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)
Emu = lr.EmulatedOpenRouterV1
FAKE_OR = "sk-or-v1-FAKE-test-key-0123456789"
SEQUENCE = ["query_events", "check_tracking_quality", "get_window_features", "compare_windows", "compare_windows",
            "compare_windows", av1.FINISH]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(live, "_get_json", blocked)
    for v in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
              "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_LOG", ia.MODEL_ENV):
        monkeypatch.delenv(v, raising=False)


def argv(i, out, *extra):
    return [str(i["zip"]), "--archive-sha256", i["sha"], "--out", str(out), "--cache-dir", str(i["cache"]),
            "--manifest", str(i["manifest"]), "--allow-synthetic", *extra]


def outputs(out):
    return (json.loads((out / "run.json").read_text(encoding="utf-8")),
            json.loads((out / "investigation_report.json").read_text(encoding="utf-8")))


def dry(i, *extra, faults=None, cost=None, name="inv", capsys=None):
    emu = Emu(faults, cost)
    out = i["tmp"] / name
    code = lr.main(argv(i, out, *extra), emulator=emu)
    if capsys:
        capsys.readouterr()
    return code, out, emu


def kinds(doc):
    return [t["kind"] for t in doc["trace"]]


# ---- a complete offline run: tool calls, finish, tokens and cost ----

def test_an_emulated_run_completes_verifies_and_accounts_every_token_and_cost(inputs, capsys):
    code, out, emu = dry(inputs, capsys=capsys)
    s, doc = outputs(out)
    assert code == 0 and s["status"] == "verified"
    assert [x["step"] for x in s["steps"]] == ["inputs", "budget", "investigate", "verify", "cost_ledger"]
    assert [t["tool"] for t in doc["trace"] if t["kind"] in ("tool_call", "finish")] == SEQUENCE
    assert doc["finished"] and doc["finding"] == "insufficient_evidence" and av1.verify_report_v1(doc) == []
    assert (out / "investigation_report.md").read_text(encoding="utf-8") == av1.render_markdown_v1(doc)
    assert s["publication"] == doc["publication"] == "not_authorized"
    assert s["model"]["language_model"] is False and s["model"]["label"].startswith("EMULATED")
    assert s["inputs"]["data_source"] == "synthetic" and "REAL" not in json.dumps(s["inputs"])
    # tokens: every model turn is one ledger attempt with the provider's own counts
    reqs, turns = s["cost"]["ledger"]["requests"], [t for t in doc["trace"] if t["kind"] == "model_turn"]
    assert len(reqs) == len(turns) == len(emu.requests) == 7 and all(r["reply_used"] for r in reqs)
    assert [t["input_tokens"] for t in turns] == [r["input_tokens"] for r in reqs]
    assert doc["usage"]["input_tokens"] == sum(r["input_tokens"] for r in reqs)
    assert doc["usage"]["output_tokens"] == 7 * 300
    # cost: the provider-reported total re-adds, the Stage 6 estimate uses the plan's conservative prices
    assert s["cost"]["actual_spend_usd"] == round(sum(r["cost_usd"] for r in reqs), 8) > 0
    assert s["cost"]["actual_spend_complete"] and not s["cost"]["over_target"] and s["cost"]["target_usd"] == 0.50
    assert s["cost"]["provider_key_limit_usd"] is None and "ceiling_usd" not in s["cost"]     # no provider: no key
    p_in, p_out = doc["budget"]["price_per_mtok"]
    assert doc["usage"]["cost_usd"] == (doc["usage"]["input_tokens"] * p_in + doc["usage"]["output_tokens"] * p_out
                                        ) / 1e6
    assert s["estimate"]["estimated_worst_case_usd"] <= 0.50 and doc["budget"]["max_cost_usd"] < 0.50
    assert lr.verify_saved(out) == []


def test_requests_carry_only_the_v1_schemas_with_no_fallbacks_and_pinned_routing(inputs, capsys):
    _, _, emu = dry(inputs, capsys=capsys)
    for r in emu.requests:
        b = r["body"]
        assert r["method"] == "POST" and r["url"] == "https://openrouter.ai/api/v1/messages"
        assert r["has_authorization"] and not r["has_x_api_key"] and not r["has_anthropic_beta"]
        assert [t["name"] for t in b["tools"]] == list(av1.TOOL_NAMES_V1) and b["tools"] == av1.TOOL_SPECS_V1
        assert b["system"][0]["text"] == av1.SYSTEM_PROMPT_V1
        assert b["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
        assert not {"fallbacks", "betas", "thinking", "models", "route"} & set(b)
        assert b["model"] == "anthropic/claude-sonnet-5.5" and b["max_tokens"] <= 4000


def test_out_of_schema_tool_calls_are_traced_refusals_and_the_run_continues(inputs, capsys):
    code, out, _ = dry(inputs, "--max-model-turns", "9", faults={1: "unknown_tool", 2: "extra_argument"},
                        capsys=capsys)
    s, doc = outputs(out)
    refused = [t for t in doc["trace"] if t.get("outcome") == "refused"]
    assert [t["refused"] for t in refused] == ["unknown_tool", "invalid_arguments"]
    assert "min_n_windows" in refused[1]["message"] and not any("run_python" == e["tool"] for e in doc["evidence"])
    assert code == 0 and s["status"] == "verified" and doc["finished"] and lr.verify_saved(out) == []


def test_a_reply_without_a_tool_call_is_rejected_and_a_refusal_stops(inputs, capsys):
    code, out, _ = dry(inputs, faults={2: "no_tool"}, capsys=capsys)
    assert code == 0 and outputs(out)[1]["finished"]
    code, out, emu = dry(inputs, faults={2: "refusal"}, name="refusal", capsys=capsys)
    s, doc = outputs(out)
    assert code == 1 and s["status"] == "incomplete" and doc["stopping_reason"] == "model_refusal"
    assert len(emu.requests) == 2 and doc["finding"] == "incomplete" and lr.verify_saved(out) == []


# ---- provider failures: one attempt, no retry, fail closed ----

@pytest.mark.parametrize("fault", ["http_500", "http_429", "timeout", "garbage"])
def test_a_failed_request_is_one_attempt_with_unknown_cost_and_stops_the_run(inputs, capsys, fault):
    code, out, emu = dry(inputs, faults={2: fault}, capsys=capsys)
    s, doc = outputs(out)
    assert len(emu.requests) == 2                                         # no retry, nothing after it
    assert code == 1 and s["status"] == "incomplete" and doc["stopping_reason"] == "model_unavailable"
    last = s["cost"]["ledger"]["requests"][-1]
    assert last["cost_status"] == "unknown_after_failure" and "reply_used" not in last
    assert not s["cost"]["actual_spend_complete"] and kinds(doc)[-2:] == ["model_error", "stop"]
    assert av1.verify_report_v1(doc) == [] and lr.verify_saved(out) == []


@pytest.mark.parametrize("fault, reason", [("no_usage", "provider_cost_unknown"), ("no_cost", "provider_cost_unknown"),
                                           ("bad_cost", "provider_cost_unknown"), ("other_model", "unexpected_model"),
                                           ("unknown_stop", "unexpected_stop_reason")])
def test_missing_usage_or_cost_and_unexpected_responses_withhold_the_reply(inputs, capsys, fault, reason):
    code, out, emu = dry(inputs, faults={2: fault}, capsys=capsys)
    s, doc = outputs(out)
    assert len(emu.requests) == 2 and code == 1 and s["status"] == "incomplete"
    assert doc["trace"][-2] == {**doc["trace"][-2], "kind": "model_error", "reason": reason}
    assert len(doc["evidence"]) == 1                                      # the withheld reply executed nothing
    last = s["cost"]["ledger"]["requests"][-1]
    assert "reply_used" not in last
    assert s["cost"]["actual_spend_complete"] is (fault in ("other_model", "unknown_stop"))   # a paid reply stays counted
    assert lr.verify_saved(out) == []


# ---- budgets: the target, the provider-reported cost, the estimate, turns, request size ----

def test_provider_reported_spending_reaching_the_target_stops_further_requests(inputs, capsys):
    code, out, emu = dry(inputs, cost=0.25, capsys=capsys)
    s, doc = outputs(out)
    assert len(emu.requests) == 2 and s["cost"]["actual_spend_usd"] == 0.5 and not s["cost"]["over_target"]
    assert doc["trace"][-2]["reason"] == "provider_reported_budget" and code == 1 and s["status"] == "incomplete"


def test_a_last_request_that_passes_the_target_is_reported_as_over_target(inputs, capsys):
    code, out, emu = dry(inputs, cost=0.3, capsys=capsys)
    s, _ = outputs(out)
    assert len(emu.requests) == 2 and s["cost"]["actual_spend_usd"] == 0.6 and s["cost"]["target_usd"] == 0.5
    assert code == 1 and s["status"] == "over_target" and s["cost"]["over_target"]


def test_the_stage6_cost_estimate_stops_a_reply_over_budget_without_executing_it(inputs, capsys):
    code, out, emu = dry(inputs, faults={2: "huge_usage"}, cost=0.01, capsys=capsys)
    s, doc = outputs(out)
    assert len(emu.requests) == 2 and doc["stopping_reason"] in ("cost_budget", "token_budget")
    assert kinds(doc)[-2:] == ["budget_exceeded", "stop"] and len(doc["evidence"]) == 1
    assert code == 1 and s["status"] == "incomplete" and lr.verify_saved(out) == []


def test_the_turn_limit_stops_the_run(inputs, capsys):
    code, out, emu = dry(inputs, "--max-model-turns", "3", capsys=capsys)
    s, doc = outputs(out)
    assert len(emu.requests) == 3 and doc["stopping_reason"] == "max_model_turns" and s["status"] == "incomplete"
    assert lr.verify_saved(out) == []


@pytest.mark.parametrize("flags", [["--target-usd", "0.51"], ["--max-tool-calls", "13"], ["--target-usd", "nan"]])
def test_plans_beyond_the_ceiling_are_refused_before_anything_is_written(inputs, capsys, flags):
    out = inputs["tmp"] / "inv"
    assert lr.main(argv(inputs, out, *flags), emulator=Emu()) == 2 and not out.exists()
    capsys.readouterr()
    with pytest.raises(ValueError):
        lv1.plan_v1(provider="anthropic")


def test_oversized_or_foreign_requests_are_never_sent():
    class Inner:
        calls = 0

        def respond(self, *a, **k):
            Inner.calls += 1
    model = lv1.GuardedModelV1(Inner(), live.Ledger(), "m", max_request_chars=10)
    for system, tools, reason in ((av1.SYSTEM_PROMPT_V1, av1.TOOL_SPECS_V1, "request_too_large"),
                                  (ia.SYSTEM_PROMPT_V2, ia.TOOL_SPECS_V2, "not_v1_schemas")):
        with pytest.raises(ia.ModelUnavailable) as e:
            model.respond(system, tools, [{"role": "user", "content": "q"}])
        assert e.value.reason == reason
    assert Inner.calls == 0


def test_stage6_entry_points_refuse_a_v1_plan():
    plan = lv1.plan_v1()
    with pytest.raises(ValueError):
        live.dry_run(plan)
    with pytest.raises(ValueError):
        live.run_live(plan, confirm_budget_usd=0.5)


# ---- the paid path: authorization, preflight, a live run against the emulated endpoint ----

CAPPED = {"limit": 0.5, "limit_remaining": 0.5, "usage": 0.0, "limit_reset": None, "is_free_tier": False}


def metadata(key=None, pricing=None, raw=None):
    calls = []

    def get(url, headers=None):
        calls.append((url, sorted((headers or {}).keys())))
        if "/models" in url:
            return {"data": [{"id": "anthropic/claude-sonnet-5.5", "context_length": 1_000_000,
                              "top_provider": {"max_completion_tokens": 128_000},
                              "supported_parameters": ["tools", "tool_choice", "reasoning"],
                              "pricing": pricing or {"prompt": "0.000002", "completion": "0.00001",
                                                     "input_cache_read": "0.0000001",
                                                     "input_cache_write": "0.0000025"}}]}
        if url.endswith("/key"):
            return raw if raw is not None else {"data": dict(CAPPED if key is None else key)}
        raise AssertionError(url)
    get.calls = calls
    return get


AUTH = ["--live", "--confirm-budget-usd", "0.50", "--authorize-paid-run", lv1.PAID_RUN_AUTHORIZATION]


@pytest.mark.parametrize("flags", [["--live"], ["--live", "--confirm-budget-usd", "0.50"],
                                   ["--live", "--confirm-budget-usd", "0.40", "--authorize-paid-run",
                                    lv1.PAID_RUN_AUTHORIZATION],
                                   ["--live", "--confirm-budget-usd", "0.50", "--authorize-paid-run", "yes"]])
def test_a_paid_run_needs_the_confirmed_budget_and_the_separate_authorization(inputs, capsys, flags):
    get, emu, out = metadata(), Emu(), inputs["tmp"] / "inv"
    code = lr.main(argv(inputs, out, *flags), env={"OPENROUTER_API_KEY": FAKE_OR}, http_get=get,
                   http_client=httpx2.Client(transport=httpx2.MockTransport(emu)))
    assert code == 2 and not out.exists() and get.calls == [] and emu.requests == []
    assert FAKE_OR not in capsys.readouterr().err


def live_run(i, monkeypatch, get, name="live", cost=None):
    monkeypatch.setattr(live, "MANIFEST", i["manifest"])
    emu, out = Emu(cost=cost), i["tmp"] / name
    code = lr.main(argv(i, out, *AUTH), env={"OPENROUTER_API_KEY": FAKE_OR}, http_get=get,
                   http_client=httpx2.Client(transport=httpx2.MockTransport(emu)))
    return code, out, emu


def test_an_authorized_run_against_the_emulated_endpoint_uses_fetched_prices_and_writes_no_key(inputs, capsys,
                                                                                              monkeypatch):
    get = metadata()
    code, out, emu = live_run(inputs, monkeypatch, get)
    capsys.readouterr()
    s, doc = outputs(out)
    assert code == 0 and s["status"] == "verified" and s["mode"].startswith("LIVE")
    assert [x["step"] for x in s["steps"]][:3] == ["inputs", "preflight", "budget"]
    assert s["preflight"]["prices_basis"] == "fetched" and doc["budget"]["price_per_mtok"] == [2.5, 10.0]
    assert [u for u, _ in get.calls] == ["https://openrouter.ai/api/v1/models?supported_parameters=tools",
                                         "https://openrouter.ai/api/v1/key"]
    assert len(emu.requests) == 7 and lr.verify_saved(out) == []
    c = s["cost"]
    assert (c["target_usd"], c["provider_key_limit_usd"], c["over_target"]) == (0.5, 0.5, False)
    assert c["actual_spend_usd"] == s["cost"]["ledger"]["known_cost_usd"] > 0 and "ceiling_usd" not in c
    for f in out.iterdir():
        assert FAKE_OR not in f.read_text(encoding="utf-8")


def refused_before_any_model_request(code, out, emu):
    s = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert code == 2 and s["status"] == "refused" and s["steps"][-1]["step"] == "preflight"
    assert emu.requests == [] and "cost" not in s and not (out / "investigation_report.json").exists()
    return s


def test_codex_a_one_dollar_key_with_thirty_cent_responses_is_refused_before_any_model_request(inputs, capsys,
                                                                                               monkeypatch):
    get = metadata({"limit": 1.0, "limit_remaining": 0.9, "usage": 0.1, "limit_reset": None, "is_free_tier": False})
    code, out, emu = live_run(inputs, monkeypatch, get, cost=0.3)
    s = refused_before_any_model_request(code, out, emu)
    assert "above $0.50" in s["steps"][-1]["message"] and FAKE_OR not in capsys.readouterr().err


BAD_KEYS = {
    "unlimited": {**CAPPED, "limit": None, "limit_remaining": None},
    "one_dollar": {**CAPPED, "limit": 1.0, "limit_remaining": 1.0},
    "daily_reset": {**CAPPED, "limit_reset": "daily"},
    "monthly_reset": {**CAPPED, "limit_reset": "monthly"},
    "missing_limit": {k: v for k, v in CAPPED.items() if k != "limit"},
    "missing_reset": {k: v for k, v in CAPPED.items() if k != "limit_reset"},
    "missing_usage": {k: v for k, v in CAPPED.items() if k != "usage"},
    "missing_remaining": {k: v for k, v in CAPPED.items() if k != "limit_remaining"},
    "text_limit": {**CAPPED, "limit": "0.5"},
    "bool_limit": {**CAPPED, "limit": True},
    "zero_limit": {**CAPPED, "limit": 0, "limit_remaining": 0},
    "negative_usage": {**CAPPED, "usage": -0.1, "limit_remaining": 0.6},
    "remaining_above_limit": {**CAPPED, "limit_remaining": 0.6},
    "inconsistent_usage": {**CAPPED, "usage": 0.1},
    "insufficient_remaining": {**CAPPED, "usage": 0.2, "limit_remaining": 0.3},
    "bad_free_tier": {**CAPPED, "is_free_tier": "no"},
}


@pytest.mark.parametrize("case", sorted(BAD_KEYS))
def test_unsafe_or_invalid_key_limits_are_refused_before_any_model_request(inputs, capsys, monkeypatch, case):
    code, out, emu = live_run(inputs, monkeypatch, metadata(BAD_KEYS[case]))
    capsys.readouterr()
    refused_before_any_model_request(code, out, emu)
    problems, limit = lv1.key_limit_problems({"data": BAD_KEYS[case]}, 0.5)
    assert problems and limit is None


@pytest.mark.parametrize("get", [metadata(raw={"data": None}), metadata(raw=["not", "an", "object"]),
                                 metadata(pricing={"prompt": "0.000002"})])
def test_malformed_key_metadata_or_missing_prices_are_refused(inputs, capsys, monkeypatch, get):
    code, out, emu = live_run(inputs, monkeypatch, get)
    capsys.readouterr()
    refused_before_any_model_request(code, out, emu)


def test_an_unreachable_key_endpoint_is_refused(inputs, capsys, monkeypatch):
    real = metadata()

    def get(url, headers=None):
        if url.endswith("/key"):
            raise OSError("unreachable")
        return real(url, headers)
    code, out, emu = live_run(inputs, monkeypatch, get)
    capsys.readouterr()
    refused_before_any_model_request(code, out, emu)


def test_a_capped_key_with_a_lower_target_and_prior_usage_is_accepted():
    key = {**CAPPED, "usage": 0.25, "limit_remaining": 0.25}
    assert lv1.key_limit_problems({"data": key}, 0.25) == ([], 0.5)
    assert lv1.key_limit_problems({"data": key}, 0.5)[0]


def test_a_provider_that_charges_past_its_own_key_limit_is_reported_truthfully(inputs, capsys, monkeypatch):
    # the emulator ignores the key's $0.50 limit and charges $0.30 per reply
    code, out, emu = live_run(inputs, monkeypatch, metadata(), cost=0.3)
    capsys.readouterr()
    s, doc = outputs(out)
    c = s["cost"]
    assert len(emu.requests) == 2 and code == 1 and s["status"] == "over_target"
    assert (c["target_usd"], c["provider_key_limit_usd"], c["actual_spend_usd"], c["over_target"]) == (
        0.5, 0.5, 0.6, True)
    assert c["actual_spend_complete"] and doc["trace"][-2]["reason"] == "provider_reported_budget"
    assert lr.verify_saved(out) == []


# ---- saved outputs: re-verified offline, tampering detected ----

@pytest.mark.parametrize("tamper", ["ledger_cost", "ledger_used", "ledger_tokens", "markdown", "report", "budget"])
def test_saved_outputs_are_reverified_and_tampering_is_detected(inputs, capsys, tamper):
    _, out, _ = dry(inputs, capsys=capsys)
    s, doc = outputs(out)
    reqs = s["cost"]["ledger"]["requests"]
    if tamper == "ledger_cost":
        reqs[0]["cost_usd"] = 0.0
    elif tamper == "ledger_used":
        del reqs[3]["reply_used"]
    elif tamper == "ledger_tokens":
        reqs[1]["input_tokens"] += 1
    elif tamper == "budget":
        s["budget"]["max_cost_usd"] = 5.0
    if tamper.startswith("ledger") or tamper == "budget":
        (out / "run.json").write_text(json.dumps(s))
    elif tamper == "markdown":
        p = out / "investigation_report.md"
        p.write_text(p.read_text(encoding="utf-8").replace("insufficient_evidence", "effect_described"))
    else:
        doc["usage"]["input_tokens"] -= 1
        (out / "investigation_report.json").write_text(json.dumps(doc))
    assert lr.verify_saved(out) != []
    assert lr.main(["--verify-saved", str(out)]) == 1
    capsys.readouterr()


def test_the_ledger_check_binds_turns_to_attempts():
    doc = {"trace": [{"kind": "model_turn", "input_tokens": 10, "output_tokens": 2}, {"kind": "stop"}],
           "stopping_reason": "max_model_turns"}
    entry = {"model_requested": "anthropic/claude-sonnet-5.5", "max_tokens": 4000, "dropped_fields": [],
             "cost_usd": 0.01, "cost_status": "reported", "input_tokens": 10, "output_tokens": 2, "reply_used": True}
    ledger = {"requests": [entry], "known_cost_usd": 0.01, "complete": True, "unknown_cost_attempts": 0,
              "reported_cost_usd": 0.01}
    plan = lv1.plan_v1()
    assert lv1.ledger_problems(doc, ledger, plan) == []
    extra = copy.deepcopy(ledger)
    extra["requests"].append({**entry, "reply_used": False})
    extra.update(known_cost_usd=0.02, reported_cost_usd=0.02)
    assert lv1.ledger_problems(doc, extra, plan)                 # an unused attempt without a model_error stop
    assert lv1.ledger_problems(doc, {**ledger, "known_cost_usd": 0.0}, plan)
    assert lv1.ledger_problems(doc, {"requests": None}, plan)


# ---- compatibility with a recorded Phase 2 run (reused, fail closed) ----

def test_a_compatible_recorded_run_is_verified_against_the_same_query(inputs, capsys):
    runs = recorded_run(inputs, "phase2", inputs["tmp"])
    code, out, _ = dry(inputs, "--validation-run", "phase2", "--runs-dir", str(runs), capsys=capsys)
    s, _ = outputs(out)
    assert code == 0 and s["report"]["validation_run"]["same_query"] is True
    assert s["steps"][-1]["step"] == "compatibility"


def test_offline_tests_read_no_real_data_and_hash_their_own_fixtures(inputs):
    assert inputs["tmp"] in Path(inputs["zip"]).parents and inputs["tmp"] in Path(inputs["cache"]).parents
    assert hashlib.sha256(Path(inputs["zip"]).read_bytes()).hexdigest() == inputs["sha"]
