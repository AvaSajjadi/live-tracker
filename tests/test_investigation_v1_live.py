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
import urllib.error
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
FAKE_MGMT = "sk-or-v1-FAKE-management-key-5555555555"
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


ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/anthropic/claude-sonnet-5.5/endpoints"
LISTED_PRICING = {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000001",
                  "input_cache_write": "0.0000025"}


def anthropic_endpoint(**kw):
    return {"provider_name": "Anthropic", "tag": "anthropic", "pricing": dict(LISTED_PRICING),
            "context_length": 200_000, "max_prompt_tokens": None, "max_completion_tokens": 64_000, "status": 0, **kw}


def forbidden(url="https://openrouter.ai/api/v1/models/x/endpoints"):
    return urllib.error.HTTPError(url, 403, "Forbidden", {}, None)


def endpoint_list(*endpoints):
    eps = endpoints or (anthropic_endpoint(),)
    return {"data": {"id": "anthropic/claude-sonnet-5.5", "endpoints": list(eps)}}


def metadata(key=None, pricing=None, raw=None, endpoints=None, management_only=False):
    """Injected metadata GETs. management_only: the endpoint list answers 403 unless the management key is sent."""
    calls, auth = [], []

    def get(url, headers=None):
        calls.append((url, sorted((headers or {}).keys())))
        auth.append((url, (headers or {}).get("Authorization")))
        if url == ENDPOINTS_URL:
            if isinstance(endpoints, Exception):
                raise endpoints
            if management_only and (headers or {}).get("Authorization") != f"Bearer {FAKE_MGMT}":
                raise forbidden(url)
            return endpoints if endpoints is not None else endpoint_list()
        if "/models" in url:
            return {"data": [{"id": "anthropic/claude-sonnet-5.5", "context_length": 1_000_000,
                              "top_provider": {"max_completion_tokens": 128_000},
                              "supported_parameters": ["tools", "tool_choice", "reasoning"],
                              "pricing": pricing or dict(LISTED_PRICING)}]}
        if url.endswith("/key"):
            return raw if raw is not None else {"data": dict(CAPPED if key is None else key)}
        raise AssertionError(url)
    get.calls, get.auth = calls, auth
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


def live_run(i, monkeypatch, get, name="live", cost=None, env=None, extra=(), faults=None):
    monkeypatch.setattr(live, "MANIFEST", i["manifest"])
    emu, out = Emu(faults, cost=cost), i["tmp"] / name
    sent_auth = []

    def handler(request):                       # the Authorization each model request carries (fake keys only)
        sent_auth.append(request.headers.get("authorization"))
        return emu(request)
    code = lr.main(argv(i, out, *AUTH, *extra), env={"OPENROUTER_API_KEY": FAKE_OR} if env is None else env,
                   http_get=get,
                   http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    emu.sent_auth = sent_auth
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
                                         "https://openrouter.ai/api/v1/key", ENDPOINTS_URL]
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


# ---- the standalone read-only preflight (--preflight-only) ----

@pytest.fixture
def no_model(monkeypatch):
    """Any model request or investigation during a preflight fails the test."""
    sent = []

    def create(self, **kw):
        sent.append(kw)
        raise AssertionError("a model request was attempted")

    def investigate(*a, **k):
        raise AssertionError("an investigation was started")
    monkeypatch.setattr(live.GuardedClient, "create", create)
    monkeypatch.setattr(av1, "investigate_v1", investigate)
    return sent


def preflight(i, monkeypatch, capsys, get, *extra, env=None):
    monkeypatch.setattr(live, "MANIFEST", i["manifest"])
    before = sorted(p.name for p in i["tmp"].iterdir())
    code = lr.main(["--preflight-only", "--manifest", str(i["manifest"]), "--cache-dir", str(i["cache"]), *extra],
                   env={"OPENROUTER_API_KEY": FAKE_OR} if env is None else env, http_get=get)
    io = capsys.readouterr()
    assert FAKE_OR not in io.out + io.err and FAKE_MGMT not in io.out + io.err
    assert sorted(p.name for p in i["tmp"].iterdir()) == before                                       # no file
    return code, (json.loads(io.out) if io.out.strip() else None), io.err


def test_a_capped_key_passes_the_read_only_preflight_without_any_model_request(inputs, monkeypatch, capsys,
                                                                               no_model):
    get = metadata()
    code, r, _ = preflight(inputs, monkeypatch, capsys, get)
    assert code == 0 and r["ready"] and r["problems"] == [] and no_model == []
    assert [u for u, _ in get.calls] == ["https://openrouter.ai/api/v1/models?supported_parameters=tools",
                                         "https://openrouter.ai/api/v1/key", ENDPOINTS_URL]
    assert get.auth == [(get.calls[0][0], None), ("https://openrouter.ai/api/v1/key", f"Bearer {FAKE_OR}"),
                        (ENDPOINTS_URL, f"Bearer {FAKE_OR}")]       # no management key: the inference key
    assert r["credentials"]["endpoint_metadata_get"] == "OPENROUTER_API_KEY"
    m = r["confirmed_provider_metadata"]
    assert m["key"] == {k: v for k, v in CAPPED.items()} and m["provider_key_limit_usd"] == 0.5
    assert m["prices_basis"] == "fetched" and m["prices_usd_per_mtok"] == {"input": 2.5, "output": 10.0}
    assert m["anthropic_endpoint"]["provider_name"] == "Anthropic"
    req = r["local_configuration"]["request"]
    assert req["routing"] == {"order": ["anthropic"], "allow_fallbacks": False} and req["requests_sent"] == 0
    assert req["messages_url"] == "https://openrouter.ai/api/v1/messages" and req["sdk_max_retries"] == 0
    assert req["retries_when_re_requested"] == 0 and req["adapter_fallbacks"] is False
    assert not {"betas", "fallbacks", "thinking"} & set(req["sent_fields"]) and req["beta_endpoint_closed"]
    e = r["estimate"]
    assert e["estimated_worst_case_usd"] <= 0.5 and e["target_usd"] == 0.5 and e["max_request_chars"] > 0
    assert any("usage.cost" in a for a in r["requires_a_model_response"])
    assert r["mode"].startswith("read-only")


@pytest.mark.parametrize("key, why", [
    ({**CAPPED, "limit": 1.0, "limit_remaining": 0.9, "usage": 0.1}, "above $0.50"),
    ({**CAPPED, "limit": None, "limit_remaining": None}, "unlimited"),
    ({**CAPPED, "limit_reset": "monthly"}, "resets automatically"),
    ({**CAPPED, "usage": 0.2, "limit_remaining": 0.3}, "does not cover"),
    ({k: v for k, v in CAPPED.items() if k != "limit_reset"}, "lacks"),
])
def test_incorrect_key_limits_fail_the_preflight(inputs, monkeypatch, capsys, no_model, key, why):
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(key))
    assert code == 2 and not r["ready"] and any(why in p for p in r["problems"]) and no_model == []
    assert r["confirmed_provider_metadata"]["provider_key_limit_usd"] is None


@pytest.mark.parametrize("pricing, endpoints, why", [
    ({"prompt": "0.000002"}, None, "pricing"),
    ({**LISTED_PRICING, "request": "0.01"}, None, "pricing"),
    (None, endpoint_list({"provider_name": "Anthropic", "tag": "anthropic",
                          "pricing": {**LISTED_PRICING, "completion": "0.00002"}}), "priced above"),
    (None, endpoint_list({"provider_name": "Anthropic", "tag": "anthropic", "pricing": {"prompt": "x"}}),
     "not usable"),
    (None, endpoint_list({"provider_name": "Anthropic", "tag": "anthropic", "pricing": dict(LISTED_PRICING),
                          "max_completion_tokens": 1000}), "output allowance"),
])
def test_invalid_pricing_fails_the_preflight(inputs, monkeypatch, capsys, no_model, pricing, endpoints, why):
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(pricing=pricing, endpoints=endpoints))
    assert code == 2 and any(why in p for p in r["problems"]) and no_model == []


@pytest.mark.parametrize("fail, why", [
    ("/models?", "model_list"), ("/key", "key_status"), ("/endpoints", "endpoint list is unavailable")])
def test_unreachable_provider_metadata_fails_the_preflight(inputs, monkeypatch, capsys, no_model, fail, why):
    real = metadata()

    def get(url, headers=None):
        if fail in url:
            raise OSError("unreachable")
        return real(url, headers)
    code, r, _ = preflight(inputs, monkeypatch, capsys, get)
    assert code == 2 and any(why in p for p in r["problems"]) and no_model == []


@pytest.mark.parametrize("endpoints, why", [
    ({"data": None}, "malformed"), ({"data": {"endpoints": "x"}}, "malformed"),
    (endpoint_list({"provider_name": "Google Vertex", "tag": "google-vertex", "pricing": dict(LISTED_PRICING)}),
     "0 Anthropic endpoints"),
    (endpoint_list(*[{"provider_name": "Anthropic", "tag": "anthropic", "pricing": dict(LISTED_PRICING)}] * 2),
     "2 Anthropic endpoints")])
def test_a_model_without_exactly_one_anthropic_endpoint_fails_the_preflight(inputs, monkeypatch, capsys, no_model,
                                                                           endpoints, why):
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(endpoints=endpoints))
    assert code == 2 and any(why in p for p in r["problems"]) and no_model == []


def test_missing_configuration_fails_the_preflight(inputs, monkeypatch, capsys, no_model):
    get = metadata()
    code, r, _ = preflight(inputs, monkeypatch, capsys, get, env={})
    assert code == 2 and "api_key_present" in r["problems"] and "the key metadata was not received" in r["problems"]
    assert all(h == [] for _, h in get.calls)                                # no key: no authenticated GET
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "x-extra: 1")
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata())
    assert code == 2 and "clean_ambient_environment" in r["problems"]
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS")
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(), "--cache-dir", str(inputs["tmp"] / "absent"))
    assert code == 2 and "validation_cache" in r["problems"]
    code, r, err = preflight(inputs, monkeypatch, capsys, metadata(), "--target-usd", "0.75")
    assert code == 2 and r is None and "at most 0.50" in err
    assert no_model == []


def test_the_preflight_refuses_another_manifest(inputs, monkeypatch, capsys, no_model):
    get = metadata()
    code = lr.main(["--preflight-only", "--manifest", str(inputs["manifest"]), "--cache-dir", str(inputs["cache"])],
                   env={"OPENROUTER_API_KEY": FAKE_OR}, http_get=get)       # live.MANIFEST is the tracked one
    r = json.loads(capsys.readouterr().out)
    assert code == 2 and "tracked split manifest" in r["problems"][0] and get.calls == [] and no_model == []


def test_a_paid_run_uses_the_same_preflight_checks(inputs, capsys, monkeypatch):
    code, out, emu = live_run(inputs, monkeypatch, metadata(endpoints=endpoint_list(
        {"provider_name": "Google Vertex", "tag": "google-vertex", "pricing": dict(LISTED_PRICING)})))
    capsys.readouterr()
    s = refused_before_any_model_request(code, out, emu)
    assert "Anthropic endpoints" in s["steps"][-1]["message"]


# ---- endpoint capacity (context, prompt and completion tokens) ----

def planned(i):
    plan = lv1.plan_v1(cache_dir=str(i["cache"]))
    est = lv1.budget_v1(plan, live.Prices.from_openrouter(LISTED_PRICING, "test"))[1]
    return plan, est, est["worst_request_input_tokens"], plan.per_request_max_tokens


def test_codex_a_one_token_context_is_not_ready(inputs, monkeypatch, capsys, no_model):
    get = metadata(endpoints=endpoint_list(anthropic_endpoint(context_length=1, max_completion_tokens=64_000)))
    code, r, _ = preflight(inputs, monkeypatch, capsys, get)
    assert code == 2 and not r["ready"] and any("context_length 1 is below" in p for p in r["problems"])
    assert no_model == []


def test_capacity_counts_planned_input_and_reserved_output(inputs):
    plan, est, need_in, need_out = planned(inputs)
    ok = lambda **kw: lv1.capacity_problems(anthropic_endpoint(**kw), plan, est)[0]    # noqa: E731
    assert ok() == [] and ok(context_length=need_in + need_out) == []
    assert ok(context_length=need_in + need_out - 1) and ok(context_length=need_in)    # the output is reserved too
    assert ok(max_prompt_tokens=need_in) == [] and ok(max_prompt_tokens=need_in - 1)
    assert ok(max_completion_tokens=need_out) == [] and ok(max_completion_tokens=need_out - 1)
    assert lv1.capacity_problems(anthropic_endpoint(), plan, None)[0]                     # no estimate: unverifiable


@pytest.mark.parametrize("fields", [{"context_length": None}, {"context_length": "200000"},
                                    {"context_length": 200_000.0}, {"context_length": True},
                                    {"context_length": 0}, {"context_length": -5}, {"max_prompt_tokens": "x"},
                                    {"max_prompt_tokens": 0}, {"max_completion_tokens": True},
                                    {"max_completion_tokens": 1.5}])
def test_missing_or_malformed_capacity_fails_the_preflight(inputs, monkeypatch, capsys, no_model, fields):
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(endpoints=endpoint_list(anthropic_endpoint(**fields))))
    assert code == 2 and any("capacity cannot be verified" in p or "malformed" in p for p in r["problems"])
    no_context = {k: v for k, v in anthropic_endpoint().items() if k != "context_length"}
    assert lv1.capacity_problems(no_context, *planned(inputs)[:2])[0]


def test_paid_mode_refuses_insufficient_capacity_before_any_model_request(inputs, capsys, monkeypatch):
    get = metadata(endpoints=endpoint_list(anthropic_endpoint(context_length=1)))
    code, out, emu = live_run(inputs, monkeypatch, get)
    capsys.readouterr()
    s = refused_before_any_model_request(code, out, emu)
    assert "context_length 1 is below" in s["steps"][-1]["message"]


# ---- endpoint-list authentication and credential separation ----

def test_a_documented_403_without_a_management_key_fails_closed(inputs, monkeypatch, capsys, no_model):
    get = metadata(management_only=True)
    code, r, _ = preflight(inputs, monkeypatch, capsys, get)
    assert code == 2 and r["confirmed_provider_metadata"]["anthropic_endpoint"] is None
    assert any("HTTP 403" in p and "OPENROUTER_MANAGEMENT_KEY" in p for p in r["problems"]) and no_model == []
    assert [u for u, _ in get.auth].count(ENDPOINTS_URL) == 1                    # no retry, no unauthenticated GET


def test_a_management_key_is_used_only_for_the_endpoint_list(inputs, monkeypatch, capsys, no_model):
    get = metadata(management_only=True)
    env = {"OPENROUTER_API_KEY": FAKE_OR, "OPENROUTER_MANAGEMENT_KEY": FAKE_MGMT}
    code, r, _ = preflight(inputs, monkeypatch, capsys, get, env=env)
    assert code == 0 and r["ready"] and no_model == []
    assert dict(get.auth) == {get.calls[0][0]: None, "https://openrouter.ai/api/v1/key": f"Bearer {FAKE_OR}",
                              ENDPOINTS_URL: f"Bearer {FAKE_MGMT}"}
    assert r["credentials"]["endpoint_metadata_get"] == "OPENROUTER_MANAGEMENT_KEY"


def test_a_403_with_the_management_key_fails_closed(inputs, monkeypatch, capsys, no_model):
    env = {"OPENROUTER_API_KEY": FAKE_OR, "OPENROUTER_MANAGEMENT_KEY": FAKE_MGMT}
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(endpoints=forbidden()), env=env)
    assert code == 2 and any("refused even with the management key" in p for p in r["problems"])


@pytest.mark.parametrize("env, why", [
    ({"OPENROUTER_API_KEY": FAKE_OR, "OPENROUTER_MANAGEMENT_KEY": FAKE_OR}, "must be different keys"),
    ({"OPENROUTER_MANAGEMENT_KEY": FAKE_MGMT}, "api_key_present")])
def test_the_management_key_never_stands_in_for_the_inference_key(inputs, monkeypatch, capsys, no_model, env, why):
    get = metadata()
    code, r, _ = preflight(inputs, monkeypatch, capsys, get, env=env)
    assert code == 2 and any(why in p for p in r["problems"]) and no_model == []
    assert all(a != f"Bearer {FAKE_MGMT}" for u, a in get.auth if u != ENDPOINTS_URL)   # never on /key


def test_a_paid_run_sends_only_the_inference_key_to_the_model_and_writes_neither_key(inputs, capsys, monkeypatch):
    env = {"OPENROUTER_API_KEY": FAKE_OR, "OPENROUTER_MANAGEMENT_KEY": FAKE_MGMT}
    get = metadata(management_only=True)
    built = []
    real = live.build_model

    def spy(plan, *, env=None, **kw):
        built.append(dict(env or {}))
        return real(plan, env=env, **kw)
    monkeypatch.setattr(live, "build_model", spy)
    code, out, emu = live_run(inputs, monkeypatch, get, env=env)
    io = capsys.readouterr()
    s, _ = outputs(out)
    assert code == 0 and s["status"] == "verified" and len(emu.requests) == 7
    assert set(emu.sent_auth) == {f"Bearer {FAKE_OR}"}                          # every model request
    assert all(FAKE_MGMT not in json.dumps(e) for e in built)                  # the adapter never had it
    assert s["preflight"]["endpoint_metadata_credential"] == "OPENROUTER_MANAGEMENT_KEY"
    assert FAKE_MGMT not in io.out + io.err and FAKE_OR not in io.out + io.err
    for f in out.iterdir():
        text = f.read_text(encoding="utf-8")
        assert FAKE_OR not in text and FAKE_MGMT not in text


def test_a_paid_run_refuses_a_403_endpoint_list_before_any_model_request(inputs, capsys, monkeypatch):
    code, out, emu = live_run(inputs, monkeypatch, metadata(management_only=True))
    capsys.readouterr()
    s = refused_before_any_model_request(code, out, emu)
    assert "HTTP 403" in s["steps"][-1]["message"] and FAKE_OR not in json.dumps(s)


# ---- metadata GETs never follow redirects (no credential forwarded) ----

MODELS_URL = "https://openrouter.ai/api/v1/models?supported_parameters=tools"
KEY_URL = "https://openrouter.ai/api/v1/key"


class FakeServer:
    """An offline urllib transport (no socket): answers by URL with (status, body, Location) and records every
    request's URL and Authorization header (fake keys only)."""

    def __init__(self, **override):
        m = metadata()
        self.routes = {MODELS_URL: (200, m(MODELS_URL), None), KEY_URL: (200, m(KEY_URL, {}), None),
                       ENDPOINTS_URL: (200, m(ENDPOINTS_URL), None),
                       "https://other.example/collect": (200, {"collected": True}, None),
                       "http://openrouter.ai/api/v1/key": (200, {"collected": True}, None),
                       "https://openrouter.ai/api/v1/elsewhere": (200, {"collected": True}, None)}
        self.routes.update(override)
        self.seen = []

    def answer(self, req):
        import email.message
        import io
        import urllib.response
        self.seen.append((req.full_url, req.get_header("Authorization")))
        code, body, location = self.routes.get(req.full_url, (404, {}, None))
        headers = email.message.Message()
        headers["Content-Type"] = "application/json"
        if location:
            headers["Location"] = location
        r = urllib.response.addinfourl(io.BytesIO(json.dumps(body).encode()), headers, req.full_url, code)
        r.msg = "emulated"
        return r

    def handlers(self):
        import urllib.request
        server = self

        class HTTPS(urllib.request.HTTPSHandler):
            def https_open(self, req):
                return server.answer(req)

        class HTTP(urllib.request.HTTPHandler):
            def http_open(self, req):
                return server.answer(req)
        return HTTPS(), HTTP()

    def get(self):
        import functools
        return functools.partial(lv1.get_json_no_redirects, handlers=self.handlers())


def redirect(location, code=302):
    return (code, {}, location)


def test_codex_a_cross_origin_302_does_not_forward_the_management_key():
    server = FakeServer(**{ENDPOINTS_URL: redirect("https://other.example/collect")})
    with pytest.raises(lv1.RedirectRefused) as e:
        server.get()(ENDPOINTS_URL, {"Authorization": "Bearer FAKE-MANAGEMENT"})
    assert e.value.code == 302 and "other.example" not in str(e.value)
    assert server.seen == [(ENDPOINTS_URL, "Bearer FAKE-MANAGEMENT")]            # nothing sent to other.example


def test_the_plain_urllib_opener_would_have_forwarded_it():
    import urllib.request
    server = FakeServer(**{ENDPOINTS_URL: redirect("https://other.example/collect")})
    req = urllib.request.Request(ENDPOINTS_URL, headers={"Authorization": "Bearer FAKE-MANAGEMENT"})
    urllib.request.build_opener(*server.handlers()).open(req).close()             # Stage 8's _get_json behaviour
    assert server.seen[-1] == ("https://other.example/collect", "Bearer FAKE-MANAGEMENT")


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("location", ["https://other.example/collect", "https://openrouter.ai/api/v1/elsewhere",
                                      "http://openrouter.ai/api/v1/key"])
def test_no_redirect_is_followed_same_origin_cross_origin_or_to_http(code, location):
    for url, key in ((ENDPOINTS_URL, f"Bearer {FAKE_MGMT}"), (KEY_URL, f"Bearer {FAKE_OR}")):
        server = FakeServer(**{url: redirect(location, code)})
        with pytest.raises(lv1.RedirectRefused):
            server.get()(url, {"Authorization": key})
        assert server.seen == [(url, key)]


@pytest.mark.parametrize("url", ["https://other.example/api/v1/key", "http://openrouter.ai/api/v1/key",
                                 "https://openrouter.ai.other.example/api/v1/key"])
def test_metadata_requests_go_only_to_openrouter_over_https(url):
    server = FakeServer()
    with pytest.raises(ValueError):
        server.get()(url, {"Authorization": f"Bearer {FAKE_OR}"})
    assert server.seen == []


def test_normal_metadata_answers_still_succeed_and_errors_keep_their_status():
    server = FakeServer()
    assert server.get()(KEY_URL, {"Authorization": f"Bearer {FAKE_OR}"}) == {"data": CAPPED}
    assert server.get()(MODELS_URL)["data"][0]["id"] == "anthropic/claude-sonnet-5.5"
    server = FakeServer(**{ENDPOINTS_URL: (403, {}, None)})
    with pytest.raises(urllib.error.HTTPError) as e:
        server.get()(ENDPOINTS_URL, {"Authorization": f"Bearer {FAKE_OR}"})
    assert e.value.code == 403
    server = FakeServer(**{KEY_URL: (204, {}, None)})
    with pytest.raises(RuntimeError):
        server.get()(KEY_URL, {"Authorization": f"Bearer {FAKE_OR}"})


def default_get(monkeypatch, server):
    """The preflight's own default GET (http_get not injected), over the offline transport."""
    monkeypatch.setattr(lv1, "get_json_no_redirects", server.get())


def test_the_default_preflight_get_succeeds_with_credentials_isolated(inputs, monkeypatch, capsys, no_model):
    server = FakeServer()
    default_get(monkeypatch, server)
    env = {"OPENROUTER_API_KEY": FAKE_OR, "OPENROUTER_MANAGEMENT_KEY": FAKE_MGMT}
    code, r, _ = preflight(inputs, monkeypatch, capsys, None, env=env)
    assert code == 0 and r["ready"] and no_model == []
    assert server.seen == [(MODELS_URL, None), (KEY_URL, f"Bearer {FAKE_OR}"), (ENDPOINTS_URL, f"Bearer {FAKE_MGMT}")]


@pytest.mark.parametrize("url, location", [(ENDPOINTS_URL, "https://other.example/collect"),
                                           (KEY_URL, "https://other.example/collect"),
                                           (KEY_URL, "http://openrouter.ai/api/v1/key"),
                                           (ENDPOINTS_URL, "https://openrouter.ai/api/v1/elsewhere")])
def test_a_redirect_fails_the_preflight_without_forwarding_any_key(inputs, monkeypatch, capsys, no_model, url,
                                                                  location):
    server = FakeServer(**{url: redirect(location)})
    default_get(monkeypatch, server)
    env = {"OPENROUTER_API_KEY": FAKE_OR, "OPENROUTER_MANAGEMENT_KEY": FAKE_MGMT}
    code, r, _ = preflight(inputs, monkeypatch, capsys, None, env=env)
    assert code == 2 and not r["ready"] and no_model == []
    if url == KEY_URL:
        assert "key_status" in r["problems"] and "the key metadata was not received" in r["problems"]
    else:
        assert any("RedirectRefused, HTTP 302" in p for p in r["problems"])
    assert location not in [u for u, _ in server.seen]                          # never followed
    assert server.seen == [(MODELS_URL, None), (KEY_URL, f"Bearer {FAKE_OR}"), (ENDPOINTS_URL, f"Bearer {FAKE_MGMT}")]


@pytest.mark.parametrize("url", [ENDPOINTS_URL, KEY_URL])
def test_paid_mode_refuses_a_redirect_before_any_model_request(inputs, capsys, monkeypatch, url):
    server = FakeServer(**{url: redirect("https://other.example/collect")})
    default_get(monkeypatch, server)
    env = {"OPENROUTER_API_KEY": FAKE_OR, "OPENROUTER_MANAGEMENT_KEY": FAKE_MGMT}
    code, out, emu = live_run(inputs, monkeypatch, None, env=env)
    io = capsys.readouterr()
    s = refused_before_any_model_request(code, out, emu)
    assert "https://other.example/collect" not in [u for u, _ in server.seen]
    assert FAKE_OR not in json.dumps(s) + io.out + io.err and FAKE_MGMT not in json.dumps(s) + io.out + io.err


# ---- best-effort mode for an existing higher-limit key (explicit opt-in) ----

HACKATHON = {"limit": 50.0, "limit_remaining": 49.0, "usage": 1.0, "limit_reset": None, "is_free_tier": False}
BEST_EFFORT = ("--key-limit-mode", "best-effort", "--acknowledge-key-limit-usd", "50")


def test_strict_mode_is_the_default_and_still_accepts_a_capped_key(inputs, monkeypatch, capsys, no_model):
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata())
    sp = r["spending"]
    assert code == 0 and sp["authorization_mode"] == "strict" and sp["provider_key_limit_usd"] == 0.5
    assert sp["acknowledged_key_limit_usd"] is None and sp["paid_run_authorized"] is False


def test_a_fifty_dollar_key_is_refused_without_explicit_best_effort_mode(inputs, monkeypatch, capsys, no_model):
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(HACKATHON))
    assert code == 2 and any("above $0.50" in p and "best-effort" in p for p in r["problems"]) and no_model == []


def test_a_fifty_dollar_key_is_accepted_with_its_limit_acknowledged(inputs, monkeypatch, capsys, no_model):
    get = metadata(HACKATHON)
    code, r, _ = preflight(inputs, monkeypatch, capsys, get, *BEST_EFFORT)
    sp = r["spending"]
    assert code == 0 and r["ready"] and no_model == []
    assert (sp["authorization_mode"], sp["local_target_usd"], sp["provider_key_limit_usd"],
            sp["provider_key_remaining_usd"], sp["acknowledged_key_limit_usd"]) == ("best_effort", 0.5, 50.0, 49.0, 50.0)
    assert "NOT a spending ceiling" in sp["exposure"] and "$49.0 remaining" in sp["exposure"]
    assert sp["paid_run_authorized"] is False and r["plan"]["target_usd"] == 0.5
    assert [u for u, _ in get.calls] == [MODELS_URL, KEY_URL, ENDPOINTS_URL]           # metadata GETs only


@pytest.mark.parametrize("key, ack, why", [
    (HACKATHON, "49", "is not the key's verified limit"),
    (HACKATHON, "0.5", "is not the key's verified limit"),
    ({**HACKATHON, "limit": None, "limit_remaining": None}, "50", "unlimited"),
    ({**HACKATHON, "limit": 60.0, "limit_remaining": 59.0}, "50", "above $50.00"),
    ({**HACKATHON, "limit_reset": "monthly"}, "50", "resets automatically"),
    ({**HACKATHON, "limit_remaining": 48.0}, "50", "inconsistent"),
    ({**HACKATHON, "usage": 49.8, "limit_remaining": 0.2}, "50", "does not cover"),
])
def test_best_effort_still_refuses_wrong_acknowledgments_and_unsafe_keys(inputs, monkeypatch, capsys, no_model, key,
                                                                         ack, why):
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(key), "--key-limit-mode", "best-effort",
                           "--acknowledge-key-limit-usd", ack)
    assert code == 2 and any(why in p for p in r["problems"]) and no_model == []
    assert r["spending"]["provider_key_limit_usd"] is None


@pytest.mark.parametrize("flags", [("--key-limit-mode", "best-effort"),
                                   ("--key-limit-mode", "best-effort", "--acknowledge-key-limit-usd", "60"),
                                   ("--key-limit-mode", "best-effort", "--acknowledge-key-limit-usd", "nan"),
                                   ("--key-limit-mode", "best-effort", "--acknowledge-key-limit-usd", "0"),
                                   ("--acknowledge-key-limit-usd", "50")])
def test_invalid_mode_options_are_refused_before_any_request(inputs, monkeypatch, capsys, no_model, flags):
    get = metadata(HACKATHON)
    code, r, err = preflight(inputs, monkeypatch, capsys, get, *flags)
    assert code == 2 and r is None and "acknowledge-key-limit-usd" in err and get.calls == []


@pytest.mark.parametrize("flags", [["--live"], ["--live", "--confirm-budget-usd", "0.50"],
                                   ["--live", "--confirm-budget-usd", "0.50", "--authorize-paid-run", "yes"]])
def test_best_effort_mode_never_authorizes_a_paid_run_by_itself(inputs, capsys, flags):
    get, emu, out = metadata(HACKATHON), Emu(), inputs["tmp"] / "inv"
    code = lr.main(argv(inputs, out, *flags, *BEST_EFFORT), env={"OPENROUTER_API_KEY": FAKE_OR}, http_get=get,
                   http_client=httpx2.Client(transport=httpx2.MockTransport(emu)))
    capsys.readouterr()
    assert code == 2 and not out.exists() and get.calls == [] and emu.requests == []


def best_effort_run(i, monkeypatch, capsys, **kw):
    code, out, emu = live_run(i, monkeypatch, metadata(HACKATHON), extra=BEST_EFFORT, **kw)
    io = capsys.readouterr()
    assert FAKE_OR not in io.out + io.err
    s, doc = outputs(out)
    return code, out, emu, s, doc


def test_the_local_target_is_enforced_best_effort_on_a_fifty_dollar_key(inputs, capsys, monkeypatch):
    code, out, emu, s, doc = best_effort_run(inputs, monkeypatch, capsys, cost=0.25)
    c = s["cost"]
    assert len(emu.requests) == 2 and doc["trace"][-2]["reason"] == "provider_reported_budget"
    assert (c["authorization_mode"], c["target_usd"], c["provider_key_limit_usd"], c["provider_key_remaining_usd"],
            c["actual_spend_usd"], c["over_target"]) == ("best_effort", 0.5, 50.0, 49.0, 0.5, False)
    assert "NOT a spending ceiling" in c["exposure"] and "ceiling_usd" not in c
    assert s["key_policy"] == {"mode": "best_effort", "acknowledged_limit_usd": 50.0}
    assert code == 1 and s["status"] == "incomplete" and s["publication"] == "not_authorized"
    assert lr.verify_saved(out) == []


def test_a_sixty_cent_overrun_on_a_fifty_dollar_key_is_recorded_truthfully(inputs, capsys, monkeypatch):
    code, out, emu, s, _ = best_effort_run(inputs, monkeypatch, capsys, cost=0.3)
    c = s["cost"]
    assert len(emu.requests) == 2 and code == 1 and s["status"] == "over_target"
    assert (c["actual_spend_usd"], c["over_target"], c["actual_spend_complete"], c["provider_key_limit_usd"]) == (
        0.6, True, True, 50.0)
    assert lr.verify_saved(out) == []


def test_a_missing_cost_stops_a_best_effort_run(inputs, capsys, monkeypatch):
    code, out, emu, s, doc = best_effort_run(inputs, monkeypatch, capsys, faults={2: "no_cost"})
    assert len(emu.requests) == 2 and doc["trace"][-2]["reason"] == "provider_cost_unknown"
    assert code == 1 and s["status"] == "incomplete" and not s["cost"]["actual_spend_complete"]
    assert lr.verify_saved(out) == []


def test_a_complete_best_effort_run_verifies(inputs, capsys, monkeypatch):
    code, out, emu, s, doc = best_effort_run(inputs, monkeypatch, capsys)
    assert code == 0 and s["status"] == "verified" and len(emu.requests) == 7 and doc["finished"]
    assert s["preflight"]["key_policy"] == s["key_policy"] and lr.verify_saved(out) == []


@pytest.mark.parametrize("tamper", ["policy_to_strict", "ack", "preflight_limit", "cost_mode", "exposure",
                                    "key_status_limit", "remaining"])
def test_tampered_best_effort_records_are_rejected(inputs, capsys, monkeypatch, tamper):
    _, out, _, s, _ = best_effort_run(inputs, monkeypatch, capsys)
    if tamper == "policy_to_strict":
        s["key_policy"] = {"mode": "strict", "acknowledged_limit_usd": None}
    elif tamper == "ack":
        s["key_policy"]["acknowledged_limit_usd"] = 40.0
    elif tamper == "preflight_limit":
        s["preflight"]["provider_key_limit_usd"] = 0.5
    elif tamper == "cost_mode":
        s["cost"]["authorization_mode"] = "strict"
    elif tamper == "exposure":
        s["cost"]["exposure"] = "strict: capped at $0.50"
    elif tamper == "key_status_limit":
        s["preflight"]["key_status"]["limit"] = 0.5
    else:
        s["cost"]["provider_key_remaining_usd"] = 0.5
    (out / "run.json").write_text(json.dumps(s))
    assert lr.verify_saved(out) != []


# ---- saved authorization integrity: the key policy is bound to the preflight evidence ----

def rewrite(out, s, policy=None, limit=None, remaining=None):
    """Save an edited run.json with its editable summaries recomputed as a forger would: the cost summary from the
    ledger and the given mode and limits, and every recorded output hash."""
    plan = live.LivePlan(**s["plan"])
    if policy is not None:
        s["cost"] = lv1.cost_summary(s["cost"]["ledger"], plan, limit, policy, remaining)
    s["outputs"] = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in s["outputs"]}
    (out / "run.json").write_text(json.dumps(s))


def strict_run(i, monkeypatch, capsys):
    code, out, emu = live_run(i, monkeypatch, metadata(), name="strict")
    capsys.readouterr()
    assert code == 0
    return out, outputs(out)[0]


def test_codex_deleting_the_preflight_and_switching_to_strict_is_rejected(inputs, capsys, monkeypatch):
    _, out, _, s, _ = best_effort_run(inputs, monkeypatch, capsys)
    del s["preflight"]
    s["key_policy"] = {"mode": "strict", "acknowledged_limit_usd": None}
    rewrite(out, s, lv1.STRICT, None, None)
    problems = lr.verify_saved(out)
    assert any("no preflight record" in p for p in problems)
    assert lr.main(["--verify-saved", str(out)]) == 1
    capsys.readouterr()


def test_a_live_run_without_its_preflight_is_rejected_in_either_mode(inputs, capsys, monkeypatch):
    out, s = strict_run(inputs, monkeypatch, capsys)
    del s["preflight"]
    rewrite(out, s, lv1.STRICT, None, None)
    assert any("no preflight record" in p for p in lr.verify_saved(out))


@pytest.mark.parametrize("field", ["key_status", "key_policy", "provider_key_limit_usd", "provider_key_remaining_usd",
                                   "checks", "prices", "anthropic_endpoint", "request_configuration", "estimate_v1"])
def test_incomplete_preflight_evidence_is_rejected(inputs, capsys, monkeypatch, field):
    _, out, _, s, _ = best_effort_run(inputs, monkeypatch, capsys)
    del s["preflight"][field]
    rewrite(out, s)
    assert any("lacks" in p and field in p for p in lr.verify_saved(out))


@pytest.mark.parametrize("edit", ["not_ready", "failed_check", "snapshot_prices", "request_sent", "no_endpoint",
                                  "malformed_prices"])
def test_contradictory_preflight_evidence_is_rejected(inputs, capsys, monkeypatch, edit):
    _, out, _, s, _ = best_effort_run(inputs, monkeypatch, capsys)
    pre = s["preflight"]
    if edit == "not_ready":
        pre["ready_for_live"] = False
    elif edit == "failed_check":
        pre["checks"][0]["ok"] = False
    elif edit == "snapshot_prices":
        pre["prices_basis"] = "snapshot (offline)"
    elif edit == "request_sent":
        pre["request_configuration"]["requests_sent"] = 1
    elif edit == "no_endpoint":
        pre["anthropic_endpoint"] = None
    else:
        pre["prices"] = {"input_usd_mtok": 2.5}
    rewrite(out, s)
    assert lr.verify_saved(out) != []


@pytest.mark.parametrize("edit", ["provider_limit", "remaining", "key_limit"])
def test_altered_provider_limits_are_rejected_even_with_a_recomputed_summary(inputs, capsys, monkeypatch, edit):
    _, out, _, s, _ = best_effort_run(inputs, monkeypatch, capsys)
    pre, be = s["preflight"], lv1.KeyPolicy("best_effort", 50.0)
    if edit == "provider_limit":
        pre["provider_key_limit_usd"] = 40.0
        rewrite(out, s, be, 40.0, 49.0)
    elif edit == "remaining":
        pre["provider_key_remaining_usd"] = 0.5
        rewrite(out, s, be, 50.0, 0.5)
    else:
        pre["key_status"]["limit"] = 0.5                                         # under the acknowledged $50
        rewrite(out, s, be, 50.0, 49.0)
    assert lr.verify_saved(out) != []


def test_switching_best_effort_to_strict_with_the_evidence_kept_is_rejected(inputs, capsys, monkeypatch):
    _, out, _, s, _ = best_effort_run(inputs, monkeypatch, capsys)
    s["key_policy"] = s["preflight"]["key_policy"] = {"mode": "strict", "acknowledged_limit_usd": None}
    rewrite(out, s, lv1.STRICT, 50.0, 49.0)
    assert any("not accepted in strict mode" in p for p in lr.verify_saved(out))


def test_switching_strict_to_best_effort_is_rejected(inputs, capsys, monkeypatch):
    out, s = strict_run(inputs, monkeypatch, capsys)
    be = lv1.KeyPolicy("best_effort", 0.5)
    s["key_policy"] = s["preflight"]["key_policy"] = {"mode": "best_effort", "acknowledged_limit_usd": 0.5}
    rewrite(out, s, be, 0.5, 0.5)
    assert any("within strict mode" in p for p in lr.verify_saved(out))


def test_best_effort_mode_refuses_a_key_that_strict_mode_accepts(inputs, monkeypatch, capsys, no_model):
    code, r, _ = preflight(inputs, monkeypatch, capsys, metadata(), "--key-limit-mode", "best-effort",
                           "--acknowledge-key-limit-usd", "0.5")
    assert code == 2 and any("within strict mode" in p for p in r["problems"]) and no_model == []


@pytest.mark.parametrize("edit", ["mode_unknown", "live_with_label", "live_with_emulated_requests",
                                  "live_without_language_model", "dry_relabel"])
def test_the_recorded_execution_mode_must_be_consistent(inputs, capsys, monkeypatch, edit):
    _, out, _, s, _ = best_effort_run(inputs, monkeypatch, capsys)
    if edit == "mode_unknown":
        s["mode"] = "something else"
    elif edit == "live_with_label":
        s["model"]["label"] = lr.EMULATED_LABEL
    elif edit == "live_with_emulated_requests":
        s["emulated_requests"] = []
    elif edit == "live_without_language_model":
        s["model"]["language_model"] = False
    else:                                        # relabelled as a dry run but its preflight evidence kept
        s["mode"], s["model"]["language_model"], s["model"]["label"] = lr.DRY_MODE, False, lr.EMULATED_LABEL
    rewrite(out, s)
    assert lr.verify_saved(out) != []


@pytest.mark.parametrize("edit", ["with_preflight", "without_requests", "foreign_tool_ids"])
def test_a_dry_run_must_be_recorded_as_one(inputs, capsys, edit):
    _, out, _ = dry(inputs, capsys=capsys)
    s, doc = outputs(out)
    if edit == "with_preflight":
        s["preflight"] = {}
    elif edit == "without_requests":
        del s["emulated_requests"]
    else:
        for t in doc["trace"]:
            for c in t.get("tool_calls") or []:
                c["id"] = "toolu_01RealLookingId"
            if t.get("kind") in ("tool_call", "finish"):
                t["tool_use_id"] = "toolu_01RealLookingId"
        doc["fingerprint"] = av1._content_fingerprint(doc)                     # a recomputed fingerprint
        (out / "investigation_report.json").write_text(json.dumps(doc))
    rewrite(out, s)
    assert lr.verify_saved(out) != []


def test_legitimate_saved_reports_verify_in_every_recorded_mode(inputs, capsys, monkeypatch):
    _, out, _ = dry(inputs, capsys=capsys)
    assert lr.verify_saved(out) == []
    _, out, _ = dry(inputs, *BEST_EFFORT, name="dry_best_effort", capsys=capsys)       # no key: mode recorded only
    assert lr.verify_saved(out) == [] and outputs(out)[0]["cost"]["provider_key_limit_usd"] is None
    out, _ = strict_run(inputs, monkeypatch, capsys)
    assert lr.verify_saved(out) == [] and lr.main(["--verify-saved", str(out)]) == 0
    _, out, _, s, _ = best_effort_run(inputs, monkeypatch, capsys, name="be")
    assert lr.verify_saved(out) == [] and s["cost"]["authorization_mode"] == "best_effort"
    capsys.readouterr()
