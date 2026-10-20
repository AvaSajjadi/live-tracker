"""Tests for the experimental Stage 8 provider setup (new file).

No test makes a network request (sockets are blocked) or spends credit:
provider endpoints are emulated in-process (the real Anthropic SDK over a
mock transport) and metadata GETs are injected. Keys are fake strings.
"""

import ast
import json
import socket
from pathlib import Path

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_live as live
from live_tracker.investigation_live import GuardedClient, LivePlan, Prices, plan_budget, preflight

httpx2 = pytest.importorskip("httpx2")
pytest.importorskip("anthropic")

FAKE_OR = "sk-or-v1-FAKE-test-key-0123456789"
FAKE_ANT = "sk-ant-FAKE-personal-key-9876543210"
HAVE_DATA = (live.CACHE_DIR / "extraction_index.json").exists() and live.MANIFEST.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="verified CalMS21 validation cache not present")


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


# ---- an emulated provider ----

def models_listing(model="anthropic/claude-sonnet-5.5", params=("tools", "tool_choice", "reasoning"),
                   pricing=None):
    return {"data": [{"id": model, "context_length": 1_000_000,
                      "top_provider": {"max_completion_tokens": 128_000},
                      "supported_parameters": list(params),
                      "pricing": pricing or {"prompt": "0.000002", "completion": "0.00001",
                                             "input_cache_read": "0.0000001", "input_cache_write": "0.0000025"}}]}


class Metadata:
    """Injected free GETs: records URL and whether a key was sent."""

    def __init__(self, models=None, key=None, fail=()):
        self.models, self.key, self.fail, self.calls = models or models_listing(), key, fail, []

    def __call__(self, url, headers=None):
        self.calls.append((url, sorted((headers or {}).keys())))
        if any(f in url for f in self.fail):
            raise OSError("unreachable")
        if "/models" in url:
            return self.models
        if url.endswith("/key"):
            return {"data": self.key or {"limit": 1.0, "limit_remaining": 0.9, "usage": 0.1, "is_free_tier": False}}
        raise AssertionError(f"unexpected URL {url}")


class Endpoint:
    """Stand-in for POST /api/v1/messages: query_events, then a finish."""

    def __init__(self, cost=0.01, status=200, error_type=None):
        self.cost, self.status, self.error_type, self.seen = cost, status, error_type, []

    def __call__(self, request):
        body = json.loads(request.content)
        self.seen.append((request, body))
        if self.status != 200:
            return httpx2.Response(self.status, json={"type": "error", "error": {"type": self.error_type,
                                                                                 "message": "bad parameter"}})
        results = [b for m in body["messages"] if isinstance(m["content"], list) for b in m["content"]
                   if b.get("type") == "tool_result"]
        if not results:
            content = [{"type": "tool_use", "id": "toolu_1", "name": "query_events",
                        "input": {"source": "validation_annotation", "behavior": "attack"}}]
        else:
            eid = json.loads(results[-1]["content"])["evidence_id"]
            content = [{"type": "tool_use", "id": f"toolu_{len(self.seen)}", "name": ia.FINISH,
                        "input": {"finding": "inconclusive", "observations": [
                            {"evidence_id": eid, "field": "measurements.n_episodes_total"}]}}]
        return httpx2.Response(200, json={
            "id": "msg", "type": "message", "role": "assistant", "model": body["model"], "content": content,
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 4000, "output_tokens": 200, "cost": self.cost}})


def client_for(endpoint):
    return httpx2.Client(transport=httpx2.MockTransport(endpoint))


# ---- providers: openrouter preferred, anthropic only when named ----

def test_openrouter_is_the_default_and_anthropic_is_never_automatic(monkeypatch):
    assert LivePlan().provider == "openrouter" and LivePlan().model_id == "anthropic/claude-sonnet-5.5"
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANT)          # only the personal key is present
    with pytest.raises(ia.ModelUnavailable) as e:
        live.build_model(LivePlan())
    assert e.value.reason == "missing_api_key" and "OPENROUTER_API_KEY" in str(e.value)
    pre = preflight(LivePlan())
    assert pre["provider"] == "openrouter" and not pre["ready_for_live"]
    assert next(c for c in pre["checks"] if c["name"] == "api_key_present")["ok"] is False
    assert LivePlan(provider="anthropic").model_id == "claude-sonnet-5-5"
    with pytest.raises(ValueError):
        LivePlan(provider="somewhere-else")


def test_openrouter_requests_never_carry_the_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANT)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")       # ignored: base_url is explicit
    endpoint = Endpoint()
    model, ledger = live.build_model(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR}, http_client=client_for(endpoint))
    model.respond(ia.SYSTEM_PROMPT, ia.TOOL_SPECS, [{"role": "user", "content": "q"}])
    request, body = endpoint.seen[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/messages"
    assert request.headers["authorization"] == f"Bearer {FAKE_OR}"
    assert "x-api-key" not in request.headers and FAKE_ANT not in str(request.headers) + json.dumps(body)
    assert "anthropic-beta" not in request.headers
    assert "fallbacks" not in body and "betas" not in body and "thinking" not in body
    assert body["model"] == "anthropic/claude-sonnet-5.5" and body["tool_choice"] == {"type": "auto"}
    assert body["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"} and body["output_config"] == {"effort": "medium"}
    assert ledger.entries[0]["cost_usd"] == 0.01 and ledger.reported_cost_usd == 0.01


def test_anthropic_provider_uses_its_own_key_only_when_named():
    endpoint = Endpoint()
    model, _ = live.build_model(LivePlan(provider="anthropic"), env={"ANTHROPIC_API_KEY": FAKE_ANT,
                                                                     "OPENROUTER_API_KEY": FAKE_OR},
                                http_client=client_for(endpoint))
    model.respond("s", [], [{"role": "user", "content": "q"}])
    request, body = endpoint.seen[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == FAKE_ANT and "authorization" not in request.headers
    assert FAKE_OR not in str(request.headers) + json.dumps(body) and "provider" not in body


# ---- graceful handling of Anthropic-specific parameters ----

def test_guarded_client_drops_unsupported_fields():
    class Inner:
        def __init__(self):
            self.kw = None
            self.messages = self

        def create(self, **kw):
            self.kw = kw
            return type("R", (), {"usage": None, "model": "m", "stop_reason": "end_turn"})()
    inner = Inner()
    g = GuardedClient(inner, "openrouter", keep_output_config=False, keep_cache_control=False)
    g.create(model="m", max_tokens=5, betas=["x"], fallbacks="default", thinking={"type": "adaptive"},
             output_config={"effort": "high"},
             system=[{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}], messages=[])
    assert set(inner.kw) == {"model", "max_tokens", "system", "messages", "extra_body"}
    assert inner.kw["system"] == [{"type": "text", "text": "s"}]
    assert g.ledger.entries[0]["dropped_fields"] == ["betas", "fallbacks", "output_config", "cache_control"]
    with pytest.raises(AttributeError):
        g.beta                                              # never the beta (server-side fallback) path


def test_preflight_decides_which_fields_to_keep():
    plain = models_listing(params=("tools", "tool_choice"), pricing={"prompt": "0.000001", "completion": "0.000002"})
    pre = preflight(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR}, online=True, http_get=Metadata(plain))
    assert pre["compatibility"]["keep_output_config"] is False and pre["compatibility"]["keep_cache_control"] is False


def test_a_provider_400_stops_the_run_gracefully():
    endpoint = Endpoint(status=400, error_type="invalid_request_error")
    model, ledger = live.build_model(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR}, http_client=client_for(endpoint))
    with pytest.raises(ia.ModelUnavailable) as e:
        model.respond("s", [], [{"role": "user", "content": "q"}])
    assert e.value.reason == "api_error" and len(endpoint.seen) == 1            # no automatic retry of a 400
    (entry,) = ledger.entries
    assert entry["status"] == 400 and entry["error_type"] == "invalid_request_error"
    assert "bad parameter" not in json.dumps(entry)                             # messages are not kept


# ---- budget ----

def test_budget_reserves_one_request_and_never_exceeds_the_target():
    prices = Prices.from_per_mtok({"prompt": 2.0, "completion": 10.0, "input_cache_write": 2.5}, "test")
    assert (prices.input_usd_mtok, prices.output_usd_mtok) == (2.5, 10.0)        # conservative input price
    budget, est = plan_budget(LivePlan(), prices)
    assert est["estimated_worst_case_usd"] == pytest.approx(0.50)
    assert budget.max_cost_usd == est["stage6_cost_cap_usd"] < 0.50 and budget.price_per_mtok == (2.5, 10.0)
    assert budget.max_model_turns == 8 and budget.max_tool_calls == 8 and budget.timeout_s == 600
    assert budget.max_output_tokens == 8 * 4000 and budget.max_input_tokens * 2.5 / 1e6 <= budget.max_cost_usd + 1e-9
    assert "not a provider-side spending cap" in est["note"]
    with pytest.raises(ValueError):                                             # no room after the reserve
        plan_budget(LivePlan(target_usd=0.05, max_model_turns=12), Prices(5.0, 25.0, "test"))


def test_openrouter_per_token_prices_are_converted_and_request_fees_refused():
    p = Prices.from_openrouter({"prompt": "0.000004", "completion": "0.00002", "input_cache_write": "0.000005"}, "t")
    assert (p.input_usd_mtok, p.output_usd_mtok) == pytest.approx((5.0, 20.0))
    with pytest.raises(ValueError):
        Prices.from_openrouter({"prompt": "0.000001", "completion": "0.000002", "request": "0.01"}, "t")
    with pytest.raises(ValueError):
        Prices.from_openrouter({"prompt": "", "completion": "0.000002"}, "t")


def test_provider_reported_cost_stops_before_another_request():
    endpoint = Endpoint(cost=0.30)
    model, ledger = live.build_model(LivePlan(target_usd=0.50), env={"OPENROUTER_API_KEY": FAKE_OR},
                                     http_client=client_for(endpoint))
    for _ in range(2):
        model.respond("s", [], [{"role": "user", "content": "q"}])
    with pytest.raises(ia.ModelUnavailable) as e:
        model.respond("s", [], [{"role": "user", "content": "q"}])
    assert e.value.reason == "provider_reported_budget" and len(endpoint.seen) == 2
    assert ledger.reported_cost_usd == pytest.approx(0.60)


# ---- preflight ----

def test_offline_preflight_makes_no_request_and_is_never_ready():
    pre = preflight(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR})
    assert pre["online"] is False and pre["ready_for_live"] is False and pre["prices_basis"] == "snapshot (offline)"
    assert FAKE_OR not in json.dumps(pre)


def test_online_preflight_uses_only_free_metadata_gets():
    meta = Metadata()
    pre = preflight(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR}, online=True, http_get=meta)
    assert [u for u, _ in meta.calls] == [live.PROVIDERS["openrouter"].models_url, live.PROVIDERS["openrouter"].key_url]
    assert meta.calls[0][1] == [] and meta.calls[1][1] == ["Authorization"]      # the key only goes to /key
    assert all("/messages" not in u for u, _ in meta.calls)
    names = {c["name"]: c["ok"] for c in pre["checks"]}
    if HAVE_DATA:
        assert pre["ready_for_live"] is True and all(names.values())
    assert pre["prices_basis"] == "fetched" and pre["key_status"]["limit_remaining"] == 0.9
    assert FAKE_OR not in json.dumps(pre)


@pytest.mark.parametrize("meta, failing", [
    (Metadata(models_listing(model="anthropic/other")), "model_listed_with_tools"),
    (Metadata(models_listing(params=("reasoning",))), "tool_support"),
    (Metadata(key={"limit": 0.2, "limit_remaining": 0.1}), "key_credit"),
    (Metadata(fail=("/models",)), "model_list"),
    (Metadata(fail=("/key",)), "key_status"),
    (Metadata(models_listing(pricing={"prompt": "0.000001", "completion": "0.000002", "request": "0.5"})), "pricing"),
])
def test_online_preflight_failures_block_a_live_run(meta, failing):
    pre = preflight(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR}, online=True, http_get=meta)
    assert not pre["ready_for_live"]
    assert any(c["name"] == failing and not c["ok"] for c in pre["checks"]), pre["checks"]


def test_custom_header_env_blocks_a_live_run():
    pre = preflight(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR, "ANTHROPIC_CUSTOM_HEADERS": "X-A: b"},
                    online=True, http_get=Metadata())
    check = next(c for c in pre["checks"] if c["name"] == "clean_ambient_environment")
    assert not pre["ready_for_live"] and not check["ok"] and "X-A" not in json.dumps(pre)


def test_expensive_model_does_not_fit_the_initial_budget():
    opus = models_listing(model="anthropic/claude-opus-5.5", pricing={
        "prompt": "0.000004", "completion": "0.00002", "input_cache_read": "0.0000002", "input_cache_write": "0.000005"})
    pre = preflight(LivePlan(model="anthropic/claude-opus-5.5"), env={"OPENROUTER_API_KEY": FAKE_OR}, online=True,
                    http_get=Metadata(opus))
    assert not pre["ready_for_live"] and not next(c for c in pre["checks"] if c["name"] == "budget_fit")["ok"]


# ---- runs ----

@needs_data
def test_dry_run_is_offline_and_end_to_end():
    out = live.dry_run()
    assert out["finished"] and out["report_verified"] and out["tools_used"] == ["query_events", ia.FINISH]
    assert len(out["emulated_requests"]) == 2
    for r in out["emulated_requests"]:
        assert r["url"] == "https://openrouter.ai/api/v1/messages" and "headers" not in r
        assert r["has_authorization"] and not r["has_x_api_key"] and not r["has_anthropic_beta"]
    assert out["provider_reported_cost_usd"] == 0.0 and out["provider_ledger_complete"] is True


def test_live_run_needs_explicit_confirmation(tmp_path):
    meta = Metadata()
    for confirm in (None, 0.49, 1.0):
        with pytest.raises(PermissionError):
            live.run_live(LivePlan(), confirm_budget_usd=confirm, env={"OPENROUTER_API_KEY": FAKE_OR},
                          out_dir=tmp_path, http_get=meta)
    assert meta.calls == []                                     # refused before even the free metadata GETs
    with pytest.raises(PermissionError):                        # confirmed, but no key
        live.run_live(LivePlan(), confirm_budget_usd=0.50, env={}, out_dir=tmp_path, http_get=Metadata())


@needs_data
def test_emulated_live_run_writes_reports_without_the_key(tmp_path):
    endpoint = Endpoint(cost=0.004)
    out = live.run_live(LivePlan(), confirm_budget_usd=0.50, env={"OPENROUTER_API_KEY": FAKE_OR}, out_dir=tmp_path,
                        http_get=Metadata(), http_client=client_for(endpoint))
    assert out["finished"] and out["report_verified"] and out["provider_reported_cost_usd"] == pytest.approx(0.008)
    files = sorted(p.name for p in Path(out["output_dir"]).iterdir())
    assert files == ["investigation_report.json", "report.json", "report.md", "run.json"]
    for p in Path(out["output_dir"]).iterdir():
        assert FAKE_OR not in p.read_text()
    run = json.loads((Path(out["output_dir"]) / "run.json").read_text())
    assert run["budget"]["max_cost_usd"] < 0.50 and run["plan"]["provider"] == "openrouter"


def test_stage8_reads_no_csv_and_no_test_split():
    src = Path(live.__file__).read_text()
    tree = ast.parse(src)
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any(s.endswith(".csv") or "/test/" in s for s in strings)
    assert "annotation_split" not in src.replace('policy.annotation_split == "validation"', "")


# ======================================================================
# Codex 2 review: ambient headers, cost accounting, retries, confirmation
# ======================================================================

AMBIENT_SECRET = "sk-ant-AMBIENT-SECRET-must-not-leak"


def text_reply(cost):
    return {"id": "m", "type": "message", "role": "assistant", "model": "anthropic/claude-sonnet-5.5",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5, **({} if cost == "absent" else {"cost": cost})}}


class Costs:
    """One response per entry: a cost value, "absent", an int HTTP status, or an exception."""

    def __init__(self, *responses):
        self.responses, self.attempts = list(responses), []

    def __call__(self, request):
        self.attempts.append(request)
        r = self.responses[min(len(self.attempts) - 1, len(self.responses) - 1)]
        if isinstance(r, BaseException):
            raise r
        if isinstance(r, int) and not isinstance(r, bool) and r >= 400:
            return httpx2.Response(r, json={"type": "error", "error": {"type": "overloaded_error",
                                                                       "message": AMBIENT_SECRET}})
        body = json.dumps(text_reply(r), allow_nan=True)          # lets NaN / Infinity reach the SDK
        return httpx2.Response(200, content=body.encode(), headers={"content-type": "application/json"})


def guarded_model(endpoint, plan=None):
    return live.build_model(plan or LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR},
                            http_client=client_for(endpoint))


def ask(model):
    return model.respond("s", [], [{"role": "user", "content": "q"}])


# ---- 1. ambient SDK headers ----

@pytest.mark.parametrize("name, value", [("ANTHROPIC_CUSTOM_HEADERS", f"X-Secret: {AMBIENT_SECRET}"),
                                         ("ANTHROPIC_LOG", "debug")])
def test_ambient_sdk_variables_are_refused_before_any_client(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", AMBIENT_SECRET)                 # a conflicting ambient credential
    endpoint = Endpoint()
    import anthropic
    built = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: built.append(k))
    with pytest.raises(live.AmbientEnvironmentError) as e:                  # injected env is clean: still refused
        live.build_model(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR}, http_client=client_for(endpoint))
    assert built == [] and endpoint.seen == [] and AMBIENT_SECRET not in str(e.value) and name in str(e.value)
    with pytest.raises(live.AmbientEnvironmentError):
        live.dry_run()                                                      # offline mode too
    pre = preflight(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR}, online=True, http_get=Metadata())
    assert not pre["ready_for_live"] and AMBIENT_SECRET not in json.dumps(pre)
    with pytest.raises(PermissionError) as e:
        live.run_live(LivePlan(), confirm_budget_usd=0.50, env={"OPENROUTER_API_KEY": FAKE_OR},
                      http_get=Metadata(), http_client=client_for(endpoint))
    assert endpoint.seen == [] and AMBIENT_SECRET not in str(e.value)


def test_cli_reports_ambient_refusal_without_values(monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cli", Path(live.ROOT) / "scripts" / "investigation_live.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"X-Secret: {AMBIENT_SECRET}")
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert "ANTHROPIC_CUSTOM_HEADERS" in captured.err and AMBIENT_SECRET not in captured.out + captured.err


@needs_data
def test_offline_run_with_conflicting_ambient_credentials_keeps_them_out(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", AMBIENT_SECRET)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", AMBIENT_SECRET)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR)                       # the dry run uses a placeholder
    out = live.dry_run()
    text = json.dumps(out)
    assert out["finished"] and AMBIENT_SECRET not in text and FAKE_OR not in text
    assert all(not r["has_x_api_key"] and r["has_authorization"] for r in out["emulated_requests"])


def test_openrouter_request_with_conflicting_ambient_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", AMBIENT_SECRET)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", AMBIENT_SECRET)
    endpoint = Costs(0.01)
    ask(guarded_model(endpoint)[0])
    (request,) = endpoint.attempts
    assert request.headers["authorization"] == f"Bearer {FAKE_OR}" and "x-api-key" not in request.headers
    assert AMBIENT_SECRET not in str(request.headers) + request.content.decode()


def test_errors_keep_no_exception_text():
    endpoint = Costs(529)
    model, ledger = guarded_model(endpoint)
    with pytest.raises(ia.ModelUnavailable) as e:
        ask(model)
    assert AMBIENT_SECRET not in str(e.value) + json.dumps(ledger.to_json())
    assert ledger.entries[0]["status"] == 529 and ledger.entries[0]["error_type"] == "overloaded_error"


# ---- 2. provider cost accounting ----

@pytest.mark.parametrize("bad", [None, "absent", -0.3, float("nan"), float("inf"), "0.1", True])
def test_unknown_or_invalid_cost_never_erases_spending_and_stops_requests(bad):
    endpoint = Costs(0.30, bad, 0.30)
    model, ledger = guarded_model(endpoint)
    ask(model)
    with pytest.raises(ia.ModelUnavailable) as e:                # the reply with the bad cost is withheld
        ask(model)
    assert e.value.reason == "provider_cost_unknown"
    with pytest.raises(ia.ModelUnavailable) as e:
        ask(model)
    assert e.value.reason == "provider_cost_unknown" and len(endpoint.attempts) == 2      # fail closed
    assert ledger.known_cost_usd == pytest.approx(0.30) and ledger.complete is False
    assert ledger.unknown_cost_attempts == 1 and ledger.reported_cost_usd is None
    assert ledger.entries[1]["cost_usd"] is None and ledger.entries[1]["cost_status"] in ("missing", "invalid")
    state = ledger.to_json()
    assert state["known_cost_usd"] == pytest.approx(0.30) and state["complete"] is False


def test_known_spending_reaching_the_target_stops_requests():
    endpoint = Costs(0.20, 0.25, 0.10, 0.10)
    model, ledger = guarded_model(endpoint, LivePlan(target_usd=0.40))
    ask(model)
    ask(model)                                                   # 0.45 >= 0.40
    with pytest.raises(ia.ModelUnavailable) as e:
        ask(model)
    assert e.value.reason == "provider_reported_budget" and len(endpoint.attempts) == 2
    assert ledger.known_cost_usd == pytest.approx(0.45) and ledger.complete


def test_valid_costs_accumulate_across_many_responses():
    endpoint = Costs(0.01, 0.0, 0.02, 0.03)
    model, ledger = guarded_model(endpoint)
    for _ in range(4):
        ask(model)
    assert ledger.known_cost_usd == pytest.approx(0.06) == ledger.reported_cost_usd and ledger.complete


def test_anthropic_reports_no_cost_but_the_estimate_still_applies():
    """No cost is ever reported there: the known-spending guard sees zero and
    the run relies on the Stage 6 estimate (one reason live mode is disabled)."""
    endpoint = Costs("absent")
    model, ledger = live.build_model(LivePlan(provider="anthropic"), env={"ANTHROPIC_API_KEY": FAKE_ANT},
                                     http_client=client_for(endpoint))
    ask(model)
    ask(model)
    assert ledger.complete is False and len(endpoint.attempts) == 2


# ---- 3. no invisible retries ----

@pytest.mark.parametrize("failure", [529, 500, 429])
def test_http_failures_are_single_attempts(failure):
    endpoint = Costs(failure, 0.01)
    model, ledger = guarded_model(endpoint)
    with pytest.raises(ia.ModelUnavailable):
        ask(model)
    assert len(endpoint.attempts) == 1 == len(ledger.entries)
    assert ledger.entries[0]["cost_status"] == "unknown_after_failure"
    with pytest.raises(ia.ModelUnavailable) as e:                # no further request after an uncertain failure
        ask(model)
    assert e.value.reason == "provider_cost_unknown" and len(endpoint.attempts) == 1


def test_transport_timeout_is_not_retried():
    endpoint = Costs(httpx2.ReadTimeout("slow"), 0.01)
    model, ledger = guarded_model(endpoint)
    with pytest.raises(ia.ModelUnavailable) as e:
        ask(model)
    assert e.value.reason == "timeout" and len(endpoint.attempts) == 1 == len(ledger.entries)
    assert ledger.entries[0]["error"] == "APITimeoutError" and ledger.entries[0]["cost_status"] == "unknown_after_failure"


def test_with_options_cannot_reenable_retries():
    endpoint = Costs(529, 0.01)
    model, ledger = guarded_model(endpoint)
    g = model.inner._client
    retrying = g.with_options(max_retries=5, timeout=10)
    with pytest.raises(Exception):
        retrying.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "q"}])
    assert len(endpoint.attempts) == 1 and retrying._client.max_retries == 0


@needs_data
def test_uncertain_failure_ends_in_a_controlled_incomplete_investigation(tmp_path):
    endpoint = Costs(httpx2.ReadTimeout("slow"))
    out = live.run_live(LivePlan(), confirm_budget_usd=0.50, env={"OPENROUTER_API_KEY": FAKE_OR}, out_dir=tmp_path,
                        http_get=Metadata(), http_client=client_for(endpoint))
    assert out["finished"] is False and out["finding"] == "incomplete" and out["stopping_reason"] == "model_unavailable"
    assert len(endpoint.attempts) == 1 == len(out["requests"]) and out["provider_ledger_complete"] is False
    assert out["report_verified"] is True
    assert sorted(p.name for p in Path(out["output_dir"]).iterdir()) == ["investigation_report.json", "report.json",
                                                                         "report.md", "run.json"]


# ---- 4. budget confirmation ----

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -0.5, 0.0, "0.50", True, None, [0.5],
                                 0.49])
def test_invalid_confirmations_are_refused_before_anything_happens(tmp_path, bad):
    meta, endpoint = Metadata(), Endpoint()
    with pytest.raises(PermissionError):
        live.run_live(LivePlan(), confirm_budget_usd=bad, env={"OPENROUTER_API_KEY": FAKE_OR}, out_dir=tmp_path,
                      http_get=meta, http_client=client_for(endpoint))
    assert meta.calls == [] and endpoint.seen == [] and list(tmp_path.iterdir()) == []


def test_cli_refuses_nan_confirmation(capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cli", Path(live.ROOT) / "scripts" / "investigation_live.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--live", "--confirm-budget-usd", "nan"]) == 2
    assert "finite positive" in capsys.readouterr().err


# ---- 5. Anthropic live mode is disabled ----

def test_anthropic_live_mode_is_disabled_with_an_explanation(tmp_path):
    meta, endpoint = Metadata(), Endpoint()
    with pytest.raises(PermissionError) as e:
        live.run_live(LivePlan(provider="anthropic"), confirm_budget_usd=0.50, env={"ANTHROPIC_API_KEY": FAKE_ANT},
                      out_dir=tmp_path, http_get=meta, http_client=client_for(endpoint))
    assert "Anthropic live mode is disabled" in str(e.value) and "--provider openrouter" in str(e.value)
    assert meta.calls == [] and endpoint.seen == [] and list(tmp_path.iterdir()) == []
    pre = preflight(LivePlan(provider="anthropic"), env={"ANTHROPIC_API_KEY": FAKE_ANT}, online=True, http_get=meta)
    assert not pre["ready_for_live"] and not next(c for c in pre["checks"] if c["name"] == "live_mode_supported")["ok"]
    assert meta.calls == []                                   # no OpenRouter metadata for the Anthropic provider


@needs_data
def test_anthropic_dry_run_still_works():
    out = live.dry_run(LivePlan(provider="anthropic"))
    assert out["finished"] and all(r["url"] == "https://api.anthropic.com/v1/messages" and r["has_x_api_key"]
                                   for r in out["emulated_requests"])


# ---- Codex 2 follow-up: unrepresentable costs and accounting failures ----

class RawCosts:
    """Responses whose cost is written as raw JSON text (e.g. a 400-digit integer)."""

    def __init__(self, *literals):
        self.literals, self.attempts = list(literals), []

    def __call__(self, request):
        self.attempts.append(request)
        lit = self.literals[min(len(self.attempts) - 1, len(self.literals) - 1)]
        body = ('{"id":"m","type":"message","role":"assistant","model":"anthropic/claude-sonnet-5.5",'
                '"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,'
                '"usage":{"input_tokens":10,"output_tokens":5,"cost":%s}}' % lit)
        return httpx2.Response(200, content=body.encode(), headers={"content-type": "application/json"})


@pytest.mark.parametrize("literal", ["1" + "0" * 400, "-" + "9" * 400, "1e400", "-1e400", "5000", "1e-400e"[:-1]])
def test_unrepresentable_costs_are_invalid_and_stop_further_requests(literal):
    endpoint = RawCosts("0.30", literal, "0.01")
    model, ledger = guarded_model(endpoint)
    ask(model)
    expected_invalid = literal != "1e-400"                       # 1e-400 underflows to 0.0: a valid zero cost
    if expected_invalid:
        with pytest.raises(ia.ModelUnavailable) as e:            # the reply with the bad cost: withheld
            ask(model)
        assert e.value.reason == "provider_cost_unknown"
        with pytest.raises(ia.ModelUnavailable) as e:
            ask(model)                                           # the valid third response is never requested
        assert e.value.reason == "provider_cost_unknown" and len(endpoint.attempts) == 2
        assert ledger.entries[1]["cost_status"] == "invalid" and ledger.entries[1]["cost_usd"] is None
        assert ledger.known_cost_usd == pytest.approx(0.30) and ledger.complete is False
        assert literal not in json.dumps(ledger.to_json())       # the provider's value is not kept
    else:
        ask(model)
        ask(model)
        assert ledger.complete and ledger.known_cost_usd == pytest.approx(0.31)
    assert len(ledger.entries) == len(endpoint.attempts)        # every physical attempt has an entry


@pytest.mark.parametrize("cost", [10 ** 400, -(10 ** 400), 1001, float("inf"), float("nan"), -0.01, True,
                                  "0.1", None])
def test_usage_cost_never_raises_and_never_overflows(cost):
    from decimal import Decimal
    from fractions import Fraction
    usage = type("U", (), {"cost": cost})()
    value, status = live._usage_cost(type("R", (), {"usage": usage})())
    assert value is None and status == ("missing" if cost is None else "invalid")
    for odd in (Fraction(10 ** 400), Decimal("1e400"), Decimal("NaN"), object()):
        assert live._usage_cost(type("R", (), {"usage": type("U", (), {"cost": odd})()})()) == (None, "invalid")


def test_numeric_conversion_overflow_is_invalid():
    class Huge(float):                      # a float subclass: not accepted as a plain float
        pass

    class Explodes:
        def __float__(self):
            raise OverflowError("int too large to convert to float")

        def __lt__(self, other):
            raise OverflowError("no")
    for odd in (Huge(1.0), Explodes()):
        assert live._usage_cost(type("R", (), {"usage": type("U", (), {"cost": odd})()})()) == (None, "invalid")


class Raising:
    def __get__(self, obj, owner):
        raise RuntimeError(f"accounting blew up {AMBIENT_SECRET}")


class StubClient:
    """An SDK-like client whose responses break accounting in a chosen way."""

    def __init__(self, *responses):
        self.responses, self.calls, self.messages = list(responses), 0, self

    def with_options(self, **kw):
        assert kw.get("max_retries") == 0
        return self

    def create(self, **kw):
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def response(**attrs):
    return type("R", (), attrs)()


@pytest.mark.parametrize("bad_response", [
    response(usage=Raising(), model="m", stop_reason="end_turn"),                    # usage access raises
    type("R", (), {"usage": type("U", (), {"cost": Raising()})(), "model": "m", "stop_reason": "end_turn"})(),
    type("R", (), {"model": Raising(), "usage": None, "stop_reason": "end_turn"})(),  # label access raises
])
def test_accounting_exceptions_still_record_the_attempt_and_stop(bad_response):
    good = response(usage=type("U", (), {"cost": 0.30, "input_tokens": 1, "output_tokens": 1})(), model="m",
                    stop_reason="end_turn")
    stub = StubClient(good, bad_response, good)
    g = GuardedClient(stub, "openrouter")
    guard = live.ReportedCostGuard(type("Inner", (), {"respond": lambda s, *a, **k: g.create(model="m",
                                                                                             max_tokens=5)})(),
                                   g.ledger, 0.50)
    guard.respond("s", [], [])
    with pytest.raises(ia.ModelUnavailable) as e:                # the request happened, but its reply is withheld
        guard.respond("s", [], [])
    assert e.value.reason == "provider_cost_unknown" and AMBIENT_SECRET not in str(e.value)
    with pytest.raises(ia.ModelUnavailable) as e:
        guard.respond("s", [], [])
    assert e.value.reason == "provider_cost_unknown" and stub.calls == 2 == len(g.ledger.entries)
    assert g.ledger.entries[1]["cost_status"] in ("accounting_error", "invalid")
    assert g.ledger.known_cost_usd == pytest.approx(0.30) and g.ledger.complete is False
    assert AMBIENT_SECRET not in json.dumps(g.ledger.to_json())


def test_an_attempt_is_in_the_ledger_before_it_is_sent():
    seen_entries = []

    class Peek(StubClient):
        def create(self, **kw):
            seen_entries.append(list(g.ledger.entries))
            raise KeyboardInterrupt                              # not even an Exception: the entry already exists
    g = GuardedClient(Peek(), "openrouter")
    with pytest.raises(KeyboardInterrupt):
        g.create(model="m", max_tokens=5)
    assert len(seen_entries[0]) == 1 and seen_entries[0][0]["cost_status"] == "pending"
    assert g.ledger.complete is False and g.ledger.unknown_cost_attempts == 1


def test_provider_values_are_sanitised_in_the_ledger():
    weird = response(usage=type("U", (), {"cost": 0.01, "input_tokens": 10 ** 400, "output_tokens": -5,
                                          "cache_read_input_tokens": "lots"})(),
                     model="x" * 500, stop_reason="end\nturn")
    g = GuardedClient(StubClient(weird), "openrouter")
    g.create(model="m", max_tokens=5)
    (e,) = g.ledger.entries
    assert e["input_tokens"] is None and e["output_tokens"] is None and e["cache_read_input_tokens"] is None
    assert e["model_served"] is None and e["stop_reason"] is None and e["cost_status"] == "reported"
    json.dumps(g.ledger.to_json())                              # always serialisable


# ---- Codex 2 follow-up: a reply whose accounting failed never reaches Stage 6 ----

import live_tracker.investigation_contracts as ic  # noqa: E402
import test_investigation_contracts as tc  # noqa: E402


class CountingTools(ic.UnavailableTools):
    """Contract-valid tools that count executions (answers are not_implemented errors)."""

    def __init__(self):
        self.executed = []

    def query_events(self, req):
        self.executed.append("query_events")
        return super().query_events(req)


def scripted_endpoint(*turns):
    """Raw-JSON replies: each turn is (content blocks, cost literal)."""
    attempts = []

    def handler(request):
        attempts.append(json.loads(request.content))
        content, cost = turns[min(len(attempts) - 1, len(turns) - 1)]
        if callable(content):
            content = content(attempts[-1])
        body = ('{"id":"m","type":"message","role":"assistant","model":"anthropic/claude-sonnet-5.5",'
                '"content":%s,"stop_reason":"tool_use","stop_sequence":null,'
                '"usage":{"input_tokens":100,"output_tokens":20,"cost":%s}}' % (json.dumps(content), cost))
        return httpx2.Response(200, content=body.encode(), headers={"content-type": "application/json"})
    return handler, attempts


FINISH_NOW = [{"type": "text", "text": "Done."},
              {"type": "tool_use", "id": "toolu_f", "name": ia.FINISH,
               "input": {"finding": "inconclusive", "observations": []}}]
QUERY = [{"type": "tool_use", "id": "toolu_q", "name": "query_events",
          "input": {"source": "validation_annotation", "behavior": "attack"}}]


def run_loop(handler, tools=None):
    model, ledger = live.build_model(LivePlan(), env={"OPENROUTER_API_KEY": FAKE_OR}, http_client=client_for(handler))
    tools = tools or CountingTools()
    report = ia.investigate("Does approach tend to precede attack?", tools, tc.POLICY, model,
                            budget=ia.Budget(max_cost_usd=0.4, price_per_mtok=(2.5, 10.0)))
    return report, ledger, tools


def assert_withheld(report, ledger, attempts):
    assert report.finished is False and report.finding == "incomplete"
    assert report.stopping_reason == "model_unavailable"
    errors = [t for t in report.trace if t["kind"] == "model_error"]
    assert errors and errors[-1]["reason"] == "provider_cost_unknown"
    assert [t["kind"] for t in report.trace[-2:]] == ["model_error", "stop"]       # the withheld turn left no trace
    assert len(attempts) == len(ledger.entries) and ledger.complete is False      # recorded, still incomplete
    text = json.dumps(report.to_json()) + json.dumps(ledger.to_json())
    assert FAKE_OR not in text and "1" + "0" * 400 not in text and "Done." not in text


@pytest.mark.parametrize("cost", ["1" + "0" * 400, "-0.5", "1e400", '"0.01"'])
def test_invalid_cost_final_answer_is_not_accepted(cost):
    handler, attempts = scripted_endpoint((FINISH_NOW, cost))
    report, ledger, _ = run_loop(handler)
    assert_withheld(report, ledger, attempts)
    assert len(attempts) == 1 and not [t for t in report.trace if t["kind"] == "finish"]
    assert ledger.entries[0]["cost_status"] == "invalid"


def test_accounting_error_final_answer_is_not_accepted(monkeypatch):
    handler, attempts = scripted_endpoint((FINISH_NOW, "0.01"))

    def broken(response):
        raise RuntimeError(f"accounting failed {FAKE_OR}")
    monkeypatch.setattr(live, "_usage_cost", broken)
    report, ledger, _ = run_loop(handler)
    assert_withheld(report, ledger, attempts)
    assert ledger.entries[0]["cost_status"] == "accounting_error" and ledger.entries[0]["accounting_error"] == "RuntimeError"
    assert not [t for t in report.trace if t["kind"] == "finish"]


def test_invalid_cost_tool_call_executes_no_tool():
    handler, attempts = scripted_endpoint((QUERY, "1" + "0" * 400), (FINISH_NOW, "0.01"))
    report, ledger, tools = run_loop(handler)
    assert_withheld(report, ledger, attempts)
    assert tools.executed == [] and report.evidence == [] and report.usage["tool_calls"] == 0
    assert len(attempts) == 1                                   # the valid follow-up is never requested


def test_known_spending_survives_a_withheld_reply():
    def finish_citing(body):
        eid = next(json.loads(b["content"])["evidence_id"] for m in body["messages"] if isinstance(m["content"], list)
                   for b in m["content"] if b.get("type") == "tool_result")
        return [{"type": "tool_use", "id": "toolu_f2", "name": ia.FINISH, "input": {
            "finding": "inconclusive", "observations": [{"evidence_id": eid, "field": "status"}]}}]
    handler, attempts = scripted_endpoint((QUERY, "0.02"), (finish_citing, "NaN"), (FINISH_NOW, "0.01"))
    report, ledger, tools = run_loop(handler)
    assert tools.executed == ["query_events"] and len(report.evidence) == 1       # the paid-for turn counted
    assert_withheld(report, ledger, attempts)
    assert len(attempts) == 2 and ledger.known_cost_usd == pytest.approx(0.02)
    assert [e["cost_status"] for e in ledger.entries] == ["reported", "invalid"]


def test_valid_cost_tool_response_still_works():
    def finish_citing(body):
        eid = next(json.loads(b["content"])["evidence_id"] for m in body["messages"] if isinstance(m["content"], list)
                   for b in m["content"] if b.get("type") == "tool_result")
        return [{"type": "tool_use", "id": "toolu_f3", "name": ia.FINISH, "input": {
            "finding": "inconclusive", "observations": [{"evidence_id": eid, "field": "status"}]}}]
    handler, attempts = scripted_endpoint((QUERY, "0.02"), (finish_citing, "0.03"))
    report, ledger, tools = run_loop(handler)
    assert report.finished and report.stopping_reason == "finished" and tools.executed == ["query_events"]
    assert len(attempts) == 2 == len(ledger.entries) and ledger.complete
    assert ledger.known_cost_usd == pytest.approx(0.05) == ledger.reported_cost_usd


# ---- full authorized coverage: preflight, dry run and live run share one setting ----

def synthetic_cache(tmp_path, sessions):
    from test_investigation_windows import build
    env = build(tmp_path, sessions=sessions, name="cov")
    return env[2], env[3]


def coverage_check(pre):
    return next(c for c in pre["checks"] if c["name"] == "validation_coverage")


def test_preflight_refuses_a_partial_cache_unless_partial_is_explicit(tmp_path, monkeypatch):
    from test_investigation_windows import VAL_A
    cache, manifest = synthetic_cache(tmp_path, (VAL_A,))
    monkeypatch.setattr(live, "MANIFEST", manifest)
    pre = preflight(LivePlan(cache_dir=str(cache)), env={"OPENROUTER_API_KEY": FAKE_OR})
    c = coverage_check(pre)
    assert not c["ok"] and "1 of 3 eligible" in c["detail"] and "--all" in c["detail"]
    assert pre["data_coverage"] == {"mode": "complete", "eligible": 3, "cached": 1, "missing": pre["data_coverage"]["missing"]}
    assert len(pre["data_coverage"]["missing"]) == 2
    pre = preflight(LivePlan(cache_dir=str(cache), coverage="partial"), env={"OPENROUTER_API_KEY": FAKE_OR})
    assert coverage_check(pre)["ok"] and "chosen explicitly" in coverage_check(pre)["detail"]


def test_preflight_reports_missing_or_corrupt_caches(tmp_path, monkeypatch):
    from test_investigation_windows import VAL_A, VAL_B, VAL_C
    cache, manifest = synthetic_cache(tmp_path, (VAL_A, VAL_B, VAL_C))
    monkeypatch.setattr(live, "MANIFEST", manifest)
    assert coverage_check(preflight(LivePlan(cache_dir=str(cache)), env={}))["ok"]
    f = next(cache.glob("task1__*.json"))
    f.write_bytes(f.read_bytes() + b" ")
    c = coverage_check(preflight(LivePlan(cache_dir=str(cache)), env={}))
    assert not c["ok"] and "does not verify" in c["detail"]
    pre = preflight(LivePlan(cache_dir=str(tmp_path / "absent")), env={})
    assert not next(x for x in pre["checks"] if x["name"] == "validation_cache")["ok"]


def test_dry_run_and_live_run_use_the_plan_coverage(tmp_path, monkeypatch):
    from test_investigation_windows import VAL_A
    cache, manifest = synthetic_cache(tmp_path, (VAL_A,))
    monkeypatch.setattr(live, "MANIFEST", manifest)
    seen = []
    real = live._tools_and_policy

    def spy(plan):
        seen.append((plan.coverage, plan.cache_dir))
        raise RuntimeError("stop after recording the configuration")
    monkeypatch.setattr(live, "_tools_and_policy", spy)
    for run in (lambda p: live.dry_run(p),):
        with pytest.raises(RuntimeError):
            run(LivePlan(cache_dir=str(cache), coverage="partial"))
    assert seen == [("partial", str(cache))]
    monkeypatch.setattr(live, "_tools_and_policy", real)
    with pytest.raises(ValueError):
        LivePlan(coverage="everything")


@needs_data
def test_real_offline_preflight_and_dry_run_cover_all_eligible_sessions():
    pre = preflight(LivePlan(), env={})
    assert coverage_check(pre)["ok"] and pre["data_coverage"]["eligible"] == pre["data_coverage"]["cached"] == 14
    out = live.dry_run()
    (g,) = out["coverage"]["groups"]
    assert out["coverage"]["status"] == "complete" and len(g["searched"]) == 14
