"""Phase 3B.2: the Stage 9 v1 agent (investigation_agent_v1.investigate_v1) on the guarded Stage 8 model
infrastructure of investigation_live (new file). Nothing here bills, prices or calls a provider itself.

Reused unchanged from investigation_live: the provider table, credential handling and ambient-environment refusal,
the Claude adapter (investigation_agent.ClaudeInvestigationModel, fallbacks=False, max_retries=0), GuardedClient
(Anthropic-only fields dropped, OpenRouter routing pinned to the Anthropic provider with allow_fallbacks False, SDK
retries forced off at every call), the per-request Ledger (every attempt recorded before it is sent),
ReportedCostGuard (no request once known provider-reported spending reaches the target or any attempt's cost is
unknown; a reply whose cost is not accounted is withheld), Prices, plan_budget (the Stage 6 Budget below the target
by a one-request reserve) and the online preflight (fetched prices, model and tool support, key status). The v1
agent's own loop, closed tool and finish schemas, runner, evidence and saved-report verification are
investigation_agent_v1's, unchanged.

Added for Stage 9 (GuardedModelV1, between the agent and ReportedCostGuard), each failing closed with
ModelUnavailable (the run stops "model_unavailable", the report is "incomplete"):
- every request carries exactly the v1 system prompt and closed tool / finish schemas (SYSTEM_PROMPT_V1,
  TOOL_SPECS_V1), and is no larger than the request the one-request reserve was sized for (max_request_chars);
- every respond() is exactly one ledger attempt; its reply is used only when the served model is the requested model,
  the stop reason is a known one and the reply's token counts are exactly the provider's usage. Otherwise the reply is
  withheld (its attempt stays in the ledger, cost counted if known).
The ledger entry of each reply given to the agent is marked reply_used, so ledger_problems() can bind the cost ledger
to the saved trace afterwards: one used, cost-reported attempt per model turn with the same tokens, at most one
unused attempt (the last, just before a model_error stop), and totals that re-add.

Spending: a plan's target is at most MAX_TARGET_USD ($0.50) and the provider is OpenRouter (the only provider that
reports a cost per request). The target is a LOCAL, client-side bound: Stage 6's estimate plus the provider-reported
cost guard stop further requests, but a request cannot be interrupted, so actual spending can pass the target by one
request. No code here can guarantee a maximum charge. The live preflight (preflight_v1) therefore also requires the
key's own provider-side credit limit, read from OpenRouter's key metadata before any paid request (key_limit_problems):
a finite configured limit of at most $0.50, remaining credit covering the target, no automatic limit reset
(limit_reset null) and consistent values (remaining = limit - usage). Unlimited keys, larger limits, resets and
missing or invalid fields are refused. That limit is the provider's to enforce; it is recorded as
provider_key_limit_usd, never as a guaranteed maximum. Saved outputs state target_usd, provider_key_limit_usd,
actual_spend_usd (the provider-reported total, complete or not) and over_target, truthfully even when the provider
charges past the target or its own limit.

Key-limit modes (KeyPolicy): "strict" (the default) requires the key limit above. "best_effort" must be chosen
explicitly with the key's actual limit acknowledged (acknowledged_limit_usd, which must equal the verified
metadata). It accepts only an existing, non-resetting key whose limit is above $0.50 (a key strict mode accepts must
use strict mode, so the mode follows from the verified limit) and at most MAX_BEST_EFFORT_KEY_LIMIT_USD ($50), with
every other key check unchanged (consistent metadata, remaining credit covering the target). The local target stays
at most $0.50 and every guard stays as it is: token budgets, cost ledger, ReportedCostGuard, retries 0, routing pin,
no fallbacks, fail-closed costs and responses. In best-effort mode, however, the provider-side limit no longer bounds
spending near the target. Exposure can exceed $0.50, up to the key's remaining credit, subject to the provider's
enforcement, and every report and cost summary says so. Neither mode authorizes a paid run (authorize_live does).

Before any live run, configure a DEDICATED, non-shared OpenRouter key with such a limit. The code cannot know whether
the key is used elsewhere at the same time, and cannot guarantee the provider's billing behaviour.

Readiness: _preflight_v1 is the one set of pre-request checks. It never makes a model request: free GETs
(model list, key status, the model's endpoint list, which must include exactly one Anthropic endpoint priced no
higher than the budget's prices and with the context, prompt and completion capacity for the plan's largest
request), the v1 budget estimate, and the request the adapter would send, built offline over a transport that
refuses every request (routing pin, no retries, no fallbacks).

Credentials: the capped inference key ($OPENROUTER_API_KEY) is the only key for /api/v1/key and for every model
request (the adapter is given an environment holding that key alone). The endpoint list is fetched with Bearer
authentication; OpenRouter may answer 403 unless a management key is used, so a separately configured
$OPENROUTER_MANAGEMENT_KEY, when set, is used for that one read-only GET and nothing else. It is never given to the
adapter, never falls back to inference, and never appears in a report or message; it must differ from the inference
key. A 403 or any other failure there is a failed check, never a skipped one.

Redirects: the v1 preflight's metadata GETs (model list, /api/v1/key, endpoint list) use get_json_no_redirects, not
Stage 8's urllib helper, whose default redirect handling would resend the Authorization header to the redirect
target. It sends requests only to https://openrouter.ai/ and follows no redirect: any 3xx is RedirectRefused (only
its status is kept, never the Location), so no credential can be forwarded to another URL or downgraded to HTTP. preflight_v1 refuses a paid run on
any problem. readiness_v1 reports the same checks for the standalone --preflight-only, keeping confirmed provider
metadata, local configuration and REQUIRES_A_MODEL_RESPONSE apart. Procedure: docs/stage9_v1_live_run.md.

A paid run needs authorize_live(): the budget confirmed equal to the target AND the separate PAID_RUN_AUTHORIZATION
text, both before any request or file. Publication stays "not_authorized" (investigation_agent_v1 has no other
status).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
import math
import os
import urllib.request
from pathlib import Path

from . import investigation_agent as ia
from . import investigation_agent_v1 as av1
from . import investigation_live as live

__all__ = ["MAX_BEST_EFFORT_KEY_LIMIT_USD", "MAX_TARGET_USD", "PAID_RUN_AUTHORIZATION", "STRICT", "GuardedModelV1",
           "KeyPolicy", "LiveRefused", "authorize_live",
           "budget_v1", "build_model_v1", "cost_summary", "key_limit_problems", "ledger_problems", "plan_v1",
           "preflight_v1", "readiness_v1"]

MAX_TARGET_USD = 0.50
MAX_BEST_EFFORT_KEY_LIMIT_USD = 50.0
PAID_RUN_AUTHORIZATION = "I authorize one paid Stage 9 v1 investigation"
KNOWN_STOP_REASONS = frozenset({"end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal"})
MANAGEMENT_KEY_ENV = "OPENROUTER_MANAGEMENT_KEY"
KEY_FIELDS = ("limit", "limit_remaining", "limit_reset", "usage")
KEY_TOLERANCE_USD = 1e-6
LEDGER_TOTALS = ("known_cost_usd", "complete", "unknown_cost_attempts", "reported_cost_usd")
DROPPABLE = frozenset({"betas", "fallbacks", "output_config", "cache_control"})


class LiveRefused(PermissionError):
    """A paid run is not authorized or not ready (nothing was sent)."""


@dataclass(frozen=True)
class KeyPolicy:
    """Which provider key limits a run accepts. "strict" (default): at most MAX_TARGET_USD. "best_effort" (explicit):
    at most MAX_BEST_EFFORT_KEY_LIMIT_USD, and only the key whose limit equals acknowledged_limit_usd."""
    mode: str = "strict"
    acknowledged_limit_usd: float | None = None

    def __post_init__(self):
        if self.mode not in ("strict", "best_effort"):
            raise ValueError("key-limit mode must be 'strict' or 'best_effort'")
        ack = self.acknowledged_limit_usd
        if self.mode == "strict" and ack is not None:
            raise ValueError("--acknowledge-key-limit-usd is only for --key-limit-mode best-effort")
        if self.mode == "best_effort" and (not _usd(ack) or not 0 < ack <= MAX_BEST_EFFORT_KEY_LIMIT_USD):
            raise ValueError("best-effort mode needs --acknowledge-key-limit-usd: the key's actual limit, a finite "
                             f"positive number of at most {MAX_BEST_EFFORT_KEY_LIMIT_USD:.2f}")

    @property
    def max_key_limit_usd(self):
        return MAX_TARGET_USD if self.mode == "strict" else MAX_BEST_EFFORT_KEY_LIMIT_USD

    def exposure(self, target_usd, key_limit_usd=None, remaining_usd=None):
        if self.mode == "strict":
            return (f"strict: the key's provider-side limit (at most ${MAX_TARGET_USD:.2f}) is the provider's to "
                    f"enforce; the ${target_usd:.2f} local target is best-effort and can be passed by one request")
        avail = "" if remaining_usd is None else f" (${remaining_usd} remaining at preflight)"
        limit = "" if key_limit_usd is None else f" ${key_limit_usd}"
        return (f"best_effort: the ${target_usd:.2f} local target is NOT a spending ceiling. Spending can exceed it, "
                f"potentially up to the available credit on the{limit} key{avail}, subject to provider enforcement")


STRICT = KeyPolicy()


def plan_v1(**kw):
    """A LivePlan for the v1 agent: format contract-v1/1, OpenRouter, target at most MAX_TARGET_USD and at most
    Contract v1's 12 tool calls."""
    plan = live.LivePlan(findings_format=av1.FINDINGS_FORMAT, **kw)
    if plan.provider != "openrouter":
        raise ValueError("the v1 integration uses OpenRouter only (provider-reported cost per request)")
    if not plan.target_usd <= MAX_TARGET_USD:
        raise ValueError(f"target_usd must be at most {MAX_TARGET_USD:.2f}")
    if not 1 <= plan.max_tool_calls <= ia.MAX_TOOL_CALLS:
        raise ValueError(f"max_tool_calls must be 1-{ia.MAX_TOOL_CALLS} (Contract v1)")
    return plan


def budget_v1(plan, prices):
    """(Budget, estimate): investigation_live.plan_budget for the v1 prompt and schemas, plus the request size the
    one-request reserve covers (max_request_chars). Refused when the worst case could pass MAX_TARGET_USD."""
    budget, estimate = live.plan_budget(plan, prices)
    if estimate["estimated_worst_case_usd"] > min(plan.target_usd, MAX_TARGET_USD) + 1e-9:
        raise ValueError("the estimated worst case exceeds the target")
    estimate["max_request_chars"] = int(estimate["worst_request_input_tokens"] * live.CHARS_PER_TOKEN)
    return budget, estimate


def _input_tokens(entry):
    """The input tokens the Claude adapter derives from this attempt's usage (cache counts included), else None."""
    n = entry.get("input_tokens")
    return None if n is None else n + (entry.get("cache_read_input_tokens") or 0) + \
        (entry.get("cache_creation_input_tokens") or 0)


class GuardedModelV1:
    """The model investigate_v1 sees: Stage 8's ReportedCostGuard (over the Claude adapter and GuardedClient) with the
    Stage 9 request and response checks of the module docstring."""

    def __init__(self, inner, ledger, model_id, max_request_chars):
        self.inner, self.ledger, self.model_id, self.max_request_chars = inner, ledger, model_id, max_request_chars

    def respond(self, system, tools, messages, limits=None):
        if system != av1.SYSTEM_PROMPT_V1 or tools != av1.TOOL_SPECS_V1:
            raise ia.ModelUnavailable("not_v1_schemas", "only the v1 prompt and closed schemas are sent")
        size = len(json.dumps({"system": system, "tools": tools, "messages": messages}))
        if size > self.max_request_chars:
            raise ia.ModelUnavailable("request_too_large", "the request is larger than the budget's reserve covers")
        before = len(self.ledger.entries)
        reply = self.inner.respond(system, tools, messages, limits=limits)
        new = self.ledger.entries[before:]
        if len(new) != 1:
            raise ia.ModelUnavailable("unexpected_attempts", "a reply must come from exactly one attempt")
        entry = new[0]
        problem = None
        if entry.get("model_served") != self.model_id:
            problem = "unexpected_model"
        elif reply.stop_reason not in KNOWN_STOP_REASONS:
            problem = "unexpected_stop_reason"
        elif (reply.input_tokens, reply.output_tokens) != (_input_tokens(entry), entry.get("output_tokens")):
            problem = "usage_mismatch"
        if problem:
            entry["withheld"] = problem
            raise ia.ModelUnavailable(problem, "the provider's reply is withheld")
        entry["reply_used"] = True
        return reply


def build_model_v1(plan, estimate, *, env=None, http_client=None, compat=None):
    """(GuardedModelV1, Ledger) over investigation_live.build_model (the one billing and accounting path)."""
    inner, ledger = live.build_model(plan, env=env, http_client=http_client, compat=compat)
    return GuardedModelV1(inner, ledger, plan.model_id, estimate["max_request_chars"]), ledger


def authorize_live(plan, confirm_budget_usd, authorization):
    """Before any request or file: the budget confirmed equal to the target, and the separate authorization text."""
    try:
        live._confirmation(confirm_budget_usd, plan.target_usd)
    except PermissionError as exc:
        raise LiveRefused(str(exc)) from None
    if authorization != PAID_RUN_AUTHORIZATION:
        raise LiveRefused(f'a paid run needs --authorize-paid-run "{PAID_RUN_AUTHORIZATION}"')


def _usd(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def key_limit_problems(raw, target_usd, policy=STRICT):
    """(problems, limit): OpenRouter's key metadata (GET /api/v1/key, as received) must show a finite configured
    credit limit within the policy (strict: at most MAX_TARGET_USD; best_effort: at most MAX_BEST_EFFORT_KEY_LIMIT_USD
    and equal to the acknowledged limit) that never resets automatically, remaining credit covering the target, and
    consistent values (0 <= remaining = limit - usage). Every field must be present; nothing is assumed."""
    d = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(d, dict):
        return ["the key metadata is malformed"], None
    missing = [k for k in KEY_FIELDS if k not in d]
    if missing:
        return [f"the key metadata lacks {missing}"], None
    limit, remaining, usage, reset = d["limit"], d["limit_remaining"], d["usage"], d["limit_reset"]
    problems = []
    if limit is None:
        problems.append("the key has no credit limit (unlimited): configure a limit of at most "
                        f"${policy.max_key_limit_usd:.2f}")
    elif not _usd(limit) or limit <= 0:
        problems.append("the key's credit limit is not a positive number")
    elif limit > policy.max_key_limit_usd + KEY_TOLERANCE_USD:
        problems.append(f"the key's credit limit ${limit} is above ${policy.max_key_limit_usd:.2f}" + (
            " (an existing higher-limit key needs --key-limit-mode best-effort with its limit acknowledged)"
            if policy.mode == "strict" and limit <= MAX_BEST_EFFORT_KEY_LIMIT_USD + KEY_TOLERANCE_USD else ""))
    elif policy.mode == "best_effort" and limit <= MAX_TARGET_USD + KEY_TOLERANCE_USD:
        problems.append(f"the key's limit ${limit} is within strict mode: use the default --key-limit-mode strict "
                        "(best-effort is only for an existing higher-limit key)")
    elif policy.mode == "best_effort" and abs(limit - policy.acknowledged_limit_usd) > KEY_TOLERANCE_USD:
        problems.append(f"the acknowledged key limit ${policy.acknowledged_limit_usd} is not the key's verified "
                        f"limit ${limit}")
    if not _usd(remaining) or remaining < 0:
        problems.append("the key's remaining credit is not a non-negative number")
    if not _usd(usage) or usage < 0:
        problems.append("the key's usage is not a non-negative number")
    if reset is not None:
        problems.append("the key's limit resets automatically (limit_reset is set): use a key whose limit never "
                        "resets (limit_reset null)")
    if "is_free_tier" in d and not isinstance(d["is_free_tier"], bool):
        problems.append("the key metadata is malformed (is_free_tier)")
    if problems:
        return problems, None
    if remaining > limit + KEY_TOLERANCE_USD or abs(limit - usage - remaining) > KEY_TOLERANCE_USD:
        return ["the key metadata is inconsistent (remaining credit is not limit - usage)"], None
    if remaining < target_usd - KEY_TOLERANCE_USD:
        return [f"the key's remaining credit ${remaining} does not cover the ${target_usd:.2f} target"], None
    return [], float(limit)


METADATA_ORIGIN = "https://openrouter.ai/"


class RedirectRefused(RuntimeError):
    """A metadata GET was answered with a redirect, which is never followed (it could carry the credential)."""

    def __init__(self, code):
        super().__init__(f"HTTP {code} redirect refused (not followed)")
        self.code = code


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise RedirectRefused(code)


def get_json_no_redirects(url, headers=None, timeout=20.0, *, handlers=()):
    """A metadata GET: only to METADATA_ORIGIN over HTTPS, no redirect followed (RedirectRefused for any 3xx), only a
    200 answer accepted. handlers: extra urllib handlers (tests inject an offline transport)."""
    if not isinstance(url, str) or not url.startswith(METADATA_ORIGIN):
        raise ValueError("metadata requests go to https://openrouter.ai/ only")
    opener = urllib.request.build_opener(_RefuseRedirects, *handlers)
    req = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})}, method="GET")
    with opener.open(req, timeout=timeout) as r:                  # noqa: S310 (fixed https origin)
        status = getattr(r, "status", None) or r.getcode()
        if status != 200:
            raise RuntimeError(f"HTTP {status} is not a metadata answer")
        return json.loads(r.read().decode("utf-8"))


def endpoints_url(model_id):
    """OpenRouter's free per-model endpoint list (GET /api/v1/models/{author}/{slug}/endpoints)."""
    return f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"


def _tokens_field(e, name, required):
    """(value, problem): a token capacity of the endpoint: a positive integer; None when absent or null and not
    required. Anything else (text, bool, float, zero or negative) is a problem."""
    v = e.get(name)
    if v is None:
        return None, (f"the Anthropic endpoint states no {name}: its capacity cannot be verified" if required
                      else None)
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        return None, f"the Anthropic endpoint's {name} is malformed"
    return v, None


def capacity_problems(e, plan, estimate):
    """(problems, record): the endpoint's capacity against the plan's largest request: the planned worst-case input
    (estimate's worst_request_input_tokens, as the one-request reserve assumes) plus the reserved output
    (per_request_max_tokens) within context_length (required); the input within max_prompt_tokens and the output
    within max_completion_tokens when they are stated."""
    problems = []
    ctx_len, p1 = _tokens_field(e, "context_length", True)
    max_prompt, p2 = _tokens_field(e, "max_prompt_tokens", False)
    max_out, p3 = _tokens_field(e, "max_completion_tokens", False)
    problems += [p for p in (p1, p2, p3) if p]
    if estimate is None:
        problems.append("the planned request size is unknown (no budget estimate): capacity cannot be verified")
        return problems, None
    need_in, need_out = estimate["worst_request_input_tokens"], plan.per_request_max_tokens
    if ctx_len is not None and need_in + need_out > ctx_len:
        problems.append(f"the Anthropic endpoint's context_length {ctx_len} is below the planned largest request "
                        f"({need_in} input + {need_out} output tokens)")
    if max_prompt is not None and need_in > max_prompt:
        problems.append(f"the Anthropic endpoint's max_prompt_tokens {max_prompt} is below the planned "
                        f"{need_in} input tokens")
    if max_out is not None and need_out > max_out:
        problems.append("the Anthropic endpoint's output allowance is below the per-request max_tokens")
    return problems, {"context_length": ctx_len, "max_prompt_tokens": max_prompt, "max_completion_tokens": max_out,
                      "planned_input_tokens": need_in, "planned_output_tokens": need_out}


def anthropic_endpoint_problems(raw, prices, plan, estimate=None):
    """(problems, summary): the model's endpoint list (as received) must include the Anthropic provider endpoint the
    routing is pinned to, priced (per token, like the model list) no higher than the prices the budget uses, with
    no per-request fee, and with the capacity for the plan's largest request (capacity_problems)."""
    d = raw.get("data") if isinstance(raw, dict) else None
    eps = d.get("endpoints") if isinstance(d, dict) else None
    if not isinstance(eps, list) or not all(isinstance(e, dict) for e in eps):
        return ["the model's endpoint list is malformed"], None
    pinned = [e for e in eps if str(e.get("tag") or "").lower() == "anthropic" or
              str(e.get("provider_name") or "").lower() == "anthropic"]
    if len(pinned) != 1:
        return [f"the model has {len(pinned)} Anthropic endpoints (exactly 1 is required for the routing pin)"], None
    e = pinned[0]
    try:
        own = live.Prices.from_openrouter(e.get("pricing") or {}, "endpoint")
    except (ValueError, TypeError) as exc:
        return [f"the Anthropic endpoint's prices are not usable ({exc})"], None
    problems = []
    if prices is None or own.input_usd_mtok > prices.input_usd_mtok + 1e-12 or \
            own.output_usd_mtok > prices.output_usd_mtok + 1e-12:
        problems.append("the Anthropic endpoint is priced above the prices the budget uses")
    p, capacity = capacity_problems(e, plan, estimate)
    problems += p
    return problems, {"provider_name": e.get("provider_name"), "tag": e.get("tag"),
                      "prices_usd_per_mtok": {"input": own.input_usd_mtok, "output": own.output_usd_mtok},
                      "capacity": capacity, "status": e.get("status")}


def _secret(env, name):
    return (env.get(name) or "").strip()


def endpoint_credential(plan, env):
    """(label, key, problem) for the endpoint-list GET: the management key when configured (for this GET only), else
    the inference key; a management key equal to the inference key, or no key at all, is a problem."""
    inference, management = _secret(env, live.PROVIDERS[plan.provider].key_env), _secret(env, MANAGEMENT_KEY_ENV)
    if management and management == inference:
        return None, None, (f"{MANAGEMENT_KEY_ENV} equals the inference key: the capped inference key and the "
                            "management key must be different keys")
    if management:
        return MANAGEMENT_KEY_ENV, management, None
    if inference:
        return live.PROVIDERS[plan.provider].key_env, inference, None
    return None, None, "no key to authenticate the endpoint-list GET"


def _get_endpoints(get, plan, env):
    """(raw, problem, credential label): the model's endpoint list, Bearer-authenticated (endpoint_credential).
    Only the error class and HTTP status are kept; a 403 is reported as such, never skipped or retried."""
    label, key, problem = endpoint_credential(plan, env)
    if problem:
        return None, problem, label
    try:
        return get(endpoints_url(plan.model_id), {"Authorization": f"Bearer {key}"}), None, label
    except Exception as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status", None)
        if status == 403:
            hint = ("refused even with the management key" if label == MANAGEMENT_KEY_ENV else
                    f"OpenRouter may require a management key: set {MANAGEMENT_KEY_ENV} (used for this read-only "
                    "GET only)")
            return None, f"the model's endpoint list was refused (HTTP 403): {hint}", label
        return None, (f"the model's endpoint list is unavailable ({type(exc).__name__}"
                      f"{f', HTTP {status}' if isinstance(status, int) else ''})"), label


def request_configuration(plan, compat):
    """(problems, record): the request the guarded adapter WOULD send for this plan, built offline (placeholder key,
    a transport that refuses every request, nothing sent): OpenRouter's Messages URL, the routing pinned to the
    Anthropic provider without provider fallbacks, no Anthropic-only fields, no thinking, the plan's model and
    max_tokens, SDK retries 0 (also when re-requested), the beta endpoint closed. Locally confirmed, not by the
    provider."""
    import httpx2
    sent = []

    def refuse(request):
        sent.append(str(request.url))
        raise RuntimeError("the preflight never sends a model request")
    model, _ = live.build_model(plan, env={live.PROVIDERS[plan.provider].key_env: "preflight-placeholder-not-a-key"},
                                http_client=httpx2.Client(transport=httpx2.MockTransport(refuse)), compat=compat)
    adapter = model.inner
    guarded, sdk = adapter._client, adapter._client._client
    kwargs, dropped = guarded.adapt(adapter.request(av1.SYSTEM_PROMPT_V1, av1.TOOL_SPECS_V1,
                                                    [{"role": "user", "content": "preflight"}]))
    try:
        guarded.beta
        beta_closed = False
    except AttributeError:
        beta_closed = True
    record = {"messages_url": str(sdk.base_url).rstrip("/") + "/v1/messages",
              "routing": kwargs.get("extra_body", {}).get("provider"), "model": kwargs.get("model"),
              "max_tokens": kwargs.get("max_tokens"), "dropped_fields": dropped,
              "sdk_max_retries": sdk.max_retries, "retries_when_re_requested": guarded.with_options(
                  max_retries=5)._client.max_retries, "adapter_fallbacks": adapter.fallbacks,
              "adapter_max_retries": adapter.max_retries, "beta_endpoint_closed": beta_closed,
              "sent_fields": sorted(kwargs), "requests_sent": len(sent)}
    want = {"messages_url": "https://openrouter.ai/api/v1/messages",
            "routing": {"order": ["anthropic"], "allow_fallbacks": False}, "model": plan.model_id,
            "sdk_max_retries": 0, "retries_when_re_requested": 0, "adapter_fallbacks": False,
            "adapter_max_retries": 0, "beta_endpoint_closed": True, "requests_sent": 0}
    problems = [f"request configuration: {k} is {record[k]!r}, not {v!r}" for k, v in want.items() if record[k] != v]
    if not isinstance(record["max_tokens"], int) or record["max_tokens"] > plan.per_request_max_tokens:
        problems.append("request configuration: max_tokens is above the plan's per-request allowance")
    banned = {"betas", "fallbacks", "thinking", "models", "route"} & set(kwargs)
    if banned:
        problems.append(f"request configuration: would send {sorted(banned)}")
    return problems, record


def _preflight_v1(plan, *, manifest, env=None, http_get=None, policy=STRICT):
    """(preflight record, problems). Never a model request: Stage 8's online preflight (free GETs: model list, key
    status), the key metadata exactly as received (key_limit_problems), the model's Anthropic endpoint
    (anthropic_endpoint_problems), the v1 budget estimate (budget_v1) and the request configuration
    (request_configuration). A required item that cannot be verified is a problem."""
    if Path(manifest).resolve() != Path(live.MANIFEST).resolve():
        return {}, ["the live preflight checks the tracked split manifest: run with the default --manifest"]
    get, received = http_get or get_json_no_redirects, {}
    key_url = live.PROVIDERS[plan.provider].key_url

    def recording_get(url, headers=None):
        out = get(url, headers)
        if url == key_url:
            received["key"] = copy.deepcopy(out)
        return out
    pre = live.preflight(plan, env=env, online=True, http_get=recording_get)
    problems = [c["name"] for c in pre["checks"] if not c["ok"]]
    if not pre["ready_for_live"]:
        problems.append("ready_for_live")
    pre["provider_key_limit_usd"] = pre["provider_key_remaining_usd"] = None
    pre["key_policy"] = {"mode": policy.mode, "acknowledged_limit_usd": policy.acknowledged_limit_usd}
    if "key" not in received:
        problems.append("the key metadata was not received")
    else:
        p, pre["provider_key_limit_usd"] = key_limit_problems(received["key"], plan.target_usd, policy)
        problems += p
        if not p:
            pre["provider_key_remaining_usd"] = float(received["key"]["data"]["limit_remaining"])
    env = os.environ if env is None else env
    prices = live.Prices(**pre["prices"]) if pre.get("prices_basis") == "fetched" else None
    estimate = None
    if prices is None:
        problems.append("no prices were fetched this run")
    else:
        try:
            estimate = pre["estimate_v1"] = budget_v1(plan, prices)[1]
        except ValueError as exc:
            problems.append(f"budget: {exc}")
    raw, problem, pre["endpoint_metadata_credential"] = _get_endpoints(get, plan, env)
    if problem:
        problems.append(problem)
        pre["anthropic_endpoint"] = None
    else:
        try:
            p, pre["anthropic_endpoint"] = anthropic_endpoint_problems(raw, prices, plan, estimate)
        except Exception as exc:
            p, pre["anthropic_endpoint"] = [f"the model's endpoint list is malformed ({type(exc).__name__})"], None
        problems += p
    compat = pre.get("compatibility") or {}
    try:
        p, pre["request_configuration"] = request_configuration(
            plan, {"output_config": compat.get("keep_output_config", True),
                   "cache_control": compat.get("keep_cache_control", True)})
    except Exception as exc:
        p, pre["request_configuration"] = [f"request configuration could not be built ({type(exc).__name__})"], None
    problems += p
    return pre, problems


PREFLIGHT_EVIDENCE = ("online", "ready_for_live", "checks", "prices", "prices_basis", "key_status", "key_policy",
                      "provider_key_limit_usd", "provider_key_remaining_usd", "anthropic_endpoint",
                      "endpoint_metadata_credential", "estimate_v1", "request_configuration")
PINNED_ROUTING = {"order": ["anthropic"], "allow_fallbacks": False}


def saved_preflight_problems(pre, plan, policy):
    """(problems, prices): a saved live run's preflight record (preflight_v1's return value) re-checked as evidence:
    every field present; a passing online preflight with fetched prices, an Anthropic endpoint and a request
    configuration that sent nothing; the key policy the run records; and the recorded key metadata accepted by the
    same validator as at run time (key_limit_problems with that policy and target), giving exactly the recorded
    provider limit and remaining credit. prices: the fetched prices (None when unusable)."""
    if not isinstance(pre, dict):
        return ["a live run has no preflight record"], None
    missing = [k for k in PREFLIGHT_EVIDENCE if k not in pre]
    if missing:
        return [f"the live run's preflight record lacks {missing}"], None
    problems = []
    if pre["key_policy"] != {"mode": policy.mode, "acknowledged_limit_usd": policy.acknowledged_limit_usd}:
        problems.append("the preflight's key policy is not the run's key policy")
    checks = pre["checks"]
    if pre["online"] is not True or pre["ready_for_live"] is not True or pre["prices_basis"] != "fetched" or \
            not isinstance(checks, list) or not checks or not all(isinstance(c, dict) and c.get("ok") is True
                                                                  for c in checks):
        problems.append("the preflight record is not a passing online preflight with fetched prices")
    rc = pre["request_configuration"]
    if not isinstance(rc, dict) or rc.get("requests_sent") != 0 or rc.get("routing") != PINNED_ROUTING:
        problems.append("the preflight's request configuration is not the pinned, unsent configuration")
    if not isinstance(pre["anthropic_endpoint"], dict):
        problems.append("the preflight record has no verified Anthropic endpoint")
    key = pre["key_status"]
    p, limit = key_limit_problems({"data": key}, plan.target_usd, policy)
    problems += [f"the recorded key is not accepted in {policy.mode} mode: {x}" for x in p]
    if not p and (limit, float(key["limit_remaining"])) != (pre["provider_key_limit_usd"],
                                                            pre["provider_key_remaining_usd"]):
        problems.append("the recorded provider limit or remaining credit is not the key metadata's")
    try:
        prices = live.Prices(**pre["prices"])
    except TypeError:
        return problems + ["the preflight's prices are malformed"], None
    return problems, prices


def preflight_v1(plan, *, manifest, env=None, http_get=None, policy=STRICT):
    """_preflight_v1 for a paid run: LiveRefused unless every check passes (before any paid request); the preflight
    record (with provider_key_limit_usd) when ready."""
    pre, problems = _preflight_v1(plan, manifest=manifest, env=env, http_get=http_get, policy=policy)
    if problems:
        raise LiveRefused(f"preflight not ready: {problems}")
    return pre


# what only a real model response can show: none of it is checked by any preflight
REQUIRES_A_MODEL_RESPONSE = (
    "the served model string equals the requested model id (a mismatch withholds the first reply)",
    "every response reports usage.cost (a missing cost stops the run after that request)",
    "OpenRouter honours the routing pin (provider order anthropic, allow_fallbacks false) on generation",
    "the effort setting (output_config) and system cache_control are accepted on this endpoint",
    "adaptive thinking blocks round-trip unchanged across turns",
    "actual input tokens stay near the chars / 3 estimate behind the one-request reserve",
    "OpenRouter enforces the key's credit limit at or before the limit",
    "no other client uses the key during the run (a dedicated key is assumed, not verifiable)",
)


def readiness_v1(plan, *, manifest, env=None, http_get=None, policy=STRICT):
    """The standalone, read-only preflight: _preflight_v1's checks as a report, never a model request and never a
    raise on a failed check. ready only when nothing failed. Confirmed provider metadata, locally confirmed
    configuration and assumptions only a model response can test are kept apart."""
    pre, problems = _preflight_v1(plan, manifest=manifest, env=env, http_get=http_get, policy=policy)
    key = pre.get("key_status")
    return {"mode": "read-only preflight (free metadata GETs only; no model request, no investigation, no cost)",
            "ready": not problems, "problems": problems,
            "plan": {"provider": plan.provider, "model": plan.model_id, "target_usd": plan.target_usd,
                     "max_model_turns": plan.max_model_turns, "max_tool_calls": plan.max_tool_calls,
                     "per_request_max_tokens": plan.per_request_max_tokens, "effort": plan.effort},
            "spending": {"authorization_mode": policy.mode, "local_target_usd": plan.target_usd,
                         "acknowledged_key_limit_usd": policy.acknowledged_limit_usd,
                         "provider_key_limit_usd": pre.get("provider_key_limit_usd"),
                         "provider_key_remaining_usd": pre.get("provider_key_remaining_usd"),
                         "exposure": policy.exposure(plan.target_usd, pre.get("provider_key_limit_usd"),
                                                     pre.get("provider_key_remaining_usd")),
                         "paid_run_authorized": False},
            "confirmed_provider_metadata": {
                "model": pre.get("model_metadata"), "prices_basis": pre.get("prices_basis"),
                "prices_usd_per_mtok": None if not pre.get("prices") else {
                    "input": pre["prices"]["input_usd_mtok"], "output": pre["prices"]["output_usd_mtok"]},
                "key": None if key is None else {k: key.get(k) for k in ("limit", "limit_remaining", "limit_reset",
                                                                        "usage", "is_free_tier")},
                "provider_key_limit_usd": pre.get("provider_key_limit_usd"),
                "anthropic_endpoint": pre.get("anthropic_endpoint")},
            "credentials": {"inference_and_key_status": live.PROVIDERS[plan.provider].key_env,
                            "endpoint_metadata_get": pre.get("endpoint_metadata_credential"),
                            "note": f"{MANAGEMENT_KEY_ENV}, when set, is used only for the read-only endpoint-list "
                                    "GET; model requests and /api/v1/key use the capped inference key only"},
            "estimate": pre.get("estimate_v1"),
            "local_configuration": {"request": pre.get("request_configuration"),
                                    "compatibility": pre.get("compatibility"),
                                    "checks": pre.get("checks")},
            "requires_a_model_response": list(REQUIRES_A_MODEL_RESPONSE)}


def cost_summary(ledger_json, plan, provider_key_limit_usd=None, policy=STRICT, provider_key_remaining_usd=None):
    """What was spent, stated apart from what was planned: target_usd is the local target (not a guaranteed maximum);
    provider_key_limit_usd the key's provider-side limit read at preflight (None: no provider, e.g. a dry run);
    actual_spend_usd the provider-reported total, complete only when every attempt reported a cost;
    authorization_mode and exposure the key-limit mode and what it means for spending."""
    spend = ledger_json["known_cost_usd"]
    return {"ledger": ledger_json, "target_usd": plan.target_usd, "provider_key_limit_usd": provider_key_limit_usd,
            "provider_key_remaining_usd": provider_key_remaining_usd, "authorization_mode": policy.mode,
            "acknowledged_key_limit_usd": policy.acknowledged_limit_usd,
            "exposure": policy.exposure(plan.target_usd, provider_key_limit_usd, provider_key_remaining_usd),
            "actual_spend_usd": spend, "actual_spend_complete": ledger_json["complete"],
            "over_target": spend > plan.target_usd,
            "note": "actual_spend_usd is the sum of provider-reported costs (OpenRouter usage.cost); when incomplete, "
                    "attempts without a reported cost are not included and the true spend may be higher. The target "
                    "is a local bound that can be passed by one request; the provider key limit is the provider's "
                    "to enforce. Neither is a guaranteed maximum charge."}


def ledger_problems(doc, ledger_json, plan):
    """The cost ledger (Ledger.to_json) against a saved contract-v1/1 report and its plan: the totals re-add, every
    attempt is a request of this plan, each model turn of the trace is one used, cost-reported attempt with the same
    tokens (in order), and the only other attempt allowed is the last, followed by a model_error stop. Returns the
    problems (empty: consistent)."""
    try:
        reqs = ledger_json["requests"]
        problems = []
        known = round(sum(r["cost_usd"] for r in reqs if r.get("cost_status") == "reported"), 8)
        unknown = sum(r.get("cost_status") != "reported" for r in reqs)
        want = {"known_cost_usd": known, "complete": unknown == 0, "unknown_cost_attempts": unknown,
                "reported_cost_usd": known if reqs and unknown == 0 else None}
        wrong = [k for k in LEDGER_TOTALS if ledger_json.get(k) != want[k]]
        if wrong:
            problems.append(f"the ledger totals do not re-add ({', '.join(wrong)})")
        for n, r in enumerate(reqs):
            if r.get("model_requested") != plan.model_id or not isinstance(r.get("max_tokens"), int) or \
                    r["max_tokens"] > plan.per_request_max_tokens or not set(r.get("dropped_fields", [])) <= DROPPABLE:
                problems.append(f"ledger request {n} is not a request of this plan")
        turns = [t for t in doc["trace"] if t.get("kind") == "model_turn"]
        used = [r for r in reqs if r.get("reply_used") is True]
        unused = [n for n, r in enumerate(reqs) if r.get("reply_used") is not True]
        if len(used) != len(turns):
            problems.append(f"{len(turns)} model turns but {len(used)} used ledger attempts")
        for n, (t, r) in enumerate(zip(turns, used)):
            if r.get("cost_status") != "reported" or (t.get("input_tokens"), t.get("output_tokens")) != (
                    _input_tokens(r), r.get("output_tokens")):
                problems.append(f"model turn {n} is not its ledger attempt (cost or tokens)")
        if unused:
            trace = doc["trace"]
            if unused != [len(reqs) - 1] or len(trace) < 2 or trace[-2].get("kind") != "model_error" or \
                    doc["stopping_reason"] != "model_unavailable":
                problems.append("an attempt whose reply was not used must be the last, before a model_error stop")
        return problems
    except (KeyError, TypeError, AttributeError) as exc:
        return [f"the cost ledger is malformed ({type(exc).__name__})"]
