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

Before any live run, configure a DEDICATED, non-shared OpenRouter key with such a limit. The code cannot know whether
the key is used elsewhere at the same time, and cannot guarantee the provider's billing behaviour.

A paid run needs authorize_live(): the budget confirmed equal to the target AND the separate PAID_RUN_AUTHORIZATION
text, both before any request or file. Publication stays "not_authorized" (investigation_agent_v1 has no other
status).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from . import investigation_agent as ia
from . import investigation_agent_v1 as av1
from . import investigation_live as live

__all__ = ["MAX_TARGET_USD", "PAID_RUN_AUTHORIZATION", "GuardedModelV1", "LiveRefused", "authorize_live",
           "budget_v1", "build_model_v1", "cost_summary", "key_limit_problems", "ledger_problems", "plan_v1",
           "preflight_v1"]

MAX_TARGET_USD = 0.50
PAID_RUN_AUTHORIZATION = "I authorize one paid Stage 9 v1 investigation"
KNOWN_STOP_REASONS = frozenset({"end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal"})
KEY_FIELDS = ("limit", "limit_remaining", "limit_reset", "usage")
KEY_TOLERANCE_USD = 1e-6
LEDGER_TOTALS = ("known_cost_usd", "complete", "unknown_cost_attempts", "reported_cost_usd")
DROPPABLE = frozenset({"betas", "fallbacks", "output_config", "cache_control"})


class LiveRefused(PermissionError):
    """A paid run is not authorized or not ready (nothing was sent)."""


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


def key_limit_problems(raw, target_usd):
    """(problems, limit): OpenRouter's key metadata (GET /api/v1/key, as received) must show a finite configured
    credit limit of at most MAX_TARGET_USD that never resets automatically, remaining credit covering the target, and
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
                        f"${MAX_TARGET_USD:.2f}")
    elif not _usd(limit) or limit <= 0:
        problems.append("the key's credit limit is not a positive number")
    elif limit > MAX_TARGET_USD + KEY_TOLERANCE_USD:
        problems.append(f"the key's credit limit ${limit} is above ${MAX_TARGET_USD:.2f}")
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


def preflight_v1(plan, *, manifest, env=None, http_get=None):
    """investigation_live.preflight online (free GETs only), required ready, plus the key's provider-side credit
    limit (key_limit_problems on the key metadata exactly as received). LiveRefused otherwise, before any paid
    request; the preflight record (with provider_key_limit_usd) when ready."""
    if Path(manifest).resolve() != Path(live.MANIFEST).resolve():
        raise LiveRefused("the live preflight checks the tracked split manifest: run with the default --manifest")
    get, received = http_get or live._get_json, {}

    def recording_get(url, headers=None):
        out = get(url, headers)
        if url == live.PROVIDERS[plan.provider].key_url:
            received["key"] = copy.deepcopy(out)
        return out
    pre = live.preflight(plan, env=env, online=True, http_get=recording_get)
    problems = [c["name"] for c in pre["checks"] if not c["ok"]]
    if not pre["ready_for_live"]:
        problems.append("ready_for_live")
    if "key" not in received:
        problems.append("the key metadata was not received")
    else:
        p, limit = key_limit_problems(received["key"], plan.target_usd)
        problems += p
        pre["provider_key_limit_usd"] = limit
    if problems:
        raise LiveRefused(f"preflight not ready: {problems}")
    return pre


def cost_summary(ledger_json, plan, provider_key_limit_usd=None):
    """What was spent, stated apart from what was planned: target_usd is the local target (not a guaranteed maximum);
    provider_key_limit_usd the key's provider-side limit read at preflight (None: no provider, e.g. a dry run);
    actual_spend_usd the provider-reported total, complete only when every attempt reported a cost."""
    spend = ledger_json["known_cost_usd"]
    return {"ledger": ledger_json, "target_usd": plan.target_usd, "provider_key_limit_usd": provider_key_limit_usd,
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
