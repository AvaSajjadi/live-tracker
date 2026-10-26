"""Phase 3B.2: the Stage 9 v1 agent with a model through the guarded Stage 8 infrastructure, on the verified
CalMS21 validation inputs of Phase 3B.1. Offline by default: no network, no key, no cost.

    python scripts/investigation_v1_live.py ARCHIVE --archive-sha256 HEX --out DIR [--validation-run run5]
        [--model anthropic/claude-sonnet-5.5] [--target-usd 0.50] [--max-model-turns 8] [--max-tool-calls 8]
    python scripts/investigation_v1_live.py ARCHIVE ... --live --confirm-budget-usd 0.50 \\
        --authorize-paid-run "I authorize one paid Stage 9 v1 investigation"       # PAID: never run in review
    python scripts/investigation_v1_live.py --preflight-only [--model M] [--target-usd T]   # read-only, no model
    python scripts/investigation_v1_live.py --verify-saved DIR                     # offline re-verification

--preflight-only (investigation_live_v1.readiness_v1) makes only free metadata GETs (model list, key status, the
model's endpoint list) and builds the request it would send offline: it never calls the model, starts an
investigation, writes a file or prints the key. It prints the same checks a paid run requires, separated into
confirmed provider metadata, locally confirmed configuration and what only a real model response can show; exit 0
ready, 2 not ready. The step-by-step live procedure is docs/stage9_v1_live_run.md.

Steps (every one reused; nothing here is new scientific, billing or verification logic):
1. inputs      investigation_v1_validation.load_data (Phase 3B.1): the verified score drop, pose cache, split
               manifest and expected scope; refused otherwise, nothing substituted. --validation-run: its
               check_validation_run (same score drop and data fingerprint) before any request.
2. preflight   live only: investigation_live_v1.preflight_v1 (Stage 8's online preflight, free GETs only, plus the
               key's provider-side credit limit: finite, at most $0.50, never resetting, covering the target,
               consistent). The dry run uses Stage 8's offline price snapshot instead.
3. investigate investigation_agent_v1.investigate_v1 with investigation_live_v1.build_model_v1: the Stage 8 Claude
               adapter, GuardedClient, Ledger and ReportedCostGuard, with the Stage 9 checks of GuardedModelV1, under
               investigation_live.plan_budget's Budget (target at most $0.50). The dry run sends the same SDK
               requests to EmulatedOpenRouterV1, an in-process stand-in for the OpenRouter endpoint answered by Phase
               3B.1's deterministic offline investigator (not a language model; its tokens and costs are emulated).
4. report      investigation_report.json, re-read and checked (Phase 3B.1 check_report: verify_report_v1, data,
               split, sessions); investigation_report.md rendered from it (render_markdown_v1); the cost ledger
               bound to the saved trace (ledger_problems); with --validation-run, Phase 3B.1's fail-closed
               check_query_against_run. The report and the ledger are always written once a model was built, so
               spending stays accounted; a failed check makes the status refused or failed, never verified.
run.json records the plan, budget, estimate, preflight, cost ledger, every step and the files' SHA-256. No file holds
an API key (the inference key or the management key). Its cost section states target_usd (local), provider_key_limit_usd, actual_spend_usd and over_target;
none of them is a guaranteed maximum charge. A live run needs a DEDICATED, non-shared OpenRouter key with a credit
limit of at most $0.50 and no reset: the code cannot know whether the key is used elsewhere, nor guarantee the
provider's billing. Publication is "not_authorized". Exit 0 verified, 2 refused, 1 otherwise (incomplete, over target,
failed). DIR must be new or empty and git-ignored (or outside the repository).

Key-limit mode: --key-limit-mode strict (default) requires a non-resetting key limit of at most $0.50.
--key-limit-mode best-effort --acknowledge-key-limit-usd L accepts an existing non-resetting key whose verified limit
is exactly L (at most $50). The target stays at most $0.50 with every guard unchanged, but it is then best-effort
only: spending can exceed it, potentially up to the key's available credit, subject to OpenRouter's enforcement.
Neither mode authorizes a paid run; --live still needs --confirm-budget-usd and --authorize-paid-run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker import investigation_agent_v1 as av1  # noqa: E402
from live_tracker import investigation_live as live  # noqa: E402
from live_tracker import investigation_live_v1 as lv1  # noqa: E402

_spec = importlib.util.spec_from_file_location("investigation_v1_validation",
                                               Path(__file__).with_name("investigation_v1_validation.py"))
iv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iv)
vr, smoke = iv.vr, iv.smoke

OUTPUT_NAMES = iv.OUTPUT_NAMES
INVESTIGATION_ID = "investigation_v1_live"
REFUSALS = iv.REFUSALS + (lv1.LiveRefused, live.AmbientEnvironmentError)
LIVE_MODE = "LIVE: paid OpenRouter requests"
DRY_MODE = "dry run (offline: emulated endpoint, no network, no model, no cost)"
EMULATED_LABEL = ("EMULATED: no language model and no provider; Phase 3B.1's deterministic offline investigator "
                  "answered through an in-process OpenRouter emulation; tokens and costs are emulated, not spent")


# ---- the emulated endpoint (offline; no network) ----

class EmulatedOpenRouterV1:
    """In-process stand-in for POST /api/v1/messages (httpx2.MockTransport handler). It records what the SDK sends
    (never header values) and answers like the endpoint: each call is chosen by Phase 3B.1's OfflineInvestigatorV1
    from the tool results in the request (refused calls are skipped), with usage and cost in OpenRouter's layout.
    faults: {request number (1-based): kind} injects a provider failure (FAULTS)."""

    FAULTS = ("http_500", "http_429", "timeout", "garbage", "no_usage", "no_cost", "bad_cost", "other_model",
              "unknown_stop", "refusal", "no_tool", "unknown_tool", "extra_argument", "huge_usage")

    def __init__(self, faults=None, cost=None, output_tokens=300):
        self.faults, self.cost, self.output_tokens, self.requests = dict(faults or {}), cost, output_tokens, []
        assert set(self.faults.values()) <= set(self.FAULTS)

    def usage(self, body_chars, fault):
        n_in = 10_000_000 if fault == "huge_usage" else math.ceil(body_chars / live.CHARS_PER_TOKEN)
        cost = self.cost if self.cost is not None else round((n_in * 2.0 + self.output_tokens * 10.0) / 1e6, 8)
        usage = {"input_tokens": n_in, "output_tokens": self.output_tokens, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0, "cost": cost}
        if fault == "no_cost":
            usage.pop("cost")
        elif fault == "bad_cost":
            usage["cost"] = -1.0
        return usage

    def __call__(self, request):
        import httpx2
        body = json.loads(request.content)
        names = {k.lower() for k in request.headers}
        self.requests.append({"method": request.method, "url": str(request.url), "body": body,
                              "has_authorization": "authorization" in names, "has_x_api_key": "x-api-key" in names,
                              "has_anthropic_beta": "anthropic-beta" in names})
        fault = self.faults.get(len(self.requests))
        if fault in ("http_500", "http_429"):
            return httpx2.Response(int(fault[-3:]), json={"type": "error", "error": {"type": "api_error",
                                                                                      "message": "emulated"}})
        if fault == "timeout":
            raise httpx2.ReadTimeout("emulated timeout")
        if fault == "garbage":
            return httpx2.Response(200, content=b"<html>not a message</html>",
                                   headers={"content-type": "application/json"})
        done = [c for c in iv._transcript(body["messages"]) if not c[3]]
        name, args = iv.OfflineInvestigatorV1().decide(done)
        if fault == "unknown_tool":
            name = "run_python"
        elif fault == "extra_argument":
            args = {**args, "min_n_windows": 1}
        content = [{"type": "tool_use", "id": f"toolu_emu_{len(self.requests)}", "name": name, "input": args}]
        stop = "tool_use"
        if fault in ("refusal", "no_tool"):
            content, stop = [{"type": "text", "text": "emulated reply without a tool call"}], (
                "refusal" if fault == "refusal" else "end_turn")
        elif fault == "unknown_stop":
            stop = "pause_turn"
        msg = {"id": f"msg_emu_{len(self.requests)}", "type": "message", "role": "assistant",
               "model": "anthropic/claude-opus-5.5" if fault == "other_model" else body["model"],
               "content": content, "stop_reason": stop, "stop_sequence": None,
               "usage": self.usage(len(request.content), fault)}
        if fault == "no_usage":
            msg.pop("usage")
        return httpx2.Response(200, json=msg)


# ---- the run ----

def _secrets(env):
    """The credentials that must never be written or printed: the inference key and the management key."""
    return tuple(k for k in ((env.get(n) or "").strip() for n in (live.PROVIDERS["openrouter"].key_env,
                                                                   lv1.MANAGEMENT_KEY_ENV)) if k)


def _write(path, data, secrets):
    if any(k.encode() in data for k in secrets):            # never write a credential
        raise RuntimeError(f"refusing to write {path.name}: it contains an API key")
    smoke.create_exclusive(path, data)


def make_policy(args):
    """The key-limit mode (strict unless best-effort is chosen explicitly, with the key's limit acknowledged)."""
    return lv1.KeyPolicy(args.key_limit_mode.replace("-", "_"), args.acknowledge_key_limit_usd)


def make_plan(args):
    return lv1.plan_v1(model=args.model, question=args.question, target_usd=args.target_usd,
                       max_model_turns=args.max_model_turns, max_tool_calls=args.max_tool_calls,
                       cache_dir=str(args.cache_dir))


# ---- live progress (presentation only) ----
# run(progress=...) reports, as they happen: each stage starting / completing / failing, each recorded trace entry
# (projected by progress_event), "waiting for the provider" before each model request, the cost ledger's recorded
# spending after each reply, each output file once written, and the final status. Nothing here decides anything:
# a failing progress sink is dropped (_Progress) and the run, its files and its verification are as without one.

# What each progress field may show: explicit values taken from the code paths that produce them (checked against
# those sources by tests/test_live_agent_progress.py). Anything else gets the field's fixed fallback, so a model- or
# provider-supplied string is never shown merely because it looks like an identifier.
KNOWN_TOOLS = frozenset(t["name"] for t in av1.TOOL_SPECS_V1) | {av1.FINISH}
UNKNOWN_TOOL = "unknown tool"
TRACE_KINDS = frozenset(av1.TRACE_KINDS)
STOP_REASONS = frozenset(lv1.KNOWN_STOP_REASONS) | {"pause_turn"}          # the provider's reply stop reasons
TURN_OUTCOMES = frozenset({"malformed", "usage_unavailable", "over_budget"})
TOOL_ERROR_CODES = frozenset({"bad_window", "budget_exhausted", "invalid_request", "no_scores", "unknown_behavior",
                              "unknown_feature", "unknown_session", "unverified_bouts"})
REFUSAL_KINDS = frozenset({"final_answer_rejected", "follow_up_limit", "invalid_arguments", "pre_window_not_in_grid",
                           "repeated_call", "tool_budget_exhausted", "tool_contract_violation", "tool_failure",
                           "unauthorized", "unknown_evidence", "unknown_tool"}) | \
    frozenset(f"tool_refused:{c}" for c in TOOL_ERROR_CODES)
MODEL_ERROR_REASONS = frozenset({
    "api_error", "auth_error", "client_error", "connection_error", "rate_limited", "timeout",      # SDK errors
    "missing_api_key", "not_v1_schemas", "provider_cost_unknown", "provider_reported_budget", "request_too_large",
    "sdk_not_installed", "unexpected_attempts",                                                   # adapters
    "unexpected_model", "unexpected_stop_reason", "usage_mismatch"})                              # withheld replies
BUDGET_STOPS = frozenset({"timeout", "token_budget", "cost_budget"})
STOPPING_REASONS = BUDGET_STOPS | frozenset({"finished", "model_refusal", "model_unavailable", "rejected_actions",
                                             "usage_unavailable", "repeated_actions", "max_model_turns",
                                             "max_tool_calls"})
RUN_STATUSES = frozenset({"verified", "incomplete", "refused", "failed", "over_target"})
STAGE_ERRORS = frozenset({"RunRefused", "LiveRefused", "ValueError", "RuntimeError", "OSError", "FileNotFoundError",
                          "PermissionError", "TimeoutError", "KeyError", "TypeError", "RunUnavailable",
                          "ReportVerificationError", "AmbientEnvironmentError"})


def _tool(name):
    """A known Contract v1 tool identifier, else the fixed label (a model-supplied name is never shown)."""
    return name if isinstance(name, str) and name in KNOWN_TOOLS else UNKNOWN_TOOL


def _allowed(value, allowed, fallback):
    """value when it is one of the field's known values, else the field's fixed fallback."""
    return value if isinstance(value, str) and value in allowed else fallback


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def progress_event(entry):
    """What a live view may show of one trace entry, every field from an explicit allowlist: its kind, step, elapsed
    time, KNOWN tool names (others: "unknown tool"), whether a call was accepted or refused and the refusal's known
    kind, the loop's known stopping reasons, the provider's known stop reasons, known model-error reasons and token
    counts. Never the model's text, a model-supplied name, a tool's arguments, results or scientific status
    (ok / insufficient_evidence), a refusal's message or a finish's content: results and the finding are shown only
    from the verified saved report. The agent's trace and the saved report are not changed."""
    kind = entry.get("kind")
    out = {"type": "trace", "kind": _allowed(kind, TRACE_KINDS, "other"), "step": _count(entry.get("step")),
           "elapsed_s": entry.get("elapsed_s") if isinstance(entry.get("elapsed_s"), (int, float)) and
           not isinstance(entry.get("elapsed_s"), bool) else None}
    if kind == "model_turn":
        if entry.get("outcome") is not None:
            out["outcome"] = _allowed(entry.get("outcome"), TURN_OUTCOMES, "not_used")
        stop = entry.get("stop_reason")
        out["stop_reason"] = None if stop is None else _allowed(stop, STOP_REASONS, "other")
        out["input_tokens"] = _count(entry.get("input_tokens"))
        out["output_tokens"] = _count(entry.get("output_tokens"))
        out["tool_names"] = [_tool(c.get("name") if isinstance(c, dict) else None)
                             for c in entry.get("tool_calls") or []][:12]
    elif kind == "model_error":
        out["reason"] = _allowed(entry.get("reason"), MODEL_ERROR_REASONS, "provider_error")
    elif kind in ("tool_call", "finish"):
        out["tool"] = _tool(entry.get("tool"))
        refused = entry.get("outcome") == "refused"
        out["outcome"] = "refused" if refused else "accepted"
        if refused:
            out["refused"] = _allowed(entry.get("refused"), REFUSAL_KINDS, "refused")
    elif kind == "budget_exceeded":
        out["stopping_reason"] = _allowed(entry.get("stopping_reason"), BUDGET_STOPS, "budget")
    elif kind == "stop":
        out["stopping_reason"] = _allowed(entry.get("stopping_reason"), STOPPING_REASONS, "stopped")
    return out


class _Progress:
    """A progress sink that can never affect the run: its first exception drops it."""

    def __init__(self, sink):
        self.sink = sink

    def __call__(self, event):
        if self.sink is None:
            return
        try:
            self.sink(event)
        except Exception:
            self.sink = None


def run(args, *, env=None, http_get=None, http_client=None, emulator=None, progress=None):
    """One investigation (see the module docstring). env, http_get and http_client are injected by tests; a live
    run without them uses the process environment, Stage 8's metadata GETs and the SDK's own HTTP client.
    progress: an optional, presentation-only callback (see progress_event); it never changes the run."""
    env = os.environ if env is None else env
    progress = _Progress(progress)
    plan, policy = make_plan(args), make_policy(args)
    if args.live:                                  # best-effort mode alone never authorizes a paid run
        lv1.authorize_live(plan, args.confirm_budget_usd, args.authorize_paid_run)  # before any request or file
    live.require_clean_environment(env)
    archive, cache, manifest = Path(args.archive).expanduser(), Path(args.cache_dir), Path(args.manifest)
    if not archive.is_file():
        raise iv.RunRefused(f"score archive {archive} not found")
    if not manifest.is_file():
        raise iv.RunRefused(f"split manifest {manifest} not found")
    if not (cache / "extraction_index.json").is_file():
        raise iv.RunRefused(f"no verified validation pose cache at {cache} (no synthetic substitute)")
    out = iv.prepare(args.out)
    args.archive = archive
    key = _secrets(env)
    summary = {"mode": LIVE_MODE if args.live else DRY_MODE,
               "steps": [], "plan": asdict(plan), "key_policy": asdict(policy), "publication": av1.PUBLICATION,
               "model": {"provider": plan.provider, "model": plan.model_id, "language_model": bool(args.live),
                         "adapter": "investigation_agent.ClaudeInvestigationModel via investigation_live "
                                    "GuardedClient / ReportedCostGuard and investigation_live_v1.GuardedModelV1",
                         **({} if args.live else {"label": EMULATED_LABEL})},
               "test_split": "not read: validation member and validation cache only"}
    status, ledger, key_limit, key_remaining = "verified", None, None, None

    def step(name, fn):                            # vr._step, with the stage reported as it starts and ends
        progress({"type": "stage", "stage": name, "state": "started"})
        try:
            out = vr._step(summary, name, fn)
        except Exception as exc:
            progress({"type": "stage", "stage": name, "state": "failed",
                      "error": _allowed(type(exc).__name__, STAGE_ERRORS, "error")})
            raise
        progress({"type": "stage", "stage": name, "state": "completed"})
        return out

    def observe(entry):                            # the agent's trace entries, as recorded
        if entry.get("kind") == "model_request":
            progress({"type": "model_request"})
            return
        progress(progress_event(entry))
        if entry.get("kind") in ("model_turn", "model_error") and ledger is not None:
            progress({"type": "spending", "basis": "provider-reported" if args.live else "simulated",
                      "known_cost_usd": ledger.known_cost_usd, "unknown_cost_attempts": ledger.unknown_cost_attempts,
                      "requests": len(ledger.entries)})
    try:
        data, scores, pose = step("inputs", lambda: iv.load_data(args, manifest, cache))
        synthetic = scores["synthetic"]
        summary["inputs"] = {"scores": {k: v for k, v in scores.items() if k != "frames_per_session"},
                             "pose_cache": pose, "data_source": data.data_source, "derived_from": data.derived_from,
                             "data_fingerprint": data.fingerprint, "sessions": list(data.sessions)}
        recorded = None
        if args.validation_run:
            recorded = step("validation_run", lambda: iv.check_validation_run(
                args.validation_run, args.runs_dir, manifest, data, scores, synthetic))
        if args.live:
            pre = step("preflight", lambda: lv1.preflight_v1(plan, manifest=manifest, env=env,
                                                             http_get=http_get, policy=policy))
            summary["preflight"], key_limit = pre, pre["provider_key_limit_usd"]
            key_remaining = pre["provider_key_remaining_usd"]
            # the adapter sees the capped inference key alone (never the management key)
            key_env = live.PROVIDERS[plan.provider].key_env
            prices, model_env = live.Prices(**pre["prices"]), {key_env: env.get(key_env, "")}
            compat = {"output_config": pre["compatibility"]["keep_output_config"],
                      "cache_control": pre["compatibility"]["keep_cache_control"]}
        else:
            import httpx2
            prices = live.snapshot_prices(plan.model_id)
            if prices is None:
                raise iv.RunRefused(f"no offline price snapshot for {plan.model_id}")
            emulator = emulator or EmulatedOpenRouterV1()
            http_client = httpx2.Client(transport=httpx2.MockTransport(emulator))
            model_env, compat = {live.PROVIDERS[plan.provider].key_env: "dry-run-placeholder-not-a-key"}, None
        budget, estimate = step("budget", lambda: lv1.budget_v1(plan, prices))
        summary["budget"], summary["estimate"] = asdict(budget), estimate
        model, ledger = lv1.build_model_v1(plan, estimate, env=model_env, http_client=http_client, compat=compat)
        report = step("investigate", lambda: av1.investigate_v1(
            plan.question, data, model, budget=budget, investigation_id=INVESTIGATION_ID, allow_synthetic=synthetic,
            observer=observe if progress.sink is not None else None))
        saved = iv._json_bytes(report.to_json())
        _write(out / OUTPUT_NAMES[0], saved, key)
        progress({"type": "saved", "file": OUTPUT_NAMES[0]})
        doc = json.loads((out / OUTPUT_NAMES[0]).read_bytes())           # what was saved, not what is in memory
        if doc != json.loads(saved):
            raise RuntimeError("the saved report is not the report that was checked")
        summary["report"] = step("verify", lambda: iv.check_report(doc, data, synthetic))
        _write(out / OUTPUT_NAMES[1], av1.render_markdown_v1(doc).encode("utf-8"), key)
        progress({"type": "saved", "file": OUTPUT_NAMES[1]})

        def check_ledger():
            problems = lv1.ledger_problems(doc, ledger.to_json(), plan)
            if problems:
                raise RuntimeError(f"the cost ledger does not match the report: {problems[:3]}")
        step("cost_ledger", check_ledger)
        summary["investigation"] = {
            "finished": doc["finished"], "stopping_reason": doc["stopping_reason"], "finding": doc["finding"],
            "primary_evidence_id": doc["primary_evidence_id"], "observations": doc["observations"],
            "fingerprint": doc["fingerprint"], "publication": doc["publication"], "usage": doc["usage"],
            "tool_sequence": [(t["tool"], t.get("refused") or t["outcome"]) for t in doc["trace"]
                              if t["kind"] in ("tool_call", "finish")]}
        if recorded is not None:
            summary["report"]["validation_run"] = step(
                "compatibility", lambda: iv.check_query_against_run(doc, recorded))
        if not doc["finished"]:
            status = "incomplete"
    except REFUSALS:
        status = "refused"
    except Exception:
        status = "failed"
    if ledger is not None:
        summary["cost"] = lv1.cost_summary(ledger.to_json(), plan, key_limit, policy, key_remaining)
        if summary["cost"]["over_target"] and status != "refused":
            status = "over_target"
    if emulator is not None:
        summary["emulated_requests"] = [{k: v for k, v in r.items() if k != "body"} for r in emulator.requests]
    summary["status"] = status
    summary["outputs"] = {n: iv._sha((out / n).read_bytes()) for n in OUTPUT_NAMES[:2] if (out / n).is_file()}
    _write(out / OUTPUT_NAMES[2], iv._json_bytes(summary), key)
    progress({"type": "saved", "file": OUTPUT_NAMES[2]})
    stopped = (summary.get("investigation") or {}).get("stopping_reason")
    progress({"type": "finished", "status": _allowed(status, RUN_STATUSES, "unknown"),
              "stopping_reason": None if stopped is None else _allowed(stopped, STOPPING_REASONS, "stopped")})
    summary["output_dir"] = str(out)
    return summary


def _execution_evidence(s, doc, plan, policy):
    """(problems, prices, provider limit, remaining credit) from the run's explicitly recorded execution mode. A live
    run needs its complete preflight record, re-checked by investigation_live_v1.saved_preflight_problems (the key
    policy, acknowledgment and provider limit bound to the key metadata through the run-time validator); a dry run
    must be labelled emulated, hold no preflight and record one emulated request per ledger attempt (its prices are
    the offline snapshot). Anything else, or a mix, is a problem."""
    model, n_attempts = s["model"], len(s["cost"]["ledger"]["requests"])
    if s["mode"] == LIVE_MODE:
        problems = []
        if model.get("language_model") is not True or "label" in model or "emulated_requests" in s:
            problems.append("a live run is recorded with emulated-run fields")
        p, prices = lv1.saved_preflight_problems(s.get("preflight"), plan, policy)
        pre = s.get("preflight") or {}
        if isinstance(pre, dict) and pre.get("estimate_v1") != s.get("estimate"):
            p.append("the run's estimate is not the preflight's")
        return problems + p, prices, pre.get("provider_key_limit_usd"), pre.get("provider_key_remaining_usd")
    if s["mode"] == DRY_MODE:
        problems = []
        reqs = s.get("emulated_requests")
        if model.get("language_model") is not False or model.get("label") != EMULATED_LABEL or "preflight" in s or \
                not isinstance(reqs, list) or len(reqs) != n_attempts:
            problems.append("a dry run must be labelled emulated, hold no preflight and record every emulated request")
        ids = [c.get("id") for t in doc["trace"] if t.get("kind") == "model_turn" for c in t.get("tool_calls") or []]
        if not all(isinstance(i, str) and i.startswith("toolu_emu_") for i in ids):
            problems.append("a dry run's trace holds tool calls the emulator did not issue")
        return problems, live.snapshot_prices(plan.model_id), None, None
    return ["the run records no known execution mode"], None, None, None


def verify_saved(out):
    """Offline re-verification of a finished output directory: the report (verify_report_v1), its Markdown
    (re-rendered, byte for byte), the cost ledger against the trace and plan, the budget and the files' SHA-256.
    Returns the problems (empty: consistent; this is consistency, not authenticity)."""
    out = Path(out)
    try:
        s = json.loads((out / OUTPUT_NAMES[2]).read_text(encoding="utf-8"))
        doc = json.loads((out / OUTPUT_NAMES[0]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"the outputs cannot be read ({type(exc).__name__})"]
    problems = list(av1.verify_report_v1(doc))
    try:
        if not problems and (out / OUTPUT_NAMES[1]).read_text(encoding="utf-8") != av1.render_markdown_v1(doc):
            problems.append("the Markdown is not the rendering of the report")
        for name, digest in s["outputs"].items():
            if iv._sha((out / name).read_bytes()) != digest:
                problems.append(f"{name} is not the file run.json recorded")
        if s["publication"] != av1.PUBLICATION or doc["publication"] != av1.PUBLICATION:
            problems.append("publication is not 'not_authorized'")
        if json.loads(json.dumps(s["budget"])) != doc["budget"]:
            problems.append("the report's budget is not the run's")
        plan, policy = live.LivePlan(**s["plan"]), lv1.KeyPolicy(**s["key_policy"])
        problems += lv1.ledger_problems(doc, s["cost"]["ledger"], plan)
        p, prices, limit, remaining = _execution_evidence(s, doc, plan, policy)
        problems += p
        if prices is not None:                       # the budget the run was bound by, re-derived from that evidence
            budget, estimate = lv1.budget_v1(plan, prices)
            if json.loads(json.dumps(asdict(budget))) != doc["budget"] or \
                    json.loads(json.dumps(estimate)) != json.loads(json.dumps(s["estimate"])):
                problems.append("the budget is not the one the recorded prices give")
        cost = lv1.cost_summary(s["cost"]["ledger"], plan, limit, policy, remaining)
        if {k: s["cost"].get(k) for k in cost} != cost:
            problems.append("the recorded cost summary is not the ledger's")
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        problems.append(f"the outputs are malformed ({type(exc).__name__})")
    return problems


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", nargs="?", help="the validation score-drop ZIP (read in place, never extracted)")
    ap.add_argument("--archive-sha256", help="the archive's expected SHA-256 (checked before reading)")
    ap.add_argument("--out", help="a new, git-ignored (or outside the repository) directory")
    ap.add_argument("--question", default=iv.DEFAULT_QUESTION)
    ap.add_argument("--validation-run", default=None, help="a recorded Phase 2 run to check compatibility against")
    ap.add_argument("--runs-dir", default=None, help="where recorded runs live (default: validation_run_view's)")
    ap.add_argument("--member", default="scores_validation_model_b.csv")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "raw" / "calms21_validation"))
    ap.add_argument("--manifest", default=str(ROOT / "splits" / "split_manifest.json"))
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="tests only: accept SYNTHETIC fixture sessions (labelled synthetic, never real evidence)")
    ap.add_argument("--model", default=None, help="OpenRouter model id (default: Stage 8's default)")
    ap.add_argument("--target-usd", type=float, default=lv1.MAX_TARGET_USD)
    ap.add_argument("--max-model-turns", type=int, default=8)
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--key-limit-mode", choices=("strict", "best-effort"), default="strict",
                    help="strict (default): the key's limit must be at most $0.50. best-effort: an existing "
                         "non-resetting key limited to at most $50; the $0.50 target is then best-effort only and "
                         "spending can exceed it, up to the key's available credit")
    ap.add_argument("--acknowledge-key-limit-usd", type=float, default=None,
                    help="with --key-limit-mode best-effort: the key's actual limit (must equal its metadata)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="PAID: one bounded investigation (needs both below)")
    mode.add_argument("--verify-saved", default=None, metavar="DIR", help="re-verify a finished output directory")
    mode.add_argument("--preflight-only", action="store_true",
                      help="read-only readiness checks (free metadata GETs; no model request, no file)")
    ap.add_argument("--confirm-budget-usd", type=float, default=None, help="with --live; must equal --target-usd")
    ap.add_argument("--authorize-paid-run", default=None, help=f'with --live: exactly "{lv1.PAID_RUN_AUTHORIZATION}"')
    return ap


def preflight_only(args, *, env=None, http_get=None):
    """(exit code, report text): the read-only preflight of the plan these arguments describe."""
    env = os.environ if env is None else env
    report = lv1.readiness_v1(make_plan(args), manifest=args.manifest, env=env, http_get=http_get,
                              policy=make_policy(args))
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if any(k in text for k in _secrets(env)):              # never print a credential
        raise RuntimeError("refusing to print the preflight: it contains an API key")
    return (0 if report["ready"] else 2), text


def main(argv=None, **inject):
    ap = parser()
    args = ap.parse_args(argv)
    if args.preflight_only:
        try:
            code, text = preflight_only(args, env=inject.get("env"), http_get=inject.get("http_get"))
        except ValueError as exc:
            print(f"refused ({type(exc).__name__}): {exc}", file=sys.stderr)
            return 2
        print(text)
        return code
    if args.verify_saved:
        problems = verify_saved(args.verify_saved)
        print(json.dumps({"verified": not problems, "problems": problems}, indent=2))
        return 0 if not problems else 1
    if not (args.archive and args.archive_sha256 and args.out):
        ap.error("ARCHIVE, --archive-sha256 and --out are required")
    try:
        summary = run(args, **inject)
    except (*REFUSALS, ValueError) as exc:         # messages name variables and limits, never values of keys
        print(f"refused ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary.get(k) for k in ("status", "steps", "output_dir")}, indent=2, default=str))
    return {"verified": 0, "refused": 2}.get(summary["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
