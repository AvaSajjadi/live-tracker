"""EXPERIMENTAL (Investigation mode, Stage 6): a bounded, autonomous LLM investigation loop (new file).

A model plans, calls the four scientific tools, observes their validated
results and decides what to do next, until it submits a final answer or a
budget stops it:

    report = investigate("Does approach tend to precede attack?", tools, policy, model)

Separate from the Day 2 rule parser (agent.py / claude_agent.py): nothing is
shared or imported from there.

Plan - act - observe
--------------------
Each model turn may contain text (its reasoning, recorded in the trace) and
tool calls. Every tool call is checked here, then executed through
investigation_contracts.call_tool, so the Stage 2 request / result
validators and the EvidencePolicy always run; the model never reaches a
tool implementation directly. The validated result (or the reason the call
was refused) goes back to the model as a tool result, and the model chooses
its next action from it. Nothing here fixes a tool sequence.

What the model controls, and what it does not
---------------------------------------------
The model chooses the tools, behaviors, sessions, episodes, windows,
features, window lengths, gaps and baseline settings. It does NOT choose
the evidence policy, the comparison's sufficiency criteria (the loop's
`sufficiency`, default PROVISIONAL_SUFFICIENCY) or its uncertainty method
("none": no session-level uncertainty method is approved, Stage 5). Those
fields are not in its tool schemas; sending them is an argument error. Long
ID lists can be passed by reference (episodes_from / windows_from: the
IDs of an earlier validated result of this run), so the model never has to
copy or invent IDs.

No fabricated evidence
----------------------
The run ends normally only through finish_investigation, whose answer is
verified before it is accepted (otherwise the model gets the problems back
and may revise):
- its arguments match the finish_investigation schema (types checked, a
  null is not an omitted field);
- every cited evidence_id is a validated result of THIS run;
- "effect_described" needs a compare_windows result with status "ok", and
  its direction must equal the sign of that result's effect;
  "insufficient_evidence" / "inconclusive" carry no direction;
- observations are references, not prose: {evidence_id, field} names a
  field of a cited result (e.g. measurements.effect.value). The loop reads
  the value from the stored validated result and writes the statement
  itself, so a measurement can only be reported as the value it has (a null
  effect stays null);
- interpretations, additional_limitations and next_steps are the model's
  prose, never evidence. They may not contain numbers or number words
  (quantities go through observations), significance, p-value,
  confidence-interval, proof, causal or prediction language, and, unless the
  finding is effect_described, no description of an effect (increase,
  decrease, faster, ...).
The report copies the effect and sufficiency from the stored result (not
from the model) and keeps every limitation the tools returned, verbatim.

Bounds (Budget; every stop is recorded with its reason)
--------------------------------------------------------
model turns, executed tool calls (when they are spent the model gets one
more turn, in which only finish_investigation is accepted), input / output
tokens, cost (only when prices are given: a cost limit without prices is
refused), wall-clock time,
repeated identical requests (refused, not re-executed) and consecutive
rejected actions (malformed replies, unknown tools, invalid or unauthorized
arguments, rejected final answers). Every reply's reported usage is
recorded as soon as it arrives, before its content is validated, so a
malformed reply is not a free call; a reply whose usage cannot be read
(missing, non-integer or negative counts) stops the run ("usage_unavailable")
when a token or cost limit is set, since its consumption is unknown. Token,
cost and time limits are checked before each model request, again as soon
as its reply arrives (a reply that overran a limit executes nothing, not
even a final answer, and a malformed one is not retried) and before each
tool action. Each request is given the remaining output-token allowance
(from the token and cost limits) and the remaining time (RequestLimits),
when the model accepts them. A synchronous request cannot be interrupted
from here and its input size is only known afterwards, so one request can
overrun the input-token or cost limit, or (within the adapter's per-request
timeout) the deadline; the overrun is recorded and stops the run.

A stop before a final answer still returns a report: finding "incomplete",
with the evidence gathered so far. Every report carries each validated
result whole, with its request (deep copies; bounded by max_tool_calls and
the contract's size limits, never truncated), and
InvestigationReport.verify_evidence() re-derives each evidence_id from its
snapshot, so an altered snapshot is detectable.

Models (dependency injection)
-----------------------------
Any object with respond(system, tools, messages) -> ModelReply; a respond
that also takes `limits` receives the RequestLimits of the request. Messages use
the Anthropic Messages layout (dict blocks), and are append-only: the
assistant content is sent back exactly as received. ClaudeInvestigationModel
is the real adapter (Anthropic SDK, optional extra
`pip install -e ".[claude]"`); tests inject deterministic fake models.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import investigation_contracts as ic

__all__ = ["AGENT_LIMITATIONS", "Budget", "ClaudeInvestigationModel", "FINDINGS", "InvestigationModel",
           "InvestigationReport", "ModelReply", "ModelUnavailable", "RequestLimits", "STOP_REASONS",
           "SYSTEM_PROMPT", "TOOL_SPECS", "investigate"]

FINISH = "finish_investigation"
SCIENCE_TOOLS = ic.TOOLS
FINDINGS = ("effect_described", "insufficient_evidence", "inconclusive")
STOP_REASONS = ("finished", "max_model_turns", "max_tool_calls", "token_budget", "cost_budget", "timeout",
                "usage_unavailable", "repeated_actions", "rejected_actions", "model_unavailable", "model_refusal")
MAX_TEXT_TRACE = 2000
MAX_STATEMENT = 600
MAX_ITEMS = 20
MAX_FIELD = 200
MAX_LIST_VALUE = 20


# ---- models ----

@dataclass
class ModelReply:
    """One model turn. content: Anthropic-layout blocks as dicts ({"type":
    "text", "text"}, {"type": "tool_use", "id", "name", "input"}; any other
    block, e.g. thinking, is kept and sent back unchanged)."""
    content: Any
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None


class ModelUnavailable(RuntimeError):
    """The model could not answer (no SDK / key, timeout, API error)."""

    def __init__(self, reason, message=""):
        super().__init__(message or reason)
        self.reason = reason


@dataclass(frozen=True)
class RequestLimits:
    """What is left of the budget for one model request."""
    max_output_tokens: int | None       # None: no token or cost limit applies
    timeout_s: float


class InvestigationModel(Protocol):
    def respond(self, system: str, tools: list, messages: list) -> ModelReply: ...


@dataclass(frozen=True)
class Budget:
    max_model_turns: int = 12
    max_tool_calls: int = 16
    max_input_tokens: int | None = 600_000
    max_output_tokens: int | None = 80_000
    max_cost_usd: float | None = None
    price_per_mtok: tuple | None = None        # (input USD, output USD) per million tokens
    timeout_s: float = 900.0
    max_repeated_calls: int = 2
    max_consecutive_rejections: int = 3
    max_result_chars: int = 16_000

    def __post_init__(self):
        if self.max_cost_usd is not None and self.price_per_mtok is None:
            raise ValueError("a cost limit needs price_per_mtok; cost cannot be tracked without prices")
        for name in ("max_model_turns", "max_tool_calls", "max_repeated_calls", "max_consecutive_rejections"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_result_chars < 2000:
            raise ValueError("max_result_chars must be >= 2000 (room for a result summary)")


# ---- tool schemas the model sees ----

_FROM = {"type": "object", "properties": {"evidence_id": {"type": "string"}}, "required": ["evidence_id"],
         "additionalProperties": False}
_WINDOWS_FROM = {"type": "object", "additionalProperties": False, "required": ["evidence_id"], "properties": {
    "evidence_id": {"type": "string", "description": "a get_window_features result of this run"},
    "status": {"enum": ["included", "excluded", "all"], "description": "default included"},
    "kind": {"enum": ["pre_event", "baseline", "all"], "description": "default all"}}}
_IDS = {"type": "array", "items": {"type": "string"}}
_FRAMES = {"type": "integer", "minimum": 0}

TOOL_SPECS = [
    {"name": "query_events",
     "description": "Behavioral episodes (maximal annotated runs) from an event source. Only "
                    "'validation_annotation' (human annotations of validation sessions) is available.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["source", "behavior"],
                      "properties": {
                          "source": {"enum": list(ic.SOURCES)},
                          "behavior": {"enum": list(ic.BEHAVIORS)},
                          "sessions": {"type": ["array", "null"], "items": {"type": "string"},
                                       "description": "null: every session the source allows"},
                          "max_episodes": {"type": "integer", "minimum": 1, "maximum": ic.MAX_EPISODES},
                          "merge_gap_frames": {"type": ["integer", "null"], "minimum": 0},
                          "min_duration_frames": {"type": ["integer", "null"], "minimum": 1}}}},
    {"name": "get_window_features",
     "description": "Causal feature summaries in pre-event windows [onset-gap-length, onset-gap) and "
                    "same-session baseline tiles, with exclusion reasons. Give episode_ids or "
                    "episodes_from (all episodes of a query_events result of this run).",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["pre_event", "features"],
                      "properties": {
                          "episode_ids": _IDS, "episodes_from": _FROM,
                          "pre_event": {"type": "object", "additionalProperties": False,
                                        "required": ["length_frames", "gap_frames"],
                                        "properties": {"length_frames": _FRAMES, "gap_frames": _FRAMES}},
                          "baseline": {"type": ["object", "null"], "additionalProperties": False,
                                       "properties": {"strategy": {"enum": ["same_session_tiles"]},
                                                      "length_frames": _FRAMES, "clearance_frames": _FRAMES,
                                                      "max_windows_per_session": _FRAMES}},
                          "features": {"type": "array", "items": {"enum": list(ic.FEATURES)}}}}},
    {"name": "check_tracking_quality",
     "description": "Per-window tracking quality: tracking_ok and derivative-endpoint counts, usable "
                    "frames by history, smoothing contamination, verdict. Never verifies motion accuracy. "
                    "Give window_ids or windows_from.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["history_frames"],
                      "properties": {"window_ids": _IDS, "windows_from": _WINDOWS_FROM,
                                     "history_frames": {"type": "array", "items": _FRAMES}}}},
    {"name": "compare_windows",
     "description": "Session-level paired comparison: per session median(pre-event window values) - "
                    "median(baseline values); effect = median of session differences, descriptive "
                    "only (no interval). Sufficiency criteria are fixed by the investigation, not by "
                    "you. List pre_window_ids / baseline_window_ids; windows_from fills any group "
                    "not listed.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["feature"],
                      "properties": {"feature": {"enum": list(ic.FEATURES)},
                                     "pre_window_ids": _IDS, "baseline_window_ids": _IDS,
                                     "windows_from": _WINDOWS_FROM}}},
    {"name": FINISH,
     "description": "Submit the final answer. It is checked against the validated results of this run "
                    "and refused (with the problems) if it is not supported.",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["finding", "observations"],
                      "properties": {
                          "finding": {"enum": list(FINDINGS)},
                          "primary_evidence_id": {"type": ["string", "null"]},
                          "direction": {"enum": ["negative", "positive", "zero", None]},
                          "observations": {"type": "array", "description": (
                              "references to result fields; the investigation reads each value itself"),
                              "items": {
                              "type": "object", "additionalProperties": False,
                              "required": ["evidence_id", "field"],
                              "properties": {"evidence_id": {"type": "string"},
                                             "field": {"type": "string", "description": (
                                                 "path in that result, e.g. status, measurements.effect.value, "
                                                 "measurements.per_session[0].difference")}}}},
                          "interpretations": {"type": "array", "items": {
                              "type": "object", "additionalProperties": False,
                              "required": ["statement", "evidence_ids"],
                              "properties": {"statement": {"type": "string"}, "evidence_ids": _IDS}}},
                          "additional_limitations": {"type": "array", "items": {"type": "string"}},
                          "next_steps": {"type": "array", "items": {"type": "string"}}}}},
]
TOOL_NAMES = tuple(t["name"] for t in TOOL_SPECS)
assert set(TOOL_NAMES) == set(SCIENCE_TOOLS) | {FINISH}

SYSTEM_PROMPT = """\
You are the investigator in Live Tracker's experimental Investigation mode. You answer a research \
question about mouse social behavior in the CalMS21 Task 1 validation sessions, using only the tools \
provided. Each tool result is a validated measurement with an evidence_id, its methodology and its \
limitations.

Work iteratively: decide what to measure, call a tool, read the result, and choose the next step from \
what it actually shows. When evidence is insufficient, excluded or affected by tracking quality, \
investigate why (other sessions, window choices, tracking quality, sensitivity checks) before you \
conclude. Stop when further calls would not change the answer.

Rules:
- Facts come only from tool results. Never invent measurements, IDs or evidence_ids.
- In finish_investigation, observations are references {evidence_id, field} to fields of results \
(e.g. measurements.effect.value, measurements.n_sessions_paired, status); the investigation reads the \
values itself. Interpretations, additional_limitations and next_steps are your prose and are not \
evidence: write no numbers or number words in them (cite quantities as observations), and unless the \
finding is effect_described do not describe an effect (increase, decrease, faster, ...).
- The comparison is descriptive: there is no confidence interval, p-value or significance test. Do \
not use the words significant, p-value, confidence interval, statistically, proves, causes or \
predicts. An association before attack is not a cause.
- Use finding "effect_described" only with a compare_windows result whose status is "ok"; otherwise \
"insufficient_evidence" or "inconclusive".
- Keep the tools' limitations in mind (investigation confound, session timing, missing tracking, \
pixel scale, unverified motion accuracy); add limitations you notice.
- Your budget of turns and tool calls is limited; repeated identical requests are refused."""

AGENT_LIMITATIONS = [
    "Investigation planned by an LLM: the tools, windows and follow-ups were chosen by the model, and a "
    "different plan could select different evidence. The trace records every choice.",
    "Observations are values the investigation read from the cited fields of validated tool results; "
    "the model chose which fields to cite. Interpretations, next steps and model-added limitations are "
    "the model's prose, checked only lexically (no numbers, no significance, causal or prediction "
    "language, no effect language without an 'ok' comparison), and are not evidence.",
    "The comparison's sufficiency criteria and uncertainty method were fixed by the investigation, not "
    "by the model; no confidence interval or significance test is available.",
]

_BANNED = [(re.compile(p, re.I), w) for p, w in (
    (r"\bsignifican", "significance"), (r"\bp\s*-?\s*values?\b", "p-value"), (r"\bp\s*[<=>]\s*0?\.\d", "p-value"),
    (r"\bconfidence\s+intervals?\b", "confidence interval"), (r"\bstatistical(ly)?\b", "statistical claim"),
    (r"\bprov(e|es|ed|en|ing)\b", "proof"), (r"\bcaus(e|es|ed|ing|al|ality|ation)\b", "causal claim"),
    (r"\bpredict", "prediction"))]
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\.?\d+(?:[,.]\d+)*(?:e[-+]?\d+)?", re.I)
_NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundreds?|"
    r"thousands?|millions?|billions?|dozens?|half|halves|twice|thrice|double[ds]?|triple[ds]?|quadruple[ds]?|"
    r"percent|percentage|fold)\b|%", re.I)
_EFFECT_WORDS = re.compile(
    r"\b(increas|decreas|higher|lower|faster|slower|greater|smaller|larger|reduc|rise|rises|rose|risen|"
    r"drop|fell|fall|declin|elevat|diminish|shrink|shrank|grew|grow|closer|farther|further apart)\w*", re.I)
_FIELD = re.compile(r"(status|errors|measurements)(\.[A-Za-z_][A-Za-z0-9_]*|\[\d{1,4}\])*")


# ---- the loop ----

class _Rejected(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class InvestigationReport:
    question: str
    stopping_reason: str
    finished: bool
    finding: str                    # FINDINGS, or "incomplete"
    direction: str | None
    primary_evidence: dict | None
    observations: list              # [{kind, evidence_id, tool, field, value, statement}]: read by the loop
    interpretations: list           # [{kind, statement, evidence_ids}]: the model's prose
    evidence: list                  # [{evidence_id, tool, status, request, result}]: every validated result
    limitations: list               # [{source, text}]: tool limitations verbatim, then agent / model ones
    next_steps: list
    usage: dict
    trace: list = field(default_factory=list)

    def to_json(self):
        return copy.deepcopy(self.__dict__)

    def verify_evidence(self):
        """evidence_ids whose request / result snapshot no longer hashes to its
        evidence_id (empty: every snapshot is as validated)."""
        bad = []
        for e in self.evidence:
            try:
                ok = (e["result"]["evidence_id"] == e["evidence_id"]
                      == ic.evidence_id(e["tool"], e["request"], e["result"]))
            except Exception:
                ok = False
            if not ok:
                bad.append(e.get("evidence_id"))
        return bad


class _Run:
    def __init__(self, question, tools, policy, model, budget, sufficiency, clock):
        if not isinstance(tools, ic.InvestigationTools):
            raise TypeError("tools must implement investigation_contracts.InvestigationTools")
        ic._require_policy(policy)
        if not isinstance(question, str) or not question.strip() or len(question) > ic.MAX_TEXT:
            raise ValueError("question must be non-empty text of at most MAX_TEXT characters")
        self.question, self.tools, self.policy, self.model = question, tools, policy, model
        self.budget, self.clock = budget, clock
        self.sufficiency = sufficiency.to_json()
        ic.SUFFICIENCY(self.sufficiency, "sufficiency")
        self.messages = [{"role": "user", "content": f"Research question: {question}"}]
        self.evidence = {}          # evidence_id -> {"tool", "request", "result"}
        self.executed = {}          # canonical request -> evidence_id
        self.trace = []
        self.turns = self.tool_calls = self.repeats = self.rejected_in_a_row = 0
        self.tools_spent_at_turn = None      # the turn in which the tool-call budget ran out
        self.tokens_in = self.tokens_out = 0
        self.overrun = None             # a limit the last model request went past
        self.unreported = 0             # replies without valid usage (only possible with no token / cost limit)
        self.pass_limits = _accepts_limits(model)
        self.start = clock()

    # -- bounds --

    def cost(self):
        if self.budget.price_per_mtok is None:
            return None
        p_in, p_out = self.budget.price_per_mtok
        return (self.tokens_in * p_in + self.tokens_out * p_out) / 1e6

    def accounts_usage(self):
        b = self.budget
        return b.max_input_tokens is not None or b.max_output_tokens is not None or b.max_cost_usd is not None

    def exceeded(self):
        """A token, cost or time limit already gone past (checked when a reply
        arrives and before each tool action: nothing more is executed)."""
        b = self.budget
        if self.clock() - self.start > b.timeout_s:
            return "timeout"
        if (b.max_input_tokens is not None and self.tokens_in > b.max_input_tokens) or \
                (b.max_output_tokens is not None and self.tokens_out > b.max_output_tokens):
            return "token_budget"
        if b.max_cost_usd is not None and self.cost() > b.max_cost_usd:
            return "cost_budget"
        return None

    def request_limits(self):
        """The remaining output-token allowance and time for the next request."""
        b = self.budget
        out = None if b.max_output_tokens is None else b.max_output_tokens - self.tokens_out
        if b.max_cost_usd is not None and b.price_per_mtok[1] > 0:
            by_cost = int((b.max_cost_usd - self.cost()) * 1e6 / b.price_per_mtok[1])
            out = by_cost if out is None else min(out, by_cost)
        return RequestLimits(max_output_tokens=None if out is None else max(1, out),
                             timeout_s=max(0.001, b.timeout_s - (self.clock() - self.start)))

    def exhausted(self):
        b = self.budget
        if self.clock() - self.start > b.timeout_s:
            return "timeout"
        if self.turns >= b.max_model_turns:
            return "max_model_turns"
        if (b.max_input_tokens is not None and self.tokens_in >= b.max_input_tokens) or \
                (b.max_output_tokens is not None and self.tokens_out >= b.max_output_tokens):
            return "token_budget"
        if b.max_cost_usd is not None and self.cost() >= b.max_cost_usd:
            return "cost_budget"
        if self.tools_spent_at_turn is not None and self.turns > self.tools_spent_at_turn:
            return "max_tool_calls"      # its one last turn was used without an accepted answer
        return None

    def log(self, kind, **kw):
        self.trace.append({"step": len(self.trace), "kind": kind, "elapsed_s": round(self.clock() - self.start, 3),
                           **kw})

    # -- one run --

    def run(self):
        while True:
            reason = self.exhausted()
            if reason:
                return self.stop(reason)
            try:
                limits = {"limits": self.request_limits()} if self.pass_limits else {}
                reply = self.model.respond(SYSTEM_PROMPT, copy.deepcopy(TOOL_SPECS), copy.deepcopy(self.messages),
                                           **limits)
            except ModelUnavailable as exc:
                self.log("model_error", reason=exc.reason)
                return self.stop("model_unavailable")
            except Exception as exc:                 # a model adapter bug; its text may hold secrets
                self.log("model_error", reason=type(exc).__name__)
                return self.stop("model_unavailable")
            self.turns += 1
            usage_in, usage_out = _usage(reply)       # recorded first: a malformed reply is not free
            self.tokens_in += usage_in or 0
            self.tokens_out += usage_out or 0
            if None in (usage_in, usage_out):
                self.unreported += 1
            if None in (usage_in, usage_out) and self.accounts_usage():
                self.log("model_turn", outcome="usage_unavailable", input_tokens=usage_in, output_tokens=usage_out)
                return self.stop("usage_unavailable")
            over = self.exceeded()
            if over:                                 # the reply arrived past a limit: execute / retry none of it
                self.overrun = over
                self.log("model_turn", outcome="over_budget", input_tokens=usage_in, output_tokens=usage_out)
                self.log("budget_exceeded", stopping_reason=over, not_executed=_use_ids(reply))
                return self.stop(over)
            problem = _reply_problem(reply)
            if problem:
                self.log("model_turn", outcome="malformed", problem=problem, input_tokens=usage_in,
                         output_tokens=usage_out)
                if self.reject(problem):
                    return self.stop("rejected_actions")
                self.messages.append({"role": "user", "content": f"Your reply was not usable: {problem}. "
                                      "Reply with tool calls, or call finish_investigation."})
                continue
            uses = [b for b in reply.content if b.get("type") == "tool_use"]
            text = " ".join(b.get("text", "") for b in reply.content if b.get("type") == "text").strip()
            self.log("model_turn", model=reply.model, stop_reason=reply.stop_reason, text=text[:MAX_TEXT_TRACE],
                     tool_calls=[{"id": u.get("id"), "name": u.get("name")} for u in uses],
                     input_tokens=reply.input_tokens, output_tokens=reply.output_tokens)
            if reply.stop_reason == "refusal":
                return self.stop("model_refusal")
            self.messages.append({"role": "assistant", "content": copy.deepcopy(reply.content)})
            if reply.stop_reason == "max_tokens":
                results = [self.error_result(u, "truncated", "your reply hit the output limit; the call "
                                             "may be incomplete and was not executed") for u in uses]
                self.messages.append({"role": "user", "content": results or
                                      "Your reply hit the output limit. Continue with tool calls."})
                if self.reject("truncated reply"):
                    return self.stop("rejected_actions")
                continue
            if not uses:
                if self.reject("no tool call"):
                    return self.stop("rejected_actions")
                self.messages.append({"role": "user", "content": "No tool was called. Call a tool, or call "
                                      "finish_investigation with your answer."})
                continue
            results, final, stop_now = [], None, None
            for use in uses:
                if not stop_now:
                    stop_now = self.exceeded()       # e.g. the deadline passed during the previous tool
                    if stop_now:
                        self.log("budget_exceeded", stopping_reason=stop_now)
                if stop_now:
                    results.append(self.error_result(use, "not_executed", f"stopped: {stop_now}"))
                    continue
                content, is_error, done = self.handle(use)
                results.append({"type": "tool_result", "tool_use_id": use["id"], "content": content,
                                **({"is_error": True} if is_error else {})})
                if done is not None:
                    final = done
                    stop_now = "final answer accepted"
                elif self.repeats >= self.budget.max_repeated_calls:
                    stop_now = "repeated_actions"
                elif self.rejected_in_a_row >= self.budget.max_consecutive_rejections:
                    stop_now = "rejected_actions"
            if self.tool_calls >= self.budget.max_tool_calls and self.tools_spent_at_turn is None:
                self.tools_spent_at_turn = self.turns
                results.append({"type": "text", "text": "The tool-call budget is spent. Your next reply must "
                                "call finish_investigation; no other tool will be executed."})
            self.messages.append({"role": "user", "content": results})
            if final is not None:
                return self.stop("finished", final)
            if stop_now:
                return self.stop(stop_now)

    def reject(self, problem):
        """Count a rejected action; True when the run must stop."""
        self.rejected_in_a_row += 1
        return self.rejected_in_a_row >= self.budget.max_consecutive_rejections

    def error_result(self, use, kind, message):
        return {"type": "tool_result", "tool_use_id": use.get("id"), "is_error": True,
                "content": json.dumps({"refused": kind, "message": message})}

    def handle(self, use):
        """(tool result content, is_error, accepted final answer or None)."""
        name, args = use.get("name"), use.get("input")
        entry = {"tool_use_id": use.get("id"), "tool": name, "arguments": _safe_copy(args)}
        try:
            if name not in TOOL_NAMES:
                raise _Rejected("unknown_tool", f"unknown tool {str(name)[:80]!r}; available: {', '.join(TOOL_NAMES)}")
            if not isinstance(args, dict):
                raise _Rejected("invalid_arguments", "tool input must be a JSON object")
            spec = next(t for t in TOOL_SPECS if t["name"] == name)["input_schema"]
            problems = []
            _schema_problems(args, spec, "arguments", problems)
            if problems:
                hint = "" if name == FINISH or not problems[0].startswith("arguments: unknown") else (
                    " (sufficiency, uncertainty, policy and contract fields are set by the investigation)")
                raise _Rejected("final_answer_rejected" if name == FINISH else "invalid_arguments",
                                "; ".join(problems)[:3000] + hint)
            if name == FINISH:
                final = _guarded(self.verify_final, args)
                self.log("finish", **entry, outcome="accepted")
                self.rejected_in_a_row = 0
                return json.dumps({"accepted": True}), False, final
            if self.tool_calls >= self.budget.max_tool_calls:
                raise _Rejected("tool_budget_exhausted", "the tool-call budget is spent; only "
                                "finish_investigation is accepted")
            request = _guarded(self.build_request, name, args)
            key = ic._canonical({"tool": name, "request": request})
            if key in self.executed:
                self.repeats += 1
                raise _Rejected("repeated_call", "an identical request was already answered: evidence_id "
                                f"{self.executed[key]}; it was not executed again")
            self.tool_calls += 1
            try:
                result = ic.call_tool(self.tools, name, request, self.policy)
            except ic.ForbiddenEvidence as exc:
                raise _Rejected("unauthorized", f"refused by the evidence policy: {str(exc)[:300]}") from None
            except ic.ContractError as exc:
                raise _Rejected("invalid_arguments", f"invalid request: {str(exc)[:300]}") from None
            except ic.ToolContractViolation:
                raise _Rejected("tool_contract_violation", "the tool returned a result that broke its "
                                "contract; it is not evidence") from None
        except _Rejected as exc:
            self.log("tool_call" if name != FINISH else "finish", **entry, outcome="refused", refused=exc.kind,
                     message=str(exc)[:MAX_TEXT_TRACE])
            self.reject(exc.kind)
            return json.dumps({"refused": exc.kind, "message": str(exc)}), True, None
        eid = result["evidence_id"]
        self.evidence[eid] = {"tool": name, "request": copy.deepcopy(request), "result": copy.deepcopy(result)}
        self.executed[key] = eid
        self.rejected_in_a_row = 0
        self.log("tool_call", **entry, request=request, outcome=result["status"], evidence_id=eid,
                 errors=[e["code"] for e in result["errors"]])
        return _render(result, self.budget.max_result_chars), False, None

    # -- arguments --

    def _from(self, ref, tool):
        if not isinstance(ref, dict) or not isinstance(ref.get("evidence_id"), str):
            raise _Rejected("invalid_arguments", "a *_from reference needs an evidence_id")
        ev = self.evidence.get(ref["evidence_id"])
        if ev is None:
            raise _Rejected("unknown_evidence", f"{ref['evidence_id'][:80]!r} is not a result of this run")
        if ev["tool"] != tool or ev["result"]["status"] == "error":
            raise _Rejected("invalid_arguments", f"{ref['evidence_id']} is not a {tool} result with measurements")
        return ev["result"]

    def _windows(self, ref, kinds):
        allowed = {"evidence_id", "status", "kind"}
        if isinstance(ref, dict) and set(ref) - allowed:
            raise _Rejected("invalid_arguments", f"windows_from has unknown fields {sorted(set(ref) - allowed)}")
        res = self._from(ref, "get_window_features")
        status, kind = ref.get("status", "included"), ref.get("kind", "all")
        if status not in ("included", "excluded", "all") or kind not in ("pre_event", "baseline", "all"):
            raise _Rejected("invalid_arguments", "windows_from status / kind not recognised")
        out = {k: [w["window_id"] for w in res["measurements"]["windows"]
                   if w["kind"] == k and status in ("all", w["status"]) and kind in ("all", k)] for k in kinds}
        if not any(out.values()):
            raise _Rejected("invalid_arguments", "windows_from selects no window")
        return out

    def build_request(self, name, args):
        """The contract request for a model's arguments (references expanded,
        investigation-level fields injected). Values are left to the contract."""
        a = copy.deepcopy(args)             # schema-checked by handle(): types, required, no unknown fields
        req = {"contract_version": ic.CONTRACT_VERSION}
        if name == "query_events":
            req.update(source=a["source"], behavior=a["behavior"], sessions=a.get("sessions"),
                       max_episodes=a.get("max_episodes", ic.MAX_EPISODES),
                       merge_gap_frames=a.get("merge_gap_frames"), min_duration_frames=a.get("min_duration_frames"))
        elif name == "get_window_features":
            if ("episode_ids" in a) == ("episodes_from" in a):
                raise _Rejected("invalid_arguments", "give exactly one of episode_ids, episodes_from")
            ids = a.get("episode_ids")
            if ids is None:
                ids = [e["episode_id"] for e in self._from(a["episodes_from"], "query_events")["measurements"]["episodes"]]
                if not ids:
                    raise _Rejected("invalid_arguments", "episodes_from selects no episode")
            req.update(episode_ids=ids, pre_event=a["pre_event"], baseline=a.get("baseline"), features=a["features"])
            if isinstance(req["baseline"], dict):
                req["baseline"] = {"strategy": "same_session_tiles", **req["baseline"]}
        elif name == "check_tracking_quality":
            if ("window_ids" in a) == ("windows_from" in a):
                raise _Rejected("invalid_arguments", "give exactly one of window_ids, windows_from")
            ids = a.get("window_ids")
            if ids is None:
                w = self._windows(a["windows_from"], ("pre_event", "baseline"))
                ids = w["pre_event"] + w["baseline"]
            req.update(window_ids=ids, history_frames=a["history_frames"])
        elif name == "compare_windows":
            pre, base = a.get("pre_window_ids"), a.get("baseline_window_ids")
            if (pre is None or base is None) != ("windows_from" in a):
                raise _Rejected("invalid_arguments", "each group needs pre_window_ids / baseline_window_ids, "
                                "or windows_from for the groups not listed")
            if "windows_from" in a:
                w = self._windows(a["windows_from"], ("pre_event", "baseline"))
                pre = w["pre_event"] if pre is None else pre
                base = w["baseline"] if base is None else base
            req.update(feature=a["feature"], statistic="median_difference", aggregation="session_level",
                       pre_window_ids=pre, baseline_window_ids=base, sufficiency=dict(self.sufficiency),
                       uncertainty={"method": "none", "level": None, "resamples": None, "seed": None})
        try:
            ic.ensure_json(req, "request")
        except ic.ContractError as exc:
            raise _Rejected("invalid_arguments", str(exc)[:300]) from None
        return req

    # -- the final answer --

    def verify_final(self, a):
        """The accepted final answer (observations resolved from the stored
        results), or _Rejected with every problem found. `a` is schema-checked."""
        problems = []
        finding, direction, primary = a["finding"], a.get("direction"), a.get("primary_evidence_id")
        if primary is not None and primary not in self.evidence:
            problems.append(f"primary_evidence_id {primary[:80]!r} is not a result of this run")
        if finding == "effect_described":
            ev = self.evidence.get(primary)
            if ev is None or ev["tool"] != "compare_windows" or ev["result"]["status"] != "ok":
                problems.append("effect_described needs primary_evidence_id of a compare_windows result with "
                                "status 'ok'")
            else:
                value = ev["result"]["measurements"]["effect"]["value"]
                sign = "negative" if value < 0 else "positive" if value > 0 else "zero"
                if direction != sign:
                    problems.append(f"direction must be {sign!r} (the sign of the cited effect)")
        elif direction is not None:
            problems.append(f"{finding} carries no direction (null)")
        observations = []
        raw = a["observations"]
        if len(raw) > MAX_ITEMS:
            problems.append(f"observations: at most {MAX_ITEMS}")
            raw = []
        for i, it in enumerate(raw):
            obs, problem = self.observe(it["evidence_id"], it["field"])
            if problem:
                problems.append(f"observations[{i}]: {problem}")
            else:
                observations.append(obs)
        if self.evidence and not raw:
            problems.append("at least one observation citing a result field is required")
        interpretations = []
        raw = a.get("interpretations", [])
        if len(raw) > MAX_ITEMS:
            problems.append(f"interpretations: at most {MAX_ITEMS}")
            raw = []
        for i, it in enumerate(raw):
            where = f"interpretations[{i}]"
            if not it["evidence_ids"] or len(it["evidence_ids"]) > MAX_ITEMS:
                problems.append(f"{where}: needs 1 to {MAX_ITEMS} evidence_ids")
                continue
            unknown = [e for e in it["evidence_ids"] if e not in self.evidence]
            if unknown:
                problems.append(f"{where}: evidence_ids not from this run: {[u[:80] for u in unknown[:3]]}")
            problems += [f"{where}: {p}" for p in _prose_problems(it["statement"], finding)]
            interpretations.append({"statement": it["statement"], "evidence_ids": list(it["evidence_ids"])})
        texts = {}
        for part in ("additional_limitations", "next_steps"):
            raw = a.get(part, [])
            if len(raw) > MAX_ITEMS:
                problems.append(f"{part}: at most {MAX_ITEMS}")
                raw = []
            for i, t in enumerate(raw):
                problems += [f"{part}[{i}]: {p}" for p in _prose_problems(t, finding)]
            texts[part] = list(raw)
        if problems:
            raise _Rejected("final_answer_rejected", "; ".join(problems)[:3000])
        return {"finding": finding, "direction": direction, "primary_evidence_id": primary,
                "observations": observations, "interpretations": interpretations, **texts}

    def observe(self, eid, path):
        """(observation, None) for a field of a stored result, or (None, problem).
        The value and statement come from the stored result, never the model."""
        ev = self.evidence.get(eid)
        if ev is None:
            return None, f"evidence_id {eid[:80]!r} is not a result of this run"
        if len(path) > MAX_FIELD or not _FIELD.fullmatch(path):
            return None, (f"field {path[:80]!r} is not a path such as status, measurements.effect.value or "
                          "measurements.per_session[0].difference")
        value = ev["result"]
        for key, index in re.findall(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", path):
            if key and isinstance(value, dict) and key in value:
                value = value[key]
            elif index and isinstance(value, list) and int(index) < len(value):
                value = value[int(index)]
            else:
                return None, f"{eid[:80]} has no field {path!r}"
        scalar = (str, int, float, bool, type(None))
        if not (isinstance(value, scalar) or isinstance(value, list) and len(value) <= MAX_LIST_VALUE
                and all(isinstance(v, scalar) for v in value)):
            return None, f"{path!r} is not a value (cite a field inside it)"
        value = copy.deepcopy(value)
        return {"evidence_id": eid, "tool": ev["tool"], "field": path, "value": value,
                "statement": f"{ev['tool']} {eid}: {path} = {json.dumps(value)}"}, None

    # -- the report --

    def stop(self, reason, final=None):
        self.log("stop", stopping_reason=reason)
        limitations, seen = [], set()
        for eid, ev in self.evidence.items():
            for text in ev["result"]["limitations"]:
                if text not in seen:
                    seen.add(text)
                    limitations.append({"source": "tool", "text": text})
        limitations += [{"source": "agent", "text": t} for t in AGENT_LIMITATIONS]
        if reason == "usage_unavailable":
            limitations.append({"source": "agent", "text": "The last model reply reported no valid token usage, so "
                                "the token / cost budget could no longer be enforced; it was not used and the "
                                "run stopped. Usage totals exclude that reply."})
        if self.overrun:
            limitations.append({"source": "agent", "text": f"The last model request went past the {self.overrun} "
                                "limit (a request cannot be interrupted); none of its actions were executed."})
        if final is None:
            limitations.append({"source": "agent", "text": f"The investigation stopped before a final answer "
                                f"({reason}); the evidence listed is partial and no finding was made."})
        else:
            limitations += [{"source": "model", "text": t} for t in final["additional_limitations"]]
        primary = None
        pid = final and final["primary_evidence_id"]
        if pid:
            ev = self.evidence[pid]
            m = ev["result"]["measurements"]
            primary = {"evidence_id": pid, "tool": ev["tool"], "status": ev["result"]["status"]}
            if ev["tool"] == "compare_windows" and m is not None:
                primary.update(feature=m["feature"], effect=copy.deepcopy(m["effect"]),
                               sufficiency=copy.deepcopy(m["sufficiency"]), n_sessions_paired=m["n_sessions_paired"],
                               n_windows_pre=m["n_windows_pre"], n_windows_baseline=m["n_windows_baseline"])
        return InvestigationReport(
            question=self.question, stopping_reason=reason, finished=final is not None,
            finding=final["finding"] if final else "incomplete", direction=final["direction"] if final else None,
            primary_evidence=primary,
            observations=[{"kind": "observation", **copy.deepcopy(o)} for o in final["observations"]] if final else [],
            interpretations=[{"kind": "interpretation", **o} for o in final["interpretations"]] if final else [],
            evidence=[{"evidence_id": e, "tool": v["tool"], "status": v["result"]["status"],
                       "request": copy.deepcopy(v["request"]), "result": copy.deepcopy(v["result"])}
                      for e, v in self.evidence.items()],
            limitations=limitations, next_steps=final["next_steps"] if final else [],
            usage={"model_turns": self.turns, "tool_calls": self.tool_calls, "repeated_calls": self.repeats,
                   "input_tokens": self.tokens_in, "output_tokens": self.tokens_out, "cost_usd": self.cost(),
                   "turns_without_usage": self.unreported,
                   "elapsed_s": round(self.clock() - self.start, 3)},
            trace=copy.deepcopy(self.trace))


def investigate(question, tools, policy, model, *, budget=None, sufficiency=ic.PROVISIONAL_SUFFICIENCY,
                clock=time.monotonic):
    """Run one bounded investigation; always returns an InvestigationReport."""
    return _Run(question, tools, policy, model, budget or Budget(), sufficiency, clock).run()


# ---- helpers ----

def _usage(reply):
    """(input tokens, output tokens) a reply reports, each None when it is
    missing or not a non-negative integer; read without trusting the reply."""
    def count(name):
        try:
            n = getattr(reply, name, None)
        except Exception:
            return None
        return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None
    return count("input_tokens"), count("output_tokens")


def _use_ids(reply):
    content = getattr(reply, "content", None)
    if not isinstance(content, list):
        return []
    return [b.get("id") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _reply_problem(reply):
    if not isinstance(reply, ModelReply):
        return "not a ModelReply"
    if not isinstance(reply.content, list) or not all(isinstance(b, dict) and isinstance(b.get("type"), str)
                                                      for b in reply.content):
        return "content must be a list of typed blocks"
    if not isinstance(reply.stop_reason, str):
        return "missing stop_reason"
    ids = []
    for b in reply.content:
        if b["type"] == "tool_use":
            if not isinstance(b.get("id"), str) or not b["id"] or not isinstance(b.get("name"), str):
                return "a tool call needs an id and a name"
            ids.append(b["id"])
        elif b["type"] == "text" and not isinstance(b.get("text"), str):
            return "a text block needs text"
    if len(set(ids)) != len(ids):
        return "duplicate tool call ids"
    return None


def _safe_copy(x):
    """A JSON copy of model-supplied arguments for the trace (repr if not JSON)."""
    try:
        return json.loads(json.dumps(x, allow_nan=False))
    except (TypeError, ValueError):
        return repr(x)[:500]


def _render(result, max_chars):
    """The validated result as JSON for the model; long lists are shortened
    (the full result stays stored under its evidence_id)."""
    text = json.dumps(result, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    for keep in (20, 10, 5, 2, 0):
        short = _shorten(result, keep)
        short["_rendering"] = (f"lists longer than {keep} items were shortened for length; the full validated "
                               f"result is stored as {result['evidence_id']}; pass IDs with episodes_from / "
                               "windows_from")
        text = json.dumps(short, separators=(",", ":"))
        if len(text) <= max_chars:
            return text
    # still too long: a small, valid summary (never cut JSON)
    m = result.get("measurements") or {}
    summary = {k: result[k] for k in ("contract_version", "tool", "status", "evidence_id", "errors")}
    summary["measurements_scalars"] = {k: v[:200] if isinstance(v, str) else v for k, v in m.items()
                                       if not isinstance(v, (list, dict))}
    summary["_rendering"] = (f"result too long to show; the full validated result is stored as "
                             f"{result['evidence_id']}; pass IDs with episodes_from / windows_from")
    for extra in ({"limitations": result.get("limitations", [])}, {}):
        text = json.dumps({**summary, **extra}, separators=(",", ":"))
        if len(text) <= max_chars:
            return text
    summary.pop("measurements_scalars")
    return json.dumps(summary, separators=(",", ":"))


def _shorten(x, keep):
    if isinstance(x, dict):
        return {k: _shorten(v, keep) for k, v in x.items()}
    if isinstance(x, list):
        if len(x) > keep and x and isinstance(x[0], dict):
            return [_shorten(v, keep) for v in x[:keep]] + [{"_omitted_items": len(x) - keep}]
        return [_shorten(v, keep) for v in x]
    return x


def _accepts_limits(model):
    try:
        params = inspect.signature(model.respond).parameters.values()
    except (TypeError, ValueError, AttributeError):
        return False
    return any(p.name == "limits" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _guarded(fn, *args):
    """Run an argument check / expansion; an unexpected exception there is a
    refused action, not a crashed run."""
    try:
        return fn(*args)
    except _Rejected:
        raise
    except Exception as exc:
        raise _Rejected("invalid_arguments", f"the arguments could not be processed ({type(exc).__name__})") from None


_JSON_TYPES = {"object": dict, "array": list, "string": str, "null": type(None), "boolean": bool}


def _json_type(x):
    if isinstance(x, bool):
        return "boolean"
    if isinstance(x, int):
        return "integer"
    if isinstance(x, float):
        return "number"
    return next((n for n, t in _JSON_TYPES.items() if isinstance(x, t)), type(x).__name__)


def _schema_problems(x, schema, path, out):
    """Check model arguments against the subset of JSON Schema TOOL_SPECS use
    (type, enum, properties / required / additionalProperties, items,
    minimum / maximum). A field given as null is a type error unless the
    schema allows null; a field left out is checked only by `required`."""
    if len(out) >= 10:
        return
    if "enum" in schema and not any(x == e and _json_type(x) == _json_type(e) for e in schema["enum"]):
        out.append(f"{path} must be one of {schema['enum']}")
        return
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else types
        actual = _json_type(x)
        if not any(t == actual or (t == "number" and actual == "integer") for t in types):
            out.append(f"{path} must be {' or '.join(types)}, not {actual}"
                       + (" (leave the field out instead of null)" if x is None else ""))
            return
    if isinstance(x, dict) and "properties" in schema:
        extra = sorted(set(x) - set(schema["properties"]))
        if extra and schema.get("additionalProperties") is False:
            out.append(f"{path}: unknown fields {[str(k)[:60] for k in extra[:10]]}")
        missing = [k for k in schema.get("required", []) if k not in x]
        if missing:
            out.append(f"{path}: missing fields {missing}")
        for k, v in x.items():
            if k in schema["properties"]:
                _schema_problems(v, schema["properties"][k], f"{path}.{k}", out)
    elif isinstance(x, list) and "items" in schema:
        for i, v in enumerate(x):
            _schema_problems(v, schema["items"], f"{path}[{i}]", out)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        if "minimum" in schema and x < schema["minimum"]:
            out.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and x > schema["maximum"]:
            out.append(f"{path} must be <= {schema['maximum']}")


def _prose_problems(text, finding):
    """Why model-written prose cannot be accepted (empty: it can)."""
    if not text.strip() or len(text) > MAX_STATEMENT:
        return [f"must be non-empty text of at most {MAX_STATEMENT} characters"]
    out = [f"{word} language is not allowed" for pat, word in _BANNED if pat.search(text)]
    quantities = [m.group() for m in _NUMBER.finditer(text)] + [m.group() for m in _NUMBER_WORDS.finditer(text)]
    if quantities:
        out.append(f"numbers or number words {quantities[:5]} are not allowed in prose; cite measurements as "
                   "observations {evidence_id, field}")
    if finding != "effect_described":
        effect = [m.group() for m in _EFFECT_WORDS.finditer(text)]
        if effect:
            out.append(f"effect language {effect[:5]} is not allowed when the finding is {finding}")
    return out


# ---- the Claude adapter ----

API_KEY_ENV = "ANTHROPIC_API_KEY"
MODEL_ENV = "LIVE_TRACKER_CLAUDE_MODEL"
DEFAULT_MODEL = "claude-opus-5-5"
_ERROR_REASONS = (("APITimeoutError", "timeout"), ("RateLimitError", "rate_limited"),
                  ("AuthenticationError", "auth_error"), ("PermissionDeniedError", "auth_error"),
                  ("APIConnectionError", "connection_error"))


class ClaudeInvestigationModel:
    """InvestigationModel over the Anthropic Messages API (manual tool-use
    loop: this module executes every tool). One respond() is one
    messages.create request, resent at most max_retries times by the SDK.

    model: argument, else $LIVE_TRACKER_CLAUDE_MODEL, else DEFAULT_MODEL.
    client: an anthropic.Anthropic-like object (tests inject one); without
    it one is built from $ANTHROPIC_API_KEY only (no ambient profiles), as
    in the Day 2 adapter. Thinking is left to the model's default (adaptive);
    effort is explicit. fallbacks=True asks the API to rerun a declined
    request on its default fallback model; the serving model is recorded in
    the trace. The system prompt and tool list are cached (stable prefix).
    With RequestLimits, max_tokens is capped by the remaining output-token
    allowance and the per-attempt timeout by the remaining time divided over
    the attempts (1 + max_retries)."""

    def __init__(self, client=None, *, model=None, max_tokens=16_000, effort="high", timeout_s=180.0,
                 max_retries=2, fallbacks=True):
        self.model = (model or os.environ.get(MODEL_ENV, "") or DEFAULT_MODEL).strip()
        self.max_tokens, self.effort, self.fallbacks = max_tokens, effort, fallbacks
        self.timeout_s, self.max_retries = timeout_s, max_retries
        self._client = client

    def _bounded(self, timeout_s):
        client = self._client
        if client is None:
            key = os.environ.get(API_KEY_ENV, "").strip()
            if not key:
                raise ModelUnavailable("missing_api_key", f"{API_KEY_ENV} is not set")
            try:
                import anthropic
            except ImportError:
                raise ModelUnavailable("sdk_not_installed", 'pip install -e ".[claude]"') from None
            client = self._client = anthropic.Anthropic(api_key=key)
        with_options = getattr(client, "with_options", None)
        if not callable(with_options):
            raise ModelUnavailable("client_error", "the client has no with_options(); limits cannot be enforced")
        return with_options(timeout=timeout_s, max_retries=self.max_retries)

    def request(self, system, tools, messages, limits=None):
        max_tokens = self.max_tokens
        if limits is not None and limits.max_output_tokens is not None:
            max_tokens = max(1, min(max_tokens, limits.max_output_tokens))
        kwargs = {"model": self.model, "max_tokens": max_tokens,
                  "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                  "tools": tools, "tool_choice": {"type": "auto"}, "messages": messages,
                  "output_config": {"effort": self.effort}}
        if self.fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return kwargs

    def respond(self, system, tools, messages, limits=None):
        timeout_s = self.timeout_s
        if limits is not None:
            timeout_s = max(0.001, min(timeout_s, limits.timeout_s / (1 + self.max_retries)))
        client = self._bounded(timeout_s)
        kwargs = self.request(system, tools, messages, limits)
        try:
            api = client.beta.messages if self.fallbacks else client.messages
            response = api.create(**kwargs)
        except Exception as exc:  # SDK errors; reported by class name only (text may hold secrets)
            names = {c.__name__ for c in type(exc).__mro__}
            raise ModelUnavailable(next((r for n, r in _ERROR_REASONS if n in names), "api_error"),
                                   type(exc).__name__) from None
        content = [_block_dict(b) for b in (getattr(response, "content", None) or [])]
        usage = getattr(response, "usage", None)

        def tok(name, optional=False):
            v = getattr(usage, name, None) if usage is not None else None
            if v is None and optional:
                return 0                      # cache counts are absent when nothing was cached
            return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
        parts = [tok("input_tokens"), tok("cache_creation_input_tokens", True), tok("cache_read_input_tokens", True)]
        # unknown usage stays None (the loop then refuses to continue under a token / cost limit)
        return ModelReply(content=content, stop_reason=str(getattr(response, "stop_reason", "") or ""),
                          input_tokens=None if None in parts else sum(parts), output_tokens=tok("output_tokens"),
                          model=getattr(response, "model", None))


def _block_dict(block):
    """A response content block as the dict sent back unchanged next turn."""
    if isinstance(block, dict):
        return copy.deepcopy(block)
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    return {"type": "unknown"}
