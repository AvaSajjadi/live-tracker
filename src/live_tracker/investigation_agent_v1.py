"""The Stage 9 autonomous investigation agent on the approved Contract v1 tools (Phase 3A, new file).

The SAME bounded plan -> act -> observe loop as investigation_agent (its _Run: model requests, usage and budget
accounting, refusal and stop rules, the trace), with the approved Contract v1 tools (investigation_tools_v1)
as the only scientific tools. Nothing here fixes a tool sequence: the model chooses every call and its
arguments from the results it has seen.

    report = investigate_v1(question, data, model)       # data: investigation_tools_v1.InvestigationData

What the model can and cannot do
--------------------------------
- Tools (TOOL_SPECS_V1): query_events, check_tracking_quality, get_window_features, compare_windows, and the
  structured finish. They take Contract v1's own request fields in seconds. compare_windows cites a query of this
  run by evidence_id (query_from); the loop expands it into the query's id, parameters and complete bout set, so
  the model never copies or edits bouts. pre_s is limited to the preselected grid {1, 2, 5} s.
- Not in any schema (sending them is an argument error): the quality policy, the overlap policy, the evidence gate
  (min_n_windows, min_n_sessions, gate_override_reason) and any uncertainty method. Contract v1 applies quality_v1,
  overlap_v1 and the 10-window / 4-session gate itself.
- Execution: every call goes through ONE investigation_tools_v1.InvestigationRunner, which enforces the 12-call
  budget, at most 2 follow-up comparisons, the pre_s grid and "a follow-up changes only pre_s"; the tools validate
  every request exactly. The loop also enforces the caller's Budget: a comparison beyond the stricter of
  Budget.max_compare_follow_ups and Contract v1's two (follow_up_limit) is refused before it reaches the runner.
  A call the loop, the runner or a tool refuses is a traced, refused action, never evidence.
- Data: an InvestigationData built by its constructors. Its sessions are validation sessions only (the real
  constructors go through the manifest-backed evidence policy and the verified cache); a session outside it is
  refused by the tool. A synthetic source is refused unless allow_synthetic=True (tests only): no synthetic
  fallback.

Evidence and findings (format "contract-v1/1")
----------------------------------------------
- Each accepted result is stored with its deterministic id (investigation_tools_v1.evidence_id of the canonical
  tool, request and result), the exact request sent, the whole result, its status and its provenance (contract,
  data source, data fingerprint, split, investigation id, call number).
- The finish is structured and holds no prose: finding, primary_evidence_id and observations, which are
  references {evidence_id, field} to fields of this run's results. The loop reads every value itself, so there
  is no model-written number, sentence, direction, significance, interval, causal or generalisation claim. A
  finish with any other field is refused.
- effect_described needs a primary compare_windows result with status "ok" and observes its mean_diff;
  insufficient_evidence needs a primary compare_windows result with status "insufficient_evidence" and observes
  its reason; inconclusive needs at least one observation. A refused finish goes back to the model and never
  enters the report.
- At the accepted finish the evidence, the execution prefix and the finding are frozen with a SHA-256
  fingerprint; at the stop the completed trace must be that prefix plus one stop event and the evidence
  unchanged, or FinishVerificationError. A run that stops without an accepted finish is "incomplete".
- verify_report_v1(report_json) re-derives a saved report field by field before it looks at a fingerprint (anyone
  can recompute a fingerprint, so a matching one proves nothing): a fixed top-level schema; the canonical
  disclosures, exactly; the trace's structure (ordered steps, every tool call or finish asked for by the model turn
  before it, exactly one final stop); every evidence id, and its provenance against the data and the execution;
  the runner's attempts against the trace's executions; usage and call counts against the trace and the recorded
  budget (including the follow-up limit); the data source against every result and every query's run id; a replay
  of every model call through the live admission (_Ledger.admit: the same closed schemas, query_from resolved only
  against queries accepted earlier, the same request expansion and limits), so each refusal must be recorded as
  that refusal and each executed request, runner attempt and accepted finish must be exactly what its recorded
  arguments produce; and the completion (the finding one of FINDINGS): a finished report needs exactly one accepted finish, immediately before the "finished" stop, whose
  arguments are the report's finding and observations; an unfinished report has no accepted finish and is
  "incomplete". Then the finish fingerprint and the whole-report fingerprint. render_markdown_v1 renders only a
  report that verifies, and prints the disclosures from the canonical text. Publication is always
  "not_authorized"; verification is consistency, not authenticity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, fields

from . import investigation_agent as ia
from . import investigation_context as ctx
from . import investigation_tools_v1 as v1
from .rules import CLASSIFIER_BEHAVIORS

__all__ = ["DISCLOSURES", "FINDINGS_FORMAT", "InvestigationReportV1", "SYSTEM_PROMPT_V1", "TOOL_SPECS_V1",
           "investigate_v1", "render_markdown_v1", "verify_report_v1"]

FINDINGS_FORMAT = "contract-v1/1"
FINISH = ia.FINISH
FINDINGS = ("effect_described", "insufficient_evidence", "inconclusive")
PUBLICATION = "not_authorized"
MAX_OBSERVATIONS = 20
MAX_LIST_VALUE = 20
# result fields that are fixed descriptions of the method, not measurements: they are not observed (the report
# states the method from Contract v1's own definitions)
NOT_OBSERVABLE = frozenset({"wording", "windows", "mean_diff_definition", "start_meaning", "quality_policy_definition",
                            "overlap_policy_definition", "exclusion_order", "usable_value_rule", "causal", "query",
                            "association_only", "contract_version", "derived_from", "data_source", "message"})
FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*|\[\d{1,4}\])*")
DISCLOSURES = (
    "LLM-planned investigation: a language model chose which Contract v1 tools to call and with which arguments; a "
    "different plan could select different evidence. The trace records every choice.",
    "Every value in this report was read by the investigation from a validated Contract v1 result of this run; the "
    "model wrote no number, sentence or claim.",
    "The bouts are classifier predictions (threshold crossings of model B probabilities), not annotated attacks.",
    "Association only: a comparison describes feature values before predicted bouts relative to a baseline; it is "
    "not a cause, not a prediction and carries no confidence interval, p-value or significance test.",
    "Validation data only; nothing here says how the result would hold on the held-out test split or other data.",
    "Quality policy quality_v1, overlap policy overlap_v1 and the 10-window / 4-session evidence gate were fixed by "
    "Contract v1, not chosen by the model.",
    "Consistency is checked (evidence identities, observed values, finding rules, fingerprint); this is not an "
    "independent authentication, and publication is not authorized.",
)

_SECONDS = {"type": "number", "minimum": 0}
TOOL_SPECS_V1 = [
    {"name": "query_events",
     "description": "Predicted bouts of one behavior: runs of frames whose classifier score is >= threshold with "
                    "tracking_ok, kept when they last at least min_duration_s (Contract v1). start_s is a "
                    "threshold crossing, not an annotated onset. sessions: null for every session of this "
                    "investigation's data (validation sessions only).",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["behavior", "min_duration_s"],
                      "properties": {
                          "behavior": {"enum": sorted(CLASSIFIER_BEHAVIORS)},
                          "sessions": {"type": ["array", "null"], "items": {"type": "string"}},
                          "min_duration_s": _SECONDS,
                          "threshold": {"type": "number", "minimum": 0, "maximum": 1}}}},
    {"name": "check_tracking_quality",
     "description": "Tracking quality of one window [t0_s, t1_s) of one session under the fixed policy quality_v1 "
                    "(usable fraction >= 0.8, no frozen / out-of-frame run > 0.5 s). Never verifies motion accuracy.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["session", "t0_s", "t1_s"],
                      "properties": {"session": {"type": "string"}, "t0_s": _SECONDS, "t1_s": _SECONDS}}},
    {"name": "get_window_features",
     "description": "Summaries (mean, min, max, n) of pair features over one window [t0_s, t1_s) of one session, "
                    "computed causally from the session truncated at t1_s.",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["session", "t0_s", "t1_s", "cols"],
                      "properties": {"session": {"type": "string"}, "t0_s": _SECONDS, "t1_s": _SECONDS,
                                     "cols": {"type": "array", "minItems": 1,
                                              "items": {"enum": list(v1.WINDOW_FEATURES)}}}}},
    {"name": "compare_windows",
     "description": "Pre-bout window [start_s - pre_s, start_s) vs an earlier baseline window, paired by session, "
                    "over the complete bouts of one query_events result of this run (query_from). Contract v1 "
                    "applies quality_v1, overlap_v1 and the evidence gate (>= 10 windows in >= 4 sessions) itself. "
                    "pre_s is one of 1, 2, 5 s; after the first comparison at most two follow-ups are allowed and a "
                    "follow-up may change only pre_s. Effect: the median of the session-paired differences.",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["query_from", "pre_s", "baseline_s", "baseline_gap_s", "feature"],
                      "properties": {
                          "query_from": {"type": "object", "additionalProperties": False, "required": ["evidence_id"],
                                         "properties": {"evidence_id": {"type": "string"}}},
                          "pre_s": {"type": "integer", "enum": [int(x) for x in v1.FOLLOW_UP_PRE_S]},
                          "baseline_s": _SECONDS, "baseline_gap_s": _SECONDS,
                          "feature": {"enum": list(v1.WINDOW_FEATURES)}}}},
    {"name": FINISH,
     "description": "Submit the final answer. It holds no prose: a finding and references to fields of this run's "
                    "results, which the investigation reads itself. It is checked and refused (with the problems) "
                    "if it is not supported.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["finding", "observations"],
                      "properties": {
                          "finding": {"enum": list(FINDINGS)},
                          "primary_evidence_id": {"type": ["string", "null"]},
                          "observations": {"type": "array", "items": {
                              "type": "object", "additionalProperties": False, "required": ["evidence_id", "field"],
                              "properties": {"evidence_id": {"type": "string"},
                                             "field": {"type": "string", "description": (
                                                 "a field of that result, e.g. status, mean_diff, n_windows, "
                                                 "per_session[0].paired_diff, reason")}}}}}}},
]
TOOL_NAMES_V1 = tuple(t["name"] for t in TOOL_SPECS_V1)
assert set(TOOL_NAMES_V1) == set(v1.TOOLS) | {FINISH}

SYSTEM_PROMPT_V1 = """\
You are the investigator in Live Tracker's Investigation mode. You answer a research question about mouse \
social behavior in CalMS21 validation sessions, using only the tools provided (Contract v1). Each tool result \
you receive carries an evidence_id.

Work iteratively: decide what to measure, call a tool, read the result, and choose the next step from what it \
actually shows. Stop when further calls would not change the answer, or when the evidence cannot support one.

Rules:
- Facts come only from tool results. Never invent sessions, times, measurements or evidence_ids.
- Bouts from query_events are classifier predictions, not annotated behavior.
- compare_windows takes the bouts of one query_events result by reference (query_from). The quality policy, \
the overlap policy and the evidence gate are fixed by the contract; you cannot set them. pre_s is 1, 2 or 5 \
seconds; after the first comparison at most two follow-ups, each changing only pre_s.
- finish_investigation holds no prose. Choose the finding and cite fields of results as observations \
{evidence_id, field}; the investigation reads the values itself. "effect_described" needs a compare_windows \
result with status "ok" as primary_evidence_id and an observation of its mean_diff; "insufficient_evidence" \
needs a compare_windows result with status "insufficient_evidence" as primary and an observation of its reason; \
otherwise use "inconclusive". The comparison is descriptive and associative only.
- Your budget of turns and tool calls is limited; identical repeated requests are refused."""


@dataclass(frozen=True)
class InvestigationReportV1:
    """A contract-v1/1 report: the frozen finding (or "incomplete"), the evidence, the trace, the runner's attempt
    record and the content fingerprint. Rendered only from verified content (render_markdown_v1)."""
    findings_format: str
    question: str
    finished: bool
    stopping_reason: str
    finding: str
    primary_evidence_id: str | None
    observations: list                  # [{evidence_id, field, value}]: values read by the investigation
    evidence: list
    trace: list
    usage: dict
    runner_record: dict
    budget: dict                        # the Budget the run was bound by
    data: dict                          # data source identity (bound to the evidence by verify_report_v1)
    finish_fingerprint: str | None      # frozen at the accepted finish (None when incomplete)
    fingerprint: str                    # every other field of the report, canonical
    publication: str = PUBLICATION
    disclosures: tuple = DISCLOSURES

    def to_json(self):
        out = copy.deepcopy(self.__dict__)
        out["disclosures"] = list(self.disclosures)
        return out


# ---- shared checks (used at the finish, at the stop and on a saved report) ----

def _resolve(result, path):
    if len(path) > ctx.MAX_FIELD or not FIELD_RE.fullmatch(path):
        raise KeyError(path)
    head = re.match(r"[A-Za-z_][A-Za-z0-9_]*", path).group(0)
    if head in NOT_OBSERVABLE:
        raise KeyError(path)
    value = result
    for key, index in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d{1,4})\]", path):
        if key:
            if not isinstance(value, dict) or key not in value:
                raise KeyError(path)
            value = value[key]
        else:
            if not isinstance(value, list) or int(index) >= len(value):
                raise KeyError(path)
            value = value[int(index)]
    return value


def _observe(evidence, eid, path):
    """(observation, None) or (None, problem): the value is read from the stored result, never from the model."""
    ev = evidence.get(eid)
    if ev is None:
        return None, f"evidence_id {str(eid)[:80]!r} is not a result of this run"
    if not isinstance(path, str):
        return None, "field must be a string"
    try:
        value = _resolve(ev["result"], path)
    except KeyError:
        return None, (f"{eid[:80]} has no observable field {path[:80]!r} (fields such as status, mean_diff, n_windows, "
                      "per_session[0].paired_diff; method descriptions are not observations)")
    scalar = (str, int, float, bool, type(None))
    if not (isinstance(value, scalar) or isinstance(value, list) and len(value) <= MAX_LIST_VALUE
            and all(isinstance(v, scalar) for v in value)):
        return None, f"{path!r} is not a value (cite a field inside it)"
    return {"evidence_id": eid, "field": path, "value": copy.deepcopy(value)}, None


def _finding_problems(proposal, evidence):
    """(problems, resolved observations) of a structured finish against the evidence (dict eid -> record)."""
    problems, observations = [], []
    finding, primary, raw = proposal["finding"], proposal.get("primary_evidence_id"), proposal["observations"]
    if not evidence:
        problems.append("no tool result of this run exists yet: call a tool before finishing")
    if primary is not None and primary not in evidence:
        problems.append(f"primary_evidence_id {str(primary)[:80]!r} is not a result of this run")
    if not raw or len(raw) > MAX_OBSERVATIONS:
        problems.append(f"give 1 to {MAX_OBSERVATIONS} observations")
        raw = raw[:MAX_OBSERVATIONS]
    seen = set()
    for i, it in enumerate(raw):
        key = (it["evidence_id"], it["field"])
        if key in seen:
            problems.append(f"observations[{i}] repeats an observation")
            continue
        seen.add(key)
        obs, problem = _observe(evidence, it["evidence_id"], it["field"])
        if problem:
            problems.append(f"observations[{i}]: {problem}")
        else:
            observations.append(obs)
    ev = evidence.get(primary) if primary is not None else None
    observed = {(o["evidence_id"], o["field"]) for o in observations}
    if finding == "effect_described":
        if ev is None or ev["tool"] != "compare_windows" or ev["result"]["status"] != "ok":
            problems.append("effect_described needs primary_evidence_id of a compare_windows result with status 'ok'")
        elif (primary, "mean_diff") not in observed:
            problems.append("effect_described must observe the primary comparison's mean_diff")
    elif finding == "insufficient_evidence":
        if ev is None or ev["tool"] != "compare_windows" or ev["result"]["status"] != "insufficient_evidence":
            problems.append("insufficient_evidence needs primary_evidence_id of a compare_windows result with status "
                            "'insufficient_evidence'")
        elif (primary, "reason") not in observed:
            problems.append("insufficient_evidence must observe the primary comparison's reason")
    return problems, observations


def follow_up_limit(budget):
    """The effective follow-up limit: the stricter of the caller's Budget and Contract v1's absolute maximum."""
    return min(budget.max_compare_follow_ups, v1.MAX_FOLLOW_UPS)


def _fingerprint(question, finding, primary, observations, evidence_list, trace_prefix):
    payload = {"format": FINDINGS_FORMAT, "question": question, "finding": finding, "primary_evidence_id": primary,
               "observations": observations, "evidence": evidence_list, "trace": trace_prefix}
    return "fpv1:" + hashlib.sha256(ctx.canonical_bytes(payload)).hexdigest()


# ---- admission: the one decision on a model call (the live loop and the saved-report replay both use it) ----

def _arguments(name, args):
    """The live argument check of one model call: a known tool, a JSON object, a whole-number pre_s as its grid
    value, then the tool's closed schema (TOOL_SPECS_V1). The checked arguments, or ia._Rejected."""
    if name not in TOOL_NAMES_V1:
        raise ia._Rejected("unknown_tool", f"unknown tool {str(name)[:80]!r}; available: {', '.join(TOOL_NAMES_V1)}")
    if not isinstance(args, dict) or ia._safe_copy(args) != args:      # the trace keeps a JSON copy of them
        raise ia._Rejected("invalid_arguments", "tool input must be a JSON object")
    pre = args.get("pre_s")
    if name == "compare_windows" and isinstance(pre, float) and pre.is_integer():
        args = {**args, "pre_s": int(pre)}            # 2.0 and 2 are the same grid value
    spec = next(t for t in TOOL_SPECS_V1 if t["name"] == name)["input_schema"]
    problems = []
    ia._schema_problems(args, spec, "arguments", problems)
    if problems:
        hint = "" if name == FINISH else (" (quality and overlap policies, the evidence gate and any "
                                          "uncertainty method are fixed by Contract v1)")
        raise ia._Rejected("final_answer_rejected" if name == FINISH else "invalid_arguments",
                           "; ".join(problems)[:3000] + hint)
    return args


class _Ledger:
    """What one investigation has admitted so far (the accepted evidence in order, the executed requests, the
    tool calls sent to the runner, the comparisons that ran) and admit(), the decision on the next model call.
    _RunV1 decides every call with it; verify_report_v1 replays a saved trace through it, so a saved call is
    judged by the same schemas, query_from resolution, request expansion and limits as at execution."""

    def __init__(self, sessions, budget):
        self.sessions, self.budget = list(sessions), budget
        self.evidence, self.executed = {}, {}       # evidence_id -> record; canonical {tool, request} -> evidence_id
        self.tool_calls = self.comparisons = 0

    def admit(self, name, args):
        """Decide the next call without executing it. A finish: (arguments, observations read from the evidence
        so far). A tool call: (the exact Contract v1 request for the runner, its canonical key). ia._Rejected
        otherwise. Changes nothing."""
        args = _arguments(name, args)
        if name == FINISH:
            return args, ia._guarded(self.finish, args)
        if self.tool_calls >= self.budget.max_tool_calls:
            raise ia._Rejected("tool_budget_exhausted", "the tool-call budget is spent; only "
                               "finish_investigation is accepted")
        request = ia._guarded(self.build_request, name, args)
        if name == "compare_windows" and self.comparisons >= 1 + follow_up_limit(self.budget):
            raise ia._Rejected("follow_up_limit", f"at most {follow_up_limit(self.budget)} follow-up "
                               "compare_windows analyses after the first (the stricter of this investigation's "
                               "budget and Contract v1's limit); this comparison was not executed")
        key = ctx.canonical_bytes({"tool": name, "request": request})
        if key in self.executed:
            raise ia._Rejected("repeated_call", f"an identical request was already answered: evidence_id "
                               f"{self.executed[key]}; it was not executed again")
        return request, key

    def finish(self, args):
        problems, observations = _finding_problems(args, self.evidence)
        if problems:
            raise ia._Rejected("final_answer_rejected", "; ".join(problems)[:3000])
        return observations

    def build_request(self, name, args):
        a = copy.deepcopy(args)
        if name == "query_events":
            req = {"session": a.get("sessions") if a.get("sessions") is not None else list(self.sessions),
                   "behavior": a["behavior"], "min_duration_s": a["min_duration_s"]}
            if "threshold" in a:
                req["threshold"] = a["threshold"]
            return req
        if name in ("check_tracking_quality", "get_window_features"):
            return a
        ref = a.pop("query_from")["evidence_id"]
        ev = self.evidence.get(ref)                 # only a query accepted earlier in this investigation
        if ev is None or ev["tool"] != "query_events":
            raise ia._Rejected("unknown_evidence", f"query_from {str(ref)[:80]!r} is not a query_events result of "
                               "this run")
        q = ev["result"]
        if not q["bouts"]:
            raise ia._Rejected("invalid_arguments", "the cited query found no bout; there is nothing to compare")
        return {"query_run_id": q["query_run_id"], "query": copy.deepcopy(q["query"]),
                "bouts": copy.deepcopy(q["bouts"]), **a}

    def accept(self, name, key, eid, record):
        self.evidence[eid], self.executed[key] = record, eid
        self.comparisons += name == "compare_windows"


# ---- the run: the Stage 6 loop with the Contract v1 tools ----

class _RunV1(ia._Run):
    """investigation_agent._Run with Contract v1 execution: run() (model turns, usage, budget, refusals, stops,
    trace) is inherited unchanged; handle() executes through the InvestigationRunner and stop() builds the
    contract-v1/1 report. Every call is decided by its _Ledger (admit)."""

    evidence = property(lambda self: self.ledger.evidence)
    tool_calls = property(lambda self: self.ledger.tool_calls)

    def __init__(self, question, data, model, budget, clock, investigation_id):
        if not isinstance(data, v1.InvestigationData):
            raise TypeError("data must be an investigation_tools_v1.InvestigationData")
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            raise ValueError("question must be non-empty text of at most 500 characters")
        self.v2, self.frozen = False, None
        self.specs, self.prompt = TOOL_SPECS_V1, SYSTEM_PROMPT_V1
        self.question, self.tools, self.policy, self.model = question, None, None, model
        self.budget, self.clock = budget, clock
        self.data = data
        self.runner = v1.InvestigationRunner(data, investigation_id, max_tool_calls=budget.max_tool_calls)
        self.messages = [{"role": "user", "content": f"Research question: {question}"}]
        self.ledger, self.trace = _Ledger(data.sessions, budget), []
        self.turns = self.repeats = self.rejected_in_a_row = 0
        self.tools_spent_at_turn = None
        self.tokens_in = self.tokens_out = 0
        self.overrun = None
        self.unreported = 0
        self.pass_limits = ia._accepts_limits(model)
        self.start = clock()

    # -- one tool call --

    def handle(self, use):
        name, args = use.get("name"), use.get("input")
        entry = {"tool_use_id": use.get("id"), "tool": name, "arguments": ia._safe_copy(args)}
        try:
            admitted, key = self.ledger.admit(name, args)
            if name == FINISH:
                finish = self.entry("finish", **entry, outcome="accepted")
                final = ia._guarded(self.verify_finish, admitted, key, finish)
                self.trace.append(finish)
                self.rejected_in_a_row = 0
                return json.dumps({"accepted": True, "fingerprint": final["fingerprint"]}), False, final
            request = admitted
            self.ledger.tool_calls += 1
            try:
                result = self.runner.call(name, request)
            except Exception as exc:                 # a programming error in a tool: never evidence
                raise ia._Rejected("tool_failure", f"the tool failed ({type(exc).__name__}); not evidence") from None
            if result.get("status") == "error":
                raise ia._Rejected(f"tool_refused:{result.get('code')}", str(result.get("message"))[:500])
        except ia._Rejected as exc:
            self.repeats += exc.kind == "repeated_call"
            self.log("tool_call" if name != FINISH else "finish", **entry, outcome="refused", refused=exc.kind,
                     message=str(exc)[:ia.MAX_TEXT_TRACE])
            self.reject(exc.kind)
            return json.dumps({"refused": exc.kind, "message": str(exc)}), True, None
        eid = v1.evidence_id(name, request, result)
        self.ledger.accept(name, key, eid, {"tool": name, "request": copy.deepcopy(request),
                                            "result": copy.deepcopy(result),
                                            "provenance": self.provenance(len(self.runner.attempts))})
        self.rejected_in_a_row = 0
        self.log("tool_call", **entry, outcome=result["status"], evidence_id=eid, call=len(self.runner.attempts))
        return self.render(eid, result), False, None

    def provenance(self, call):
        return {"contract_version": v1.CONTRACT_VERSION, "data_source": self.data.data_source,
                "derived_from": self.data.derived_from, "data_fingerprint": self.data.fingerprint,
                "split": "synthetic" if self.data.data_source == "synthetic" else "validation",
                "investigation_id": self.runner.investigation_id, "call": call}

    def render(self, eid, result):
        """The validated result for the model, with its evidence_id; long lists shortened (stored whole)."""
        shown = {"evidence_id": eid, **result}
        text = json.dumps(shown, separators=(",", ":"))
        for keep in (40, 20, 10, 5, 0):
            if len(text) <= self.budget.max_result_chars:
                break
            short = {"evidence_id": eid, **ia._shorten(result, keep)}
            short["_rendering"] = (f"lists longer than {keep} items shortened for length; the full result is "
                                   f"stored as {eid}")
            text = json.dumps(short, separators=(",", ":"))
        return text

    # -- the finish --

    def evidence_list(self):
        return [{"evidence_id": e, "tool": v["tool"], "status": v["result"]["status"],
                 "request": copy.deepcopy(v["request"]), "result": copy.deepcopy(v["result"]),
                 "provenance": copy.deepcopy(v["provenance"])} for e, v in self.evidence.items()]

    def verify_finish(self, args, observations, finish):
        """Freeze an admitted finish (its observations already read by _Ledger.finish)."""
        evidence = self.evidence_list()
        prefix = copy.deepcopy(self.trace) + [copy.deepcopy(finish)]
        fp = _fingerprint(self.question, args["finding"], args.get("primary_evidence_id"), observations, evidence,
                          prefix)
        self.frozen = {"proposal": copy.deepcopy(args), "observations": observations,
                       "evidence": ctx.canonical_bytes(evidence), "prefix": ctx.canonical_bytes(prefix),
                       "fingerprint": fp}
        return {"fingerprint": fp}

    # -- the report --

    def stop(self, reason, final=None):
        self.log("stop", stopping_reason=reason)
        evidence, trace = self.evidence_list(), copy.deepcopy(self.trace)
        usage = {"model_turns": self.turns, "tool_calls": self.tool_calls, "repeated_calls": self.repeats,
                 "input_tokens": self.tokens_in, "output_tokens": self.tokens_out, "cost_usd": self.cost(),
                 "turns_without_usage": self.unreported, "elapsed_s": round(self.clock() - self.start, 3)}
        if final is not None:
            f = self.frozen
            ok = reason == "finished" and ctx.canonical_bytes(evidence) == f["evidence"] and \
                ctx.canonical_bytes(trace[:-1]) == f["prefix"] and ctx.prefix_extends(trace[:-1], trace)
            problems, observations = _finding_problems(f["proposal"], {e["evidence_id"]: e for e in evidence})
            fp = _fingerprint(self.question, f["proposal"]["finding"], f["proposal"].get("primary_evidence_id"),
                              observations, evidence, trace[:-1])
            if not ok or problems or fp != f["fingerprint"]:
                raise ia.FinishVerificationError("the completed run does not reproduce the frozen finish")
            finding, primary = f["proposal"]["finding"], f["proposal"].get("primary_evidence_id")
        else:
            finding, primary, observations, fp = "incomplete", None, [], None
        content = {"findings_format": FINDINGS_FORMAT, "question": self.question, "finished": final is not None,
                   "stopping_reason": reason, "finding": finding, "primary_evidence_id": primary,
                   "observations": observations, "evidence": evidence, "trace": trace, "usage": usage,
                   "runner_record": self.runner.record(), "budget": json.loads(json.dumps(asdict(self.budget))),
                   "data": {"data_source": self.data.data_source, "derived_from": self.data.derived_from,
                            "fingerprint": self.data.fingerprint, "sessions": list(self.data.sessions)},
                   "finish_fingerprint": fp, "publication": PUBLICATION, "disclosures": list(DISCLOSURES)}
        content["fingerprint"] = _content_fingerprint(content)
        return InvestigationReportV1(**{**content, "disclosures": DISCLOSURES})


def investigate_v1(question, data, model, *, budget=None, investigation_id=None, allow_synthetic=False,
                   clock=time.monotonic):
    """One bounded, model-directed investigation over Contract v1 (see the module docstring). allow_synthetic is
    for tests only: real runs refuse a synthetic data source (no synthetic fallback)."""
    if isinstance(data, v1.InvestigationData) and data.data_source == "synthetic" and not allow_synthetic:
        raise ValueError("synthetic data is refused: a real investigation needs a real data source")
    budget = budget or ia.Budget()
    investigation_id = investigation_id or f"agent_v1_{hashlib.sha256(question.encode()).hexdigest()[:12]}"
    return _RunV1(question, data, model, budget, clock, investigation_id).run()


# ---- a saved report: re-derived from its own content, then rendered ----

class ReportVerificationError(ValueError):
    """A saved contract-v1/1 report does not verify."""


REPORT_KEYS = frozenset({"findings_format", "question", "finished", "stopping_reason", "finding",
                         "primary_evidence_id", "observations", "evidence", "trace", "usage", "runner_record",
                         "budget", "data", "finish_fingerprint", "fingerprint", "publication", "disclosures"})
TRACE_KINDS = ("model_turn", "model_error", "tool_call", "finish", "stop", "budget_exceeded")
USAGE_KEYS = frozenset({"model_turns", "tool_calls", "repeated_calls", "input_tokens", "output_tokens", "cost_usd",
                        "turns_without_usage", "elapsed_s"})
RUNNER_KEYS = frozenset({"investigation_id", "contract_version", "max_tool_calls", "calls_used", "data_source",
                         "attempts"})
DATA_KEYS = frozenset({"data_source", "derived_from", "fingerprint", "sessions"})


def _content_fingerprint(content):
    """"fpv1:" + SHA-256 of every report field except the fingerprint itself (consistency, not authenticity)."""
    return "fpv1:" + hashlib.sha256(ctx.canonical_bytes({k: v for k, v in content.items() if k != "fingerprint"})
                                    ).hexdigest()


def _tokens(x):
    """A reported token count as the loop counted it (investigation_agent._usage): a non-negative int, else 0."""
    return x if isinstance(x, int) and not isinstance(x, bool) and x >= 0 else 0


def _reaches_runner(entry):
    """Trace tool calls that the InvestigationRunner received (and therefore recorded as attempts)."""
    return entry.get("kind") == "tool_call" and ("evidence_id" in entry or str(entry.get("refused", "")).startswith(
        "tool_refused:") or entry.get("refused") == "tool_failure")


def _trace_problems(trace):
    problems = []
    if not isinstance(trace, list) or not trace or not all(isinstance(t, dict) for t in trace):
        return ["the trace is not a list of events"]
    last_elapsed, calls = -1.0, {}
    for n, t in enumerate(trace):
        if t.get("step") != n or t.get("kind") not in TRACE_KINDS or not isinstance(t.get("elapsed_s"), (int, float)) \
                or t["elapsed_s"] < last_elapsed:
            problems.append(f"trace step {n} is out of order or of an unknown kind")
            break
        last_elapsed = t["elapsed_s"]
        if t["kind"] == "model_turn":
            calls = {c.get("id"): c.get("name") for c in t.get("tool_calls") or [] if isinstance(c, dict)}
        elif t["kind"] in ("tool_call", "finish") and calls.get(t.get("tool_use_id")) != t.get("tool"):
            problems.append(f"trace step {n}: no model turn asked for this {t.get('tool')!r} call")
    stops = [t for t in trace if t.get("kind") == "stop"]
    if len(stops) != 1 or trace[-1].get("kind") != "stop":
        problems.append("the trace must end with exactly one stop event")
    return problems


def _budget_problems(doc):
    b = doc["budget"]
    names = {f.name for f in fields(ia.Budget)}
    if not isinstance(b, dict) or set(b) != names:
        return ["the recorded budget is not a Budget"], None
    try:
        budget = ia.Budget(**{**b, "price_per_mtok": tuple(b["price_per_mtok"]) if b["price_per_mtok"] else None,
                              "pre_window_grid_frames": tuple(b["pre_window_grid_frames"])
                              if b["pre_window_grid_frames"] else None})
    except (TypeError, ValueError) as exc:
        return [f"the recorded budget is invalid ({exc})"], None
    return [], budget


def _data_problems(doc):
    d, problems = doc["data"], []
    if not isinstance(d, dict) or set(d) != DATA_KEYS or d["data_source"] not in v1.DATA_SOURCES or \
            not isinstance(d["sessions"], list) or not all(isinstance(x, str) for x in d["sessions"]):
        return ["the recorded data source is malformed"]
    sessions = set(d["sessions"])
    if doc["runner_record"].get("data_source") != d["data_source"]:
        problems.append("the runner record names another data source")
    for e in doc["evidence"]:
        r, req = e["result"], e["request"]
        if (r.get("data_source"), r.get("derived_from")) != (d["data_source"], d["derived_from"]):
            problems.append(f"evidence {e['evidence_id'][:40]} was computed on another data source")
        named = req.get("session")
        named = named if isinstance(named, list) else [named] if named is not None else \
            [b.get("session") for b in req.get("bouts", [])]
        if not set(named) <= sessions:
            problems.append(f"evidence {e['evidence_id'][:40]} names sessions outside the recorded data")
        if e["tool"] == "query_events" and r.get("query_run_id") != \
                "inv_" + v1._sha(v1._canonical(r.get("query")) + d["fingerprint"])[:16]:
            problems.append(f"query {e['evidence_id'][:40]} was not run on the recorded data (query_run_id)")
    return problems


def _evidence_problems(doc):
    problems, evidence = [], {}
    rec, d = doc["runner_record"], doc["data"]
    for n, e in enumerate(doc["evidence"]):
        if not isinstance(e, dict) or set(e) != {"evidence_id", "tool", "status", "request", "result", "provenance"}:
            problems.append(f"evidence[{n}] is malformed")
            continue
        if v1.evidence_id(e["tool"], e["request"], e["result"]) != e["evidence_id"] or \
                e["status"] != e["result"].get("status") or e["result"].get("status") == "error":
            problems.append(f"evidence {str(e.get('evidence_id'))[:60]} does not match its content")
        if e["tool"] == "compare_windows":
            p = v1.comparison_problems(e["request"], e["result"])
            if p:                                   # contradictory: it cannot support a finding
                problems += [f"comparison {str(e['evidence_id'])[:60]}: {x}" for x in p]
                continue
        evidence[e["evidence_id"]] = e
    executed = [t for t in doc["trace"] if t.get("kind") == "tool_call" and "evidence_id" in t]
    listed = [e for e in doc["evidence"] if isinstance(e, dict)]
    if [t["evidence_id"] for t in executed] != [e.get("evidence_id") for e in listed] or \
            [t.get("tool") for t in executed] != [e.get("tool") for e in listed]:
        problems.append("the trace's executed calls are not the evidence, in order")
    for t, e in zip(executed, listed):
        want = {"contract_version": v1.CONTRACT_VERSION, "data_source": d.get("data_source"),
                "derived_from": d.get("derived_from"), "data_fingerprint": d.get("fingerprint"),
                "split": "synthetic" if d.get("data_source") == "synthetic" else "validation",
                "investigation_id": rec.get("investigation_id"), "call": t.get("call")}
        if e["provenance"] != want:
            problems.append(f"evidence {e['evidence_id'][:40]} has another provenance than its execution")
    return problems, evidence


def _runner_problems(doc, budget):
    rec, problems = doc["runner_record"], []
    if not isinstance(rec, dict) or set(rec) != RUNNER_KEYS or not isinstance(rec["attempts"], list):
        return ["the runner record is malformed"]
    reached = [t for t in doc["trace"] if _reaches_runner(t)]
    by_id = {e["evidence_id"]: e for e in doc["evidence"]}
    if rec["contract_version"] != v1.CONTRACT_VERSION or rec["max_tool_calls"] != budget.max_tool_calls or \
            rec["calls_used"] != len(rec["attempts"]) or len(rec["attempts"]) != len(reached):
        return ["the runner record does not match the trace's tool executions (attempts, count or budget)"]
    for n, (a, t) in enumerate(zip(rec["attempts"], reached), start=1):
        if not isinstance(a, dict) or a.get("call") != n or a.get("tool") != t.get("tool") or \
                a.get("investigation_id") != rec["investigation_id"] or a.get("counted") is not True:
            problems.append(f"runner attempt {n} is not the trace's call {n}")
            continue
        if "evidence_id" in t:
            e = by_id.get(t["evidence_id"], {})
            if t.get("call") != n or a.get("status") != e.get("status") or a.get("request") != e.get("request") or \
                    a.get("result") != e.get("result") or a.get("result_sha256") != v1._sha(
                        json.dumps(e.get("result"), sort_keys=True, default=str)):
                problems.append(f"runner attempt {n} is not the evidence it produced")
        elif t.get("refused") == "tool_failure":
            if a.get("status") != "exception":
                problems.append(f"runner attempt {n} is not the failed call")
        elif a.get("status") != "error" or "tool_refused:" + str(a.get("code")) != t.get("refused"):
            problems.append(f"runner attempt {n} is not the refused call")
    return problems


def _replay_problems(doc, budget):
    """Every model call of the trace decided again, in order, by the live admission (_Ledger.admit) from its
    recorded arguments, with only the evidence accepted before it. A refusal must be recorded as that refusal
    (kind and message); an admitted call must have reached the runner, whose attempt and (if accepted) whose
    evidence must carry exactly the request those arguments expand to; an admitted finish must be recorded as
    accepted. So saved arguments are bound to what was executed, and later evidence cannot justify an earlier
    call. Comparisons that reached the runner are also replayed through the runner's own fixed-follow-up rule
    (investigation_tools_v1.FollowUps): a follow-up it refuses must be recorded as that refusal, never as evidence."""
    problems, ledger, follow_ups = [], _Ledger(doc["data"]["sessions"], budget), v1.FollowUps()
    by_id = {e["evidence_id"]: e for e in doc["evidence"]}
    attempts = iter(doc["runner_record"]["attempts"])
    for t in doc["trace"]:
        if t["kind"] not in ("tool_call", "finish"):
            continue
        n, name = t["step"], t.get("tool")
        if (t["kind"] == "finish") != (name == FINISH):
            problems.append(f"trace step {n}: a {t['kind']} event cannot be a {str(name)[:40]!r} call")
            continue
        try:
            request, key = ledger.admit(name, copy.deepcopy(t.get("arguments")))
        except ia._Rejected as exc:
            if (t.get("outcome"), t.get("refused"), t.get("message")) != (
                    "refused", exc.kind, str(exc)[:ia.MAX_TEXT_TRACE]):
                problems.append(f"trace step {n}: its arguments are refused by the live checks ({exc.kind}: "
                                f"{str(exc)[:200]}), but the trace does not record that refusal")
            continue
        if name == FINISH:
            if t.get("outcome") != "accepted" or "refused" in t:
                problems.append(f"trace step {n}: the live checks accept this finish, but it is recorded as refused")
            continue
        if not _reaches_runner(t):
            problems.append(f"trace step {n}: the live checks admit this call, but it is recorded as refused "
                            "before execution")
            continue
        ledger.tool_calls += 1
        attempt = next(attempts, None)
        if not isinstance(attempt, dict) or attempt.get("tool") != name or attempt.get("request") != request:
            problems.append(f"trace step {n}: the runner attempt is not the request its arguments produce")
        refusal = follow_ups.problem(request) if name == "compare_windows" else None
        if refusal is not None:
            result = attempt.get("result") if isinstance(attempt, dict) else None
            if t.get("refused") != "tool_refused:invalid_request" or not isinstance(result, dict) or \
                    (result.get("status"), result.get("code"), result.get("message")) != (
                        "error", "invalid_request", refusal[:500]):
                problems.append(f"trace step {n}: the runner's follow-up rule refuses this comparison "
                                f"({refusal[:200]}), but the trace does not record that refusal")
            continue
        if "evidence_id" in t:
            e = by_id.get(t["evidence_id"])
            if e is None or e["tool"] != name or e["request"] != request or t.get("outcome") != e["status"]:
                problems.append(f"trace step {n}: the accepted evidence is not the request its arguments produce")
                continue
            ledger.accept(name, key, t["evidence_id"], e)
            if name == "compare_windows":
                follow_ups.ran(request)
    return problems


def _usage_problems(doc, budget):
    u, trace, problems = doc["usage"], doc["trace"], []
    if not isinstance(u, dict) or set(u) != USAGE_KEYS:
        return ["the recorded usage is malformed"]
    turns = [t for t in trace if t.get("kind") == "model_turn"]
    tokens_in = sum(_tokens(t.get("input_tokens")) for t in turns)
    tokens_out = sum(_tokens(t.get("output_tokens")) for t in turns)
    unreported = sum(1 for t in turns if _tokens(t.get("input_tokens")) != t.get("input_tokens") or
                     _tokens(t.get("output_tokens")) != t.get("output_tokens"))
    cost = None if budget.price_per_mtok is None else \
        (tokens_in * budget.price_per_mtok[0] + tokens_out * budget.price_per_mtok[1]) / 1e6
    want = {"model_turns": len(turns), "tool_calls": len([t for t in trace if _reaches_runner(t)]),
            "repeated_calls": sum(1 for t in trace if t.get("refused") == "repeated_call"),
            "input_tokens": tokens_in, "output_tokens": tokens_out, "turns_without_usage": unreported,
            "cost_usd": cost}
    wrong = [k for k, v in want.items() if u.get(k) != v]
    if wrong:
        problems.append(f"the recorded usage contradicts the trace in {', '.join(wrong)}")
    if not isinstance(u.get("elapsed_s"), (int, float)) or u["elapsed_s"] < trace[-1].get("elapsed_s", 0):
        problems.append("the recorded elapsed time is before the trace's stop")
    if want["model_turns"] > budget.max_model_turns or want["tool_calls"] > budget.max_tool_calls:
        problems.append("the trace exceeds the recorded budget")
    comparisons = sum(1 for e in doc["evidence"] if e["tool"] == "compare_windows")
    if comparisons > 1 + follow_up_limit(budget):
        problems.append("more comparisons were executed than the follow-up limit allows")
    return problems


def _completion_problems(doc, evidence):
    """Completion established from the trace itself, independently of any fingerprint."""
    trace, problems = doc["trace"], []
    stop = trace[-1]
    accepted = [t for t in trace if t.get("kind") == "finish" and t.get("outcome") == "accepted"]
    if stop.get("stopping_reason") != doc["stopping_reason"] or \
            doc["finished"] != (doc["stopping_reason"] == "finished"):
        problems.append("the report's stop is not the trace's stop")
    if doc["finding"] not in FINDINGS + ("incomplete",):
        problems.append(f"unknown finding {str(doc['finding'])[:40]!r}")
        return problems
    if not doc["finished"]:
        if accepted or doc["finding"] != "incomplete" or doc["primary_evidence_id"] is not None or \
                doc["observations"] or doc["finish_fingerprint"] is not None:
            problems.append("an unfinished run must be 'incomplete', with no accepted finish, finding or observation")
        return problems
    if len(accepted) != 1 or len(trace) < 2 or trace[-2] is not accepted[0]:
        problems.append("a finished run needs exactly one accepted finish, immediately before its stop (nothing may "
                        "be executed after it)")
        return problems
    args = accepted[0].get("arguments") or {}
    refs = [{"evidence_id": o.get("evidence_id"), "field": o.get("field")} for o in doc["observations"]]
    if not isinstance(args, dict) or doc["finding"] not in FINDINGS or args.get("finding") != doc["finding"] or \
            args.get("primary_evidence_id") != doc["primary_evidence_id"] or args.get("observations") != refs:
        problems.append("the report's finding is not the accepted finish's arguments")
    proposal = {"finding": doc["finding"], "primary_evidence_id": doc["primary_evidence_id"], "observations": refs}
    p, observations = _finding_problems(proposal, evidence)
    problems += p
    if observations != doc["observations"]:
        problems.append("the observed values are not the values in the cited results")
    if doc["finish_fingerprint"] != _fingerprint(doc["question"], doc["finding"], doc["primary_evidence_id"],
                                                 observations, doc["evidence"], trace[:-1]):
        problems.append("the finish fingerprint does not match the accepted finish")
    return problems


def verify_report_v1(doc):
    """Re-derive a saved report (InvestigationReportV1.to_json()) from its own content, field by field: the fixed
    schema and canonical disclosures; every evidence id; the trace's structure and completion; the runner's
    attempts against the trace; the usage against the trace and budget; the data source against every result and
    query; the finding, its observed values and its finish fingerprint; then the whole-report fingerprint. A
    matching fingerprint alone proves nothing (anyone can recompute it): every field is checked first. Returns the
    problems (empty: it verifies; this is consistency, not authenticity)."""
    if not isinstance(doc, dict) or set(doc) != REPORT_KEYS:
        extra = sorted(set(doc) - REPORT_KEYS) if isinstance(doc, dict) else []
        return [f"not a contract-v1/1 report: fields {extra or 'missing'}"]
    problems = []
    try:
        if doc["findings_format"] != FINDINGS_FORMAT or doc["publication"] != PUBLICATION:
            problems.append("not a contract-v1/1 report, or its publication status is not 'not_authorized'")
        if doc["disclosures"] != list(DISCLOSURES):
            problems.append("the disclosures are not the canonical disclosures")
        if not isinstance(doc["question"], str) or not doc["question"].strip() or len(doc["question"]) > 500:
            problems.append("the question is malformed")
        p, budget = _budget_problems(doc)
        problems += p
        tp = _trace_problems(doc["trace"])
        problems += tp
        if budget is None or tp:
            return problems
        p, evidence = _evidence_problems(doc)
        problems += p + _runner_problems(doc, budget) + _usage_problems(doc, budget) + _data_problems(doc)
        problems += _replay_problems(doc, budget)
        problems += _completion_problems(doc, evidence)
        if doc["fingerprint"] != _content_fingerprint(doc):
            problems.append("the fingerprint does not match the report's content")
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        problems.append(f"the report is malformed ({type(exc).__name__})")
    return problems


FINDING_TEXT = {
    "effect_described": "The primary comparison met Contract v1's evidence gate; its effect is described below "
                        "(association only).",
    "insufficient_evidence": "The primary comparison did not meet Contract v1's evidence gate: no effect is reported.",
    "inconclusive": "The evidence gathered does not support a finding.",
    "incomplete": "The investigation stopped before an accepted final answer; the evidence listed is partial and no "
                  "finding was made.",
}


def render_markdown_v1(doc):
    """Markdown from a saved report, only after it verifies (ReportVerificationError otherwise). Every value shown
    is a recorded, verified value; the sentences are fixed templates."""
    problems = verify_report_v1(doc)
    if problems or doc["finding"] not in FINDING_TEXT:
        raise ReportVerificationError("; ".join(problems) or "unsupported finding")
    lines = [f"# Investigation report ({FINDINGS_FORMAT})", "",
             f"**Researcher's question (as asked, not a finding):** `{doc['question']}`", "",
             f"**Finding:** `{doc['finding']}`. {FINDING_TEXT[doc['finding']]}",
             f"**Stopped:** {doc['stopping_reason']}. **Publication:** {doc['publication']}. "
             f"**Fingerprint:** `{doc['fingerprint']}`", "",
             f"**Data:** {doc['data'].get('data_source')} ({doc['data'].get('derived_from')}), "
             f"{len(doc['data'].get('sessions', []))} sessions.", "", "## Observations (read from the evidence)", ""]
    if doc["observations"]:
        lines += ["| evidence | field | value |", "|---|---|---|"]
        lines += [f"| `{o['evidence_id']}` | `{o['field']}` | `{json.dumps(o['value'])}` |"
                  for o in doc["observations"]]
    else:
        lines.append("None.")
    lines += ["", "## Evidence (validated Contract v1 results of this run)", "",
              "| evidence | tool | status |", "|---|---|---|"]
    lines += [f"| `{e['evidence_id']}` | {e['tool']} | {e['status']} |" for e in doc["evidence"]]
    lines += ["", "## Trace (every model-chosen action, including refused ones)", "",
              "| step | kind | tool | outcome |", "|---|---|---|---|"]
    for t in doc["trace"]:
        if t.get("kind") in ("tool_call", "finish", "stop", "model_error", "budget_exceeded"):
            outcome = t.get("refused") or t.get("outcome") or t.get("stopping_reason") or t.get("reason") or ""
            lines.append(f"| {t['step']} | {t['kind']} | {t.get('tool') or ''} | {outcome} |")
    lines += ["", "## Method (Contract v1's own definitions)", "",
              f"- Windows: pre {v1.WINDOW_DEFINITIONS['pre']}; baseline {v1.WINDOW_DEFINITIONS['baseline']}; "
              f"anchored on {v1.WINDOW_DEFINITIONS['anchor']}.",
              f"- Effect: {v1.MEAN_DIFF_DEFINITION}.",
              f"- Quality policy {v1.DEFAULT_QUALITY_POLICY}: "
              f"{v1.QUALITY_POLICIES[v1.DEFAULT_QUALITY_POLICY].to_json()}.",
              f"- Overlap policy {v1.DEFAULT_OVERLAP_POLICY}: "
              f"{v1.OVERLAP_POLICIES[v1.DEFAULT_OVERLAP_POLICY].to_json()}.",
              f"- Evidence gate: at least {v1.DEFAULT_MIN_N_WINDOWS} usable windows in at least "
              f"{v1.DEFAULT_MIN_N_SESSIONS} sessions.", "", "## Disclosures", ""]
    lines += [f"- {d}" for d in DISCLOSURES]              # the canonical text, never the report's copy
    return "\n".join(lines) + "\n"
