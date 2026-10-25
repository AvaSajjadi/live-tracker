"""Researcher rule approval workflow, without any UI (new file).

pages/1_Rules.py only renders this; all state transitions live here so they
can be tested without a browser. Plain Python, no Streamlit import.

States of one instruction
-------------------------
    submit(text) --> RuleProposal --> pending (shown for review, NOT in the RuleSet)
                                        |-- confirm(token) -> RuleSet.add + RuleSet.approve -> active
                                        |-- reject(token)  -> RuleSet.add + RuleSet.reject  -> inactive
                                        |-- text changed / new submit -> discarded (never added)
                 --> Clarification / ClaudeUnavailable --> shown, nothing to approve

- Only RuleSet.approve() makes a rule active (RuleSet.active() is what the
  monitor uses). A proposal is added to the RuleSet only when the researcher
  decides, so an undecided or stale proposal can never be active.
- Every pending proposal has a fresh token. confirm / reject must name it, so
  a click on a button rendered for an older proposal, a second click on the
  same button, or a click after the instruction changed is refused
  (StaleActionError) instead of acting on the wrong proposal.
- Rule IDs (R1, R2, ...) are never reused within a workflow, so decided and
  pending rules can never collide.
- Confirming a rule that is identical (behavior, duration, mode, thresholds)
  to an already approved rule is refused: it would only duplicate every event.
- Decisions are atomic. RuleSet has no delete, and approve() / reject()
  change Rule objects in place, so a decision is staged on a deep copy of the
  RuleSet (add + approve/reject) and checked against the exact expected
  state: the original rules, unchanged in every field, in the same order,
  plus the validated proposal with only its status changed. The workflow's
  `ruleset` reference is swapped to the copy only when everything matches. If any
  step raises, the live RuleSet is untouched (no inserted rule, nothing
  approved), the proposal stays pending with the same token, and
  DecisionError is raised so the caller can show it and the researcher can
  retry. Callers must therefore read `workflow.ruleset` each time instead of
  keeping an old reference.
- Backend failures never escape submit(): an unexpected exception, or a
  result of the wrong type / not a fresh proposal, becomes a
  ClaudeUnavailable("api_error") for the Claude backend or a BackendFailure
  for the offline parser, with only the exception's class name (never its
  text). Pending and last-result state are cleared before the backend runs.

Backends: "offline" (agent.parse_rule, the default) and "claude"
(claude_agent.propose_rule_with_claude, an optional cross-check that accepts
at most what the offline parser accepts).
"""

from __future__ import annotations

import copy
import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Optional, Union

from . import agent
from .agent import Clarification, RuleProposal, parse_rule
from .claude_agent import ClaudeUnavailable, propose_rule_with_claude
from .claude_interpreter import interpret_rule_with_claude
from .flex_reader import read_flexibly
from .rules import CLASSIFIER_BEHAVIORS, STATUSES, Rule, RuleSet, validate_rule

__all__ = ["BACKENDS", "BackendFailure", "DecisionError", "DuplicateRuleError", "Pending",
           "RuleWorkflow", "StaleActionError", "describe_rule", "is_valid_workflow",
           "monitoring_allowed", "preview"]

BACKENDS = ("offline", "claude", "claude_interpret")
_BACKEND_NAMES = {"offline": "offline parser", "claude": "Claude cross-check", "claude_interpret": "Claude reading"}
_DECISION_FIELDS = ("behavior", "min_duration_s", "alert_mode", "alert_threshold", "ask_threshold")
_DEFAULTS = {k: Rule.__dataclass_fields__[k].default for k in ("alert_threshold", "ask_threshold")}

MODE_TEXT = {
    "alert": "alert: you are alerted when the event's confidence reaches the alert threshold; "
             "below it you are asked to confirm",
    "ask": "ask: you are asked about each event (the exact meaning of this mode is still "
           "being agreed by the team)",
    "log": "log: events are only recorded, you are not interrupted",
}


class StaleActionError(RuntimeError):
    """confirm / reject named a proposal that is no longer pending."""


class DuplicateRuleError(ValueError):
    """The proposal is identical to an already approved rule."""


class DecisionError(RuntimeError):
    """Confirm / Reject failed part-way; nothing was changed and the proposal
    is still pending. The message is safe to show (no raw exception text)."""


@dataclass(frozen=True)
class BackendFailure:
    """The offline parser failed unexpectedly (a bug, not a researcher error).
    Nothing was proposed. (Claude failures use claude_agent.ClaudeUnavailable.)"""
    backend: str
    message: str  # safe to show: the exception's class name only


@dataclass(frozen=True)
class Pending:
    """A proposal shown to the researcher, waiting for Confirm or Reject."""
    token: str
    instruction: str      # exactly the text that was submitted
    backend: str
    proposal: RuleProposal


Result = Union[RuleProposal, Clarification, ClaudeUnavailable, BackendFailure]


def _number_text(x):
    """2.0 -> '2', 0.85 -> '0.85', no exponent."""
    text = format(Decimal(repr(float(x))), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _named_thresholds(instruction):
    """Thresholds the researcher named in the text (same grammar item as the parser)."""
    t = re.sub(r"\s+", " ", instruction.strip().lower())
    return {f"{m['which']}_threshold" for m in agent._G_THRESHOLD_ITEM.finditer(t)}


def preview(pending: Pending) -> dict:
    """Everything the researcher must see before confirming, as plain data."""
    rule = pending.proposal.rule_dict
    named = _named_thresholds(pending.instruction)
    seconds = _number_text(rule["min_duration_s"])

    def threshold(key):
        value = _number_text(rule[key])
        if key in named:
            return f"{value} (from your instruction)"
        return f"{value} (untuned example default from rules.py, not chosen on data)"

    return {
        "rule_id": rule["rule_id"],
        "status": "proposed: NOT active until you click Confirm",
        "behavior": rule["behavior"],
        "min_duration": f"at least {seconds} {'second' if seconds == '1' else 'seconds'}",
        "alert_mode": MODE_TEXT[rule["alert_mode"]],
        "alert_threshold": threshold("alert_threshold"),
        "ask_threshold": threshold("ask_threshold"),
        "uses_default_thresholds": sorted({"alert_threshold", "ask_threshold"} - named),
        "assumptions": list(pending.proposal.assumptions),
        "instruction": pending.instruction,
        "backend": pending.backend,
        "rule_json": pending.proposal.rule_data(),
    }


def describe_rule(rule: Rule) -> dict:
    """One row of the rules table."""
    status = {"approved": "approved (active)", "rejected": "rejected (inactive)",
              "proposed": "proposed (inactive)"}[rule.status]
    seconds = _number_text(rule.min_duration_s)
    return {"rule_id": rule.rule_id, "status": status, "behavior": rule.behavior,
            "min_duration": f"at least {seconds} {'second' if seconds == '1' else 'seconds'}",
            "alert_mode": rule.alert_mode, "alert_threshold": rule.alert_threshold,
            "ask_threshold": rule.ask_threshold, "instruction": rule.description}


def monitoring_allowed(ruleset: RuleSet) -> bool:
    """Monitoring may only start with at least one approved rule."""
    return len(ruleset.active()) > 0


_RULE_FIELDS = tuple(Rule.__dataclass_fields__)
_NUMBER_FIELDS = ("min_duration_s", "alert_threshold", "ask_threshold")
_TOKEN = re.compile(r"[0-9a-f]{32}")
_RULE_ID = re.compile(r"R([1-9][0-9]*)")


def _rule_fields(rule):
    """A stored Rule's field values read from its INSTANCE attributes only.
    getattr / asdict / to_dict would silently fall back to the dataclass's
    class-level defaults for a deleted attribute (a deleted alert_threshold of
    0.95 would read as 0.8), so every Rule field must be present in vars(rule)
    and nothing else may be. Raises otherwise."""
    if type(rule) is not Rule:
        raise ValueError("not a Rule")
    inst = vars(rule)
    if type(inst) is not dict or set(inst) != set(_RULE_FIELDS):
        raise ValueError("Rule instance attributes do not match the Rule fields")
    return {k: inst[k] for k in _RULE_FIELDS}


def _rule_snapshot(rule_id, rule):
    """Exact, typed contents of one stored rule (every Rule field, with its
    type, so 2 vs 2.0 or "attack" vs "mount" both differ). Raises if it is not
    a complete Rule stored under its own id."""
    fields = _rule_fields(rule)
    if not isinstance(rule_id, str) or fields["rule_id"] != rule_id:
        raise ValueError("Rule not stored under its own rule_id")
    return tuple((k, type(v), v) for k, v in fields.items())


def _ruleset_snapshot(ruleset):
    """Ordered, exact snapshot of a whole RuleSet; raises if malformed."""
    if type(ruleset) is not RuleSet or type(ruleset.rules) is not dict:
        raise ValueError("not a RuleSet with a plain dict of rules")
    return [(rid, _rule_snapshot(rid, r)) for rid, r in ruleset.rules.items()]


def _check_rule_data(data, statuses):
    """A rule dict (or Rule.to_dict()) must be exactly what rules.validate_rule
    produces, with finite numbers, a classifier behavior and an allowed status.
    Raises on any problem; never modifies data."""
    if type(data) is not dict or set(data) != set(_RULE_FIELDS):
        raise ValueError("rule fields do not match the Rule schema")
    if data["status"] not in statuses:
        raise ValueError("unexpected status")
    checked, errors = validate_rule({**data, "status": "proposed"})
    if errors:
        raise ValueError("rule fails rules.validate_rule")
    if checked.to_dict() != {**data, "status": "proposed"}:
        raise ValueError("rule is not in validate_rule's normalized form")
    if any(type(data[k]) is not float or not math.isfinite(data[k]) for k in _NUMBER_FIELDS):
        raise ValueError("rule numbers must be finite floats")
    if data["behavior"] not in CLASSIFIER_BEHAVIORS:
        raise ValueError("behavior has no classifier score")
    if not _RULE_ID.fullmatch(data["rule_id"]):
        raise ValueError("rule_id is not of the form R<n>")


def _check_workflow(obj):
    """Raise ValueError (or anything) if obj is not a consistent RuleWorkflow."""
    if type(obj) is not RuleWorkflow:
        raise ValueError("not a RuleWorkflow")
    state = vars(obj)  # instance attributes only; a missing one fails here
    for name in ("ruleset", "pending", "last_result", "last_backend", "notice", "_next_id",
                 "claude_fn", "offline_fn", "interpret_fn"):
        if name not in state:
            raise ValueError(f"missing attribute {name}")
    _ruleset_snapshot(state["ruleset"])  # structure: Rule objects under their own ids
    rules = state["ruleset"].rules
    for rule in rules.values():
        _check_rule_data(_rule_fields(rule), STATUSES)  # instance values, never class defaults
    next_id = state["_next_id"]
    if type(next_id) is not int or next_id < 1:  # (allocation itself skips ids in use)
        raise ValueError("_next_id must be a positive int")
    pending = state["pending"]
    if pending is not None:
        if type(pending) is not Pending:
            raise ValueError("pending is not a Pending")
        if not (isinstance(pending.token, str) and _TOKEN.fullmatch(pending.token)):
            raise ValueError("pending token is malformed")
        if not isinstance(pending.instruction, str) or pending.backend not in BACKENDS:
            raise ValueError("pending instruction / backend is malformed")
        proposal = pending.proposal
        if type(proposal) is not RuleProposal:
            raise ValueError("pending proposal is not a RuleProposal")
        if not (isinstance(proposal.source_text, str) and type(proposal.assumptions) is tuple
                and all(isinstance(a, str) for a in proposal.assumptions)):
            raise ValueError("pending proposal text / assumptions are malformed")
        # must be a real mapping (read-only ones are fine): preview() indexes it
        # by key, and dict() would also accept a list of pairs
        if not isinstance(proposal.rule_dict, Mapping):
            raise ValueError("pending rule_dict is not a mapping")
        data = dict(proposal.rule_dict)  # a copy: validation never modifies the proposal
        _check_rule_data(data, {"proposed"})
        if data["rule_id"] in rules or int(_RULE_ID.fullmatch(data["rule_id"]).group(1)) >= next_id:
            raise ValueError("pending rule_id collides with a decided rule or the id counter")
    if state["last_backend"] is not None and state["last_backend"] not in BACKENDS:
        raise ValueError("last_backend is malformed")
    if state["last_result"] is not None and not isinstance(
            state["last_result"], (RuleProposal, Clarification, ClaudeUnavailable, BackendFailure)):
        raise ValueError("last_result is malformed")
    if state["notice"] is not None and not isinstance(state["notice"], str):
        raise ValueError("notice is malformed")
    if not (callable(state["claude_fn"]) and callable(state["offline_fn"]) and callable(state["interpret_fn"])):
        raise ValueError("backend functions are not callable")


def is_valid_workflow(obj) -> bool:
    """True only if obj (e.g. a value restored from Streamlit session state) is
    a fully consistent RuleWorkflow: every stored rule passes the shared rule
    contract with an allowed status, ids match and are never reused, and any
    pending proposal is a well-formed, unapproved, validated rule. Never raises
    and never modifies obj; anything else must be replaced, not read as rules."""
    try:
        _check_workflow(obj)
    except Exception:
        return False
    return True


@dataclass
class RuleWorkflow:
    """Per-session state: the RuleSet plus at most one pending proposal."""
    ruleset: RuleSet = field(default_factory=RuleSet)
    pending: Optional[Pending] = None
    last_result: Optional[Result] = None     # what the last submit returned
    last_backend: Optional[str] = None
    notice: Optional[str] = None             # e.g. why a pending proposal was discarded
    _next_id: int = 1
    claude_fn: Callable[..., Result] = propose_rule_with_claude
    offline_fn: Callable[..., Result] = read_flexibly
    interpret_fn: Callable[..., Result] = interpret_rule_with_claude

    # ---- ids ----

    def _new_rule_id(self):
        while f"R{self._next_id}" in self.ruleset.rules:
            self._next_id += 1
        rule_id = f"R{self._next_id}"
        self._next_id += 1  # never reused, even if this proposal is discarded
        return rule_id

    # ---- transitions ----

    def submit(self, text: str, backend: str = "offline", **claude_kwargs) -> Result:
        """Parse one instruction. Replaces (discards) any pending proposal."""
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # nothing from an earlier submission may survive a failure below
        self.pending = None
        self.last_result = None
        self.last_backend = None
        self.notice = None
        rule_id = self._new_rule_id()
        try:
            if backend == "claude":
                result = self.claude_fn(text, rule_id, **claude_kwargs)
            elif backend == "claude_interpret":
                result = self.interpret_fn(text, rule_id, **claude_kwargs)
            else:
                result = self.offline_fn(text, rule_id)
            problem = self._check_result(result, rule_id)
        except Exception as exc:  # any backend bug; its text may hold secrets
            problem = f"it raised {type(exc).__name__}"
        if problem is not None:
            result = self._backend_failure(backend, problem)
        self.last_result, self.last_backend = result, backend
        if isinstance(result, RuleProposal):
            self.pending = Pending(uuid.uuid4().hex, text, backend, result)
        return result

    @staticmethod
    def _check_result(result, rule_id):
        """None if the backend returned a usable result, else what is wrong."""
        if isinstance(result, (Clarification, ClaudeUnavailable, BackendFailure)):
            return None
        if not isinstance(result, RuleProposal):
            return f"it returned an unexpected {type(result).__name__}"
        if result.rule_dict.get("status") != "proposed" or result.rule_dict.get("rule_id") != rule_id:
            return "it returned a rule that is not a fresh, unapproved proposal"
        return None

    @staticmethod
    def _backend_failure(backend, problem):
        text = (f"The {_BACKEND_NAMES[backend]} failed unexpectedly ({problem}). Nothing was proposed.")
        if backend in ("claude", "claude_interpret"):
            return ClaudeUnavailable("api_error", text + " Use the offline parser or try again later.")
        return BackendFailure(backend, text + " Please report this; your approved rules are unchanged.")

    def sync_instruction(self, current_text: str) -> bool:
        """Call on every rerun with the text box's current value. If it no
        longer equals the submitted instruction, the pending proposal is
        discarded (its Confirm can no longer act). Returns True if discarded."""
        if self.pending is not None and current_text != self.pending.instruction:
            self.pending = None
            self.last_result = None
            self.notice = ("The instruction was changed after the rule was proposed, so that "
                           "proposal was discarded. Click Propose rule again.")
            return True
        return False

    def _take(self, token):
        if self.pending is None or token != self.pending.token:
            raise StaleActionError("this proposal is no longer pending (already decided, replaced "
                                   "or its instruction changed)")
        return self.pending

    def _decide(self, token, approve):
        """Stage add + approve/reject on a copy; swap it in only if all succeeded."""
        pending = self._take(token)
        data = pending.proposal.rule_data()
        verb, status = ("Confirm", "approved") if approve else ("Reject", "rejected")
        if approve:
            same = [r.rule_id for r in self.ruleset.active()
                    if all(getattr(r, k) == data[k] for k in _DECISION_FIELDS)]
            if same:
                raise DuplicateRuleError(f"identical to approved rule {same[0]}; confirming it "
                                         "again would duplicate every event")
        live = self.ruleset
        try:
            # the exact state this decision must produce: every existing rule
            # unchanged, plus the validated proposal with only its status changed
            expected_rule, errors = validate_rule(data)
            if errors or expected_rule.to_dict() != data:
                raise ValueError("the pending proposal no longer validates")
            expected_rule.status = status
            expected = (_ruleset_snapshot(live)
                        + [(expected_rule.rule_id, _rule_snapshot(expected_rule.rule_id, expected_rule))])

            staged = copy.deepcopy(live)
            rule = staged.add(data)                  # RuleSet.add validates again
            if approve:
                staged.approve(rule.rule_id)         # the only way a rule becomes active
            else:
                staged.reject(rule.rule_id)
            # every rule, every field (with types), the order, nothing added or removed
            if _ruleset_snapshot(staged) != expected:
                raise RuntimeError("the staged RuleSet is not in the expected state")
            rule = staged.rules[expected_rule.rule_id]
        except Exception as exc:  # the live RuleSet and the pending proposal are untouched
            self.notice = None
            raise DecisionError(f"{verb} failed ({type(exc).__name__}). Nothing was changed and "
                                "the proposal is still pending; you can try again.") from exc
        self.ruleset = staged                        # commit: one reference swap
        self.pending = None
        self.last_result = None
        self.notice = (f"Rule {rule.rule_id} approved. It is active for monitoring." if approve
                       else f"Rule {rule.rule_id} rejected. It will not be used.")
        return staged.rules[rule.rule_id]

    def confirm(self, token: str) -> Rule:
        """The researcher's explicit Confirm: add the pending rule and approve it
        (atomically; DecisionError if any step fails)."""
        return self._decide(token, approve=True)

    def reject(self, token: str) -> Rule:
        """The researcher's explicit Reject: record the rule as rejected
        (atomically; DecisionError if any step fails)."""
        return self._decide(token, approve=False)

    # ---- views ----

    def rules_table(self) -> list[dict]:
        return [describe_rule(r) for r in self.ruleset.rules.values()]

    def monitoring_allowed(self) -> bool:
        return monitoring_allowed(self.ruleset)
