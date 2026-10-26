"""Tests for the rule approval workflow (no UI) (new file)."""

from collections.abc import Mapping
from dataclasses import fields as _dc_fields
from dataclasses import replace as _dc_replace
from types import MappingProxyType, SimpleNamespace

import pytest

from live_tracker.agent import Clarification, Issue, RuleProposal, parse_rule
from live_tracker.claude_agent import ClaudeUnavailable
from live_tracker.fake_scores import demo_scores
from live_tracker.monitor import run_monitor
from live_tracker.rule_workflow import (BackendFailure, DecisionError, DuplicateRuleError, RuleWorkflow,
                                        StaleActionError, describe_rule, is_valid_workflow,
                                        monitoring_allowed, preview)
from live_tracker.rules import Rule, RuleSet

EXAMPLE = "Alert me when the mice attack for at least 2 seconds"


def submitted(text=EXAMPLE, **kw):
    wf = RuleWorkflow(**kw)
    result = wf.submit(text)
    assert isinstance(result, RuleProposal), result
    return wf


# ---- preview accuracy ----

def test_preview_shows_every_field():
    wf = submitted()
    p = preview(wf.pending)
    assert p["rule_id"] == "R1"
    assert p["behavior"] == "attack"
    assert p["min_duration"] == "at least 2 seconds"
    assert p["alert_mode"].startswith("alert:")
    assert p["alert_threshold"].startswith("0.8 (untuned example default")
    assert p["ask_threshold"].startswith("0.5 (untuned example default")
    assert p["uses_default_thresholds"] == ["alert_threshold", "ask_threshold"]
    assert p["assumptions"] == list(parse_rule(EXAMPLE, "R1").assumptions)
    assert p["instruction"] == EXAMPLE
    assert "NOT active" in p["status"]
    assert p["rule_json"] == parse_rule(EXAMPLE, "R1").rule_data()
    assert p["backend"] == "offline"


@pytest.mark.parametrize("text, duration, mode, alert, ask, defaults", [
    ("Tell me if mounting lasts at least 1 minute", "at least 60 seconds", "alert:", "0.8 (untuned",
     "0.5 (untuned", ["alert_threshold", "ask_threshold"]),
    ("Alert me if they attack for a second", "at least 1 second", "alert:", "0.8 (untuned",
     "0.5 (untuned", ["alert_threshold", "ask_threshold"]),
    ("Log investigation bouts of 2.5 s or more", "at least 2.5 seconds", "log:", "0.8 (untuned",
     "0.5 (untuned", ["alert_threshold", "ask_threshold"]),
    ("Ask me when they mount for 2 seconds with alert threshold 0.9", "at least 2 seconds", "ask:",
     "0.9 (from your instruction)", "0.5 (untuned", ["ask_threshold"]),
    ("Alert me if they attack for 2 s with alert threshold 0.8 and ask threshold 60%",
     "at least 2 seconds", "alert:", "0.8 (from your instruction)", "0.6 (from your instruction)", []),
])
def test_preview_variants(text, duration, mode, alert, ask, defaults):
    p = preview(submitted(text).pending)
    assert p["min_duration"] == duration
    assert p["alert_mode"].startswith(mode)
    assert p["alert_threshold"].startswith(alert)
    assert p["ask_threshold"].startswith(ask)
    assert p["uses_default_thresholds"] == defaults


def test_preview_flags_implicit_minimum_and_provisional_ask():
    p = preview(submitted("Ask me when they mount for 2 seconds").pending)
    assert "Interpreted duration as at least 2 seconds." in p["assumptions"]
    assert any("provisional" in a for a in p["assumptions"])
    assert "still being agreed" in p["alert_mode"]


def test_explicit_default_value_is_not_called_a_default():
    """0.8 typed by the researcher is theirs, even though it equals the default."""
    p = preview(submitted(EXAMPLE + " with alert threshold 0.8").pending)
    assert p["alert_threshold"] == "0.8 (from your instruction)"
    assert p["uses_default_thresholds"] == ["ask_threshold"]


# ---- approval and rejection ----

def test_proposal_is_not_in_the_ruleset_until_decided():
    wf = submitted()
    assert wf.ruleset.rules == {} and wf.ruleset.active() == []
    assert not wf.monitoring_allowed()


def test_confirm_approves_through_ruleset():
    wf = submitted()
    rule = wf.confirm(wf.pending.token)
    assert rule.status == "approved" and rule.rule_id == "R1"
    assert [r.rule_id for r in wf.ruleset.active()] == ["R1"]
    assert wf.pending is None and wf.monitoring_allowed()
    assert wf.ruleset.rules["R1"].to_dict() == {**parse_rule(EXAMPLE, "R1").rule_data(),
                                                "status": "approved"}


def test_confirm_uses_ruleset_add_and_approve(monkeypatch):
    calls = []
    real_add, real_approve = RuleSet.add, RuleSet.approve
    monkeypatch.setattr(RuleSet, "add", lambda self, d: calls.append(("add", d["status"])) or real_add(self, d))
    monkeypatch.setattr(RuleSet, "approve", lambda self, rid: calls.append(("approve", rid)) or real_approve(self, rid))
    wf = submitted()
    assert calls == []  # nothing until the decision
    wf.confirm(wf.pending.token)
    assert calls == [("add", "proposed"), ("approve", "R1")]


def test_reject_never_activates():
    wf = submitted()
    rule = wf.reject(wf.pending.token)
    assert rule.status == "rejected" and wf.ruleset.active() == []
    assert not wf.monitoring_allowed()
    assert describe_rule(rule)["status"] == "rejected (inactive)"


def test_nothing_is_monitored_before_approval():
    wf = submitted()
    assert run_monitor(demo_scores(), wf.ruleset, run_id="t") == []
    wf.confirm(wf.pending.token)
    events = run_monitor(demo_scores(), wf.ruleset, run_id="t")
    assert [(e["rule_id"], e["action"]) for e in events] == [("R1", "alert"), ("R1", "ask")]


def test_rejected_rule_is_never_monitored():
    wf = submitted()
    wf.reject(wf.pending.token)
    assert run_monitor(demo_scores(), wf.ruleset, run_id="t") == []


# ---- clarification and backend errors ----

@pytest.mark.parametrize("text", [
    "", "Alert me when the mice attack", "Alert me if they groom for 2 s",     # "fight" is now read as attack (flex_reader)
    "Alert me if they attack for 2 seconds at night", "Log attacks for 2 seconds",
])
def test_clarification_is_never_approvable(text):
    wf = RuleWorkflow()
    result = wf.submit(text)
    assert isinstance(result, Clarification)
    assert wf.pending is None and wf.last_result is result
    with pytest.raises(StaleActionError):
        wf.confirm("anything")
    assert wf.ruleset.rules == {}


def test_claude_unavailable_is_surfaced_and_not_approvable():
    down = ClaudeUnavailable("missing_api_key", "ANTHROPIC_API_KEY is not set")
    wf = RuleWorkflow(claude_fn=lambda text, rule_id, **kw: down)
    assert wf.submit(EXAMPLE, backend="claude") is down
    assert wf.pending is None and wf.last_result is down and wf.last_backend == "claude"


def test_claude_backend_without_configuration_is_unavailable(monkeypatch):
    """Real adapter, no key / model configured: no network, a clear result."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LIVE_TRACKER_CLAUDE_MODEL", raising=False)
    wf = RuleWorkflow()
    result = wf.submit(EXAMPLE, backend="claude")
    assert isinstance(result, ClaudeUnavailable) and result.reason == "missing_model"
    assert wf.pending is None


def test_claude_proposal_can_be_confirmed_and_is_labelled():
    def fake_claude(text, rule_id, **kw):
        offline = parse_rule(text, rule_id)
        return RuleProposal(offline.source_text, offline.rule_dict,
                            ("Cross-checked with the Claude API (model m).",) + offline.assumptions)

    wf = RuleWorkflow(claude_fn=fake_claude)
    wf.submit(EXAMPLE, backend="claude")
    p = preview(wf.pending)
    assert p["backend"] == "claude" and p["assumptions"][0].startswith("Cross-checked")
    wf.confirm(wf.pending.token)
    assert [r.rule_id for r in wf.ruleset.active()] == ["R1"]


def test_backend_is_never_chosen_silently():
    calls = []
    wf = RuleWorkflow(claude_fn=lambda *a, **k: calls.append("claude"),
                      offline_fn=lambda t, rid: calls.append("offline") or parse_rule(t, rid))
    wf.submit(EXAMPLE)
    assert calls == ["offline"]
    with pytest.raises(ValueError):
        wf.submit(EXAMPLE, backend="auto")


def test_a_parser_that_returns_an_approved_rule_is_refused():
    def bad_parser(text, rule_id):
        offline = parse_rule(text, rule_id)
        return RuleProposal(offline.source_text, {**offline.rule_dict, "status": "approved"}, ())

    wf = RuleWorkflow(offline_fn=bad_parser)
    result = wf.submit(EXAMPLE)
    assert isinstance(result, BackendFailure) and "not a fresh, unapproved proposal" in result.message
    assert wf.pending is None and wf.ruleset.rules == {}


# ---- stale and duplicate actions ----

def test_double_confirm_is_refused():
    wf = submitted()
    token = wf.pending.token
    wf.confirm(token)
    with pytest.raises(StaleActionError):
        wf.confirm(token)
    with pytest.raises(StaleActionError):
        wf.reject(token)
    assert len(wf.ruleset.rules) == 1 and wf.ruleset.rules["R1"].status == "approved"


def test_confirm_after_reject_is_refused():
    wf = submitted()
    token = wf.pending.token
    wf.reject(token)
    with pytest.raises(StaleActionError):
        wf.confirm(token)
    assert wf.ruleset.active() == []


def test_changed_instruction_discards_the_pending_proposal():
    wf = submitted()
    token = wf.pending.token
    assert wf.sync_instruction(EXAMPLE + " ") is True  # any change, even whitespace
    assert wf.pending is None and "changed" in wf.notice
    with pytest.raises(StaleActionError):
        wf.confirm(token)
    assert wf.ruleset.rules == {}


def test_unchanged_instruction_keeps_the_proposal():
    wf = submitted()
    token = wf.pending.token
    assert wf.sync_instruction(EXAMPLE) is False
    assert wf.pending.token == token


def test_new_submission_replaces_the_old_proposal():
    wf = submitted()
    old = wf.pending.token
    wf.submit("Log mounts of 5+ seconds")
    assert wf.pending.token != old and wf.pending.proposal.rule_dict["rule_id"] == "R2"
    with pytest.raises(StaleActionError):
        wf.confirm(old)
    wf.confirm(wf.pending.token)
    assert [r.behavior for r in wf.ruleset.active()] == ["mount"]


def test_clarification_after_a_proposal_discards_it():
    wf = submitted()
    old = wf.pending.token
    wf.submit("Alert me when the mice attack")
    assert wf.pending is None
    with pytest.raises(StaleActionError):
        wf.confirm(old)


def test_rule_ids_are_unique_and_never_reused():
    wf = RuleWorkflow()
    seen = []
    for text in (EXAMPLE, "Log mounts of 5+ seconds", "Alert me when the mice attack",
                 "Ask me when they investigate for 3 s"):
        wf.submit(text)
        if wf.pending:
            seen.append(wf.pending.proposal.rule_dict["rule_id"])
            wf.reject(wf.pending.token)
    assert seen == ["R1", "R2", "R4"]  # R3 went to the clarification and is not reused
    assert len(set(wf.ruleset.rules)) == len(wf.ruleset.rules) == 3


def test_ids_skip_rules_already_in_a_given_ruleset():
    rs = RuleSet()
    rs.add({"rule_id": "R1", "behavior": "attack", "min_duration_s": 2})
    wf = RuleWorkflow(ruleset=rs)
    wf.submit(EXAMPLE)
    assert wf.pending.proposal.rule_dict["rule_id"] == "R2"


def test_identical_approved_rule_is_not_approved_twice():
    wf = submitted()
    wf.confirm(wf.pending.token)
    wf.submit(EXAMPLE)
    token = wf.pending.token
    with pytest.raises(DuplicateRuleError, match="R1"):
        wf.confirm(token)
    assert [r.rule_id for r in wf.ruleset.active()] == ["R1"]
    assert wf.pending is not None  # still pending: the researcher can reject it
    wf.reject(token)
    assert wf.ruleset.rules["R2"].status == "rejected"


def test_different_threshold_is_not_a_duplicate():
    wf = submitted()
    wf.confirm(wf.pending.token)
    wf.submit(EXAMPLE + " with alert threshold 0.9")
    wf.confirm(wf.pending.token)
    assert [r.rule_id for r in wf.ruleset.active()] == ["R1", "R2"]


# ---- views ----

def test_rules_table_and_monitoring_flag():
    wf = submitted()
    wf.confirm(wf.pending.token)
    wf.submit("Log mounts of 5+ seconds")
    wf.reject(wf.pending.token)
    rows = wf.rules_table()
    assert [(r["rule_id"], r["status"]) for r in rows] == [("R1", "approved (active)"),
                                                           ("R2", "rejected (inactive)")]
    assert rows[0]["min_duration"] == "at least 2 seconds"
    assert monitoring_allowed(wf.ruleset)
    assert not monitoring_allowed(RuleSet())


def test_issue_type_is_reused_from_agent():
    """The workflow adds no new result types for the page to handle."""
    wf = RuleWorkflow()
    result = wf.submit("Alert me when the mice attack")
    assert all(isinstance(i, Issue) for i in result.issues)


# ---- Codex review: atomic decisions ----

SECRET = "sk-ant-secret-0123456789"


def _fault(monkeypatch, method, *, after):
    """Make RuleSet.<method> raise, before or after changing the rule's status."""
    real = getattr(RuleSet, method)

    def faulty(self, rule_id):
        if after:
            real(self, rule_id)
        raise RuntimeError(f"injected failure {SECRET}")

    monkeypatch.setattr(RuleSet, method, faulty)


def _state(wf):
    return {rid: r.to_dict() for rid, r in wf.ruleset.rules.items()}


def _with_history():
    """A workflow with one approved (R1) and one rejected (R2) rule, and a
    third proposal (R3) pending."""
    wf = submitted()
    wf.confirm(wf.pending.token)
    wf.submit("Log mounts of 5+ seconds")
    wf.reject(wf.pending.token)
    wf.submit("Ask me when they investigate for 3 s")
    return wf


@pytest.mark.parametrize("method, decide", [("approve", "confirm"), ("reject", "reject")])
@pytest.mark.parametrize("after", [False, True], ids=["before_mutation", "after_mutation"])
def test_failed_decision_changes_nothing(monkeypatch, method, decide, after):
    wf = _with_history()
    before, live, pending = _state(wf), wf.ruleset, wf.pending
    _fault(monkeypatch, method, after=after)
    with pytest.raises(DecisionError) as err:
        getattr(wf, decide)(pending.token)
    assert SECRET not in str(err.value) and "RuntimeError" in str(err.value)
    assert "still pending" in str(err.value)
    # nothing changed: same RuleSet object, same rules, no R3, nothing newly active
    assert wf.ruleset is live and _state(wf) == before
    assert "R3" not in wf.ruleset.rules
    assert [r.rule_id for r in wf.ruleset.active()] == ["R1"]
    assert wf.pending is pending  # same proposal, same token: retryable
    assert run_monitor(demo_scores(), wf.ruleset, run_id="t") == \
        run_monitor(demo_scores(), RuleSet(rules={"R1": wf.ruleset.rules["R1"]}), run_id="t")


@pytest.mark.parametrize("method, decide, status", [("approve", "confirm", "approved"),
                                                    ("reject", "reject", "rejected")])
def test_retry_succeeds_after_the_fault_is_removed(monkeypatch, method, decide, status):
    wf = _with_history()
    token = wf.pending.token
    _fault(monkeypatch, method, after=True)
    with pytest.raises(DecisionError):
        getattr(wf, decide)(token)
    monkeypatch.undo()
    rule = getattr(wf, decide)(token)  # same token, no duplicate-ID error
    assert rule.rule_id == "R3" and wf.ruleset.rules["R3"].status == status
    assert wf.ruleset.rules["R1"].status == "approved" and wf.ruleset.rules["R2"].status == "rejected"
    assert wf.pending is None
    with pytest.raises(StaleActionError):
        getattr(wf, decide)(token)


def test_failure_inside_add_changes_nothing(monkeypatch):
    wf = _with_history()
    before = _state(wf)
    monkeypatch.setattr(RuleSet, "add", lambda self, data: (_ for _ in ()).throw(ValueError(SECRET)))
    with pytest.raises(DecisionError) as err:
        wf.confirm(wf.pending.token)
    assert SECRET not in str(err.value) and _state(wf) == before and wf.pending is not None


def test_add_that_inserts_then_raises_changes_nothing(monkeypatch):
    real_add = RuleSet.add

    def add_then_fail(self, data):
        real_add(self, data)
        raise ValueError("late failure")

    wf = _with_history()
    before = _state(wf)
    monkeypatch.setattr(RuleSet, "add", add_then_fail)
    with pytest.raises(DecisionError):
        wf.confirm(wf.pending.token)
    assert _state(wf) == before and "R3" not in wf.ruleset.rules
    monkeypatch.undo()
    wf.confirm(wf.pending.token)  # retry: no "duplicate rule_id R3"
    assert wf.ruleset.rules["R3"].status == "approved"


@pytest.mark.parametrize("bad", ["status_not_set", "touches_other_rule"])
def test_silently_wrong_decision_is_refused(monkeypatch, bad):
    """approve() returning without the expected effect, or with a side effect
    on another rule, is a failure too, not a success."""
    def faulty(self, rule_id):
        if bad == "touches_other_rule":
            self.rules[rule_id].status = "approved"
            self.rules["R2"].status = "approved"  # would activate a rejected rule

    wf = _with_history()
    before = _state(wf)
    monkeypatch.setattr(RuleSet, "approve", faulty)
    with pytest.raises(DecisionError):
        wf.confirm(wf.pending.token)
    assert _state(wf) == before and [r.rule_id for r in wf.ruleset.active()] == ["R1"]


def test_successful_decision_swaps_in_a_consistent_ruleset():
    wf = _with_history()
    old = wf.ruleset
    wf.confirm(wf.pending.token)
    assert wf.ruleset is not old  # committed by one reference swap
    assert [(r.rule_id, r.status) for r in wf.ruleset.rules.values()] == \
        [("R1", "approved"), ("R2", "rejected"), ("R3", "approved")]
    assert "R3" not in old.rules  # the old object was never touched


# ---- Codex review: unexpected backend exceptions ----

@pytest.mark.parametrize("backend", ["offline", "claude"])
def test_backend_exception_after_a_valid_proposal(backend):
    def boom(*a, **k):
        raise RuntimeError(f"connection to proxy with {SECRET} failed")

    wf = RuleWorkflow(claude_fn=boom, offline_fn=parse_rule)
    wf.submit(EXAMPLE)
    wf.confirm(wf.pending.token)                 # an approved rule exists
    wf.submit("Log mounts of 5+ seconds")        # and a valid pending proposal
    old_token = wf.pending.token
    if backend == "offline":
        wf.offline_fn = boom
    result = wf.submit("Ask me when they investigate for 3 s", backend=backend)
    expected = ClaudeUnavailable if backend == "claude" else BackendFailure
    assert isinstance(result, expected)
    assert SECRET not in result.message and "RuntimeError" in result.message
    assert wf.last_result is result and wf.last_backend == backend  # not the old proposal
    assert wf.pending is None
    with pytest.raises(StaleActionError):
        wf.confirm(old_token)
    assert [(r.rule_id, r.status) for r in wf.ruleset.rules.values()] == [("R1", "approved")]


@pytest.mark.parametrize("returned", [None, "a rule", {"rule_id": "R2"}, 42])
def test_backend_returning_garbage_is_a_controlled_failure(returned):
    wf = RuleWorkflow(offline_fn=lambda t, rid: returned)
    result = wf.submit(EXAMPLE)
    assert isinstance(result, BackendFailure) and "it returned an unexpected" in result.message
    assert wf.pending is None


def test_state_is_cleared_before_the_backend_runs():
    seen = []

    def spy(text, rule_id):
        seen.append((wf.pending, wf.last_result, wf.last_backend))
        return parse_rule(text, rule_id)

    wf = submitted()
    wf.offline_fn = spy
    wf.submit("Log mounts of 5+ seconds")
    assert seen == [(None, None, None)]


# ---- Codex review: session-state validation ----

@pytest.mark.parametrize("value", [{}, None, "R1", [], 0, RuleSet(), object()])
def test_invalid_stored_values_are_not_workflows(value):
    assert not is_valid_workflow(value)


def test_corrupted_workflow_is_not_valid():
    wf = submitted()
    assert is_valid_workflow(wf)
    for broken in ({"ruleset": None}, {"ruleset": {"R1": "approved"}}, {"pending": "token"}):
        bad = RuleWorkflow()
        for k, v in broken.items():
            setattr(bad, k, v)
        assert not is_valid_workflow(bad), broken
    bad = RuleWorkflow()
    bad.ruleset.rules["R1"] = {"status": "approved"}
    assert not is_valid_workflow(bad)


# ---- Codex second review: the whole staged rule is verified ----

_CHANGES = {
    "behavior": "mount",
    "min_duration_s": 7.5,  # the pending R3 rule has 3.0
    "alert_threshold": 0.95,
    "ask_threshold": 0.1,
    "alert_mode": "log",
    "description": "something else",
    "rule_id": "R99",
}


def _silent(monkeypatch, method, field, value):
    """RuleSet.<method> sets the right status but silently changes one field."""
    status = {"approve": "approved", "reject": "rejected"}[method]

    def wrong(self, rule_id):
        self.rules[rule_id].status = status
        setattr(self.rules[rule_id], field, value)

    monkeypatch.setattr(RuleSet, method, wrong)


@pytest.mark.parametrize("method, decide", [("approve", "confirm"), ("reject", "reject")])
@pytest.mark.parametrize("field", list(_CHANGES))
def test_silent_field_change_is_refused(monkeypatch, method, decide, field):
    wf = _with_history()
    before, live, pending = _state(wf), wf.ruleset, wf.pending
    _silent(monkeypatch, method, field, _CHANGES[field])
    with pytest.raises(DecisionError):
        getattr(wf, decide)(pending.token)
    assert wf.ruleset is live and _state(wf) == before and wf.pending is pending
    assert [r.rule_id for r in wf.ruleset.active()] == ["R1"]
    monkeypatch.undo()
    rule = getattr(wf, decide)(pending.token)  # retry with the same token
    assert rule.to_dict() == {**pending.proposal.rule_data(),
                              "status": "approved" if decide == "confirm" else "rejected"}


def test_codex_behavior_reproduction(monkeypatch):
    def incorrect_approve(self, rule_id):
        self.rules[rule_id].status = "approved"
        self.rules[rule_id].behavior = "mount"

    wf = submitted()
    monkeypatch.setattr(RuleSet, "approve", incorrect_approve)
    with pytest.raises(DecisionError):
        wf.confirm(wf.pending.token)
    assert wf.ruleset.rules == {} and wf.pending.proposal.rule_dict["behavior"] == "attack"


@pytest.mark.parametrize("value", [2, "2.0", None])
def test_type_change_alone_is_refused(monkeypatch, value):
    """2.0 -> 2 compares equal as a number but is still not the validated rule."""
    wf = submitted()
    _silent(monkeypatch, "approve", "min_duration_s", value)
    with pytest.raises(DecisionError):
        wf.confirm(wf.pending.token)
    assert wf.ruleset.rules == {}


@pytest.mark.parametrize("damage", ["insert", "remove", "modify_other", "reorder", "replace_object"])
@pytest.mark.parametrize("method, decide", [("approve", "confirm"), ("reject", "reject")])
def test_unrelated_rules_must_be_untouched(monkeypatch, damage, method, decide):
    real = getattr(RuleSet, method)

    def wrong(self, rule_id):
        real(self, rule_id)
        if damage == "insert":
            self.rules["R50"] = self.rules["R1"].__class__(**{**self.rules["R1"].to_dict(),
                                                            "rule_id": "R50"})
        elif damage == "remove":
            del self.rules["R2"]
        elif damage == "modify_other":
            self.rules["R1"].min_duration_s = 0.5
        elif damage == "reorder":
            self.rules = dict(reversed(list(self.rules.items())))
        else:  # same content, but R1 is no longer a Rule under its own id
            self.rules["R1"] = self.rules["R2"]

    wf = _with_history()
    before = _state(wf)
    monkeypatch.setattr(RuleSet, method, wrong)
    with pytest.raises(DecisionError):
        getattr(wf, decide)(wf.pending.token)
    assert _state(wf) == before and [r.rule_id for r in wf.ruleset.active()] == ["R1"]


def test_normal_decisions_still_match_exactly():
    wf = _with_history()
    rule = wf.confirm(wf.pending.token)
    assert rule.to_dict() == {**parse_rule("Ask me when they investigate for 3 s", "R3").rule_data(),
                              "status": "approved"}


# ---- Codex second review: restored workflow state is fully validated ----

def _healthy():
    wf = _with_history()  # R1 approved, R2 rejected, R3 pending
    assert is_valid_workflow(wf)
    return wf


def _corrupt_pending_rule(wf, **change):
    p = wf.pending
    data = {**p.proposal.rule_dict, **change}
    for k in [k for k, v in change.items() if v is _DROP]:
        del data[k]
    wf.pending = type(p)(p.token, p.instruction, p.backend,
                         RuleProposal(p.proposal.source_text, data, p.proposal.assumptions))
    return wf


_DROP = object()


def _without(wf, name):
    delattr(wf, name)
    return wf


def _set(wf, name, value):
    setattr(wf, name, value)
    return wf


def _replace_pending(wf, **change):
    from dataclasses import replace
    wf.pending = replace(wf.pending, **change)
    return wf


def _replace_proposal(wf, **change):
    from dataclasses import replace
    wf.pending = replace(wf.pending, proposal=replace(wf.pending.proposal, **change))
    return wf


def _rename_rule(wf, old, new):
    rule = wf.ruleset.rules.pop(old)
    rule.rule_id = new
    wf.ruleset.rules[new] = rule
    return wf


def _rule(wf, rid, **change):
    for k, v in change.items():
        object.__setattr__(wf.ruleset.rules[rid], k, v)
    return wf


CORRUPTIONS = {
    # the five Codex reproductions
    "approved rule status corrupt": lambda wf: _rule(wf, "R1", status="corrupt"),
    "rule duration None": lambda wf: _rule(wf, "R1", min_duration_s=None),
    "_next_id None": lambda wf: _set(wf, "_next_id", None),
    "missing ruleset": lambda wf: _without(wf, "ruleset"),
    "pending missing duration": lambda wf: _corrupt_pending_rule(wf, min_duration_s=_DROP),
    # more stored-rule damage
    "rule behavior unknown": lambda wf: _rule(wf, "R1", behavior="fighting"),
    "rule behavior not a classifier score": lambda wf: _rule(wf, "R1", behavior="immobility"),
    "rule duration negative": lambda wf: _rule(wf, "R1", min_duration_s=-1.0),
    "rule duration inf": lambda wf: _rule(wf, "R1", min_duration_s=float("inf")),
    "rule duration nan": lambda wf: _rule(wf, "R1", min_duration_s=float("nan")),
    "rule duration int": lambda wf: _rule(wf, "R1", min_duration_s=2),
    "rule duration bool": lambda wf: _rule(wf, "R1", min_duration_s=True),
    "rule thresholds swapped": lambda wf: _rule(wf, "R1", ask_threshold=0.9, alert_threshold=0.1),
    "rule threshold out of range": lambda wf: _rule(wf, "R1", alert_threshold=1.5),
    "rule mode unknown": lambda wf: _rule(wf, "R1", alert_mode="email"),
    "rule description not str": lambda wf: _rule(wf, "R1", description=None),
    "rule id mismatch": lambda wf: _rule(wf, "R1", rule_id="R7"),
    "rule id padded": lambda wf: _rule(wf, "R1", rule_id=" R1"),
    "rule stored as dict": lambda wf: _set(wf.ruleset, "rules", {"R1": {"status": "approved"}}),
    "rules not a dict": lambda wf: _set(wf.ruleset, "rules", [("R1", None)]),
    "rules key not str": lambda wf: _set(wf.ruleset, "rules", {1: wf.ruleset.rules["R1"]}),
    "ruleset not a RuleSet": lambda wf: _set(wf, "ruleset", {"rules": {}}),
    "rule extra attribute": lambda wf: _set(wf.ruleset.rules["R1"], "approved_by", "nobody") or wf,
    # counter
    "_next_id zero": lambda wf: _set(wf, "_next_id", 0),
    "_next_id bool": lambda wf: _set(wf, "_next_id", True),
    "_next_id str": lambda wf: _set(wf, "_next_id", "4"),
    "_next_id below pending id": lambda wf: _set(wf, "_next_id", 2),
    # pending damage
    "pending not a Pending": lambda wf: _set(wf, "pending", "R3"),
    "pending token empty": lambda wf: _replace_pending(wf, token=""),
    "pending token None": lambda wf: _replace_pending(wf, token=None),
    "pending token short": lambda wf: _replace_pending(wf, token="abc"),
    "pending backend unknown": lambda wf: _replace_pending(wf, backend="auto"),
    "pending instruction None": lambda wf: _replace_pending(wf, instruction=None),
    "pending proposal None": lambda wf: _replace_pending(wf, proposal=None),
    "pending proposal is a Clarification": lambda wf: _replace_pending(
        wf, proposal=Clarification("x", (Issue("missing_duration", "x"),))),
    "pending already approved": lambda wf: _corrupt_pending_rule(wf, status="approved"),
    "pending behavior unknown": lambda wf: _corrupt_pending_rule(wf, behavior="fighting"),
    "pending duration str": lambda wf: _corrupt_pending_rule(wf, min_duration_s="2"),
    "pending threshold out of range": lambda wf: _corrupt_pending_rule(wf, alert_threshold=2.0),
    "pending extra field": lambda wf: _corrupt_pending_rule(wf, approved=True),
    "pending id collides": lambda wf: _corrupt_pending_rule(wf, rule_id="R1"),
    "pending rule_dict not a mapping": lambda wf: _replace_proposal(wf, rule_dict=["attack"]),
    "pending rule_dict a number": lambda wf: _replace_proposal(wf, rule_dict=5),  # TypeError inside
    "pending duck-typed": lambda wf: _set(wf, "pending", SimpleNamespace(**vars(wf.pending))),
    "rule id not R<n>": lambda wf: _rename_rule(wf, "R1", "X1"),
    "_next_id None, nothing pending": lambda wf: _set(_set(wf, "pending", None), "_next_id", None),
    "pending assumptions not tuple": lambda wf: _replace_proposal(wf, assumptions="note"),
    "pending assumptions not str": lambda wf: _replace_proposal(wf, assumptions=(None,)),
    # other attributes
    "missing pending": lambda wf: _without(wf, "pending"),
    "missing _next_id": lambda wf: _without(wf, "_next_id"),
    "last_backend unknown": lambda wf: _set(wf, "last_backend", "auto"),
    "last_result garbage": lambda wf: _set(wf, "last_result", {"rule_id": "R1"}),
    "notice not str": lambda wf: _set(wf, "notice", 5),
    "offline_fn not callable": lambda wf: _set(wf, "offline_fn", None),
}


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_corrupted_workflow_is_invalid_and_validation_never_raises(name):
    wf = CORRUPTIONS[name](_healthy())
    snapshot = repr(vars(wf)) if hasattr(wf, "__dict__") else None
    assert is_valid_workflow(wf) is False
    assert (repr(vars(wf)) if hasattr(wf, "__dict__") else None) == snapshot  # not modified


def test_validation_survives_hostile_objects():
    class Hostile(RuleWorkflow):
        def __getattribute__(self, name):
            raise RuntimeError("boom")

    assert is_valid_workflow(Hostile()) is False


@pytest.mark.parametrize("build", [
    lambda: RuleWorkflow(),
    lambda: submitted(),
    lambda: _with_history(),
])
def test_normal_sessions_stay_valid(build):
    wf = build()
    assert is_valid_workflow(wf)
    if wf.pending:
        wf.confirm(wf.pending.token)
        assert is_valid_workflow(wf)
    wf.submit("Alert me when the mice attack")  # clarification as last_result
    assert is_valid_workflow(wf)


# ---- Codex2 review: complete Rule instances and mapping proposals ----
RULE_FIELDS = [f.name for f in _dc_fields(Rule)]


def test_codex2_missing_alert_threshold_reproduction():
    wf = RuleWorkflow()
    wf.submit(EXAMPLE)
    wf.confirm(wf.pending.token)
    del wf.ruleset.rules["R1"].alert_threshold
    assert is_valid_workflow(wf) is False


def test_deleted_attribute_would_otherwise_read_as_the_default():
    """Why the check matters: the class default silently replaces the value."""
    wf = submitted(EXAMPLE + " with alert threshold 0.95")
    wf.confirm(wf.pending.token)
    rule = wf.ruleset.rules["R1"]
    del rule.alert_threshold
    assert rule.alert_threshold == 0.8 and rule.to_dict()["alert_threshold"] == 0.8
    assert is_valid_workflow(wf) is False


@pytest.mark.parametrize("field_name", RULE_FIELDS)
@pytest.mark.parametrize("rid", ["R1", "R2"], ids=["approved", "rejected"])
def test_deleting_any_rule_field_is_invalid(field_name, rid):
    wf = _with_history()
    delattr(wf.ruleset.rules[rid], field_name)
    assert is_valid_workflow(wf) is False


def test_extra_rule_attribute_is_invalid():
    wf = _with_history()
    wf.ruleset.rules["R1"].approved_by = "nobody"
    assert is_valid_workflow(wf) is False


@pytest.mark.parametrize("field_name", RULE_FIELDS)
def test_decision_that_deletes_a_field_is_refused(monkeypatch, field_name):
    real = RuleSet.approve

    def approve_and_drop(self, rule_id):
        real(self, rule_id)
        delattr(self.rules[rule_id], field_name)

    wf = _with_history()
    before = _state(wf)
    monkeypatch.setattr(RuleSet, "approve", approve_and_drop)
    with pytest.raises(DecisionError):
        wf.confirm(wf.pending.token)
    assert _state(wf) == before and wf.pending is not None
    monkeypatch.undo()
    assert wf.confirm(wf.pending.token).status == "approved"
    assert is_valid_workflow(wf)


def test_codex2_pending_pairs_reproduction():
    wf = RuleWorkflow()
    wf.submit(EXAMPLE)
    p = wf.pending
    wf.pending = _dc_replace(p, proposal=_dc_replace(p.proposal,
                                                    rule_dict=list(p.proposal.rule_dict.items())))
    assert is_valid_workflow(wf) is False


@pytest.mark.parametrize("make", [
    lambda d: list(d.items()), lambda d: tuple(d.items()), lambda d: [list(kv) for kv in d.items()],
    lambda d: iter(d.items()), lambda d: frozenset(d.items()), lambda d: str(d),
])
def test_non_mapping_pending_data_is_invalid(make):
    wf = submitted()
    p = wf.pending
    wf.pending = _dc_replace(p, proposal=_dc_replace(p.proposal, rule_dict=make(dict(p.proposal.rule_dict))))
    assert is_valid_workflow(wf) is False


class _ReadOnlyMapping(Mapping):
    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


@pytest.mark.parametrize("wrap", [MappingProxyType, _ReadOnlyMapping, dict])
def test_valid_mappings_are_accepted_and_previewable(wrap):
    wf = submitted()
    p = wf.pending
    wf.pending = _dc_replace(p, proposal=_dc_replace(p.proposal, rule_dict=wrap(dict(p.proposal.rule_dict))))
    assert is_valid_workflow(wf)
    assert preview(wf.pending)["min_duration"] == "at least 2 seconds"  # renders
    assert wf.confirm(wf.pending.token).status == "approved"


def test_every_accepted_pending_state_can_be_previewed():
    """Any workflow the validator accepts must render without errors."""
    for build in (submitted, _with_history):
        wf = build()
        assert is_valid_workflow(wf)
        p = preview(wf.pending)
        assert p["rule_json"]["status"] == "proposed"


def test_validation_does_not_modify_rules_or_proposals():
    wf = _with_history()
    before_rules = {rid: dict(vars(r)) for rid, r in wf.ruleset.rules.items()}
    before_pending = (wf.pending, dict(wf.pending.proposal.rule_dict))
    assert is_valid_workflow(wf)
    assert {rid: dict(vars(r)) for rid, r in wf.ruleset.rules.items()} == before_rules
    assert (wf.pending, dict(wf.pending.proposal.rule_dict)) == before_pending
