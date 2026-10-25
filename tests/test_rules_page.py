"""Tests for the rule confirmation page (new file).

Uses Streamlit's AppTest, so no browser is needed. State transitions are
tested in test_rule_workflow.py; these check what the page shows and that
clicks go through the workflow.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from live_tracker.agent import RuleProposal, parse_rule
from live_tracker.rule_workflow import RuleWorkflow

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "1_Rules.py")
EXAMPLE = "Alert me when the mice attack for at least 2 seconds"


@pytest.fixture(autouse=True)
def no_claude_config(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LIVE_TRACKER_CLAUDE_MODEL", raising=False)


_UNSET = object()


def start(workflow=_UNSET):
    at = AppTest.from_file(PAGE, default_timeout=30)
    if workflow is not _UNSET:
        at.session_state["rule_workflow"] = workflow
    at.run()
    assert not at.exception
    return at


def propose(at, text=EXAMPLE):
    at.text_area(key="rule_instruction").input(text).run()
    button(at, "Propose rule").click().run()
    assert not at.exception
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


def has_button(at, label):
    return any(b.label == label for b in at.button)


def texts(elements):
    return " ".join(e.value for e in elements)


def why(at):
    """The hover help (?) next to the instruction box: the reasons and the assumptions."""
    return at.text_area[0].help or ""


def wf(at) -> RuleWorkflow:
    return at.session_state["rule_workflow"]


def test_page_starts_with_offline_default_and_no_monitoring():
    at = start()
    assert at.radio[0].value.startswith("Smart reading (default")
    assert not wf(at).monitoring_allowed()
    assert not has_button(at, "Confirm")


def test_proposal_preview_shows_all_fields():
    at = propose(start())
    body = texts(at.markdown)
    assert "Proposed rule" not in texts(at.subheader)                                # section removed on request
    for removed in ("**Behavior:**", "**Minimum duration:**", "**Alert mode:**", "**Alert threshold:**",
                    "**Ask threshold:**", "**Interpreted by:**"):
        assert removed not in body, removed
    assert "rules.py defaults are used" in why(at)                                   # in the (?), not in bullets on the page
    assert "Assumptions to check" not in body
    assert "NOT active until you click Confirm" not in texts(at.warning)           # removed on request; Confirm / Reject decide
    assert "Untuned default thresholds: alert_threshold, ask_threshold" in why(at) and not any(w.value.startswith("Untuned") for w in at.warning)
    assert "Your instruction" not in body and EXAMPLE not in texts(at.text)       # section removed on request
    assert '"status": "proposed"' in at.code[0].value
    assert has_button(at, "Confirm") and has_button(at, "Reject")
    assert wf(at).ruleset.rules == {}  # nothing added before the decision
    assert not wf(at).monitoring_allowed()


def test_implicit_minimum_assumption_is_shown():
    at = propose(start(), "Alert me if they attack for 2 s")
    assert "Interpreted duration as at least 2 seconds." in why(at)


def test_confirm_activates_and_enables_monitoring():
    at = propose(start())
    button(at, "Confirm").click().run()
    assert not at.exception
    assert [r.rule_id for r in wf(at).ruleset.active()] == ["R1"]
    assert "Rule R1 approved" in texts(at.info)
    assert wf(at).monitoring_allowed() and len(wf(at).ruleset.active()) == 1
    assert not has_button(at, "Confirm")  # nothing left to approve twice
    assert [r.status for r in wf(at).ruleset.rules.values()] == ["approved"]


def test_reject_keeps_monitoring_blocked():
    at = propose(start())
    button(at, "Reject").click().run()
    assert not at.exception
    assert wf(at).ruleset.active() == []
    assert wf(at).ruleset.rules["R1"].status == "rejected"
    assert not wf(at).monitoring_allowed()
    assert [r.status for r in wf(at).ruleset.rules.values()] == ["rejected"]


@pytest.mark.parametrize("text", [
    "Alert me when the mice attack",
    "Alert me if they attack for 2 seconds at night",
    "Avoid alerting me if they attack for 2 seconds",
])
def test_clarification_has_no_confirm_button(text):
    at = propose(start(), text)
    assert "cannot be turned into a rule yet" in texts(at.warning)
    for issue in parse_rule(text, "R1").issues:  # every reason is shown to the researcher, in plain words (no code names)
        assert f"- {issue.message}" in why(at) and issue.code not in why(at)
        assert issue.message not in texts(at.markdown)                                # not shown as bullets on the page
    assert not has_button(at, "Confirm") and not has_button(at, "Reject")
    assert wf(at).ruleset.rules == {}


def test_editing_the_instruction_discards_the_proposal():
    at = propose(start())
    at.text_area(key="rule_instruction").input(EXAMPLE + " with alert threshold 0.9").run()
    assert not at.exception
    assert "proposal was discarded" in texts(at.info)
    assert not has_button(at, "Confirm")
    assert wf(at).ruleset.rules == {} and wf(at).pending is None


def test_confirm_clicked_together_with_an_edit_is_not_applied():
    """The edit and the click arrive in the same rerun: the edit wins."""
    at = propose(start())
    at.text_area(key="rule_instruction").set_value("Log mounts of 5+ seconds")
    button(at, "Confirm").click()
    at.run()
    assert not at.exception
    assert wf(at).ruleset.rules == {} and wf(at).ruleset.active() == []
    assert "proposal was discarded" in texts(at.info)


def test_identical_rule_cannot_be_approved_twice():
    at = propose(start())
    button(at, "Confirm").click().run()
    button(at, "Propose rule").click().run()
    button(at, "Confirm").click().run()
    assert not at.exception
    assert "Not approved: identical to approved rule R1" in texts(at.error)
    assert [r.rule_id for r in wf(at).ruleset.active()] == ["R1"]


def test_state_survives_reruns():
    at = propose(start())
    workflow = wf(at)
    token = workflow.pending.token
    at.run()
    at.run()
    assert wf(at) is workflow and workflow.pending.token == token


def test_claude_option_explains_its_limits():
    at = start()
    at.radio[0].set_value(at.radio[0].options[1]).run()
    info = texts(at.info) + " " + texts(at.caption)      # short line in the box, full text under "More info"
    assert "needs an API key" in texts(at.info)
    assert "only accepts instructions that the offline parser already understands" in info
    assert "never interpret an instruction the offline parser rejects" in info


def test_claude_unavailable_is_shown_and_not_approvable():
    at = start()
    at.radio[0].set_value(at.radio[0].options[1]).run()
    propose(at)
    assert "Claude cross-check unavailable (missing_model)" in texts(at.error)
    assert not has_button(at, "Confirm")
    assert wf(at).ruleset.rules == {}


def test_claude_cross_checked_proposal_can_be_confirmed():
    def fake_claude(text, rule_id, **kw):
        offline = parse_rule(text, rule_id)
        return RuleProposal(offline.source_text, offline.rule_dict,
                            ("Cross-checked with the Claude API (model m).",) + offline.assumptions)

    at = start(RuleWorkflow(claude_fn=fake_claude))
    at.radio[0].set_value(at.radio[0].options[1]).run()
    propose(at)
    assert "offline parser + Claude cross-check" not in texts(at.markdown)           # the section was removed on request
    assert "Cross-checked with the Claude API" in why(at)
    button(at, "Confirm").click().run()
    assert [r.rule_id for r in wf(at).ruleset.active()] == ["R1"]


def test_claude_clarification_is_shown():
    at = start()
    at.radio[0].set_value(at.radio[0].options[1]).run()
    propose(at, "Log attacks for 2 seconds")  # refused offline, before any API call
    assert "cannot be turned into a rule yet" in texts(at.warning)
    assert not has_button(at, "Confirm")


# ---- Codex review: invalid session state, failed decisions, backend errors ----

@pytest.mark.parametrize("value", [{}, None, "R1 approved", [], 0])
def test_invalid_session_state_is_recovered(value):
    at = start(value)
    assert "stored rule session was invalid and has been reset" in texts(at.error)
    assert isinstance(wf(at), RuleWorkflow) and wf(at).ruleset.rules == {}
    assert not wf(at).monitoring_allowed()
    propose(at)  # and the page works normally afterwards
    button(at, "Confirm").click().run()
    assert not at.exception and [r.rule_id for r in wf(at).ruleset.active()] == ["R1"]
    assert "invalid" not in texts(at.error)  # the recovery message is shown once


def test_valid_session_state_is_kept():
    workflow = RuleWorkflow()
    workflow.submit(EXAMPLE)
    workflow.confirm(workflow.pending.token)
    at = start(workflow)
    assert wf(at) is workflow and "invalid" not in texts(at.error)
    assert wf(at).monitoring_allowed() and len(wf(at).ruleset.active()) == 1


@pytest.mark.parametrize("method, label", [("approve", "Confirm"), ("reject", "Reject")])
@pytest.mark.parametrize("after", [False, True])
def test_failed_decision_shows_a_controlled_error(monkeypatch, method, label, after):
    from live_tracker.rules import RuleSet
    real = getattr(RuleSet, method)

    def faulty(self, rule_id):
        if after:
            real(self, rule_id)
        raise RuntimeError("injected sk-ant-secret")

    at = propose(start())
    monkeypatch.setattr(RuleSet, method, faulty)
    button(at, label).click().run()
    assert not at.exception
    assert f"{label} failed (RuntimeError)" in texts(at.error)
    assert "sk-ant-secret" not in texts(at.error)
    assert wf(at).ruleset.rules == {} and wf(at).ruleset.active() == []
    assert has_button(at, label)  # still pending
    monkeypatch.undo()
    button(at, label).click().run()  # retry works
    assert not at.exception and "R1" in wf(at).ruleset.rules


def test_backend_exception_is_shown_without_details():
    def boom(*a, **k):
        raise RuntimeError("proxy password sk-ant-secret")

    at = start(RuleWorkflow(claude_fn=boom))
    propose(at)  # a valid offline proposal first
    at.radio[0].set_value(at.radio[0].options[1]).run()
    button(at, "Propose rule").click().run()
    assert not at.exception
    assert "Claude cross-check unavailable (api_error)" in texts(at.error)
    assert "sk-ant-secret" not in texts(at.error)
    assert not has_button(at, "Confirm") and wf(at).pending is None


def test_offline_backend_exception_is_shown_without_details():
    workflow = RuleWorkflow()
    at = start(workflow)
    workflow.offline_fn = lambda t, rid: (_ for _ in ()).throw(KeyError("sk-ant-secret"))
    propose(at)
    assert "offline parser failed unexpectedly (it raised KeyError)" in texts(at.error)
    assert "sk-ant-secret" not in texts(at.error) and not has_button(at, "Confirm")


# ---- Codex second review: corrupted restored workflows ----

def _history_workflow():
    """R1 approved, R2 rejected, R3 pending."""
    w = RuleWorkflow()
    w.submit(EXAMPLE)
    w.confirm(w.pending.token)
    w.submit("Log mounts of 5+ seconds")
    w.reject(w.pending.token)
    w.submit("Ask me when they investigate for 3 s")
    return w


def _corrupt(name):
    from dataclasses import replace
    w = _history_workflow()
    if name == "status corrupt":
        w.ruleset.rules["R1"].status = "corrupt"
    elif name == "duration None":
        w.ruleset.rules["R1"].min_duration_s = None
    elif name == "_next_id None":
        w._next_id = None
    elif name == "missing ruleset":
        del w.ruleset
    elif name == "pending missing duration":
        data = {k: v for k, v in w.pending.proposal.rule_dict.items() if k != "min_duration_s"}
        w.pending = replace(w.pending, proposal=replace(w.pending.proposal, rule_dict=data))
    elif name == "pending token corrupt":
        w.pending = replace(w.pending, token=None)
    elif name == "pending already approved":
        data = {**w.pending.proposal.rule_dict, "status": "approved"}
        w.pending = replace(w.pending, proposal=replace(w.pending.proposal, rule_dict=data))
    elif name == "rules mapping malformed":
        w.ruleset.rules = {"R1": {"status": "approved", "behavior": "attack"}}
    return w


@pytest.mark.parametrize("name", [
    "status corrupt", "duration None", "_next_id None", "missing ruleset",
    "pending missing duration", "pending token corrupt", "pending already approved",
    "rules mapping malformed",
])
def test_corrupted_workflow_is_replaced_not_used(name):
    at = start(_corrupt(name))
    assert not at.exception
    assert "stored rule session was invalid and has been reset" in texts(at.error)
    fresh = wf(at)
    assert fresh.ruleset.rules == {} and fresh.pending is None  # nothing carried over
    assert not has_button(at, "Confirm")
    assert not wf(at).monitoring_allowed()
    propose(at)
    button(at, "Confirm").click().run()  # the fresh session works normally
    assert not at.exception and [r.rule_id for r in wf(at).ruleset.active()] == ["R1"]


def test_healthy_history_survives_reruns_and_decisions():
    workflow = _history_workflow()
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["rule_workflow"] = workflow
    at.session_state["rule_instruction"] = workflow.pending.instruction  # the restored text box
    at.run()
    assert not at.exception
    assert wf(at) is workflow and "invalid" not in texts(at.error)
    assert has_button(at, "Confirm")  # R3 still pending after restore
    button(at, "Confirm").click().run()
    assert not at.exception
    assert [(r.rule_id, r.status) for r in wf(at).ruleset.rules.values()] == \
        [("R1", "approved"), ("R2", "rejected"), ("R3", "approved")]
    at.run()
    assert "invalid" not in texts(at.error)
    assert wf(at).monitoring_allowed() and len(wf(at).ruleset.active()) == 2


# ---- Codex2 review: missing Rule attributes, non-mapping pending data ----

@pytest.mark.parametrize("field_name", ["alert_threshold", "ask_threshold", "status", "alert_mode",
                                        "description", "behavior", "min_duration_s", "rule_id"])
def test_rule_with_a_deleted_field_resets_the_session(field_name):
    w = _history_workflow()
    delattr(w.ruleset.rules["R1"], field_name)
    at = start(w)
    assert not at.exception
    assert "stored rule session was invalid and has been reset" in texts(at.error)
    assert wf(at).ruleset.rules == {} and not wf(at).monitoring_allowed()


@pytest.mark.parametrize("make", [lambda d: list(d.items()), lambda d: tuple(d.items())])
def test_pending_pairs_reset_the_session_instead_of_crashing(make):
    from dataclasses import replace
    w = _history_workflow()
    p = w.pending
    w.pending = replace(p, proposal=replace(p.proposal, rule_dict=make(dict(p.proposal.rule_dict))))
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["rule_workflow"] = w
    at.session_state["rule_instruction"] = p.instruction  # so the pending preview would render
    at.run()
    assert not at.exception
    assert "stored rule session was invalid and has been reset" in texts(at.error)
    assert wf(at).pending is None and wf(at).ruleset.rules == {}


def test_read_only_mapping_pending_still_renders_and_confirms():
    from types import MappingProxyType
    from dataclasses import replace
    w = _history_workflow()
    p = w.pending
    w.pending = replace(p, proposal=replace(p.proposal, rule_dict=MappingProxyType(dict(p.proposal.rule_dict))))
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["rule_workflow"] = w
    at.session_state["rule_instruction"] = p.instruction
    at.run()
    assert not at.exception and "invalid" not in texts(at.error)
    assert "Minimum duration" not in texts(at.markdown) and has_button(at, "Confirm")
    button(at, "Confirm").click().run()
    assert [(r.rule_id, r.status) for r in wf(at).ruleset.rules.values()] == \
        [("R1", "approved"), ("R2", "rejected"), ("R3", "approved")]
