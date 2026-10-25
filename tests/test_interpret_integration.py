"""The Claude reading as a third backend: in the rule workflow and on the Rules page (fake client, nothing is paid)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from live_tracker import agent
from live_tracker import claude_interpreter as ci
from live_tracker.agent import Clarification, Issue, RuleProposal
from live_tracker.claude_agent import ClaudeConfig, ClaudeUnavailable
from live_tracker.rule_workflow import BACKENDS, BackendFailure, RuleWorkflow, is_valid_workflow, preview

from test_claude_interpreter import SENTENCE, FakeClient, reading

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "1_Rules.py")
THIRD = "Claude reads the sentence (stronger with loose wording, paid API call)"


def fake_interpreter(data=None):
    return lambda text, rule_id, **kw: ci.interpret_rule_with_claude(
        text, rule_id, client=FakeClient(reading() if data is None else data), config=ClaudeConfig(model="test-model"))


def texts(elements):
    return " ".join(e.value for e in elements)


# ---- the workflow ----

def test_the_new_backend_is_listed_and_unknown_ones_are_still_refused():
    assert BACKENDS == ("offline", "claude", "claude_interpret")
    with pytest.raises(ValueError):
        RuleWorkflow().submit(SENTENCE, backend="telepathy")


def test_a_reading_becomes_a_pending_proposal_that_can_be_confirmed():
    wf = RuleWorkflow(interpret_fn=fake_interpreter())
    result = wf.submit(SENTENCE, backend="claude_interpret")
    assert type(result) is RuleProposal and wf.pending.backend == "claude_interpret" and is_valid_workflow(wf)
    p = preview(wf.pending)
    assert p["backend"] == "claude_interpret" and p["min_duration"] == "at least 2 seconds"
    assert any("Read by Claude" in a for a in p["assumptions"]) and p["uses_default_thresholds"] == ["alert_threshold", "ask_threshold"]
    wf.confirm(wf.pending.token)
    assert [r.rule_id for r in wf.ruleset.active()] == ["R1"] and wf.monitoring_allowed()


def test_a_refusal_or_an_unavailable_claude_leaves_nothing_pending():
    wf = RuleWorkflow(interpret_fn=lambda t, rid, **kw: Clarification(t, (Issue("unsupported_condition", "no places"),)))
    assert isinstance(wf.submit(SENTENCE, backend="claude_interpret"), Clarification) and wf.pending is None
    wf = RuleWorkflow(interpret_fn=lambda t, rid, **kw: ClaudeUnavailable("missing_model", "no model"))
    assert isinstance(wf.submit(SENTENCE, backend="claude_interpret"), ClaudeUnavailable) and wf.pending is None


def test_a_broken_backend_is_reported_without_its_text():
    def boom(text, rule_id, **kw):
        raise RuntimeError("secret-key-sk-123")

    wf = RuleWorkflow(interpret_fn=boom)
    result = wf.submit(SENTENCE, backend="claude_interpret")
    assert isinstance(result, ClaudeUnavailable) and "Claude reading failed unexpectedly" in result.message
    assert "secret" not in result.message and wf.pending is None
    wf = RuleWorkflow(interpret_fn=lambda t, rid, **kw: "not a result")
    assert "unexpected str" in wf.submit(SENTENCE, backend="claude_interpret").message


def test_a_reading_cannot_arrive_already_approved_or_with_the_wrong_id():
    def sneaky(text, rule_id, **kw):
        good = ci.interpret_rule_with_claude(text, rule_id, client=FakeClient(reading()), config=ClaudeConfig(model="m"))
        return RuleProposal(good.source_text, {**good.rule_dict, "status": "approved"}, good.assumptions)

    wf = RuleWorkflow(interpret_fn=sneaky)
    result = wf.submit(SENTENCE, backend="claude_interpret")
    assert isinstance(result, ClaudeUnavailable) and wf.pending is None and not wf.ruleset.active()


def test_the_offline_and_cross_check_backends_are_not_touched():
    wf = RuleWorkflow(interpret_fn=lambda *a, **k: pytest.fail("the reading must not be called"))
    assert isinstance(wf.submit("alert me when the mice attack for at least 2 seconds"), RuleProposal)
    assert isinstance(agent.parse_rule("alert me when mice attack for more that 2 seconds", "R1"), Clarification)   # the strict parser still refuses it
    assert isinstance(RuleWorkflow().submit("alert me when mice attack for more that 2 seconds"), RuleProposal)    # the default reading understands it


# ---- the page ----

def open_page(wf=None):
    at = AppTest.from_file(PAGE, default_timeout=60)
    if wf is not None:
        at.session_state["rule_workflow"] = wf
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_the_page_offers_the_reading_as_a_third_choice_and_explains_it():
    at = open_page()
    assert len(at.radio[0].options) == 3 and at.radio[0].options[2] == THIRD
    at.radio[0].set_value(THIRD).run()
    assert "Claude reads the sentence" in texts(at.info) and "one paid call per click" in texts(at.info)
    assert "refused, not dropped" in texts(at.caption) and "ANTHROPIC_API_KEY" in texts(at.caption)


def test_the_page_turns_the_loose_sentence_into_a_rule_the_researcher_confirms():
    at = open_page(RuleWorkflow(interpret_fn=fake_interpreter()))
    at.radio[0].set_value(THIRD).run()
    at.text_area(key="rule_instruction").set_value(SENTENCE).run()
    [b for b in at.button if b.label == "Propose rule"][0].click().run()
    assert not at.exception
    shown = texts(at.markdown)
    assert "Proposed rule R1" in texts(at.subheader) and "Interpreted by:** Claude (paid), checked by code" in shown
    assert "Read by Claude" in shown and "treated as at least 2 seconds" in shown and "Minimum duration:** at least 2 seconds" in shown
    [b for b in at.button if b.label == "Confirm"][0].click().run()
    assert "approved" in texts(at.success) or "Monitoring can start" in texts(at.success)


def test_without_a_model_the_page_says_claude_is_unavailable_and_proposes_nothing(monkeypatch):
    monkeypatch.delenv("LIVE_TRACKER_CLAUDE_MODEL", raising=False)
    at = open_page()
    at.radio[0].set_value(THIRD).run()
    at.text_area(key="rule_instruction").set_value(SENTENCE).run()
    [b for b in at.button if b.label == "Propose rule"][0].click().run()
    assert "Claude unavailable (missing_model)" in texts(at.error) and "Proposed rule" not in texts(at.subheader)


def test_a_constraint_the_rule_cannot_express_is_shown_as_a_refusal_with_the_reason():
    text = "alert me when mice attack near the wall for at least 2 seconds"
    data = reading(duration_quote="for at least 2 seconds", decision="clarify", assumptions=[],
                   unsupported=[{"quote": "near the wall", "category": "location"}], questions=["Rules cannot use places."])
    at = open_page(RuleWorkflow(interpret_fn=fake_interpreter(data)))
    at.radio[0].set_value(THIRD).run()
    at.text_area(key="rule_instruction").set_value(text).run()
    [b for b in at.button if b.label == "Propose rule"][0].click().run()
    assert "cannot be turned into a rule yet" in texts(at.warning)
    assert "location condition ('near the wall')" in texts(at.markdown)


# ---- the smart reading is the default on the page ----

def test_the_first_choice_is_the_smart_reading_and_it_understands_a_loose_sentence_with_its_notes():
    at = open_page()
    assert at.radio[0].value.startswith("Smart reading (default") and "nothing is sent anywhere" in texts(at.caption)
    at.text_area(key="rule_instruction").set_value("let me know if the mice are fighting for more that 2 seconds").run()
    [b for b in at.button if b.label == "Propose rule"][0].click().run()
    assert not at.exception
    shown = texts(at.markdown)
    assert "Interpreted by:** smart reading (offline, free)" in shown
    assert "Your sentence was read as: 'let me know if the mice are attacking for at least 2 seconds'." in shown
    assert "Read 'fighting' as attack" in shown and "inclusive minimum" in shown
    assert "Minimum duration:** at least 2 seconds" in shown and "Behavior:** attack" in shown


def test_a_refusal_is_shown_in_plain_words_with_an_example_and_no_code_names():
    at = open_page()
    at.text_area(key="rule_instruction").set_value("alert me when mice attack for less than 2 seconds").run()
    [b for b in at.button if b.label == "Propose rule"][0].click().run()
    assert "cannot be turned into a rule yet" in texts(at.warning)
    assert "unsupported_duration" not in texts(at.markdown) and "Try other words" in texts(at.caption)
    assert "Proposed rule" not in texts(at.subheader)


def test_without_a_key_the_page_says_claude_needs_one_and_with_a_key_it_offers_the_button(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LIVE_TRACKER_CLAUDE_MODEL", raising=False)
    at = open_page()
    at.text_area(key="rule_instruction").set_value("alert me when mice attack near the wall for 2 seconds").run()
    [b for b in at.button if b.label == "Propose rule"][0].click().run()
    assert "needs an API key" in texts(at.caption) and not any(b.key == "ask_claude" for b in at.button)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("LIVE_TRACKER_CLAUDE_MODEL", "test-model")
    at = open_page()
    at.text_area(key="rule_instruction").set_value("alert me when mice attack near the wall for 2 seconds").run()
    [b for b in at.button if b.label == "Propose rule"][0].click().run()
    assert any(b.key == "ask_claude" for b in at.button)


def test_the_ask_claude_button_runs_the_paid_reading_only_when_clicked():
    calls = []

    def reading_fn(text, rule_id, **kw):
        calls.append(text)
        return fake_interpreter()(text, rule_id)

    import os
    os.environ["ANTHROPIC_API_KEY"], os.environ["LIVE_TRACKER_CLAUDE_MODEL"] = "sk-test-not-real", "test-model"
    try:
        at = open_page(RuleWorkflow(interpret_fn=reading_fn))
        at.text_area(key="rule_instruction").set_value("alert me when mice attack near the wall for 2 seconds").run()
        [b for b in at.button if b.label == "Propose rule"][0].click().run()
        assert calls == []                                               # the smart reading refused; nothing was sent
        [b for b in at.button if b.key == "ask_claude"][0].click().run()
        assert calls == ["alert me when mice attack near the wall for 2 seconds"]
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("LIVE_TRACKER_CLAUDE_MODEL", None)
