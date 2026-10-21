"""Tests for the offline sentence -> rule parser (new file)."""

import ast
import socket
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import live_tracker.agent as agent
from live_tracker.agent import CLARIFICATION_CODES, Clarification, RuleProposal, parse_rule
from live_tracker.fake_scores import demo_scores
from live_tracker.monitor import run_monitor
from live_tracker.rules import CLASSIFIER_BEHAVIORS, Rule, RuleSet, validate_rule

EXAMPLE = "Alert me when the mice attack for at least 2 seconds."


def proposal(text, rule_id="R1"):
    result = parse_rule(text, rule_id)
    assert isinstance(result, RuleProposal), getattr(result, "question", result)
    return result


def clarification(text, rule_id="R1"):
    result = parse_rule(text, rule_id)
    assert isinstance(result, Clarification), dict(result.rule_dict)
    assert result.issues and set(result.codes) <= set(CLARIFICATION_CODES)
    assert all(i.message for i in result.issues)
    return result


# ---- valid instructions -> schema-compatible proposals ----

@pytest.mark.parametrize("text, behavior, seconds, mode", [
    (EXAMPLE, "attack", 2.0, "alert"),
    ("Alert me if the mice attack for at least 2 seconds", "attack", 2.0, "alert"),
    ("Notify me when an attack lasts at least 3 s", "attack", 3.0, "alert"),
    ("Warn me about attacking for 2s or more", "attack", 2.0, "alert"),
    ("Let me know when they attack for at least 1.5 sec", "attack", 1.5, "alert"),
    ("Ask me to confirm when the mice are attacking each other for 2+ s", "attack", 2.0, "ask"),
    ("Can you alert me when there is an attack lasting at least 2 seconds?", "attack", 2.0, "alert"),
    ("Please log attacks lasting at least 2 s", "attack", 2.0, "log"),
    ("Alert me about any attack bouts that last at least 4 seconds", "attack", 4.0, "alert"),
    ("Alert me whenever both mice attack for a minimum of 2 seconds!", "attack", 2.0, "alert"),
    ("Alert me if they attack for at least 2 seconds or longer", "attack", 2.0, "alert"),
    ("Ask me if a mount lasts at least 3 s", "mount", 3.0, "ask"),
    ("Alert me when a mounting bout continues for at least 2 s", "mount", 2.0, "alert"),
    ("Alert me when the mice mount for at least 2 seconds", "mount", 2.0, "alert"),
    ("Tell me if mounting lasts at least 1 minute", "mount", 60.0, "alert"),
    ("Log mounts of 5+ seconds", "mount", 5.0, "log"),
    ("Log investigation bouts of 3 s or more", "investigation", 3.0, "log"),
    ("Record when the mice investigate each other for two seconds", "investigation", 2.0, "log"),
    ("Check with me when they are investigating for at least 0.5 minutes", "investigation", 30.0, "ask"),
    ("Alert me when they attack for a minute", "attack", 60.0, "alert"),
    ("Alert me if they attack for 1.5 minutes", "attack", 90.0, "alert"),
    ("Alert me if they attack for 0.1 min", "attack", 6.0, "alert"),
    ("alert me if they attack for 2 mins", "attack", 120.0, "alert"),
    ("Alert me if they attack for no less than 2 s", "attack", 2.0, "alert"),
    ("ALERT ME IF THE TWO MICE ATTACK FOR AT LEAST 2 SECONDS", "attack", 2.0, "alert"),
    ("Notify me when there's an investigation episode that lasts 2 seconds", "investigation", 2.0, "alert"),
    ("Alert me if they attack for 02.50 s", "attack", 2.5, "alert"),
    ("Alert me if they attack for .5 minutes", "attack", 30.0, "alert"),
    ("Alert me if they attack for 2 sec.", "attack", 2.0, "alert"),  # '.' ends the sentence
    ("Alert me  if they\tattack for 2 s", "attack", 2.0, "alert"),
])
def test_valid_instructions(text, behavior, seconds, mode):
    p = proposal(text)
    data = p.rule_data()
    assert (data["behavior"], data["min_duration_s"], data["alert_mode"]) == (behavior, seconds, mode)
    assert isinstance(data["min_duration_s"], float)
    assert data["rule_id"] == "R1" and data["description"] == text.strip()
    assert data["status"] == "proposed"
    # the proposal is exactly what the shared contract accepts
    rule, errors = validate_rule(data)
    assert errors == [] and rule.to_dict() == data
    assert set(data) == set(Rule.__dataclass_fields__)


def test_readme_example_matches_interfaces_rule():
    """Same values as the INTERFACES.md §3 / rules.py example rule."""
    data = proposal(EXAMPLE).rule_data()
    expected = {"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0,
                "alert_threshold": 0.8, "ask_threshold": 0.5, "alert_mode": "alert",
                "description": EXAMPLE, "status": "proposed"}
    assert data == expected


def test_every_classifier_behavior_is_reachable():
    got = {proposal(f"Alert me when they {w} for 2 s").rule_data()["behavior"]
           for w in ("attack", "mount", "investigate")}
    assert got == CLASSIFIER_BEHAVIORS


def test_strict_wording_is_not_weakened_to_a_minimum():
    """The schema's duration is inclusive; 'more than' must not become '>='."""
    c = clarification("Notify me if mounting lasts more than 1.5 minutes")
    assert c.codes == ("unsupported_duration",)
    assert "at least" in c.question


def test_named_thresholds_are_used():
    p = proposal("Alert me when they attack for 2 s with alert threshold 90% and ask threshold 0.6")
    data = p.rule_data()
    assert (data["alert_threshold"], data["ask_threshold"]) == (0.9, 0.6)
    assert not any("default" in a for a in p.assumptions)
    data = proposal("Alert me when they attack for 2 s, alert_threshold=0.95").rule_data()
    assert (data["alert_threshold"], data["ask_threshold"]) == (0.95, 0.5)


def test_default_thresholds_are_never_silent():
    p = proposal(EXAMPLE)
    assert any("alert_threshold = 0.8" in a and "ask_threshold = 0.5" in a and "not tuned" in a
               for a in p.assumptions)
    # the defaults come from rules.py, not from the parser
    defaults = Rule("x", "attack", 1.0)
    assert p.rule_data()["alert_threshold"] == defaults.alert_threshold
    assert p.rule_data()["ask_threshold"] == defaults.ask_threshold


def test_ask_mode_is_flagged_as_provisional():
    p = proposal("Ask me when they attack for 2 s")
    assert p.rule_data()["alert_mode"] == "ask"
    assert any("provisional" in a for a in p.assumptions)
    assert not any("provisional" in a for a in proposal(EXAMPLE).assumptions)


# ---- clarifications ----

@pytest.mark.parametrize("text, code", [
    # missing information
    ("", "empty_instruction"),
    ("   ", "empty_instruction"),
    ("x" * 501, "instruction_too_long"),
    ("Alert me when the mice attack", "missing_duration"),
    ("Alert me for at least 2 seconds", "no_behavior"),
    ("Watch for attacks lasting 2 s", "missing_action"),
    ("Monitor mounting for 2 seconds", "missing_action"),
    # behaviors
    ("Alert me if they attack or mount for 2 s", "multiple_behaviors"),
    ("Alert me if they mount and investigate for 2 s", "multiple_behaviors"),
    ("Alert me if they fight for 2 s", "unsupported_behavior"),
    ("Alert me about aggression lasting 2 s", "unsupported_behavior"),
    ("Alert me if they are sniffing for 2 s", "unsupported_behavior"),
    ("Alert me when a mouse is grooming for 2 s", "unsupported_behavior"),
    ("Alert me on immobility for 2 s", "unsupported_behavior"),
    ("Alert me on tracking loss for 2 s", "unsupported_behavior"),
    ("Alert me if they attack while sniffing for 2 s", "unsupported_behavior"),
    # durations
    ("Alert me if they attack for 2-3 seconds", "ambiguous_duration"),
    ("Alert me if they attack for 2 to 3 seconds", "ambiguous_duration"),
    ("Alert me if they attack for between 2 and 3 seconds", "ambiguous_duration"),
    ("Alert me if they attack for twenty five seconds", "ambiguous_duration"),
    ("Alert me if they attack for 2 to 3 seconds or more", "ambiguous_duration"),
    ("Alert me if they attack for twenty five seconds or more", "ambiguous_duration"),
    ("Alert me about a 2 or 3 second attack", "ambiguous_duration"),
    ("Alert me if they attack for 2,5 seconds", "ambiguous_duration"),
    ("Alert me if they attack for a few seconds", "ambiguous_duration"),
    ("Alert me if they attack for a while", "missing_duration"),
    ("Alert me about attacks of 2 seconds", "ambiguous_duration"),
    ("Alert me if they attack 5 minutes into the session", "ambiguous_duration"),
    ("Alert me if they attack for a second time", "ambiguous_duration"),
    ("Alert me if they attack for less than 2 seconds", "unsupported_duration"),
    ("Alert me if they attack for at most 2 seconds", "unsupported_duration"),
    ("Alert me if they attack within 10 seconds", "unsupported_duration"),
    ("Alert me if they attack for exactly 2 seconds", "unsupported_duration"),
    ("Alert me if they attack for 500 ms", "unsupported_duration"),
    ("Alert me if they attack for 60 frames", "unsupported_duration"),
    ("Alert me if they attack for 1 hour", "unsupported_duration"),
    ("Alert me if they attack for 2 s, at least 3 s", "conflicting_durations"),
    ("Alert me if they attack for 2 seconds within 5 minutes", "unsupported_duration"),
    # actions
    ("Alert me and log when they attack for 2 s", "conflicting_actions"),
    ("Alert me when they attack for 2 s and ask me if unsure", "conflicting_actions"),
    # thresholds and other conditions
    ("Alert me when attack confidence is above 0.9 for 2 s", "ambiguous_threshold"),
    ("Alert me if the attack probability stays over 80% for 2 s", "ambiguous_threshold"),
    ("Alert me if attack lasts 2 s and you are sure", "ambiguous_threshold"),
    ("Alert me when they attack for 2 s, alert threshold 0.9, alert threshold 0.7",
     "conflicting_thresholds"),
    ("Alert me if they attack 3 times for 2 seconds", "unsupported_condition"),
    ("Alert me if they attack twice for 2 seconds", "unsupported_condition"),
    ("Alert me if mouse 2 attacks for 2 seconds", "unsupported_condition"),
    # negation
    ("Don't alert me if they attack for 2 s", "negation"),
    ("Do not alert me when they attack for 2 s", "negation"),
    ("Alert me if they attack for 2 s unless tracking is bad", "negation"),
    ("Never log attacks for 2 s", "negation"),
])
def test_clarifications(text, code):
    assert code in clarification(text).codes


def test_clarification_explains_supported_options():
    c = clarification("Alert me if they fight for 2 s")
    assert "attack" in c.question and "investigation" in c.question and "mount" in c.question
    assert c.source_text == "Alert me if they fight for 2 s"
    c = clarification("Watch for attacks lasting 2 s")
    assert "alert" in c.question and "ask" in c.question and "log" in c.question


def test_unsupported_synonyms_are_not_mapped():
    """'fighting' gets a hint, never a silent mapping to attack."""
    c = clarification("Alert me if they fight for 2 s")
    assert "attack" in c.question
    assert "no_behavior" not in c.codes


def test_all_problems_reported_at_once():
    c = clarification("please watch the mice")
    assert {"no_behavior", "missing_duration", "missing_action"} <= set(c.codes)


@pytest.mark.parametrize("text", [
    "Alert me if they attack for 0 seconds",
    "Alert me if they attack for 0 minutes",
    "Alert me if they attack for 0.0 s",
])
def test_zero_duration_fails_shared_validation(text):
    c = clarification(text)
    assert c.codes == ("invalid_rule",)
    assert any("min_duration_s must be > 0" in i.message for i in c.issues)


@pytest.mark.parametrize("text", [
    "Alert me if they attack for -2 seconds",
    "Alert me if they attack for minus 2 seconds",
])
def test_negative_durations_are_not_in_the_grammar(text):
    assert "unsupported_duration" in clarification(text).codes


def test_non_finite_duration_is_rejected():
    c = clarification("Alert me if they attack for " + "9" * 400 + " seconds")
    assert c.codes == ("invalid_rule",)
    assert "min_duration_s must be finite" in c.question


@pytest.mark.parametrize("thresholds, msg", [
    ("alert threshold 1.5", "alert_threshold must be between 0 and 1"),
    ("alert threshold 150%", "alert_threshold must be between 0 and 1"),
    ("ask threshold 0.9", "ask_threshold must be <= alert_threshold"),  # default alert 0.8
    ("alert threshold 0.4 and ask threshold 0.6", "ask_threshold must be <= alert_threshold"),
])
def test_invalid_thresholds_fail_shared_validation(thresholds, msg):
    """No clamping or swapping: the validator's own errors come back."""
    c = clarification(f"Alert me when they attack for 2 s with {thresholds}")
    assert set(c.codes) == {"invalid_rule"}
    assert msg in [i.message for i in c.issues]


def test_invalid_rule_id_fails_shared_validation():
    c = clarification(EXAMPLE, rule_id="  ")
    assert c.codes == ("invalid_rule",)
    assert "rule_id must be a non-empty string" in c.question


def test_non_string_text_is_a_type_error():
    with pytest.raises(TypeError):
        parse_rule(None, "R1")


# ---- uses the shared contract; never approves ----

def test_validate_rule_is_always_called(monkeypatch):
    calls = []
    real = agent.validate_rule

    def spy(data):
        calls.append(dict(data))
        return real(data)

    monkeypatch.setattr(agent, "validate_rule", spy)
    proposal(EXAMPLE)
    assert calls and calls[-1]["behavior"] == "attack"

    # if the shared validator says no, the parser does not override it
    monkeypatch.setattr(agent, "validate_rule", lambda data: (None, ["vetoed by rules.py"]))
    c = clarification(EXAMPLE)
    assert c.codes == ("invalid_rule",) and c.question == "vetoed by rules.py"


def test_proposal_is_never_approved():
    p = proposal(EXAMPLE)
    assert p.rule_data()["status"] == "proposed" and p.rule.status == "proposed"
    rs = RuleSet()
    rs.add(p.rule_data())
    assert rs.rules["R1"].status == "proposed"
    assert rs.active() == []
    # nothing runs until the researcher confirms
    assert run_monitor(demo_scores(), rs, run_id="t") == []


def test_proposal_cannot_be_modified_into_an_approved_rule():
    p = proposal(EXAMPLE)
    with pytest.raises(TypeError):
        p.rule_dict["status"] = "approved"
    with pytest.raises(FrozenInstanceError):
        p.assumptions = ()
    data = p.rule_data()
    data["status"] = "approved"
    r = p.rule
    r.status = "approved"
    assert p.rule_data()["status"] == "proposed" and p.rule.status == "proposed"
    # and an approved dict is refused by the shared contract anyway
    with pytest.raises(ValueError, match="proposed"):
        RuleSet().add(data)


def test_confirmed_proposal_runs_in_the_monitor():
    """End to end: sentence -> proposal -> RuleSet.add -> approve -> monitor."""
    rs = RuleSet()
    rs.add(proposal(EXAMPLE, "R1").rule_data())
    rs.add(proposal("Log investigation bouts of 2 s or more", "R2").rule_data())
    rs.approve("R1")
    rs.approve("R2")
    events = run_monitor(demo_scores(), rs, run_id="t")
    attack = [e for e in events if e["rule_id"] == "R1"]
    assert [(e["start_s"], e["action"]) for e in attack] == [(2.0, "alert"), (7.0, "ask")]
    assert [e["action"] for e in events if e["rule_id"] == "R2"] == ["log"]


def test_deterministic():
    results = {repr(parse_rule(EXAMPLE, "R1")) for _ in range(5)}
    assert len(results) == 1
    assert parse_rule(EXAMPLE, "R1") == parse_rule(EXAMPLE, "R1")


# ---- offline ----

def test_parser_makes_no_network_calls(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(parse_rule(EXAMPLE, "R1"), RuleProposal)
    assert isinstance(parse_rule("watch the mice", "R1"), Clarification)


def test_agent_imports_only_local_and_stdlib_modules():
    tree = ast.parse(Path(agent.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add("." if node.level else node.module.split(".")[0])
    assert imported <= {"__future__", "math", "re", "dataclasses", "decimal", "types", "typing", "."}


# ---- Codex review regressions: the whole sentence must be represented ----

CODEX_REPRODUCTIONS = [
    # A. unconsumed restrictions
    "Alert me if they attack for 2 seconds only for the intruder",
    "Alert me if they attack for 2 seconds at night",
    "Alert me if mouse A attacks for 2 seconds",
    "Alert me if they attack for 2 seconds after contact",
    "Alert me if they attack for 2 seconds only once",
    "Alert me if they attack for 2 seconds while grooming",
    "Alert me if they attack for 2 seconds near the wall",
    # B. suppression
    "Avoid alerting me if they attack for 2 seconds",
    "Disable alerts for attacks lasting 2 seconds",
    "Refrain from notifying me when they attack for 2 seconds",
    # C. non-inclusive duration comparisons
    "Alert me if they attack for 2 seconds or less",
    "Alert me if they attack for 2 seconds exactly",
    "Alert me if they attack for > 2 seconds",
    "Alert me if they attack for more than 2 seconds",
    # D. duration not bound to the behavior
    "Alert me for 2 seconds when they attack",
    "Alert me if they attack for 2 seconds, then remain apart for 2 seconds",
    # E. malformed numeric tokens
    "Alert me if they attack for 2 seconds with ask threshold 0.5％",
    "Alert me if they attack for 2 seconds with alert threshold .9e",
]


@pytest.mark.parametrize("text", CODEX_REPRODUCTIONS)
def test_codex_reproductions_are_refused(text):
    clarification(text)


@pytest.mark.parametrize("text, code", [
    ("Avoid alerting me if they attack for 2 seconds", "negation"),
    ("Disable alerts for attacks lasting 2 seconds", "negation"),
    ("Refrain from notifying me when they attack for 2 seconds", "negation"),
    ("Alert me if they attack for 2 seconds or less", "unsupported_duration"),
    ("Alert me if they attack for 2 seconds exactly", "unsupported_duration"),
    ("Alert me if they attack for > 2 seconds", "unsupported_duration"),
    ("Alert me if they attack for more than 2 seconds", "unsupported_duration"),
    ("Alert me if they attack for 2 seconds, then remain apart for 2 seconds", "conflicting_durations"),
    ("Alert me if they attack for 2 seconds with alert threshold .9e", "unsupported_phrasing"),
    ("Alert me if they attack for 2 seconds with ask threshold 0.5％", "unsupported_phrasing"),
])
def test_codex_reproductions_get_specific_reasons(text, code):
    assert code in clarification(text).codes


def test_unconsumed_clause_is_quoted_back():
    c = clarification("Alert me if they attack for 2 seconds at night")
    assert c.codes == ("unsupported_phrasing",)
    assert "'at night'" in c.question


VALID_BASES = [
    EXAMPLE.rstrip("."),
    "Notify me when an attack lasts at least 3 s",
    "Log investigation bouts of 3 s or more",
    "Ask me when they mount for 2 seconds",
    "Alert me when they attack for 2 s with alert threshold 0.9",
]
RESTRICTIONS = [
    "only for the intruder", "at night", "after contact", "only once", "while grooming",
    "near the wall", "in the home cage", "during the first minute", "before they separate",
    "then remain apart", "unless tracking is bad", "but not at night", "and also mount",
    "if the resident starts it", "on weekdays", "please", "or not", "and stop",
    "in session s01", "for the resident", "after 5 minutes", "twice",
]


@pytest.mark.parametrize("base", VALID_BASES)
@pytest.mark.parametrize("extra", RESTRICTIONS)
def test_any_trailing_restriction_is_refused(base, extra):
    proposal(base)  # the base sentence is supported...
    for sep in (" ", ", "):
        clarification(base + sep + extra)  # ...but nothing may be silently dropped


@pytest.mark.parametrize("text", [
    "Alert me if the resident attacks the intruder for at least 2 seconds",
    "Alert me if the intruder attacks for 2 seconds",
    "Alert me if mouse 1 attacks for 2 seconds",
    "Alert me if the resident mouse mounts for 2 s",
    "Alert me if one mouse investigates the other for 2 s",
    "Alert me if they suddenly attack for 2 s",
    "Alert me if they attack the wall for 2 s",
    "Alert me if they attack again for 2 s",
    "Alert me in the second half if they attack for 2 s",
    "Alert me only if they attack for 2 s",
    "Alert me if they attack for 2 s in the recorded session",
    # named animals, even with the plural verb form the grammar uses
    "Alert me if the resident attack for 2 s",
    "Alert me if mouse a attack for 2 s",
    "Alert me if the intruder mount for 2 s",
    "Alert me if the males investigate for 2 s",
])
def test_restrictions_inside_the_sentence_are_refused(text):
    clarification(text)


@pytest.mark.parametrize("text", [
    "Avoid alerting me if they attack for 2 seconds",
    "Don't alert me if they attack for 2 s",
    "Don’t alert me if they attack for 2 s",
    "Never alert me when they attack for 2 s",
    "Stop alerting me when they attack for 2 s",
    "Do not log attacks lasting at least 2 s",
    "Disable alerts for attacks lasting 2 seconds",
    "Mute alerts when they attack for 2 s",
    "Suppress notifications if they attack for 2 s",
    "Skip logging when they attack for 2 s",
    "Turn off alerts when they attack for 2 s",
    "Alert me unless they attack for 2 s",
    "Alert me if they don't attack for 2 s",
    "Alert me if they do not attack for 2 s",
    "Alert me if they stop attacking for 2 s",
    "No alerts when they attack for 2 s",
    "Alert me if they attack for 2 s, except at night",
])
def test_suppression_and_negation_are_refused(text):
    clarification(text)


@pytest.mark.parametrize("text", [
    "Alert me if they attack for 2 seconds or less",
    "Alert me if they attack for 2 seconds or shorter",
    "Alert me if they attack for 2 seconds exactly",
    "Alert me if they attack for exactly 2 seconds",
    "Alert me if they attack for precisely 2 s",
    "Alert me if they attack for 2 s at most",
    "Alert me if they attack for at most 2 s",
    "Alert me if they attack for 2 s max",
    "Alert me if they attack for > 2 seconds",
    "Alert me if they attack for >= 2 seconds",
    "Alert me if they attack for < 2 seconds",
    "Alert me if they attack for more than 2 seconds",
    "Alert me if they attack for longer than 2 seconds",
    "Alert me if they attack for over 2 seconds",
    "Alert me if an attack lasts more than 2 s",
    "Alert me about attacks lasting over 2 s",
    "Alert me if they attack for up to 2 s",
    "Alert me if they attack for under 2 s",
    "Alert me if they attack within 2 s",
    "Alert me if they attack for about 2 s",
    "Alert me if they attack for roughly 2 s",
    "Alert me if they attack for 2-3 s",
    "Ask me when there is a 3-second attack",
    "Log attacks of 2 seconds",
])
def test_non_inclusive_or_unclear_comparisons_are_refused(text):
    clarification(text)


@pytest.mark.parametrize("text", [
    "Alert me for 2 seconds when they attack",
    "For 2 seconds, alert me when they attack",
    "Log attacks for 2 s",  # could be 'log for 2 s'
    "Alert me about attacks for 2 s",
    "Alert me if they attack for 2 seconds, then remain apart for 2 seconds",
    "Alert me if they attack for 2 seconds for 2 seconds",
    "Alert me if they attack for 2 s and mount for 2 s",
    "Alert me if they attack for 2 seconds within 2 seconds",
    "Alert me if they attack for 2 minutes (120 seconds)",
    "Alert me if they attack for 2 s after 2 s",
    "Wait 2 seconds and alert me if they attack for 2 s",
])
def test_duration_must_describe_the_behavior(text):
    clarification(text)


def test_equal_durations_are_not_collapsed():
    c = clarification("Alert me if they attack for 2 seconds, then remain apart for 2 seconds")
    assert "conflicting_durations" in c.codes
    assert "conflicting_durations" in clarification(
        "Alert me if they attack for 2 seconds for 2 seconds").codes


@pytest.mark.parametrize("text", [
    "Alert me if they attack for 2 seconds with alert threshold .9e",
    "Alert me if they attack for 2 s with alert threshold 0.9x",
    "Alert me if they attack for 2 s with alert threshold 0.9.1",
    "Alert me if they attack for 2 s with alert threshold 0..9",
    "Alert me if they attack for 2 s with alert threshold 9e-1",
    "Alert me if they attack for 2 s with alert threshold 0,9",
    "Alert me if they attack for 2 s with alert threshold 90%%",
    "Alert me if they attack for 2 s with alert threshold 90%x",
    "Alert me if they attack for 2 s with alert threshold -0.9",
    "Alert me if they attack for 2 s with alert threshold +0.9",
    "Alert me if they attack for 2 s with alert threshold 0.9 0.8",
    "Alert me if they attack for 2 s with alert threshold",
    "Alert me if they attack for 2 s with alert threshold nan",
    "Alert me if they attack for 2 s with alert threshold inf",
    "Alert me if they attack for 2.5.1 s",
    "Alert me if they attack for 2e1 s",
    "Alert me if they attack for 2x s",
    "Alert me if they attack for 2,000 s",
    "Alert me if they attack for 1_000 s",
    "Alert me if they attack for 0x10 s",
    "Alert me if they attack for 2 ss",
    "Alert me if they attack for 2 secondss",
])
def test_malformed_numbers_are_refused(text):
    clarification(text)


@pytest.mark.parametrize("text", [
    "Alert me if they attack for ２ seconds",              # fullwidth digit
    "Alert me if they attack for ٢ seconds",              # Arabic-Indic digit
    "Alert me if they attack for ² seconds",              # superscript two
    "Alert me if they attack for 2 seconds",              # no-break space
    "Alert me if they attack for 2 s",                    # thin space
    "Alert me if they attack for ≥ 2 seconds",            # >=
    "Alert me if they attack for 2–3 seconds",            # en dash range
    "Alert me if they attacK for 2 seconds",              # Kelvin sign 'K'
    "Alert me if they аttack for 2 seconds",              # Cyrillic 'a'
    "Alert me if they attack for 2 seconds with ask threshold 0.5％",
    "Alert me if they attack for 2 seconds​",             # zero-width space
    "Alert me if they attack for 2 seconds with alert threshold ½",
])
def test_unicode_lookalikes_are_refused(text):
    c = clarification(text)
    assert c.codes == ("unsupported_phrasing",)


def test_grammar_is_the_only_way_to_a_proposal(monkeypatch):
    """Even with every diagnostic switched off, a sentence outside the grammar
    stays a Clarification: the diagnostics only explain, they never accept."""
    monkeypatch.setattr(agent, "_diagnose", lambda t: [])
    for text in CODEX_REPRODUCTIONS:
        assert isinstance(parse_rule(text, "R1"), Clarification), text
    assert isinstance(parse_rule(EXAMPLE, "R1"), RuleProposal)


def test_several_readings_are_never_guessed(monkeypatch):
    monkeypatch.setattr(agent, "_GRAMMAR", agent._GRAMMAR + agent._GRAMMAR[:1])
    assert isinstance(parse_rule("Alert me when the mice attack for at least 2 seconds", "R1"),
                      Clarification)


def test_repeated_threshold_is_refused_even_with_equal_values():
    c = clarification("Alert me if they attack for 2 s, alert threshold 0.9 and alert threshold 0.9")
    assert c.codes == ("conflicting_thresholds",)


@pytest.mark.parametrize("value, expected", [
    ("0.9", 0.9), (".9", 0.9), ("90%", 0.9), ("90 %", 0.9), ("90 percent", 0.9),
    ("1", 1.0), ("100%", 1.0), ("85.5%", 0.855),
])
def test_legitimate_threshold_values_still_work(value, expected):
    data = proposal(f"Alert me if they attack for 2 s with alert threshold {value}").rule_data()
    assert data["alert_threshold"] == pytest.approx(expected)


@pytest.mark.parametrize("value", ["0.6%", "0.6 %", "0.5 percent", ".9%"])
def test_tiny_percentages_are_refused_as_likely_typos(value):
    c = clarification(f"Alert me if they attack for 2 s with ask threshold {value}")
    assert c.codes == ("ambiguous_threshold",)


# ---- second Codex review: exact decimals, visible duration interpretation ----

@pytest.mark.parametrize("thresholds, msg", [
    ("alert threshold 1.00000000000000000001", "alert_threshold must be between 0 and 1"),
    ("alert threshold .8 and ask threshold .80000000000000000001",
     "ask_threshold must be <= alert_threshold"),
    ("ask threshold 0.80000000000000000001", "ask_threshold must be <= alert_threshold"),  # default 0.8
    ("alert threshold 100.000000000000000001%", "alert_threshold must be between 0 and 1"),
    ("ask threshold 0.00000000000000000000", None),  # exactly 0 is allowed
    ("ask threshold 0.79", None),  # below the default alert 0.8
    ("alert threshold 0.51", None),  # above the default ask 0.5
])
def test_threshold_checks_use_exact_decimals(thresholds, msg):
    """Float rounding must not turn an out-of-range or mis-ordered value valid."""
    result = parse_rule(f"Alert me if they attack for 2 seconds with {thresholds}", "R1")
    if msg is None:
        assert isinstance(result, RuleProposal)
        return
    assert isinstance(result, Clarification)
    assert set(result.codes) == {"invalid_rule"} and msg in result.question


def test_codex_decimal_reproductions():
    for text in ("Alert me if they attack for 2 seconds with alert threshold 1.00000000000000000001",
                 "Alert me if they attack for 2 seconds with alert threshold .8 and "
                 "ask threshold .80000000000000000001"):
        assert isinstance(parse_rule(text, "R1"), Clarification), text


def test_exact_valid_thresholds_still_pass_shared_validation(monkeypatch):
    calls = []
    real = agent.validate_rule
    monkeypatch.setattr(agent, "validate_rule", lambda d: calls.append(dict(d)) or real(d))
    p = proposal("Alert me if they attack for 2 s with alert threshold 0.95 and ask threshold 0.6")
    assert calls and (calls[-1]["alert_threshold"], calls[-1]["ask_threshold"]) == (0.95, 0.6)
    assert isinstance(calls[-1]["alert_threshold"], float)
    assert validate_rule(p.rule_data())[1] == []


@pytest.mark.parametrize("text, note", [
    ("Alert me if they attack for 2 seconds", "Interpreted duration as at least 2 seconds."),
    ("Alert me if they attack for 2 s", "Interpreted duration as at least 2 seconds."),
    ("Notify me when an attack lasts 3 s", "Interpreted duration as at least 3 seconds."),
    ("Tell me if mounting lasts 1 minute", "Interpreted duration as at least 60 seconds."),
    ("Alert me when they attack for 1.5 minutes", "Interpreted duration as at least 90 seconds."),
    ("Alert me if they attack for 0.1 min", "Interpreted duration as at least 6 seconds."),
    ("Alert me if they attack for 2.5 s", "Interpreted duration as at least 2.5 seconds."),
    ("Alert me if they attack for a second", "Interpreted duration as at least 1 second."),
    ("Warn me about attacking for 2 s", "Interpreted duration as at least 2 seconds."),
    ("Alert me when there is an attack lasting 2 s", "Interpreted duration as at least 2 seconds."),
])
def test_implicit_minimum_is_stated(text, note):
    p = proposal(text)
    assert note in p.assumptions
    assert p.assumptions[0] == note
    # the threshold-default note is kept
    assert any("rules.py defaults" in a for a in p.assumptions)


@pytest.mark.parametrize("text", [
    "Alert me if they attack for at least 2 seconds",
    "Alert me if they attack for no less than 2 s",
    "Alert me if they attack for a minimum of 2 s",
    "Alert me if an attack lasts at least 3 s",
    "Alert me if they attack for 2 s or more",
    "Alert me if they attack for 2 s or longer",
    "Log mounts of 5+ seconds",
    "Log investigation bouts of at least 3 s",
])
def test_explicit_minimum_has_no_interpretation_note(text):
    assert not any(a.startswith("Interpreted duration") for a in proposal(text).assumptions)


def test_duration_note_coexists_with_other_assumptions():
    p = proposal("Ask me when they attack for 2 s with alert threshold 0.9")
    assert p.assumptions[0] == "Interpreted duration as at least 2 seconds."
    assert any("ask_threshold = 0.5" in a for a in p.assumptions)
    assert not any("alert_threshold =" in a for a in p.assumptions)  # given explicitly
    assert any("provisional" in a for a in p.assumptions)


# ---- thresholds must survive float conversion unchanged ----

@pytest.mark.parametrize("thresholds", [
    # the two distinct near-1.0 values from the previous report: both -> 1.0
    "alert threshold 0.99999999999999999999 and ask threshold 0.99999999999999999998",
    "alert threshold 0.99999999999999999999",
    "ask threshold 0.79999999999999999999",          # -> 0.8, equal to the default alert
    "alert threshold 0.50000000000000000001",         # -> 0.5, equal to the default ask
    "alert threshold 0.9 and ask threshold 0.90000000000000000001",
    "alert threshold 0.80000000000000001",            # -> 0.8
    "alert threshold 99.9999999999999999999%",        # -> 1.0
    "ask threshold 0.1000000000000000000000001",
])
def test_thresholds_that_float_would_change_are_refused(thresholds):
    c = clarification(f"Alert me if they attack for 2 seconds with {thresholds}")
    assert set(c.codes) == {"invalid_rule"}
    assert "more digits than can be stored exactly" in c.question


def test_distinct_thresholds_that_collapse_to_one_float_are_refused():
    a, b = "0.99999999999999999999", "0.99999999999999999998"
    assert float(a) == float(b) == 1.0  # the collapse this guards against
    c = clarification(f"Alert me if they attack for 2 s with alert threshold {a} and ask threshold {b}")
    assert sum("more digits" in i.message for i in c.issues) == 2


@pytest.mark.parametrize("value, expected", [
    ("0.8", 0.8), ("0.95", 0.95), ("75%", 0.75), (".9", 0.9), ("0.1", 0.1), ("0.3", 0.3),
    ("0.80", 0.8), ("0.800000", 0.8), ("1", 1.0), ("1.0", 1.0), ("0", 0.0), ("0.0", 0.0),
    ("33.3%", 0.333), ("85.5%", 0.855), ("1%", 0.01), ("100%", 1.0),
    ("0.123456789012345", 0.123456789012345),
])
def test_ordinary_decimals_are_not_refused_for_binary_inexactness(value, expected):
    """0.8 is not exact in binary, but repr(0.8) == '0.8', so it is kept."""
    data = proposal(f"Alert me if they attack for 2 s with alert threshold {value} "
                    "and ask threshold 0").rule_data()
    assert data["alert_threshold"] == expected
