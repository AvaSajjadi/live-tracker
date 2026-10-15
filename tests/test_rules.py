"""Tests for rule validation and the approval gate (new file)."""

import json

import pytest

from live_tracker.rules import RuleSet, validate_rule

GOOD = {"rule_id": "R1", "behavior": "attack", "min_duration_s": 2,
        "alert_threshold": 0.8, "ask_threshold": 0.5, "alert_mode": "alert",
        "description": "Alert me if the mice attack for at least 2 seconds"}


def test_good_rule_passes():
    rule, errors = validate_rule(GOOD)
    assert errors == []
    assert rule.min_duration_s == 2.0
    assert rule.status == "proposed"


def test_defaults_fill_in():
    rule, errors = validate_rule({"rule_id": "R2", "behavior": "mount", "min_duration_s": 1})
    assert errors == [] and rule.alert_mode == "alert"


@pytest.mark.parametrize("change, msg", [
    ({"behavior": "fighting"}, "behavior"),
    ({"min_duration_s": 0}, "min_duration_s"),
    ({"min_duration_s": "2"}, "number"),
    ({"alert_threshold": 1.5}, "between 0 and 1"),
    ({"ask_threshold": 0.9}, "ask_threshold must be <="),
    ({"alert_mode": "email"}, "alert_mode"),
    ({"status": "approved"}, "proposed"),
    ({"min_durration_s": 2}, "unknown fields"),
])
def test_bad_rules_rejected(change, msg):
    rule, errors = validate_rule({**GOOD, **change})
    assert rule is None
    assert any(msg in e for e in errors)


def test_missing_field_rejected():
    _, errors = validate_rule({"rule_id": "R1", "behavior": "attack"})
    assert "missing required field: min_duration_s" in errors


def test_not_a_dict_rejected():
    assert validate_rule("attack 2s")[0] is None


def test_unapproved_rule_is_not_active():
    rs = RuleSet()
    rs.add(GOOD)
    assert rs.active() == []
    rs.approve("R1")
    assert [r.rule_id for r in rs.active()] == ["R1"]


def test_rejected_rule_is_not_active():
    rs = RuleSet()
    rs.add(GOOD)
    rs.reject("R1")
    assert rs.active() == []


def test_duplicate_and_invalid_add_raise():
    rs = RuleSet()
    rs.add(GOOD)
    with pytest.raises(ValueError):
        rs.add(GOOD)
    with pytest.raises(ValueError):
        rs.add({**GOOD, "rule_id": "R9", "behavior": "dance"})


def test_json_round_trip():
    rs = RuleSet()
    rs.add(GOOD)
    data = json.loads(rs.to_json())
    assert data[0]["rule_id"] == "R1" and data[0]["status"] == "proposed"
