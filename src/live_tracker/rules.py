"""Monitoring rules: format, validation, and the researcher-approval gate.

New file for Live Tracker (written by us, not part of the starter).

This is plain deterministic Python. Later the Claude agent will only
*propose* a rule dict from the researcher's sentence; this module decides
whether that dict is valid, and nothing can be monitored until the
researcher approves it.

Example rule (JSON):
{
  "rule_id": "R1",
  "behavior": "attack",
  "min_duration_s": 2.0,
  "alert_threshold": 0.8,
  "ask_threshold": 0.5,
  "alert_mode": "alert",
  "description": "Alert me if the mice attack for at least 2 seconds"
}

Meaning: if the classifier score stays above ask_threshold for at least
min_duration_s, an event is created. If its confidence is >= alert_threshold
we alert; between ask and alert threshold we ask the researcher.
The threshold numbers above are only an example, not tuned values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

# behaviors the classifier will score (CalMS21 Task 1 labels, still to be
# checked against the official readme) plus two non-classifier checks
CLASSIFIER_BEHAVIORS = {"attack", "mount", "investigation"}
OTHER_BEHAVIORS = {"immobility", "tracking_loss"}
BEHAVIORS = CLASSIFIER_BEHAVIORS | OTHER_BEHAVIORS
ALERT_MODES = {"alert", "ask", "log"}
STATUSES = {"proposed", "approved", "rejected"}


@dataclass
class Rule:
    rule_id: str
    behavior: str
    min_duration_s: float
    alert_threshold: float = 0.8
    ask_threshold: float = 0.5
    alert_mode: str = "alert"
    description: str = ""
    status: str = "proposed"  # every new rule starts unapproved

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


def validate_rule(data):
    """Check a rule dict. Returns (Rule or None, list of error strings).

    We never trust input from the language model, so every field is checked.
    Unknown keys are errors too, so typos do not get silently ignored.
    """
    errors = []
    if not isinstance(data, dict):
        return None, ["rule must be a JSON object"]

    allowed = set(Rule.__dataclass_fields__)
    extra = set(data) - allowed
    if extra:
        errors.append(f"unknown fields: {sorted(extra)}")

    for key in ("rule_id", "behavior", "min_duration_s"):
        if key not in data:
            errors.append(f"missing required field: {key}")
    if errors:
        return None, errors

    if not isinstance(data["rule_id"], str) or not data["rule_id"].strip():
        errors.append("rule_id must be a non-empty string")
    if data["behavior"] not in BEHAVIORS:
        errors.append(f"behavior must be one of {sorted(BEHAVIORS)}")

    def number(key, default=None):
        v = data.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{key} must be a number")
            return None
        return float(v)

    dur = number("min_duration_s")
    alert_t = number("alert_threshold", 0.8)
    ask_t = number("ask_threshold", 0.5)

    if dur is not None and dur <= 0:
        errors.append("min_duration_s must be > 0")
    for name, v in (("alert_threshold", alert_t), ("ask_threshold", ask_t)):
        if v is not None and not 0.0 <= v <= 1.0:
            errors.append(f"{name} must be between 0 and 1")
    if alert_t is not None and ask_t is not None and ask_t > alert_t:
        errors.append("ask_threshold must be <= alert_threshold")

    mode = data.get("alert_mode", "alert")
    if mode not in ALERT_MODES:
        errors.append(f"alert_mode must be one of {sorted(ALERT_MODES)}")

    # a rule can never arrive already approved; only approve() does that
    if data.get("status", "proposed") != "proposed":
        errors.append("new rules must have status 'proposed'")

    if errors:
        return None, errors
    rule = Rule(rule_id=data["rule_id"].strip(), behavior=data["behavior"],
                min_duration_s=dur, alert_threshold=alert_t, ask_threshold=ask_t,
                alert_mode=mode, description=str(data.get("description", "")))
    return rule, []


@dataclass
class RuleSet:
    """Holds rules for one monitoring session."""
    rules: dict = field(default_factory=dict)

    def add(self, data):
        rule, errors = validate_rule(data)
        if errors:
            raise ValueError("; ".join(errors))
        if rule.rule_id in self.rules:
            raise ValueError(f"duplicate rule_id {rule.rule_id}")
        self.rules[rule.rule_id] = rule
        return rule

    def approve(self, rule_id):
        """Called only from the researcher's Confirm button."""
        self.rules[rule_id].status = "approved"

    def reject(self, rule_id):
        self.rules[rule_id].status = "rejected"

    def active(self):
        """Only approved rules are used by the monitor."""
        return [r for r in self.rules.values() if r.status == "approved"]

    def to_json(self):
        return json.dumps([r.to_dict() for r in self.rules.values()], indent=2)
