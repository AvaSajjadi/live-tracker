"""Researcher sentence -> proposed monitoring rule (new file).

Day 2, part 1: the offline parser (no API key, no network, no LLM). It is
plain deterministic Python: the same sentence always gives the same result.

    >>> result = parse_rule("Alert me when the mice attack for at least 2 seconds", "R1")
    >>> isinstance(result, RuleProposal)
    True
    >>> result.rule_data()["min_duration_s"], result.rule_data()["status"]
    (2.0, 'proposed')

parse_rule() returns exactly one of:

- RuleProposal: a rule dict that already passed rules.validate_rule(), still
  with status "proposed". Nothing is approved here; the researcher's Confirm
  goes through RuleSet.add(proposal.rule_data()) and RuleSet.approve(rule_id).
- Clarification: one or more Issues (a code from CLARIFICATION_CODES and a
  message for the researcher) saying what is missing, ambiguous, conflicting
  or unsupported.

How it parses: whole-sentence grammar
-------------------------------------
A rule is proposed only if the WHOLE sentence matches one of a few supported
sentence forms. Every word must belong to the form; there is no keyword
search on the success path, so an extra clause ("only for the intruder",
"at night", "while grooming", "then ...") or a malformed number makes the
sentence fail, and the parser returns a Clarification instead of dropping
the part it does not understand. Only ASCII text is accepted (curly quotes
aside), so look-alike Unicode digits, signs or letters cannot slip through.

The forms (ACTION, SUBJECT, ... defined below; [x] = optional):

  F1 [please] ACTION when|if|whenever SUBJECT VERB for DURATION
       "Alert me when the mice attack for at least 2 seconds"
  F2 [please] ACTION when|if|whenever [a|an|the|any] NOUN lasts|continues [for] DURATION
       "Notify me if an attack lasts at least 3 s", "Tell me if mounting lasts at least 1 minute"
  F3 [please] ACTION when|if|whenever there is|there's a|an|any NOUN lasting|that lasts [for] DURATION
  F4 [please] ACTION [about|of|on] [any] NOUNS lasting|that last|which last [for] DURATION
     [please] ACTION [about|of|on] [any] NOUNS of MIN_DURATION
     [please] ACTION [about|of|on] [any] GERUND for DURATION
       "Log investigation bouts of 3 s or more", "Warn me about attacking for 2 s or more"
     ("Log attacks for 2 s" is refused: "for 2 s" could be how long to log.)

  each optionally followed by: [,] [with|and] THRESHOLD [[,] and THRESHOLD] [. ! ?]

- ACTION (rules.ALERT_MODES): alert = "alert me", "notify me", "warn me",
  "tell me", "let me know"; ask = "ask me", "ask me to confirm", "check with
  me"; log = "log", "record". alert_mode "ask" is provisional (team agreement
  pending); the parser only sets the field.
- SUBJECT: they, the mice, mice, the two mice, both mice. Named animals
  ("the resident", "mouse A") are not supported: a rule cannot say who does
  the behavior.
- behavior: attack, mount, investigation (rules.CLASSIFIER_BEHAVIORS) in
  verb ("attack", "are mounting", "investigate each other"), noun ("an attack",
  "mount bout", "investigation episode", "mounting") or plural form
  ("attacks", "investigation bouts").
- DURATION is an inclusive minimum, as in the schema (the monitor fires once
  the bout has lasted >= min_duration_s). Without a minimum marker ("for 2 s",
  "lasts 2 s") it is read as a minimum and the proposal's assumptions say
  "Interpreted duration as at least 2 seconds.": [at least | no less than | not less
  than | a minimum of] NUMBER[+] UNIT [or more | or longer]. MIN_DURATION
  (after "of") must carry one of those minimum markers. NUMBER is digits
  ("2", "1.5", ".5") or a word from one to ninety ("a minute"); UNIT is s, sec,
  secs, second(s), min, mins, minute(s). Strict ("more than", ">", "over"),
  maximum ("or less", "under", "within"), exact ("exactly") and range forms
  are refused, as are other units and the "a 3-second attack" adjective form
  (it does not say whether 3 s is a minimum).
- THRESHOLD: [a|an|the] alert|ask threshold [of|at|set to|=|:] VALUE, VALUE =
  digits with an optional "%" or " percent" (a percentage below 1 %, e.g.
  "0.6 %", is refused as a likely typo for 0.6). Bounds and ordering are
  checked on the exact decimal typed, before float conversion, so
  "1.00000000000000000001" is refused instead of rounding to 1.0, and a value
  that float conversion would change (more digits than a float keeps, e.g.
  0.99999999999999999999 -> 1.0) is refused too. Thresholds that are not given
  come from the rules.py defaults (example values, not tuned) and are listed
  in the proposal's assumptions.

When the grammar does not match, keyword scans (behavior, action, duration,
negation / suppression, unnamed confidence conditions, other numbers) are used
only to explain WHY, so the researcher gets a specific question. They never
turn a non-matching sentence into a rule.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping, Union

from .rules import CLASSIFIER_BEHAVIORS, Rule, validate_rule

__all__ = ["CLARIFICATION_CODES", "Clarification", "Issue", "ParseResult",
           "RuleProposal", "parse_rule"]

MAX_INSTRUCTION_CHARS = 500

CLARIFICATION_CODES = (
    "empty_instruction",      # blank text
    "instruction_too_long",   # more than MAX_INSTRUCTION_CHARS
    "unsupported_phrasing",   # not one of the supported sentence forms, or non-ASCII text
    "no_behavior",            # no behavior word at all
    "unsupported_behavior",   # a behavior the classifier does not score
    "multiple_behaviors",     # more than one classifier behavior
    "missing_duration",       # no minimum duration
    "ambiguous_duration",     # a time we cannot read or place
    "unsupported_duration",   # strict / maximum / exact / range / unit we cannot use
    "conflicting_durations",  # more than one duration clause
    "missing_action",         # not said whether to alert, ask or log
    "conflicting_actions",    # more than one of alert / ask / log
    "ambiguous_threshold",    # confidence / score condition without a named threshold
    "conflicting_thresholds", # the same threshold given more than once
    "negation",               # "don't", "avoid", "disable", "unless", ...
    "unsupported_condition",  # other numbers, e.g. counts ("3 times")
    "invalid_rule",           # the assembled rule failed rules.validate_rule()
)


@dataclass(frozen=True)
class Issue:
    """One reason the sentence could not become a rule."""
    code: str     # one of CLARIFICATION_CODES
    message: str  # shown to the researcher


@dataclass(frozen=True)
class Clarification:
    """The parser needs the researcher to rephrase or add information."""
    source_text: str
    issues: tuple[Issue, ...]

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(i.code for i in self.issues)

    @property
    def question(self) -> str:
        """All issue messages as one text for the researcher."""
        return " ".join(i.message for i in self.issues)


@dataclass(frozen=True)
class RuleProposal:
    """A validated, unapproved rule. rule_dict is read-only; use rule_data()
    (a fresh dict for RuleSet.add) or rule (a fresh Rule)."""
    source_text: str
    rule_dict: Mapping[str, object]
    assumptions: tuple[str, ...]

    def rule_data(self) -> dict:
        return dict(self.rule_dict)

    @property
    def rule(self) -> Rule:
        return Rule(**self.rule_dict)


ParseResult = Union[RuleProposal, Clarification]


# ---- the grammar (success path) ----

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORDS = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_DIGITS = r"\d+(?:\.\d+)?|\.\d+"
_SECONDS = ("seconds", "second", "secs", "sec", "s")
_MINUTES = ("minutes", "minute", "mins", "min")

_G_ACTIONS = {
    "alert": r"alert me|notify me|warn me|tell me|let me know",
    "ask": r"ask me to confirm|ask me|check with me",
    "log": r"log|record",
}
_G_BEHAVIORS = {
    "attack": {
        "verb": r"attack(?: each other)?|are attacking(?: each other)?",
        "noun": r"attack(?: bout| episode)?|attacking",
        "nouns": r"attacks|attack (?:bouts|episodes)|attacking",
        "gerund": r"attacking",
    },
    "mount": {
        "verb": r"mount(?: each other)?|are mounting(?: each other)?",
        "noun": r"mount(?:ing)? (?:bout|episode)|mount|mounting",
        "nouns": r"mounts|mount(?:ing)? (?:bouts|episodes)|mounting",
        "gerund": r"mounting",
    },
    "investigation": {
        "verb": r"investigate(?: each other)?|are investigating(?: each other)?",
        "noun": r"investigation(?: bout| episode)?|investigating",
        "nouns": r"investigations|investigation (?:bouts|episodes)|investigation|investigating",
        "gerund": r"investigating",
    },
}
assert set(_G_BEHAVIORS) == CLASSIFIER_BEHAVIORS

_G_PRE = r"(?:(?:please|can you|could you|can you please|could you please) )?"
_G_ACTION = "(?P<action>" + "|".join(rf"(?P<a_{a}>{p})" for a, p in _G_ACTIONS.items()) + ")"
_G_CONN = r"(?:when|if|whenever)"
_G_SUBJECT = r"(?:they|the mice|mice|the two mice|both mice)"
_G_MIN_QUAL = r"(?P<qual>at least|no less than|not less than|a minimum of) "
_G_NUMBER = rf"(?:(?P<num>{_DIGITS})(?P<plus>\+)? ?|(?P<wnum>{_WORDS}) )"
_G_UNIT = "(?P<unit>" + "|".join(_SECONDS + _MINUTES) + ")"
_G_DURATION = rf"(?:{_G_MIN_QUAL})?{_G_NUMBER}{_G_UNIT}(?P<post> or (?:more|longer))?"
_G_VALUE = rf"(?:{_DIGITS})(?: ?%| percent)?"
_G_THRESHOLD = rf"(?:(?:an?|the) )?(?:alert|ask)[ _]threshold(?: of | at | set to | ?[=:] ?| ){_G_VALUE}"
_G_TAIL = (rf"(?P<thr>(?:,? (?:with |and )?|, ){_G_THRESHOLD}(?:(?:,? and |, ){_G_THRESHOLD})?)?"
           r"[.!?]?")
_G_THRESHOLD_ITEM = re.compile(
    rf"(?P<which>alert|ask)[ _]threshold(?: of | at | set to | ?[=:] ?| )"
    rf"(?P<value>{_DIGITS})(?P<pct> ?%| percent)?", re.ASCII)


def _forms(b):
    """The supported sentence forms for behavior b (see module docstring)."""
    g = _G_BEHAVIORS[b]
    head = rf"{_G_PRE}{_G_ACTION}"
    return [
        rf"{head} {_G_CONN} {_G_SUBJECT} (?:{g['verb']}) for {_G_DURATION}",
        rf"{head} {_G_CONN} (?:(?:an?|the|any) )?(?:{g['noun']}) (?:lasts|continues) (?:for )?{_G_DURATION}",
        rf"{head} {_G_CONN} there(?: is|'s) (?:an?|any) (?:{g['noun']}) (?:lasting|that lasts) "
        rf"(?:for )?{_G_DURATION}",
        rf"{head} (?:(?:about|of|on) )?(?:any )?(?:(?:{g['nouns']}) "
        rf"(?:(?:lasting|that last|which last) (?:for )?|(?P<of>of ))|(?:{g['gerund']}) for ){_G_DURATION}",
    ]


_GRAMMAR = [(b, re.compile(form + _G_TAIL, re.ASCII))
            for b in sorted(_G_BEHAVIORS) for form in _forms(b)]

_SUPPORTED_LIST = ", ".join(sorted(CLASSIFIER_BEHAVIORS))
_FORMS_HELP = ("I can only turn sentences of these forms into a rule: 'Alert me when the mice "
               "attack for at least 2 seconds', 'Ask me if a mount lasts at least 3 s', 'Log "
               "investigation bouts of 5 seconds or more', optionally followed by 'with alert "
               "threshold 0.9 and ask threshold 0.6'. Actions: alert me / ask me / log. "
               f"Behaviors: {_SUPPORTED_LIST}. A rule cannot be limited to one animal, a place, "
               "a time of day, a context or a count, and has exactly one minimum duration.")


def _match_grammar(t):
    """Return (behavior, match) for a full-sentence match, else (None, longest
    partial match or None)."""
    full, best = [], None
    for behavior, rx in _GRAMMAR:
        m = rx.match(t)
        if m is None:
            continue
        if m.end() == len(t):
            full.append((behavior, m))
        elif best is None or m.end() > best.end():
            best = m
    if len(full) == 1:
        return full[0]
    # several readings (should not happen while the forms stay disjoint): never guess
    return None, (full[0][1] if full else best)


# ---- diagnostics (only used to explain a sentence the grammar refused) ----

_BEHAVIOR_WORDS = {
    "attack": r"attack(?:s|ed|ing|er|ers)?",
    "mount": r"mount(?:s|ed|ing)?",
    "investigation": r"investigat(?:e|es|ed|ing|ion|ions)",
}

# (pattern, name shown to the researcher, hint) for behaviors we must not map
_UNSUPPORTED_BEHAVIORS = (
    (r"fight(?:s|ing)?", "fighting", "attack"),
    (r"aggressi(?:on|ve)", "aggression", "attack"),
    (r"bit(?:e|es|ing)", "biting", "attack"),
    (r"chas(?:e|es|ed|ing)", "chasing", None),
    (r"sniff(?:s|ed|ing)?", "sniffing", "investigation"),
    (r"mat(?:e|es|ed|ing)", "mating", "mount"),
    (r"groom(?:s|ed|ing)?", "grooming", None),
    (r"rearing", "rearing", None),
    (r"freez(?:e|es|ing)|froze", "freezing", None),
    (r"immobil(?:e|ity)|motionless", "immobility", None),
    (r"tracking[\s_]+(?:loss|lost)|lost\s+tracking|los(?:e|es|ing)\s+tracking", "tracking loss", None),
    (r"sleep(?:s|ing)?|asleep", "sleeping", None),
    (r"eat(?:s|ing)?|drink(?:s|ing)?", "eating/drinking", None),
)

_ACTIONS = {
    "alert": r"alert(?:s|ed|ing)?|notif(?:y|ied|ying|ication|ications)|warn(?:s|ed|ing)?"
             r"|tell\s+me|let\s+me\s+know",
    "ask": r"ask(?:\s+me)?|check\s+with\s+me|confirm\s+with\s+me",
    "log": r"log(?:s|ged|ging)?|records?",
}

_COUNT_WORDS = "|".join(sorted(set(_NUMBER_WORDS) - {"a", "an"}, key=len, reverse=True))
_UNITS = ("milliseconds|millisecond|msecs|msec|ms|minutes|minute|mins|min|"
          "seconds|second|secs|sec|s|hours|hour|hrs|hr|h|frames|frame")

_MIN_QUALS = (r"at\s+least|a\s+minimum\s+of|minimum\s+of|no\s+less\s+than|not\s+less\s+than"
              r"|no\s+shorter\s+than|not\s+shorter\s+than")
_STRICT_QUALS = r"more\s+than|longer\s+than|greater\s+than|in\s+excess\s+of|exceeding|over|above"
_MAX_QUALS = (r"no\s+more\s+than|not\s+more\s+than|no\s+longer\s+than|not\s+longer\s+than"
              r"|less\s+than|fewer\s+than|shorter\s+than|under|below|at\s+most|up\s+to"
              r"|a\s+maximum\s+of|maximum\s+of|within")
_EXACT_QUALS = r"exactly|precisely"

_DURATION = re.compile(rf"""
    (?P<lead>\b(?:(?:that\s+|which\s+)?(?:lasts?|lasted|lasting)(?:\s+for)?
              |(?:sustained|continues|continuing|persists|persisting|keeps\s+going)\s+for
              |for
              |(?:a\s+)?(?:min(?:imum)?\s+)?duration\s*(?:of|is|=|:)?
              )\s*)?
    (?P<qual>(?:\b(?:{_MIN_QUALS}|{_STRICT_QUALS}|{_MAX_QUALS}|{_EXACT_QUALS})\b
              |>=|=>|<=|=<|>|<)\s*)?
    (?P<sign>(?<![\w.,\-])(?:-\s*|\bminus\s+))?
    (?(sign)|(?<![\w.,\-]))(?P<num>{_DIGITS}|\b(?:{_WORDS})\b)
    \s*(?P<plus>\+)?\s*-?\s*
    (?P<unit>{_UNITS})(?!\w)
    (?P<post>\s*(?:\+|\bor\s+(?:more|longer)\b|\band\s+(?:more|longer|up)\b))?
    (?P<badpost>\s*(?:\bor\s+(?:less|shorter|fewer|under|below)\b|\bexactly\b|\bprecisely\b
                     |\bat\s+most\b|\bmax(?:imum)?\b|\bat\s+the\s+most\b))?
""", re.X | re.ASCII)

_QUAL_KIND = [(re.compile(rf"^(?:{p})$"), kind) for p, kind in (
    (_MIN_QUALS + r"|>=|=>", "min"),
    (_STRICT_QUALS + r"|>", "strict"),
    (_MAX_QUALS + r"|<=|=<|<", "max"),
    (_EXACT_QUALS, "exact"),
)]

# a number right before the match means a range ("2-3 s", "2 to 3 s") or a
# compound number ("twenty five seconds")
_NUMBER_BEFORE = re.compile(rf"(?:\d|\b(?:{_COUNT_WORDS})\b)[\s,]*(?:-|\bto\b|\bor\b|\band\b)?\s*$")
_BETWEEN_BEFORE = re.compile(r"\bbetween\b[^.;]*\band\s*$")
_ORDINAL_AFTER = re.compile(r"\s+times?\b")
_ORPHAN_UNIT = re.compile(r"\b(?:seconds?|secs?|minutes?|mins|milliseconds?|msecs?|hours?|hrs?|frames?)\b")

_THRESHOLD = re.compile(rf"""
    \b(?P<which>alert|ask)[\s_-]*threshold\b\s*
    (?:(?:of|at|to|is|=|:|set\s+to|equal\s+to)\s*)?
    (?P<num>-?(?:{_DIGITS})[^\s,]*?)(?=[.!?]?(?:[\s,]|$))
""", re.X | re.ASCII)
_UNNAMED_THRESHOLD = re.compile(
    r"\b(?:threshold|confidence|confident|probabilit(?:y|ies)|prob|scores?|certain(?:ty)?|sure"
    r"|likelihood|likely)\b")
_NEGATION = re.compile(
    r"\b(?:not|no|never|don't|dont|do\s+not|doesn't|does\s+not|isn't|stop|without|except"
    r"|unless|ignore|neither|nor|avoid(?:s|ing)?|disabl(?:e|es|ed|ing)|refrain(?:ing)?"
    r"|suppress(?:es|ed|ing)?|mute|silenc(?:e|ing)|skip(?:ping)?|turn\s+off|switch\s+off"
    r"|cancel|prevent|block|no\s+longer)\b|n't\b")
_MICE = re.compile(r"\b(?:one|two|both)\s+(?:of\s+the\s+)?(?:mouse|mice|animals?)\b")
_LEFTOVER_NUMBER = re.compile(rf"\d|\b(?:{_COUNT_WORDS}|twice|thrice|times)\b")


def _words(pattern):
    return re.compile(rf"\b(?:{pattern})\b")


_BEHAVIOR_RES = {b: _words(p) for b, p in _BEHAVIOR_WORDS.items()}
_UNSUPPORTED_RES = [(_words(p), name, hint) for p, name, hint in _UNSUPPORTED_BEHAVIORS]
_ACTION_RES = {a: _words(p) for a, p in _ACTIONS.items()}
_ACTION_QUESTION = ("Should I alert you, ask you to confirm each event, or only log it? "
                    "Say e.g. 'alert me', 'ask me' or 'log'.")
_DURATION_EXAMPLE = "e.g. 'for at least 2 seconds'"


def _blank(text, spans):
    """Replace spans with spaces (keeps every other offset unchanged)."""
    chars = list(text)
    for a, b in spans:
        chars[a:b] = " " * (b - a)
    return "".join(chars)


def _qual_kind(qual):
    for pattern, kind in _QUAL_KIND:
        if pattern.match(qual):
            return kind
    raise AssertionError(f"unclassified qualifier {qual!r}")  # the regexes are out of sync


def _diagnose_thresholds(t, issues):
    spans, seen = [], {}
    for m in _THRESHOLD.finditer(t):
        spans.append(m.span())
        seen[m["which"]] = seen.get(m["which"], 0) + 1
        if not _G_THRESHOLD_ITEM.fullmatch(t, m.start(), m.end()):
            issues.append(Issue("unsupported_phrasing",
                                f"I could not read the threshold value in '{m.group(0)}'. Use a plain "
                                "number like 0.9 or a percentage like 90%."))
    for which, n in seen.items():
        if n > 1:
            issues.append(Issue("conflicting_thresholds",
                                f"The {which} threshold is given {n} times. Give it once."))
    return spans


def _diagnose_durations(t, issues):
    spans, n_ok = [], 0
    for m in _DURATION.finditer(t):
        spans.append(m.span())
        snippet = m.group(0).strip()
        before, after = t[:m.start()], t[m.end():]
        kind = _qual_kind(m["qual"].strip()) if m["qual"] else None
        bare = not m["lead"] and not m["qual"]
        if bare and (_NUMBER_BEFORE.search(before) or _BETWEEN_BEFORE.search(before)):
            issues.append(Issue("ambiguous_duration",
                                f"I could not read the duration near '{snippet}' (a range or a number "
                                f"in several words?). Give one minimum duration in digits, {_DURATION_EXAMPLE}."))
        elif m["sign"]:
            issues.append(Issue("unsupported_duration",
                                f"'{snippet}': a duration must be a positive number of seconds or minutes."))
        elif m["unit"] not in _SECONDS and m["unit"] not in _MINUTES:
            issues.append(Issue("unsupported_duration",
                                f"'{snippet}': please give the duration in seconds or minutes."))
        elif _ORDINAL_AFTER.match(after):
            issues.append(Issue("ambiguous_duration",
                                f"'{snippet}' looks like part of a count, not a duration. Give how long "
                                f"the behavior must last, {_DURATION_EXAMPLE}."))
        elif kind in ("max", "exact") or m["badpost"]:
            issues.append(Issue("unsupported_duration",
                                f"'{snippet}': a rule can only require that the behavior lasts AT LEAST "
                                f"some time ({_DURATION_EXAMPLE}), not at most or exactly."))
        elif kind == "strict":
            issues.append(Issue("unsupported_duration",
                                f"'{snippet}': a rule's duration is an inclusive minimum (the monitor "
                                "reacts once the behavior has lasted that long), so it cannot express "
                                "'more than'. Say 'at least' with the duration you want."))
        elif not (kind == "min" or m["lead"] or m["plus"] or m["post"]):
            issues.append(Issue("ambiguous_duration",
                                f"I found '{snippet}' but not what it refers to. If it is how long the "
                                f"behavior must last, say {_DURATION_EXAMPLE}."))
        else:
            n_ok += 1
    if len(spans) > 1:
        issues.append(Issue("conflicting_durations",
                            "The sentence has more than one duration. A rule has exactly one minimum "
                            "duration, and it must describe the behavior being watched."))
    return spans, n_ok


def _diagnose(t):
    """Specific reasons a sentence the grammar refused cannot be a rule."""
    issues = []
    t = _blank(t, _diagnose_thresholds(t, issues))
    spans, n_ok = _diagnose_durations(t, issues)
    t = _blank(t, spans)
    if _ORPHAN_UNIT.search(t):
        issues.append(Issue("ambiguous_duration",
                            "I could not read a duration in the sentence. Give it as a number of "
                            f"seconds or minutes, {_DURATION_EXAMPLE}."))
    elif not spans:
        issues.append(Issue("missing_duration",
                            f"How long must the behavior last before the rule reacts? {_DURATION_EXAMPLE}."))

    behaviors = [b for b, rx in _BEHAVIOR_RES.items() if rx.search(t)]
    unsupported = [(name, hint) for rx, name, hint in _UNSUPPORTED_RES if rx.search(t)]
    for name, hint in unsupported:
        msg = f"'{name}' is not a behavior the classifier scores (it scores {_SUPPORTED_LIST})."
        if name in ("immobility", "tracking loss"):
            msg += " The monitor only watches classifier scores; tracking warnings are logged separately."
        elif hint:
            msg += f" If you mean '{hint}', please say '{hint}'."
        issues.append(Issue("unsupported_behavior", msg))
    if len(behaviors) > 1:
        issues.append(Issue("multiple_behaviors",
                            f"The sentence mentions {', '.join(behaviors)}. One rule watches one behavior; "
                            "please write one sentence per behavior."))
    elif not behaviors and not unsupported:
        issues.append(Issue("no_behavior",
                            f"Which behavior should I watch? The classifier scores {_SUPPORTED_LIST}."))

    actions = [a for a, rx in _ACTION_RES.items() if rx.search(t)]
    if len(actions) > 1:
        issues.append(Issue("conflicting_actions",
                            f"The sentence asks for several actions ({', '.join(actions)}). A rule has "
                            "one: " + _ACTION_QUESTION))
    elif not actions:
        issues.append(Issue("missing_action", _ACTION_QUESTION))

    if _UNNAMED_THRESHOLD.search(t):
        issues.append(Issue("ambiguous_threshold",
                            "I can't tell which threshold the confidence / score condition means. Say "
                            "'alert threshold 0.9' (confidence needed to alert instead of ask) or "
                            "'ask threshold 0.6' (score a frame needs to count as the behavior)."))
    if _NEGATION.search(t):
        issues.append(Issue("negation",
                            "A rule can only ask for an alert, a question or a log entry when a behavior "
                            "happens; it cannot suppress alerts or express 'not', 'unless', 'except', "
                            "'avoid', 'disable', etc. Please rephrase as what you want to happen."))
    if _LEFTOVER_NUMBER.search(_blank(t, [m.span() for m in _MICE.finditer(t)])):
        issues.append(Issue("unsupported_condition",
                            "The sentence has a number or count I can't use in a rule (only one minimum "
                            "duration and named thresholds). Please remove it or rephrase."))
    return issues


# ---- building the rule from a full match ----

def _seconds(m):
    value = Decimal(m["num"]) if m["num"] is not None else Decimal(_NUMBER_WORDS[m["wnum"]])
    return float(value * (60 if m["unit"] in _MINUTES else 1))


def _implicit_minimum(m):
    """True when the duration has no explicit minimum marker ("for 2 s",
    "lasts 2 s"), so reading it as an inclusive minimum is our interpretation."""
    return not (m["qual"] or m["plus"] or m["post"])


def _seconds_text(seconds):
    """Plain decimal text without exponent: 2.0 -> '2', 1e-07 -> '0.0000001'."""
    text = format(Decimal(repr(seconds)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


_DEFAULT_THRESHOLDS = {k: Decimal(repr(Rule.__dataclass_fields__[k].default))
                       for k in ("alert_threshold", "ask_threshold")}


def _exact_threshold_errors(values):
    """Bounds and ordering on the exact decimals the researcher typed, before
    any float rounding (1.00000000000000000001 would round to 1.0). Messages
    match rules.validate_rule. Unset thresholds use the rules.py defaults."""
    errors = []
    for key, v in values.items():
        # repr() is the shortest decimal that round-trips the float, so 0.8
        # or 75% pass while 0.99999999999999999999 (-> 1.0) is refused
        stored = Decimal(repr(float(v)))
        if stored != v:
            errors.append(f"{key} {v} has more digits than can be stored exactly "
                          f"(it would become {stored}); please use fewer decimal places")
        if not Decimal(0) <= v <= Decimal(1):
            errors.append(f"{key} must be between 0 and 1")
    full = {**_DEFAULT_THRESHOLDS, **values}
    if full["ask_threshold"] > full["alert_threshold"]:
        errors.append("ask_threshold must be <= alert_threshold")
    return errors


def _thresholds(m):
    """Named thresholds of a full match -> ({field: exact Decimal}, issues)."""
    values, issues = {}, []
    if m["thr"]:
        for item in _G_THRESHOLD_ITEM.finditer(m["thr"]):
            key = f"{item['which']}_threshold"
            if key in values:
                issues.append(Issue("conflicting_thresholds",
                                    f"The {item['which']} threshold is given more than once. Give it once."))
                continue
            value = Decimal(item["value"])
            if item["pct"] and value < 1:
                # "0.6 %" is 0.006: almost surely a fraction typed with a % sign
                issues.append(Issue("ambiguous_threshold",
                                    f"Did you mean {item['which']} threshold {item['value']} or "
                                    f"{item['value']}% (= {value / 100:g})? Percentages below 1% are "
                                    "not accepted; write the fraction, e.g. 0.6, or e.g. 60%."))
                continue
            values[key] = value / (100 if item["pct"] else 1)
    return values, issues


# ---- public API ----

def parse_rule(text: str, rule_id: str) -> ParseResult:
    """Turn one researcher instruction into a RuleProposal or a Clarification.

    A RuleProposal is returned only if the whole sentence matches a supported
    form (module docstring) and the rule passes rules.validate_rule().
    rule_id is chosen by the caller (it must be unique in the RuleSet); the
    sentence is kept as the rule's description. Offline and deterministic.
    The returned rule is never approved: approval only happens through
    RuleSet.approve() after the researcher confirms.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    source = text.strip()
    if not source:
        return Clarification(source, (Issue("empty_instruction",
                                            "Please type what you want to monitor, e.g. "
                                            "'Alert me when the mice attack for at least 2 seconds'."),))
    if len(source) > MAX_INSTRUCTION_CHARS:
        return Clarification(source, (Issue("instruction_too_long",
                                            f"Please describe one rule in at most "
                                            f"{MAX_INSTRUCTION_CHARS} characters."),))

    t = source.replace("’", "'").replace("‘", "'")
    odd = sorted({c for c in t if not c.isascii()})
    if odd:
        return Clarification(source, (Issue("unsupported_phrasing",
                                            "Please use plain ASCII text; I can't read these characters: "
                                            + " ".join(f"'{c}' (U+{ord(c):04X})" for c in odd) + "."),))
    t = re.sub(r"\s+", " ", t.lower())

    behavior, m = _match_grammar(t)
    if behavior is None:
        issues = _diagnose(t)
        if not issues:
            msg = _FORMS_HELP
            if m is not None and m.end() < len(t):
                msg = f"I can't represent '{t[m.end():].strip()}' in a rule. " + msg
            issues = [Issue("unsupported_phrasing", msg)]
        return Clarification(source, tuple(issues))

    if m.groupdict().get("of") and not (m["qual"] or m["plus"] or m["post"]):
        return Clarification(source, (Issue(
            "ambiguous_duration",
            "'of N seconds' does not say whether N is a minimum. Say e.g. 'of at least 2 seconds' "
            "or 'of 2 seconds or more'."),))
    exact, issues = _thresholds(m)
    if issues:
        return Clarification(source, tuple(issues))
    exact_errors = _exact_threshold_errors(exact)
    # without errors, each float has exactly the decimal value typed (checked
    # above), so the exact checks hold for it; validate_rule checks it again
    thresholds = {k: float(v) for k, v in exact.items()}
    action = next(a for a in _G_ACTIONS if m[f"a_{a}"] is not None)
    data = {"rule_id": rule_id, "behavior": behavior, "min_duration_s": _seconds(m),
            "alert_mode": action, "description": source, **thresholds}

    rule, errors = validate_rule(data)
    errors = exact_errors + [e for e in errors if e not in exact_errors]
    # rules.validate_rule does not reject inf / nan yet; the monitor would
    for key in ("min_duration_s", "alert_threshold", "ask_threshold"):
        if key in data and not math.isfinite(data[key]):
            errors.append(f"{key} must be finite")
    if errors:
        return Clarification(source, tuple(Issue("invalid_rule", e) for e in errors))

    assumptions = []
    if _implicit_minimum(m):
        n = _seconds_text(rule.min_duration_s)
        assumptions.append(f"Interpreted duration as at least {n} {'second' if n == '1' else 'seconds'}.")
    defaults = [k for k in ("alert_threshold", "ask_threshold") if k not in thresholds]
    if defaults:
        assumptions.append(
            "Not given, so the rules.py defaults are used (example values, not tuned): "
            + ", ".join(f"{k} = {getattr(rule, k):g}" for k in defaults) + ".")
    if rule.alert_mode == "ask":
        assumptions.append("alert_mode 'ask' is provisional: its exact meaning still needs team agreement.")
    return RuleProposal(source, MappingProxyType(rule.to_dict()), tuple(assumptions))
