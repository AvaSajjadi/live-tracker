"""A tolerant reader for rule sentences: free, offline, no key (new file).

The strict parser (agent.parse_rule) understands one family of sentences. A biologist should not have to learn it. This
reader tries the strict parser first. If that refuses, it rewrites the sentence into the strict form in small, safe steps,
and parses the rewritten sentence with the same strict parser:

    1. repair typos ("secnds", "attck", "more that") against a small closed vocabulary (one unique close word only)
    2. remove polite filler ("please", "I would like you to", "could you")
    3. everyday words for the action ("let me know", "ping me", "be notified" -> alert me) and for the behavior
       ("fighting", "aggression", "biting" -> attack; "sniffing" -> investigation; "mating" -> mount)
    4. other ways of saying a minimum ("more than 2 seconds", "over 2 seconds", "2 seconds straight")
    5. word order ("when the mice fight for 2 seconds, let me know")

Every step that changed something is written down as an assumption, together with the sentence the rule was really read
from, so the researcher sees exactly what was understood and still clicks Confirm.

What it never does: guess a missing number ("for a while"), turn "less than", "under", "at most", "exactly" into a minimum,
drop a place, a time of day or an "if" condition, or map a behavior the classifier does not score (grooming, chasing).
Those are refused by the strict parser with its own reason, and that reason is returned.
"""

from __future__ import annotations

import re
from types import MappingProxyType

from . import agent
from .agent import Clarification, RuleProposal
from .rules import validate_rule

__all__ = ["read_flexibly", "repair_typos"]

# ---- 1. typos ----

_VOCAB = frozenset("""
alert alerts alerted notify notified warn warned tell told let know ask confirm check log record message text ping inform
informed signal alarm flag attack attacks attacking attacked mount mounts mounting investigate investigates investigating
investigation investigations bout bouts episode episodes fight fights fighting aggression aggressive biting bite sniff sniffing
sniffs explore exploring mating mate seconds second secs sec minutes minute mins min least more longer than over above
exceeding excess minimum when whenever there lasting lasts continues continuing each other with and threshold percent
please kindly could would both mice mouse animals the two they them for straight consecutive continuously
""".split())
_PROTECTED = frozenset("""that this then them what with from have been will your mine mind rice dice nice mile wall near side time
times night days seconds hour hours half area into onto upon over under below less most exact exactly maximum within""".split())


_SHORT_SLIPS = {"wen": "when", "whn": "when", "teh": "the", "hte": "the", "adn": "and"}


def _distance(a, b):
    """Edit distance where swapping two neighbouring letters costs 1 (the usual typing slip)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def repair_typos(t):
    """Replace a word that is one slip away from exactly one vocabulary word. Returns (text, [(wrong, right)])."""
    changes = []

    def fix(m):
        word = m.group()
        if len(word) < 4 or word in _VOCAB or word in _PROTECTED or any(c.isdigit() for c in word):
            return word
        limit = 1 if len(word) < 8 else 2
        scored = sorted((_distance(word, v), v) for v in _VOCAB if abs(len(v) - len(word)) <= limit)
        near = [v for dist, v in scored if dist <= limit]
        best = [v for dist, v in scored if dist == scored[0][0] and dist <= limit]
        if len(best) != 1 or not near:
            return word                                   # nothing close, or two equally close words: do not guess
        changes.append((word, best[0]))
        return best[0]

    return re.sub(r"[a-z']+", fix, t), changes


# ---- 2 to 5. rewrites ----

_FILLER = re.compile(
    r"\b(?:please|kindly|hey|hi|hello|thanks|thank you|i (?:would|'d) like (?:you )?to|i want you to|i need you to|"
    r"i would like|i'd like|i want|i need|make sure (?:you|to)|can you|could you|would you|will you|just|really|"
    r"at any time|any time|at all|ever)\b,?\s*")

_ACTION_WORDS = [
    (r"(?:to )?(?:be|get) (?:alerted|notified|warned|informed|told)", "alert me"),
    (r"to know", "alert me"),
    (r"(?:message|text|ping|buzz|inform|signal|alarm) me", "alert me"),
    (r"(?:send|give) me (?:an? )?(?:alert|alarm|notification|warning|message)", "alert me"),
    (r"(?:raise|sound|trigger) (?:an? |the )?(?:alert|alarm)", "alert me"),
    (r"confirm with me|ask me to check", "ask me"),
    (r"(?:write down|make a note of|keep a log of|note)", "log"),
]
_BEHAVIOR_WORDS = [
    (r"fighting", "attacking", "attack"), (r"fights", "attacks", "attack"), (r"fight", "attack", "attack"),
    (r"aggressive|violent", "attacking", "attack"), (r"aggression|violence", "attack", "attack"),
    (r"biting", "attacking", "attack"), (r"bites", "attacks", "attack"), (r"bite", "attack", "attack"),
    (r"sniffing", "investigating", "investigation"), (r"sniffs", "investigate", "investigation"),
    (r"sniff", "investigate", "investigation"), (r"exploring", "investigating", "investigation"),
    (r"explore", "investigate", "investigation"),
    (r"mating", "mounting", "mount"), (r"mates", "mount", "mount"), (r"mate", "mount", "mount"),
]
_SUBJECTS = [(r"(?:my|the|these|those) (?:mice|animals|rodents)", "the mice"), (r"both (?:of )?(?:the )?(?:mice|animals)", "both mice"),
             (r"(?:the )?(?:two|2) (?:animals|mice)", "the two mice"), (r"the animals", "the mice")]
_UNIT = r"(?P<unit>seconds?|secs?|s|minutes?|mins?)"
_NUM = rf"(?P<num>\d+(?:\.\d+)?|\.\d+|(?:{agent._WORDS}))"
_STRICT_MIN = re.compile(
    rf"\b(?:for |lasting |lasts |that lasts |that last )?(?:more than|longer than|greater than|over|above|exceeding|in excess of) "
    rf"{_NUM} ?{_UNIT}\b")
_FILLER_AFTER_DURATION = re.compile(r"\b(?P<unit>seconds?|secs?|s|minutes?|mins?)\b (?:straight|in a row|continuously|continually|"
                                    r"consecutively|without (?:a break|stopping)|nonstop|non-stop)\b")
_MISSING_FOR = re.compile(rf"\b((?:attack|attacks|attacking|mount|mounts|mounting|investigate|investigates|investigating)"
                          rf"(?: each other)?) (?=(?:\d+(?:\.\d+)?|\.\d+|(?:{agent._WORDS})) ?(?:seconds?|secs?|s|minutes?|mins?)\b)")
_ACTION_END = re.compile(r"^(?P<conn>when|if|whenever) (?P<body>.+?)[, ]+(?P<act>alert me|notify me|warn me|tell me|let me know|"
                         r"ask me to confirm|ask me|check with me|log|record)[.!?]?$")


def _rewrite(t):
    """The strict-form candidate for t, and the notes that explain every change."""
    notes = []

    for wrong, right in _SHORT_SLIPS.items():             # very short words are only repaired from this fixed list
        if re.search(rf"\b{wrong}\b", t):
            t = re.sub(rf"\b{wrong}\b", right, t)
            notes.append(f"Read '{wrong}' as '{right}' (a typing slip).")
    t2, typos = repair_typos(t)
    for wrong, right in typos:
        notes.append(f"Read '{wrong}' as '{right}' (a typing slip).")
    t = t2

    t = _FILLER.sub("", t).strip()
    t = re.sub(r"^[,;:\s]+|[,;:\s]+$", "", t)             # a comma left behind by a removed "hey," or ", thanks"
    t = re.sub(r"\bmore that\b", "more than", t)
    t = re.sub(r"\b(are|is) being (?=aggressive|violent)", r"\1 ", t)
    for pattern, strict in _ACTION_WORDS:
        found = re.search(rf"\b{pattern}\b", t)
        if found:
            notes.append(f"Read '{found.group()}' as '{strict}'.")
            t = t[:found.start()] + strict + t[found.end():]
    for pattern, strict in _SUBJECTS:
        t = re.sub(rf"\b{pattern}\b", strict, t)
    for pattern, strict, behavior in _BEHAVIOR_WORDS:
        m = re.search(rf"\b{pattern}\b", t)
        if m:
            notes.append(f"Read '{m.group()}' as {behavior}, the closest behavior the classifier scores.")
            t = re.sub(rf"\b{pattern}\b", strict, t)
    # "mice are fighting" became "mice are attacking"; "the mouse attacks" and "mice attacks" become "mice attack"
    t = re.sub(r"\b(mice|they|both mice|the mice|the two mice) (?:attacks|attacked)\b", r"\1 attack", t)

    # "the mice attack 3 secs" (no "for") -> "the mice attack for 3 secs"
    t = _MISSING_FOR.sub(r"\1 for ", t)

    m = _STRICT_MIN.search(t)
    if m:
        n, unit = m["num"], m["unit"]
        notes.append(f"Read '{m.group().strip()}' as 'at least {n} {unit}': a rule's duration is an inclusive minimum "
                     "(the monitor reacts once the behavior has lasted that long).")
        t = t[:m.start()] + f"for at least {n} {unit}" + t[m.end():]
    t = _FILLER_AFTER_DURATION.sub(r"\g<unit>", t)
    t = re.sub(r"\s+", " ", t).strip()

    m = _ACTION_END.match(t)
    if m:
        t = f"{m['act']} {m['conn']} {m['body']}"
        notes.append("Moved the action to the front of the sentence.")
    return t, notes


def read_flexibly(text: str, rule_id: str):
    """RuleProposal or Clarification. The strict parser decides; this only rewrites the sentence into its form."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    strict = agent.parse_rule(text, rule_id)
    if isinstance(strict, RuleProposal):
        return strict
    source = text.strip()
    t = re.sub(r"\s+", " ", source.replace("’", "'").replace("‘", "'").lower())
    if not t.isascii() or not t:
        return strict
    candidate, notes = _rewrite(t)
    if candidate == t and not notes:
        return strict
    attempt = agent.parse_rule(candidate, rule_id)
    if not isinstance(attempt, RuleProposal):
        return strict                                       # still not readable: the strict parser's own reason is the best help
    data = dict(attempt.rule_dict)
    data["description"] = source
    rule, errors = validate_rule(data)
    if errors:
        return Clarification(source, tuple(agent.Issue("invalid_rule", e) for e in errors))
    assumptions = (f"Your sentence was read as: '{candidate}'.",) + tuple(notes) + attempt.assumptions
    return RuleProposal(source, MappingProxyType(rule.to_dict()), assumptions)
