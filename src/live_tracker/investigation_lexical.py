"""EXPERIMENTAL (Investigation mode): the legacy/1 lexical prose checks (new
file at Findings v2 checkpoint M4; the code is moved unchanged from
investigation_agent).

Legacy reports (findings_format "legacy/1") carry model-written prose
(interpretations, next steps, model limitations). These patterns refuse
numbers, number words, significance / causal / proof / prediction language
and, unless the finding is effect_described, effect language. They are a
lexical screen with known limits (tests/test_investigation_agent.py,
test_known_lexical_limits: 11 strict xfails), not evidence verification.

Only the legacy paths import this module, lazily: the legacy finish check
(investigation_agent._Run.verify_final) and the legacy Stage 7 check
(investigation_report._check). findings/2 never imports it: its finish has
no prose, its claims are checked by the claims/2.0 registry and its report
is rendered from fixed templates (investigation_report_v2).
"""

from __future__ import annotations

import re

__all__ = ["MAX_STATEMENT", "prose_problems"]

MAX_STATEMENT = 600

_BANNED = [(re.compile(p, re.I), w) for p, w in (
    (r"\bsignifican", "significance"), (r"\bp\s*-?\s*values?\b", "p-value"), (r"\bp\s*[<=>]\s*0?\.\d", "p-value"),
    (r"\bconfidence\s+intervals?\b", "confidence interval"), (r"\bstatistical(ly)?\b", "statistical claim"),
    (r"\bprov(e|es|ed|en|ing)\b", "proof"), (r"\bcaus(e|es|ed|ing|al|ality|ation)\b", "causal claim"),
    (r"\bpredict", "prediction"))]
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\.?\d+(?:[,.]\d+)*(?:e[-+]?\d+)?", re.I)
_NUMBER_WORDS = re.compile(
    r"\b(zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundreds?|"
    r"thousands?|millions?|billions?|dozens?|half|halves|twice|thrice|double[ds]?|triple[ds]?|quadruple[ds]?|"
    r"percent|percentage|fold)\b|%", re.I)
# Exemptions are allowlists (a context not listed keeps the refusal):
# - "one" is accepted only in fixed non-quantitative forms ("no one", "one of", "one or more", "one by one")
#   or before a count noun ("one session", "one annotated session") not followed by a rate;
# - a number directly after an identifier noun ("CalMS21 Task 1", "Stage 4", "annotator 1") is a name only
#   when the end of the sentence, punctuation or a listed word follows it (never a unit or a value).
_COUNT_NOUNS = (r"(?:sessions?|windows?|bouts?|episodes?|attacks?|mouse|mice|animals?|annotators?|videos?|"
                r"recordings?|datasets?|comparisons?|results?|tools?|groups?|conditions?|tasks?)")
_UNIT_WORDS = (r"(?:px|pixels?|met(?:re|er)s?|millimet(?:re|er)s?|centimet(?:re|er)s?|kilomet(?:re|er)s?|mm|cm|km|m|"
               r"hz|hertz|fps|degrees?|radians?|deg|rad|percent|units?|standard\s+deviations?|sd|sds|"
               r"inch(?:es)?|feet|foot)")
_RATE = r"(?:[A-Za-z]+\s*/\s*[A-Za-z]+|[A-Za-z]+\s+per\s+[A-Za-z]+)"
_TIME = r"(?:seconds?|secs?|s|frames?|minutes?|mins?|hours?|hrs?|ms|milliseconds?)"
# a normalisation: "per second", "every second", "each frame", "a minute" (not an ordinal: "a second session")
_PER_TIME = (rf"(?:per|every|each)\s+(?:few\s+|couple\s+of\s+|several\s+)?{_TIME}\b|"
             rf"(?:a|an)\s+{_TIME}\b(?!\s+(?!and\b|or\b|but\b|before\b|after\b|during\b|while\b|until\b|in\b|"
             rf"on\b|at\b|of\b|for\b|to\b|than\b)[A-Za-z])")
# lengths measured in an object ("one mouse length", "a body width")
_OBJECT_LENGTH = r"(?:lengths?|widths?|diameters?|steps?|strides?)\b"
_ONE_ALLOWED = re.compile(
    rf"\b(?:no|any|each|every)\s+one\b|\bone(?=\s+(?:of|another|or\s+more|or\s+the\s+other|by\s+one)\b)|"
    rf"(?<=\bone by )one\b|"
    rf"\bone(?=\s+(?:[a-z]+(?:-[a-z]+)?\s+)?{_COUNT_NOUNS}\b(?!\s*(?:/|per\b))(?!\s+{_OBJECT_LENGTH})"
    rf"(?![^.;:!?]*?\b(?:{_PER_TIME})))", re.I)
_ONE = re.compile(r"\bone\b", re.I)
_IDENTIFIER_NUMBER = re.compile(
    r"\b(?:Task|Stage|Day|Experiment|annotator)\s+\d{1,2}(?=\s*$|\s*[.,;:)!?]|\s+(?:has|have|had|does|did|contains?|"
    r"includes?|lacks?|provides?|defines?|uses?|data|dataset|annotations?|labels?|sessions?|videos?|recordings?|"
    r"windows?|rules|results|evidence|split|and|or|but|which|that|where)\b)", re.I)
# a quantifier before a physical unit or a rate is a measured value ("a metre per second", "several px/s")
_QUANT = r"(?:a|an|one|once|single|several|few|couple|many|some|\d[\d.,]*)"
_QUANTIFIED_UNIT = re.compile(
    rf"\b{_QUANT}\s+(?:of\s+)?(?:{_UNIT_WORDS}\b|{_RATE})|"
    # a quantity normalised by time later in the clause: "one bout every second", "a mouse length per second"
    rf"\b{_QUANT}\s+[^.;:!?]*?\b(?:{_PER_TIME})|\b{_QUANT}\s+(?:{_PER_TIME})|"
    rf"\b(?:every|each|per)\s+(?:few|couple\s+of|several)\s+{_TIME}\b|"
    # a length measured in an object: "one mouse length", "a body width"
    rf"\b{_QUANT}\s+(?:[a-z]+\s+){{0,2}}?{_OBJECT_LENGTH}", re.I)
# claims about an effect's size, refused like other effect language when there is no effect to describe
_MAGNITUDE = re.compile(
    r"\b(?:large|small|big|huge|tiny|strong|weak|marked|substantial|considerable|pronounced|dramatic|modest|slight|"
    r"negligible|robust|notable|noticeable|sizable|sizeable)\s+(?:effect|difference|change|shift|association|"
    r"increase|decrease)s?\b|\b(?:effect|difference|change|association)s?\s+(?:was|were|is|are|appears?|seems?|"
    r"looks?)\s+(?:very\s+)?(?:large|small|big|huge|tiny|strong|weak|marked|substantial|considerable|pronounced|"
    r"dramatic|modest|slight|negligible|robust|notable|noticeable|sizable|sizeable)\b", re.I)
_EFFECT_WORDS = re.compile(
    r"\b(increas|decreas|higher|lower|faster|slower|greater|smaller|larger|reduc|rise|rises|rose|risen|"
    r"drop|fell|fall|declin|elevat|diminish|shrink|shrank|grew|grow|closer|farther|further apart)\w*", re.I)

def _prose_problems(text, finding):
    """Why model-written prose cannot be accepted (empty: it can)."""
    if not text.strip() or len(text) > MAX_STATEMENT:
        return [f"must be non-empty text of at most {MAX_STATEMENT} characters"]
    out = [f"{word} language is not allowed" for pat, word in _BANNED if pat.search(text)]
    named = _IDENTIFIER_NUMBER.sub(" ", text)        # identifiers are names, not quantities
    allowed_one = {m.start() for m in _ONE_ALLOWED.finditer(text)}
    allowed_one |= {m.end() - 3 for m in _ONE_ALLOWED.finditer(text) if m.group().lower().endswith("one")}
    quantities = ([m.group() for m in _NUMBER.finditer(named)] + [m.group() for m in _NUMBER_WORDS.finditer(text)]
                  + [m.group() for m in _ONE.finditer(text) if m.start() not in allowed_one]
                  + [m.group() for m in _QUANTIFIED_UNIT.finditer(text)])
    if quantities:
        out.append(f"numbers or number words {quantities[:5]} are not allowed in prose; cite measurements as "
                   "observations {evidence_id, field}")
    if finding != "effect_described":
        effect = [m.group() for m in _EFFECT_WORDS.finditer(text)] + [m.group() for m in _MAGNITUDE.finditer(text)]
        if effect:
            out.append(f"effect language {effect[:5]} is not allowed when the finding is {finding}")
    return out


prose_problems = _prose_problems
