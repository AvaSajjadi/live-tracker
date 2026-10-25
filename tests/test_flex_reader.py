"""Tests for flex_reader: loose wording and typos are understood; unsafe readings stay refused."""

import random

import pytest

from live_tracker import agent
from live_tracker import flex_reader as fr
from live_tracker.agent import Clarification, RuleProposal


def read(text):
    return fr.read_flexibly(text, "R1")


# what must be understood: sentence -> (behavior, seconds, alert mode)
UNDERSTOOD = [
    # the sentence that started this
    ("alert me when mice attack for more that 2 seconds", "attack", 2.0, "alert"),
    ("alert me when mice attack for more than 2 seconds", "attack", 2.0, "alert"),
    # other ways of saying a minimum
    ("alert me when the mice attack for over 3 seconds", "attack", 3.0, "alert"),
    ("alert me when the mice attack for longer than 4 sec", "attack", 4.0, "alert"),
    ("alert me when the mice attack for above 5 secs", "attack", 5.0, "alert"),
    ("alert me when the mice attack for 2 seconds straight", "attack", 2.0, "alert"),
    ("let me know when the mice attack 3 secs in a row", "attack", 3.0, "alert"),
    ("alert me when the mice attack for two minutes continuously", "attack", 120.0, "alert"),
    ("alert me when mice attack for 2 seconds or longer", "attack", 2.0, "alert"),
    # typos
    ("alert me when mice attck for at least 2 secnds", "attack", 2.0, "alert"),
    ("alrt me wehn the mice mount for two secnds or more", "mount", 2.0, "alert"),
    ("alert me wen the mice atack for atleast 2 seconds".replace("atleast", "at least"), "attack", 2.0, "alert"),
    ("notfy me whenever mice investigte for 3 seconds", "investigation", 3.0, "alert"),
    # polite wording and filler
    ("Please could you alert me when the mice attack for at least 2 seconds", "attack", 2.0, "alert"),
    ("I would like you to let me know if the mice are fighting for over 3 seconds", "attack", 3.0, "alert"),
    ("Hey, just notify me whenever the mice attack for 2 secs, thanks", "attack", 2.0, "alert"),
    ("I want to be notified when the mice attack for longer than 4 sec", "attack", 4.0, "alert"),
    ("I'd like to know if the mice attack for at least 2 seconds", "attack", 2.0, "alert"),
    # everyday words for the action
    ("ping me when the mice attack for at least 2 seconds", "attack", 2.0, "alert"),
    ("message me if the mice attack for at least 2 seconds", "attack", 2.0, "alert"),
    ("send me an alert when the mice attack for at least 2 seconds", "attack", 2.0, "alert"),
    ("write down when the mice mount for at least 2 seconds", "mount", 2.0, "log"),
    ("confirm with me when the mice mount for at least 2 seconds", "mount", 2.0, "ask"),
    # everyday words for the behavior
    ("alert me when the mice are fighting for at least 2 seconds", "attack", 2.0, "alert"),
    ("alert me when the mice fight for at least 2 seconds", "attack", 2.0, "alert"),
    ("Tell me if they are being aggressive for 2 seconds or more", "attack", 2.0, "alert"),
    ("alert me when the mice are violent for at least 2 seconds", "attack", 2.0, "alert"),
    ("alert me when the mice are biting for at least 2 seconds", "attack", 2.0, "alert"),
    ("ping me when the mice are sniffing each other for at least 1 second", "investigation", 1.0, "alert"),
    ("ask me if mice are mating for at least 2 seconds", "mount", 2.0, "ask"),
    # word order and the subject
    ("when the mice fight for more than 5 seconds, let me know", "attack", 5.0, "alert"),
    ("if my mice attack for at least 2 seconds, alert me", "attack", 2.0, "alert"),
    ("alert me when both animals attack for at least 2 seconds", "attack", 2.0, "alert"),
    ("alert me when the mice attacks for 2 seconds", "attack", 2.0, "alert"),
]

# what must stay refused, and the reason code the strict parser gives
REFUSED = [
    ("alert me when mice attack for less than 2 seconds", "unsupported_duration"),
    ("alert me when mice attack for at most 2 seconds", "unsupported_duration"),
    ("alert me when mice attack for under 2 seconds", "unsupported_duration"),
    ("alert me when mice attack for no more than 2 seconds", "unsupported_duration"),
    ("alert me when mice attack for exactly 2 seconds", "unsupported_duration"),
    ("alert me when mice attack for up to 2 seconds", "unsupported_duration"),
    ("alert me when mice attack for a while", "missing_duration"),
    ("alert me when mice attack", "missing_duration"),
    ("alert me when mice attack for a few seconds", None),
    ("alert me when the mice are grooming for 2 seconds", "unsupported_behavior"),
    ("alert me when the mice are chasing for at least 2 seconds", "unsupported_behavior"),
    ("don't alert me when mice attack for 2 seconds", "negation"),
    ("alert me unless the mice attack for 2 seconds", None),
    ("alert me when mice attack near the wall for 2 seconds", None),
    ("alert me at night when mice attack for 2 seconds", None),
    ("alert me when mice attack or mount for 2 seconds", None),
    ("hello", None),
    ("", None),
]


@pytest.mark.parametrize("text,behavior,seconds,mode", UNDERSTOOD)
def test_loose_wording_and_typos_are_understood(text, behavior, seconds, mode):
    result = read(text)
    assert isinstance(result, RuleProposal), (text, getattr(result, "issues", None))
    rule = result.rule_dict
    assert (rule["behavior"], rule["min_duration_s"], rule["alert_mode"]) == (behavior, seconds, mode)
    assert rule["status"] == "proposed" and rule["rule_id"] == "R1" and rule["description"] == text.strip()
    assert result.source_text == text.strip()


@pytest.mark.parametrize("text,code", REFUSED)
def test_unsafe_or_unreadable_sentences_stay_refused(text, code):
    result = read(text)
    assert isinstance(result, Clarification), (text, result)
    if code is not None:
        assert code in [i.code for i in result.issues]


def test_every_change_is_written_down_and_the_rewritten_sentence_is_shown():
    result = read("alert me when mice attck for more that 2 secnds")
    notes = " | ".join(result.assumptions)
    assert "Your sentence was read as: 'alert me when mice attack for at least 2 seconds'." in notes
    assert "Read 'attck' as 'attack'" in notes and "Read 'secnds' as 'seconds'" in notes
    assert "as 'at least 2 seconds': a rule's duration is an inclusive minimum" in notes
    assert "defaults are used" in notes


def test_a_synonym_for_a_behavior_is_named_in_the_assumptions():
    notes = " | ".join(read("alert me when the mice are fighting for at least 2 seconds").assumptions)
    assert "Read 'fighting' as attack, the closest behavior the classifier scores." in notes
    assert "Read 'let me know' as" not in notes


def test_a_sentence_the_strict_parser_already_understands_is_returned_unchanged():
    for text in ("alert me when the mice attack for at least 2 seconds", "Log investigation bouts of 5 seconds or more",
                 "alert me when the mice attack for at least 0.5 seconds, ask threshold 0.6, alert threshold 0.8"):
        strict, flexible = agent.parse_rule(text, "R1"), read(text)
        assert isinstance(strict, RuleProposal) and flexible == strict
        assert not any("Your sentence was read as" in a for a in flexible.assumptions)


def test_named_thresholds_survive_a_loose_sentence():
    result = read("alert me when the mice fight for more than 0.5 seconds, ask threshold 0.6, alert threshold 0.8")
    assert isinstance(result, RuleProposal)
    assert (result.rule_dict["ask_threshold"], result.rule_dict["alert_threshold"]) == (0.6, 0.8)


def test_the_refusal_reason_comes_from_the_strict_parser_untouched():
    text = "alert me when mice attack for less than 2 seconds"
    assert read(text) == agent.parse_rule(text, "R1")


def test_typos_that_hide_a_refusal_are_still_refused():
    for text in ("alert me when mice attack for lss than 2 secnds", "alrt me when mice attck for at mst 2 seconds"):
        result = read(text)
        assert isinstance(result, Clarification), text


def test_non_ascii_text_and_wrong_types():
    assert isinstance(read("alert me when mice attack for 2 secondes é"), Clarification)
    with pytest.raises(TypeError):
        fr.read_flexibly(None, "R1")


# ---- the typo repair on its own ----

def test_distance_counts_a_swap_of_two_letters_as_one_slip():
    assert fr._distance("when", "wehn") == 1 and fr._distance("attack", "attck") == 1 and fr._distance("a", "b") == 1
    assert fr._distance("seconds", "seconds") == 0 and fr._distance("abc", "xyz") == 3


def test_repair_changes_only_a_word_that_is_one_slip_from_exactly_one_vocabulary_word():
    text, changes = fr.repair_typos("alrt me wehn mice attck for 2 secnds")
    assert text == "alert me when mice attack for 2 seconds" and len(changes) == 4
    assert fr.repair_typos("the then")[0] == "the then"                   # 'then' is protected and close to two words
    assert fr.repair_typos("teh wll")[0] == "teh wll"                      # too short to repair
    assert fr.repair_typos("near the wall for 2 seconds")[0] == "near the wall for 2 seconds"
    assert fr.repair_typos("zebra quartz")[0] == "zebra quartz"            # nothing close


def test_it_never_raises_on_odd_sentences():
    rng = random.Random(7)
    pool = ("alert me when the mice attack mount investigate for at least more than less over 2 3.5 seconds sec minutes "
            "if whenever near wall night please ping fight fighting attck wehn not don't unless and or threshold 0.6 "
            "ask log, . ! ? 'x' ) ( é -").split()
    for _ in range(400):
        text = " ".join(rng.choice(pool) for _ in range(rng.randint(1, 14)))
        result = read(text)
        assert isinstance(result, (RuleProposal, Clarification)), text
        if isinstance(result, RuleProposal):
            assert result.rule_dict["status"] == "proposed" and result.rule_dict["min_duration_s"] > 0
