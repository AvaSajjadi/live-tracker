"""Replay -> Contract v1 investigation bridge (replay_investigation.py) and its
section on the Replay page. New file. SYNTHETIC sessions only; no network."""

import dataclasses
import json

import pytest

import live_tracker.investigation_tools_v1 as v1
import live_tracker.replay_investigation as ri
import live_tracker.replay_view as rv
from live_tracker.monitor import run_monitor
from live_tracker.rules import RuleSet
from test_replay_page import TRACK, start, workflow


def approved(*rules):
    rs = RuleSet()
    for r in rules:
        rs.add(r)
        rs.approve(r["rule_id"])
    return rs


ATTACK = {"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0}
MOUNT = {"rule_id": "R2", "behavior": "mount", "min_duration_s": 1.0, "alert_mode": "ask"}


@pytest.mark.parametrize("entry", rv.synthetic_catalog(), ids=lambda e: e.session)
def test_query_bouts_are_the_monitored_bouts_of_the_replay(entry):
    rules = approved(ATTACK, MOUNT)
    res = ri.investigate_replay(entry, rules.active(), "run1")
    events = run_monitor(entry.scores, rules)                         # what the replay monitors
    for item in res["rules"]:
        mine = [(e["start_s"], e["end_s"]) for e in events if e["rule_id"] == item["rule_id"]]
        assert [(b["start_s"], b["end_s"]) for b in item["query"]["bouts"]] == mine
        assert item["query"]["data_source"] == "synthetic" and item["query"]["query_run_id"].startswith("inv_")


def test_comparison_is_honestly_insufficient_and_association_only():
    entry = rv.synthetic_catalog()[0]
    res = ri.investigate_replay(entry, approved(ATTACK).active(), "run1")
    (item,) = res["rules"]
    c = item["compare"]
    assert c["status"] == "insufficient_evidence" and c["reason"] == "too_few_bouts"
    assert c["association_only"] is True and c["data_source"] == "synthetic"
    assert "mean_diff" not in c and "pre_mean" not in c                 # no effect value behind a failed gate
    assert "not a scientific finding" in res["note"] and res["data_source"] == "synthetic"
    rec = item["record"]
    assert rec["investigation_id"] == "run1_R1" and rec["calls_used"] == 2 and rec["max_tool_calls"] == 12
    assert [a["tool"] for a in rec["attempts"]] == ["query_events", "compare_windows"]
    json.dumps(res)                                                   # JSON-ready (the page offers it as a download)


def test_each_rule_is_a_separate_bounded_investigation():
    res = ri.investigate_replay(rv.synthetic_catalog()[1], approved(ATTACK, MOUNT).active(), "run9")
    assert [i["investigation_id"] for i in res["rules"]] == ["run9_R1", "run9_R2"]
    assert all(i["record"]["calls_used"] <= 2 for i in res["rules"])


def test_only_synthetic_sessions_with_pose_are_investigated():
    entry = rv.synthetic_catalog()[0]
    with pytest.raises(ValueError, match="not a SYNTHETIC session"):
        ri.investigate_replay(dataclasses.replace(entry, session="mouse001_task1"), [], "r")
    with pytest.raises(ValueError, match="no pose"):
        ri.investigate_replay(dataclasses.replace(entry, pose=None), [], "r")


def test_researcher_rules_and_files_are_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rules = approved(ATTACK)
    before = rules.to_json()
    ri.investigate_replay(rv.synthetic_catalog()[0], rules.active(), "run1")
    assert rules.to_json() == before and list(tmp_path.iterdir()) == []


# ---- the page section ----

def section(at):
    (exp,) = [e for e in at.expander if e.label == "Investigation evidence (Contract v1 tools, SYNTHETIC)"]
    return exp


def end_of_session(at):
    next(b for b in at.button if b.label == "End of session").click().run()
    assert not at.exception, at.exception
    return at


def test_the_end_view_shows_the_investigation_evidence_labelled_synthetic():
    at = end_of_session(start(workflow()))
    exp = section(at)
    assert any("not a scientific finding" in w.value for w in exp.warning)
    text = " ".join(c.value for c in exp.caption)
    assert "query_events: 2 bout(s)" in text and "insufficient_evidence" in text and "too_few_bouts" in text
    assert "Association only: True" in text
    assert any(b.label == "Download investigation record (JSON)" for b in at.get("download_button")
               if hasattr(b, "label"))


def test_the_investigation_section_appears_only_at_the_end():
    at = start(workflow())
    assert not [e for e in at.expander if "Investigation evidence" in e.label]


def test_the_section_follows_the_selected_session():
    at = start(workflow(), replay_session=TRACK)
    at = end_of_session(at)
    text = " ".join(c.value for c in section(at).caption)
    assert "query_events:" in text and "insufficient_evidence" in text


# ---- cache identity: pose, scores, rules, session and configuration (Codex review) ----

def excluded_text(at):
    return " ".join(c.value for c in section(at).caption if c.value.startswith("Excluded:"))


def with_nan_pose(entry, t0, t1):
    pose = entry.pose.copy()
    sel = (pose["time_s"] >= t0) & (pose["time_s"] < t1)
    pose.loc[sel, ["x", "y"]] = float("nan")
    return dataclasses.replace(entry, pose=pose)


def test_a_changed_pose_recomputes_the_evidence_codex_reproduction(monkeypatch):
    at = end_of_session(start(workflow(), replay_session=TRACK))
    assert "17.00 s baseline_window_unreliable" in excluded_text(at)   # the stuck tracker inside the baseline
    demo, track = rv.synthetic_catalog()
    changed = with_nan_pose(track, 15.0, 17.0)                          # same scores, session and rules
    monkeypatch.setattr(rv, "synthetic_catalog", lambda: (demo, changed))
    at.run()
    assert not at.exception, at.exception
    assert "17.00 s pre_window_unreliable" in excluded_text(at)          # recomputed, not the stale result
    fresh = ri.investigate_replay(changed, workflow().ruleset.active(), at.session_state["replay_active_run"])
    shown = [v for k, v in at.session_state["replay_investigation_cache"].items()
             if v["inputs"]["pose_sha256"] == rv.content_digest(changed.pose)]
    assert len(shown) == 1 and shown[0]["rules"] == fresh["rules"]      # the panel's (and download's) result
    assert any(rv.content_digest(changed.pose)[:12] in c.value for c in section(at).caption)


def compare_caption(at):
    return " ".join(c.value for c in section(at).caption if c.value.startswith("compare_windows"))


def test_unchanged_inputs_reuse_the_result_and_a_real_preset_change_recomputes(monkeypatch):
    calls = []
    real = ri.investigate_replay
    monkeypatch.setattr(ri, "investigate_replay", lambda *a: calls.append(1) or real(*a))
    at = end_of_session(start(workflow()))
    at.run()
    at.run()
    assert len(calls) == 1 and "pre 2 s" in compare_caption(at)        # ordinary reruns reuse
    monkeypatch.setitem(ri.PRESET, "pre_s", 1.0)                       # an actual change of the effective settings
    at.run()
    assert len(calls) == 2 and "pre 1 s" in compare_caption(at)


def test_an_effective_quality_policy_change_invalidates_and_is_recorded(monkeypatch):
    at = end_of_session(start(workflow(), replay_session=TRACK))
    cache = at.session_state["replay_investigation_cache"]
    (first_key,) = list(cache)
    assert cache[first_key]["inputs"]["config"]["quality_policy"] == "quality_v1"
    monkeypatch.setattr(v1, "DEFAULT_QUALITY_POLICY", "quality_v1_strict90")
    at.run()
    assert not at.exception, at.exception
    cache = at.session_state["replay_investigation_cache"]
    (new_key,) = [k for k in cache if k != first_key]                  # a new identity, a new computation
    new = cache[new_key]
    cfg = new["inputs"]["config"]
    assert cfg["quality_policy"] == "quality_v1_strict90"
    assert cfg["quality_policy_definition"] == v1.QUALITY_POLICIES["quality_v1_strict90"].to_json()
    assert new["inputs"]["config_sha256"] == ri.config_digest(cfg) == new_key[1]
    (item,) = new["rules"]
    assert item["compare"]["quality_policy"] == "quality_v1_strict90"   # executed with the recorded policy
    assert cache[first_key]["rules"][0]["compare"]["quality_policy"] == "quality_v1"   # the old result is intact
    assert any(new["inputs"]["config_sha256"][:12] in c.value for c in section(at).caption)


def test_a_saved_result_never_changes_when_module_settings_change(monkeypatch):
    entry = rv.synthetic_catalog()[1]
    res = ri.investigate_replay(entry, approved(ATTACK).active(), "run1")
    before = json.dumps(res, sort_keys=True)
    monkeypatch.setitem(ri.PRESET, "pre_s", 1.0)                       # Codex reproduction 1
    monkeypatch.setattr(v1, "DEFAULT_QUALITY_POLICY", "quality_v1_strict90")   # Codex reproduction 2
    assert json.dumps(res, sort_keys=True) == before
    cfg = res["inputs"]["config"]
    assert cfg["preset"]["pre_s"] == 2.0 and cfg["quality_policy"] == "quality_v1"
    assert ri.config_digest(cfg) == res["inputs"]["config_sha256"]
    (item,) = res["rules"]
    assert item["compare"]["pre_s"] == 2.0 and item["compare"]["quality_policy"] == "quality_v1"
    now = ri.investigate_replay(entry, approved(ATTACK).active(), "run1")        # resolved at run time
    assert now["inputs"]["config"]["preset"]["pre_s"] == 1.0 and now["rules"][0]["compare"]["pre_s"] == 1.0
    assert now["rules"][0]["compare"]["quality_policy"] == "quality_v1_strict90"
    assert now["inputs"]["config_sha256"] != res["inputs"]["config_sha256"]


def test_the_snapshot_is_independent_of_module_state():
    cfg = ri.effective_config()
    cfg["preset"]["pre_s"] = 99.0
    assert ri.PRESET["pre_s"] == 2.0 and ri.effective_config()["preset"]["pre_s"] == 2.0
    json.dumps(ri.effective_config())                                  # JSON-only


def test_the_result_records_what_it_was_computed_from():
    entry = rv.synthetic_catalog()[1]
    res = ri.investigate_replay(entry, approved(ATTACK).active(), "run1")
    assert res["inputs"] == {"session": entry.session, "replay_run_id": "run1",
                             "scores_sha256": rv.content_digest(entry.scores),
                             "pose_sha256": rv.content_digest(entry.pose),
                             "config_sha256": ri.config_digest(ri.effective_config()), "config": ri.effective_config()}
    other = ri.investigate_replay(with_nan_pose(entry, 15.0, 17.0), approved(ATTACK).active(), "run1")
    assert other["inputs"]["pose_sha256"] != res["inputs"]["pose_sha256"]


# ---- rules Contract v1 cannot investigate (ask threshold 0) ----

ZERO = "Alert me when the mice attack for at least 2 seconds with ask threshold 0"


def test_a_zero_threshold_rule_is_disclosed_not_substituted():
    rule = {**ATTACK, "ask_threshold": 0.0}
    res = ri.investigate_replay(rv.synthetic_catalog()[0], approved(rule).active(), "run1")
    (item,) = res["rules"]
    assert item["unsupported"] == ri.UNSUPPORTED_ZERO
    assert item["query"]["status"] == "error" and item["query"]["code"] == "invalid_request"
    assert item["compare"] is None and item["record"]["calls_used"] == 1     # the refusal itself is recorded
    assert run_monitor(rv.synthetic_catalog()[0].scores, approved(rule))      # monitoring still runs the rule


def test_the_page_discloses_a_zero_threshold_rule():
    at = end_of_session(start(workflow(ZERO)))
    infos = " ".join(i.value for i in section(at).info)
    assert "Rule R1: ask threshold 0 is not supported by Contract v1 investigation" in infos
    assert "Deterministic monitoring still runs this rule" in infos
