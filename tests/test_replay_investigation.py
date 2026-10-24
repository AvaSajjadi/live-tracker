"""Replay -> Contract v1 investigation bridge (replay_investigation.py) and its
section on the Replay page. New file. SYNTHETIC sessions only; no network."""

import copy
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
    assert c["overlap_policy"] == res["inputs"]["config"]["overlap_policy"] == "overlap_v1"
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


def test_the_overlap_policy_is_named_in_the_recorded_configuration_and_the_export():
    cfg = ri.effective_config()
    assert cfg["overlap_policy"] == v1.DEFAULT_OVERLAP_POLICY == "overlap_v1"
    assert cfg["overlap_policy_definition"] == v1.OVERLAP_POLICIES["overlap_v1"].to_json()
    assert cfg["overlap_policy_definition"]["windows_checked"] == ["pre", "baseline"]
    res = ri.investigate_replay(rv.synthetic_catalog()[1], approved(ATTACK).active(), "run1")
    exported = json.loads(json.dumps(res, sort_keys=True))             # what the page's download contains
    assert exported["inputs"]["config"]["overlap_policy"] == "overlap_v1"
    assert exported["rules"][0]["compare"]["overlap_policy_definition"] == cfg["overlap_policy_definition"]


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


# ---- the recorded overlap policy must be the one executed (Codex review: provenance) ----

ENTRY = rv.synthetic_catalog()[0]


def run(cfg=None):
    return ri.investigate_replay(ENTRY, approved(ATTACK).active(), "review", config=cfg)


def pre_only_registry(monkeypatch):
    """The registry's overlap_v1 replaced by a pre-window-only definition (another scientific policy)."""
    monkeypatch.setitem(v1.OVERLAP_POLICIES, "overlap_v1",
                        dataclasses.replace(v1.OVERLAP_POLICIES["overlap_v1"], windows=("pre",)))


def spy_compare(monkeypatch, before=None, after=None):
    """Wrap the runner's compare_windows: `before()` runs just before the real tool, `after(result)` may alter it."""
    calls, real = [], v1._TOOL_FUNCS["compare_windows"]

    def wrapped(data, req):
        calls.append(1)
        if before:
            before()
        r = real(data, req)
        return after(r) if after else r
    monkeypatch.setitem(v1._TOOL_FUNCS, "compare_windows", wrapped)
    return calls


def test_a_canonical_snapshot_runs_and_the_result_carries_one_policy():
    cfg = ri.effective_config()
    res = run(cfg)
    c = res["rules"][0]["compare"]
    assert c["status"] == "insufficient_evidence"
    for k in ri.OVERLAP_KEYS:
        assert c[k] == res["inputs"]["config"][k] == cfg[k]
    assert cfg["overlap_policy_definition"]["windows_checked"] == ["pre", "baseline"]
    assert cfg["exclusion_order"] == list(v1.EXCLUSION_REASONS)
    assert res["inputs"]["config_sha256"] == ri.config_digest(cfg)


def test_tampered_windows_checked_is_refused_before_anything_runs(monkeypatch):
    calls = spy_compare(monkeypatch)
    cfg = ri.effective_config()
    cfg["overlap_policy_definition"]["windows_checked"] = ["pre"]        # Codex reproduction
    with pytest.raises(ri.PolicyMismatch, match="overlap_policy_definition"):
        run(cfg)
    assert calls == []


@pytest.mark.parametrize("change", [
    lambda c: c.pop("exclusion_order"),                                # a snapshot from before the order was recorded
    lambda c: c.pop("overlap_policy_definition"),
    lambda c: c.update(overlap_policy="overlap_v0"),                    # an older policy name
    lambda c: c["exclusion_order"].reverse(),
    lambda c: c["overlap_policy_definition"].update(other_bouts="every threshold run"),
])
def test_missing_or_stale_overlap_metadata_is_refused_even_with_a_consistent_hash(change):
    cfg = ri.effective_config()
    change(cfg)
    assert ri.config_digest(cfg) == ri.config_digest(json.loads(json.dumps(cfg)))    # internally consistent
    with pytest.raises(ri.PolicyMismatch):
        run(cfg)


def test_a_snapshot_captured_before_the_registry_changed_is_refused(monkeypatch):
    cfg = ri.effective_config()                                        # valid when captured
    pre_only_registry(monkeypatch)
    calls = spy_compare(monkeypatch)
    with pytest.raises(ri.PolicyMismatch, match="disagrees"):
        run(cfg)
    assert calls == []                                                 # refused before execution


def test_a_registry_change_between_validation_and_execution_fails_closed(monkeypatch):
    calls = spy_compare(monkeypatch, before=lambda: pre_only_registry(monkeypatch))
    with pytest.raises(ri.PolicyMismatch):
        run(ri.effective_config())
    assert calls == [1]                                                # it ran, and its result was not accepted


def test_a_policy_removed_from_the_registry_mid_run_fails_closed(monkeypatch):
    spy_compare(monkeypatch, before=lambda: monkeypatch.delitem(v1.OVERLAP_POLICIES, "overlap_v1"))
    with pytest.raises(ri.PolicyMismatch, match="not a Contract v1 policy"):
        run(ri.effective_config())


@pytest.mark.parametrize("field, value", [
    ("overlap_policy", "overlap_v2"),
    ("overlap_policy_definition", {"name": "overlap_v1", "windows_checked": ["pre"]}),
    ("exclusion_order", ["pre_window_overlaps_bout"]),
])
def test_a_completed_result_with_other_policy_metadata_is_refused(monkeypatch, field, value):
    spy_compare(monkeypatch, after=lambda r: {**r, field: value})
    with pytest.raises(ri.PolicyMismatch):
        run(ri.effective_config())


def test_a_completed_result_without_policy_metadata_is_refused(monkeypatch):
    spy_compare(monkeypatch, after=lambda r: {k: v for k, v in r.items() if k != "overlap_policy_definition"})
    with pytest.raises(ri.PolicyMismatch, match="reports no overlap_policy_definition"):
        run(ri.effective_config())


def test_a_mismatch_is_never_cached(monkeypatch):
    spy_compare(monkeypatch, before=lambda: pre_only_registry(monkeypatch))
    cache = {}
    with pytest.raises(ri.PolicyMismatch):
        ri.cached_investigation(cache, ("identity",), ENTRY, approved(ATTACK).active(), "review")
    assert cache == {}


def test_the_replay_page_refuses_a_mismatch_with_no_record_and_no_download(tmp_path, monkeypatch):
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(tmp_path / "replay"))      # isolated journals
    spy_compare(monkeypatch, before=lambda: pre_only_registry(monkeypatch))
    at = end_of_session(start(workflow(), replay_session=TRACK))
    assert any("The investigation was refused" in e.value for e in section(at).error)
    assert not [b for b in at.get("download_button") if b.label == "Download investigation record (JSON)"]
    assert at.session_state["replay_investigation_cache"] == {}


# ---- every execution-defining field of the snapshot must be what the tools execute (Codex review 2) ----

def spy_tool(monkeypatch, tool, before=None, after=None):
    """Like spy_compare, for any v1 tool the runner dispatches."""
    calls, real = [], v1._TOOL_FUNCS[tool]

    def wrapped(data, req):
        calls.append(1)
        if before:
            before()
        r = real(data, req)
        return after(r) if after else r
    monkeypatch.setitem(v1._TOOL_FUNCS, tool, wrapped)
    return calls


def strict_registry(monkeypatch, **changes):
    """quality_v1 in the registry replaced by an altered definition (another scientific policy)."""
    monkeypatch.setitem(v1.QUALITY_POLICIES, "quality_v1",
                        dataclasses.replace(v1.QUALITY_POLICIES["quality_v1"], **changes))


def refused_before_any_tool(monkeypatch, cfg, match=None):
    q, c = spy_tool(monkeypatch, "query_events"), spy_compare(monkeypatch)
    with pytest.raises(ri.PolicyMismatch, match=match):
        run(cfg)
    assert q == [] and c == []                                         # not a single scientific tool call


def test_codex_quality_definition_reproduction_is_refused(monkeypatch):
    cfg = ri.effective_config()
    cfg["quality_policy_definition"]["min_usable_fraction"] = 1.0       # quality_v1 executes 0.8
    refused_before_any_tool(monkeypatch, cfg, "quality_policy_definition")


@pytest.mark.parametrize("field, value", [
    ("threshold_from", "rule.alert_threshold"),                        # Codex: execution uses rule.ask_threshold
    ("contract_version", "invented/v99"),                              # Codex: execution is Contract v1
    ("preset", {**ri.PRESET, "pre_s": 5.0}),
    ("preset", {**ri.PRESET, "feature": "rel_speed"}),
    ("preset", {k: v for k, v in ri.PRESET.items() if k != "baseline_gap_s"}),
    ("evidence_gate", {"min_n_windows": 3, "min_n_sessions": 1}),
    ("max_tool_calls", 11),
    ("quality_policy_definition", {**v1.QUALITY_POLICIES["quality_v1"].to_json(), "veto_max_s": 1.0}),
])
def test_altered_execution_descriptors_are_refused(monkeypatch, field, value):
    cfg = ri.effective_config()
    cfg[field] = value
    refused_before_any_tool(monkeypatch, cfg, field)


@pytest.mark.parametrize("field", ["quality_policy_definition", "quality_policy", "threshold_from",
                                   "contract_version", "preset", "evidence_gate", "max_tool_calls"])
def test_missing_execution_descriptors_are_refused(monkeypatch, field):
    cfg = ri.effective_config()
    cfg.pop(field)
    refused_before_any_tool(monkeypatch, cfg)


@pytest.mark.parametrize("name", ["quality_v9", "", None, 7, ["quality_v1"]])
def test_an_invalid_quality_policy_name_is_refused(monkeypatch, name):
    cfg = ri.effective_config()
    cfg["quality_policy"] = name
    refused_before_any_tool(monkeypatch, cfg, "not a Contract v1 policy")


def test_unknown_extra_fields_and_non_objects_are_refused(monkeypatch):
    cfg = ri.effective_config()
    cfg["gate_override_reason"] = "trust me"
    refused_before_any_tool(monkeypatch, cfg, "do not execute")
    refused_before_any_tool(monkeypatch, ["not", "a", "config"], "JSON object")


def test_a_snapshot_captured_before_the_quality_registry_changed_is_refused(monkeypatch):
    cfg = ri.effective_config()                                        # valid when captured
    strict_registry(monkeypatch, min_usable_fraction=0.95)
    refused_before_any_tool(monkeypatch, cfg, "quality_policy_definition")


def test_a_quality_registry_change_during_execution_fails_closed(monkeypatch):
    calls = spy_compare(monkeypatch, before=lambda: strict_registry(monkeypatch, min_usable_fraction=0.95))
    with pytest.raises(ri.PolicyMismatch, match="quality_policy_definition"):
        run(ri.effective_config())
    assert calls == [1]                                                # executed, and its result refused


def test_a_quality_policy_removed_during_execution_fails_closed_without_fabricated_metadata(monkeypatch):
    seen = []
    spy_compare(monkeypatch, before=lambda: monkeypatch.delitem(v1.QUALITY_POLICIES, "quality_v1"),
                after=lambda r: seen.append(r) or r)
    with pytest.raises(ri.PolicyMismatch, match="not a Contract v1 policy"):
        run(ri.effective_config())
    (refusal,) = seen                                                  # the tool refused; it claims no analysis
    assert refusal["status"] == "error" and "quality_policy" not in refusal and "overlap_policy" not in refusal


@pytest.mark.parametrize("tool, field, value", [
    ("compare_windows", "quality_policy", "quality_v1_strict90"),
    ("compare_windows", "quality_policy_definition", {"name": "quality_v1", "min_usable_fraction": 1.0}),
    ("compare_windows", "pre_s", 5.0),
    ("compare_windows", "feature", "rel_speed"),
    ("compare_windows", "evidence_gate", {"min_n_windows": 1, "min_n_sessions": 1, "overridden": True,
                                          "override_reason": "x"}),
    ("compare_windows", "contract_version", "invented/v99"),
    ("query_events", "contract_version", "invented/v99"),
    ("query_events", "threshold", 0.99),
])
def test_executed_metadata_that_disagrees_with_the_snapshot_is_refused(monkeypatch, tool, field, value):
    spy_tool(monkeypatch, tool, after=lambda r: {**r, field: value})
    with pytest.raises(ri.PolicyMismatch, match="refused"):
        run(ri.effective_config())


def test_a_tool_refusal_is_kept_as_a_refusal_and_given_no_policy_metadata(monkeypatch):
    spy_compare(monkeypatch, after=lambda r: {"status": "error", "code": "bad_window", "message": "synthetic",
                                              "association_only": True, "contract_version": v1.CONTRACT_VERSION})
    res = run(ri.effective_config())
    c = res["rules"][0]["compare"]
    assert c["status"] == "error" and not set(c) & {"quality_policy", "quality_policy_definition", *ri.OVERLAP_KEYS}


def test_a_selectable_nondefault_quality_policy_runs_with_its_own_definition():
    cfg = ri.effective_config()
    cfg["quality_policy"] = "quality_v1_strict90"
    cfg["quality_policy_definition"] = v1.QUALITY_POLICIES["quality_v1_strict90"].to_json()
    res = run(cfg)
    c = res["rules"][0]["compare"]
    assert c["quality_policy"] == "quality_v1_strict90"
    assert c["quality_policy_definition"] == res["inputs"]["config"]["quality_policy_definition"] == \
        v1.QUALITY_POLICIES["quality_v1_strict90"].to_json()
    cfg["quality_policy_definition"] = v1.QUALITY_POLICIES["quality_v1"].to_json()     # name and definition disagree
    with pytest.raises(ri.PolicyMismatch, match="quality_policy_definition"):
        run(cfg)


def test_a_canonical_snapshot_matches_the_executed_metadata_field_by_field():
    cfg = ri.effective_config()
    res = run(cfg)
    (item,) = res["rules"]
    q, c, rec = item["query"], item["compare"], item["record"]
    assert res["inputs"]["config"] == cfg and res["inputs"]["config_sha256"] == ri.config_digest(cfg)
    assert q["threshold"] == approved(ATTACK).active()[0].ask_threshold
    assert cfg["threshold_from"] == "rule.ask_threshold"
    assert q["contract_version"] == c["contract_version"] == rec["contract_version"] == cfg["contract_version"]
    assert rec["max_tool_calls"] == cfg["max_tool_calls"]
    for k in ("quality_policy", "quality_policy_definition", *ri.OVERLAP_KEYS):
        assert c[k] == cfg[k]
    assert {k: c[k] for k in cfg["preset"]} == cfg["preset"]
    assert {k: c["evidence_gate"][k] for k in cfg["evidence_gate"]} == cfg["evidence_gate"]


def test_a_quality_mismatch_is_never_cached_or_exported(tmp_path, monkeypatch):
    drift = iter([0.95, 0.96, 0.97, 0.98])                            # every execution sees a NEW definition
    spy_compare(monkeypatch, before=lambda: strict_registry(monkeypatch, min_usable_fraction=next(drift)))
    cache = {}
    with pytest.raises(ri.PolicyMismatch):
        ri.cached_investigation(cache, ("identity",), ENTRY, approved(ATTACK).active(), "review")
    assert cache == {}
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(tmp_path / "replay"))      # isolated journals
    at = end_of_session(start(workflow(), replay_session=TRACK))
    assert any("The investigation was refused" in e.value for e in section(at).error)
    assert not [b for b in at.get("download_button") if b.label == "Download investigation record (JSON)"]
    assert at.session_state["replay_investigation_cache"] == {}


# ---- the executed query must be the approved rule's query (Codex review 3) ----

def query_with(monkeypatch, request=None, response=None, only=None):
    """Run the REAL query_events with an altered request and/or an altered copy of its real response (only for
    rules of behavior `only`, when given)."""
    real = v1._TOOL_FUNCS["query_events"]

    def wrapped(data, req):
        if only is not None and req.get("behavior") != only:
            return real(data, req)
        r = real(data, {**req, **(request or {})})
        return response(copy.deepcopy(r)) if response else r
    monkeypatch.setitem(v1._TOOL_FUNCS, "query_events", wrapped)


def test_codex_wrong_duration_reproduction_is_refused(monkeypatch):
    query_with(monkeypatch, request={"min_duration_s": 0.5})          # the approved rule requires 2.0 s
    with pytest.raises(ri.PolicyMismatch, match="another query than the approved rule"):
        run()


@pytest.mark.parametrize("change", [{"behavior": "mount"}, {"threshold": 0.9}, {"min_duration_s": 3.0}])
def test_a_query_run_with_other_rule_arguments_is_refused(monkeypatch, change):
    query_with(monkeypatch, request=change)
    with pytest.raises(ri.PolicyMismatch, match="another query than the approved rule"):
        run()


def set_path(r, path, value):
    *head, last = path
    target = r
    for k in head:
        target = target[k]
    if value is KeyError:
        target.pop(last)
    else:
        target[last] = value
    return r


@pytest.mark.parametrize("path, value, why", [
    (("query", "session"), ["SYNTHETIC_another_session"], "another query"),            # wrong session selection
    (("query", "session"), ENTRY.session, "another query"),                            # not the canonical list form
    (("query", "threshold"), KeyError, "another query"),                               # a missing canonical field
    (("query", "rule_id"), "R1", "another query"),                                     # an extra field
    (("query", "min_duration_s"), 0.5, "another query"),                               # top level says 2.0: conflict
    (("min_duration_s",), 0.5, "min_duration_s"),                                      # query says 2.0: conflict
    (("behavior",), "mount", "behavior"),
    (("threshold",), 0.9, "threshold"),
    (("n_sessions_searched",), 2, "n_sessions_searched"),
    (("query_run_id",), "inv_0000000000000000", "query_run_id"),
    (("query_run_id",), KeyError, "query_run_id"),
    (("n_bouts",), 99, "n_bouts"),
    (("data_source",), "classifier", "data_source"),
    (("latency_gap_s",), 1.0, "latency_gap_s"),
])
def test_conflicting_missing_or_altered_query_metadata_is_refused(monkeypatch, path, value, why):
    query_with(monkeypatch, response=lambda r: set_path(r, path, value))
    with pytest.raises(ri.PolicyMismatch, match=why):
        run()


def test_altered_bouts_are_refused(monkeypatch):
    query_with(monkeypatch, response=lambda r: {**r, "bouts": r["bouts"][1:], "n_bouts": len(r["bouts"]) - 1})
    with pytest.raises(ri.PolicyMismatch, match="bouts"):
        run()


def test_a_comparison_over_another_number_of_bouts_is_refused(monkeypatch):
    spy_compare(monkeypatch, after=lambda r: {**r, "n_input": r["n_input"] + 1})
    with pytest.raises(ri.PolicyMismatch, match="another number of bouts"):
        run()


def test_a_valid_query_returned_unchanged_is_accepted(monkeypatch):
    plain = run()
    query_with(monkeypatch)                                            # the real tool, untouched
    res = run()
    assert res == plain
    q = res["rules"][0]["query"]
    rule = approved(ATTACK).active()[0]
    assert q["query"] == {"session": [ENTRY.session], "behavior": rule.behavior,
                          "min_duration_s": rule.min_duration_s, "threshold": rule.ask_threshold}
    cited = [a["request"] for a in res["rules"][0]["record"]["attempts"] if a["tool"] == "compare_windows"]
    assert cited[0]["query_run_id"] == q["query_run_id"] and cited[0]["bouts"] == q["bouts"]


def test_a_refused_query_stays_a_refusal(monkeypatch):
    rule = {**ATTACK, "ask_threshold": 0.0}                            # query_events refuses threshold 0
    res = ri.investigate_replay(ENTRY, approved(rule).active(), "review")
    (item,) = res["rules"]
    assert item["query"]["status"] == "error" and item["compare"] is None and item["unsupported"]
    assert "query" not in item["query"] and "bouts" not in item["query"]       # no executed query is claimed


def test_a_mismatch_in_a_later_rule_refuses_the_whole_investigation(monkeypatch):
    query_with(monkeypatch, request={"min_duration_s": 0.5}, only="mount")  # R2 (mount) runs after R1 (attack)
    compared = spy_compare(monkeypatch)
    cache = {}
    with pytest.raises(ri.PolicyMismatch, match="rule R2"):
        ri.cached_investigation(cache, ("identity",), ENTRY, approved(ATTACK, MOUNT).active(), "review")
    assert compared == [1]                                             # R1 was fully investigated first ...
    assert cache == {}                                                 # ... and nothing partial is kept
    with pytest.raises(ri.PolicyMismatch, match="rule R2"):
        ri.investigate_replay(ENTRY, approved(ATTACK, MOUNT).active(), "review")    # nor returned


def test_a_query_mismatch_is_never_cached_or_exported(tmp_path, monkeypatch):
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(tmp_path / "replay"))      # isolated journals
    query_with(monkeypatch, request={"min_duration_s": 0.5})
    at = end_of_session(start(workflow(), replay_session=TRACK))
    assert any("The investigation was refused" in e.value for e in section(at).error)
    assert not [b for b in at.get("download_button") if b.label == "Download investigation record (JSON)"]
    assert at.session_state["replay_investigation_cache"] == {}
