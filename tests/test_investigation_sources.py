"""Tests for Stage 9 S9-3: classifier_prediction and monitor_event evidence
sources (investigation_sources.py). New file.

Synthetic score drops with hand-placed boundaries (SYNTHETIC_ sessions),
packaged and verified through score_drops; no annotation cache is involved.
"""

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

import live_tracker.event_log as el
import live_tracker.investigation_contracts as ic
import live_tracker.investigation_events as ie
import live_tracker.score_drops as sd
from live_tracker.investigation_events import CalMS21InvestigationTools
from live_tracker.investigation_sources import ClassifierEventSource, MonitorEventSource
from live_tracker.monitor import Monitor
from live_tracker.rules import RuleSet
from test_score_drops import MANIFEST, TEST_MEMBER, VAL, VAL_MEMBER, archive, csv_bytes, load, manifest  # noqa: F401

V1, V2 = VAL


def scores():
    """Two sessions with known attack runs (threshold 0.5):
    V1: [10,30) 0.9 + frame 30 exactly 0.5 -> [10,31); 31 = 0.49
        [40,50) 0.8 with tracking lost at 45 -> [40,45) and [46,50)
        [60,75) 0.95 (15 frames, exactly the minimum); [80,94) 0.95 (14 frames)
    V2: no attack; mount [5,40) 0.9."""
    def session(sid, n):
        f = np.arange(n)
        return pd.DataFrame({"session": sid, "frame": f, "time_s": f / 30.0, "attack": 0.1, "mount": 0.1,
                             "investigation": 0.1, "tracking_ok": True})
    a, b = session(V1, 120), session(V2, 90)
    a.loc[10:29, "attack"] = 0.9
    a.loc[30, "attack"] = 0.5
    a.loc[31, "attack"] = 0.49
    a.loc[40:49, "attack"] = 0.8
    a.loc[45, "tracking_ok"] = False
    a.loc[60:74, "attack"] = 0.95
    a.loc[80:93, "attack"] = 0.95
    b.loc[5:39, "mount"] = 0.9
    return pd.concat([a, b], ignore_index=True)


@pytest.fixture
def drop(tmp_path, manifest):  # noqa: F811
    return load(archive(tmp_path, {VAL_MEMBER: csv_bytes(scores()), TEST_MEMBER: csv_bytes()}), manifest)


@pytest.fixture
def policy():
    return ic.EvidencePolicy(MANIFEST, allow_synthetic=True)


def request(source, behavior="attack", **kw):
    return {"contract_version": ic.CONTRACT_VERSION, "source": source, "behavior": behavior, "sessions": None,
            "max_episodes": 500, "merge_gap_frames": None, "min_duration_frames": None, **kw}


def spans(result):
    return [(e["session"], e["start_frame"], e["end_frame"]) for e in result["measurements"]["episodes"]]


def classifier_tools(drop, policy, **kw):
    kw = {"threshold": 0.5, "min_duration_frames": 15, **kw}
    return CalMS21InvestigationTools(policy, [ClassifierEventSource(drop, policy, **kw)])


# ======================================================================
# classifier_prediction
# ======================================================================

def test_classifier_episodes_have_exact_half_open_boundaries(drop, policy):
    res = ic.call_tool(classifier_tools(drop, policy), "query_events", request("classifier_prediction"), policy)
    assert res["status"] == "ok"
    assert spans(res) == [(V1, 10, 31), (V1, 60, 75)]          # 0.5 qualifies (>=); 15 frames is enough; 14 is not
    e = res["measurements"]["episodes"][0]
    assert (e["start_s"], e["end_s"], e["fps"]) == (10 / 30, 31 / 30, 30.0) and "annotated_frames" not in e
    assert e["episode_id"] == f"ep:classifier_prediction:attack:{V1}:10"
    assert res["measurements"]["sessions_searched"] == sorted(VAL)
    assert res["measurements"]["sessions_without_episodes"] == [V2]


@pytest.mark.parametrize("source_min, req, expected", [
    (5, {}, [(V1, 10, 31), (V1, 40, 45), (V1, 60, 75), (V1, 80, 94)]),          # tracking loss splits [40,50)
    (5, {"min_duration_frames": 15}, [(V1, 10, 31), (V1, 60, 75)]),            # the stricter minimum applies
    (15, {"min_duration_frames": 5}, [(V1, 10, 31), (V1, 60, 75)]),
    (1, {}, [(V1, 10, 31), (V1, 40, 45), (V1, 46, 50), (V1, 60, 75), (V1, 80, 94)]),
    (5, {"merge_gap_frames": 1}, [(V1, 10, 31), (V1, 40, 50), (V1, 60, 75), (V1, 80, 94)]),   # merged over frame 45
])
def test_minimum_duration_tracking_gaps_and_merging(drop, policy, source_min, req, expected):
    tools = classifier_tools(drop, policy, min_duration_frames=source_min)
    assert spans(ic.call_tool(tools, "query_events", request("classifier_prediction", **req), policy)) == expected


def test_classifier_provenance_and_disclosures(drop, policy):
    res = ic.call_tool(classifier_tools(drop, policy), "query_events", request("classifier_prediction"), policy)
    prov = res["measurements"]["episodes"][0]["provenance"]
    assert prov == {"model_id": "unverified-label:model_b", "score_file_sha256": drop.provenance.member_sha256,
                    "threshold": 0.5, "min_duration_frames": 15}
    method = {m["name"]: m for m in res["methodology"]}
    assert "threshold crossing, not annotated behavioral onset" in method["episode_definition"]["note"]
    assert method["episode_rules_version"]["value"] == "prediction-episodes/1"
    assert "UNVERIFIED" in method["model_identity"]["note"]
    assert any("not human annotations" in x for x in res["limitations"])
    assert any("threshold crossing" in x for x in res["limitations"])
    assert {d["sha256"] for d in res["provenance"]["data"]} == {drop.provenance.archive_sha256,
                                                                drop.provenance.member_sha256}
    other = ic.call_tool(classifier_tools(drop, policy), "query_events",
                         request("classifier_prediction", behavior="mount"), policy)
    assert spans(other) == [(V2, 5, 40)]


def test_never_falls_back_to_annotations(drop, policy, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("annotations were read")
    monkeypatch.setattr(ie, "session_labels", forbidden)
    monkeypatch.setattr(ie, "load_calms21", forbidden)
    tools = classifier_tools(drop, policy)
    assert ic.call_tool(tools, "query_events", request("classifier_prediction"), policy)["status"] == "ok"
    res = ic.call_tool(tools, "query_events", request("validation_annotation"), policy)
    assert res["status"] == "error" and res["errors"][0]["code"] == "source_unavailable"
    assert "no other source is substituted" in res["errors"][0]["message"]


def test_only_issued_unchanged_drops_are_accepted(drop, policy):
    for fake in (sd.ScoreDrop(provenance=drop.provenance, _table=drop.scores()), drop.provenance, None):
        with pytest.raises(ic.ForbiddenEvidence, match="not an issued, verified ScoreDrop"):
            ClassifierEventSource(fake, policy, threshold=0.5, min_duration_frames=15)
    tools = classifier_tools(drop, policy)
    drop._table.loc[12, "attack"] = 0.0                                          # changed after construction
    res = ic.call_tool(tools, "query_events", request("classifier_prediction"), policy)
    assert res["status"] == "error" and "changed since this source was created" in res["errors"][0]["message"]


@pytest.mark.parametrize("kw, text", [({"threshold": 0.0}, "threshold"), ({"threshold": 1.5}, "threshold"),
                                      ({"threshold": True}, "threshold"), ({"min_duration_frames": 0}, "min_dur"),
                                      ({"min_duration_frames": 15.0}, "min_dur")])
def test_source_settings_are_explicit_and_checked(drop, policy, kw, text):
    with pytest.raises(ValueError, match=text):
        ClassifierEventSource(drop, policy, **{"threshold": 0.5, "min_duration_frames": 15, **kw})


def test_test_sessions_and_mismatched_splits_are_refused(drop, policy):
    tools = classifier_tools(drop, policy)
    res = ic.call_tool(tools, "query_events", request("classifier_prediction", sessions=["SYNTHETIC_test_1"]), policy)
    assert res["status"] == "error" and res["errors"][0]["code"] == "forbidden_split"
    assert "held-out test session" in res["errors"][0]["message"] and res["measurements"] is None
    bigger = ic.EvidencePolicy({**MANIFEST, "validation": VAL + ["SYNTHETIC_val_3"]}, allow_synthetic=True)
    with pytest.raises(ic.ForbiddenEvidence, match="not exactly the policy's validation split"):
        ClassifierEventSource(drop, bigger, threshold=0.5, min_duration_frames=15)
    assert spans(ic.call_tool(tools, "query_events", request("classifier_prediction", sessions=[V2]), policy)) == []


def test_synthetic_evidence_needs_a_synthetic_policy(drop):
    strict = ic.EvidencePolicy(MANIFEST)                                        # synthetic fixtures not allowed
    tools = CalMS21InvestigationTools(strict, [ClassifierEventSource(drop, strict, threshold=0.5,
                                                                     min_duration_frames=15)])
    res = ic.call_tool(tools, "query_events", request("classifier_prediction"), strict)
    assert res["status"] == "error" and res["measurements"] is None             # synthetic data is not evidence


# ======================================================================
# monitor_event
# ======================================================================

def approved(rid="R1", behavior="attack", min_s=0.5):
    rules = RuleSet()
    rules.add({"rule_id": rid, "behavior": behavior, "min_duration_s": min_s, "alert_threshold": 0.8,
               "ask_threshold": 0.5})
    rules.approve(rid)
    return rules.active()


def frozen_run(drop, provenance="drop", rules=None, run_id="mon_run"):
    m = Monitor(rules or approved(), run_id=run_id, provenance=drop if provenance == "drop" else provenance)
    for row in drop.rows():
        m.step(row)
    m.finish()
    return el.freeze_run(m.journal, run_id)


def test_monitor_episodes_distinguish_onset_firing_and_closure(drop, policy):
    snap = frozen_run(drop)
    source = MonitorEventSource(snap, drop, policy, rule_id="R1")
    tools = CalMS21InvestigationTools(policy, [source])
    res = ic.call_tool(tools, "query_events", request("monitor_event"), policy)
    # 0.5 s needs 16 frames by elapsed time: only [10,31) fires; [60,75) (15 frames) does not (unlike the
    # classifier source's 15-frame minimum: the two rules are different and both are kept as they are)
    assert spans(res) == [(V1, 10, 31)]
    e = res["measurements"]["episodes"][0]
    life = source.lifecycle(e["provenance"]["event_id"])
    assert (life["detection_onset_frame"], life["alert_fired_frame"], life["closure_end_frame"]) == (10, 25, 31)
    assert life["last_frame"] == 30 and life["end_reason"] == "below_threshold" and life["action"] == "alert"
    assert (e["start_frame"], e["end_frame"]) == (life["detection_onset_frame"], life["closure_end_frame"])
    assert e["provenance"] == {"journal_sha256": snap.sha256, "run_id": "mon_run", "rule_id": "R1",
                               "event_id": life["event_id"]}
    method = {m["name"]: m for m in res["methodology"]}
    assert "detection onset" in method["episode_definition"]["note"] and "not a boundary" in \
        method["episode_definition"]["note"]
    assert method["journal_snapshot_sha256"]["value"] == snap.sha256
    assert any("not at the alert" in x for x in res["limitations"])
    assert snap.sha256 in {d["sha256"] for d in res["provenance"]["data"]}


def test_monitor_source_refuses_unverified_or_mismatched_runs(drop, policy, tmp_path, manifest):  # noqa: F811
    declared = frozen_run(drop, provenance=drop.provenance)                     # declared, not verified
    with pytest.raises(ic.ForbiddenEvidence, match="declared_unverified"):
        MonitorEventSource(declared, drop, policy, rule_id="R1")
    legacy = frozen_run(drop, provenance=None)
    with pytest.raises(ic.ForbiddenEvidence, match="not complete|no_source_provenance"):
        MonitorEventSource(legacy, drop, policy, rule_id="R1")
    other_drop = load(archive(tmp_path, name="other.zip"), manifest)           # a different verified drop
    with pytest.raises(ic.ForbiddenEvidence, match="not monitored from this score drop"):
        MonitorEventSource(frozen_run(drop), other_drop, policy, rule_id="R1")
    with pytest.raises(ValueError, match="not a rule of this monitor run"):
        MonitorEventSource(frozen_run(drop), drop, policy, rule_id="R9")
    with pytest.raises(ic.ForbiddenEvidence, match="RunSnapshot"):
        MonitorEventSource({"sha256": "x"}, drop, policy, rule_id="R1")


def test_tampered_or_incomplete_snapshots_are_refused(drop, policy):
    snap = frozen_run(drop)
    forged = el.RunSnapshot(snap.run_id, snap.provenance_status, snap.completion, snap.n_events, snap.sha256,
                            snap.canonical.replace(b'"last_frame":30', b'"last_frame":31'))
    with pytest.raises(ic.ForbiddenEvidence, match="does not verify"):
        MonitorEventSource(forged, drop, policy, rule_id="R1")
    payload = snap.payload()
    payload["records"] = []                                                    # deleted event pair, resealed
    data = el.canonical_bytes(payload)
    resealed = el.RunSnapshot(snap.run_id, snap.provenance_status, snap.completion, 0,
                              hashlib.sha256(data).hexdigest(), data)
    with pytest.raises(ic.ForbiddenEvidence, match="does not verify"):
        MonitorEventSource(resealed, drop, policy, rule_id="R1")
    m = Monitor(approved(), run_id="unfinished", provenance=drop)
    for row in drop.rows()[:50]:
        m.step(row)
    with pytest.raises(el.JournalIntegrityError, match="never frozen"):       # no snapshot of an unfinished run
        el.freeze_run(m.journal, "unfinished")


def test_monitor_source_query_rules(drop, policy):
    tools = CalMS21InvestigationTools(policy, [MonitorEventSource(frozen_run(drop), drop, policy, rule_id="R1")])
    res = ic.call_tool(tools, "query_events", request("monitor_event", merge_gap_frames=2), policy)
    assert res["status"] == "error" and "never merged" in res["errors"][0]["message"]
    assert spans(ic.call_tool(tools, "query_events", request("monitor_event", behavior="mount"), policy)) == []
    assert spans(ic.call_tool(tools, "query_events", request("monitor_event", min_duration_frames=22), policy)) == []
    res = ic.call_tool(tools, "query_events", request("monitor_event", sessions=["SYNTHETIC_test_1"]), policy)
    assert res["errors"][0]["code"] == "forbidden_split"
    res = ic.call_tool(tools, "query_events", request("classifier_prediction"), policy)
    assert res["errors"][0]["code"] == "source_unavailable"                      # no substitution between sources


def test_multi_rule_runs_answer_for_one_rule(drop, policy):
    rules = RuleSet()
    for rid, b in (("RA", "attack"), ("RM", "mount")):
        rules.add({"rule_id": rid, "behavior": b, "min_duration_s": 0.5, "alert_threshold": 0.8,
                   "ask_threshold": 0.5})
        rules.approve(rid)
    snap = frozen_run(drop, rules=rules.active())
    mount = CalMS21InvestigationTools(policy, [MonitorEventSource(snap, drop, policy, rule_id="RM")])
    assert spans(ic.call_tool(mount, "query_events", request("monitor_event", behavior="mount"), policy)) == \
        [(V2, 5, 40)]
    assert spans(ic.call_tool(mount, "query_events", request("monitor_event"), policy)) == []


def test_the_contract_validator_still_checks_monitor_provenance(drop, policy):
    tools = CalMS21InvestigationTools(policy, [MonitorEventSource(frozen_run(drop), drop, policy, rule_id="R1")])
    res = ic.call_tool(tools, "query_events", request("monitor_event"), policy)
    bad = copy.deepcopy(res)
    bad["measurements"]["episodes"][0]["provenance"]["journal_sha256"] = "not-a-digest"
    with pytest.raises(ic.ContractError):
        ic.validate_result("query_events", request("monitor_event"), bad, policy)
