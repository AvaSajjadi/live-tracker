"""Tests for Stage 9 S9-4: get_window_features, check_tracking_quality and
compare_windows over classifier_prediction and monitor_event episodes
(investigation_windows.evidence_context). New file.

The synthetic pose cache (test_investigation_windows.build) annotates attacks
at A [100,120) [130,140) [300,320), B [200,230); the synthetic classifier
predicts attacks ELSEWHERE (A [150,180) [250,270), B [60,90), C [330,360)),
so any use of annotations by the prediction paths is visible.
"""

import copy

import numpy as np
import pandas as pd
import pytest

import live_tracker.calms21 as calms21
import live_tracker.event_log as el
import live_tracker.investigation_contracts as ic
import live_tracker.investigation_events as ie
import live_tracker.investigation_windows as iw
import live_tracker.score_drops as sd
from live_tracker.investigation_events import CalMS21InvestigationTools
from live_tracker.investigation_sources import ClassifierEventSource, MonitorEventSource
from live_tracker.monitor import Monitor
from live_tracker.rules import RuleSet
from test_investigation_agent import LOOSE
from test_investigation_windows import N, VAL_A, VAL_B, VAL_C, build
from test_score_drops import TEST_MEMBER, VAL_MEMBER, archive, csv_bytes

PREDICTED = {VAL_A: [(150, 180), (250, 270)], VAL_B: [(60, 90)], VAL_C: [(330, 360)]}


def score_table(n=N):
    parts = []
    for sid in (VAL_A, VAL_B, VAL_C):
        f = np.arange(n)
        d = pd.DataFrame({"session": sid, "frame": f, "time_s": f / 30.0, "attack": 0.1, "mount": 0.1,
                          "investigation": 0.1, "tracking_ok": True})
        for a, b in PREDICTED[sid]:
            d.loc[a:b - 1, "attack"] = 0.9
        d.loc[20:29, "attack"] = 0.9 if sid == VAL_C else 0.1       # a 10-frame run: below the source minimum
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def world(tmp_path):
    tools0, policy, cache, manifest = build(tmp_path)
    drop = sd.load_validation_scores(
        archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table()), TEST_MEMBER: csv_bytes(score_table())}),
        VAL_MEMBER, manifest_path=manifest)
    ann = tools0.sources["validation_annotation"]
    clf = ClassifierEventSource(drop, policy, threshold=0.5, min_duration_frames=15)
    rules = RuleSet()
    rules.add({"rule_id": "R1", "behavior": "attack", "min_duration_s": 0.5, "alert_threshold": 0.8,
               "ask_threshold": 0.5})
    rules.approve("R1")
    m = Monitor(rules.active(), run_id="mon", provenance=drop)
    for row in drop.rows():
        m.step(row)
    m.finish()
    snap = el.freeze_run(m.journal, "mon")
    mon = MonitorEventSource(snap, drop, policy, rule_id="R1")
    return {"policy": policy, "drop": drop, "ann_only": tools0, "snap": snap, "mon_source": mon,
            "clf": CalMS21InvestigationTools(policy, [ann, clf]),
            "mon": CalMS21InvestigationTools(policy, [ann, mon]), "cache": cache, "manifest": manifest}


def call(tools, tool, request, policy):
    return ic.call_tool(tools, tool, {"contract_version": ic.CONTRACT_VERSION, **request}, policy)


def episodes(tools, policy, source):
    res = call(tools, "query_events", {"source": source, "behavior": "attack", "sessions": None, "max_episodes": 500,
                                       "merge_gap_frames": None, "min_duration_frames": None}, policy)
    return [e["episode_id"] for e in res["measurements"]["episodes"]]


def features(tools, policy, episode_ids, clearance=10, length=20, gap=0):
    return call(tools, "get_window_features", {
        "episode_ids": episode_ids, "pre_event": {"length_frames": length, "gap_frames": gap},
        "baseline": {"strategy": "same_session_tiles", "length_frames": length, "clearance_frames": clearance,
                     "max_windows_per_session": 500}, "features": ["dist_rate"]}, policy)


def compare(tools, policy, pre, base):
    return call(tools, "compare_windows", {
        "feature": "dist_rate", "statistic": "median_difference", "aggregation": "session_level",
        "pre_window_ids": pre, "baseline_window_ids": base, "sufficiency": LOOSE.to_json(),
        "uncertainty": {"method": "none", "level": None, "resamples": None, "seed": None}}, policy)


def by_span(res, sid, kind, start):
    return next(w for w in res["measurements"]["windows"]
                if w["session"] == sid and w["kind"] == kind and w["start_frame"] == start)


def methods(res):
    return {m["name"]: m for m in res["methodology"]}


# ======================================================================
# classifier_prediction windows
# ======================================================================

def test_classifier_feature_windows(world):
    t, p = world["clf"], world["policy"]
    eps = episodes(t, p, "classifier_prediction")
    assert eps == [f"ep:classifier_prediction:attack:{VAL_A}:150", f"ep:classifier_prediction:attack:{VAL_A}:250",
                   f"ep:classifier_prediction:attack:{VAL_B}:60", f"ep:classifier_prediction:attack:{VAL_C}:330"]
    res = features(t, p, eps)
    assert res["status"] == "ok"
    pre = by_span(res, VAL_A, "pre_event", 130)
    assert (pre["start_frame"], pre["end_frame"]) == (130, 150)                   # [onset - 20, onset), exclusive
    assert all(w["label_context"] is None for w in res["measurements"]["windows"])
    m = methods(res)
    assert m["event_source"]["value"] == "classifier_prediction"
    assert m["onset_meaning"]["value"] == "classifier threshold crossing"
    assert "pre-threshold-crossing, not necessarily pre-behavior" in m["onset_meaning"]["note"]
    assert "never an annotated frame" in m["event_overlap_rule"]["note"]
    assert any("not necessarily before the behavior" in x for x in res["limitations"])
    assert iw.NOT_ASSESSED_NOTE in res["limitations"]
    assert not any(x == iw.ANNOTATION_NOTE for x in res["limitations"])
    assert world["drop"].provenance.member_sha256 in {d["sha256"] for d in res["provenance"]["data"]}


def test_exclusions_use_the_declared_source_not_annotations(world):
    t, p = world["clf"], world["policy"]
    res = features(t, p, episodes(t, p, "classifier_prediction"))
    # pre-window [130,150) holds the ANNOTATED attack [130,140): included, because annotations are not read
    assert by_span(res, VAL_A, "pre_event", 130)["exclusion_reasons"] == []
    # baseline tile [100,120) is an annotated attack, far from predicted frames: not excluded as an event
    assert not {"overlaps_event", "near_event"} & set(by_span(res, VAL_A, "baseline", 100)["exclusion_reasons"])
    # baseline tile [160,180) holds predicted frames: overlaps_event
    assert "overlaps_event" in by_span(res, VAL_A, "baseline", 160)["exclusion_reasons"]
    # a tile right after a predicted episode is near_event (clearance 10 over [history start, end))
    assert "near_event" in by_span(res, VAL_A, "baseline", 180)["exclusion_reasons"]
    ann = features(world["ann_only"], p, [f"ep:validation_annotation:attack:{VAL_A}:100"])
    assert "overlaps_event" in by_span(ann, VAL_A, "baseline", 100)["exclusion_reasons"]   # the annotation path


def test_no_annotation_is_read_on_prediction_paths(world, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("annotations were read")
    for mod, name in ((iw, "session_labels"), (iw, "_label_context"), (iw, "load_calms21"), (ie, "session_labels"),
                      (calms21, "_session_labels")):
        monkeypatch.setattr(mod, name, forbidden)
    for key, source in (("clf", "classifier_prediction"), ("mon", "monitor_event")):
        t, p = world[key], world["policy"]
        res = features(t, p, episodes(t, p, source))
        assert res["status"] == "ok", res.get("errors")
        included = [w for w in res["measurements"]["windows"] if w["status"] == "included"]
        pre = [w["window_id"] for w in included if w["kind"] == "pre_event"]
        base = [w["window_id"] for w in included if w["kind"] == "baseline"]
        q = call(t, "check_tracking_quality", {"window_ids": pre[:3] + base[:3], "history_frames": [1]}, p)
        assert q["status"] == "ok" and methods(q)["event_source"]["value"] == source
        c = compare(t, p, pre, base)
        assert c["status"] in ("ok", "insufficient_evidence")
        assert methods(c)["label_frames_pre"]["value"] is None and "NOT COMPUTED" in \
            methods(c)["label_frames_pre"]["note"]


def test_comparison_is_descriptive_deterministic_and_source_labelled(world):
    t, p = world["clf"], world["policy"]
    res = features(t, p, episodes(t, p, "classifier_prediction"))
    included = [w for w in res["measurements"]["windows"] if w["status"] == "included"]
    pre = [w["window_id"] for w in included if w["kind"] == "pre_event"]
    base = [w["window_id"] for w in included if w["kind"] == "baseline"]
    a, b = compare(t, p, pre, base), compare(t, p, pre, base)
    assert a == b and a["evidence_id"] == b["evidence_id"]                     # deterministic evidence
    assert a["measurements"]["effect"]["ci_low"] is None and methods(a)["uncertainty_method"]["value"] == "none"
    assert methods(a)["event_source"]["value"] == "classifier_prediction"
    assert any("Association, not causation" in x for x in a["limitations"])     # descriptive, not causal
    assert features(t, p, episodes(t, p, "classifier_prediction")) == res


# ======================================================================
# monitor_event windows
# ======================================================================

def test_monitor_feature_windows_are_pre_detection(world):
    t, p = world["mon"], world["policy"]
    eps = episodes(t, p, "monitor_event")
    assert eps == [f"ep:monitor_event:attack:{VAL_A}:150", f"ep:monitor_event:attack:{VAL_A}:250",
                   f"ep:monitor_event:attack:{VAL_B}:60", f"ep:monitor_event:attack:{VAL_C}:330"]
    res = features(t, p, eps)
    m = methods(res)
    assert m["event_source"]["value"] == "monitor_event" and m["onset_meaning"]["value"] == "monitor detection onset"
    assert "the alert fires later" in m["onset_meaning"]["note"]
    assert m["journal_snapshot_sha256"]["value"] == world["snap"].sha256
    assert world["snap"].sha256 in {d["sha256"] for d in res["provenance"]["data"]}
    assert any("Detection onset precedes alert firing" in x and "precedes both" in x for x in res["limitations"])
    assert not any("not before the alert" in x for x in res["limitations"])
    assert any("not interchangeable" in x for x in res["limitations"])
    life = world["mon_source"].lifecycle(next(x["event_id"] for x in world["mon_source"]._lifecycles
                                              if x["detection_onset_frame"] == 150))
    assert (life["detection_onset_frame"], life["alert_fired_frame"], life["closure_end_frame"]) == (150, 165, 180)
    assert by_span(res, VAL_A, "pre_event", 130)["end_frame"] == life["detection_onset_frame"]


# ======================================================================
# refusals
# ======================================================================

def test_mixed_sources_are_refused(world):
    t, p = world["clf"], world["policy"]
    clf_eps = episodes(t, p, "classifier_prediction")
    with pytest.raises(ic.ContractError, match="mixes sources"):
        features(t, p, clf_eps[:1] + [f"ep:validation_annotation:attack:{VAL_A}:100"])
    res = features(t, p, clf_eps)
    ann = features(t, p, [f"ep:validation_annotation:attack:{VAL_A}:100"])
    pre = [w["window_id"] for w in res["measurements"]["windows"] if w["kind"] == "pre_event"][:1]
    base = [w["window_id"] for w in ann["measurements"]["windows"] if w["kind"] == "baseline"][:1]
    with pytest.raises(ic.ContractError, match="mixes sources"):
        compare(t, p, pre, base)
    q = call(t, "check_tracking_quality", {"window_ids": pre + base, "history_frames": [1]}, p)
    assert q["status"] == "error" and "one evidence source only" in q["errors"][0]["message"]


@pytest.mark.parametrize("episode, text", [
    (f"ep:classifier_prediction:attack:{VAL_A}:151", "is not the first frame of a classifier_prediction"),
    (f"ep:classifier_prediction:attack:{VAL_C}:20", "is not the first frame"),      # a run below the minimum
    (f"ep:classifier_prediction:attack:{VAL_A}:100", "is not the first frame"),     # an ANNOTATED onset only
])
def test_episodes_the_source_cannot_produce_are_unknown(world, episode, text):
    res = features(world["clf"], world["policy"], [episode])
    assert res["status"] == "error" and res["errors"][0]["code"] == "unknown_id" and text in res["errors"][0]["message"]


def test_missing_sources_pose_cache_or_alignment_are_unavailable(world, tmp_path):
    p = world["policy"]
    eps = episodes(world["clf"], p, "classifier_prediction")
    for tools, text in ((world["ann_only"], "not available"),                    # no classifier source plugged in
                        (CalMS21InvestigationTools(p, [world["clf"].sources["classifier_prediction"]]), "pose cache")):
        res = features(tools, p, eps[:1])
        assert res["status"] == "error" and res["errors"][0]["code"] == "source_unavailable"
        assert text in res["errors"][0]["message"]
    short = sd.load_validation_scores(archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table(N - 1)),
                                                         TEST_MEMBER: csv_bytes(score_table())}, name="short.zip"),
                                      VAL_MEMBER, manifest_path=world["manifest"])
    tools = CalMS21InvestigationTools(p, [world["ann_only"].sources["validation_annotation"],
                                          ClassifierEventSource(short, p, threshold=0.5, min_duration_frames=15)])
    res = features(tools, p, eps[:1])
    assert res["status"] == "error" and "frame alignment" in res["errors"][0]["message"]


def test_a_changed_score_drop_is_refused_before_measuring(world):
    t, p = world["clf"], world["policy"]
    eps = episodes(t, p, "classifier_prediction")
    world["drop"]._table.loc[0, "attack"] = 0.5
    res = features(t, p, eps)
    assert res["status"] == "error" and "changed since this source was created" in res["errors"][0]["message"]


def test_insufficient_and_edge_paths(world):
    t, p = world["clf"], world["policy"]
    res = features(t, p, [f"ep:classifier_prediction:attack:{VAL_B}:60"], length=90)
    assert by_span(res, VAL_B, "pre_event", 0)["exclusion_reasons"][0] == "starts_before_session"
    eps = episodes(t, p, "classifier_prediction")
    f = features(t, p, eps)
    pre = [w["window_id"] for w in f["measurements"]["windows"] if w["kind"] == "pre_event" and w["status"] == "included"]
    base = [w["window_id"] for w in f["measurements"]["windows"] if w["kind"] == "baseline" and w["status"] == "included"]
    strict = dict(LOOSE.to_json(), min_sessions=4)
    c = call(t, "compare_windows", {"feature": "dist_rate", "statistic": "median_difference",
                                    "aggregation": "session_level", "pre_window_ids": pre[:2],
                                    "baseline_window_ids": base[:2], "sufficiency": strict,
                                    "uncertainty": {"method": "none", "level": None, "resamples": None, "seed": None}},
             p)
    assert c["status"] == "insufficient_evidence" and c["measurements"]["effect"]["value"] is None


# ======================================================================
# annotation regressions
# ======================================================================

def test_annotation_windows_are_unchanged_by_extra_sources(world):
    p = world["policy"]
    ep = f"ep:validation_annotation:attack:{VAL_A}:100"
    alone, together = features(world["ann_only"], p, [ep]), features(world["clf"], p, [ep])
    assert alone == together
    assert methods(alone)["event_source"]["value"] == "validation_annotation"
    assert any(w["label_context"] is not None for w in alone["measurements"]["windows"])
    assert iw.ANNOTATION_NOTE in alone["limitations"] and "onset_meaning" not in methods(alone)


# ======================================================================
# Codex S9-4 follow-up: limitations by source, merged episodes, monitor timing
# ======================================================================

import live_tracker.investigation_compare as icmp  # noqa: E402

POSITIVE_ANNOTATION_CLAIMS = ("mostly annotated", "annotations after the window are read", "label_frames_* in",
                              "Behavior frames are human annotations", "frame annotated with the behavior",
                              "see label_context", "Investigation confound (not resolved)")


def all_texts(res):
    return list(res["limitations"]) + [m["note"] or "" for m in res["methodology"]] + \
        [str(m["value"]) for m in res["methodology"]]


def three_results(t, p, source):
    f = features(t, p, episodes(t, p, source))
    inc = [w for w in f["measurements"]["windows"] if w["status"] == "included"]
    pre = [w["window_id"] for w in inc if w["kind"] == "pre_event"]
    base = [w["window_id"] for w in inc if w["kind"] == "baseline"]
    q = call(t, "check_tracking_quality", {"window_ids": pre[:2] + base[:2], "history_frames": [1]}, p)
    return f, q, compare(t, p, pre, base)


@pytest.mark.parametrize("key, source", [("clf", "classifier_prediction"), ("mon", "monitor_event")])
def test_prediction_limitations_are_complete_and_never_claim_annotations(world, key, source):
    t, p = world[key], world["policy"]
    src = t.sources[source]
    f, q, c = three_results(t, p, source)
    synthetic = ["SYNTHETIC test fixture: not CalMS21 data, not evidence."]
    tail = [iw.NOT_ASSESSED_NOTE]
    assert f["limitations"] == synthetic + src.window_limitations() + iw.FEATURE_LIMITATIONS_PREDICTION + tail
    assert q["limitations"] == synthetic + src.window_limitations() + iw.QUALITY_LIMITATIONS_PREDICTION + tail
    assert c["limitations"] == synthetic + src.window_limitations() + icmp.LIMITATIONS_PREDICTION + tail
    for res in (f, q, c):
        texts = all_texts(res)
        for phrase in POSITIVE_ANNOTATION_CLAIMS:
            assert not any(phrase in x for x in texts), (phrase, [x for x in texts if phrase in x])
        assert iw.ANNOTATION_NOTE not in texts and any("NOT assessed" in x for x in texts)
        causal = next(m for m in res["methodology"] if m["name"] == "causal_computation")["note"]
        assert "annotations are never read" in causal


def test_annotation_limitations_are_unchanged(world):
    p = world["policy"]
    f = features(world["ann_only"], p, [f"ep:validation_annotation:attack:{VAL_A}:100"])
    assert f["limitations"] == ["SYNTHETIC test fixture: not CalMS21 data, not evidence."] + iw.FEATURE_LIMITATIONS
    causal = next(m for m in f["methodology"] if m["name"] == "causal_computation")["note"]
    assert causal.endswith("annotations after the window are read by the exclusion rules")


def test_codex_merged_episode_round_trip(world):
    t, p = world["clf"], world["policy"]
    q = call(t, "query_events", {"source": "classifier_prediction", "behavior": "attack", "sessions": [VAL_C],
                                 "max_episodes": 500, "merge_gap_frames": 300, "min_duration_frames": None}, p)
    merged = next(e for e in q["measurements"]["episodes"] if e["start_frame"] == 20)
    assert (merged["start_frame"], merged["end_frame"]) == (20, 360)        # [20,30) + [330,360), half-open
    res = features(t, p, [merged["episode_id"]])
    assert res["status"] in ("ok", "no_data") and "errors" not in res or not res["errors"], res.get("errors")
    m = methods(res)
    assert m["merged_episodes"]["value"] == 1 and "merge_gap_frames 300" in m["merged_episodes"]["note"]
    assert "below threshold or without tracking" in m["merged_episodes"]["note"]
    pre = [w["window_id"] for w in res["measurements"]["windows"] if w["kind"] == "pre_event"]
    qual = call(t, "check_tracking_quality", {"window_ids": pre, "history_frames": [1]}, p)
    assert qual["status"] in ("ok", "no_data") and not qual["errors"]
    # [20,30) is shorter than the source minimum: valid only because this source returned the merged episode
    fresh = CalMS21InvestigationTools(p, [world["ann_only"].sources["validation_annotation"],
                                          ClassifierEventSource(world["drop"], p, threshold=0.5,
                                                                min_duration_frames=15)])
    assert features(fresh, p, [merged["episode_id"]])["errors"][0]["code"] == "unknown_id"


@pytest.mark.parametrize("frame", [21, 25, 29, 30, 345])
def test_forged_merged_starts_are_refused(world, frame):
    t, p = world["clf"], world["policy"]
    call(t, "query_events", {"source": "classifier_prediction", "behavior": "attack", "sessions": [VAL_C],
                             "max_episodes": 500, "merge_gap_frames": 300, "min_duration_frames": None}, p)
    res = features(t, p, [f"ep:classifier_prediction:attack:{VAL_C}:{frame}"])
    assert res["status"] == "error" and res["errors"][0]["code"] == "unknown_id"


# ---- Codex: an issuance record is re-derived, never trusted for membership ----

def test_codex_fabricated_issuance_record_is_refused(world):
    src = world["clf"].sources["classifier_prediction"]
    eid = ic.episode_id("classifier_prediction", "attack", VAL_C, 20)
    assert not src.is_episode_start(VAL_C, "attack", 20)
    src._issued[eid] = (30, 0, 15)                         # Codex: [20,30) is 10 frames, below the minimum
    assert not src.is_episode_start(VAL_C, "attack", 20)
    res = features(world["clf"], world["policy"], [eid])
    assert res["status"] == "error" and res["errors"][0]["code"] == "unknown_id"


def _merged_query(world):
    t, p = world["clf"], world["policy"]
    call(t, "query_events", {"source": "classifier_prediction", "behavior": "attack", "sessions": [VAL_C],
                             "max_episodes": 500, "merge_gap_frames": 300, "min_duration_frames": None}, p)
    src = t.sources["classifier_prediction"]
    eid = ic.episode_id("classifier_prediction", "attack", VAL_C, 20)
    assert src._issued[eid] == (360, 300, 15) and src.is_episode_start(VAL_C, "attack", 20)
    return src, eid


@pytest.mark.parametrize("altered", [
    (359, 300, 15), (361, 300, 15), (30, 300, 15),          # end frame
    (360, 5, 15), (360, 0, 15), (360, None, 15), (360, -1, 15), (360, 299, 15),   # merge gap
    (360, 300, 14), (360, 300, 5), (360, 300, 400),         # minimum (below the source's, or not met)
    (360.0, 300, 15), (360, "300", 15), (360, 300, True), [360, 300, 15], (360, 300),   # malformed
])
def test_altered_issuance_records_are_refused(world, altered):
    src, eid = _merged_query(world)
    src._issued[eid] = altered
    assert not src.is_episode_start(VAL_C, "attack", 20)


def test_a_genuine_merged_episode_still_round_trips(world):
    src, eid = _merged_query(world)
    res = features(world["clf"], world["policy"], [eid])
    assert not res.get("errors") and methods(res)["merged_episodes"]["value"] == 1
    assert src.is_episode_start(VAL_C, "attack", 330)      # an unmerged run of 30 frames stays valid on its own


@pytest.mark.parametrize("frame", [21, 25, 29, 331, 359])
def test_mid_run_starts_are_refused_even_with_an_issuance_record(world, frame):
    src, _ = _merged_query(world)
    eid = ic.episode_id("classifier_prediction", "attack", VAL_C, frame)
    src._issued[eid] = (360, 300, 15)
    assert not src.is_episode_start(VAL_C, "attack", frame)
