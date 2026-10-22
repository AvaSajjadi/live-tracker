"""Tests for the Contract v1 tool adapters (investigation_tools_v1.py). New file.

Everything here is SYNTHETIC: made-up scores (fake_scores) and made-up pose in
the CalMS21 layout. Nothing reads CalMS21, a cache or a score archive.
"""

import copy
import os

import numpy as np
import pandas as pd
import pytest

from live_tracker import investigation_tools_v1 as v1
from live_tracker import investigation_windows as iw
from live_tracker import timebase as tb
from live_tracker.calms21 import KEYPOINTS, _session_to_long
from live_tracker.fake_scores import Segment, build_scores
from live_tracker.rules import RuleSet

TORSO_IDX = [KEYPOINTS.index(k) for k in ("neck", "left_hip", "right_hip")]
N = 1800                                             # 60 s per session
BOUT_STARTS = (12.0, 25.0, 40.0)                     # 2-s attack bouts, 3 per session
SESSIONS = [f"SYNTHETIC_v1_{i}" for i in range(5)]


# ---- synthetic builders ----

def pose_arrays(n, seed):
    """Two mice with jittered torsos (never identical between frames)."""
    rng = np.random.default_rng(seed)
    kp = np.zeros((n, 2, 2, len(KEYPOINTS)))
    for m, cx in enumerate((300.0, 650.0)):
        walk = np.cumsum(rng.normal(0, 1.5, (n, 2)), axis=0) + [cx, 280.0]
        for k in range(len(KEYPOINTS)):
            kp[:, m, 0, k] = walk[:, 0] + 8 * k + rng.normal(0, 0.8, n)
            kp[:, m, 1, k] = walk[:, 1] + rng.normal(0, 0.8, n)
    return kp, np.full((n, 2, len(KEYPOINTS)), 0.9)


def freeze(kp, a, b, mouse=0):
    kp[a:b, mouse][:, :, TORSO_IDX] = kp[a, mouse][:, TORSO_IDX]


def lose(kp, a, b, mouse=0, k=0):
    kp[a:b, mouse, :, TORSO_IDX[k]] = np.nan


def out_of_frame(kp, a, b, mouse=0):
    kp[a:b, mouse, 0, TORSO_IDX[0]] = 5000.0 + np.arange(b - a)     # moving, so never frozen


def make(sessions=SESSIONS, n=N, starts=BOUT_STARTS, edit=None, extra_segments=None):
    scores, poses = [], []
    for i, sid in enumerate(sessions):
        segs = [Segment("attack", s, s + 2.0, 0.85, 0.95) for s in starts]
        segs += (extra_segments or {}).get(sid, [])
        scores.append(build_scores(sid, n / 30, segs, seed=i))
        kp, sc = pose_arrays(n, 100 + i)
        if edit:
            edit(sid, kp, sc)
        poses.append(_session_to_long(sid, {"keypoints": kp, "scores": sc}))
    return v1.InvestigationData.from_synthetic(pd.concat(scores, ignore_index=True),
                                               pd.concat(poses, ignore_index=True))


@pytest.fixture(scope="module")
def data():
    return make()


@pytest.fixture(scope="module")
def query(data):
    r = v1.query_events(data, {"session": list(data.sessions), "behavior": "attack", "min_duration_s": 1.0})
    assert r["status"] == "ok"
    return r


@pytest.fixture(scope="module")
def bouts(query):
    return query["bouts"]


def cmp_req(q, **kw):
    """A compare_windows request citing query_events result q (its id, parameters and bouts, unchanged)."""
    return {"query_run_id": q["query_run_id"], "query": q["query"], "bouts": q["bouts"], "pre_s": 2.0,
            "baseline_s": 5.0, "baseline_gap_s": 1.0, "feature": "dist_center", **kw}


# ======================================================================
# 1. query_events
# ======================================================================

def test_query_events_keeps_start_fired_and_end_separate(data, bouts):
    assert len(bouts) == 15
    for b in bouts:
        assert set(b) == {"event_id", "session", "behavior", "start_s", "fired_s", "end_s", "end_reason",
                          "confidence", "bout_mean_score"}
        assert b["start_s"] in BOUT_STARTS                       # the first frame at or above threshold
        assert b["fired_s"] == pytest.approx(b["start_s"] + 1.0)  # min_duration_s later: never the anchor
        assert b["end_s"] == pytest.approx(b["start_s"] + 2.0) and b["end_reason"] == "below_threshold"
        assert 0.85 <= b["confidence"] <= 0.95


def test_query_events_counts_source_and_determinism(data):
    req = {"session": SESSIONS[:2], "behavior": "attack", "min_duration_s": 1.0, "threshold": 0.5}
    r1, r2 = v1.query_events(data, req), v1.query_events(data, copy.deepcopy(req))
    assert r1 == r2
    assert (r1["n_bouts"], r1["n_sessions"], r1["n_sessions_searched"]) == (6, 2, 2)
    assert r1["data_source"] == "synthetic" and r1["association_only"] is True
    assert r1["query_run_id"].startswith("inv_") and r1["latency_gap_s"] is None
    assert "threshold crossing" in r1["start_meaning"]


def test_short_bouts_are_not_events(data):
    r = v1.query_events(data, {"session": SESSIONS[0], "behavior": "attack", "min_duration_s": 2.5})
    assert r["status"] == "ok" and r["n_bouts"] == 0 and r["n_sessions"] == 0


@pytest.mark.parametrize("req, code", [
    ({"session": "SYNTHETIC_nope", "behavior": "attack", "min_duration_s": 1.0}, "unknown_session"),
    ({"session": SESSIONS[0], "behavior": "immobility", "min_duration_s": 1.0}, "unknown_behavior"),
    ({"session": SESSIONS[0], "behavior": "attack", "min_duration_s": 0.0}, "invalid_request"),
    ({"session": SESSIONS[0], "behavior": "attack", "min_duration_s": 1.0, "threshold": 1.5}, "invalid_request"),
    ({"session": SESSIONS[0], "behavior": "attack", "min_duration_s": 1.0, "rule_id": "R1"}, "invalid_request"),
    ({"session": [SESSIONS[0], SESSIONS[0]], "behavior": "attack", "min_duration_s": 1.0}, "invalid_request"),
    ("not a dict", "invalid_request"),
])
def test_query_events_errors(data, req, code):
    r = v1.query_events(data, req)
    assert r["status"] == "error" and r["code"] == code and r["message"]
    assert code in v1.ERROR_CODES + v1.EXTENSION_ERROR_CODES


def test_missing_scores_are_no_scores():
    sid = "SYNTHETIC_noscore"
    scores = build_scores(sid, 10.0)
    scores["mount"] = np.nan
    kp, sc = pose_arrays(300, 1)
    d = v1.InvestigationData.from_synthetic(scores, _session_to_long(sid, {"keypoints": kp, "scores": sc}))
    r = v1.query_events(d, {"session": sid, "behavior": "mount", "min_duration_s": 1.0})
    assert r["status"] == "error" and r["code"] == "no_scores"


def test_query_events_writes_nothing_and_never_touches_researcher_rules(data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rules = RuleSet()
    rules.add({"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0})
    before = rules.to_json()
    v1.query_events(data, {"session": SESSIONS[0], "behavior": "attack", "min_duration_s": 1.0})
    assert rules.to_json() == before
    assert rules.active() == []                                   # still proposed: nothing approved it
    assert list(tmp_path.iterdir()) == []                         # in-memory journal only


# ======================================================================
# 2. get_window_features
# ======================================================================

def test_window_features_match_the_truncated_pipeline(data):
    sid = SESSIONS[1]
    r = v1.get_window_features(data, {"session": sid, "t0_s": 10.0, "t1_s": 12.0, "cols": ["dist_center",
                                                                                           "rel_speed"]})
    assert r["status"] == "ok" and r["n_frames"] == 60 and r["n_usable_frames"] == 60
    pose = data._get(sid).pose
    _, ok, feats, _ = iw._pipeline(pose.torso[pose.torso["frame"] < 360])     # independent truncated run
    ref = feats["dist_center"][300:360]
    assert r["summary"]["dist_center"]["mean"] == pytest.approx(ref.mean(), rel=1e-12)
    assert r["summary"]["dist_center"]["min"] == pytest.approx(ref.min(), rel=1e-12)
    assert r["summary"]["dist_center"]["n"] == 60
    assert "rows" not in r and set(r["summary"]) == {"dist_center", "rel_speed"}


@pytest.mark.parametrize("t0, t1", [(2.0, 1.0), (1.0, 1.0), (-0.5, 1.0), (59.0, 60.5), (0.02, 0.03)])
def test_window_features_bad_windows(data, t0, t1):
    r = v1.get_window_features(data, {"session": SESSIONS[0], "t0_s": t0, "t1_s": t1, "cols": ["dist_center"]})
    assert r["status"] == "error" and r["code"] == "bad_window"


@pytest.mark.parametrize("col", ["speed", "no_such_feature"])
def test_window_features_unknown_feature(data, col):
    r = v1.get_window_features(data, {"session": SESSIONS[0], "t0_s": 0, "t1_s": 1, "cols": [col]})
    assert r["status"] == "error" and r["code"] == "unknown_feature"


def test_window_features_ignore_frames_after_the_window():
    def poison(sid, kp, sc):
        kp[400:, :, :, :] = np.nan
    clean, poisoned = make(SESSIONS[:1], n=900, starts=()), make(SESSIONS[:1], n=900, starts=(), edit=poison)
    req = {"session": SESSIONS[0], "t0_s": 10.0, "t1_s": 13.0, "cols": list(v1.WINDOW_FEATURES)}
    a, b = v1.get_window_features(clean, req), v1.get_window_features(poisoned, req)
    assert a["summary"] == b["summary"]


# ======================================================================
# 3. check_tracking_quality
# ======================================================================

def quality(d, t0, t1, policy=None):
    req = {"session": d.sessions[0], "t0_s": t0, "t1_s": t1}
    if policy:
        req["quality_policy"] = policy
    return v1.check_tracking_quality(d, req)


def test_clean_window_is_reliable_and_the_policy_is_named(data):
    r = quality(data, 0.0, 5.0)
    assert r["status"] == "ok" and r["verdict"] == "reliable" and r["reason"] is None
    assert r["usable_fraction"] == 1.0 and r["flags_fired"] == [] and r["quality_policy"] == "quality_v1"
    assert r["quality_policy_definition"] == {"name": "quality_v1", "min_usable_fraction": 0.8,
                                              "veto_flags": ["frozen", "out_of_frame"], "veto_max_s": 0.5}


def test_frozen_run_over_half_a_second_vetoes_quality_v1_only():
    d = make(SESSIONS[:1], n=600, starts=(), edit=lambda sid, kp, sc: freeze(kp, 100, 116))   # 16 frames
    v, strict = quality(d, 0.0, 200 / 30), quality(d, 0.0, 200 / 30, "quality_v1_strict90")
    assert v["usable_fraction"] == pytest.approx(184 / 200) and "frozen" in v["flags_fired"]
    assert v["verdict"] == "unreliable" and v["reason"] == "frozen_or_out_of_frame_run"
    assert v["longest_veto_run_s"] == pytest.approx(16 / 30)
    assert strict["verdict"] == "reliable" and strict["quality_policy"] == "quality_v1_strict90"


def test_a_run_of_exactly_half_a_second_is_not_vetoed():
    d = make(SESSIONS[:1], n=600, starts=(), edit=lambda sid, kp, sc: freeze(kp, 100, 115))   # 15 frames
    r = quality(d, 0.0, 200 / 30)
    assert "frozen" in r["flags_fired"] and r["longest_veto_run_s"] == pytest.approx(0.5)
    assert r["verdict"] == "reliable"


def test_out_of_frame_run_vetoes():
    d = make(SESSIONS[:1], n=600, starts=(), edit=lambda sid, kp, sc: out_of_frame(kp, 50, 70))
    r = quality(d, 0.0, 200 / 30)
    assert "out_of_frame" in r["flags_fired"] and r["reason"] == "frozen_or_out_of_frame_run"


def test_usable_fraction_boundary_is_exact():
    d = make(SESSIONS[:1], n=600, starts=(), edit=lambda sid, kp, sc: lose(kp, 0, 20))
    at = quality(d, 0.0, 100 / 30)                                    # 80 of 100 usable: exactly 0.8
    assert at["usable_fraction"] == 0.8 and at["verdict"] == "reliable"
    below = quality(d, 0.0, 99 / 30)                                  # 79 of 99
    assert below["verdict"] == "unreliable" and below["reason"] == "usable_fraction_below_min"
    assert "missing" in below["flags_fired"]


def test_a_freeze_continuing_after_the_window_is_judged_causally():
    d = make(SESSIONS[:1], n=600, starts=(), edit=lambda sid, kp, sc: freeze(kp, 190, 240))
    full_flags = d._get(d.sessions[0]).pose.flags
    assert full_flags[(full_flags["frame"] >= 190) & (full_flags["frame"] < 200)]["frozen"].any()  # look-ahead
    r = quality(d, 0.0, 200 / 30)                                     # only 10 identical frames before t1
    assert "frozen" not in r["flags_fired"] and r["verdict"] == "reliable"


def test_unknown_quality_policy_is_refused(data):
    assert quality(data, 0, 1, "quality_v2")["code"] == "invalid_request"


# ======================================================================
# 4. compare_windows
# ======================================================================

def test_compare_windows_counts_and_session_paired_summary(data, query, bouts):
    r = v1.compare_windows(data, cmp_req(query))
    assert r["status"] == "ok" and r["association_only"] is True
    assert (r["n_input"], r["n_used"], r["n_excluded"]) == (15, 15, 0)
    assert (r["n_sessions"], r["n_bouts"], r["n_windows"]) == (5, 15, 15)
    assert "ci95" not in r and "effect_size_dz" not in r and r["wording"] == v1.WORDING
    # composed from tool 2 exactly, then per-session means, median across sessions
    per = {}
    for b in bouts:
        def mean(t0, t1):
            return v1.get_window_features(data, {"session": b["session"], "t0_s": t0, "t1_s": t1,
                                                 "cols": ["dist_center"]})["summary"]["dist_center"]["mean"]
        s = b["start_s"]
        per.setdefault(b["session"], []).append(mean(s - 2, s) - mean(s - 8, s - 3))
    expected = {sid: float(np.mean(d)) for sid, d in per.items()}
    assert {p["session"]: p["paired_diff"] for p in r["per_session"]} == pytest.approx(expected)
    assert r["mean_diff"] == pytest.approx(float(np.median(list(expected.values()))))
    assert all(p["n_bouts"] == 3 for p in r["per_session"])


def test_windows_are_anchored_on_start_s(data, query):
    r = v1.compare_windows(data, cmp_req(query))
    assert r["windows"]["anchor"] == "start_s (never fired_s)"
    # a pre window ending at fired_s would contain the bout itself; anchored on start_s it ends at the onset
    b = query["bouts"][0]
    pre = tb.window_frames(b["start_s"] - 2.0, b["start_s"])
    assert pre[1] == tb.boundary_frame(b["start_s"]) < tb.boundary_frame(b["fired_s"])


def test_overlap_edge_and_quality_exclusions_are_reported():
    def edit(sid, kp, sc):
        if sid == SESSIONS[0]:
            lose(kp, 25 * 30 - 40, 25 * 30 - 10)          # pre window of the 25-s bout: 30 of 60 frames lost
        if sid == SESSIONS[1]:
            lose(kp, 1000, 1040)                         # baseline of the 40-s bout: 40 of 150 lost
    extra = {SESSIONS[2]: [Segment("attack", 21.5, 23.5, 0.85, 0.95)],   # ends 1.5 s before the 25-s bout
             SESSIONS[3]: [Segment("attack", 4.0, 6.0, 0.85, 0.95)]}     # baseline would start before 0 s
    d = make(edit=edit, extra_segments=extra)
    q = v1.query_events(d, {"session": list(d.sessions), "behavior": "attack", "min_duration_s": 1.0})
    r = v1.compare_windows(d, cmp_req(q))
    why = {(e["session"], e["start_s"]): e["why"] for e in r["excluded"]}
    assert why == {(SESSIONS[0], 25.0): "pre_window_unreliable", (SESSIONS[1], 40.0): "baseline_window_unreliable",
                   (SESSIONS[2], 25.0): "pre_window_overlaps_bout", (SESSIONS[3], 4.0): "window_before_session_start"}
    assert r["n_input"] == 17 and r["n_used"] == 13 and r["n_excluded"] == 4
    assert set(why.values()) <= set(v1.EXCLUSION_REASONS)


def query2(data):
    return v1.query_events(data, {"session": SESSIONS[:2], "behavior": "attack", "min_duration_s": 1.0})


def test_too_few_bouts_and_too_few_sessions_are_insufficient(data):
    few = v1.compare_windows(data, cmp_req(query2(data)))
    assert few["status"] == "insufficient_evidence" and few["reason"] == "too_few_bouts"
    assert (few["n_available"], few["n_required"]) == (6, 10)
    d3 = make(SESSIONS[:3], starts=(12.0, 25.0, 40.0, 52.0))       # 12 bouts, but only 3 sessions
    q3 = v1.query_events(d3, {"session": list(d3.sessions), "behavior": "attack", "min_duration_s": 1.0})
    r = v1.compare_windows(d3, cmp_req(q3))
    assert r["status"] == "insufficient_evidence" and r["reason"] == "too_few_usable_windows"
    assert r["failed_gate"] == "min_n_sessions" and (r["n_available"], r["n_required"]) == (3, 4)
    assert r["n_windows"] == 12 and r["n_sessions"] == 3
    assert "mean_diff" not in r and "pre_mean" not in r                # no effect value behind a failed gate


def test_lowering_the_gate_needs_a_documented_reason(data):
    r = v1.compare_windows(data, cmp_req(query2(data), min_n_windows=5))
    assert r["status"] == "error" and r["code"] == "invalid_request"
    r = v1.compare_windows(data, cmp_req(query2(data), min_n_windows=5, min_n_sessions=2,
                                         gate_override_reason="synthetic unit test"))
    assert r["status"] == "ok" and r["evidence_gate"]["overridden"] is True


@pytest.mark.parametrize("change, code", [
    ({"pre_s": 0.01}, "bad_window"), ({"baseline_s": 0.0}, "bad_window"), ({"feature": "speed"}, "unknown_feature"),
    ({"bouts": [{"session": SESSIONS[0], "start_s": 12.0}]}, "unverified_bouts"),
    ({"bouts": []}, "invalid_request"),
])
def test_compare_windows_errors(data, query, change, code):
    r = v1.compare_windows(data, {**cmp_req(query), **change})
    assert r["status"] == "error" and r["code"] == code


def test_tools_are_deterministic(data, query):
    assert v1.compare_windows(data, cmp_req(query)) == v1.compare_windows(data, cmp_req(copy.deepcopy(query)))


# ======================================================================
# budget and approval boundary
# ======================================================================

def test_budget_of_twelve_calls_records_every_attempt(data):
    run = v1.InvestigationRunner(data, "inv-test-1")
    for _ in range(12):
        run.call("check_tracking_quality", {"session": SESSIONS[0], "t0_s": 0, "t1_s": 1})
    r = run.call("check_tracking_quality", {"session": SESSIONS[0], "t0_s": 0, "t1_s": 1})
    assert r["status"] == "error" and r["code"] == "budget_exhausted"
    rec = run.record()
    assert rec["calls_used"] == 12 and len(rec["attempts"]) == 13 and rec["attempts"][-1]["counted"] is False
    assert all(a["investigation_id"] == "inv-test-1" for a in rec["attempts"])
    assert rec["attempts"][0]["result"]["verdict"] == "reliable" and rec["attempts"][-1]["result"]["code"] == \
        "budget_exhausted"
    with pytest.raises(ValueError):
        v1.InvestigationRunner(data, "x", max_tool_calls=13)


def test_follow_ups_use_only_the_preselected_grid(data, query):
    run = v1.InvestigationRunner(data, "inv-test-2")
    assert run.call("compare_windows", cmp_req(query, pre_s=3.0))["code"] == "invalid_request"   # not in grid
    assert run.call("compare_windows", cmp_req(query))["status"] == "ok"                       # first: 2 s
    assert run.call("compare_windows", cmp_req(query, feature="rel_speed", pre_s=1.0))["code"] == "invalid_request"
    assert run.call("compare_windows", cmp_req(query, pre_s=2.0))["code"] == "invalid_request"  # repeat
    assert run.call("compare_windows", cmp_req(query, pre_s=1.0))["status"] == "ok"             # follow-up 1
    assert run.call("compare_windows", cmp_req(query, pre_s=5.0))["status"] in ("ok", "insufficient_evidence")
    assert run.call("compare_windows", cmp_req(query, pre_s=1.0))["code"] == "invalid_request"  # a 3rd follow-up
    statuses = [(a["tool"], a["status"]) for a in run.record()["attempts"]]
    assert len(statuses) == 7 and statuses.count(("compare_windows", "error")) == 4           # failures kept


def test_propose_rule_is_deferred_and_cannot_approve(data):
    run = v1.InvestigationRunner(data, "inv-test-3")
    r = run.call("propose_rule", {"rule_id": "R9", "behavior": "attack", "min_duration_s": 1.0,
                                  "status": "approved"})
    assert r["status"] == "error" and "deferred" in r["message"]
    src = open(v1.__file__).read()
    assert ".approve(" not in src and "RuleSet" not in src.split('"""', 2)[2]


# ======================================================================
# source labelling
# ======================================================================

def test_synthetic_sessions_cannot_be_labelled_real(data):
    s = data._sessions
    with pytest.raises(ValueError, match="synthetic"):
        v1.InvestigationData("classifier", "classifier_probabilities", s, [])
    with pytest.raises(ValueError):
        v1.InvestigationData("model_b", "x", s, [])


def test_misaligned_scores_and_pose_are_refused():
    sid = "SYNTHETIC_mis"
    scores = build_scores(sid, 10.0)
    kp, sc = pose_arrays(299, 1)                                       # one frame short
    with pytest.raises(ValueError, match="aligned"):
        v1.InvestigationData.from_synthetic(scores, _session_to_long(sid, {"keypoints": kp, "scores": sc}))


def test_verified_bindings_reuse_the_stage9_sources(tmp_path):
    """The real constructors on the Stage 9 synthetic cache and an issued synthetic
    score drop: verification is reused, and SYNTHETIC data stays labelled synthetic."""
    from test_investigation_windows import ATTACKS, VAL_A, build
    from test_prediction_windows import score_table
    from test_score_drops import TEST_MEMBER, VAL_MEMBER, archive, csv_bytes
    import live_tracker.score_drops as sd

    tools, policy, cache, manifest = build(tmp_path)
    ann = tools.sources["validation_annotation"]
    labels = v1.InvestigationData.from_annotations(ann)
    assert labels.data_source == "synthetic" and labels.derived_from == "synthetic_scores"
    r = v1.query_events(labels, {"session": VAL_A, "behavior": "attack", "min_duration_s": 0.3})
    assert [(tb.boundary_frame(b["start_s"]), tb.boundary_frame(b["end_s"])) for b in r["bouts"]] == ATTACKS[VAL_A]
    drop = sd.load_validation_scores(
        archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table()), TEST_MEMBER: csv_bytes(score_table())}),
        VAL_MEMBER, manifest_path=manifest)
    clf = v1.InvestigationData.from_classifier(drop, ann, policy)
    assert clf.data_source == "synthetic" and clf.provenance[0]["kind"] == "classifier_scores"
    assert v1.query_events(clf, {"session": VAL_A, "behavior": "attack", "min_duration_s": 0.5})["status"] == "ok"
    forged = sd.ScoreDrop(drop.provenance, drop.scores())                # not issued by the loader
    with pytest.raises(ValueError, match="verified"):
        v1.InvestigationData.from_classifier(forged, ann, policy)
    assert not os.path.exists("events.journal.jsonl")


# ======================================================================
# compare_windows converts each ABSOLUTE endpoint (Ava's rule), never
# start_frame minus independently rounded duration counts
# ======================================================================

def consumed(data, query, monkeypatch, **change):
    """compare_windows on the cited query; returns (result, {session-start: [(first, stop), ...]}) where the
    list holds the frame ranges actually given to the quality / feature steps (pre first, then baseline)."""
    calls = []
    real_assess, real_values = v1._assess, v1._values
    monkeypatch.setattr(v1, "_assess", lambda pose, a, z, pol: calls.append((pose.session, a, z)) or
                        real_assess(pose, a, z, pol))
    monkeypatch.setattr(v1, "_values", lambda pose, a, z, f: calls.append((pose.session, a, z)) or
                        real_values(pose, a, z, f))
    r = v1.compare_windows(data, {**cmp_req(query), **change})
    monkeypatch.undo()
    out = {}
    for sid, a, z in calls:
        out.setdefault(sid, [])
        if (a, z) not in out[sid]:
            out[sid].append((a, z))
    return r, out


def test_codex_case_baseline_of_005_s_is_frames_299_to_300(data, query, monkeypatch):
    r, out = consumed(data, query, monkeypatch, pre_s=2.0, baseline_s=0.05, baseline_gap_s=0.0,
                      min_n_windows=1, min_n_sessions=1, gate_override_reason="synthetic regression")
    assert r["status"] == "ok"
    assert (300, 360) in out[SESSIONS[0]]                       # pre [10.0, 12.0) s
    assert (299, 300) in out[SESSIONS[0]]                       # baseline [9.95, 10.0) s: frame(9.95) = 299
    assert (298, 300) not in out[SESSIONS[0]]                   # NOT 300 - frame(0.05) = 300 - 2


def test_fractional_gap_and_nonaligned_pre_boundaries(data, query, monkeypatch):
    # gap 0.05 s: baseline end = frame(9.95) = 299 (subtracting frame(0.05) = 2 would give 298)
    _, out = consumed(data, query, monkeypatch, pre_s=2.0, baseline_s=1.0, baseline_gap_s=0.05)
    assert (269, 299) in out[SESSIONS[0]] and (300, 360) in out[SESSIONS[0]]
    # pre 1.95 s: pre start = frame(10.05) = floor(302.0) = 302 (subtracting frame(1.95) = 59 would give 301)
    _, out = consumed(data, query, monkeypatch, pre_s=1.95, baseline_s=1.0, baseline_gap_s=0.0)
    assert (302, 360) in out[SESSIONS[0]] and (301, 360) not in out[SESSIONS[0]]
    assert (272, 302) in out[SESSIONS[0]]                       # baseline [9.05, 10.05): adjacent to the pre window


def test_every_consumed_window_equals_the_endpoint_conversion(data, query, monkeypatch):
    pre, base, gap = 1.95, 0.75, 0.3
    _, out = consumed(data, query, monkeypatch, pre_s=pre, baseline_s=base, baseline_gap_s=gap)
    for b in query["bouts"]:
        s = b["start_s"]
        want_pre = (tb.boundary_frame(s - pre), tb.boundary_frame(s))
        want_base = (tb.boundary_frame(s - pre - gap - base), tb.boundary_frame(s - pre - gap))
        assert want_pre in out[b["session"]] and want_base in out[b["session"]]


def test_adjacent_pre_and_baseline_windows_share_their_boundary_frame(data, query, monkeypatch):
    _, out = consumed(data, query, monkeypatch, pre_s=1.0, baseline_s=1.25, baseline_gap_s=0.0)
    for b in query["bouts"]:
        s = b["start_s"]
        p0 = tb.boundary_frame(s - 1.0)
        assert (p0, tb.boundary_frame(s)) in out[b["session"]]
        assert (tb.boundary_frame(s - 1.0 - 1.25), p0) in out[b["session"]]   # no frame missing or shared


@pytest.mark.parametrize("change", [{"baseline_s": 0.01}, {"pre_s": 0.01}, {"baseline_s": 1e-12},
                                    {"pre_s": 1e-12}])
def test_windows_that_convert_to_no_frame_are_bad_window(data, query, change):
    r = v1.compare_windows(data, {**cmp_req(query), **change})
    assert r["status"] == "error" and r["code"] == "bad_window" and "empty frame range" in r["message"]
