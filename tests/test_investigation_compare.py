"""Tests for the experimental compare_windows tool (new file).

Synthetic parts reuse the Stage 4 SYNTHETIC fixtures (CalMS21 layout,
published through the real Stage 1 extraction code); nothing synthetic is
CalMS21 data. The real-data tests read the verified validation cache under
data/raw and are skipped when it is not present.
"""

import json
import statistics
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import live_tracker.investigation_events as ev
import live_tracker.investigation_windows as iw
from live_tracker.investigation_contracts import (CONTRACT_VERSION, PROVISIONAL_SUFFICIENCY, ContractError,
                                                  EvidencePolicy, ForbiddenEvidence, SufficiencyCriteria,
                                                  baseline_window_id, call_tool, episode_id, make_result,
                                                  pre_window_id, validate_request, validate_result)
from live_tracker.investigation_data import INDEX_NAME
from live_tracker.investigation_events import CalMS21InvestigationTools
from test_investigation_windows import (MANIFEST, REAL_CACHE, REAL_MANIFEST, TEST, TRAIN, VAL_A, VAL_B,
                                        VAL_C, arrays, base, build, ep, freq, record, windows)

SRC = "validation_annotation"
LOOSE = SufficiencyCriteria(min_sessions=1, min_windows_per_group=1, min_usable_fraction=0.9)
NONE = {"method": "none", "level": None, "resamples": None, "seed": None}
# one attack per session (spaced so every pre-event window is clean), three paired sessions
SPACED = {VAL_A: [(100, 105), (200, 205), (300, 305)], VAL_B: [(200, 205)], VAL_C: [(300, 305)]}


def boot(level=0.7, resamples=2000, seed=7):
    return {"method": "bootstrap_sessions", "level": level, "resamples": resamples, "seed": seed}


def pre(sid, onset, length=20, gap=0, behavior="attack", source=SRC):
    return pre_window_id(episode_id(source, behavior, sid, onset), length, gap)


def tile(sid, start, length=20, behavior="attack", source=SRC):
    return baseline_window_id(source, behavior, sid, start, length)


def creq(pre_ids, base_ids, crit=LOOSE, unc=NONE, feature="dist_rate"):
    return {"contract_version": CONTRACT_VERSION, "feature": feature, "statistic": "median_difference",
            "aggregation": "session_level", "pre_window_ids": list(pre_ids),
            "baseline_window_ids": list(base_ids), "sufficiency": crit.to_json(), "uncertainty": dict(unc)}


def compare(env, pre_ids, base_ids, **kw):
    req = creq(pre_ids, base_ids, **kw)
    res = call_tool(env[0], "compare_windows", req, env[1])     # validated against the contract
    if res["status"] != "error":
        validate_result("compare_windows", req, res, env[1])
    return res


def rows(res):
    return {r["session"]: r for r in res["measurements"]["per_session"]}


def method(res, name):
    return next(m for m in res["methodology"] if m["name"] == name)


@pytest.fixture
def spaced(tmp_path):
    return build(tmp_path, attacks=SPACED)


SPACED_PRE = [pre(VAL_A, 100), pre(VAL_A, 200), pre(VAL_A, 300), pre(VAL_B, 200), pre(VAL_C, 300)]
SPACED_BASE = [tile(s, a) for s in (VAL_A, VAL_B, VAL_C) for a in (20, 40)]


# ---- statistic ----

def test_values_are_stage4_measurements(spaced):
    """Every window value and count is exactly get_window_features' median and
    n_usable for the same window: no alternative feature computation."""
    res = compare(spaced, SPACED_PRE, SPACED_BASE)
    stage4 = {}
    for sid, onsets in ((VAL_A, (100, 200, 300)), (VAL_B, (200,)), (VAL_C, (300,))):
        f = call_tool(spaced[0], "get_window_features",
                      freq([ep(sid, o) for o in onsets], baseline=base(clearance=0)), spaced[1])
        stage4.update({wid: w for wid, w in windows(f).items() if w["status"] == "included"})
    used = [w for r in res["measurements"]["per_session"] for w in r["pre_windows"] + r["baseline_windows"]]
    assert len(used) == len(SPACED_PRE) + len(SPACED_BASE)
    for w in used:
        s4 = stage4[w["window_id"]]["features"]["dist_rate"]
        assert (w["n_frames"], w["n_usable"], w["value"]) == (s4["n_frames"], s4["n_usable"], s4["median"])


def test_session_level_paired_differences(spaced):
    res = compare(spaced, SPACED_PRE, SPACED_BASE)
    m = res["measurements"]
    diffs = []
    for sid, r in rows(res).items():
        p = statistics.median(w["value"] for w in r["pre_windows"])
        b = statistics.median(w["value"] for w in r["baseline_windows"])
        assert (r["pre_summary"], r["baseline_summary"], r["difference"]) == (p, b, p - b)
        diffs.append(p - b)
    assert res["status"] == "ok" and m["n_sessions_paired"] == 3
    assert m["effect"]["value"] == statistics.median(diffs)
    assert (m["n_windows_pre"], m["n_windows_baseline"]) == (5, 6)
    # the synthetic approach (frames 70..100) makes VAL_A's [80, 100) window strongly negative
    a = {w["window_id"]: w["value"] for w in rows(res)[VAL_A]["pre_windows"]}
    assert a[pre(VAL_A, 100)] < -200


def test_many_windows_in_one_session_do_not_outweigh_other_sessions(spaced, monkeypatch):
    """Adversarial: one session contributes three extreme windows. Pooling
    windows would give -100; the session-level effect is the median of the
    three session differences."""
    real = iw.measure_window
    planted = {(VAL_A, 80): -100.0, (VAL_A, 180): -100.0, (VAL_A, 280): -100.0,
               (VAL_B, 180): 10.0, (VAL_C, 280): 5.0}

    def fake(sess, start, end, features):
        out = real(sess, start, end, features)
        for f in features:
            out[f] = dict(out[f], n_usable=out[f]["n_frames"], median=planted.get((sess.session, start), 0.0))
        return out
    monkeypatch.setattr(iw, "measure_window", fake)
    m = compare(spaced, SPACED_PRE, SPACED_BASE)["measurements"]
    assert [r["difference"] for r in m["per_session"]] == [-100.0, 10.0, 5.0]
    assert m["effect"]["value"] == 5.0
    pooled = statistics.median([-100.0] * 3 + [10.0, 5.0])
    assert pooled == -100.0 != m["effect"]["value"]


def test_descriptive_effect_has_no_interval_and_stable_evidence(spaced):
    res = compare(spaced, SPACED_PRE, SPACED_BASE)
    eff = res["measurements"]["effect"]
    assert res["status"] == "ok" and eff["value"] is not None
    assert (eff["ci_low"], eff["ci_high"], eff["ci_level"]) == (None, None, None)
    assert compare(spaced, SPACED_PRE, SPACED_BASE)["evidence_id"] == res["evidence_id"]   # deterministic
    m = method(res, "uncertainty_method")
    assert m["value"] == "none" and m["status"] == "provisional"
    assert "no session-level uncertainty method has been approved" in m["note"]
    assert "descriptive, over n = 3 paired sessions" in method(res, "effect")["note"]


# ---- uncertainty: no population interval until a method is approved ----

FOUR = [f"task1/train/SYNTHETIC_four_{i}" for i in range(4)]


@pytest.fixture
def four_sessions(monkeypatch):
    """Four paired sessions whose differences are exactly [0, 1, 2, 3] (Codex
    Stage 5 review counterexample): in-memory records, planted window medians."""
    policy = EvidencePolicy({"train": [], "validation": FOUR, "test": ["task1/test/SYNTHETIC_four_test"]},
                            allow_synthetic=True)
    records = {}
    for sid in FOUR:
        kp, sc = arrays(VAL_A)
        records[sid] = {"annotator-id_0": {sid: record(sid, kp, sc, attacks={sid: [(100, 105)]})}}

    class Source:
        source, read = SRC, []

        def __init__(self):
            self.policy = policy

        def snapshot(self, sessions):
            self.read.append(sessions)
            return SimpleNamespace(sessions={s: records[s] for s in sessions}, synthetic=True,
                                   finalize=lambda: None, data=[{"kind": "synthetic_fixture", "path": None,
                                                                 "sha256": None, "description": "in-memory"}])
    real = iw.measure_window

    def planted(sess, start, end, features):
        out = real(sess, start, end, features)
        value = float(FOUR.index(sess.session)) if start == 80 else 0.0
        return {f: dict(out[f], n_usable=out[f]["n_frames"], median=value) for f in features}
    monkeypatch.setattr(iw, "measure_window", planted)
    src = Source()
    return CalMS21InvestigationTools(policy, [src]), policy, src


FOUR_PRE = [pre(s, 100) for s in FOUR]
FOUR_BASE = [tile(s, 20) for s in FOUR]


def test_four_session_counterexample_reports_no_interval(four_sessions):
    res = compare(four_sessions, FOUR_PRE, FOUR_BASE)
    m = res["measurements"]
    assert res["status"] == "ok" and [r["difference"] for r in m["per_session"]] == [0.0, 1.0, 2.0, 3.0]
    assert m["effect"] == {"name": "median_of_session_differences", "value": 1.5, "ci_low": None,
                           "ci_high": None, "ci_level": None}


def test_four_session_counterexample_bootstrap_is_refused(four_sessions):
    """The removed method returned [0.5, 2.5] at a nominal 0.875 for these
    differences; it must now be refused, not reported or converted."""
    tools, policy, src = four_sessions
    res = compare(four_sessions, FOUR_PRE, FOUR_BASE, unc=boot(0.875, 100_000, 7))
    assert res["status"] == "error" and res["measurements"] is None
    assert res["errors"][0]["code"] == "not_implemented"
    assert "bootstrap_sessions" in res["errors"][0]["message"] and "'none'" in res["errors"][0]["message"]
    assert src.read == []                                       # refused before anything is read


def test_why_the_percentile_bootstrap_was_removed():
    """Pins the defect: for differences [0, 1, 2, 3] the exact percentile
    bootstrap interval at 0.875 is [0.5, 2.5], but under iid sampling that
    interval covers the population median only ~0.75 of the time; the
    1 - 2^(1-n) = 0.875 figure is the coverage of the full range [min, max]."""
    d = np.arange(4.0)
    medians = np.median(d[np.array(list(product(range(4), repeat=4)))], axis=1)
    assert list(np.quantile(medians, [0.0625, 0.9375])) == [0.5, 2.5]
    x = np.sort(np.random.default_rng(0).uniform(size=(200_000, 4)), axis=1)
    full_range = np.mean((x[:, 0] <= 0.5) & (0.5 <= x[:, 3]))
    percentile = np.mean(((x[:, 0] + x[:, 1]) / 2 <= 0.5) & (0.5 <= (x[:, 2] + x[:, 3]) / 2))
    assert abs(full_range - 0.875) < 0.005 and abs(percentile - 0.75) < 0.005


@pytest.mark.parametrize("level", [0.5, 0.875, 0.95, 0.999])
@pytest.mark.parametrize("crit", [LOOSE, PROVISIONAL_SUFFICIENCY])
def test_bootstrap_requests_are_refused_before_reading(env_default, monkeypatch, level, crit):
    """The contract still accepts the request (an approved method can be added
    later); this implementation refuses it whatever the data would show."""
    req = creq([pre(VAL_A, 100)], [tile(VAL_A, 20)], crit=crit, unc=boot(level))
    validate_request("compare_windows", req, env_default[1])

    def no_read(*a, **k):
        raise AssertionError("the cache was read")
    monkeypatch.setattr(ev, "verified_sessions", no_read)
    res = call_tool(env_default[0], "compare_windows", req, env_default[1])
    assert res["status"] == "error" and res["errors"][0]["code"] == "not_implemented"


def test_contract_refuses_an_interval_without_an_uncertainty_method(spaced):
    req = creq(SPACED_PRE, SPACED_BASE)
    res = compare(spaced, SPACED_PRE, SPACED_BASE)
    m = json.loads(json.dumps(res["measurements"]))
    m["effect"].update(ci_low=m["effect"]["value"], ci_high=m["effect"]["value"], ci_level=0.9)
    forged = make_result("compare_windows", req, "ok", m, methodology=res["methodology"],
                         limitations=res["limitations"], provenance=res["provenance"])
    with pytest.raises(ContractError):
        validate_result("compare_windows", req, forged, spaced[1])


# ---- sufficiency, incomplete and missing data ----

def test_unmet_criteria_report_no_effect(spaced):
    res = compare(spaced, SPACED_PRE, SPACED_BASE, crit=PROVISIONAL_SUFFICIENCY)
    m = res["measurements"]
    assert res["status"] == "insufficient_evidence"
    assert m["sufficiency"] == {"criteria": PROVISIONAL_SUFFICIENCY.to_json(), "met": False,
                                "reasons_not_met": ["min_sessions", "min_windows_per_group_pre",
                                                    "min_windows_per_group_baseline"]}
    assert m["effect"] == {"name": "median_of_session_differences", "value": None, "ci_low": None,
                           "ci_high": None, "ci_level": None}
    assert all(r["difference"] is not None for r in m["per_session"])   # measured facts stay visible


def test_missing_baselines_leave_sessions_unpaired(spaced):
    """Every baseline overlaps an attack: no session is paired, no effect."""
    res = compare(spaced, SPACED_PRE, [tile(VAL_A, 100), tile(VAL_B, 200)])
    m = res["measurements"]
    assert res["status"] == "insufficient_evidence" and m["n_sessions_paired"] == 0
    assert m["sufficiency"]["reasons_not_met"] == ["min_sessions", "min_windows_per_group_baseline"]
    assert all(r["baseline_summary"] is None and r["difference"] is None for r in m["per_session"])
    assert {x["window_id"]: x["reason"] for x in m["excluded_windows"]} == \
        {tile(VAL_A, 100): "overlaps_event", tile(VAL_B, 200): "overlaps_event"}
    assert m["effect"]["value"] is None


def test_zero_usable_windows_is_no_data(tmp_path):
    def blind(sid, kp, sc):
        sc[:] = 0.01                                    # nothing passes the score flag
    env = build(tmp_path, blind, attacks=SPACED)
    res = compare(env, SPACED_PRE, SPACED_BASE)
    m = res["measurements"]
    assert res["status"] == "no_data" and m["per_session"] == [] and m["n_sessions_paired"] == 0
    # without valid centre samples the feature history reaches frame 0, so windows after
    # an earlier attack also fail overlaps_event (first in contract order)
    assert {x["reason"] for x in m["excluded_windows"]} == {"missing_data", "overlaps_event"}
    assert all((x["n_frames"], x["n_usable"]) == (20, 0) for x in m["excluded_windows"])
    assert "baseline missing_data 6" in method(res, "exclusion_counts")["value"]
    assert "pre missing_data 5" in method(res, "exclusion_counts")["value"]
    assert m["effect"]["value"] is None


def test_invalid_confidence_scores_never_become_evidence(tmp_path):
    """Stage 4 finding 1 carried through: null scores in VAL_A's pre-event window."""
    def edit(sid, rec):
        if sid == VAL_A:
            for t in range(80, 100):
                rec["scores"][t] = [[None] * 7 for _ in range(2)]
    env = build(tmp_path, attacks=SPACED, edit=edit)
    res = compare(env, SPACED_PRE, SPACED_BASE)
    x = {x["window_id"]: x for x in res["measurements"]["excluded_windows"]}
    assert x[pre(VAL_A, 100)]["reason"] == "missing_data"
    assert pre(VAL_A, 100) not in {w["window_id"] for w in rows(res)[VAL_A]["pre_windows"]}


def test_min_usable_fraction_comes_from_the_request(tmp_path):
    def degrade(sid, kp, sc):
        if sid == VAL_B:
            sc[185:188, 0, 3] = 0.01                    # [180, 200): 16 of 20 frames usable
    env = build(tmp_path, degrade, attacks=SPACED)
    strict = compare(env, SPACED_PRE, SPACED_BASE)
    x = {x["window_id"]: x for x in strict["measurements"]["excluded_windows"]}[pre(VAL_B, 200)]
    assert (x["reason"], x["n_frames"], x["n_usable"]) == ("insufficient_usable_frames", 20, 16)
    lenient = compare(env, SPACED_PRE, SPACED_BASE, crit=SufficiencyCriteria(1, 1, 0.8))
    assert rows(lenient)[VAL_B]["pre_windows"][0]["n_usable"] == 16
    assert method(lenient, "min_usable_fraction")["value"] == 0.8


def test_stage4_exclusion_rules_are_applied(env_default):
    """Pre-event: earlier attack in the history, start before the session.
    Baseline: inside an attack, attack in its feature history (clearance 0),
    overlapping a requested pre-event window. One reason each, in contract
    order; every failed rule is counted."""
    pre_ids = [pre(VAL_A, 100), pre(VAL_A, 130)]
    base_ids = [tile(VAL_A, a) for a in (100, 160, 80, 140, 200)]
    res = compare(env_default, pre_ids, base_ids)
    x = {x["window_id"]: x for x in res["measurements"]["excluded_windows"]}
    assert {k: v["reason"] for k, v in x.items()} == {
        pre(VAL_A, 130): "overlaps_event", tile(VAL_A, 100): "overlaps_event",
        tile(VAL_A, 160): "near_event",                 # attack 130..140 in its feature history
        tile(VAL_A, 80): "overlaps_pre_window",
        tile(VAL_A, 140): "overlaps_pre_window"}        # also near_event; reported in contract order
    assert [w["window_id"] for w in rows(res)[VAL_A]["baseline_windows"]] == [tile(VAL_A, 200)]
    counts = method(res, "exclusion_counts")["value"]
    assert "baseline near_event 3" in counts and "baseline overlaps_pre_window 3" in counts   # 80, 100, 140
    assert method(res, "baseline_clearance_frames")["value"] == 0


def test_window_starting_before_the_session_is_excluded_without_counts(env_default):
    early = pre(VAL_A, 100, gap=90)                     # [-10, 10): clipped
    res = compare(env_default, [early, pre(VAL_A, 300, gap=90)], [tile(VAL_A, 360)])
    x = res["measurements"]["excluded_windows"]
    assert x == [{"window_id": early, "reason": "starts_before_session", "n_frames": None, "n_usable": None}]


@pytest.fixture
def env_default(tmp_path):
    return build(tmp_path)


# ---- refusals ----

@pytest.mark.parametrize("pre_ids, base_ids, code", [
    ([pre(VAL_A, 100), pre(VAL_A, 300, length=30)], [tile(VAL_A, 20)], "invalid_request"),     # mixed lengths
    ([pre(VAL_A, 100), pre(VAL_A, 300, gap=10)], [tile(VAL_A, 20)], "invalid_request"),        # mixed gaps
    ([pre(VAL_A, 100)], [tile(VAL_A, 30, length=30)], "invalid_request"),                      # other duration
    ([pre(VAL_A, 101)], [tile(VAL_A, 20)], "unknown_id"),                                      # not an onset
    ([pre(VAL_A, 100)], [tile(VAL_A, 30)], "unknown_id"),                                      # off the tile grid
    ([pre(VAL_A, 100)], [tile(VAL_A, 400)], "unknown_id"),                                     # past the end
])
def test_incompatible_or_unknown_windows_are_refused(env_default, pre_ids, base_ids, code):
    res = compare(env_default, pre_ids, base_ids)
    assert res["status"] == "error" and res["errors"][0]["code"] == code


def test_other_sources_are_not_substituted(env_default):
    res = compare(env_default, [pre(VAL_A, 100, source="classifier_prediction")],
                  [tile(VAL_A, 20, source="classifier_prediction")])
    assert res["errors"][0]["code"] == "source_unavailable"


def test_mixed_behaviors_are_refused_by_the_contract(env_default):
    with pytest.raises(ContractError):
        compare(env_default, [pre(VAL_A, 100)], [tile(VAL_A, 20, behavior="mount")])


@pytest.mark.parametrize("sid", [TEST, TRAIN])
def test_test_and_train_windows_are_refused_before_reading(env_default, monkeypatch, sid):
    def no_read(*a, **k):
        raise AssertionError("the cache was read")
    monkeypatch.setattr(ev, "verified_sessions", no_read)
    with pytest.raises((ContractError, ForbiddenEvidence)):
        compare(env_default, [pre(sid, 100)], [tile(sid, 20)])


def test_session_missing_from_the_cache_is_unavailable(tmp_path):
    env = build(tmp_path, sessions=(VAL_A,))
    res = compare(env, [pre(VAL_A, 100)], [tile(VAL_B, 20)])
    assert res["errors"][0]["code"] == "source_unavailable"


def test_manifest_changed_during_the_comparison_is_refused(env_default, monkeypatch):
    real = iw.load_session

    def load_then_rewrite(sid, data):
        env_default[3].write_text(json.dumps(MANIFEST, indent=1))
        return real(sid, data)
    monkeypatch.setattr(iw, "load_session", load_then_rewrite)
    res = compare(env_default, [pre(VAL_A, 100)], [tile(VAL_A, 20)])
    assert res["errors"][0]["code"] == "source_unavailable" and "manifest changed" in res["errors"][0]["message"]


def test_index_changed_during_the_comparison_is_refused(env_default, monkeypatch):
    real = iw.load_session

    def load_then_touch(sid, data):
        index = json.loads((env_default[2] / INDEX_NAME).read_text())
        index["extraction_run"]["parse_seconds"] = 99.0
        (env_default[2] / INDEX_NAME).write_text(json.dumps(index))
        return real(sid, data)
    monkeypatch.setattr(iw, "load_session", load_then_touch)
    res = compare(env_default, [pre(VAL_A, 100)], [tile(VAL_A, 20)])
    assert res["errors"][0]["code"] == "source_unavailable"


# ---- causality ----

def test_frames_after_every_window_never_change_the_comparison(tmp_path):
    """Corrupt every frame from the last window end on (coordinates, scores):
    the comparison is identical."""
    pre_ids, base_ids = [pre(VAL_A, 100), pre(VAL_B, 200)], [tile(VAL_A, 20), tile(VAL_B, 40)]

    def corrupt(sid, kp, sc):
        rng = np.random.default_rng(99)
        kp[200:] = rng.uniform(0, 1000, kp[200:].shape)
        sc[200:] = rng.uniform(0, 1, sc[200:].shape)
    a = compare(build(tmp_path, name="clean", attacks=SPACED), pre_ids, base_ids)
    b = compare(build(tmp_path, corrupt, name="corrupt", attacks=SPACED), pre_ids, base_ids)
    assert a["measurements"] == b["measurements"]
    assert a["measurements"]["n_sessions_paired"] == 2


# ---- disclosure ----

def test_confounds_and_provisional_choices_are_disclosed(spaced):
    res = compare(spaced, SPACED_PRE, SPACED_BASE)
    text = " ".join(res["limitations"]).lower()
    for phrase in ("investigation confound", "session-timing confound", "missing tracking", "pixel scale",
                   "motion accuracy is not verified", "no p-value, significance or confidence interval", "synthetic"):
        assert phrase in text
    names = {m["name"] for m in res["methodology"]}
    assert {"label_frames_pre", "label_frames_baseline", "window_start_s_pre", "window_start_s_baseline",
            "usable_frame_fraction_pre", "usable_frame_fraction_baseline", "smoothing_contamination_windows_pre",
            "smoothing_contamination_windows_baseline", "uncertainty_method", "baseline_clearance_frames",
            "min_sessions", "min_windows_per_group", "min_usable_fraction", "causal_computation",
            "score_validation", "units"} <= names
    assert all(m["status"] == "provisional" for m in res["methodology"])
    assert method(res, "units")["value"] == "px/s"
    assert method(res, "window_start_s_pre")["value"] == "2.66667 / 6 / 9.33333"
    assert not names & {"max_supported_ci_level", "uncertainty_resamples"}
    assert "confidence interval" in text and "not a population estimate" in text
    assert "significan" not in json.dumps(res["measurements"]).lower()


def test_approved_criteria_carry_who_decided(spaced):
    crit = SufficiencyCriteria(1, 1, 0.9, status="approved", decided_by="team review")
    res = compare(spaced, SPACED_PRE, SPACED_BASE, crit=crit)
    m = method(res, "min_sessions")
    assert (m["status"], m["decided_by"]) == ("approved", "team review")


def test_provenance_names_the_compare_code(spaced):
    res = compare(spaced, SPACED_PRE, SPACED_BASE)
    assert "investigation_compare.py" in method(res, "code_files")["value"]
    assert res["provenance"]["implementation_version"] == ev.IMPLEMENTATION_VERSION
    assert {d["kind"] for d in res["provenance"]["data"]} == {"synthetic_fixture"}


# ---- real data (verified validation cache; skipped when absent) ----

REAL = "task1/train/mouse005_task1_annotator1"
REAL_NO_ATTACK = "task1/train/mouse009_task1_annotator1"
REAL_ONSETS = (79, 251, 736, 934, 1147, 1969, 2072)


@pytest.fixture(scope="module")
def real_env():
    if not (REAL_CACHE / INDEX_NAME).exists() or not REAL_MANIFEST.exists():
        pytest.skip("verified CalMS21 validation cache not present")
    policy = EvidencePolicy.from_manifest(REAL_MANIFEST)
    tools = CalMS21InvestigationTools.for_validation_cache(REAL_CACHE, REAL_MANIFEST, policy)
    f = call_tool(tools, "get_window_features",
                  freq([ep(REAL, o) for o in REAL_ONSETS], length=30,
                       baseline=base(length=30, clearance=150, max_windows=500)), policy)
    included = [w["window_id"] for w in f["measurements"]["windows"]
                if w["kind"] == "baseline" and w["status"] == "included"]
    return tools, policy, [pre(REAL, o, length=30) for o in REAL_ONSETS], included, f


def test_real_comparison_is_insufficient_with_one_attack_session(real_env):
    """Of the three cached validation sessions only mouse005 has attacks: one
    paired session. With the provisional criteria no effect is reported."""
    tools, policy, pre_ids, base_ids, _ = real_env
    res = compare((tools, policy), pre_ids, base_ids, crit=PROVISIONAL_SUFFICIENCY)
    m = res["measurements"]
    assert res["status"] == "insufficient_evidence"
    assert m["sufficiency"]["reasons_not_met"] == ["min_sessions", "min_windows_per_group_pre"]
    assert (m["n_sessions_paired"], m["n_windows_pre"], m["n_windows_baseline"]) == (1, 5, 128)
    assert m["effect"]["value"] is None and m["effect"]["ci_low"] is None
    assert {x["window_id"]: (x["reason"], x["n_usable"]) for x in m["excluded_windows"]} == {
        pre(REAL, 934, length=30): ("overlaps_event", 30),
        pre(REAL, 1147, length=30): ("insufficient_usable_frames", 26)}
    r = rows(res)[REAL]
    assert [round(w["value"], 3) for w in r["pre_windows"]] == [-211.473, -74.735, -8.359, -1.134, -3.646]
    assert r["pre_summary"] == pytest.approx(-8.359411455944091, abs=1e-9)
    assert r["baseline_summary"] == pytest.approx(-3.2846561530421514, abs=1e-9)
    assert r["difference"] == pytest.approx(-5.07475530290194, abs=1e-9)
    assert method(res, "label_frames_pre")["value"] == "attack 0, investigation 101, mount 0, other 49"
    assert method(res, "label_frames_baseline")["value"] == "attack 0, investigation 274, mount 0, other 3566"
    assert method(res, "window_start_s_pre")["value"] == "1.63333 / 23.5333 / 68.0667"
    assert method(res, "window_start_s_baseline")["value"] == "18 / 123.5 / 187"
    assert {d["kind"] for d in res["provenance"]["data"]} == {"calms21_validation_cache"}


def test_real_values_match_stage4(real_env):
    tools, policy, pre_ids, base_ids, f = real_env
    res = compare((tools, policy), pre_ids, base_ids)
    s4 = {w["window_id"]: w["features"]["dist_rate"] for w in f["measurements"]["windows"]}
    for w in rows(res)[REAL]["pre_windows"] + rows(res)[REAL]["baseline_windows"]:
        assert (w["n_usable"], w["value"]) == (s4[w["window_id"]]["n_usable"], s4[w["window_id"]]["median"])


def test_real_single_session_effect_has_no_interval(real_env):
    """Loose criteria give the one-session difference as a descriptive effect,
    without an interval; a bootstrap request is refused."""
    tools, policy, pre_ids, base_ids, _ = real_env
    res = compare((tools, policy), pre_ids, base_ids)
    assert res["status"] == "ok"
    assert res["measurements"]["effect"]["value"] == pytest.approx(-5.07475530290194, abs=1e-9)
    assert res["measurements"]["effect"]["ci_low"] is None and res["measurements"]["effect"]["ci_level"] is None
    refused = compare((tools, policy), pre_ids, base_ids, unc=boot(0.6))
    assert refused["status"] == "error" and refused["errors"][0]["code"] == "not_implemented"


def test_real_session_without_attacks_is_listed_unpaired(real_env):
    tools, policy, pre_ids, base_ids, _ = real_env
    extra = [tile(REAL_NO_ATTACK, 30 * k, length=30) for k in (10, 20, 30)]
    res = compare((tools, policy), pre_ids, base_ids + extra)
    r = rows(res)[REAL_NO_ATTACK]
    assert r["pre_windows"] == [] and r["difference"] is None and len(r["baseline_windows"]) == 3
    assert res["measurements"]["n_sessions_paired"] == 1
