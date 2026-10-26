"""Tests for the experimental investigation tool contracts (new file).

Every request / result here is a SYNTHETIC test fixture (session IDs contain
SYNTHETIC, provenance kind "synthetic_fixture"); none is CalMS21 evidence.
"""

import copy
import json
from pathlib import Path

import pytest

import live_tracker.investigation_contracts as ic
from live_tracker.calms21 import FPS
from live_tracker.features import DYADIC_COLS
from live_tracker.investigation_contracts import (CONTRACT_VERSION, PROVISIONAL_SUFFICIENCY,
                                                  ContractError, EvidencePolicy, ForbiddenEvidence,
                                                  SufficiencyCriteria, ToolContractViolation,
                                                  UnavailableTools, baseline_window_id, call_tool,
                                                  check_request_schema, check_result_schema,
                                                  episode_id, make_result, parse_episode_id,
                                                  parse_window_id, pre_window_id, validate_request,
                                                  validate_result)
from live_tracker.rules import CLASSIFIER_BEHAVIORS

ROOT = Path(__file__).resolve().parents[1]
VAL_A, VAL_B = "task1/train/SYNTHETIC_val_a", "task1/train/SYNTHETIC_val_b"
TRAIN, TEST = "task1/train/SYNTHETIC_train", "task1/test/SYNTHETIC_test"
MANIFEST = {"train": [TRAIN], "validation": [VAL_A, VAL_B], "test": [TEST]}
POLICY = EvidencePolicy(MANIFEST, allow_synthetic=True)
SRC, BEH = "validation_annotation", "attack"
EP = episode_id(SRC, BEH, VAL_A, 90)
PRE = pre_window_id(EP, 30, 0)
BASE = baseline_window_id(SRC, BEH, VAL_A, 0, 30)
PROV = {"implementation": "synthetic-test-fixture", "implementation_version": "0",
        "contract_version": CONTRACT_VERSION, "code_sha256": None,
        "data": [{"kind": "synthetic_fixture", "path": None, "sha256": None,
                  "description": "SYNTHETIC test fixture, not CalMS21 data"}]}
METHOD = [{"name": "episode_definition", "value": "maximal run of attack frames",
           "status": "provisional", "decided_by": None, "note": None}]


def t(frame):
    return frame / FPS


# ---- valid synthetic fixtures, one per tool ----

def query_request(**change):
    return {"contract_version": CONTRACT_VERSION, "source": SRC, "behavior": BEH,
            "sessions": [VAL_A], "max_episodes": 10, "merge_gap_frames": None,
            "min_duration_frames": None, **change}


def query_measurements():
    return {"episodes": [{"episode_id": EP, "session": VAL_A, "source": SRC, "behavior": BEH,
                          "start_frame": 90, "end_frame": 120, "start_s": t(90), "end_s": t(120),
                          "fps": FPS, "provenance": {"split": "validation", "annotator": "1",
                                                     "file_sha256": "a" * 64},
                          "annotated_frames": 30}],
            "n_episodes_total": 1, "truncated": False, "sessions_searched": [VAL_A],
            "sessions_without_episodes": []}


def features_request(**change):
    return {"contract_version": CONTRACT_VERSION, "episode_ids": [EP],
            "pre_event": {"length_frames": 30, "gap_frames": 0},
            "baseline": {"strategy": "same_session_tiles", "length_frames": 30,
                         "clearance_frames": 0, "max_windows_per_session": 2},
            "features": ["dist_rate"], **change}


def window(wid, kind, start, end, *, episode=None, truncated=False, usable=28,
           median=-12.5, mean=-10.0, status="included", reasons=(), labels=True):
    return {"window_id": wid, "kind": kind, "session": VAL_A, "episode_id": episode,
            "start_frame": start, "end_frame": end, "start_s": t(start), "end_s": t(end),
            "status": status, "exclusion_reasons": list(reasons),
            "features": {"dist_rate": {"n_frames": end - start, "n_usable": usable,
                                       "median": median, "mean": mean}},
            "label_context": ({"attack": 0, "investigation": 20, "mount": 0, "other": 10}
                              if labels else None),
            "preprocessing": {"truncated_at_onset": truncated, "interpolation": False,
                              "smoothing": "one_euro"}}


def features_measurements():
    return {"windows": [window(PRE, "pre_event", 60, 90, episode=EP, truncated=True),
                        window(BASE, "baseline", 0, 30, median=-3.0, mean=-2.0)]}


def quality_request(**change):
    return {"contract_version": CONTRACT_VERSION, "window_ids": [PRE, BASE],
            "history_frames": [1, 15], **change}


def quality_item(wid, start, end, **change):
    item = {"window_id": wid, "start_frame": start, "end_frame": end, "frames": end - start,
            "tracking_ok_frames": 29, "endpoint_valid_frames": 28,
            "usable_by_history": {"1": 28, "15": 20},
            "excluded_frames": {"not_tracking_ok": [start + 5], "endpoint_invalid": [start + 6]},
            "smoothing_contamination": {"status": "not_assessed", "frames": [],
                                        "note": "no counterfactual run for this window"},
            "tracking_ok_meaning": ic.TRACKING_OK_MEANING, "motion_accuracy_verified": False,
            "verdict": "usable_with_caveats"}
    item.update(change)
    return item


def quality_measurements():
    return {"windows": [quality_item(PRE, 60, 90), quality_item(BASE, 0, 30)]}


def compare_request(sufficiency=None, **change):
    return {"contract_version": CONTRACT_VERSION, "feature": "dist_rate",
            "statistic": "median_difference", "aggregation": "session_level",
            "pre_window_ids": [PRE], "baseline_window_ids": [BASE],
            "sufficiency": (sufficiency or PROVISIONAL_SUFFICIENCY).to_json(),
            "uncertainty": {"method": "bootstrap_sessions", "level": 0.95, "resamples": 1000, "seed": 0},
            **change}


def wv(wid, value, n_usable=28, n_frames=30):
    return {"window_id": wid, "n_frames": n_frames, "n_usable": n_usable, "value": value}


def compare_measurements(req, met):
    return {"feature": "dist_rate", "aggregation": "session_level",
            "per_session": [{"session": VAL_A, "pre_windows": [wv(PRE, -12.5)],
                             "baseline_windows": [wv(BASE, -3.0, 29)],
                             "pre_summary": -12.5, "baseline_summary": -3.0, "difference": -9.5}],
            "excluded_windows": [], "n_sessions_paired": 1, "n_windows_pre": 1,
            "n_windows_baseline": 1,
            "effect": ({"name": "median_of_session_differences", "value": -9.5, "ci_low": -9.5,
                        "ci_high": -9.5, "ci_level": 0.95} if met else
                       {"name": "median_of_session_differences", "value": None, "ci_low": None,
                        "ci_high": None, "ci_level": None}),
            "sufficiency": {"criteria": req["sufficiency"], "met": met,
                            # PROVISIONAL (4 sessions, 10 windows per group) vs 1 / 1 / 1: all unmet
                            "reasons_not_met": [] if met else ["min_sessions", "min_windows_per_group_pre", "min_windows_per_group_baseline"]}}


LOOSE = SufficiencyCriteria(min_sessions=1, min_windows_per_group=1, min_usable_fraction=0.5)


def fixture(tool, variant="default"):
    """(request, result) for a tool; results are built with make_result."""
    if tool == "query_events":
        req, status, m = query_request(), "ok", query_measurements()
    elif tool == "get_window_features":
        req, status, m = features_request(), "ok", features_measurements()
    elif tool == "check_tracking_quality":
        req, status, m = quality_request(), "ok", quality_measurements()
    elif variant == "ok":
        req = compare_request(LOOSE)
        status, m = "ok", compare_measurements(req, True)
    else:
        req = compare_request()
        status, m = "insufficient_evidence", compare_measurements(req, False)
    return req, make_result(tool, req, status, m, methodology=METHOD,
                            limitations=["SYNTHETIC fixture"], provenance=PROV)


def rebuild(tool, req, res):
    """Recompute the evidence id after a deliberate change (to test the content check)."""
    res = dict(res)
    res["evidence_id"] = ic.evidence_id(tool, req, res)
    return res


ALL = [("query_events", "default"), ("get_window_features", "default"),
       ("check_tracking_quality", "default"), ("compare_windows", "default"),
       ("compare_windows", "ok")]


# ---- valid fixtures pass ----

@pytest.mark.parametrize("tool, variant", ALL)
def test_valid_fixtures_pass(tool, variant):
    req, res = fixture(tool, variant)
    validate_result(tool, req, res, POLICY)
    assert json.loads(json.dumps(res)) == res  # plain JSON


def test_contract_reuses_existing_constants():
    assert ic.FEATURES == tuple(DYADIC_COLS)
    assert set(ic.BEHAVIORS) == CLASSIFIER_BEHAVIORS
    assert ic.SOURCES == ("validation_annotation", "classifier_prediction", "monitor_event")


# ---- identifiers ----

def test_ids_are_deterministic_and_parse_back():
    assert EP == "ep:validation_annotation:attack:task1/train/SYNTHETIC_val_a:90"
    assert parse_episode_id(EP) == {"source": SRC, "behavior": BEH, "session": VAL_A, "start_frame": 90}
    assert pre_window_id(EP, 30, 0) == PRE
    assert parse_window_id(PRE)["onset_frame"] == 90 and parse_window_id(PRE)["kind"] == "pre_event"
    assert parse_window_id(BASE) == {"kind": "baseline", "length_frames": 30, "source": SRC,
                                     "behavior": BEH, "session": VAL_A, "start_frame": 0,
                                     "episode_id": None}


@pytest.mark.parametrize("call", [
    lambda: episode_id("annotations", BEH, VAL_A, 1),
    lambda: episode_id(SRC, "fighting", VAL_A, 1),
    lambda: episode_id(SRC, BEH, "a:b", 1),
    lambda: episode_id(SRC, BEH, "../x", 1),
    lambda: episode_id(SRC, BEH, VAL_A, -1),
    lambda: episode_id(SRC, BEH, VAL_A, 1.0),
    lambda: episode_id(SRC, BEH, VAL_A, True),
    lambda: pre_window_id("ep:bad", 30, 0),
    lambda: pre_window_id(EP, 0, 0),
    lambda: pre_window_id(EP, 30.0, 0),
    lambda: baseline_window_id(SRC, BEH, VAL_A, -5, 30),
    lambda: parse_window_id("win:other:1"),
    lambda: parse_window_id(None),
])
def test_invalid_ids_are_refused(call):
    with pytest.raises(ContractError):
        call()


# ---- generic request validation ----

@pytest.mark.parametrize("tool, change", [
    ("query_events", {"extra": 1}),                               # unknown field
    ("query_events", {"source": "annotations"}),                  # not a declared source
    ("query_events", {"behavior": "fighting"}),
    ("query_events", {"max_episodes": True}),                     # bool is not an int
    ("query_events", {"max_episodes": 10.0}),
    ("query_events", {"max_episodes": 0}),
    ("query_events", {"max_episodes": ic.MAX_EPISODES + 1}),      # oversized
    ("query_events", {"sessions": []}),
    ("query_events", {"sessions": [VAL_A, VAL_A]}),
    ("query_events", {"sessions": [f"task1/train/SYNTHETIC_{i}" for i in range(ic.MAX_SESSIONS + 1)]}),
    ("query_events", {"sessions": ["a:b"]}),
    ("query_events", {"contract_version": "investigation-tools/9.9"}),
    ("get_window_features", {"features": ["dist_rate", "speed_a"]}),     # not an allowed feature
    ("get_window_features", {"features": []}),
    ("get_window_features", {"pre_event": {"length_frames": ic.MAX_WINDOW_FRAMES + 1, "gap_frames": 0}}),
    ("get_window_features", {"pre_event": {"length_frames": 30, "gap_frames": 0, "smooth": 1}}),
    ("get_window_features", {"episode_ids": ["not-an-id"]}),
    ("get_window_features", {"episode_ids": [EP, episode_id("classifier_prediction", BEH, VAL_A, 5)]}),
    ("check_tracking_quality", {"history_frames": [1, 2, 3, 4, 5, 6]}),
    ("check_tracking_quality", {"history_frames": [0]}),
    ("check_tracking_quality", {"window_ids": [PRE, PRE]}),
    ("compare_windows", {"statistic": "t_test"}),
    ("compare_windows", {"aggregation": "episode_level"}),         # session independence only
    ("compare_windows", {"pre_window_ids": [BASE]}),               # wrong window kind
    ("compare_windows", {"baseline_window_ids": [PRE]}),
    ("compare_windows", {"uncertainty": {"method": "bootstrap_sessions", "level": 0.95,
                                         "resamples": 1000, "seed": None}}),
    ("compare_windows", {"uncertainty": {"method": "none", "level": 0.95, "resamples": None, "seed": None}}),
    ("compare_windows", {"pre_window_ids": [pre_window_id(episode_id("monitor_event", BEH, VAL_A, 90), 30, 0)]}),
])
def test_invalid_requests_are_refused(tool, change):
    req = fixture(tool)[0]
    req.update(change)
    with pytest.raises(ContractError):
        validate_request(tool, req, POLICY)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numbers_are_refused(bad):
    req = compare_request()
    req["sufficiency"]["min_usable_fraction"] = bad
    with pytest.raises(ContractError, match="non-finite"):
        validate_request("compare_windows", req, POLICY)


@pytest.mark.parametrize("bad", [("a",), {1: "x"}, {"set"}, b"bytes", object()])
def test_non_json_values_are_refused(bad):
    req = query_request(sessions=bad)
    with pytest.raises(ContractError):
        validate_request("query_events", req, POLICY)


def test_oversized_request_bytes_are_refused():
    wids = [baseline_window_id(SRC, BEH, VAL_A, 30 * i, 30) for i in range(1000)]
    req = quality_request(window_ids=wids)
    assert len(json.dumps(req)) > ic.MAX_REQUEST_BYTES
    with pytest.raises(ContractError, match="bytes"):
        validate_request("check_tracking_quality", req, POLICY)


def test_unknown_tool_is_refused():
    with pytest.raises(ContractError, match="unknown tool"):
        validate_request("delete_rules", {}, POLICY)


def test_approved_sufficiency_must_name_who_decided():
    req = compare_request(SufficiencyCriteria(4, 10, 0.9, status="approved"))
    with pytest.raises(ContractError, match="who decided"):
        validate_request("compare_windows", req, POLICY)
    validate_request("compare_windows", compare_request(SufficiencyCriteria(
        4, 10, 0.9, status="approved", decided_by="team meeting 2026-10-20")), POLICY)


# ---- held-out test annotations never become evidence ----

def test_annotation_requests_on_test_or_train_sessions_are_forbidden():
    for sid in (TEST, TRAIN):
        with pytest.raises(ForbiddenEvidence):
            validate_request("query_events", query_request(sessions=[sid]), POLICY)
    with pytest.raises(ForbiddenEvidence):
        validate_request("get_window_features",
                         features_request(episode_ids=[episode_id(SRC, BEH, TEST, 90)]), POLICY)
    with pytest.raises(ForbiddenEvidence):
        validate_request("check_tracking_quality",
                         quality_request(window_ids=[baseline_window_id(SRC, BEH, TEST, 0, 30)]), POLICY)


def test_predictions_and_monitor_events_may_cover_test_sessions():
    """Predictions carry no labels; deployment runs on held-out sessions."""
    for source in ("classifier_prediction", "monitor_event"):
        validate_request("query_events", query_request(source=source, sessions=[TEST]), POLICY)


def test_sessions_outside_the_manifest_are_refused():
    with pytest.raises(ForbiddenEvidence):
        validate_request("query_events", query_request(source="classifier_prediction",
                                                       sessions=["task1/train/SYNTHETIC_unknown"]), POLICY)


def test_label_context_for_a_test_session_is_forbidden():
    tid = episode_id("classifier_prediction", BEH, TEST, 90)
    req = features_request(episode_ids=[tid], baseline=None)
    w = window(pre_window_id(tid, 30, 0), "pre_event", 60, 90, episode=tid, truncated=True)
    w["session"] = TEST  # predictions on a test session are fine, its annotations are not
    res = make_result("get_window_features", req, "ok", {"windows": [w]}, provenance=PROV)
    with pytest.raises(ForbiddenEvidence, match="held-out test"):
        validate_result("get_window_features", req, res, POLICY)
    w["label_context"] = None
    validate_result("get_window_features", req,
                    make_result("get_window_features", req, "ok", {"windows": [w]}, provenance=PROV), POLICY)


def test_label_context_needs_a_policy():
    req, res = fixture("get_window_features")
    with pytest.raises(ForbiddenEvidence):
        validate_result("get_window_features", req, res, None)


def test_annotation_episode_from_a_test_session_is_forbidden_in_results():
    req = query_request(sessions=None)
    m = query_measurements()
    tep = episode_id(SRC, BEH, TEST, 90)
    m["episodes"][0].update(episode_id=tep, session=TEST)
    m["sessions_searched"] = [TEST]
    res = make_result("query_events", req, "ok", m, provenance=PROV)
    with pytest.raises(ForbiddenEvidence):
        validate_result("query_events", req, res, POLICY)


def test_annotation_provenance_must_say_validation():
    req, _ = fixture("query_events")
    m = query_measurements()
    m["episodes"][0]["provenance"]["split"] = "test"
    with pytest.raises(ContractError):
        validate_result("query_events", req, make_result("query_events", req, "ok", m, provenance=PROV), POLICY)


def test_policy_refuses_test_annotations_and_reads_the_real_manifest():
    with pytest.raises(ForbiddenEvidence):
        EvidencePolicy(MANIFEST, annotation_split="test")
    real = EvidencePolicy.from_manifest(ROOT / "splits" / "split_manifest.json")
    manifest = json.loads((ROOT / "splits" / "split_manifest.json").read_text())
    real.check_annotation_session(manifest["validation"][0], "x")
    for sid in (manifest["test"][0], manifest["train"][0]):
        with pytest.raises(ForbiddenEvidence):
            real.check_annotation_session(sid, "x")


def test_synthetic_fixtures_are_not_evidence_by_default():
    req, res = fixture("query_events")
    strict = EvidencePolicy(MANIFEST)
    with pytest.raises(ForbiddenEvidence, match="synthetic"):
        validate_result("query_events", req, res, strict)


# ---- result envelope ----

def test_evidence_id_binds_request_and_content():
    req, res = fixture("query_events")
    assert res["evidence_id"].startswith("ev:query_events:")
    assert fixture("query_events")[1]["evidence_id"] == res["evidence_id"]  # deterministic
    tampered = copy.deepcopy(res)
    tampered["measurements"]["episodes"][0]["end_frame"] = 121
    tampered["measurements"]["episodes"][0]["end_s"] = t(121)
    with pytest.raises(ContractError, match="evidence_id"):
        validate_result("query_events", req, tampered, POLICY)
    other_req = query_request(max_episodes=11)
    with pytest.raises(ContractError, match="evidence_id"):
        validate_result("query_events", other_req, res, POLICY)


@pytest.mark.parametrize("change", [
    lambda r: r.__setitem__("extra", 1),
    lambda r: r.__setitem__("tool", "compare_windows"),
    lambda r: r.__setitem__("status", "maybe"),
    lambda r: r["provenance"].__setitem__("code_sha256", "xyz"),
    lambda r: r["provenance"]["data"][0].__setitem__("kind", "csv"),
    lambda r: r["methodology"][0].__setitem__("status", "approved"),          # approved w/o decided_by
    lambda r: r["methodology"][0].__setitem__("value", {"nested": 1}),
    lambda r: r.__setitem__("errors", [{"code": "internal_error", "message": "x"}]),  # errors on ok
    lambda r: r.__setitem__("measurements", None),
    lambda r: r["limitations"].append(1),
])
def test_invalid_envelopes_are_refused(change):
    req, res = fixture("query_events")
    res = copy.deepcopy(res)
    change(res)
    with pytest.raises(ContractError):
        validate_result("query_events", req, rebuild("query_events", req, res), POLICY)


def test_error_results_have_errors_and_no_measurements():
    req = query_request()
    ok = make_result("query_events", req, "error", None, provenance=PROV,
                     errors=[{"code": "source_unavailable", "message": "no score file"}])
    validate_result("query_events", req, ok, POLICY)
    for bad in (make_result("query_events", req, "error", None, provenance=PROV),
                make_result("query_events", req, "error", query_measurements(), provenance=PROV,
                            errors=[{"code": "internal_error", "message": "x"}])):
        with pytest.raises(ContractError):
            validate_result("query_events", req, bad, POLICY)


# ---- query_events ----

@pytest.mark.parametrize("change", [
    lambda m: m["episodes"][0].__setitem__("source", "monitor_event"),
    lambda m: m["episodes"][0].__setitem__("episode_id", episode_id(SRC, BEH, VAL_A, 91)),
    lambda m: m["episodes"][0].__setitem__("end_frame", 90),
    lambda m: m["episodes"][0].__setitem__("start_s", 3.1),
    lambda m: m["episodes"][0].__setitem__("fps", 25.0),
    lambda m: m["episodes"][0].__setitem__("label", "attack"),            # unknown field
    lambda m: m["episodes"][0]["provenance"].__setitem__("model_id", "x"),  # wrong provenance shape
    lambda m: m.__setitem__("episodes", m["episodes"] * 2),
    lambda m: m.__setitem__("truncated", True),
    lambda m: m.__setitem__("sessions_searched", [VAL_A, VAL_B]),         # VAL_B was not requested
    lambda m: m.__setitem__("sessions_without_episodes", [VAL_A]),
])
def test_invalid_query_results_are_refused(change):
    req = query_request()
    m = query_measurements()
    change(m)
    with pytest.raises(ContractError):
        validate_result("query_events", req, make_result("query_events", req, "ok", m, provenance=PROV), POLICY)


def test_no_data_is_explicit():
    req = query_request()
    m = {"episodes": [], "n_episodes_total": 0, "truncated": False, "sessions_searched": [VAL_A],
         "sessions_without_episodes": [VAL_A]}
    validate_result("query_events", req, make_result("query_events", req, "no_data", m, provenance=PROV), POLICY)
    with pytest.raises(ContractError, match="no_data"):
        validate_result("query_events", req, make_result("query_events", req, "ok", m, provenance=PROV), POLICY)


# ---- get_window_features ----

def _features_with(change, req=None):
    req = req or features_request()
    m = features_measurements()
    change(m)
    return req, make_result("get_window_features", req, "ok", m, provenance=PROV)


@pytest.mark.parametrize("change", [
    lambda m: m["windows"][0].__setitem__("episode_id", episode_id(SRC, BEH, VAL_A, 500)),  # not requested
    lambda m: m["windows"][0].__setitem__("window_id", pre_window_id(EP, 31, 0)),
    lambda m: m["windows"][0]["preprocessing"].__setitem__("truncated_at_onset", False),
    lambda m: m["windows"][0]["preprocessing"].__setitem__("interpolation", True),
    lambda m: m["windows"][0].__setitem__("end_frame", 89),
    lambda m: m["windows"][0]["features"].__setitem__("speed_corr_w", m["windows"][0]["features"]["dist_rate"]),
    lambda m: m["windows"][0]["features"]["dist_rate"].__setitem__("n_usable", 31),
    lambda m: m["windows"][0]["features"]["dist_rate"].__setitem__("median", None),  # null median, usable > 0
    lambda m: m["windows"][0]["features"]["dist_rate"].__setitem__("mean", float("nan")),
    lambda m: m["windows"][0].__setitem__("status", "excluded"),                   # no reason given
    lambda m: m["windows"][0].__setitem__("exclusion_reasons", ["near_event"]),    # reason on included
    lambda m: m["windows"].pop(0),                                                 # episode unanswered
    lambda m: m["windows"].append(copy.deepcopy(m["windows"][0])),                 # duplicate
    lambda m: m["windows"][1].__setitem__("start_frame", 1),
    lambda m: m["windows"][1].__setitem__("kind", "pre_event"),
])
def test_invalid_feature_results_are_refused(change):
    with pytest.raises(ContractError):  # make_result itself refuses non-JSON content
        req, res = _features_with(change)
        validate_result("get_window_features", req, res, POLICY)


def test_baseline_windows_need_a_baseline_request():
    req, res = _features_with(lambda m: None, features_request(baseline=None))
    with pytest.raises(ContractError, match="baseline"):
        validate_result("get_window_features", req, res, POLICY)


def test_missing_data_and_exclusions_are_explicit():
    def no_usable(m):
        m["windows"][1] = window(BASE, "baseline", 0, 30, usable=0, median=None, mean=None,
                                 status="excluded", reasons=["insufficient_usable_frames"])
    req, res = _features_with(no_usable)
    validate_result("get_window_features", req, res, POLICY)


# ---- check_tracking_quality ----

def _quality_with(change):
    req = quality_request()
    m = quality_measurements()
    change(m["windows"][0])
    return req, make_result("check_tracking_quality", req, "ok", m, provenance=PROV)


@pytest.mark.parametrize("change", [
    lambda q: q.__setitem__("motion_accuracy_verified", True),       # never claimable in v0.1
    lambda q: q.__setitem__("tracking_ok_meaning", "accurate"),
    lambda q: q.__setitem__("verdict", "accurate"),
    lambda q: q.__setitem__("endpoint_valid_frames", 30),            # more than tracking_ok
    lambda q: q["usable_by_history"].__setitem__("15", 29),          # grows with history
    lambda q: q["usable_by_history"].pop("15"),
    lambda q: q["usable_by_history"].__setitem__("2", 20),
    lambda q: q["excluded_frames"].__setitem__("not_tracking_ok", []),
    lambda q: q["excluded_frames"].__setitem__("not_tracking_ok", [10]),          # outside the window
    lambda q: q["excluded_frames"].__setitem__("endpoint_invalid", [65]),         # same frame twice
    lambda q: q.__setitem__("smoothing_contamination", {"status": "detected", "frames": [], "note": "x"}),
    lambda q: q.__setitem__("smoothing_contamination", {"status": "none_detected", "frames": [70],
                                                       "note": "x"}),
    lambda q: q.__setitem__("smoothing_contamination", {"status": "not_assessed", "frames": []}),
    lambda q: q.update(start_frame=59, end_frame=89),                 # does not match the window id
    lambda q: q.update(tracking_ok_frames=0, endpoint_valid_frames=0, usable_by_history={"1": 0, "15": 0},
                       excluded_frames={"not_tracking_ok": list(range(60, 90)), "endpoint_invalid": []}),
])
def test_invalid_quality_results_are_refused(change):
    req, res = _quality_with(change)
    with pytest.raises(ContractError):
        validate_result("check_tracking_quality", req, res, POLICY)


def test_detected_contamination_is_representable():
    req, res = _quality_with(lambda q: q.update(smoothing_contamination={
        "status": "detected", "frames": [70, 71, 72], "note": "counterfactual shows a carried-over spike"}))
    validate_result("check_tracking_quality", req, res, POLICY)


def test_quality_must_answer_exactly_the_requested_windows():
    req, res = _quality_with(lambda q: q.update(window_id=pre_window_id(EP, 30, 1), end_frame=89,
                                                start_frame=59))
    with pytest.raises(ContractError):
        validate_result("check_tracking_quality", req, res, POLICY)


# ---- compare_windows ----

def _compare_with(change, variant="ok", status=None):
    req, res = fixture("compare_windows", variant)
    m = copy.deepcopy(res["measurements"])
    change(m)
    return req, make_result("compare_windows", req, status or res["status"], m, provenance=PROV)


@pytest.mark.parametrize("change, variant, status", [
    (lambda m: m["effect"].__setitem__("value", -9.5), "default", None),        # effect while insufficient
    (lambda m: m["sufficiency"].__setitem__("met", True), "default", None),
    (lambda m: m["sufficiency"].__setitem__("reasons_not_met", []), "default", None),
    (lambda m: m["sufficiency"].__setitem__("met", False), "ok", None),
    (lambda m: m["effect"].__setitem__("value", None), "ok", None),
    (lambda m: m["effect"].__setitem__("ci_high", -10.0), "ok", None),          # value outside its interval
    (lambda m: m["effect"].__setitem__("ci_level", 0.9), "ok", None),
    (lambda m: m["per_session"].append(copy.deepcopy(m["per_session"][0])), "ok", None),  # same session twice
    (lambda m: m["per_session"][0].__setitem__("difference", -9.0), "ok", None),
    (lambda m: m["per_session"][0].__setitem__("baseline_windows", []), "ok", None),
    (lambda m: m["per_session"][0].__setitem__("session", VAL_B), "ok", None),
    (lambda m: m["excluded_windows"].append({"window_id": PRE, "reason": "missing_data",
                                             "n_frames": None, "n_usable": None}), "ok", None),
    (lambda m: m["per_session"][0]["pre_windows"].append(wv(pre_window_id(EP, 30, 5), 1.0)), "ok", None),
    (lambda m: m.__setitem__("n_sessions_paired", 2), "ok", None),
    (lambda m: m.__setitem__("n_windows_pre", 2), "ok", None),
    (lambda m: m.__setitem__("aggregation", "episode_level"), "ok", None),
    (lambda m: m["sufficiency"]["criteria"].__setitem__("min_sessions", 1), "default", None),
    (lambda m: None, "default", "ok"),
])
def test_invalid_comparisons_are_refused(change, variant, status):
    req, res = _compare_with(change, variant, status)
    with pytest.raises(ContractError):
        validate_result("compare_windows", req, res, POLICY)


def test_ok_needs_enough_paired_sessions():
    req = compare_request(SufficiencyCriteria(min_sessions=2, min_windows_per_group=1, min_usable_fraction=0.5))
    m = compare_measurements(req, True)
    res = make_result("compare_windows", req, "ok", m, provenance=PROV)
    with pytest.raises(ContractError, match="min_sessions"):
        validate_result("compare_windows", req, res, POLICY)


def test_sufficiency_is_configurable_and_marked_provisional():
    assert PROVISIONAL_SUFFICIENCY.status == "provisional" and PROVISIONAL_SUFFICIENCY.decided_by is None
    req, res = fixture("compare_windows")
    assert res["measurements"]["sufficiency"]["criteria"]["status"] == "provisional"
    req2, res2 = fixture("compare_windows", "ok")  # same data, other criteria, other verdict
    assert res2["status"] == "ok" and res["status"] == "insufficient_evidence"


def test_excluded_windows_are_accounted_for():
    def exclude_baseline(m):
        m["per_session"][0].update(baseline_windows=[], baseline_summary=None, difference=None)
        m["excluded_windows"] = [{"window_id": BASE, "reason": "insufficient_usable_frames",
                                  "n_frames": 30, "n_usable": 12}]
        m["n_sessions_paired"], m["n_windows_baseline"] = 0, 0
        m["effect"].update(value=None, ci_low=None, ci_high=None, ci_level=None)
        m["sufficiency"].update(met=False, reasons_not_met=["min_sessions", "min_windows_per_group_pre", "min_windows_per_group_baseline"])
    req, res = _compare_with(exclude_baseline, "default")
    validate_result("compare_windows", req, res, POLICY)


# ---- the interface and call_tool ----

class FakeTools:
    """SYNTHETIC implementation returning the fixtures above."""
    implementation, implementation_version = "synthetic-fake", "0"

    def __init__(self, mutate=None, raise_exc=None):
        self.mutate, self.raise_exc, self.seen = mutate, raise_exc, []

    def _answer(self, tool, request):
        self.seen.append(request)
        if self.raise_exc:
            raise self.raise_exc
        req, res = fixture(tool)
        assert request == req
        request["max_episodes"] = 999  # mutating its copy must not affect the caller
        if self.mutate:
            res = copy.deepcopy(res)
            self.mutate(res)
        return res

    def query_events(self, request):
        return self._answer("query_events", request)

    def get_window_features(self, request):
        return self._answer("get_window_features", request)

    def check_tracking_quality(self, request):
        return self._answer("check_tracking_quality", request)

    def compare_windows(self, request):
        return self._answer("compare_windows", request)


def test_call_tool_validates_both_sides():
    tools = FakeTools()
    assert isinstance(tools, ic.InvestigationTools)
    req = query_request()
    res = call_tool(tools, "query_events", req, POLICY)
    assert res == fixture("query_events")[1]
    assert req["max_episodes"] == 10  # the implementation only saw a copy


def test_invalid_requests_never_reach_the_implementation():
    tools = FakeTools()
    with pytest.raises(ForbiddenEvidence):
        call_tool(tools, "query_events", query_request(sessions=[TEST]), POLICY)
    assert tools.seen == []


def test_implementation_exceptions_become_sanitized_error_results():
    tools = FakeTools(raise_exc=RuntimeError("secret /path/to/key"))
    res = call_tool(tools, "query_events", query_request(), POLICY)
    assert res["status"] == "error" and res["measurements"] is None
    assert res["errors"] == [{"code": "internal_error", "message": "query_events failed (RuntimeError)"}]
    validate_result("query_events", query_request(), res, POLICY)


def test_contract_breaking_results_are_violations_not_evidence():
    tools = FakeTools(mutate=lambda r: r["measurements"]["episodes"][0].__setitem__("fps", 25.0))
    with pytest.raises(ToolContractViolation):
        call_tool(tools, "query_events", query_request(), POLICY)


@pytest.mark.parametrize("tool", ic.TOOLS)
def test_unavailable_tools_answer_with_explicit_errors(tool):
    tools = UnavailableTools()
    assert isinstance(tools, ic.InvestigationTools)
    req = fixture(tool)[0]
    res = call_tool(tools, tool, req, POLICY)
    assert res["status"] == "error" and res["errors"][0]["code"] == "not_implemented"


# ---- targeted checks (each would otherwise be masked by another rule) ----

def test_one_row_per_session_even_with_disjoint_windows():
    """Splitting one session's windows into two rows would double-count it."""
    ep2 = episode_id(SRC, BEH, VAL_A, 300)
    pre2, base2 = pre_window_id(ep2, 30, 0), baseline_window_id(SRC, BEH, VAL_A, 150, 30)
    req = compare_request(LOOSE, pre_window_ids=[PRE, pre2], baseline_window_ids=[BASE, base2])
    m = compare_measurements(req, True)
    row = m["per_session"][0]
    m["per_session"] = [row, {**row, "pre_windows": [wv(pre2, -12.5)], "baseline_windows": [wv(base2, -3.0)]}]
    m.update(n_sessions_paired=2, n_windows_pre=2, n_windows_baseline=2)
    res = make_result("compare_windows", req, "ok", m, provenance=PROV)
    with pytest.raises(ContractError, match="one row per session"):
        validate_result("compare_windows", req, res, POLICY)


def test_endpoint_invalid_frames_must_all_be_listed():
    req, res = _quality_with(lambda q: q["excluded_frames"].__setitem__("endpoint_invalid", []))
    with pytest.raises(ContractError, match="endpoint_invalid"):
        validate_result("check_tracking_quality", req, res, POLICY)


def test_windows_for_unrequested_episodes_are_refused():
    other = episode_id(SRC, BEH, VAL_A, 500)

    def add_unrequested(m):
        m["windows"].append(window(pre_window_id(other, 30, 0), "pre_event", 470, 500,
                                   episode=other, truncated=True))
    req, res = _features_with(add_unrequested)
    with pytest.raises(ContractError, match="not a requested episode"):
        validate_result("get_window_features", req, res, POLICY)


# ======================================================================
# Codex review 1 regressions (all SYNTHETIC fixtures)
# ======================================================================

# ---- 1. fail-closed annotation policy ----

class LookalikePolicy:
    """Has the right methods but is not a manifest-backed EvidencePolicy."""
    allow_synthetic = True

    def check_annotation_session(self, sid, path):
        return None

    def check_known_session(self, sid, path):
        return None


@pytest.mark.parametrize("policy", [None, {}, MANIFEST, LookalikePolicy()])
def test_evidence_acceptance_requires_a_manifest_backed_policy(policy):
    req, res = fixture("query_events")
    with pytest.raises(ForbiddenEvidence, match="EvidencePolicy"):
        validate_request("query_events", req, policy)
    with pytest.raises(ForbiddenEvidence, match="EvidencePolicy"):
        validate_result("query_events", req, res, policy)
    tools = FakeTools()
    with pytest.raises(ForbiddenEvidence, match="EvidencePolicy"):
        call_tool(tools, "query_events", req, policy)
    assert tools.seen == []


def test_call_tool_has_no_policy_default():
    with pytest.raises(TypeError):
        call_tool(FakeTools(), "query_events", query_request())


def test_schema_checks_are_separate_and_never_accept_evidence():
    """A test-session annotation request is structurally valid (schema check
    passes) but is never accepted as evidence."""
    req = query_request(sessions=[TEST])
    check_request_schema("query_events", req)
    with pytest.raises(ForbiddenEvidence):
        validate_request("query_events", req, POLICY)
    req, res = fixture("query_events")
    check_result_schema("query_events", req, res)


@pytest.mark.parametrize("sid, ok", [(VAL_A, True), (TEST, False), (TRAIN, False),
                                     ("task1/train/SYNTHETIC_unlisted", False)])
def test_annotation_sessions_by_split(sid, ok):
    req = query_request(sessions=[sid])
    if ok:
        validate_request("query_events", req, POLICY)
    else:
        with pytest.raises(ForbiddenEvidence):
            validate_request("query_events", req, POLICY)


def test_nested_annotation_provenance_on_a_test_session_is_forbidden():
    """Episode provenance says 'validation', but the session is a test session."""
    req = query_request(sessions=None)
    m = query_measurements()
    tep = episode_id(SRC, BEH, TEST, 90)
    m["episodes"][0].update(episode_id=tep, session=TEST)
    m["sessions_searched"] = [TEST]
    res = make_result("query_events", req, "ok", m, provenance=PROV)
    check_result_schema("query_events", req, res)      # structurally fine
    with pytest.raises(ForbiddenEvidence, match="held-out test"):
        validate_result("query_events", req, res, POLICY)


def test_prediction_episode_cannot_carry_annotation_provenance():
    req = query_request(source="classifier_prediction")
    m = query_measurements()
    cep = episode_id("classifier_prediction", BEH, VAL_A, 90)
    m["episodes"][0].update(episode_id=cep, source="classifier_prediction")  # annotation-shaped provenance
    del m["episodes"][0]["annotated_frames"]
    res = make_result("query_events", req, "ok", m, provenance=PROV)
    with pytest.raises(ContractError):
        validate_result("query_events", req, res, POLICY)


@pytest.mark.parametrize("manifest", [
    {"train": [], "validation": [], "test": []},                  # no validation sessions
    {"train": [VAL_A], "validation": [VAL_A], "test": []},        # overlapping splits
    {"validation": [VAL_A], "test": []},                          # missing split
    {"train": [], "validation": ["../x"], "test": []},            # invalid id
    ["not", "a", "manifest"],
])
def test_policy_rejects_malformed_manifests(manifest):
    with pytest.raises(ContractError):
        EvidencePolicy(manifest)


# ---- 2. every sufficiency criterion is enforced ----

def _ok_compare(req, change):
    m = compare_measurements(req, True)
    change(m)
    return make_result("compare_windows", req, "ok", m, provenance=PROV)


def test_met_cannot_be_claimed_with_too_few_windows():
    """Codex: min_windows_per_group unmet, yet met=True and status ok."""
    req = compare_request(SufficiencyCriteria(min_sessions=1, min_windows_per_group=5,
                                              min_usable_fraction=0.5))
    res = _ok_compare(req, lambda m: None)
    with pytest.raises(ContractError, match="min_windows_per_group"):
        validate_result("compare_windows", req, res, POLICY)


def test_met_cannot_be_claimed_with_too_little_usable_coverage():
    """Codex: a used window below min_usable_fraction, yet met=True."""
    req = compare_request(SufficiencyCriteria(min_sessions=1, min_windows_per_group=1,
                                              min_usable_fraction=0.95))
    res = _ok_compare(req, lambda m: None)  # pre window has 28/30 = 0.933 usable
    with pytest.raises(ContractError, match="min_usable_fraction"):
        validate_result("compare_windows", req, res, POLICY)


def test_unmet_cannot_be_claimed_when_every_criterion_is_met():
    req = compare_request(LOOSE)
    m = compare_measurements(req, False)
    m["per_session"][0]["pre_windows"][0]["value"] = -12.5
    res = make_result("compare_windows", req, "insufficient_evidence", m, provenance=PROV)
    with pytest.raises(ContractError, match="must be True"):
        validate_result("compare_windows", req, res, POLICY)


def test_coverage_exclusions_must_show_the_counts():
    req = compare_request(LOOSE)

    def exclude(m, n_usable):
        m["per_session"][0].update(baseline_windows=[], baseline_summary=None, difference=None)
        m["excluded_windows"] = [{"window_id": BASE, "reason": "insufficient_usable_frames",
                                  "n_frames": 30 if n_usable is not None else None, "n_usable": n_usable}]
        m["n_sessions_paired"], m["n_windows_baseline"] = 0, 0
        m["effect"].update(value=None, ci_low=None, ci_high=None, ci_level=None)
        m["sufficiency"].update(met=False, reasons_not_met=["min_sessions", "min_windows_per_group_baseline"])

    for n_usable in (29, None):  # 29/30 is above 0.5; None hides the counts
        m = compare_measurements(req, False)
        exclude(m, n_usable)
        res = make_result("compare_windows", req, "insufficient_evidence", m, provenance=PROV)
        with pytest.raises(ContractError, match="insufficient_usable_frames"):
            validate_result("compare_windows", req, res, POLICY)
    m = compare_measurements(req, False)
    exclude(m, 10)
    validate_result("compare_windows", req,
                    make_result("compare_windows", req, "insufficient_evidence", m, provenance=PROV), POLICY)


# ---- 3. statistics are recomputed (a fresh evidence hash does not help) ----

def _two_session_ok(effect, ci=(None, None), uncertainty="none", diffs=(-9.5, -1.5)):
    ep_b = episode_id(SRC, BEH, VAL_B, 90)
    pre_b, base_b = pre_window_id(ep_b, 30, 0), baseline_window_id(SRC, BEH, VAL_B, 0, 30)
    u = ({"method": "none", "level": None, "resamples": None, "seed": None} if uncertainty == "none"
         else {"method": "bootstrap_sessions", "level": 0.95, "resamples": 1000, "seed": 0})
    req = compare_request(LOOSE, pre_window_ids=[PRE, pre_b], baseline_window_ids=[BASE, base_b],
                          uncertainty=u)
    m = compare_measurements(req, True)
    m["per_session"] = [
        {"session": VAL_A, "pre_windows": [wv(PRE, diffs[0])], "baseline_windows": [wv(BASE, 0.0)],
         "pre_summary": diffs[0], "baseline_summary": 0.0, "difference": diffs[0]},
        {"session": VAL_B, "pre_windows": [wv(pre_b, diffs[1])], "baseline_windows": [wv(base_b, 0.0)],
         "pre_summary": diffs[1], "baseline_summary": 0.0, "difference": diffs[1]}]
    m.update(n_sessions_paired=2, n_windows_pre=2, n_windows_baseline=2)
    m["effect"].update(value=effect, ci_low=ci[0], ci_high=ci[1],
                       ci_level=None if uncertainty == "none" else 0.95)
    return req, make_result("compare_windows", req, "ok", m, provenance=PROV)


def test_effect_must_be_the_median_of_session_differences():
    req, res = _two_session_ok(-5.5)                   # median of -9.5 and -1.5
    validate_result("compare_windows", req, res, POLICY)
    for wrong in (-9.5, -1.5, -5.4, 0.0):
        req, res = _two_session_ok(wrong)              # evidence hash recomputed by make_result
        with pytest.raises(ContractError, match="median of the paired session differences"):
            validate_result("compare_windows", req, res, POLICY)


def test_effect_tolerance_is_explicit():
    req, res = _two_session_ok(-5.5 + ic.STAT_ABS_TOL / 2)
    validate_result("compare_windows", req, res, POLICY)
    req, res = _two_session_ok(-5.5 + 1e-6)
    with pytest.raises(ContractError):
        validate_result("compare_windows", req, res, POLICY)


def test_session_summaries_must_be_medians_of_window_values():
    req = compare_request(LOOSE)
    res = _ok_compare(req, lambda m: m["per_session"][0].update(pre_summary=-12.0, difference=-9.0))
    with pytest.raises(ContractError, match="median of its window values"):
        validate_result("compare_windows", req, res, POLICY)


@pytest.mark.parametrize("ci", [(-12.0, -1.5), (-9.5, 0.0), (-4.0, -1.5)])
def test_interval_must_contain_the_value_and_stay_within_the_differences(ci):
    req, res = _two_session_ok(-5.5, ci, uncertainty="bootstrap")
    with pytest.raises(ContractError, match="interval"):
        validate_result("compare_windows", req, res, POLICY)
    req, res = _two_session_ok(-5.5, (-9.5, -1.5), uncertainty="bootstrap")
    validate_result("compare_windows", req, res, POLICY)


# ---- 4. impossible windows, caps, label counts ----

def test_label_context_must_match_the_window_length():
    req, res = _features_with(lambda m: m["windows"][0]["label_context"].__setitem__("other", 999))
    with pytest.raises(ContractError, match="add up to the window length"):
        validate_result("get_window_features", req, res, POLICY)


def test_pre_window_starting_before_the_session_is_clipped_and_excluded():
    early = episode_id(SRC, BEH, VAL_A, 10)
    req = features_request(episode_ids=[early], baseline=None)
    w = window(pre_window_id(early, 30, 0), "pre_event", 0, 10, episode=early, truncated=True,
               usable=10, status="excluded", reasons=["starts_before_session"], labels=False)
    res = make_result("get_window_features", req, "no_data", {"windows": [w]}, provenance=PROV)
    validate_result("get_window_features", req, res, POLICY)
    for change in ({"status": "included", "exclusion_reasons": []},       # impossible: included
                   {"start_frame": -20},                                   # negative frames
                   {"exclusion_reasons": ["near_event"]}):                 # wrong reason
        bad = {**w, **change}
        bad_res = make_result("get_window_features", req, "no_data", {"windows": [bad]}, provenance=PROV)
        with pytest.raises(ContractError):
            validate_result("get_window_features", req, bad_res, POLICY)


def test_onset_at_frame_zero_gives_an_empty_excluded_window():
    first = episode_id(SRC, BEH, VAL_A, 0)
    req = features_request(episode_ids=[first], baseline=None)
    w = window(pre_window_id(first, 30, 0), "pre_event", 0, 0, episode=first, truncated=True,
               usable=0, median=None, mean=None, status="excluded", reasons=["starts_before_session"],
               labels=False)
    validate_result("get_window_features", req,
                    make_result("get_window_features", req, "no_data", {"windows": [w]}, provenance=PROV), POLICY)


def test_full_pre_window_cannot_claim_it_starts_before_the_session():
    req, res = _features_with(lambda m: m["windows"][0].update(
        status="excluded", exclusion_reasons=["starts_before_session"]))
    with pytest.raises(ContractError):
        validate_result("get_window_features", req, res, POLICY)


@pytest.mark.parametrize("start", [15, 45])
def test_baseline_tiles_must_be_aligned(start):
    def move(m):
        bid = baseline_window_id(SRC, BEH, VAL_A, start, 30)
        m["windows"][1] = window(bid, "baseline", start, start + 30)
    req, res = _features_with(move)
    with pytest.raises(ContractError, match="baseline tiles"):
        validate_result("get_window_features", req, res, POLICY)


def test_baseline_cap_is_enforced():
    def three_tiles(m):
        m["windows"][1:] = [window(baseline_window_id(SRC, BEH, VAL_A, s, 30), "baseline", s, s + 30)
                            for s in (0, 120, 150)]
    req, res = _features_with(three_tiles)   # max_windows_per_session is 2
    with pytest.raises(ContractError, match="max_windows_per_session"):
        validate_result("get_window_features", req, res, POLICY)
    req, res = _features_with(lambda m: m["windows"].append(
        window(baseline_window_id(SRC, BEH, VAL_A, 150, 30), "baseline", 150, 180, status="excluded",
               reasons=["near_event"])))
    validate_result("get_window_features", req, res, POLICY)  # excluded tiles do not count


def test_included_baseline_cannot_overlap_an_included_pre_window():
    req, res = _features_with(lambda m: m["windows"].__setitem__(
        1, window(baseline_window_id(SRC, BEH, VAL_A, 60, 30), "baseline", 60, 90)))
    with pytest.raises(ContractError, match="overlaps"):
        validate_result("get_window_features", req, res, POLICY)


def test_baseline_exclusion_reasons_are_baseline_reasons():
    req, res = _features_with(lambda m: m["windows"][1].update(
        status="excluded", exclusion_reasons=["starts_before_session"]))
    with pytest.raises(ContractError, match="baseline exclusion reason"):
        validate_result("get_window_features", req, res, POLICY)


def test_included_window_needs_usable_frames():
    req, res = _features_with(lambda m: m["windows"][1]["features"]["dist_rate"].update(
        n_usable=0, median=None, mean=None))
    with pytest.raises(ContractError, match="included window needs usable frames"):
        validate_result("get_window_features", req, res, POLICY)


# ---- 5. status invariants for every tool ----

def test_features_no_data_exactly_when_nothing_is_included():
    req, res = fixture("get_window_features")
    with pytest.raises(ContractError, match="no_data"):
        validate_result("get_window_features", req,
                        make_result("get_window_features", req, "no_data", res["measurements"],
                                    provenance=PROV), POLICY)
    with pytest.raises(ContractError, match="sufficiency"):
        validate_result("get_window_features", req,
                        make_result("get_window_features", req, "insufficient_evidence",
                                    res["measurements"], provenance=PROV), POLICY)


def _all_invalid(q):
    q.update(tracking_ok_frames=0, endpoint_valid_frames=0, usable_by_history={"1": 0, "15": 0},
             excluded_frames={"not_tracking_ok": list(range(q["start_frame"], q["end_frame"])),
                              "endpoint_invalid": []}, verdict="insufficient_data")


def test_quality_no_data_exactly_when_no_window_has_valid_frames():
    req = quality_request()
    m = quality_measurements()
    with pytest.raises(ContractError, match="no_data"):
        validate_result("check_tracking_quality", req,
                        make_result("check_tracking_quality", req, "no_data", m, provenance=PROV), POLICY)
    for q in m["windows"]:
        _all_invalid(q)
    validate_result("check_tracking_quality", req,
                    make_result("check_tracking_quality", req, "no_data", m, provenance=PROV), POLICY)
    with pytest.raises(ContractError, match="no_data"):
        validate_result("check_tracking_quality", req,
                        make_result("check_tracking_quality", req, "ok", m, provenance=PROV), POLICY)


def _empty_comparison(req, **change):
    m = compare_measurements(req, False)
    m.update(per_session=[], n_sessions_paired=0, n_windows_pre=0, n_windows_baseline=0,
             excluded_windows=[{"window_id": PRE, "reason": "missing_data", "n_frames": None, "n_usable": None},
                               {"window_id": BASE, "reason": "missing_data", "n_frames": None, "n_usable": None}])
    m["sufficiency"].update(met=False, reasons_not_met=["min_sessions", "min_windows_per_group_pre", "min_windows_per_group_baseline"])
    for k, v in change.items():
        if k == "effect":
            m["effect"].update(v)
        elif k == "sufficiency":
            m["sufficiency"].update(v)
    return m


@pytest.mark.parametrize("status, change", [
    ("ok", {}),
    ("insufficient_evidence", {}),
    ("no_data", {"sufficiency": {"met": True, "reasons_not_met": []}}),
    ("no_data", {"effect": {"value": 0.0}}),
    ("no_data", {"effect": {"ci_low": -1.0, "ci_high": 1.0, "ci_level": 0.95}}),
])
def test_empty_comparisons_have_unmet_sufficiency_and_no_effect(status, change):
    req = compare_request(LOOSE)
    res = make_result("compare_windows", req, status, _empty_comparison(req, **change), provenance=PROV)
    with pytest.raises(ContractError):
        validate_result("compare_windows", req, res, POLICY)


def test_empty_comparison_is_valid_no_data():
    req = compare_request(LOOSE)
    validate_result("compare_windows", req,
                    make_result("compare_windows", req, "no_data", _empty_comparison(req), provenance=PROV),
                    POLICY)


def test_insufficient_comparison_cannot_carry_an_interval():
    req, res = fixture("compare_windows")
    m = copy.deepcopy(res["measurements"])
    m["effect"].update(ci_low=-10.0, ci_high=-9.0, ci_level=0.95)
    with pytest.raises(ContractError, match="no effect or interval"):
        validate_result("compare_windows", req,
                        make_result("compare_windows", req, "insufficient_evidence", m, provenance=PROV), POLICY)


# ---- 6. huge integers are controlled errors ----

@pytest.mark.parametrize("big", [10 ** 400, -(10 ** 400), 2 ** 53 + 1])
def test_huge_integers_are_contract_errors(big):
    with pytest.raises(ContractError):
        validate_request("query_events", query_request(max_episodes=big), POLICY)
    req = compare_request()
    req["sufficiency"]["min_usable_fraction"] = big
    with pytest.raises(ContractError):
        validate_request("compare_windows", req, POLICY)
    with pytest.raises(ContractError):
        ic.N()(big, "x")
    m = query_measurements()
    m["n_episodes_total"] = big
    with pytest.raises(ContractError):
        make_result("query_events", query_request(), "ok", m, provenance=PROV)


def test_largest_safe_integer_is_accepted_by_ensure_json():
    ic.ensure_json({"n": 2 ** 53, "m": -(2 ** 53)})


# ---- 7. validated evidence cannot be changed by the implementation ----

class KeepingTools(FakeTools):
    """Keeps a reference to every result it returns (a misbehaving backend)."""

    def __init__(self):
        super().__init__()
        self.returned = []

    def _answer(self, tool, request):
        res = super()._answer(tool, request)
        self.returned.append(res)
        return res


def test_returned_result_is_a_private_copy():
    tools = KeepingTools()
    res = call_tool(tools, "query_events", query_request(), POLICY)
    tools.returned[0]["measurements"]["episodes"][0]["start_frame"] = 0  # backend mutates later
    tools.returned[0]["status"] = "error"
    assert res["status"] == "ok" and res["measurements"]["episodes"][0]["start_frame"] == 90
    validate_result("query_events", query_request(), res, POLICY)


def test_uncopyable_results_are_violations():
    class Weird(FakeTools):
        def _answer(self, tool, request):
            res = super()._answer(tool, request)
            res = dict(res)
            res["measurements"] = {"x": (lambda: None)}  # not copyable as JSON evidence
            return res

    with pytest.raises(ToolContractViolation):
        call_tool(Weird(), "query_events", query_request(), POLICY)


def test_min_windows_applies_to_each_group_separately():
    """Enough baseline windows must not hide too few pre-event windows."""
    base2 = baseline_window_id(SRC, BEH, VAL_A, 150, 30)
    req = compare_request(SufficiencyCriteria(min_sessions=1, min_windows_per_group=2,
                                              min_usable_fraction=0.5),
                          baseline_window_ids=[BASE, base2])

    def two_baselines(m):
        m["per_session"][0]["baseline_windows"] = [wv(BASE, -3.0, 29), wv(base2, -3.0, 29)]
        m["n_windows_baseline"] = 2
    res = _ok_compare(req, two_baselines)
    with pytest.raises(ContractError, match=r"min_windows_per_group_pre"):
        validate_result("compare_windows", req, res, POLICY)


def test_used_window_without_usable_frames_is_refused_even_at_zero_coverage():
    req = compare_request(SufficiencyCriteria(min_sessions=1, min_windows_per_group=1,
                                              min_usable_fraction=0.0))
    res = _ok_compare(req, lambda m: m["per_session"][0]["pre_windows"][0].update(n_usable=0))
    with pytest.raises(ContractError, match="min_usable_fraction"):
        validate_result("compare_windows", req, res, POLICY)


# ---- Codex review 2: episode durations respect every minimum ----

def _query_case(end_frame, *, request_min=None, source=SRC, provenance_min=None):
    """A one-episode query result [90, end_frame), built with make_result so the
    evidence hash is recomputed for every modified fixture."""
    req = query_request(source=source, min_duration_frames=request_min)
    m = query_measurements()
    ep = m["episodes"][0]
    ep.update(episode_id=episode_id(source, BEH, VAL_A, 90), source=source,
              end_frame=end_frame, end_s=t(end_frame), annotated_frames=end_frame - 90)
    if source == "classifier_prediction":
        ep["provenance"] = {"model_id": "SYNTHETIC-model", "score_file_sha256": "b" * 64,
                            "threshold": 0.75, "min_duration_frames": provenance_min}
        del ep["annotated_frames"]  # predictions are not annotations
    res = make_result("query_events", req, "ok", m, provenance=PROV)
    assert res["evidence_id"] == ic.evidence_id("query_events", req, res)
    return req, res


@pytest.mark.parametrize("end_frame, request_min, ok", [
    (120, 60, False),   # 30 frames < 60
    (150, 60, True),    # exactly 60: boundary accepted
    (149, 60, False),   # one frame short
    (151, 60, True),
    (120, None, True),  # no minimum requested
])
def test_episode_duration_respects_the_request_minimum(end_frame, request_min, ok):
    req, res = _query_case(end_frame, request_min=request_min)
    if ok:
        validate_result("query_events", req, res, POLICY)
    else:
        with pytest.raises(ContractError, match="request.min_duration_frames"):
            validate_result("query_events", req, res, POLICY)


@pytest.mark.parametrize("end_frame, provenance_min, ok", [
    (120, 120, False),  # 30 frames < 120
    (120, 30, True),    # exactly 30: boundary accepted
    (119, 30, False),
    (210, 120, True),   # exactly 120
])
def test_classifier_episode_respects_its_provenance_minimum(end_frame, provenance_min, ok):
    req, res = _query_case(end_frame, source="classifier_prediction", provenance_min=provenance_min)
    if ok:
        validate_result("query_events", req, res, POLICY)
    else:
        with pytest.raises(ContractError, match="provenance.min_duration_frames"):
            validate_result("query_events", req, res, POLICY)


@pytest.mark.parametrize("end_frame, request_min, provenance_min, ok, which", [
    (150, 60, 90, False, "provenance"),   # 60 frames: meets the request (60), not provenance (90)
    (150, 90, 60, False, "request"),      # 60 frames: meets provenance (60), not the request (90)
    (180, 60, 90, True, None),            # 90 frames: meets the stricter bound exactly
    (180, 90, 60, True, None),
    (150, 60, 60, True, None),            # both bounds equal, exact
])
def test_the_stricter_of_both_minimums_decides(end_frame, request_min, provenance_min, ok, which):
    req, res = _query_case(end_frame, request_min=request_min, source="classifier_prediction",
                           provenance_min=provenance_min)
    if ok:
        validate_result("query_events", req, res, POLICY)
    else:
        with pytest.raises(ContractError, match=f"{which}.min_duration_frames"):
            validate_result("query_events", req, res, POLICY)


@pytest.mark.parametrize("source", ["validation_annotation", "monitor_event"])
def test_sources_without_a_provenance_minimum_only_use_the_request(source):
    """Annotation and monitor provenance declare no minimum; only the request's applies."""
    if source == "monitor_event":
        req = query_request(source=source, min_duration_frames=60)
        m = query_measurements()
        ep = m["episodes"][0]
        ep.update(episode_id=episode_id(source, BEH, VAL_A, 90), source=source,
                  provenance={"journal_sha256": "c" * 64, "run_id": "SYNTHETIC-run",
                              "rule_id": "R1", "event_id": "E-SYNTHETIC-1"})
        del ep["annotated_frames"]  # monitor events are not annotations
        res = make_result("query_events", req, "ok", m, provenance=PROV)
        with pytest.raises(ContractError, match="request.min_duration_frames"):
            validate_result("query_events", req, res, POLICY)
    else:
        req, res = _query_case(120, request_min=60)
        with pytest.raises(ContractError, match="request.min_duration_frames"):
            validate_result("query_events", req, res, POLICY)
    req, res = _query_case(120, request_min=30)
    validate_result("query_events", req, res, POLICY)


def test_duration_check_cannot_be_bypassed_by_rehashing():
    """Shrinking an accepted episode below the minimum and recomputing the
    evidence hash still fails."""
    req, res = _query_case(150, request_min=60)
    validate_result("query_events", req, res, POLICY)
    shrunk = copy.deepcopy(res)
    shrunk["measurements"]["episodes"][0].update(end_frame=120, end_s=t(120), annotated_frames=30)
    shrunk = rebuild("query_events", req, shrunk)
    with pytest.raises(ContractError, match="min_duration_frames"):
        validate_result("query_events", req, shrunk, POLICY)


# ======================================================================
# Antigravity audit regressions (SYNTHETIC fixtures; hashes rebuilt by make_result)
# ======================================================================

ALL_UNMET = ["min_sessions", "min_windows_per_group_pre", "min_windows_per_group_baseline"]


# ---- 1. reasons_not_met is exactly the recomputed list ----

@pytest.mark.parametrize("reasons", [
    ["min_sessions", "min_sessions", "min_windows_per_group_pre", "min_windows_per_group_baseline"],
    ["min_sessions"],                                                 # incomplete
    ["min_windows_per_group_pre", "min_sessions", "min_windows_per_group_baseline"],  # order
    ["the moon is full"],                                             # fabricated
    ALL_UNMET + ["min_usable_fraction"],                              # not a reason
])
def test_reasons_not_met_must_equal_the_recomputed_criteria(reasons):
    req = compare_request()
    m = compare_measurements(req, False)
    m["sufficiency"]["reasons_not_met"] = reasons
    res = make_result("compare_windows", req, "insufficient_evidence", m, provenance=PROV)
    with pytest.raises(ContractError, match="reasons_not_met"):
        validate_result("compare_windows", req, res, POLICY)


def test_exact_reasons_are_accepted():
    req = compare_request()
    m = compare_measurements(req, False)
    assert m["sufficiency"]["reasons_not_met"] == ALL_UNMET
    validate_result("compare_windows", req,
                    make_result("compare_windows", req, "insufficient_evidence", m, provenance=PROV), POLICY)
    assert ic.SUFFICIENCY_REASONS == tuple(ALL_UNMET)


# ---- 2. temporal rules for window features ----

def _with_clearance(clearance, tile_start, status="included", reasons=()):
    req = features_request(baseline={"strategy": "same_session_tiles", "length_frames": 30,
                                     "clearance_frames": clearance, "max_windows_per_session": 5})
    m = features_measurements()
    m["windows"][1] = window(baseline_window_id(SRC, BEH, VAL_A, tile_start, 30), "baseline",
                             tile_start, tile_start + 30, status=status, reasons=reasons)
    return req, make_result("get_window_features", req, "ok", m, provenance=PROV)


@pytest.mark.parametrize("clearance, tile_start, ok", [
    (0, 90, False),     # tile contains the onset
    (0, 120, True),     # after the onset, no clearance asked
    (60, 120, False),   # 30 frames after onset 90: within 60
    (60, 150, False),   # frame 150 is exactly 60 from the onset: within
    (60, 180, True),    # first frame 180 is 90 away
    (60, 0, True),      # last frame 29 is 61 before the onset
    (61, 0, False),     # ... and within 61
])
def test_baseline_clearance_around_requested_onsets(clearance, tile_start, ok):
    req, res = _with_clearance(clearance, tile_start)
    if ok:
        validate_result("get_window_features", req, res, POLICY)
    else:
        with pytest.raises(ContractError, match="clearance_frames|pre-event window"):
            validate_result("get_window_features", req, res, POLICY)


def test_excluded_baseline_near_an_onset_is_fine():
    req, res = _with_clearance(60, 120, status="excluded", reasons=["near_event"])
    validate_result("get_window_features", req, res, POLICY)


def test_included_baseline_cannot_overlap_an_excluded_pre_window():
    ep2 = episode_id(SRC, BEH, VAL_A, 300)
    req = features_request(episode_ids=[EP, ep2])
    m = features_measurements()
    m["windows"] += [window(pre_window_id(ep2, 30, 0), "pre_event", 270, 300, episode=ep2, truncated=True,
                            status="excluded", reasons=["overlaps_event"]),
                     window(baseline_window_id(SRC, BEH, VAL_A, 270, 30), "baseline", 270, 300)]
    res = make_result("get_window_features", req, "ok", m, provenance=PROV)
    with pytest.raises(ContractError, match="included or excluded"):
        validate_result("get_window_features", req, res, POLICY)


def test_included_pre_window_cannot_contain_another_requested_onset():
    ep3 = episode_id(SRC, BEH, VAL_A, 80)   # inside EP's pre window [60, 90)
    req = features_request(episode_ids=[EP, ep3], baseline=None)
    m = {"windows": [window(PRE, "pre_event", 60, 90, episode=EP, truncated=True),
                     window(pre_window_id(ep3, 30, 0), "pre_event", 50, 80, episode=ep3, truncated=True)]}
    res = make_result("get_window_features", req, "ok", m, provenance=PROV)
    with pytest.raises(ContractError, match="contains a requested onset"):
        validate_result("get_window_features", req, res, POLICY)
    m["windows"][0].update(status="excluded", exclusion_reasons=["overlaps_event"])
    validate_result("get_window_features", req,
                    make_result("get_window_features", req, "ok", m, provenance=PROV), POLICY)


@pytest.mark.parametrize("reason", ["near_event", "overlaps_pre_window"])
def test_pre_event_windows_cannot_use_baseline_only_reasons(reason):
    req, res = _features_with(lambda m: m["windows"][0].update(status="excluded", exclusion_reasons=[reason]))
    with pytest.raises(ContractError, match="not a pre-event exclusion reason"):
        validate_result("get_window_features", req, res, POLICY)


# ---- 3. constructors and parser agree ----

@pytest.mark.parametrize("length, gap", [(1, 0), (ic.MAX_WINDOW_FRAMES, 0), (30, ic.MAX_WINDOW_FRAMES)])
def test_every_constructible_pre_window_id_parses_back(length, gap):
    wid = pre_window_id(EP, length, gap)
    p = parse_window_id(wid)
    assert (p["length_frames"], p["gap_frames"]) == (length, gap)
    assert pre_window_id(p["episode_id"], p["length_frames"], p["gap_frames"]) == wid


@pytest.mark.parametrize("start, length", [(0, 1), (ic.MAX_FRAMES, ic.MAX_WINDOW_FRAMES)])
def test_every_constructible_baseline_id_parses_back(start, length):
    wid = baseline_window_id(SRC, BEH, VAL_A, start, length)
    p = parse_window_id(wid)
    assert (p["start_frame"], p["length_frames"]) == (start, length)


@pytest.mark.parametrize("call", [
    lambda: pre_window_id(EP, ic.MAX_WINDOW_FRAMES + 1, 0),
    lambda: pre_window_id(EP, 30, ic.MAX_WINDOW_FRAMES + 1),
    lambda: pre_window_id(EP, 10_000, 0),
    lambda: baseline_window_id(SRC, BEH, VAL_A, 0, ic.MAX_WINDOW_FRAMES + 1),
    lambda: baseline_window_id(SRC, BEH, VAL_A, ic.MAX_FRAMES + 1, 30),
    lambda: episode_id(SRC, BEH, VAL_A, ic.MAX_FRAMES + 1),
])
def test_constructors_refuse_what_the_parser_would_refuse(call):
    with pytest.raises(ContractError):
        call()


@pytest.mark.parametrize("wid", [
    f"win:pre:030:0:{EP}",                 # leading zero
    f"win:pre:30:00:{EP}",
    f"win:pre:0:0:{EP}",                   # length 0
    f"win:pre:901:0:{EP}",                 # above MAX_WINDOW_FRAMES
    f"win:pre:30:0:ep:{SRC}:{BEH}:{VAL_A}:090",
    f"win:base:30:{SRC}:{BEH}:{VAL_A}:00",
    f"win:base:30:{SRC}:{BEH}:{VAL_A}:99999999",   # above MAX_FRAMES
    f"win:base:1000:{SRC}:{BEH}:{VAL_A}:0",
])
def test_parser_refuses_non_canonical_or_out_of_range_ids(wid):
    with pytest.raises(ContractError):
        parse_window_id(wid)


def test_non_canonical_episode_ids_are_refused():
    for eid in (f"ep:{SRC}:{BEH}:{VAL_A}:090", f"ep:{SRC}:{BEH}:{VAL_A}:99999999"):
        with pytest.raises(ContractError):
            parse_episode_id(eid)


# ---- 4. exclusion reasons by window kind in compare_windows ----

def _compare_excluding(window_id, reason):
    req = compare_request(LOOSE)
    m = compare_measurements(req, False)
    row = m["per_session"][0]
    if window_id == BASE:
        row.update(baseline_windows=[], baseline_summary=None, difference=None)
        m.update(n_windows_baseline=0)
        unmet = ["min_sessions", "min_windows_per_group_baseline"]
    else:
        row.update(pre_windows=[], pre_summary=None, difference=None)
        m.update(n_windows_pre=0)
        unmet = ["min_sessions", "min_windows_per_group_pre"]
    m["excluded_windows"] = [{"window_id": window_id, "reason": reason, "n_frames": None, "n_usable": None}]
    m["n_sessions_paired"] = 0
    m["effect"].update(value=None, ci_low=None, ci_high=None, ci_level=None)
    m["sufficiency"].update(met=False, reasons_not_met=unmet)
    return req, make_result("compare_windows", req, "insufficient_evidence", m, provenance=PROV)


@pytest.mark.parametrize("window_id, reason, ok", [
    (BASE, "starts_before_session", False),
    (BASE, "near_event", True),
    (BASE, "overlaps_pre_window", True),
    (PRE, "near_event", False),
    (PRE, "overlaps_pre_window", False),
    (PRE, "starts_before_session", True),
    (PRE, "missing_data", True),
])
def test_compare_exclusion_reasons_must_fit_the_window_kind(window_id, reason, ok):
    req, res = _compare_excluding(window_id, reason)
    if ok:
        validate_result("compare_windows", req, res, POLICY)
    else:
        with pytest.raises(ContractError, match="impossible for a"):
            validate_result("compare_windows", req, res, POLICY)


# ---- 5. truncation is internally consistent ----

def _query_with(n_returned, total, truncated, max_episodes=10):
    req = query_request(max_episodes=max_episodes)
    m = query_measurements()
    base = m["episodes"][0]
    m["episodes"] = []
    for i in range(n_returned):
        e = copy.deepcopy(base)
        s = 90 + 40 * i
        e.update(start_frame=s, end_frame=s + 30, start_s=t(s), end_s=t(s + 30),
                 episode_id=episode_id(SRC, BEH, VAL_A, s))
        m["episodes"].append(e)
    m.update(n_episodes_total=total, truncated=truncated)
    return req, make_result("query_events", req, "ok", m, provenance=PROV)


@pytest.mark.parametrize("n, total, truncated, ok", [
    (1, 5, True, False),     # audit: truncated but only 1 of max 10 returned
    (1, 5, False, False),
    (10, 15, True, True),    # a full page, more exist
    (10, 10, False, True),
    (10, 10, True, False),
    (3, 3, False, True),
    (3, 2, False, False),    # more returned than exist
])
def test_truncation_must_be_internally_consistent(n, total, truncated, ok):
    req, res = _query_with(n, total, truncated)
    if ok:
        validate_result("query_events", req, res, POLICY)
    else:
        with pytest.raises(ContractError):
            validate_result("query_events", req, res, POLICY)


# ---- 6. no overlapping bouts within a session ----

def _two_episodes(second_session, second_start, second_end):
    req = query_request(sessions=sorted({VAL_A, second_session}))
    m = query_measurements()
    e2 = copy.deepcopy(m["episodes"][0])
    e2.update(session=second_session, start_frame=second_start, end_frame=second_end,
              start_s=t(second_start), end_s=t(second_end), annotated_frames=second_end - second_start,
              episode_id=episode_id(SRC, BEH, second_session, second_start))
    m["episodes"].append(e2)
    m.update(n_episodes_total=2, sessions_searched=sorted({VAL_A, second_session}))
    return req, make_result("query_events", req, "ok", m, provenance=PROV)


@pytest.mark.parametrize("session, start, end, ok", [
    (VAL_A, 100, 130, False),   # overlaps [90, 120)
    (VAL_A, 80, 95, False),     # overlaps from before
    (VAL_A, 119, 140, False),   # one frame of overlap
    (VAL_A, 120, 150, True),    # adjacent: half-open spans do not overlap
    (VAL_B, 90, 120, True),     # same frames, another session
])
def test_overlapping_episodes_in_one_session_are_refused(session, start, end, ok):
    req, res = _two_episodes(session, start, end)
    if ok:
        validate_result("query_events", req, res, POLICY)
    else:
        with pytest.raises(ContractError, match="overlap"):
            validate_result("query_events", req, res, POLICY)


# ---- 7. sessions_without_episodes vs searched sessions and episodes ----

def _sessions_case(searched, without, truncated=False, requested="same"):
    req = query_request(sessions=sorted(searched) if requested == "same" else requested,
                        max_episodes=1)
    m = query_measurements()
    m.update(sessions_searched=list(searched), sessions_without_episodes=list(without),
             n_episodes_total=2 if truncated else 1, truncated=truncated)
    return req, make_result("query_events", req, "ok", m, provenance=PROV)


@pytest.mark.parametrize("searched, without, truncated, ok", [
    ([VAL_A, VAL_B], [VAL_B], False, True),           # complete and consistent
    ([VAL_A, VAL_B], [], False, False),               # audit: B searched, no episode, not listed
    ([VAL_A, VAL_B], [VAL_A, VAL_B], False, False),   # A has an episode
    ([VAL_A, VAL_B], [], True, True),                 # truncated: B may have episodes beyond the page
    ([VAL_A, VAL_B], [VAL_B], True, True),
    ([VAL_A, VAL_B], [VAL_A], True, False),           # A has a returned episode
    ([VAL_A, VAL_A], [], False, False),               # duplicate searched session
])
def test_sessions_without_episodes_must_be_consistent(searched, without, truncated, ok):
    req, res = _sessions_case(searched, without, truncated)
    if ok:
        validate_result("query_events", req, res, POLICY)
    else:
        with pytest.raises(ContractError):
            validate_result("query_events", req, res, POLICY)


def test_requested_sessions_must_all_be_searched():
    req, res = _sessions_case([VAL_A], [], requested=[VAL_A, VAL_B])
    with pytest.raises(ContractError, match="exactly the requested sessions"):
        validate_result("query_events", req, res, POLICY)
    req, res = _sessions_case([VAL_A], [], requested=None)   # null: the source decides
    validate_result("query_events", req, res, POLICY)


@pytest.mark.parametrize("searched, without", [([VAL_A, VAL_A], []), ([VAL_A, VAL_B], [VAL_B, VAL_B])])
def test_duplicate_sessions_in_results_are_refused(searched, without):
    """The request leaves sessions to the source (null), so only the result rule applies."""
    req, res = _sessions_case(searched, without, requested=None)
    with pytest.raises(ContractError, match="sessions must be unique"):
        validate_result("query_events", req, res, POLICY)


# ======================================================================
# annotated_frames on episodes (approved contract extension)
# ======================================================================

def _annotated_case(value, *, merge=None, end_frame=120, source=SRC, drop=False):
    req = query_request(merge_gap_frames=merge, source=source)
    m = query_measurements()
    ep = m["episodes"][0]
    ep.update(end_frame=end_frame, end_s=t(end_frame), annotated_frames=value)
    if source != SRC:
        ep.update(source=source, episode_id=episode_id(source, BEH, VAL_A, 90),
                  provenance={"model_id": "SYNTHETIC-model", "score_file_sha256": "b" * 64,
                              "threshold": 0.75, "min_duration_frames": 1}
                  if source == "classifier_prediction" else
                  {"journal_sha256": "c" * 64, "run_id": "SYNTHETIC-run", "rule_id": "R1", "event_id": "E1"})
    if drop:
        del ep["annotated_frames"]
    return req, m


def _check(req, m):
    validate_result("query_events", req, make_result("query_events", req, "ok", m, provenance=PROV), POLICY)


@pytest.mark.parametrize("value, merge, ok", [
    (30, None, True),          # unmerged: equals the 30-frame span
    (29, None, False),         # unmerged but partial
    (12, 5, True),             # merged: fewer annotated frames than the span
    (30, 5, True),             # merging asked, nothing merged
    (31, 5, False),            # more than the span
    (12, 0, False),            # a 0-frame gap never merges maximal runs
])
def test_annotated_frames_against_the_span(value, merge, ok):
    req, m = _annotated_case(value, merge=merge)
    if ok:
        _check(req, m)
    else:
        with pytest.raises(ContractError, match="annotated_frames"):
            _check(req, m)


@pytest.mark.parametrize("value", [0, -1, 1.5, 30.0, True, "30", None])
def test_malformed_annotated_frames_are_refused(value):
    req, m = _annotated_case(value)
    with pytest.raises(ContractError):
        _check(req, m)


def test_annotation_episodes_require_annotated_frames():
    req, m = _annotated_case(30, drop=True)
    with pytest.raises(ContractError, match="required for annotation episodes"):
        _check(req, m)


@pytest.mark.parametrize("source", ["classifier_prediction", "monitor_event"])
def test_predictions_and_monitor_events_carry_no_annotation_count(source):
    req, m = _annotated_case(30, source=source)
    with pytest.raises(ContractError, match="no annotations"):
        _check(req, m)
    req, m = _annotated_case(30, source=source, drop=True)
    _check(req, m)


def test_annotated_frames_are_bound_by_the_evidence_hash():
    req, m = _annotated_case(12, merge=5)
    res = make_result("query_events", req, "ok", m, provenance=PROV)
    validate_result("query_events", req, res, POLICY)
    tampered = copy.deepcopy(res)
    tampered["measurements"]["episodes"][0]["annotated_frames"] = 13   # plausible but altered
    with pytest.raises(ContractError, match="evidence_id"):
        validate_result("query_events", req, tampered, POLICY)


def test_merged_episode_cannot_have_zero_annotated_frames():
    """Only the positive-integer rule catches this: 0 is below a merged span."""
    req, m = _annotated_case(0, merge=5)
    with pytest.raises(ContractError, match="annotated_frames"):
        _check(req, m)
