"""Adversarial and malformed-input regressions for the Contract v1 adapter
(investigation_tools_v1) and timebase. New file. SYNTHETIC data only.

1. compare_windows accepts only the exact bouts of a cited, authentic query on
   the same data source; fabricated, altered, stale or cross-source bouts are
   refused (unverified_bouts), whatever event ids they carry.
2. Malformed values (wrong types, NaN, infinity, integers too large for a
   float) get a structured error, never an exception, and every attempt is
   recorded and counted against the 12-call budget.
"""

import copy

import numpy as np
import pytest

from live_tracker import investigation_tools_v1 as v1
from live_tracker import timebase as tb
from live_tracker.fake_scores import Segment
from test_investigation_tools_v1 import SESSIONS, cmp_req, make

HUGE = 10 ** 400                                   # float(HUGE) raises OverflowError
ALLOWED_CODES = set(v1.ERROR_CODES) | set(v1.EXTENSION_ERROR_CODES)


@pytest.fixture(scope="module")
def data():
    return make()


@pytest.fixture(scope="module")
def query(data):
    return v1.query_events(data, {"session": list(data.sessions), "behavior": "attack", "min_duration_s": 1.0})


def refused(data, req, code="unverified_bouts"):
    r = v1.compare_windows(data, req)
    assert r["status"] == "error" and r["code"] == code, r
    assert "mean_diff" not in r and "n_bouts" not in r
    return r


# ======================================================================
# 1. bout provenance
# ======================================================================

def test_untampered_query_results_still_compare(data, query):
    r = v1.compare_windows(data, cmp_req(query))
    assert r["status"] == "ok" and r["n_bouts"] == 15


def test_codex_reproduction_attack_bouts_relabelled_as_mount_are_refused(data, query):
    fake = [{**b, "behavior": "mount"} for b in query["bouts"]]
    refused(data, {**cmp_req(query), "bouts": fake})                                       # bouts changed
    refused(data, {**cmp_req(query), "bouts": fake, "query": {**query["query"], "behavior": "mount"}})  # + query
    # even a genuine mount query id does not vouch for fabricated mount bouts: the query is re-run
    mount = v1.query_events(data, {"session": list(data.sessions), "behavior": "mount", "min_duration_s": 1.0})
    assert mount["status"] == "ok" and mount["n_bouts"] == 0
    refused(data, {**cmp_req(mount), "bouts": fake})


@pytest.mark.parametrize("field, change", [
    ("start_s", lambda v: v - 1.0), ("end_s", lambda v: v + 0.5), ("fired_s", lambda v: v + 0.7),
    ("event_id", lambda v: v[:-1] + "9"), ("session", lambda v: SESSIONS[(SESSIONS.index(v) + 1) % 5]),
    ("behavior", lambda v: "investigation"), ("confidence", lambda v: v + 0.01),
    ("end_reason", lambda v: "stream_end"), ("bout_mean_score", lambda v: 0.99)])
def test_any_altered_bout_field_is_refused(data, query, field, change):
    bouts = copy.deepcopy(query["bouts"])
    bouts[4][field] = change(bouts[4][field])
    r = refused(data, {**cmp_req(query), "bouts": bouts})
    assert "bouts[4]" in r["message"]


@pytest.mark.parametrize("edit", [
    lambda b: b[:-1], lambda b: b[1:], lambda b: b + [b[0]], lambda b: list(reversed(b)),
    lambda b: b + [{**b[0], "event_id": "E-forged-99", "start_s": 50.0, "fired_s": 51.0, "end_s": 52.0}]])
def test_subsets_duplicates_reordering_and_additions_are_refused(data, query, edit):
    refused(data, {**cmp_req(query), "bouts": edit(copy.deepcopy(query["bouts"]))})


@pytest.mark.parametrize("change", [{"threshold": 0.6}, {"min_duration_s": 1.5}, {"session": SESSIONS[:4]},
                                    {"behavior": "investigation"}])
def test_altered_query_parameters_are_refused(data, query, change):
    refused(data, {**cmp_req(query), "query": {**query["query"], **change}})


def test_forged_or_foreign_query_ids_are_refused(data, query):
    refused(data, {**cmp_req(query), "query_run_id": "inv_" + "0" * 16})
    other = v1.query_events(data, {"session": SESSIONS[:2], "behavior": "attack", "min_duration_s": 1.0})
    refused(data, {**cmp_req(query), "query_run_id": other["query_run_id"]})


def test_bouts_from_another_or_changed_source_are_refused(data, query):
    changed = make(extra_segments={SESSIONS[0]: [Segment("attack", 50.0, 53.0, 0.85, 0.95)]})
    assert changed.fingerprint != data.fingerprint
    refused(changed, cmp_req(query))                                  # stale: the scores changed since the query
    other = make(sessions=SESSIONS)                                   # identical scores, rebuilt: same identity
    assert other.fingerprint == data.fingerprint
    assert v1.compare_windows(other, cmp_req(query))["status"] == "ok"


def test_the_overlap_rule_uses_the_authentic_bouts(data, query):
    r = v1.compare_windows(data, cmp_req(query))
    assert r["n_input"] == len(query["bouts"])                        # nothing dropped or invented


# ======================================================================
# 2. malformed values: structured errors, every attempt counted
# ======================================================================

def bad_requests(query):
    w = {"session": SESSIONS[0], "t0_s": 0.0, "t1_s": 1.0}
    q = {"session": SESSIONS[0], "behavior": "attack", "min_duration_s": 1.0}
    c = cmp_req(query)
    return [
        ("query_events", {**q, "behavior": []}, "invalid_request"),
        ("query_events", {**q, "behavior": {}}, "invalid_request"),
        ("query_events", {**q, "behavior": 5}, "invalid_request"),
        ("query_events", {**q, "behavior": "fighting"}, "unknown_behavior"),
        ("query_events", {**q, "min_duration_s": HUGE}, "invalid_request"),
        ("query_events", {**q, "min_duration_s": float("nan")}, "invalid_request"),
        ("query_events", {**q, "min_duration_s": float("inf")}, "invalid_request"),
        ("query_events", {**q, "min_duration_s": "1"}, "invalid_request"),
        ("query_events", {**q, "min_duration_s": [1.0]}, "invalid_request"),
        ("query_events", {**q, "threshold": HUGE}, "invalid_request"),
        ("query_events", {**q, "session": [[]]}, "invalid_request"),
        ("query_events", {**q, "session": {}}, "invalid_request"),
        ("query_events", {**q, "session": [1, 2]}, "invalid_request"),
        ("get_window_features", {**w, "t0_s": HUGE, "cols": ["dist_center"]}, "bad_window"),
        ("get_window_features", {**w, "t1_s": HUGE, "cols": ["dist_center"]}, "bad_window"),
        ("get_window_features", {**w, "t0_s": "0", "cols": ["dist_center"]}, "bad_window"),
        ("get_window_features", {**w, "t1_s": float("nan"), "cols": ["dist_center"]}, "bad_window"),
        ("get_window_features", {**w, "cols": {"dist_center": 1}}, "invalid_request"),
        ("get_window_features", {**w, "cols": [[]]}, "invalid_request"),
        ("get_window_features", {**w, "cols": [{}]}, "invalid_request"),
        ("get_window_features", {**w, "cols": "dist_center"}, "invalid_request"),
        ("get_window_features", {**w, "session": {}, "cols": ["dist_center"]}, "invalid_request"),
        ("check_tracking_quality", {**w, "quality_policy": []}, "invalid_request"),
        ("check_tracking_quality", {**w, "quality_policy": {}}, "invalid_request"),
        ("check_tracking_quality", {**w, "t1_s": float("-inf")}, "bad_window"),
        ("check_tracking_quality", {**w, "t0_s": 5.0, "t1_s": 2.0}, "bad_window"),
        ("compare_windows", {**c, "pre_s": HUGE}, "bad_window"),
        ("compare_windows", {**c, "pre_s": []}, "bad_window"),
        ("compare_windows", {**c, "baseline_s": HUGE}, "bad_window"),
        ("compare_windows", {**c, "baseline_gap_s": "1"}, "bad_window"),
        ("compare_windows", {**c, "feature": []}, "invalid_request"),
        ("compare_windows", {**c, "min_n_windows": HUGE}, "invalid_request"),
        ("compare_windows", {**c, "quality_policy": {}}, "invalid_request"),
        ("compare_windows", {**c, "query": []}, "invalid_request"),
        ("compare_windows", {**c, "query_run_id": 5}, "invalid_request"),
        ("compare_windows", {**c, "bouts": "all"}, "invalid_request"),
        ("query_events", ["not", "a", "dict"], "invalid_request"),
        (["not a tool"], {}, "invalid_request"),
    ]


def test_every_malformed_call_is_a_structured_error_and_is_counted(data, query):
    cases = bad_requests(query)
    for i, (tool, req, code) in enumerate(cases):
        run = v1.InvestigationRunner(data, f"inv-bad-{i}")
        r = run.call(tool, req)                                     # never raises
        assert r["status"] == "error" and r["code"] == code, (tool, req, r)
        assert code in ALLOWED_CODES and r["association_only"] is True
        rec = run.record()
        assert rec["calls_used"] == 1 and rec["attempts"][0]["status"] == "error"
        if callable(getattr(v1, str(tool), None)) and str(tool) in v1.TOOLS:
            assert getattr(v1, tool)(data, copy.deepcopy(req)) == r   # the bare tool answers the same


def test_repeated_invalid_calls_exhaust_the_budget(data):
    run = v1.InvestigationRunner(data, "inv-bad-repeat")
    for _ in range(12):
        assert run.call("query_events", {"session": SESSIONS[0], "behavior": [], "min_duration_s": HUGE})["code"] \
            == "invalid_request"
    r = run.call("query_events", {"session": SESSIONS[0], "behavior": "attack", "min_duration_s": 1.0})
    assert r["code"] == "budget_exhausted"                          # a valid call can no longer run
    rec = run.record()
    assert rec["calls_used"] == 12 and len(rec["attempts"]) == 13 and not rec["attempts"][-1]["counted"]


def test_a_programming_error_is_recorded_counted_and_not_hidden(data, monkeypatch):
    def broken(d, r):
        raise RuntimeError("bug")
    monkeypatch.setitem(v1._TOOL_FUNCS, "check_tracking_quality", broken)
    run = v1.InvestigationRunner(data, "inv-bug")
    with pytest.raises(RuntimeError, match="bug"):
        run.call("check_tracking_quality", {"session": SESSIONS[0], "t0_s": 0.0, "t1_s": 1.0})
    (a,) = run.record()["attempts"]
    assert a["status"] == "exception" and a["reason"] == "RuntimeError" and a["counted"] and run.calls_used == 1


@pytest.mark.parametrize("value", [HUGE, -HUGE, float("nan"), float("inf"), [], {}, "1", None, True])
def test_timebase_rejects_values_that_are_not_finite_seconds(value):
    with pytest.raises(tb.WindowError) as err:
        tb.boundary_frame(value)
    assert err.value.code == "bad_window"
    with pytest.raises(tb.WindowError):
        tb.window_frames(0.0, value)
    with pytest.raises(tb.WindowError):
        tb.duration_frames(value)


@pytest.mark.parametrize("first, stop", [(HUGE, HUGE + 1), ([], 3), (0, []), (0, HUGE), (5, 5)])
def test_frame_conversion_rejects_bad_frames(first, stop):
    with pytest.raises(tb.WindowError):
        tb.window_s(first, stop)


def test_huge_but_finite_boundaries_are_outside_any_session():
    assert tb.boundary_frame(1e300) > tb.MAX_FRAME
    with pytest.raises(tb.WindowError, match="past the session end"):
        tb.window_frames(0.0, 1e300, n_frames=1800)
    assert np.isfinite(tb.frame_time_s(tb.MAX_FRAME))


# ======================================================================
# durations that round to zero frames
# ======================================================================

@pytest.mark.parametrize("tiny", [1e-12, 5e-10, 1e-9, 5e-324])
def test_a_positive_duration_below_one_frame_is_bad_window(tiny):
    with pytest.raises(tb.WindowError, match="shorter than one frame") as err:
        tb.duration_frames(tiny)                     # under half a frame: rounds to 0 frames
    assert err.value.code == "bad_window"


def test_one_frame_and_zero_durations():
    assert tb.duration_frames(1 / 30) == 1 and tb.duration_frames(1 / 30 + 5e-10) == 1
    assert tb.duration_frames(0.0, allow_zero=True) == 0
    assert tb.duration_frames(1e-12, allow_zero=True) == 0           # rounds to a zero gap
    with pytest.raises(tb.WindowError, match="> 0"):
        tb.duration_frames(0.0)


@pytest.mark.parametrize("field", ["baseline_s", "pre_s"])
@pytest.mark.parametrize("tiny", [1e-12, 1e-10])
def test_compare_windows_with_a_sub_frame_duration_is_a_structured_error(data, query, field, tiny):
    r = v1.compare_windows(data, {**cmp_req(query), field: tiny})    # used to raise ZeroDivisionError
    assert r["status"] == "error" and r["code"] == "bad_window" and "empty frame range" in r["message"]
    run = v1.InvestigationRunner(data, "inv-tiny")
    if field == "pre_s":
        assert run.call("compare_windows", {**cmp_req(query), field: tiny})["code"] == "invalid_request"  # grid
    else:
        assert run.call("compare_windows", {**cmp_req(query), field: tiny})["code"] == "bad_window"
    assert run.calls_used == 1


def test_compare_windows_accepts_one_frame_baseline_and_zero_gap(data, query):
    r = v1.compare_windows(data, {**cmp_req(query), "baseline_s": 1 / 30, "baseline_gap_s": 0.0})
    assert r["status"] in ("ok", "insufficient_evidence")
    r = v1.compare_windows(data, {**cmp_req(query), "baseline_gap_s": 1e-12})       # rounds to a zero gap
    assert r["status"] in ("ok", "insufficient_evidence")
