"""Presentation helpers of the Investigation page (investigation_view.py). New file.

Every input is a real record: produced by replay_investigation on the Replay
page's SYNTHETIC sessions, or by the Contract v1 tools on the synthetic
multi-session fixture of test_investigation_tools_v1. No value is made up."""

import json

import pytest

import live_tracker.investigation_tools_v1 as v1
import live_tracker.investigation_view as iv
import live_tracker.replay_investigation as ri
import live_tracker.replay_view as rv
from live_tracker.rules import RuleSet
from test_investigation_tools_v1 import cmp_req, make

DEMO, TRACK = rv.synthetic_catalog()


def rules(*specs):
    rs = RuleSet()
    for s in specs:
        rs.add(s)
        rs.approve(s["rule_id"])
    return rs.active()


ATTACK = {"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0}


@pytest.fixture(scope="module")
def track():
    (item,) = ri.investigate_replay(TRACK, rules(ATTACK), "run1")["rules"]
    return item


def test_trace_rows_follow_the_record_in_order(track):
    rows = iv.trace_rows(track["record"])
    assert [(r["step"], r["tool"], r["outcome"]) for r in rows] == [
        (1, "query_events", "executed: ok"), (2, "compare_windows", "executed: insufficient_evidence")]
    assert all(r["counted toward budget"] for r in rows)
    assert rows[1]["arguments"].startswith("baseline_gap_s: 1.0") and "bouts: 3 (the cited query's" in \
        rows[1]["arguments"]
    assert rows[0]["result sha256"] == track["record"]["attempts"][0]["result_sha256"][:12]


def test_refusals_budget_and_failures_are_labelled():
    data = make(sessions=["SYNTHETIC_v1_0"], n=600, starts=())
    run = v1.InvestigationRunner(data, "x", max_tool_calls=1)
    run.call("query_events", {"session": "SYNTHETIC_v1_0", "behavior": [], "min_duration_s": 1.0})
    run.call("query_events", {"session": "SYNTHETIC_v1_0", "behavior": "attack", "min_duration_s": 1.0})
    outcomes = [r["outcome"] for r in iv.trace_rows(run.record())]
    assert outcomes == ["refused: invalid_request", "refused: budget exhausted"]
    assert iv._outcome({"status": "exception", "code": None}) == "failed (programming error)"


def test_evidence_status_for_every_kind(track):
    assert iv.evidence_status(track)[0] == "insufficient"
    assert "No effect value is reported" in iv.evidence_status(track)[1]
    (zero,) = ri.investigate_replay(DEMO, rules({**ATTACK, "ask_threshold": 0.0}), "r")["rules"]
    assert iv.evidence_status(zero) == ("unsupported", ri.UNSUPPORTED_ZERO)
    (none,) = ri.investigate_replay(DEMO, rules({**ATTACK, "min_duration_s": 60.0}), "r")["rules"]
    assert iv.evidence_status(none)[0] == "no_bouts"
    # a comparison that meets the gate: five synthetic sessions (real tool output)
    data = make()
    q = v1.query_events(data, {"session": list(data.sessions), "behavior": "attack", "min_duration_s": 1.0})
    ok = {"unsupported": None, "query": q, "compare": v1.compare_windows(data, cmp_req(q))}
    kind, text = iv.evidence_status(ok)
    assert kind == "supported" and "Association only" in text and f"{ok['compare']['mean_diff']:.4g}" in text
    bad = {"unsupported": None, "query": q, "compare": v1.compare_windows(data, {**cmp_req(q), "pre_s": 0.01})}
    assert iv.evidence_status(bad)[0] == "error"


def test_bout_windows_are_the_windows_compare_windows_uses(track):
    c = track["compare"]
    for i, b in enumerate(track["query"]["bouts"]):
        S, p0, b0, b1, before = v1._bout_windows(i, b, c["pre_s"], c["baseline_s"], c["baseline_gap_s"])
        pre, base = iv.bout_windows(b, c)
        assert pre["frames"] == ("starts before 0 s" if b["start_s"] - c["pre_s"] < 0 else f"[{p0}, {S})")
        assert base["frames"] == ("starts before 0 s" if before else f"[{b0}, {b1})")


def test_bout_outcome_reads_the_recorded_exclusions(track):
    c = track["compare"]
    outcomes = {b["start_s"]: iv.bout_outcome(b, c) for b in track["query"]["bouts"]}
    assert outcomes == {e["start_s"]: e["why"] for e in c["excluded"]} | \
        {b["start_s"]: "used" for b in track["query"]["bouts"]
         if b["start_s"] not in {e["start_s"] for e in c["excluded"]}}
    assert outcomes[17.0] == "baseline_window_unreliable"            # the stuck tracker inside the baseline
    assert iv.bout_outcome(track["query"]["bouts"][0], None) is None


def test_bout_tracking_lists_overlapping_warnings(track):
    warnings = rv.entry_warnings(TRACK)
    late = next(b for b in track["query"]["bouts"] if b["start_s"] == 17.0)
    rows = iv.bout_tracking(late, warnings, iv.bout_windows(late, track["compare"]))
    assert any(r["warning"] == "frozen_tracker" and "baseline" in r["overlaps"] for r in rows)
    ask = next(b for b in track["query"]["bouts"] if b["start_s"] == 6.0)     # 6-9 s; low confidence 8-9.5 s
    assert [r["warning"] for r in iv.bout_tracking(ask, warnings) if "bout" in r["overlaps"]] == ["low_confidence"]
    first = track["query"]["bouts"][0]             # 1.00-3.50 s; missing coordinates from 3.50 s: half-open, no overlap
    assert iv.bout_tracking(first, warnings) == []


def test_timeline_rows_hold_only_recorded_intervals(track):
    warnings = rv.entry_warnings(TRACK)
    sel = track["query"]["bouts"][-1]
    rows = iv.timeline_rows(track, warnings, sel)
    bouts = [r for r in rows if r["kind"] == "bout"]
    assert [(r["start"], r["end"]) for r in bouts] == [(b["start_s"], b["end_s"]) for b in track["query"]["bouts"]]
    assert sum(r["selected"] for r in bouts) == 1
    assert {r["kind"] for r in rows} == {"bout", "pre-event", "baseline", "tracking warning"}
    assert len([r for r in rows if r["kind"] == "tracking warning"]) == len(warnings)


def test_export_is_the_replay_pages_bytes():
    res = ri.investigate_replay(DEMO, rules(ATTACK), "run1")
    assert iv.export_json(res) == json.dumps(res, indent=2, sort_keys=True)


def test_timeline_never_draws_outside_the_session(track):
    warnings = rv.entry_warnings(TRACK)
    first = track["query"]["bouts"][0]                 # 1.00 s: pre [-1, 1) is clipped, baseline [-7, -2) is dropped
    rows = iv.timeline_rows(track, warnings, first)
    assert all(0 <= r["start"] < r["end"] for r in rows)
    kinds = {r["kind"]: (r["start"], r["end"]) for r in rows if r["kind"] in ("pre-event", "baseline")}
    assert kinds == {"pre-event": (0.0, 1.0)}
