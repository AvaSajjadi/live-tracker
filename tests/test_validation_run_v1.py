"""Tests for the Phase 2 validation run (scripts/validation_run_v1.py) on SYNTHETIC inputs only: the Stage 9 synthetic
pose cache (test_investigation_windows.build) and an issued synthetic score drop (test_prediction_windows). The real
archive and cache are never read here. New file."""

import copy
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

import live_tracker.investigation_tools_v1 as v1
import live_tracker.score_drops as sd
import test_investigation_tools_v1 as T
from test_investigation_windows import VAL_A, VAL_B, build
from test_prediction_windows import score_table
from test_score_drops import TEST_MEMBER, VAL_MEMBER, archive, csv_bytes

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validation_run_v1", ROOT / "scripts" / "validation_run_v1.py")
vr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vr)


@pytest.fixture
def inputs(tmp_path):
    _, _, cache, manifest = build(tmp_path)
    zip_path = archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table()), TEST_MEMBER: csv_bytes(score_table())})
    return {"zip": zip_path, "sha": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "cache": cache,
            "manifest": manifest, "tmp": tmp_path}


def argv(i, out, *extra, confirm=True, synthetic=True, sha=None):
    a = [str(i["zip"]), "--archive-sha256", sha or i["sha"], "--out", str(out), "--cache-dir", str(i["cache"]),
         "--manifest", str(i["manifest"])]
    a += ["--confirm", vr.RULE_ID] if confirm else []
    a += ["--allow-synthetic"] if synthetic else []
    return a + list(extra)


def summary(out):
    return json.loads((out / "run_summary.json").read_text(encoding="utf-8"))


def test_a_synthetic_run_completes_and_is_labelled_synthetic_everywhere(inputs, capsys):
    out = inputs["tmp"] / "run1"
    assert vr.main(argv(inputs, out)) == 0
    capsys.readouterr()
    s = summary(out)
    assert s["status"] == "verified" and all(x["status"] == "ok" for x in s["steps"])
    assert [x["step"] for x in s["steps"]] == ["scores", "pose_cache", "approval", "monitor", "labels", "evaluation",
                                                "contract_v1"]
    assert set(s["data_labels"].values()) == {vr.SYNTHETIC_LABELS["scores"]} and "REAL" not in json.dumps(s)
    assert s["scores"]["synthetic"] is True and s["contract_v1"]["data_source"] == "synthetic"
    assert s["rule"] == {"rule_id": "ATTACK_RULE_V1", "behavior": "attack", "ask_threshold": 0.6,
                         "min_duration_s": 0.5, "alert_threshold": 0.8, "alert_mode": "alert",
                         "source": "evaluate.ATTACK_RULE_V1",
                         "approval": "approved for this run by the operator (--confirm ATTACK_RULE_V1)"}
    m = s["monitor"]
    assert m["provenance_status"] == "verified_score_drop" and m["completion"] == "complete"
    assert m["n_events"] == len(m["events"]) == sum(m["events_per_session"].values())
    c = s["contract_v1"]
    assert c["query_matches_monitor_bouts"] is True and c["query_events"]["threshold"] == 0.6
    assert c["query_events"]["min_duration_s"] == 0.5 and c["calls_used"] <= c["max_tool_calls"] == 12
    assert [x["pre_s"] for x in c["compare_windows"]] == [2.0, 1.0, 5.0]
    for x in c["compare_windows"]:                    # 3 synthetic sessions cannot meet the 4-session gate
        assert x["status"] == "insufficient_evidence" and "mean_diff" not in x and x["overlap_policy"] == "overlap_v1"
        assert x["evidence_gate"]["overridden"] is False
    assert sorted(p.name for p in out.iterdir()) == sorted(vr.OUTPUT_NAMES)
    for name, digest in s["outputs"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_the_run_is_reproducible(inputs, capsys):
    a, b = inputs["tmp"] / "a", inputs["tmp"] / "b"
    assert vr.main(argv(inputs, a)) == 0 and vr.main(argv(inputs, b)) == 0
    capsys.readouterr()
    for name in vr.OUTPUT_NAMES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_synthetic_sessions_are_refused_without_the_test_flag(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert vr.main(argv(inputs, out, synthetic=False)) == 2
    s = summary(out)
    assert s["status"] == "refused" and s["steps"] == [{**s["steps"][0], "step": "scores", "status": "failed"}]
    assert "SYNTHETIC" in s["steps"][0]["message"] and "data_labels" not in s


def test_without_explicit_approval_nothing_is_monitored(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert vr.main(argv(inputs, out, confirm=False)) == 2
    s = summary(out)
    assert s["status"] == "refused" and s["steps"][-1]["step"] == "approval"
    assert "--confirm" in s["steps"][-1]["message"]
    assert not (out / "events.journal.jsonl").exists() and "monitor" not in s


def test_a_wrong_archive_digest_is_refused_before_any_score_is_used(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert vr.main(argv(inputs, out, sha="0" * 64)) == 2
    s = summary(out)
    assert s["steps"] == [{**s["steps"][0], "step": "scores", "status": "failed"}] and "scores" not in s


def test_a_missing_pose_cache_is_refused_with_no_substitute(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert vr.main(argv(inputs, out, "--cache-dir", str(inputs["tmp"] / "no_cache"))) == 2
    assert "no verified validation pose cache" in capsys.readouterr().err and not out.exists()


def test_an_incomplete_pose_cache_is_refused(tmp_path, capsys):
    _, _, cache, manifest = build(tmp_path, sessions=(VAL_A, VAL_B))        # one validation session missing
    zip_path = archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table())})
    i = {"zip": zip_path, "sha": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "cache": cache,
         "manifest": manifest}
    out = tmp_path / "run"
    assert vr.main(argv(i, out)) == 2
    s = summary(out)
    assert s["steps"][-1]["step"] == "pose_cache" and "incomplete" in s["steps"][-1]["message"]
    assert not (out / "events.journal.jsonl").exists()


def test_scores_and_pose_with_different_frame_counts_are_refused(tmp_path, capsys):
    _, _, cache, manifest = build(tmp_path)
    zip_path = archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table(n=399))})      # the cache has 400 frames
    i = {"zip": zip_path, "sha": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "cache": cache,
         "manifest": manifest}
    out = tmp_path / "run"
    assert vr.main(argv(i, out)) == 2
    s = summary(out)
    assert s["steps"][-1]["step"] == "pose_cache" and "disagree" in s["steps"][-1]["message"]


def test_the_test_member_is_never_read(inputs, capsys, monkeypatch):
    read = []
    real = sd._read_member
    monkeypatch.setattr(sd, "_read_member", lambda z, info, limit: read.append(info.filename) or real(z, info, limit))
    assert vr.main(argv(inputs, inputs["tmp"] / "run")) == 0
    assert TEST_MEMBER not in read and VAL_MEMBER in read


def test_existing_outputs_are_never_overwritten(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert vr.main(argv(inputs, out)) == 0
    before = {n: (out / n).read_bytes() for n in vr.OUTPUT_NAMES}
    assert vr.main(argv(inputs, out)) == 2
    assert {n: (out / n).read_bytes() for n in vr.OUTPUT_NAMES} == before


def test_the_contract_v1_record_holds_every_attempt_with_its_result(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert vr.main(argv(inputs, out)) == 0
    rec = json.loads((out / "contract_v1_record.json").read_text(encoding="utf-8"))
    s = summary(out)
    assert rec["investigation_id"] == vr.INVESTIGATION_ID and rec["calls_used"] == s["contract_v1"]["calls_used"]
    tools = [a["tool"] for a in rec["attempts"]]
    assert tools[0] == "query_events" and tools.count("compare_windows") == 3
    assert all(a["result"]["association_only"] is True for a in rec["attempts"])


# ---- acceptance (Codex review): a required call that did not execute, or a monitor / query disagreement, is never
# "verified"; an insufficient_evidence comparison is an executed, explicitly inconclusive outcome ----

REQUIRED = ("query_events", "check_tracking_quality", "get_window_features", "compare_windows")


def run_main(inputs, name="run"):
    out = inputs["tmp"] / name
    return vr.main(argv(inputs, out)), out


def not_verified(out, status="failed", step="contract_v1"):
    """The run ended with `status` at `step`, and nothing success-labelled was produced."""
    s = summary(out)
    assert s["status"] == status and s["steps"][-1]["step"] == step and s["steps"][-1]["status"] == "failed"
    assert "contract_v1" not in s                                      # no tool outcome is reported as accepted
    assert not (out / "contract_v1_record.json").exists()
    assert set(s["outputs"]) <= {"events.journal.jsonl", "journal_snapshot.json"}
    return s["steps"][-1]["message"]


def test_codex_compare_refusal_reproduction_fails_the_run(inputs, capsys, monkeypatch):
    monkeypatch.setitem(v1._TOOL_FUNCS, "compare_windows",
                        lambda data, request: v1._error("bad_window", "injected refusal"))
    rc, out = run_main(inputs)
    assert rc == 1
    msg = not_verified(out)
    assert "compare_windows (pre_s 2.0) did not execute" in msg and "injected refusal" in msg


@pytest.mark.parametrize("tool", REQUIRED)
def test_an_error_from_any_required_tool_fails_the_run(inputs, capsys, monkeypatch, tool):
    monkeypatch.setitem(v1._TOOL_FUNCS, tool, lambda data, request: v1._error("invalid_request", f"injected {tool}"))
    rc, out = run_main(inputs)
    assert rc == 1 and f"injected {tool}" in not_verified(out)


def altered(monkeypatch, tool, change):
    real = v1._TOOL_FUNCS[tool]
    monkeypatch.setitem(v1._TOOL_FUNCS, tool, lambda data, request: change(copy.deepcopy(real(data, request))))


@pytest.mark.parametrize("tool, change", [
    ("query_events", lambda r: None),                                              # no result object
    ("query_events", lambda r: {k: v for k, v in r.items() if k != "status"}),     # missing status
    ("query_events", lambda r: {**r, "query": {**r["query"], "threshold": 0.5}}),   # another query
    ("query_events", lambda r: {**r, "n_bouts": r["n_bouts"] + 1}),
    ("check_tracking_quality", lambda r: {**r, "verdict": None}),
    ("check_tracking_quality", lambda r: {**r, "quality_policy": "quality_v1_strict90"}),
    ("get_window_features", lambda r: {**r, "summary": {}}),
    ("get_window_features", lambda r: {**r, "association_only": False}),
    ("compare_windows", lambda r: {**r, "pre_s": 9.0}),                            # contradicts the request
    ("compare_windows", lambda r: {**r, "overlap_policy": "pre_only"}),
    ("compare_windows", lambda r: {**r, "n_excluded": r["n_excluded"] + 1}),
    ("compare_windows", lambda r: {**r, "mean_diff": 1.0}),                        # an effect behind a failed gate
    ("compare_windows", lambda r: {**r, "reason": "looked fine"}),
    ("compare_windows", lambda r: {**r, "status": "ok"}),                          # "ok" without an effect
    ("compare_windows", lambda r: {**r, "contract_version": "invented/v99"}),
])
def test_malformed_or_contradictory_required_results_fail_the_run(inputs, capsys, monkeypatch, tool, change):
    altered(monkeypatch, tool, change)
    rc, out = run_main(inputs)
    assert rc == 1
    not_verified(out)


def test_insufficient_evidence_is_an_executed_inconclusive_outcome(inputs, capsys):
    rc, out = run_main(inputs)
    assert rc == 0
    s = summary(out)
    assert s["status"] == "verified" and (out / "contract_v1_record.json").exists()
    outcome = s["contract_v1"]["scientific_outcome"]
    assert [o["status"] for o in outcome] == ["insufficient_evidence"] * 3
    assert all(o["outcome"].startswith("inconclusive: insufficient_evidence (too_few_bouts") for o in outcome)
    assert all("mean_diff" not in c for c in s["contract_v1"]["compare_windows"])


def monitor_returns(monkeypatch, change):
    real = vr.run_monitor
    monkeypatch.setattr(vr, "run_monitor", lambda drop, rule, out: (lambda ev, snap: (change(copy.deepcopy(ev)),
                                                                                      snap))(*real(drop, rule, out)))


def shift_start(ev):
    ev[1]["start_s"] += 1 / 30                                         # one frame later, same count
    return ev


def shift_end(ev):
    ev[0]["end_s"] -= 1 / 30
    return ev


def swap_order(ev):
    ev[0], ev[1] = ev[1], ev[0]                                        # same bouts, another firing order
    return ev


def other_reason(ev):
    ev[0]["evidence"]["end_reason"] = "session_end"
    return ev


@pytest.mark.parametrize("change", [lambda ev: ev[1:], shift_start, shift_end, swap_order, other_reason],
                         ids=["one_missing", "start_frame", "end_frame", "order", "end_reason"])
def test_a_monitor_query_disagreement_is_refused_before_any_window(inputs, capsys, monkeypatch, change):
    compared = []
    real = v1._TOOL_FUNCS["compare_windows"]
    monkeypatch.setitem(v1._TOOL_FUNCS, "compare_windows", lambda d, r: compared.append(1) or real(d, r))
    monitor_returns(monkeypatch, change)
    rc, out = run_main(inputs)
    assert rc == 2 and compared == []
    msg = not_verified(out, status="refused")
    assert "nothing is reconciled" in msg


def test_bout_identities_use_frame_boundaries_and_firing_order():
    ev = [{"event_id": "E-r-1", "session": "s", "start_s": 1.0, "end_s": 2.0,
           "evidence": {"fired_s": 1.5, "end_reason": "below_threshold"}}]
    bout = [{"event_id": "E-inv_x-1", "session": "s", "start_s": 1.0 + 1e-9, "fired_s": 1.5, "end_s": 2.0,
             "end_reason": "below_threshold"}]
    vr.check_same_bouts(ev, bout)                                       # the run ids differ; identities agree
    with pytest.raises(vr.RunRefused):
        vr.check_same_bouts(ev, [{**bout[0], "event_id": "E-inv_x-2"}])
    with pytest.raises(vr.RunRefused, match="malformed event id"):
        vr.check_same_bouts(ev, [{**bout[0], "event_id": "E-inv_x-?"}])


# ---- evidence-gate acceptance (Codex review): "ok" only for a genuinely met gate, computed by Contract v1 ----


def compare(d, **kw):
    q = v1.query_events(d, {"session": list(d.sessions), "behavior": "attack", "min_duration_s": 1.0})
    req = T.cmp_req(q, **kw)
    return v1.compare_windows(d, copy.deepcopy(req)), req, d


def accept(result, req, d):
    return vr._require_compare(result, req["pre_s"], d, req)


@pytest.fixture(scope="module")
def ok_case():
    c, req, d = compare(T.make(T.SESSIONS, starts=(12.0, 25.0)))            # 10 windows, 5 sessions: gate met
    assert c["status"] == "ok" and (c["n_windows"], c["n_sessions"]) == (10, 5)
    return c, req, d


def promote(c, req):
    used = {}
    excluded = {(e["session"], e["start_s"]) for e in c["excluded"]}
    for b in req["bouts"]:
        if (b["session"], b["start_s"]) not in excluded:
            used[b["session"]] = used.get(b["session"], 0) + 1
    per = [{"session": sid, "n_bouts": n, "paired_diff": 1.0} for sid, n in sorted(used.items())]
    out = {k: v for k, v in c.items() if k not in ("reason", "n_available", "n_required", "failed_gate")}
    return {**out, "status": "ok", "per_session": per, "mean_diff": statistics.median(p["paired_diff"] for p in per),
            "pre_mean": 1.0, "baseline_mean": 0.0, "mean_diff_definition": "x", "wording": v1.WORDING}


def test_a_valid_comparison_meeting_the_gate_is_accepted(ok_case):
    c, req, d = ok_case
    assert accept(c, req, d) is c


def test_nine_usable_windows_are_below_the_gate():
    # 5 sessions x 2 bouts, one pre window made unreliable: 9 used windows in 5 sessions
    c, req, d = compare(T.make(T.SESSIONS, starts=(12.0, 25.0),
                               edit=lambda sid, kp, sc: T.lose(kp, 25 * 30 - 40, 25 * 30 - 10)
                               if sid == T.SESSIONS[0] else None))
    assert c["status"] == "insufficient_evidence" and (c["n_windows"], c["n_sessions"]) == (9, 5)
    assert accept(c, req, d) is c                                      # the honest result is accepted
    with pytest.raises(vr.ToolFailure, match="9 usable windows from 5 sessions, below the approved gate"):
        accept(promote(c, req), req, d)


def test_ten_windows_from_three_sessions_are_below_the_gate():
    c, req, d = compare(T.make(T.SESSIONS[:3], starts=(12.0, 25.0, 40.0, 52.0)))
    assert c["status"] == "insufficient_evidence" and c["failed_gate"] == "min_n_sessions"
    assert accept(c, req, d) is c
    with pytest.raises(vr.ToolFailure, match="12 usable windows from 3 sessions, below the approved gate"):
        accept(promote(c, req), req, d)


@pytest.mark.parametrize("forge, why", [
    (lambda c: {**c, "n_sessions": c["n_sessions"] + 1}, "counts disagree"),
    (lambda c: {**c, "per_session": c["per_session"] + [c["per_session"][0]]}, "duplicated contributing sessions"),
    (lambda c: {**c, "per_session": [{**c["per_session"][0], "session": "SYNTHETIC_forged"}] + c["per_session"][1:]},
     "not the used bouts"),
    (lambda c: {**c, "per_session": [{**c["per_session"][0], "n_bouts": 3}, {**c["per_session"][1], "n_bouts": 1}]
                + c["per_session"][2:]}, "not the used bouts"),
    (lambda c: {**c, "mean_diff": c["mean_diff"] + 1.0}, "not the median"),
    (lambda c: {**c, "per_session": [{**c["per_session"][0], "paired_diff": float("nan")}] + c["per_session"][1:]},
     "not the median of finite"),
    (lambda c: {**c, "per_session": [{**p, "paired_diff": p["paired_diff"] + 5.0} for p in c["per_session"]],
                "mean_diff": c["mean_diff"] + 5.0}, "differs from what Contract v1 computes"),   # self-consistent
    (lambda c: {**c, "excluded": [{"session": c["per_session"][0]["session"], "start_s": 12.0, "why": "bored"}],
                "n_excluded": 1}, "known reason"),
])
def test_forged_or_inconsistent_ok_results_are_refused(ok_case, forge, why):
    c, req, d = ok_case
    with pytest.raises(vr.ToolFailure, match=why):
        accept(forge(copy.deepcopy(c)), req, d)


def test_insufficient_evidence_must_be_genuinely_unmet_and_carry_no_effect(ok_case):
    c, req, d = compare(T.make(T.SESSIONS[:3], starts=(12.0, 25.0, 40.0)))     # 9 bouts: too_few_bouts
    assert (c["reason"], c["failed_gate"]) == ("too_few_bouts", "min_n_windows") and accept(c, req, d) is c
    with pytest.raises(vr.ToolFailure, match="carries an effect"):
        accept({**c, "mean_diff": 0.0}, req, d)
    with pytest.raises(vr.ToolFailure, match="not the unmet condition"):
        accept({**c, "reason": "too_few_usable_windows"}, req, d)
    with pytest.raises(vr.ToolFailure, match="not the unmet condition"):
        accept({**c, "n_available": 10}, req, d)
    ok, oreq, od = ok_case                                              # a met gate declared insufficient
    demoted = {k: v for k, v in ok.items() if k not in ("per_session", "mean_diff", "pre_mean", "baseline_mean",
                                                         "mean_diff_definition", "wording")}
    demoted.update(status="insufficient_evidence", reason="too_few_usable_windows", failed_gate="min_n_windows",
                   n_available=10, n_required=10)
    with pytest.raises(vr.ToolFailure, match="although the gate is met"):
        accept(demoted, oreq, od)


def test_codex_forged_success_reproduction_fails_the_run(inputs, capsys, monkeypatch):
    real = v1._TOOL_FUNCS["compare_windows"]

    def promote_exact(data, request):                                  # Codex's wrapper, verbatim in effect
        result = real(data, request)
        result.update(status="ok", mean_diff=0.0,
                      per_session=[{"session": "SYNTHETIC_forged", "n_bouts": result["n_used"], "paired_diff": 0.0}])
        return result
    monkeypatch.setitem(v1._TOOL_FUNCS, "compare_windows", promote_exact)
    rc, out = run_main(inputs)
    assert rc == 1
    msg = not_verified(out)
    assert "compare_windows (pre_s 2.0) is not acceptable" in msg and "below the approved gate" in msg
    s = summary(out)
    assert s["investigation_evidence"].startswith("none accepted") and "evidence gate met" not in json.dumps(s)
