"""Tests for the experimental evidence-grounded report renderer (new file).

Reports are rendered from real Stage 6 InvestigationReports: the deterministic
fake investigators of test_investigation_agent over the Stage 4 SYNTHETIC
fixtures, in-memory contract fixtures (classifier and missing-provenance
cases), and the verified CalMS21 validation cache (skipped when absent). No
model or network is used.
"""

import ast
import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_contracts as ic
import live_tracker.investigation_report as ir
import test_investigation_contracts as tc
from live_tracker.investigation_agent import Budget
from test_investigation_agent import investigate  # legacy/1, explicitly (M3)
from live_tracker.investigation_contracts import PROVISIONAL_SUFFICIENCY, make_result
from live_tracker.investigation_events import CalMS21InvestigationTools
from live_tracker.investigation_report import ReportIntegrityError, build_report, render, render_markdown
from test_investigation_agent import (GOOD, LOOSE, QUESTION, Investigator, Scripted, _until_compare, call,
                                      finish, interp, no_network, obs, query, real_env)  # noqa: F401 (fixtures)
from test_investigation_windows import build

assert no_network and real_env       # autouse network block and the real-data fixture, reused


@pytest.fixture
def env(tmp_path):
    return build(tmp_path)


@pytest.fixture
def effect_report(env):
    return investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)


@pytest.fixture
def insufficient_report(env):
    return investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=PROVISIONAL_SUFFICIENCY)


def snapshot(report, tool, k=0):
    return [e for e in report.evidence if e["tool"] == tool][k]


def section(md, title):
    start = md.index(f"## {title}")
    end = md.find("\n## ", start + 1)
    return md[start:end if end > 0 else None]


# ---- successful descriptive finding ----

def test_successful_descriptive_finding(effect_report, env):
    doc, md = render(effect_report, policy=env[1])
    compare = snapshot(effect_report, "compare_windows")["result"]["measurements"]
    p = doc["primary_result"]
    assert doc["status"]["finding"] == "effect_described" and doc["status"]["finished"] is True
    assert p["effect_value"] == compare["effect"]["value"] and p["effect_reported"] is True
    assert p["direction"] == effect_report.direction and p["unit"] == "px/s"
    assert p["uncertainty"] == {"method": "none", "interval": None,
                                "note": "no approved session-level uncertainty method (Stage 5)"}
    assert p["sufficiency"] == compare["sufficiency"]
    assert doc["verification"]["status"] == "verified" and doc["verification"]["policy_revalidated"] is True
    summary = section(md, "Executive summary")
    assert f"was {ir._fmt(compare['effect']['value'])} (px/s)" in summary
    assert "Descriptive only" in summary and "2 independent sessions paired" in summary
    # nothing beyond the descriptive policy: no interval, test or causal claim is generated
    generated = "\n".join(doc["executive_summary"]).replace(ir.DESCRIPTIVE_POLICY, "")
    assert not re.search(r"significan|p-value|confidence interval|caus|prov(e|es|en)\b", generated, re.I)


# ---- insufficient evidence ----

def test_insufficient_evidence_reports_no_effect(insufficient_report):
    doc, md = render(insufficient_report)
    p = doc["primary_result"]
    assert doc["status"]["finding"] == "insufficient_evidence" and doc["status"]["direction"] is None
    assert p["effect_value"] is None and p["effect_reported"] is False and p["direction"] is None
    assert p["sufficiency"]["met"] is False and p["sufficiency"]["reasons_not_met"]
    summary = section(md, "Executive summary")
    assert "no aggregate effect is reported" in summary and "direction" not in summary
    assert "| Aggregate effect | not reported (null) |" in md
    assert "shown for audit only and are not a finding" in md
    for r in p["per_session"]:                        # per-session values never reach the summary
        if r["difference"] is not None:
            assert ir._fmt(r["difference"]) not in summary


# ---- incomplete investigation ----

def test_incomplete_investigation_lists_evidence_without_a_finding(env):
    report = investigate(QUESTION, env[0], env[1], Scripted(*_until_compare(LOOSE)), sufficiency=LOOSE,
                         budget=Budget(max_model_turns=3))
    assert report.finding == "incomplete" and snapshot(report, "compare_windows")["status"] == "ok"
    doc, md = render(report)
    assert doc["status"] == {"finding": "incomplete", "finding_label": "Incomplete (no finding)", "finished": False,
                             "stopping_reason": "max_model_turns", "direction": None, "primary_evidence_id": None}
    assert doc["primary_result"] is None and doc["observations"] == [] and doc["interpretations"] == []
    summary = section(md, "Executive summary")
    assert "Incomplete: the investigation stopped before a final answer (max_model_turns)" in summary
    assert "3 validated results were collected" in summary and "none presented as a finding" in summary
    assert "## Primary comparison" not in md
    # the ok comparison is still inspectable in the appendix, as evidence
    eid = snapshot(report, "compare_windows")["evidence_id"]
    assert f"### `{eid}`" in md and any(a["evidence_id"] == eid for a in doc["evidence_appendix"])
    assert any("stopped before a final answer" in t for t in doc["limitations"]["agent"])


# ---- tampered, missing or inconsistent evidence: fail closed ----

def _with(report, **fields):
    return dataclasses.replace(report, **{k: copy.deepcopy(v) for k, v in fields.items()})


def _tamper_value(r):
    ev = copy.deepcopy(r.evidence)
    next(e for e in ev if e["tool"] == "compare_windows")["result"]["measurements"]["n_windows_baseline"] += 1
    return _with(r, evidence=ev)


def _rehash_effect(r):
    """Change the effect and recompute the hash in the snapshot: the trace,
    primary result and observation still disagree (and the effect no longer
    follows from the per-session rows)."""
    ev = copy.deepcopy(r.evidence)
    e = next(e for e in ev if e["tool"] == "compare_windows")
    e["result"]["measurements"]["effect"]["value"] = 123.0
    e["result"]["evidence_id"] = e["evidence_id"] = ic.evidence_id(e["tool"], e["request"], e["result"])
    return _with(r, evidence=ev)


def _forge_everywhere(r):
    """A fully re-hashed forgery: every reference to the old ID is rewritten."""
    old = next(e for e in r.evidence if e["tool"] == "compare_windows")
    e = copy.deepcopy(old)
    e["result"]["measurements"]["effect"]["value"] = 123.0
    new = ic.evidence_id(e["tool"], e["request"], e["result"])
    text = json.dumps(r.to_json()).replace(old["evidence_id"], new)
    d = json.loads(text)
    for x in d["evidence"]:
        if x["evidence_id"] == new:
            x["result"]["measurements"]["effect"]["value"] = 123.0
    d["primary_evidence"]["effect"]["value"] = 123.0
    for o in d["observations"]:
        if o["field"] == "measurements.effect.value":
            o["value"] = 123.0
            o["statement"] = f"compare_windows {new}: measurements.effect.value = 123.0"
    return ia.InvestigationReport(**d)


def _drop_evidence(r):
    return _with(r, evidence=[e for e in r.evidence if e["tool"] != "check_tracking_quality"])


def _observation(r, **change):
    obs_ = copy.deepcopy(r.observations)
    obs_[-1].update(change)
    return _with(r, observations=obs_)


@pytest.mark.parametrize("tamper, problem", [
    (_tamper_value, "failed verification"),
    (_rehash_effect, "failed verification"),
    (_forge_everywhere, "failed verification"),
    (_drop_evidence, "trace's executed tool calls do not match"),
    (lambda r: _with(r, evidence=r.evidence + r.evidence[:1]), "duplicate evidence_id"),
    (lambda r: _with(r, evidence=r.evidence[:1] + [{"evidence_id": "x"}]), "not a request / result snapshot"),
    (lambda r: _with(r, trace=[t for t in r.trace if t["kind"] != "stop"]), "stopping reason"),
    (lambda r: _observation(r, value=-1.0), "observations[4]: value or statement differs"),
    (lambda r: _observation(r, statement="The effect was large."), "value or statement differs"),
    (lambda r: _observation(r, field="measurements.no_such"), "cites a field that is not in its evidence"),
    (lambda r: _observation(r, evidence_id="ev:compare_windows:" + "0" * 24), "not in its evidence"),
    (lambda r: _with(r, primary_evidence={**r.primary_evidence, "n_sessions_paired": 9}),
     "primary evidence differs"),
    (lambda r: _with(r, direction="positive" if r.direction == "negative" else "negative"), "direction differs"),
    (lambda r: _with(r, limitations=[l for l in r.limitations if "Investigation confound" not in l["text"]]),
     "tool limitations differ"),
    (lambda r: _with(r, interpretations=[{"kind": "interpretation", "statement": "Approach causes attack.",
                                          "evidence_ids": [r.primary_evidence["evidence_id"]]}]), "causal claim"),
    (lambda r: _with(r, next_steps=["Report the p-value."]), "p-value"),
    (lambda r: _with(r, finished=False), "disagree"),
])
def test_tampered_or_inconsistent_reports_are_not_rendered(effect_report, tamper, problem):
    bad = tamper(effect_report)
    with pytest.raises(ReportIntegrityError) as e:
        render(bad)
    assert any(problem in p for p in e.value.problems), e.value.problems
    render(effect_report)                              # the original is untouched and still renders


def test_effect_cannot_be_claimed_from_an_insufficient_comparison(insufficient_report):
    bad = _with(insufficient_report, finding="effect_described", direction="negative")
    with pytest.raises(ReportIntegrityError) as e:
        build_report(bad)
    assert any("effect_described without an 'ok' comparison" in p for p in e.value.problems)


def test_intervals_outside_the_descriptive_policy_are_refused(env):
    """A comparison carrying an interval (contract-valid bootstrap fixture)
    is never rendered as a finding."""
    req, res = tc.fixture("compare_windows", "ok")
    report = ia.InvestigationReport(
        question=QUESTION, stopping_reason="max_model_turns", finished=False, finding="incomplete", direction=None,
        primary_evidence=None, observations=[], interpretations=[],
        evidence=[{"evidence_id": res["evidence_id"], "tool": "compare_windows", "status": "ok", "request": req,
                   "result": res}],
        limitations=[{"source": "tool", "text": t} for t in res["limitations"]], next_steps=[], usage={},
        trace=[{"step": 0, "kind": "tool_call", "tool": "compare_windows", "request": req,
                "evidence_id": res["evidence_id"]}, {"step": 1, "kind": "stop", "stopping_reason": "max_model_turns"}])
    with pytest.raises(ReportIntegrityError) as e:
        render(report)
    assert any("outside the descriptive policy" in p for p in e.value.problems)


def test_not_a_report_and_strict_policy_fail_closed(effect_report):
    with pytest.raises(ReportIntegrityError):
        render({"finding": "effect_described"})
    strict = ic.EvidencePolicy(tc.MANIFEST)            # no synthetic evidence allowed
    with pytest.raises(ReportIntegrityError) as e:
        render(effect_report, policy=strict)
    assert any("failed verification" in p for p in e.value.problems)


# ---- provenance: annotations, classifier predictions, synthetic, missing ----

class FixtureTools(ic.UnavailableTools):
    """In-memory tools answering from contract fixtures (built with make_result)."""

    def __init__(self, provenance=tc.PROV):
        self.provenance = provenance

    def query_events(self, req):
        m = tc.query_measurements()
        ep = m["episodes"][0]
        if req["source"] == "classifier_prediction":
            ep.update(episode_id=ic.episode_id("classifier_prediction", tc.BEH, tc.VAL_A, 90),
                      source="classifier_prediction",
                      provenance={"model_id": "SYNTHETIC-classifier-v3", "score_file_sha256": "b" * 64,
                                  "threshold": 0.75, "min_duration_frames": 1})
            del ep["annotated_frames"]
        return make_result("query_events", req, "ok", m, provenance=self.provenance, limitations=["fixture"])

    def compare_windows(self, req):
        (pre,), (base,) = req["pre_window_ids"], req["baseline_window_ids"]
        m = {"feature": req["feature"], "aggregation": "session_level",
             "per_session": [{"session": tc.VAL_A, "pre_windows": [tc.wv(pre, -12.5)],
                              "baseline_windows": [tc.wv(base, -3.0, 29)],
                              "pre_summary": -12.5, "baseline_summary": -3.0, "difference": -9.5}],
             "excluded_windows": [], "n_sessions_paired": 1, "n_windows_pre": 1, "n_windows_baseline": 1,
             "effect": {"name": "median_of_session_differences", "value": None, "ci_low": None, "ci_high": None,
                        "ci_level": None},
             "sufficiency": {"criteria": req["sufficiency"], "met": False,
                             "reasons_not_met": ["min_sessions", "min_windows_per_group_pre",
                                                 "min_windows_per_group_baseline"]}}
        return make_result("compare_windows", req, "insufficient_evidence", m, provenance=self.provenance,
                           limitations=["fixture"])


def _query_report(source, tools=None):
    model = Scripted([call("query_events", source=source, behavior="attack", sessions=[tc.VAL_A])],
                     lambda m: [call(ia.FINISH, finding="inconclusive", observations=[
                         {"evidence_id": next(json.loads(b["content"])["evidence_id"] for x in m
                                              if isinstance(x["content"], list) for b in x["content"]
                                              if b.get("type") == "tool_result"),
                          "field": "measurements.n_episodes_total"}])])
    return investigate(QUESTION, tools or FixtureTools(), tc.POLICY, model)


def test_classifier_and_annotation_provenance_are_distinguished():
    ann = build_report(_query_report("validation_annotation"))["provenance"]
    clf = build_report(_query_report("classifier_prediction"))["provenance"]
    assert [s["source"] for s in ann["event_sources"]] == ["validation_annotation"]
    assert [s["source"] for s in clf["event_sources"]] == ["classifier_prediction"]
    assert ann["classifier"] == {"model_ids": None, "thresholds": None, "note": "no classifier-generated evidence"}
    assert clf["classifier"]["model_ids"] == ["SYNTHETIC-classifier-v3"] and clf["classifier"]["thresholds"] == [0.75]
    assert ann["split"] == {"values": ["validation"], "basis": "provenance of query_events episodes"}
    assert clf["split"]["values"] is None and clf["annotators"] is None      # predictions record no split / annotator
    assert ann["annotators"] == ["1"]
    for p in (ann, clf):                               # both from synthetic fixtures: never called ground truth
        assert p["synthetic"] is True and "synthetic_fixture" in p["event_sources"][0]["class"]
    md = render(_query_report("classifier_prediction"))[1]
    assert "SYNTHETIC-classifier-v3" in md and "SYNTHETIC DATA" in md


def test_classifier_identity_is_not_inferred_from_window_ids():
    pre = ic.pre_window_id(ic.episode_id("classifier_prediction", tc.BEH, tc.VAL_A, 90), 30, 0)
    base = ic.baseline_window_id("classifier_prediction", tc.BEH, tc.VAL_A, 0, 30)
    model = Scripted([call("compare_windows", feature="dist_rate", pre_window_ids=[pre], baseline_window_ids=[base])])
    report = investigate(QUESTION, FixtureTools(), tc.POLICY, model, budget=Budget(max_model_turns=1))
    prov = build_report(report)["provenance"]
    assert prov["event_sources"][0]["source"] == "classifier_prediction"
    assert prov["classifier"]["model_ids"] is None and "not recorded" in prov["classifier"]["note"]


def test_missing_provenance_is_reported_as_unknown():
    bare = {"implementation": "bare-tools", "implementation_version": "0", "contract_version": ic.CONTRACT_VERSION,
            "code_sha256": None, "data": []}
    pre = ic.pre_window_id(tc.EP, 30, 0)
    model = Scripted([call("compare_windows", feature="dist_rate", pre_window_ids=[pre],
                           baseline_window_ids=[tc.BASE])])
    report = investigate(QUESTION, FixtureTools(bare), tc.POLICY, model, budget=Budget(max_model_turns=1))
    doc, md = render(report)
    prov = doc["provenance"]
    assert prov["data"] is None and prov["synthetic"] is None
    assert prov["split"] == {"values": None, "basis": "not recorded in this evidence"}
    assert prov["annotators"] is None and prov["event_definitions"] is None and prov["investigator_model_ids"] is None
    assert prov["tracking_quality"]["assessed"] is False and prov["tool_configuration"][0]["code_sha256"] is None
    assert doc["counts"]["behavior_bouts"] is None
    assert "| Synthetic data | unknown |" in md and "| Data | unknown |" in md and "| Split | unknown (not recorded" in md
    assert "synthetic data: unknown (no data provenance recorded)" in section(md, "Executive summary")
    assert "the unit is not recorded" not in md and doc["evidence_appendix"][0]["summary"]["unit"] is None


# ---- bouts, windows and sessions ----

def test_multiple_windows_from_one_session_are_not_replicates(effect_report):
    doc, md = render(effect_report)
    p = doc["primary_result"]
    rows = {r["session"]: r for r in p["per_session"]}
    assert p["independent_sessions"] == len(rows) == 2
    assert p["n_windows_pre"] == sum(r["pre_windows_used"] for r in rows.values()) == 3
    multi = [r for r in rows.values() if r["pre_windows_used"] > 1]
    assert multi and multi[0]["bouts_with_used_pre_window"] == multi[0]["pre_windows_used"]
    assert doc["counts"]["independent_sessions"] == 2 and doc["counts"]["windows_used"]["pre_event"] == 3
    assert doc["counts"]["behavior_bouts"][0]["n_bouts_total"] == 4
    assert ir.UNIT_OF_ANALYSIS in md and "2 independent sessions paired, from 3 pre-event windows" in md
    assert "3 independent" not in md and "4 independent" not in md


# ---- observations ----

def test_observation_values_are_rendered_from_the_snapshots(effect_report):
    doc, md = render(effect_report)
    for o, src in zip(doc["observations"], effect_report.observations):
        snap = next(e for e in effect_report.evidence if e["evidence_id"] == o["evidence_id"])
        assert o["value"] == ir._resolve(snap["result"], o["field"]) == src["value"]
        row = f"| {o['field']} | {ir._cell(o['value'])} | {o['unit'] or '-'} | {o['evidence_id']} |"
        assert row in md
    units = {o["field"]: o["unit"] for o in doc["observations"]}
    assert units["measurements.effect.value"] == "px/s" and units["measurements.n_sessions_paired"] == "sessions"
    assert units["measurements.n_windows_pre"] == "windows" and units["status"] is None


def test_interpretations_and_next_steps_are_labelled(insufficient_report):
    doc, md = render(insufficient_report)
    assert all(i["kind"] == "model_interpretation" for i in doc["interpretations"])
    assert all(s["kind"] == "model_suggestion" for s in doc["next_steps"])
    body = section(md, "Model interpretations (not evidence; model-authored, unverified)")
    assert body.count("_Interpretation (model):_") == len(doc["interpretations"])
    assert "Extract the remaining validation sessions" in section(md, "Suggested next steps (model suggestions, not evidence; "
                                                                     "unverified)")


# ---- JSON / Markdown ----

def test_json_and_markdown_are_consistent_and_deterministic(effect_report, insufficient_report):
    for report in (effect_report, insufficient_report):
        doc, md = render(report)
        text = ir.to_json_text(doc)
        assert text == ir.to_json_text(build_report(report)) and md == render(report)[1]     # deterministic
        assert render_markdown(json.loads(text)) == md                        # Markdown from the JSON alone
        for eid in doc["provenance"]["evidence_ids"]:
            assert eid in md
        for o in doc["observations"]:
            assert ir._cell(o["value"]) in md
        assert doc["status"]["finding_label"] in md and doc["status"]["stopping_reason"] in md
        for s in doc["executive_summary"]:
            assert s in md
        for r in (doc["primary_result"] or {}).get("per_session", []):
            assert f"| {r['session']} | {r['bouts_with_used_pre_window']} | {r['pre_windows_used']} |" in md
        for t in doc["limitations"]["tool"]:
            assert t in md
        assert doc["evidence_snapshots"] == [{k: e[k] for k in ("evidence_id", "tool", "status", "request", "result")}
                                             for e in report.evidence]


def test_rendering_does_not_alias_the_report(effect_report):
    before = json.dumps(effect_report.to_json(), sort_keys=True)
    doc = build_report(effect_report)
    doc["evidence_snapshots"][0]["result"]["status"] = "error"
    doc["observations"][0]["value"] = "x"
    assert json.dumps(effect_report.to_json(), sort_keys=True) == before and effect_report.verify_evidence() == []


def test_separate_from_the_journal_and_from_the_agent_loop():
    tree = ast.parse(Path(ir.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in imported if m and re.search(r"event_log|monitor|journal|anthropic|agent\b", m)}
    assert "investigate(" not in Path(ir.__file__).read_text()


# ---- real validation data ----

def test_real_validation_report(real_env):
    report = investigate(QUESTION, *real_env, Investigator(length=30, clearance=150))
    doc, md = render(report, policy=real_env[1])
    p, prov = doc["primary_result"], doc["provenance"]
    assert doc["status"]["finding"] == "insufficient_evidence" and p["effect_value"] is None
    assert (p["independent_sessions"], p["n_windows_pre"], p["n_windows_baseline"]) == (2, 30, 164)
    assert p["per_session"][0]["pre_windows_used"] == 5 and p["unit"] == "px/s"
    assert prov["synthetic"] is False and prov["split"]["values"] == ["validation"]
    assert [s["class"] for s in prov["event_sources"]] == ["ground_truth_annotation"]
    assert {d["kind"] for d in prov["data"]} == {"calms21_validation_cache"}
    assert prov["classifier"]["model_ids"] is None and prov["investigator_model_ids"] == ["fake-investigator"]
    assert prov["tracking_quality"]["assessed"] and prov["tracking_quality"]["motion_accuracy_verified"] is False
    assert doc["counts"]["behavior_bouts"][0]["n_bouts_total"] == 56
    assert len(doc["limitations"]["confounders"]) >= 2
    assert "no aggregate effect is reported" in section(md, "Executive summary") and "SYNTHETIC" not in md
    assert all(s.startswith("task1/") for s in prov["session_ids"])


# ======================================================================
# Codex 2 review: five reporting defects and the limitation trust boundary
# ======================================================================

import statistics  # noqa: E402

from test_investigation_agent import contaminated  # noqa: E402


def forge(report, tool, mutate, k=0, check=True):
    """Change a snapshot, re-hash it and rewrite every reference, so each
    snapshot stays individually valid (a consistent forgery)."""
    d = report.to_json()
    e = [x for x in d["evidence"] if x["tool"] == tool][k]
    mutate(e["request"], e["result"])
    old, new = e["evidence_id"], ic.evidence_id(tool, e["request"], e["result"])
    d = json.loads(json.dumps(d).replace(old, new))
    e = next(x for x in d["evidence"] if x["evidence_id"] == new)
    m = e["result"]["measurements"]
    if d["primary_evidence"] and d["primary_evidence"]["evidence_id"] == new and m is not None:
        d["primary_evidence"].update({k: m[k] for k in ("feature", "effect", "sufficiency", "n_sessions_paired",
                                                         "n_windows_pre", "n_windows_baseline")})
    for o in d["observations"]:
        if o["evidence_id"] == new:
            o["value"] = ir._resolve(e["result"], o["field"])
            o["statement"] = f"{tool} {new}: {o['field']} = {json.dumps(o['value'])}"
    if d["direction"] is not None and m is not None and tool == "compare_windows":
        d["direction"] = ir._sign(m["effect"]["value"])
    forged = ia.InvestigationReport(**d)
    for x in forged.evidence if check else ():         # every snapshot is individually valid
        ic.check_result_schema(x["tool"], x["request"], x["result"])
    return forged


def inflate_pre_window(req, res):
    """One pre-event window value -> -1000, with the session summary,
    difference and aggregate effect recomputed as the contract defines them."""
    m = res["measurements"]
    s = m["per_session"][0]
    s["pre_windows"][0]["value"] = -1000.0
    s["pre_summary"] = statistics.median(w["value"] for w in s["pre_windows"])
    s["difference"] = s["pre_summary"] - s["baseline_summary"]
    m["effect"]["value"] = statistics.median(x["difference"] for x in m["per_session"] if x["difference"] is not None)


def set_method(name, value):
    def mutate(req, res):
        for x in res["methodology"]:
            if x["name"] == name:
                x["value"] = value
    return mutate


def both(*mutations):
    def mutate(req, res):
        for f in mutations:
            f(req, res)
    return mutate


# ---- 1. contradictory snapshots ----

def test_contradictory_window_values_across_snapshots_are_refused(effect_report):
    forged = forge(effect_report, "compare_windows", inflate_pre_window)
    with pytest.raises(ReportIntegrityError) as e:
        render(forged)
    assert any("disagree on dist_rate" in p and "-1000.0" in p for p in e.value.problems), e.value.problems


def test_agreeing_values_without_a_unit_on_one_side_are_not_verified(effect_report):
    """get_window_features records no unit: its values agree with the
    comparison, but agreement is not verified without explicit units."""
    doc, md = render(effect_report)
    cross = doc["verification"]["cross_snapshot"]
    assert cross["status"] == "not_verified" and cross["unit_conflicts"] == []
    (pair,) = cross["pairs"]
    assert pair["outcome"] == "unit_unverified" and pair["units"] == [None, "px/s"]
    assert pair["features"] == ["dist_rate"] and pair["shared_window_features"] > 20
    assert "without explicit unit evidence" in pair["reasons"][0]
    assert "(unit_unverified: 1; see Verification)" in section(md, "Executive summary")


@pytest.mark.parametrize("mutation, reason", [
    (set_method("smoothing", "none"), "processing configuration differs: ['smoothing']"),
    (set_method("smoother_memory_frames", 5), "processing configuration differs: ['smoother_memory_frames']"),
    (set_method("window_value", "mean of the feature over usable frames"), "different or unrecognised statistics"),
    (lambda req, res: res["methodology"].__setitem__(slice(None), [x for x in res["methodology"]
                                                                    if x["name"] != "fps"]), "does not record processing"),
    (lambda req, res: res["provenance"]["data"][0].update(sha256="0" * 64), "data provenance differs"),
])
def test_non_comparable_snapshots_are_disclosed_not_trusted(effect_report, mutation, reason):
    """Different processing, statistic or data: the values are not compared
    (even when they differ) and the report says agreement was not verified."""
    forged = forge(effect_report, "compare_windows", both(mutation, inflate_pre_window))
    doc, md = render(forged)
    cross = doc["verification"]["cross_snapshot"]
    assert cross["status"] == "not_verified"
    (pair,) = cross["pairs"]
    assert pair["outcome"] == "not_comparable" and any(reason in r for r in pair["reasons"]), pair["reasons"]
    assert "Cross-snapshot agreement was not verified for 1 pair" in section(md, "Executive summary")
    assert "Cross-snapshot agreement: `not_verified`." in md and reason.split(":")[0] in md


def test_unhashed_data_provenance_is_not_comparable(effect_report):
    forged = forge(effect_report, "compare_windows",
                   both(lambda req, res: res["provenance"]["data"][0].update(sha256=None), inflate_pre_window))
    pair = build_report(forged)["verification"]["cross_snapshot"]["pairs"][0]
    assert pair["outcome"] == "not_comparable" and "data provenance is not recorded with hashes" in pair["reasons"]


# ---- 2. a finding other than effect_described never presents an effect ----

@pytest.mark.parametrize("finding", ["insufficient_evidence", "inconclusive"])
def test_ok_comparison_under_another_finding_is_not_promoted(effect_report, finding):
    changed = _with(effect_report, finding=finding, direction=None, interpretations=[])
    doc, md = render(changed)
    p = doc["primary_result"]
    value = ir._fmt(p["effect_value"])
    assert p["status"] == "ok" and p["effect_reported"] and not p["presented_as_finding"] and p["direction"] is None
    summary = section(md, "Executive summary")
    assert value not in summary and "direction" not in summary and "no aggregate effect is presented" in summary
    assert "| Aggregate effect | recorded but not presented as a finding (see appendix) |" in md
    assert "| Direction | null |" in md
    assert f"| effect_value | {value} |" in section(md, "Evidence appendix")       # kept for audit


def test_finding_and_comparison_status_must_agree(effect_report, insufficient_report):
    with pytest.raises(ReportIntegrityError):              # a direction on insufficient evidence
        render(_with(insufficient_report, direction="negative"))
    # an "ok" status without the sufficiency criteria met is inconsistent (re-hashed)
    mutate = lambda req, res: res["measurements"]["sufficiency"].update(met=False)  # noqa: E731
    with pytest.raises(ic.ContractError):                  # the contract refuses it ...
        forge(effect_report, "compare_windows", mutate)
    with pytest.raises(ReportIntegrityError):              # ... and so does the renderer
        render(forge(effect_report, "compare_windows", mutate, check=False))


# ---- 3. units belong to their own snapshot ----

def test_conflicting_units_are_kept_per_snapshot(tmp_path):
    env = build(tmp_path, contaminated)
    report = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    assert [e["tool"] for e in report.evidence].count("compare_windows") == 2
    forged = forge(report, "compare_windows", set_method("units", "mm/s"), k=1)
    doc, md = render(forged)
    first, second = [e for e in forged.evidence if e["tool"] == "compare_windows"]
    by_eid = {a["evidence_id"]: a["summary"]["unit"] for a in doc["evidence_appendix"] if a["tool"] == "compare_windows"}
    assert by_eid == {first["evidence_id"]: "px/s", second["evidence_id"]: "mm/s"}
    assert doc["primary_result"]["unit"] == "px/s"                         # the primary keeps its own unit
    units = {(o["evidence_id"], o["field"]): o["unit"] for o in doc["observations"]}
    assert units.get((first["evidence_id"], "measurements.effect.value")) == "px/s"
    (conflict,) = doc["verification"]["cross_snapshot"]["unit_conflicts"]
    assert conflict == {"feature": "dist_rate", "kind": "conflicting",
                        "units": [{"unit": "mm/s", "evidence_ids": [second["evidence_id"]]},
                                  {"unit": "px/s", "evidence_ids": [first["evidence_id"]]}]}
    assert "Comparisons record different units for the same feature: dist_rate (conflicting)" in \
        section(md, "Executive summary")


def test_feature_snapshots_have_no_borrowed_unit(effect_report):
    doc = build_report(effect_report)
    feats = snapshot(effect_report, "get_window_features")
    assert ir._unit_of(feats, "measurements.windows[0].features.dist_rate.median") is None
    summary = next(a["summary"] for a in doc["evidence_appendix"] if a["tool"] == "get_window_features")
    assert summary["units"] == {"dist_rate": None} and "records no unit" in summary["units_note"]


def test_a_comparison_without_a_unit_reports_unknown(effect_report):
    forged = forge(effect_report, "compare_windows", lambda req, res: res["methodology"].__setitem__(
        slice(None), [x for x in res["methodology"] if x["name"] != "units"]))
    doc, md = render(forged)
    assert doc["primary_result"]["unit"] is None and "(unit not recorded)" in section(md, "Executive summary")


# ---- 4. error comparisons render ----

class FailingCompare(FixtureTools):
    def compare_windows(self, req):
        raise RuntimeError("backend down")             # call_tool turns this into a valid error result


def test_error_comparison_renders_with_its_error_codes():
    pre = ic.pre_window_id(tc.EP, 30, 0)

    def answer(m):
        eid = next(json.loads(b["content"])["evidence_id"] for x in m if isinstance(x["content"], list)
                   for b in x["content"] if b.get("type") == "tool_result")
        return [call(ia.FINISH, finding="insufficient_evidence", primary_evidence_id=eid,
                     observations=[obs({"evidence_id": eid}, "status"), obs({"evidence_id": eid}, "errors[0].code")])]
    model = Scripted([call("compare_windows", feature="dist_rate", pre_window_ids=[pre],
                           baseline_window_ids=[tc.BASE])], answer)
    report = investigate(QUESTION, FailingCompare(), tc.POLICY, model)
    assert report.finished and report.evidence[0]["status"] == "error"
    doc, md = render(report)
    assert doc["primary_result"] is None
    assert doc["primary_evidence"] == {"evidence_id": report.evidence[0]["evidence_id"], "tool": "compare_windows",
                                       "status": "error", "error_codes": ["internal_error"]}
    assert "has status error (error codes: internal_error)" in section(md, "Executive summary")
    assert "| internal_error | compare_windows failed (RuntimeError) |" in md
    assert [o["value"] for o in doc["observations"]] == ["error", "internal_error"]
    assert doc["trace"]["steps"][1]["outcome"] == "error"


def test_error_results_in_an_incomplete_run_render():
    pre = ic.pre_window_id(tc.EP, 30, 0)
    model = Scripted([call("compare_windows", feature="dist_rate", pre_window_ids=[pre], baseline_window_ids=[tc.BASE])])
    report = investigate(QUESTION, FailingCompare(), tc.POLICY, model, budget=Budget(max_model_turns=1))
    doc, md = render(report)
    assert doc["status"]["finding"] == "incomplete" and doc["evidence_appendix"][0]["summary"]["measurements"] is None
    assert "internal_error" in section(md, "Evidence appendix")


# ---- 5. truncated queries ----

class TruncatingQuery(FixtureTools):
    def query_events(self, req):
        eps = []
        for sid in (tc.VAL_A, tc.VAL_B):
            ep = copy.deepcopy(tc.query_measurements()["episodes"][0])
            ep.update(episode_id=ic.episode_id(tc.SRC, tc.BEH, sid, 90), session=sid)
            eps.append(ep)
        kept = eps[:req["max_episodes"]]
        m = {"episodes": kept, "n_episodes_total": len(eps), "truncated": len(eps) > len(kept),
             "sessions_searched": [tc.VAL_A, tc.VAL_B], "sessions_without_episodes": []}
        return make_result("query_events", req, "ok", m, provenance=self.provenance, limitations=["fixture"])


def test_truncated_query_counts_sessions_from_the_query_summary():
    model = Scripted([call("query_events", source="validation_annotation", behavior="attack",
                           sessions=[tc.VAL_A, tc.VAL_B], max_episodes=1)])
    report = investigate(QUESTION, TruncatingQuery(), tc.POLICY, model, budget=Budget(max_model_turns=1))
    doc, md = render(report)
    (b,) = doc["counts"]["behavior_bouts"]
    assert (b["n_bouts_total"], b["n_bouts_returned"], b["truncated"]) == (2, 1, True)
    assert (b["n_sessions_searched"], b["n_sessions_with_bouts"], b["n_sessions_among_returned_bouts"]) == (2, 2, 1)
    assert b["truncation_note"] == ("Truncated: 1 of 2 bouts were returned; counts per session of returned bouts "
                                    "are not the full distribution.")
    assert b["truncation_note"] in section(md, "Executive summary")
    assert f"| {b['evidence_id']} | attack | 2 | 1 | 2 | 2 | 1 | true |" in md
    summary = doc["evidence_appendix"][0]["summary"]
    assert summary["returned_episodes_per_session"] == {tc.VAL_A: 1} and "episodes_per_session" not in summary


def test_untruncated_query_has_no_truncation_note(effect_report):
    (b,) = build_report(effect_report)["counts"]["behavior_bouts"]
    assert b["truncated"] is False and b["truncation_note"] is None
    assert b["n_sessions_with_bouts"] == b["n_sessions_among_returned_bouts"] == 2


# ---- trust boundary: only the loop's fixed operational texts are "agent" limitations ----

@pytest.mark.parametrize("text", ["Approach causes attack.", "The effect was 10 px/s.",
                                  ia.AGENT_LIMITATIONS[0] + " "])
def test_unknown_agent_limitations_are_refused(effect_report, text):
    with pytest.raises(ReportIntegrityError) as e:
        render(_with(effect_report, limitations=effect_report.limitations + [{"source": "agent", "text": text}]))
    assert any("not one of the loop's fixed texts" in p for p in e.value.problems)


def test_model_limitations_cannot_appear_in_an_incomplete_run(env):
    report = investigate(QUESTION, env[0], env[1], Scripted([query()]), budget=Budget(max_model_turns=1))
    bad = _with(report, limitations=report.limitations + [{"source": "model", "text": "Only one session."}])
    with pytest.raises(ReportIntegrityError):
        render(bad)


@pytest.mark.parametrize("budget, reply_tokens", [(Budget(max_output_tokens=1), (100, 100)),
                                                  (Budget(max_model_turns=1), (10, 5))])
def test_operational_limitations_of_budget_stops_render(env, budget, reply_tokens):
    report = investigate(QUESTION, env[0], env[1], Scripted([query()], tokens=reply_tokens), budget=budget)
    doc, md = render(report)
    assert doc["limitations"]["agent"] and doc["limitations"]["trust"]["agent"] == ir.TRUST["agent"]
    assert "Investigation limitations (investigation loop, fixed operational text)" in md
    assert "Limitations added by the model (model-authored, unverified (lexical checks only))" in md


def test_usage_unavailable_limitation_renders(env):
    reply = ia.ModelReply(content=[query()], stop_reason="tool_use", input_tokens=None, output_tokens=None)
    report = investigate(QUESTION, env[0], env[1], Scripted(reply))
    assert report.stopping_reason == "usage_unavailable"
    assert any("no valid token usage" in t for t in build_report(report)["limitations"]["agent"])


# ---- Codex 2 follow-up: units take part in cross-snapshot comparability ----

@pytest.fixture
def two_comparisons(tmp_path):
    """Two comparisons of dist_rate sharing windows (the sensitivity rerun)."""
    env = build(tmp_path, contaminated)
    report = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    assert [e["tool"] for e in report.evidence].count("compare_windows") == 2
    return report


def drop_units(req, res):
    res["methodology"][:] = [x for x in res["methodology"] if x["name"] != "units"]


def compare_pair(doc, report):
    a, b = [e["evidence_id"] for e in report.evidence if e["tool"] == "compare_windows"]
    return next(p for p in doc["verification"]["cross_snapshot"]["pairs"] if p["evidence_ids"] == [a, b])


def exports_are_consistent(report):
    doc, md = render(report)
    text = ir.to_json_text(doc)
    assert text == ir.to_json_text(build_report(report))                      # deterministic JSON
    assert render_markdown(json.loads(text)) == md == render(report)[1]      # Markdown from the JSON alone
    return doc, md


def test_matching_known_units_verify_agreement(two_comparisons):
    doc, md = exports_are_consistent(two_comparisons)
    pair = compare_pair(doc, two_comparisons)
    assert pair["outcome"] == "agree" and pair["units"] == ["px/s", "px/s"] and pair["reasons"] == []
    cross = doc["verification"]["cross_snapshot"]
    assert cross["status"] == "partially_verified" and cross["unit_conflicts"] == []   # feature pairs: no unit
    assert "| px/s / px/s | agree | - |" in md


def test_known_and_missing_units_render_and_are_not_verified(two_comparisons):
    forged = forge(two_comparisons, "compare_windows", drop_units, k=1)
    doc, md = exports_are_consistent(forged)                                 # no TypeError on null units
    first, second = [e["evidence_id"] for e in forged.evidence if e["tool"] == "compare_windows"]
    pair = compare_pair(doc, forged)
    assert pair["outcome"] == "unit_unverified" and pair["units"] == ["px/s", None]
    assert f"no unit is recorded by {second}" in pair["reasons"][0]
    (records,) = doc["verification"]["cross_snapshot"]["unit_conflicts"]
    assert records == {"feature": "dist_rate", "kind": "partly_unrecorded",
                       "units": [{"unit": "px/s", "evidence_ids": [first]}, {"unit": None, "evidence_ids": [second]}]}
    assert doc["verification"]["cross_snapshot"]["status"] == "not_verified"
    assert f"not recorded in {second}" in section(md, "Executive summary") and "| px/s / not recorded |" in md


def test_conflicting_units_are_not_verified_and_not_compared(two_comparisons):
    forged = forge(two_comparisons, "compare_windows", set_method("units", "mm/s"), k=1)
    doc, md = exports_are_consistent(forged)
    pair = compare_pair(doc, forged)
    assert pair["outcome"] == "unit_conflict" and pair["units"] == ["px/s", "mm/s"]
    assert "values are not compared and no conversion is assumed" in pair["reasons"][0]
    assert doc["verification"]["cross_snapshot"]["status"] == "not_verified"
    assert "unit_conflict: 1" in section(md, "Executive summary")


def test_both_units_unknown_cannot_verify_agreement(two_comparisons):
    forged = forge(forge(two_comparisons, "compare_windows", drop_units, k=0), "compare_windows", drop_units, k=1)
    doc, md = exports_are_consistent(forged)
    pair = compare_pair(doc, forged)
    assert pair["outcome"] == "unit_unverified" and pair["units"] == [None, None]
    assert doc["verification"]["cross_snapshot"]["status"] == "not_verified"
    assert doc["verification"]["cross_snapshot"]["unit_conflicts"] == []    # one value (null): no conflict record
    assert doc["primary_result"]["unit"] is None


def test_several_known_units_are_recorded_deterministically(two_comparisons):
    forged = forge(two_comparisons, "compare_windows", set_method("units", "cm/s"), k=1)
    forged = forge(forged, "compare_windows", set_method("units", "mm/s"), k=0)
    doc, _ = exports_are_consistent(forged)
    (records,) = doc["verification"]["cross_snapshot"]["unit_conflicts"]
    assert [r["unit"] for r in records["units"]] == ["cm/s", "mm/s"] and records["kind"] == "conflicting"


def test_contradiction_with_matching_units_is_still_refused(two_comparisons):
    """Both comparisons record px/s; the second's shared window value is
    changed (aggregates recomputed, re-hashed): refused."""
    forged = forge(two_comparisons, "compare_windows", inflate_pre_window, k=1)
    with pytest.raises(ReportIntegrityError) as e:
        render(forged)
    a, b = [x["evidence_id"] for x in forged.evidence if x["tool"] == "compare_windows"]
    assert any(f"{a} and {b} disagree on dist_rate" in p for p in e.value.problems), e.value.problems


# ---- Stage 8 live-run audit: tracking quality is surfaced from the evidence ----

LIVE_RUN = Path(__file__).resolve().parents[1] / "outputs" / "investigation_live" / "20261019T220059Z_openrouter"


def tracking(doc):
    return doc["limitations"]["tracking"]


def test_contaminated_windows_used_by_the_primary_comparison_are_surfaced(tmp_path):
    env = build(tmp_path, contaminated)
    report = investigate(QUESTION, env[0], env[1], Scripted(*_until_compare(LOOSE), finish(GOOD)), sufficiency=LOOSE)
    doc, md = render(report)
    pid = report.primary_evidence["evidence_id"]
    compare = snapshot(report, "compare_windows")["result"]
    note = next(m["note"] for m in compare["methodology"] if m["name"] == "smoothing_contamination_windows_pre")
    n = next(m["value"] for m in compare["methodology"] if m["name"] == "smoothing_contamination_windows_pre")
    item = next(t for t in tracking(doc) if t["source"] == "methodology.smoothing_contamination_windows_pre")
    assert n >= 1 and item["evidence_id"] == pid and item["primary"] and item["tool_note"] == note   # verbatim note
    summary = section(md, "Executive summary")
    assert "Tracking quality (from the evidence; no severity assigned)" in summary and item["text"] in summary
    assert "### Tracking quality (from the evidence; no severity assigned)" in md and note in md
    assert any(t["source"] == "trace" and "check_tracking_quality was never called" in t["text"] for t in tracking(doc))


def test_a_quality_check_run_replaces_the_not_assessed_notice(tmp_path):
    env = build(tmp_path, contaminated)
    report = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    items = tracking(build_report(report))
    assert not any(t["source"] == "trace" for t in items)
    q = snapshot(report, "check_tracking_quality")
    detected = sum(w["smoothing_contamination"]["status"] == "detected" for w in q["result"]["measurements"]["windows"])
    found = [t for t in items if t["source"] == "measurements.windows[].smoothing_contamination"]
    assert detected and found and found[0]["evidence_id"] == q["evidence_id"] and str(detected) in found[0]["text"]


def test_clean_evidence_lists_only_what_is_recorded(effect_report):
    items = tracking(build_report(effect_report))
    assert all(t["evidence_id"] in {e["evidence_id"] for e in effect_report.evidence} | {None} for t in items)
    for t in items:                                         # every item points at a real recorded fact
        if t["source"].startswith("methodology."):
            e = next(x for x in effect_report.evidence if x["evidence_id"] == t["evidence_id"])
            value = next(m["value"] for m in e["result"]["methodology"] if m["name"] == t["source"].split(".", 1)[1])
            assert value not in (0, 1.0)                     # zero counts / full fractions are not listed


def test_tracking_items_survive_the_json_round_trip(tmp_path):
    env = build(tmp_path, contaminated)
    report = investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)
    doc, md = render(report)
    assert render_markdown(json.loads(ir.to_json_text(doc))) == md
    for t in tracking(doc):
        assert t["text"] in md


@pytest.mark.skipif(not (LIVE_RUN / "investigation_report.json").exists(), reason="live-run artifacts not present")
def test_live_run_contamination_is_now_surfaced_without_touching_the_artifacts(real_env):
    import hashlib
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in LIVE_RUN.iterdir()}
    report = ia.InvestigationReport(**json.loads((LIVE_RUN / "investigation_report.json").read_text()))
    doc, md = render(report, policy=real_env[1])                       # rendered in memory only
    summary = section(md, "Executive summary")
    assert "1 used pre-event window with smoothing contamination" in summary
    assert "check_tracking_quality was never called" in summary
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in LIVE_RUN.iterdir()} == before


# ---- Codex 1 review: tracking notes belong to their windows; three quality-check states ----

def quality_run(tmp_path, window_ids, change=contaminated, name="q"):
    env = build(tmp_path, change, name=name)
    model = Scripted([call("check_tracking_quality", window_ids=window_ids, history_frames=[1])])
    return investigate(QUESTION, env[0], env[1], model, budget=Budget(max_model_turns=1)), env


def window_pair():
    from test_investigation_windows import VAL_A, ep
    return "win:pre:20:0:" + ep(VAL_A, 300), "win:pre:20:0:" + ep(VAL_A, 100)     # clean, contaminated


def test_contamination_note_comes_from_the_contaminated_window_not_window_0(tmp_path):
    clean, flagged = window_pair()
    report, _ = quality_run(tmp_path, [clean, flagged])
    windows = snapshot(report, "check_tracking_quality")["result"]["measurements"]["windows"]
    assert [w["window_id"] for w in windows] == [clean, flagged]
    assert [w["smoothing_contamination"]["status"] for w in windows] == ["none_detected", "detected"]
    marker = "CLEAN WINDOW NOTE - must never be quoted for contamination"

    def relabel_window_0(req, res):
        res["measurements"]["windows"][0]["smoothing_contamination"]["note"] = marker
    forged = forge(report, "check_tracking_quality", relabel_window_0)
    doc, md = render(forged)
    real_note = windows[1]["smoothing_contamination"]["note"]
    (item,) = [t for t in tracking(doc) if t["source"] == "measurements.windows[].smoothing_contamination"]
    q = snapshot(forged, "check_tracking_quality")
    assert item["evidence_id"] == q["evidence_id"] and item["window_ids"] == [flagged]
    assert item["tool_note"] == real_note and item["text"] == "1 window with detected smoothing contamination"
    rendered = {k: v for k, v in json.loads(ir.to_json_text(doc)).items() if k != "evidence_snapshots"}
    assert marker not in json.dumps(rendered["limitations"]) and marker in json.dumps(doc["evidence_snapshots"])
    line = next(l for l in md.splitlines() if "with detected smoothing contamination —" in l)
    assert marker not in line and flagged in line and clean not in line and real_note in line
    assert render_markdown(json.loads(ir.to_json_text(doc))) == md


def two_contaminated(sid, kp, sc):
    from test_investigation_windows import VAL_A
    if sid == VAL_A:
        sc[68, 0, 3] = 0.01          # smoother memory of [80, 100)
        sc[268, 0, 3] = 0.01         # smoother memory of [280, 300)


def test_windows_with_identical_notes_are_grouped(tmp_path):
    clean, flagged = window_pair()
    report, _ = quality_run(tmp_path, [clean, flagged], change=two_contaminated)
    doc, md = render(report)
    (item,) = [t for t in tracking(doc) if t["source"] == "measurements.windows[].smoothing_contamination"]
    assert item["window_ids"] == [clean, flagged] and item["text"].startswith("2 windows")
    assert f"windows: `{clean}`, `{flagged}`" in md


def test_tracking_check_never_called(effect_report_no_quality):
    items = tracking(build_report(effect_report_no_quality))
    assert [t["text"] for t in items if t["source"] in ("trace", "status")] == [
        "check_tracking_quality was never called in this investigation: per-window tracking quality was not "
        "assessed beyond the counts other tools record"]


@pytest.fixture
def effect_report_no_quality(env):
    return investigate(QUESTION, env[0], env[1], Scripted(*_until_compare(LOOSE), finish(GOOD)), sufficiency=LOOSE)


def test_tracking_check_called_but_unavailable(tmp_path, monkeypatch):
    from live_tracker.investigation_events import CalMS21InvestigationTools as Tools

    def unavailable(self, request):
        raise RuntimeError("quality backend unavailable")         # call_tool returns a valid error result
    monkeypatch.setattr(Tools, "check_tracking_quality", unavailable)
    clean, flagged = window_pair()
    report, _ = quality_run(tmp_path, [clean, flagged])
    q = snapshot(report, "check_tracking_quality")
    assert q["status"] == "error"
    doc, md = render(report)
    texts = {t["source"]: t for t in tracking(doc)}
    assert texts["status"]["evidence_id"] == q["evidence_id"]
    assert texts["status"]["text"] == ("check_tracking_quality was called but returned status error (error codes: "
                                       "internal_error); it assessed no window")
    assert texts["trace"]["text"].startswith("check_tracking_quality was called but no call returned an assessment")
    assert "never called" not in md and "was called but returned status error" in section(md, "Executive summary")


def test_tracking_check_with_usable_results_has_no_missing_notice(tmp_path):
    clean, flagged = window_pair()
    report, _ = quality_run(tmp_path, [clean, flagged])
    items = tracking(build_report(report))
    assert not [t for t in items if t["source"] in ("trace", "status")]


def test_a_failed_and_a_successful_quality_check_are_both_reported(tmp_path, monkeypatch):
    from live_tracker.investigation_events import CalMS21InvestigationTools as Tools
    real = Tools.check_tracking_quality
    calls = []

    def flaky(self, request):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return real(self, request)
    monkeypatch.setattr(Tools, "check_tracking_quality", flaky)
    clean, flagged = window_pair()
    env = build(tmp_path, contaminated, name="flaky")
    model = Scripted([call("check_tracking_quality", window_ids=[clean, flagged], history_frames=[1])],
                     [call("check_tracking_quality", window_ids=[flagged], history_frames=[1])])
    report = investigate(QUESTION, env[0], env[1], model, budget=Budget(max_model_turns=2))
    items = tracking(build_report(report))
    assert [t["source"] for t in items if t["source"] in ("trace", "status")] == ["status"]   # attempt + success
    assert any(t["source"] == "measurements.windows[].smoothing_contamination" for t in items)


# ---- coverage per compatible query group, primary coverage from provenance ----

from test_investigation_agent import last_result  # noqa: E402
from test_investigation_windows import VAL_A, VAL_B, VAL_C  # noqa: E402


def coverage(doc):
    return doc["coverage"]


def scripted(env, *turns, turns_budget=None):
    return investigate(QUESTION, env[0], env[1], Scripted(*turns), budget=Budget(max_model_turns=turns_budget or len(turns)))


def test_attack_in_one_session_and_investigation_in_the_others_are_never_combined(env):
    report = scripted(env, [query(sessions=[VAL_A])], [call("query_events", source="validation_annotation",
                                                             behavior="investigation", sessions=[VAL_B, VAL_C])])
    doc, md = render(report, policy=env[1])
    cov = coverage(doc)
    by = {g["behavior"]: g for g in cov["groups"]}
    assert cov["status"] == "per_query_group" and cov["primary"] is None
    assert (by["attack"]["status"], by["attack"]["searched"]) == ("partial", [VAL_A])
    assert (by["investigation"]["status"], by["investigation"]["searched"]) == ("partial", [VAL_B, VAL_C])
    summary = section(md, "Executive summary")
    assert "searched 3 of 3" not in summary and not re.search(r"\bcomplete\b", summary.split("Descriptive only")[0])
    assert "PARTIAL: searched 1 of 3 eligible sessions" in summary and "PARTIAL: searched 2 of 3" in summary
    assert cov["sessions_touched"]["count"] == 3 and "not scientific coverage" in cov["sessions_touched"]["note"]


def test_same_behavior_in_disjoint_sessions_with_compatible_settings_is_combined(env):
    report = scripted(env, [query(sessions=[VAL_A])], [query(sessions=[VAL_B, VAL_C])])
    (g,) = coverage(build_report(report, policy=env[1]))["groups"]
    assert g["status"] == "complete" and g["searched"] == [VAL_A, VAL_B, VAL_C] and len(g["query_evidence_ids"]) == 2


@pytest.mark.parametrize("second", [dict(merge_gap_frames=5), dict(min_duration_frames=10)])
def test_same_behavior_with_incompatible_event_definitions_is_not_combined(env, second):
    report = scripted(env, [query(sessions=[VAL_A])], [query(sessions=[VAL_B, VAL_C], **second)])
    cov = coverage(build_report(report, policy=env[1]))
    assert len(cov["groups"]) == 2 and {g["status"] for g in cov["groups"]} == {"partial"}


def _fake_query(eid, source, sessions, with_episodes=()):
    return {"evidence_id": eid, "tool": "query_events", "status": "ok",
            "request": {"source": source, "behavior": "attack", "sessions": sessions, "merge_gap_frames": None,
                        "min_duration_frames": None},
            "result": {"evidence_id": eid, "status": "ok", "methodology": [], "provenance": {"data": []}, "limitations": [],
                       "measurements": {"sessions_searched": sessions, "episodes": [], "truncated": False,
                                        "sessions_without_episodes": [x for x in sessions if x not in with_episodes]}}}


def test_same_behavior_from_incompatible_sources_is_not_combined(env):
    snaps = {"q1": _fake_query("q1", "validation_annotation", [VAL_A]),
             "q2": _fake_query("q2", "classifier_prediction", [VAL_B, VAL_C])}
    cov = ir._coverage(snaps, env[1], None)
    by = {g["source"]: g for g in cov["groups"]}
    assert by["validation_annotation"]["status"] == "partial"
    assert by["classifier_prediction"]["status"] == "unknown"           # no eligible set for predictions
    assert by["classifier_prediction"]["basis"] == "not established for this source"


def test_primary_without_query_provenance_is_unknown(env):
    """Windows built from explicit episode IDs, no query: never complete."""
    from test_investigation_windows import ep
    feats = call("get_window_features", episode_ids=[ep(VAL_A, 100), ep(VAL_B, 200)],
                 pre_event={"length_frames": 20, "gap_frames": 0},
                 baseline={"length_frames": 20, "clearance_frames": 10, "max_windows_per_session": 500},
                 features=["dist_rate"])
    compare = lambda m: [call("compare_windows", feature="dist_rate",                       # noqa: E731
                              windows_from={"evidence_id": last_result(m)["evidence_id"]})]
    report = investigate(QUESTION, env[0], env[1], Scripted([feats], compare, finish(GOOD)), sufficiency=LOOSE)
    doc, md = render(report, policy=env[1])
    prim = coverage(doc)["primary"]
    assert prim["status"] == "unknown" and "no compatible query_events group" in prim["reason"]
    assert doc["coverage"]["status"] == "unknown" and "not established" in section(md, "Executive summary")


def test_primary_with_a_truncated_query_is_unknown(env):
    turns = [[query(max_episodes=1)]] + _until_compare(LOOSE)[1:] + [finish(GOOD)]
    report = investigate(QUESTION, env[0], env[1], Scripted(*turns), sufficiency=LOOSE)
    cov = coverage(build_report(report, policy=env[1]))
    prim = cov["primary"]
    assert cov["groups"][0]["truncated"] and cov["groups"][0]["status"] == "complete"   # every session searched ...
    assert prim["status"] == "partial" and "truncated" in prim["reason"]               # ... but not every bout


def test_partial_coverage_propagates_through_downstream_tools_without_query(tmp_path):
    from test_investigation_windows import ep
    env = build(tmp_path, sessions=(VAL_A, VAL_B), name="partial")
    tools = CalMS21InvestigationTools.for_validation_cache(env[2], env[3], env[1], coverage="partial")
    feats = call("get_window_features", episode_ids=[ep(VAL_A, 100), ep(VAL_B, 200)],
                 pre_event={"length_frames": 20, "gap_frames": 0},
                 baseline={"length_frames": 20, "clearance_frames": 10, "max_windows_per_session": 500},
                 features=["dist_rate"])
    compare = lambda m: [call("compare_windows", feature="dist_rate",                       # noqa: E731
                              windows_from={"evidence_id": last_result(m)["evidence_id"]})]
    report = investigate(QUESTION, tools, env[1], Scripted([feats], compare, finish(GOOD)), sufficiency=LOOSE)
    for e in report.evidence:                                    # every downstream result records it
        methods = {m["name"]: m["value"] for m in e["result"]["methodology"]}
        assert methods["session_coverage"].startswith("partial") and methods["available_sessions"] == 2
        assert any(l.startswith("PARTIAL COVERAGE") for l in e["result"]["limitations"])
    doc, md = render(report, policy=env[1])
    prim = coverage(doc)["primary"]
    assert prim["status"] == "partial" and "partial cache coverage recorded by" in prim["reason"]
    assert "Session coverage of the primary comparison is PARTIAL" in section(md, "Executive summary")


def test_downstream_tools_on_a_complete_cache_still_need_a_query_for_complete(env):
    from test_investigation_windows import ep
    feats = call("get_window_features", episode_ids=[ep(VAL_A, 100)], pre_event={"length_frames": 20, "gap_frames": 0},
                 baseline={"length_frames": 20, "clearance_frames": 10, "max_windows_per_session": 500},
                 features=["dist_rate"])
    report = scripted(env, [feats])
    methods = {m["name"]: m["value"] for m in report.evidence[0]["result"]["methodology"]}
    assert methods["session_coverage"] == "complete"            # the cache is complete ...
    assert coverage(build_report(report, policy=env[1]))["status"] == "unknown"     # ... but nothing was queried


def test_complete_primary_coverage_from_a_full_compatible_query(effect_report, env):
    cov = coverage(build_report(effect_report, policy=env[1]))
    prim = cov["primary"]
    assert prim["status"] == "complete" == cov["status"] and prim["reason"] is None
    g = cov["groups"][prim["group"]]
    assert g["searched"] == g["eligible"] and prim["query_evidence_ids"] == g["query_evidence_ids"]
    assert all(x["session"] not in prim["contributing"] for x in prim["excluded"])


def test_coverage_without_a_policy_comes_from_the_recorded_metadata(effect_report):
    (g,) = coverage(build_report(effect_report))["groups"]
    assert g["status"] == "complete" and g["eligible"] is None and g["eligible_count"] == 3
    assert "recorded by the tools" in g["basis"]


def test_coverage_metadata_that_disagrees_with_the_policy_is_refused(effect_report):
    other = ic.EvidencePolicy({"train": [], "validation": ["task1/train/SYNTHETIC_val_a"],
                               "test": []}, allow_synthetic=True)
    with pytest.raises(ReportIntegrityError):
        render(effect_report, policy=other)


def test_explicit_partial_cache_with_a_full_query_is_partial(tmp_path):
    env = build(tmp_path, sessions=(VAL_A, VAL_B), name="partial2")
    tools = CalMS21InvestigationTools.for_validation_cache(env[2], env[3], env[1], coverage="partial")
    report = investigate(QUESTION, tools, env[1], Scripted([query()]), budget=Budget(max_model_turns=1))
    (g,) = coverage(build_report(report, policy=env[1]))["groups"]
    assert g["status"] == "partial" and g["available_count"] == 2 and g["not_searched"] == [VAL_C]


@pytest.mark.skipif(not (LIVE_RUN / "investigation_report.json").exists(), reason="live-run artifacts not present")
def test_the_original_live_run_is_reported_as_partial_coverage(real_env):
    import hashlib
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in LIVE_RUN.iterdir()}
    report = ia.InvestigationReport(**json.loads((LIVE_RUN / "investigation_report.json").read_text()))
    doc, md = render(report, policy=real_env[1])
    prim = coverage(doc)["primary"]
    g = coverage(doc)["groups"][prim["group"]]
    assert prim["status"] == "partial" and len(g["searched"]) == 3 and g["eligible_count"] == 14
    assert "PARTIAL: searched 3 of 14 eligible sessions" in section(md, "Executive summary")
    assert coverage(build_report(report))["status"] == "unknown"    # no policy and no recorded metadata
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in LIVE_RUN.iterdir()} == before


def test_real_full_report_coverage_and_overlapping_exclusions(real_env):
    report = investigate(QUESTION, *real_env, Investigator(length=30, clearance=150))
    doc, md = render(report, policy=real_env[1])
    prim = coverage(doc)["primary"]
    g = coverage(doc)["groups"][prim["group"]]
    assert prim["status"] == "complete" and len(g["searched"]) == 14 and len(g["with_behavior"]) == 4
    assert [x.split("/")[-1][:8] for x in prim["contributing"]] == ["mouse005", "mouse064"]
    assert [(x["session"].split("/")[-1][:8], x["reason"]) for x in prim["excluded"]] == [
        ("mouse028", "no eligible baseline window"), ("mouse045", "no eligible baseline window")]
    feats = next(a for a in doc["evidence_appendix"] if a["tool"] == "get_window_features")["summary"]
    pre_reasons = sum(n for k, n in feats["exclusion_reason_counts"].items() if k.startswith("pre_event:"))
    assert feats["excluded_windows"]["pre_event"] == 26 and pre_reasons == 29           # 3 windows have 2 reasons
    assert "can overlap" in feats["exclusion_reason_note"]
    assert any("(reasons can overlap)" in t["text"] for t in doc["limitations"]["tracking"])


def test_a_complete_query_that_does_not_contain_the_compared_episodes_gives_unknown(env):
    """The query searched every session but with a 25-frame minimum duration
    that drops VAL_A's 20-frame bout at frame 100; the comparison uses that bout
    through explicit IDs, so the query's complete coverage does not apply to it."""
    from test_investigation_windows import ep
    feats = call("get_window_features", episode_ids=[ep(VAL_A, 100), ep(VAL_B, 200)],
                 pre_event={"length_frames": 20, "gap_frames": 0},
                 baseline={"length_frames": 20, "clearance_frames": 10, "max_windows_per_session": 500},
                 features=["dist_rate"])
    compare = lambda m: [call("compare_windows", feature="dist_rate",                       # noqa: E731
                              windows_from={"evidence_id": last_result(m)["evidence_id"]})]
    report = investigate(QUESTION, env[0], env[1], Scripted([query(min_duration_frames=25)], [feats], compare,
                                                            finish(GOOD)), sufficiency=LOOSE)
    cov = coverage(build_report(report, policy=env[1]))
    (g,) = cov["groups"]
    compared = snapshot(report, "compare_windows")["request"]["pre_window_ids"]
    assert g["status"] == "complete" and any(w.endswith(ep(VAL_A, 100)) for w in compared)
    assert ep(VAL_A, 100) not in {x["episode_id"] for x in snapshot(report, "query_events")["result"]["measurements"]["episodes"]}
    assert cov["primary"]["status"] == "unknown" and "every compared attack episode" in cov["primary"]["reason"]


# ---- data identity along the primary comparison's provenance chain ----

from live_tracker.investigation_data import INDEX_NAME  # noqa: E402

def _prov(*items):
    return {"evidence_id": "ev:x", "result": {"provenance": {"data": [dict(i) if isinstance(i, dict) else i
                                                                      for i in items]}}}


INDEX = {"kind": "calms21_validation_cache", "path": "/c/extraction_index.json", "sha256": "a" * 64, "description": "i"}
SOURCE = {"kind": "calms21_validation_cache", "path": "/src/train.json", "sha256": "b" * 64, "description": "s"}
S1 = {"kind": "calms21_validation_cache", "path": "/c/s1.json", "sha256": "c" * 64, "description": "s1"}
S2 = {"kind": "calms21_validation_cache", "path": "/c/s2.json", "sha256": "d" * 64, "description": "s2"}


@pytest.mark.parametrize("a, b, expected", [
    (_prov(INDEX, SOURCE, S1, S2), _prov(INDEX, SOURCE, S1), None),          # a query lists more session files
    (_prov(INDEX, SOURCE, S1), _prov(INDEX, SOURCE, S1), None),
    (_prov(), _prov(INDEX, SOURCE, S1), "not recorded with hashes"),         # missing provenance
    (_prov(INDEX, {**SOURCE, "sha256": None}), _prov(INDEX, SOURCE), "not recorded with hashes"),
    (_prov(SOURCE, S1), _prov(SOURCE, S1), "exactly one verified extraction index"),   # no index identity
    (_prov(INDEX, SOURCE), _prov({**INDEX, "sha256": "e" * 64}, SOURCE), "differs"),   # other index / manifest
    (_prov(INDEX, SOURCE, S1), _prov(INDEX, SOURCE, {**S1, "sha256": "f" * 64}), "differs"),  # conflicting file
    (_prov(INDEX, SOURCE), _prov({**INDEX, "kind": "synthetic_fixture"}, SOURCE), "differs"),
    (_prov(INDEX, S1), _prov({**INDEX, "path": "/other/extraction_index.json", "sha256": "e" * 64}, S1),
     "differs"),                                                              # another cache: index digests differ
])
def test_data_identity_helper(a, b, expected):
    problem = ir._data_identity_problem(a, b)
    assert (problem is None) if expected is None else (expected in problem)


def _drop_feature_provenance(req, res):
    res["provenance"]["data"] = []


def test_codex_counterexample_missing_feature_provenance_is_unknown(effect_report, env):
    assert coverage(build_report(effect_report, policy=env[1]))["primary"]["status"] == "complete"   # before
    forged = forge(effect_report, "get_window_features", _drop_feature_provenance)   # contract-valid, re-hashed
    doc, md = render(forged, policy=env[1])
    prim = coverage(doc)["primary"]
    assert prim["status"] == "unknown" and "not recorded with hashes" in prim["reason"]
    assert doc["verification"]["cross_snapshot"]["status"] == "not_verified"          # the same identity notion
    pair = doc["verification"]["cross_snapshot"]["pairs"][0]
    assert "data provenance is not recorded with hashes" in pair["reasons"]
    assert "Session coverage of the primary comparison not established" in section(md, "Executive summary")
    (g,) = coverage(doc)["groups"]
    assert g["status"] == "complete"                         # the query group's own coverage is unchanged


@pytest.mark.parametrize("tool", ["get_window_features", "query_events", "compare_windows"])
def test_conflicting_provenance_never_establishes_primary_coverage(effect_report, env, tool):
    def other_index(req, res):
        for d in res["provenance"]["data"]:
            if d["path"].endswith(INDEX_NAME):
                d["sha256"] = "0" * 64                       # another index / manifest binding
    forged = forge(effect_report, tool, other_index)
    doc, md = render(forged, policy=env[1])
    prim = coverage(doc)["primary"]
    assert prim["status"] == "unknown" and "data provenance differs" in prim["reason"]
    assert "complete" not in section(md, "Executive summary").split("\n")[2]


def test_a_conflicting_session_file_hash_never_establishes_primary_coverage(effect_report, env):
    def other_file(req, res):
        d = next(x for x in res["provenance"]["data"] if not x["path"].endswith(INDEX_NAME)
                 and "source file" not in x["description"])
        d["sha256"] = "1" * 64
    prim = coverage(build_report(forge(effect_report, "get_window_features", other_file), policy=env[1]))["primary"]
    assert prim["status"] == "unknown" and "data provenance differs" in prim["reason"]


def test_compatible_provenance_with_missing_units_keeps_complete_coverage(effect_report, env):
    forged = forge(effect_report, "compare_windows", lambda req, res: res["methodology"].__setitem__(
        slice(None), [x for x in res["methodology"] if x["name"] != "units"]))
    doc, _ = render(forged, policy=env[1])
    assert coverage(doc)["primary"]["status"] == "complete"                           # identity, not units
    assert doc["primary_result"]["unit"] is None


def test_queries_over_different_data_are_never_combined_into_complete(env):
    report = scripted(env, [query(sessions=[VAL_A])], [query(sessions=[VAL_B, VAL_C])])

    def other_index(req, res):
        for d in res["provenance"]["data"]:
            if d["path"].endswith(INDEX_NAME):
                d["sha256"] = "2" * 64
    forged = forge(report, "query_events", other_index, k=1)
    (g,) = coverage(build_report(forged, policy=env[1]))["groups"]
    assert g["status"] == "unknown" and "do not share data identity" in g["basis"]


def test_identity_failure_keeps_partial_cache_disclosure_in_the_reason(tmp_path):
    env = build(tmp_path, sessions=(VAL_A, VAL_B), name="partial_id")
    tools = CalMS21InvestigationTools.for_validation_cache(env[2], env[3], env[1], coverage="partial")
    turns = [[query()]] + _until_compare(LOOSE)[1:] + [finish(GOOD)]
    report = investigate(QUESTION, tools, env[1], Scripted(*turns), sufficiency=LOOSE)
    assert coverage(build_report(report, policy=env[1]))["primary"]["status"] == "partial"   # disclosed as before
    forged = forge(report, "get_window_features", _drop_feature_provenance)
    prim = coverage(build_report(forged, policy=env[1]))["primary"]
    assert prim["status"] == "unknown" and "partial cache coverage recorded by" in prim["reason"]


# ---- provenance entries are validated before they are indexed by path ----

CONFLICT = {**SOURCE, "sha256": "9" * 64}


@pytest.mark.parametrize("entries", [(INDEX, CONFLICT, SOURCE), (INDEX, SOURCE, CONFLICT)])
def test_conflicting_duplicate_paths_refuse_identity_in_either_order(entries):
    for a, b in ((_prov(*entries), _prov(INDEX, SOURCE)), (_prov(INDEX, SOURCE), _prov(*entries))):  # both sides
        assert "conflicting entries for /src/train.json" in ir._data_identity_problem(a, b)


def test_conflicting_kind_for_one_path_refuses_identity():
    other_kind = {**SOURCE, "kind": "synthetic_fixture"}
    assert "conflicting entries" in ir._data_identity_problem(_prov(INDEX, SOURCE, other_kind), _prov(INDEX, SOURCE))


def test_identical_duplicates_and_reordering_are_accepted():
    assert ir._data_identity_problem(_prov(INDEX, SOURCE, SOURCE, S1), _prov(S1, SOURCE, INDEX)) is None
    same_after_normalizing = {**SOURCE, "path": "/src/./train.json"}
    assert ir._data_identity_problem(_prov(INDEX, SOURCE, same_after_normalizing), _prov(INDEX, SOURCE)) is None
    assert "conflicting" in ir._data_identity_problem(
        _prov(INDEX, SOURCE, {**same_after_normalizing, "sha256": "8" * 64}), _prov(INDEX, SOURCE))


@pytest.mark.parametrize("bad, problem", [
    ({**SOURCE, "path": None}, "malformed entry"), ({**SOURCE, "path": "  "}, "malformed entry"),
    ({**SOURCE, "kind": ""}, "malformed entry"), ({k: v for k, v in SOURCE.items() if k != "kind"}, "malformed entry"),
    ({**SOURCE, "sha256": ""}, "not recorded with hashes"), ({**SOURCE, "sha256": "xyz"}, "malformed entry"),
    ({**SOURCE, "sha256": 7}, "malformed entry"), ("not an object", "malformed entry"),
])
def test_malformed_entries_refuse_identity_without_raising(bad, problem):
    for a, b in ((_prov(INDEX, bad), _prov(INDEX, SOURCE)), (_prov(INDEX, SOURCE), _prov(INDEX, bad))):
        assert problem in ir._data_identity_problem(a, b)
    assert "missing or malformed" in ir._data_identity_problem({"evidence_id": "x", "result": {}}, _prov(INDEX))


@pytest.mark.parametrize("first", [True, False])
def test_conflicting_duplicate_provenance_cannot_establish_complete_primary_coverage(effect_report, env, first):
    """A re-hashed, contract-valid features snapshot that records one session
    file twice with different hashes (in either order)."""
    def duplicate_conflict(req, res):
        data = res["provenance"]["data"]
        session = next(d for d in data if not d["path"].endswith(INDEX_NAME) and "source file" not in d["description"])
        clash = {**session, "sha256": "7" * 64}
        data.insert(data.index(session) + (0 if first else 1), clash)
    assert coverage(build_report(effect_report, policy=env[1]))["primary"]["status"] == "complete"
    forged = forge(effect_report, "get_window_features", duplicate_conflict)    # checks every snapshot's contract
    doc, md = render(forged, policy=env[1])
    prim = coverage(doc)["primary"]
    assert prim["status"] == "unknown" and "conflicting entries" in prim["reason"]
    assert "Session coverage of the primary comparison not established" in section(md, "Executive summary")
    assert doc["verification"]["cross_snapshot"]["status"] == "not_verified"
