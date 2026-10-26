"""Findings v2 checkpoint M4: final isolation and end-to-end verification (new file).

The whole offline pipeline, per scenario: research question -> the model's
own tool choices -> evidence -> structured finish -> M0/M1 verification at
the freeze point -> InvestigationReportV2 -> saved Stage 6 JSON -> v2
document -> saved report JSON -> Markdown re-derived from the saved JSON
alone (verify_saved_v2). And the M4 gate: no findings/2 path imports the
legacy lexical checks (investigation_lexical), which stay in use for
explicit legacy/1 runs.

SYNTHETIC fixtures only; no model, network or paid provider is used.
"""

import ast
import copy
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_claims as cl
import live_tracker.investigation_context as ctx
import live_tracker.investigation_live as live
import live_tracker.investigation_report as ir
import live_tracker.investigation_report_v2 as v2
from live_tracker.investigation_agent import Budget
from live_tracker.investigation_contracts import PROVISIONAL_SUFFICIENCY
from test_investigation_agent import (LOOSE, QUESTION, Scripted, call, contaminated, last_result,  # noqa: F401
                                      last_result_of, no_network, obs, query, windows_of)
from test_investigation_findings_v2 import C, InvestigatorV2, finish
from test_investigation_windows import build

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "live_tracker"
LEXICAL = "live_tracker.investigation_lexical"
BANNED = re.compile(r"\b(proves?|proven|shows that|demonstrates?|establishes|causes|significant(ly)?|"
                    r"predicts?|confirms?|large|small|strong(ly)?|weak(ly)?|substantial(ly)?|dramatic(ally)?|"
                    r"huge|clearly|definitely)\b", re.I)


def tools_used(report):
    return [t["tool"] for t in report.trace if t["kind"] in ("tool_call", "finish")]


def pipeline(report, policy, tmp_path):
    """Stage 6 report -> saved JSON -> v2 document -> saved JSON -> Markdown
    from the saved document alone. Returns (doc, markdown, saved document)."""
    stage6 = tmp_path / "investigation_report.json"
    stage6.write_text(json.dumps(report.to_json(), sort_keys=True, indent=2))
    doc, md = v2.render_v2(json.loads(stage6.read_text()), policy=policy)
    saved = tmp_path / "report.json"
    saved.write_text(ir.to_json_text(doc))
    loaded = json.loads(saved.read_text())
    assert v2.render_markdown_v2(loaded, policy=policy) == md                      # independently re-derived
    assert ir.to_json_text(v2.verify_saved_v2(loaded, policy=policy)) == saved.read_text()
    assert doc["verification"]["fingerprint"]["recomputed"] == report.fingerprint   # Stage 6 == Stage 7
    reasoning = [t["text"] for t in report.trace if t["kind"] == "model_turn" and t.get("text")]
    assert not any(r in md for r in reasoning)                                     # model text never rendered
    return doc, md, loaded


def no_data_model():
    def excluded_only(m):                      # the model's own (odd but allowed) choice: excluded windows only
        ws = windows_of(last_result(m))
        pre = [w["window_id"] for w in ws if w["kind"] == "pre_event" and w["status"] == "excluded"]
        base = [w["window_id"] for w in ws if w["kind"] == "baseline" and "overlaps_event" in w["exclusion_reasons"]]
        return [call("compare_windows", feature="dist_rate", pre_window_ids=pre, baseline_window_ids=base)]

    def stop(m):
        c = last_result_of(m, "compare_windows")
        return [finish(finding="insufficient_evidence", primary_evidence_id=c["evidence_id"],
                       observations=[obs(c, "status")], claims=[C("comparison_no_data", c["evidence_id"])])]
    return Scripted([query()], lambda m: [call(
        "get_window_features", episodes_from={"evidence_id": last_result(m)["evidence_id"]},
        pre_event={"length_frames": 20, "gap_frames": 0},
        baseline={"length_frames": 20, "clearance_frames": 10, "max_windows_per_session": 500},
        features=["dist_rate"])], excluded_only, stop)


def recovering_model():
    """Two refused finishes (prose, then an invented reference), then a valid one."""
    def prose(m):
        c = last_result_of(m, "compare_windows")
        return [call(ia.FINISH, finding="effect_described", primary_evidence_id=c["evidence_id"], observations=[],
                     claims=[C("descriptive_effect", c["evidence_id"])], next_investigations=[],
                     interpretations=[{"statement": "Approach causes attack.", "evidence_ids": [c["evidence_id"]]}])]

    def invented(m):
        c = last_result_of(m, "compare_windows")
        return [finish(finding="effect_described", primary_evidence_id=c["evidence_id"],
                       claims=[C("descriptive_effect", "ev:compare_windows:" + "9" * 24)])]

    def good(m):
        c = last_result_of(m, "compare_windows")
        return [finish(finding="effect_described", primary_evidence_id=c["evidence_id"],
                       observations=[obs(c, "measurements.effect.value")], claims=[C("descriptive_effect", c["evidence_id"])])]
    from test_investigation_agent import _until_compare
    return Scripted(*_until_compare(LOOSE), prose, invented, good)


SCENARIOS = {
    "descriptive": lambda e: (e["clean"], InvestigatorV2(), {"sufficiency": LOOSE}),
    "insufficient": lambda e: (e["clean"], InvestigatorV2(), {"sufficiency": PROVISIONAL_SUFFICIENCY}),
    "no_data": lambda e: (e["clean"], no_data_model(), {"sufficiency": LOOSE, "budget": Budget(max_model_turns=4)}),
    "contaminated": lambda e: (e["dirty"], InvestigatorV2(), {"sufficiency": LOOSE}),
    "budget_stopped": lambda e: (e["clean"], InvestigatorV2(), {"sufficiency": LOOSE, "budget": Budget(max_model_turns=2)}),
    "recovered": lambda e: (e["clean"], recovering_model(), {"sufficiency": LOOSE, "budget": Budget(max_model_turns=7)}),
}
EXPECTED = {  # finding, publication, the claim types the report carries
    "descriptive": ("effect_described", "publishable", ["descriptive_effect", "coverage"]),
    "insufficient": ("insufficient_evidence", "publishable", ["sufficiency_not_met", "methodological_limitation"]),
    "no_data": ("insufficient_evidence", "publishable", ["comparison_no_data"]),
    "contaminated": ("effect_described", "publishable", ["descriptive_effect", "coverage",
                                                         "sensitivity_excluding_flagged"]),
    "budget_stopped": ("incomplete", "record_only", []),
    "recovered": ("effect_described", "publishable", ["descriptive_effect"]),
}


@pytest.fixture(scope="module")
def envs(tmp_path_factory):
    base = tmp_path_factory.mktemp("e2e")
    return {"clean": build(base), "dirty": build(base, contaminated, name="dirty")}


@pytest.fixture(scope="module")
def runs(envs):
    out = {}
    for name, make in SCENARIOS.items():
        env, model, kw = make(envs)
        out[name] = (ia.investigate(QUESTION, env[0], env[1], model, **kw), env[1])
    return out


# ======================================================================
# 1. the offline pipeline, scenario by scenario
# ======================================================================

@pytest.mark.parametrize("name", list(SCENARIOS))
def test_end_to_end_pipeline(runs, name, tmp_path):
    report, policy = runs[name]
    assert isinstance(report, ia.InvestigationReportV2)
    doc, md, _ = pipeline(report, policy, tmp_path)
    finding, publication, types = EXPECTED[name]
    assert (doc["status"]["finding"], doc["status"]["publication"]) == (finding, publication)
    assert [c["type"] for c in doc["claims"]] == types
    assert doc["format"]["findings_format"] == "findings/2" and doc["verification"]["authorization"] == "rechecked"
    body = md.replace(ir.DESCRIPTIVE_POLICY, "").replace(QUESTION, "")
    assert not BANNED.search(body), BANNED.search(body)
    if finding == "incomplete":
        assert report.claims == [] and "No finding: the run is incomplete." in md


def test_scenario_specifics(runs):
    sens = next(c for c in runs["contaminated"][0].claims if c["type"] == "sensitivity_excluding_flagged")
    assert len(sens["evidence_ids"]) == 3
    doc = v2.build_report_v2(runs["contaminated"][0], policy=runs["contaminated"][1])
    s = next(c["sentence"] for c in doc["claims"] if c["type"] == "sensitivity_excluding_flagged")
    assert "it is not an assessment of robustness" in s and "not scientific agreement" in s
    assert not any("Sensitivity" in x for x in doc["executive_summary"])
    rec = runs["recovered"][0]
    finishes = [t for t in rec.trace if t["kind"] == "finish"]
    assert [t["outcome"] for t in finishes] == ["refused", "refused", "accepted"]
    assert "not part of the findings/2" in finishes[0]["message"] and "not evidence of this run" in finishes[1]["message"]
    assert rec.claims == finishes[2]["arguments"]["claims"]
    stopped = runs["budget_stopped"][0]
    assert stopped.stopping_reason == "max_model_turns" and not stopped.finished


def test_no_hard_coded_tool_sequence(runs, tmp_path, monkeypatch):
    pytest.importorskip("anthropic")             # live.dry_run drives the SDK: the investigation extra
    pytest.importorskip("httpx2")
    sequences = {name: tuple(tools_used(r)) for name, (r, _) in runs.items()}
    assert len(set(sequences.values())) >= 4, sequences                           # the evidence drives the path
    assert sequences["descriptive"] != sequences["contaminated"]                   # same investigator, other data
    assert sequences["descriptive"] != sequences["insufficient"]                  # other criteria, other follow-up
    assert runs["descriptive"][0].finding != runs["insufficient"][0].finding
    e = build(tmp_path)                                                           # the Stage 8 emulator adapts too
    monkeypatch.setattr(live, "_tools_and_policy", lambda plan: (e[0], e[1]))
    out = live.dry_run()
    assert out["tools_used"].count("get_window_features") == 2 and out["report_verified"]


# ======================================================================
# 2. integrity regression on the saved documents
# ======================================================================

TAMPER = {
    "invented evidence reference": lambda d: d["canonical"]["claims"].append(C("no_comparison")) or
    d["canonical"]["claims"][0].update(evidence_ids=["ev:compare_windows:" + "9" * 24]),
    "model-authored value": lambda d: d["canonical"]["evidence"][-1]["result"].update(status="ok") or
    d["canonical"]["evidence"][-1]["result"]["limitations"].append("The effect is -100 px/s."),
    # flip the publication status / completion to the other state (whichever the document has)
    "publication flipped": lambda d: d["status"].update(
        publication="diagnostic" if d["status"]["publication"] != "diagnostic" else "publishable"),
    "completion flipped": lambda d: d["canonical"].update(
        **({"finished": False, "stopping_reason": "max_model_turns"} if d["canonical"]["finished"] else
           {"finished": True, "stopping_reason": "finished"})),
    "sentence edit": lambda d: d["executive_summary"].append("This proves approach causes attack."),
    "fingerprint edit": lambda d: d["canonical"].update(fingerprint="0" * 64),
    "trace prefix edit": lambda d: d["canonical"]["trace"].insert(-1, copy.deepcopy(
        next(t for t in d["canonical"]["trace"] if t["kind"] == "tool_call"))),
    "evidence removed": lambda d: d["canonical"]["evidence"].pop(),
    "legacy prose added": lambda d: d["canonical"].update(interpretations=[{"statement": "faster approach"}]),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("tamper", list(TAMPER))
def test_saved_documents_resist_tampering(runs, name, tamper):
    report, policy = runs[name]
    saved = json.loads(ir.to_json_text(v2.build_report_v2(report, policy=policy)))
    bad = copy.deepcopy(saved)
    TAMPER[tamper](bad)
    with pytest.raises(ir.ReportIntegrityError):
        v2.render_markdown_v2(bad, policy=policy)
    assert v2.render_markdown_v2(copy.deepcopy(json.loads(ir.to_json_text(v2.build_report_v2(report, policy=policy)))),
                                 policy=policy)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_authorization_rules(runs, name):
    report, policy = runs[name]
    publishable = v2.build_report_v2(report, policy=policy)
    diagnostic = v2.build_report_v2(report)
    assert diagnostic["status"]["publication"] == "diagnostic"
    assert diagnostic["verification"]["authorization"] == "not_rechecked"
    with pytest.raises(ir.ReportIntegrityError):                       # a policy-checked document needs the policy
        v2.render_markdown_v2(json.loads(ir.to_json_text(publishable)))
    assert publishable["status"]["publication"] in ("publishable", "record_only")
    assert (publishable["status"]["publication"] == "record_only") == (report.finding == "incomplete")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_prefix_completed_trace_and_frozen_fingerprint_agree(runs, name):
    report, policy = runs[name]
    proposal = {k: getattr(report, k) for k in cl.PROPOSAL_KEYS}
    if report.finished:
        frozen = {"proposal": proposal, "evidence": ctx.canonical_bytes(report.evidence),
                  "prefix": ctx.canonical_bytes(report.trace[:-1]), "fingerprint": report.fingerprint}
        assert ia.finished_v2_problems(frozen, report.evidence, report.trace, policy, "finished") == []
        prefix = cl.validate_findings(proposal, evidence=report.evidence, trace=report.trace[:-1], policy=policy,
                                      mode="prefix")
        assert prefix.accepted and prefix.fingerprint == report.fingerprint
    else:
        r = cl.incomplete_findings(evidence=report.evidence, trace=report.trace, policy=policy,
                                   stopping_reason=report.stopping_reason)
        assert r.accepted and r.fingerprint == report.fingerprint


def test_malformed_prose_cannot_enter_any_finish_slot(envs):
    env = envs["clean"]
    slots = [  # every string slot of the finish, carrying prose
        lambda c: {"primary_evidence_id": "the attack comparison"},
        lambda c: {"observations": [{"evidence_id": c["evidence_id"], "field": "effect was large and negative"}]},
        lambda c: {"observations": [{"evidence_id": "approach causes attack", "field": "status"}]},
        lambda c: {"claims": [{"type": "descriptive_effect", "evidence_ids": ["it is significant"], "params": {}}]},
        lambda c: {"claims": [C("coverage", aspect="the whole population")]},
        lambda c: {"claims": [C("descriptive_effect", c["evidence_id"]), C("coverage", aspect="primary_comparison")],
                   "next_investigations": [{"action": "include_more_sessions", "params": {}, "because": [0],
                                            "why": "to prove it"}]},
    ]
    for slot in slots:
        def bad(m, slot=slot):
            c = last_result_of(m, "compare_windows")
            args = {"finding": "effect_described", "primary_evidence_id": c["evidence_id"], "observations": [],
                    "claims": [C("descriptive_effect", c["evidence_id"])], "next_investigations": []}
            return [call(ia.FINISH, **{**args, **slot(c)})]
        from test_investigation_agent import _until_compare
        r = ia.investigate(QUESTION, env[0], env[1], Scripted(*_until_compare(LOOSE), bad), sufficiency=LOOSE,
                           budget=Budget(max_model_turns=4))
        assert not r.finished and r.claims == [] and r.finding == "incomplete"
        assert [t["outcome"] for t in r.trace if t["kind"] == "finish"] == ["refused"]


# ======================================================================
# 3. the M4 gate: findings/2 never imports the legacy lexical checks
# ======================================================================

V2_PATH_MODULES = ("investigation_agent", "investigation_claims", "investigation_context", "investigation_report_v2",
                   "investigation_live", "investigation_report", "investigation_contracts", "investigation_events",
                   "investigation_windows", "investigation_compare", "investigation_data")


def test_no_module_level_import_of_the_lexical_checks():
    def imports_lexical(n):
        return isinstance(n, (ast.Import, ast.ImportFrom)) and any(
            "investigation_lexical" in x for x in [getattr(n, "module", None) or ""] + [a.name for a in n.names])
    lazy = []
    for name in V2_PATH_MODULES:
        tree = ast.parse((SRC / f"{name}.py").read_text())
        assert not any(imports_lexical(n) for n in tree.body), name    # module level: never
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for n in ast.walk(fn):
                    if isinstance(n, ast.ImportFrom) and "investigation_lexical" in ((n.module or "") +
                                                                                     "".join(a.name for a in n.names)):
                        lazy.append(f"{name}.{fn.name}")
    # only the two legacy checks (and the module attribute shim for legacy names) import it, lazily
    assert sorted(lazy) == ["investigation_agent.__getattr__", "investigation_agent.verify_final",
                            "investigation_report._check"]


def test_a_full_v2_pipeline_never_loads_the_lexical_checks(tmp_path):
    pytest.importorskip("anthropic")             # live.dry_run drives the SDK: the investigation extra
    pytest.importorskip("httpx2")
    script = textwrap.dedent(f"""
        import json, sys, socket
        sys.path.insert(0, {str(ROOT / 'tests')!r})
        socket.socket.connect = None
        import live_tracker.investigation_agent as ia
        import live_tracker.investigation_live as live
        import live_tracker.investigation_report as ir
        import live_tracker.investigation_report_v2 as v2
        from test_investigation_agent import LOOSE, QUESTION, contaminated
        from test_investigation_findings_v2 import InvestigatorV2
        from test_investigation_windows import build
        from pathlib import Path
        tmp = Path({str(tmp_path)!r})
        for name, env in (("clean", build(tmp)), ("dirty", build(tmp, contaminated, name="dirty"))):
            r = ia.investigate(QUESTION, env[0], env[1], InvestigatorV2(), sufficiency=LOOSE)
            doc, md = v2.render_v2(json.loads(json.dumps(r.to_json())), policy=env[1])
            assert v2.render_markdown_v2(json.loads(ir.to_json_text(doc)), policy=env[1]) == md
        e = build(tmp, name="emulated")
        live._tools_and_policy = lambda plan: (e[0], e[1])
        out = live.dry_run()
        assert out["report_verified"] and out["format"]["findings_format"] == "findings/2"
        print(json.dumps({{"lexical_loaded": {LEXICAL!r} in sys.modules}}))
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"lexical_loaded": False}


def test_explicit_legacy_runs_still_use_the_lexical_checks(envs, tmp_path):
    env = envs["clean"]

    def numbered(m):
        c = last_result_of(m, "compare_windows")
        return [call(ia.FINISH, finding="inconclusive", observations=[obs(c, "status")],
                     interpretations=[{"statement": "Approach was 3 times faster.", "evidence_ids": [c["evidence_id"]]}])]

    def clean_(m):
        c = last_result_of(m, "compare_windows")
        return [call(ia.FINISH, finding="inconclusive", observations=[obs(c, "status")])]
    from test_investigation_agent import _until_compare
    legacy = ia.investigate(QUESTION, env[0], env[1], Scripted(*_until_compare(LOOSE), numbered, clean_),
                            sufficiency=LOOSE, findings_format="legacy/1")
    finishes = [t for t in legacy.trace if t["kind"] == "finish"]
    assert isinstance(legacy, ia.InvestigationReport) and legacy.finished
    assert finishes[0]["outcome"] == "refused" and "number" in finishes[0]["message"]      # lexical, as before
    rendered = v2.render_payload(json.loads(json.dumps(legacy.to_json())), policy=env[1])
    assert (rendered.format, rendered.verification) == ("legacy/1", "legacy_verified")
    assert LEXICAL in sys.modules
    with pytest.raises(ir.ReportIntegrityError):
        v2.build_report_v2(legacy.to_json(), policy=env[1])                               # never upgraded
    with pytest.raises(ir.ReportIntegrityError):
        ir.build_report(ia.investigate(QUESTION, env[0], env[1], InvestigatorV2(), sufficiency=LOOSE), policy=env[1])


def test_legacy_lexical_names_keep_resolving_through_the_agent_module():
    import live_tracker.investigation_lexical as lx
    assert ia._prose_problems is lx._prose_problems is lx.prose_problems and ia.MAX_STATEMENT == lx.MAX_STATEMENT
    assert ia._prose_problems("The p-value was small.", "inconclusive")
    with pytest.raises(AttributeError):
        ia.not_a_lexical_name
