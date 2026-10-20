"""Tests for the Findings v2 M0 shared validation foundation (new file).

investigation_context: shared field grammar and resolver, snapshot validation
and authorization assessment, execution assessment (prefix / completed
trace), strict format detection, recursive freezing and the canonical
content fingerprint; plus the pinned historical Stage 8 artifacts and the
byte identity of their legacy rendering. No model or network is used.
"""

import copy
import dataclasses
import hashlib
import json
import math
import types
from pathlib import Path

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_context as ctx
import live_tracker.investigation_contracts as ic
import live_tracker.investigation_report as ir
from live_tracker.investigation_agent import Budget, investigate
from test_investigation_agent import LOOSE, QUESTION, Investigator, Scripted, _until_compare, no_network  # noqa: F401
from test_investigation_windows import MANIFEST, build

ROOT = Path(__file__).resolve().parents[1]
LIVE_RUN = ROOT / "outputs" / "investigation_live" / "20261019T220059Z_openrouter"
REAL_MANIFEST = ROOT / "splits" / "split_manifest.json"
ARTIFACTS = {
    "investigation_report.json": "fbda46968830ecc73ae3c120d3f11ac3bd09a174c73db5c43b5df535ff4dc1cb",
    "report.json": "db308eddbadf644a00df42339df79b12e346c082e1b1f7fddfdc126b163b5fdf",
    "report.md": "f98b239e3e46f9f9e2766ad09223ba6e9b098c065e8135906dbe8ee947689d06",
    "run.json": "479338031d70dd7be372f0b086d66fabceca09d0ca41cfb82bcf5390acf8360c",
}
# legacy rendering of the live run (JSON text, Markdown), fixed at checkpoint M0
LEGACY_RENDER = {
    "policy": ("23e20d518cdf7ae6912a6a8ae526379b23165e47fd4f00c7866f45dce586b389",
               "126094a07e3a1a4c22d4824c819540545dacf92372f0336e25514f9c2a8297ba"),
    "no_policy": ("95c9aa2ca3e7853f3de5ad756fefe1f8567eab3b22296e0aedb0e4066b0bcbcb",
                  "ba1797d8870c00e3fa3bf8c25915491927a3e6d28116c8770b922fcc517e8dd5"),
}
needs_live = pytest.mark.skipif(not all((LIVE_RUN / n).exists() for n in ARTIFACTS) or not REAL_MANIFEST.exists(),
                                reason="original live-run artifacts or manifest not present")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


@pytest.fixture
def env(tmp_path):
    return build(tmp_path)


@pytest.fixture
def finished(env):
    return investigate(QUESTION, env[0], env[1], Investigator(), sufficiency=LOOSE)


@pytest.fixture
def incomplete(env):
    return investigate(QUESTION, env[0], env[1], Scripted(*_until_compare(LOOSE)), sufficiency=LOOSE,
                       budget=Budget(max_model_turns=3))


def snaps_of(report):
    return {e["evidence_id"]: e for e in report.evidence}


# ---- historical artifacts and legacy byte identity ----

@needs_live
def test_historical_artifacts_are_pinned():
    assert {n: sha((LIVE_RUN / n).read_bytes()) for n in ARTIFACTS} == ARTIFACTS


@needs_live
def test_legacy_rendering_of_the_live_run_is_byte_identical():
    report = ir.load_report(json.loads((LIVE_RUN / "investigation_report.json").read_text()))
    policy = ic.EvidencePolicy.from_manifest(REAL_MANIFEST)
    for label, p in (("policy", policy), ("no_policy", None)):
        doc, md = ir.render(report, policy=p)
        assert (sha(ir.to_json_text(doc)), sha(md)) == LEGACY_RENDER[label], label
    assert {n: sha((LIVE_RUN / n).read_bytes()) for n in ARTIFACTS} == ARTIFACTS     # nothing was written


def test_legacy_rendering_is_deterministic_and_round_trips(finished, env):
    doc, md = ir.render(finished, policy=env[1])
    again = ir.load_report(json.loads(json.dumps(finished.to_json())))
    doc2, md2 = ir.render(again, policy=env[1])
    assert ir.to_json_text(doc) == ir.to_json_text(doc2) and md == md2


# ---- strict format detection ----

def legacy_payload(report):
    return report.to_json()


def test_legacy_payload_is_detected(finished):
    assert ctx.detect_format(legacy_payload(finished)) == ctx.LEGACY_FORMAT
    assert ir.report_format(finished) == ctx.LEGACY_FORMAT


@pytest.mark.parametrize("field", ctx.V2_ONLY_FIELDS[1:])
def test_missing_format_with_any_v2_field_is_a_mixed_payload(finished, field):
    data = {**legacy_payload(finished), field: [] if field in ("claims", "next_investigations") else "x"}
    with pytest.raises(ctx.FormatError, match="mixed payload"):
        ctx.detect_format(data)
    with pytest.raises(ir.ReportIntegrityError, match="mixed payload"):
        ir.load_report(data)


V2_MINIMAL = {"findings_format": "findings/2", "registry_version": "claims/2.0", "grid_version": "none",
              "claims": [], "next_investigations": [], "fingerprint": "0" * 64}


@pytest.mark.parametrize("change, problem", [
    ({"findings_format": "findings/3"}, "unknown findings_format"),
    ({"findings_format": None}, "unknown findings_format"),
    ({"registry_version": "claims/9.9"}, "unknown registry_version"),
    ({"registry_version": None}, "unknown registry_version"),
    ({"grid_version": "window-grid/1"}, "unknown grid_version"),
    ({"interpretations": [{"statement": "x"}]}, "legacy model-authored prose"),
    ({"next_steps": ["x"]}, "legacy model-authored prose"),
    ({"limitations": [{"source": "model", "text": "x"}]}, "legacy model-authored prose"),
])
def test_unknown_or_mixed_v2_payloads_are_refused(change, problem):
    with pytest.raises(ctx.FormatError, match=problem):
        ctx.detect_format({**V2_MINIMAL, **change})


def test_known_v2_payload_is_detected_but_not_rendered_before_m2():
    assert ctx.detect_format(V2_MINIMAL) == ctx.V2_FORMAT
    with pytest.raises(ir.ReportIntegrityError, match="not rendered before checkpoint M2"):
        ir.load_report(V2_MINIMAL)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(unexpected=1), lambda d: d.pop("evidence"), lambda d: d.pop("finding")])
def test_payloads_that_are_not_exactly_legacy_are_refused(finished, mutate):
    data = legacy_payload(finished)
    mutate(data)
    with pytest.raises(ctx.FormatError, match="not a legacy report"):
        ctx.detect_format(data)


def test_non_mapping_payloads_and_unknown_schema_markers_are_refused(finished, env):
    for bad in (None, [], "report"):
        with pytest.raises(ctx.FormatError):
            ctx.detect_format(bad)
    doc, _ = ir.render(finished, policy=env[1])
    for schema in ("live-tracker-investigation-report/0.2", None):
        with pytest.raises(ir.ReportIntegrityError, match="unknown report schema"):
            ir.render_markdown({**doc, "schema": schema})


def test_legacy_fields_match_the_report_dataclass():
    assert ctx.LEGACY_FIELDS == {f.name for f in dataclasses.fields(ia.InvestigationReport)}


# ---- nested immutability ----

def test_freeze_is_recursive_and_thaw_returns_independent_copies():
    data = {"a": [1, {"b": [2, 3]}], "c": {"d": {"e": "f"}}}
    frozen = ctx.freeze(data)
    with pytest.raises(TypeError):
        frozen["x"] = 1
    with pytest.raises(TypeError):
        frozen["c"]["d"]["e"] = "g"
    with pytest.raises(TypeError):
        frozen["a"][1]["b"][0] = 9
    assert isinstance(frozen["a"], tuple) and isinstance(frozen["a"][1], types.MappingProxyType)
    thawed = ctx.thaw(frozen)
    thawed["a"][1]["b"].append(4)
    assert thawed != data and ctx.thaw(frozen) == data
    data["c"]["d"]["e"] = "changed"
    assert frozen["c"]["d"]["e"] == "f"                         # frozen from a copy, not a view of the input


def test_the_validated_context_is_immutable(finished, env):
    context = ir._verify(finished, env[1])
    assert isinstance(context, ctx.ValidatedContext)
    eid = next(iter(context.snapshots))
    with pytest.raises(TypeError):
        context.snapshots[eid]["result"]["measurements"]["x"] = 1
    with pytest.raises(TypeError):
        context.coverage["status"] = "complete"
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.primary_id = None
    copies = context.snapshot_dicts()
    copies[eid]["result"]["status"] = "error"
    assert context.snapshots[eid]["result"]["status"] != "error"
    finished.evidence[0]["result"]["status"] = "tampered after validation"
    assert context.snapshots[finished.evidence[0]["evidence_id"]]["result"]["status"] != "tampered after validation"


# ---- the canonical content fingerprint ----

def payload_of(report, **change):
    args = dict(evidence=report.evidence, finding=report.finding,
                primary_evidence_id=report.primary_evidence["evidence_id"] if report.primary_evidence else None,
                observations=[{"evidence_id": o["evidence_id"], "field": o["field"]} for o in report.observations])
    args.update(change)
    return ctx.fingerprint_payload(**args)


def test_fingerprint_is_stable_across_json_round_trips_and_evidence_order(finished):
    fp = ctx.content_fingerprint(payload_of(finished))
    assert fp == ctx.content_fingerprint(json.loads(json.dumps(payload_of(finished))))
    assert fp == ctx.content_fingerprint(payload_of(finished, evidence=list(reversed(finished.evidence))))
    loaded = ir.load_report(json.loads(json.dumps(finished.to_json())))
    assert fp == ctx.content_fingerprint(payload_of(loaded))
    assert len(fp) == 64 and ctx.canonical_bytes(payload_of(finished)) == ctx.canonical_bytes(
        json.loads(ctx.canonical_bytes(payload_of(finished))))


def test_fingerprint_ignores_execution_metadata_and_authorization(finished, env):
    fp = ctx.content_fingerprint(payload_of(finished))
    altered = dataclasses.replace(finished, trace=[], usage={"model_turns": 99}, stopping_reason="timeout",
                                  limitations=[], interpretations=[], next_steps=[])
    assert ctx.content_fingerprint(payload_of(altered)) == fp
    strict = ic.EvidencePolicy(MANIFEST)                                   # a different authorization outcome
    assert ctx.validate_snapshots(finished.evidence, strict)[2].status == "refused"
    assert ctx.content_fingerprint(payload_of(finished)) == fp


def test_fingerprint_changes_with_content(finished):
    fp = ctx.content_fingerprint(payload_of(finished))
    evidence = copy.deepcopy(finished.evidence)
    evidence[0]["result"]["limitations"].append("changed")
    assert ctx.content_fingerprint(payload_of(finished, evidence=evidence)) != fp
    assert ctx.content_fingerprint(payload_of(finished, finding="inconclusive")) != fp
    obs = payload_of(finished)["observations"]
    assert ctx.content_fingerprint(payload_of(finished, observations=list(reversed(obs)))) != fp   # submitted order
    claims = [{"type": "a", "evidence_ids": [], "params": {}}, {"type": "b", "evidence_ids": [], "params": {}}]
    one = ctx.content_fingerprint(payload_of(finished, claims=claims))
    assert one != ctx.content_fingerprint(payload_of(finished, claims=list(reversed(claims))))


def test_incomplete_fingerprint_needs_no_finish_and_refuses_findings(incomplete):
    payload = payload_of(incomplete)
    assert payload["finding"] == "incomplete" and payload["primary_evidence_id"] is None
    assert payload["claims"] == payload["observations"] == payload["next_investigations"] == []
    ctx.content_fingerprint(payload)
    for change in (dict(observations=[{"evidence_id": incomplete.evidence[0]["evidence_id"], "field": "status"}]),
                   dict(claims=[{"type": "x", "evidence_ids": [], "params": {}}]),
                   dict(primary_evidence_id=incomplete.evidence[-1]["evidence_id"])):
        with pytest.raises(ctx.FingerprintError, match="incomplete run"):
            payload_of(incomplete, **change)


@pytest.mark.parametrize("change, problem", [
    (dict(finding="proven"), "finding must be one of"),
    (dict(primary_evidence_id="ev:compare_windows:" + "0" * 24), "primary evidence"),
    (dict(observations=[{"evidence_id": "x", "field": "status", "statement": "x"}]), "observation"),
    (dict(claims=[{"type": "x"}]), "claim"),
])
def test_fingerprint_inputs_are_validated(finished, change, problem):
    with pytest.raises(ctx.FingerprintError, match=problem):
        payload_of(finished, **change)


def test_fingerprint_refuses_duplicate_evidence_and_non_finite_numbers(finished):
    with pytest.raises(ctx.FingerprintError, match="duplicate"):
        payload_of(finished, evidence=finished.evidence + finished.evidence[:1])
    with pytest.raises(ValueError):
        ctx.canonical_bytes({"x": math.nan})
    with pytest.raises(ctx.FingerprintError):
        ctx.content_fingerprint({"fingerprint_version": "fp/0"})


# ---- execution assessment: freeze-time prefix vs completed trace ----

def test_finished_run_prefix_and_completed_trace(finished):
    snaps = snaps_of(finished)
    prefix = finished.trace[:-1]                                 # up to and including the accepted finish
    pre = ctx.assess_execution(prefix, snaps, mode="prefix")
    assert pre.status == "consistent" and pre.accepted_finish and prefix[-1]["kind"] == "finish"
    done = ctx.assess_execution(finished.trace, snaps, mode="complete", stopping_reason="finished", finished=True)
    assert done.status == "consistent" and done.stopping_reason == "finished" and done.model_ids == (
        "fake-investigator",)
    assert ctx.prefix_extends(prefix, finished.trace)
    assert ctx.assess_execution(prefix, snaps, mode="complete", stopping_reason="finished", finished=True).problems


def test_incomplete_run_never_needs_an_accepted_finish(incomplete):
    snaps = snaps_of(incomplete)
    pre = ctx.assess_execution(incomplete.trace[:-1], snaps, mode="prefix")
    assert pre.status == "consistent" and not pre.accepted_finish
    done = ctx.assess_execution(incomplete.trace, snaps, mode="complete",
                                stopping_reason="max_model_turns", finished=False)
    assert done.status == "consistent" and done.stopping_reason == "max_model_turns"


@pytest.mark.parametrize("damage, problem", [
    (lambda t: t + [dict(t[-1])], "more than one stop event"),
    (lambda t: t[:-1], "does not end with the report's stopping reason"),
    (lambda t: t[:-1] + [{**t[-1], "stopping_reason": "timeout"}], "does not end with"),
    (lambda t: [x for x in t if x["kind"] != "finish"], "accepted finish exists exactly when"),
    (lambda t: t[:-1] + [{"kind": "finish", "outcome": "accepted"}, t[-1]], "more than one accepted finish"),
])
def test_completed_trace_problems(finished, damage, problem):
    trace = damage(copy.deepcopy(finished.trace))
    a = ctx.assess_execution(trace, snaps_of(finished), mode="complete", stopping_reason="finished", finished=True)
    assert a.status == "inconsistent" and any(problem in p for p in a.problems), a.problems


def test_prefix_problems(finished):
    snaps = snaps_of(finished)
    assert "an execution prefix has no stop event" in ctx.assess_execution(finished.trace, snaps, mode="prefix").problems
    calls = [i for i, t in enumerate(finished.trace) if t["kind"] == "tool_call"]
    reordered = copy.deepcopy(finished.trace[:-1])
    reordered[calls[0]], reordered[calls[1]] = reordered[calls[1]], reordered[calls[0]]
    assert ctx.assess_execution(reordered, snaps, mode="prefix").status == "inconsistent"
    late = copy.deepcopy(finished.trace[:-1]) + [copy.deepcopy(finished.trace[calls[0]])]
    assert any("after the accepted finish" in p or "do not match" in p
               for p in ctx.assess_execution(late, snaps, mode="prefix").problems)
    assert not ctx.prefix_extends(finished.trace[:-2], finished.trace)
    with pytest.raises(ValueError):
        ctx.assess_execution(finished.trace, snaps, mode="partial")


def test_stage7_reports_the_completed_trace_problems(finished):
    bad = dataclasses.replace(finished, trace=finished.trace + [dict(finished.trace[-1])])
    with pytest.raises(ir.ReportIntegrityError, match="more than one stop event"):
        ir.render(bad)


# ---- authorization and data identity: one implementation ----

def test_authorization_assessment_layers(finished, env):
    _, problems, rechecked = ctx.validate_snapshots(finished.evidence, env[1])
    assert not problems and rechecked.status == "rechecked" and set(rechecked.per_snapshot.values()) == {"rechecked"}
    _, _, diagnostic = ctx.validate_snapshots(finished.evidence, None)
    assert diagnostic.status == "not_rechecked" and diagnostic.policy_digest is None
    assert set(diagnostic.per_snapshot.values()) == {"not_rechecked"}
    snaps, problems, refused = ctx.validate_snapshots(finished.evidence, ic.EvidencePolicy(MANIFEST))
    assert refused.status == "refused" and problems and len(snaps) < len(finished.evidence)   # synthetic refused
    assert rechecked.policy_digest != ctx.policy_digest(ic.EvidencePolicy(MANIFEST))
    with pytest.raises(TypeError):
        rechecked.per_snapshot["x"] = "rechecked"
    assert ir._verify(finished, None).authorization.status == "not_rechecked"
    assert ir._verify(finished, env[1]).authorization.status == "rechecked"


def test_stage6_and_stage7_share_one_implementation(finished):
    assert ir._data_identity_problem is ctx._data_identity_problem
    assert ir._comparability is ctx._comparability and ir._coverage is ctx._coverage
    assert ir._resolve is ctx._resolve and ir._tracking_concerns is ctx._tracking_concerns
    assert ia._FIELD is ctx.FIELD_RE and ia.MAX_FIELD == ctx.MAX_FIELD
    assert finished.verify_evidence() == ctx.content_consistency(finished.evidence) == []
    tampered = copy.deepcopy(finished.evidence)
    tampered[0]["result"]["limitations"].append("x")
    assert dataclasses.replace(finished, evidence=tampered).verify_evidence() == [tampered[0]["evidence_id"]]
    for o in finished.observations:                               # Stage 6 values = the shared resolver's
        snap = snaps_of(finished)[o["evidence_id"]]
        assert ctx.resolve_field(snap["result"], o["field"]) == o["value"]


def test_context_coverage_and_identity_match_stage7_output(finished, env):
    doc = ir.build_report(finished, policy=env[1])
    context = ir._verify(finished, env[1])
    assert doc["coverage"] == ctx.thaw(context.coverage)
    assert doc["verification"]["cross_snapshot"] == ctx.thaw(context.cross_snapshot)
    assert doc["limitations"]["tracking"] == ctx.thaw(context.tracking)
    pairs = doc["verification"]["cross_snapshot"]["pairs"]
    snaps = snaps_of(finished)
    for p in pairs:                                               # the pair outcomes use the shared identity helper
        assert ctx._data_identity_problem(snaps[p["evidence_ids"][0]], snaps[p["evidence_ids"][1]]) is None


# ---- Codex 2 M0 follow-up: execution equivalence, malformed traces, malformed markers ----

@pytest.mark.parametrize("trace_of, reason, is_finished, problem", [
    (lambda r: r.trace[:-1] + [{**r.trace[-1], "stopping_reason": "timeout"}], "timeout", True,
     "finished and the stopping reason disagree"),                                    # accepted finish, reason timeout
    (lambda r: [t for t in r.trace if t["kind"] != "finish"], "finished", False,
     "finished and the stopping reason disagree"),                                    # reason finished, no finish
    (lambda r: r.trace, "finished", "yes", "boolean finished flag"),
    (lambda r: r.trace, "finished", None, "boolean finished flag"),
])
def test_completed_trace_equivalence_is_enforced(finished, trace_of, reason, is_finished, problem):
    a = ctx.assess_execution(trace_of(finished), snaps_of(finished), mode="complete", stopping_reason=reason,
                             finished=is_finished)
    assert a.status == "inconsistent" and any(problem in p for p in a.problems), a.problems


def test_completed_trace_valid_boundaries(finished, incomplete):
    ok = ctx.assess_execution(finished.trace, snaps_of(finished), mode="complete", stopping_reason="finished",
                              finished=True)
    assert ok.status == "consistent" and ok.accepted_finish
    ok = ctx.assess_execution(incomplete.trace, snaps_of(incomplete), mode="complete",
                              stopping_reason="max_model_turns", finished=False)
    assert ok.status == "consistent" and not ok.accepted_finish


@pytest.mark.parametrize("mode", ["prefix", "complete"])
@pytest.mark.parametrize("make, problem", [
    (lambda t: None, "not a list"), (lambda t: tuple(t), "not a list"), (lambda t: {"kind": "stop"}, "not a list"),
    (lambda t: "trace", "not a list"),
    (lambda t: t[:2] + [7] + t[2:], "malformed trace entries"),                       # interior non-dict
    (lambda t: t[:2] + [None] + t[2:], "malformed trace entries"),
    (lambda t: t + [None], "malformed trace entries"),                                # final None
    (lambda t: t[:1] + [{"step": 1}] + t[1:], "malformed trace entries"),             # no kind
    (lambda t: t[:1] + [{"kind": 3}] + t[1:], "malformed trace entries"),
])
def test_malformed_traces_are_inconsistent_without_raising(finished, mode, make, problem):
    trace = make(copy.deepcopy(finished.trace if mode == "complete" else finished.trace[:-1]))
    a = ctx.assess_execution(trace, snaps_of(finished), mode=mode, stopping_reason="finished", finished=True)
    assert a.status == "inconsistent" and any(problem in p for p in a.problems), a.problems
    assert a.tool_calls == 0 and not a.accepted_finish                       # nothing interpreted


def test_stage7_refuses_a_malformed_trace_with_a_report_error(finished):
    with pytest.raises(ir.ReportIntegrityError, match="malformed trace entries"):
        ir.render(dataclasses.replace(finished, trace=finished.trace + [None]))


MALFORMED_MARKERS = [[], {}, None, 1, 0, 2.0, True, False, "", ["findings/2"], {"findings/2": 1}]


@pytest.mark.parametrize("marker", ["findings_format", "registry_version", "grid_version"])
@pytest.mark.parametrize("value", MALFORMED_MARKERS)
def test_malformed_markers_are_controlled_format_errors(marker, value):
    data = {**V2_MINIMAL, marker: value}
    with pytest.raises(ctx.FormatError, match=f"unknown {marker}"):
        ctx.detect_format(data)
    with pytest.raises(ir.ReportIntegrityError, match=f"unknown {marker}"):     # the public interface
        ir.load_report(data)


@pytest.mark.parametrize("value", MALFORMED_MARKERS)
def test_codex_reproduction_load_report_with_a_malformed_format(value):
    with pytest.raises(ir.ReportIntegrityError, match="unknown findings_format"):
        ir.load_report({"findings_format": value})


@pytest.mark.parametrize("field, value", [("interpretations", "x"), ("interpretations", 3), ("next_steps", {}),
                                          ("limitations", 5), ("limitations", "model")])
def test_malformed_prose_fields_in_v2_payloads_are_refused(field, value):
    with pytest.raises(ctx.FormatError, match="malformed|mixed payload"):
        ctx.detect_format({**V2_MINIMAL, field: value})


def test_marker_valid_boundaries(finished):
    assert ctx.detect_format(V2_MINIMAL) == ctx.V2_FORMAT
    assert ctx.detect_format({**V2_MINIMAL, "interpretations": [], "next_steps": [], "limitations": []}) == \
        ctx.V2_FORMAT
    assert ctx.detect_format(finished.to_json()) == ctx.LEGACY_FORMAT
    assert ir.report_format(ir.load_report(finished.to_json())) == ctx.LEGACY_FORMAT
