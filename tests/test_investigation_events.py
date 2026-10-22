"""Tests for the experimental real `query_events` tool (new file).

Synthetic parts use SYNTHETIC test fixtures in the CalMS21 layout, published
through the real Stage 1 extraction code; nothing synthetic is CalMS21 data.
The real-data tests read the verified validation cache under data/raw and are
skipped when it is not present.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import live_tracker.investigation_events as ev
from live_tracker.calms21 import FPS, load_calms21
from live_tracker.investigation_contracts import (CONTRACT_VERSION, EvidencePolicy, ForbiddenEvidence,
                                                  ToolContractViolation, call_tool, episode_id,
                                                  validate_result)
from live_tracker.investigation_data import INDEX_NAME, extract_sessions, session_filename
from live_tracker.investigation_events import (AnnotationEventSource, CalMS21InvestigationTools,
                                               behavior_runs, merge_runs)

ROOT = Path(__file__).resolve().parents[1]
VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
VAL_A, VAL_B, VAL_C = (f"task1/train/SYNTHETIC_val_{x}" for x in "abc")
TRAIN, TEST = "task1/train/SYNTHETIC_train", "task1/test/SYNTHETIC_test"
MANIFEST = {"train": [TRAIN], "validation": [VAL_A, VAL_B, VAL_C], "test": [TEST]}
N = 150
# known attack runs (start, end exclusive); A and B overlap in time
ATTACKS = {VAL_A: [(10, 20), (23, 30), (50, 51), (90, 120)],
           VAL_B: [(10, 20), (40, 100)],
           VAL_C: []}
INVESTIGATION = {VAL_A: [(60, 70)], VAL_B: [], VAL_C: [(0, 5)]}


def record(attacks, investigation=(), n=N):
    labels = np.full(n, VOCAB["other"])
    for a, b in investigation:
        labels[a:b] = VOCAB["investigation"]
    for a, b in attacks:
        labels[a:b] = VOCAB["attack"]
    kp = np.zeros((n, 2, 2, 7))
    kp[:, 0, 0, :], kp[:, 0, 1, :] = 300.0, 300.0
    kp[:, 1, 0, :], kp[:, 1, 1, :] = 600.0, 300.0
    return {"keypoints": kp.tolist(), "scores": np.full((n, 2, 7), 0.9).tolist(),
            "annotations": labels.tolist(), "metadata": {"annotator-id": 0, "vocab": dict(VOCAB)}}


def source_data():
    recs = {sid: record(ATTACKS[sid], INVESTIGATION[sid]) for sid in (VAL_A, VAL_B, VAL_C)}
    recs[TRAIN] = record([(5, 50)])
    return {"annotator-id_0": recs}


@pytest.fixture
def env(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST))
    src = tmp_path / "source.json"
    src.write_text(json.dumps(source_data()))
    cache = tmp_path / "cache"
    extract_sessions(src, cache, [VAL_A, VAL_B, VAL_C], manifest)
    policy = EvidencePolicy(MANIFEST, allow_synthetic=True)
    tools = CalMS21InvestigationTools.for_validation_cache(cache, manifest, policy)
    return tools, policy, cache, manifest, src


def request(**change):
    return {"contract_version": CONTRACT_VERSION, "source": "validation_annotation", "behavior": "attack",
            "sessions": None, "max_episodes": 100, "merge_gap_frames": None, "min_duration_frames": None,
            **change}


def spans(res):
    return [(e["session"], e["start_frame"], e["end_frame"]) for e in res["measurements"]["episodes"]]


def run(env, **change):
    tools, policy = env[0], env[1]
    req = request(**change)
    res = call_tool(tools, "query_events", req, policy)
    validate_result("query_events", req, res, policy)  # always contract-compliant
    return res


# ---- pure helpers ----

def test_behavior_runs_and_merging():
    lab = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1])
    assert behavior_runs(lab == 1) == [(1, 3), (5, 6), (7, 9)]
    assert behavior_runs([]) == []
    # (start, end, annotated frames): merged spans keep their gap frames out of the count
    assert merge_runs([(1, 3), (5, 6), (7, 9)], None) == [(1, 3, 2), (5, 6, 1), (7, 9, 2)]
    assert merge_runs([(1, 3), (5, 6), (7, 9)], 1) == [(1, 3, 2), (5, 9, 3)]
    assert merge_runs([(1, 3), (5, 6), (7, 9)], 2) == [(1, 9, 5)]
    assert merge_runs([], 5) == []


# ---- exact episodes ----

def test_exact_episode_boundaries_across_sessions(env):
    res = run(env)
    expected = [(sid, a, b) for sid in (VAL_A, VAL_B, VAL_C) for a, b in ATTACKS[sid]]
    assert spans(res) == expected
    m = res["measurements"]
    assert res["status"] == "ok" and m["n_episodes_total"] == 6 and m["truncated"] is False
    assert m["sessions_searched"] == [VAL_A, VAL_B, VAL_C]
    assert m["sessions_without_episodes"] == [VAL_C]
    e = m["episodes"][3]
    assert e["episode_id"] == episode_id("validation_annotation", "attack", VAL_A, 90)
    assert (e["start_s"], e["end_s"], e["fps"]) == (90 / FPS, 120 / FPS, FPS)


def test_overlapping_timestamps_in_different_sessions_stay_distinct(env):
    res = run(env)
    tenth = [e for e in res["measurements"]["episodes"] if e["start_frame"] == 10]
    assert [e["session"] for e in tenth] == [VAL_A, VAL_B]
    assert len({e["episode_id"] for e in tenth}) == 2
    assert {e["provenance"]["file_sha256"] for e in tenth} == {
        json.loads((env[2] / INDEX_NAME).read_text())["sessions"][s]["sha256"] for s in (VAL_A, VAL_B)}


def test_session_subset(env):
    res = run(env, sessions=[VAL_B])
    assert spans(res) == [(VAL_B, 10, 20), (VAL_B, 40, 100)]
    assert res["measurements"]["sessions_searched"] == [VAL_B]


def test_other_behaviors(env):
    res = run(env, behavior="investigation")
    assert spans(res) == [(VAL_A, 60, 70), (VAL_C, 0, 5)]


# ---- merging and minimum durations ----

def test_merge_gap_boundary(env):
    assert (VAL_A, 10, 30) in spans(run(env, merge_gap_frames=3))      # gap 20..23 is 3 frames
    assert (VAL_A, 10, 30) not in spans(run(env, merge_gap_frames=2))
    assert (VAL_A, 10, 20) in spans(run(env, merge_gap_frames=2))


@pytest.mark.parametrize("minimum, expected", [
    (30, [(VAL_A, 90, 120), (VAL_B, 40, 100)]),   # 30 frames exactly: kept
    (31, [(VAL_B, 40, 100)]),
    (60, [(VAL_B, 40, 100)]),                     # 60 frames exactly: kept
    (61, []),
    (1, [(s, a, b) for s in (VAL_A, VAL_B) for a, b in ATTACKS[s]]),
])
def test_minimum_duration_boundaries(env, minimum, expected):
    res = run(env, min_duration_frames=minimum)
    assert spans(res) == expected
    m = res["measurements"]
    assert m["sessions_without_episodes"] == [s for s in (VAL_A, VAL_B, VAL_C)
                                              if s not in {x[0] for x in expected}]
    assert res["status"] == ("ok" if expected else "no_data")


def test_minimum_applies_after_merging(env):
    res = run(env, merge_gap_frames=3, min_duration_frames=20)
    assert (VAL_A, 10, 30) in spans(res)          # 10 + 7 frames, merged to 20
    assert (VAL_B, 10, 20) not in spans(res)      # 10 frames


# ---- truncation and no data ----

@pytest.mark.parametrize("limit", [1, 2, 4, 5, 6])
def test_truncation_keeps_the_first_episodes_in_order(env, limit):
    full = spans(run(env))
    res = run(env, max_episodes=limit)
    m = res["measurements"]
    assert spans(res) == full[:limit]
    assert m["n_episodes_total"] == 6 and m["truncated"] is (limit < 6)
    assert m["sessions_without_episodes"] == [VAL_C]  # truncated sessions still had episodes


def test_no_data(env):
    res = run(env, behavior="mount")
    m = res["measurements"]
    assert res["status"] == "no_data" and m["episodes"] == [] and m["n_episodes_total"] == 0
    assert m["sessions_without_episodes"] == m["sessions_searched"] == [VAL_A, VAL_B, VAL_C]


# ---- sources: no substitution ----

@pytest.mark.parametrize("source", ["classifier_prediction", "monitor_event"])
def test_unsupported_sources_are_refused_without_reading_annotations(env, monkeypatch, source):
    def no_read(*a, **k):
        raise AssertionError("annotations were read")

    monkeypatch.setattr(ev, "verified_sessions", no_read)
    res = run(env, source=source)
    assert res["status"] == "error" and res["errors"][0]["code"] == "source_unavailable"
    assert "no other source is substituted" in res["errors"][0]["message"]


def test_sources_are_pluggable_without_changing_the_interface(env):
    """A future source only implements EventSource.query."""
    class FakePredictions:
        source = "classifier_prediction"
        policy = env[1]

        def query(self, request):
            ep = {"episode_id": episode_id(self.source, "attack", VAL_A, 12), "session": VAL_A,
                  "source": self.source, "behavior": "attack", "start_frame": 12, "end_frame": 42,
                  "start_s": 12 / FPS, "end_s": 42 / FPS, "fps": FPS,
                  "provenance": {"model_id": "SYNTHETIC-model", "score_file_sha256": "d" * 64,
                                 "threshold": 0.75, "min_duration_frames": 30}}
            return ev.EventQuery([ep], [VAL_A], [{"kind": "synthetic_fixture", "path": None,
                                                  "sha256": None, "description": "SYNTHETIC"}])

    tools, policy = env[0], env[1]
    plugged = CalMS21InvestigationTools(policy, list(tools.sources.values()) + [FakePredictions()])
    req = request(source="classifier_prediction")
    res = call_tool(plugged, "query_events", req, policy)
    assert spans(res) == [(VAL_A, 12, 42)] and res["measurements"]["episodes"][0]["source"] == \
        "classifier_prediction"
    assert spans(call_tool(plugged, "query_events", request(), policy))[0] == (VAL_A, 10, 20)


def test_duplicate_or_unknown_sources_are_refused(env):
    tools, policy, cache, manifest, _ = env
    src = AnnotationEventSource(cache, manifest, policy)
    with pytest.raises(Exception, match="two sources"):
        CalMS21InvestigationTools(policy, [src, src])
    bad = AnnotationEventSource(cache, manifest, policy)
    bad.source = "rumours"
    with pytest.raises(Exception, match="unknown event source"):
        CalMS21InvestigationTools(policy, [bad])


# ---- test / train exclusion and the policy boundary ----

@pytest.mark.parametrize("sid", [TEST, TRAIN, "task1/train/SYNTHETIC_unlisted"])
def test_non_validation_sessions_are_refused_before_reading(env, monkeypatch, sid):
    def no_read(*a, **k):
        raise AssertionError("annotations were read")

    monkeypatch.setattr(ev, "verified_sessions", no_read)
    tools, policy = env[0], env[1]
    with pytest.raises(ForbiddenEvidence):
        call_tool(tools, "query_events", request(sessions=[sid]), policy)
    with pytest.raises(ForbiddenEvidence):        # the tool enforces it too when called directly
        tools.query_events(request(sessions=[sid]))


def test_tool_requires_a_matching_manifest_backed_policy(env, tmp_path):
    _, policy, cache, manifest, _ = env
    with pytest.raises(ForbiddenEvidence):
        CalMS21InvestigationTools.for_validation_cache(cache, manifest, None)
    other = EvidencePolicy({"train": [], "validation": [VAL_A], "test": []}, allow_synthetic=True)
    with pytest.raises(ForbiddenEvidence, match="does not match"):
        CalMS21InvestigationTools.for_validation_cache(cache, manifest, other)
    with pytest.raises(ForbiddenEvidence, match="validation annotations"):
        CalMS21InvestigationTools(EvidencePolicy(MANIFEST, annotation_split="train"), [])


def test_synthetic_data_is_never_real_evidence(env):
    _, _, cache, manifest, _ = env
    strict = EvidencePolicy(MANIFEST)  # synthetic fixtures not allowed
    tools = CalMS21InvestigationTools.for_validation_cache(cache, manifest, strict)
    res = call_tool(tools, "query_events", request(), strict)
    assert res["status"] == "error" and "ForbiddenEvidence" in res["errors"][0]["message"]


# ---- invalid, missing or altered extraction metadata ----

def _edit_index(cache, change):
    index = json.loads((cache / INDEX_NAME).read_text())
    change(index)
    (cache / INDEX_NAME).write_text(json.dumps(index))


ALTERATIONS = {
    "missing index": lambda cache, manifest: (cache / INDEX_NAME).unlink(),
    "corrupt index": lambda cache, manifest: (cache / INDEX_NAME).write_text("{not json"),
    "old index format": lambda cache, manifest: _edit_index(cache, lambda i: i.__setitem__("format", "v1")),
    "file hash altered": lambda cache, manifest: _edit_index(
        cache, lambda i: i["sessions"][VAL_A].__setitem__("sha256", "0" * 64)),
    "frame count altered": lambda cache, manifest: _edit_index(
        cache, lambda i: i["sessions"][VAL_A].__setitem__("frames", 999)),
    "summary altered": lambda cache, manifest: _edit_index(
        cache, lambda i: i["split_label_summary"][VAL_A]["label_runs"].__setitem__("attack", 99)),
    "test session listed": lambda cache, manifest: _edit_index(
        cache, lambda i: i["sessions"].__setitem__(TEST, dict(i["sessions"][VAL_A]))),
    "session file tampered": lambda cache, manifest: (cache / session_filename(VAL_A)).write_text(
        (cache / session_filename(VAL_A)).read_text().replace("300.0", "301.0", 1)),
    "session file removed": lambda cache, manifest: (cache / session_filename(VAL_B)).unlink(),
    "manifest edited after extraction": lambda cache, manifest: manifest.write_text(
        json.dumps({**MANIFEST, "note": "edited"})),
}


@pytest.mark.parametrize("name", list(ALTERATIONS))
def test_altered_extraction_metadata_is_an_explicit_error(env, name):
    tools, policy, cache, manifest, _ = env
    ALTERATIONS[name](cache, manifest)
    res = call_tool(tools, "query_events", request(), policy)
    assert res["status"] == "error" and res["measurements"] is None
    assert res["errors"][0]["code"] == "source_unavailable"
    assert str(cache) not in res["errors"][0]["message"]  # no file paths in agent-visible text


def test_unlisted_session_files_are_never_read(env):
    tools, policy, cache, _, _ = env
    (cache / session_filename(TEST)).write_text(json.dumps({"annotator-id_0": {TEST: record([(0, 50)])}}))
    res = run(env)
    assert TEST not in {e["session"] for e in res["measurements"]["episodes"]}
    assert TEST not in res["measurements"]["sessions_searched"]


def test_requested_session_missing_from_the_cache_is_an_error(env, tmp_path):
    manifest = tmp_path / "m2.json"
    manifest.write_text(json.dumps(MANIFEST))
    src = tmp_path / "s2.json"
    src.write_text(json.dumps(source_data()))
    cache = tmp_path / "c2"
    extract_sessions(src, cache, [VAL_A], manifest)   # VAL_B not extracted
    policy = EvidencePolicy(MANIFEST, allow_synthetic=True)
    # default (complete coverage): a cache missing eligible sessions refuses every query
    tools = CalMS21InvestigationTools.for_validation_cache(cache, manifest, policy)
    for req in (request(), request(sessions=[VAL_A])):
        res = call_tool(tools, "query_events", req, policy)
        assert res["status"] == "error" and "holds 1 of " in res["errors"][0]["message"]
    # explicit partial coverage: searched what is there, and says what is not
    tools = CalMS21InvestigationTools.for_validation_cache(cache, manifest, policy, coverage="partial")
    res = call_tool(tools, "query_events", request(sessions=[VAL_A, VAL_B]), policy)
    assert res["status"] == "error" and "not in the validated cache" in res["errors"][0]["message"]
    res = call_tool(tools, "query_events", request(), policy)
    assert res["measurements"]["sessions_searched"] == [VAL_A]
    method = {m["name"]: m for m in res["methodology"]}
    assert method["session_coverage"]["value"].startswith("partial") and VAL_B in method["session_coverage"]["note"]
    assert method["eligible_sessions"]["value"] == len(MANIFEST["validation"]) and \
        method["available_sessions"]["value"] == 1
    assert any(l.startswith("PARTIAL COVERAGE: only 1 of") for l in res["limitations"])


# ---- contract compliance, provenance, determinism ----

def test_result_provenance_and_methodology(env):
    res = run(env)
    prov = res["provenance"]
    assert prov["implementation"] == ev.IMPLEMENTATION
    assert prov["code_sha256"] == hashlib.sha256(Path(ev.__file__).read_bytes()).hexdigest()
    kinds = {d["kind"] for d in prov["data"]}
    assert kinds == {"synthetic_fixture"}  # synthetic sessions are labelled as such
    assert prov["data"][0]["sha256"] == ev._canonical_sha(json.loads((env[2] / INDEX_NAME).read_text()))
    names = [m["name"] for m in res["methodology"]]
    assert names == ["event_source", "episode_definition", "merge_gap_frames", "min_duration_frames",
                     "ordering_and_truncation", "fps", "eligible_sessions", "available_sessions",
                     "session_coverage"]
    assert {m["name"]: m["value"] for m in res["methodology"]}["session_coverage"] == "complete"
    assert all(m["status"] == "provisional" for m in res["methodology"])
    assert any("investigation confound" in x for x in res["limitations"])
    for e in res["measurements"]["episodes"]:
        assert e["provenance"]["split"] == "validation" and e["provenance"]["annotator"] == "annotator-id_0"


def test_evidence_ids_are_deterministic_and_content_bound(env):
    a, b = run(env), run(env)
    assert a == b and a["evidence_id"].startswith("ev:query_events:")
    assert run(env, max_episodes=5)["evidence_id"] != a["evidence_id"]


def test_returned_results_are_contract_checked(env, monkeypatch):
    """A broken episode list would never leave the tool as evidence."""
    real = AnnotationEventSource.query

    def broken(self, req):
        q = real(self, req)
        q.episodes[0]["end_s"] += 1.0
        return q

    monkeypatch.setattr(AnnotationEventSource, "query", broken)
    tools, policy = env[0], env[1]
    res = call_tool(tools, "query_events", request(), policy)  # the tool's self-check raised
    assert res["status"] == "error" and res["errors"][0]["code"] == "internal_error"


# ---- real CalMS21 validation data (skipped without the local cache) ----

REAL_CACHE = ROOT / "data" / "raw" / "calms21_validation"
REAL_MANIFEST = ROOT / "splits" / "split_manifest.json"
real = pytest.mark.skipif(not (REAL_CACHE / INDEX_NAME).exists(),
                          reason="verified CalMS21 validation cache not present")


@pytest.fixture
def real_tools():
    policy = EvidencePolicy.from_manifest(REAL_MANIFEST)
    return CalMS21InvestigationTools.for_validation_cache(REAL_CACHE, REAL_MANIFEST, policy), policy


@real
def test_real_attack_episodes_match_stage_1(real_tools):
    tools, policy = real_tools
    req = request()
    res = call_tool(tools, "query_events", req, policy)
    validate_result("query_events", req, res, policy)
    m = res["measurements"]
    mouse005 = "task1/train/mouse005_task1_annotator1"
    manifest = json.loads(REAL_MANIFEST.read_text())
    assert m["sessions_searched"] == sorted(manifest["validation"]) and len(m["sessions_searched"]) == 14
    assert m["n_episodes_total"] == 56 and not m["truncated"]           # counted from the source, all sessions
    with_attacks = sorted({e["session"] for e in m["episodes"]})
    assert [s.split("/")[-1].split("_")[0] for s in with_attacks] == ["mouse005", "mouse028", "mouse045", "mouse064"]
    # mouse005's onsets and duration range reported by Stage 1 (docs/investigation_experiment_01.md)
    m005 = [e for e in m["episodes"] if e["session"] == mouse005]
    assert [e["start_frame"] for e in m005] == [79, 251, 736, 934, 1147, 1969, 2072]
    durations = [(e["end_frame"] - e["start_frame"]) / FPS for e in m005]
    assert round(min(durations), 2) == 0.33 and round(max(durations), 2) == 5.5
    assert {d["kind"] for d in res["provenance"]["data"]} == {"calms21_validation_cache"}
    index = json.loads((REAL_CACHE / INDEX_NAME).read_text())
    assert res["provenance"]["data"][1]["sha256"] == index["source"]["sha256"]
    assert set(m["sessions_without_episodes"]) == set(m["sessions_searched"]) - set(with_attacks)
    assert {x["name"]: x["value"] for x in res["methodology"]}["session_coverage"] == "complete"
    assert any(l.startswith("In the manifest's validation split, 4 of 14 sessions contain any annotated attack "
                            "(56 runs") for l in res["limitations"])


@real
def test_real_counts_agree_with_the_independent_label_summary(real_tools):
    """Episode counts without merging equal the run counts that extraction
    computed from the source (and Stage 1 cross-checked)."""
    tools, policy = real_tools
    index = json.loads((REAL_CACHE / INDEX_NAME).read_text())
    for behavior in ("attack", "investigation", "mount"):
        res = call_tool(tools, "query_events", request(behavior=behavior, max_episodes=500), policy)
        per_session = {}
        for e in res["measurements"]["episodes"]:
            per_session[e["session"]] = per_session.get(e["session"], 0) + 1
        for sid in res["measurements"]["sessions_searched"]:
            assert per_session.get(sid, 0) == index["split_label_summary"][sid]["label_runs"][behavior]
        assert res["measurements"]["n_episodes_total"] == sum(per_session.values())


@real
def test_real_minimum_duration_and_truncation(real_tools):
    tools, policy = real_tools
    mouse005 = "task1/train/mouse005_task1_annotator1"
    res = call_tool(tools, "query_events", request(min_duration_frames=30), policy)
    assert [(e["start_frame"], e["end_frame"]) for e in res["measurements"]["episodes"]
            if e["session"] == mouse005] == [(251, 347), (736, 901), (934, 1033)]
    res = call_tool(tools, "query_events", request(max_episodes=2), policy)
    assert res["measurements"]["truncated"] and res["measurements"]["n_episodes_total"] == 56


@real
def test_real_test_and_train_sessions_are_refused(real_tools):
    tools, policy = real_tools
    manifest = json.loads(REAL_MANIFEST.read_text())
    for sid in (manifest["test"][0], manifest["train"][0]):
        with pytest.raises(ForbiddenEvidence):
            call_tool(tools, "query_events", request(sessions=[sid]), policy)


@real
def test_real_episodes_are_independently_recomputed(real_tools):
    """Recompute the attack runs straight from the loader as a cross-check."""
    tools, policy = real_tools
    res = call_tool(tools, "query_events", request(), policy)
    index = json.loads((REAL_CACHE / INDEX_NAME).read_text())
    expected = []
    for sid in res["measurements"]["sessions_searched"]:
        _, labels = load_calms21(REAL_CACHE / index["sessions"][sid]["file"])
        lab = labels.sort_values("frame")["label"].to_numpy() == "attack"
        edges = np.diff(np.r_[0, lab.astype(int), 0])
        expected += [(sid, int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1),
                                                           np.flatnonzero(edges == -1))]
    assert spans(res) == expected


def test_index_rebuilt_between_verification_and_reading_is_refused(env, monkeypatch):
    """A concurrent rebuild must not mix two cache versions in one result."""
    tools, policy, cache = env[0], env[1], env[2]
    real = ev.verified_sessions

    def verify_then_rebuild(*a, **k):
        out = real(*a, **k)
        _edit_index(cache, lambda i: i["sessions"][VAL_A].__setitem__("sha256", "e" * 64))
        return out

    monkeypatch.setattr(ev, "verified_sessions", verify_then_rebuild)
    res = call_tool(tools, "query_events", request(), policy)
    assert res["status"] == "error" and "changed while reading" in res["errors"][0]["message"]


def test_labels_at_another_frame_rate_are_refused(env, monkeypatch):
    real = ev.load_calms21

    def at_25fps(data):
        poses, labels = real(data)
        labels["time_s"] = labels["frame"] / 25.0
        return poses, labels

    monkeypatch.setattr(ev, "load_calms21", at_25fps)
    res = call_tool(env[0], "query_events", request(), env[1])
    assert res["status"] == "error" and f"{FPS} fps" in res["errors"][0]["message"]


# ======================================================================
# Codex 2 Stage 3 regressions (synthetic fixtures unless marked real)
# ======================================================================

class ReadLog:
    """Records every file opened through open() or Path.read_bytes/read_text."""

    def __init__(self, monkeypatch):
        import builtins
        self.paths = []
        real_open, real_rb, real_rt = builtins.open, Path.read_bytes, Path.read_text

        def spy_open(file, *a, **k):
            self.paths.append(str(file))
            return real_open(file, *a, **k)

        def spy_rb(path):
            self.paths.append(str(path))
            return real_rb(path)

        def spy_rt(path, *a, **k):
            self.paths.append(str(path))
            return real_rt(path, *a, **k)

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(Path, "read_bytes", spy_rb)
        monkeypatch.setattr(Path, "read_text", spy_rt)

    def session_files(self):
        return [p for p in self.paths if Path(p).name.startswith("task1__")]


def _manifest_with_a_in_test():
    return {"train": [TRAIN], "validation": [VAL_B, VAL_C], "test": [TEST, VAL_A]}


# ---- Finding 1: authorization before any annotation is read ----

def test_codex_manifest_swapped_back_after_construction_reads_nothing(env, monkeypatch):
    """Tool built while the manifest puts VAL_A in the held-out split; the
    manifest file is then changed back to the one the cache was built with.
    The query must be refused before any session file is opened."""
    _, _, cache, manifest, _ = env
    original = manifest.read_bytes()
    manifest.write_text(json.dumps(_manifest_with_a_in_test()))
    held_out = EvidencePolicy(_manifest_with_a_in_test(), allow_synthetic=True)
    tools = CalMS21InvestigationTools.for_validation_cache(cache, manifest, held_out)
    manifest.write_bytes(original)                       # changed before querying
    log = ReadLog(monkeypatch)
    res = call_tool(tools, "query_events", request(sessions=None), held_out)
    assert res["status"] == "error" and "manifest changed" in res["errors"][0]["message"]
    assert log.session_files() == []


def test_cache_bound_to_another_manifest_is_refused_before_reading(env, tmp_path, monkeypatch):
    """Policy and manifest snapshot agree (VAL_A held out), but the cache was
    built with VAL_A in validation: refused before any session file is read."""
    _, _, cache, _, _ = env
    other = tmp_path / "held_out_manifest.json"
    other.write_text(json.dumps(_manifest_with_a_in_test()))
    held_out = EvidencePolicy(_manifest_with_a_in_test(), allow_synthetic=True)
    tools = CalMS21InvestigationTools.for_validation_cache(cache, other, held_out)
    log = ReadLog(monkeypatch)
    res = call_tool(tools, "query_events", request(sessions=None), held_out)
    assert res["status"] == "error" and res["errors"][0]["code"] == "source_unavailable"
    assert log.session_files() == []


def test_source_rejects_a_policy_that_disagrees_with_its_manifest(env):
    _, _, cache, manifest, _ = env
    with pytest.raises(ForbiddenEvidence, match="does not match"):
        AnnotationEventSource(cache, manifest, EvidencePolicy(_manifest_with_a_in_test(),
                                                              allow_synthetic=True))
    with pytest.raises(ForbiddenEvidence):
        AnnotationEventSource(cache, manifest, None)


def test_tool_rejects_a_source_bound_to_another_policy_object(env):
    _, policy, cache, manifest, _ = env
    twin = EvidencePolicy(MANIFEST, allow_synthetic=True)  # same content, different object
    with pytest.raises(ForbiddenEvidence, match="another evidence policy"):
        CalMS21InvestigationTools(policy, [AnnotationEventSource(cache, manifest, twin)])


def test_policy_changed_after_construction_reads_nothing(env, monkeypatch):
    tools, policy = env[0], env[1]
    policy.split_of[VAL_A] = "test"
    log = ReadLog(monkeypatch)
    with pytest.raises(ForbiddenEvidence):  # request-level: VAL_A now held out
        call_tool(tools, "query_events", request(sessions=[VAL_A]), policy)
    res = call_tool(tools, "query_events", request(sessions=[VAL_B]), policy)
    assert res["status"] == "error" and "no longer matches" in res["errors"][0]["message"]
    assert log.session_files() == []


def test_every_query_rechecks_the_binding(env):
    tools, policy, _, manifest, _ = env
    assert run(env)["status"] == "ok"
    manifest.write_text(json.dumps({**MANIFEST, "note": "edited"}))
    res = call_tool(tools, "query_events", request(), policy)
    assert res["status"] == "error" and "manifest changed" in res["errors"][0]["message"]


# ---- Finding 2: every backend episode is validated before totals / truncation ----

def _alter_backend(monkeypatch, change):
    real = AnnotationEventSource.query

    def altered(self, req):
        q = real(self, req)
        q.episodes.sort(key=lambda e: (e["session"], e["start_frame"]))
        change(q.episodes)
        return q

    monkeypatch.setattr(AnnotationEventSource, "query", altered)


@pytest.mark.parametrize("name, change", [
    ("one frame long (Codex)", lambda eps: eps[-1].update(end_frame=eps[-1]["start_frame"] + 1,
                                                          end_s=(eps[-1]["start_frame"] + 1) / FPS)),
    ("bad timestamp", lambda eps: eps[-1].update(end_s=eps[-1]["end_s"] + 0.5)),
    ("wrong source", lambda eps: eps[-1].update(source="monitor_event")),
    ("unstable id", lambda eps: eps[-1].update(episode_id=episode_id(
        "validation_annotation", "attack", eps[-1]["session"], eps[-1]["start_frame"] + 1))),
    ("duplicate", lambda eps: eps.append(copy.deepcopy(eps[-1]))),
    ("unsearched session", lambda eps: eps[-1].update(
        session=TRAIN, episode_id=episode_id("validation_annotation", "attack", TRAIN, eps[-1]["start_frame"]))),
    ("held-out session", lambda eps: eps[-1].update(
        session=TEST, episode_id=episode_id("validation_annotation", "attack", TEST, eps[-1]["start_frame"]))),
    ("bad provenance", lambda eps: eps[-1]["provenance"].update(split="test")),
])
def test_invalid_episodes_beyond_the_page_refuse_the_query(env, monkeypatch, name, change):
    """min_duration 30 / max_episodes 1: only VAL_A [90,120) would be returned;
    the altered later episode must still refuse the whole query."""
    _alter_backend(monkeypatch, change)
    tools, policy = env[0], env[1]
    req = request(min_duration_frames=30, max_episodes=1)
    res = call_tool(tools, "query_events", req, policy)    # evidence hashes rebuilt by make_result
    assert res["status"] == "error" and res["measurements"] is None, name
    validate_result("query_events", req, res, policy)
    with pytest.raises(Exception):
        tools.query_events(req)


def test_valid_backend_still_truncates_normally(env):
    res = run(env, min_duration_frames=30, max_episodes=1)
    assert spans(res) == [(VAL_A, 90, 120)]
    assert res["measurements"]["n_episodes_total"] == 2 and res["measurements"]["truncated"]


# ---- Finding 3: one immutable metadata snapshot ----

def _change_after_verification(monkeypatch, cache, change):
    real = ev.verified_sessions

    def verify_then_change(*a, **k):
        out = real(*a, **k)
        _edit_index(cache, change)
        return out

    monkeypatch.setattr(ev, "verified_sessions", verify_then_change)


@pytest.mark.parametrize("change", [
    lambda i: i["source"].__setitem__("sha256", "f" * 64),          # Codex: source-level only
    lambda i: i["source"].__setitem__("path", "/elsewhere/source.json"),
    lambda i: i["extraction_run"].__setitem__("parse_seconds", 1.0),
    lambda i: i.__setitem__("code", {"investigation_data.py": "0" * 64}),
])
def test_metadata_changed_after_verification_is_refused(env, monkeypatch, change):
    tools, policy, cache = env[0], env[1], env[2]
    _change_after_verification(monkeypatch, cache, change)
    res = call_tool(tools, "query_events", request(), policy)
    assert res["status"] == "error" and "verified snapshot" in res["errors"][0]["message"]


def test_manifest_changed_while_computing_is_refused(env, monkeypatch):
    """The binding is rechecked just before the result is returned: a manifest
    rewritten (same sessions, different bytes) after verification, while the
    index stays unchanged, refuses the query."""
    tools, policy, cache, manifest = env[:4]
    real = ev.session_labels

    def labels_then_rewrite(*a, **k):
        out = real(*a, **k)
        manifest.write_text(json.dumps(MANIFEST, indent=1))
        return out
    monkeypatch.setattr(ev, "session_labels", labels_then_rewrite)
    index_before = (cache / INDEX_NAME).read_bytes()
    res = call_tool(tools, "query_events", request(), policy)
    assert res["status"] == "error" and res["errors"][0]["code"] == "source_unavailable"
    assert "manifest changed" in res["errors"][0]["message"]
    assert (cache / INDEX_NAME).read_bytes() == index_before


def test_provenance_comes_from_the_verified_snapshot(env):
    res = run(env)
    index = json.loads((env[2] / INDEX_NAME).read_text())
    data = res["provenance"]["data"]
    assert data[0]["sha256"] == ev._canonical_sha(index)
    assert (data[1]["path"], data[1]["sha256"]) == (index["source"]["path"], index["source"]["sha256"])
    assert [d["sha256"] for d in data[2:]] == [index["sessions"][s]["sha256"] for s in (VAL_A, VAL_B, VAL_C)]


# ---- Finding 4: merged spans are not continuous behavior ----

def test_merged_span_carries_its_annotated_frames(env):
    res = run(env, merge_gap_frames=3)
    ep = next(e for e in res["measurements"]["episodes"] if (e["session"], e["start_frame"]) == (VAL_A, 10))
    assert (ep["end_frame"] - ep["start_frame"], ep["annotated_frames"]) == (20, 17)  # [10,20) + [23,30)
    notes = {m["name"]: m["note"] for m in res["methodology"]}
    assert "not continuous behavior" in notes["merge_gap_frames"]
    assert "annotated_frames counts only the annotated frames" in notes["merge_gap_frames"]
    assert "merged gap frames count toward it" in notes["min_duration_frames"]
    assert any("annotated_frames is the observed behavior" in x for x in res["limitations"])
    assert not any(": 17 of 20 frames" in x for x in res["limitations"])  # no text workaround


def test_unmerged_results_carry_no_merge_limitation(env):
    res = run(env)
    assert not any("Merged episodes span" in x for x in res["limitations"])


@real
def test_real_mouse005_merged_span_is_not_continuous_attack(real_tools):
    """Merging [1969,1979) and [2072,2094) gives a 125-frame span with only 32
    annotated attack frames; a 100-frame (span) minimum still keeps it."""
    tools, policy = real_tools
    req = request(sessions=["task1/train/mouse005_task1_annotator1"], merge_gap_frames=93,
                  min_duration_frames=100)
    res = call_tool(tools, "query_events", req, policy)
    validate_result("query_events", req, res, policy)
    ep = next(e for e in res["measurements"]["episodes"] if e["start_frame"] == 1969)
    assert (ep["end_frame"], ep["end_frame"] - ep["start_frame"], ep["annotated_frames"]) == (2094, 125, 32)


@real
def test_real_query_rechecks_the_whole_index(real_tools):
    tools, policy = real_tools
    res = call_tool(tools, "query_events", request(), policy)
    assert res["status"] == "ok"
    index = json.loads((REAL_CACHE / INDEX_NAME).read_text())
    assert res["provenance"]["data"][0]["sha256"] == ev._canonical_sha(index)


def test_manifest_swapped_between_binding_check_and_verification_reads_nothing(env, tmp_path, monkeypatch):
    """Data access uses the frozen snapshot, not the live file, so a swap right
    after the binding check cannot authorize reading held-out annotations."""
    _, _, cache, manifest, _ = env
    original = manifest.read_bytes()
    held = tmp_path / "held.json"
    held.write_text(json.dumps(_manifest_with_a_in_test()))
    held_out = EvidencePolicy(_manifest_with_a_in_test(), allow_synthetic=True)
    tools = CalMS21InvestigationTools.for_validation_cache(cache, held, held_out)
    real_check = AnnotationEventSource._check_binding

    def check_then_swap(self):
        real_check(self)
        held.write_bytes(original)  # now matches the cache, which lists VAL_A as validation

    monkeypatch.setattr(AnnotationEventSource, "_check_binding", check_then_swap)
    log = ReadLog(monkeypatch)
    res = call_tool(tools, "query_events", request(sessions=None), held_out)
    assert res["status"] == "error" and log.session_files() == []


def test_duplicates_across_validation_pages_are_refused(env, monkeypatch):
    """More than MAX_EPISODES backend episodes are validated in pages; a
    duplicate split across two pages must still be caught."""
    from live_tracker.investigation_contracts import MAX_EPISODES

    def many(eps):
        base = copy.deepcopy(eps[0])
        eps.clear()
        for i in range(MAX_EPISODES + 20):
            e = copy.deepcopy(base)
            start = 1000 + 2 * i
            e.update(start_frame=start, end_frame=start + 1, start_s=start / FPS, end_s=(start + 1) / FPS,
                     episode_id=episode_id("validation_annotation", "attack", e["session"], start))
            eps.append(e)
        # a copy of the LAST episode, put first: the two copies sit on different validation
        # pages, and after sorting both fall outside the returned page of 10
        eps.insert(0, copy.deepcopy(eps[-1]))

    _alter_backend(monkeypatch, many)
    res = call_tool(env[0], "query_events", request(max_episodes=10), env[1])
    assert res["status"] == "error" and "ContractError" in res["errors"][0]["message"]
    with pytest.raises(Exception, match="duplicate episode ids"):
        env[0].query_events(request(max_episodes=10))


# ======================================================================
# annotated_frames (approved contract extension)
# ======================================================================

def test_unmerged_episodes_are_annotated_on_every_frame(env):
    for e in run(env)["measurements"]["episodes"]:
        assert e["annotated_frames"] == e["end_frame"] - e["start_frame"]


def test_merging_three_runs_counts_each_annotated_frame_once(env):
    """VAL_A runs [10,20) [23,30) [50,51) merge at gap 20 into [10,51): 10 + 7 + 1 = 18."""
    res = run(env, merge_gap_frames=20)
    ep = next(e for e in res["measurements"]["episodes"] if (e["session"], e["start_frame"]) == (VAL_A, 10))
    assert (ep["end_frame"], ep["annotated_frames"]) == (51, 18)
    assert ep["annotated_frames"] < ep["end_frame"] - ep["start_frame"]


def test_annotated_frames_survive_truncation_and_dispatch(env):
    full = run(env, merge_gap_frames=3)
    page = run(env, merge_gap_frames=3, max_episodes=1)
    assert page["measurements"]["truncated"]
    assert page["measurements"]["episodes"][0] == full["measurements"]["episodes"][0]
    assert page["measurements"]["episodes"][0]["annotated_frames"] == 17


def test_a_backend_mislabelling_counts_is_refused(env, monkeypatch):
    for change in (lambda eps: eps[0].update(annotated_frames=eps[0]["end_frame"] - eps[0]["start_frame"] + 1),
                   lambda eps: eps[0].update(annotated_frames=1),       # unmerged but partial
                   lambda eps: eps[0].pop("annotated_frames")):
        monkeypatch.undo()
        _alter_backend(monkeypatch, change)
        res = call_tool(env[0], "query_events", request(), env[1])
        assert res["status"] == "error"


def test_predictions_are_never_described_as_annotations(env):
    class Predictions:
        source = "classifier_prediction"
        policy = env[1]

        def __init__(self, with_count):
            self.with_count = with_count

        def query(self, request):
            ep = {"episode_id": episode_id(self.source, "attack", VAL_A, 12), "session": VAL_A,
                  "source": self.source, "behavior": "attack", "start_frame": 12, "end_frame": 42,
                  "start_s": 12 / FPS, "end_s": 42 / FPS, "fps": FPS,
                  "provenance": {"model_id": "SYNTHETIC-model", "score_file_sha256": "d" * 64,
                                 "threshold": 0.75, "min_duration_frames": 30}}
            if self.with_count:
                ep["annotated_frames"] = 30
            return ev.EventQuery([ep], [VAL_A], [{"kind": "synthetic_fixture", "path": None,
                                                  "sha256": None, "description": "SYNTHETIC"}])

    req = request(source="classifier_prediction")
    ok = call_tool(CalMS21InvestigationTools(env[1], [Predictions(False)]), "query_events", req, env[1])
    assert ok["status"] == "ok" and "annotated_frames" not in ok["measurements"]["episodes"][0]
    bad = call_tool(CalMS21InvestigationTools(env[1], [Predictions(True)]), "query_events", req, env[1])
    assert bad["status"] == "error"


@real
def test_real_annotated_frames_equal_the_label_counts(real_tools):
    """Independent check: annotated_frames equals the number of frames the
    loader labels with the behavior inside each returned span."""
    tools, policy = real_tools
    index = json.loads((REAL_CACHE / INDEX_NAME).read_text())
    for merge in (None, 93):
        res = call_tool(tools, "query_events", request(merge_gap_frames=merge), policy)
        for e in res["measurements"]["episodes"]:
            _, labels = load_calms21(REAL_CACHE / index["sessions"][e["session"]]["file"])
            lab = labels.sort_values("frame")["label"].to_numpy()[e["start_frame"]:e["end_frame"]]
            assert e["annotated_frames"] == int((lab == "attack").sum())
