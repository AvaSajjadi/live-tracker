"""Phase 3B.1 (scripts/investigation_v1_validation.py): the Stage 9 v1 agent on verified validation inputs with the
deterministic offline investigator. New file.

Isolated tests use SYNTHETIC fixtures only (the Stage 9 synthetic pose cache, test_investigation_windows.build, and an
issued synthetic score drop, test_prediction_windows.score_table); they are labelled synthetic everywhere and are
never real validation results. The one real-data test reads the REAL validation archive, the verified validation
cache and the recorded run5 read-only when they are all present locally (it is skipped otherwise; deselect it
with -k "not real" where real data is not to be read). The test split is never read.
"""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_agent_v1 as av1
from test_investigation_windows import VAL_A, VAL_B, build
from test_prediction_windows import score_table
from test_score_drops import TEST_MEMBER, VAL_MEMBER, archive, csv_bytes

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("investigation_v1_validation",
                                              ROOT / "scripts" / "investigation_v1_validation.py")
iv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv)
vr = iv.vr


@pytest.fixture
def inputs(tmp_path):
    _, _, cache, manifest = build(tmp_path)
    zip_path = archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table()), TEST_MEMBER: csv_bytes(score_table())})
    return {"zip": zip_path, "sha": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "cache": cache,
            "manifest": manifest, "tmp": tmp_path}


def argv(i, out, *extra, synthetic=True, sha=None):
    a = [str(i["zip"]), "--archive-sha256", sha or i["sha"], "--out", str(out), "--cache-dir", str(i["cache"]),
         "--manifest", str(i["manifest"])]
    return a + (["--allow-synthetic"] if synthetic else []) + list(extra)


def summary(out):
    return json.loads((out / "run.json").read_text(encoding="utf-8"))


def report(out):
    return json.loads((out / "investigation_report.json").read_text(encoding="utf-8"))


def refused(code, out, step=None):
    assert code == 2
    s = summary(out)
    assert s["status"] == "refused" and not (out / "investigation_report.json").exists()
    if step:
        assert s["steps"][-1]["step"] == step and s["steps"][-1]["status"] == "failed"
    return s


# ---- a complete offline run (SYNTHETIC fixture) ----

def test_a_synthetic_offline_investigation_completes_and_verifies(inputs, capsys):
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out)) == 0
    capsys.readouterr()
    s, doc = summary(out), report(out)
    assert s["status"] == "verified" and [x["step"] for x in s["steps"]] == ["inputs", "investigate", "verify"]
    assert s["model"]["language_model"] is False and s["model"]["adapter"] == iv.OFFLINE_MODEL
    assert s["inputs"]["data_source"] == "synthetic" and s["inputs"]["scores"]["synthetic"] is True
    assert "REAL" not in json.dumps(s) and set(vr.SYNTHETIC_LABELS.values()) >= {s["inputs"]["scores"]["label"]}
    assert av1.verify_report_v1(doc) == [] and doc["finished"] and doc["data"]["data_source"] == "synthetic"
    assert all(e["provenance"]["split"] == "synthetic" for e in doc["evidence"])
    assert (out / "investigation_report.md").read_text(encoding="utf-8") == av1.render_markdown_v1(doc)
    for name, digest in s["outputs"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert sorted(p.name for p in out.iterdir()) == sorted(iv.OUTPUT_NAMES)


def test_the_investigator_chooses_from_results_and_the_gate_and_budget_hold(inputs, capsys):
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out)) == 0
    capsys.readouterr()
    doc = report(out)
    assert [t["tool"] for t in doc["trace"] if t["kind"] in ("tool_call", "finish")] == [
        "query_events", "check_tracking_quality", "get_window_features", "compare_windows", "compare_windows",
        "compare_windows", av1.FINISH]
    comps = [e for e in doc["evidence"] if e["tool"] == "compare_windows"]
    assert [c["request"]["pre_s"] for c in comps] == [2, 1, 5]               # 1 + Contract v1's two follow-ups
    for c in comps:                                                            # 3 sessions: the 4-session gate
        r = c["result"]
        assert r["status"] == "insufficient_evidence" and "mean_diff" not in r
        assert r["evidence_gate"] == {"min_n_windows": 10, "min_n_sessions": 4, "overridden": False,
                                      "override_reason": None}
        assert (r["quality_policy"], r["overlap_policy"]) == ("quality_v1", "overlap_v1")
    assert doc["finding"] == "insufficient_evidence" and doc["primary_evidence_id"] == comps[-1]["evidence_id"]
    assert doc["budget"] == json.loads(json.dumps(ia.Budget().__dict__, default=list))
    assert doc["runner_record"]["max_tool_calls"] == 12 and doc["usage"]["tool_calls"] == 6
    assert doc["usage"]["input_tokens"] == doc["usage"]["output_tokens"] == 0          # no model was called


def test_the_run_is_reproducible(inputs, capsys):
    a, b = inputs["tmp"] / "a", inputs["tmp"] / "b"
    assert iv.main(argv(inputs, a)) == 0 and iv.main(argv(inputs, b)) == 0
    capsys.readouterr()
    ra, rb = report(a), report(b)
    assert ra["evidence"] == rb["evidence"] and (ra["finding"], ra["observations"]) == (rb["finding"],
                                                                                         rb["observations"])


def test_an_offline_investigator_that_stops_gives_an_incomplete_verified_report(inputs, capsys, monkeypatch):
    real = iv.OfflineInvestigatorV1.decide

    def stop_after_query(self, done):
        if done:
            raise ia.ModelUnavailable("test_stop")
        return real(self, done)
    monkeypatch.setattr(iv.OfflineInvestigatorV1, "decide", stop_after_query)
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out)) == 1
    capsys.readouterr()
    doc = report(out)
    assert summary(out)["status"] == "incomplete" and doc["finding"] == "incomplete" and not doc["finished"]
    assert av1.verify_report_v1(doc) == [] and "stopped before an accepted final answer" in \
        (out / "investigation_report.md").read_text(encoding="utf-8")


def test_the_investigator_stops_on_a_refusal_instead_of_guessing():
    with pytest.raises(ia.ModelUnavailable):
        iv.OfflineInvestigatorV1().decide([("query_events", {}, {"refused": "x"}, True)])


# ---- missing or invalid inputs: refused, nothing substituted ----

def test_synthetic_inputs_are_refused_without_the_test_flag(inputs, capsys):
    out = inputs["tmp"] / "inv"
    refused(iv.main(argv(inputs, out, synthetic=False)), out, "inputs")
    capsys.readouterr()


@pytest.mark.parametrize("missing", ["zip", "manifest", "cache"])
def test_missing_inputs_are_refused_before_anything_runs(inputs, capsys, missing):
    i = dict(inputs, **{missing: inputs["tmp"] / "nowhere"})
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(i, out)) == 2
    assert "refused" in capsys.readouterr().err and not out.exists()


def test_a_wrong_archive_digest_is_refused(inputs, capsys):
    out = inputs["tmp"] / "inv"
    refused(iv.main(argv(inputs, out, sha="0" * 64)), out, "inputs")
    capsys.readouterr()


def test_an_incomplete_pose_cache_is_refused(tmp_path, capsys):
    _, _, cache, manifest = build(tmp_path, sessions=(VAL_A, VAL_B))           # one validation session missing
    zip_path = archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table())})
    i = {"zip": zip_path, "sha": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "cache": cache,
         "manifest": manifest}
    out = tmp_path / "inv"
    s = refused(iv.main(argv(i, out)), out, "inputs")
    assert "incomplete" in s["steps"][-1]["message"]
    capsys.readouterr()


def test_scores_and_pose_with_different_frame_counts_are_refused(tmp_path, capsys):
    _, _, cache, manifest = build(tmp_path)
    zip_path = archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table(n=399))})          # the cache has 400 frames
    i = {"zip": zip_path, "sha": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "cache": cache,
         "manifest": manifest}
    out = tmp_path / "inv"
    refused(iv.main(argv(i, out)), out, "inputs")
    capsys.readouterr()


def test_the_test_member_is_never_read(inputs, capsys, monkeypatch):
    seen = []
    real = iv.score_drops.load_validation_scores

    def spy(zip_path, member, **kw):
        seen.append(member)
        return real(zip_path, member, **kw)
    monkeypatch.setattr(iv.score_drops, "load_validation_scores", spy)
    assert iv.main(argv(inputs, inputs["tmp"] / "inv")) == 0
    capsys.readouterr()
    assert seen == [VAL_MEMBER]
    out = iv.main(argv(inputs, inputs["tmp"] / "inv2", "--member", TEST_MEMBER))
    assert out == 2 and "test" not in json.dumps(summary(inputs["tmp"] / "inv2").get("inputs", {}))
    capsys.readouterr()


def test_existing_outputs_are_never_overwritten(inputs, capsys):
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out)) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert iv.main(argv(inputs, out)) == 2
    capsys.readouterr()
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


# ---- scope, provenance and session checks ----

def stub(sessions, source="classifier", derived="classifier_probabilities", frames=100):
    return SimpleNamespace(data_source=source, derived_from=derived, sessions=tuple(sessions),
                           n_frames=lambda sid: frames)


def scores(sessions, frames=100):
    return {"frames_per_session": {s: frames for s in sessions}, "n_frames": frames * len(sessions)}


REAL14 = [f"task1/train/mouse{n:03d}_task1_annotator1" for n in range(14)]


def test_real_data_must_be_exactly_the_expected_validation_scope():
    frames = iv.EXPECTED_SCOPE["n_frames"] // 14
    with pytest.raises(iv.RunRefused, match="expected scope"):                 # 14 sessions, too few frames
        iv.check_scope(stub(REAL14, frames=frames), scores(REAL14, frames), REAL14, synthetic=False)
    with pytest.raises(iv.RunRefused, match="expected scope"):                 # 13 sessions
        iv.check_scope(stub(REAL14[:13], frames=frames), scores(REAL14[:13], frames), REAL14[:13], synthetic=False)


@pytest.mark.parametrize("data, sc, manifest, why", [
    (stub(REAL14[:13]), scores(REAL14[:13]), REAL14, "not exactly the manifest's validation sessions"),
    (stub(REAL14 + ["task1/test/mouse999"]), scores(REAL14), REAL14, "not exactly the manifest's validation"),
    (stub(REAL14), scores(REAL14[1:]), REAL14, "not exactly the manifest's validation sessions"),
    (stub(REAL14, source="ground_truth_labels", derived="annotation_indicators"), scores(REAL14), REAL14,
     "contradicts the score drop"),
    (stub(REAL14, source="synthetic"), scores(REAL14), REAL14, "contradicts the score drop"),
    (stub(REAL14), {**scores(REAL14), "n_frames": 1}, REAL14, "frames"),
])
def test_provenance_and_session_mismatches_are_refused(data, sc, manifest, why):
    with pytest.raises(iv.RunRefused, match=why):
        iv.check_scope(data, sc, manifest, synthetic=False)


def recorded_run(i, name, tmp, table=None):
    """A recorded Phase 2 run (scripts/validation_run_v1.py) on SYNTHETIC inputs, for the compatibility checks."""
    zip_path = i["zip"] if table is None else archive(tmp, {VAL_MEMBER: csv_bytes(table)}, name=f"{name}.zip")
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    runs = tmp / "runs"
    code = vr.main([str(zip_path), "--archive-sha256", sha, "--out", str(runs / name), "--cache-dir",
                    str(i["cache"]), "--manifest", str(i["manifest"]), "--confirm", vr.RULE_ID, "--allow-synthetic"])
    assert code == 0
    return runs


def test_a_compatible_recorded_run_gives_identical_results(inputs, capsys):
    runs = recorded_run(inputs, "phase2", inputs["tmp"])
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out, "--validation-run", "phase2", "--runs-dir", str(runs))) == 0
    capsys.readouterr()
    compat = summary(out)["report"]["validation_run"]
    assert compat["same_query"] and [m["tool"] for m in compat["identical_results"]] == [
        "query_events", "check_tracking_quality", "get_window_features", "compare_windows", "compare_windows",
        "compare_windows"]


def test_a_recorded_run_on_another_score_drop_is_refused(inputs, capsys):
    other = score_table()
    other.loc[other.index[:5], "mount"] = 0.2                                  # different rows, same sessions
    runs = recorded_run(inputs, "other", inputs["tmp"], table=other)
    out = inputs["tmp"] / "inv"
    s = refused(iv.main(argv(inputs, out, "--validation-run", "other", "--runs-dir", str(runs))), out,
                "validation_run")
    assert "rows_sha256" in s["steps"][-1]["message"] and "archive_sha256" in s["steps"][-1]["message"]
    capsys.readouterr()


def test_an_unknown_or_inconsistent_recorded_run_is_refused(inputs, capsys):
    runs = recorded_run(inputs, "phase2", inputs["tmp"])
    out = inputs["tmp"] / "inv"
    refused(iv.main(argv(inputs, out, "--validation-run", "nope", "--runs-dir", str(runs))), out, "validation_run")
    rec = runs / "phase2" / "run_summary.json"
    original = rec.read_text()
    for n, change in enumerate((lambda s: s["scores"].update(rows_sha256="0" * 64),
                                lambda s: s["scores"]["provenance"].update(rows_sha256="0" * 64),
                                lambda s: s["contract_v1"].update(fingerprint="0" * 64))):
        s = json.loads(original)
        change(s)
        rec.write_text(json.dumps(s))
        out2 = inputs["tmp"] / f"inv{n}"
        refused(iv.main(argv(inputs, out2, "--validation-run", "phase2", "--runs-dir", str(runs))), out2,
                "validation_run")
    capsys.readouterr()


def test_a_recorded_result_that_differs_for_the_same_request_is_refused(inputs, capsys):
    runs = recorded_run(inputs, "phase2", inputs["tmp"])
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out, "--validation-run", "phase2", "--runs-dir", str(runs))) == 0
    capsys.readouterr()
    run = iv.vrv.load_run("phase2", root=runs, manifest_path=inputs["manifest"], allow_synthetic=True)
    forged = copy.deepcopy(run)
    object.__setattr__(forged, "record", copy.deepcopy(run.record))
    forged.record["attempts"][0]["result"]["n_bouts"] += 1
    with pytest.raises(iv.RunRefused, match="disagrees"):
        iv.check_query_against_run(report(out), forged)


# ---- the saved report ----

def test_a_saved_report_that_is_altered_no_longer_verifies(inputs, capsys):
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out)) == 0
    capsys.readouterr()
    doc = report(out)
    data = SimpleNamespace(data_source=doc["data"]["data_source"], derived_from=doc["data"]["derived_from"],
                           fingerprint=doc["data"]["fingerprint"], sessions=tuple(doc["data"]["sessions"]))
    assert iv.check_report(doc, data, synthetic=True)["verify_report_v1"] == "passed"
    bad = copy.deepcopy(doc)
    bad["finding"] = "effect_described"
    with pytest.raises(RuntimeError, match="does not verify"):
        iv.check_report(bad, data, synthetic=True)
    with pytest.raises(av1.ReportVerificationError):
        av1.render_markdown_v1(bad)
    with pytest.raises(RuntimeError, match="other data"):
        iv.check_report(doc, SimpleNamespace(**{**data.__dict__, "fingerprint": "0" * 64}), synthetic=True)
    with pytest.raises(RuntimeError, match="another split"):
        iv.check_report(doc, data, synthetic=False)


# ---- REAL validation data (read-only; skipped when not present locally) ----

REAL_ARCHIVE = Path(os.environ.get("LIVE_TRACKER_VALIDATION_SCORES",
                                   "~/Downloads/igor_scores_validation_model_b_repacked.zip")).expanduser()
REAL_SHA = "5c1a81e40ca12f00dea0ae511708b7d534d42e8b1feb43948af338b8b5c23427"
REAL_CACHE = ROOT / "data" / "raw" / "calms21_validation"
RUN5 = ROOT / "outputs" / "validation_run_v1" / "run5"


@pytest.mark.skipif(not (REAL_ARCHIVE.is_file() and (REAL_CACHE / "extraction_index.json").is_file()
                         and (RUN5 / "run_summary.json").is_file()),
                    reason="the real validation archive, the verified validation cache or run5 is not present")
def test_real_validation_offline_investigation(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)                          # run5 was recorded with these relative input paths
    out = tmp_path / "real"
    code = iv.main([str(REAL_ARCHIVE), "--archive-sha256", REAL_SHA, "--out", str(out), "--validation-run", "run5",
                    "--cache-dir", "data/raw/calms21_validation", "--manifest", "splits/split_manifest.json"])
    capsys.readouterr()
    s, doc = summary(out), report(out)
    assert code == 0 and s["status"] == "verified"
    assert s["inputs"]["scope"] == iv.EXPECTED_SCOPE and s["inputs"]["data_source"] == "classifier"
    assert s["inputs"]["scores"]["synthetic"] is False and s["inputs"]["scores"]["rows_sha256"].startswith("b22df250")
    assert av1.verify_report_v1(doc) == [] and doc["data"]["data_source"] == "classifier"
    assert all(e["provenance"]["split"] == "validation" for e in doc["evidence"])
    assert all("/test/" not in sid for sid in doc["data"]["sessions"]) and len(doc["data"]["sessions"]) == 14
    compat = s["report"]["validation_run"]
    assert compat["same_query"] and len(compat["identical_results"]) == len(doc["evidence"])
    assert doc["finished"] and doc["finding"] in av1.FINDINGS


# ---- run compatibility fails closed on the initial query (Codex re-review) ----

def investigator_with(monkeypatch, change):
    """OfflineInvestigatorV1 with one decision altered (change(name, args) -> (name, args))."""
    real = iv.OfflineInvestigatorV1.decide
    monkeypatch.setattr(iv.OfflineInvestigatorV1, "decide", lambda self, done: change(done, *real(self, done)))


def test_codex_unmatched_initial_query_is_refused(inputs, capsys, monkeypatch):
    """Codex's reproduction: the initial query's min_duration_s 1.0 instead of the run's 0.5."""
    runs = recorded_run(inputs, "phase2", inputs["tmp"])

    def longer(done, name, args):
        return (name, {**args, "min_duration_s": 1.0}) if name == "query_events" else (name, args)
    investigator_with(monkeypatch, longer)
    out = inputs["tmp"] / "inv"
    s = refused(iv.main(argv(inputs, out, "--validation-run", "phase2", "--runs-dir", str(runs))), out,
                "compatibility")
    capsys.readouterr()
    assert "initial query_events is not the request" in s["steps"][-1]["message"]
    assert "validation_run" not in s.get("report", {}) and not (out / "investigation_report.md").exists()


def test_a_matching_initial_query_with_identical_results_is_compatible(inputs, capsys):
    runs = recorded_run(inputs, "phase2", inputs["tmp"])
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out, "--validation-run", "phase2", "--runs-dir", str(runs))) == 0
    capsys.readouterr()
    compat = summary(out)["report"]["validation_run"]
    assert compat["compatible"] and compat["same_query"] and compat["unmatched"] == []
    assert compat["identical_results"][0]["tool"] == "query_events" and compat["identical_results"][0]["run_call"] == 1
    assert len(compat["identical_results"]) == len(report(out)["evidence"])


def test_additional_independent_requests_are_listed_as_unmatched_not_equivalent(inputs, capsys, monkeypatch):
    runs = recorded_run(inputs, "phase2", inputs["tmp"])

    def wider(done, name, args):                     # a feature request the run never made
        return (name, {**args, "cols": ["dist_center", "rel_speed"]}) if name == "get_window_features" else \
            (name, args)
    investigator_with(monkeypatch, wider)
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out, "--validation-run", "phase2", "--runs-dir", str(runs))) == 0
    capsys.readouterr()
    compat = summary(out)["report"]["validation_run"]
    assert compat["compatible"] and [u["tool"] for u in compat["unmatched"]] == ["get_window_features"]
    assert "get_window_features" not in [m["tool"] for m in compat["identical_results"]]
    assert "not equivalent" in compat["unmatched_meaning"]
    assert [m["tool"] for m in compat["identical_results"]] == ["query_events", "check_tracking_quality",
                                                               "compare_windows", "compare_windows", "compare_windows"]


@pytest.fixture
def compatible(inputs, capsys):
    runs = recorded_run(inputs, "phase2", inputs["tmp"])
    out = inputs["tmp"] / "inv"
    assert iv.main(argv(inputs, out, "--validation-run", "phase2", "--runs-dir", str(runs))) == 0
    capsys.readouterr()
    run = iv.vrv.load_run("phase2", root=runs, manifest_path=inputs["manifest"], allow_synthetic=True)
    return report(out), run


def with_record(run, change):
    forged = copy.deepcopy(run)
    object.__setattr__(forged, "record", copy.deepcopy(run.record))
    change(forged.record)
    return forged


def test_a_matching_initial_query_with_a_different_result_is_refused(compatible):
    doc, run = compatible
    forged = with_record(run, lambda r: r["attempts"][0]["result"]["bouts"].pop())
    with pytest.raises(iv.RunRefused, match="initial query_events disagrees"):
        iv.check_query_against_run(doc, forged)


def test_a_later_matched_request_with_a_different_result_is_refused(compatible):
    doc, run = compatible
    forged = with_record(run, lambda r: r["attempts"][3]["result"].update(n_excluded=99))
    with pytest.raises(iv.RunRefused, match="compare_windows disagrees"):
        iv.check_query_against_run(doc, forged)


def test_compatibility_is_never_claimed_without_the_initial_query(compatible):
    doc, run = compatible
    q = doc["evidence"][0]
    for d in ({**doc, "evidence": doc["evidence"][1:]},                                         # no query at all
              {**doc, "evidence": [{**q, "request": {**q["request"], "min_duration_s": 1.0}}] + doc["evidence"]}):
        with pytest.raises(iv.RunRefused, match="initial query_events is not the request"):   # a later match
            iv.check_query_against_run(d, run)                                                  # does not count
    for change in (lambda r: r.update(attempts=[]), lambda r: r["attempts"][0].update(status="error"),
                   lambda r: r["attempts"][0].update(tool="compare_windows")):
        with pytest.raises(iv.RunRefused, match="no executed initial query_events"):
            iv.check_query_against_run(doc, with_record(run, change))
