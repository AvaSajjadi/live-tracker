"""Verified real validation runs on the Replay and Investigation pages (validation_run_view, validation_run_ui).
New file.

The integrity checks are exercised on runs written by the REAL runner (scripts/validation_run_v1.py) over the
SYNTHETIC Stage 9 fixtures (--allow-synthetic), copied and tampered with in tmp_path. The default loader refuses
such runs (they are synthetic), so the checks are reached with allow_synthetic=True, which the pages never pass.
The genuine run outputs/validation_run_v1/run5, when present locally, is only READ (smoke tests; skipped
otherwise) and its files are checked unchanged afterwards.
"""

import copy
import functools
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import live_tracker.investigation_tools_v1 as v1
import live_tracker.validation_run_ui as vui
import live_tracker.validation_run_view as vv
import test_investigation_tools_v1 as T
from test_investigation_windows import build
from test_prediction_windows import score_table
from test_score_drops import TEST_MEMBER, VAL_MEMBER, archive, csv_bytes

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validation_run_v1", ROOT / "scripts" / "validation_run_v1.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
REAL_ROOT = ROOT / "outputs" / "validation_run_v1"
REAL_RUN = "run5"


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """A runner-written SYNTHETIC run: (runs root, run name, manifest)."""
    tmp = tmp_path_factory.mktemp("vrun")
    _, _, cache, manifest = build(tmp)
    zip_path = archive(tmp, {VAL_MEMBER: csv_bytes(score_table()), TEST_MEMBER: csv_bytes(score_table())})
    root = tmp / "runs"
    rc = runner.main([str(zip_path), "--archive-sha256", hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                      "--out", str(root / "good"), "--cache-dir", str(cache), "--manifest", str(manifest),
                      "--confirm", runner.RULE_ID, "--allow-synthetic"])
    assert rc == 0
    return root, "good", manifest


@pytest.fixture
def run_copy(made, tmp_path):
    """A fresh copy of the made run that a test may tamper with: (root, name, manifest)."""
    root, name, manifest = made
    dst = tmp_path / "runs"
    shutil.copytree(root / name, dst / "copy")
    return dst, "copy", manifest


def load(root, name, manifest, **kw):
    return vv.load_run(name, root, manifest_path=manifest, allow_synthetic=True, **kw)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def rehash(run_dir):
    """Make the summary's recorded digests match the current files (so a deeper check, not the digest, decides)."""
    s = read_json(run_dir / vv.SUMMARY)
    sha = {f: hashlib.sha256((run_dir / f).read_bytes()).hexdigest() for f in (vv.RECORD, vv.JOURNAL, vv.SNAPSHOT)}
    s["outputs"] = sha
    s["contract_v1"]["record_sha256"] = sha[vv.RECORD]
    s["monitor"]["journal_snapshot_sha256"] = sha[vv.SNAPSHOT]
    write_json(run_dir / vv.SUMMARY, s)


def edit_record(run_dir, change):
    """Change the Contract v1 record, re-sign each attempt's result digest, and rehash (a consistent forgery)."""
    rec = read_json(run_dir / vv.RECORD)
    change(rec)
    for a in rec["attempts"]:
        a["status"] = a["result"].get("status")
        a["result_sha256"] = hashlib.sha256(json.dumps(a["result"], sort_keys=True, default=str).encode()).hexdigest()
    write_json(run_dir / vv.RECORD, rec)
    rehash(run_dir)


# ---- loading ----

def test_a_verified_run_loads_with_its_recorded_values(made):
    root, name, manifest = made
    run = load(root, name, manifest)
    s = read_json(root / name / vv.SUMMARY)
    assert len(run.events) == s["monitor"]["n_events"] and run.query["n_bouts"] == len(run.events)
    assert [e["event_id"] for e in run.events] == [e["event_id"] for e in s["monitor"]["events"]]   # order, ids
    assert [c["pre_s"] for c in run.comparisons] == [2.0, 1.0, 5.0]
    assert all(c["status"] == "insufficient_evidence" for c in run.comparisons)         # 3 synthetic sessions
    assert run.header["rules"][0]["rule_id"] == "ATTACK_RULE_V1" and run.header["source"]["split"] == "validation"


def test_a_synthetic_run_is_never_accepted_as_real(made):
    root, name, manifest = made
    with pytest.raises(vv.RunUnavailable, match="not labelled as real data"):
        vv.load_run(name, root, manifest_path=manifest)


@pytest.mark.parametrize("missing", vv.RUN_FILES)
def test_a_missing_file_is_refused(run_copy, missing):
    root, name, manifest = run_copy
    (root / name / missing).unlink()
    with pytest.raises(vv.RunUnavailable, match="missing"):
        load(root, name, manifest)


@pytest.mark.parametrize("target", [vv.RECORD, vv.JOURNAL, vv.SNAPSHOT])
def test_a_modified_file_is_refused(run_copy, target):
    root, name, manifest = run_copy
    p = root / name / target
    p.write_bytes(p.read_bytes().replace(b"0.6", b"0.7", 1))
    with pytest.raises(vv.RunUnavailable, match="SHA-256 mismatch"):
        load(root, name, manifest)


@pytest.mark.parametrize("change, why", [
    (lambda s: s.update(status="failed"), "not 'verified'"),
    (lambda s: s.update(status="refused"), "not 'verified'"),
    (lambda s: s["steps"].pop(), "runner's steps"),
    (lambda s: s.update(test_split="read once"), "test split"),
    (lambda s: s["scores"]["provenance"].update(archive_sha256="0" * 64), "different score drops"),
    (lambda s: s["scores"]["provenance"].update(split="train"), "another split"),
    (lambda s: s["scores"]["frames_per_session"].update({"task1/test/SYNTHETIC_test": 10}), "exactly the validation"),
    (lambda s: s["monitor"].update(n_events=s["monitor"]["n_events"] + 1), "monitor n_events contradicts"),
    (lambda s: s["monitor"]["events"][0].update(action="ask" if s["monitor"]["events"][0]["action"] == "alert"
                                                 else "alert"), "not the journal's events"),
    (lambda s: s["monitor"]["events"].reverse(), "not the journal's events"),
    (lambda s: s["rule"].update(ask_threshold=0.5), "not ATTACK_RULE_V1"),
    (lambda s: s["contract_v1"]["compare_windows"][0].update(n_used=99), "misreports comparison"),
    (lambda s: s["contract_v1"]["query_events"].update(query_run_id="inv_0"), "canonical query"),
])
def test_conflicting_status_or_provenance_in_the_summary_is_refused(run_copy, change, why):
    root, name, manifest = run_copy
    s = read_json(root / name / vv.SUMMARY)
    change(s)
    write_json(root / name / vv.SUMMARY, s)
    with pytest.raises(vv.RunUnavailable, match=why):
        load(root, name, manifest)


def promote_to_ok(r):
    """An insufficient comparison rewritten as a complete, well-formed "ok" (every ok field, no insufficient field),
    so that only the gate itself can refuse it."""
    for k in ("reason", "n_available", "n_required", "failed_gate"):
        r.pop(k)
    r.update(status="ok", pre_mean=1.0, baseline_mean=0.0, mean_diff=1.0, mean_diff_definition=v1.MEAN_DIFF_DEFINITION,
             wording=v1.WORDING, per_session=[{"session": "x", "n_bouts": 1, "paired_diff": 1.0}])


def first_compare(rec):
    return next(a for a in rec["attempts"] if a["tool"] == "compare_windows")


@pytest.mark.parametrize("change, why", [
    (lambda r: r["attempts"][0]["result"]["bouts"].pop(), "counts contradict its bouts"),
    (lambda r: (r["attempts"][0]["result"]["bouts"].pop(), r["attempts"][0]["result"].update(
        n_bouts=r["attempts"][0]["result"]["n_bouts"] - 1)), "not the rule's canonical query|not the journal"),
    (lambda r: r["attempts"][0]["result"]["bouts"][0].update(start_s=r["attempts"][0]["result"]["bouts"][0]["start_s"]
                                                             + 1 / 30), "not the journal's events"),
    (lambda r: first_compare(r)["result"].update(status="error"), "not an executed comparison"),
    (lambda r: first_compare(r)["result"].update(overlap_policy="pre_only"), "contradicts the approved configuration"),
    (lambda r: first_compare(r)["result"].update(mean_diff=0.0), "Contract v1's fields"),
    (lambda r: first_compare(r)["result"].update(status="ok", mean_diff=0.0), "Contract v1's fields"),
    (lambda r: promote_to_ok(first_compare(r)["result"]), "below the approved gate"),
    (lambda r: first_compare(r)["request"].update(pre_s=5.0), "did not cite the verified query"),
    (lambda r: r["attempts"].insert(1, copy.deepcopy(r["attempts"][1])), "order"),
    (lambda r: (r["attempts"].__setitem__(slice(1, 3), r["attempts"][2:0:-1]),
                [a.update(call=n) for n, a in enumerate(r["attempts"], start=1)]), "fixed sequence"),
    (lambda r: r.update(investigation_id="someone_else"), "runner's investigation"),
    (lambda r: r.update(data_source="ground_truth_labels"), "data source"),
])
def test_a_forged_but_consistently_hashed_contract_v1_record_is_refused(run_copy, change, why):
    root, name, manifest = run_copy
    edit_record(root / name, change)
    with pytest.raises(vv.RunUnavailable, match=why):
        load(root, name, manifest)


def test_an_attempt_that_does_not_match_its_recorded_digest_is_refused(run_copy):
    root, name, manifest = run_copy
    rec = read_json(root / name / vv.RECORD)
    rec["attempts"][1]["result"]["verdict"] = "unreliable"               # digest left as it was
    write_json(root / name / vv.RECORD, rec)
    rehash(root / name)
    with pytest.raises(vv.RunUnavailable, match="recorded digest"):
        load(root, name, manifest)


def test_a_snapshot_that_is_not_the_journal_run_is_refused(run_copy):
    root, name, manifest = run_copy
    snap = read_json(root / name / vv.SNAPSHOT)
    snap["records"] = snap["records"][2:]                                # drop one fired/closed pair
    (root / name / vv.SNAPSHOT).write_bytes(json.dumps(snap).encode())
    rehash(root / name)
    with pytest.raises(vv.RunUnavailable, match="snapshot"):
        load(root, name, manifest)


def test_a_journal_that_does_not_validate_is_refused(run_copy):
    root, name, manifest = run_copy
    lines = (root / name / vv.JOURNAL).read_bytes().splitlines(keepends=True)
    (root / name / vv.JOURNAL).write_bytes(b"".join(lines[:1] + lines[3:]))   # an event pair removed
    rehash(root / name)
    with pytest.raises(vv.RunUnavailable, match="journal"):
        load(root, name, manifest)


def test_another_split_manifest_is_refused(made):
    root, name, _ = made
    with pytest.raises(vv.RunUnavailable, match="another split manifest"):
        load(root, name, ROOT / "splits" / "split_manifest.json")


# ---- safe selection ----

@pytest.mark.parametrize("name", ["../good", "/etc", "good/..", "", ".hidden", "a" * 80, None, ["good"]])
def test_only_plain_run_names_are_accepted(made, name):
    root, _, manifest = made
    with pytest.raises(vv.RunUnavailable):
        load(root, name, manifest)


def test_symlinked_runs_are_neither_listed_nor_loaded(made, tmp_path):
    root, name, manifest = made
    other = tmp_path / "runs"
    other.mkdir()
    (other / "linked").symlink_to(root / name, target_is_directory=True)
    shutil.copytree(root / name, other / "real")
    (other / "real" / vv.SUMMARY).rename(other / "real" / "x.json")      # not a run: no summary
    assert vv.list_runs(other) == []
    with pytest.raises(vv.RunUnavailable, match="symlinks are refused"):
        load(other, "linked", manifest)


def test_the_runs_directory_comes_from_the_environment_or_the_default(tmp_path):
    assert vv.runs_dir({}) == vv.DEFAULT_RUNS_DIR
    assert vv.runs_dir({vv.RUNS_DIR_ENV: str(tmp_path)}) == tmp_path
    assert vv.list_runs(tmp_path / "absent") == []


# ---- rows ----

def test_rows_are_the_recorded_values(made):
    root, name, manifest = made
    run = load(root, name, manifest)
    sessions = vv.session_rows(run)
    assert len(sessions) == len(run.validation_sessions)
    assert sum(r["events"] for r in sessions) == len(run.events)
    assert [r["event id"] for r in vv.event_rows(run)] == [e["event_id"] for e in run.events]
    sid = run.events[0]["session"]
    assert all(r["session"] == sid.rsplit("/", 1)[-1] for r in vv.event_rows(run, sid))
    rows = vv.comparison_rows(run)
    assert [r["role"] for r in rows] == ["primary", "follow-up 1", "follow-up 2"]
    assert all(r["median session-paired difference (pre - baseline)"] == "none (insufficient evidence)" for r in rows)
    assert sum(r["excluded windows"] for r in vv.exclusion_rows(run, 2.0)) == run.comparisons[0]["n_excluded"]
    assert [r["call"] for r in vv.trace_rows(run)] == [a["call"] for a in run.record["attempts"]]
    prov = {r["item"]: r["value"] for r in vv.provenance_rows(run)}
    assert "UNAVAILABLE (unverified)" in prov["model"] and run.files[vv.RECORD] in prov["Contract v1 record"]


# ---- the pages ----

def page(path, monkeypatch, root, manifest=None, mode=vui.REAL):
    monkeypatch.setenv(vv.RUNS_DIR_ENV, str(root))
    if manifest is not None:                                            # tests only: accept the synthetic fixture run
        monkeypatch.setattr(vv, "load_run", functools.partial(vv.load_run, manifest_path=manifest,
                                                              allow_synthetic=True))
    at = AppTest.from_file(str(ROOT / "pages" / path), default_timeout=60)
    at.session_state[vui.MODE_KEY] = mode
    at.run()
    assert not at.exception, at.exception
    return at


def texts(elements):
    return " ".join(e.value for e in elements)


@pytest.mark.parametrize("path", ["2_Replay.py", "3_Investigation.py"])
def test_real_mode_refuses_a_synthetic_run_with_no_fallback(made, monkeypatch, path):
    root, _, _ = made
    at = page(path, monkeypatch, root)
    assert "cannot be shown" in texts(at.error) and "not labelled as real data" in texts(at.error)
    assert "Nothing is substituted" in texts(at.error)
    assert not at.metric and not at.dataframe and "SYNTHETIC test data" not in texts(at.warning)


@pytest.mark.parametrize("path", ["2_Replay.py", "3_Investigation.py"])
def test_real_mode_without_runs_says_so(tmp_path, monkeypatch, path):
    at = page(path, monkeypatch, tmp_path / "none")
    assert "No validation run found" in texts(at.error) and not at.dataframe


def test_real_mode_renders_a_verified_run_insufficient_evidence_explicitly(made, monkeypatch):
    root, _, manifest = made
    at = page("3_Investigation.py", monkeypatch, root, manifest)
    assert "verified" in texts(at.success)
    assert texts(at.warning).count("Insufficient evidence") == 3 and "No effect value is reported" in texts(at.warning)
    assert "no language model chose or interpreted them" in texts(at.info)
    rec = at.get("download_button")[0]
    assert rec.label.startswith("Download the recorded Contract v1 record") and "not authenticated" in rec.label
    at = page("2_Replay.py", monkeypatch, root, manifest)
    assert {m.label for m in at.metric} >= {"Sessions", "Events (predicted)", "Alerts", "Asks"}
    assert "browsing is not one" in texts(at.info)                     # browsing is not an approval


def test_the_synthetic_demonstration_is_still_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRACKER_REPLAY_DIR", str(tmp_path / "replay"))
    at = AppTest.from_file(str(ROOT / "pages" / "2_Replay.py"), default_timeout=60)
    at.run()
    assert not at.exception and at.radio[0].value == vui.SYNTHETIC
    assert "No approved rule" in texts(at.error)                       # the unchanged synthetic path


# ---- the genuine run (local, read only) ----

real = pytest.mark.skipif(not (REAL_ROOT / REAL_RUN / vv.SUMMARY).is_file(),
                          reason="the genuine validation run is not present locally")


def digests(run_dir):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run_dir.iterdir())}


@real
def test_genuine_run_loads_read_only():
    before = digests(REAL_ROOT / REAL_RUN)
    run = vv.load_run(REAL_RUN, REAL_ROOT)
    assert digests(REAL_ROOT / REAL_RUN) == before
    s = run.summary
    assert s["scores"]["n_sessions"] == len(run.validation_sessions) and len(run.events) == s["monitor"]["n_events"]
    assert run.header["source"]["model_artifact_sha256"] is None
    assert all(c["status"] == "ok" for c in run.comparisons) and run.comparisons[0]["pre_s"] == 2.0
    blob = json.dumps(run.record, indent=2, sort_keys=True, default=str) + "\n"     # the page's download
    assert hashlib.sha256(blob.encode()).hexdigest() == run.files[vv.RECORD]


@real
@pytest.mark.parametrize("path", ["2_Replay.py", "3_Investigation.py"])
def test_genuine_run_renders_on_both_pages(monkeypatch, path):
    before = digests(REAL_ROOT / REAL_RUN)
    at = page(path, monkeypatch, REAL_ROOT)
    at.session_state[vui.RUN_KEY] = REAL_RUN
    at.run()
    assert not at.exception and f"Run '{REAL_RUN}': the recorded execution is CONSISTENT" in texts(at.success)
    assert digests(REAL_ROOT / REAL_RUN) == before


# ======================================================================
# Codex review 2B: forged scientific results, summary claims, filesystem races
# ======================================================================

@pytest.fixture
def real_copy(tmp_path):
    """A temporary COPY of the genuine run (the only run whose comparisons meet the gate); skipped without it."""
    if not (REAL_ROOT / REAL_RUN / vv.SUMMARY).is_file():
        pytest.skip("the genuine validation run is not present locally")
    root = tmp_path / "runs"
    shutil.copytree(REAL_ROOT / REAL_RUN, root / "c")
    return root, "c"


def forge(run_dir, change_record, change_summary=None):
    """A consistently re-hashed forgery: change the record (attempt digests re-signed) and the summary, then make
    every recorded file digest match (the summary itself is not hashed by anything)."""
    rec = read_json(run_dir / vv.RECORD)
    change_record(rec)
    for a in rec["attempts"]:
        a["status"] = a["result"].get("status")
        a["result_sha256"] = hashlib.sha256(json.dumps(a["result"], sort_keys=True, default=str).encode()).hexdigest()
    write_json(run_dir / vv.RECORD, rec)
    if change_summary is not None:
        s = read_json(run_dir / vv.SUMMARY)
        change_summary(s, rec)
        write_json(run_dir / vv.SUMMARY, s)
    rehash(run_dir)


def compares(rec):
    return [a for a in rec["attempts"] if a["tool"] == "compare_windows"]


def sync_compare(s, rec, i=0):
    """Copy a forged comparison's reported fields into the summary, as a careful forger would."""
    r, summ = compares(rec)[i]["result"], s["contract_v1"]["compare_windows"][i]
    for k in list(summ):
        if k in r:
            summ[k] = copy.deepcopy(r[k])
        elif k != "exclusions_by_reason":
            del summ[k]
    by = {}
    for e in r["excluded"]:
        by[e["why"]] = by.get(e["why"], 0) + 1
    summ["exclusions_by_reason"] = by


def forged_compare(change):
    return lambda rec: change(compares(rec)[0]["result"])


@pytest.mark.parametrize("change, why", [
    (lambda r: r.update(mean_diff=999999.0), "not the median of finite per-session"),            # Codex
    (lambda r: r.pop("per_session"), "Contract v1's fields"),                                       # Codex
    (lambda r: r.update(per_session=[]), "contributing sessions are not the used bouts"),
    (lambda r: r.update(per_session=r["per_session"] + [r["per_session"][0]]), "contributing sessions"),
    (lambda r: r["per_session"][0].update(session="task1/train/mouse999_invented"), "contributing sessions"),
    (lambda r: r["per_session"][0].update(paired_diff=float("nan")), "not the median of finite"),
    (lambda r: r.update(n_sessions=r["n_sessions"] + 1), "n_sessions disagree"),
    (lambda r: r.update(excluded=r["excluded"][1:]), "disagree with the verified bouts"),
    (lambda r: r.update(excluded=r["excluded"] + [r["excluded"][0]], n_excluded=r["n_excluded"] + 1),
     "not distinct bouts"),
    (lambda r: r["excluded"][0].update(start_s=12345.0), "not distinct bouts of the verified query"),
    (lambda r: r["excluded"][0].update(why="looked odd"), "known reason"),
])
def test_forged_comparisons_with_recomputed_hashes_are_refused(real_copy, change, why):
    root, name = real_copy
    forge(root / name, forged_compare(change), sync_compare)
    with pytest.raises(vv.RunUnavailable, match=why):
        vv.load_run(name, root)


def probe_change(field_path, value):
    def change(rec):
        for a in rec["attempts"]:
            if a["tool"] in ("check_tracking_quality", "get_window_features"):
                for where in ("request", "result"):
                    target = a[where]
                    if field_path[0] in target:
                        target[field_path[0]] = value
    return change


def sync_probe(s, rec):
    for a in rec["attempts"]:
        if a["tool"] in ("check_tracking_quality", "get_window_features"):
            s["contract_v1"][a["tool"]] = copy.deepcopy(a["result"])
    s["contract_v1"]["probe_bout"]["session"] = rec["attempts"][1]["result"]["session"]


@pytest.mark.parametrize("change, why", [
    (probe_change(("session",), "unauthorized/session"), "probe"),                                   # Codex
    (probe_change(("t0_s",), 0.0), "probe window"),
    (lambda rec: rec["attempts"][1]["result"].update(verdict="reliable", reasons=["usable_fraction_below_min"]),
     "verdict contradicts its measurements"),
    (lambda rec: rec["attempts"][1]["result"].update(quality_policy="quality_v1_strict90"),
     "tracking-quality result"),
    (lambda rec: rec["attempts"][2]["result"]["summary"]["dist_center"].update(min=1e9),
     "feature result is malformed or contradictory"),
    (lambda rec: rec["attempts"][2]["request"].update(cols=["rel_speed"]), "probe window"),
])
def test_rehashed_contradictory_probe_evidence_is_refused(real_copy, change, why):
    root, name = real_copy
    forge(root / name, change, sync_probe)
    with pytest.raises(vv.RunUnavailable, match=why):
        vv.load_run(name, root)


def test_a_self_consistent_forged_effect_cannot_be_detected_so_only_consistency_is_claimed(real_copy, monkeypatch):
    """The honest limit: other per-session differences with a matching median, re-hashed everywhere, are
    consistent. Nothing can tell them from the original without re-running the tools, so the pages claim
    consistency, never authentication."""
    import statistics

    root, name = real_copy

    def change(r):
        for p in r["per_session"]:
            p["paired_diff"] += 1.0
        r["mean_diff"] = statistics.median(p["paired_diff"] for p in r["per_session"])
    forge(root / name, forged_compare(change), sync_compare)
    run = vv.load_run(name, root)                                       # accepted: consistent
    assert run.comparisons[0]["mean_diff"] != read_json(REAL_ROOT / REAL_RUN / vv.RECORD)["attempts"][3][
        "result"]["mean_diff"]
    at = page("3_Investigation.py", monkeypatch, root)
    shown = texts(at.success) + texts(at.caption)
    assert "CONSISTENT" in shown and "Not independently authenticated" in shown
    assert "authenticated" not in texts(at.success).replace("not independently authenticated", "")


# ---- summary values: derived from the journal, or recorded and labelled ----

def edit_summary(run_dir, change):
    s = read_json(run_dir / vv.SUMMARY)
    change(s)
    write_json(run_dir / vv.SUMMARY, s)                                 # no hash is updated (none covers it)


@pytest.mark.parametrize("change, why", [
    (lambda s: s["monitor"].update(actions={"alert": 0, "ask": 80}), "monitor actions contradicts"),        # Codex
    (lambda s: s["evaluation"]["metrics"].update(event_precision=1.0), "arithmetically inconsistent"),     # Codex
    (lambda s: s["monitor"].update(end_reasons={"below_threshold": 80}), "end_reasons contradicts"),
    (lambda s: s["monitor"].update(events_ended_by_tracking_loss=0), "events_ended_by_tracking_loss"),
    (lambda s: s["monitor"]["events_per_session"].popitem(), "events_per_session contradicts"),
    (lambda s: s["monitor"]["duration_s"].update(max=99.0), "durations contradict"),
    (lambda s: s["monitor"]["tracking_interruptions"]["task1/train/mouse064_task1_annotator1"].update(
        tracking_lost_frames=0, tracking_lost_runs=0), "ended by tracking loss in a session recorded without any"),
    (lambda s: s["monitor"]["tracking_interruptions"]["task1/train/mouse005_task1_annotator1"].update(
        tracking_lost_runs=10 ** 6), "impossible"),
    (lambda s: s["monitor"]["tracking_interruptions"].popitem(), "cover exactly the validation sessions"),
    (lambda s: s["evaluation"]["metrics"].update(n_events=79), "does not match the journal"),
    (lambda s: s["evaluation"]["metrics"].update(detected_true=40), "arithmetically inconsistent"),
    (lambda s: s["evaluation"]["true_bouts_per_session"].append(
        {"session": "task1/test/mouse999", "true_bouts": 1, "detected": 0}), "malformed"),
])
def test_forged_summary_claims_are_refused(real_copy, change, why):
    root, name = real_copy
    edit_summary(root / name, change)
    with pytest.raises(vv.RunUnavailable, match=why):
        vv.load_run(name, root)


def test_monitor_counts_are_derived_from_the_journal(real_copy):
    root, name = real_copy
    run = vv.load_run(name, root)
    assert run.derived["n_events"] == len(run.events)
    assert run.derived["actions"] == {a: sum(e["action"] == a for e in run.events) for a in run.derived["actions"]}
    rows = vv.session_rows(run)
    assert sum(r["alerts"] for r in rows) == run.derived["actions"]["alert"]


@pytest.mark.parametrize("change", [
    lambda s: s["evaluation"].update(true_bouts_per_session=[]),                                         # Codex
    lambda s: s["evaluation"]["true_bouts_per_session"].pop(),
    lambda s: s["evaluation"].pop("true_bouts_per_session"),
])
def test_missing_or_incomplete_annotation_counts_are_unknown_not_zero(real_copy, monkeypatch, change):
    root, name = real_copy
    edit_summary(root / name, change)
    run = vv.load_run(name, root)
    assert run.evaluation["status"] == "recorded_incomplete" and run.evaluation["reason"]
    assert {r["annotated attack bouts"] for r in vv.session_rows(run)} == {vv.UNKNOWN}       # never 0
    assert {r["annotated attack bouts in session"] for r in vv.per_session_rows(run, 2.0)} == {vv.UNKNOWN}
    at = page("2_Replay.py", monkeypatch, root)
    assert "Per-session annotation coverage" in texts(at.warning)


def test_complete_recorded_annotation_counts_are_labelled_unverified(real_copy):
    root, name = real_copy
    run = vv.load_run(name, root)
    assert run.evaluation["status"] == "recorded_complete"
    labels = {r["annotated attack bouts"] for r in vv.session_rows(run)}
    assert all(str(x).endswith("(recorded, unverified)") for x in labels)


def test_no_false_alert_or_prevalence_conclusion_is_drawn(real_copy, monkeypatch):
    root, name = real_copy
    at = page("3_Investigation.py", monkeypatch, root)
    shown = " ".join(m.value for m in at.markdown)
    assert "false alerts" not in shown.replace("no conclusion about false alerts", "")
    assert "contain no annotated attack" not in shown and "Annotated attack occurs in" not in shown
    assert "NOT established here" in shown


@pytest.mark.parametrize("path", ["2_Replay.py", "3_Investigation.py"])
def test_a_rejected_run_shows_no_conclusion_and_no_download(real_copy, monkeypatch, path):
    root, name = real_copy
    edit_summary(root / name, lambda s: s["monitor"].update(actions={"alert": 0, "ask": 80}))
    at = page(path, monkeypatch, root)
    assert "cannot be shown" in texts(at.error) and "Nothing is substituted" in texts(at.error)
    assert not at.get("download_button") and not at.metric and not at.dataframe
    assert "evidence gate is met" not in " ".join(m.value for m in at.markdown)


# ---- filesystem safety ----

def test_a_file_swapped_for_a_symlink_just_before_opening_is_refused(run_copy, monkeypatch, tmp_path):
    root, name, manifest = run_copy
    outside = tmp_path / "outside.json"
    outside.write_text('{"status": "verified"}')
    real_open = vv._openat

    def racing(n, flags, dir_fd=None):
        if n == vv.SUMMARY and (root / name / vv.SUMMARY).is_file() and not (root / name / vv.SUMMARY).is_symlink():
            (root / name / vv.SUMMARY).unlink()
            (root / name / vv.SUMMARY).symlink_to(outside)              # the swap, between selection and open
        return real_open(n, flags, dir_fd)
    monkeypatch.setattr(vv, "_openat", racing)
    with pytest.raises(vv.RunUnavailable, match="symlink"):
        load(root, name, manifest)


def test_a_run_directory_swapped_during_loading_cannot_redirect_reads(run_copy, monkeypatch, tmp_path):
    root, name, manifest = run_copy
    expected = load(root, name, manifest).files
    evil = tmp_path / "evil"
    shutil.copytree(root / name, evil)
    (evil / vv.RECORD).write_text("{}")                                 # what a redirect would read
    real_open = vv._openat
    swapped = []

    def racing(n, flags, dir_fd=None):
        if n == vv.RECORD and not swapped:                              # the run directory is already open
            (root / name).rename(root / "old")
            (root / name).symlink_to(evil, target_is_directory=True)
            swapped.append(1)
        return real_open(n, flags, dir_fd)
    monkeypatch.setattr(vv, "_openat", racing)
    run = load(root, name, manifest)
    assert swapped and run.files == expected                            # read from the directory that was opened


def test_an_oversized_file_is_refused_while_reading(run_copy, monkeypatch):
    root, name, manifest = run_copy
    monkeypatch.setattr(vv, "MAX_FILE_BYTES", 1000)
    with pytest.raises(vv.RunUnavailable, match="larger than 1000 bytes"):
        load(root, name, manifest)


def test_a_symlinked_runs_root_is_refused(made, tmp_path):
    root, name, manifest = made
    link = tmp_path / "linkroot"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(vv.RunUnavailable, match="symlinks are refused"):
        load(link, name, manifest)


def test_unsupported_platforms_fail_closed(made, monkeypatch):
    root, name, manifest = made
    monkeypatch.setattr(vv, "_platform_ok", lambda: False)
    with pytest.raises(vv.RunUnavailable, match="fail closed"):
        load(root, name, manifest)


def open_fds():
    return len(os.listdir("/dev/fd"))


def test_descriptors_are_closed_on_success_and_on_refusal(run_copy):
    root, name, manifest = run_copy
    before = open_fds()
    load(root, name, manifest)
    (root / name / vv.JOURNAL).unlink()
    with pytest.raises(vv.RunUnavailable):
        load(root, name, manifest)
    assert open_fds() == before


# ======================================================================
# Codex review 2B (2): execution-defining metadata and quality verdicts
# ======================================================================

CODEX_METADATA = {"quality_policy_definition": {"min_usable_fraction": 0.0},
                  "overlap_policy_definition": "No exclusion of overlapping events",
                  "windows": {"pre": "after event", "baseline": "during event", "anchor": "end_s"}}


def with_field(key, value):
    return lambda r: r.update({key: value})


def altered(key, change):
    def edit(r):
        x = copy.deepcopy(r[key])
        change(x)
        r[key] = x
    return edit


@pytest.mark.parametrize("change, why", [
    (lambda r: r.update(copy.deepcopy(CODEX_METADATA)), "contradicts the approved configuration"),         # Codex
    (altered("quality_policy_definition", lambda d: d.update(min_usable_fraction=0.5)), "quality_policy_definition"),
    (altered("quality_policy_definition", lambda d: d.update(veto_max_s=5.0)), "quality_policy_definition"),
    (altered("overlap_policy_definition", lambda d: d.update(windows_checked=["pre"])), "overlap_policy_definition"),
    (with_field("overlap_policy_definition", "No exclusion of overlapping events"), "overlap_policy_definition"),
    (altered("exclusion_order", lambda o: o.reverse()), "exclusion_order"),
    (altered("windows", lambda w: w.update(pre="after event")), "windows"),
    (altered("windows", lambda w: w.update(baseline="during event")), "windows"),
    (altered("windows", lambda w: w.update(anchor="end_s")), "windows"),
    (altered("windows", lambda w: w.pop("truncation")), "windows"),
    (with_field("baseline_s", 4.0), "baseline_s"),
    (with_field("baseline_gap_s", 0.0), "baseline_gap_s"),
    (with_field("mean_diff_definition", "the mean"), "mean_diff_definition"),
    (with_field("wording", "causes"), "wording"),
    (with_field("data_source", "ground_truth_labels"), "data_source"),
])
def test_altered_execution_defining_metadata_is_refused(real_copy, change, why):
    root, name = real_copy
    forge(root / name, forged_compare(change), sync_compare)
    with pytest.raises(vv.RunUnavailable, match=why):
        vv.load_run(name, root)


@pytest.mark.parametrize("missing", ["quality_policy_definition", "overlap_policy", "overlap_policy_definition",
                                     "exclusion_order", "windows", "quality_policy", "evidence_gate"])
def test_missing_policy_metadata_is_refused(real_copy, missing):
    root, name = real_copy
    forge(root / name, forged_compare(lambda r: r.pop(missing)), sync_compare)
    with pytest.raises(vv.RunUnavailable, match="Contract v1's fields"):
        vv.load_run(name, root)


def test_an_extra_field_in_a_comparison_is_refused(real_copy):
    root, name = real_copy
    forge(root / name, forged_compare(with_field("note", "trust me")), sync_compare)
    with pytest.raises(vv.RunUnavailable, match="Contract v1's fields"):
        vv.load_run(name, root)


# ---- quality verdicts: re-decided with Contract v1's own rule from the recorded measurements ----

@pytest.fixture(scope="module")
def genuine_quality_results():
    """Real check_tracking_quality results (the adapter itself) on synthetic pose: reliable, a usable fraction below
    0.8, the exact 0.8 boundary, and a frozen run longer than 0.5 s."""
    out = {}
    d = T.make(T.SESSIONS[:1], n=600, starts=())
    out["reliable"] = T.quality(d, 0.0, 5.0)
    d = T.make(T.SESSIONS[:1], n=600, starts=(), edit=lambda sid, kp, sc: T.lose(kp, 0, 20))
    out["boundary_0.8"] = T.quality(d, 0.0, 100 / 30)
    out["below"] = T.quality(d, 0.0, 99 / 30)
    d = T.make(T.SESSIONS[:1], n=600, starts=(), edit=lambda sid, kp, sc: T.freeze(kp, 100, 116))
    out["frozen"] = T.quality(d, 0.0, 200 / 30)
    assert [out[k]["verdict"] for k in ("reliable", "boundary_0.8", "below", "frozen")] == \
        ["reliable", "reliable", "unreliable", "unreliable"]
    return out


@pytest.mark.parametrize("kind", ["reliable", "boundary_0.8", "below", "frozen"])
def test_genuine_quality_results_are_accepted(genuine_quality_results, kind):
    r = genuine_quality_results[kind]
    assert vv.check_quality_result(r) == r["reasons"]


def mutate(r, **changes):
    x = copy.deepcopy(r)
    x.update(changes)
    return x


@pytest.mark.parametrize("kind, changes, why", [
    ("reliable", {"usable_fraction": 0.0}, "usable-frame counts are inconsistent"),                         # Codex
    ("reliable", {"n_usable_frames": 0, "usable_fraction": 0.0}, "verdict contradicts its measurements"),
    ("below", {"verdict": "reliable", "reason": None, "reasons": []}, "verdict contradicts"),                 # 79/99
    ("boundary_0.8", {"verdict": "unreliable", "reason": "usable_fraction_below_min",
                      "reasons": ["usable_fraction_below_min"]}, "verdict contradicts"),                     # 80/100
    ("frozen", {"verdict": "reliable", "reason": None, "reasons": []}, "verdict contradicts"),
    ("frozen", {"reasons": ["usable_fraction_below_min"], "reason": "usable_fraction_below_min"}, "verdict contradicts"),
    ("frozen", {"reason": None}, "verdict contradicts"),
    ("below", {"verdict": "unreliable", "reasons": []}, "verdict contradicts"),
    ("frozen", {"flags_fired": []}, "longest veto run is inconsistent"),
    ("reliable", {"longest_veto_run_s": 0.01}, "longest veto run is inconsistent"),
    ("reliable", {"n_frames": 10}, "usable-frame counts are inconsistent"),
    ("reliable", {"quality_policy_definition": {"min_usable_fraction": 0.0}}, "approved policy definition"),
    ("reliable", {"quality_policy": "quality_v1_strict90"}, "approved policy definition"),
    ("reliable", {"flags_fired": ["frozen", "missing"]}, "flags are malformed"),
])
def test_contradictory_quality_results_are_refused(genuine_quality_results, kind, changes, why):
    with pytest.raises(vv.RunUnavailable, match=why):
        vv.check_quality_result(mutate(genuine_quality_results[kind], **changes))


def test_a_missing_quality_field_is_refused(genuine_quality_results):
    r = copy.deepcopy(genuine_quality_results["reliable"])
    r.pop("longest_veto_run_s")
    with pytest.raises(vv.RunUnavailable, match="Contract v1's fields"):
        vv.check_quality_result(r)


def forge_probe_quality(**changes):
    def change(rec):
        q = next(a for a in rec["attempts"] if a["tool"] == "check_tracking_quality")["result"]
        q.update(changes)
        f = next(a for a in rec["attempts"] if a["tool"] == "get_window_features")["result"]
        f["n_usable_frames"] = q["n_usable_frames"]
        f["summary"]["dist_center"]["n"] = min(f["summary"]["dist_center"]["n"], q["n_usable_frames"])
    return change


@pytest.mark.parametrize("changes, why", [
    ({"usable_fraction": 0.0, "verdict": "reliable", "reasons": []}, "usable-frame counts are inconsistent"),  # Codex
    ({"n_usable_frames": 0, "usable_fraction": 0.0}, "verdict contradicts"),
    ({"n_usable_frames": 47, "usable_fraction": 47 / 60}, "verdict contradicts"),                     # just below
    ({"longest_veto_run_s": 0.6, "flags_fired": ["frozen"]}, "verdict contradicts"),
])
def test_rehashed_contradictory_probe_verdicts_are_refused(real_copy, changes, why):
    root, name = real_copy
    forge(root / name, forge_probe_quality(**changes), sync_probe)
    with pytest.raises(vv.RunUnavailable, match=why):
        vv.load_run(name, root)


def test_a_rehashed_probe_at_the_exact_boundary_is_consistent(real_copy):
    """48 of 60 usable frames is exactly 0.8: quality_v1 calls it reliable, so a record saying so is consistent
    (the counts themselves stay recorded values, which is what the page says)."""
    root, name = real_copy
    forge(root / name, forge_probe_quality(n_usable_frames=48, usable_fraction=48 / 60), sync_probe)
    run = vv.load_run(name, root)
    assert run.probe("check_tracking_quality")["verdict"] == "reliable"


@pytest.mark.parametrize("path", ["2_Replay.py", "3_Investigation.py"])
def test_codex_metadata_forgery_shows_no_findings_and_no_download(real_copy, monkeypatch, path):
    root, name = real_copy
    forge(root / name, forged_compare(lambda r: r.update(copy.deepcopy(CODEX_METADATA))), sync_compare)
    at = page(path, monkeypatch, root)
    assert "cannot be shown" in texts(at.error) and "Nothing is substituted" in texts(at.error)
    assert not at.get("download_button") and not at.dataframe and not at.metric
    shown = " ".join(m.value for m in at.markdown) + texts(at.caption)
    assert "after event" not in shown and "No exclusion of overlapping events" not in shown


@pytest.mark.parametrize("path", ["2_Replay.py", "3_Investigation.py"])
def test_codex_quality_forgery_shows_no_findings_and_no_download(real_copy, monkeypatch, path):
    root, name = real_copy
    forge(root / name, forge_probe_quality(usable_fraction=0.0, verdict="reliable", reasons=[]), sync_probe)
    at = page(path, monkeypatch, root)
    assert "cannot be shown" in texts(at.error) and not at.get("download_button") and not at.dataframe


def test_the_page_shows_contract_v1_definitions_not_record_text(monkeypatch):
    if not (REAL_ROOT / REAL_RUN / vv.SUMMARY).is_file():
        pytest.skip("the genuine validation run is not present locally")
    at = page("3_Investigation.py", monkeypatch, REAL_ROOT)
    shown = texts(at.caption)
    assert v1.WINDOW_DEFINITIONS["pre"] in shown and v1.WINDOW_DEFINITIONS["anchor"] in shown
    assert "re-decided from these recorded counts" in shown and "recorded values" in shown
