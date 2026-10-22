"""Tests for Stage 9 S9-6: the offline end-to-end integration
(scripts/stage9_integration.py) on small synthetic inputs. New file.

Synthetic pose cache (test_investigation_windows.build) + a synthetic score
drop over the same three validation sessions (test_prediction_windows); the
real archive is not read. No network, no API key, no paid call.
"""

import copy
import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest

import live_tracker.investigation_claims as cl
import live_tracker.investigation_live as live
import live_tracker.investigation_report as ir
import live_tracker.investigation_report_v2 as v2
from test_investigation_agent import no_network  # noqa: F401  (blocks sockets)
from test_investigation_windows import build
from test_prediction_windows import score_table
from test_score_drops import TEST_MEMBER, VAL_MEMBER, archive, csv_bytes, sums_for

ROOT = Path(__file__).resolve().parents[1]
# the deterministic emulator drives the anthropic SDK through httpx2: pip install -e ".[investigation]"
def _installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


needs_sdk = pytest.mark.skipif(not (_installed("anthropic") and _installed("httpx2")),
                               reason="the investigation extra (anthropic, httpx2) is not installed")
spec = importlib.util.spec_from_file_location("stage9_integration", ROOT / "scripts" / "stage9_integration.py")
integ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integ)


@pytest.fixture
def inputs(tmp_path):
    _, policy, cache, manifest = build(tmp_path)
    zip_path = archive(tmp_path, {VAL_MEMBER: csv_bytes(score_table()), TEST_MEMBER: csv_bytes(score_table())})
    return {"zip": zip_path, "cache": cache, "manifest": manifest, "policy": policy, "tmp": tmp_path}


def argv(inputs, out, *extra):
    return [str(inputs["zip"]), "--out", str(out), "--cache-dir", str(inputs["cache"]), "--manifest",
            str(inputs["manifest"]), "--allow-synthetic", "--min-duration-s", "0.5", *extra]


def text(path):
    """An output file as text. The script writes every output as UTF-8 bytes, so it is read back as UTF-8 on every
    platform: Path.read_text() without an encoding uses the locale's (cp1252 on Windows) and garbles non-ASCII text
    such as the report's "·" separators."""
    return path.read_text(encoding="utf-8")


def summary(out):
    return json.loads(text(out / "integration_summary.json"))


@needs_sdk
def test_full_offline_integration(inputs, capsys):
    out = inputs["tmp"] / "run1"
    assert integ.main(argv(inputs, out)) == 0
    capsys.readouterr()
    s = summary(out)
    assert s["status"] == "verified" and [x["step"] for x in s["steps"]] == [
        "scores", "monitor", "snapshot", "classifier_prediction", "monitor_event"]
    assert all(x["status"] == "ok" for x in s["steps"])
    assert s["monitor"]["provenance_status"] == "verified_score_drop" and s["monitor"]["completion"] == "complete"
    assert s["monitor"]["rule"]["rule_id"] == "LOCAL_SMOKE_TEST" and "not a validated" in s["rule_note"]
    assert "not a researcher decision" in s["monitor"]["rule_approval"] and s["monitor"]["events"] == 4
    assert sorted(p.name for p in out.iterdir()) == sorted(integ.OUTPUT_NAMES)
    for source in integ.SOURCES:
        inv = s["investigations"][source]
        assert inv["evidence_sources"] == [source] and inv["registry_version"] == "claims/2.1"
        assert inv["publication"] in ("not_authorized", "record_only") and inv["provider_reported_cost_usd"] == 0.0
        assert {"query_events", "get_window_features", "check_tracking_quality", "compare_windows"} <= \
            set(inv["tools_used"])
        saved = text(out / f"{source}_report.json")
        rebuilt = v2.verify_saved_v2(json.loads(saved), policy=inputs["policy"])     # independent re-verification
        assert rebuilt["source"]["source"] == source and rebuilt["source"]["disclosures"]
        rendered = v2.render_markdown_v2(json.loads(saved), policy=inputs["policy"])
        assert rendered == text(out / f"{source}_report.md")
        # byte-level, independent of the platform's default encoding: the saved file is exactly the UTF-8 of the
        # rendering, and the rendering really contains non-ASCII text (so a locale decode would differ)
        assert (out / f"{source}_report.md").read_bytes() == rendered.encode("utf-8") and not rendered.isascii()
    snap = (out / "journal_snapshot.json").read_bytes()
    assert hashlib.sha256(snap).hexdigest() == s["monitor"]["snapshot_sha256"]


@needs_sdk
def test_fingerprints_are_reproducible(inputs, capsys):
    a, b = inputs["tmp"] / "a", inputs["tmp"] / "b"
    assert integ.main(argv(inputs, a)) == 0 and integ.main(argv(inputs, b)) == 0
    capsys.readouterr()
    fa, fb = summary(a)["investigations"], summary(b)["investigations"]
    assert {k: v["fingerprint"] for k, v in fa.items()} == {k: v["fingerprint"] for k, v in fb.items()}
    assert summary(a)["monitor"]["snapshot_sha256"] == summary(b)["monitor"]["snapshot_sha256"]


def test_missing_inputs_fail_clearly(inputs, capsys):
    out = inputs["tmp"] / "missing"
    assert integ.main(argv({**inputs, "zip": inputs["tmp"] / "absent.zip"}, out)) == 2
    assert "score archive 'absent.zip' not found" in capsys.readouterr().err and not out.exists()
    assert integ.main(argv({**inputs, "cache": inputs["tmp"] / "no_cache"}, out)) == 2
    assert "validation pose cache" in capsys.readouterr().err


def test_invalid_score_provenance_is_a_recorded_failure(inputs, capsys):
    members = {VAL_MEMBER: csv_bytes(score_table()), TEST_MEMBER: csv_bytes(score_table())}
    bad = archive(inputs["tmp"], members, sums=sums_for({VAL_MEMBER: b"other", TEST_MEMBER: b"x"}), name="bad.zip")
    out = inputs["tmp"] / "bad"
    assert integ.main(argv({**inputs, "zip": bad}, out)) == 1
    capsys.readouterr()
    s = summary(out)
    assert s["status"] == "failed" and s["steps"] == [{**s["steps"][0], "step": "scores", "status": "failed",
                                                       "error": "ArchiveIntegrityError"}]
    assert not (out / "classifier_prediction_report.json").exists()


def test_the_test_member_is_never_read(inputs, capsys):
    out = inputs["tmp"] / "test_member"
    assert integ.main(argv(inputs, out, "--member", TEST_MEMBER)) == 1
    capsys.readouterr()
    assert summary(out)["steps"][0]["error"] == "SplitRefused"


@needs_sdk
def test_an_incomplete_journal_is_refused(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert integ.main(argv(inputs, out)) == 0
    capsys.readouterr()
    journal = out / "events.journal.jsonl"
    cut = inputs["tmp"] / "cut.jsonl"
    cut.write_bytes("".join(x + "\n" for x in text(journal).splitlines()          # exact bytes: no CRLF on Windows
                            if '"run_finished"' not in x).encode("utf-8"))
    with pytest.raises(integ.IntegrationError, match="does not give a verified frozen run"):
        integ.frozen_snapshot(cut)


@needs_sdk
def test_cross_source_evidence_is_refused(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert integ.main(argv(inputs, out)) == 0
    capsys.readouterr()
    clf = json.loads(text(out / "classifier_prediction_investigation.json"))
    mon = json.loads(text(out / "monitor_event_investigation.json"))
    mixed = types.SimpleNamespace(evidence=clf["evidence"] + mon["evidence"][:1], registry_version="claims/2.1")
    with pytest.raises(integ.IntegrationError, match="holds evidence of"):
        integ.check_single_source(mixed, "classifier_prediction")
    wrong = types.SimpleNamespace(evidence=clf["evidence"], registry_version="claims/2.0")
    with pytest.raises(integ.IntegrationError, match="not claims/2.1"):
        integ.check_single_source(wrong, "classifier_prediction")


@needs_sdk
def test_report_status_cannot_be_promoted(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert integ.main(argv(inputs, out)) == 0
    capsys.readouterr()
    saved = json.loads(text(out / "monitor_event_report.json"))
    md = text(out / "monitor_event_report.md")
    promoted = copy.deepcopy(saved)
    promoted["status"]["publication"] = "publishable"
    with pytest.raises(ir.ReportIntegrityError):
        integ.verify_saved(json.dumps(promoted), inputs["policy"], md, "monitor_event")
    with pytest.raises(integ.IntegrationError, match="not the rendering"):
        integ.verify_saved(json.dumps(saved), inputs["policy"], md + "edited", "monitor_event")
    with pytest.raises(integ.IntegrationError, match="lacks its source-specific disclosures"):
        integ.verify_saved(json.dumps(saved), inputs["policy"], md, "classifier_prediction")


@needs_sdk
def test_outputs_are_never_overwritten_or_tracked(inputs, capsys):
    out = inputs["tmp"] / "run"
    assert integ.main(argv(inputs, out)) == 0
    assert integ.main(argv(inputs, out)) == 2
    assert "already contains" in capsys.readouterr().err
    assert integ.main(argv(inputs, ROOT / "docs" / "stage9_out_should_not_exist")) == 2
    assert "is not ignored by Git" in capsys.readouterr().err
    assert not (ROOT / "docs" / "stage9_out_should_not_exist").exists()


def test_stage8_emulator_defaults_are_unchanged():
    e = live._EmulatedOpenRouter()
    assert (e.source, e.with_quality, e.max_baselines) == ("validation_annotation", False, 500)
    assert e.decide([])[2] == {"source": "validation_annotation", "behavior": "attack"}
