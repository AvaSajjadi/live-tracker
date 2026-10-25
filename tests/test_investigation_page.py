"""The Investigation page (pages/3_Investigation.py) through the real multipage
app (AppTest, no browser). New file. SYNTHETIC sessions only; no network, no
model call, journals in tmp_path."""

import dataclasses
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import live_tracker.investigation_view as iv
import live_tracker.replay_investigation as ri
import live_tracker.replay_view as rv
from test_replay_page import DEMO, TRACK, workflow

ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/3_Investigation.py"
ZERO = "Alert me when the mice attack for at least 2 seconds with ask threshold 0"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(tmp_path / "replay"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def open_page(wf=None, **state):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    if wf is not None:
        at.session_state["rule_workflow"] = wf
    for k, v in state.items():
        at.session_state[k] = v
    at.switch_page(PAGE).run()
    assert not at.exception, at.exception
    return at


def rerun(at):
    at.run()
    assert not at.exception, at.exception
    return at


def texts(elements):
    return " ".join(e.value for e in elements)


def provenance(at):
    df = at.dataframe[-1].value                                       # the provenance table is the last one
    return dict(zip(df["field"], df["value"]))


def cached(at):
    return at.session_state[ri.CACHE_KEY]


# ---- selection ----

def test_opens_on_the_replay_session_and_selects_bouts():
    at = open_page(workflow(), replay_session=TRACK)
    assert at.selectbox(key="investigation_session").value == TRACK
    run = provenance(at)["replay run id"]
    bout = at.selectbox(key=f"investigation_bout_{run}_R1")
    assert len(bout.options) == 3                                     # 1.0, 6.0 and 17.0 s
    bout.set_value(2)
    rerun(at)
    md = texts(at.markdown)
    assert "start 17.00 s" in md and "fired 19.00 s" in md
    assert "Excluded from the comparison: baseline_window_unreliable." in texts(at.warning)
    assert any("frozen_tracker" in str(df.value.to_dict()) for df in at.dataframe)


def test_session_switch_shows_the_other_sessions_evidence():
    at = open_page(workflow(), replay_session=DEMO)
    first = provenance(at)
    at.selectbox(key="investigation_session").set_value(TRACK)
    rerun(at)
    second = provenance(at)
    assert first["session"] == "SYNTHETIC_demo_01" and second["session"] == "SYNTHETIC_tracking_02"
    assert first["replay run id"] != second["replay run id"]
    assert first["scores sha256"] != second["scores sha256"] and first["pose sha256"] != second["pose sha256"]
    assert len(cached(at)) == 2                                       # one record per input identity


# ---- correspondence and provenance ----

def test_displayed_provenance_is_the_cached_records():
    at = open_page(workflow(), replay_session=TRACK)
    (res,) = cached(at).values()
    p = provenance(at)
    assert p["session"] == res["inputs"]["session"] and p["replay run id"] == res["inputs"]["replay_run_id"]
    assert p["pose sha256"] == res["inputs"]["pose_sha256"] == rv.content_digest(rv.synthetic_catalog()[1].pose)
    assert p["configuration sha256"] == res["inputs"]["config_sha256"] == ri.config_digest(res["inputs"]["config"])
    assert p["query run id"] == res["rules"][0]["query"]["query_run_id"]
    assert at.get("download_button")[0].label == "Download investigation record (JSON)"




# ---- insufficient evidence, exclusions, quality ----

def test_insufficient_evidence_is_explicit_and_no_effect_is_shown():
    at = open_page(workflow(), replay_session=TRACK)
    assert "Insufficient evidence (too_few_bouts): 3 available, 10 required" in texts(at.warning)
    assert "No effect value is reported" in texts(at.warning)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Bouts found"] == "3" and metrics["Paired sessions"] == "0"
    assert "Per-session paired differences" not in texts(at.markdown)  # no chart without an effect
    assert "quality_v1" in texts(at.caption)
    assert "Overlap policy: `overlap_v1`" in texts(at.caption)
    assert "baseline_window_overlaps_bout" in texts(at.caption)        # the exclusion order is listed


# ---- trace labelling ----

def test_trace_is_labelled_deterministic_and_lists_the_recorded_steps():
    at = open_page(workflow(), replay_session=TRACK)
    assert iv.RUN_KIND in texts(at.info)
    assert "LLM-driven runs are not started from this page" in texts(at.caption)
    trace = next(df.value for df in at.dataframe if "outcome" in df.value.columns)
    assert list(trace["outcome"]) == ["executed: ok", "executed: insufficient_evidence"]
    assert "reasoning" not in " ".join(trace.astype(str).to_numpy().ravel()).lower()


def test_a_refused_rule_is_shown_as_unsupported_with_its_refusal():
    at = open_page(workflow(ZERO))
    assert "ask threshold 0 is not supported by Contract v1 investigation" in texts(at.info)
    trace = next(df.value for df in at.dataframe if "outcome" in df.value.columns)
    assert list(trace["outcome"]) == ["refused: invalid_request"]
    assert "No evidence exists for this rule" in texts(at.caption)


# ---- findings ----

def test_findings_tab_reports_no_report_and_association_only():
    at = open_page(workflow(), replay_session=TRACK)
    assert iv.REPORT_NOTE in texts(at.info)
    md = texts(at.markdown)
    assert "Association only" in md and "not verified behavior" in md and "Publication status: not applicable" in md


# ---- cache invalidation ----

def test_a_changed_pose_is_never_shown_stale(monkeypatch):
    at = open_page(workflow(), replay_session=TRACK)
    before = provenance(at)["pose sha256"]
    demo, track = rv.synthetic_catalog()
    pose = track.pose.copy()
    pose.loc[(pose["time_s"] >= 15.0) & (pose["time_s"] < 17.0), ["x", "y"]] = float("nan")
    changed = dataclasses.replace(track, pose=pose)
    monkeypatch.setattr(rv, "synthetic_catalog", lambda: (demo, changed))
    rerun(at)
    assert provenance(at)["pose sha256"] == rv.content_digest(pose) != before
    run = provenance(at)["replay run id"]
    at.selectbox(key=f"investigation_bout_{run}_R1").set_value(2)
    rerun(at)
    assert "Excluded from the comparison: pre_window_unreliable." in texts(at.warning)


def test_an_effective_policy_change_recomputes(monkeypatch):
    import live_tracker.investigation_tools_v1 as v1
    at = open_page(workflow(), replay_session=TRACK)
    monkeypatch.setattr(v1, "DEFAULT_QUALITY_POLICY", "quality_v1_strict90")
    rerun(at)
    assert len(cached(at)) == 2 and "quality_v1_strict90" in texts(at.caption)


# ---- errors ----

def test_without_an_approved_rule_the_page_says_so():
    at = open_page()
    assert "No approved rule yet" in texts(at.warning) and not at.tabs


def test_a_broken_catalog_and_an_unusable_pose_are_errors(monkeypatch):
    monkeypatch.setattr(rv, "synthetic_catalog", lambda: (_ for _ in ()).throw(RuntimeError("fixture broke")))
    at = open_page(workflow())
    assert "could not be built (RuntimeError: fixture broke)" in texts(at.error) and not at.tabs
    monkeypatch.undo()
    demo, track = rv.synthetic_catalog()
    monkeypatch.setattr(rv, "synthetic_catalog", lambda: (dataclasses.replace(demo, pose=None), track))
    at = open_page(workflow(), replay_session=DEMO)
    assert "needs the session's pose" in texts(at.error) and not at.tabs


def test_the_page_file_runs_on_its_own_without_navigation_errors():
    at = AppTest.from_file(str(ROOT / PAGE), default_timeout=60)
    at.session_state["rule_workflow"] = workflow()
    at.run()
    assert not at.exception, at.exception
    assert "Replay and Rules are in the sidebar." in texts(at.caption)


def test_the_investigation_page_writes_nothing(tmp_path, monkeypatch):
    directory = tmp_path / "never_created"
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(directory))
    at = open_page(workflow(), replay_session=TRACK)
    assert at.tabs and not directory.exists()                         # read-only: no journal, no file
