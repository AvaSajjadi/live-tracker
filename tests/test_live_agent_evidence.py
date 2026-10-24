"""UI-2: the Live Agent's scientific evidence charts (investigation_v1_ui.render_evidence). New file.

Through the real multipage app (AppTest). New investigations use SYNTHETIC fixtures and the offline emulator only (no
network, no spending). Every expected value is read from the saved report, never hardcoded; the charts are checked
through the data Vega-Lite actually receives. A verified Stage 9 v1 investigation cannot record an ok comparison on
the 3-session synthetic fixture, nor any gate override (the agent refuses gate arguments: Contract v1 fixes the
gate), so those displays are tested on an IN-MEMORY view of a verified rehearsal (nothing written; display logic
only). The genuine saved investigations are only read, when present, and checked to be unchanged.
"""

import builtins
import copy
import hashlib
import io
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import live_tracker.investigation_tools_v1 as v1
import live_tracker.investigation_v1_runs as runs
import live_tracker.investigation_v1_ui as ui
from live_tracker.investigation_data import extract_sessions
from live_tracker.validation_run_view import RUNS_DIR_ENV
from test_investigation_v1_validation import argv, inputs, iv, recorded_run  # noqa: F401 (fixture)
from test_investigation_windows import TEST, TRAIN, record
from test_score_drops import TEST_MEMBER, VAL_MEMBER, archive, csv_bytes
from test_live_agent_page import offline, open_page, page_text, rehearse_asking, run, synthetic  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
GENUINE = {"live_run1": ROOT / "outputs" / "investigation_v1_live" / "live_run1",
           "dashboard_run": ROOT / "outputs" / "investigation_v1_live" / "20261024T021351Z_69bed91b",
           "run3": ROOT / "outputs" / "investigation_v1" / "phase3b1_run3"}
SELECT = "live_agent_comparison_"


# ---- reading what the charts receive ----

def charts(at):
    """[(layers, data)] of every Altair chart: each layer's spec and the records it plots."""
    out = []
    for c in at.get("vega_lite_chart"):
        spec = json.loads(c.proto.spec)
        data = {d.name: pa.ipc.open_stream(d.data.data).read_pandas().to_dict("records") for d in c.proto.datasets}
        layers = spec.get("layer") or [spec]
        out.append([(layer, data[(layer.get("data") or spec["data"])["name"]]) for layer in layers])  # or shared
    return out


def chart(at, x=None, y=None):
    """The one chart whose first layer plots field x (and y)."""
    found = [c for c in charts(at) if (x is None or c[0][0]["encoding"]["x"].get("field") == x)
             and (y is None or c[0][0]["encoding"]["y"].get("field") == y)]
    assert len(found) <= 1, f"{len(found)} charts plot {x}/{y}"
    return found[0] if found else None


def saved(tmp, name):
    return json.loads((tmp / "live" / name / "investigation_report.json").read_text())


def evidence(doc, evidence_id):
    return next(e for e in doc["evidence"] if e["evidence_id"] == evidence_id)


def cited_query(doc, comparison):
    """The query evidence the comparison's recorded call cited (query_from)."""
    call = next(t for t in doc["trace"] if t.get("evidence_id") == comparison["evidence_id"])
    return evidence(doc, call["arguments"]["query_from"]["evidence_id"])


def section(at):
    """The text of the Scientific evidence section, in render order (up to the timeline's heading)."""
    out, inside = [], False

    def walk(node):
        nonlocal inside
        for child in getattr(node, "children", {}).values():
            value = getattr(child, "value", None) if child.type in ("markdown", "caption", "alert", "info",
                                                                    "success", "warning", "error") else None
            if isinstance(value, str):
                if value == "#### Scientific evidence":
                    inside = True
                elif value == "#### Recorded investigation activity":
                    inside = False
                elif inside:
                    out.append(value)
            walk(child)
    walk(at._tree)
    return out


def markdown(at):
    return [m.value for m in at.markdown]


def short(session):
    return session.rsplit("/", 1)[-1]


def check_bouts(at, query):
    """The bout explorer shows exactly the recorded bouts of this query evidence."""
    r = query["result"]
    per = Counter(short(b["session"]) for b in r["bouts"])
    searched = [short(s) for s in r["query"]["session"]]
    (bars, rows), = chart(at, x="bouts", y="session")
    assert rows == [{"session": s, "bouts": per.get(s, 0)} for s in searched]        # zeros included
    hist = chart(at, x="duration_s")
    durations = sorted(round(row["duration_s"], 9) for row in hist[0][1])
    assert durations == sorted(round(b["end_s"] - b["start_s"], 9) for b in r["bouts"])
    assert hist[1][1] == [{"min_duration_s": r["min_duration_s"]}]                    # the minimum-duration line
    m = {x.label: x.value for x in at.metric}
    assert m["Predicted bouts"] == str(r["n_bouts"]) and m["Applied threshold"] == f"{r['threshold']:.4g}"
    assert m["Sessions with bouts"] == f"{r['n_sessions']} of {r['n_sessions_searched']}"
    captions = [c.value for c in at.caption]
    assert any(c.startswith(f"Classifier-predicted `{r['behavior']}` bouts") and "not annotated behavior" in c
               for c in captions)
    assert any(r["start_meaning"] in c and "never used as the anchor" in c for c in captions)
    labels = json.dumps([layer for c in charts(at) for layer, _ in c]).lower()
    assert "annotat" not in labels                                     # no chart calls a predicted bout annotated
    for text in section(at):                                           # and the text only ever says what they are not
        low = text.lower()
        for i in [i for i in range(len(low)) if low.startswith("annotat", i)]:
            assert re.search(r"\bnot (the )?$", low[max(0, i - 8):i]), f"unnegated 'annotat…' in: {text}"


def check_exclusions(at, r):
    """The exclusion chart and table show the recorded tally, once per bout, in Contract v1's order."""
    tally = Counter(x["why"] for x in r["excluded"])
    (bars, rows), = chart(at, x="bouts", y="category")
    want = [("Used in the comparison", r["n_used"])] + [(runs.EXCLUSION_LABELS[w], tally[w])
                                                         for w in r["exclusion_order"] if tally[w]]
    assert [(x["category"], x["bouts"]) for x in sorted(rows, key=lambda x: x["order"])] == want
    assert sum(x["bouts"] for x in rows) == r["n_input"]                       # used + excluded: nothing twice
    table = next(df.value for df in at.dataframe if "recorded reason (why)" in df.value.columns)
    assert table["recorded reason (why)"].tolist() == r["exclusion_order"]
    assert table["bouts"].tolist() == [tally[w] for w in r["exclusion_order"]]
    assert table["meaning"].tolist() == [runs.EXCLUSION_TEXT[w] for w in r["exclusion_order"]]
    m = {x.label: x.value for x in at.metric}
    assert (m["Input bouts"], m["Used"], m["Excluded"]) == tuple(str(r[k]) for k in ("n_input", "n_used",
                                                                                      "n_excluded"))


# ---- a verified rehearsal: insufficient evidence, three separate comparisons ----

def test_an_insufficient_comparison_charts_its_evidence_but_no_effect(synthetic):
    name = rehearse_asking(synthetic, "Does the distance change before predicted attack bouts?")
    doc = saved(synthetic["tmp"], name)
    at = open_page()
    primary = evidence(doc, doc["primary_evidence_id"])
    r = primary["result"]
    assert r["status"] == "insufficient_evidence" and "per_session" not in r
    assert at.selectbox(key=SELECT + name).value == primary["evidence_id"]          # the primary by default
    assert "#### Scientific evidence" in markdown(at)
    assert chart(at, x="paired_diff") is None                                        # no effect is drawn
    assert "Reported mean_diff (median across sessions)" not in {x.label for x in at.metric}
    assert any(f"status is `insufficient_evidence` ({r['reason']}" in i.value and "nothing to plot" in i.value
               for i in at.info)
    check_exclusions(at, r)
    query = cited_query(doc, primary)
    check_bouts(at, query)
    assert any(f"Bouts of query evidence `{query['evidence_id']}`" in c.value for c in at.caption)
    gate = r["evidence_gate"]
    rows = [m for m in markdown(at) if m.startswith(("**Valid comparison windows**", "**Contributing sessions**"))]
    assert rows[0] == (f"**Valid comparison windows** (`min_n_windows`): observed {r['n_windows']}, required ≥ "
                       f"{gate['min_n_windows']} :red-badge[not met] (recorded: {r['n_available']} available, "
                       f"{r['n_required']} required)") and r["failed_gate"] == "min_n_windows"
    assert rows[1].endswith(":gray-badge[not evaluated] (Contract v1 stops at the first failed gate)")
    assert any(f"Evidence gate not met (`{r['failed_gate']}`; {r['reason']}" in i.value
               and "No effect is reported" in i.value for i in at.info)
    assert any("not overridden" in c.value for c in at.caption) and not any("overridden:" in w.value
                                                                             for w in at.warning)
    (bars, observed), (ticks, required) = chart(at, x="observed")
    assert [(x["observed"], x["required"]) for x in observed] == [(r["n_windows"], gate["min_n_windows"]),
                                                                  (r["n_sessions"], gate["min_n_sessions"])]


def test_each_comparison_is_shown_on_its_own(synthetic):
    name = rehearse_asking(synthetic, "Several comparisons")
    doc = saved(synthetic["tmp"], name)
    comps = [e for e in doc["evidence"] if e["tool"] == "compare_windows"]
    assert len(comps) == 3
    at = open_page()
    assert at.selectbox(key=SELECT + name).options == [c["label"] for c in runs.comparison_choices(doc)]
    seen = []
    for e in comps:
        at.selectbox(key=SELECT + name).set_value(e["evidence_id"])
        at = run(at)
        r = e["result"]
        check_exclusions(at, r)                                                    # this call's tally only
        (_, observed), _ = chart(at, x="observed")
        assert [x["observed"] for x in observed] == [r["n_windows"], r["n_sessions"]]
        assert any(f"pre_s {r['pre_s']:g} s" in o for o in at.selectbox(key=SELECT + name).options)
        seen.append(tuple(sorted(Counter(x["why"] for x in r["excluded"]).items())))
    assert len(set(seen)) > 1                                                       # the calls differ: not merged


def test_the_query_is_the_one_the_comparison_used(synthetic):
    name = rehearse_asking(synthetic, "Association")
    doc = saved(synthetic["tmp"], name)
    comp = evidence(doc, doc["primary_evidence_id"])
    real = next(e for e in doc["evidence"] if e["tool"] == "query_events")
    assert runs.query_for(doc, comp)[0] is real
    decoy = copy.deepcopy(real)                                                     # in memory only
    decoy["evidence_id"] = "ev:query_events:decoy"
    decoy["result"]["query_run_id"] = "inv_another_query"
    doc["evidence"].insert(0, decoy)
    call = next(t for t in doc["trace"] if t.get("evidence_id") == comp["evidence_id"])
    call["arguments"]["query_from"]["evidence_id"] = decoy["evidence_id"]           # a citation that does not match
    q, basis = runs.query_for(doc, comp)
    assert q is real and "query_run_id and query" in basis                          # the recorded run id decides
    real["result"]["query_run_id"] = "inv_yet_another"
    assert runs.query_for(doc, comp)[0] is None                                     # no unrelated query is used
    assert runs.evidence_view(doc, comp["evidence_id"])["query"] is None


# ---- display logic on an in-memory view (no saved file changes) ----

def in_memory_ok(synthetic, monkeypatch, gate):
    """A verified rehearsal whose primary comparison is replaced, IN MEMORY, by an ok result with per-session values
    and the given gate (the display logic only: a saved report like this cannot come from the v1 agent here)."""
    name = rehearse_asking(synthetic, "In-memory display")
    path = synthetic["tmp"] / "live" / name
    view = runs.load_saved(path)
    assert view.verified
    r = evidence(view.doc, view.doc["primary_evidence_id"])["result"]
    for k in ("reason", "n_available", "n_required", "failed_gate"):
        r.pop(k)
    sessions = sorted({b["session"] for b in evidence(view.doc, runs.query_for(view.doc, evidence(
        view.doc, view.doc["primary_evidence_id"]))[0]["evidence_id"])["result"]["bouts"]})
    per = [{"session": s, "n_bouts": 1, "paired_diff": d} for s, d in zip(sessions, (-3.5, 1.25, 0.5))]
    r.update(status="ok", per_session=per, mean_diff=statistics.median(p["paired_diff"] for p in per),
             n_sessions=len(per), n_windows=len(per), evidence_gate=gate)
    monkeypatch.setattr(runs, "load_saved", lambda p: view)
    return name, r


def test_an_ok_comparison_plots_each_recorded_session_with_zero_and_median_lines(synthetic, monkeypatch):
    gate = {"min_n_windows": 10, "min_n_sessions": 4, "overridden": False, "override_reason": None}
    name, r = in_memory_ok(synthetic, monkeypatch, gate)
    at = open_page()
    (bars, rows), (zero, z), (median, md) = chart(at, x="paired_diff")
    assert [(x["full_session"], x["n_bouts"], x["paired_diff"]) for x in rows] == \
        [(p["session"], p["n_bouts"], p["paired_diff"]) for p in r["per_session"]]
    assert [x["direction"] for x in rows] == ["lower before the bout", "higher before the bout",
                                              "higher before the bout"]          # the recorded signs, unchanged
    assert zero["mark"]["type"] == "rule" and z == [{"zero": 0.0}]
    assert median["mark"]["type"] == "rule" and median["mark"]["strokeDash"] and md == [{"mean_diff": r["mean_diff"]}]
    assert bars["encoding"]["x"]["title"] == f"{r['feature']}: pre − baseline (px)"
    m = {x.label: x.value for x in at.metric}
    assert m["Contributing sessions"] == str(r["n_sessions"])
    assert m["Reported mean_diff (median across sessions)"] == f"{r['mean_diff']:.4g} px"
    cap = next(c.value for c in at.caption if c.value.startswith("Each bar is one session"))
    assert "the mean, over that session's used bouts, of the pre-bout window value minus the baseline" in cap
    assert "the median of these session values" in cap
    assert "no confidence interval or significance test" in cap
    text = " ".join(section(at)).lower()
    assert "each bar is one session" in text
    for banned in ("p-value", "p =", "confidence interval of", "statistically significant", "95%"):
        assert banned not in text
    assert any(s.value.startswith("Evidence gate met:") for s in at.success)


def test_a_recorded_gate_override_is_shown_with_its_reason(synthetic, monkeypatch):
    gate = {"min_n_windows": 2, "min_n_sessions": 2, "overridden": True, "override_reason": "pilot: 3 sessions only"}
    name, r = in_memory_ok(synthetic, monkeypatch, gate)
    at = open_page()
    warning = next(w.value for w in at.warning if w.value.startswith("Gate overridden"))
    assert (f"defaults (≥ {v1.DEFAULT_MIN_N_WINDOWS} windows, ≥ {v1.DEFAULT_MIN_N_SESSIONS} sessions) to ≥ 2 "
            "windows and ≥ 2 sessions") in warning and "“pilot: 3 sessions only”" in warning
    (_, observed), (_, required) = chart(at, x="observed")
    assert [x["required"] for x in required] == [2, 2]                             # the lowered, applied gate
    assert all(m.endswith(":green-badge[met]") for m in markdown(at)
               if m.startswith(("**Valid comparison windows**", "**Contributing sessions**")))


# ---- fail closed, read only ----

def test_an_unverified_investigation_shows_no_evidence(synthetic):
    name = rehearse_asking(synthetic, "Tampered")
    path = synthetic["tmp"] / "live" / name / "investigation_report.json"
    doc = json.loads(path.read_text())
    evidence(doc, doc["primary_evidence_id"])["result"]["n_excluded"] += 1          # one changed count
    path.write_text(json.dumps(doc))
    at = open_page()
    assert at.error and "does NOT verify" in at.error[0].value
    assert "#### Scientific evidence" not in markdown(at) and not charts(at)


def test_viewing_evidence_makes_no_request_and_no_calculation(synthetic, monkeypatch):
    name = rehearse_asking(synthetic, "Read only")
    calls = []

    def forbidden(label):
        def f(*a, **k):
            calls.append(label)
            raise AssertionError(f"{label} called while viewing")
        return f
    for fn in ("query_events", "compare_windows", "get_window_features", "check_tracking_quality"):
        monkeypatch.setattr(v1, fn, forbidden(fn))                                  # no Contract v1 computation
    for fn in ("execute", "preflight"):
        monkeypatch.setattr(runs, fn, forbidden(fn))                                # no investigation, no request
    # saved-report verification recomputes the median to check mean_diff (unchanged, required); the evidence
    # section itself must only read it
    rendering = []
    real_render = ui.render_evidence

    def render(saved):
        rendering.append(1)
        try:
            real_render(saved)
        finally:
            rendering.pop()
    monkeypatch.setattr(ui, "render_evidence", render)
    for fn in ("median", "fmean", "mean"):
        real_fn = getattr(statistics, fn)
        monkeypatch.setattr(statistics, fn, lambda *a, _f=real_fn, _n=fn, **k: forbidden(_n)() if rendering
                            else _f(*a, **k))
    at = open_page()
    assert "#### Scientific evidence" in markdown(at) and charts(at)
    for e in [e for e in saved(synthetic["tmp"], name)["evidence"] if e["tool"] == "compare_windows"]:
        at.selectbox(key=SELECT + name).set_value(e["evidence_id"])
        at = run(at)
    assert calls == [] and not at.exception


# ---- a genuinely verified ok comparison, end to end (SYNTHETIC, no local outputs) ----
# Five synthetic validation sessions of 40 s, each with three 1 s predicted attack bouts (score 0.9 >= the offline
# investigator's 0.6, >= its 0.5 s minimum), each with 8 s of steady history, so every pre-bout (2 s) and baseline
# (5 s, 1 s gap) window is clean and none overlaps another bout. Mouse b approaches during each pre-bout window
# (700 -> 500 px; in the last session it retreats, 700 -> 900 px) and returns after the bout. Contract v1's DEFAULT
# gate (10 windows, 4 sessions) is met by the data itself: nothing is lowered or overridden.

GATE_SESSIONS = tuple(f"task1/train/SYNTHETIC_gate_{x}" for x in "abcde")
GATE_MANIFEST = {"train": [TRAIN], "validation": list(GATE_SESSIONS), "test": [TEST]}
GATE_FRAMES = 1200
BOUT_STARTS = (300, 700, 1100)                                        # frames; each bout lasts 30 frames
PRE_X = {**{s: 500.0 for s in GATE_SESSIONS[:4]}, GATE_SESSIONS[4]: 900.0}


def gate_arrays(sid, seed):
    """Keypoints (n, mouse, xy, kp) with sub-pixel jitter (never frozen); mouse b moves to PRE_X over each pre-bout
    window, stays during the bout and returns over the next 2 s."""
    rng = np.random.default_rng(seed)
    bx = np.full(GATE_FRAMES, 700.0)
    for s in BOUT_STARTS:
        bx[s - 60:s] = np.linspace(700.0, PRE_X[sid], 60)
        bx[s:s + 30] = PRE_X[sid]
        bx[s + 30:s + 90] = np.linspace(PRE_X[sid], 700.0, 60)
    kp = np.zeros((GATE_FRAMES, 2, 2, 7))
    for k in range(7):
        kp[:, 0, 0, k] = 300.0 + 3 * k
        kp[:, 0, 1, k] = 300.0 + 2 * k
        kp[:, 1, 0, k] = bx + 3 * k
        kp[:, 1, 1, k] = 300.0 + 2 * k
    kp += rng.normal(0, 0.5, kp.shape)
    return kp, np.full((GATE_FRAMES, 2, 7), 0.9)


def gate_scores():
    parts = []
    for sid in GATE_SESSIONS:
        f = np.arange(GATE_FRAMES)
        d = pd.DataFrame({"session": sid, "frame": f, "time_s": f / 30.0, "attack": 0.1, "mount": 0.1,
                          "investigation": 0.1, "tracking_ok": True})
        for s in BOUT_STARTS:
            d.loc[s:s + 29, "attack"] = 0.9
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def build_gate_inputs(root):
    """The five-session SYNTHETIC validation inputs (pose cache, score archive, manifest) written under root, as
    `inputs` gives them (also used by tests/live_agent_progress_demo.py)."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(GATE_MANIFEST))
    recs = {sid: record(sid, *gate_arrays(sid, i), n=GATE_FRAMES,
                        attacks={sid: [(s, s + 30) for s in BOUT_STARTS]}) for i, sid in enumerate(GATE_SESSIONS)}
    src = root / "source.json"
    src.write_text(json.dumps({"annotator-id_0": recs}))
    cache = root / "cache"
    extract_sessions(src, cache, list(GATE_SESSIONS), manifest)
    table = csv_bytes(gate_scores())
    zip_path = archive(root, {VAL_MEMBER: table, TEST_MEMBER: table})
    return {"zip": zip_path, "sha": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "cache": cache,
            "manifest": manifest, "tmp": root}


@pytest.fixture
def gate_inputs(tmp_path):
    return build_gate_inputs(tmp_path / "five_sessions")


@pytest.fixture
def no_local_outputs(monkeypatch):
    """Fail on any file opened under the repository's outputs/ (the git-ignored genuine investigations)."""
    forbidden = str(ROOT / "outputs")
    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

    def check(path):
        if isinstance(path, (str, os.PathLike)) and os.path.abspath(os.fspath(path)).startswith(forbidden):
            raise AssertionError(f"opened a local output: {path}")
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (check(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(io, "open", lambda f, *a, **k: (check(f), real_io_open(f, *a, **k))[1])
    monkeypatch.setattr(os, "open", lambda f, *a, **k: (check(f), real_os_open(f, *a, **k))[1])


def all_chart_values(at):
    return {json.dumps(v, sort_keys=True) for c in charts(at) for _, rows in c for row in rows for v in row.values()}


def test_a_verified_ok_investigation_renders_and_switching_clears_it(gate_inputs, inputs, tmp_path, monkeypatch,
                                                                     capsys, no_local_outputs):
    root = Path(os.environ[runs.OFFLINE_DIR_ENV])
    met, not_met = root / "gate_met", root / "gate_not_met"
    assert iv.main(argv(gate_inputs, met)) == 0                     # the real Phase 3B.1 run and report writer
    assert iv.main(argv(inputs, not_met)) == 0                      # the 3-session fixture: insufficient
    capsys.readouterr()
    no_runs = tmp_path / "no_reference_runs"                        # the run5 check reads nothing local
    no_runs.mkdir()
    monkeypatch.setenv(RUNS_DIR_ENV, str(no_runs))
    monkeypatch.setattr(runs, "default_inputs", lambda: runs.ValidationInputs(
        label="SYNTHETIC five-session inputs (tests only)", archive=str(gate_inputs["zip"]),
        archive_sha256=gate_inputs["sha"], cache_dir=str(gate_inputs["cache"]),
        manifest=str(gate_inputs["manifest"]), runs_dir=str(no_runs), allow_synthetic=True))

    view = runs.load_saved(met)                                     # the real reload and re-verification
    assert view.verified and view.problems == [] and view.kind == "offline"
    doc = json.loads((met / "investigation_report.json").read_text())
    assert doc["finding"] == "effect_described" and doc["data"]["data_source"] == "synthetic"
    comp = evidence(doc, doc["primary_evidence_id"])
    r = comp["result"]
    assert r["status"] == "ok" and comp["request"].keys().isdisjoint({"min_n_windows", "min_n_sessions",
                                                                     "gate_override_reason"})
    assert r["evidence_gate"] == {"min_n_windows": v1.DEFAULT_MIN_N_WINDOWS,
                                  "min_n_sessions": v1.DEFAULT_MIN_N_SESSIONS, "overridden": False,
                                  "override_reason": None}
    assert r["n_windows"] >= v1.DEFAULT_MIN_N_WINDOWS and r["n_sessions"] >= v1.DEFAULT_MIN_N_SESSIONS
    assert (r["n_windows"], r["n_sessions"]) == (len(GATE_SESSIONS) * len(BOUT_STARTS), len(GATE_SESSIONS))
    assert r["mean_diff"] == statistics.median(p["paired_diff"] for p in r["per_session"])
    signs = {p["paired_diff"] < 0 for p in r["per_session"]}
    assert signs == {True, False}                                   # both directions are recorded

    at = open_page(live_agent_open=str(met))
    assert next(m for m in markdown(at) if m.startswith("**Showing `")).startswith("**Showing `gate_met`")
    assert at.selectbox(key=SELECT + "gate_met").value == comp["evidence_id"]
    (bars, rows), (_, zero), (_, median) = chart(at, x="paired_diff")
    assert [(x["full_session"], x["n_bouts"], x["paired_diff"]) for x in rows] == \
        [(p["session"], p["n_bouts"], p["paired_diff"]) for p in r["per_session"]]
    assert zero == [{"zero": 0.0}] and median == [{"mean_diff": r["mean_diff"]}]
    m = {x.label: x.value for x in at.metric}
    assert m["Reported mean_diff (median across sessions)"] == f"{r['mean_diff']:.4g} px"
    assert m["Contributing sessions"] == str(r["n_sessions"])
    check_exclusions(at, r)
    check_bouts(at, cited_query(doc, comp))
    assert all(x.endswith(":green-badge[met]") for x in markdown(at)
               if x.startswith(("**Valid comparison windows**", "**Contributing sessions**")))
    old = {json.dumps(v) for v in [p["paired_diff"] for p in r["per_session"]] + [r["mean_diff"]]}
    old_sessions = {json.dumps(short(s)) for s in GATE_SESSIONS} | {json.dumps(s) for s in GATE_SESSIONS}

    at.selectbox(key="live_agent_saved").set_value(str(not_met))     # another saved investigation
    at = run(at)
    weak = json.loads((not_met / "investigation_report.json").read_text())
    wcomp = evidence(weak, weak["primary_evidence_id"])
    assert runs.load_saved(not_met).verified and wcomp["result"]["status"] == "insufficient_evidence"
    assert next(m for m in markdown(at) if m.startswith("**Showing `")).startswith("**Showing `gate_not_met`")
    assert chart(at, x="paired_diff") is None                       # the effect is gone
    assert "Reported mean_diff (median across sessions)" not in {x.label for x in at.metric}
    values = all_chart_values(at)
    assert not values & old and not values & old_sessions          # no old bar, median line or session remains
    check_exclusions(at, wcomp["result"])                           # only the new evidence is drawn
    check_bouts(at, cited_query(weak, wcomp))
    (_, observed), _ = chart(at, x="observed")
    assert [x["observed"] for x in observed] == [wcomp["result"]["n_windows"], wcomp["result"]["n_sessions"]]

    at.selectbox(key="live_agent_saved").set_value(str(met))         # and back: the recorded effect again
    at = run(at)
    (_, rows), _, (_, median) = chart(at, x="paired_diff")
    assert [x["paired_diff"] for x in rows] == [p["paired_diff"] for p in r["per_session"]]
    assert median == [{"mean_diff": r["mean_diff"]}]


# ---- the genuine saved investigations (read only) ----

def digests(path):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(path.iterdir()) if f.is_file()}


@pytest.mark.parametrize("key", sorted(GENUINE))
def test_the_genuine_investigations_chart_their_recorded_evidence(monkeypatch, key):
    path = GENUINE[key]
    if not (path / "run.json").is_file():
        pytest.skip(f"{path.name} is not present")
    monkeypatch.delenv(runs.LIVE_DIR_ENV)
    monkeypatch.delenv(runs.OFFLINE_DIR_ENV)
    before = digests(path)
    doc = json.loads((path / "investigation_report.json").read_text())
    comp = evidence(doc, doc["primary_evidence_id"])
    r = comp["result"]
    at = open_page(live_agent_open=str(path))
    (_, rows), (_, z), (_, md) = chart(at, x="paired_diff")
    assert [(x["full_session"], x["paired_diff"]) for x in rows] == [(p["session"], p["paired_diff"])
                                                                     for p in r["per_session"]]
    assert z == [{"zero": 0.0}] and md == [{"mean_diff": r["mean_diff"]}]
    check_exclusions(at, r)
    check_bouts(at, cited_query(doc, comp))
    assert all(m.endswith(":green-badge[met]") for m in markdown(at)
               if m.startswith(("**Valid comparison windows**", "**Contributing sessions**")))
    assert digests(path) == before                                                  # nothing was written
