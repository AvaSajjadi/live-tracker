"""UI-1: the Live Agent's scientific overview and recorded investigation activity (pages/5_Live_Agent.py). New file.

Through the real multipage app (AppTest). New investigations use SYNTHETIC fixtures and the offline OpenRouter
emulator only (no network, no spending). Every expected value is read from the saved report, never hardcoded,
except the reference thresholds the brief states for live_run1. The genuine saved investigations are only read, when
present, and checked to be unchanged.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

import live_tracker.investigation_tools_v1 as v1
import live_tracker.investigation_v1_runs as runs
import live_tracker.investigation_v1_ui as ui
from test_investigation_v1_validation import argv, inputs, iv, recorded_run  # noqa: F401 (fixture)
from test_live_agent_page import (_click, authorize, offline, open_page, page_text, preflight, provider,  # noqa
                                  rehearse_asking, run, saved_dirs, synthetic)

ROOT = Path(__file__).resolve().parents[1]
GENUINE = {"live_run1": ROOT / "outputs" / "investigation_v1_live" / "live_run1",
           "dashboard_run": ROOT / "outputs" / "investigation_v1_live" / "20261024T021351Z_69bed91b",
           "run3": ROOT / "outputs" / "investigation_v1" / "phase3b1_run3"}


def headers(at):
    """The timeline's step headers, in order (each carries a status badge; the timeline follows its heading)."""
    md = [m.value for m in at.markdown]
    start = md.index("#### Recorded investigation activity")
    return [m for m in md[start:] if "-badge[" in m]


def title(header):
    return header.split("**")[1]


def metrics(at):
    return {m.label: m.value for m in at.metric}


def saved(tmp, name):
    path = tmp / "live" / name
    return (json.loads((path / "investigation_report.json").read_text()),
            json.loads((path / "run.json").read_text()))


def g4(x):
    return f"{x:.4g}"


def paid(synthetic, monkeypatch, *, turns=None, **kw):
    _, emu = provider(monkeypatch, **kw)
    at = open_page()
    if turns:
        at.number_input(key="live_agent_turns").set_value(turns)
        at = run(at)
    at = run(_click(authorize(preflight(at)), "Start paid investigation"))
    (name,) = saved_dirs(synthetic["tmp"])
    return at, name, emu


# ---- the overview ----

def test_the_overview_reads_every_value_from_the_saved_report(synthetic):
    name = rehearse_asking(synthetic, "Does the distance between the mice change before predicted attack bouts?")
    doc, s = saved(synthetic["tmp"], name)
    at = open_page()
    comp = next(e["result"] for e in doc["evidence"] if e["evidence_id"] == doc["primary_evidence_id"])
    m = metrics(at)
    assert doc["finding"] == "insufficient_evidence"
    assert m[ui.MAIN_DIFFERENCE] == "not reported"   # no effect is shown
    assert m["Valid comparison windows"] == str(comp["n_windows"])
    assert m["Contributing sessions"] == str(comp["n_sessions"])
    assert m["Actual API spending"] == "$0.00"
    assert m[ui.SIMULATED_COST] == f"${s['cost']['actual_spend_usd']:.5f}"
    md = " ".join(x.value for x in at.markdown)
    assert "> Does the distance between the mice change before predicted attack bouts?" in md
    assert "### Not enough usable evidence for an effect" in md
    assert any(comp["reason"] in c.value and "No effect is reported" in c.value for c in at.caption)
    q = next(e for e in doc["evidence"] if e["tool"] == "query_events")
    assert f"threshold {g4(q['result']['threshold'])}" in md and f"{q['result']['n_bouts']} bouts" in md
    assert f"pre_s = {comp['pre_s']:g} s" in md and f"baseline_gap_s = {comp['baseline_gap_s']:g} s" in md
    windows = comp.get("windows") or v1.WINDOW_DEFINITIONS                 # Contract v1's own definitions
    assert windows["pre"] in md and windows["baseline"] in md
    assert comp.get("mean_diff_definition", v1.MEAN_DIFF_DEFINITION) in md
    overview = [x.value for x in at.markdown if x.value.startswith(("### ", "- **", "> "))]
    overview += [c.value for c in at.caption if c.value.startswith(("Sign:", "The comparison did not meet"))]
    overview.append(next(x.value for x in at.markdown if x.value == runs.av1.FINDING_TEXT[doc["finding"]]))
    for banned in ("p-value", "p =", "confidence interval of", "significant", "causes", "caused", "leads to"):
        assert banned not in " ".join(overview)                 # the disclosures' negations are elsewhere


def test_a_described_effect_reports_value_unit_and_definitional_sign_only(synthetic):
    name = rehearse_asking(synthetic, "Effect question")
    view = runs.load_saved(synthetic["tmp"] / "live" / name)
    assert runs.overview(view)["effect"] is None                # the synthetic fixture never meets the gate
    comp = next(e for e in view.doc["evidence"] if e["evidence_id"] == view.doc["primary_evidence_id"])
    comp["result"].update(status="ok", mean_diff=-12.5)         # an in-memory view only: nothing is written
    view.doc["finding"] = "effect_described"
    assert runs.overview(view)["effect"] == {"value": -12.5, "unit": "px", "feature": "dist_center",
                                             "sign": "negative"}
    comp["result"]["mean_diff"] = 3.0
    assert runs.overview(view)["effect"]["sign"] == "positive"


# ---- the recorded investigation activity ----

def test_the_timeline_follows_the_recorded_trace_with_follow_up_comparisons(synthetic):
    name = rehearse_asking(synthetic, "Timeline question")
    doc, _ = saved(synthetic["tmp"], name)
    at = open_page()
    tools = [t["tool"] for t in doc["trace"] if t["kind"] in ("tool_call", "finish")]
    want = [runs.TOOL_TITLES[t][0] for t in tools] + [f"Finding: {doc['finding']}"]
    hs = headers(at)
    assert [title(h) for h in hs] == want
    assert tools.count("compare_windows") == 3 and "check_tracking_quality" in tools
    compares = [h for h in hs if "`compare_windows`" in h]
    assert all(":orange-badge[accepted: insufficient evidence]" in h for h in compares)
    chosen = [m.value for m in at.markdown if m.value.startswith("Scripted by the emulator (no model):")]
    pre = [t["arguments"]["pre_s"] for t in doc["trace"] if t.get("tool") == "compare_windows"]
    assert [next(c for c in chosen if f"`pre_s` = {p}" in c) for p in pre]          # each follow-up's own pre_s
    assert any("`bouts from` = step 1's query" in c for c in chosen)
    support = next(m.value for m in at.markdown if m.value.startswith("Supporting evidence:"))
    for o in doc["observations"]:
        assert f"`{o['field']}`" in support
    assert any("Recorded investigation activity" in m.value for m in at.markdown)
    assert any("not a live progress stream" in c.value for c in at.caption)
    assert hs[-1].endswith(":green-badge[investigation finished]")


def test_refused_calls_appear_as_refused_steps(synthetic, monkeypatch):
    at, name, _ = paid(synthetic, monkeypatch, turns=9, faults={1: "unknown_tool", 2: "extra_argument"})
    doc, _ = saved(synthetic["tmp"], name)
    hs = headers(at)
    assert title(hs[0]) == "Unknown tool run_python" and ":red-badge[refused]" in hs[0]
    assert title(hs[1]) == "Find predicted behavior bouts" and ":red-badge[refused]" in hs[1]
    assert any("message: **" in m.value and "min_n_windows" in m.value for m in at.markdown)
    assert ":green-badge[accepted as evidence]" in hs[2] and doc["finished"]
    refused = [t for t in doc["trace"] if t.get("outcome") == "refused"]
    assert len([h for h in hs if ":red-badge[refused]" in h]) == len(refused) == 2


def test_an_incomplete_investigation_ends_without_a_finding(synthetic, monkeypatch):
    at, name, emu = paid(synthetic, monkeypatch, cost=0.25)
    doc, _ = saved(synthetic["tmp"], name)
    assert not doc["finished"] and len(emu.requests) == 2
    hs = headers(at)
    assert [title(h) for h in hs][-2:] == ["Model request stopped", "Ended without a finding"]
    assert ":orange-badge[incomplete]" in hs[-1]
    md = " ".join(m.value for m in at.markdown)
    assert "### Stopped before a finding" in md
    assert any("Scientific report: verified, incomplete." in w.value for w in at.warning)
    assert metrics(at)[ui.MAIN_DIFFERENCE] == "not reported"
    assert any(f"stopped ({doc['stopping_reason']})" in c.value for c in at.caption)


def test_a_budget_stop_is_shown_in_the_timeline(synthetic, monkeypatch):
    at, name, _ = paid(synthetic, monkeypatch, cost=0.01, faults={2: "huge_usage"})
    titles = [title(h) for h in headers(at)]
    assert titles[-3:] == ["Model reply not used", "Budget limit reached", "Ended without a finding"]


def test_a_reference_mismatch_keeps_the_verified_report_prominent(synthetic, monkeypatch):
    iv = runs.script("investigation_v1_validation")
    real = iv.OfflineInvestigatorV1.decide

    def own_threshold(self, done):
        name, args = real(self, done)
        return name, ({k: v for k, v in args.items() if k != "threshold"} if name == "query_events" else args)
    monkeypatch.setattr(iv.OfflineInvestigatorV1, "decide", own_threshold)
    name = rehearse_asking(synthetic, "Mismatch question")
    _, s = saved(synthetic["tmp"], name)
    at = open_page()
    ref = runs.reference_check(runs.load_saved(synthetic["tmp"] / "live" / name))
    row = next(r for r in ref["rows"] if r["parameter"] == "threshold")
    panel = next(i.value for i in at.info if "Reference comparison" in i.value)
    assert s["status"] == "refused"
    assert f"not equivalent.** Agent threshold {row['investigation (applied)']} (not set by the agent" in panel
    assert f"reference threshold {row[runs.REFERENCE_RUN + ' (applied)']}" in panel
    assert any("Scientific report: verified." in x.value for x in at.success)
    assert any("set by the reference compatibility check; the scientific report itself verifies" in c.value
               for c in at.caption)
    assert not any("refused" in x.value.lower() for x in at.error)


# ---- who chose the actions, and what a finish is ----

# (the canonical disclosures, kept verbatim, say "the model wrote no number" generically, so they are not scanned for)
MODEL_WORDING = ("Chosen by the model", "What the model did", "Model's recorded text", "Model reply", "Model request")


def chosen_lines(at):
    return [m.value for m in at.markdown if m.value.startswith(("Chosen by", "Scripted by", "Set by", "Recorded par"))]


def test_an_emulated_rehearsal_is_never_attributed_to_a_model(synthetic):
    name = rehearse_asking(synthetic, "Attribution question")
    at = open_page()
    text = page_text(at)
    assert chosen_lines(at) and all(c.startswith("Scripted by the emulator (no model): ") for c in chosen_lines(at))
    assert not any(w in text for w in MODEL_WORDING)
    assert any("Emulated rehearsal (no model, nothing spent)" in c.value and "No model ran" in c.value
               for c in at.caption)
    view = runs.load_saved(synthetic["tmp"] / "live" / name)
    view.name, view.path = "live_run1", Path("live_run1")                  # a directory name decides nothing
    assert runs.attribution(view)["mode"] == "emulated"
    view.summary = {**view.summary, "mode": runs.script("investigation_v1_live").LIVE_MODE}    # in memory only
    assert runs.attribution(view)["mode"] == "paid_model"                   # the recorded mode does


def test_a_paid_run_is_attributed_to_its_recorded_model(synthetic, monkeypatch):
    at, name, _ = paid(synthetic, monkeypatch)
    _, s = saved(synthetic["tmp"], name)
    model = s["model"]["model"]
    assert chosen_lines(at) and all(c.startswith(f"Chosen by the model (`{model}`): ") for c in chosen_lines(at))
    assert any(f"requested by the model `{model}` through paid OpenRouter requests" in c.value for c in at.caption)


def test_an_offline_deterministic_investigation_is_never_attributed_to_a_model(synthetic, capsys):
    path = Path(os.environ[runs.OFFLINE_DIR_ENV]) / "offline_run"            # a SYNTHETIC Phase 3B.1 investigation
    assert iv.main(argv(synthetic, path)) == 0
    capsys.readouterr()
    view = runs.load_saved(path)
    assert view.verified and view.kind == "offline" and runs.attribution(view)["mode"] == "offline"
    at = open_page(live_agent_open=str(path))
    assert chosen_lines(at) and all(c.startswith("Set by the deterministic procedure (no model): ")
                                    for c in chosen_lines(at))
    assert not any(w in page_text(at) for w in MODEL_WORDING)
    assert any("offline deterministic investigator (no model)" in x.value for x in at.success)


def test_a_finish_is_a_finding_accepted_and_never_evidence(synthetic):
    name = rehearse_asking(synthetic, "Finish question")
    doc, s = saved(synthetic["tmp"], name)
    at = open_page()
    hs = headers(at)
    finish = [h for h in hs if "`finish_investigation`" in h]
    assert len(finish) == 1 and finish[0].endswith(":green-badge[finding accepted]")
    assert not any("finish_investigation" in h and "evidence" in h.split("-badge[")[1] for h in hs)
    ok = [t for t in doc["trace"] if t["kind"] == "tool_call" and
          next(e for e in doc["evidence"] if e["evidence_id"] == t["evidence_id"])["status"] == "ok"]
    assert len([h for h in hs if h.endswith(":green-badge[accepted as evidence]")]) == len(ok) > 0
    assert hs[-1].endswith(":green-badge[investigation finished]")
    steps = runs.timeline(doc, "Emulated")
    fin = next(x for x in steps if x.get("tool") == "finish_investigation")
    assert fin["status"] == "finding_accepted" and fin["evidence_id"] is None
    support = next(m.value for m in at.markdown if m.value.startswith("Supporting evidence:"))
    for o in doc["observations"]:                                           # the observations are kept
        assert f"`{o['field']}`" in support
    labels = metrics(at)
    assert "simulated" in ui.SIMULATED_COST.lower() and "not spent" in ui.SIMULATED_COST       # the qualifier
    assert labels[ui.SIMULATED_COST] == f"${s['cost']['actual_spend_usd']:.5f}"


def test_the_plain_language_comparison_uses_the_investigations_parameters(synthetic):
    name = rehearse_asking(synthetic, "Plain question")
    doc, _ = saved(synthetic["tmp"], name)
    at = open_page()
    comp = next(e["result"] for e in doc["evidence"] if e["evidence_id"] == doc["primary_evidence_id"])
    q = next(e["result"] for e in doc["evidence"] if e["tool"] == "query_events")
    md = [m.value for m in at.markdown]
    order = [i for i, m in enumerate(md) if m in ("**In plain terms**", "**Contract v1 definitions**")]
    assert len(order) == 2 and order[0] < order[1]                          # before the exact formulas
    plain = " ".join(md[order[0]:order[1]])
    pre, base, gap = comp["pre_s"], comp["baseline_s"], comp["baseline_gap_s"]
    assert f"the {q['n_bouts']} predicted `{q['behavior']}` bouts" in plain
    assert f"`{comp['feature']}`" in plain and f"the {pre:g} s immediately before the bout starts" in plain
    assert f"from {pre + gap + base:g} s to {pre + gap:g} s before the bout starts" in plain
    assert "pre-event value minus the baseline value" in plain and "median of those session values" in plain
    for banned in ("p-value", "confidence interval", "significant", "causes", "caused", "leads to"):
        assert banned not in plain


# ---- the genuine saved investigations (read only) ----

def digests(path):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(path.iterdir()) if f.is_file()}


@pytest.mark.parametrize("key", sorted(GENUINE))
def test_the_genuine_investigations_are_summarized_from_their_reports(monkeypatch, key):
    path = GENUINE[key]
    if not (path / "run.json").is_file():
        pytest.skip(f"{path.name} is not present")
    monkeypatch.delenv(runs.LIVE_DIR_ENV)
    monkeypatch.delenv(runs.OFFLINE_DIR_ENV)
    before = digests(path)
    doc = json.loads((path / "investigation_report.json").read_text())
    s = json.loads((path / "run.json").read_text())
    comp = next(e["result"] for e in doc["evidence"] if e["evidence_id"] == doc["primary_evidence_id"])
    at = open_page(live_agent_open=str(path))
    m = metrics(at)
    assert m[ui.MAIN_DIFFERENCE] == f"{g4(comp['mean_diff'])} px"
    assert m["Valid comparison windows"] == str(comp["n_windows"])
    assert m["Contributing sessions"] == str(comp["n_sessions"])
    assert m["Actual API spending"] == (f"${s['cost']['actual_spend_usd']:.5f}" if s.get("mode") == "LIVE: paid "
                                        "OpenRouter requests" else "$0.00")
    assert any("Scientific report: verified." in x.value for x in at.success)
    tools = [t["tool"] for t in doc["trace"] if t["kind"] in ("tool_call", "finish")]
    assert [title(h) for h in headers(at)] == [runs.TOOL_TITLES[t][0] for t in tools] + ["Finding: effect_described"]
    md = " ".join(x.value for x in at.markdown)
    assert f"baseline_gap_s = {comp['baseline_gap_s']:g} s" in md
    want = ("Set by the deterministic procedure (no model): " if key == "run3" else
            f"Chosen by the model (`{s['model']['model']}`): ")
    assert chosen_lines(at) and all(c.startswith(want) for c in chosen_lines(at))
    if key == "run3":
        assert not any(w in page_text(at) for w in MODEL_WORDING)
    assert headers(at)[-2].endswith(":green-badge[finding accepted]")                  # the finish step
    if key == "run3":
        assert any("Reference comparison (run5): equivalent." in x.value for x in at.success)
    else:
        panel = next(i.value for i in at.info if "Reference comparison" in i.value)
        assert "not equivalent" in panel and "Agent threshold 0.5" in panel and "reference threshold 0.6" in panel
    assert digests(path) == before                                          # nothing was written
