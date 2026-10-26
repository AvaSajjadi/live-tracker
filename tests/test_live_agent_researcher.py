"""UI-4: the researcher's journey on the Live Agent page (investigation_v1_ui.render_page). New file.

Through the real multipage app (AppTest). SYNTHETIC fixtures and the in-process emulated OpenRouter endpoint only: no
network, no spending, no real data, no genuine saved output.
"""

import json

import pytest

import live_tracker.investigation_live_v1 as lv1
import live_tracker.investigation_v1_runs as runs
import live_tracker.investigation_v1_ui as ui
from test_investigation_v1_validation import inputs, recorded_run  # noqa: F401 (fixture)
from test_live_agent_page import (_click, authorize, button, offline, open_page, page_text, preflight,  # noqa: F401
                                  provider, rehearse_asking, run, saved_dirs, showing, synthetic)
from test_run_lock import child


def markdown(at):
    return [m.value for m in at.markdown]


def checklist(at):
    """The paid run's requirement lines: {text: met}."""
    lines = next(m for m in markdown(at) if "Authorization text typed exactly" in m).split("  \n")
    return {x.split("] ", 1)[1]: x.startswith(":green-badge[ready]") for x in lines}


def live(synthetic, name):
    return synthetic["tmp"] / "live" / name


# ---- A/B: the page explains itself; every action says what it is ----

def test_the_page_introduces_what_it_does_and_labels_every_action(synthetic):
    at = open_page()
    intro = " ".join(ui.INTRO_POINTS)
    for point in ("chooses every Contract v1 tool call", "predicted by a classifier", "not experimentally confirmed",
                  "verified CalMS21 validation inputs", "descriptive association", "No causal claim",
                  "reloaded and verified", "not a validated scientific instrument",
                  "deterministic workflow"):
        assert point in intro
    assert any(e.label.startswith("What this page does") for e in at.expander)
    md = markdown(at)
    for step in ("Step 1 · Your research question and the data", "Step 2 · Model and execution limits",
                 "Step 3 · Choose how to run"):
        assert f"##### {step}" in md
    assert any(ui.ACTION_BADGES["rehearsal"] in m and "Free rehearsal" in m for m in md)
    assert any(ui.ACTION_BADGES["preflight"] in m and "Readiness check" in m for m in md)
    assert any(ui.ACTION_BADGES["paid"] in m and "Paid autonomous investigation" in m for m in md)
    assert any("not a guaranteed maximum charge" in c.label for c in at.checkbox)
    assert button(at, "Start paid investigation").disabled
    assert not button(at, "Run free rehearsal").disabled


# ---- B: readiness, stale preflight, one use per preflight ----

def test_the_readiness_checklist_follows_every_requirement(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    at = open_page()
    first = checklist(at)
    assert first == {"Verified validation inputs present": True,
                     "Run the readiness check (step B1) for this configuration": False,
                     "Spending target confirmed": False, "Authorization text typed exactly": False,
                     "No other investigation running": True}
    at = preflight(at)
    assert checklist(at)["Readiness check passed for exactly this configuration (usable once)"]
    at.text_area(key="live_agent_question").input("A changed question")       # another configuration
    at = run(at)
    assert "The configuration changed after the last readiness check: run it again" in checklist(at)
    assert button(at, "Start paid investigation").disabled
    at = authorize(preflight(at))
    assert all(checklist(at).values()) and not button(at, "Start paid investigation").disabled
    at = run(_click(at, "Start paid investigation"))
    assert len(saved_dirs(synthetic["tmp"])) == 1 and emu.requests                # one paid (emulated) run
    after = checklist(at)
    assert after["That readiness check was already used for one paid run: run it again for another"] is False
    assert button(at, "Start paid investigation").disabled                         # one use per preflight


def test_a_running_investigation_disables_both_runs_and_says_why(synthetic):
    holder = child(synthetic["tmp"] / "live", "hold")
    try:
        assert holder.stdout.readline().strip() == "entered"
        at = open_page()
        assert button(at, "Run free rehearsal").disabled and button(at, "Start paid investigation").disabled
        assert any("Only one investigation can run at a time" in w.value for w in at.warning)
        assert checklist(at)["No other investigation running"] is False
    finally:
        holder.kill()
        holder.wait(10)
    assert saved_dirs(synthetic["tmp"]) == []


def test_missing_inputs_disable_every_run_and_nothing_is_substituted(synthetic, monkeypatch):
    missing = runs.ValidationInputs(label="CalMS21 validation inputs (missing in this test)",
                                    archive=str(synthetic["tmp"] / "nowhere.zip"), archive_sha256="0" * 64,
                                    cache_dir=str(synthetic["tmp"] / "no_cache"), manifest=str(synthetic["manifest"]),
                                    allow_synthetic=True)
    monkeypatch.setattr(runs, "default_inputs", lambda: missing)
    at = open_page()
    assert any("score archive not found" in e.value and "Nothing is substituted" in e.value for e in at.error)
    assert any("Nothing synthetic is ever substituted" in i.value for i in at.info)
    assert button(at, "Run free rehearsal").disabled and button(at, "Start paid investigation").disabled
    assert checklist(at)["Verified validation inputs present"] is False


# ---- C: the end of a run, and the way to its results ----

def test_a_completed_run_opens_its_verified_report_without_running_again(synthetic):
    at = open_page()
    at = run(_click(at, "Run free rehearsal"))
    (name,) = saved_dirs(synthetic["tmp"])
    assert any(f"Investigation {name} finished" in s.value and "Outcome:" in s.value for s in at.success)
    at = run(_click(at, "Open this investigation in Saved investigations"))
    assert at.session_state[ui.TAB_KEY] == "Saved investigations"
    assert showing(at) == (name, name) and saved_dirs(synthetic["tmp"]) == [name]  # nothing ran again
    assert not any("finished with recorded status" in s.value for s in at.success)  # the message was one-time
    at = run(at)
    assert at.session_state[ui.TAB_KEY] == "Saved investigations" and saved_dirs(synthetic["tmp"]) == [name]


def test_an_incomplete_run_is_reported_as_such(synthetic, monkeypatch):
    provider(monkeypatch, faults={2: "http_500"})
    at = run(_click(authorize(preflight(open_page())), "Start paid investigation"))
    (name,) = saved_dirs(synthetic["tmp"])
    text = next(w.value for w in at.warning if f"Investigation {name} finished" in w.value)
    assert "stopped before a finding (`model_unavailable`)" in text and "verified but incomplete" in text
    assert "A descriptive association" not in text
    label = at.selectbox(key=ui.SAVED_KEY).format_func(str(live(synthetic, name)))
    assert "incomplete: stopped before a finding (model_unavailable)" in label


def test_an_unverifiable_report_is_never_described(synthetic, monkeypatch):
    real = runs.load_saved

    def tampered(path):
        saved = real(path)
        saved.problems = ["the report's fingerprint does not match its content"]
        return saved
    monkeypatch.setattr(runs, "load_saved", tampered)
    at = run(_click(open_page(), "Run free rehearsal"))
    (name,) = saved_dirs(synthetic["tmp"])
    text = next(e.value for e in at.error if f"Investigation {name} finished" in e.value)
    assert "None of its results are shown" in text and "Outcome" not in text
    assert at.selectbox(key=ui.SAVED_KEY).format_func(str(live(synthetic, name))).endswith("does NOT verify")
    assert any("Nothing from its files is trusted" in c.value for c in at.caption)


def test_the_outcome_message_reports_recorded_states_only(synthetic):
    name = rehearse_asking(synthetic, "Outcome question")
    saved = runs.load_saved(live(synthetic, name))
    summary = json.loads((live(synthetic, name) / "run.json").read_text())
    kind, text, path = ui.outcome_message(name, summary, saved, "verified", str(live(synthetic, name)))
    assert kind == ("success" if saved.doc["finished"] else "warning") and path.endswith(name)
    assert f"finding `{saved.doc['finding']}`" in text and "association only" in text
    over = {**summary, "status": "over_target", "cost": {"actual_spend_complete": False}}
    kind, text, _ = ui.outcome_message(name, over, saved, "verified", "p")          # a rehearsal: simulated only
    assert kind == "warning" and "simulated cost went past the local target" in text and "may be higher" not in text
    kind, text, _ = ui.outcome_message(name, summary, None, "the saved investigation does NOT verify", "p")
    assert kind == "error" and saved.doc["finding"] not in text


# ---- the completion banner's spending: actual for a paid run, simulated (and $0 actual) for a rehearsal ----

def banner_after(synthetic, monkeypatch, paid, faults):
    """Run one investigation on the page (emulated endpoint, synthetic inputs; a paid run through the mocked
    provider); (the completion banner, its run.json)."""
    from test_live_agent_page import FAKE_OR
    if paid:
        provider(monkeypatch, faults=faults)
        at = authorize(preflight(open_page()))
        at = run(_click(at, "Start paid investigation"))
    else:
        offline_emu = runs.script("investigation_v1_live").EmulatedOpenRouterV1
        monkeypatch.setattr(runs, "REHEARSAL_EMULATOR_FACTORY", lambda: offline_emu(faults))
        at = run(_click(open_page(), "Run free rehearsal"))
    (name,) = saved_dirs(synthetic["tmp"])
    summary = json.loads((live(synthetic, name) / "run.json").read_text())
    (banner,) = [m for m in (*at.success, *at.warning, *at.error) if "finished with recorded status" in m.value]
    assert FAKE_OR not in page_text(at)
    return banner.value, summary


@pytest.mark.parametrize("paid, faults, complete", [(False, {}, True), (False, {3: "http_500"}, False),
                                                    (True, {}, True), (True, {3: "http_500"}, False)])
def test_the_completion_banner_states_actual_and_simulated_spending_correctly(synthetic, monkeypatch, paid, faults,
                                                                              complete):
    text, summary = banner_after(synthetic, monkeypatch, paid, faults)
    assert summary["mode"] == (runs.script("investigation_v1_live").LIVE_MODE if paid else
                               runs.script("investigation_v1_live").DRY_MODE)
    assert summary["cost"]["actual_spend_complete"] is complete                  # the case under test
    unknown_actual = "Some requests reported no provider cost, so the actual spend may be higher than shown."
    if paid:
        assert ("may be higher" in text) is (not complete)
        assert (unknown_actual in text) is (not complete) and "simulated" not in text
    else:
        assert "may be higher" not in text and "actual spend" not in text        # never unknown ACTUAL charges
        assert "actual API spending is $0" in text and "usage and costs are simulated" in text
        assert ("simulated accounting is incomplete" in text) is (not complete)
    assert "ceiling" not in text and "guarantee" not in text


# ---- D: finding saved investigations ----

def test_no_saved_investigation_yet_says_how_to_start(synthetic):
    at = open_page()
    assert any("No saved investigation yet" in i.value and "free rehearsal" in i.value for i in at.info)


def test_runs_are_listed_by_save_time_not_by_their_names(synthetic, monkeypatch):
    names = ["20261024T120000Z_ffffffff", "20261024T120000Z_00000000"]   # same second, reverse lexical order
    queue_ = list(names)
    monkeypatch.setattr(runs, "new_run_name", lambda: queue_.pop(0))
    rehearse_asking(synthetic, "First saved")
    rehearse_asking(synthetic, "Second saved")
    at = open_page()
    options = [o for o in at.selectbox(key=ui.SAVED_KEY).options]
    assert "Second saved" in options[0] and "First saved" in options[1]          # newest first, by save time
    assert names[1] in at.selectbox(key=ui.SAVED_KEY).value          # though "…00000000" sorts first by name
    assert any("2 saved investigations, most recently saved first" in c.value for c in at.caption)


def test_a_damaged_run_is_listed_by_name_only_and_never_trusted(synthetic):
    good = rehearse_asking(synthetic, "A good run")
    bad = rehearse_asking(synthetic, "A run to damage")
    report = live(synthetic, bad) / "investigation_report.json"
    doc = json.loads(report.read_text())
    doc["question"] = "PRIVATE-MARKER injected question"
    report.write_text(json.dumps(doc))
    at = open_page()
    sb = at.selectbox(key=ui.SAVED_KEY)
    labels = [sb.format_func(str(live(synthetic, n))) for n in (good, bad)]
    assert "A good run" in labels[0] and labels[1].endswith(f"{bad} · does NOT verify")
    assert "PRIVATE-MARKER" not in " ".join(sb.options)
    sb.set_value(str(live(synthetic, bad)))
    at = run(at)
    assert any("does NOT verify" in e.value for e in at.error)
    assert "PRIVATE-MARKER" not in page_text(at)
    assert any("1 not verifying" in c.value for c in at.caption)


# ---- E: from the summary to the details, for the selected run only ----

def test_the_evidence_path_is_the_selected_runs_verified_chain(synthetic):
    a = rehearse_asking(synthetic, "Path question A")
    b = rehearse_asking(synthetic, "Path question B")
    at = open_page()
    for name, question in ((a, "Path question A"), (b, "Path question B"), (a, "Path question A")):
        at.selectbox(key=ui.SAVED_KEY).set_value(str(live(synthetic, name)))
        at = run(at)
        saved = runs.load_saved(live(synthetic, name))
        lines = ui.evidence_path(saved, runs.overview(saved, runs.reference_check(saved)))
        shown = next(m for m in markdown(at) if m.startswith("1. **Question:**"))
        other = "Path question B" if question.endswith("A") else "Path question A"
        assert f"“{question}”" in shown and other not in shown              # the selected run's chain only
        for i, (title, text, anchor) in enumerate(lines, 1):
            assert f"{i}. **{title}:** {text} · [details](#{anchor})" in shown
        titles = [t for t, _, _ in lines]
        assert titles[0] == "Question" and titles[-1] == "Outcome" and "Evidence used" in titles
        d = saved.doc
        assert next(x for t, x, _ in lines if t == "Outcome").endswith(
            f"(`{d['finding']}`)" if d["finished"] else f"(`{d['stopping_reason']}`)")
        assert "#### How this result was reached" in markdown(at)


def test_navigating_and_reopening_never_calls_a_model(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    a = rehearse_asking(synthetic, "Navigate A")
    b = rehearse_asking(synthetic, "Navigate B")
    at = preflight(open_page())                                      # metadata only
    for name in (a, b, a):
        at.selectbox(key=ui.SAVED_KEY).set_value(str(live(synthetic, name)))
        at = run(at)
    assert emu.requests == [] and len(saved_dirs(synthetic["tmp"])) == 2


# ---- Codex audit 1: the catalog trusts only what verifies NOW ----

def test_codex_tampering_with_preserved_size_and_times_is_never_trusted(synthetic):
    import os
    kept = rehearse_asking(synthetic, "An untouched run")
    name = rehearse_asking(synthetic, "Question QQQ about distance")
    at = open_page()
    sb = at.selectbox(key=ui.SAVED_KEY)
    assert "“Question QQQ about distance”" in sb.format_func(str(live(synthetic, name)))
    assert any("2 with a finding" in c.value for c in at.caption)
    report = live(synthetic, name) / "investigation_report.json"
    before = report.stat()
    raw = report.read_bytes()
    tampered = raw.replace(b"Question QQQ about distance", b"Question ZZZ about distance")
    assert tampered != raw and len(tampered) == len(raw)              # same length
    report.write_bytes(tampered)
    os.utime(report, ns=(before.st_atime_ns, before.st_mtime_ns))     # the original times restored
    assert (report.stat().st_size, report.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert not runs.load_saved(live(synthetic, name)).verified        # fresh verification rejects it
    at = run(at)                                                      # the catalog, refreshed
    sb = at.selectbox(key=ui.SAVED_KEY)
    label = sb.format_func(str(live(synthetic, name)))
    assert label.endswith(f"{name} · does NOT verify") and "QQQ" not in label and "ZZZ" not in label
    assert "“An untouched run”" in sb.format_func(str(live(synthetic, kept)))   # unchanged runs keep their label
    counts = next(c.value for c in at.caption if "saved investigations, most recently saved first" in c.value)
    assert "1 not verifying" in counts and "2 with a finding" not in counts
    sb.set_value(str(live(synthetic, name)))
    at = run(at)
    assert any("does NOT verify" in e.value for e in at.error)
    text = page_text(at)
    assert "ZZZ" not in text and "#### How this result was reached" not in markdown(at)


# ---- Codex audit 2: a malformed run is contained; the others stay usable ----

OFFLINE_MODEL = runs.script("investigation_v1_validation").OFFLINE_MODEL
MALFORMED_RUN_JSON = ["[]", "null", '"text"', "[1, 2, 3]", "{not json", '{"model": [1]}',
                      json.dumps({"model": {"adapter": OFFLINE_MODEL}, "outputs": [1]})]


@pytest.mark.parametrize("content", MALFORMED_RUN_JSON)
def test_codex_a_malformed_run_json_is_contained_and_other_runs_stay_readable(synthetic, content):
    good = rehearse_asking(synthetic, "Valid run A")
    bad = rehearse_asking(synthetic, "Run B before damage")
    (live(synthetic, bad) / "run.json").write_text(content)
    at = open_page()
    assert not at.exception
    sb = at.selectbox(key=ui.SAVED_KEY)
    assert sb.format_func(str(live(synthetic, bad))).endswith(f"{bad} · does NOT verify")
    for target in (good, bad, good, bad):                             # valid -> damaged -> valid
        at.selectbox(key=ui.SAVED_KEY).set_value(str(live(synthetic, target)))
        at = run(at)
        assert not at.exception                                       # contained, not an uncaught error
        if target == bad:
            assert any("could not be loaded or verified" in e.value or "does NOT verify" in e.value
                       for e in at.error)
            assert not any(m.startswith("**Showing `") for m in markdown(at))      # no report, A's included
            assert "Valid run A" not in " ".join(markdown(at)) and "Run B before damage" not in page_text(at)
            assert "#### Scientific evidence" not in markdown(at)
        else:
            assert showing(at) == (good, good) and "#### Scientific evidence" in markdown(at)
    assert (live(synthetic, bad) / "run.json").read_text() == content        # nothing repaired or rewritten


def files_of(synthetic):
    return {p: p.read_bytes() for p in sorted((synthetic["tmp"] / "live").rglob("*")) if p.is_file()}


@pytest.mark.parametrize("stage", ["catalog", "selected"])
@pytest.mark.parametrize("error", [AttributeError, TypeError])
def test_a_programming_error_in_the_loader_propagates_and_is_never_called_malformed(synthetic, monkeypatch, error,
                                                                                     stage):
    """Valid synthetic files; the loader itself fails. Injected while building the catalog (the first load of each
    run) or only while loading the selected report (its second load, after the catalog has listed it)."""
    _, emu = provider(monkeypatch)
    rehearse_asking(synthetic, "Any valid run")
    before, dirs = files_of(synthetic), saved_dirs(synthetic["tmp"])
    real, calls = runs.load_saved, []

    def broken(path):
        calls.append(path)
        if stage == "catalog" or len(calls) > 1:
            raise error("injected programming error")
        return real(path)
    monkeypatch.setattr(runs, "load_saved", broken)
    from streamlit.testing.v1 import AppTest
    from test_live_agent_page import PAGE, ROOT
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    at.switch_page(PAGE).run()                                       # (open_page would assert no exception)
    assert len(calls) == (1 if stage == "catalog" else 2)
    assert at.exception and "injected programming error" in at.exception[0].message     # the original, propagated
    assert at.exception[0].proto.type == error.__name__                  # the same exception class
    text = " ".join(e.value for e in at.error)
    assert "could not be loaded or verified" not in text and "malformed" not in text   # not called bad data
    assert "does NOT verify" not in text and not any("not verifying" in c.value for c in at.caption)  # no verdict
    assert not any(m.startswith("**Showing `") for m in markdown(at))
    assert emu.requests == [] and saved_dirs(synthetic["tmp"]) == dirs and files_of(synthetic) == before


# ---- Codex audit 3: only what Contract v1 exposes ----

def test_every_example_question_uses_a_supported_behavior_and_feature(synthetic):
    import live_tracker.investigation_tools_v1 as v1
    for question, behavior, feature in ui.EXAMPLE_QUESTIONS:
        assert behavior in ui.QUERY_BEHAVIORS and feature in v1.WINDOW_FEATURES
        assert behavior in question and "classifier-predicted" in question
    assert set(ui.QUERY_BEHAVIORS) == {"attack", "investigation", "mount"}
    intro = " ".join(ui.INTRO_POINTS)
    for feature in v1.WINDOW_FEATURES:
        assert f"`{feature}`" in intro
    assert "not available to this agent" in intro
    at = open_page()
    examples = next(c.value for c in at.caption if c.value.startswith("Examples:"))
    for question, _, _ in ui.EXAMPLE_QUESTIONS:
        assert question in examples
    for unsupported in ("facing", "face each other", "resident's speed", "orientation"):
        assert unsupported not in examples


# ---- UI-5: the app's shared style, and colours that carry a meaning ----

def test_the_page_uses_the_shared_style_once_with_one_title_and_one_set_of_tabs(synthetic):
    from live_tracker import ui_style
    at = open_page()
    assert [m.value.strip() for m in at.markdown].count(ui_style._CSS.strip()) == 1   # Ava's stylesheet, once
    assert [t.value for t in at.title] == ["Live Agent"]
    assert len(at.tabs) == 2 and [t.label for t in at.tabs] == list(ui.TABS)
    assert sum(1 for e in at.expander if e.label.startswith("What this page does")) == 1
    intro = markdown(at)
    assert "  ".join(f":primary-badge[{i}] {s}" for i, s in enumerate(ui.JOURNEY, 1)) in intro
    assert any(c.value.startswith("An autonomous agent investigates your question") for c in at.caption)
    assert ui.NOTICE_MEANINGS in intro and "red means something was refused" in ui.NOTICE_MEANINGS


PALETTE = (":green-badge[", ":orange-badge[", ":red-badge[", ":gray-badge[", ":primary-badge[")


def test_every_badge_is_in_the_shared_palette_and_paid_asks_for_a_decision(synthetic):
    import re
    assert ui.ACTION_BADGES["paid"].startswith(":orange-badge[") and "needs your authorization" in \
        ui.ACTION_BADGES["paid"]
    assert not any(b.startswith((":green-badge[", ":red-badge[")) for b in ui.ACTION_BADGES.values())
    rehearse_asking(synthetic, "Palette question")
    at = open_page()
    at = run(_click(at, "Run free rehearsal"))
    view_lines = "\n".join(at.session_state[ui.PROGRESS_KEY]["lines"])
    shown = "\n".join(markdown(at)) + "\n" + view_lines
    used = set(re.findall(r":[a-z]+-badge\[", shown))
    assert used and all(b in PALETTE for b in used), used
    assert ":primary-badge[reply]" in view_lines or ":gray-badge[waiting]" in view_lines


def test_the_readiness_summary_counts_the_checklist(synthetic, monkeypatch):
    provider(monkeypatch)
    at = open_page()
    assert any(m.startswith("**Requirements for the paid run: 2 of 5 met** :gray-badge[not ready]")
               for m in markdown(at))
    at = authorize(preflight(at))
    assert any(m.startswith("**Requirements for the paid run: 5 of 5 met** :green-badge[ready to start]")
               for m in markdown(at))
    assert not button(at, "Start paid investigation").disabled


def test_long_explanations_are_short_notices_with_more_info(synthetic):
    at = open_page()
    captions = [c.value for c in at.caption]
    assert "The same pipeline over an emulated endpoint. Nothing is spent." in captions
    assert any("its costs are simulated" in c for c in captions)             # the full text, under More info
    assert sum(1 for e in at.expander if e.label == "More info") >= 2


def test_the_selected_saved_run_shows_its_verification_state_as_badges(synthetic):
    good = rehearse_asking(synthetic, "Badge question")
    bad = rehearse_asking(synthetic, "Badge PRIVATE question")
    report = live(synthetic, bad) / "investigation_report.json"
    report.write_text(report.read_text().replace("Badge PRIVATE question", "Badge PRIVATE qu3stion"))
    at = open_page()
    for name in (good, bad, good):
        at.selectbox(key=ui.SAVED_KEY).set_value(str(live(synthetic, name)))
        at = run(at)
        md = markdown(at)
        if name == good:
            saved = runs.load_saved(live(synthetic, name))
            state = ":green-badge[verified · finding]" if saved.doc["finished"] else \
                ":orange-badge[verified · incomplete]"
            assert f"{state} :gray-badge[free rehearsal]" in md
        else:
            assert ":red-badge[does NOT verify] nothing of it is shown" in md
            assert "PRIVATE" not in page_text(at) and not any(m.startswith("**Showing `") for m in md)
