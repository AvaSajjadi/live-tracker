"""Phase 3D / 4: the Live Agent page (pages/5_Live_Agent.py) and its backend (investigation_v1_runs), end to end.

Through the real multipage app (AppTest, no browser). No network (sockets blocked) and no spending: the OpenRouter
Messages endpoint is the in-process emulator (the real Anthropic SDK over httpx2.MockTransport), metadata GETs are
injected and keys are fake strings. New investigations use SYNTHETIC fixtures only (labelled synthetic everywhere).
The genuine saved outputs (live_run1, run3) are only read, when present, and checked to be unchanged.
"""

import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import live_tracker.investigation_live as live
import live_tracker.investigation_live_v1 as lv1
import live_tracker.investigation_v1_runs as runs
from test_investigation_v1_live import CAPPED, FAKE_OR, HACKATHON, metadata
from test_investigation_v1_validation import inputs, recorded_run  # noqa: F401 (fixture)

httpx2 = pytest.importorskip("httpx2")
ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/5_Live_Agent.py"
LR = runs.script("investigation_v1_live")                  # the page's own instance of the reviewed CLI
LIVE_RUN1 = ROOT / "outputs" / "investigation_v1_live" / "live_run1"
RUN3 = ROOT / "outputs" / "investigation_v1" / "phase3b1_run3"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def blocked(*a, **k):
        raise AssertionError("network access attempted")
    real_connect = socket.socket.connect

    def guarded_connect(self, address, *a, **k):
        # Windows has no native socketpair: socket.socketpair() (asyncio's event loop uses it) connects a loopback
        # pair inside the socket module. Only that connection is allowed; any other, loopback included, is refused.
        caller = sys._getframe(1)
        if caller.f_globals.get("__name__") == "socket" and \
                caller.f_code.co_name in ("socketpair", "_fallback_socketpair") and \
                isinstance(address, tuple) and address and address[0] in ("127.0.0.1", "::1"):
            return real_connect(self, address, *a, **k)
        blocked()
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(live, "_get_json", blocked)
    monkeypatch.setattr(lv1, "get_json_no_redirects", blocked)
    monkeypatch.setattr(runs, "HTTP_CLIENT_FACTORY", None)
    for v in ("OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS",
              "ANTHROPIC_LOG"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(runs.LIVE_DIR_ENV, str(tmp_path / "live"))
    monkeypatch.setenv(runs.OFFLINE_DIR_ENV, str(tmp_path / "offline"))


@pytest.fixture
def synthetic(inputs, monkeypatch):
    """The page wired to SYNTHETIC inputs, a synthetic recorded reference run and an emulated provider."""
    ref_runs = recorded_run(inputs, "phase2", inputs["tmp"])
    monkeypatch.setenv("LIVE_TRACKER_VALIDATION_RUNS", str(ref_runs))
    monkeypatch.setattr(runs, "REFERENCE_RUN", "phase2")
    monkeypatch.setattr(live, "MANIFEST", inputs["manifest"])
    vi = runs.ValidationInputs(label="SYNTHETIC fixture inputs (tests only)", archive=str(inputs["zip"]),
                               archive_sha256=inputs["sha"], cache_dir=str(inputs["cache"]),
                               manifest=str(inputs["manifest"]), allow_synthetic=True)
    monkeypatch.setattr(runs, "default_inputs", lambda: vi)
    return {**inputs, "vi": vi}


def provider(monkeypatch, key=None, cost=None, faults=None):
    """The fake key in the environment, injected metadata GETs and the emulated Messages endpoint."""
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR)
    get = metadata(key)
    monkeypatch.setattr(lv1, "get_json_no_redirects", get)
    emu = LR.EmulatedOpenRouterV1(faults, cost)
    monkeypatch.setattr(runs, "HTTP_CLIENT_FACTORY", lambda: httpx2.Client(transport=httpx2.MockTransport(emu)))
    return get, emu


def test_the_network_guard_allows_only_the_socketpair_loopback():
    with pytest.raises(AssertionError, match="network access attempted"):
        socket.create_connection(("openrouter.ai", 443))
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
        except OSError:                                         # (no IPv6 here)
            continue
        with s, pytest.raises(AssertionError, match="network access attempted"):
            s.connect((host, 9))                                # a direct loopback connection: refused too
    pair = getattr(socket, "_fallback_socketpair", None) or (socket.socketpair if os.name == "nt" else None)
    if pair is None:
        pytest.skip("this Python has no pure-Python socketpair (the Windows path) to exercise")
    a, b = pair()                                               # what asyncio does on Windows: allowed
    with a, b:
        a.sendall(b"x")
        assert b.recv(1) == b"x"


def open_page(**state):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    for k, v in state.items():
        at.session_state[k] = v
    at.switch_page(PAGE).run()
    assert not at.exception, at.exception
    return at


def run(at):
    at.run()
    assert not at.exception, at.exception
    return at


def page_text(at):
    parts = []
    for kind in ("markdown", "caption", "success", "info", "warning", "error", "metric", "text"):
        for e in getattr(at, kind):
            parts.append(str(getattr(e, "value", "")) + str(getattr(e, "label", "")))
    for df in at.dataframe:
        parts.append(df.value.to_json())
    return " ".join(parts)


def table(at, *columns, last=False):
    """The first (or last) table on the page with exactly these columns."""
    found = [df.value for df in at.dataframe if list(df.value.columns) == list(columns)]
    return found[-1 if last else 0]


def button(at, label):
    return next(b for b in at.button if b.label == label)


def _click(at, label):
    button(at, label).click()
    return at


def preflight(at):
    return run(_click(at, "Run preflight"))


def authorize(at, gen=0, phrase=lv1.PAID_RUN_AUTHORIZATION):
    at.checkbox(key=f"live_agent_confirm_{gen}").check()
    at.text_input(key=f"live_agent_phrase_{gen}").input(phrase)
    return run(at)


def saved_dirs(tmp):
    root = tmp / "live"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


# ---- the end-to-end acceptance path ----

def test_acceptance_question_preflight_paid_run_evidence_report_verification_display(synthetic, monkeypatch):
    get, emu = provider(monkeypatch)
    at = open_page()
    at.text_area(key="live_agent_question").input("Does the distance between the mice change before predicted "
                                                  "attack bouts?")
    at = run(at)
    assert button(at, "Start paid investigation").disabled                     # no preflight yet
    at = preflight(at)
    assert any("Ready" in s.value for s in at.success) and emu.requests == []   # preflight: metadata only
    assert [u for u, _ in get.calls][-1].endswith("/endpoints")
    at = authorize(at)
    assert not button(at, "Start paid investigation").disabled
    at = run(_click(at, "Start paid investigation"))
    (name,) = saved_dirs(synthetic["tmp"])
    assert len(emu.requests) == 7                                              # model tool calls + finish
    s = json.loads((synthetic["tmp"] / "live" / name / "run.json").read_text())
    assert s["status"] == "verified" and s["mode"] == LR.LIVE_MODE and s["publication"] == "not_authorized"
    text = page_text(at)
    assert f"Investigation {name} finished" in text
    assert "Verified autonomous investigation" in text and "insufficient_evidence" in text
    assert "Verdict: equivalent" in text and FAKE_OR not in text
    assert LR.verify_saved(synthetic["tmp"] / "live" / name) == []
    tools = table(at, *runs.trace_rows(json.loads((synthetic["tmp"] / "live" / name / "investigation_report.json")
                                           .read_text()))[0])["tool"].tolist()  # the tool sequence tab
    assert "query_events" in tools and "compare_windows" in tools and "finish_investigation" in tools


# ---- the paid path: authorization, duplicate prevention, budgets, costs ----

def test_best_effort_selection_alone_never_enables_a_paid_run(synthetic, monkeypatch):
    provider(monkeypatch, key=HACKATHON)
    at = open_page()
    at.radio(key="live_agent_key_mode").set_value("best-effort")
    at = run(at)
    assert any("NOT a spending ceiling" in w.value for w in at.warning)
    at = preflight(at)
    assert any("Ready" in s.value for s in at.success)
    assert button(at, "Start paid investigation").disabled                     # no confirmation, no phrase
    at = authorize(at, phrase="yes")
    assert button(at, "Start paid investigation").disabled                     # wrong authorization text
    at = authorize(at)
    assert not button(at, "Start paid investigation").disabled


def test_a_changed_configuration_needs_a_new_preflight(synthetic, monkeypatch):
    provider(monkeypatch)
    at = authorize(preflight(open_page()))
    assert not button(at, "Start paid investigation").disabled
    at.number_input(key="live_agent_turns").set_value(5)
    at = run(at)
    assert button(at, "Start paid investigation").disabled
    assert any("configuration changed" in i.value for i in at.info)


def test_one_preflight_allows_one_paid_run_only(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    at = run(_click(authorize(preflight(open_page())), "Start paid investigation"))
    assert len(saved_dirs(synthetic["tmp"])) == 1 and len(emu.requests) == 7
    at = authorize(at, gen=1)                                                  # re-confirmed, same preflight
    assert button(at, "Start paid investigation").disabled
    assert len(saved_dirs(synthetic["tmp"])) == 1


def test_a_held_lock_blocks_every_new_run(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    root = synthetic["tmp"] / "live"
    with runs.RunLock(root):
        at = authorize(preflight(open_page()))
        assert button(at, "Start paid investigation").disabled and button(at, "Run free rehearsal").disabled
        assert any("Another investigation is running" in w.value for w in at.warning)
        with pytest.raises(runs.RunBusy):
            runs.execute(runs.RunConfig(), synthetic["vi"], confirm_budget_usd=0.5,
                         authorization=lv1.PAID_RUN_AUTHORIZATION, root=root)
    assert emu.requests == [] and saved_dirs(synthetic["tmp"]) == []
    (root / runs.LOCK_NAME).write_text(json.dumps({"pid": 999_999_999}))       # left on disk, held by no one
    with runs.RunLock(root):
        pass
    assert (root / runs.LOCK_NAME).exists()                                    # reused, never unlinked


def test_execute_refuses_without_the_separate_authorization(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    for kw in ({"confirm_budget_usd": 0.5, "authorization": None},
               {"confirm_budget_usd": None, "authorization": lv1.PAID_RUN_AUTHORIZATION},
               {"confirm_budget_usd": 0.4, "authorization": lv1.PAID_RUN_AUTHORIZATION}):
        with pytest.raises(lv1.LiveRefused):
            runs.execute(runs.RunConfig(), synthetic["vi"], **kw)
    assert emu.requests == [] and saved_dirs(synthetic["tmp"]) == []


def test_budget_stopping_is_displayed(synthetic, monkeypatch):
    _, emu = provider(monkeypatch, cost=0.25)
    at = run(_click(authorize(preflight(open_page())), "Start paid investigation"))
    assert len(emu.requests) == 2
    text = page_text(at)
    assert "Verified but incomplete" in text and "model_unavailable" in text and "$0.50000" in text


def test_a_missing_model_cost_stops_and_is_displayed_as_unknown(synthetic, monkeypatch):
    _, emu = provider(monkeypatch, faults={2: "no_cost"})
    at = run(_click(authorize(preflight(open_page())), "Start paid investigation"))
    assert len(emu.requests) == 2
    text = page_text(at)
    assert "Verified but incomplete" in text and "+unknown" in text and "provider_cost_unknown" in text


def test_credentials_never_reach_the_page_or_the_files(synthetic, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "sk-or-v1-FAKE-management-key-5555555555")
    provider(monkeypatch)
    at = run(_click(authorize(preflight(open_page())), "Start paid investigation"))
    text = page_text(at)
    assert FAKE_OR not in text and "FAKE-management" not in text
    for f in (synthetic["tmp"] / "live").rglob("*"):
        if f.is_file():
            body = f.read_text(encoding="utf-8")
            assert FAKE_OR not in body and "FAKE-management" not in body
    assert any("OPENROUTER_API_KEY is set" in c.value for c in at.caption)


def test_a_preflight_that_fails_keeps_the_paid_run_disabled(synthetic, monkeypatch):
    provider(monkeypatch, key={**CAPPED, "limit": 1.0, "limit_remaining": 1.0})
    at = authorize(preflight(open_page()))
    assert any("Not ready" in e.value and "above $0.50" in e.value for e in at.error)
    assert button(at, "Start paid investigation").disabled


# ---- autonomous investigation vs reference compatibility ----

def test_a_rehearsal_that_does_not_reproduce_the_reference_is_still_a_verified_investigation(synthetic, monkeypatch):
    iv = runs.script("investigation_v1_validation")
    real = iv.OfflineInvestigatorV1.decide

    def own_threshold(self, done):                         # the "model" leaves the threshold to the default
        name, args = real(self, done)
        if name == "query_events":
            args = {k: v for k, v in args.items() if k != "threshold"}
        return name, args
    monkeypatch.setattr(iv.OfflineInvestigatorV1, "decide", own_threshold)
    at = run(_click(open_page(), "Run free rehearsal"))
    (name,) = saved_dirs(synthetic["tmp"])
    assert json.loads((synthetic["tmp"] / "live" / name / "run.json").read_text())["status"] == "refused"
    text = page_text(at)
    assert "Verified autonomous investigation" in text
    assert "Verdict: not equivalent" in text and "threshold" in text and "not sent" in text
    assert "equivalence, not the investigation's own validity" in text
    assert "Scientific report: verified." in text
    assert "actual API spending $0" in text                                    # a rehearsal spends nothing


# ---- reopening and invalid reports ----

def rehearse(synthetic):
    name, _ = runs.execute(runs.RunConfig(validation_run=runs.REFERENCE_RUN), synthetic["vi"], confirm_budget_usd=None,
                           authorization=None, paid=False)
    return synthetic["tmp"] / "live" / name


def test_reopening_a_saved_investigation_makes_no_request(synthetic, monkeypatch):
    path = rehearse(synthetic)
    calls = []
    monkeypatch.setattr(lv1, "get_json_no_redirects", lambda *a, **k: calls.append(a) or {})
    monkeypatch.setattr(runs, "HTTP_CLIENT_FACTORY", lambda: calls.append("client"))
    at = open_page()
    assert "Verified autonomous investigation" in page_text(at) and calls == []
    assert runs.load_saved(path).verified


@pytest.mark.parametrize("damage", ["missing_report", "tampered_report", "tampered_cost", "missing_evidence"])
def test_an_invalid_saved_investigation_shows_only_why(synthetic, damage):
    path = rehearse(synthetic)
    report = path / "investigation_report.json"
    if damage == "missing_report":
        report.unlink()
    elif damage == "tampered_report":
        doc = json.loads(report.read_text())
        doc["observations"][0]["value"] = "fabricated"
        report.write_text(json.dumps(doc))
    elif damage == "tampered_cost":
        s = json.loads((path / "run.json").read_text())
        s["cost"]["actual_spend_usd"] = 0.0
        (path / "run.json").write_text(json.dumps(s))
    else:
        doc = json.loads(report.read_text())
        doc["evidence"] = doc["evidence"][:1]
        report.write_text(json.dumps(doc))
    saved = runs.load_saved(path)
    assert not saved.verified and runs.autonomous_status(saved)[0] == "unverified"
    at = open_page()
    assert any("does NOT verify" in e.value for e in at.error)
    assert len(at.dataframe) == 0 and not any("autonomous investigation" in s.value for s in at.success)


# ---- the genuine saved outputs (read only) ----

def digests(path):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(path.iterdir()) if f.is_file()}


@pytest.mark.skipif(not (LIVE_RUN1 / "run.json").is_file() or not (RUN3 / "run.json").is_file(),
                    reason="the genuine saved investigations are not present")
def test_the_genuine_live_run1_and_run3_are_shown_without_modification(monkeypatch):
    monkeypatch.delenv(runs.LIVE_DIR_ENV)
    monkeypatch.delenv(runs.OFFLINE_DIR_ENV)
    before = {p: digests(p) for p in (LIVE_RUN1, RUN3)}
    live1, run3 = runs.load_saved(LIVE_RUN1), runs.load_saved(RUN3)
    assert live1.verified and live1.kind == "live" and live1.doc["finding"] == "effect_described"
    assert run3.verified and run3.kind == "offline"
    ref = runs.reference_check(live1)
    assert not ref["equivalent"] and ref["differences"] == ["threshold"]
    row = next(r for r in ref["rows"] if r["parameter"] == "threshold")
    assert (row["investigation (sent)"], row["investigation (applied)"], row["run5 (applied)"]) == ("not sent",
                                                                                                    "0.5", "0.6")
    assert runs.reference_check(run3)["equivalent"]
    at = open_page(live_agent_open=str(LIVE_RUN1))
    text = page_text(at)
    assert "Verified autonomous investigation" in text and "effect_described" in text
    assert "Verdict: not equivalent" in text and "$0.03224" in text and "best_effort" in text
    assert {p: digests(p) for p in (LIVE_RUN1, RUN3)} == before                 # nothing was written


# ---- the selected saved investigation is always the one displayed ----

def showing(at):
    """The run the Saved tab says it shows, and the run its selector holds."""
    shown = next(m.value for m in at.markdown if m.value.startswith("**Showing `"))
    return shown.split("`")[1], Path(at.selectbox(key="live_agent_saved").value).name


def displayed_question(at):
    return next(m.value for m in at.markdown if m.value.startswith("> "))[2:]


def rehearse_asking(synthetic, question):
    name, _ = runs.execute(runs.RunConfig(question=question, validation_run=runs.REFERENCE_RUN), synthetic["vi"],
                           confirm_budget_usd=None, authorization=None, paid=False)
    return name


def test_switching_saved_runs_shows_exactly_the_selected_run(synthetic):
    a = rehearse_asking(synthetic, "Question A: does the distance change before predicted attacks?")
    b = rehearse_asking(synthetic, "Question B: does the distance change before predicted attacks?")
    at = open_page()
    for name, question in ((a, "Question A"), (b, "Question B"), (a, "Question A")):
        at.selectbox(key="live_agent_saved").set_value(str(synthetic["tmp"] / "live" / name))
        at = run(at)
        assert showing(at) == (name, name) and displayed_question(at).startswith(question)
        doc = json.loads((synthetic["tmp"] / "live" / name / "investigation_report.json").read_text())
        obs = table(at, "evidence_id", "field", "value")                       # the Finding tab's observations
        assert obs["evidence_id"].tolist() == [o["evidence_id"] for o in doc["observations"]]
        prov = table(at, "field", "value", last=True)                          # the provenance tab
        prov = dict(zip(prov["field"], prov["value"]))
        assert prov["directory"] == name and prov["report fingerprint"] == doc["fingerprint"]


def test_a_completed_rehearsal_is_selected_in_the_browser_too(synthetic):
    old = rehearse_asking(synthetic, "Question A: an earlier rehearsal")
    at = open_page()
    at.selectbox(key="live_agent_saved").set_value(str(synthetic["tmp"] / "live" / old))
    at = run(at)
    assert showing(at) == (old, old)
    at.text_area(key="live_agent_question").input("Question C: the new rehearsal")
    at = run(_click(run(at), "Run free rehearsal"))
    (new,) = set(saved_dirs(synthetic["tmp"])) - {old}
    sb = at.selectbox(key="live_agent_saved")
    assert showing(at) == (new, new) and displayed_question(at).startswith("Question C")
    assert sb.proto.set_value                     # the new selection is pushed to the browser, not only the server
    label = sb.format_func(sb.value)                 # UI-4: a verified run is labelled by what it is
    assert "free rehearsal" in label and "“Question C: the new rehearsal”" in label
    at.selectbox(key="live_agent_saved").set_value(str(synthetic["tmp"] / "live" / old))
    at = run(at)                                  # and the researcher can go back
    assert showing(at) == (old, old) and displayed_question(at).startswith("Question A")
    at = run(at)                                  # a plain rerun keeps the researcher's choice
    assert showing(at) == (old, old)


def test_reopening_a_saved_investigation_by_path_and_after_it_is_removed(synthetic):
    a = rehearse_asking(synthetic, "Question A: kept")
    b = rehearse_asking(synthetic, "Question B: removed later")
    at = open_page(live_agent_open=str(synthetic["tmp"] / "live" / b))
    assert showing(at) == (b, b)
    for f in (synthetic["tmp"] / "live" / b).iterdir():
        f.unlink()
    (synthetic["tmp"] / "live" / b).rmdir()
    at = run(at)                                  # the selected run vanished: another run, shown consistently
    assert showing(at) == (a, a)


# ---- emulated costs are simulated, not spending ----

def test_a_rehearsal_shows_simulated_cost_and_zero_actual_spending(synthetic):
    name = rehearse_asking(synthetic, "Question R: a rehearsal")
    simulated = json.loads((synthetic["tmp"] / "live" / name / "run.json").read_text())["cost"]["actual_spend_usd"]
    assert simulated > 0                                                       # the emulated ledger is unchanged
    at = open_page()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Actual API spending"] == "$0.00"
    assert metrics["Simulated cost (emulated, not spent)"] == f"${simulated:.5f}"
    assert "Actual spend (provider-reported)" not in metrics
    assert any(f"Simulated cost ${simulated:.5f}; actual API spending $0" in i.value for i in at.info)
    ledger = next(df.value for df in at.dataframe if "cost status" in df.value.columns)
    assert "simulated cost (USD, not spent)" in ledger.columns and "cost (USD)" not in ledger.columns


def test_a_paid_run_still_shows_provider_reported_spending(synthetic, monkeypatch):
    provider(monkeypatch)
    at = run(_click(authorize(preflight(open_page())), "Start paid investigation"))
    (name,) = saved_dirs(synthetic["tmp"])
    spend = json.loads((synthetic["tmp"] / "live" / name / "run.json").read_text())["cost"]["actual_spend_usd"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Actual spend (provider-reported)"] == metrics["Actual API spending"] == f"${spend:.5f}"
    assert "Simulated cost (emulated, not spent)" not in metrics
    ledger = next(df.value for df in at.dataframe if "cost status" in df.value.columns)
    assert "cost (USD)" in ledger.columns
