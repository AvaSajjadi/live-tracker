"""UI-3: live progress of a Live Agent run (runs.execute(progress=), the CLI's progress events, the agent's
presentation-only observer and investigation_v1_ui.LiveProgress). New file.

SYNTHETIC fixtures and the in-process emulated OpenRouter endpoint only: no network, no spending, no real data.
"""

import dataclasses
import json
import threading
import time
from pathlib import Path

import pytest
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException, StopException

import live_tracker.investigation_agent as ia
import live_tracker.investigation_agent_v1 as av1
import live_tracker.investigation_live_v1 as lv1
import live_tracker.investigation_tools_v1 as v1
import live_tracker.investigation_v1_runs as runs
import live_tracker.investigation_v1_ui as ui
from test_investigation_v1_validation import inputs, iv, recorded_run  # noqa: F401 (fixture)
from test_run_lock import child
from test_live_agent_page import (_click, offline, open_page, page_text, provider, run,  # noqa: F401
                                  saved_dirs, synthetic)

LR = runs.script("investigation_v1_live")


class Recorder:
    """A progress sink recording each event with what was on disk when it arrived."""

    def __init__(self, root):
        self.root, self.events = root, []

    def __call__(self, event):
        dirs = [p for p in self.root.iterdir() if p.is_dir()] if self.root.is_dir() else []
        on_disk = {f for d in dirs for f in ("investigation_report.json", "run.json") if (d / f).is_file()}
        self.events.append((json.loads(json.dumps(event)), on_disk, time.monotonic()))

    def of(self, kind):
        return [e for e, _, _ in self.events if e["type"] == kind]


def execute(synthetic, sink=None, paid=False, **kw):
    config = runs.RunConfig(validation_run=runs.REFERENCE_RUN, **kw)
    auth = dict(confirm_budget_usd=0.5, authorization=lv1.PAID_RUN_AUTHORIZATION) if paid else \
        dict(confirm_budget_usd=None, authorization=None)
    return runs.execute(config, synthetic["vi"], paid=paid, progress=sink, **auth)


def report(synthetic, name):
    return json.loads((synthetic["tmp"] / "live" / name / "investigation_report.json").read_text())


def stages(rec):
    return [(e["stage"], e["state"]) for e in rec.of("stage")]


# ---- events: live, ordered, the recorded ones ----

def test_progress_arrives_while_the_run_is_in_progress_and_in_stage_order(synthetic):
    rec = Recorder(synthetic["tmp"] / "live")
    name, summary = execute(synthetic, rec)
    assert summary["status"] == "verified" and runs.load_saved(synthetic["tmp"] / "live" / name).verified
    order = ["lock", "inputs", "validation_run", "budget", "investigate", "verify", "cost_ledger", "compatibility"]
    want = [("lock", "completed")] + [(s, x) for s in order[1:] for x in ("started", "completed")]
    assert stages(rec) == want                                       # every stage, in the recorded order
    assert [s["step"] for s in summary["steps"]] == order[1:]        # the same stages as the run recorded
    trace = [(e, disk) for e, disk, _ in rec.events if e["type"] == "trace"]
    assert trace and all(not disk for _, disk in trace)              # every turn and tool: before any report file
    events = [e for e, _, _ in rec.events]
    start = events.index({"type": "stage", "stage": "investigate", "state": "started"})
    end = events.index({"type": "stage", "stage": "investigate", "state": "completed"})
    inside = [i for i, e in enumerate(events) if e["type"] in ("trace", "model_request", "spending")]
    assert inside and start < min(inside) and max(inside) < end      # every turn and tool: inside the stage
    assert end < events.index({"type": "saved", "file": "investigation_report.json"})
    assert rec.events[-1][0] == {"type": "finished", "status": "verified", "stopping_reason": "finished"}
    assert rec.events[-1][1] == {"investigation_report.json", "run.json"}


def test_the_trace_events_are_the_saved_trace_projected(synthetic):
    rec = Recorder(synthetic["tmp"] / "live")
    name, _ = execute(synthetic, rec)
    doc = report(synthetic, name)
    live = rec.of("trace")
    assert [(e["kind"], e["step"]) for e in live] == [(t["kind"], t["step"]) for t in doc["trace"]]
    for e, t in zip(live, doc["trace"]):
        for f in ("tool", "refused", "stopping_reason", "input_tokens", "output_tokens"):
            assert e.get(f) == t.get(f) or f not in e
        if t["kind"] in ("tool_call", "finish"):                    # accepted / refused only: no scientific status
            assert e["outcome"] == ("refused" if t["outcome"] == "refused" else "accepted")
    turns = [t for t in doc["trace"] if t["kind"] == "model_turn"]
    assert len(rec.of("model_request")) == len(turns) == doc["usage"]["model_turns"]
    for e in live:                                                   # no text, arguments, messages or findings
        assert not {"text", "arguments", "message", "evidence_id", "observations", "finding"} & set(e)
    dumped = json.dumps([e for e, _, _ in rec.events])
    statuses = {e["status"] for e in doc["evidence"]} | {doc["finding"]}
    assert not any(s in dumped for s in statuses - {"ok"}) and doc["finding"] == "insufficient_evidence"
    assert "finding" not in json.dumps([sorted(e) for e, _, _ in rec.events])


def test_refused_and_executed_calls_are_distinct_and_shown_apart(synthetic, monkeypatch):
    _, emu = provider(monkeypatch, faults={1: "unknown_tool", 2: "extra_argument"})
    rec = Recorder(synthetic["tmp"] / "live")
    name, _ = execute(synthetic, rec, paid=True, max_model_turns=9)
    doc = report(synthetic, name)
    calls = [e for e in rec.of("trace") if e["kind"] in ("tool_call", "finish")]
    refused = [e for e in calls if e["outcome"] == "refused"]
    assert [(e["tool"], e["refused"]) for e in refused] == [                # unknown names: never shown as given
        (LR._tool(t["tool"]), t["refused"]) for t in doc["trace"] if t.get("outcome") == "refused"]
    assert [e["tool"] for e in refused] == [LR.UNKNOWN_TOOL, "query_events"]
    view = replay(rec, paid=True, max_turns=9)
    lines = [x for x in view.lines if x.startswith((":red-badge[refused]", ":green-badge[executed]"))]
    assert lines[0].startswith(":red-badge[refused] `unknown tool`: unknown_tool")
    assert lines[1].startswith(":red-badge[refused] `query_events`: invalid_arguments")
    assert lines[2].startswith(":green-badge[executed] `query_events`: accepted")
    assert len(view.refused) == 2 and view.turns == doc["usage"]["model_turns"] == len(emu.requests)


def test_model_unavailability_and_an_early_stop_are_reported_as_recorded(synthetic, monkeypatch):
    _, emu = provider(monkeypatch, faults={2: "http_500"})
    rec = Recorder(synthetic["tmp"] / "live")
    name, summary = execute(synthetic, rec, paid=True)
    doc = report(synthetic, name)
    assert not doc["finished"] and summary["status"] == "incomplete"
    kinds = [e["kind"] for e in rec.of("trace")]
    assert kinds[-2:] == ["model_error", "stop"] and rec.of("trace")[-1]["stopping_reason"] == doc["stopping_reason"]
    assert rec.events[-1][0]["stopping_reason"] == doc["stopping_reason"] == "model_unavailable"
    view = replay(rec, paid=True)
    assert any(x.startswith(":red-badge[provider] Model request failed") for x in view.lines)
    assert view.stop == "model_unavailable" and not any("finding submitted" in x for x in view.lines)
    state, verdict = ui.verdict_of(runs.load_saved(synthetic["tmp"] / "live" / name))
    assert (state, verdict) == ("complete", "verified, incomplete (no finding)")


# ---- the observer changes nothing ----

class FixedClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.25
        return self.t


def test_a_failing_observer_changes_no_decision_and_no_report(synthetic):
    data = iv.load_data(*_load_args(synthetic))[0]

    def model():
        return iv.OfflineInvestigatorV1()

    def report_with(observer):
        r = av1.investigate_v1("Observer question", data, model(), budget=ia.Budget(), investigation_id="obs",
                               allow_synthetic=True, clock=FixedClock(), observer=observer)
        return r.to_json()
    calls = []

    def boom(event):
        calls.append(event)
        raise RuntimeError("a broken display")
    plain, broken, seen = report_with(None), report_with(boom), []
    assert broken == plain and len(calls) == 1                       # dropped after its first failure
    assert report_with(seen.append) == plain and len(seen) > 5        # a working observer: the same report too
    av1.verify_report_v1(json.loads(json.dumps(broken)))


def _load_args(synthetic):
    class A:
        archive, archive_sha256 = Path(synthetic["zip"]), synthetic["sha"]
        member, allow_synthetic = "scores_validation_model_b.csv", True
    return A, Path(synthetic["manifest"]), Path(synthetic["cache"])


def test_a_failing_sink_leaves_the_run_and_its_verification_intact(synthetic):
    def boom(event):
        raise RuntimeError("display failed")
    a, sa = execute(synthetic, boom)
    b, sb = execute(synthetic, None)
    da, db = report(synthetic, a), report(synthetic, b)
    assert runs.load_saved(synthetic["tmp"] / "live" / a).verified and sa["status"] == sb["status"] == "verified"
    assert (da["finding"], da["observations"], da["evidence"]) == (db["finding"], db["observations"], db["evidence"])
    assert [(t["kind"], t.get("tool"), t.get("outcome")) for t in da["trace"]] == \
        [(t["kind"], t.get("tool"), t.get("outcome")) for t in db["trace"]]


def test_a_rerun_or_stop_request_during_a_run_is_held_until_it_is_saved(synthetic):
    for control in (RerunException(None), StopException()):
        view = ui.LiveProgress(False, 8)
        raised = []

        def draw():
            if not raised:
                raised.append(1)
                raise control                                        # a click / Stop / closed tab, mid-run
        view.draw = draw
        name, summary = ui.run_with_live_view(lambda progress: execute(synthetic, progress), view)
        assert summary["status"] == "verified" and runs.load_saved(synthetic["tmp"] / "live" / name).verified
        assert view.deferred is control and view.turns > 0           # the run went on, events still counted
        with pytest.raises(type(control)):
            view.release()                                           # then Streamlit gets its request


# ---- authorization, the lock, spending ----

def test_paid_authorization_is_unchanged_with_a_progress_sink(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    rec = Recorder(synthetic["tmp"] / "live")
    config = runs.RunConfig(validation_run=runs.REFERENCE_RUN)
    for kw in ({"confirm_budget_usd": 0.5, "authorization": None},
               {"confirm_budget_usd": None, "authorization": lv1.PAID_RUN_AUTHORIZATION}):
        with pytest.raises(lv1.LiveRefused):
            runs.execute(config, synthetic["vi"], progress=rec, **kw)
    assert emu.requests == [] and saved_dirs(synthetic["tmp"]) == []
    assert all(e["type"] == "stage" and e["stage"] == "lock" for e, _, _ in rec.events)   # nothing else happened


def test_a_held_lock_refuses_before_any_progress(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    root = synthetic["tmp"] / "live"
    holder = child(root, "hold")
    try:
        assert holder.stdout.readline().strip() == "entered"
        rec = Recorder(root)
        with pytest.raises(runs.RunBusy):
            execute(synthetic, rec, paid=True)
        assert rec.events == [] and emu.requests == []               # refused before the lock: no progress at all
    finally:
        holder.kill()
        holder.wait(10)


def test_spending_is_the_ledgers_provider_reported_cost_never_an_estimate(synthetic, monkeypatch):
    provider(monkeypatch, faults={3: "no_cost"})
    rec = Recorder(synthetic["tmp"] / "live")
    name, summary = execute(synthetic, rec, paid=True)
    ledger = summary["cost"]["ledger"]["requests"]
    spent = rec.of("spending")
    assert spent and all(e["basis"] == "provider-reported" for e in spent)
    running, unknown = 0.0, 0
    for e, entry in zip(spent, ledger):                              # one per reply, cumulative, as recorded
        running += entry["cost_usd"] if entry.get("cost_status") == "reported" else 0.0
        unknown += entry.get("cost_status") != "reported"
        assert (round(e["known_cost_usd"], 8), e["unknown_cost_attempts"]) == (round(running, 8), unknown)
    assert spent[-1]["known_cost_usd"] == summary["cost"]["actual_spend_usd"] and unknown == 1
    view = replay(rec, paid=True)
    assert view.spent() == ("Provider-reported spending", f"${summary['cost']['actual_spend_usd']:.5f} + 1 unreported")


def test_a_rehearsal_reports_simulated_cost_only(synthetic):
    rec = Recorder(synthetic["tmp"] / "live")
    execute(synthetic, rec)
    assert {e["basis"] for e in rec.of("spending")} == {"simulated"}
    assert replay(rec).spent()[0] == "Simulated cost (not spent)"


# ---- the page ----

def replay(rec, paid=False, max_turns=8):
    """The events through a LiveProgress, outside a running app (its drawing is a no-op there)."""
    view = ui.LiveProgress(paid, max_turns)
    view.draw = lambda: None
    for e, _, _ in rec.events:
        view(e)
    return view


def test_the_page_runs_a_rehearsal_with_live_progress_then_points_to_the_verified_report(synthetic):
    at = open_page()
    at = run(_click(at, "Run free rehearsal"))
    (name,) = saved_dirs(synthetic["tmp"])
    rec = at.session_state[ui.PROGRESS_KEY]
    assert rec["run"] == name and rec["verdict"] == "verified" and rec["status"] == "verified"
    text = page_text(at)
    assert f"Investigation {name} finished with recorded status `verified`" in text
    assert "re-verified: verified" in text and "Open the Saved investigations tab" in text
    lines = "\n".join(rec["lines"])
    for label in ("Loading the verified validation inputs", "Investigation (model turns and Contract v1 tools)",
                  "Checking the saved report", "Re-verifying the saved report from disk"):
        assert label in lines
    assert ":green-badge[executed] `query_events`" in lines and ":green-badge[verified]" in lines
    doc = report(synthetic, name)
    assert doc["finding"] not in lines                               # the finding: only from the saved report
    assert any("Live progress of the last run started in this tab" in e.label for e in at.expander)
    assert any("not a monitor of other tabs" in c.value for c in at.caption)
    at.selectbox(key=ui.SAVED_KEY).set_value(str(synthetic["tmp"] / "live" / name))
    at = run(at)
    assert any(m.value == "#### Scientific evidence" for m in at.markdown)    # UI-2 on the verified report
    assert any(m.value == "#### Recorded investigation activity" for m in at.markdown)   # UI-1


def test_an_unverifiable_saved_report_is_reported_as_such(synthetic, monkeypatch):
    real = runs.load_saved

    def tampered(path):
        saved = real(path)
        saved.problems = ["the report's fingerprint does not match its content"]
        return saved
    monkeypatch.setattr(runs, "load_saved", tampered)
    at = run(_click(open_page(), "Run free rehearsal"))
    rec = at.session_state[ui.PROGRESS_KEY]
    assert rec["verdict"].startswith("the saved investigation does NOT verify")
    assert any("does NOT verify" in e.value for e in at.error)
    assert ":red-badge[not verified]" in "\n".join(rec["lines"])


# ---- Codex audit 1: rendering never takes time from the investigation ----

class SlowEmulator(LR.EmulatedOpenRouterV1):
    """The emulated endpoint, answering after a measurable delay (the offline investigator decides)."""

    def __init__(self, delay_s):
        super().__init__()
        self.delay_s = delay_s

    def __call__(self, request):
        time.sleep(self.delay_s)
        return super().__call__(request)


def short_deadline(monkeypatch, timeout_s):
    real = lv1.budget_v1

    def budget(plan, prices):
        b, estimate = real(plan, prices)
        return dataclasses.replace(b, timeout_s=timeout_s), estimate
    monkeypatch.setattr(lv1, "budget_v1", budget)


def sequence(doc):
    return [(t["kind"], t.get("tool"), t.get("outcome"), t.get("refused"), t.get("stopping_reason"))
            for t in doc["trace"]]


def test_slow_rendering_does_not_change_the_run_or_its_deadline(synthetic, monkeypatch):
    """Real clock, a 6 s investigation deadline, an emulated provider answering after 0.15 s, and a renderer taking
    1 s per draw. Synchronous delivery (the defect) times the run out; the worker/script-thread split does not."""
    short_deadline(monkeypatch, 6.0)
    emulators = []

    def factory():
        emulators.append(SlowEmulator(0.15))
        return emulators[-1]
    monkeypatch.setattr(runs, "REHEARSAL_EMULATOR_FACTORY", factory)
    base_name, base = execute(synthetic)                             # no rendering at all
    base_doc = report(synthetic, base_name)
    assert base_doc["finished"] and base_doc["stopping_reason"] == "finished"

    draws, before_end = [], []
    view = ui.LiveProgress(False, 8)

    def slow_draw():
        draws.append(time.monotonic())
        if view.turns and view.status is None:
            before_end.append(view.turns)                            # drawn while the investigation was running
        time.sleep(1.0)
    view.draw = slow_draw
    name, summary = ui.run_with_live_view(lambda progress: execute(synthetic, progress), view)
    doc = report(synthetic, name)
    assert sequence(doc) == sequence(base_doc) and doc["stopping_reason"] == "finished"
    assert len(emulators[1].requests) == len(emulators[0].requests) == doc["usage"]["model_turns"]
    assert [r["body"] for r in emulators[1].requests] == [r["body"] for r in emulators[0].requests]   # identical
    assert doc["usage"]["elapsed_s"] < 6.0 and len(draws) >= 2 and before_end      # live, and not in its time
    assert runs.load_saved(synthetic["tmp"] / "live" / name).verified and summary["status"] == "verified"

    control = []                                                     # the defect, reproduced: drawing inline
    inline = ui.LiveProgress(False, 8)
    inline.draw = lambda: (control.append(1), time.sleep(1.0))
    cname, _ = execute(synthetic, inline)
    cdoc = report(synthetic, cname)
    assert cdoc["stopping_reason"] == "timeout" and sequence(cdoc) != sequence(base_doc)


def test_the_progress_queue_is_bounded_never_blocks_and_counts_what_it_drops():
    q = ui.ProgressQueue(maxsize=3)
    started = time.monotonic()
    for i in range(10):
        q.put({"type": "saved", "file": str(i)})                    # never blocks, never raises
    assert time.monotonic() - started < 0.5 and q.dropped == 7
    assert [e["file"] for e in q.drain(0)] == ["0", "1", "2"] and q.drain(0) == []
    view = ui.LiveProgress(False, 8)
    view.draw = lambda: None
    view.show([], q.dropped)
    rec = view.record("r", "verified")
    assert rec["dropped"] == 7 and "7 progress events could not be shown" in rec["lines"][-1]


def test_a_worker_failure_reaches_the_script_thread_after_the_worker_ended(synthetic):
    view = ui.LiveProgress(False, 8)
    view.draw = lambda: None
    workers = []

    def work(progress):
        workers.append(threading.current_thread())
        progress({"type": "stage", "stage": "lock", "state": "completed"})
        raise lv1.LiveRefused("refused in the worker")
    with pytest.raises(lv1.LiveRefused, match="refused in the worker"):
        ui.run_with_live_view(work, view)
    assert workers[0] is not threading.current_thread() and not workers[0].is_alive()
    assert len(view.events) == 1


# ---- Codex audit 2: every exit of _run is settled ----

@pytest.fixture
def held_rerun(monkeypatch):
    """Session state as a plain dict, and a rerun request raised by the first draw of every LiveProgress."""
    state = {}
    monkeypatch.setattr(ui.st, "session_state", state)
    real = ui.LiveProgress.draw

    def draw(self):
        if not getattr(self, "_raised", False):
            self._raised = True
            raise RerunException(None)
        return real(self)
    monkeypatch.setattr(ui.LiveProgress, "draw", draw)
    return state


def rehearse_via_run(synthetic):
    ui._run(runs.RunConfig(validation_run=runs.REFERENCE_RUN), synthetic["vi"], False, confirm_budget_usd=None,
            authorization=None)


def test_run_success_settles_the_held_request_after_verification(synthetic, held_rerun):
    with pytest.raises(RerunException):
        rehearse_via_run(synthetic)
    (name,) = saved_dirs(synthetic["tmp"])
    assert held_rerun[ui.PROGRESS_KEY]["verdict"] == "verified" and held_rerun[ui.MESSAGE_KEY][0] == "success"


def test_run_refusal_settles_the_held_request_and_runs_nothing(synthetic, monkeypatch, held_rerun):
    _, emu = provider(monkeypatch)
    with pytest.raises(RerunException):
        ui._run(runs.RunConfig(validation_run=runs.REFERENCE_RUN), synthetic["vi"], True, confirm_budget_usd=0.5,
                authorization=None)
    assert held_rerun[ui.MESSAGE_KEY][1].startswith("Not started (LiveRefused)")
    assert emu.requests == [] and saved_dirs(synthetic["tmp"]) == [] and ui.PROGRESS_KEY not in held_rerun


def test_run_verification_failure_is_reported_never_as_verified(synthetic, monkeypatch, held_rerun):
    real = runs.load_saved

    def unverified(path):
        saved = real(path)
        saved.problems = ["the report's fingerprint does not match its content"]
        return saved
    monkeypatch.setattr(runs, "load_saved", unverified)
    with pytest.raises(RerunException):
        rehearse_via_run(synthetic)
    assert held_rerun[ui.MESSAGE_KEY][0] == "error" and "does NOT verify" in held_rerun[ui.MESSAGE_KEY][1]
    assert held_rerun[ui.PROGRESS_KEY]["verdict"].startswith("the saved investigation does NOT verify")


def test_run_an_exception_from_load_saved_is_settled_and_never_verified(synthetic, monkeypatch, held_rerun):
    def broken(path):
        raise RuntimeError("disk vanished")
    monkeypatch.setattr(runs, "load_saved", broken)
    with pytest.raises(RerunException):                               # the held request, settled at the end
        rehearse_via_run(synthetic)
    (name,) = saved_dirs(synthetic["tmp"])                           # the run happened exactly once
    verdict = held_rerun[ui.PROGRESS_KEY]["verdict"]
    assert verdict.startswith("the saved report could not be loaded or checked (RuntimeError)")
    assert held_rerun[ui.MESSAGE_KEY][0] == "error" and "re-verified: verified" not in held_rerun[ui.MESSAGE_KEY][1]
    assert ":red-badge[not verified]" in "\n".join(held_rerun[ui.PROGRESS_KEY]["lines"])


def test_run_an_unexpected_failure_stays_primary_over_a_held_request(synthetic, monkeypatch, held_rerun):
    def record(self, name, verdict):
        raise KeyError("unexpected")
    monkeypatch.setattr(ui.LiveProgress, "record", record)
    with pytest.raises(KeyError, match="unexpected"):                # not masked by the held RerunException
        rehearse_via_run(synthetic)
    assert len(saved_dirs(synthetic["tmp"])) == 1                    # and the run was not repeated


def test_run_busy_settles_without_a_run(synthetic, monkeypatch, held_rerun):
    holder = child(synthetic["tmp"] / "live", "hold")
    try:
        assert holder.stdout.readline().strip() == "entered"
        rehearse_via_run(synthetic)                                  # nothing drawn: no request was held
    finally:
        holder.kill()
        holder.wait(10)
    assert held_rerun[ui.MESSAGE_KEY][1].startswith("Not started: another investigation is running")
    assert saved_dirs(synthetic["tmp"]) == []


# ---- Codex audit 3: no model-supplied text in progress events ----

MARKER = "SECRET-7f3a91 <private chain of thought: the key is sk-or-v1-abc>"


def test_a_malicious_unknown_tool_name_never_reaches_progress(synthetic, monkeypatch):
    iv_ = runs.script("investigation_v1_validation")
    real = iv_.OfflineInvestigatorV1.decide
    calls = []

    def decide(self, done):
        calls.append(1)
        name, args = real(self, done)
        return (MARKER, args) if len(calls) == 1 else (name, args)
    monkeypatch.setattr(iv_.OfflineInvestigatorV1, "decide", decide)
    rec = Recorder(synthetic["tmp"] / "live")
    view = ui.LiveProgress(False, 8)
    view.draw = lambda: None
    name, _ = ui.run_with_live_view(lambda progress: execute(synthetic, lambda e: (rec(e), progress(e))), view)
    doc = report(synthetic, name)
    assert any(t.get("tool") == MARKER for t in doc["trace"])        # the trace itself is kept as recorded
    assert runs.load_saved(synthetic["tmp"] / "live" / name).verified
    for text in (json.dumps([e for e, _, _ in rec.events]), json.dumps(view.events), "\n".join(view.lines),
                 json.dumps(view.record(name, "verified"))):
        assert "SECRET-7f3a91" not in text and "sk-or-v1" not in text and "chain of thought" not in text
    refused = [e for e in rec.of("trace") if e["kind"] == "tool_call" and e["outcome"] == "refused"]
    assert refused[0]["tool"] == LR.UNKNOWN_TOOL and refused[0]["refused"] == "unknown_tool"
    replies = [e for e in rec.of("trace") if e["kind"] == "model_turn"]
    assert replies[0]["tool_names"] == [LR.UNKNOWN_TOOL]
    assert any(x.startswith(":red-badge[refused] `unknown tool`: unknown_tool") for x in view.lines)
    assert any(x.startswith(":green-badge[executed] `query_events`") for x in view.lines)   # accepted stays apart


@pytest.mark.parametrize("entry", [
    {"kind": "tool_call", "tool": MARKER, "outcome": "refused", "refused": MARKER},
    {"kind": "finish", "tool": MARKER, "outcome": "accepted", "arguments": {"finding": MARKER}},
    {"kind": "model_turn", "outcome": MARKER, "stop_reason": MARKER, "text": MARKER,
     "tool_calls": [{"id": MARKER, "name": MARKER}, MARKER], "input_tokens": MARKER, "output_tokens": 3},
    {"kind": "model_error", "reason": MARKER},
    {"kind": "stop", "stopping_reason": MARKER},
    {"kind": "budget_exceeded", "stopping_reason": MARKER, "not_executed": [MARKER]},
    {"kind": MARKER, "step": MARKER, "elapsed_s": MARKER},
])
def test_every_projected_field_is_allowlisted(entry):
    out = LR.progress_event(entry)
    assert "SECRET-7f3a91" not in json.dumps(out) and "chain of thought" not in json.dumps(out)
    assert set(out) <= {"type", "kind", "step", "elapsed_s", "outcome", "stop_reason", "input_tokens",
                        "output_tokens", "tool_names", "reason", "tool", "refused", "stopping_reason"}


def test_known_codes_and_tools_pass_through():
    out = LR.progress_event({"kind": "tool_call", "tool": "compare_windows", "outcome": "refused",
                             "refused": "tool_refused:bad_window", "step": 4, "elapsed_s": 1.5})
    assert out == {"type": "trace", "kind": "tool_call", "step": 4, "elapsed_s": 1.5, "tool": "compare_windows",
                   "outcome": "refused", "refused": "tool_refused:bad_window"}
    out = LR.progress_event({"kind": "model_turn", "stop_reason": "tool_use", "input_tokens": 10, "output_tokens": 2,
                             "tool_calls": [{"name": "query_events"}], "step": 0, "elapsed_s": 0.1})
    assert out["tool_names"] == ["query_events"] and out["stop_reason"] == "tool_use"


# ---- second Codex audit, 1: unexpected failures stay primary ----

def fake_execute(monkeypatch, events, error, calls):
    """runs.execute replaced by a worker that reports `events`, then raises `error` (no investigation runs)."""
    def execute_(config, inputs, *, paid, progress, **kw):
        calls.append(threading.current_thread())
        for e in events:
            progress(e)
        time.sleep(0.05)
        raise error
    monkeypatch.setattr(runs, "execute", execute_)


LOCK = {"type": "stage", "stage": "lock", "state": "completed", "run": "r"}
INPUTS = {"type": "stage", "stage": "inputs", "state": "started"}


def test_codex_a_worker_runtime_error_is_not_replaced_by_a_held_rerun(synthetic, monkeypatch, held_rerun):
    calls = []
    fake_execute(monkeypatch, [LOCK, INPUTS], RuntimeError("worker broke"), calls)
    with pytest.raises(RuntimeError, match="worker broke"):          # not RerunException
        rehearse_via_run(synthetic)
    assert len(calls) == 1 and not calls[0].is_alive()               # joined, and never started twice
    kind, text = held_rerun[ui.MESSAGE_KEY]
    assert kind == "error" and "unexpected error (RuntimeError)" in text and "Not started" not in text
    assert ui.PROGRESS_KEY not in held_rerun                         # nothing is recorded as verified


def test_an_unexpected_error_before_any_stage_is_still_not_a_refusal(synthetic, monkeypatch, held_rerun):
    calls = []
    fake_execute(monkeypatch, [LOCK], RuntimeError("after the lock"), calls)
    with pytest.raises(RuntimeError, match="after the lock"):
        rehearse_via_run(synthetic)
    assert "Not started" not in held_rerun[ui.MESSAGE_KEY][1] and len(calls) == 1


def test_a_documented_refusal_after_the_run_started_is_not_called_not_started(synthetic, monkeypatch, held_rerun):
    calls = []
    fake_execute(monkeypatch, [LOCK, INPUTS], lv1.LiveRefused("late refusal"), calls)
    with pytest.raises(lv1.LiveRefused, match="late refusal"):
        rehearse_via_run(synthetic)
    assert "Not started" not in held_rerun[ui.MESSAGE_KEY][1] and len(calls) == 1


def test_codex_a_rendering_failure_does_not_hide_the_worker_failure():
    view = ui.LiveProgress(False, 8)
    view.draw = lambda: None
    workers = []

    def broken_show(batch, dropped=0):
        if batch:
            raise ValueError("render broke")
    view.show = broken_show

    def work(progress):
        workers.append(threading.current_thread())
        progress(LOCK)
        time.sleep(0.3)
        raise RuntimeError("worker broke")
    with pytest.raises(RuntimeError, match="worker broke") as err:   # precedence: the investigation's failure
        ui.run_with_live_view(work, view)
    assert "(the live progress view also failed: ValueError)" in err.value.__notes__
    assert not workers[0].is_alive() and view.live is False


def test_a_rendering_failure_never_discards_a_completed_run():
    view = ui.LiveProgress(False, 8)
    view.draw = lambda: None
    real_show = view.show

    def broken_show(batch, dropped=0):
        real_show(batch, dropped)
        if batch:
            raise ValueError("render broke")
    view.show = broken_show

    def work(progress):
        progress(LOCK)
        time.sleep(0.2)
        return ("name", {"status": "verified"})
    assert ui.run_with_live_view(work, view) == ("name", {"status": "verified"})
    assert view.render_failure == "ValueError"
    assert "The live view stopped updating (ValueError)" in view.record("name", "verified")["lines"][-1]


def test_an_interrupt_while_rendering_is_raised_only_after_the_worker_ended():
    view = ui.LiveProgress(False, 8)
    finished = []

    def interrupted(batch, dropped=0):
        if batch:
            raise KeyboardInterrupt
    view.show = interrupted

    def work(progress):
        progress(LOCK)
        time.sleep(0.3)
        finished.append(1)
        return "done"
    with pytest.raises(KeyboardInterrupt):
        ui.run_with_live_view(work, view)
    assert finished == [1]                                           # the investigation was not cut short


# ---- second Codex audit, 2: omissions are cumulative ----

def quiet_view():
    view = ui.LiveProgress(False, 8)
    view.draw = lambda: None
    return view


def saved_event(i):
    return {"type": "saved", "file": f"f{i}"}


def test_codex_view_cap_omissions_survive_an_empty_final_drain():
    view = quiet_view()
    view.show([saved_event(i) for i in range(ui.MAX_PROGRESS_EVENTS)], 0)
    view.show([saved_event(-1), saved_event(-2)], 0)                 # past the view's cap
    view.show([], 0)                                                 # the empty final drain
    assert (view.view_omitted, view.queue_dropped, view.dropped) == (2, 0, 2)
    rec = view.record("r", "verified")
    assert rec["dropped"] == 2 and "2 progress events could not be shown" in rec["lines"][-1]
    assert len(view.events) == ui.MAX_PROGRESS_EVENTS


def test_queue_drops_and_view_omissions_add_up_across_batches(monkeypatch):
    monkeypatch.setattr(ui, "MAX_PROGRESS_EVENTS", 2)
    q = ui.ProgressQueue(maxsize=3)
    for i in range(10):
        q.put(saved_event(i))                                        # 3 queued, 7 dropped by the queue
    view = quiet_view()
    view.show(q.drain(0), q.dropped)                                 # 2 kept, 1 past the view's cap
    view.show(q.drain(0), q.dropped)                                 # empty
    view.show([], q.dropped)                                         # a repeated render: nothing new
    assert (view.queue_dropped, view.view_omitted, view.dropped) == (7, 1, 8)
    q.put(saved_event(99))
    view.show(q.drain(0), q.dropped)                                 # 1 more past the cap
    view.show([], 5)                                                 # a stale, lower total never lowers it
    assert (view.queue_dropped, view.view_omitted, view.dropped) == (7, 2, 9)
    assert "9 progress events could not be shown (7 not queued, 2 past the view's limit)" in view.incomplete()


# ---- second Codex audit, 3: explicit allowlists per field ----

SECRET = "private_provider_secret"


@pytest.mark.parametrize("entry", [
    {"kind": "tool_call", "tool": SECRET, "outcome": "refused", "refused": SECRET},
    {"kind": "tool_call", "tool": "query_events", "outcome": "refused", "refused": f"tool_refused:{SECRET}"},
    {"kind": "finish", "tool": SECRET, "outcome": SECRET, "arguments": {"finding": SECRET}},
    {"kind": "model_turn", "outcome": SECRET, "stop_reason": SECRET, "text": SECRET, "model": SECRET,
     "tool_calls": [{"id": SECRET, "name": SECRET}, {"name": {"nested": SECRET}}, SECRET], "input_tokens": SECRET},
    {"kind": "model_error", "reason": SECRET},
    {"kind": "budget_exceeded", "stopping_reason": SECRET, "not_executed": [SECRET]},
    {"kind": "stop", "stopping_reason": SECRET},
    {"kind": SECRET, "step": SECRET, "elapsed_s": SECRET, "tool": SECRET},
])
def test_a_valid_looking_secret_never_reaches_any_progress_field(entry):
    out = LR.progress_event(entry)
    view = quiet_view()
    view.show([out, {"type": "finished", "status": "verified", "stopping_reason": "finished"}], 0)
    for text in (json.dumps(out), json.dumps(view.events), "\n".join(view.lines),
                 json.dumps(view.record("r", "verified"))):
        assert SECRET not in text


def test_a_provider_supplied_model_error_reason_is_not_shown(synthetic):
    data = iv.load_data(*_load_args(synthetic))[0]

    class Failing:
        def respond(self, system, tools, messages):
            raise ia.ModelUnavailable(SECRET, "the provider said something private")
    events = []
    r = av1.investigate_v1("Q", data, Failing(), budget=ia.Budget(), investigation_id="x", allow_synthetic=True,
                           observer=lambda e: events.append(LR.progress_event(e)) if e.get("kind") != "model_request"
                           else None)
    doc = r.to_json()
    assert any(t.get("reason") == SECRET for t in doc["trace"])     # the trace keeps what was recorded
    assert SECRET not in json.dumps(events) and any(e.get("reason") == "provider_error" for e in events)


def test_a_failed_stage_shows_only_a_known_error_type(synthetic, monkeypatch):
    iv_ = runs.script("investigation_v1_validation")
    secret_error = type(SECRET, (RuntimeError,), {})

    def load_data(*a, **k):
        raise secret_error("details")
    monkeypatch.setattr(iv_, "load_data", load_data)
    rec = Recorder(synthetic["tmp"] / "live")
    _, summary = execute(synthetic, rec)
    assert summary["status"] == "failed"
    failed = [e for e in rec.of("stage") if e["state"] == "failed"]
    assert failed == [{"type": "stage", "stage": "inputs", "state": "failed", "error": "error"}]
    assert SECRET not in json.dumps([e for e, _, _ in rec.events])


ALLOWED = [("refused", k) for k in sorted(LR.REFUSAL_KINDS)] + \
    [("reason", k) for k in sorted(LR.MODEL_ERROR_REASONS)] + \
    [("stopping_reason", k) for k in sorted(LR.STOPPING_REASONS)] + \
    [("budget", k) for k in sorted(LR.BUDGET_STOPS)] + \
    [("stop_reason", k) for k in sorted(LR.STOP_REASONS)] + \
    [("outcome", k) for k in sorted(LR.TURN_OUTCOMES)] + [("tool", k) for k in sorted(LR.KNOWN_TOOLS)]


@pytest.mark.parametrize("field, value", ALLOWED)
def test_every_legitimate_status_is_preserved(field, value):
    entry = {"refused": {"kind": "tool_call", "tool": "compare_windows", "outcome": "refused", "refused": value},
             "reason": {"kind": "model_error", "reason": value},
             "stopping_reason": {"kind": "stop", "stopping_reason": value},
             "budget": {"kind": "budget_exceeded", "stopping_reason": value},
             "stop_reason": {"kind": "model_turn", "stop_reason": value, "tool_calls": []},
             "outcome": {"kind": "model_turn", "outcome": value, "tool_calls": []},
             "tool": {"kind": "tool_call", "tool": value, "outcome": "accepted"}}[field]
    out = LR.progress_event(entry)
    assert out[{"budget": "stopping_reason"}.get(field, field)] == value


def test_the_allowlists_cover_every_status_the_code_can_produce():
    """Scanned from the sources, so a new status is never silently hidden as a fallback."""
    import ast
    import re
    src = ROOT_SRC / "live_tracker"
    agent = (src / "investigation_agent.py").read_text() + (src / "investigation_agent_v1.py").read_text()
    tools = (src / "investigation_tools_v1.py").read_text()
    live_ = (src / "investigation_live.py").read_text() + (src / "investigation_live_v1.py").read_text()
    kinds = set(re.findall(r'_Rejected\("([a-z_]+)"', agent))
    codes = set(re.findall(r'_Refused\("([a-z_]+)"', tools)) | set(re.findall(r'_error\("([a-z_]+)"', tools))
    reasons = set(re.findall(r'ModelUnavailable\("([a-z_]+)"', agent + live_)) | \
        {r for _, r in ia._ERROR_REASONS} | {"api_error"} | \
        set(re.findall(r'problem = "([a-z_]+)"', live_))
    stops = set(re.findall(r'self\.stop\("([a-z_]+)"', agent)) | set(re.findall(r'stop_now = "([a-z_]+)"', agent))
    tree = ast.parse((src / "investigation_agent.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("exhausted", "exceeded"):
            stops |= {r.value.value for r in ast.walk(node) if isinstance(r, ast.Return)
                      and isinstance(r.value, ast.Constant) and isinstance(r.value.value, str)}
    assert kinds and codes and reasons and stops
    assert kinds <= LR.REFUSAL_KINDS and codes <= LR.TOOL_ERROR_CODES
    assert {f"tool_refused:{c}" for c in codes} <= LR.REFUSAL_KINDS
    assert reasons <= LR.MODEL_ERROR_REASONS, reasons - LR.MODEL_ERROR_REASONS
    assert stops <= LR.STOPPING_REASONS, stops - LR.STOPPING_REASONS
    assert set(lv1.KNOWN_STOP_REASONS) <= LR.STOP_REASONS and set(av1.TRACE_KINDS) == LR.TRACE_KINDS


ROOT_SRC = Path(__file__).resolve().parents[1] / "src"


# ---- third Codex audit: worker failures survive a persistent Stop / rerun (real Streamlit request state) ----

from streamlit.runtime.scriptrunner_utils.script_requests import (RerunData, ScriptRequests,  # noqa: E402
                                                                   ScriptRequestType)
from streamlit.runtime.state import SafeSessionState, SessionState  # noqa: E402


def streamlit_state(kind):
    """st.session_state as Streamlit builds it: a real SafeSessionState whose yield callback handles a real
    ScriptRequests the way ScriptRunner._maybe_handle_execution_control_request does. kind "stop": a Stop request
    stays pending (every access raises StopException); "rerun": the user keeps asking for a rerun."""
    requests = ScriptRequests()
    if kind == "stop":
        requests.request_stop()

    def on_yield():
        if kind == "rerun":
            requests.request_rerun(RerunData())
        request = requests.on_scriptrunner_yield()
        if request is None:
            return
        if request.type == ScriptRequestType.RERUN:
            raise RerunException(request.rerun_data)
        raise StopException()
    inner = SessionState()
    return SafeSessionState(inner, on_yield), inner


@pytest.mark.parametrize("kind, control", [("stop", StopException), ("rerun", RerunException)])
def test_codex_a_worker_error_survives_a_persistent_control_request(synthetic, monkeypatch, kind, control):
    state, inner = streamlit_state(kind)
    monkeypatch.setattr(ui.st, "session_state", state)
    with pytest.raises(control):                                     # the request really is raised on access
        state["probe"] = 1
    calls = []

    def failing_run(args, **kw):                                     # the real execute and RunLock around it
        calls.append(threading.current_thread())
        kw["progress"]({"type": "stage", "stage": "inputs", "state": "started"})
        raise RuntimeError("worker broke")
    monkeypatch.setattr(LR, "run", failing_run)
    with pytest.raises(RuntimeError, match="worker broke") as err:   # primary: not StopException / RerunException
        rehearse_via_run(synthetic)
    assert not isinstance(err.value, (StopException, RerunException))
    assert any(f.name == "failing_run" for f in __import__("traceback").extract_tb(err.value.__traceback__))
    assert len(calls) == 1 and not calls[0].is_alive()               # ran once, joined
    assert runs.RunLock(synthetic["tmp"] / "live").holder() is None # the lock was released
    assert saved_dirs(synthetic["tmp"]) == [] and ui.PROGRESS_KEY not in inner   # nothing marked verified


@pytest.mark.parametrize("kind, control", [("stop", StopException), ("rerun", RerunException)])
def test_a_persistent_control_request_alone_still_stops_a_successful_run(synthetic, monkeypatch, kind, control):
    state, inner = streamlit_state(kind)
    monkeypatch.setattr(ui.st, "session_state", state)
    with pytest.raises(control):                                     # no other failure: the request is honoured
        rehearse_via_run(synthetic)
    (name,) = saved_dirs(synthetic["tmp"])                           # after the run completed, exactly once
    assert runs.load_saved(synthetic["tmp"] / "live" / name).verified
    assert runs.RunLock(synthetic["tmp"] / "live").holder() is None


def test_a_persistent_stop_does_not_weaken_paid_authorization(synthetic, monkeypatch):
    _, emu = provider(monkeypatch)
    state, _ = streamlit_state("stop")
    monkeypatch.setattr(ui.st, "session_state", state)
    with pytest.raises(StopException):                               # the refusal's message write is stopped
        ui._run(runs.RunConfig(validation_run=runs.REFERENCE_RUN), synthetic["vi"], True, confirm_budget_usd=0.5,
                authorization=None)
    assert emu.requests == [] and saved_dirs(synthetic["tmp"]) == []  # still refused before any request or file


# ---- third Codex audit: Python 3.10 has no add_note ----

def test_a_note_is_attached_without_add_note_and_never_masks_the_error():
    class NoNotes(RuntimeError):
        add_note = None                                              # as on Python 3.10
    error = NoNotes("primary")
    ui._add_note(error, "(the live progress view also failed: ValueError)")
    assert error.__notes__ == ["(the live progress view also failed: ValueError)"]

    class BrokenNotes(RuntimeError):
        def add_note(self, note):
            raise TypeError("notes unavailable")
    ui._add_note(BrokenNotes("primary"), "x")                        # swallowed: never raises


def test_without_add_note_the_worker_error_keeps_precedence_over_a_rendering_failure():
    class NoNotes(RuntimeError):
        add_note = None
    view = ui.LiveProgress(False, 8)

    def broken_show(batch, dropped=0):
        if batch:
            raise ValueError("render broke")
    view.show = broken_show

    def work(progress):
        progress(LOCK)
        time.sleep(0.2)
        raise NoNotes("worker broke")
    with pytest.raises(NoNotes, match="worker broke") as err:
        ui.run_with_live_view(work, view)
    assert err.value.__notes__ == ["(the live progress view also failed: ValueError)"]


# ---- the production page keeps the tabs in place (the selected tab survives the one-time message) ----

def tabs_position(at):
    """Where the page's tabs sit among its top-level elements (Streamlit identifies an element by its position: a
    tabs element that moves is a new one to the browser, which then resets it to its first tab)."""
    return next(k for k, v in at.main.children.items() if getattr(v, "type", None) == "tab_container")


def finished_messages(at):
    return [i.value for i in [*at.info, *at.success, *at.warning, *at.error]
            if "finished with recorded status" in i.value]


def rehearse_on_page(at, synthetic, question):
    """Ask `question`, run one free rehearsal on the page; (page, the ONE run it created). Identified by set
    difference, never by name order (run names are a second-resolution timestamp plus a random suffix)."""
    before = set(saved_dirs(synthetic["tmp"]))
    at.text_area(key="live_agent_question").input(question)
    at = run(_click(at, "Run free rehearsal"))
    created = set(saved_dirs(synthetic["tmp"])) - before
    assert len(created) == 1, created                                # exactly one new run per execution
    return at, created.pop()


@pytest.mark.parametrize("names", [None, ["20261024T120000Z_ffffffff", "20261024T120000Z_00000000"]],
                         ids=["generated names", "same second, reverse lexical order"])
def test_the_one_time_message_never_moves_the_tabs_and_the_selected_report_renders(synthetic, monkeypatch, names):
    from test_live_agent_page import showing
    if names:                                                        # the first run sorts AFTER the second
        queue_ = list(names)
        monkeypatch.setattr(runs, "new_run_name", lambda: queue_.pop(0))
    at = open_page()
    position = tabs_position(at)                                     # no message
    at, first = rehearse_on_page(at, synthetic, "Question one: the first run")
    assert len(finished_messages(at)) == 1 and tabs_position(at) == position     # message shown, tabs in place
    at, second = rehearse_on_page(at, synthetic, "Question two: the second run")
    assert first != second and (names is None or (first, second) == tuple(names))
    assert len(finished_messages(at)) == 1 and second in finished_messages(at)[0] and showing(at) == (second, second)
    at.selectbox(key=ui.SAVED_KEY).set_value(str(synthetic["tmp"] / "live" / first))
    at = run(at)                                                     # the selection change: the next rerun
    assert finished_messages(at) == [] and tabs_position(at) == position         # one-time, tabs still in place
    doc = json.loads((synthetic["tmp"] / "live" / first / "investigation_report.json").read_text())
    assert showing(at) == (first, first)                             # the intended run, not the newer one
    assert doc["question"] == "Question one: the first run"
    assert any(m.value == f"> {doc['question']}" for m in at.markdown)
    assert not any("Question two" in m.value for m in at.markdown)  # nothing stale from the other run
    assert at.selectbox(key=ui.EVIDENCE_KEY + first).value == doc["primary_evidence_id"]   # its own evidence
    assert any(b.label == "Run free rehearsal" for b in at.button)  # New investigation is still there
