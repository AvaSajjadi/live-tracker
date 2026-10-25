"""Streamlit rendering of the Live Agent page (Phase 3D, new file). All logic is investigation_v1_runs'.

A new investigation: question, verified validation inputs, model, turn and tool budgets, local USD target and
key-limit mode; then the free metadata preflight (no model request); then, only for the exact configuration that
preflight approved, the paid run, which needs an explicit confirmation of the target AND the typed authorization
text (choosing best-effort mode authorizes nothing). Each preflight issues a one-use token and the runs directory is
locked while an investigation runs, so a double click, a rerun or a second browser tab cannot start a second paid
run. A free rehearsal (the emulated endpoint, no model, no cost) exercises the same pipeline.

A saved investigation is always reloaded from disk and re-verified offline before anything of it is shown; one that
does not verify shows only why. Reopening never calls the network or the model.
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from . import investigation_agent_v1 as av1
from . import investigation_live_v1 as lv1
from . import investigation_tools_v1 as v1
from . import investigation_v1_runs as runs
from .ui_style import note

PREFLIGHT_KEY = "live_agent_preflight"            # {"digest", "ready", "report", "token"}
USED_KEY = "live_agent_used_tokens"
GEN_KEY = "live_agent_generation"                 # renews the confirmation widgets after every run
OPEN_KEY = "live_agent_open"                      # a saved investigation to select once (its directory)
SAVED_KEY = "live_agent_saved"                    # the saved-investigation selectbox: its value (a directory) is
                                                  # the ONLY source of what the Saved tab shows
MESSAGE_KEY = "live_agent_message"
STATUS_BOX = {"verified": st.success, "incomplete": st.warning, "unverified": st.error}


def _state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---- a new investigation (UI-4: a guided sequence; the settings and their checks are unchanged) ----

# colours by meaning (the app's shared scheme, ui_style): grey neutral, amber asks for the researcher's decision, green
# a check passed or a result verified, red refused / failed / not verifying; the accent marks the agent's own actions
ACTION_BADGES = {"rehearsal": ":gray-badge[free · emulated · no model]",
                 "preflight": ":gray-badge[free · metadata only · no model request]",
                 "paid": ":orange-badge[paid · needs your authorization]"}
# (question, Contract v1 behavior, Contract v1 pair feature): only what query_events and compare_windows expose
EXAMPLE_QUESTIONS = (
    ("Does the distance between the mice change before classifier-predicted attack bouts?", "attack", "dist_center"),
    ("Are the mice closing the distance between them (a falling center distance) before classifier-predicted mount "
     "bouts?", "mount", "dist_rate"),
    ("Is the mice's relative speed different before classifier-predicted investigation bouts?", "investigation",
     "rel_speed"))


def _step(title):
    box = st.container(border=True)
    box.markdown(f"##### {title}")
    return box


def configure():
    """(RunConfig, ValidationInputs, input problems) from the widgets: steps 1 and 2 of the guided sequence. The
    settings, their keys and their limits are those of the reviewed configuration."""
    inputs = runs.default_inputs()
    problems = inputs.problems()
    with _step("Step 1 · Your research question and the data"):
        question = st.text_area("Research question", value=runs.RunConfig.question, max_chars=500,
                                key="live_agent_question",
                                help="Plain language. The agent turns it into Contract v1 tool calls. It can only "
                                     "compare one of Contract v1's pair features (" + ", ".join(v1.WINDOW_FEATURES)
                                     + ") before classifier-predicted attack, investigation or mount bouts with an "
                                     "earlier baseline.")
        st.caption("Examples: " + " · ".join(f"“{q}”" for q, _, _ in EXAMPLE_QUESTIONS))
        st.selectbox("Validation inputs", [inputs.label], key="live_agent_inputs",
                     help="The verified Phase 3B.1 inputs: the validation member only, the expected archive SHA-256, "
                          "the verified pose cache and the tracked split manifest. The test split is never read.")
        for p in problems:
            st.error(f"Inputs: {p}. Nothing is substituted.")
        if problems:
            st.info("The verified validation inputs are not all present, so no rehearsal or investigation can start "
                    "from this page. Nothing synthetic is ever substituted for them.")
    with _step("Step 2 · Model and execution limits"):
        c = st.columns(4)
        model = c[0].selectbox("Model (via OpenRouter)", runs.MODELS, key="live_agent_model",
                               help="Used only by a paid investigation; a free rehearsal uses no model.")
        turns = c[1].number_input("Max model turns", 1, 12, 8, key="live_agent_turns",
                                  help="How many replies the model may give before the run stops.")
        tools = c[2].number_input("Max tool calls", 1, 12, 8, key="live_agent_tools",
                                  help="How many Contract v1 tool calls the model may make (Contract v1 allows 12).")
        target = c[3].number_input("Local spending target (USD)", 0.01, lv1.MAX_TARGET_USD, lv1.MAX_TARGET_USD,
                                   step=0.05, format="%.2f", key="live_agent_target",
                                   help="A target this app checks between requests. One request can pass it, so it "
                                        "is not a guaranteed maximum charge.")
        with st.expander("Advanced settings: provider key limit and reference check"):
            mode = st.radio("Key-limit mode", ("strict", "best-effort"), horizontal=True, key="live_agent_key_mode",
                            help="strict (default): the OpenRouter key's own limit must be at most $0.50. "
                                 "best-effort: an existing non-resetting key limited to at most $50, its limit "
                                 "acknowledged.")
            ack = None
            if mode == "best-effort":
                ack = st.number_input("Acknowledge the key's actual limit (USD)", 0.51,
                                      lv1.MAX_BEST_EFFORT_KEY_LIMIT_USD, lv1.MAX_BEST_EFFORT_KEY_LIMIT_USD,
                                      key="live_agent_ack")
            check = st.checkbox(f"Reference compatibility check against {runs.REFERENCE_RUN} (optional, exact "
                                "matching)", value=True, key="live_agent_reference",
                                help="Compares the investigation's first query with a recorded reference run. It "
                                     "never changes the investigation; a mismatch does not invalidate it.")
        if mode == "best-effort":
            st.warning(f"Best-effort mode: the ${target:.2f} target is NOT a spending ceiling. Spending can exceed "
                       f"it, potentially up to the available credit on the ${ack:g} key, subject to OpenRouter's "
                       "enforcement. Choosing this mode does not authorize a paid run.")
    config = runs.RunConfig(question=question.strip(), model=model, max_model_turns=int(turns),
                            max_tool_calls=int(tools), target_usd=float(target), key_mode=mode,
                            acknowledged_key_limit_usd=ack, validation_run=runs.REFERENCE_RUN if check else None)
    return config, inputs, problems


def show_preflight(report):
    if "refused" in report:
        st.error(f"Refused before any request: {report['refused']}")
        return
    sp = report.get("spending") or {}
    (st.success if report["ready"] else st.error)(
        "Ready: every check passed (metadata only; no model request was made)." if report["ready"] else
        f"Not ready: {'; '.join(map(str, report['problems']))[:1500]}")
    if not report["ready"]:
        st.caption("Nothing was sent to a model. Fix the problems listed (for example the key, its limit or the "
                   "chosen model), then run the readiness check again.")
    m = st.columns(4)
    m[0].metric("Key-limit mode", sp.get("authorization_mode", "?"))
    m[1].metric("Local target", f"${sp.get('local_target_usd', 0):.2f}")
    m[2].metric("Provider key limit", "unverified" if sp.get("provider_key_limit_usd") is None
                else f"${sp['provider_key_limit_usd']:g}")
    m[3].metric("Remaining credit", "unverified" if sp.get("provider_key_remaining_usd") is None
                else f"${sp['provider_key_remaining_usd']:g}")
    if sp.get("exposure"):
        st.caption(f"Exposure: {sp['exposure']}")
    est = report.get("estimate") or {}
    if est:
        st.caption(f"Estimate: typical run ~${est.get('typical_run_usd')}, worst case "
                   f"${est.get('estimated_worst_case_usd')} (client-side estimate, not a cap); Stage 6 cost cap "
                   f"${est.get('stage6_cost_cap_usd')} plus a one-request reserve ${est.get('one_request_reserve_usd')}.")
    with st.expander("Preflight details (confirmed metadata, local configuration, assumptions)"):
        st.markdown("**Only a real model response can show:**")
        for a in report.get("requires_a_model_response", []):
            st.markdown(f"- {a}")
        st.json({k: report.get(k) for k in ("plan", "confirmed_provider_metadata", "credentials", "estimate",
                                            "local_configuration")}, expanded=False)


# ---- UI-3: live progress of a run in the tab that started it (presentation only) ----

try:                                              # a rerun / stop request is raised from inside st.* calls
    from streamlit.runtime.scriptrunner_utils.exceptions import ScriptControlException
except ImportError:  # pragma: no cover - another Streamlit: no live updates during a run (as before UI-3)
    ScriptControlException = None

PROGRESS_KEY = "live_agent_last_progress"         # the last run's progress record in this browser tab
MAX_PROGRESS_EVENTS = 400                         # a run has at most 12 turns and 12 tool calls: ample
STAGE_LABELS = {"lock": "Investigation lock acquired (one investigation at a time)",
                "inputs": "Loading the verified validation inputs",
                "validation_run": "Checking the recorded reference run",
                "preflight": "Provider preflight (metadata only, no model request)",
                "budget": "Setting the budget",
                "investigate": "Investigation (model turns and Contract v1 tools)",
                "verify": "Checking the saved report",
                "cost_ledger": "Checking the cost ledger against the report",
                "compatibility": "Reference compatibility check"}


class ProgressQueue:
    """The bridge from the investigation (worker thread) to the page (Streamlit script thread): bounded and
    thread-safe. put() never blocks and never raises: when the queue is full the event is dropped and counted, so
    the view can say it is incomplete instead of pretending otherwise."""

    def __init__(self, maxsize=MAX_PROGRESS_EVENTS):
        self._q, self._lock, self.dropped = queue.Queue(maxsize), threading.Lock(), 0

    def put(self, event):
        try:
            self._q.put_nowait(event)
        except queue.Full:
            with self._lock:
                self.dropped += 1

    def drain(self, timeout):
        """Every queued event, waiting up to `timeout` seconds for the first one."""
        out = []
        try:
            out.append(self._q.get(timeout=timeout) if timeout else self._q.get_nowait())
            while True:
                out.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return out


def _add_note(error, text):
    """Attach a diagnostic note (a type name, never a message) without ever raising: add_note() on Python 3.11+,
    else the same __notes__ list by hand (Python 3.10 keeps it, its traceback just does not print it)."""
    try:
        add = getattr(error, "add_note", None)
        if callable(add):
            add(text)
        else:
            notes = getattr(error, "__notes__", None)
            error.__notes__ = (list(notes) if isinstance(notes, list) else []) + [text]
    except Exception:
        pass                                      # a note is optional: the error itself is what matters


def _is_control(exc):
    return ScriptControlException is not None and isinstance(exc, ScriptControlException)


def _report_unexpected(view, exc):
    """Best effort while an unexpected failure is leaving _run: close the view and leave a message. Streamlit may
    refuse (a pending stop or rerun is raised from session-state access and page updates) or fail; either is
    swallowed HERE ONLY, so the original failure stays the one that propagates. Interrupts are not swallowed."""
    try:
        view.close("error", "Stopped by an unexpected error")
        st.session_state[MESSAGE_KEY] = (
            "error", f"The investigation stopped with an unexpected error ({type(exc).__name__}). Nothing from it "
            "is shown as verified and it is not repeated; open Saved investigations to see whether a report was "
            "saved and whether it verifies.")
    except Exception:
        pass
    except BaseException as secondary:
        if not _is_control(secondary):
            raise


def run_with_live_view(work, view, poll_s=0.1):
    """work(progress) runs on ONE worker thread (the investigation: runs.execute, unchanged); its progress events
    cross a ProgressQueue and are rendered here, on the Streamlit script thread, in batches. The worker never waits
    for rendering, and never calls Streamlit. Returns work's result, or re-raises its exception on this thread after
    the worker has ended (it is always joined: no investigation outlives this call, and none is started twice)."""
    events, outcome = ProgressQueue(), {}

    def worker():
        try:
            outcome["result"] = work(events.put)
        except BaseException as exc:              # handed to the script thread
            outcome["error"] = exc
    thread = threading.Thread(target=worker, name="live-agent-investigation", daemon=False)
    thread.start()
    render_error = None
    try:
        while thread.is_alive():
            view.show(events.drain(poll_s), events.dropped)
        view.show(events.drain(0), events.dropped)
    except BaseException as exc:                  # the display failed: the run is still waited for and inspected
        render_error = exc
        view.live = False
    finally:
        thread.join()                             # even if this thread is interrupted, the run completes first
    # precedence: (1) the investigation's own failure, with the display failure noted by type only; (2) a display
    # failure that is not an ordinary Exception (e.g. KeyboardInterrupt); (3) the investigation's result (an
    # ordinary display failure never discards a run that completed; the view then says it stopped updating)
    if "error" in outcome:
        error = outcome["error"]
        if render_error is not None:
            _add_note(error, f"(the live progress view also failed: {type(render_error).__name__})")
        raise error
    if render_error is not None:
        if not isinstance(render_error, Exception):
            raise render_error
        view.render_failure = type(render_error).__name__
    return outcome["result"]


class LiveProgress:
    """The live view of one run in the tab that started it: the CLI's progress events (runs.execute(progress=)),
    rendered into an st.status box on the Streamlit script thread (run_with_live_view), never on the investigation's
    thread, so rendering takes no time from the investigation. Presentation only. Rendering is where Streamlit raises
    a pending rerun or stop request (a click, the Stop button, a closed tab): such a request is HELD and re-raised by
    release() after the run has finished and been saved, so the investigation is never interrupted (as with the
    former blocking spinner). Any other display error just ends the live updates."""

    def __init__(self, paid, max_turns):
        self.paid, self.max_turns = paid, max_turns
        self.events, self.lines = [], []
        self.turns, self.accepted, self.refused = 0, [], []
        self.stage = self.failed = self.spending = self.stop = self.status = None
        self.waiting, self.deferred, self.live = False, None, ScriptControlException is not None
        self.queue_dropped = 0                    # events the bounded queue could not take (the producer's total)
        self.view_omitted = 0                     # events past the view's own cap (MAX_PROGRESS_EVENTS)
        self.render_failure = None                # the type of a display failure, if the live view stopped
        self.box = st.status("Starting the investigation...", expanded=True)
        # placeholders created on the box itself: leaving a `with st.status(...)` block would mark it complete
        self.now, self.counts, self.log = self.box.empty(), self.box.empty(), self.box.empty()

    # -- state from the events --

    @property
    def dropped(self):
        """Every event this view did not get or did not keep: never decreases."""
        return self.queue_dropped + self.view_omitted

    def __call__(self, event):
        self.show([event], self.queue_dropped)

    def show(self, batch, queue_dropped=0):
        """Apply a batch of events, then draw once. queue_dropped is the queue's running total (it only grows)."""
        before = self.dropped
        self.queue_dropped = max(self.queue_dropped, queue_dropped)
        for event in batch:
            if len(self.events) >= MAX_PROGRESS_EVENTS:
                self.view_omitted += 1
                continue
            self.events.append(event)
            self.apply(event)
        if batch or self.dropped != before:
            self.render()

    def started(self):
        """True once anything beyond taking the lock was reported (the investigation's own stages began)."""
        return any(not (e.get("type") == "stage" and e.get("stage") == "lock") for e in self.events)

    def apply(self, e):
        kind = e.get("type")
        if kind == "stage":
            label = STAGE_LABELS.get(e["stage"], e["stage"])
            if e["state"] == "failed":
                self.failed = e["stage"]
                self.lines.append(f":red-badge[failed] {label} ({e.get('error')})")
            elif e["state"] == "started":
                self.stage = e["stage"]
                self.lines.append(f":gray-badge[started] {label}")
            else:
                self.stage = None if self.stage == e["stage"] else self.stage
                self.lines.append(f":green-badge[done] {label}")
        elif kind == "model_request":
            self.waiting = True
            self.lines.append(f":gray-badge[waiting] Request sent for model turn {self.turns + 1} of "
                              f"{self.max_turns}" + ("" if self.paid else " (emulated endpoint)"))
        elif kind == "trace":
            self.trace(e)
        elif kind == "spending":
            self.spending = e
        elif kind == "saved":
            self.lines.append(f":green-badge[saved] `{e['file']}` written")
        elif kind == "finished":
            self.status = e.get("status")
            self.stop = e.get("stopping_reason") or self.stop
            self.lines.append(f":green-badge[finished] Run finished with recorded status `{self.status}`")

    def trace(self, e):
        k = e.get("kind")
        if k == "model_turn":
            self.turns += 1
            self.waiting = False
            if e.get("outcome"):
                self.lines.append(f":red-badge[not used] Model reply {self.turns}: {e['outcome']}")
            else:
                asked = ", ".join(f"`{n}`" for n in e.get("tool_names") or []) or "no tool"
                self.lines.append(f":primary-badge[reply] Model reply {self.turns} of {self.max_turns}: asked for "
                                  f"{asked}")
        elif k == "model_error":
            self.waiting = False
            self.lines.append(f":red-badge[provider] Model request failed ({e.get('reason')}): no reply was used")
        elif k in ("tool_call", "finish"):
            tool = f"`{e.get('tool')}`"
            if e.get("outcome") == "refused":
                self.refused.append(e.get("tool"))
                self.lines.append(f":red-badge[refused] {tool}: {e.get('refused')} (not executed as evidence)")
            elif k == "finish":
                self.accepted.append(e.get("tool"))
                self.lines.append(f":green-badge[finding submitted] {tool}: accepted by the loop; the finding is "
                                  "shown only from the saved report once it verifies")
            else:
                self.accepted.append(e.get("tool"))
                self.lines.append(f":green-badge[executed] {tool}: accepted (result shown after verification)")
        elif k == "budget_exceeded":
            self.lines.append(f":orange-badge[budget] Limit reached: {e.get('stopping_reason')}")
        elif k == "stop":
            self.stop = e.get("stopping_reason")
            self.lines.append(f":orange-badge[stopped] The investigation loop stopped: `{self.stop}`")

    # -- the page --

    def render(self):
        if not self.live:
            return
        try:
            self.draw()
        except Exception:                         # a display problem: no more live updates, the run goes on
            self.live = False
        except BaseException as exc:
            if ScriptControlException is None or not isinstance(exc, ScriptControlException):
                raise
            self.deferred = self.deferred or exc  # held until the run is over (release())

    def draw(self):
        if self.failed:
            self.now.error(f"Stage failed: {STAGE_LABELS.get(self.failed, self.failed)}. The run stops; its "
                           "recorded status will say why.")
        elif self.waiting:
            who = "the provider (OpenRouter)" if self.paid else "the emulated endpoint"
            self.now.info(f"Waiting for {who} to reply to model turn {self.turns + 1} of {self.max_turns}. "
                          "Nothing is known about this reply until it arrives.")
        elif self.stage:
            self.now.info(f"In progress: {STAGE_LABELS.get(self.stage, self.stage)}")
        elif self.stop:
            self.now.info(f"The investigation loop stopped (`{self.stop}`); saving and checking the report.")
        with self.counts.container():
            c = st.columns(4)
            c[0].metric("Model turns", f"{self.turns} of {self.max_turns}")
            c[1].metric("Tool calls executed", len([t for t in self.accepted if t != av1.FINISH]))
            c[2].metric("Tool calls refused", len(self.refused))
            c[3].metric(*self.spent())
        self.log.markdown("  \n".join(self.lines[-40:]) + self.incomplete())
        self.box.update(label=self.label(), expanded=True)

    def incomplete(self):
        out = ""
        if self.dropped:
            out += (f"  \n:orange-badge[incomplete] {self.dropped} progress events could not be shown "
                    f"({self.queue_dropped} not queued, {self.view_omitted} past the view's limit), so this is not "
                    "a complete timeline (the saved report is not affected)")
        if self.render_failure:
            out += (f"  \n:orange-badge[incomplete] The live view stopped updating ({self.render_failure}); the "
                    "saved report is not affected")
        return out

    def spent(self):
        s = self.spending
        if s is None:
            return ("Provider-reported spending" if self.paid else "Simulated cost (not spent)"), "none reported yet"
        value = f"${s['known_cost_usd']:.5f}" + (f" + {s['unknown_cost_attempts']} unreported"
                                                 if s["unknown_cost_attempts"] else "")
        return ("Provider-reported spending" if s["basis"] == "provider-reported" else
                "Simulated cost (not spent)"), value

    def label(self):
        if self.failed:
            return "Stopped: a stage failed"
        if self.waiting:
            return f"Waiting for the {'provider' if self.paid else 'emulated endpoint'} (turn {self.turns + 1})..."
        if self.stage:
            return STAGE_LABELS.get(self.stage, self.stage) + "..."
        return "Investigating..."

    def close(self, state, label, line=None):
        if line:
            self.lines.append(line)
        if not self.live:
            return
        try:
            self.now.empty()
            self.log.markdown("  \n".join(self.lines[-40:]) + self.incomplete())
            self.box.update(label=label, state=state, expanded=True)
        except Exception:
            self.live = False
        except BaseException as exc:
            if ScriptControlException is None or not isinstance(exc, ScriptControlException):
                raise
            self.deferred = self.deferred or exc

    def record(self, name, verdict):
        """What the tab keeps after the run (projected events only: no text, arguments or findings)."""
        missing = self.incomplete().strip()
        return {"run": name, "paid": self.paid, "lines": list(self.lines) + ([missing] if missing else []),
                "dropped": self.dropped, "queue_dropped": self.queue_dropped, "view_omitted": self.view_omitted,
                "render_failure": self.render_failure, "turns": self.turns,
                "max_turns": self.max_turns, "executed": len([t for t in self.accepted if t != av1.FINISH]),
                "refused": len(self.refused), "stop": self.stop, "status": self.status, "verdict": verdict,
                "spending": self.spent()}

    def release(self):
        """Re-raise a rerun / stop request held during the run."""
        if self.deferred is not None:
            raise self.deferred


def verdict_of(saved):
    """(state, text) of the saved report's re-verification (load_saved), for the end of a run."""
    if not saved.verified:
        return "error", ("the saved investigation does NOT verify, so none of its results are shown: "
                         + "; ".join(map(str, saved.problems[:2])))
    kind, _ = runs.autonomous_status(saved)
    return ("complete", "verified") if kind == "verified" else ("complete", "verified, incomplete (no finding)")


def render_progress_record(rec):
    with st.expander(f"Live progress of the last run started in this tab ({rec['run']})", expanded=False):
        note("caption", "Recorded in this browser tab only. Results come from the saved, re-verified report.",
             "Recorded as the run happened, in this browser tab only (not a monitor of other tabs or "
             "sessions). Results are shown only from the saved, re-verified report.")
        c = st.columns(4)
        c[0].metric("Model turns", f"{rec['turns']} of {rec['max_turns']}")
        c[1].metric("Tool calls executed", rec["executed"])
        c[2].metric("Tool calls refused", rec["refused"])
        c[3].metric(*rec["spending"])
        st.markdown("  \n".join(rec["lines"]))


def outcome_message(name, summary, saved, verdict, path):
    """(kind, text, path to open) for the end of a run. saved: the re-VERIFIED SavedInvestigation, else None (then
    nothing of the report is described). Only what the run recorded and what verification established."""
    status = summary.get("status")
    lead = f"Investigation {name} finished with recorded status `{status}`; the saved report was reloaded from disk " \
           f"and re-verified: {verdict}."
    if saved is None:
        return ("error", f"{lead} None of its results are shown. Open the Saved investigations tab to see why it "
                         "does not verify.", path)
    d = saved.doc
    if d["finished"]:
        kind = "success"
        outcome = (f"Outcome: {runs.FINDING_HEADLINES.get(d['finding'], d['finding'])} "
                   f"(finding `{d['finding']}`, association only).")
    else:
        kind = "warning"
        outcome = (f"Outcome: stopped before a finding (`{d['stopping_reason']}`); the report is verified but "
                   "incomplete, and its evidence is partial.")
    notes = []
    rehearsal = saved.kind == "rehearsal"         # the emulated endpoint: its costs are simulated, nothing is spent
    if status == "over_target":
        kind, notes = "warning", notes + ["The simulated cost went past the local target (nothing was spent)."
                                          if rehearsal else "The provider-reported spend went past the local target."]
    elif status not in ("verified", "incomplete"):
        notes.append(f"The recorded status `{status}` comes from a later check (for example the reference "
                     "comparison); see the report's technical details.")
    cost = summary.get("cost") or {}
    if rehearsal:
        notes.append("Free rehearsal: no provider was called, so actual API spending is $0; its usage and costs are "
                     "simulated.")
        if cost and cost.get("actual_spend_complete") is False:
            notes.append("Some emulated replies reported no cost, so the simulated accounting is incomplete.")
    elif cost and cost.get("actual_spend_complete") is False:
        notes.append("Some requests reported no provider cost, so the actual spend may be higher than shown.")
    return (kind, " ".join([lead, outcome, *notes, "Open the Saved investigations tab for its overview, evidence "
                                                     "and activity."]), path)


# ---- the page (shared by pages/5_Live_Agent.py and the synthetic demo) ----

TAB_KEY = "live_agent_tab"                        # the selected tab (tracked, so a button can open the results)
TABS = ("New investigation", "Saved investigations")


def _open_saved(path):
    """Button callback (runs before the page): select the run and open the Saved tab. Starts nothing."""
    st.session_state[OPEN_KEY] = path
    st.session_state[TAB_KEY] = TABS[1]


QUERY_BEHAVIORS = tuple(next(s for s in av1.TOOL_SPECS_V1 if s["name"] == "query_events")
                        ["input_schema"]["properties"]["behavior"]["enum"])
INTRO_POINTS = (
    "**What you can ask:** whether one of the pair features Contract v1 exposes to the agent ("
    + "; ".join(f"`{f}`: {runs.FEATURE_MEANINGS.get(f, f)}" for f in v1.WINDOW_FEATURES)
    + ") differs just before classifier-predicted "
    + ", ".join(QUERY_BEHAVIORS) + " bouts compared with an earlier baseline. Other pose measurements in the "
    "wider project (for example a single mouse's speed or its orientation) are not available to this agent.",
    "**Who decides:** the agent chooses every Contract v1 tool call itself (which events to query, which windows and "
    "feature to compare, when to finish). A paid run uses a real model; a free rehearsal uses a fixed offline "
    "investigator instead.",
    "**What the events are:** bouts *predicted by a classifier* (threshold crossings of its scores), not "
    "experimentally confirmed or annotated behavior.",
    "**What data:** the verified CalMS21 validation inputs only; the test split is never read, and nothing synthetic "
    "is ever substituted.",
    "**What a finding means:** a descriptive association (a median of session-paired differences). No causal claim, "
    "no significance test, no confidence interval.",
    "**When it counts:** only after its saved report is reloaded and verified. This is a research prototype, not a "
    "validated scientific instrument.",
    "**How it differs from the other pages:** Rules, Replay, Investigation and Real Session are the deterministic "
    "workflow (researcher-approved rules, monitoring, a fixed analysis sequence). Here an agent plans the analysis "
    "itself, within Contract v1's fixed tools and checks.")


JOURNEY = ("Ask a question", "Set the limits", "Rehearse free, or authorize one paid run",
           "Open the verified report")
NOTICE_MEANINGS = ("**How to read the notices:** green means a check passed or a saved report verified; amber "
                   "means something needs your attention or decision (an authorization, an incomplete run, a limit "
                   "reached); red means something was refused, failed or does not verify; blue panels and grey "
                   "labels are neutral information.")


def render_intro():
    st.caption("An autonomous agent investigates your question on the verified validation split. Association only; "
               "publication not authorized.")
    st.markdown("  ".join(f":primary-badge[{i}] {step}" for i, step in enumerate(JOURNEY, 1)))
    with st.expander("What this page does, and what its evidence can and cannot establish"):
        st.markdown("\n".join(f"- {x}" for x in INTRO_POINTS))
        st.markdown(NOTICE_MEANINGS)


def render_message(container, message):
    kind, text = message[0], message[1]
    show = {"success": container.success, "warning": container.warning, "error": container.error}.get(
        kind, container.info)
    show(text)
    if len(message) > 2 and message[2]:
        container.button("Open this investigation in Saved investigations", key="live_agent_open_result",
                         on_click=_open_saved, args=(message[2],), icon=":material/folder_open:")


def render_page(title="Live Agent", banner=None):
    """The Live Agent page: introduction, the one-time completion message and the two tabs. The caller applies the
    app's shared style (ui_style.apply()) first, as every page does."""
    st.title(title)
    if banner:
        st.warning(banner)
    render_intro()
    message = st.session_state.pop(MESSAGE_KEY, None)
    # The one-time completion message sits in a container that is ALWAYS rendered, so the tabs below never move; the
    # tabs are also keyed, so the browser keeps (and a button can set) the selected tab.
    notice = st.container()
    if message:
        render_message(notice, message)
    tab_new, tab_saved = st.tabs(list(TABS), key=TAB_KEY, on_change="rerun")
    with tab_new:
        render_new()
    with tab_saved:
        render_open()


# a run that is refused before it starts (documented: authorization, environment, inputs, configuration); anything
# else from runs.execute is an unexpected failure and stays the primary one
EXPECTED_REFUSALS = runs.script("investigation_v1_live").REFUSALS + (ValueError,)


def _run(config, inputs, paid, **kw):
    """One run with its live view. Every exit settles the view: a rerun / stop request held during the run is
    re-raised at the end, unless another failure is already leaving this function, which stays the primary one (the
    held request is then dropped). The run itself is never repeated here."""
    view = LiveProgress(paid, config.max_model_turns)
    failure = None
    try:
        try:
            name, summary = run_with_live_view(
                lambda progress: runs.execute(config, inputs, paid=paid, progress=progress, **kw), view)
        except runs.RunBusy as exc:
            view.close("error", "Not started")
            st.session_state[MESSAGE_KEY] = ("error", f"Not started: {exc}. Only one investigation can run at a "
                                                      "time; wait for it to finish, then start again.")
            return
        except EXPECTED_REFUSALS as exc:
            if view.started():                     # (never "not started" once the investigation's stages began)
                raise
            view.close("error", "Not started")     # the CLI's refusals: before any request or file
            st.session_state[MESSAGE_KEY] = ("error", f"Not started ({type(exc).__name__}): {exc}. Nothing was sent "
                                                      "to a provider and no investigation file was written.")
            return
        view.close("running", "Re-verifying the saved report from disk...",
                   ":gray-badge[started] Re-verifying the saved report from disk")
        saved = None
        try:
            saved = runs.load_saved(runs.live_dir() / name)
            state, verdict = verdict_of(saved)
        except Exception as exc:                   # never "verified" when the check itself failed
            saved, state, verdict = None, "error", (f"the saved report could not be loaded or checked "
                                                    f"({type(exc).__name__}), so none of its results are shown")
        view.close(state, f"Run {name}: recorded status {summary.get('status')}; saved report "
                          f"{verdict.split(':')[0].split(' (')[0]}",
                   (":green-badge[verified]" if state == "complete" else ":red-badge[not verified]") +
                   f" Saved report re-verified: {verdict}")
        st.session_state[PROGRESS_KEY] = view.record(name, verdict)
        # select the new run through the Session State API (sent to the browser), never by dropping the widget's
        # state: a dropped state falls back to a default the browser is not told about, so the selector would keep
        # showing the previous run above the new run's results
        st.session_state[OPEN_KEY] = str(runs.live_dir() / name)
        st.session_state[MESSAGE_KEY] = outcome_message(name, summary, saved if state == "complete" else None,
                                                        verdict, str(runs.live_dir() / name))
    except BaseException as exc:
        failure = exc
        if not _is_control(exc):                   # an unexpected failure: reporting it is secondary
            _report_unexpected(view, exc)
        raise                                      # the original exception, type and traceback
    finally:
        if failure is None:
            view.release()                         # the held rerun / stop request, now that the run is settled
        else:
            view.deferred = None                   # the original failure stays the primary one


def readiness(problems, pf, digest, used, confirm, phrase, busy):
    """The paid run's requirements as (met, text), in order; the Start button needs every one."""
    if pf is None:
        check = (False, "Run the readiness check (step B1) for this configuration")
    elif pf["digest"] != digest:
        check = (False, "The configuration changed after the last readiness check: run it again")
    elif not pf["ready"]:
        check = (False, "The last readiness check did not pass: fix its problems and run it again")
    elif pf["token"] in used:
        check = (False, "That readiness check was already used for one paid run: run it again for another")
    else:
        check = (True, "Readiness check passed for exactly this configuration (usable once)")
    return [(not problems, "Verified validation inputs present"), check,
            (bool(confirm), "Spending target confirmed"),
            (phrase == lv1.PAID_RUN_AUTHORIZATION, "Authorization text typed exactly"),
            (not busy, "No other investigation running")]


def render_new():
    config, inputs, problems = configure()
    digest = config.digest(inputs)
    used, gen = _state(USED_KEY, set()), _state(GEN_KEY, 0)
    busy = runs.RunLock(runs.live_dir()).holder() is not None
    if busy:
        st.warning("Another investigation is running: wait for it to finish. Only one investigation can run at a "
                   "time, across every tab and session; nothing new can start until it ends.")
    with _step("Step 3 · Choose how to run"):
        with st.container(border=True):
            st.markdown(f"**Option A · Free rehearsal** {ACTION_BADGES['rehearsal']}")
            note("caption", "The same pipeline over an emulated endpoint. Nothing is spent.",
                 "The offline deterministic investigator answers instead of a model: no network, no provider, no "
                 "cost. Useful to check the inputs and see the report's form; its choices are scripted, not a "
                 "model's, and its costs are simulated.")
            if st.button("Run free rehearsal", key=f"live_agent_rehearse_{gen}", disabled=busy or bool(problems)):
                st.session_state[GEN_KEY] = gen + 1
                _run(config, inputs, False, confirm_budget_usd=None, authorization=None)
                st.rerun()
        with st.container(border=True):
            st.markdown(f"**Option B · Paid autonomous investigation** {ACTION_BADGES['paid']}")
            note("caption", "A real model chooses every Contract v1 call; each reply is billed. Two free steps come "
                            "first.",
                 "The model is reached through OpenRouter. B1 reads provider metadata only; B2 is your explicit "
                 "authorization of exactly one run of exactly this configuration.")
            st.markdown(f"**B1 · Readiness check** {ACTION_BADGES['preflight']}")
            st.caption("Reads the model list, the key's status and limit and the endpoint list; never asks a model. "
                       f"OPENROUTER_API_KEY is {'set' if runs.key_present() else 'NOT set'} in the app's environment "
                       "(its value is never shown or written).")
            if st.button("Run preflight", key="live_agent_preflight_button"):
                ready, report = runs.preflight(config, inputs)
                st.session_state[PREFLIGHT_KEY] = {"digest": digest, "ready": ready, "report": report,
                                                   "token": secrets.token_hex(8)}
            pf = st.session_state.get(PREFLIGHT_KEY)
            if pf is not None:
                if pf["digest"] != digest:
                    st.info("The configuration changed since the last preflight: run the preflight again.")
                else:
                    show_preflight(pf["report"])
            st.markdown(f"**B2 · Authorize one paid investigation** {ACTION_BADGES['paid']}")
            fresh = pf is not None and pf["digest"] == digest and pf["ready"] and pf["token"] not in used
            confirm = st.checkbox(f"I confirm ONE paid investigation with a local target of ${config.target_usd:.2f} "
                                  "(a best-effort target, not a guaranteed maximum charge)",
                                  key=f"live_agent_confirm_{gen}")
            phrase = st.text_input("Type the authorization text exactly", key=f"live_agent_phrase_{gen}",
                                   placeholder=lv1.PAID_RUN_AUTHORIZATION)
            st.caption(f'Authorization text: "{lv1.PAID_RUN_AUTHORIZATION}"')
            checks = readiness(problems, pf, digest, used, confirm, phrase, busy)
            met = sum(ok for ok, _ in checks)
            st.markdown(f"**Requirements for the paid run: {met} of {len(checks)} met** "
                        + (":green-badge[ready to start]" if met == len(checks) else ":gray-badge[not ready]"))
            st.markdown("  \n".join((":green-badge[ready]" if ok else ":gray-badge[not yet]") + f" {text}"
                                    for ok, text in checks))
            if not fresh and not busy:
                st.caption("A paid run needs a ready preflight of exactly this configuration (one paid run per "
                           "preflight).")
            if st.button("Start paid investigation", type="primary", key=f"live_agent_start_{gen}",
                         disabled=not (fresh and confirm and phrase == lv1.PAID_RUN_AUTHORIZATION and not busy
                                       and not problems)):
                used.add(pf["token"])                  # consumed before anything runs: no second submission
                st.session_state[GEN_KEY] = gen + 1
                _run(config, inputs, True, confirm_budget_usd=config.target_usd, authorization=phrase)
                st.rerun()
    if st.session_state.get(PROGRESS_KEY):
        with _step("Step 4 · Review"):
            note("caption", "The last run's live progress. Its results are in Saved investigations.",
                 "Results are shown there only after the saved report is reloaded from disk and re-verified.")
            render_progress_record(st.session_state[PROGRESS_KEY])


# ---- a saved investigation ----

def render_reference(saved, ref=None):
    st.markdown(f"**Reference compatibility check against {runs.REFERENCE_RUN}** (separate from the investigation; "
                "exact initial request and result matching, fail closed)")
    ref = ref or runs.reference_check(saved)
    if ref["recorded"] is not None:
        st.caption(f"Recorded at run time: {ref['recorded']['status']}"
                   + (f" ({ref['recorded']['message']})" if ref["recorded"].get("message") else ""))
    elif saved.kind != "offline":
        st.caption("No reference check was requested for this run; the check below is computed now, offline.")
    (st.success if ref["equivalent"] else st.warning)(f"Verdict: {ref['verdict']}")
    if ref.get("explanation"):
        st.markdown(ref["explanation"])
    if ref["rows"]:
        st.dataframe(pd.DataFrame(ref["rows"]), hide_index=True, width="stretch")
        st.caption("'not sent' means the model left the parameter to Contract v1's default; 'applied' is the value "
                   "the tool recorded in its result.")


def render_cost(saved):
    c = runs.cost_view(saved)
    if c is None:
        st.caption("No model was called (offline deterministic investigator): no cost.")
        return
    m = st.columns(5 if saved.kind == "rehearsal" else 4)
    if saved.kind == "rehearsal":                 # an emulated ledger: nothing was spent
        m[0].metric("Actual API spending", "$0.00")
        m[4].metric("Simulated cost (emulated, not spent)", f"${c['actual_spend_usd']:.5f}")
        st.info(f"Simulated cost ${c['actual_spend_usd']:.5f}; actual API spending $0. This rehearsal used the "
                "emulated endpoint: its tokens and costs are emulated, no provider was called.")
    else:
        m[0].metric("Actual spend (provider-reported)", f"${c['actual_spend_usd']:.5f}"
                    + ("" if c["actual_spend_complete"] else " +unknown"))
    m[1].metric("Local target", f"${c['target_usd']:.2f}")
    m[2].metric("Provider key limit", "none (no provider)" if c["provider_key_limit_usd"] is None
                else f"${c['provider_key_limit_usd']:g}")
    m[3].metric("Over target", "yes" if c["over_target"] else "no")
    if saved.kind == "rehearsal":
        st.caption(f"Key-limit mode `{c['authorization_mode']}` (recorded; no key was used).")
    else:
        st.caption(f"Key-limit mode `{c['authorization_mode']}`. {c['exposure']}. {c['note']}")
    st.dataframe(pd.DataFrame(runs.ledger_rows(saved.summary, simulated=saved.kind == "rehearsal")),
                 hide_index=True, width="stretch")


# ---- UI-1: the scientific overview and the recorded agent activity ----

BADGES = {"accepted": ":green-badge[accepted as evidence]", "accepted_insufficient":
          ":orange-badge[accepted: insufficient evidence]", "accepted_other": ":gray-badge[accepted]",
          "finding_accepted": ":green-badge[finding accepted]", "refused": ":red-badge[refused]",
          "problem": ":red-badge[stopped]", "finished": ":green-badge[investigation finished]",
          "incomplete": ":orange-badge[incomplete]"}
ICONS = {"accepted": ":material/check_circle:", "finding_accepted": ":material/task_alt:",
         "accepted_insufficient": ":material/error:",
         "accepted_other": ":material/check:", "refused": ":material/block:", "problem": ":material/stop_circle:",
         "finished": ":material/flag:", "incomplete": ":material/hourglass_disabled:"}


MAIN_DIFFERENCE = "Main difference (pre − baseline)"
SIMULATED_COST = "Simulated cost (emulated, not spent)"
# Metric labels, qualifiers and values wrap instead of being cut off (a value only between number and unit); small
# scientific captions get more contrast and size; long identifiers wrap instead of widening the page.
READABLE = """<style>
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {white-space: normal !important;
  overflow: visible !important; text-overflow: clip !important; height: auto !important;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {white-space: normal !important;
  overflow: visible !important; text-overflow: clip !important; overflow-wrap: normal !important;
  font-size: 1.75rem; line-height: 1.2;}
[data-testid="stCaptionContainer"] {color: inherit !important; opacity: 0.86; font-size: 0.92rem; line-height: 1.45;}
[data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {overflow-wrap: anywhere;}
[data-testid="stMarkdownContainer"] code {white-space: normal; overflow-wrap: anywhere;}
</style>"""


def _num(x, digits=4):
    return f"{x:.{digits}g}" if isinstance(x, float) else ("—" if x is None else str(x))


def _seconds(x):
    return "—" if x is None else f"{x:g} s"


def render_status_panels(saved, ov):
    kind, text = runs.autonomous_status(saved)
    left, right = st.columns(2)
    with left:
        (st.success if kind == "verified" else st.warning)(
            f"**Scientific report: {'verified' if kind == 'verified' else 'verified, incomplete'}.** {text}")
    ref = ov.get("reference") or {}
    name = ref.get("reference", runs.REFERENCE_RUN)
    with right:
        if ref.get("equivalent"):
            st.success(f"**Reference comparison ({name}): equivalent.** The initial request and result are exactly "
                       f"{name}'s.")
        elif ref.get("available"):
            th = (f" Agent threshold {ref['agent_threshold']}"
                  + ("" if ref.get("agent_threshold_sent") else " (not set by the agent: Contract v1's default)")
                  + f"; reference threshold {ref['reference_threshold']}." if ref.get("agent_threshold") else "")
            st.info(f"**Reference comparison ({name}): not equivalent.**{th} Differs in: "
                    f"{', '.join(ref['differences']) or 'a result'}. This concerns {name} equivalence, not the "
                    "investigation's own validity.")
        else:
            st.info(f"**Reference comparison ({name}): not checked.** {ref.get('verdict', '')}")
    status = saved.summary.get("status")
    st.caption(f"Recorded run status `{status}`" + (
        " (set by the reference compatibility check; the scientific report itself verifies)"
        if status == "refused" and kind != "unverified" else "") + ".")


def render_overview(saved, ov):
    """The scientific overview: every value from the re-verified report or its accepted evidence."""
    st.markdown("#### Research question")
    st.markdown(f"> {ov['question']}")
    render_status_panels(saved, ov)
    st.markdown(f"### {ov['headline']}")
    st.markdown(ov["finding_text"])
    c, e, sp = ov["comparison"] or {}, ov["effect"], ov["spending"]
    m = st.columns(5 if sp["simulated_usd"] is not None else 4)
    m[0].metric(MAIN_DIFFERENCE, f"{_num(e['value'])} {e['unit']}".strip() if e else "not reported",
                help="The median, across sessions, of each session's paired difference (pre-bout window minus "
                     f"baseline). Contract v1: {c.get('mean_diff_definition')}")
    m[1].metric("Valid comparison windows", _num(c.get("n_windows")))
    m[2].metric("Contributing sessions", _num(c.get("n_sessions")))
    actual = "$0.00" if saved.kind != "live" else f"${sp['actual_usd']:.5f}" + ("" if sp["complete"] else " +unknown")
    m[3].metric("Actual API spending", actual, help=sp["basis"])
    if sp["simulated_usd"] is not None:
        m[4].metric(SIMULATED_COST, f"${sp['simulated_usd']:.5f}", help="Computed by the emulated endpoint; no "
                    "request reached a provider and nothing was spent.")
    if e:
        note("caption", f"Sign: {e['sign']}. Association only: no causal claim, no interval, no significance test.",
             f"Sign: Contract v1 defines each session's difference as pre-window minus baseline and reports their "
             f"median; a {e['sign']} value means the median session-paired `{e['feature']}` was "
             f"{'lower' if e['sign'] == 'negative' else 'higher' if e['sign'] == 'positive' else 'no different'} "
             "in the pre-bout window than in the baseline. Association only: no causal claim, no confidence "
             "interval, no significance test.")
    elif c.get("status") == "insufficient_evidence":
        st.caption(f"The comparison did not meet the evidence gate ({c.get('reason')}): {c.get('n_available')} "
                   f"available, {c.get('n_required')} required. No effect is reported.")
    if not ov["finished"]:
        st.caption(f"The investigation stopped ({ov['stopping_reason']}) before an accepted finding; the values above "
                   "come from the partial evidence and are not a finding.")
    render_compared(ov)


def plain_comparison(c, q):
    """A plain-language reading of the comparison: the investigation's own parameters put into Contract v1's window
    and effect definitions (the window offsets are only those definitions' sums). No result is interpreted."""
    out = []
    if q is not None:
        out.append(f"**Events:** the {_num(q['n_bouts'])} predicted `{q['behavior']}` bouts where the classifier's "
                   f"score stayed at or above {_num(q['threshold'])} for at least {_seconds(q['min_duration_s'])}; "
                   "each bout is timed from its start.")
    if c is None:
        return out
    pre, base, gap = c.get("pre_s"), c.get("baseline_s"), c.get("baseline_gap_s")
    unit = f", in {c['unit']}" if c["unit"] else ""
    out.append(f"**Measured:** `{c['feature']}` ({c['feature_meaning']}{unit}), averaged over each period.")
    if isinstance(pre, (int, float)):
        out.append(f"**Pre-event period:** the {_seconds(pre)} immediately before the bout starts.")
    if all(isinstance(x, (int, float)) for x in (pre, base, gap)):
        out.append(f"**Baseline period:** a {_seconds(base)} period from {_seconds(pre + gap + base)} to "
                   f"{_seconds(pre + gap)} before the bout starts"
                   + (" (directly before the pre-event period)." if gap == 0 else
                      f" (separated from the pre-event period by a {_seconds(gap)} gap)."))
    out.append("**Pre − baseline:** for each bout, the pre-event value minus the baseline value; each session "
               "averages its bouts' differences, and the main difference is the median of those session values. "
               "Negative means lower before the bout than at baseline, positive means higher. It describes an "
               "association, not a cause.")
    return out


def render_compared(ov):
    c, q = ov["comparison"], ov["query"]
    if c is None and q is None:
        return
    with st.expander("What was compared (Contract v1 definitions and the parameters this investigation used)",
                     expanded=True):
        plain = plain_comparison(c, q)
        if plain:
            st.markdown("**In plain terms**")
            for line in plain:
                st.markdown(f"- {line}")
            st.markdown("**Contract v1 definitions**")
        if q is not None:
            th = f"{_num(q['threshold'])}" + ("" if q["threshold_sent"] else " (not set by the agent: Contract v1's "
                                                                           "default applied)")
            st.markdown(f"- **Events:** predicted `{q['behavior']}` bouts from {q['data_source']} scores: score ≥ "
                        f"threshold {th}, lasting ≥ {_seconds(q['min_duration_s'])}; {q['n_bouts']} bouts in "
                        f"{q['n_sessions']} of {q['n_sessions_searched']} sessions searched.")
            st.caption(q["start_meaning"])
        if c is not None:
            w = c.get("windows") or {}
            st.markdown(f"- **Feature:** `{c['feature']}` ({c['feature_meaning']}{', ' + c['unit'] if c['unit'] else ''}).")
            st.markdown(f"- **Pre-bout window:** {w.get('pre', '—')}, with pre_s = {_seconds(c['pre_s'])}.")
            st.markdown(f"- **Baseline window:** {w.get('baseline', '—')}, with baseline_s = {_seconds(c['baseline_s'])} "
                        f"and baseline_gap_s = {_seconds(c['baseline_gap_s'])}.")
            st.markdown(f"- **Anchor:** {w.get('anchor', '—')}; {w.get('truncation', '')}.")
            gate = c.get("evidence_gate") or {}
            st.markdown(f"- **Policies (fixed by Contract v1):** quality `{c['quality_policy']}`, overlap "
                        f"`{c['overlap_policy']}`, evidence gate ≥ {gate.get('min_n_windows')} windows in ≥ "
                        f"{gate.get('min_n_sessions')} sessions. {c['n_used']} of {c['n_input']} bouts used "
                        f"({c['n_excluded']} excluded).")
            st.markdown(f"- **Effect definition:** {c['mean_diff_definition']}.")


# ---- UI-2: scientific evidence (charts of one accepted comparison's recorded evidence) ----

EVIDENCE_KEY = "live_agent_comparison_"           # + the saved investigation's name: its selected comparison
SIGN_COLORS = {"lower before the bout": "#4c78a8", "higher before the bout": "#f58518", "no difference": "#9d9d9d"}
VERDICT_BADGES = {"met": ":green-badge[met]", "not met": ":red-badge[not met]",
                  "not evaluated": ":gray-badge[not evaluated]", "not recorded": ":gray-badge[not recorded]"}


def _height(n, legend_rows=0):
    """The whole chart's height (Streamlit fits the axis title and the legend inside it): room for every row's
    label, so none is dropped as overlapping, plus the legend's rows."""
    return max(120, 32 * n + 60 + 22 * legend_rows)


def render_evidence(saved):
    """Charts of the recorded evidence of a VERIFIED investigation: one accepted comparison at a time (the primary
    one by default) with the query evidence whose bouts it used. Every value is read from the report
    (runs.evidence_view); nothing is recomputed and no interval or significance test is shown."""
    d = saved.doc
    st.markdown("#### Scientific evidence")
    note("caption", "The recorded evidence behind the finding. Nothing is recomputed.",
         "The recorded evidence behind the finding, read from the verified report: one accepted comparison at a "
         "time, with the predicted bouts it used. Nothing is recomputed; no interval or significance test is "
         "shown.")
    choices = runs.comparison_choices(d)
    if not choices:
        st.info("This investigation recorded no accepted comparison, so there is no comparison evidence to chart.")
        queries = [e for e in d["evidence"] if e["tool"] == "query_events"]
        if queries:
            st.caption("The predicted bouts of its last accepted query (no comparison used them):")
            render_bouts(runs.bout_view(queries[-1]))
        return
    ids = [c["evidence_id"] for c in choices]
    labels = {c["evidence_id"]: c["label"] for c in choices}
    key = EVIDENCE_KEY + saved.name
    if st.session_state.get(key) not in ids:                    # first view: the primary comparison, else the last
        st.session_state[key] = next((c["evidence_id"] for c in choices if c["primary"]), ids[-1])
    selected = st.selectbox("Comparison", ids, format_func=labels.__getitem__, key=key,
                            help="Each accepted compare_windows call separately; results are never combined.")
    v = runs.evidence_view(d, selected)
    tabs = st.tabs(["Session comparison", "Predicted bouts", "Exclusions", "Evidence gate"])
    with tabs[0]:
        render_sessions(v)
    with tabs[1]:
        if v["query"] is None:
            st.warning(f"The bouts of this comparison cannot be shown: {v['query_basis']}.")
        else:
            st.caption(f"Bouts of query evidence `{v['query']['evidence_id']}` ({v['query_basis']}); this "
                       f"comparison took {_num(v['n_input'])} of them as input.")
            render_bouts(v["query"])
    with tabs[2]:
        render_exclusions(v)
    with tabs[3]:
        render_gate(v)


def render_sessions(v):
    if v["status"] != "ok" or not v["sessions"]:
        why = f" ({v['reason']}: {v['reason_text']})" if v.get("reason") else ""
        note("info", f"No session-level effect is shown: this comparison's status is `{v['status']}`{why}, "
                     "so there is nothing to plot.",
             "Contract v1 records per-session differences only for a comparison that meets the evidence gate.")
        return
    unit = v["unit"]
    m = st.columns(2)
    m[0].metric("Contributing sessions", _num(v["n_sessions"]))
    m[1].metric("Reported mean_diff (median across sessions)", f"{_num(v['mean_diff'])} {unit}".strip())
    df = pd.DataFrame(v["sessions"])
    df["direction"] = ["lower before the bout" if x < 0 else "higher before the bout" if x > 0 else "no difference"
                       for x in df["paired_diff"]]
    title = f"{v['feature']}: pre − baseline" + (f" ({unit})" if unit else "")
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("paired_diff:Q", title=title),
        y=alt.Y("session:N", sort=None, title=None, axis=alt.Axis(labelLimit=170, labelOverlap=False)),
        color=alt.Color("direction:N", title=None, scale=alt.Scale(domain=list(SIGN_COLORS),
                                                                   range=list(SIGN_COLORS.values())),
                        legend=alt.Legend(orient="bottom", direction="vertical")),
        tooltip=[alt.Tooltip("full_session:N", title="session"), alt.Tooltip("n_bouts:Q", title="used bouts"),
                 alt.Tooltip("paired_diff:Q", title="session mean of pre − baseline", format=".4g")])
    zero = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(color="#666666", strokeWidth=1.5).encode(x="zero:Q")
    median = alt.Chart(pd.DataFrame({"mean_diff": [v["mean_diff"]]})).mark_rule(
        color="#222222", strokeDash=[6, 4], strokeWidth=2).encode(
        x="mean_diff:Q", tooltip=[alt.Tooltip("mean_diff:Q", title="mean_diff (median across sessions)",
                                              format=".4g")])
    st.altair_chart((bars + zero + median).properties(height=_height(len(df), len(SIGN_COLORS))), width="stretch")
    st.caption(f"Each bar is one session: the mean, over that session's used bouts, of the pre-bout window value "
               f"minus the baseline window value of `{v['feature']}` ({v['feature_meaning']}). The dashed line is "
               "the reported mean_diff, the median of these session values; the solid line is zero (no "
               "difference). Colour shows only the sign. Association only: no confidence interval or significance "
               "test.")
    with st.expander("Plotted values"):
        st.dataframe(df[["full_session", "n_bouts", "paired_diff"]].rename(columns={
            "full_session": "session", "n_bouts": "used bouts", "paired_diff": f"pre − baseline ({unit})"}),
            hide_index=True, width="stretch")


def render_bouts(q):
    st.caption(f"Classifier-predicted `{q['behavior']}` bouts: threshold crossings of the {q['data_source']} "
               "scores, not annotated behavior.")
    m = st.columns(4)
    m[0].metric("Predicted bouts", _num(q["n_bouts"]))
    m[1].metric("Sessions with bouts", f"{_num(q['n_sessions'])} of {_num(q['n_sessions_searched'])}")
    m[2].metric("Applied threshold", _num(q["threshold"]),
                help=None if q["threshold_sent"] else "Not set by the investigation: Contract v1's default applied.")
    m[3].metric("Minimum duration", _seconds(q["min_duration_s"]))
    per = pd.DataFrame(q["per_session"])
    if not per.empty:
        st.altair_chart(alt.Chart(per).mark_bar(color="#4c78a8").encode(
            x=alt.X("bouts:Q", title="predicted bouts", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("session:N", sort=None, title=None, axis=alt.Axis(labelLimit=170, labelOverlap=False)),
            tooltip=["session:N", "bouts:Q"]).properties(height=_height(len(per))), width="stretch")
    rows = pd.DataFrame(q["bouts"])
    if q["durations_recorded"]:
        durations = rows.dropna(subset=["duration_s"])
        hist = alt.Chart(durations).mark_bar(color="#72b7b2").encode(
            x=alt.X("duration_s:Q", bin=alt.Bin(maxbins=20), title="bout duration, end_s − start_s (s)"),
            y=alt.Y("count():Q", title="predicted bouts"))
        floor = alt.Chart(pd.DataFrame({"min_duration_s": [q["min_duration_s"]]})).mark_rule(
            color="#222222", strokeDash=[6, 4]).encode(x="min_duration_s:Q")
        st.altair_chart((hist + floor).properties(height=200), width="stretch")
        st.caption(f"Durations of the {q['durations_recorded']} bouts with a recorded end_s; the dashed line is the "
                   "minimum duration.")
    elif not rows.empty:
        st.info("No bout has a recorded end_s, so their durations are not available.")
    st.caption(f"{q['start_meaning']}. {q['fired_meaning']}")
    if not rows.empty:
        with st.expander(f"All {len(rows)} predicted bouts"):
            st.dataframe(rows, hide_index=True, width="stretch")


def render_exclusions(v):
    m = st.columns(3)
    m[0].metric("Input bouts", _num(v["n_input"]))
    m[1].metric("Used", _num(v["n_used"]))
    m[2].metric("Excluded", _num(v["n_excluded"]))
    if not v["exclusions_consistent"]:
        st.warning("The recorded exclusion counts do not add up; they are shown exactly as recorded.")
    rows = [{"category": "Used in the comparison", "bouts": v["n_used"], "kind": "used", "order": 0}]
    rows += [{"category": x["label"], "bouts": x["bouts"], "kind": "excluded", "order": i}
             for i, x in enumerate(v["exclusions"], 1) if x["bouts"]]
    df = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(df).mark_bar().encode(
        x=alt.X("bouts:Q", title="bouts", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("category:N", sort=alt.EncodingSortField("order"), title=None,
                axis=alt.Axis(labelLimit=200, labelOverlap=False)),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["used", "excluded"],
                                                              range=["#4c78a8", "#bab0ac"]), legend=None),
        tooltip=["category:N", "bouts:Q"]).properties(height=_height(len(df))), width="stretch")
    st.caption("The blue bar is the bouts the comparison used; each grey bar is one recorded exclusion reason. Each "
               "excluded bout is counted once, under the first reason Contract v1 checks, in this order:")
    st.dataframe(pd.DataFrame(v["exclusions"])[["reason", "description", "bouts"]].rename(
                 columns={"reason": "recorded reason (why)", "description": "meaning"}),
                 hide_index=True, width="stretch")


def render_gate(v):
    g = v["gate"]
    if v["status"] == "ok":
        st.success(f"Evidence gate met: {_num(v['n_windows'])} valid windows in {_num(v['n_sessions'])} sessions.")
    elif v["status"] == "insufficient_evidence":
        st.info(f"Evidence gate not met (`{v['failed_gate']}`; {v['reason']}: {v['reason_text']}). No effect is "
                "reported for this comparison.")
    if g["overridden"]:
        st.warning(f"Gate overridden: lowered from the Contract v1 defaults (≥ {g['default_min_n_windows']} windows, "
                   f"≥ {g['default_min_n_sessions']} sessions) to ≥ {g['min_n_windows']} windows and ≥ "
                   f"{g['min_n_sessions']} sessions. Recorded reason: “{g['override_reason']}”.")
    else:
        st.caption(f"The Contract v1 default gate, not overridden: ≥ {g['min_n_windows']} windows in ≥ "
                   f"{g['min_n_sessions']} sessions.")
    for row in v["gates"]:
        extra = (f" (recorded: {_num(row['n_available'])} available, {_num(row['n_required'])} required)"
                 if row["verdict"] == "not met" else
                 " (Contract v1 stops at the first failed gate)" if row["verdict"] == "not evaluated" else "")
        st.markdown(f"**{row['gate']}** (`{row['field']}`): observed {_num(row['observed'])}, required ≥ "
                    f"{_num(row['required'])} {VERDICT_BADGES[row['verdict']]}{extra}")
    df = pd.DataFrame(v["gates"])
    if df[["observed", "required"]].notna().all().all():
        base = alt.Chart(df).encode(y=alt.Y("gate:N", sort=None, title=None,
                                            axis=alt.Axis(labelLimit=170, labelOverlap=False)))
        observed = base.mark_bar(color="#4c78a8").encode(x=alt.X("observed:Q", title="count"),
                                                         tooltip=["gate:N", "observed:Q", "required:Q", "verdict:N"])
        required = base.mark_tick(color="#222222", thickness=3, size=30).encode(x="required:Q")
        st.altair_chart((observed + required).properties(height=_height(len(df))), width="stretch")
        st.caption("Bars: the recorded counts; black ticks: the required minimums.")


def render_timeline(steps, who):
    st.markdown("#### Recorded investigation activity")
    st.caption(f"{who['label'].capitalize()}. The saved trace, in order (a record of a finished run, not a live "
               f"progress stream). {who['intro']}")             # who chose the actions stays visible: no More info
    for s in steps:
        with st.container(border=True):
            left, right = st.columns([1, 14])
            left.markdown(f"{ICONS.get(s['status'], '')}  \n**{s.get('number', '')}**")
            with right:
                tool = f" · `{s['tool']}`" if s.get("tool") else ""
                st.markdown(f"**{s['title']}**{tool} {BADGES.get(s['status'], '')}")
                st.caption(s["description"])
                if s.get("parameters"):
                    st.markdown(f"{who['chooser']}: " + " · ".join(f"`{k}` = {v}" for k, v in s["parameters"]))
                if s.get("results"):
                    st.markdown(" · ".join(f"{k}: **{v}**" for k, v in s["results"]))
                if s.get("supporting"):
                    st.markdown("Supporting evidence: " + " · ".join(
                        f"`{x['field']}` = **{_num(x['value']) if isinstance(x['value'], float) else x['value']}**"
                        + (f" (step {x['step']})" if x["step"] else "") for x in s["supporting"]))
                with st.expander("Technical details"):
                    if s.get("model_note"):
                        st.markdown(f"Recorded reply text: {s['model_note']}")
                    st.json({k: s.get(k) for k in ("trace_step", "tool", "arguments", "evidence_id", "details",
                                                   "supporting") if s.get(k) is not None}, expanded=False)


def _path_value(x, digits=4):
    return "—" if x is None else _num(x, digits)


def evidence_path(saved, ov):
    """The verified report's chain, question to outcome, as (title, text, anchor) lines: values read from the
    verified report only (runs.overview and runs.evidence_view of its primary or last comparison)."""
    d = saved.doc
    calls = [t for t in d["trace"] if t["kind"] in ("tool_call", "finish")]
    refused = sum(1 for t in calls if t.get("outcome") == "refused")
    lines = [("Question", f"“{d['question']}”", "research-question"),
             ("Agent's calls", f"{len(calls)} recorded tool calls ({refused} refused), chosen during "
                               f"{d['usage']['model_turns']} turns", "recorded-investigation-activity")]
    q, c = ov.get("query"), ov.get("comparison")
    if q:
        lines.append(("Predicted events", f"{_path_value(q['n_bouts'])} classifier-predicted `{q['behavior']}` bouts "
                                          f"in {_path_value(q['n_sessions'])} sessions (threshold "
                                          f"{_path_value(q['threshold'])})", "scientific-evidence"))
    if c:
        comps = [e for e in d["evidence"] if e["tool"] == "compare_windows"]     # as runs.overview chooses
        primary = next((e for e in comps if e["evidence_id"] == d["primary_evidence_id"]), comps[-1])
        v = runs.evidence_view(d, primary["evidence_id"])
        lines += [
            ("Windows compared", f"`{c['feature']}` in the {_seconds(c['pre_s'])} before each bout vs a "
                                 f"{_seconds(c['baseline_s'])} baseline ({_seconds(c['baseline_gap_s'])} gap)",
             "research-question"),
            ("Evidence used", f"{_path_value(v['n_used'])} of {_path_value(v['n_input'])} bouts used, "
                              f"{_path_value(v['n_excluded'])} excluded (Exclusions tab)", "scientific-evidence"),
            ("Per-session differences", f"{len(v['sessions'])} sessions (Session comparison tab)"
             if v["sessions"] else "none recorded (the comparison did not meet the evidence gate)",
             "scientific-evidence"),
            ("Aggregate", f"median of session differences {_path_value(v['mean_diff'])} {v['unit']}".rstrip()
             if v["mean_diff"] is not None else "not reported", "scientific-evidence"),
            ("Evidence gate", ", ".join(f"{g['gate'].lower()} {g['verdict']}" for g in v["gates"]),
             "scientific-evidence")]
    lines.append(("Outcome", f"{ov['headline']} (`{d['finding']}`)" if d["finished"] else
                  f"stopped before a finding (`{d['stopping_reason']}`)", "technical-details"))
    return lines


def render_evidence_path(saved, ov):
    with st.container(border=True):
        st.markdown("#### How this result was reached")
        note("caption", "Each step in one line, from the verified report; follow a link for its details.",
             "Limitations, provenance, the reference check and costs are in Technical details.")
        st.markdown("\n".join(f"{i}. **{title}:** {text} · [details](#{anchor})"
                               for i, (title, text, anchor) in enumerate(evidence_path(saved, ov), 1)))


def render_saved(saved):
    if not saved.verified:
        st.error("This saved investigation does NOT verify, so none of its findings, evidence or costs are shown: "
                 + "; ".join(map(str, saved.problems[:5])))
        note("caption", "Nothing from its files is trusted. Other saved investigations are unaffected.",
             "Its files may have been changed, moved or only partly written. A new run always creates a new "
             "directory, so nothing here is repaired or overwritten.")
        return
    d = saved.doc
    st.markdown(f"**Showing `{saved.name}`** ({runs.KIND_LABELS.get(saved.kind, saved.kind)}); everything below is "
                "this investigation's saved, re-verified files.")
    st.markdown(READABLE, unsafe_allow_html=True)
    ref, who = runs.reference_check(saved), runs.attribution(saved)
    ov = runs.overview(saved, ref)
    render_overview(saved, ov)
    render_evidence_path(saved, ov)
    render_evidence(saved)
    render_timeline(runs.timeline(d, who["replies"]), who)
    st.markdown("#### Technical details")
    tabs = st.tabs(["Finding", "Tool sequence", "Evidence", "Cost ledger", "Reference check (run5)",
                    "Provenance & limitations", "Report"])
    with tabs[0]:
        st.markdown(f"**Finding:** `{d['finding']}`. {av1.FINDING_TEXT[d['finding']]}")
        if d["primary_evidence_id"]:
            st.caption(f"Primary evidence `{d['primary_evidence_id']}`.")
        if d["observations"]:
            st.markdown("**Observations** (values read by the investigation from the cited results; no number was "
                        "written by the investigator)")
            st.dataframe(pd.DataFrame(runs.observation_rows(d)), hide_index=True, width="stretch")
        st.caption("Association only: no causal claim, no confidence interval, validation data only. Publication: "
                   f"{d['publication']}.")
    with tabs[1]:
        st.caption(f"Every recorded turn ({who['label']}) and every call it asked for, including refused ones.")
        st.dataframe(pd.DataFrame(runs.trace_rows(d)), hide_index=True, width="stretch")
    with tabs[2]:
        st.dataframe(pd.DataFrame(runs.evidence_rows(d)), hide_index=True, width="stretch")
        for e in d["evidence"]:
            with st.expander(f"{e['tool']} {e['evidence_id']} ({e['status']})"):
                st.json({"request": e["request"], "result": e["result"], "provenance": e["provenance"]},
                        expanded=False)
    with tabs[3]:
        render_cost(saved)
    with tabs[4]:
        render_reference(saved, ref)
    with tabs[5]:
        st.dataframe(pd.DataFrame(runs.provenance_rows(saved)), hide_index=True, width="stretch")
        st.markdown("**Limitations** (the report's canonical disclosures)")
        for x in av1.DISCLOSURES:
            st.markdown(f"- {x}")
    with tabs[6]:
        st.download_button("Download the verified report (Markdown)", saved.markdown,
                           file_name=f"{saved.name}_investigation_report.md", mime="text/markdown",
                           key=f"live_agent_md_{saved.name}")
        st.download_button("Download the verified report (JSON)", json.dumps(d, indent=2, sort_keys=True) + "\n",
                           file_name=f"{saved.name}_investigation_report.json", mime="application/json",
                           key=f"live_agent_json_{saved.name}")
        st.markdown(saved.markdown)


TYPE_LABELS = {"paid_model": "paid investigation", "emulated": "free rehearsal",
               "offline": "offline deterministic run", "unknown": "investigation"}


def summary_shape_problem(path):
    """Why run.json does not have the JSON shape runs.load_saved dereferences before any verifier runs (an object;
    its "model" an object when set; an offline run's "outputs" an object when set), or None. Shapes only: no value
    read here is trusted or shown. An unreadable file or invalid JSON is left to load_saved, which reports it as not
    verifying; a Live Agent run's own fields are contained by its verifier (verify_saved)."""
    try:
        s = json.loads((Path(path) / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(s, dict):
        return "run.json is not a JSON object"
    if "key_policy" in s and "cost" in s:
        return None
    model = s.get("model") or {}
    if not isinstance(model, dict):
        return "run.json's model is not a JSON object"
    if model.get("adapter") == runs.script("investigation_v1_validation").OFFLINE_MODEL and \
            not isinstance(s.get("outputs") or {}, dict):
        return "run.json's outputs are not a JSON object"
    return None


def load_verified(path):
    """(SavedInvestigation or None, why its saved data is malformed or None), always freshly from disk. Only the
    explicit shape check above classifies data as malformed; any exception from runs.load_saved propagates."""
    problem = summary_shape_problem(path)
    if problem:
        return None, problem
    return runs.load_saved(path), None


def catalog_entry(path):
    """What the selector may say about one saved run, re-verified NOW (runs.load_saved; no cache): only a run that
    verifies is described by its content (type, outcome, question); any other is shown by its directory name only."""
    return describe(load_verified(path)[0])


def describe(saved):
    """catalog_entry's description of one loaded SavedInvestigation (or None)."""
    if saved is not None and saved.verified:
        d = saved.doc
        return {"verified": True, "type": TYPE_LABELS[runs.attribution(saved)["mode"]],
                "finished": d["finished"], "question": d["question"],
                "outcome": (runs.FINDING_HEADLINES.get(d["finding"], d["finding"]) if d["finished"] else
                            f"incomplete: stopped before a finding ({d['stopping_reason']})")}
    return {"verified": False, "type": None, "finished": None, "question": None,
            "outcome": "does NOT verify: nothing of it is shown"}


def saved_catalog():
    """The saved runs, most recently saved first by the file system's modification time of run.json (never by the
    order of run names, whose random suffixes say nothing about time)."""
    out = []
    for label, path in runs.list_saved():
        try:
            saved_ns = (path / "run.json").stat().st_mtime_ns
        except OSError:
            saved_ns = 0
        out.append({"path": str(path), "name": path.name, "group": label.split(":")[0], "saved_ns": saved_ns,
                    **catalog_entry(path)})
    return sorted(out, key=lambda e: e["saved_ns"], reverse=True)


def catalog_badges(e):
    """The selected run's state as badges, from its fresh verification (catalog_entry): content only if it verifies."""
    if not e["verified"]:
        return ":red-badge[does NOT verify] nothing of it is shown"
    state = ":green-badge[verified · finding]" if e["finished"] else ":orange-badge[verified · incomplete]"
    return f"{state} :gray-badge[{e['type']}]"


def catalog_label(e):
    when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(e["saved_ns"] / 1e9)) if e["saved_ns"] else "?"
    if not e["verified"]:
        return f"{when} · {e['name']} · does NOT verify"
    q = e["question"] if len(e["question"]) <= 70 else e["question"][:67] + "..."
    return f"{when} · {e['type']} · {e['outcome']} · “{q}”"


def render_open():
    """Reopen a saved investigation (offline: reload and re-verify; no network, no model)."""
    entries = saved_catalog()
    if not entries:
        st.info("No saved investigation yet. Start one in New investigation: a free rehearsal needs no key and "
                "costs nothing. Each finished run appears here, reloaded and re-verified before it is shown.")
        return
    label_of = {e["path"]: catalog_label(e) for e in entries}
    by_path = {e["path"]: e for e in entries}
    want = st.session_state.pop(OPEN_KEY, None)               # a just-completed run: selected once, explicitly
    if want in label_of:
        st.session_state[SAVED_KEY] = want
    elif st.session_state.get(SAVED_KEY) not in label_of:     # first view, or the selected run is gone
        st.session_state[SAVED_KEY] = next(iter(label_of))
    counts = {"with a finding": sum(1 for e in entries if e["verified"] and e["finished"]),
              "incomplete": sum(1 for e in entries if e["verified"] and not e["finished"]),
              "not verifying": sum(1 for e in entries if not e["verified"])}
    st.caption(f"{len(entries)} saved investigation{'s' if len(entries) != 1 else ''}, most recently saved first: "
               + ", ".join(f"{n} {k}" for k, n in counts.items() if n) + ". Each is reloaded and re-verified when "
               "you open it.")
    path = st.selectbox("Saved investigation", list(label_of), format_func=label_of.__getitem__, key=SAVED_KEY)
    e = by_path[path]
    saved, failure = load_verified(path)
    st.markdown(catalog_badges(describe(saved)))                # from this load, the one shown below
    st.caption(f"Run `{e['name']}` · {e['group']} · files in {Path(path).parent}")
    if saved is None:
        st.error(f"This investigation could not be loaded or verified ({failure}): its saved files are unreadable "
                 "or malformed. Nothing from it is shown, and nothing was repaired, changed or deleted.")
        return
    render_saved(saved)
