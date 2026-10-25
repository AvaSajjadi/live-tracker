"""Rules page: researcher sentence -> proposed rule -> Confirm / Reject (new file).

Rendering only; every state change is in live_tracker.rule_workflow
(tested without a browser). The workflow, and with it the session's RuleSet,
lives in st.session_state[WORKFLOW_KEY] so it survives reruns; the replay
page will read the approved rules from there (RuleSet.active()).

Order on every rerun matters: the text box is compared with the submitted
instruction BEFORE any button is handled, so a Confirm clicked together with
an edit is never applied to the old proposal. Buttons are handled inline (no
on_click callbacks, which would run before that check), and their keys carry
the proposal's token, so a stale or repeated click cannot act twice.
"""

import json
import os

import streamlit as st

from live_tracker.agent import Clarification, RuleProposal
from live_tracker.ui_style import apply as apply_style
from live_tracker.ui_style import note
from live_tracker import flow, schedule_ui
from live_tracker.claude_agent import ClaudeUnavailable
from live_tracker.rule_workflow import (BackendFailure, DecisionError, DuplicateRuleError, RuleWorkflow,
                                        StaleActionError, is_valid_workflow, preview)

WORKFLOW_KEY = "rule_workflow"
TEXT_KEY = "rule_instruction"
BACKEND_LABELS = {
    "Smart reading (default, no API key, no cost)": "offline",
    "Offline parser + Claude cross-check (paid API call)": "claude",
    "Claude reads the sentence (stronger with loose wording, paid API call)": "claude_interpret",
}
INTERPRETED_BY = {"offline": "smart reading (offline, free)", "claude": "offline parser + Claude cross-check",
                  "claude_interpret": "Claude (paid), checked by code"}

# a missing or malformed stored value is replaced, never read as rules
recovered = WORKFLOW_KEY in st.session_state and not is_valid_workflow(st.session_state[WORKFLOW_KEY])
if recovered or WORKFLOW_KEY not in st.session_state:
    st.session_state[WORKFLOW_KEY] = RuleWorkflow()
wf: RuleWorkflow = st.session_state[WORKFLOW_KEY]

apply_style()
flow.render_bar("rules")
st.title("Monitoring rules")
if recovered:
    st.error("The stored rule session was invalid and has been reset. No rules are approved; "
             "please enter and confirm your rules again.")
st.caption("Write what you want to monitor in your own words. A rule is only a proposal until you click Confirm; "
           "only confirmed rules are used for monitoring.")

backend_label = st.radio("Interpretation", list(BACKEND_LABELS), index=0)
backend = BACKEND_LABELS[backend_label]
if backend == "claude":
    note("info", "Claude cross-check: needs an API key and makes one paid call per click on Propose rule.",
         "The Claude cross-check only accepts instructions that the offline parser already "
         "understands; it can block a proposal but never interpret an instruction the offline "
         "parser rejects. It needs ANTHROPIC_API_KEY and LIVE_TRACKER_CLAUDE_MODEL and makes one "
         "paid API call per click on Propose rule.")

if backend == "claude_interpret":
    note("info", "Claude reads the sentence: it copes with typos and loose wording. Needs an API key; one paid call per click.",
         "Claude fills a strict form and code checks every number against your own words. Anything a rule cannot express "
         "(a place, a time of day, an extra condition) is refused, not dropped, and every guess is shown as an assumption "
         "before you confirm. It needs ANTHROPIC_API_KEY and LIVE_TRACKER_CLAUDE_MODEL. The offline parser stays the default.")

def why_text():
    """The reasons and assumptions behind the last answer, for the hover help (?) next to the instruction box."""
    lines = []
    if isinstance(wf.last_result, Clarification):
        lines += [f"- {issue.message}" for issue in wf.last_result.issues]
    if wf.pending is not None and isinstance(wf.pending.proposal, RuleProposal):
        info = preview(wf.pending)
        if info["uses_default_thresholds"]:
            lines.append("- Untuned default thresholds: " + ", ".join(info["uses_default_thresholds"])
                         + ". These are example values from rules.py, not chosen on data.")
        lines += [f"- {a}" for a in info["assumptions"]]
    return "\n".join(lines) or None


text = st.text_area("Instruction", key=TEXT_KEY, help=why_text(),
                    placeholder="For example: let me know if the mice fight for more than 2 seconds")

# 1. a changed instruction invalidates the pending proposal before any click is handled
wf.sync_instruction(text)

# 2. new proposal (the only place a parser / the API is called)
if st.button("Propose rule", type="primary"):
    with st.spinner("Interpreting..."):
        wf.submit(text, backend=backend)
    st.rerun()                                       # so the (?) next to the box shows the new reasons

if wf.notice:
    st.info(wf.notice)

# 3. the result of the last submission
result = wf.last_result
if isinstance(result, ClaudeUnavailable):
    st.error(f"{'Claude' if wf.last_backend == 'claude_interpret' else 'Claude cross-check'} unavailable ({result.reason}): {result.message}")
    st.caption("Nothing was proposed. Choose the offline parser to continue without Claude.")
elif isinstance(result, BackendFailure):
    st.error(result.message)
elif isinstance(result, Clarification):
    st.warning("This instruction cannot be turned into a rule yet. Nothing was proposed.")
    if wf.last_backend == "offline":
        st.caption("Try other words, and say how long the behavior must last, in seconds. For example: "
                   "\"Let me know if the mice fight for at least 2 seconds\".")
        if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("LIVE_TRACKER_CLAUDE_MODEL"):
            if st.button("Ask Claude to read this sentence (paid call)", key="ask_claude"):
                with st.spinner("Claude is reading..."):
                    wf.submit(text, backend="claude_interpret")
                st.rerun()
        else:
            st.caption("Claude can read looser wording, but needs an API key. See docs/claude_interpreter.md.")

pending = wf.pending
if pending is not None and isinstance(pending.proposal, RuleProposal):
    p = preview(pending)
    with st.expander("Rule JSON"):
        st.code(json.dumps(p["rule_json"], indent=2), language="json")

    col_confirm, col_reject = st.columns(2)
    confirm = col_confirm.button("Confirm", key=f"confirm_{pending.token}", type="primary")
    reject = col_reject.button("Reject", key=f"reject_{pending.token}")
    try:
        if confirm:
            wf.confirm(pending.token)
            st.rerun()
        elif reject:
            wf.reject(pending.token)
            st.rerun()
    except DuplicateRuleError as exc:
        st.error(f"Not approved: {exc}.")
    except DecisionError as exc:  # nothing changed; the proposal is still pending
        st.error(str(exc))
    except StaleActionError as exc:
        st.error(f"Ignored: {exc}.")

schedule_ui.render()

st.divider()
flow.render_nav("rules", can_next=wf.monitoring_allowed(),
                next_hint="Confirm a rule first, then press Next.")
