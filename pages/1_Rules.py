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

import streamlit as st

from live_tracker.agent import Clarification, RuleProposal
from live_tracker import schedule_ui
from live_tracker.claude_agent import ClaudeUnavailable
from live_tracker.rule_workflow import (BackendFailure, DecisionError, DuplicateRuleError, RuleWorkflow,
                                        StaleActionError, is_valid_workflow, preview)

WORKFLOW_KEY = "rule_workflow"
TEXT_KEY = "rule_instruction"
BACKEND_LABELS = {
    "Offline parser (default, no API key, no cost)": "offline",
    "Offline parser + Claude cross-check (paid API call)": "claude",
}

# a missing or malformed stored value is replaced, never read as rules
recovered = WORKFLOW_KEY in st.session_state and not is_valid_workflow(st.session_state[WORKFLOW_KEY])
if recovered or WORKFLOW_KEY not in st.session_state:
    st.session_state[WORKFLOW_KEY] = RuleWorkflow()
wf: RuleWorkflow = st.session_state[WORKFLOW_KEY]

st.title("Monitoring rules")
if recovered:
    st.error("The stored rule session was invalid and has been reset. No rules are approved; "
             "please enter and confirm your rules again.")
st.caption("Type what you want to monitor. A rule is only a proposal until you click Confirm; "
           "only confirmed rules are used for monitoring.")

backend_label = st.radio("Interpretation", list(BACKEND_LABELS), index=0)
backend = BACKEND_LABELS[backend_label]
if backend == "claude":
    st.info("The Claude cross-check only accepts instructions that the offline parser already "
            "understands; it can block a proposal but never interpret an instruction the offline "
            "parser rejects. It needs ANTHROPIC_API_KEY and LIVE_TRACKER_CLAUDE_MODEL and makes one "
            "paid API call per click on Propose rule.")

text = st.text_area("Instruction", key=TEXT_KEY,
                    placeholder="Alert me when the mice attack for at least 2 seconds")

# 1. a changed instruction invalidates the pending proposal before any click is handled
wf.sync_instruction(text)

# 2. new proposal (the only place a parser / the API is called)
if st.button("Propose rule", type="primary"):
    with st.spinner("Interpreting..."):
        wf.submit(text, backend=backend)

if wf.notice:
    st.info(wf.notice)

# 3. the result of the last submission
result = wf.last_result
if isinstance(result, ClaudeUnavailable):
    st.error(f"Claude cross-check unavailable ({result.reason}): {result.message}")
    st.caption("Nothing was proposed. Choose the offline parser to continue without Claude.")
elif isinstance(result, BackendFailure):
    st.error(result.message)
elif isinstance(result, Clarification):
    st.warning("This instruction cannot be turned into a rule yet. Nothing was proposed.")
    for issue in result.issues:
        st.markdown(f"- **{issue.code}**: {issue.message}")

pending = wf.pending
if pending is not None and isinstance(pending.proposal, RuleProposal):
    p = preview(pending)
    st.subheader(f"Proposed rule {p['rule_id']}")
    st.warning(f"Status: {p['status']}.")
    st.markdown(
        f"- **Behavior:** {p['behavior']}\n"
        f"- **Minimum duration:** {p['min_duration']}\n"
        f"- **Alert mode:** {p['alert_mode']}\n"
        f"- **Alert threshold:** {p['alert_threshold']}\n"
        f"- **Ask threshold:** {p['ask_threshold']}\n"
        f"- **Interpreted by:** {'offline parser + Claude cross-check' if p['backend'] == 'claude' else 'offline parser'}"
    )
    if p["uses_default_thresholds"]:
        st.warning("Untuned default thresholds: " + ", ".join(p["uses_default_thresholds"])
                   + ". These are example values from rules.py, not chosen on data.")
    st.markdown("**Assumptions to check before confirming:**")
    for a in p["assumptions"]:
        st.markdown(f"- {a}")
    st.markdown("**Your instruction:**")
    st.text(p["instruction"])
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

# 4. all rules of this session
st.subheader("Rules in this session")
rows = wf.rules_table()
if rows:
    st.dataframe(rows, width="stretch", hide_index=True)
else:
    st.caption("No rules decided yet.")
if wf.monitoring_allowed():
    n = len(wf.ruleset.active())
    st.success(f"Monitoring can start: {n} approved rule{'s' if n != 1 else ''}.")
else:
    st.warning("Monitoring cannot start: no rule is approved yet.")

schedule_ui.render()
