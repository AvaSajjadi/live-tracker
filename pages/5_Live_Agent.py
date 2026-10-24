"""Live Agent page: the Stage 9 autonomous investigation agent on the verified CalMS21 validation inputs (new file).

A researcher asks a question; the agent (Claude through the guarded OpenRouter adapter) chooses every Contract v1
call; the saved report is re-verified before it is shown. Rendering: live_tracker.investigation_v1_ui; logic:
live_tracker.investigation_v1_runs (the reviewed CLI underneath). Paid requests happen only after a ready preflight,
an explicit confirmation and the typed authorization text.
"""

import streamlit as st

from live_tracker import investigation_v1_ui as ui

st.title("Live Agent")
st.caption("An autonomous, model-planned investigation over the Contract v1 tools on the verified CalMS21 "
           "validation split. Every value shown is read from a verified tool result; association only; publication "
           "not authorized.")
message = st.session_state.pop(ui.MESSAGE_KEY, None)
if message:
    (st.error if message[0] == "error" else st.info)(message[1])
tab_new, tab_saved = st.tabs(["New investigation", "Saved investigations"])
with tab_new:
    ui.render_new()
with tab_saved:
    ui.render_open()
