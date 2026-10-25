"""The "Run schedule" part of the Rules page: rendering only (new file).

Every state change is in schedule_workflow.py (tested without a browser). The workflow lives in
st.session_state[WORKFLOW_KEY]. As with rules, a Confirm click is handled after the inputs are
compared with the ones the proposal was made from, so an edit and a Confirm clicked together never
approve the old proposal.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo, available_timezones

import streamlit as st

from . import schedule_workflow as sw

WORKFLOW_KEY = "schedule_workflow"
MODE_HOURS, MODE_WINDOW = "For N hours", "From - To"
START_NOW, START_AT = "Now", "At a chosen time"
DEFAULT_TZ = "America/New_York"


def _workflow():
    if not isinstance(st.session_state.get(WORKFLOW_KEY), sw.ScheduleWorkflow):
        st.session_state[WORKFLOW_KEY] = sw.ScheduleWorkflow()
    return st.session_state[WORKFLOW_KEY]


def render():
    wf = _workflow()
    st.subheader("Run schedule")
    st.caption("Choose how long the monitor should run. A schedule is only a proposal until you click Confirm.")

    mode = st.radio("How long should it run?", [MODE_HOURS, MODE_WINDOW], horizontal=True, key="schedule_mode")
    zones = sorted(available_timezones())
    tz = st.selectbox("Time zone of the lab", zones, index=zones.index(DEFAULT_TZ) if DEFAULT_TZ in zones else 0,
                      key="schedule_tz")

    if mode == MODE_HOURS:
        h_col, m_col = st.columns(2)
        whole_hours = h_col.number_input("Hours", min_value=0, max_value=168, value=8, step=1, key="schedule_hours")
        minutes = m_col.number_input("Minutes", min_value=0, max_value=59, value=0, step=1, key="schedule_minutes")
        hours = whole_hours + minutes / 60
        start_choice = st.radio("Start", [START_NOW, START_AT], horizontal=True, key="schedule_start_choice")
        start_day = start_clock = None
        if start_choice == START_AT:
            c1, c2 = st.columns(2)
            start_day = c1.date_input("Start date", value=date.today(), key="schedule_start_day")
            start_clock = c2.time_input("Start time", value=time(18, 0), key="schedule_start_clock")
        signature = (mode, tz, hours, start_choice, start_day, start_clock)
    else:
        c1, c2 = st.columns(2)
        from_clock = c1.time_input("From", value=time(18, 0), key="schedule_from")
        to_clock = c2.time_input("To", value=time(6, 0), key="schedule_to")
        signature = (mode, tz, from_clock, to_clock)

    wf.sync(signature)                                    # a changed input drops the pending proposal first

    if st.button("Propose schedule", type="primary", key="schedule_propose"):
        if mode == MODE_HOURS:
            start = None
            if start_choice == START_AT:
                start = datetime.combine(start_day, start_clock).replace(tzinfo=ZoneInfo(tz))
            wf.propose_for_hours(hours, tz, start=start, signature=signature)
        else:
            wf.propose_between(from_clock, to_clock, tz, signature=signature)

    if wf.error:
        st.error(f"Not proposed: {wf.error}.")

    pending = wf.pending
    if pending is not None:
        s = pending.schedule
        st.markdown(f"#### Proposed schedule {s.schedule_id}")
        st.warning("Status: proposed. Not active until you click Confirm.")
        st.markdown(f"- **From:** {s.start:%A %d %B %Y, %H:%M}\n"
                    f"- **To:** {s.end:%A %d %B %Y, %H:%M}\n"
                    f"- **Length:** {s.length_text()}\n"
                    f"- **Time zone:** {s.tz}")
        col_confirm, col_reject = st.columns(2)
        confirm = col_confirm.button("Confirm", key=f"schedule_confirm_{pending.token}", type="primary")
        reject = col_reject.button("Reject", key=f"schedule_reject_{pending.token}")
        try:
            if confirm:
                wf.confirm(pending.token)
                st.rerun()
            elif reject:
                wf.reject(pending.token)
                st.rerun()
        except sw.DecisionError as exc:
            st.error(str(exc))
        except sw.StaleActionError as exc:
            st.error(f"Ignored: {exc}.")

    st.markdown("#### Schedules in this session")
    rows = wf.table()
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("No schedule decided yet.")
    st.caption("A confirmed schedule is saved as a file for the monitor. "
               "Starting and stopping a run from it is not connected yet.")
