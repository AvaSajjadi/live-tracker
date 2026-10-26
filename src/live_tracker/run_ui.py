"""The Run page: start a run, watch it, answer its questions, read its report (new file).

Rendering only. What is done to a run is in run_control.py (tested without a browser); the run
itself is the separate program monitor_runner, so nothing here has to stay open for it to go on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from .ui_style import note, color_actions, event_notice

from . import monitor_runner as mr
from . import run_control as rc
from . import run_schedule as rs
from . import schedule_workflow as sw
from .rule_workflow import is_valid_workflow

RULE_WORKFLOW_KEY = "rule_workflow"            # set by the Rules page
SCHEDULE_WORKFLOW_KEY = "schedule_workflow"    # set by the Rules page, run schedule part
SELECTED_KEY = "run_selected"
STATE_TEXT = {"waiting": ":orange[Waiting for the start time]", "ready": ":orange[Ready, waiting for GO]", "running": ":green[Running]",
              "finishing": ":orange[Writing the report]", "finished": ":green[Finished]", "failed": ":red[Failed]"}
OUTCOME_TEXT = {"complete": "complete", "partial": "partial: part of the planned time is missing",
                "stopped": "stopped early", "no_data": "no data", "failed": "failed"}


def _clock(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def usable_schedules():
    """Approved schedules that have not ended: the ones confirmed in this session and the saved files."""
    found, now = {}, datetime.now(timezone.utc)
    wf = st.session_state.get(SCHEDULE_WORKFLOW_KEY)
    candidates = list(wf.approved()) if isinstance(wf, sw.ScheduleWorkflow) else []
    folder = sw.save_dir()
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                candidates.append(rs.RunSchedule.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, rs.ScheduleError):
                continue
    for s in candidates:
        if s.status == "approved" and s.status_at(now) != "finished":
            found[(s.schedule_id, s.start.isoformat())] = s
    return sorted(found.values(), key=lambda s: s.start)


def approved_rules():
    wf = st.session_state.get(RULE_WORKFLOW_KEY)
    return list(wf.ruleset.active()) if is_valid_workflow(wf) else []


def render():
    st.title("Run")
    st.caption("A run happens in a separate program on the lab computer. Closing this page does not stop it.")
    render_start()
    st.divider()
    render_runs()


# ---- start ----

def render_start():
    st.subheader("Start a run")
    rules, schedules = approved_rules(), usable_schedules()
    missing = [label for label, ready in (("rule", rules), ("run schedule", schedules)) if not ready]
    if missing:
        st.warning("Before starting, confirm a " + " and a ".join(missing) + " on the Rules page.")

    sources = rc.available_sources()
    enabled = [s for s in sources if s["enabled"]]
    chosen_schedule = st.selectbox("Schedule", schedules, format_func=lambda s: f"{s.schedule_id}: {s.describe()}",
                                   key="run_schedule_choice") if schedules else None
    source = st.radio("Frames come from", enabled, format_func=lambda s: s["label"], key="run_source_choice")
    with st.expander("More info"):
        st.caption("Recorded keypoints are scored by Model B; camera image inference is unavailable.")
        for s in sources:
            if not s["enabled"]:
                st.caption(f"{s['label']}: not available, {s['reason']}.")
    spec = {"kind": source["kind"]}
    if source["kind"] == "keypoints":
        sessions = rc.validation_sessions()
        spec.update(session=st.selectbox("Validation session", sessions, key="run_session_choice") if sessions else None,
                    json="data/raw/calms21/calms21_task1_train.json")
    speed = st.number_input("Demo speed (1 = real time)", min_value=1.0, max_value=600.0, value=1.0, step=1.0,
                            key="run_speed", help="Above 1 only plays a recording faster, for demos. At a very high speed "
                            "the time the computer needs counts as part of the night, so the run may end early.")

    can_start = bool(rules and chosen_schedule and (source["kind"] != "keypoints" or spec.get("session")))
    if st.button("Start run", type="primary", disabled=not can_start, key="run_start"):
        try:
            run_dir = rc.new_run_dir(chosen_schedule.schedule_id)
            cfg = rc.build_config(run_dir, chosen_schedule, rules, spec,
                                  model="models/model_b" if source["kind"] == "keypoints" else None, speed=speed)
            rc.start_run(cfg)
        except (mr.RunRefusedError, rs.ScheduleError, ValueError, OSError) as exc:
            st.error(f"Not started: {exc}.")
        else:
            st.session_state[SELECTED_KEY] = run_dir.name
            st.success(f"Started. The run is in {run_dir.name}. It appears in the list below.")


# ---- the runs ----

def render_runs():
    st.subheader("Runs")
    runs = rc.list_runs()
    if not runs:
        st.caption("No runs yet.")
        return
    names = [r["name"] for r in runs]
    default = names.index(st.session_state[SELECTED_KEY]) if st.session_state.get(SELECTED_KEY) in names else 0
    state_of = {r["name"]: (r["status"] or {}).get("state", "starting") for r in runs}
    name = st.selectbox("Run", names, index=default, format_func=lambda n: f"{n}: {state_of[n]}", key="run_pick")
    run_panel(next(r["path"] for r in runs if r["name"] == name))


@st.fragment(run_every=3)
def run_panel(run_dir):
    run_dir = Path(run_dir)
    status = rc.read_status(run_dir)
    if status is None:
        st.info("Waiting for the program to write its first status.")
        log = run_dir / "runner.log"
        if log.exists() and log.stat().st_size:
            with st.expander("Program output"):
                st.code(log.read_text(encoding="utf-8", errors="replace")[-2000:])
        return

    st.markdown(f"**{STATE_TEXT.get(status['state'], status['state'])}**"
                + (f" ({OUTCOME_TEXT.get(status.get('outcome'), status.get('outcome'))})" if status.get("outcome") else ""))
    if status.get("message"):
        note("error" if status["state"] == "failed" else "caption", status["message"])
    if rc.is_stale(status):
        if rc.run_is_over(run_dir, status):
            note("error", "The run program is no longer running and no report was written.",
                 "A power cut or error may have stopped it; see runner.log in the run folder. "
                 "Answers you give now go straight into the event log.")
        else:
            note("error", "The run program has not reported for a while and may have stopped.",
                 "Check runner.log and error.txt in the run folder for a power cut or error.")
    if status.get("interrupted_now"):
        st.warning("The video source is interrupted right now. The program keeps trying to reconnect.")
    if status.get("schedule"):
        st.caption(rs.RunSchedule.from_dict(status["schedule"]).describe())
    st.progress(rc.progress(status), text=f"{_clock(status.get('seconds_left', 0))} left")

    render_events(run_dir, status)
    with st.expander("Run details"):
        st.caption(f"Frames: {status.get('frames', 0)} · Alerts: {status.get('alerts', 0)} · "
                   f"Questions: {status.get('asks', 0)} · Unanswered: {status.get('unanswered', 0)} · "
                   f"Interruptions: {status.get('interruptions', 0)}")
        st.caption(f"Source: {status.get('source') or 'unknown'}. Last update: {status.get('heartbeat_at', '?')}.")

    if status["state"] in rc.ACTIVE_STATES:
        if rc.stop_requested(run_dir):
            st.caption("Stop requested. The program will finish and write the report.")
        elif st.button("Stop run", key=f"run_stop_{run_dir.name}"):
            rc.stop_run(run_dir)
            st.rerun()

    report = rc.read_report(run_dir)
    if report is not None:
        md, js = report
        with st.expander("Report", expanded=True):
            st.markdown(md)
        d1, d2 = st.columns(2)
        d1.download_button("Download report (Markdown)", md, file_name=f"{run_dir.name}_report.md", mime="text/markdown",
                           key=f"run_md_{run_dir.name}")
        d2.download_button("Download report (JSON)", js, file_name=f"{run_dir.name}_report.json", mime="application/json",
                           key=f"run_json_{run_dir.name}")
    log = run_dir / "notifications.log"
    if log.exists():
        with st.expander("Messages sent"):
            st.code(log.read_text(encoding="utf-8")[-4000:] or "(none)")


def render_events(run_dir, status):
    st.markdown("#### Events")
    events = rc.read_events(run_dir)
    if not events:
        st.caption("No event yet.")
        return
    important = next((e for e in reversed(events) if e["action"] in ("alert", "ask")), None)
    if important:
        event_notice(important, _clock(important["start_s"]))
    pending = rc.pending_answers(run_dir)
    rows = [{"event": e["event_id"].rsplit("-", 1)[-1], "action": e["action"], "behavior": e["behavior"],
             "from": _clock(e["start_s"]), "confidence": round(e["confidence"], 2),
             "answer": e.get("researcher_answer") or ("sent, waiting for the run" if e["event_id"] in pending else "")}
            for e in events]
    st.dataframe(color_actions(pd.DataFrame(rows)), hide_index=True, width="stretch")
    for e in events:
        if e["action"] != "ask" or e.get("researcher_answer") or e["event_id"] in pending:
            continue
        if e is not important:
            event_notice(e, _clock(e["start_s"]))
        st.markdown(f"**Review question {e['event_id'].rsplit('-', 1)[-1]}**")
        a, b = st.columns(2)
        if a.button("Confirm", key=f"run_confirm_{e['event_id']}", type="primary"):
            rc.answer_event(run_dir, e["event_id"], "confirmed")
            st.rerun()
        if b.button("Reject", key=f"run_reject_{e['event_id']}"):
            rc.answer_event(run_dir, e["event_id"], "rejected")
            st.rerun()
