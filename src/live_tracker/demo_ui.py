"""The Live demo page: the video and the run, started together (new file).

Rendering only; the logic is in demo.py and run_control.py. The run is the real run program on a recorded
session, so closing the page does not stop it. The page says plainly that this is a simulated live stream
from a recording.

Order matters: the video element is drawn first and the GO signal is sent right after it in the same script
run, so both start within a fraction of a second. While the video plays, only the live panel (a fragment)
reruns, so the video is never redrawn. Questions are answered on the Run page, not here, because a full
rerun here could restart the video.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from . import demo
from . import monitor_runner as mr
from . import run_control as rc
from . import run_schedule as rs

KEY_RUN, KEY_VIDEO, KEY_SENT = "demo_run", "demo_video", "demo_go_sent"
DEFAULT_OFFSET_S = 0.3                      # the video needs a moment to start after its element is drawn


def _mmss(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def render():
    st.title("Live demo")
    st.caption("Simulated live stream from a recording (CalMS21, a validation session). The video and the monitor "
               "start together. No camera is used.")
    demos = demo.find_demos()
    if not demos:
        st.info("No demo video found. Put a CalMS21 video and its poses file in data/raw/calms21_video/ "
                "(see docs/lab_run.md and scripts/prepare_demo_session.py).")
        return
    if not demo.model_available():
        st.warning("Model B is not on this computer (models/model_b), so the demo cannot score the keypoints.")
        return
    run_dir = st.session_state.get(KEY_RUN)
    if run_dir is None:
        render_intro(demos)
    else:
        render_running(Path(run_dir), next((d for d in demos if d.name == st.session_state.get("demo_name")), demos[0]))


def render_intro(demos):
    chosen = st.selectbox("Recording", demos, format_func=lambda d: d.name, key="demo_choice")
    seconds = demo.length_s(chosen)
    st.markdown(f"- **Rule:** {demo.rule_text()}.\n"
                f"- **Length:** {_mmss(seconds)}. The run starts when you click and ends on its own.\n"
                f"- **Session:** `{chosen.session}`. Scores are Model B's raw scores (not calibrated).")
    with st.expander("Sync"):
        st.number_input("Wait before the run starts, after the video is drawn (seconds)", min_value=0.0, max_value=3.0,
                        value=DEFAULT_OFFSET_S, step=0.1, key="demo_offset",
                        help="Raise it if the alerts come before the fight in the video; lower it if they come late.")
    if st.button("Confirm the rule and start the demo", type="primary", key="demo_start"):
        try:
            run_dir = demo.start_demo(chosen)
        except (mr.RunRefusedError, rs.ScheduleError, ValueError, OSError) as exc:
            st.error(f"Not started: {exc}.")
            return
        st.session_state.update({KEY_RUN: str(run_dir), "demo_name": chosen.name, KEY_VIDEO: False, KEY_SENT: False})
        st.rerun()


def render_running(run_dir, chosen):
    left, right = st.columns([3, 2])
    with left:
        if st.session_state.get(KEY_VIDEO):
            st.video(str(chosen.video), autoplay=True, muted=True)
            if not st.session_state.get(KEY_SENT):             # the video element is drawn; now start the run
                time.sleep(float(st.session_state.get("demo_offset", DEFAULT_OFFSET_S)))
                rc.send_go(run_dir)
                st.session_state[KEY_SENT] = True
        else:
            st.info("Getting ready: loading the model and the keypoints.")
    with right:
        panel(run_dir, demo.length_s(chosen))


@st.fragment(run_every=1)
def panel(run_dir, length_s):
    run_dir = Path(run_dir)
    status = rc.read_status(run_dir)
    if status is None:
        st.caption("Waiting for the run program to start.")
        return
    problem = demo.failed_message(status)
    if problem:
        st.error(f"The run could not start: {problem}")
    if demo.go_if_ready(run_dir) and not st.session_state.get(KEY_VIDEO):
        st.session_state[KEY_VIDEO] = True                     # draw the video, then send GO (see render_running)
        st.rerun()

    frames = status.get("frames", 0)
    st.progress(min(1.0, frames / demo.FPS / length_s), text=f"{_mmss(frames / demo.FPS)} of {_mmss(length_s)}")
    c = st.columns(2)
    c[0].metric("Alerts", status.get("alerts", 0))
    c[1].metric("Questions", status.get("asks", 0))

    events = rc.read_events(run_dir)
    last = demo.latest_alert(events)
    if last is not None:
        text = f"**{'ALERT' if last['action'] == 'alert' else 'Question'}** at {_mmss(last['start_s'])}: possible " \
               f"{last['behavior']}, confidence {last['confidence']:.2f}"
        (st.error if last["action"] == "alert" else st.warning)(text)
    elif status.get("state") == "running":
        st.caption("No alert yet.")

    lag = status.get("max_lag_s", 0.0)
    if status.get("state") in ("running", "finished") and frames:
        if lag < 1.0:
            st.caption(f"Keeping up with real time (slowest delay {lag:.1f} s).")
        else:
            st.warning(f"The laptop fell behind at times (slowest delay {lag:.1f} s). Alerts were that late.")

    shown = [e for e in events if e["action"] in ("alert", "ask")]
    if shown:
        rows = [{"time": _mmss(e["start_s"]), "action": e["action"], "confidence": round(e["confidence"], 2)}
                for e in sorted(shown, key=lambda e: e["start_s"], reverse=True)]
        st.dataframe(rows, hide_index=True, width="stretch", height=260)

    if status.get("state") == "finished":
        st.success("The demo has finished. The report is on the Run page.")
        report = rc.read_report(run_dir)
        if report is not None:
            with st.expander("Report"):
                st.markdown(report[0])
    if status.get("state") in ("finished", "failed") or problem:
        if st.button("Start again", key="demo_again"):
            for key in (KEY_RUN, KEY_VIDEO, KEY_SENT, "demo_name"):
                st.session_state.pop(key, None)
            st.rerun()
    elif status.get("state") in rc.ACTIVE_STATES and st.button("Stop the demo", key="demo_stop"):
        rc.stop_run(run_dir)
