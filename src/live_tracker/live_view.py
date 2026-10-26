"""The live session view on the Real session page: the analysis on the left, the video (about 45% of the width) on the right, with two small count boxes above it (new file).

Rendering only; the logic is in live_session.py, run_control.py and video_web.py. The run is the real run program,
so closing the page does not stop it. The page says plainly that this is a simulated live stream from a recording.

Order matters: the video element is drawn first and the GO signal is sent right after it in the same script run, so
both start within a fraction of a second. While the video plays, only the live panel (a fragment) reruns, so the
video is never redrawn. Questions are answered on the Run page, because a full rerun here could restart the video.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from . import demo
from . import flow
from . import biologist_report as bio
from . import live_session as ls
from . import monitor_runner as mr
from . import camera_live
from . import pose_model
from . import run_control as rc
from . import run_schedule as rs
from . import schedule_workflow as sw
from . import video_web as vw
from .run_ui import approved_rules
from .uploads import SessionSource
from .ui_style import note, badge, color_actions, event_notice

KEY_RUN, KEY_VIDEO, KEY_SENT, KEY_SOURCE, KEY_SERIES = "live_run", "live_video", "live_go_sent", "live_source", "live_series"
DEFAULT_OFFSET_S = 0.3                      # the video needs a moment to start after its element is drawn
CHART_SECONDS = 30


def _mmss(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def chosen_limit_s():
    """The length in seconds of the schedule the researcher confirmed on the Rules page, or None."""
    wf = st.session_state.get("schedule_workflow")
    approved = wf.approved() if isinstance(wf, sw.ScheduleWorkflow) else []
    return approved[-1].planned_s if approved else None


def _reset():
    for key in (KEY_RUN, KEY_VIDEO, KEY_SENT, KEY_SERIES):
        st.session_state.pop(key, None)
    camera_live.release()


def render():
    st.caption("Recorded-data simulation: behavior scoring uses saved keypoints, not camera images.")
    source = flow.get_source()
    if source is None:
        st.info("Nothing is chosen yet. Go back to the Upload step and choose a recording.")
        return
    if st.session_state.get(KEY_SOURCE) != source.to_dict():          # a different recording was chosen: start clean
        _reset()
        st.session_state[KEY_SOURCE] = source.to_dict()
    if source.kind == "camera":
        render_camera(source)
        return
    if source.kind == "centers":
        render_centers(source)
        return
    rules = approved_rules()
    if not rules:
        st.warning("No approved rule yet. Go back to the Rules step and confirm a rule: the agent needs to know what to look for.")
        return
    if not demo.model_available():
        st.warning("Model B is not on this computer (models/model_b), so the keypoints cannot be scored.")
        return
    run_dir = st.session_state.get(KEY_RUN)
    if run_dir is None:
        render_intro(source, rules)
    else:
        render_running(Path(run_dir), source)


# ---- a real camera ----

def render_camera(source):
    rules = approved_rules()
    if not rules:
        st.warning("No approved rule yet. Go back to the Rules step and confirm a rule: the agent needs to know what to look for.")
        return
    if not demo.model_available():
        st.warning("Model B is not on this computer (models/model_b), so the keypoints cannot be scored.")
        return
    run_dir = st.session_state.get(KEY_RUN)
    if run_dir is not None:
        counts(Path(run_dir))
        panel(Path(run_dir), ls.camera_limit_s(chosen_limit_s()), source.fps)
        return
    estimator, why = pose_model.load()
    limit = chosen_limit_s()
    st.markdown(f"- **Camera:** {source.label}. This is a real camera, not a recording.\n"
                f"- **Rules:** " + "; ".join(f"{r.behavior} for at least {r.min_duration_s:g} s "
                                             f"(ask above {r.ask_threshold:g}, alert at {r.alert_threshold:g})" for r in rules)
                + f".\n- **Time limit:** {_mmss(ls.camera_limit_s(limit))}"
                + (" (the time you confirmed on the Rules page)." if limit else " (no time chosen on the Rules page)."))
    if estimator is None:
        st.error(why)
        return
    if st.button("Start the camera session", type="primary", key="live_start_camera"):
        camera_live.release()                                     # the camera run opens the camera in its own program
        try:
            run_dir = ls.start_camera(source, rules, limit_s=limit)
        except (mr.RunRefusedError, rs.ScheduleError, ValueError, OSError) as exc:
            st.error(f"Not started: {exc}.")
            return
        st.session_state.update({KEY_RUN: str(run_dir), KEY_SENT: True})
        st.rerun()


# ---- a centers file: motion only ----

def render_centers(source):
    import pandas as pd
    df = pd.read_csv(source.centers)
    series = ls.centers_series(df, source.fps)
    st.caption("Motion and tracking only. A centers file has one point per mouse, so there are no behavior scores "
               "and no alerts; those need the 7 keypoints of each mouse.")
    c = st.columns(3)
    c[0].metric("Frames", f"{len(series):,}")
    c[1].metric("Length", _mmss(len(series) / source.fps))
    c[2].metric("Both mice tracked", f"{series['tracking_ok'].mean():.1%}")
    st.markdown("**Speed of each mouse (px/s)**")
    st.line_chart(series.set_index("time_s")[["speed_a", "speed_b"]], height=220)
    st.markdown("**Distance between the mice (px)**")
    st.line_chart(series.set_index("time_s")[["distance"]], height=160)


# ---- before the session starts ----

STAGED_WARNING = ("STAGED DEMO. The camera picture is shown live, but the alerts are computed from the SAVED keypoints of the "
                  "demo video, not from what the camera sees. There is no pose model in this demo.")


def render_staged_intro(source, rules):
    st.warning(STAGED_WARNING)
    seconds = ls.length_s(source.poses, source.fps)
    st.markdown(f"- **Camera:** {source.label}.\n"
                f"- **Video on the screen:** {Path(source.video).name}, {_mmss(seconds)} long. Play it from the start.\n"
                f"- **Rules:** " + "; ".join(f"{r.behavior} for at least {r.min_duration_s:g} s "
                                             f"(ask above {r.ask_threshold:g}, alert at {r.alert_threshold:g})" for r in rules)
                + ".\n- **Scores:** Model B's raw scores (not calibrated).")
    limit = chosen_limit_s()
    if limit is not None and limit < seconds:
        st.markdown(f"- **Time limit:** the analysis stops after {_mmss(limit)} (the time you confirmed on the Rules page).")
    st.caption("Steps: 1. Press Start. 2. Wait for 'Ready'. 3. Press GO and press play on the screen at the same moment.")
    if st.button("Start the staged session", type="primary", key="live_start_staged"):
        try:
            run_dir = ls.start_live(source, rules, wait_for_go=True, limit_s=limit)
        except (mr.RunRefusedError, rs.ScheduleError, ValueError, OSError) as exc:
            st.error(f"Not started: {exc}.")
            return
        st.session_state.update({KEY_RUN: str(run_dir), KEY_VIDEO: False, KEY_SENT: False})
        st.rerun()


@st.fragment(run_every=0.5)
def camera_preview(device, screen):
    """The live camera picture (stretched to the video part when corners were set). Only shown, never analysed."""
    image, why = camera_live.read(device)
    if image is None:
        st.error(f"The camera cannot be shown: {why}.")
        return
    import cv2
    if screen:
        from . import screen_map as sm
        image = sm.warp(image, screen)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Live camera picture (shown only)", width="stretch", output_format="JPEG")


def render_staged_running(run_dir, source):
    st.warning(STAGED_WARNING)
    if KEY_SERIES not in st.session_state:
        st.session_state[KEY_SERIES] = ls.motion_series(source.poses, source.fps)
    length = ls.length_s(source.poses, source.fps)
    left, right = st.columns([11, 9])
    with right:
        counts(run_dir)
        camera_preview(source.device, source.screen)
        status = rc.read_status(run_dir) or {}
        if not st.session_state.get(KEY_SENT):
            if status.get("state") == "ready":
                st.success("Ready.")
                if st.button("GO (press play on the screen at the same moment)", type="primary", key="live_go_staged"):
                    rc.send_go(run_dir)
                    st.session_state[KEY_SENT] = True
                    st.rerun()
            elif status.get("state") == "failed":
                st.error("The run could not start.")
            else:
                st.info("Getting ready: loading the model and the keypoints.")
    with left:
        panel(run_dir, length, source.fps)


def render_intro(source, rules):
    if source.staged_camera:
        render_staged_intro(source, rules)
        return
    seconds = ls.length_s(source.poses, source.fps)
    playable = vw.playable_video(source.video) if source.has_video else None
    recording = source.label if source.kind == "demo" else f"{source.label}, {_mmss(seconds)} long"
    st.markdown(f"- **Recording:** {recording}.\n"
                f"- **Rules:** " + "; ".join(f"{r.behavior} for at least {r.min_duration_s:g} s "
                                             f"(ask above {r.ask_threshold:g}, alert at {r.alert_threshold:g})" for r in rules)
                + ".\n- **Scores:** Model B's raw scores (not calibrated).")
    if source.has_video and playable is None:
        ok, why = vw.is_browser_playable(source.video)
        st.warning(f"The picture cannot be shown in the browser: {why}. The analysis can still run without it.")
        if vw.converter_available():
            if st.button("Prepare the video for the browser", key="live_convert"):
                with st.spinner("Converting the video (about a minute)..."):
                    try:
                        vw.make_web_copy(source.video)
                    except (vw.ConversionFailed, vw.ConverterMissing) as exc:
                        st.error(f"Not converted: {exc}")
                    else:
                        st.rerun()
        else:
            note("caption", "No video converter is installed.",
                 "Install the optional package imageio-ffmpeg in the project's Python, then press the button that appears here "
                 "to make a browser-friendly copy next to the original. The original is never changed.")
    limit = chosen_limit_s()
    if limit is not None and limit < seconds:
        st.markdown(f"- **Time limit:** the analysis stops after {_mmss(limit)} (the time you confirmed on the Rules page).")
    with st.expander("Sync"):
        st.number_input("Wait before the run starts, after the video is drawn (seconds)", min_value=0.0, max_value=3.0,
                        value=DEFAULT_OFFSET_S, step=0.1, key="live_offset",
                        help="Raise it if the alerts come before the fight in the video; lower it if they come late.")
    label = "Start recorded session" if (playable or not source.has_video) else "Start without the picture"
    if st.button(label, type="primary", key="live_start"):
        try:
            run_dir = ls.start_live(source, rules, wait_for_go=playable is not None,
                                 limit_s=limit)
        except (mr.RunRefusedError, rs.ScheduleError, ValueError, OSError) as exc:
            st.error(f"Not started: {exc}.")
            return
        st.session_state.update({KEY_RUN: str(run_dir), KEY_VIDEO: False, KEY_SENT: playable is None})
        st.rerun()


# ---- while it runs ----

def render_running(run_dir, source):
    if source.staged_camera:
        render_staged_running(run_dir, source)
        return
    playable = vw.playable_video(source.video) if source.has_video else None
    if KEY_SERIES not in st.session_state:
        st.session_state[KEY_SERIES] = ls.motion_series(source.poses, source.fps)
    length = ls.length_s(source.poses, source.fps)
    if playable is None:
        counts(run_dir)
        panel(run_dir, length, source.fps)
        return
    left, right = st.columns([11, 9])                         # run details on the left, the video (about 45%) on the right
    with right:
        counts(run_dir)
        if st.session_state.get(KEY_VIDEO):
            st.video(str(playable), autoplay=True, muted=True)
            if not st.session_state.get(KEY_SENT):             # the video element is drawn; now start the run
                time.sleep(float(st.session_state.get("live_offset", DEFAULT_OFFSET_S)))
                rc.send_go(run_dir)
                st.session_state[KEY_SENT] = True
        else:
            st.info("Getting ready: loading the model and the keypoints.")
    with left:
        panel(run_dir, length, source.fps)


@st.fragment(run_every=1)
def counts(run_dir):
    """Two small boxes with the number of alerts and questions (above the video)."""
    status = rc.read_status(Path(run_dir)) or {}
    st.markdown(badge(f"Alerts: {status.get('alerts', 0)}", "alert") + " "
                + badge(f"Questions: {status.get('asks', 0)}", "ask"), unsafe_allow_html=True)


def _small_headings(text):
    """Page view only: the theme makes Markdown headings very large, so a heading line becomes a bold line."""
    lines = []
    for line in text.split("\n"):
        if line.startswith("# ") or line.startswith("## "):
            line = f"**{line.lstrip('#').strip()}**"
        lines.append(line)
    return "\n".join(lines)


def _show_report(text):
    """The report text; a Markdown table is shown as a small data table so it fits the box."""
    text = _small_headings(text)
    rows = [ln for ln in text.split("\n") if ln.startswith("|")]
    if len(rows) < 3:
        st.markdown(text)
        return
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in rows]
    before, _, after = text.partition(rows[0])
    after = after.split(rows[-1], 1)[1]
    st.markdown(before)
    st.dataframe([dict(zip(cells[0], r)) for r in cells[2:]], hide_index=True, width="stretch")
    st.markdown(after)


def render_biologist_report(run_dir, report_json):
    """The written report for the biologist: Claude writes it when the button is clicked (one paid request)."""
    key = f"live_bio_{run_dir.name}"
    saved = run_dir / bio.FILE_NAME
    if key not in st.session_state and saved.exists():
        st.session_state[key] = saved.read_text(encoding="utf-8")
    text = st.session_state.get(key)
    if text is None:
        st.caption("Claude writes a short report in plain words for the biologist. It sends the facts of this run "
                   "(no video, no keypoints) to Claude and uses your Claude key, so it costs a little.")
        if st.button("Write report with Claude (paid call)", key="live_bio_write"):
            with st.spinner("Claude is writing the report..."):
                text, problem = bio.write(report_json)
            if problem:
                text, plain_problem = bio.write_plain(report_json)       # no Claude: the program writes the facts
                if plain_problem:
                    st.error(f"Not written: {problem}")
                    return
                st.session_state[key + "_why"] = f"Claude could not write it ({problem}) so this is the plain report."
            st.session_state[key] = text
            st.session_state[key + "_note"] = bio.save_and_notify(run_dir, text)
        else:
            return
    with st.container(border=True):
        _show_report(text)
    st.download_button("Download the report (Markdown)", text, file_name=f"{run_dir.name}_report.md",
                       mime="text/markdown", key="live_bio_download", on_click="ignore")
    if st.session_state.get(key + "_why"):
        st.caption(st.session_state[key + "_why"])
    if st.session_state.get(key + "_note"):
        st.caption(st.session_state[key + "_note"])


@st.fragment(run_every=1)
def panel(run_dir, length_s, fps):
    run_dir = Path(run_dir)
    status = rc.read_status(run_dir)
    if status is None:
        st.caption("Waiting for the run program to start.")
        return
    problem = ls.failed_message(status)
    if problem:
        st.error(f"The run could not start: {problem}")
    if ls.go_if_ready(run_dir) and not st.session_state.get(KEY_VIDEO):
        st.session_state[KEY_VIDEO] = True                     # draw the video, then send GO (see render_running)
        st.rerun()

    events = rc.read_events(run_dir)
    last = ls.latest_alert(events)
    if last is not None:
        event_notice(last, _mmss(last['start_s']))
    elif status.get("state") == "running":
        st.caption("No alert yet.")

    frames = status.get("frames", 0)
    st.progress(min(1.0, frames / fps / length_s), text=f"{_mmss(frames / fps)} of {_mmss(length_s)}")
    series = st.session_state.get(KEY_SERIES)
    if series is not None and frames:
        now = frames / fps
        view = series[(series["time_s"] <= now) & (series["time_s"] >= now - CHART_SECONDS)]
        if len(view):
            st.markdown(f"**Motion, last {CHART_SECONDS} s** (px/s, px)")
            st.line_chart(view.set_index("time_s")[["speed_a", "speed_b", "distance"]], height=260)
            if not bool(view["tracking_ok"].iloc[-1]):
                st.warning("Tracking is lost for at least one mouse right now. A gap is not a still mouse.")

    lag = status.get("max_lag_s", 0.0)
    if status.get("state") in ("running", "finished") and frames:
        if lag < 1.0:
            st.caption(f"Keeping up with real time (slowest delay {lag:.1f} s).")
        else:
            st.warning(f"The laptop fell behind at times (slowest delay {lag:.1f} s). Alerts were that late.")

    shown = [e for e in events if e["action"] in ("alert", "ask")]
    if shown:
        rows = [{"time": _mmss(e["start_s"]), "action": e["action"], "behavior": e["behavior"], "confidence": round(e["confidence"], 2)}
                for e in sorted(shown, key=lambda e: e["start_s"], reverse=True)]
        st.dataframe(color_actions(pd.DataFrame(rows)), hide_index=True, width="stretch", height=240)

    if status.get("state") == "finished":
        report = rc.read_report(run_dir)
        if report is None:
            st.warning("The session has finished; the report is not available yet.")
        elif status.get("message") == "The schedule ended.":
            st.success("The chosen time is over; analysis stopped and the report is ready (video may still play).")
        else:
            st.success("The session has finished; its report is ready on the Run page.")
        if report is not None:
            with st.expander("Report"):
                st.markdown(report[0])
            render_biologist_report(run_dir, report[1])
    if status.get("state") in ("finished", "failed") or problem:
        if st.button("Start again", key="live_again"):
            _reset()
            st.rerun()
    elif status.get("state") in rc.ACTIVE_STATES and st.button("Stop the session", key="live_stop"):
        rc.stop_run(run_dir)
