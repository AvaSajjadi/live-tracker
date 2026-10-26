"""The Camera page: connect a USB camera, set the screen corners, analyze it live or record and analyze (new file).

Rendering only. This used to be part of the Upload step (upload_ui.py). The Upload step is now Syed's page (recorded data only); the camera
screens live here, on their own page, so that nothing of the camera work is lost. The camera is a first step like Upload: "Next" goes to Rules.

Two ways to use a camera (see docs/camera.md):
  - Live real pose: the pose model (models/pose/pose_model.json, on a Mac pose_model_mac.json) finds the mice in each picture;
    see docs/pose_model.md.
  - Record, then analyze with MARS (real pose, not live): the camera is recorded with the screen stretched to 1024 x 570, MARS runs on the
    recording in the background, and the result becomes the source.
"""

from __future__ import annotations

import streamlit as st

from . import camera_check as cc
from . import camera_live
from . import flow
from . import mars_job
from . import uploads as up


def show_preview(preview):
    if preview is None:
        st.caption("The live video appears here as soon as you test the camera.")
    elif preview[0] == "live":
        live_preview(preview[1])
    elif preview[0] == "image":
        st.image(preview[1], caption="One picture from the camera", width="stretch")


def render_usb_camera():
    """A real camera: test it, watch live video from it, drag the screen corners, then use it."""
    if not cc.opencv_available():
        st.error("OpenCV is not installed, so a camera cannot be opened. Run: pip install opencv-python-headless")
        return None
    device = st.text_input("Camera number or stream address", value="0", key="upload_device",
                           help="0 is the first camera, 1 the second. A network camera has an address like rtsp://...")
    current = flow.get_source()                                  # the demo camera or another camera number is no longer the choice
    if current is not None and current.kind == "demo" and not current.device:
        st.session_state.pop(flow.SOURCE_KEY, None)
    state = st.session_state.get("upload_camera_test")
    if st.button("Test the camera", key="upload_test_camera"):
        camera_live.release()                                    # a camera opened by this page would block the test
        with st.spinner("Opening the camera..."):
            png, why = cc.snapshot(device)
        state = st.session_state["upload_camera_test"] = {"device": device, "png": png, "why": why}
        st.session_state["upload_live"] = png is not None
    if not state or state["device"] != device:
        st.caption("Press Test the camera to see live video from it.")
        return None
    if state["png"] is None:
        st.error(f"The camera cannot be used: {state['why']}.")
        return None
    st.success("The camera works.")
    live = bool(st.session_state.get("upload_live"))
    if live and st.button("Stop the camera", key="upload_stop_camera"):
        camera_live.release()
        st.session_state["upload_live"] = live = False
    screen_on = st.checkbox("The camera films a screen (for example an iPad that plays the video)", value=True, key="upload_screen_on")
    screen, png = None, state["png"]
    if screen_on:
        st.caption("Drag the 4 red corners on the video until the green shape sits on the edge of the video: 1 = top-left, "
                   "2 = top-right, 3 = bottom-right, 4 = bottom-left.")
        if st.button("Reset the corners", key="upload_reset_corners"):
            st.session_state[KEY_VERSION] = st.session_state.get(KEY_VERSION, 0) + 1   # a new picker starts at the default corners
        screen = _corners_now()
        if screen is None:
            st.error("These corners cannot be used. Press Reset the corners.")
        elif not live:
            png = _marked_png(state["png"], screen)
    mode = st.radio("What should the program analyze?", [REAL_POSE, OFFLINE], key="upload_cam_mode",
                    help="Live real pose: the pose model finds the mice in each camera picture. Record then analyze: real pose "
                         "from your recording, with MARS, not live (about 25 minutes for each minute of video).")
    current = flow.get_source()                                  # another camera or corners than the saved ones: choose again
    if current is not None and current.device is not None and (
            current.device != device.strip() or current.screen != screen or current.staged_camera):
        st.session_state.pop(flow.SOURCE_KEY, None)
    if mode == OFFLINE:
        render_record_and_analyze(device, live, screen_on, screen)
        return ("live", device.strip()) if live else ("image", png)
    _, reason = cc.pose_model.check()                              # check, not load: loading a trained model takes seconds
    if reason:
        st.warning(f"{reason} You can choose this camera now, but the analysis cannot start until a pose model is set up.")
    if st.button("Use this camera", type="primary", key="upload_use_camera"):
        flow.set_source(up.SessionSource("camera", f"USB camera {device}", device=device.strip(), screen=screen))
    return ("live", device.strip()) if live else ("image", png)


OFFLINE = "Record, then analyze with MARS (real pose, not live)"


REAL_POSE = "Live real pose"


def recordings_root():
    import os
    from pathlib import Path
    return Path(os.environ.get("LIVE_TRACKER_RECORDINGS") or up.ROOT / "outputs" / "recordings")


def render_record_and_analyze(device, live, screen_on, screen):
    """Record the camera (the screen stretched to 1024 x 570), run MARS on the recording in the background, use the result."""
    from datetime import datetime, timezone
    st.info("Real pose, offline. The camera is recorded with the screen stretched to 1024 x 570. Then MARS finds the mice in the "
            "recording. This laptop needs about 25 minutes for each minute of video, so record 10 to 20 seconds first. "
            "MARS needs a top view with one black and one white mouse.")
    ok, why = mars_job.available()
    if not ok:
        st.error(why)
    g = None
    if live:
        try:
            g = camera_live.grabber(device)              # opens the camera now, so the buttons below can use it
        except camera_live.rsrc.CameraUnavailableError as exc:
            st.error(f"The camera cannot be used: {exc}.")
    if g is None:
        st.caption("Press Test the camera first, so the live video runs. Then you can record.")
    elif g.recording:
        st.error("RECORDING. Play the video on the screen now. Press Stop recording when it is over.")
        if st.button("Stop recording", type="primary", key="upload_rec_stop"):
            st.session_state["upload_recording"] = g.stop_recording()
            st.session_state.pop("upload_job", None)
            st.rerun()
    else:
        blocked = screen_on and screen is None
        if st.button("Start recording", type="primary", key="upload_rec_start", disabled=blocked):
            folder = recordings_root() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            folder.mkdir(parents=True, exist_ok=True)
            try:
                g.start_recording(folder / "camera.mp4", screen if screen_on else None)
            except (RuntimeError, OSError) as exc:
                st.error(f"Not started: {exc}")
            else:
                st.session_state.pop("upload_recording", None)
                st.session_state.pop("upload_job", None)
                st.rerun()
    rec = st.session_state.get("upload_recording")
    if not rec:
        return
    st.success(f"Recorded {rec['seconds']:.0f} s ({rec['frames']} frames, {rec['width']} x {rec['height']}). Saved in {rec['path']}.")
    if "upload_job" not in st.session_state:
        minutes = mars_job.estimate_minutes(rec["frames"])
        if st.button(f"Analyze with MARS (about {minutes:.0f} min)", type="primary", key="upload_mars_start", disabled=not ok):
            from pathlib import Path
            job = Path(rec["path"]).parent / "mars"
            try:
                mars_job.start(rec["path"], job, fps=rec["fps"])
            except (RuntimeError, OSError) as exc:
                st.error(f"Not started: {exc}")
            else:
                camera_live.release()                           # the camera is not needed now; the CPU is
                st.session_state["upload_live"] = False
                st.session_state["upload_job"] = str(job)
                st.rerun()
    else:
        job_panel(st.session_state["upload_job"], rec)


@st.fragment(run_every=2)
def job_panel(job_dir, rec):
    s = mars_job.status(job_dir)
    if s["state"] == "running":
        total = s["total"] or rec["frames"]
        st.progress(min(1.0, s["done"] / total) if total else 0.0,
                    text=f"MARS: {s['done']} of {total} frames ({s['message']}). You can leave this page; the job goes on.")
    elif s["state"] == "failed":
        st.error(f"The analysis did not finish: {s['message']}. The log is in {job_dir}.")
    elif s["state"] == "finished":
        st.success(f"MARS finished ({s['message']}).")
        if st.button("Use this recording", type="primary", key="upload_use_recording"):
            from pathlib import Path
            try:
                source = up.build_keypoints_source(("keypoints.parquet", mars_job.result(job_dir).read_bytes()),
                                                   (Path(rec["path"]).name, Path(rec["path"]).read_bytes()), fps=rec["fps"])
            except (up.UploadError, OSError) as exc:
                st.error(f"Not used: {exc}")
            else:
                flow.set_source(source)
                st.rerun()


STRETCHED_CAPTION = "The screen stretched to 1024 x 570. The whole video should fill this picture, with no border."


KEY_VERSION = "upload_corner_version"


def _picker_key():
    """A new key makes a new picker, which starts at the default corners (the Reset button)."""
    return f"upload_corner_picker_{st.session_state.get(KEY_VERSION, 0)}"


def _corners_now():
    """The corners the researcher dragged to (the default ones before any drag), or None when they cannot be used."""
    from . import corner_picker as cp
    from . import screen_map as sm
    try:
        return sm.check_corners(cp.current_corners(_picker_key(), sm.DEFAULT_CORNERS))
    except sm.ScreenError:
        return None


def _marked_png(png, corners):
    import cv2
    import numpy as np
    from . import screen_map as sm
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    return cv2.imencode(".png", sm.outline(image, corners))[1].tobytes()


@st.fragment(run_every=0.1)
def live_preview(device):
    """The camera as live video (about 10 pictures a second). With a screen: the corners can be dragged on it, and the
    stretched screen is shown below."""
    image, why = camera_live.read(device)
    if image is None:
        st.error(f"The camera cannot be shown: {why}.")
        return
    import cv2
    g = camera_live.current()
    status = g.recording_status() if g is not None else None
    if status is not None:
        st.error(f"RECORDING  {status['seconds']:.0f} s  ({status['frames']} frames)")
    if not st.session_state.get("upload_screen_on", True):
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Live camera video", width="stretch", output_format="JPEG")
        return
    from . import corner_picker as cp
    from . import screen_map as sm
    key = _picker_key()
    cp.corner_picker(image, cp.current_corners(key, sm.DEFAULT_CORNERS), key=key)
    corners = _corners_now()
    if corners is None:
        st.error("These corners cannot be used. Press Reset the corners.")
        return
    stretched = cv2.resize(sm.warp(image, corners), (640, 356), interpolation=cv2.INTER_AREA)      # shown smaller: a big PNG ten times a second used a whole core
    st.image(cv2.cvtColor(stretched, cv2.COLOR_BGR2RGB), caption=STRETCHED_CAPTION, width="stretch", output_format="JPEG")
    st.caption("Corners for the Colab notebook (copy this line into CORNERS):")
    st.code(sm.corners_text(corners), language=None)


def render():
    flow.render_bar("upload")
    st.title("Camera")
    st.caption("Connect a USB camera, set the screen corners, then analyze it live or record and analyze. "
               "Recorded data is on the Upload page.")
    left, right = st.columns([2, 3])
    with left:
        preview = render_usb_camera()
    with right:
        show_preview(preview)

    current = flow.get_source()
    st.divider()
    if current is None:
        st.caption("Nothing chosen yet.")
    flow.render_nav("upload", can_next=current is not None, next_hint="Choose a source first, then press Next.")
