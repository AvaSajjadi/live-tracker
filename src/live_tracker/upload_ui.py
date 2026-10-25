"""The Upload step: choose what the agent should look at (new file). Rendering only.

Three choices: a camera, a video, a CSV. The left side holds the choice and its controls; the right side shows the
video as soon as there is one.

No camera can be analysed yet: turning each camera picture into mouse keypoints needs a mouse pose model, and this
project does not have one. So the camera choice is a DEMO camera: a recorded video stands in for the camera feed, so the
whole flow can be tried. The page says so in plain words and never claims that a camera is connected.

A video alone cannot be analysed (the behavior scores come from keypoints), so a video needs the keypoints of the same
recording. Files are checked and saved by uploads.py.
"""

from __future__ import annotations

import streamlit as st

from . import camera_check as cc
from . import camera_live
from . import demo
from . import flow
from . import uploads as up
from . import video_web as vw

CAMERA = "Connect a camera"
VIDEO = "Upload a video and its keypoints"
CENTERS = "Upload a CSV of mouse positions"
CHOICES = [CAMERA, VIDEO, CENTERS]
KIND_TEXT = {"demo": "demo camera, a recorded video stands in for the feed", "video_keypoints": "video with keypoints",
             "keypoints": "keypoints, no picture", "centers": "mouse positions, motion only", "camera": "camera"}


def camera_source(d):
    """The demo camera: the demo recording (video and keypoints) used as the stand-in feed."""
    return up.SessionSource("demo", "Demo camera (live stream camera)", video=str(d.video), poses=str(d.poses),
                            session=d.session)


def video_state(name, data):
    """('play', None) if the browser can play this video, else ('blocked', reason)."""
    ok, why = vw.is_playable_bytes(name, data)
    return ("play", None) if ok else ("blocked", why)


# ---- the right side: the video, or a table ----

def show_path(path):
    playable = vw.playable_video(path)
    if playable is not None:
        st.video(str(playable), autoplay=True, muted=True, loop=True)
        return
    ok, why = vw.is_browser_playable(path)
    st.warning(f"The picture cannot be shown in the browser: {why}.")
    if vw.converter_available():
        if st.button("Prepare the video for the browser", key="upload_convert"):
            with st.spinner("Converting the video (about a minute)..."):
                try:
                    vw.make_web_copy(path)
                except (vw.ConversionFailed, vw.ConverterMissing) as exc:
                    st.error(f"Not converted: {exc}")
                else:
                    st.rerun()
    else:
        st.caption("No video converter is installed. Install the optional package imageio-ffmpeg in the project's Python "
                   "(see docs/demo_recording.md), then a button appears here that makes a browser-friendly copy. The "
                   "original is never changed. The analysis does not need the picture.")


def show_preview(preview):
    if preview is None:
        st.caption("The video appears here as soon as you choose one.")
    elif preview[0] == "path":
        show_path(preview[1])
    elif preview[0] == "bytes":
        state, why = video_state(preview[1], preview[2])
        if state == "play":
            st.video(preview[2], muted=True)
        else:
            st.warning(f"This video cannot be shown in the browser: {why}. It can still be used for the analysis.")
    elif preview[0] == "live":
        live_preview(preview[1])
    elif preview[0] == "image":
        st.image(preview[1], caption="One picture from the camera", width="stretch")
    elif preview[0] == "table":
        st.markdown("**First rows of the file**")
        st.dataframe(preview[1].head(8), hide_index=True, width="stretch")


# ---- the three choices ----

DEMO_CAMERA, USB_CAMERA = "Demo camera (a recorded video)", "USB camera"


def render_demo_camera():
    demos = demo.find_demos()
    if not demos:
        st.warning("No recorded video is available on this computer to stand in for a camera. "
                   "A real camera cannot be analysed yet, because that needs a mouse pose model.")
        return None
    source = camera_source(demos[0])
    current = flow.get_source()
    if current is None or current.to_dict() != source.to_dict():
        flow.set_source(source)                                  # choosing the demo camera is the whole step
    st.success("Camera connected successfully!")
    return ("path", demos[0].video)


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
    mode = st.radio("What should the program analyze?", [REAL_POSE, STAGED], key="upload_cam_mode",
                    help="Real: a pose model finds the mice in the camera picture (none is set up yet). "
                         "Staged: the camera shows the iPad, but the numbers come from saved keypoints of the demo video.")
    staged = mode == STAGED
    current = flow.get_source()                                  # another camera, corners or mode than the saved ones: choose again
    if current is not None and current.device is not None and (
            current.device != device.strip() or current.screen != screen or current.staged_camera != staged):
        st.session_state.pop(flow.SOURCE_KEY, None)
    if staged:
        render_staged_choice(device, screen)
        return ("live", device.strip()) if live else ("image", png)
    _, reason = cc.pose_model.load()
    if reason:
        st.warning(f"{reason} You can choose this camera now, but the analysis cannot start until a pose model is set up.")
    if st.button("Use this camera", type="primary", key="upload_use_camera"):
        flow.set_source(up.SessionSource("camera", f"USB camera {device}", device=device.strip(), screen=screen))
    return ("live", device.strip()) if live else ("image", png)


REAL_POSE, STAGED = "Real pose model (needs a pose model)", "Staged demo (saved keypoints of the demo video)"


def render_staged_choice(device, screen):
    """The camera films a screen that plays the demo video. The picture is only shown; the keypoints are read from a file."""
    st.warning("STAGED DEMO. The camera picture is shown live, but the alerts are computed from the SAVED keypoints of the demo "
               "video, not from what the camera sees. Say this plainly when you present it.")
    demos = demo.find_demos()
    if not demos:
        st.error("No demo video with saved keypoints is on this computer (data/raw/calms21_video/), so there is nothing to stage.")
        return
    d = demos[0]
    st.caption(f"Play this video on the screen the camera films: {d.video.name}")
    if st.button("Use the staged demo", type="primary", key="upload_use_staged"):
        flow.set_source(up.SessionSource("demo", f"Staged demo: camera {device} films the screen", video=str(d.video),
                                         poses=str(d.poses), session=d.session, device=device.strip(), screen=screen))


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
    if not st.session_state.get("upload_screen_on", True):
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Live camera video", width="stretch")
        return
    from . import corner_picker as cp
    from . import screen_map as sm
    key = _picker_key()
    cp.corner_picker(image, cp.current_corners(key, sm.DEFAULT_CORNERS), key=key)
    corners = _corners_now()
    if corners is None:
        st.error("These corners cannot be used. Press Reset the corners.")
        return
    st.image(cv2.cvtColor(sm.warp(image, corners), cv2.COLOR_BGR2RGB), caption=STRETCHED_CAPTION, width="stretch")
    st.caption("Corners for the Colab notebook (copy this line into CORNERS):")
    st.code(sm.corners_text(corners), language=None)


def render_camera():
    kind = st.radio("Camera type", [DEMO_CAMERA, USB_CAMERA], key="upload_camera_kind")
    return render_usb_camera() if kind == USB_CAMERA else render_demo_camera()


def render_video():
    st.markdown("The analysis needs the **keypoints of the same recording** (7 per mouse, CalMS21 layout, in pixels of a "
                "1024 x 570 frame). A video alone is not enough.")
    video = st.file_uploader("Video", type=list(up.VIDEO_TYPES), key="upload_video",
                             help="mp4, webm, mov or m4v. Browsers play H.264 and VP9; other formats can be converted later.")
    keypoints = st.file_uploader("Keypoints file", type=list(up.TABLE_TYPES), key="upload_keypoints",
                                 help="Columns: mouse (a or b), keypoint (nose, left_ear, right_ear, neck, left_hip, "
                                      "right_hip, tail_base), frame, x, y; optional score (0 to 1) and session.")
    with st.expander("Frame rate"):
        fps = st.number_input("Camera frame rate (frames per second)", min_value=1.0, value=30.0, key="upload_fps",
                              help="Frames per second (fps): how many pictures of the recording there are in each second. "
                                   "Model B was trained on 30 fps recordings.")
    if st.button("Use these files", type="primary", key="upload_use_files"):
        try:
            source = up.build_keypoints_source((keypoints.name, keypoints.getvalue()) if keypoints else None,
                                               (video.name, video.getvalue()) if video else None, fps=fps)
        except up.UploadError as exc:
            st.error(f"Not used: {exc}")
        else:
            flow.set_source(source)
    return ("bytes", video.name, video.getvalue()) if video else None


def render_centers():
    st.markdown("A CSV with one row per frame and four columns: `x_a`, `y_a` (mouse a) and `x_b`, `y_b` (mouse b), in pixels. "
                "This gives motion and tracking quality only: no behavior scores and no alerts, because that needs the "
                "7 keypoints of each mouse.")
    file = st.file_uploader("Positions CSV", type=["csv"], key="upload_centers")
    preview = None
    if file:
        try:
            preview = ("table", up.check_centers(file.name, file.getvalue()))
        except up.UploadError as exc:
            st.error(f"This file cannot be used: {exc}")
    if st.button("Use this file", type="primary", key="upload_use_centers"):
        try:
            source = up.build_centers_source((file.name, file.getvalue()) if file else ("", b""))
        except up.UploadError as exc:
            st.error(f"Not used: {exc}")
        else:
            flow.set_source(source)
    return preview


def render():
    flow.render_bar("upload")
    st.title("Upload")
    st.caption("Choose what the agent should look at. Next, tell it what to look for.")
    left, right = st.columns([2, 3])
    with left:
        choice = st.radio("Source", CHOICES, index=None, key="upload_choice")
        preview = {CAMERA: render_camera, VIDEO: render_video, CENTERS: render_centers}.get(choice, lambda: None)()
    with right:
        show_preview(preview)

    current = flow.get_source()
    st.divider()
    if current is None:
        st.caption("Nothing chosen yet.")
    flow.render_nav("upload", can_next=current is not None, next_hint="Choose a source first, then press Next.")
