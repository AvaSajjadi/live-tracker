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
    return up.SessionSource("demo", f"Demo camera (recorded video {d.name})", video=str(d.video), poses=str(d.poses),
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
    elif preview[0] == "table":
        st.markdown("**First rows of the file**")
        st.dataframe(preview[1].head(8), hide_index=True, width="stretch")


# ---- the three choices ----

def render_camera():
    demos = demo.find_demos()
    if not demos:
        st.warning("No recorded video is available on this computer to stand in for a camera. A real camera cannot be "
                   "analysed yet, because that needs a mouse pose model.")
        return None
    source = camera_source(demos[0])
    current = flow.get_source()
    if current is None or current.to_dict() != source.to_dict():
        flow.set_source(source)                                  # choosing the demo camera is the whole step
    st.success("Camera connected successfully!")
    return ("path", demos[0].video)


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
