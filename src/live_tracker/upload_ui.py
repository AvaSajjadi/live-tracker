"""The Upload step: choose what the agent should look at (new file). Rendering only.

Files are checked and saved by uploads.py. A video alone cannot be analysed (the behavior scores come from keypoints),
and a camera cannot be analysed yet (no mouse pose model), so the page says so instead of letting the researcher
walk into a dead end.
"""

from __future__ import annotations

import streamlit as st

from . import demo
from . import flow
from . import uploads as up

DEMO = "Use the demo recording"
CAMERA = "Connect a camera"
VIDEO = "Upload a video and its keypoints"
CENTERS = "Upload a CSV of mouse positions"
KIND_TEXT = {"demo": "demo recording, with picture", "video_keypoints": "video with keypoints",
             "keypoints": "keypoints, no picture", "centers": "mouse positions, motion only", "camera": "camera"}


def demo_source(d):
    return up.SessionSource("demo", f"Demo recording ({d.name})", video=str(d.video), poses=str(d.poses), session=d.session)


def render():
    flow.render_bar("upload")
    st.title("Upload")
    st.caption("Choose what the agent should look at. Next, tell it what to look for.")
    demos = demo.find_demos()
    choices = ([DEMO] if demos else []) + [CAMERA, VIDEO, CENTERS]
    choice = st.radio("Source", choices, key="upload_choice")

    if choice == DEMO:
        chosen = st.selectbox("Recording", demos, format_func=lambda d: d.name, key="upload_demo") if len(demos) > 1 else demos[0]
        st.markdown("A CalMS21 video of two mice, with its saved keypoints: the validation session with the most events. "
                    "It is the best case, not a typical session.")
        if st.button("Use the demo recording", type="primary", key="upload_use_demo"):
            flow.set_source(demo_source(chosen))
    elif choice == CAMERA:
        st.info("A camera cannot be analysed yet. Turning each camera picture into mouse keypoints needs a mouse pose model, "
                "and this project does not have one. Until then, choose a recording.")
    elif choice == VIDEO:
        st.markdown("The analysis needs the **keypoints of the same recording** (7 per mouse, CalMS21 layout, in pixels of a "
                    "1024 x 570 frame). A video alone is not enough.")
        video = st.file_uploader("Video (optional, to see the picture)", type=list(up.VIDEO_TYPES), key="upload_video",
                                 help="mp4, webm, mov or m4v. Browsers play H.264 and VP9; other formats can be converted later.")
        keypoints = st.file_uploader("Keypoints file", type=list(up.TABLE_TYPES), key="upload_keypoints",
                                     help="Columns: mouse (a or b), keypoint (nose, left_ear, right_ear, neck, left_hip, right_hip, "
                                          "tail_base), frame, x, y; optional score (0 to 1) and session.")
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
    else:
        st.markdown("A CSV with one row per frame and four columns: `x_a`, `y_a` (mouse a) and `x_b`, `y_b` (mouse b), in pixels. "
                    "This gives motion and tracking quality only: no behavior scores and no alerts, because that needs the "
                    "7 keypoints of each mouse.")
        file = st.file_uploader("Positions CSV", type=["csv"], key="upload_centers")
        if st.button("Use this file", type="primary", key="upload_use_centers"):
            try:
                source = up.build_centers_source((file.name, file.getvalue()) if file else ("", b""))
            except up.UploadError as exc:
                st.error(f"Not used: {exc}")
            else:
                flow.set_source(source)

    current = flow.get_source()
    st.divider()
    if current is None:
        st.caption("Nothing chosen yet.")
    else:
        st.success(f"Chosen: {current.label} ({KIND_TEXT[current.kind]}).")
        if current.can_score:
            st.caption("Keypoints present: the agent can score behaviors and raise alerts.")
    flow.render_nav("upload", can_next=current is not None,
                    next_hint="Choose a recording first, then press Next.")
