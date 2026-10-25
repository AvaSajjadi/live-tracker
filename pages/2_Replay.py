"""Replay page: the real video next to the mouse skeletons, to check that the model finds the mice (new file).

Nothing about the run is shown here. The video is the one chosen on the Upload step. The skeleton drawing follows
the video's own time (see live_tracker.skeleton_player). The run details are on the Real session page.
"""

import streamlit as st
import streamlit.components.v1 as components

from live_tracker import flow, ui_style
from live_tracker import skeleton_player as sp
from live_tracker import video_web as vw

ui_style.apply()
flow.render_bar("replay")
st.title("Replay")
flow.render_nav("replay")


@st.cache_data(show_spinner="Reading the keypoints...", max_entries=2)
def _frames(poses):
    return sp.pose_frames(poses)


source = flow.get_source()
if source is not None and source.kind == "camera":
    st.info("A real camera has no recording to replay, so this step has nothing to show. Press Next to continue. "
            "(For the staged demo, choose 'Staged demo' on the Upload step: it has a video and saved keypoints to replay.)")
    st.stop()
if source is None or not source.has_video or not source.poses:
    st.info("Nothing to replay yet. On the Upload step, choose a video that has a keypoints file.")
    st.stop()
playable = vw.playable_video(source.video)
if playable is None:
    st.warning("The picture cannot be shown in the browser. Open the Real session page to prepare the video.")
    st.stop()

video, skeletons = st.columns(2)
with video:
    st.video(str(playable), muted=True)
with skeletons:
    components.html(sp.html(_frames(source.poses), source.fps), height=420)
