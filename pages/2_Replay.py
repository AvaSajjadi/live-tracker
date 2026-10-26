"""Read-only recorded video/pose inspection, or the live camera with its pose; monitoring belongs to Real session and Run."""
from pathlib import Path

import streamlit as st

from live_tracker import flow, live_pose_view, ui_style
from live_tracker import skeleton_player as sp
from live_tracker import video_web as vw

ui_style.apply()
flow.render_bar("replay")
st.title("Replay")
flow.render_nav("replay")


@st.cache_data(show_spinner="Reading the recorded keypoints...", max_entries=2)
def _frames(poses, session, stamp, require_observed=False):
    return sp.pose_frames(poses, session=session, require_observed=require_observed)


source = flow.get_source()
if source is not None and source.kind == "camera":
    live_pose_view.render(source)
    st.stop()
st.caption("Inspect recorded video with the matching recorded keypoints. This page does not run monitoring.")
if source is None or not source.has_video or not source.poses:
    st.info("Nothing to replay yet. On the Upload step, choose a video that has a keypoints file.")
    st.stop()
playable = vw.playable_video(source.video)
if playable is None:
    st.warning("This recording's codec cannot be shown in the browser; no synchronized skeleton is claimed. "
               "Replay never converts or changes the source video.")
    st.stop()
try:
    stat = Path(source.poses).stat()
    frames = _frames(source.poses, source.session, (stat.st_mtime_ns, stat.st_size),
                     source.kind == "video_keypoints")
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Cannot display this pose source: {exc}")
    st.stop()
sp.render(playable, frames, source.fps, session=source.session, poses_path=source.poses,
          original_video=source.video)
