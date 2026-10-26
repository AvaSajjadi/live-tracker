"""The live camera picture with the keypoints the pose model finds, drawn on it (new file).

The Replay step for a real camera: there is no recording, so it shows what the pose model sees NOW. Only shown, never scored:
alerts and the report come from the Real session run, which opens the camera in its own program.
"""

from __future__ import annotations

import streamlit as st

from . import camera_live
from . import pose_model as pm
from .pose_preview import draw, to_frame

REFRESH_S = 0.15
BLACK_LEVEL = 3.0                                     # mean pixel value below this: a black picture, not a dark cage


@st.cache_resource(show_spinner="Loading the pose model...")
def _estimator():
    return pm.load()


def render(source):
    estimator, why = _estimator()
    if estimator is None:
        st.error(why)
        return
    live_panel(source.device, source.screen)


@st.fragment(run_every=REFRESH_S)
def live_panel(device, screen):
    import cv2
    image, why = camera_live.read(device)
    if image is None:
        st.error(f"The camera cannot be shown: {why}.")
        return
    if image.mean() < BLACK_LEVEL:                    # the Nikon sends black pictures when it sleeps or leaves live view
        st.warning("The camera sends a black picture. Wake it (press the shutter button halfway) or switch it off and on.")
        return
    estimator, _ = _estimator()
    frame = to_frame(image, screen)
    try:
        found = pm.check_output(estimator(frame))
    except pm.PoseModelError as exc:
        st.error(f"The pose model answer cannot be used: {exc}.")
        return
    mice = sorted(found["mouse"].unique()) if found is not None else []
    st.image(cv2.cvtColor(draw(frame, found), cv2.COLOR_BGR2RGB), width="stretch", output_format="JPEG",
             caption=f"Mice found: {', '.join(mice) or 'none'}")
