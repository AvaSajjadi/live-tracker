"""Live demo page: a recorded video and its run, started together (new file).

Rendering only; see live_tracker.demo_ui and live_tracker.demo. A simulated live stream from a recording,
not a camera.
"""

import streamlit as st

from live_tracker import demo_ui

from live_tracker import ui_style

st.set_page_config(page_title="Live demo", layout="wide")
ui_style.apply()
demo_ui.render()
