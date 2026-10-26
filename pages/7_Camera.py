"""Camera page: connect a USB camera, set the screen corners, analyze it live or record and analyze (new file).

Rendering only; see live_tracker.camera_ui. The Upload page (recorded data) is Syed's; the camera screens live here."""

import streamlit as st

from live_tracker import camera_ui, ui_style

st.set_page_config(page_title="Camera", layout="wide")
ui_style.apply()
camera_ui.render()
