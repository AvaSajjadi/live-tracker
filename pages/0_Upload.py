"""Upload page: step 1 of the guided flow (new file). Rendering only; see live_tracker.upload_ui."""

import streamlit as st

from live_tracker import ui_style, upload_ui

st.set_page_config(page_title="Upload", layout="wide")
ui_style.apply()
upload_ui.render()
