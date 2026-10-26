"""Front door of the dashboard: the guided flow first, the other pages after it (router, new file).

`streamlit run app.py` shows the menu defined in live_tracker.flow.navigation_pages(): the four steps in order
(Upload, Rules, Replay, Real session), then the other pages. Each page is its own file in pages/ and can still
be tested on its own.
"""
import streamlit as st

from live_tracker import flow

st.navigation(flow.navigation_pages()).run()
