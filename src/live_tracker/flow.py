"""The guided flow: Upload -> Rules -> Replay -> Real session, with Back and Next buttons (new file).

Each step is a page. `render_bar()` shows where the researcher is and `render_nav()` the Back and Next buttons.
What the researcher chose on the Upload page lives in st.session_state[SOURCE_KEY] (a SessionSource, see
uploads.py). The pages stay usable on their own from the left menu; the flow only adds direction.
"""

from __future__ import annotations

import streamlit as st

from .uploads import SessionSource

SOURCE_KEY = "flow_source"
STEPS = (
    ("upload", "Upload", "pages/0_Upload.py"),
    ("rules", "Rules", "pages/1_Rules.py"),
    ("replay", "Replay", "pages/2_Replay.py"),
    ("session", "Real session", "pages/4_Real_Session.py"),
)
KEYS = [key for key, _, _ in STEPS]


OTHER_PAGES = (
    ("Investigation", "pages/3_Investigation.py"),
    ("Live Agent", "pages/5_Live_Agent.py"),
    ("Run", "pages/6_Run.py"),
    ("Motion tool", "pages/8_Motion_Tool.py"),
)


def menu():
    """The sidebar menu as plain data: {section: [(title, path, is_default)]}. The numbered steps come first, and
    Upload is the front page."""
    steps = [(f"{i + 1}. {label}", path, i == 0) for i, (_, label, path) in enumerate(STEPS)]
    more = [(title, path, False) for title, path in OTHER_PAGES]
    return {"Guided flow": steps, "More": more}


def navigation_pages():
    """The same menu as st.Page objects, for st.navigation (call it inside a running app)."""
    return {section: [st.Page(path, title=title, default=default) for title, path, default in items]
            for section, items in menu().items()}


def step_index(key):
    return KEYS.index(key)


def next_page(key):
    """Path of the page after `key`, or None on the last step."""
    i = step_index(key)
    return STEPS[i + 1][2] if i + 1 < len(STEPS) else None


def previous_page(key):
    i = step_index(key)
    return STEPS[i - 1][2] if i > 0 else None


def get_source():
    """The source chosen on the Upload page, or None."""
    value = st.session_state.get(SOURCE_KEY)
    return value if isinstance(value, SessionSource) else None


def set_source(source):
    st.session_state[SOURCE_KEY] = source


def bar_text(key):
    """One line such as '1 Upload  >  2 Rules  >  3 Replay  >  4 Real session' with the current step in bold."""
    here = step_index(key)
    parts = []
    for i, (_, label, _) in enumerate(STEPS):
        text = f"{i + 1}. {label}"
        parts.append(f"**{text}**" if i == here else (f"~~{text}~~" if i < here else text))
    return "  >  ".join(parts)


def render_bar(key):
    st.markdown(bar_text(key))


def render_nav(key, can_next=True, next_hint=None):
    """Back and Next buttons. Next is disabled (with a hint) until the step is complete."""
    back, nxt = previous_page(key), next_page(key)
    left, right = st.columns(2)
    if back and left.button("Back", key=f"flow_back_{key}"):
        st.switch_page(back)
    if nxt:
        if right.button("Next", key=f"flow_next_{key}", type="primary", disabled=not can_next):
            st.switch_page(nxt)
        if not can_next and next_hint:
            right.caption(next_hint)
