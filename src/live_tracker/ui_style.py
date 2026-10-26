"""Shared look for the Streamlit pages: one stylesheet, status badges, short notices.

Colors: green = ok, amber = ask / needs a look, red = alert / error, grey = neutral,
terracotta = the accent of the app.
`note()` shows one short line and keeps the full text in a "More info" expander,
so long explanations stay available without crowding the page.
"""

from __future__ import annotations

from html import escape

import streamlit as st

COLORS = {"ok": "#15803D", "ask": "#B45309", "alert": "#B91C1C", "neutral": "#475569", "accent": "#B76A54"}
_BG = {"ok": "#DCFCE7", "ask": "#FEF3C7", "alert": "#FEE2E2", "neutral": "#E2E8F0", "accent": "#F3E3DD"}

_CSS = """
<style>
/* less empty space at the top of the page and the sidebar */
[data-testid="stMainBlockContainer"], .block-container {padding-top: 3rem !important; padding-bottom: 3rem; max-width: 70rem;}
h1 {padding-top: 0 !important; font-weight: 700; letter-spacing: -0.02em; margin-bottom: .2rem;}
h2, h3 {font-weight: 650; letter-spacing: -0.01em;}
[data-testid="stSidebarHeader"] {height: 1.5rem; padding: 0 1rem;}
[data-testid="stSidebarNav"] {padding-top: 0;}
[data-testid="stSidebarUserContent"] {padding-top: 1rem;}

/* system font: no download, works offline; icon fonts are left alone */
.stApp :where(p, h1, h2, h3, h4, h5, h6, label, li, td, th, a, input, textarea, button, pre) {
  font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded {
  font-family: "Material Symbols Rounded" !important;
}

div[data-testid="stMetric"] {background: #F1F5F4; border: 1px solid #DDE6E3; border-radius: .7rem; padding: .6rem .9rem;}
div[data-testid="stMetricLabel"] p {font-size: .8rem; color: #475569;}
div[data-testid="stAlert"] {border-radius: .7rem; padding: .55rem .9rem; font-size: .92rem;}
div[data-testid="stCaptionContainer"] {font-size: .82rem;}
details {border-radius: .7rem !important; border-color: #DDE6E3 !important;}
.lt-badge {display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600;}
.lt-badge {white-space: normal; overflow-wrap: anywhere;}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .block-container {padding: 2rem 1rem !important;}
  [data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: .75rem;}
  [data-testid="stColumn"] {min-width: min(100%, 16rem); flex: 1 1 16rem;}
  h1 {font-size: 2rem;}
}
</style>
"""


def apply():
    """Call once at the top of every page, after st.set_page_config if the page uses it."""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text, kind="neutral"):
    """Small colored label as HTML for st.markdown(..., unsafe_allow_html=True)."""
    return (f'<span class="lt-badge" style="background:{_BG[kind]};color:{COLORS[kind]}">{escape(str(text))}</span>')


_ACTION_STYLE = {"alert": f"background-color:{_BG['alert']};color:{COLORS['alert']};font-weight:600",
                 "ask": f"background-color:{_BG['ask']};color:{COLORS['ask']};font-weight:600",
                 "log": f"background-color:{_BG['neutral']};color:{COLORS['neutral']};font-weight:600"}


def color_actions(df, column="action"):
    """A table with the alert / ask / log cells colored (red / amber / grey). Same data, only the look."""
    if column not in df.columns:
        return df
    return df.style.map(lambda v: _ACTION_STYLE.get(str(v), ""), subset=[column])


def note(kind, short, more=None):
    """kind: info | warning | error | success | caption. `more` goes into a collapsed expander."""
    getattr(st, kind)(short)
    if more:
        with st.expander("More info"):
            st.caption(more)


def event_notice(event, time_text):
    """Consistent event severity; confidence is a score, never verified behavior."""
    action = event["action"]
    label = {"alert": "ALERT", "ask": "ASK · researcher review", "log": "LOG"}.get(action, action.upper())
    text = (f"{label} · possible {event['behavior']} at {time_text} · "
            f"confidence {event['confidence']:.2f} (model score, not verified behavior)")
    note({"alert": "error", "ask": "warning"}.get(action, "caption"), text)
