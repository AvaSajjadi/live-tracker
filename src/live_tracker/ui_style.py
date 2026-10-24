"""Shared look for the Streamlit pages: one stylesheet, status badges, short notices.

Colors: green = ok, amber = ask / needs a look, red = alert / error, grey = neutral.
`note()` shows one short line and keeps the full text in a "More info" expander,
so long explanations stay available without crowding the page.
"""

from __future__ import annotations

import streamlit as st

COLORS = {"ok": "#15803D", "ask": "#B45309", "alert": "#B91C1C", "neutral": "#475569", "accent": "#0F766E"}
_BG = {"ok": "#DCFCE7", "ask": "#FEF3C7", "alert": "#FEE2E2", "neutral": "#E2E8F0", "accent": "#CCFBF1"}

_CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 70rem;}
h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: .2rem;}
h2, h3 {font-weight: 650; letter-spacing: -0.01em;}
div[data-testid="stMetric"] {background: #F1F5F4; border: 1px solid #DDE6E4; border-radius: .7rem; padding: .6rem .9rem;}
div[data-testid="stMetricLabel"] p {font-size: .8rem; color: #475569;}
div[data-testid="stAlert"] {border-radius: .7rem; padding: .55rem .9rem; font-size: .92rem;}
div[data-testid="stCaptionContainer"] {font-size: .82rem;}
details {border-radius: .7rem !important;}
.lt-badge {display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600;}
</style>
"""


def apply():
    """Call once at the top of every page, after st.set_page_config if the page uses it."""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text, kind="neutral"):
    """Small colored label as HTML for st.markdown(..., unsafe_allow_html=True)."""
    return (f'<span class="lt-badge" style="background:{_BG[kind]};color:{COLORS[kind]}">{text}</span>')


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
