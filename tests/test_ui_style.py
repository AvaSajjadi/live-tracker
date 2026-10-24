"""Tests for ui_style: colors by meaning, short notices with the full text kept under More info."""

import pandas as pd
from streamlit.testing.v1 import AppTest

from live_tracker import ui_style


def test_badge_uses_the_color_of_its_kind():
    html = ui_style.badge("ALERT", "alert")
    assert "ALERT" in html and ui_style.COLORS["alert"] in html


def test_color_actions_changes_the_look_not_the_data():
    df = pd.DataFrame({"action": ["alert", "ask", "log", "other"], "n": [1, 2, 3, 4]})
    styled = ui_style.color_actions(df)
    pd.testing.assert_frame_equal(styled.data, df)
    css = styled._compute().ctx
    colored = {row for (row, col) in css if col == 0 and css[(row, col)]}
    assert colored == {0, 1, 2}                       # "other" stays unstyled


def test_color_actions_ignores_a_table_without_the_column():
    df = pd.DataFrame({"n": [1]})
    assert ui_style.color_actions(df) is df


def test_note_shows_the_short_line_and_keeps_the_long_text():
    def app():
        from live_tracker import ui_style as u
        u.note("info", "Short line.", "The long explanation.")

    at = AppTest.from_function(app).run()
    assert not at.exception
    assert [i.value for i in at.info] == ["Short line."]
    assert [c.value for c in at.caption] == ["The long explanation."]
    assert [e.label for e in at.expander] == ["More info"]


def test_note_without_more_has_no_expander():
    def app():
        from live_tracker import ui_style as u
        u.note("caption", "Only a short line.")

    at = AppTest.from_function(app).run()
    assert [c.value for c in at.caption] == ["Only a short line."] and len(at.expander) == 0
