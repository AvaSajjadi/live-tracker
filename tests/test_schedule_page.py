"""The "Run schedule" part of the Rules page, driven with Streamlit's AppTest (no browser)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "1_Rules.py")


def texts(elements):
    return " ".join(e.value for e in elements)


def button(at, label):
    return [b for b in at.button if b.label == label]


@pytest.fixture
def at(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_TRACKER_SCHEDULE_DIR", str(tmp_path / "schedules"))
    app = AppTest.from_file(PAGE, default_timeout=30)
    app.run()
    assert not app.exception
    return app


def test_the_section_is_on_the_rules_page_and_says_what_is_not_connected(at):
    assert "Run schedule" in texts(at.subheader)
    assert "No schedule decided yet." in texts(at.caption)
    assert "not connected yet" in texts(at.caption)
    assert at.radio(key="schedule_mode").value == "For N hours"


def test_propose_for_hours_shows_a_preview_with_confirm_and_reject(at):
    at.number_input(key="schedule_hours").set_value(8).run()
    button(at, "Propose schedule")[0].click().run()
    assert "Proposed schedule S1" in texts(at.markdown)
    assert "Not active until you click Confirm" in texts(at.warning)
    assert "8 h 00 min" in texts(at.markdown)
    assert len(button(at, "Confirm")) >= 1 and len(button(at, "Reject")) >= 1


def test_a_run_of_only_minutes_can_be_proposed(at):
    at.number_input(key="schedule_hours").set_value(0)
    at.number_input(key="schedule_minutes").set_value(4).run()
    button(at, "Propose schedule")[0].click().run()
    assert not at.exception and "0 h 04 min" in texts(at.markdown)


def test_confirm_adds_an_approved_row_and_saves_a_file(at, tmp_path):
    button(at, "Propose schedule")[0].click().run()
    [b for b in at.button if b.label == "Confirm" and b.key.startswith("schedule_confirm_")][0].click().run()
    assert not at.exception
    row = at.dataframe[-1].value.iloc[0]
    assert row["schedule"] == "S1" and row["status"] == "approved" and row["length"] == "8 h 00 min"
    assert len(list((tmp_path / "schedules").glob("*.json"))) == 1
    assert "Proposed schedule" not in texts(at.markdown)


def test_reject_adds_a_rejected_row_and_saves_nothing(at, tmp_path):
    button(at, "Propose schedule")[0].click().run()
    [b for b in at.button if b.label == "Reject" and b.key.startswith("schedule_reject_")][0].click().run()
    assert at.dataframe[-1].value.iloc[0]["status"] == "rejected"
    assert not (tmp_path / "schedules").exists()


def test_from_to_window_across_midnight_is_twelve_hours(at):
    at.radio(key="schedule_mode").set_value("From - To").run()
    button(at, "Propose schedule")[0].click().run()
    assert "12 h 00 min" in texts(at.markdown) and "Proposed schedule S1" in texts(at.markdown)


def test_changing_an_input_drops_the_proposal(at):
    button(at, "Propose schedule")[0].click().run()
    assert "Proposed schedule S1" in texts(at.markdown)
    at.number_input(key="schedule_hours").set_value(10).run()
    assert "Proposed schedule" not in texts(at.markdown)
    assert not [b for b in at.button if b.key and b.key.startswith("schedule_confirm_")]


def test_an_equal_from_and_to_is_refused_with_a_message(at):
    at.radio(key="schedule_mode").set_value("From - To").run()
    from datetime import time
    at.time_input(key="schedule_to").set_value(time(18, 0)).run()
    button(at, "Propose schedule")[0].click().run()
    assert "same" in texts(at.error) and "Proposed schedule" not in texts(at.markdown)
    assert not [b for b in at.button if b.key and b.key.startswith("schedule_confirm_")]


def test_start_at_a_chosen_time_shows_the_date_and_time_inputs(at):
    at.radio(key="schedule_start_choice").set_value("At a chosen time").run()
    assert at.date_input(key="schedule_start_day") is not None and at.time_input(key="schedule_start_clock") is not None
