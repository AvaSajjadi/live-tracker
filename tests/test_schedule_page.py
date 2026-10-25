"""The "Run schedule" part was removed from the Rules page on request (the schedule modules stay)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "1_Rules.py")


def test_the_rules_page_has_no_run_schedule_section():
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.run()
    assert not at.exception
    assert "Run schedule" not in " ".join(s.value for s in at.subheader)
    assert not [b for b in at.button if b.label == "Propose schedule"]
