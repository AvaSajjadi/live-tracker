"""Smoke tests for the Streamlit app (new file).

Uses Streamlit's built-in AppTest, so no browser is needed.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "pages" / "8_Motion_Tool.py")


def run_demo(inject=False):
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.checkbox[0].check().run()  # synthetic demo
    if inject:
        at.checkbox[1].check().run()  # inside "Advanced settings"
    assert not at.exception
    return at


def test_app_starts_without_data():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception


def test_demo_is_labeled_synthetic():
    at = run_demo()
    assert any("SYNTHETIC" in w.value for w in at.warning)
    assert any("No tracking loss" in s.value for s in at.success)


def test_injected_failures_show_warnings():
    at = run_demo(inject=True)
    assert any("Tracking lost for mouse b" in e.value for e in at.error)
    assert any("stuck for mouse a" in w.value for w in at.warning)


def test_starter_section_is_labeled_offline_only():
    at = run_demo()
    assert any("OFFLINE ONLY" in s.value for s in at.subheader)


def test_engine_settings_are_not_shown_but_advanced_settings_are():
    at = run_demo()
    assert len(at.sidebar.number_input) == 0 and len(at.sidebar.slider) == 0   # nothing in the sidebar
    labels = {w.label for w in list(at.number_input) + list(at.slider) + list(at.checkbox)}
    assert not any("One-Euro" in l or "Frozen if" in l or "Jump if" in l or "Quality window" in l for l in labels)
    assert [e.label for e in at.expander] == ["Advanced settings"]
    assert {"Warn if window quality below", "Report tracking loss longer than (s)"} <= labels


def test_upload_box_carries_the_file_format_and_no_extra_intro_lines():
    at = AppTest.from_file(APP, default_timeout=30).run()
    page_text = " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
    assert "Milestone 1" not in page_text and "Upload a CSV with columns" not in page_text
    assert "OFFLINE ONLY: never for live" not in page_text


def test_input_labels_are_in_plain_words():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.number_input[0].label == "Recording frame rate (frames per second)"
    assert "how many pictures of the recording there are in each second" in at.number_input[0].help
    assert at.checkbox[0].label == "Try synthetic demo data"
    assert any("x_a, y_a (mouse a) and x_b, y_b (mouse b)" in c.value for c in at.caption)
    labels = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "CalMS21" not in labels
