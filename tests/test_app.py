"""Smoke tests for the Streamlit app (new file).

Uses Streamlit's built-in AppTest, so no browser is needed.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_demo(inject=False):
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.checkbox[0].check().run()  # synthetic demo
    if inject:
        at.sidebar.checkbox[0].check().run()
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
    assert any("never for live" in c.value for c in at.caption)
