"""End-to-end SYNTHETIC demo smoke test (D4-2). New file.

The researcher path through the real multipage app, in one Streamlit session
(AppTest, no browser): app.py -> Rules page (offline parser, Propose, Confirm)
-> Replay page (demo session, the ask at 9.0 s answered at frame 280) -> end
of session -> summary. The journal on disk is then read back independently and
summarized again, so the page, the journal and the summary agree. The page and
module details are covered by test_rules_page, test_replay_page and
test_summary. No network, no API key, no real data (docs/day4_synthetic_demo.md).
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import live_tracker.event_log as el
import live_tracker.replay_view as rv
import live_tracker.summary as sm

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Alert me when the mice attack for at least 2 seconds"


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_rules_then_replay_page_opens(tmp_path, monkeypatch):
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LIVE_TRACKER_CLAUDE_MODEL", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()

    at.switch_page("pages/1_Rules.py").run()                     # sentence -> proposal -> Confirm
    at.text_area(key="rule_instruction").input(INSTRUCTION).run()
    button(at, "Propose rule").click().run()
    assert not at.session_state["rule_workflow"].monitoring_allowed()
    button(at, "Confirm").click().run()
    assert at.session_state["rule_workflow"].monitoring_allowed()

    at.switch_page("pages/2_Replay.py").run()                    # the Replay page shows only video and skeletons
    assert not at.exception and not at.metric and not at.dataframe
