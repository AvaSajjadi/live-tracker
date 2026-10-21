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


def test_rules_to_replay_to_answer_to_summary(tmp_path, monkeypatch):
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LIVE_TRACKER_CLAUDE_MODEL", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()

    at.switch_page("pages/1_Rules.py").run()                     # sentence -> proposal -> Confirm
    at.text_area(key="rule_instruction").input(INSTRUCTION).run()
    button(at, "Propose rule").click().run()
    assert "Proposed rule R1" in " ".join(s.value for s in at.subheader)
    button(at, "Confirm").click().run()
    assert any("Monitoring can start: 1 approved rule." in s.value for s in at.success)

    at.switch_page("pages/2_Replay.py").run()                    # the same session keeps the rule
    assert not at.exception and at.warning[0].value == "Replay of a recorded session, not a live camera"
    run = at.session_state["replay_active_run"]
    at.session_state[f"replay_frame_{run}"] = 280                 # the ask fired at frame 270 (9.0 s)
    at.run()
    button(at, "Confirm").click().run()
    assert any("Answered: confirmed at frame 280" in m.value for m in at.markdown)
    button(at, "End of session").click().run()
    shown = " ".join(m.value for m in at.markdown)
    assert "# Session summary: SYNTHETIC_demo_01" in shown and "1 confirmed, 0 rejected, 0 unanswered" in shown
    assert len(at.get("download_button")) == 2

    (journal,) = tmp_path.glob(f"{run}.journal.jsonl")             # what persisted, read back independently
    state = el.load_journal(journal)
    assert [r["researcher_answer"] for r in state.records if r["record"] == "event_answered"] == ["confirmed"]
    again = sm.build_summary(state.records, run_id=run, session="SYNTHETIC_demo_01").as_dict()
    assert again["counts"] == {"alert": 1, "ask": 1, "log": 0} and again["answers"]["confirmed"] == 1
    assert again["provenance"]["label"] == "SYNTHETIC"
