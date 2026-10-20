"""Tests for the Replay page (pages/2_Replay.py), D3-2. New file.

Streamlit AppTest (no browser): what the page shows at each replay position,
the controls, answers and errors. The page logic itself is tested in
test_replay_view.py and test_replay.py. Rules are approved through the real
RuleWorkflow (offline parser, then Confirm). SYNTHETIC data only; journals go
to a temporary directory (LIVE_TRACKER_REPLAY_DIR). No network, no API key.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import live_tracker.replay as rp
import live_tracker.replay_view as rv
from live_tracker.rule_workflow import RuleWorkflow

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "2_Replay.py")
DEMO, TRACK = (e.label for e in rv.synthetic_catalog())


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(tmp_path / "replay"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LIVE_TRACKER_CLAUDE_MODEL", raising=False)


def workflow(*instructions):
    wf = RuleWorkflow()
    for text in instructions or ("Alert me when the mice attack for at least 2 seconds",):
        wf.submit(text)
        wf.confirm(wf.pending.token)
    return wf


def start(wf=None, **state):
    at = AppTest.from_file(PAGE, default_timeout=60)
    if wf is not None:
        at.session_state["rule_workflow"] = wf
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def run_id(at):
    return at.session_state["replay_active_run"]


def seek(at, frame):
    at.session_state[f"replay_frame_{run_id(at)}"] = frame
    at.run()
    assert not at.exception, at.exception
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception
    return at


def labels(at):
    return [b.label for b in at.button]


def text(elements):
    return " | ".join(str(e.value) for e in elements)


def table(at, column):
    """The first dataframe that has `column`."""
    for df in at.dataframe:
        if column in df.value.columns:
            return df.value
    return None


def frame(at):
    return at.session_state[f"replay_frame_{run_id(at)}"]


# ======================================================================
# rules, banner and sessions
# ======================================================================

def test_no_approved_rule_is_an_error():
    at = start()
    assert "No approved rule" in text(at.error) and not at.slider
    pending = RuleWorkflow()
    pending.submit("Alert me when the mice attack for at least 2 seconds")        # proposed, never confirmed
    at = start(pending)
    assert "No approved rule" in text(at.error)


def test_banner_and_synthetic_sessions():
    at = start(workflow())
    assert at.warning[0].value == "Replay of a recorded session, not a live camera"
    assert "SYNTHETIC test data" in text(at.warning)
    assert at.selectbox[0].options == [DEMO, TRACK]
    assert "SYNTHETIC_demo_01" in text(at.caption)
    assert at.slider[0].value == 0 and at.slider[0].min == 0 and at.slider[0].max == 719


# ======================================================================
# seeking, playback and event visibility
# ======================================================================

def test_events_appear_only_once_they_fire():
    at = seek(start(workflow()), 119)
    assert "No event has fired yet." in text(at.caption) and table(at, "fired") is None
    at = seek(at, 120)
    assert "ALERT" in text(at.error) and "attack (rule R1)" in text(at.error)       # announced on its frame
    rows = table(at, "fired")
    assert list(rows["action"]) == ["alert"] and rows["closure"][0] == "open (not ended yet)"
    assert rows["onset"][0] == "2.00 s (frame 60)" and rows["fired"][0] == "4.00 s (frame 120)"
    at = seek(at, 130)
    assert "ALERT" not in text(at.error)                                           # only on the firing frame
    at = seek(at, 150)
    assert table(at, "fired")["closure"][0] == "5.00 s (below_threshold, last frame 149)"
    at = seek(at, 100)                                                             # seeking back hides it again
    assert table(at, "fired") is None


def test_frame_time_and_tracking_status_are_shown():
    at = seek(start(workflow()), 490)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Frame"] == "490 (491/720)" and metrics["Time"] == "16.33 s" and metrics["Tracking"] == "LOST"
    assert rp.TRACKING_GAP_NOTE in text(at.warning)
    assert "not drawn as tracked" in text(at.warning)
    at = seek(at, 30)
    assert {m.label: m.value for m in at.metric}["Tracking"] == "ok" and rp.TRACKING_GAP_NOTE not in text(at.warning)


def test_navigation_buttons():
    at = start(workflow())
    assert frame(click(at, "+1 s")) == 30
    assert frame(click(at, "+1 frame")) == 31
    assert frame(click(at, "−1 frame")) == 30
    assert frame(click(at, "−1 s")) == 0
    assert frame(click(at, "−1 frame")) == 0                                       # never before the first row
    click(at, "End of session")
    assert "End of session" in text(at.info) and {m.label: m.value for m in at.metric}["Frame"] == "end"
    assert list(table(at, "fired")["action"]) == ["alert", "ask"]
    assert frame(click(at, "⏮ Start")) == 0 and "End of session" not in text(at.info)


def test_recorded_speed_playback(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(rv, "now", lambda: clock["t"])
    at = start(workflow())
    click(at, "Play")
    assert at.session_state[f"replay_playing_{run_id(at)}"] is True and "Pause" in labels(at)
    clock["t"] += 1.0
    at.run()                                                                       # one tick: 1 s at 1x
    assert frame(at) == 30
    at.radio(key="replay_speed").set_value("5×").run()                             # full rerun, still playing
    clock["t"] += 1.0
    at.run()
    assert frame(at) == 180
    clock["t"] += 100.0
    at.run()
    assert frame(at) == 719 and at.session_state[f"replay_playing_{run_id(at)}"] is False   # stops at the end
    click(at, "Play")                                                              # plays again from the start
    assert frame(at) == 0
    click(at, "Pause")
    clock["t"] += 5.0
    at.run()
    assert frame(at) == 0


# ======================================================================
# answers
# ======================================================================

def test_confirm_an_ask_and_its_states(tmp_path):
    at = seek(start(workflow()), 260)
    assert "Confirm" not in labels(at)                                             # the ask fires on frame 270
    at = seek(at, 280)
    assert "Unanswered" in text(at.markdown)
    click(at, "Confirm")
    assert "Answered: confirmed at frame 280" in text(at.markdown) and "Confirm" not in labels(at)
    assert table(at, "fired")["answer"][1] == "confirmed"
    at = seek(at, 275)                                                             # before the answer was given
    assert "given later in this replay" in text(at.markdown) and "Confirm" not in labels(at)
    assert table(at, "fired")["answer"][1] == ""
    directory = tmp_path / "replay"
    (answers,) = directory.glob("*.answers.jsonl")
    rec = json.loads(answers.read_text())
    assert rec["answered_frame"] == 280 and rec["researcher_answer"] == "confirmed"


def test_answers_persist_and_unknown_timing_is_explicit(tmp_path):
    at = seek(start(workflow()), 290)
    click(at, "Reject")
    again = seek(start(workflow()), 290)                                           # a new browser session
    assert "Answered: rejected at frame 290" in text(again.markdown)
    for f in (tmp_path / "replay").glob("*.answers.jsonl"):
        f.unlink()                                                                 # positions lost
    blind = seek(start(workflow()), 300)
    assert "when it was given is not known" in text(blind.markdown) and "Reject" not in labels(blind)
    click(blind, "End of session")
    assert "Answered: rejected (replay time unknown)" in text(blind.markdown)


def test_an_answer_from_another_tab_is_seen_and_never_contradicted():
    tab1, tab2 = seek(start(workflow()), 280), seek(start(workflow()), 300)        # two browser sessions
    click(tab1, "Confirm")
    tab2.run()                                                                     # its next rerun re-reads
    assert "Answered: confirmed at frame 280" in text(tab2.markdown) and "Reject" not in labels(tab2)


def test_a_stale_button_cannot_write_a_conflicting_answer():
    tab1, tab2 = seek(start(workflow()), 280), seek(start(workflow()), 300)
    replay2 = next(iter(tab2.session_state["replay_cache"].values()))
    eid = replay2.end_state().events[1]["event_id"]
    click(tab1, "Confirm")
    with pytest.raises(rp.ReplayError, match="answers are final"):                # the stale tab, directly
        replay2.answer(eid, "rejected", at_frame=300)
    assert replay2.answer_status(eid)["researcher_answer"] == "confirmed"
    tab3 = start(workflow())                                                       # the journal is still valid
    assert not tab3.error


# ======================================================================
# tracking warnings and skeletons
# ======================================================================

def test_tracking_warnings_panel():
    at = start(workflow())
    at.selectbox(key="replay_session").set_value(TRACK).run()
    at = seek(at, 100)
    assert "No tracking warning so far." in text(at.caption)
    at = seek(at, 125)
    w = table(at, "kind")
    assert list(w["kind"]) == ["missing_coordinates"] and list(w["end"]) == ["ongoing"]
    assert "Kept separate from the event log" in text(at.caption)
    at = seek(at, 300)
    w = table(at, "kind")
    assert {"missing_coordinates", "low_confidence"} <= set(w["kind"])
    assert w.loc[w["kind"] == "missing_coordinates", "end"].item() == "4.50 s"
    at = seek(at, 255)
    assert "mouse b: 3 low confidence" in text(at.warning)


def test_skeleton_image_and_legend():
    at = seek(start(workflow()), 30)
    assert len(at.get("image")) == 1
    assert "PROVISIONAL" in text(at.caption) and "hollow amber: low confidence" in text(at.caption)
    click(at, "End of session")
    assert len(at.get("image")) == 0


# ======================================================================
# state isolation and cache invalidation
# ======================================================================

def test_sessions_keep_their_own_state():
    at = seek(start(workflow()), 280)
    click(at, "Confirm")
    demo_run = run_id(at)
    at.selectbox(key="replay_session").set_value(TRACK).run()
    assert run_id(at) != demo_run and frame(at) == 0
    assert "Answered" not in text(at.markdown) and table(at, "fired") is None
    at = seek(at, 280)
    track = next(r for r in at.session_state["replay_cache"].values() if r.session == rv.TRACKING_SESSION)
    assert track.run_id == run_id(at) and all(e["evidence"]["run_id"] == run_id(at) for e in track.state_at(280).events)
    assert "Answered: confirmed" not in text(at.markdown)
    at.selectbox(key="replay_session").set_value(DEMO).run()
    assert run_id(at) == demo_run and frame(at) == 280 and "Answered: confirmed at frame 280" in text(at.markdown)


def test_switching_sessions_stops_playback(monkeypatch):
    monkeypatch.setattr(rv, "now", lambda: 0.0)
    at = start(workflow())
    click(at, "Play")
    demo_run = run_id(at)
    at.selectbox(key="replay_session").set_value(TRACK).run()
    assert at.session_state[f"replay_playing_{demo_run}"] is False and "Play" in labels(at)


def test_a_newly_approved_rule_is_a_new_run():
    wf = workflow()
    at = seek(start(wf), 400)
    first = run_id(at)
    wf.submit("Alert me when the mice attack for at least 1 second")
    wf.confirm(wf.pending.token)
    at.run()
    assert run_id(at) != first
    at = seek(at, 400)
    assert set(table(at, "fired")["rule"]) == {"R1", "R2"}


# ======================================================================
# missing and invalid data
# ======================================================================

def _catalog(monkeypatch, edit):
    entries = rv.synthetic_catalog()
    monkeypatch.setattr(rv, "synthetic_catalog", lambda: (edit(entries[0]),) + entries[1:])


@pytest.mark.parametrize("edit, match", [
    (lambda e: dataclasses.replace(e, session="task1/test/mouse001"), "not a SYNTHETIC session"),
    (lambda e: dataclasses.replace(e, scores=e.scores.drop(columns=["tracking_ok"])), "lacks columns"),
    (lambda e: dataclasses.replace(e, scores=e.scores.iloc[0:0]), "no synthetic score rows"),
    (lambda e: dataclasses.replace(e, scores=e.scores.assign(attack="high")), "could not be built"),
])
def test_invalid_score_data_is_an_error(monkeypatch, edit, match):
    _catalog(monkeypatch, edit)
    at = start(workflow())
    assert match in text(at.error) and not at.slider


def test_invalid_pose_is_an_error_but_events_still_replay(monkeypatch):
    _catalog(monkeypatch, lambda e: dataclasses.replace(e, pose=e.pose.drop(columns=["score"])))
    at = seek(start(workflow()), 400)
    assert "Skeletons and tracking warnings are unavailable" in text(at.error)
    assert "lacks columns" in text(at.error) and len(at.get("image")) == 0
    assert list(table(at, "fired")["action"]) == ["alert", "ask"]


def test_a_broken_catalog_is_an_error(monkeypatch):
    def broken():
        raise RuntimeError("fixture failed")
    monkeypatch.setattr(rv, "synthetic_catalog", broken)
    at = start(workflow())
    assert "could not be built (RuntimeError: fixture failed)" in text(at.error)


def test_an_unusable_journal_directory_is_an_error(tmp_path, monkeypatch):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    monkeypatch.setenv(rv.REPLAY_DIR_ENV, str(blocker / "replay"))
    at = start(workflow())
    assert "journal directory" in text(at.error) and not at.slider


def test_a_corrupt_answers_file_is_an_error(tmp_path):
    at = seek(start(workflow()), 280)
    click(at, "Confirm")
    (answers,) = (tmp_path / "replay").glob("*.answers.jsonl")
    rec = json.loads(answers.read_text())
    answers.write_text(json.dumps({**rec, "researcher_answer": "rejected"}) + "\n")
    at = start(workflow())
    assert "could not be built" in text(at.error) and "contradicts the journal" in text(at.error)


def test_the_frame_slider_seeks():
    at = start(workflow())
    at.slider[0].set_value(150).run()
    assert not at.exception and frame(at) == 150
    assert table(at, "fired")["closure"][0] == "5.00 s (below_threshold, last frame 149)"
    click(at, "End of session")
    at.slider[0].set_value(100).run()                                              # leaves the end-of-session view
    assert frame(at) == 100 and "End of session" not in text(at.info) and table(at, "fired") is None


# ======================================================================
# Codex D3-2 review: content-bound cache, playback timing, malformed pose
# ======================================================================

def _replace_demo(monkeypatch, edit):
    entries = rv.synthetic_catalog()
    monkeypatch.setattr(rv, "synthetic_catalog", lambda: (edit(entries[0]),) + entries[1:])


def test_replaced_scores_are_a_new_run_never_the_cached_one(monkeypatch):
    at = seek(start(workflow()), 400)
    first = run_id(at)
    assert list(table(at, "fired")["action"]) == ["alert", "ask"]
    _replace_demo(monkeypatch, lambda e: dataclasses.replace(e, scores=e.scores.assign(attack=0.0)))
    at.run()
    assert not at.exception and run_id(at) != first
    assert frame(at) == 0 and table(at, "fired") is None                        # a new run starts from its start
    at = seek(at, 400)
    assert table(at, "fired") is None and "No event has fired yet." in text(at.caption)


def test_malformed_replacement_scores_are_an_error_not_the_cached_run(monkeypatch):
    at = seek(start(workflow()), 400)
    assert table(at, "fired") is not None
    _replace_demo(monkeypatch, lambda e: dataclasses.replace(e, scores=e.scores.assign(attack="high")))
    at.run()
    assert "could not be built" in text(at.error) and "attack" in text(at.error)
    assert table(at, "fired") is None and not at.slider                          # nothing cached is shown


def test_changed_pose_rebuilds_but_keeps_the_runs_position(monkeypatch):
    at = seek(start(workflow()), 300)
    first = run_id(at)
    old = next(iter(at.session_state["replay_cache"].values()))
    _replace_demo(monkeypatch, lambda e: dataclasses.replace(e, pose=e.pose.assign(x=e.pose["x"] + 40.0)))
    at.run()
    assert run_id(at) == first and frame(at) == 300                              # same events: same run state
    cache = at.session_state["replay_cache"]
    assert len(cache) == 2
    new = next(r for k, r in cache.items() if r is not old)
    xs = lambda r: [k.x for k in r.state_at(300).skeletons[0].keypoints]        # noqa: E731
    assert xs(new) == [x + 40.0 for x in xs(old)]
    _replace_demo(monkeypatch, lambda e: dataclasses.replace(e, pose=e.pose.drop(columns=["score"])))
    at.run()
    assert "Skeletons and tracking warnings are unavailable" in text(at.error) and len(at.get("image")) == 0


def test_malformed_pose_frames_are_an_error(monkeypatch):
    _replace_demo(monkeypatch, lambda e: dataclasses.replace(
        e, pose=e.pose.assign(frame=e.pose["frame"].map(lambda f: f"f{f}"))))
    at = seek(start(workflow()), 400)
    assert "Skeletons and tracking warnings are unavailable" in text(at.error)
    assert "frame numbers are not numeric" in text(at.error) and len(at.get("image")) == 0
    assert list(table(at, "fired")["action"]) == ["alert", "ask"]               # events still replay


@pytest.mark.parametrize("speed, expected", [("1×", 60), ("2×", 120), ("5×", 300)])
def test_eight_quarter_second_ticks_on_the_page(monkeypatch, speed, expected):
    clock = {"t": 100.0}
    monkeypatch.setattr(rv, "now", lambda: clock["t"])
    at = start(workflow())
    at.radio(key="replay_speed").set_value(speed).run()
    click(at, "Play")
    for _ in range(8):
        clock["t"] += 0.25
        at.run()
    assert frame(at) == expected


def test_seek_pause_and_speed_change_re_anchor(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(rv, "now", lambda: clock["t"])
    at = start(workflow())
    click(at, "Play")
    clock["t"] += 1.0
    at.run()
    assert frame(at) == 30
    at.radio(key="replay_speed").set_value("2×").run()                         # re-anchored at 1.0 s
    clock["t"] += 1.0
    at.run()
    assert frame(at) == 90
    click(at, "+1 s")                                                            # seeking stops playback
    assert frame(at) == 120 and at.session_state[f"replay_anchor_{run_id(at)}"] is None
    clock["t"] += 5.0
    at.run()
    assert frame(at) == 120
    click(at, "Play")                                                            # resumes from the new position
    clock["t"] += 0.5
    at.run()
    assert frame(at) == 150


# ======================================================================
# D4-1: the session summary in the end-of-session view only
# ======================================================================

def summary_shown(at):
    return any(s.value == "Session summary" for s in at.subheader)


def test_the_summary_appears_only_at_the_end_of_the_session():
    at = start(workflow())
    for f in (0, 280, 719):                                                     # every frame, even the last row
        at = seek(at, f)
        assert not summary_shown(at) and not at.get("download_button")
    click(at, "End of session")
    assert summary_shown(at) and len(at.get("download_button")) == 2
    md = text(at.markdown)
    assert "# Session summary: SYNTHETIC_demo_01" in md and "**SYNTHETIC test data" in md
    assert "Alerts: 1, asks: 1, logs: 0" in md and "0 confirmed, 0 rejected, 1 unanswered" in md
    assert "Total: 1 (missing_coordinates: 1)" in md                           # the separate warning source
    replay = next(iter(at.session_state["replay_cache"].values()))
    assert replay.run_id in md
    at.slider[0].set_value(300).run()                                          # back into the replay: hidden
    assert frame(at) == 300 and not summary_shown(at) and not at.get("download_button")


def test_the_summary_includes_answers_and_only_this_run():
    at = seek(start(workflow()), 280)
    click(at, "Confirm")
    click(at, "End of session")
    md = text(at.markdown)
    assert "1 confirmed, 0 rejected, 0 unanswered" in md and "confirmed (frame 280)" in md
    at.selectbox(key="replay_session").set_value(TRACK).run()
    click(at, "End of session")
    md = text(at.markdown)
    assert "# Session summary: SYNTHETIC_tracking_02" in md and "confirmed (frame 280)" not in md
    assert "SYNTHETIC_demo_01" not in md


def test_without_a_warning_source_the_summary_says_so(monkeypatch):
    _replace_demo(monkeypatch, lambda e: dataclasses.replace(e, pose=e.pose.drop(columns=["score"])))
    at = start(workflow())
    click(at, "End of session")
    assert "No tracking-warning source supplied." in text(at.markdown)


def test_a_summary_that_cannot_be_written_is_an_error(monkeypatch):
    def refuse(*a, **k):
        raise sm.SummaryError("journal contradiction (test)")
    import live_tracker.summary as sm
    monkeypatch.setattr(sm, "summary_from_replay", refuse)
    at = start(workflow())
    click(at, "End of session")
    assert "The session summary cannot be written: journal contradiction (test)" in text(at.error)
    assert not at.get("download_button")
