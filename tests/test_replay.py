"""Tests for the deterministic causal replay model (replay.py) and the skeleton
geometry (skeleton.py), D3-1. New file.

SYNTHETIC data only: fake_scores (the labelled demo session and small built
sessions) and a small generated pose table; every session ID starts with
SYNTHETIC_. No CalMS21 file, cache, score archive, model or network.
"""

import copy

import numpy as np
import pandas as pd
import pytest

import live_tracker.event_log as el
import live_tracker.replay as rp
from live_tracker.calms21 import KEYPOINTS, TORSO_KEYPOINTS
from live_tracker.fake_scores import DEMO_SESSION, Segment, TrackingLoss, build_scores, demo_scores
from live_tracker.monitor import Monitor
from live_tracker.quality import flag_tracking
from live_tracker.rules import RuleSet
from live_tracker.skeleton import PROVISIONAL_EDGES, keypoint_status, mouse_skeleton

FPS = 30.0
S_END = "SYNTHETIC_bout_to_end"
S_GAP = "SYNTHETIC_frame_gap"


def ruleset(*rules):
    rs = RuleSet()
    for r in rules:
        rs.add(r)
        rs.approve(r["rule_id"])
    return rs


R_ALERT = {"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0, "alert_threshold": 0.8,
           "ask_threshold": 0.5, "alert_mode": "alert", "description": "attack for 2 s"}
R_SHORT = {**R_ALERT, "rule_id": "R2", "min_duration_s": 1.0, "description": "attack for 1 s"}
R_ASK = {**R_ALERT, "rule_id": "R3", "alert_mode": "ask", "description": "ask about every attack"}
R_LOG = {"rule_id": "R4", "behavior": "investigation", "min_duration_s": 1.0, "alert_threshold": 0.8,
         "ask_threshold": 0.5, "alert_mode": "log", "description": "log investigation"}


def end_session():
    """An attack bout that is still going when the session ends, and a tracking-lost closure."""
    return build_scores(S_END, 10.0, (Segment("attack", 1.0, 4.0, 0.9, 0.95), Segment("attack", 7.0, 10.0, 0.9, 0.95)),
                        (TrackingLoss(3.5, 4.5),), seed=3)


def gap_session():
    """Frames 120-134 are absent from the recording: the bout (fired on frame 90) closes with frame_gap."""
    s = build_scores(S_GAP, 8.0, (Segment("attack", 1.0, 6.0, 0.9, 0.95),), seed=4)
    return s[(s["frame"] < 120) | (s["frame"] >= 135)].reset_index(drop=True)


def all_scores():
    return pd.concat([demo_scores(), end_session(), gap_session()], ignore_index=True)


def pose_table(session, n, *, missing=(), low_conf=(), seed=0):
    """Synthetic long pose (session, frame, time_s, mouse, keypoint, x, y, score).
    missing / low_conf: (mouse, start_s, end_s) stretches on the torso keypoints."""
    rng = np.random.default_rng(seed)
    rows = []
    for mi, mouse in enumerate(("a", "b")):
        base = np.cumsum(rng.normal(0, 1.0, (n, 2)), axis=0) + (300 + 200 * mi, 250)
        for ki, kp in enumerate(KEYPOINTS):
            x, y = base[:, 0] + 8 * ki, base[:, 1] + 3 * ki + rng.normal(0, 0.3, n)
            score = np.full(n, 0.9)
            t = np.arange(n) / FPS
            for m, a, b in missing:
                if m == mouse and kp in TORSO_KEYPOINTS:
                    sel = (t >= a - 1e-9) & (t < b - 1e-9)
                    x, y = np.where(sel, np.nan, x), np.where(sel, np.nan, y)
            for m, a, b in low_conf:
                if m == mouse and kp in TORSO_KEYPOINTS:
                    score = np.where((t >= a - 1e-9) & (t < b - 1e-9), 0.05, score)
            rows.append(pd.DataFrame({"session": session, "frame": np.arange(n), "time_s": t, "mouse": mouse,
                                      "keypoint": kp, "x": x, "y": y, "score": score}))
    return pd.concat(rows, ignore_index=True)


def flags_of(pose):
    return flag_tracking(pose, score_col="score", min_score=0.15)


@pytest.fixture(scope="module")
def demo():
    return rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT, R_LOG))


def ids(state):
    return [e["event_id"] for e in state.events]


def by_rule(state, rule_id):
    return [e for e in state.events if e["rule_id"] == rule_id]


# ======================================================================
# 1. event firing and closure timing
# ======================================================================

def test_an_alert_appears_on_its_firing_frame_and_closes_after_its_last_frame(demo):
    assert not by_rule(demo.state_at(119), "R1")                # 2 s bout from 2.0 s: fires on frame 120
    s = demo.state_at(120)
    (alert,) = by_rule(s, "R1")
    assert alert["action"] == "alert" and alert["evidence"]["fired_frame"] == 120
    assert alert["event_id"] in s.new_event_ids and alert["event_id"] in s.open_event_ids
    assert alert["end_s"] == pytest.approx(120 / FPS + 1 / FPS)            # what was seen at firing
    assert alert["evidence"]["bout_closed"] is False and alert["evidence"]["end_reason"] is None
    assert alert["evidence"]["bout_frames"] is None
    still = by_rule(demo.state_at(149), "R1")[0]                # last bout frame: closure not known yet
    assert still == alert and alert["event_id"] in demo.state_at(149).open_event_ids
    closed = by_rule(demo.state_at(150), "R1")[0]               # first frame below threshold
    assert closed["end_s"] == pytest.approx(5.0) and closed["evidence"]["end_reason"] == "below_threshold"
    assert closed["evidence"]["last_frame"] == 149 and closed["evidence"]["bout_frames"] == 90
    assert closed["confidence"] == alert["confidence"]          # decided at firing, never changed
    assert alert["event_id"] not in demo.state_at(150).open_event_ids


def test_the_demo_scenario_events(demo):
    end = demo.end_state()
    r1 = by_rule(end, "R1")
    assert [(e["action"], e["evidence"]["fired_frame"]) for e in r1] == [("alert", 120), ("ask", 270)]
    assert not [e for e in r1 if e["start_s"] >= 12.0]          # short bout and tracking-split bout: no event
    (log,) = by_rule(end, "R4")
    assert log["action"] == "log" and log["evidence"]["fired_frame"] == 630


def test_a_bout_open_at_the_session_end_closes_only_in_the_end_state():
    r = rp.build_replay(end_session(), S_END, ruleset(R_ALERT))
    last = r.state_at(r.frames[-1])
    first, second = by_rule(last, "R1")
    assert first["evidence"]["end_reason"] == "tracking_lost"                # tracking ended the first bout
    assert first["end_s"] == pytest.approx(3.5)
    assert second["evidence"]["bout_closed"] is False and second["event_id"] in last.open_event_ids
    end = r.end_state()
    assert end.at_end and end.frame is None and not end.open_event_ids
    closed = by_rule(end, "R1")[1]
    assert closed["evidence"]["end_reason"] == "stream_end" and closed["end_s"] == pytest.approx(10.0)


def test_a_frame_gap_closure_is_visible_on_the_next_recorded_row():
    r = rp.build_replay(gap_session(), S_GAP, ruleset(R_ALERT))
    assert 120 not in r.frames and r.index_of(125) == r.index_of(119)       # seeking into the gap
    (open_,) = by_rule(r.state_at(119), "R1")
    assert open_["evidence"]["bout_closed"] is False and open_["evidence"]["fired_frame"] == 90
    assert by_rule(r.state_at(130), "R1")[0]["evidence"]["bout_closed"] is False
    first = by_rule(r.state_at(135), "R1")[0]                  # the next recorded row after the gap
    assert first["evidence"]["end_reason"] == "frame_gap" and first["evidence"]["last_frame"] == 119


# ======================================================================
# 2. no future information: the replay equals a monitor fed only the rows so far
# ======================================================================

@pytest.mark.parametrize("make, session, rules", [
    (demo_scores, DEMO_SESSION, (R_ALERT, R_LOG)), (demo_scores, DEMO_SESSION, (R_SHORT, R_ASK)),
    (end_session, S_END, (R_ALERT,)), (gap_session, S_GAP, (R_SHORT,))])
def test_every_frame_shows_exactly_what_a_live_monitor_had_written(make, session, rules):
    scores = make()
    r = rp.build_replay(scores, session, ruleset(*rules))
    live = Monitor(ruleset(*rules).active(), run_id=r.run_id)          # the oracle: fed row by row
    for i, row in enumerate(scores.sort_values("frame").to_dict("records")):
        live.step(row)
        expected = sorted(live.journal.events(r.run_id), key=lambda e: e["evidence"]["fired_frame"])
        got = r.state_at_index(i).events
        assert [e["event_id"] for e in got] == [e["event_id"] for e in expected], i
        assert list(got) == expected, i
    live.finish()
    assert list(r.end_state().events) == sorted(live.journal.events(r.run_id),
                                                key=lambda e: e["evidence"]["fired_frame"])


def test_scores_shown_are_the_current_frame_only(demo):
    s = demo.state_at(60)
    assert set(s.scores) == {"attack", "investigation"}
    row = demo_scores().set_index("frame").loc[60]
    assert s.scores["attack"] == pytest.approx(row["attack"]) and s.time_s == pytest.approx(2.0)


def test_before_the_first_row_nothing_is_known(demo):
    s = demo.state_at(-5)
    assert s.index == -1 and s.frame is None and not s.events and not s.warnings and s.scores == {}


# ======================================================================
# 3. seeking and deterministic playback
# ======================================================================

def test_seeking_back_hides_what_had_not_happened_and_repeated_seeks_agree(demo):
    late = demo.state_at(400)
    early = demo.state_at(100)
    assert len(late.events) == 2 and not early.events
    assert demo.state_at(400) == late and demo.state_at(100) == early
    for f in (300, 10, 700, 150, 149, 0, 300):
        assert demo.state_at(f) == demo.state_at(f)
    late.events[0]["confidence"] = -1                       # a caller changing a state changes nothing
    assert demo.state_at(400).events[0]["confidence"] != -1


def test_rebuilding_gives_the_same_run_and_states():
    a = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT))
    b = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT))
    assert a.run_id == b.run_id and len(a.run_id) <= 64
    assert all(a.state_at(f) == b.state_at(f) for f in range(0, 720, 7)) and a.end_state() == b.end_state()


def test_playback_at_recorded_speed(demo):
    assert demo.advance(0, 1.0, 1.0) == 30 and demo.advance(0, 1.0, 2.0) == 60 and demo.advance(0, 1.0, 5.0) == 150
    assert demo.advance(30, 0.0, 1.0) == 30
    assert demo.advance(700, 100.0, 5.0) == demo.n_frames - 1                 # never past the last row
    assert demo.advance(-1, 0.0) == -1 and demo.advance(-1, 0.5) == 15
    path = [0]
    for _ in range(10):
        path.append(demo.advance(path[-1], 0.1, 2.0))
    assert path == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    with pytest.raises(rp.ReplayError):
        demo.advance(0, 1.0, 3.0)
    r = rp.build_replay(gap_session(), S_GAP, ruleset(R_ALERT))
    assert r.frames[r.advance(r.index_of(119), 0.1)] == 119                  # time_s based across the gap
    assert r.frames[r.advance(r.index_of(119), 0.6)] == 137


# ======================================================================
# 4. sessions, runs and rule sets
# ======================================================================

def test_sessions_are_separate_runs(tmp_path):
    reps = rp.build_replays(all_scores(), ruleset(R_ALERT), journal_dir=tmp_path)
    assert sorted(reps) == sorted([DEMO_SESSION, S_END, S_GAP])
    runs = {s: r.run_id for s, r in reps.items()}
    assert len(set(runs.values())) == 3
    seen = set()
    for s, r in reps.items():
        end = r.end_state()
        assert end.events and all(e["session"] == s and e["evidence"]["run_id"] == r.run_id for e in end.events)
        assert not seen & set(ids(end))
        seen |= set(ids(end))
        assert (tmp_path / f"{r.run_id}.journal.jsonl").exists()
    a, b = reps[DEMO_SESSION], reps[S_END]
    before = b.state_at(100)
    a.state_at(500), a.end_state()                                          # switching never leaks state
    assert b.state_at(100) == before and b.session == S_END


def test_other_rules_are_another_run_in_the_same_journal(tmp_path):
    path = tmp_path / "shared.journal.jsonl"
    a = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT), journal_path=path)
    b = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_SHORT), journal_path=path)   # no conflict
    assert a.run_id != b.run_id and set(el.load_journal(path).run_ids()) == {a.run_id, b.run_id}
    assert all(e["evidence"]["run_id"] == b.run_id for e in b.end_state().events)
    assert len(b.end_state().events) > len(a.end_state().events)          # the 1 s rule fires on the short bout
    assert len(el.load_journal(path).events()) == len(a.end_state().events) + len(b.end_state().events)
    size = path.stat().st_size
    again = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT), journal_path=path)
    assert again.run_id == a.run_id and path.stat().st_size == size        # idempotent: nothing rewritten
    assert again.end_state() == a.end_state()


def test_changed_scores_are_another_run():
    changed = demo_scores()
    changed.loc[changed["frame"] == 5, "attack"] = 0.11
    assert rp.build_replay(changed, DEMO_SESSION, ruleset(R_ALERT)).run_id != \
        rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT)).run_id


def test_rule_modes_and_short_bouts():
    end = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_SHORT, R_ASK)).end_state()
    short = by_rule(end, "R2")
    # 15.0-16.0 s is 30 frames: elapsed 29/30 s on its last frame, so the 1 s rule does not fire before tracking
    # is lost; 17.0-18.5 s fires
    assert [round(e["start_s"], 2) for e in short] == [2.0, 7.0, 12.0, 17.0]
    assert all(e["action"] == "ask" for e in by_rule(end, "R3"))
    assert [e["evidence"]["fired_frame"] for e in by_rule(end, "R3")] == [120, 270]


def test_only_approved_rules_replay():
    rs = RuleSet()
    rs.add(R_ALERT)                                      # proposed, never confirmed
    with pytest.raises(rp.ReplayError, match="no approved rule"):
        rp.build_replay(demo_scores(), DEMO_SESSION, rs)
    with pytest.raises(rp.ReplayError, match="not approved"):
        rp.build_replay(demo_scores(), DEMO_SESSION, list(rs.rules.values()))


def test_bad_inputs_are_refused():
    with pytest.raises(rp.ReplayError, match="no score rows"):
        rp.build_replay(demo_scores(), "SYNTHETIC_absent", ruleset(R_ALERT))
    dup = pd.concat([demo_scores(), demo_scores().iloc[:1]], ignore_index=True)
    with pytest.raises(rp.ReplayError, match="duplicate frame"):
        rp.build_replay(dup, DEMO_SESSION, ruleset(R_ALERT))
    with pytest.raises(rp.ReplayError, match="lack columns"):
        rp.build_replay(demo_scores().drop(columns=["attack"]), DEMO_SESSION, ruleset(R_ALERT))
    shuffled = demo_scores().sample(frac=1.0, random_state=1)          # recorded order is restored by frame
    assert rp.build_replay(shuffled, DEMO_SESSION, ruleset(R_ALERT)).end_state() == \
        rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT)).end_state()


def test_synthetic_sessions_are_labelled(demo):
    assert demo.synthetic and demo.end_state().synthetic and "SYNTHETIC" in demo.source_note
    assert demo.banner == "Replay of a recorded session, not a live camera"


# ======================================================================
# 5. researcher answers
# ======================================================================

def test_answers_only_for_visible_ask_events_and_written_once():
    r = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT))
    alert, ask = by_rule(r.end_state(), "R1")
    with pytest.raises(rp.ReplayError, match="not an event of this run visible"):
        r.answer(ask["event_id"], "confirmed", at_frame=200)                 # fires on frame 270
    with pytest.raises(rp.ReplayError, match="only 'ask'"):
        r.answer(alert["event_id"], "confirmed", at_frame=300)
    with pytest.raises(rp.ReplayError, match="answer must be"):
        r.answer(ask["event_id"], "maybe", at_frame=300)
    assert r.answer(ask["event_id"], "confirmed", at_frame=280) is True
    assert by_rule(r.state_at(280), "R1")[1]["researcher_answer"] == "confirmed"
    assert len(by_rule(r.state_at(200), "R1")) == 1                          # seeking back hides the event
    assert r.answer(ask["event_id"], "confirmed", at_frame=290) is False     # same answer: no-op
    with pytest.raises(rp.ReplayError, match="answers are final"):
        r.answer(ask["event_id"], "rejected", at_frame=290)
    with pytest.raises(rp.ReplayError, match="not an event of this run"):
        r.answer("E-other-1", "confirmed", at_frame=700)


# ======================================================================
# 6. tracking gaps and warnings
# ======================================================================

def test_a_tracking_gap_is_not_presented_as_stillness(demo):
    gap = demo.state_at(490)                                     # tracking lost on 16.0-17.0 s
    assert gap.tracking_ok is False and "not the mouse standing still" in gap.tracking_note
    assert demo.state_at(400).tracking_ok is True and demo.state_at(400).tracking_note is None


def _warn_pose(session, n=300, seed=0):
    return pose_table(session, n, seed=seed, missing=[("a", 2.0, 3.0), ("a", 9.0, 10.0)],
                      low_conf=[("b", 5.0, 6.0)])


def test_warning_kinds_distinguish_low_confidence_from_missing_coordinates():
    w = rp.tracking_warnings(flags_of(_warn_pose("SYNTHETIC_w")), "SYNTHETIC_w")
    kinds = [(x.kind, x.mouse, x.coordinates_present, round(x.start_s, 2)) for x in w]
    assert kinds == [("missing_coordinates", "a", False, 2.0), ("low_confidence", "b", True, 5.0),
                     ("missing_coordinates", "a", False, 9.0)]
    assert [x.message for x in w][:1] == ["Tracking lost for mouse a"]      # Ava's text, unchanged
    assert [x.ongoing_at_session_end for x in w] == [False, False, True]


def test_warning_timing_is_causal():
    pose = _warn_pose("SYNTHETIC_w")
    scores = build_scores("SYNTHETIC_w", 10.0, seed=2)
    w = rp.tracking_warnings(flags_of(pose), "SYNTHETIC_w")
    r = rp.build_replay(scores, "SYNTHETIC_w", ruleset(R_ALERT), warnings=w)
    first = w[0].warning_id
    assert first not in [x["event_id"] for x in r.state_at(74).warnings]     # 2.0 s + 0.5 s minimum = frame 75
    s = r.state_at(75)
    (entry,) = [x for x in s.warnings if x["event_id"] == first]
    assert entry["end_s"] is None and entry["evidence"]["end_known"] is False and first in s.active_warning_ids
    assert entry["evidence"]["duration_s"] is None
    assert [x for x in r.state_at(89).warnings if x["event_id"] == first][0]["end_s"] is None
    ended = [x for x in r.state_at(90).warnings if x["event_id"] == first][0]  # first good frame, 3.0 s
    assert ended["end_s"] == pytest.approx(3.0) and first not in r.state_at(90).active_warning_ids
    last = w[2].warning_id                                       # still missing when the session ends
    at_last = [x for x in r.state_at(r.frames[-1]).warnings if x["event_id"] == last][0]
    assert at_last["end_s"] is None and last in r.state_at(r.frames[-1]).active_warning_ids
    end_entry = [x for x in r.end_state().warnings if x["event_id"] == last][0]
    assert end_entry["evidence"]["end_known"] is True and end_entry["evidence"]["ongoing_at_session_end"] is True
    assert all(e["action"] in el.ACTIONS for e in r.journal.events())       # warnings never enter the journal
    assert {x["action"] for x in r.log_entries()} <= {"alert", "ask", "log", "tracking_warning"}


def test_warning_ids_never_collide_across_sessions():
    flags = pd.concat([flags_of(_warn_pose("SYNTHETIC_w1", seed=1)), flags_of(_warn_pose("SYNTHETIC_w2", seed=2))],
                      ignore_index=True)
    w1, w2 = rp.tracking_warnings(flags, "SYNTHETIC_w1"), rp.tracking_warnings(flags, "SYNTHETIC_w2")
    assert [x.source_event_id for x in w1] == [x.source_event_id for x in w2] == ["T1", "T2", "T3"]   # Ava's restart
    assert not {x.warning_id for x in w1} & {x.warning_id for x in w2}
    assert len({x.warning_id for x in w1}) == len(w1)
    assert rp.tracking_warnings(flags, "SYNTHETIC_w1") == w1                  # deterministic
    with pytest.raises(rp.ReplayError, match="another session"):
        rp.build_replay(build_scores("SYNTHETIC_w1", 10.0), "SYNTHETIC_w1", ruleset(R_ALERT), warnings=w2)


def test_low_conf_is_only_flagged_when_asked():
    pose = _warn_pose("SYNTHETIC_w")
    w = rp.tracking_warnings(flag_tracking(pose), "SYNTHETIC_w")          # no score threshold: no low_conf
    assert [x.kind for x in w] == ["missing_coordinates", "missing_coordinates"]


# ======================================================================
# 7. keypoints and skeleton geometry
# ======================================================================

def test_keypoint_statuses():
    assert keypoint_status(1.0, 2.0, 0.9) == "tracked"
    assert keypoint_status(1.0, 2.0, 0.05) == "low_confidence"
    assert keypoint_status(1.0, 2.0, None) == "unreliable"                    # unknown confidence is never tracked
    assert keypoint_status(float("nan"), 2.0, 0.9) == "missing" and keypoint_status(None, None) == "missing"
    assert keypoint_status(1.0, 2.0, 0.9, flags={"frozen": True}) == "unreliable"
    assert keypoint_status(1.0, 2.0, 0.9, flags={"imputed": True}) == "imputed"
    assert keypoint_status(1.0, 2.0, 0.9, frame_tracking_ok=False) == "unreliable"
    assert keypoint_status(1.0, 2.0, 0.9, flags={"low_conf": True}) == "low_confidence"


def test_only_tracked_keypoints_are_solid_and_joined():
    rows = {kp: {"x": 10.0 + i, "y": 5.0, "score": 0.9} for i, kp in enumerate(KEYPOINTS)}
    rows["neck"]["score"] = 0.05
    rows["nose"].update(x=float("nan"), y=float("nan"))
    del rows["tail_base"]
    sk = mouse_skeleton("a", rows)
    st = {k.name: (k.status, k.style, k.x) for k in sk.keypoints}
    assert st["neck"] == ("low_confidence", "hollow", 13.0)
    assert st["nose"] == ("missing", None, None) and st["tail_base"][0] == "missing"
    assert st["left_hip"][:2] == ("tracked", "solid")
    assert not any("neck" in e or "nose" in e or "tail_base" in e for e in sk.edges)
    assert len(sk.edges) + len(sk.omitted_edges) == len(PROVISIONAL_EDGES) and sk.edges == ()
    assert "PROVISIONAL" in sk.edge_status
    whole = mouse_skeleton("a", {kp: {"x": 1.0, "y": 1.0, "score": 0.9} for kp in KEYPOINTS})
    assert whole.edges == PROVISIONAL_EDGES and whole.n_tracked == 7
    lost = mouse_skeleton("a", {kp: {"x": 1.0, "y": 1.0, "score": 0.9} for kp in KEYPOINTS},
                          frame_tracking_ok=False)
    assert lost.n_tracked == 0 and lost.edges == ()


def test_frame_states_carry_skeletons_from_the_pose():
    pose = flags_of(_warn_pose("SYNTHETIC_w"))
    r = rp.build_replay(build_scores("SYNTHETIC_w", 10.0, tracking_loss=(TrackingLoss(2.0, 3.0),), seed=2),
                        "SYNTHETIC_w", ruleset(R_ALERT), pose=pose)
    ok = r.state_at(30)
    assert [s.mouse for s in ok.skeletons] == ["a", "b"] and all(s.n_tracked == 7 for s in ok.skeletons)
    gap = {s.mouse: s for s in r.state_at(70).skeletons}                     # missing torso, tracking_ok False
    assert {k.name: k.status for k in gap["a"].keypoints}["neck"] == "missing"
    assert gap["b"].n_tracked == 0 and gap["a"].edges == ()                    # nothing drawn as tracked
    low = {s.mouse: s for s in r.state_at(160).skeletons}["b"]               # low confidence, coordinates kept
    assert {k.name: k.status for k in low.keypoints}["neck"] == "low_confidence"
    assert {k.name: k.x for k in low.keypoints}["neck"] is not None
    assert rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT)).state_at(30).skeletons == ()
    assert copy.deepcopy(ok) == ok



# ======================================================================
# 8. Codex D3-1 review: regressions for the five blockers
# ======================================================================

def _ask_replay(**kw):
    return rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT), **kw)


def _ask(r):
    return by_rule(r.end_state(), "R1")[1]                     # the ask fires on frame 270


def _answer_at(r, frame):
    return by_rule(r.state_at(frame), "R1")[1]["researcher_answer"]


def test_b1_an_answer_given_at_280_is_not_visible_at_270():
    r = _ask_replay()
    ask = _ask(r)
    assert r.answer(ask["event_id"], "confirmed", at_frame=280) is True
    assert _answer_at(r, 270) is None and _answer_at(r, 279) is None          # before it was given
    assert _answer_at(r, 280) == "confirmed" and _answer_at(r, 700) == "confirmed"
    assert _answer_at(r, 270) is None                                        # forward, then back again
    assert by_rule(r.end_state(), "R1")[1]["researcher_answer"] == "confirmed"
    assert r.answer_status(ask["event_id"]) == {"researcher_answer": "confirmed", "answered_frame": 280}
    assert r.answer(ask["event_id"], "confirmed", at_frame=300) is False      # idempotent, keeps frame 280
    assert r.answer_status(ask["event_id"])["answered_frame"] == 280
    with pytest.raises(rp.ReplayError, match="answers are final"):
        r.answer(ask["event_id"], "rejected", at_frame=300)


def test_b1_answer_positions_survive_a_rebuild_and_untimed_answers_stay_unknown(tmp_path):
    journal, answers = tmp_path / "j.jsonl", tmp_path / "a.jsonl"
    r = _ask_replay(journal_path=journal, answers_path=answers)
    eid = _ask(r)["event_id"]
    r.answer(eid, "rejected", at_frame=290)
    again = _ask_replay(journal_path=journal, answers_path=answers)          # rebuilt with the positions
    assert _answer_at(again, 289) is None and _answer_at(again, 290) == "rejected"
    blind = _ask_replay(journal_path=journal)                               # journal only: timing unknown
    assert blind.answer_status(eid) == {"researcher_answer": "rejected", "answered_frame": None}
    assert all(_answer_at(blind, f) is None for f in (270, 290, 500, 719))   # never placed at firing
    assert by_rule(blind.end_state(), "R1")[1]["researcher_answer"] == "rejected"
    assert blind.answer(eid, "rejected", at_frame=600) is False               # still timing-unknown
    assert blind.answer_status(eid)["answered_frame"] is None
    with pytest.raises(rp.ReplayError, match="answers are final"):
        blind.answer(eid, "confirmed", at_frame=600)


@pytest.mark.parametrize("edit, match", [
    (lambda rec: rec.update(researcher_answer="confirmed"), "contradicts the journal"),
    (lambda rec: rec.update(answered_frame=100), "before the event fired"),
    (lambda rec: rec.update(answered_frame=99999), "unknown event or frame"),
    (lambda rec: rec.update(extra=1), "not an answer-position record"),
])
def test_b1_a_contradictory_answers_file_is_refused(tmp_path, edit, match):
    import json
    journal, answers = tmp_path / "j.jsonl", tmp_path / "a.jsonl"
    r = _ask_replay(journal_path=journal, answers_path=answers)
    r.answer(_ask(r)["event_id"], "rejected", at_frame=290)
    rec = json.loads(answers.read_text())
    edit(rec)
    answers.write_text(json.dumps(rec) + "\n")
    with pytest.raises(rp.ReplayError, match=match):
        _ask_replay(journal_path=journal, answers_path=answers)


def test_b2_renamed_behavior_columns_are_another_run(tmp_path):
    scores = demo_scores()
    swapped = scores.rename(columns={"attack": "mount", "mount": "attack"})[list(scores.columns)]
    a = rp.build_replay(scores, DEMO_SESSION, ruleset(R_ALERT), journal_path=tmp_path / "j.jsonl")
    b = rp.build_replay(swapped, DEMO_SESSION, ruleset(R_ALERT), journal_path=tmp_path / "j.jsonl")   # no conflict
    assert a.run_id != b.run_id
    assert len(a.end_state().events) == 2 and b.end_state().events == ()      # mount values under "attack"


def test_b2_types_order_alignment_and_extra_columns_bind_identity():
    base = demo_scores()
    as_text = base.assign(tracking_ok=base["tracking_ok"].map({True: "true", False: "false"}))
    plus = base.assign(note="x")
    run = lambda s, fps=30.0: rp.build_replay(s, DEMO_SESSION, ruleset(R_ALERT), fps=fps).run_id   # noqa: E731
    ids_ = {run(base), run(as_text), run(plus)}
    assert len(ids_) == 3                                                     # typed cells and column names
    assert run(base) == run(base.sample(frac=1.0, random_state=3))           # fed in frame order: identical
    assert run(base) == run(base.copy())
    slow = base.assign(time_s=base["frame"] / 25.0)
    assert run(slow, 25.0) != run(base)


def test_b2_reversed_rule_order_executes_identically(tmp_path):
    forward = ruleset(R_ALERT, R_SHORT, R_LOG)
    backward = ruleset(R_LOG, R_SHORT, R_ALERT)
    a = rp.build_replay(demo_scores(), DEMO_SESSION, forward, journal_path=tmp_path / "j.jsonl")
    b = rp.build_replay(demo_scores(), DEMO_SESSION, backward, journal_path=tmp_path / "j.jsonl")   # no conflict
    assert a.run_id == b.run_id and [r.rule_id for r in b.rules] == ["R1", "R2", "R4"]
    assert a.end_state() == b.end_state()
    assert [(e["event_id"], e["rule_id"]) for e in a.end_state().events] == \
        [(e["event_id"], e["rule_id"]) for e in b.end_state().events]
    other = ruleset({**R_ALERT, "min_duration_s": 2.5}, R_SHORT, R_LOG)
    assert rp.build_replay(demo_scores(), DEMO_SESSION, other).run_id != a.run_id


@pytest.mark.parametrize("ok_true, ok_false", [("true", "false"), ("True", "FALSE"), (1, 0), ("1", "0")])
def test_b3_tracking_strings_follow_the_monitor(ok_true, ok_false):
    scores = demo_scores()
    scores["tracking_ok"] = scores["tracking_ok"].map({True: ok_true, False: ok_false}).astype(object)
    r = rp.build_replay(scores, DEMO_SESSION, ruleset(R_ALERT))
    gap = r.state_at(490)
    assert gap.tracking_ok is False and gap.tracking_note == rp.TRACKING_GAP_NOTE
    assert r.state_at(400).tracking_ok is True
    plain = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT)).end_state().events
    strip = lambda evs: [(e["action"], e["start_s"], e["end_s"], e["evidence"]["fired_frame"]) for e in evs]  # noqa
    assert strip(r.end_state().events) == strip(plain)                # same events (run_id differs: typed cells)


def test_b3_tracking_loss_from_strings_makes_every_keypoint_unreliable():
    scores = build_scores("SYNTHETIC_w", 10.0, tracking_loss=(TrackingLoss(2.0, 3.0),), seed=2)
    scores["tracking_ok"] = scores["tracking_ok"].map({True: "true", False: "false"})
    r = rp.build_replay(scores, "SYNTHETIC_w", ruleset(R_ALERT), pose=pose_table("SYNTHETIC_w", 300))
    assert all(sk.n_tracked == 0 and sk.edges == () for sk in r.state_at(70).skeletons)
    assert all(sk.n_tracked == 7 for sk in r.state_at(30).skeletons)


@pytest.mark.parametrize("bad", ["yes", "", 2, None, 0.5])
def test_b3_invalid_tracking_values_are_refused(bad):
    scores = demo_scores().astype({"tracking_ok": object})
    scores.loc[5, "tracking_ok"] = bad
    with pytest.raises(rp.ReplayError, match="tracking_ok"):
        rp.build_replay(scores, DEMO_SESSION, ruleset(R_ALERT))


def test_b4_the_session_end_outcome_is_only_in_the_end_state():
    pose = _warn_pose("SYNTHETIC_w")
    w = rp.tracking_warnings(flags_of(pose), "SYNTHETIC_w")
    r = rp.build_replay(build_scores("SYNTHETIC_w", 10.0, seed=2), "SYNTHETIC_w", ruleset(R_ALERT), warnings=w)
    last = w[2].warning_id
    for f in (285, r.frames[-1]):
        (entry,) = [x for x in r.state_at(f).warnings if x["event_id"] == last]
        assert entry["evidence"]["ongoing_at_session_end"] is None and entry["end_s"] is None
        assert entry["evidence"]["end_known"] is False and entry["evidence"]["duration_s"] is None
    recovered = [x for x in r.state_at(r.frames[-1]).warnings if x["event_id"] == w[0].warning_id][0]
    assert recovered["evidence"]["ongoing_at_session_end"] is False                  # it ended at 3.0 s
    (final,) = [x for x in r.end_state().warnings if x["event_id"] == last]
    assert final["evidence"]["ongoing_at_session_end"] is True and final["end_s"] is not None
    assert [x for x in r.state_at(74).warnings if x["event_id"] == w[0].warning_id] == []   # onset unchanged
    assert [x for x in r.state_at(90).warnings if x["event_id"] == w[0].warning_id][0]["end_s"] == \
        pytest.approx(3.0)                                                           # recovery unchanged


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), float("-inf")])
def test_b5_unknown_confidence_never_draws_solid(score):
    rows = {kp: {"x": 1.0 + i, "y": 1.0, "score": score} for i, kp in enumerate(KEYPOINTS)}
    sk = mouse_skeleton("a", rows)
    assert sk.n_tracked == 0 and sk.edges == ()
    assert all(k.status == "unreliable" and k.style == "hollow" and "confidence" in k.reason for k in sk.keypoints)
    with pytest.raises(ValueError, match="min_score"):
        keypoint_status(1.0, 1.0, 0.9, min_score=float("nan"))
    pose = pose_table("SYNTHETIC_s", 60)
    pose["score"] = score
    r = rp.build_replay(build_scores("SYNTHETIC_s", 2.0), "SYNTHETIC_s", ruleset(R_ALERT), pose=pose)
    assert all(s.n_tracked == 0 and s.edges == () for s in r.state_at(30).skeletons)


def _stuck_pose(session="SYNTHETIC_f", n=90, until=30):
    """Mouse a's neck is identical on frames [0, until), then moves."""
    pose = pose_table(session, n)
    sel = (pose["mouse"] == "a") & (pose["keypoint"] == "neck") & (pose["frame"] < until)
    pose.loc[sel, ["x", "y"]] = (321.0, 123.0)
    return pose


def _neck(state):
    return next(k for sk in state.skeletons if sk.mouse == "a" for k in sk.keypoints if k.name == "neck")


def test_b5_a_frozen_run_is_not_known_before_enough_identical_frames():
    pose = _stuck_pose()
    assert flag_tracking(pose).query("mouse == 'a' and keypoint == 'neck' and frame == 0")["frozen"].item()  # Ava's
    r = rp.build_replay(build_scores("SYNTHETIC_f", 3.0), "SYNTHETIC_f", ruleset(R_ALERT), pose=pose)
    assert _neck(r.state_at(0)).status == "tracked" and _neck(r.state_at(13)).status == "tracked"
    stuck = _neck(r.state_at(14))                                     # the 15th identical position
    assert stuck.status == "unreliable" and "frozen" in stuck.reason
    assert _neck(r.state_at(29)).status == "unreliable" and _neck(r.state_at(30)).status == "tracked"
    flags = rp.causal_pose_flags(pose).query("mouse == 'a' and keypoint == 'neck'")
    assert flags["frozen"].tolist() == [False] * 14 + [True] * 16 + [False] * 60
    # prefix-invariance: flags for frames <= k never depend on frames after k
    for k in (0, 13, 14, 20, 29, 45):
        part = rp.causal_pose_flags(pose[pose["frame"] <= k])
        full = rp.causal_pose_flags(pose)
        full = full[full["frame"] <= k].reset_index(drop=True)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), full)


def test_b5_a_frozen_warning_appears_only_once_frozen_is_knowable():
    pose = _stuck_pose()
    w = rp.tracking_warnings(flag_tracking(pose), "SYNTHETIC_f", min_duration_s=0.1)
    (frozen,) = [x for x in w if x.flag == "frozen"]
    assert frozen.start_s == 0.0 and frozen.knowable_after_s == pytest.approx(14 / 30)
    r = rp.build_replay(build_scores("SYNTHETIC_f", 3.0), "SYNTHETIC_f", ruleset(R_ALERT), warnings=w)
    assert not r.state_at(13).warnings and r.state_at(14).warnings[0]["evidence"]["kind"] == "frozen_tracker"


def test_b5_an_interpolated_gap_is_missing_on_its_own_frames():
    from live_tracker.quality import interpolate_short_gaps
    pose = pose_table("SYNTHETIC_i", 60, missing=[("a", 1.0, 1.1)])            # 3 frames, later filled
    filled = interpolate_short_gaps(flag_tracking(pose))
    assert filled.query("mouse == 'a' and keypoint == 'neck' and frame == 31")["imputed"].item()
    r = rp.build_replay(build_scores("SYNTHETIC_i", 2.0), "SYNTHETIC_i", ruleset(R_ALERT), pose=filled)
    neck = _neck(r.state_at(31))
    assert neck.status == "missing" and neck.x is None
    assert _neck(r.state_at(33)).status == "tracked"
