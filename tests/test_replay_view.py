"""Tests for replay_view (what the Replay page shows, without Streamlit). New file, D3-2.

SYNTHETIC data only (fake_scores and generated pose). No CalMS21, cache,
score archive, model or network.
"""

import dataclasses

import pandas as pd
import pytest

import live_tracker.replay as rp
import live_tracker.replay_view as rv
from live_tracker.calms21 import KEYPOINTS
from live_tracker.fake_scores import DEMO_SESSION, demo_scores
from live_tracker.skeleton import mouse_skeleton
from test_replay import R_ALERT, ruleset


@pytest.fixture(scope="module")
def catalog():
    return rv.synthetic_catalog()


def demo_replay(**kw):
    entry = rv.synthetic_catalog()[0]
    return rp.build_replay(entry.scores, entry.session, ruleset(R_ALERT), pose=entry.pose, **kw)


def test_the_catalog_is_synthetic_deterministic_and_valid(catalog):
    assert [e.session for e in catalog] == [DEMO_SESSION, rv.TRACKING_SESSION]
    for e in catalog:
        assert e.session.startswith("SYNTHETIC_")
        rv.check_entry(e)
        assert rv.check_pose(e) is None
        assert set(e.pose["session"]) == {e.session} and len(e.pose) == len(e.scores) * 14
    again = rv.synthetic_catalog()
    for a, b in zip(catalog, again):
        pd.testing.assert_frame_equal(a.scores, b.scores)
        pd.testing.assert_frame_equal(a.pose, b.pose)


def test_the_tracking_session_has_every_warning_kind(catalog):
    kinds = {w.kind for w in rv.entry_warnings(catalog[1])}
    assert {"missing_coordinates", "low_confidence", "frozen_tracker"} <= kinds


@pytest.mark.parametrize("edit, match", [
    (lambda e: dataclasses.replace(e, session="task1/test/mouse001"), "not a SYNTHETIC session"),
    (lambda e: dataclasses.replace(e, scores=e.scores.iloc[0:0]), "no synthetic score rows"),
    (lambda e: dataclasses.replace(e, scores=e.scores.drop(columns=["tracking_ok"])), "lacks columns"),
    (lambda e: dataclasses.replace(e, scores=e.scores.assign(session="SYNTHETIC_other")), "another session"),
    (lambda e: "not an entry", "invalid entry"),
])
def test_entries_the_page_cannot_show_honestly_are_refused(catalog, edit, match):
    with pytest.raises(rp.ReplayError, match=match):
        rv.check_entry(edit(catalog[0]))


@pytest.mark.parametrize("edit, match", [
    (lambda p: None, "no pose"),
    (lambda p: p.drop(columns=["score"]), "lacks columns"),
    (lambda p: p.assign(session="SYNTHETIC_other"), "another session"),
    (lambda p: p.assign(keypoint="elbow"), "unknown keypoint"),
    (lambda p: p.assign(mouse="c"), "unknown mouse"),
    (lambda p: pd.concat([p, p.iloc[:1]]), "duplicate pose rows"),
    (lambda p: p.assign(x="12"), "not numeric"),
    (lambda p: p.assign(frame=p["frame"].map(lambda f: f"f{f}")), "frame numbers are not numeric"),
    (lambda p: p.assign(frame=p["frame"] + 0.5), "whole numbers"),
    (lambda p: p.assign(frame=p["frame"].where(p.index != 3)), "missing or not finite"),
    (lambda p: p.assign(frame=p["frame"].astype(float).where(p.index != 3, float("inf"))), "missing or not finite"),
    (lambda p: p.assign(frame=p["frame"] - 1), "whole numbers >= 0"),
    (lambda p: p.assign(frame=p["frame"] + 10_000), "not in the score rows"),
    (lambda p: p.assign(frame=p["frame"] == 0), "not numeric"),
    (lambda p: p.assign(time_s=p["time_s"] + 1.0), "do not match the score rows"),
    (lambda p: p.assign(time_s=float("nan")), "times are missing"),
])
def test_invalid_pose_is_reported(catalog, edit, match):
    entry = dataclasses.replace(catalog[0], pose=edit(catalog[0].pose))
    assert match in rv.check_pose(entry)


def _play(r, speed, ticks, dt=0.25, start=0, t0=100.0):
    anchor = rv.start_playback(r, start, t0, speed)
    index, playing = start, True
    for k in range(1, ticks + 1):
        index, anchor, playing = rv.playback_position(r, anchor, t0 + k * dt, speed)
    return index, anchor, playing


@pytest.mark.parametrize("speed, frame", [(1.0, 60), (2.0, 120), (5.0, 300)])
def test_eight_quarter_second_ticks_lose_no_time(speed, frame):
    r = demo_replay()
    index, _, playing = _play(r, speed, 8)
    assert r.frames[index] == frame and playing                    # 2 s of wall time, no rounding drift
    assert r.frames[_play(r, speed, 80, dt=0.025)[0]] == frame      # finer ticks reach the same frame


def test_playback_anchor_rules():
    r = demo_replay()
    anchor = rv.start_playback(r, 0, 10.0, 1.0)
    i, anchor, _ = rv.playback_position(r, anchor, 11.0, 1.0)      # 1 s at 1x
    assert r.frames[i] == 30
    i, anchor, _ = rv.playback_position(r, anchor, 11.0, 2.0)      # speed change re-anchors at 1.0 s exactly
    assert anchor.recorded_s == pytest.approx(1.0) and anchor.speed == 2.0
    i, anchor, _ = rv.playback_position(r, anchor, 12.0, 2.0)
    assert r.frames[i] == 90
    i, back, _ = rv.playback_position(r, anchor, 11.5, 2.0)        # a clock going back never rewinds
    assert r.frames[i] == 90 and back.wall == 12.0 and back.recorded_s == pytest.approx(3.0)
    assert rv.playback_position(r, rv.start_playback(r, 700, 0.0, 5.0), 100.0, 5.0)[1:][1] is False
    assert rv.start_playback(r, -1, 0.0, 1.0).recorded_s == 0.0
    with pytest.raises(rp.ReplayError):
        rv.start_playback(r, 0, 0.0, 3.0)
    with pytest.raises(rp.ReplayError):
        rv.playback_position(r, anchor, 13.0, 0.5)


def test_playback_follows_recorded_time_across_skipped_frames():
    s = demo_scores()
    s = s[(s["frame"] < 100) | (s["frame"] >= 130)].reset_index(drop=True)      # frames 100-129 not recorded
    r = rp.build_replay(s, DEMO_SESSION, ruleset(R_ALERT))
    i, _, _ = _play(r, 1.0, 14)                                                   # 3.5 s: inside the gap
    assert r.frames[i] == 99
    i, _, _ = _play(r, 1.0, 16)                                                   # 4.0 s: still not recorded
    assert r.frames[i] == 99
    i, _, _ = _play(r, 1.0, 18)                                                   # 4.5 s: recording resumes
    assert r.frames[i] == 135


def test_event_times_are_distinct_labels():
    r = demo_replay()
    open_ = rv.event_times(r.state_at(130).events[0])
    assert open_ == {"onset": "2.00 s (frame 60)", "fired": "4.00 s (frame 120)", "closure": "open (not ended yet)"}
    closed = rv.event_times(r.state_at(150).events[0])
    assert closed["closure"] == "5.00 s (below_threshold, last frame 149)"
    rows = rv.timeline_rows(r.state_at(400))
    assert [(x["action"], x["fired"]) for x in rows] == [("alert", "4.00 s (frame 120)"), ("ask", "9.00 s (frame 270)")]
    assert rv.timeline_rows(r.state_at(400)) == rows and rv.timeline_rows(r.state_at(100)) == []


def test_answer_states(tmp_path):
    r = demo_replay(journal_path=tmp_path / "j.jsonl", answers_path=tmp_path / "a.jsonl")
    alert, ask = r.end_state().events
    assert rv.answer_state(r, r.state_at(300), alert) == (None, "")
    assert rv.answer_state(r, r.state_at(300), r.state_at(300).events[1])[0] == "unanswered"
    r.answer(ask["event_id"], "confirmed", at_frame=280)
    code, text = rv.answer_state(r, r.state_at(280), r.state_at(280).events[1])
    assert code == "answered" and text == "Answered: confirmed at frame 280"
    code, text = rv.answer_state(r, r.state_at(275), r.state_at(275).events[1])
    assert code == "answered_later" and "confirmed" not in text                # the later answer is not revealed
    blind = demo_replay(journal_path=tmp_path / "j.jsonl")                    # no answer positions
    assert rv.answer_state(blind, blind.state_at(300), blind.state_at(300).events[1])[0] == "timing_unknown"
    assert rv.answer_state(blind, blind.end_state(), blind.end_state().events[1]) == \
        ("answered", "Answered: confirmed (replay time unknown)")


def test_warning_rows_say_ongoing_until_the_end_is_known(catalog):
    entry = catalog[1]
    r = rp.build_replay(entry.scores, entry.session, ruleset(R_ALERT), warnings=rv.entry_warnings(entry),
                        pose=entry.pose)
    assert rv.warning_rows(r.state_at(100)) == []
    (row,) = rv.warning_rows(r.state_at(125))                               # missing torso since 3.5 s
    assert row["kind"] == "missing_coordinates" and row["end"] == "ongoing" and row["coordinates"] == "missing"
    done = rv.warning_rows(r.state_at(140))[0]
    assert done["end"] == "4.50 s"
    low = [x for x in rv.warning_rows(r.state_at(280)) if x["kind"] == "low_confidence"][0]
    assert low["coordinates"] == "present (unreliable)"


def test_tracking_summary_counts_untracked_keypoints(catalog):
    entry = catalog[1]
    r = rp.build_replay(entry.scores, entry.session, ruleset(R_ALERT), pose=entry.pose)
    assert rv.tracking_summary(r.state_at(30)) == {}
    assert rv.tracking_summary(r.state_at(255))["b"] == {"low_confidence": 3}
    lost = rv.tracking_summary(r.state_at(110))                              # tracking_ok False: nothing tracked
    assert lost["a"]["missing"] == 3 and sum(lost["b"].values()) == 7


def _state_with(sk):
    r = demo_replay()
    return dataclasses.replace(r.state_at(30), skeletons=(sk,), tracking_ok=True)


def test_the_skeleton_image_draws_only_what_is_known():
    tracked = mouse_skeleton("a", {kp: {"x": 100.0 + 40 * i, "y": 200.0, "score": 0.9} for i, kp in enumerate(KEYPOINTS)})
    img = rv.render_skeletons(_state_with(tracked))
    assert img.size == (512, 285)
    assert img.getpixel((int(100 * 0.5), int(200 * 0.5))) == rv.MOUSE_COLOURS["a"]          # solid tracked point
    missing = mouse_skeleton("a", {})
    blank = rv.render_skeletons(_state_with(missing))
    assert {c for _, c in blank.getcolors(1 << 20)} == {(250, 250, 250)}                         # nothing invented for missing points
    low = mouse_skeleton("a", {kp: {"x": 100.0 + 40 * i, "y": 200.0, "score": 0.01} for i, kp in enumerate(KEYPOINTS)})
    hollow = rv.render_skeletons(_state_with(low))
    assert hollow.getpixel((50, 100)) == (250, 250, 250)                     # hollow: the centre is not filled
    assert rv.MOUSE_COLOURS["a"] not in {c for _, c in hollow.getcolors(1 << 20)}                # and no edge is drawn
    r = demo_replay()
    assert rv.render_skeletons(r.state_at(30)) is not None and rv.render_skeletons(r.end_state()) is None
    lost = rv.render_skeletons(r.state_at(490))                               # tracking lost: red frame border
    assert lost.getpixel((0, 0)) == (210, 40, 40)


def test_valid_poses_pass_and_shifted_frames_need_matching_times(catalog):
    assert all(rv.check_pose(e) is None for e in catalog)
    as_float = dataclasses.replace(catalog[0], pose=catalog[0].pose.assign(frame=catalog[0].pose["frame"] * 1.0))
    assert rv.check_pose(as_float) is None                                      # whole-number floats are frames
    part = catalog[0].pose[catalog[0].pose["frame"] < 100]
    assert rv.check_pose(dataclasses.replace(catalog[0], pose=part)) is None    # a pose for part of the session


def test_the_cache_identity_follows_the_content(catalog, tmp_path):
    e = catalog[0]
    rules = ruleset(R_ALERT).active()
    base = rv.cache_identity(e, rules, tmp_path, True)
    assert rv.cache_identity(e, rules, tmp_path, True) == base
    zero = dataclasses.replace(e, scores=e.scores.assign(attack=0.0))
    moved = dataclasses.replace(e, pose=e.pose.assign(x=e.pose["x"] + 1.0))
    assert rv.cache_identity(zero, rules, tmp_path, True)[0] != base[0]          # a new run
    assert rv.cache_identity(moved, rules, tmp_path, True)[0] == base[0]         # same events, other pose
    assert rv.cache_identity(moved, rules, tmp_path, True) != base
    assert rv.cache_identity(e, rules, tmp_path, False) != base
    assert rv.cache_identity(e, rules, tmp_path / "other", True) != base
