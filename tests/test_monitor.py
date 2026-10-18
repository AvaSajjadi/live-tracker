"""Tests for the streaming monitor (new file).

Journal / §4 export tests are in test_event_log.py.
"""

import math

import numpy as np
import pytest

from live_tracker.fake_scores import demo_scores
from live_tracker.monitor import Monitor, MonitorInputError, run_monitor
from live_tracker.rules import Rule, RuleSet

FPS = 30.0
R1 = {"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0,
      "alert_threshold": 0.8, "ask_threshold": 0.5, "alert_mode": "alert",
      "description": "Alert me if the mice attack for at least 2 seconds"}


def approved(*rule_dicts):
    rs = RuleSet()
    for d in rule_dicts:
        rs.add(d)
        rs.approve(d["rule_id"])
    return rs


def rule(**change):
    return approved({**R1, **change})


def rows(attack, ok=None, session="SYNTHETIC_t", start_frame=0, fps=FPS, **other):
    """Regularly sampled score rows from a list of attack scores."""
    n = len(attack)
    ok = [True] * n if ok is None else ok
    out = []
    for i in range(n):
        f = start_frame + i
        r = {"session": session, "frame": f, "time_s": f / fps, "attack": attack[i],
             "mount": 0.0, "investigation": 0.0, "tracking_ok": ok[i]}
        for k, v in other.items():
            r[k] = v[i]
        out.append(r)
    return out


def run(data, rules, **kw):
    return run_monitor(data, rules, run_id="t", **kw)


def fired_log(data, rules, **kw):
    """(frame, event as returned at firing) for every firing, in stream order."""
    m = Monitor(rules, run_id="t", **kw)
    log = []
    for r in data:
        for e in m.step(r):
            log.append((r["frame"], e))
    return log, m.finish()


# ---- approval gate and rule revalidation ----

def test_unapproved_rules_never_run():
    rs = RuleSet()
    rs.add(R1)  # proposed only
    assert run(demo_scores(), rs) == []
    rs.reject("R1")
    assert run(demo_scores(), rs) == []


@pytest.mark.parametrize("status", ["proposed", "rejected"])
def test_unapproved_rule_objects_are_refused(status):
    r = RuleSet().add(R1)
    r.status = status
    with pytest.raises(ValueError, match="not approved"):
        Monitor([r])


def test_raw_dict_rules_are_refused():
    with pytest.raises(TypeError):
        Monitor([{**R1, "status": "approved"}])


@pytest.mark.parametrize("change", [
    {"min_duration_s": -1}, {"min_duration_s": 0}, {"min_duration_s": math.nan},
    {"min_duration_s": math.inf}, {"min_duration_s": True}, {"min_duration_s": "2"},
    {"alert_threshold": math.nan}, {"ask_threshold": math.inf}, {"alert_threshold": 1.5},
    {"ask_threshold": 0.9, "alert_threshold": 0.8}, {"alert_mode": "email"},
    {"behavior": "dance"}, {"rule_id": " R1 "}, {"rule_id": ""},
])
def test_invalid_approved_rule_objects_are_refused(change):
    # Rule objects can be built or edited without validate_rule; the monitor re-checks them
    fields = {k: v for k, v in R1.items()}
    fields.update(change)
    r = Rule(**fields, status="approved")
    with pytest.raises(ValueError):
        Monitor([r])


def test_codex_invalid_direct_rule_is_refused():
    with pytest.raises(ValueError, match="min_duration_s"):
        Monitor([Rule("R1", "attack", -1, alert_mode="email", status="approved")])


@pytest.mark.parametrize("behavior", ["immobility", "tracking_loss"])
def test_non_classifier_behavior_is_refused(behavior):
    with pytest.raises(ValueError, match="no classifier score"):
        Monitor(approved({"rule_id": "R2", "behavior": behavior, "min_duration_s": 3}))


def test_duplicate_rule_ids_refused():
    r = rule().active()[0]
    with pytest.raises(ValueError, match="duplicate"):
        Monitor([r, r])


def test_rule_snapshot_cannot_be_changed_from_outside():
    rs = rule()
    m = Monitor(rs, run_id="t")
    rs.rules["R1"].min_duration_s = 0.1
    rs.reject("R1")
    m.rules[0].min_duration_s = 0.1  # a copy
    for r in demo_scores().to_dict("records"):
        m.step(r)
    assert len(m.finish()) == 2
    assert m.rules[0].min_duration_s == 2.0 and m.rules[0].status == "approved"


# ---- the synthetic demo scenario ----

def test_demo_scenario_events():
    clear, uncertain = run(demo_scores(), rule())
    assert clear["action"] == "alert" and clear["confidence"] >= 0.8
    assert clear["start_s"] == pytest.approx(2.0) and clear["end_s"] == pytest.approx(5.0)
    assert clear["evidence"]["fired_s"] == pytest.approx(4.0)
    assert clear["evidence"]["fired_frame"] == 120
    assert clear["evidence"]["bout_frames"] == 90
    assert uncertain["action"] == "ask" and 0.5 <= uncertain["confidence"] < 0.8
    assert uncertain["start_s"] == pytest.approx(7.0) and uncertain["end_s"] == pytest.approx(10.0)
    assert uncertain["evidence"]["fired_s"] == pytest.approx(9.0)


def test_demo_multi_rule():
    inv = {"rule_id": "R2", "behavior": "investigation", "min_duration_s": 2.0}
    events = run(demo_scores(), approved(R1, inv))
    assert [(e["event_id"], e["rule_id"]) for e in events] == [
        ("E-t-1", "R1"), ("E-t-2", "R1"), ("E-t-3", "R2")]


# ---- firing time (observation-time contract) ----

def test_fires_on_first_frame_at_or_after_min_duration():
    log, events = fired_log(rows([0.9] * 300), rule())
    assert [f for f, _ in log] == [60]  # t = 2.0 s, the earliest observation with 2 s elapsed
    snap = log[0][1]
    assert snap["evidence"]["fired_s"] == 2.0
    assert snap["end_s"] == pytest.approx(61 / FPS)  # exclusive end of what was seen
    assert snap["evidence"]["bout_closed"] is False
    assert events[0]["end_s"] == pytest.approx(10.0)
    assert events[0]["evidence"]["end_reason"] == "stream_end"


@pytest.mark.parametrize("fps", [10.0, 29.97, 30.0, 60.0])
def test_never_fires_before_elapsed_minimum(fps):
    data = demo_scores(fps=fps).to_dict("records")
    log, _ = fired_log(data, rule(), fps=fps)
    assert len(log) == 2
    for frame, e in log:
        t = data[frame]["time_s"]
        assert t - e["start_s"] >= 2.0 - 1e-9
        # one frame earlier would have been too early
        assert t - 1 / fps - e["start_s"] < 2.0 - 1e-9


@pytest.mark.parametrize("n_frames, expected", [(60, 0), (61, 1), (62, 1)])
def test_duration_boundary(n_frames, expected):
    data = [0.0] + [0.9] * n_frames + [0.0]
    assert len(run(rows(data), rule())) == expected


def test_non_integer_duration():
    r = rule(min_duration_s=1.01)  # elapsed 31/30 = 1.033 s is the first >= 1.01
    assert run(rows([0.9] * 31 + [0.0]), r) == []
    assert len(run(rows([0.9] * 32 + [0.0]), r)) == 1


def test_tiny_duration_needs_two_observations():
    r = rule(min_duration_s=0.001)
    assert run(rows([0.9, 0.0]), r) == []  # one observation has 0 s elapsed
    assert len(run(rows([0.9, 0.9]), r)) == 1


JITTER = 2e-5  # < 0.1 % of a 30 fps frame period (3.3e-5 s): accepted by validation


def _bout_with_frame60_at(t60):
    data = rows([0.9] * 70)
    data[60] = dict(data[60], time_s=t60)
    return data


def test_early_jitter_on_boundary_frame_delays_firing():
    # Codex: frame 60 stamped at 1.99998 s must not fire a 2 s rule
    log, _ = fired_log(_bout_with_frame60_at(2.0 - JITTER), rule())
    assert [f for f, _ in log] == [61]
    frame, e = log[0]
    assert e["evidence"]["fired_s"] - e["start_s"] >= 2.0


def test_codex_jitter_case():
    m = Monitor(rule(), run_id="t")
    for r in rows([0.6] * 60):
        m.step(r)
    early = dict(rows([0.6], start_frame=60)[0], time_s=2.0 - JITTER)
    for event in m.step(early):
        assert early["time_s"] - event["start_s"] >= 2.0 - 1e-9


def test_late_jitter_and_exact_boundary_fire_on_frame_60():
    for t60 in (2.0, 2.0 + JITTER):
        log, _ = fired_log(_bout_with_frame60_at(t60), rule())
        assert [f for f, _ in log] == [60]
        assert log[0][1]["evidence"]["fired_s"] == t60


def test_jitter_on_start_frame_is_measured_from_observed_start():
    data = rows([0.9] * 70)
    data[0] = dict(data[0], time_s=JITTER)  # start observed late -> frame 60 is 1.99998 s later
    log, _ = fired_log(data, rule())
    assert [f for f, _ in log] == [61]


def test_jitter_beyond_tolerance_is_rejected():
    with pytest.raises(MonitorInputError, match="regular sampling"):
        run(_bout_with_frame60_at(2.0 - 4e-5), rule())


def test_never_fires_before_observed_duration_under_random_jitter():
    rng = np.random.default_rng(7)
    data = rows([0.9] * 400)
    for r in data[1:]:  # keep frame 0 exact so the session origin is clean
        r["time_s"] += float(rng.uniform(-0.9, 0.9)) * 1e-3 / FPS
    log, _ = fired_log(data, rule(min_duration_s=0.5))
    by_frame = {r["frame"]: r for r in data}
    assert log
    for frame, e in log:
        assert by_frame[frame]["time_s"] - e["start_s"] >= 0.5 - 1e-9
        # and it fired on the first frame that qualified, not later
        assert by_frame[frame - 1]["time_s"] - e["start_s"] < 0.5 - 1e-9


def test_negative_session_time_is_rejected():
    with pytest.raises(MonitorInputError, match="since session start"):
        run([dict(rows([0.9])[0], time_s=-0.1)], rule())


def test_pose_confidence_column_is_not_a_behavior_score():
    # Igor's pose table has a keypoint "score" (pose confidence 0..1). It must
    # never stand in for a classifier probability.
    r = dict(rows([0.9])[0], score=0.99)
    del r["attack"]
    with pytest.raises(MonitorInputError, match="missing columns"):
        Monitor(rule()).step(r)
    data = [dict(x, score=0.99) for x in rows([0.1] * 61)]
    assert run(data, rule()) == []  # high pose confidence, low attack: no event


# ---- thresholds and confidence ----

def test_score_exactly_at_ask_threshold_counts():
    assert len(run(rows([0.5] * 61), rule())) == 1
    assert run(rows([0.4999] * 61), rule()) == []


def test_confidence_at_alert_threshold_is_alert():
    (e,) = run(rows([0.8] * 61), rule())
    assert e["action"] == "alert" and e["confidence"] == pytest.approx(0.8)
    (e,) = run(rows([0.7999] * 61), rule())
    assert e["action"] == "ask"


def test_confidence_is_mean_of_bout_so_far():
    # mean 0.7525 -> ask, even though half the frames are above alert_threshold
    (e,) = run(rows([0.6] * 30 + [0.9] * 31 + [0.0]), rule())
    assert e["confidence"] == pytest.approx((0.6 * 30 + 0.9 * 31) / 61) and e["action"] == "ask"


@pytest.mark.parametrize("mode, score, action", [
    ("alert", 0.95, "alert"), ("alert", 0.6, "ask"),
    ("ask", 0.95, "ask"), ("ask", 0.6, "ask"),
    ("log", 0.95, "log"), ("log", 0.6, "log"),
])
def test_alert_modes(mode, score, action):
    (e,) = run(rows([score] * 90), rule(alert_mode=mode))
    assert e["action"] == action


def test_decision_fixed_at_firing_and_bout_stats_separate():
    (e,) = run(rows([0.9] * 61 + [0.55] * 60), rule())
    ev = e["evidence"]
    assert e["action"] == "alert" and e["confidence"] == pytest.approx(0.9)
    # §4 evidence fields = what the decision was based on
    assert ev["mean_score"] == pytest.approx(0.9) and ev["min_score"] == pytest.approx(0.9)
    assert ev["frames"] == 61
    # retrospective whole-bout statistics in separate fields
    assert ev["bout_frames"] == 121
    assert ev["bout_mean_score"] == pytest.approx((0.9 * 61 + 0.55 * 60) / 121)
    assert ev["bout_min_score"] == pytest.approx(0.55) and ev["bout_max_score"] == pytest.approx(0.9)
    # uncertain first, then very high: stays one "ask", never upgraded or repeated
    (e,) = run(rows([0.6] * 61 + [0.99] * 60), rule())
    assert e["action"] == "ask" and e["confidence"] == pytest.approx(0.6)


def test_returned_event_is_a_copy():
    m = Monitor(rule(), run_id="t")
    for r in rows([0.9] * 61):
        fired = m.step(r)
    fired[0]["researcher_answer"] = "confirmed"
    fired[0]["evidence"]["frames"] = 999
    (e,) = m.finish()
    assert e["researcher_answer"] is None and e["evidence"]["frames"] == 61


# ---- duplicates ----

def test_long_bout_gives_one_event():
    log, events = fired_log(rows([0.95] * 3000), rule())  # 100 s
    assert len(log) == 1 and len(events) == 1


def test_one_low_frame_breaks_a_bout():
    assert run(rows([0.9] * 60 + [0.1] + [0.9] * 60), rule()) == []


def test_two_bouts_give_two_events():
    events = run(rows([0.9] * 70 + [0.1] + [0.9] * 70), rule())
    assert [e["event_id"] for e in events] == ["E-t-1", "E-t-2"]
    assert events[1]["start_s"] == pytest.approx(71 / FPS)


def test_two_rules_same_behavior_are_independent():
    strict = {**R1, "rule_id": "R2", "ask_threshold": 0.85, "alert_threshold": 0.9}
    assert [e["rule_id"] for e in run(rows([0.7] * 90), approved(R1, strict))] == ["R1"]
    events = run(rows([0.95] * 90), approved(R1, strict))
    assert sorted(e["rule_id"] for e in events) == ["R1", "R2"]
    assert len({e["event_id"] for e in events}) == 2


# ---- tracking loss, missing scores, gaps ----

def test_tracking_loss_interrupts_bout():
    ok = [True] * 30 + [False] * 15 + [True] * 45
    assert run(rows([0.95] * 90, ok), rule()) == []


def test_high_scores_during_tracking_loss_are_ignored():
    assert run(rows([0.99] * 120, [False] * 120), rule()) == []


def test_tracking_loss_after_firing_ends_event():
    ok = [True] * 75 + [False] * 30 + [True] * 75
    first, second = run(rows([0.95] * 180, ok), rule())
    assert first["end_s"] == pytest.approx(2.5)  # = time of the first lost frame
    assert first["evidence"]["end_reason"] == "tracking_lost"
    assert first["evidence"]["bout_frames"] == 75
    assert second["start_s"] == pytest.approx(105 / FPS)


@pytest.mark.parametrize("false", [False, 0, np.bool_(False), np.int64(0), "False", "false", " 0 "])
def test_tracking_ok_false_values(false):
    assert run(rows([0.95] * 90, [false] * 90), rule()) == []


@pytest.mark.parametrize("true", [True, 1, np.bool_(True), np.int64(1), "True", "TRUE", "1"])
def test_tracking_ok_true_values(true):
    assert len(run(rows([0.95] * 90, [true] * 90), rule())) == 1


@pytest.mark.parametrize("bad", [None, math.nan, np.float32("nan"), math.inf, 2, -1,
                                 0.5, 1.0, "yes", "", "ok"])
def test_invalid_tracking_values_are_rejected(bad):
    with pytest.raises(MonitorInputError, match="tracking_ok"):
        run(rows([0.95] * 90, [bad] * 90), rule())


@pytest.mark.parametrize("missing", [math.nan, None, np.float64("nan")])
def test_missing_score_breaks_bout(missing):
    assert run(rows([0.9] * 40 + [missing] + [0.9] * 40), rule()) == []


@pytest.mark.parametrize("bad", [-0.1, 1.1, math.inf, -math.inf, "0.9", True, [0.9]])
def test_invalid_probabilities_are_rejected(bad):
    with pytest.raises(MonitorInputError, match="attack"):
        run(rows([0.9] * 5 + [bad]), rule())


def test_probability_bounds_are_accepted():
    assert len(run(rows([1.0] * 61), rule())) == 1
    assert run(rows([0.0] * 61), rule(ask_threshold=0.0, alert_threshold=0.0)) != []


def test_missing_score_column_is_rejected():
    r = rows([0.9])[0]
    del r["attack"]
    with pytest.raises(MonitorInputError, match="missing columns"):
        Monitor(rule()).step(r)


def test_unused_score_columns_are_not_required():
    r = rows([0.9])[0]
    del r["mount"]
    Monitor(rule()).step(r)  # only "attack" is monitored


def test_skipped_frames_break_bout():
    # 80 frames that would span 2.67 s if the gap were ignored
    assert run(rows([0.9] * 40) + rows([0.9] * 40, start_frame=41), rule()) == []


def test_timestamp_jump_is_rejected():
    with pytest.raises(MonitorInputError, match="regular sampling"):
        run([dict(rows([0.9])[0]), dict(rows([0.9], start_frame=1)[0], time_s=10.0)], rule())


@pytest.mark.parametrize("t", [0.01, 1 / FPS + 0.001, 1 / FPS - 0.001])
def test_timestamp_inconsistent_with_fps_is_rejected(t):
    second = dict(rows([0.9], start_frame=1)[0], time_s=t)
    with pytest.raises(MonitorInputError, match="regular sampling"):
        run(rows([0.9]) + [second], rule())


def test_frame_gap_needs_matching_timestamp():
    gap_ok = rows([0.9], start_frame=10)[0]
    run(rows([0.9]) + [gap_ok], rule())  # 10 frames later, 10/30 s later: fine
    with pytest.raises(MonitorInputError, match="regular sampling"):
        run(rows([0.9]) + [dict(gap_ok, time_s=1 / FPS)], rule())


def test_session_may_start_at_any_frame_and_time():
    data = rows([0.9] * 61, start_frame=500)
    (e,) = run(data, rule())
    assert e["start_s"] == pytest.approx(500 / FPS)


@pytest.mark.parametrize("field, bad", [
    ("frame", -1), ("frame", 1.0), ("frame", "3"), ("frame", True),
    ("time_s", math.nan), ("time_s", math.inf), ("time_s", -1e-6), ("time_s", "0.0"), ("time_s", None),
    ("session", ""), ("session", "  "), ("session", 7), ("session", None),
])
def test_invalid_row_fields_are_rejected(field, bad):
    r = dict(rows([0.9])[0], **{field: bad})
    with pytest.raises(MonitorInputError, match=field):
        Monitor(rule()).step(r)


@pytest.mark.parametrize("order", [[0, 1, 1], [0, 2, 1]])
def test_out_of_order_frames_are_rejected(order):
    base = rows([0.1] * 3)
    with pytest.raises(MonitorInputError, match="frame"):
        run([base[i] for i in order], rule())


def test_pandas_series_rows_are_accepted():
    df = demo_scores()
    m = Monitor(rule(), run_id="t")
    for _, r in df.iterrows():
        m.step(r)
    assert len(m.finish()) == 2


# ---- atomicity of rejected rows ----

def two_rules(duration):
    return approved({**R1, "min_duration_s": duration},
                    {"rule_id": "R2", "behavior": "mount", "min_duration_s": duration})


def test_rejected_first_row_can_be_retried():
    m = Monitor(two_rules(1 / FPS), run_id="t")
    good = rows([0.9] * 2, mount=[0.9] * 2)
    bad = dict(good[0])
    del bad["mount"]
    with pytest.raises(MonitorInputError):
        m.step(bad)
    assert m.step(good[0]) == []
    assert {e["rule_id"] for e in m.step(good[1])} == {"R1", "R2"}


def test_rejected_later_row_loses_no_frame_or_alert():
    m = Monitor(two_rules(2 / FPS), run_id="t")
    good = rows([0.9] * 3, mount=[0.9] * 3)
    m.step(good[0])
    m.step(good[1])
    for bad in (dict(good[2], mount=1.5), dict(good[2], tracking_ok="maybe"),
                dict(good[2], time_s=5.0), {k: v for k, v in good[2].items() if k != "mount"}):
        with pytest.raises(MonitorInputError):
            m.step(bad)
    assert {e["rule_id"] for e in m.step(good[2])} == {"R1", "R2"}
    assert [e["evidence"]["bout_frames"] for e in m.finish()] == [3, 3]


def test_rejected_new_session_row_keeps_current_session_open():
    m = Monitor(rule(), run_id="t")
    a = rows([0.9] * 90, session="SYNTHETIC_a")
    for r in a[:45]:
        m.step(r)
    with pytest.raises(MonitorInputError):
        m.step(dict(rows([0.9], session="SYNTHETIC_b")[0], attack=2.0))
    for r in a[45:]:  # session a continues as if nothing happened
        m.step(r)
    (e,) = m.finish()
    assert e["session"] == "SYNTHETIC_a" and e["evidence"]["bout_frames"] == 90


# ---- sessions ----

def test_bout_does_not_cross_session_boundary():
    data = rows([0.9] * 45, session="SYNTHETIC_a") + rows([0.9] * 45, session="SYNTHETIC_b")
    assert run(data, rule()) == []


def test_event_closed_at_session_end():
    data = (rows([0.1] * 10 + [0.9] * 70, session="SYNTHETIC_a")
            + rows([0.9] * 61, session="SYNTHETIC_b"))
    a, b = run(data, rule())
    assert a["session"] == "SYNTHETIC_a" and a["evidence"]["end_reason"] == "session_end"
    assert a["end_s"] == pytest.approx(80 / FPS) and a["evidence"]["bout_frames"] == 70
    assert b["session"] == "SYNTHETIC_b" and b["start_s"] == 0.0
    assert b["evidence"]["end_reason"] == "stream_end"


def test_event_ids_are_unique_across_sessions():
    data = rows([0.9] * 61, session="SYNTHETIC_a") + rows([0.9] * 61, session="SYNTHETIC_b")
    assert [e["event_id"] for e in run(data, rule())] == ["E-t-1", "E-t-2"]


def test_session_returning_is_rejected():
    data = (rows([0.1] * 5, session="SYNTHETIC_a") + rows([0.1] * 5, session="SYNTHETIC_b")
            + rows([0.1] * 5, session="SYNTHETIC_a", start_frame=5))
    with pytest.raises(MonitorInputError, match="appeared again"):
        run(data, rule())


def test_finish_is_idempotent_and_step_after_finish_fails():
    m = Monitor(rule(), run_id="t")
    for r in rows([0.9] * 61):
        m.step(r)
    a = m.finish()
    assert m.finish() == a and len(a) == 1
    with pytest.raises(RuntimeError):
        m.step(rows([0.1], start_frame=61)[0])
    assert run([], rule()) == []


@pytest.mark.parametrize("fps", [0, -30, math.nan, math.inf, True])
def test_bad_fps(fps):
    with pytest.raises(ValueError):
        Monitor(rule(), fps=fps)


@pytest.mark.parametrize("run_id", ["", "a-b", "x" * 65, "a b", 5])
def test_bad_run_id(run_id):
    with pytest.raises(ValueError):
        Monitor(rule(), run_id=run_id)


# ---- causality ----

def _key(frame, e):
    return (frame, e["event_id"], e["action"], e["confidence"], e["start_s"],
            e["evidence"]["fired_s"], e["evidence"]["frames"])


def test_future_frames_do_not_change_past_decisions():
    full = demo_scores().to_dict("records")
    full_log, _ = fired_log(full, rule())
    rng = np.random.default_rng(42)
    for cut in (100, 119, 120, 269, 270, 400, len(full) - 1):
        prefix_log, _ = fired_log(full[:cut + 1], rule())
        expected = [_key(f, e) for f, e in full_log if f <= cut]
        assert [_key(f, e) for f, e in prefix_log] == expected
        # scramble everything after the cut: decisions up to the cut stay the same
        future = [dict(r, attack=float(rng.uniform()), tracking_ok=bool(rng.uniform() < 0.8))
                  for r in full[cut + 1:]]
        mixed_log, _ = fired_log(full[:cut + 1] + future, rule())
        assert [_key(f, e) for f, e in mixed_log if f <= cut] == expected


def test_deterministic_runs():
    assert run(demo_scores(), rule()) == run(demo_scores(), rule())
