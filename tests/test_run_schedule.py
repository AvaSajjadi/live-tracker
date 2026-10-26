"""Tests for run_schedule: two ways to set a run, the approval gate, clock changes, completeness."""

from dataclasses import FrozenInstanceError
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from live_tracker import run_schedule as rs

NY = "America/New_York"
Z = ZoneInfo(NY)


def at(y, m, d, hh=0, mm=0, zone=Z):
    return datetime(y, m, d, hh, mm, tzinfo=zone)


# ---- for N hours ----

def test_for_hours_from_now():
    s = rs.for_hours(8, now=at(2026, 10, 24, 18), tz=NY)
    assert s.mode == "duration" and s.status == "proposed"
    assert s.start == at(2026, 10, 24, 18) and s.end == at(2026, 10, 25, 2)
    assert s.planned_s == 8 * 3600


def test_for_hours_at_a_chosen_start():
    s = rs.for_hours(2.5, now=at(2026, 10, 24, 9), tz=NY, start=at(2026, 10, 24, 20))
    assert s.start == at(2026, 10, 24, 20) and s.end == at(2026, 10, 24, 22, 30)


def test_for_hours_counts_real_hours_across_the_fall_back_night():
    # 2026-11-01: the clocks go back at 02:00, so 20:00 + 12 real hours is 07:00 (11 hours on the wall clock)
    s = rs.for_hours(12, now=at(2026, 10, 31, 20), tz=NY)
    assert s.planned_s == 12 * 3600
    assert (s.end.hour, s.end.minute) == (7, 0) and s.end.day == 1


def test_for_hours_rejects_bad_values():
    now = at(2026, 10, 24, 18)
    for bad in (0, -1, 0.001, 169, float("nan"), True, "8", None):
        with pytest.raises(rs.ScheduleError):
            rs.for_hours(bad, now=now, tz=NY)


# ---- from A to B ----

def test_window_in_one_day():
    s = rs.between(time(9), time(17), now=at(2026, 10, 24, 7), tz=NY)
    assert s.mode == "window" and s.start == at(2026, 10, 24, 9) and s.end == at(2026, 10, 24, 17)


def test_window_across_midnight_ends_the_next_day():
    s = rs.between(time(18), time(6), now=at(2026, 10, 24, 17), tz=NY)
    assert s.start == at(2026, 10, 24, 18) and s.end == at(2026, 10, 25, 6)
    assert s.planned_s == 12 * 3600


def test_asked_in_the_middle_of_the_previous_night_it_is_running_now():
    now = at(2026, 10, 25, 2)
    s = rs.between(time(18), time(6), now=now, tz=NY)
    assert s.start == at(2026, 10, 24, 18) and s.end == at(2026, 10, 25, 6)
    assert s.status_at(now) == "running"


def test_asked_after_the_window_has_ended_it_means_the_next_one():
    s = rs.between(time(18), time(6), now=at(2026, 10, 25, 7), tz=NY)
    assert s.start == at(2026, 10, 25, 18) and s.end == at(2026, 10, 26, 6)


def test_window_with_equal_times_is_refused():
    with pytest.raises(rs.ScheduleError, match="same"):
        rs.between(time(6), time(6), now=at(2026, 10, 24), tz=NY)


def test_window_needs_clock_times():
    with pytest.raises(rs.ScheduleError):
        rs.between("18:00", time(6), now=at(2026, 10, 24), tz=NY)


def test_now_in_another_zone_still_gives_the_lab_day():
    now = datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc)          # still Oct 24, 23:00 in New York
    s = rs.between(time(18), time(6), now=now, tz=NY)
    assert s.start == at(2026, 10, 24, 18) and s.status_at(now) == "running"


# ---- clock changes ----

def test_spring_forward_window_is_one_real_hour():
    s = rs.between(time(1, 30), time(3, 30), now=at(2026, 3, 8, 0, 10), tz=NY)       # 02:00-03:00 does not exist
    assert s.planned_s == 3600


def test_a_start_time_that_does_not_exist_moves_forward():
    s = rs.between(time(2, 30), time(5), now=at(2026, 3, 8, 0, 10), tz=NY)
    assert (s.start.hour, s.start.minute) == (3, 30)


def test_fall_back_window_is_three_real_hours():
    s = rs.between(time(0, 30), time(2, 30), now=at(2026, 11, 1, 0, 10), tz=NY)       # 01:00-02:00 happens twice
    assert s.planned_s == 3 * 3600


# ---- checks on the schedule itself ----

def make(**kw):
    base = dict(schedule_id="S1", mode="window", tz=NY, start=at(2026, 10, 24, 18), end=at(2026, 10, 25, 6))
    base.update(kw)
    return rs.RunSchedule(**base)


@pytest.mark.parametrize("kw", [
    {"mode": "forever"}, {"status": "maybe"}, {"tz": "Mars/Base"}, {"tz": ""},
    {"start": datetime(2026, 10, 24, 18)},                                  # no time zone
    {"end": at(2026, 10, 24, 18)},                                          # zero length
    {"end": at(2026, 10, 24, 17)},                                          # before the start
    {"end": at(2026, 10, 24, 18, 0) + timedelta(seconds=30)},               # under a minute
    {"end": at(2026, 10, 24, 18) + timedelta(days=8)},                      # over a week
])
def test_invalid_schedules_are_refused(kw):
    with pytest.raises(rs.ScheduleError):
        make(**kw)


def test_times_are_stored_in_the_lab_zone():
    s = make(start=datetime(2026, 10, 24, 22, 0, tzinfo=timezone.utc), end=datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc))
    assert s.start.utcoffset() == timedelta(hours=-4) and s.start.hour == 18


# ---- status over time ----

def test_status_pending_running_finished_with_the_end_not_included():
    s = make()
    assert s.status_at(s.start - timedelta(seconds=1)) == "pending"
    assert s.status_at(s.start) == "running"
    assert s.status_at(s.end - timedelta(seconds=1)) == "running"
    assert s.status_at(s.end) == "finished"


def test_seconds_left():
    s = make()
    assert s.seconds_left(s.start) == 12 * 3600
    assert s.seconds_left(s.end + timedelta(hours=1)) == 0.0


def test_now_must_have_a_time_zone():
    with pytest.raises(rs.ScheduleError):
        make().status_at(datetime(2026, 10, 24, 19))


# ---- the approval gate ----

def test_only_an_approved_schedule_inside_its_time_may_run():
    s, mid = make(), at(2026, 10, 25, 1)
    assert not s.may_run(mid)                                   # proposed
    assert not s.reject().may_run(mid)                          # rejected
    ok = s.approve()
    assert ok.may_run(mid)                                      # also true after a restart in the middle
    assert not ok.may_run(s.start - timedelta(minutes=1))       # too early
    assert not ok.may_run(s.end)                                # too late


def test_approve_does_not_change_the_original():
    s = make()
    assert s.approve().status == "approved" and s.status == "proposed"
    with pytest.raises(FrozenInstanceError):
        s.status = "approved"


# ---- saving ----

def test_round_trip_through_a_dict():
    s = rs.between(time(18), time(6), now=at(2026, 10, 24, 17), tz=NY, description="night").approve()
    back = rs.RunSchedule.from_dict(s.to_dict())
    assert back == s and back.to_dict() == s.to_dict()
    assert back.to_dict()["start"].endswith("-04:00")


def test_from_dict_reports_problems():
    good = make().to_dict()
    for broken in ({**good, "start": "not a time"}, {**good, "mode": "x"}, {k: v for k, v in good.items() if k != "end"}):
        with pytest.raises(rs.ScheduleError):
            rs.RunSchedule.from_dict(broken)


def test_describe_is_readable():
    text = make().describe()
    assert "18:00" in text and "06:00" in text and "12 h 00 min" in text and NY in text


# ---- was the run complete? ----

def test_complete_run():
    s = make()
    r = rs.completeness(s, s.start + timedelta(seconds=2), s.end - timedelta(seconds=1))
    assert r["outcome"] == "complete" and r["complete"] and r["planned_s"] == 12 * 3600


def test_run_that_started_late_is_partial():
    s = make()
    r = rs.completeness(s, s.start + timedelta(hours=1), s.end)
    assert r["outcome"] == "partial" and r["missing_at_start_s"] == 3600 and r["missing_at_end_s"] == 0


def test_run_that_stopped_early_is_partial():
    s = make()                                                  # e.g. the power failed at 02:00
    r = rs.completeness(s, s.start, at(2026, 10, 25, 2))
    assert r["outcome"] == "partial" and r["missing_at_end_s"] == 4 * 3600 and r["recorded_s"] == 8 * 3600


def test_no_frames_means_no_data():
    s = make()
    assert rs.completeness(s, None, None)["outcome"] == "no_data"
    after = rs.completeness(s, s.end + timedelta(hours=1), s.end + timedelta(hours=2))
    assert after["outcome"] == "no_data" and after["recorded_s"] == 0.0


def test_frames_outside_the_window_are_clipped_and_tolerance_can_change():
    s = make()
    r = rs.completeness(s, s.start - timedelta(hours=1), s.end + timedelta(hours=1))
    assert r["outcome"] == "complete" and r["recorded_s"] == 12 * 3600
    late = rs.completeness(s, s.start + timedelta(minutes=5), s.end)
    assert late["outcome"] == "partial"
    assert rs.completeness(s, s.start + timedelta(minutes=5), s.end, tolerance_s=600)["outcome"] == "complete"


def test_time_left_is_real_time_across_the_fall_back_night():
    s = rs.for_hours(12, now=at(2026, 10, 31, 20), tz=NY)
    one_real_hour_before_the_end = s.end - timedelta(hours=1)          # same zone: the wall clock shows 06:00
    assert s.seconds_left(one_real_hour_before_the_end.astimezone(timezone.utc)) == 3600
    assert s.status_at(one_real_hour_before_the_end.astimezone(timezone.utc)) == "running"
    assert s.seconds_left(s.start.astimezone(timezone.utc)) == 12 * 3600


def test_seconds_to_start():
    s = make()
    assert s.seconds_to_start(s.start - timedelta(minutes=5)) == 300
    assert s.seconds_to_start(s.start) == 0.0 and s.seconds_to_start(s.end) == 0.0
