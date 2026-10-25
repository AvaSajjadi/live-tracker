"""Tests for schedule_workflow: propose -> Confirm / Reject, stale clicks, saving, no browser needed."""

import json
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from live_tracker import schedule_workflow as sw

NY = "America/New_York"
NOW = datetime(2026, 10, 24, 17, 0, tzinfo=ZoneInfo(NY))


@pytest.fixture
def wf(tmp_path):
    return sw.ScheduleWorkflow(directory=tmp_path / "schedules", now_fn=lambda: NOW)


def test_nothing_is_approved_at_the_start(wf):
    assert wf.pending is None and wf.decided == [] and wf.approved() == [] and wf.table() == []


def test_a_proposal_is_pending_and_not_approved(wf):
    s = wf.propose_for_hours(8, NY)
    assert s.status == "proposed" and wf.pending.schedule is s and wf.approved() == []
    assert s.planned_s == 8 * 3600 and s.start == NOW


def test_confirm_approves_and_saves_a_file(wf, tmp_path):
    wf.propose_between(time(18), time(6), NY)
    token = wf.pending.token
    approved = wf.confirm(token)
    assert approved.status == "approved" and wf.pending is None and wf.approved() == [approved]
    files = list((tmp_path / "schedules").glob("*.json"))
    assert len(files) == 1 and not list((tmp_path / "schedules").glob("*.tmp"))
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["status"] == "approved" and saved["tz"] == NY and saved["mode"] == "window"
    assert saved["start"].endswith("-04:00") and saved["end"].startswith("2026-10-25T06:00")


def test_reject_saves_nothing(wf, tmp_path):
    wf.propose_for_hours(2, NY)
    wf.reject(wf.pending.token)
    assert [s.status for s in wf.decided] == ["rejected"] and wf.approved() == []
    assert not (tmp_path / "schedules").exists()


def test_a_stale_token_is_refused_and_changes_nothing(wf):
    wf.propose_for_hours(2, NY)
    old = wf.pending.token
    wf.propose_for_hours(3, NY)                       # a newer proposal replaces the old one
    with pytest.raises(sw.StaleActionError):
        wf.confirm(old)
    with pytest.raises(sw.StaleActionError):
        wf.reject(old)
    assert wf.decided == [] and wf.pending.schedule.planned_s == 3 * 3600


def test_the_same_proposal_cannot_be_confirmed_twice(wf):
    wf.propose_for_hours(2, NY)
    token = wf.pending.token
    wf.confirm(token)
    with pytest.raises(sw.StaleActionError):
        wf.confirm(token)
    assert len(wf.approved()) == 1


def test_a_bad_request_gives_a_message_and_no_proposal(wf):
    assert wf.propose_for_hours(0, NY) is None
    assert wf.pending is None and wf.error == "hours must be more than 0"
    assert wf.propose_between(time(6), time(6), NY) is None and "same" in wf.error
    assert wf.propose_for_hours(2, "Mars/Base") is None and "time zone" in wf.error


def test_ids_are_only_used_by_valid_proposals(wf):
    wf.propose_for_hours(0, NY)                       # refused
    assert wf.propose_for_hours(2, NY).schedule_id == "S1"
    wf.confirm(wf.pending.token)
    assert wf.propose_for_hours(2, NY).schedule_id == "S2"


def test_changed_inputs_drop_the_pending_proposal(wf):
    wf.propose_for_hours(2, NY, signature=("a",))
    assert wf.sync(("a",)) is False and wf.pending is not None
    assert wf.sync(("b",)) is True and wf.pending is None


def test_a_save_failure_keeps_the_proposal_pending(tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x", encoding="utf-8")                           # a file where the folder should be
    wf = sw.ScheduleWorkflow(directory=blocker / "schedules", now_fn=lambda: NOW)
    wf.propose_for_hours(2, NY)
    with pytest.raises(sw.DecisionError, match="still pending"):
        wf.confirm(wf.pending.token)
    assert wf.pending is not None and wf.decided == []


def test_the_table_lists_decided_schedules(wf):
    wf.propose_for_hours(8, NY)
    wf.confirm(wf.pending.token)
    wf.propose_between(time(18), time(6), NY)
    wf.reject(wf.pending.token)
    rows = wf.table()
    assert [r["schedule"] for r in rows] == ["S1", "S2"] and [r["status"] for r in rows] == ["approved", "rejected"]
    assert rows[0]["length"] == "8 h 00 min" and rows[1]["length"] == "12 h 00 min" and rows[0]["time zone"] == NY


def test_default_folder_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(sw.SAVE_DIR_ENV, str(tmp_path / "from_env"))
    assert sw.save_dir() == tmp_path / "from_env"
    wf = sw.ScheduleWorkflow(now_fn=lambda: NOW)
    wf.propose_for_hours(1, NY)
    wf.confirm(wf.pending.token)
    assert len(list((tmp_path / "from_env").glob("*.json"))) == 1
