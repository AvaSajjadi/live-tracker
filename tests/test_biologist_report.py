"""The written report for the biologist: the facts Claude gets, the request, the saved file and the phone message."""

import json
from types import SimpleNamespace

import pytest

from live_tracker import biologist_report as bio

REPORT = {
    "run": {"outcome": "complete", "message": "The schedule ended.", "plan": "From 03:02 to 03:05 (0 h 03 min)",
            "completeness": {"planned_s": 180.0, "recorded_s": 176.0}, "interruptions": [], "max_lag_s": 0.4},
    "summary": {
        "recording": {"duration_s": 176.2},
        "rules": {"rules": [{"rule_id": "R2", "behavior": "attack", "min_duration_s": 2.0,
                             "description": "tell me if mice fight"}]},
        "counts": {"alert": 1, "ask": 0, "log": 0}, "answers": {"confirmed": 0, "rejected": 0, "unanswered": 0},
        "events": [{"rule_id": "R2", "behavior": "attack", "action": "alert", "onset_s": 15.8, "fired_s": 17.8,
                    "confidence_model_score": 0.8928,
                    "closure": {"ending": "tracking was lost (the behavior may have continued)", "duration_s": 2.9}}],
        "tracking_warnings": {"supplied": False},
    },
}


class FakeClient:
    def __init__(self, text="Two mice fought at 0:16.", fail=False):
        self.calls, self.text, self.fail = [], text, fail
        self.messages = self

    def with_options(self, **kw):
        return self

    def create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("secret detail")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


def test_facts_hold_times_scores_and_no_paths():
    f = bio.facts(json.dumps(REPORT))
    assert f["events"][0]["started_at"] == "0:16" and f["events"][0]["model_score"] == 0.89
    assert f["counts"]["alert"] == 1 and f["recording_length"] == "2:56"
    assert "path" not in json.dumps(f).lower()


def test_write_makes_one_request_and_adds_the_exact_table():
    client = FakeClient()
    md, problem = bio.write(json.dumps(REPORT), client=client)
    assert problem is None and len(client.calls) == 1
    assert "Two mice fought at 0:16." in md and "Exact facts" in md and "| R2 | attack | alert | 0:16 | 0:18 | 0.89 |" in md
    assert client.calls[0]["model"] == bio.DEFAULT_MODEL and "biologist" in client.calls[0]["system"]


def test_a_failed_request_is_reported_without_the_error_text():
    md, problem = bio.write(json.dumps(REPORT), client=FakeClient(fail=True))
    assert md is None and "failed" in problem and "secret" not in problem


def test_an_empty_reply_and_a_missing_key_are_reported(monkeypatch):
    md, problem = bio.write(json.dumps(REPORT), client=FakeClient(text=""))
    assert md is None and "no text" in problem
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    md, problem = bio.write(json.dumps(REPORT))
    assert md is None and "ANTHROPIC_API_KEY" in problem


def test_an_unreadable_run_report_is_reported():
    md, problem = bio.write("{}", client=FakeClient())
    assert md is None and "could not be read" in problem


def test_save_writes_the_file_and_says_the_phone_message_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("LIVE_TRACKER_NTFY_TOPIC", raising=False)
    note = bio.save_and_notify(tmp_path, "# hi", env={})
    assert (tmp_path / bio.FILE_NAME).read_text(encoding="utf-8") == "# hi"
    assert "off" in note and "Written report ready" in (tmp_path / "notifications.log").read_text(encoding="utf-8")


def test_save_sends_a_phone_message_when_a_topic_is_set(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr("live_tracker.notify.NtfyNotifier.send", lambda self, title, message, priority="default": sent.append(title))
    note = bio.save_and_notify(tmp_path, "# hi", env={"LIVE_TRACKER_NTFY_TOPIC": "longrandomtopic"})
    assert sent == ["Written report ready"] and "phone message was sent" in note
