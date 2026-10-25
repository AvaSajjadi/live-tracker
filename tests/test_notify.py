"""Tests for notify: log file, optional ntfy push, failures never stop a run, plain-ASCII headers."""

import pytest

from live_tracker import notify


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b"ok"


class FakeOpener:
    def __init__(self, fail=False):
        self.requests, self.fail = [], fail

    def __call__(self, request, timeout=None):
        self.requests.append((request, timeout))
        if self.fail:
            raise OSError("network down")
        return FakeResponse()


def test_log_notifier_appends_lines(tmp_path):
    from datetime import datetime, timezone
    log = notify.LogNotifier(tmp_path / "run" / "notifications.log", now_fn=lambda: datetime(2026, 10, 25, 2, 14, tzinfo=timezone.utc))
    log.send("ALERT: possible attack", "Run S1: ...", "high")
    log.send("Run finished", "done")
    lines = (tmp_path / "run" / "notifications.log").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "2026-10-25 02:14:00Z | high | ALERT: possible attack | Run S1: ..."
    assert lines[1].endswith("| default | Run finished | done")


def test_ntfy_posts_the_message_with_ascii_headers():
    opener = FakeOpener()
    n = notify.NtfyNotifier("my-long-random-topic", "https://ntfy.example/", opener=opener)
    n.send("ALERT: possible attack — now", "Run S1: possible attack, confidence 0.93", "high")
    (request, timeout), = opener.requests
    assert request.full_url == "https://ntfy.example/my-long-random-topic" and request.get_method() == "POST"
    assert request.data == b"Run S1: possible attack, confidence 0.93" and timeout == 5.0
    assert request.get_header("Priority") == "high"
    request.get_header("Title").encode("ascii")                       # no error: plain ASCII only


@pytest.mark.parametrize("topic", ["", "  ", "a/b", "two words", None])
def test_ntfy_refuses_a_bad_topic(topic):
    with pytest.raises(ValueError):
        notify.NtfyNotifier(topic)


def test_ntfy_refuses_an_unknown_priority():
    with pytest.raises(ValueError):
        notify.NtfyNotifier("topic", opener=FakeOpener()).send("t", "m", "urgent!!")


def test_a_failing_channel_is_recorded_and_the_others_still_get_the_message(tmp_path):
    log = notify.LogNotifier(tmp_path / "n.log")
    multi = notify.MultiNotifier([notify.NtfyNotifier("topic", opener=FakeOpener(fail=True)), log])
    multi.send("title", "message")                                   # must not raise
    assert len(multi.failures) == 1 and "network down" in multi.failures[0]
    assert "title | message" in (tmp_path / "n.log").read_text(encoding="utf-8")


def test_from_env_is_log_only_without_a_topic(tmp_path):
    multi = notify.from_env(tmp_path / "n.log", env={})
    assert [type(c).__name__ for c in multi.channels] == ["LogNotifier"]


def test_from_env_adds_ntfy_when_a_topic_is_set(tmp_path):
    multi = notify.from_env(tmp_path / "n.log", env={notify.TOPIC_ENV: "abc123", notify.SERVER_ENV: "https://s.example"})
    assert [type(c).__name__ for c in multi.channels] == ["LogNotifier", "NtfyNotifier"]
    assert multi.channels[1].url == "https://s.example/abc123"
    default = notify.from_env(tmp_path / "n.log", env={notify.TOPIC_ENV: "abc123"})
    assert default.channels[1].url == "https://ntfy.sh/abc123"


def test_event_messages_by_action():
    ev = {"action": "alert", "behavior": "attack", "confidence": 0.934, "start_s": 7425.0, "rule_id": "R1"}
    title, message, priority = notify.event_message(ev, "S1", when="02:04")
    assert (title, priority) == ("ALERT: possible attack", "high")
    assert "confidence 0.93" in message and "at 02:04" in message and "rule R1" in message
    t, m, p = notify.event_message({**ev, "action": "ask"}, "S1")
    assert t.startswith("Question") and p == "default" and "2:03:45 into the run" in m and "confirm or reject" in m
    t, m, p = notify.event_message({**ev, "action": "log", "confidence": None}, "S1")
    assert t.startswith("Logged") and p == "low" and "confidence" not in m


def test_finish_messages():
    t, m, p = notify.finish_message("S1", "complete", 2, 1, 0)
    assert t == "Run finished" and p == "default" and m == "Run S1: 2 alerts, 1 question. Report: report.md."
    t, m, p = notify.finish_message("S1", "partial", 1, 3, 2)
    assert t == "Run finished (partial)" and "2 questions waiting" in m and "1 alert," in m
    assert notify.finish_message("S1", "failed", 0, 0, 0)[2] == "high"
    assert notify.finish_message("S1", "no_data", 0, 0, 0)[2] == "high"
