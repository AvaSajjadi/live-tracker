"""Notifications for a run: short messages to the researcher (new file).

Two channels:
  - LogNotifier: one line per message in a text file of the run folder. Always on.
  - NtfyNotifier: a push message to a phone through an ntfy server. OFF unless the researcher sets
    LIVE_TRACKER_NTFY_TOPIC (and optionally LIVE_TRACKER_NTFY_SERVER). The topic works like a
    password: anyone who knows it can read the messages, so pick a long random name.

A message names the behavior, the confidence and the time. It never contains video, positions or
file paths. A failing channel never stops a run: MultiNotifier records the failure and goes on.
Headers are plain ASCII on purpose (ntfy sends them as HTTP headers).
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TOPIC_ENV = "LIVE_TRACKER_NTFY_TOPIC"
SERVER_ENV = "LIVE_TRACKER_NTFY_SERVER"
DEFAULT_SERVER = "https://ntfy.sh"
PRIORITIES = ("low", "default", "high")


class Notifier:
    def send(self, title, message, priority="default"):
        raise NotImplementedError


def _ascii(text):
    return str(text).encode("ascii", "replace").decode("ascii").replace("\n", " ").replace("\r", " ")


class LogNotifier(Notifier):
    """Appends 'time | priority | title | message' to a file."""

    def __init__(self, path, now_fn=lambda: datetime.now(timezone.utc)):
        self.path, self.now_fn = Path(path), now_fn

    def send(self, title, message, priority="default"):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = f"{self.now_fn():%Y-%m-%d %H:%M:%S}Z | {priority} | {title} | {message}\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)


class NtfyNotifier(Notifier):
    def __init__(self, topic, server=DEFAULT_SERVER, timeout=5.0, opener=urllib.request.urlopen):
        topic = str(topic or "").strip()
        if not topic or "/" in topic or " " in topic:
            raise ValueError("the ntfy topic must be one word without slashes or spaces")
        self.url = f"{str(server).rstrip('/')}/{topic}"
        self.timeout, self.opener = float(timeout), opener

    def send(self, title, message, priority="default"):
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        request = urllib.request.Request(
            self.url, data=str(message).encode("utf-8"), method="POST",
            headers={"Title": _ascii(title), "Priority": priority, "Tags": "mouse"})
        with self.opener(request, timeout=self.timeout) as response:
            response.read()


class MultiNotifier(Notifier):
    """Sends to every channel; a channel that fails is recorded in `failures` and never stops the run."""

    def __init__(self, channels):
        self.channels, self.failures = list(channels), []

    def send(self, title, message, priority="default"):
        for channel in self.channels:
            try:
                channel.send(title, message, priority)
            except Exception as exc:                         # a broken phone link must not stop a night run
                self.failures.append(f"{type(channel).__name__}: {type(exc).__name__}: {exc}")


def from_env(log_path, env=None):
    """The log channel, plus ntfy when the researcher has set a topic."""
    env = os.environ if env is None else env
    channels = [LogNotifier(log_path)]
    topic = env.get(TOPIC_ENV, "").strip()
    if topic:
        channels.append(NtfyNotifier(topic, env.get(SERVER_ENV, "").strip() or DEFAULT_SERVER))
    return MultiNotifier(channels)


# ---- the texts ----

def _clock(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def event_message(event, schedule_id, when=None):
    """(title, message, priority) for one fired event. `when` is a clock text such as '02:14'."""
    action = str(event.get("action", "log"))
    behavior = str(event.get("behavior", "behavior"))
    conf = event.get("confidence")
    conf_text = f", confidence {conf:.2f}" if isinstance(conf, (int, float)) else ""
    at = f" at {when}" if when else f", {_clock(event.get('start_s', 0))} into the run"
    if action == "alert":
        title, priority = f"ALERT: possible {behavior}", "high"
    elif action == "ask":
        title, priority = f"Question: possible {behavior}", "default"
    else:
        title, priority = f"Logged: {behavior}", "low"
    tail = " Please confirm or reject it on the dashboard." if action == "ask" else ""
    return title, f"Run {schedule_id}: possible {behavior}{conf_text}{at} (rule {event.get('rule_id', '?')}).{tail}", priority


def finish_message(schedule_id, outcome, n_alerts, n_asks, n_unanswered, report_name="report.md"):
    """(title, message, priority) at the end of a run. outcome: complete | partial | stopped | no_data | failed."""
    title = {"complete": "Run finished", "partial": "Run finished (partial)", "stopped": "Run stopped",
             "no_data": "Run ended with no data", "failed": "Run FAILED"}.get(outcome, "Run ended")
    priority = "high" if outcome in ("failed", "no_data") else "default"
    waiting = f" {n_unanswered} question{'s' if n_unanswered != 1 else ''} waiting." if n_unanswered else ""
    return title, (f"Run {schedule_id}: {n_alerts} alert{'s' if n_alerts != 1 else ''}, "
                   f"{n_asks} question{'s' if n_asks != 1 else ''}.{waiting} Report: {report_name}."), priority
