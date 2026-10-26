"""A written report for the biologist, from a finished run (new file).

Claude writes the text from the facts of the run's report.json (one request, only when the researcher clicks the
button). It sees no video, no keypoints and no file paths. The exact facts (a table made by this program) are always
added after Claude's text, so a number never rests on Claude's wording alone. The text is saved in the run folder and
a short phone message goes out when the researcher has set up ntfy (see notify.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import notify
from .claude_agent import MODEL_ENV, ClaudeConfig, ClaudeUnavailable, _bounded, _error_reason, _make_client

DEFAULT_MODEL = "claude-sonnet-5-5"
FILE_NAME = "biologist_report.md"
MAX_TOKENS = 1200
SYSTEM_PROMPT = (
    "You write a short report for a biologist about one analysed mouse video. You get the facts of the analysis as "
    "JSON. Write in plain, clear English, as a short letter with these parts: what was analysed, what happened (in "
    "time order), how much to trust it, and what to check by eye. Rules: use only the facts given; never invent a "
    "time, a number or a behavior; give times as minutes and seconds; say that the events are detections by a "
    "computer model on keypoint scores, not human labels, and that confidence is a model score, not a probability; "
    "if tracking was lost or the video source stopped, say the behavior may have lasted longer; do not claim any "
    "scientific or welfare finding; if nothing was detected, say so plainly. Keep it under 250 words. No headings "
    "except the four parts above.")


def _mmss(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def facts(report_json):
    """The small set of facts Claude is given, from the text of report.json."""
    data = json.loads(report_json)
    run, summary = data["run"], data["summary"]
    events = []
    for e in summary.get("events", []):
        closure = e.get("closure") or {}
        events.append({"rule": e.get("rule_id"), "behavior": e.get("behavior"), "action": e.get("action"),
                       "started_at": _mmss(e["onset_s"]), "alert_fired_at": _mmss(e["fired_s"]),
                       "model_score": round(e["confidence_model_score"], 2),
                       "duration_s": round(closure["duration_s"], 1) if closure.get("duration_s") is not None else None,
                       "how_it_ended": closure.get("ending")})
    done = run.get("completeness") or {}
    return {"outcome": run.get("outcome"), "why_it_ended": run.get("message"), "plan": run.get("plan"),
            "recording_length": _mmss(summary["recording"]["duration_s"]),
            "planned_seconds": done.get("planned_s"), "recorded_seconds": done.get("recorded_s"),
            "video_source_interruptions": len(run.get("interruptions") or []),
            "slowest_delay_s": run.get("max_lag_s"),
            "rules": [{"rule": r["rule_id"], "behavior": r["behavior"], "your_words": r.get("description"),
                       "min_duration_s": r["min_duration_s"]} for r in summary["rules"]["rules"]],
            "counts": summary["counts"], "answers": summary["answers"], "events": events,
            "tracking_warnings_supplied": bool((summary.get("tracking_warnings") or {}).get("supplied"))}


def facts_table(f):
    """The same facts as a plain table, made by this program (not by Claude)."""
    lines = ["## Exact facts (made by the program)", "",
             f"- Outcome: {f['outcome']}. {f['why_it_ended']}",
             f"- Plan: {f['plan']}",
             f"- Alerts: {f['counts'].get('alert', 0)}, questions: {f['counts'].get('ask', 0)}", ""]
    if f["events"]:
        lines += ["| rule | behavior | action | started | fired | model score | duration (s) | how it ended |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {e['rule']} | {e['behavior']} | {e['action']} | {e['started_at']} | {e['alert_fired_at']} | "
                  f"{e['model_score']} | {e['duration_s']} | {e['how_it_ended']} |" for e in f["events"]]
    else:
        lines.append("No event fired.")
    return "\n".join(lines)


def write(report_json, *, client=None, config=None):
    """(markdown, None) with Claude's text and the exact facts, or (None, message) when it cannot be written.
    At most one API request."""
    try:
        f = facts(report_json)
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"The run report could not be read ({type(exc).__name__})."
    cfg = config or ClaudeConfig(max_tokens=MAX_TOKENS, timeout_s=40.0)
    model = (cfg.model or os.environ.get(MODEL_ENV, "")).strip() or DEFAULT_MODEL
    if client is None:
        client, unavailable = _make_client(cfg)
        if unavailable:
            return None, unavailable.message
    client, unavailable = _bounded(client, cfg)
    if unavailable:
        return None, unavailable.message
    try:
        response = client.messages.create(
            model=model, max_tokens=cfg.max_tokens, system=SYSTEM_PROMPT, timeout=cfg.timeout_s,
            messages=[{"role": "user", "content": json.dumps(f, indent=1)}])
    except Exception as exc:                          # the SDK's errors; reported by class name, never by text
        return None, f"The Claude request failed ({_error_reason(exc)}, {type(exc).__name__}). Try again later."
    text = "".join(getattr(b, "text", "") for b in (getattr(response, "content", None) or [])
                   if getattr(b, "type", None) == "text").strip()
    if not text:
        return None, "Claude's reply had no text."
    return (f"# Report for the biologist\n\n{text}\n\n_Written by Claude ({model}) from the facts below. "
            f"Check the table; it is not written by Claude._\n\n{facts_table(f)}\n"), None


def write_plain(report_json):
    """(markdown, None) made by this program alone (no Claude, no cost), or (None, message). Same facts table."""
    try:
        f = facts(report_json)
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"The run report could not be read ({type(exc).__name__})."
    return (f"# Report for the biologist\n\n_Made by the program from the facts of the run. Claude did not write "
            f"this text. Events are detections by a computer model on keypoint scores, not human labels. "
            f"Check them by eye._\n\nRecording length: {f['recording_length']}.\n\n{facts_table(f)}\n"), None


def save_and_notify(run_dir, markdown, env=None):
    """Save the report in the run folder and send a short phone message when ntfy is set up. Returns a sentence."""
    run_dir = Path(run_dir)
    (run_dir / FILE_NAME).write_text(markdown, encoding="utf-8")
    channels = notify.from_env(run_dir / "notifications.log", env)
    channels.send("Written report ready", f"The written report for run {run_dir.name} is ready on the dashboard "
                  f"(Real session page).", "default")
    if channels.failures:
        return "The report is saved, but the phone message failed."
    sent = bool((os.environ if env is None else env).get(notify.TOPIC_ENV, "").strip())
    return "The report is saved. A phone message was sent." if sent else (
        "The report is saved. The phone message is off (no ntfy topic is set).")


__all__ = ["DEFAULT_MODEL", "FILE_NAME", "facts", "facts_table", "write", "write_plain", "save_and_notify", "ClaudeUnavailable"]
