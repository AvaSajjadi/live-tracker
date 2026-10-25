"""What the dashboard does to a run, without Streamlit (new file).

The run itself is a separate program (monitor_runner). The dashboard only:
  - writes a run folder with a config and starts the program (start_run),
  - asks it to stop (stop_run: a STOP file),
  - passes the researcher's answers on (answer_event: a line in answers.jsonl; the runner is the only
    writer of the event log),
  - reads what the program wrote (read_status, read_events, read_report).
Closing the browser or the dashboard never stops a run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import monitor_runner as mr
from . import run_schedule as rs
from .event_log import EventJournal, JournalIntegrityError, load_journal

ROOT = Path(__file__).resolve().parents[2]
RUNS_ENV = "LIVE_TRACKER_RUNS_DIR"
ANSWERS = ("confirmed", "rejected")
ACTIVE_STATES = ("waiting", "running", "finishing")
STALE_AFTER_S = 30.0                  # heartbeats come every few seconds; this long without one is a warning


def runs_root():
    return Path(os.environ.get(RUNS_ENV) or ROOT / "outputs" / "runs")


# ---- start and stop ----

def new_run_dir(schedule_id, now=None, root=None):
    now = now or datetime.now(timezone.utc)
    base = Path(root) if root is not None else runs_root()
    name = f"{schedule_id}_{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"
    path, n = base / name, 1
    while path.exists():                       # two starts in the same second never share a folder
        n += 1
        path = base / f"{name}_{n}"
    return path


def build_config(run_dir, schedule, rules, source, *, model=None, speed=1.0, fps=30.0):
    """A RunConfig from an approved RunSchedule and approved Rule objects. Refuses anything not approved."""
    if schedule.status != "approved":
        raise mr.RunRefusedError("the schedule is not approved")
    if not rules or any(r.status != "approved" for r in rules):
        raise mr.RunRefusedError("every rule must be approved")
    return mr.RunConfig(run_dir=str(run_dir), schedule=schedule.to_dict(), rules=[r.to_dict() for r in rules],
                        source=dict(source), model=model, speed=float(speed), fps=float(fps))


def default_launcher(config_path):
    """Start the run program so that it keeps going when this process ends."""
    run_dir = Path(config_path).parent
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with open(run_dir / "runner.log", "ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "live_tracker.monitor_runner", "--config", str(config_path)],
                                cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kwargs)
    return proc.pid


LAUNCHER = default_launcher            # tests replace this so that no program is started


def start_run(config, launcher=None):
    """Write the config into the run folder and start the program. Returns the folder."""
    run_dir = Path(config.run_dir)
    rs.RunSchedule.from_dict(config.schedule)                 # fail here, not in the background, on a bad schedule
    mr.rules_from_dicts(config.rules)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    path.write_text(config.to_json(), encoding="utf-8")
    pid = (launcher or LAUNCHER)(path)
    (run_dir / "pid.txt").write_text(str(pid) + "\n", encoding="utf-8")
    return run_dir


def stop_run(run_dir):
    (Path(run_dir) / "STOP").write_text("stop requested from the dashboard\n", encoding="utf-8")


def stop_requested(run_dir):
    return (Path(run_dir) / "STOP").exists()


# ---- answers ----

def pid_alive(pid):
    """Is a process with this id still running? (No extra packages: ctypes on Windows, signal 0 elsewhere.)"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                                 # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_is_over(run_dir, status=None):
    """True when no program is writing to this run any more: it finished or failed, or its heartbeat has
    stopped and its process is gone (a power cut, a crash)."""
    status = status if status is not None else read_status(run_dir)
    if status is None:
        return False
    if status.get("state") not in ACTIVE_STATES:
        return True
    if not is_stale(status):
        return False
    try:
        pid = int((Path(run_dir) / "pid.txt").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False                                          # no way to know: assume it may still be alive
    return not pid_alive(pid)


def answer_event(run_dir, event_id, answer):
    """The researcher's answer to an "ask" event. Returns 'applied' or 'queued'.

    A run that is still going is the only writer of its log, so the answer waits in answers.jsonl and the
    program applies it. A run that is over has no writer left: the answer goes straight into the log and the
    report is written again, so a question left for the morning can still be answered.
    """
    if answer not in ANSWERS:
        raise ValueError(f"answer must be one of {ANSWERS}")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")
    run_dir = Path(run_dir)
    if run_is_over(run_dir):
        try:
            EventJournal(run_dir / "journal.jsonl").record_answer(event_id, answer)
        except (KeyError, JournalIntegrityError) as exc:
            raise ValueError(f"the answer was not recorded: {exc}") from exc
        if (run_dir / "report.json").exists():
            mr.rebuild_report(run_dir)
        return "applied"
    with open(run_dir / "answers.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"event_id": event_id, "answer": answer,
                            "at": datetime.now(timezone.utc).isoformat()}) + "\n")
    return "queued"


def pending_answers(run_dir):
    """{event_id: answer} for answers the dashboard wrote that the log does not show yet."""
    path = Path(run_dir) / "answers.jsonl"
    given = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                given[item["event_id"]] = item["answer"]
            except (ValueError, KeyError, TypeError):
                continue
    return {e: a for e, a in given.items() if e not in {x["event_id"] for x in read_events(run_dir) if x.get("researcher_answer")}}


# ---- reading what the program wrote ----

def read_status(run_dir):
    path = Path(run_dir) / "status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):                              # missing, or caught in the middle of a rewrite
        return None


def is_stale(status, now=None, limit_s=STALE_AFTER_S):
    """True when a run that should be alive has not written a heartbeat for a while."""
    if not status or status.get("state") not in ACTIVE_STATES:
        return False
    try:
        beat = datetime.fromisoformat(status["heartbeat_at"])
    except (KeyError, ValueError, TypeError):
        return True
    return ((now or datetime.now(timezone.utc)) - beat).total_seconds() > limit_s


def read_events(run_dir):
    """The §4 events in the run's log (an empty list when there is no log yet or it cannot be read)."""
    path = Path(run_dir) / "journal.jsonl"
    if not path.exists():
        return []
    try:
        return load_journal(path).events()
    except (OSError, ValueError, JournalIntegrityError):
        return []


def read_report(run_dir):
    """(markdown, json text) of the finished run, or None."""
    d = Path(run_dir)
    if not (d / "report.md").exists():
        return None
    try:
        return (d / "report.md").read_text(encoding="utf-8"), (d / "report.json").read_text(encoding="utf-8")
    except OSError:
        return None


def list_runs(root=None):
    """Run folders, newest first: [{'name', 'path', 'status'}]."""
    base = Path(root) if root is not None else runs_root()
    if not base.is_dir():
        return []
    out = []
    for p in sorted((q for q in base.iterdir() if q.is_dir() and (q / "config.json").exists()),
                    key=lambda q: q.stat().st_mtime, reverse=True):
        out.append({"name": p.name, "path": p, "status": read_status(p)})
    return out


def progress(status):
    """0..1 share of the planned window that has passed, from the heartbeat."""
    if not status or "schedule" not in status or "clock_now" not in status:
        return 0.0
    try:
        sched = rs.RunSchedule.from_dict(status["schedule"])
        now = datetime.fromisoformat(status["clock_now"])
        done = (now.astimezone(timezone.utc) - sched.start.astimezone(timezone.utc)).total_seconds()
        return min(1.0, max(0.0, done / sched.planned_s))
    except (ValueError, rs.ScheduleError, KeyError):
        return 0.0


# ---- what can be started here ----

def available_sources(root=None):
    """The source choices for the start form, each with whether it can run and why not."""
    base = Path(root) if root is not None else ROOT
    have_data = (base / "data/raw/calms21/calms21_task1_train.json").is_file()
    have_model = (base / "models/model_b.joblib").is_file() and (base / "models/model_b.json").is_file()
    have_manifest = (base / "splits/split_manifest.json").is_file()
    keypoints_ok = have_data and have_model and have_manifest
    missing = [n for n, ok in (("the CalMS21 train file", have_data), ("models/model_b", have_model),
                               ("splits/split_manifest.json", have_manifest)) if not ok]
    return [
        {"kind": "synthetic", "label": "Demo: made-up scores (SYNTHETIC)", "enabled": True, "reason": ""},
        {"kind": "keypoints", "label": "Recorded validation session: keypoints, then Model B, then the monitor",
         "enabled": keypoints_ok, "reason": "" if keypoints_ok else "missing: " + ", ".join(missing)},
        {"kind": "camera", "label": "Camera", "enabled": False,
         "reason": "needs a mouse pose model, which this repository does not have yet"},
    ]


def validation_sessions(root=None):
    base = Path(root) if root is not None else ROOT
    path = base / "splits/split_manifest.json"
    try:
        return list(json.loads(path.read_text(encoding="utf-8"))["validation"])
    except (OSError, ValueError, KeyError):
        return []
