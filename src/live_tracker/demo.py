"""The live demo: a recorded video and its run, started together (new file).

A demo is a video in data/raw/calms21_video/ (git-ignored) with a file of the same session's pose keypoints
next to it (`<video name>_poses.parquet`, made by scripts/prepare_demo_session.py). Starting the demo runs the
real run program on those keypoints (keypoints, live scores from Model B, the validated demo rule, the monitor,
the event log) at real speed, and the page plays the video at the same moment.

It is a simulated live stream from a recording, and the page says so. It needs no camera. It is not a camera test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import monitor_runner as mr
from . import run_control as rc
from . import run_schedule as rs
from .evaluate import ATTACK_RULE_V1
from .rules import RuleSet

VIDEO_DIR = "data/raw/calms21_video"
MODEL = "models/model_b"
FPS = 30.0
EXTRA_S = 120.0                         # the schedule lasts this much longer than the video, so it never ends first
DEMO_RULE_ID = "DEMO1"


@dataclass(frozen=True)
class Demo:
    name: str                           # the video's file name without ".mp4"
    session: str                        # the CalMS21 session this video shows
    video: Path
    poses: Path


def find_demos(root=None):
    """Videos that have a poses file and a session in the train or validation split (never a test session)."""
    base = Path(root) if root is not None else rc.ROOT
    folder = base / VIDEO_DIR
    try:
        manifest = json.loads((base / "splits/split_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    allowed = {s.rsplit("/", 1)[-1]: s for s in manifest.get("train", []) + manifest.get("validation", [])}
    found = []
    for video in sorted(folder.glob("*.mp4")) if folder.is_dir() else []:
        poses = folder / f"{video.stem}_poses.parquet"
        if poses.exists() and video.stem in allowed:
            found.append(Demo(video.stem, allowed[video.stem], video, poses))
    return found


def model_available(root=None):
    base = Path(root) if root is not None else rc.ROOT
    return all((base / f"{MODEL}{ext}").is_file() for ext in (".joblib", ".json"))


def length_s(demo):
    """Length of the recording in seconds (from its frames)."""
    frames = pd.read_parquet(demo.poses, columns=["frame"])["frame"]
    return (int(frames.max()) + 1) / FPS


def demo_rule():
    """The validated demo rule (evaluate.ATTACK_RULE_V1), approved through the normal RuleSet gate.

    Only the researcher's click on the page calls this, so the page's button is the approval."""
    data = {k: ATTACK_RULE_V1[k] for k in ("behavior", "min_duration_s", "alert_threshold", "ask_threshold", "alert_mode")}
    data.update(rule_id=DEMO_RULE_ID, description="Demo attack rule v1 (chosen on validation)")
    rules = RuleSet()
    rules.add(data)
    rules.approve(DEMO_RULE_ID)
    return rules.active()


def rule_text():
    r = ATTACK_RULE_V1
    return (f"{r['behavior'].capitalize()} for at least {r['min_duration_s']:g} s: ask when the score is above "
            f"{r['ask_threshold']:g}, alert when the confidence reaches {r['alert_threshold']:g}")


def demo_schedule(seconds, now=None, tz="America/New_York"):
    """An approved schedule that starts now and lasts the video plus a margin."""
    now = now or datetime.now(timezone.utc)
    return rs.for_hours((seconds + EXTRA_S) / 3600.0, now=now, tz=tz, schedule_id="DEMO",
                        description="Live demo").approve()


def start_demo(demo, now=None, launcher=None):
    """Start the run for this demo. It gets ready, then waits for the GO signal. Returns the run folder."""
    now = now or datetime.now(timezone.utc)
    run_dir = rc.new_run_dir("DEMO", now)
    config = rc.build_config(run_dir, demo_schedule(length_s(demo), now), demo_rule(),
                             {"kind": "poses_file", "path": str(demo.poses), "session": demo.session},
                             model=MODEL, speed=1.0, fps=FPS)
    config.wait_for_go = True
    return rc.start_run(config, launcher=launcher)


def go_if_ready(run_dir):
    """True (and the signal is sent) when the run is ready and has not been told to start yet."""
    status = rc.read_status(run_dir)
    if status and status.get("state") == "ready" and not (Path(run_dir) / mr.GO_FILE).exists():
        return True
    return False


def latest_alert(events):
    """The most recent alert or question, or None."""
    shown = [e for e in events if e.get("action") in ("alert", "ask")]
    return max(shown, key=lambda e: e["start_s"]) if shown else None


def failed_message(status):
    """Why the run is not going to start, or None."""
    if status and status.get("state") == "failed":
        return status.get("message") or "the run program failed"
    return None
