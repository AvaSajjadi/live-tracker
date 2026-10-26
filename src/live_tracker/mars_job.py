"""Run MARS pose estimation on a recording in the background and read how far it is (new file).

MARS (scripts/mars_pose_offline.py) needs TensorFlow, which the project's own Python does not have. It runs in its own
environment in %USERPROFILE%\\mars_local (set MARS_HOME to put it elsewhere): `env\\` (Python with TensorFlow) and `models\\`
(the three .pb files). See docs/camera.md, section 8. The job runs as its own program, so closing the page does not stop it.

A job folder holds: job.json (what was started), mars.log (the program's messages, with lines like "120/450 frames"),
keypoints.parquet (the result) and keypoints.parquet.done.json (written at the end: {"ok": true, ...} or {"ok": false, "error": ...}).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "mars_pose_offline.py"
HOME_ENV = "MARS_HOME"
PB_FILES = ("detection/MARS_top_detection_black.pb", "detection/MARS_top_detection_white.pb", "pose/MARS_top_pose.pb")
FRAMES_PER_SECOND = 1.2                 # measured on the project laptop (CPU), after a 20 s warm-up
WARMUP_S = 45.0
STALE_S = 240.0                         # no new message in the log for this long and no done file: the job stopped
PROGRESS = re.compile(r"(\d+)/(\d+|\?) frames")


def home():
    return Path(os.environ.get(HOME_ENV) or Path.home() / "mars_local")


def python_exe():
    env = home() / "env"
    return env / "Scripts" / "python.exe" if os.name == "nt" else env / "bin" / "python"


def available():
    """(True, "") when MARS can run on this computer, else (False, what is missing)."""
    if not python_exe().is_file():
        return False, f"The MARS environment is missing ({python_exe()}). See docs/camera.md, section 8."
    missing = [p for p in PB_FILES if not (home() / "models" / p).is_file()]
    if missing:
        return False, f"MARS model files are missing in {home() / 'models'}: {', '.join(missing)}. See docs/camera.md, section 8."
    if not SCRIPT.is_file():
        return False, f"{SCRIPT} is missing."
    return True, ""


def estimate_minutes(frames):
    """About how long MARS needs for this many frames on this laptop."""
    return (WARMUP_S + frames / FRAMES_PER_SECOND) / 60.0


def command(video, job_dir, fps=30.0):
    return [str(python_exe()), str(SCRIPT), str(video), "--out", str(Path(job_dir) / "keypoints.parquet"),
            "--models-dir", str(home() / "models"), "--batch", "4", "--session", "camera", "--fps", str(float(fps))]


def default_launcher(cmd, job_dir):
    """Start the job so that it keeps going when this process ends."""
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with open(Path(job_dir) / "mars.log", "ab") as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **kwargs)
    return proc.pid


LAUNCHER = default_launcher             # tests replace this so that no program is started


def start(video, job_dir, *, fps=30.0, launcher=None):
    """Start MARS on `video`. The result goes to job_dir. Returns job_dir."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    if not Path(video).is_file():
        raise FileNotFoundError(f"the recording {video} does not exist")
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(video, job_dir, fps)
    (job_dir / "job.json").write_text(json.dumps({"video": str(video), "started_at": datetime.now(timezone.utc).isoformat(),
                                                  "command": cmd}, indent=1), encoding="utf-8")
    (launcher or LAUNCHER)(cmd, job_dir)
    return job_dir


def _progress(log):
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - 4096))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return 0, None
    found = PROGRESS.findall(text)
    if not found:
        return 0, None
    done, total = found[-1]
    return int(done), (int(total) if total != "?" else None)


def status(job_dir, now=None):
    """{'state': 'none'|'running'|'finished'|'failed', 'done': frames done, 'total': frames or None, 'message': text}"""
    job_dir = Path(job_dir)
    if not (job_dir / "job.json").is_file():
        return {"state": "none", "done": 0, "total": None, "message": "no job here"}
    done_file = job_dir / "keypoints.parquet.done.json"
    log = job_dir / "mars.log"
    done, total = _progress(log)
    if done_file.is_file():
        try:
            info = json.loads(done_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {"ok": False, "error": "the result file cannot be read"}
        if info.get("ok"):
            return {"state": "finished", "done": info.get("frames", done), "total": info.get("frames", total),
                    "message": f"done in {info.get('seconds', 0):.0f} s"}
        return {"state": "failed", "done": done, "total": total, "message": info.get("error", "the job failed")}
    now = time.time() if now is None else now
    last = max((p.stat().st_mtime for p in (log, job_dir / "job.json") if p.is_file()), default=0)
    if now - last > STALE_S:
        return {"state": "failed", "done": done, "total": total, "message": "the job stopped without finishing"}
    return {"state": "running", "done": done, "total": total, "message": "starting" if not done else "analyzing"}


def result(job_dir):
    return Path(job_dir) / "keypoints.parquet"
