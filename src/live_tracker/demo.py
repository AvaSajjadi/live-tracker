"""The demo recording: a CalMS21 video with its pose keypoints, offered on the Upload step (new file).

A demo is a video in data/raw/calms21_video/ (git-ignored) with a file of the same session's pose keypoints next to
it (`<video name>_poses.parquet`, made by scripts/prepare_demo_session.py). Only sessions of the train or validation
split are offered; a test session never is. What happens after it is chosen is in live_session.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import run_control as rc

VIDEO_DIR = "data/raw/calms21_video"
MODEL = "models/model_b"
FPS = 30.0


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
