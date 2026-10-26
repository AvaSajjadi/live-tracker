"""The slot for the mouse pose model: a picture in, mouse keypoints out (new file).

This repository has no pose model. This module is the place where one is plugged in, so that a real camera can be
scored. A pose model is a function: it takes one camera picture and returns a table with the columns mouse ("a" or
"b"), keypoint (the 7 CalMS21 names), x, y (pixels of the CalMS21 1024 x 570 frame) and score (0 to 1), or None when
no mouse is found. The model must also do the work of mapping the camera picture to that frame.

How a model is found: the file models/pose/pose_model.json (or the file named in LIVE_TRACKER_POSE_MODEL) names a
backend and its settings, for example {"backend": "onnx", "path": "models/pose/mouse.onnx"}. A backend is a function
that takes the settings and the project folder and returns the pose function. Backends are added with
register_backend(). No backend ships yet, so today the answer is always "no pose model", and the camera run is refused
with a clear message. The configuration only names a backend; it never runs code from a file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .calms21 import IMAGE_SIZE, KEYPOINTS, MICE

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ENV = "LIVE_TRACKER_POSE_MODEL"
DEFAULT_CONFIG = "models/pose/pose_model.json"
COLUMNS = ("mouse", "keypoint", "x", "y", "score")
MAX_OUTSIDE_SHARE = 0.25                    # more points than this outside the frame: the model's coordinates do not fit

_BACKENDS = {}


class PoseModelError(ValueError):
    """The pose model's answer does not have the layout the scorer needs."""


def register_backend(name, factory):
    """Add a backend: factory(settings, root) -> a function(image) -> keypoint table or None."""
    _BACKENDS[name] = factory


def config_path(root=None, env=None):
    env = os.environ if env is None else env
    named = (env.get(CONFIG_ENV) or "").strip()
    return Path(named) if named else Path(root or ROOT) / DEFAULT_CONFIG


def check_output(found):
    """The keypoint table, checked and cleaned, or None when no mouse was found. Raises PoseModelError if it does not fit."""
    if found is None or len(found) == 0:
        return None
    df = pd.DataFrame(found)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise PoseModelError(f"the pose model answer lacks the columns {missing}")
    df = df[list(COLUMNS)].copy()
    if not set(df["mouse"]) <= set(MICE):
        raise PoseModelError(f"mouse must be one of {list(MICE)}")
    if not set(df["keypoint"]) <= set(KEYPOINTS):
        raise PoseModelError(f"keypoint must be one of {list(KEYPOINTS)}")
    if df.duplicated(["mouse", "keypoint"]).any():
        raise PoseModelError("a keypoint of a mouse is given twice")
    for col in ("x", "y", "score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if ((df["score"] < 0) | (df["score"] > 1)).any():
        raise PoseModelError("score must be between 0 and 1")
    width, height = IMAGE_SIZE
    placed = df.dropna(subset=["x", "y"])
    outside = ~(placed["x"].between(0, width) & placed["y"].between(0, height))
    if len(placed) and outside.mean() > MAX_OUTSIDE_SHARE:
        raise PoseModelError(f"too many points are outside the {width} x {height} frame: the coordinates do not fit")
    return df


def _read_settings(root=None, env=None):
    """(settings, None) when the settings file is there and names a backend this program has, else (None, the reason in plain words)."""
    path = config_path(root, env)
    if not path.is_file():
        return None, (f"There is no pose model. A camera needs one to turn each picture into mouse keypoints. "
                      f"Put its settings in {path.name} (see docs/pose_model.md).")
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
        backend = settings["backend"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, f"The pose model settings in {path.name} cannot be read (they need a \"backend\" name)."
    if _BACKENDS.get(backend) is None:
        return None, f"The pose model backend {backend!r} is not available in this program."
    return settings, None


def check(root=None, env=None):
    """(True, None) when a pose model looks set up, else (False, the reason). FAST: nothing is loaded and no model is prepared, so a page
    can ask this on every click (load() takes seconds: it imports the model library and prepares the Intel GPU)."""
    settings, reason = _read_settings(root, env)
    if settings is None:
        return False, reason
    for key in ("weights", "poses"):                          # the files the backend will need, if the settings name them
        if settings.get(key):
            named = Path(settings[key])
            if not (named if named.is_absolute() else Path(root or ROOT) / named).exists():
                return False, "The pose model could not be loaded (FileNotFoundError)."
    return True, None


def load(root=None, env=None):
    """(pose function, None) when a pose model is set up, else (None, the reason in plain words). SLOW for a trained model: use check() to ask."""
    settings, reason = _read_settings(root, env)
    if settings is None:
        return None, reason
    try:
        function = _BACKENDS[settings["backend"]](settings, Path(root or ROOT))
    except Exception as exc:                                  # a backend can fail in many ways; say which one
        return None, f"The pose model could not be loaded ({type(exc).__name__})."
    if settings.get("stabilize"):                             # optional, off by default: see point_stabilizer.py
        from . import point_stabilizer
        function = point_stabilizer.wrap(function, settings["stabilize"] if isinstance(settings["stabilize"], dict) else {})
    return (lambda image: check_output(function(image))), None


def _register_builtin_backends():
    """The backends that ship with the program (their heavy imports happen only when one is loaded)."""
    from . import pose_backends
    register_backend("yolo", pose_backends.yolo_factory)
    register_backend("stub", pose_backends.stub_factory)


_register_builtin_backends()
