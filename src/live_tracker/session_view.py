"""One real CalMS21 VALIDATION session as the motion engine sees it (new file).

Pure functions behind pages/4_Real_Session.py: nothing here draws anything.

Where the data comes from: only the hash-verified validation cache that
investigation_data builds (python -m live_tracker.investigation_data --all).
That module refuses any session outside the manifest's validation split, any
file whose hash differs from the index, and any symlink, so a test session can
never reach this page.

What is computed is exactly the classifier's own preprocessing
(classify.session_features): calms21.flag_quality (team quality settings), then
features.compute_features with the torso keypoints. The keypoint features are
the opt-in features.keypoint_features. Everything is a MEASUREMENT of motion or
tracking; none of it is a behavior label or a model score.

The annotation bouts are CalMS21's own ground-truth labels, returned separately
so a page can show them as context. They are never an input to any feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calms21 import FPS, TORSO_KEYPOINTS, flag_quality, load_calms21
from .features import compute_features, keypoint_features, tracking_warning_events
from .investigation_data import CacheError, ExtractionRefused, verified_sessions

__all__ = ["BUILD_COMMAND", "DEFAULT_CACHE", "DEFAULT_MANIFEST", "SessionView", "SessionUnavailable",
           "build_view", "label_bouts", "list_sessions", "load_view", "short_name", "tracking_runs"]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "raw" / "calms21_validation"
DEFAULT_MANIFEST = ROOT / "splits" / "split_manifest.json"
BUILD_COMMAND = ("python -m live_tracker.investigation_data --source data/raw/calms21/calms21_task1_train.json "
                 "--all")


class SessionUnavailable(RuntimeError):
    """The verified validation cache is missing or does not verify; the reason is the message."""


@dataclass(frozen=True)
class SessionView:
    session: str
    n_frames: int
    duration_s: float
    table: pd.DataFrame          # compute_features(): INTERFACES.md §5 (session, frame, time_s, tracking_ok, ...)
    keypoints: pd.DataFrame      # keypoint_features(): nose/tail distances and facing angles
    warnings: list              # tracking_warning_events(): INTERFACES.md §4 dicts, torso keypoints only
    labels: pd.DataFrame         # session, frame, time_s, label  (ground truth, context only)
    flag_share: dict             # share of torso keypoint-frames with each quality flag

    @property
    def tracking_ok_share(self):
        return float(self.table["tracking_ok"].mean())


def short_name(session_id):
    """task1/train/mouse005_task1_annotator1 -> mouse005"""
    return session_id.rsplit("/", 1)[-1].split("_", 1)[0]


def _load_verified(cache_dir, manifest_path):
    try:
        sessions, _ = verified_sessions(cache_dir, manifest_path, "validation")
    except (CacheError, ExtractionRefused, OSError, ValueError) as exc:
        raise SessionUnavailable(f"{type(exc).__name__}: {exc}") from None
    return sessions


def _paths(cache_dir, manifest_path):
    """Defaults are resolved at call time, so a test can point them at a fixture."""
    return Path(cache_dir or DEFAULT_CACHE), Path(manifest_path or DEFAULT_MANIFEST)


def list_sessions(cache_dir=None, manifest_path=None):
    """Session ids in the verified validation cache (sorted). SessionUnavailable if it is missing or
    fails verification."""
    cache_dir, manifest_path = _paths(cache_dir, manifest_path)
    if not cache_dir.is_dir():
        raise SessionUnavailable(f"no validation cache at {cache_dir}. Build it once with: {BUILD_COMMAND}")
    return [sid for sid, _ in _load_verified(cache_dir, manifest_path)]


def load_view(session, cache_dir=None, manifest_path=None):
    """The SessionView of one session of the verified validation cache."""
    cache_dir, manifest_path = _paths(cache_dir, manifest_path)
    if not cache_dir.is_dir():
        raise SessionUnavailable(f"no validation cache at {cache_dir}. Build it once with: {BUILD_COMMAND}")
    for sid, data in _load_verified(cache_dir, manifest_path):
        if sid == session:
            return build_view(sid, data)
    raise SessionUnavailable(f"{session!r} is not in the verified validation cache")


def build_view(session, data):
    """SessionView from one verified record in the CalMS21 layout {group: {session: record}}."""
    poses, labels = load_calms21(data)
    flags = flag_quality(poses)                       # the classifier's own quality step
    table = compute_features(flags, torso_keypoints=TORSO_KEYPOINTS)
    keypoints = keypoint_features(flags, torso_keypoints=TORSO_KEYPOINTS)
    torso = flags[flags["keypoint"].isin(TORSO_KEYPOINTS)]
    flag_share = {name: float(torso[name].mean()) for name in ("missing", "out_of_frame", "jump", "frozen",
                                                               "low_conf") if name in torso.columns}
    n = int(table["frame"].max()) + 1
    return SessionView(session=session, n_frames=n, duration_s=n / FPS, table=table, keypoints=keypoints,
                       warnings=tracking_warning_events(flags, keypoints=TORSO_KEYPOINTS), labels=labels,
                       flag_share=flag_share)


def label_bouts(labels, skip=("other",)):
    """Runs of the same annotated label: session, label, start_s, end_s, duration_s (end is exclusive).
    Ground truth for context; `skip` labels (default the background class) are left out."""
    lab = labels.sort_values("frame").reset_index(drop=True)
    if lab.empty:
        return pd.DataFrame(columns=["session", "label", "start_s", "end_s", "duration_s"])
    change = lab["label"].ne(lab["label"].shift())
    run = change.cumsum()
    rows = []
    for _, part in lab.groupby(run, sort=True):
        name = part["label"].iloc[0]
        if name in skip:
            continue
        start, stop = int(part["frame"].iloc[0]), int(part["frame"].iloc[-1]) + 1
        rows.append({"session": part["session"].iloc[0], "label": name, "start_s": start / FPS,
                     "end_s": stop / FPS, "duration_s": (stop - start) / FPS})
    return pd.DataFrame(rows, columns=["session", "label", "start_s", "end_s", "duration_s"])


def tracking_runs(table):
    """Runs where tracking_ok is False: start_s, end_s (exclusive), duration_s."""
    ok = table.sort_values("frame")["tracking_ok"].to_numpy(bool)
    t = table.sort_values("frame")["time_s"].to_numpy(float)
    bad = ~ok
    starts = np.flatnonzero(bad & ~np.r_[False, bad[:-1]])
    ends = np.flatnonzero(bad & ~np.r_[bad[1:], False]) + 1
    step = 1.0 / FPS
    return pd.DataFrame({"start_s": t[starts], "end_s": t[ends - 1] + step,
                         "duration_s": (ends - starts) * step})
