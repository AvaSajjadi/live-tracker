"""What the researcher gives the system on the Upload step (new file).

Three kinds of input, plus the camera:
  - a video, with the pose keypoints of the same recording (so the video can be shown and analysed together)
  - a keypoints file alone (analysed without a picture)
  - a centers CSV (x_a, y_a, x_b, y_b): motion and tracking quality only, no behavior scores
  - the demo recording
  - a camera: listed, but it needs a mouse pose model that does not exist yet, so it cannot be started

A video alone cannot be analysed: the behavior scores come from keypoints, and turning a picture into
keypoints needs a pose model. Everything uploaded is saved in a new folder under outputs/uploads/ (git-ignored),
checked, and never run as code.

Keypoint files use the CalMS21 layout: mouse "a" / "b", the 7 CalMS21 keypoint names, frame numbers, x and y in
pixels of the CalMS21 image frame (1024 x 570), and an optional confidence "score".
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .calms21 import FPS, IMAGE_SIZE, KEYPOINTS, MICE

ROOT = Path(__file__).resolve().parents[2]
UPLOADS_ENV = "LIVE_TRACKER_UPLOADS_DIR"
KINDS = ("demo", "video_keypoints", "keypoints", "centers", "camera")
SCORING_KINDS = ("demo", "video_keypoints", "keypoints")
VIDEO_TYPES = ("mp4", "webm", "mov", "m4v")
TABLE_TYPES = ("csv", "parquet")
MIN_FRAMES = 30
MAX_FRAMES = 2_000_000                      # about 18 hours at 30 fps
MAX_OUTSIDE_SHARE = 0.05                    # more points than this outside the image frame: wrong coordinates
POSE_COLUMNS = ["session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score", "score_observed"]


class UploadError(ValueError):
    """The uploaded file cannot be used; the message says what to fix."""


@dataclass(frozen=True)
class SessionSource:
    kind: str
    label: str
    video: str | None = None            # path of a video file
    poses: str | None = None            # Parquet in the CalMS21 long layout, one session
    session: str | None = None
    centers: str | None = None          # CSV with x_a, y_a, x_b, y_b
    fps: float = FPS
    device: str | None = None           # a real camera: its number ("0") or stream address; none means camera 0
    screen: tuple | None = None         # a camera that films a screen: the 4 corners of the video (see screen_map.py)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise UploadError(f"unknown source kind {self.kind!r}")
        if self.screen is not None:
            from .screen_map import ScreenError, check_corners
            try:
                object.__setattr__(self, "screen", check_corners(self.screen))
            except ScreenError as exc:
                raise UploadError(f"screen corners: {exc}") from exc
        if self.kind in SCORING_KINDS and not (self.poses and self.session):
            raise UploadError("this source needs a keypoints file")
        if self.kind == "centers" and not self.centers:
            raise UploadError("this source needs a centers file")
        if self.kind in ("video_keypoints", "demo") and not self.video:
            raise UploadError("this source needs a video")

    @property
    def can_score(self):
        return self.kind in SCORING_KINDS

    @property
    def staged_camera(self):
        """A real camera that films a recording (an iPad), while the numbers come from the recording's saved keypoints."""
        return self.kind == "demo" and bool(self.device)

    @property
    def has_video(self):
        return bool(self.video)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ---- saving ----

def uploads_root():
    return Path(os.environ.get(UPLOADS_ENV) or ROOT / "outputs" / "uploads")


def new_upload_dir(root=None, now=None):
    base = Path(root) if root is not None else uploads_root()
    now = now or datetime.now(timezone.utc)
    name = f"{now:%Y%m%dT%H%M%SZ}"
    path, n = base / name, 1
    while path.exists():
        n += 1
        path = base / f"{name}_{n}"
    path.mkdir(parents=True)
    return path


def safe_name(name):
    """A file name that cannot point anywhere else: no folders, only letters, digits, dot, dash, underscore."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", Path(str(name).replace("\\", "/")).name).strip("._")
    if not base:
        raise UploadError("the file has no usable name")
    return base[-80:]


def extension(name):
    return Path(name).suffix.lower().lstrip(".")


def save_bytes(folder, name, data):
    path = Path(folder) / safe_name(name)
    path.write_bytes(data)
    return path


def save_video(folder, name, data):
    if extension(name) not in VIDEO_TYPES:
        raise UploadError(f"a video must be one of: {', '.join(VIDEO_TYPES)}")
    if not data:
        raise UploadError("the video file is empty")
    return save_bytes(folder, name, data)


# ---- keypoints ----

def read_table(name, data):
    kind = extension(name)
    if kind not in TABLE_TYPES:
        raise UploadError(f"a table must be a .csv or .parquet file, not .{kind or '?'}")
    try:
        return pd.read_csv(io.BytesIO(data)) if kind == "csv" else pd.read_parquet(io.BytesIO(data))
    except Exception as exc:                                       # a damaged file can fail in many ways
        raise UploadError(f"the file could not be read as {kind}: {type(exc).__name__}") from exc


def normalize_keypoints(df, name="upload", fps=FPS):
    """The CalMS21 long layout, one frame after another, every keypoint of both mice on every frame."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("mouse", "keypoint", "frame", "x", "y") if c not in df.columns]
    if missing:
        raise UploadError(f"the keypoints file lacks the columns {missing} (it needs mouse, keypoint, frame, x, y)")
    df["mouse"] = df["mouse"].astype(str).str.strip().str.lower()
    df["keypoint"] = df["keypoint"].astype(str).str.strip().str.lower()
    if not set(df["mouse"]) <= set(MICE):
        raise UploadError(f"mouse must be 'a' or 'b', found {sorted(set(df['mouse']) - set(MICE))[:3]}")
    if not set(df["keypoint"]) <= set(KEYPOINTS):
        raise UploadError(f"keypoint names must be among {list(KEYPOINTS)}; found {sorted(set(df['keypoint']) - set(KEYPOINTS))[:3]}")
    for col in ("frame", "x", "y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["frame"].isna().any() or (df["frame"] < 0).any() or (df["frame"] % 1 != 0).any():
        raise UploadError("frame must be whole numbers, 0 or more")
    df["frame"] = df["frame"].astype(int)
    if df.duplicated(["mouse", "keypoint", "frame"]).any():
        raise UploadError("some keypoint appears twice on the same frame")
    if df["frame"].nunique() < MIN_FRAMES:
        raise UploadError(f"the file has only {df['frame'].nunique()} frames; at least {MIN_FRAMES} are needed")
    if df["frame"].max() >= MAX_FRAMES:
        raise UploadError("the frame numbers are too large")

    session = f"UPLOAD_{Path(name).stem[:40]}"
    if "session" in df.columns:
        names = df["session"].dropna().astype(str).unique()
        if len(names) > 1:
            raise UploadError("the file holds several sessions; upload one session at a time")
        if len(names) == 1:
            session = names[0]
    if re.search(r"(^|/)test(/|$)", session):
        raise UploadError("test-split sessions are never used by the agent")

    if "score" in df.columns:
        score = pd.to_numeric(df["score"], errors="coerce")
        if (score.dropna() < 0).any() or (score.dropna() > 1).any():
            raise UploadError("score must be between 0 and 1")
    else:
        score = pd.Series(np.nan, index=df.index)
    df["score"] = score
    observed = score.notna()
    if "score_observed" in df:
        observed &= df["score_observed"].eq(True)
    df["score_observed"] = observed

    inside = df["x"].between(0, IMAGE_SIZE[0]) & df["y"].between(0, IMAGE_SIZE[1])
    present = df["x"].notna() & df["y"].notna()
    if present.sum() == 0:
        raise UploadError("no keypoint has a position")
    if (present & ~inside).sum() / present.sum() > MAX_OUTSIDE_SHARE:
        raise UploadError(f"many positions are outside the CalMS21 image frame ({IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} pixels); "
                          "the model was trained in that frame, so the keypoints must be converted to it first")

    # every keypoint of both mice on every frame: a missing one is NaN with score 0, never absent
    frames = np.sort(df["frame"].unique())
    grid = pd.MultiIndex.from_product([frames, MICE, KEYPOINTS], names=["frame", "mouse", "keypoint"]).to_frame(index=False)
    # Display-only supplied quality; no new quality computation or scorer input.
    display_flags = [c for c in ("missing", "low_conf", "frozen", "jump", "out_of_frame", "imputed", "tracking_ok") if c in df]
    full = grid.merge(df[["frame", "mouse", "keypoint", "x", "y", "score", "score_observed", *display_flags]], on=["frame", "mouse", "keypoint"], how="left")
    full["score_observed"] = full["score_observed"].eq(True)
    has_xy = full["x"].notna() & full["y"].notna()
    full["score"] = full["score"].where(full["score"].notna(), np.where(has_xy, 1.0, 0.0))
    full.loc[~has_xy, "score"] = 0.0
    full["session"] = session
    full["time_s"] = full["frame"] / float(fps)
    return full[POSE_COLUMNS + display_flags].sort_values(["frame", "mouse", "keypoint"]).reset_index(drop=True)


def save_keypoints(folder, name, data, fps=FPS):
    """Check a keypoints file and save it as Parquet. Returns (path, session)."""
    poses = normalize_keypoints(read_table(name, data), name, fps)
    path = Path(folder) / f"{Path(safe_name(name)).stem}_poses.parquet"
    poses.to_parquet(path, index=False)
    return path, str(poses["session"].iloc[0])


# ---- centers ----

CENTER_COLUMNS = ("x_a", "y_a", "x_b", "y_b")


def check_centers(name, data):
    df = read_table(name, data)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in CENTER_COLUMNS if c not in df.columns]
    if missing:
        raise UploadError(f"the file lacks the columns {missing} (it needs x_a, y_a, x_b, y_b)")
    df = df[list(CENTER_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if len(df) < 10:
        raise UploadError("the file has fewer than 10 rows")
    if df.notna().sum().sum() == 0:
        raise UploadError("the file has no numbers in those columns")
    return df


def save_centers(folder, name, data):
    df = check_centers(name, data)
    path = Path(folder) / f"{Path(safe_name(name)).stem}_centers.csv"
    df.to_csv(path, index=False)
    return path


# ---- from uploaded files to a source ----

def build_keypoints_source(keypoints, video=None, *, fps=FPS, root=None):
    """keypoints and video are (file name, bytes). A video alone cannot be analysed, so keypoints are required.

    Saves everything in a new folder. If any file is bad, the folder is removed and UploadError is raised."""
    import shutil
    if keypoints is None:
        raise UploadError("a video alone cannot be analysed: the behavior scores come from keypoints. "
                          "Upload the keypoints file of the same recording too.")
    folder = new_upload_dir(root)
    try:
        poses, session = save_keypoints(folder, keypoints[0], keypoints[1], fps)
        video_path = save_video(folder, video[0], video[1]) if video else None
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    kind = "video_keypoints" if video_path else "keypoints"
    label = (safe_name(video[0]) if video_path else safe_name(keypoints[0]))
    return SessionSource(kind, label, video=str(video_path) if video_path else None, poses=str(poses), session=session,
                         fps=float(fps))


def build_centers_source(centers, *, fps=FPS, root=None):
    """centers is (file name, bytes)."""
    import shutil
    folder = new_upload_dir(root)
    try:
        path = save_centers(folder, centers[0], centers[1])
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return SessionSource("centers", safe_name(centers[0]), centers=str(path), fps=float(fps))
