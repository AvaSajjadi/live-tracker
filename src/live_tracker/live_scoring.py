"""Live scoring: pose keypoints, one frame at a time -> behavior scores (new file).

The monitor wants one score row per frame (INTERFACES.md section 2). The classifier was trained on
features computed over whole sessions. This module computes the same features from a short moving
buffer of the most recent frames, so a live stream can be scored without the future.

Why a short buffer is enough: every feature uses only a trailing window (1 s by default) and a
filter that forgets quickly. Checked on one real CalMS21 validation session (7150 frames, model B):
a 6 s buffer scored in chunks of 5 frames gave the same scores as the whole-session run (largest
difference 5e-13) and took about 95 ms per chunk against a 167 ms budget. That check is repeated
by a test when the data is present locally. It is one session, not a guarantee for every setup.

Scores are produced in chunks, so a score row appears up to `chunk` frames (0.17 s at 30 fps,
chunk 5) after its frame. This is only a delay: each row still uses frames up to its own frame.

Limits:
  - Pose rows must use the CalMS21 layout: mouse "a" / "b", the 7 CalMS21 keypoint names, pixels in
    the CalMS21 image frame (1024 x 570) and a confidence "score". A camera has to be mapped to this
    frame, with the same kind of top-down view, or the scores are meaningless.
  - A frame that was never pushed (dropped) is simply absent from the output; the monitor treats
    skipped frame numbers as a gap that ends any bout.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from .classify import feature_columns, score_frames, session_features

POSE_COLUMNS = ("session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score")
SCORE_ROW_KEYS = ("session", "frame", "time_s", "attack", "mount", "investigation", "tracking_ok")


class LiveScoringError(ValueError):
    """The pose rows or the model do not fit what the scorer needs."""


class LiveScorer:
    def __init__(self, model, expected_columns, *, session, features="basic,motion", fps=30.0,
                 buffer_s=6.0, chunk=5):
        names = [n.strip() for n in features.split(",") if n.strip()]
        if not names or not set(names) <= {"basic", "motion"} or "basic" not in names:
            raise LiveScoringError("features must be 'basic' or 'basic,motion'")
        if chunk < 1 or buffer_s * fps < chunk:
            raise LiveScoringError("the buffer must hold at least one chunk of frames")
        self.model, self.expected_columns = model, list(expected_columns)
        self.session, self.fps, self.chunk = session, float(fps), int(chunk)
        self.use_motion = "motion" in names
        self.buffer_frames = int(round(buffer_s * fps))
        self._buffer = deque()            # (frame, rows of that frame)
        self._pending = []                # frames not scored yet, in order
        self._last_frame = None

    def push(self, frame, pose_rows):
        """Add one frame of pose rows. Returns the score rows (dicts) that are ready, often none."""
        if self._last_frame is not None and frame <= self._last_frame:
            raise LiveScoringError(f"frame {frame} is not after frame {self._last_frame}")
        rows = pd.DataFrame(pose_rows)
        missing = [c for c in POSE_COLUMNS if c not in rows.columns]
        if missing:
            raise LiveScoringError(f"pose rows lack columns {missing}")
        if not (rows["session"] == self.session).all() or not (rows["frame"] == frame).all():
            raise LiveScoringError("pose rows must belong to this session and this frame")
        self._last_frame = frame
        self._buffer.append((frame, rows[list(POSE_COLUMNS)]))
        while self._buffer[0][0] < frame - self.buffer_frames + 1:
            self._buffer.popleft()
        self._pending.append(frame)
        return self._score() if len(self._pending) >= self.chunk else []

    def flush(self):
        """Score the frames still waiting (end of a run)."""
        return self._score() if self._pending else []

    def _score(self):
        poses = pd.concat([rows for _, rows in self._buffer], ignore_index=True)
        basic, motion = session_features(poses)
        # tracking_ok always comes from the motion table (model A has no motion features, but its rows still need it)
        extra = ([c for c in motion.columns if c not in basic.columns or c in ("session", "frame")]
                 if self.use_motion else ["session", "frame", "tracking_ok"])
        feats = basic.merge(motion[extra], on=["session", "frame"], how="inner")
        if feature_columns(feats) != self.expected_columns:
            raise LiveScoringError("the computed feature columns differ from the ones the model was trained on")
        waiting = feats[feats["frame"].isin(self._pending)]
        self._pending = []
        if waiting.empty:
            return []
        scored = score_frames(self.model, waiting, [self.session])
        return [{"session": r["session"], "frame": int(r["frame"]), "time_s": float(r["time_s"]),
                 "attack": float(r["attack"]), "mount": float(r["mount"]),
                 "investigation": float(r["investigation"]), "tracking_ok": bool(r["tracking_ok"])}
                for r in scored.to_dict("records")]
