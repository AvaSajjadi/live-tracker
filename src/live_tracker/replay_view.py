"""What the Replay page shows, without Streamlit (new file, D3-2).

pages/2_Replay.py only renders; everything it decides is here and tested
directly: the synthetic session catalog, input checks, playback ticks,
labels for event times and answer states, the timeline rows and the
skeleton image. It reads replay.FrameState / ReplaySession (D3-1) only, so
nothing shown can come from a later frame than the replay position.

SYNTHETIC data only for now: every catalog session starts with SYNTHETIC_
(fake_scores) and its pose is generated here. Real sessions wait for H3 and
the held-out-session policy; until then a non-synthetic session is refused.
"""

from __future__ import annotations

import bisect
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .calms21 import IMAGE_SIZE, KEYPOINTS, MICE, TORSO_KEYPOINTS
from .fake_scores import (DEMO_SESSION, SCORE_COLS, SYNTHETIC_PREFIX, Segment, TrackingLoss, build_scores,
                          demo_scores)
from .quality import flag_tracking
from .monitor import DEFAULT_FPS
from .quality import SUGGESTED_MIN_SCORE
from .replay import FROZEN_MIN_FRAMES, PLAYBACK_SPEEDS, WARNING_FLAGS, ReplayError, tracking_warnings
from .skeleton import EDGE_STATUS

ROOT = Path(__file__).resolve().parents[2]
REPLAY_DIR_ENV = "LIVE_TRACKER_REPLAY_DIR"
POSE_COLUMNS = ("session", "frame", "time_s", "mouse", "keypoint", "x", "y", "score")
WARNING_MIN_SCORE = 0.15         # quality.SUGGESTED_MIN_SCORE, for the synthetic low-confidence warnings


def now():
    """Wall clock for playback (monotonic; replaced in tests)."""
    return time.monotonic()


def replay_dir():
    """Where journals and answer positions are kept (git-ignored outputs/ by default)."""
    return Path(os.environ.get(REPLAY_DIR_ENV) or ROOT / "outputs" / "replay")


# ---- synthetic catalog ----

@dataclass(frozen=True)
class CatalogEntry:
    label: str
    session: str
    description: str
    scores: pd.DataFrame
    pose: pd.DataFrame | None


def synthetic_pose(session, n, *, missing=(), low_conf=(), frozen=(), seed=0, fps=30.0):
    """A SYNTHETIC two-mouse pose table (calms21 keypoint names, image pixels).
    missing / low_conf / frozen: (mouse, start_s, end_s) stretches on the torso keypoints."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    parts = []
    for mi, mouse in enumerate(MICE):
        centre = np.c_[350 + 250 * mi + 120 * np.sin(t / 3 + mi), 285 + 90 * np.cos(t / 4 + 2 * mi)]
        heading = t / 3 + mi
        for kp in KEYPOINTS:
            # a mouse-sized body (about 140 px nose to tail at 1024 px), so the skeleton is readable
            off = {"nose": 75, "left_ear": 50, "right_ear": 50, "neck": 35, "left_hip": -25, "right_hip": -25,
                   "tail_base": -65}[kp]
            side = {"left_ear": 14, "right_ear": -14, "left_hip": 20, "right_hip": -20}.get(kp, 0)
            x = centre[:, 0] + off * np.cos(heading) - side * np.sin(heading) + rng.normal(0, 0.4, n)
            y = centre[:, 1] + off * np.sin(heading) + side * np.cos(heading) + rng.normal(0, 0.4, n)
            score = np.full(n, 0.9)
            if kp in TORSO_KEYPOINTS:
                for m, a, b in missing:
                    sel = (m == mouse) & (t >= a - 1e-9) & (t < b - 1e-9)
                    x, y = np.where(sel, np.nan, x), np.where(sel, np.nan, y)
                for m, a, b in low_conf:
                    score = np.where((m == mouse) & (t >= a - 1e-9) & (t < b - 1e-9), 0.05, score)
                for m, a, b in frozen:
                    sel = np.flatnonzero((m == mouse) & (t >= a - 1e-9) & (t < b - 1e-9))
                    if len(sel):
                        x, y = x.copy(), y.copy()
                        x[sel], y[sel] = x[sel[0]], y[sel[0]]
            parts.append(pd.DataFrame({"session": session, "frame": np.arange(n), "time_s": t, "mouse": mouse,
                                       "keypoint": kp, "x": x, "y": y, "score": score}))
    return pd.concat(parts, ignore_index=True)


TRACKING_SESSION = "SYNTHETIC_tracking_02"


def synthetic_catalog():
    """The labelled SYNTHETIC sessions the page offers (deterministic)."""
    demo = demo_scores()
    track = build_scores(TRACKING_SESSION, 20.0, (
        Segment("attack", 1.0, 4.0, 0.88, 0.95, "alert, interrupted by lost tracking at 3.5 s"),
        Segment("attack", 6.0, 9.0, 0.6, 0.75, "uncertain attack -> ask"),
        Segment("mount", 11.0, 14.0, 0.85, 0.95, "mount (for a mount rule)"),
        Segment("attack", 17.0, 20.0, 0.9, 0.95, "attack still going when the session ends")),
        (TrackingLoss(3.5, 4.5, "tracking lost"),), seed=7)
    return (
        CatalogEntry("Demo session (alert, ask, short bout, lost tracking)", DEMO_SESSION,
                     "fake_scores demo: an alert at 4.0 s, an ask at 9.0 s, a 1.5 s bout below a 2 s rule, "
                     "and lost tracking on 16-17 s", demo,
                     synthetic_pose(DEMO_SESSION, len(demo), missing=[("a", 16.0, 17.0)], seed=1)),
        CatalogEntry("Tracking problems (missing, low confidence, stuck tracker)", TRACKING_SESSION,
                     "an alert ended by lost tracking, an ask, a mount bout, a bout running to the end; pose with "
                     "missing torso points (3.5-4.5 s), low confidence (8-9.5 s) and a stuck tracker (12-13.5 s)",
                     track, synthetic_pose(TRACKING_SESSION, len(track), missing=[("a", 3.5, 4.5)],
                                           low_conf=[("b", 8.0, 9.5)], frozen=[("a", 12.0, 13.5)], seed=2)),
    )


def check_entry(entry):
    """Refuse a catalog entry the page cannot show honestly (ReplayError says why)."""
    if not isinstance(entry, CatalogEntry):
        raise ReplayError("the session catalog holds an invalid entry")
    if not str(entry.session).startswith(SYNTHETIC_PREFIX):
        raise ReplayError(f"{entry.session!r} is not a SYNTHETIC session: real sessions are not available in the "
                          "replay yet (H3 and the held-out-session policy)")
    s = entry.scores
    if not isinstance(s, pd.DataFrame) or s.empty:
        raise ReplayError(f"{entry.session}: no synthetic score rows")
    missing = [c for c in ("session", "frame", "time_s", "tracking_ok") if c not in s.columns]
    if missing:
        raise ReplayError(f"{entry.session}: score data lacks columns {missing}")
    if set(s["session"].unique()) != {entry.session}:
        raise ReplayError(f"{entry.session}: score rows belong to another session")


def check_pose(entry):
    """None when the pose is usable for skeletons, else the reason (shown as an error)."""
    p = entry.pose
    if p is None:
        return "no pose for this session"
    if not isinstance(p, pd.DataFrame):
        return "the pose is not a table"
    missing = [c for c in POSE_COLUMNS if c not in p.columns]
    if missing:
        return f"the pose lacks columns {missing}"
    if set(p["session"].unique()) != {entry.session}:
        return "the pose belongs to another session"
    if not set(p["keypoint"].unique()) <= set(KEYPOINTS):
        return f"unknown keypoint names {sorted(set(p['keypoint'].unique()) - set(KEYPOINTS))[:5]}"
    if not set(p["mouse"].astype(str).unique()) <= set(MICE):
        return "unknown mouse IDs"
    for c in ("x", "y", "score", "time_s"):
        if not pd.api.types.is_numeric_dtype(p[c]) or pd.api.types.is_bool_dtype(p[c]):
            return f"pose column {c!r} is not numeric"
    f = p["frame"]
    if not pd.api.types.is_numeric_dtype(f) or pd.api.types.is_bool_dtype(f):
        return "pose frame numbers are not numeric"
    fv = f.to_numpy(dtype=float)
    if not np.isfinite(fv).all():
        return "pose frame numbers are missing or not finite"
    if (fv != np.round(fv)).any() or (fv < 0).any():
        return "pose frame numbers must be whole numbers >= 0"
    if not np.isfinite(p["time_s"].to_numpy(dtype=float)).all():
        return "pose times are missing or not finite"
    if p.duplicated(["frame", "mouse", "keypoint"]).any():
        return "duplicate pose rows for a frame, mouse and keypoint"
    times = dict(zip(entry.scores["frame"].astype(int), entry.scores["time_s"].astype(float)))
    frames = fv.astype(np.int64)
    outside = sorted(set(frames.tolist()) - set(times))
    if outside:
        return f"pose frames not in the score rows (e.g. {outside[:3]})"
    expected = np.array([times[x] for x in frames])
    if not np.allclose(p["time_s"].to_numpy(dtype=float), expected, rtol=0, atol=1e-6):
        return "pose times do not match the score rows' times for the same frames"
    return None


def content_digest(df):
    """SHA-256 of a table's column names, dtypes and values (row order included)."""
    if df is None:
        return None
    h = hashlib.sha256(repr([(str(c), str(t)) for c, t in df.dtypes.items()]).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


# replay configuration the page uses (part of the cache identity: a change rebuilds)
PAGE_CONFIG = {"warning_min_score": WARNING_MIN_SCORE, "warning_flags": WARNING_FLAGS,
               "frozen_min_frames": FROZEN_MIN_FRAMES, "keypoint_min_score": SUGGESTED_MIN_SCORE, "fps": DEFAULT_FPS}


def cache_identity(entry, rules, directory, pose_ok):
    """What a cached replay must match to be reused: the score-bound run_id
    (session, rules, fps and the exact score rows), the pose content (which
    also determines the tracking warnings), the configuration, and storage."""
    from .replay import run_paths
    run_id, _, _ = run_paths(entry.scores, entry.session, rules, directory)
    return (run_id, content_digest(entry.pose) if pose_ok else None, repr(sorted(PAGE_CONFIG.items())),
            str(directory))


def entry_warnings(entry):
    """Typed tracking warnings of the entry's pose (Ava's warning function via replay.tracking_warnings)."""
    flags = flag_tracking(entry.pose, score_col="score", min_score=WARNING_MIN_SCORE)
    return tracking_warnings(flags, entry.session)


# ---- playback ----

@dataclass(frozen=True)
class PlaybackAnchor:
    """Where playback was (re)started: the wall clock, the exact recorded time
    (not rounded to a frame) and the speed. Positions are always computed from
    the anchor, so elapsed fractions of a frame are never lost between ticks."""
    wall: float
    recorded_s: float
    speed: float


def start_playback(session, index, now_wall, speed):
    """The anchor for playing from row `index` (before the first row: from the first row)."""
    if speed not in PLAYBACK_SPEEDS:
        raise ReplayError(f"speed must be one of {PLAYBACK_SPEEDS}")
    return PlaybackAnchor(float(now_wall), session.times[min(max(int(index), 0), session.n_frames - 1)], speed)


def playback_position(session, anchor, now_wall, speed):
    """(row index, new anchor, still playing) at wall time now_wall. The new
    anchor is the exact recorded time reached (a float, never rounded to a
    frame), so no fraction is lost between ticks, a speed change applies from
    now on, and a clock that goes back never rewinds. Recorded time is
    time_s, so skipped frames are followed as recorded."""
    if speed not in PLAYBACK_SPEEDS:
        raise ReplayError(f"speed must be one of {PLAYBACK_SPEEDS}")
    recorded = anchor.recorded_s + max(0.0, now_wall - anchor.wall) * anchor.speed
    anchor = PlaybackAnchor(max(float(now_wall), anchor.wall), recorded, speed)
    index = max(0, bisect.bisect_right(session.times, recorded + 1e-9) - 1)
    return index, anchor, index < session.n_frames - 1


# ---- labels ----

def fmt_s(t):
    return "—" if t is None else f"{t:.2f} s"


def event_times(event):
    """Onset, firing and closure of a §4 event as distinct labels."""
    ev = event["evidence"]
    onset = f"{fmt_s(event['start_s'])} (frame {ev['first_frame']})"
    fired = f"{fmt_s(ev['fired_s'])} (frame {ev['fired_frame']})"
    if ev["bout_closed"]:
        closure = f"{fmt_s(event['end_s'])} ({ev['end_reason']}, last frame {ev['last_frame']})"
    else:
        closure = "open (not ended yet)"
    return {"onset": onset, "fired": fired, "closure": closure}


def answer_state(session, state, event):
    """(code, text) of an ask event's answer at this replay position:
    unanswered / answered / answered_later / timing_unknown; non-ask events: (None, "")."""
    if event["action"] != "ask":
        return None, ""
    if event["researcher_answer"] is not None:
        st = session.answer_status(event["event_id"])
        where = f" at frame {st['answered_frame']}" if st["answered_frame"] is not None else \
            " (replay time unknown)"
        return "answered", f"Answered: {event['researcher_answer']}{where}"
    status = session.answer_status(event["event_id"])
    if status["researcher_answer"] is None:
        return "unanswered", "Unanswered: confirm or reject"
    if status["answered_frame"] is None:
        return "timing_unknown", ("An answer is in the journal, but when it was given is not known; it is shown "
                                  "in the end-of-session view. Answers are final.")
    return "answered_later", ("An answer was given later in this replay; it is not shown before that point. "
                              "Answers are final.")


def timeline_rows(state):
    """Deterministic event timeline rows (firing order), from the visible events only."""
    rows = []
    for e in state.events:
        times = event_times(e)
        rows.append({"event": e["event_id"].rsplit("-", 1)[-1], "event_id": e["event_id"], "rule": e["rule_id"],
                     "behavior": e["behavior"], "action": e["action"], "onset": times["onset"],
                     "fired": times["fired"], "closure": times["closure"], "confidence": round(e["confidence"], 3),
                     "answer": e["researcher_answer"] or ""})
    return rows


def warning_rows(state):
    rows = []
    for w in state.warnings:
        ev = w["evidence"]
        if w["end_s"] is not None:
            end = fmt_s(w["end_s"]) + (" (still going when the session ended)" if ev["ongoing_at_session_end"]
                                       else "")
        else:
            end = "ongoing"
        rows.append({"warning": w["event_id"], "kind": ev["kind"], "mouse": ev["mouse"],
                     "coordinates": "present (unreliable)" if ev["coordinates_present"] else "missing",
                     "start": fmt_s(w["start_s"]), "end": end, "message": ev["message"]})
    return rows


def tracking_summary(state):
    """Keypoint status counts per mouse that are not 'tracked' (for the unreliable-tracking warning)."""
    out = {}
    for sk in state.skeletons:
        counts = {}
        for k in sk.keypoints:
            if k.status != "tracked":
                counts[k.status] = counts.get(k.status, 0) + 1
        if counts:
            out[sk.mouse] = counts
    return out


# ---- skeleton image ----

MOUSE_COLOURS = {"a": (40, 40, 40), "b": (200, 120, 20)}        # resident (black), intruder (white -> orange)
STATUS_OUTLINE = {"low_confidence": (230, 160, 0), "unreliable": (210, 40, 40), "imputed": (120, 120, 220)}
LEGEND = ("filled dot: tracked; hollow amber: low confidence; hollow red: unreliable (stuck, flagged, unknown "
          "confidence or lost frame); hollow blue: interpolated; nothing: missing. Lines join two tracked "
          "keypoints only. " + EDGE_STATUS)


def render_skeletons(state, width=512):
    """A small RGB image of the frame's skeletons (None without pose)."""
    if not state.skeletons:
        return None
    w0, h0 = IMAGE_SIZE
    scale = width / w0
    img = Image.new("RGB", (width, int(h0 * scale)), (250, 250, 250))
    d = ImageDraw.Draw(img)
    r = max(2, int(3 * scale * 2))
    for sk in state.skeletons:
        colour = MOUSE_COLOURS.get(sk.mouse, (0, 0, 0))
        pts = {k.name: (k.x * scale, k.y * scale) for k in sk.keypoints if k.x is not None}
        for a, b in sk.edges:
            d.line([pts[a], pts[b]], fill=colour, width=2)
        for k in sk.keypoints:
            if k.x is None:
                continue                                    # missing: no position is invented
            x, y = pts[k.name]
            box = [x - r, y - r, x + r, y + r]
            if k.status == "tracked":
                d.ellipse(box, fill=colour, outline=colour)
            else:
                d.ellipse(box, outline=STATUS_OUTLINE.get(k.status, (210, 40, 40)), width=2)
    if state.tracking_ok is False:
        d.rectangle([0, 0, img.width - 1, img.height - 1], outline=(210, 40, 40), width=4)
    return img


__all__ = ["CatalogEntry", "LEGEND", "SCORE_COLS", "check_entry", "check_pose", "entry_warnings", "event_times",
           "answer_state", "cache_identity", "content_digest", "PlaybackAnchor", "playback_position", "render_skeletons", "start_playback", "replay_dir", "synthetic_catalog",
           "synthetic_pose", "timeline_rows", "tracking_summary", "warning_rows"]
