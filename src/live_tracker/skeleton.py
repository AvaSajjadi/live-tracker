"""Skeleton geometry for the replay page (new file, D3-1). Pure: no drawing
library, no Streamlit; the page (D3-2) turns this geometry into pixels.

Keypoints follow the CalMS21 Task 1 names (calms21.KEYPOINTS). Each keypoint
of one frame gets a status, and only confidently tracked keypoints may be
drawn as solid points or joined by edges:

  tracked         coordinates present, a FINITE score >= min_score, no
                  quality flag, and the frame's tracking_ok is not False
  low_confidence  coordinates present but the tracker score is below min_score
                  (or a low_conf flag)
  unreliable      coordinates present but the confidence is not known (no
                  score, NaN or infinite), or flagged frozen / jump /
                  out_of_frame, or the whole frame has tracking_ok False
  imputed         filled in by quality.interpolate_short_gaps (not measured)
  missing         no coordinates: nothing is drawn, the position is unknown
                  (missing is not standing still)

Draw style per status (STYLE): solid point for tracked, hollow point for
low_confidence / unreliable / imputed, nothing for missing. Each keypoint also
carries the reason for its status. Flags must be causal for a replay (see
replay.causal_pose_flags): this module only applies the flags it is given.

The edge list is PROVISIONAL: the CalMS21 readme does not define one here.
PROVISIONAL_EDGES joins keypoints by anatomical name only, and it is labelled
as such (EDGE_STATUS) until Ava / Igor confirm it. An edge is drawn only when
both ends are tracked; every other edge is listed as omitted with the reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .calms21 import KEYPOINTS
from .quality import SUGGESTED_MIN_SCORE

__all__ = ["EDGE_STATUS", "PROVISIONAL_EDGES", "STATUSES", "STYLE", "Keypoint", "MouseSkeleton",
           "keypoint_status", "mouse_skeleton"]

# Unconfirmed connectivity: from keypoint names, not from the dataset's documentation.
PROVISIONAL_EDGES = (
    ("nose", "left_ear"), ("nose", "right_ear"),
    ("left_ear", "neck"), ("right_ear", "neck"),
    ("neck", "left_hip"), ("neck", "right_hip"),
    ("left_hip", "tail_base"), ("right_hip", "tail_base"),
)
EDGE_STATUS = ("PROVISIONAL: joined by anatomical keypoint name only; not from the CalMS21 readme; "
               "needs confirmation by Ava / Igor")
STATUSES = ("tracked", "low_confidence", "unreliable", "imputed", "missing")
STYLE = {"tracked": "solid", "low_confidence": "hollow", "unreliable": "hollow", "imputed": "hollow",
         "missing": None}
_UNRELIABLE_FLAGS = ("frozen", "jump", "out_of_frame")
assert all(a in KEYPOINTS and b in KEYPOINTS for a, b in PROVISIONAL_EDGES)


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float | None            # None when missing (never a substitute position)
    y: float | None
    score: float | None
    status: str
    style: str | None          # STYLE[status]
    reason: str = ""           # why the keypoint has this status


@dataclass(frozen=True)
class MouseSkeleton:
    mouse: str
    keypoints: tuple           # Keypoint, in calms21.KEYPOINTS order (absent rows count as missing)
    edges: tuple               # ((name, name), ...) drawn: both ends tracked
    omitted_edges: tuple       # ((name, name, reason), ...)
    edge_status: str = EDGE_STATUS

    @property
    def n_tracked(self):
        return sum(k.status == "tracked" for k in self.keypoints)


def _finite(v):
    return v is not None and not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def keypoint_status(x, y, score=None, *, min_score=SUGGESTED_MIN_SCORE, flags=None, frame_tracking_ok=True):
    """The status of one keypoint (module docstring)."""
    return _status(x, y, score, min_score, flags or {}, frame_tracking_ok)[0]


def _status(x, y, score, min_score, flags, frame_tracking_ok):
    """(status, reason). `flags`: optional causal quality flags for this row
    (missing, low_conf, frozen, jump, out_of_frame, imputed). Only a finite
    score at or above min_score can make a keypoint tracked."""
    if not _finite(min_score):
        raise ValueError("min_score must be a finite number")
    if not (_finite(x) and _finite(y)) or flags.get("missing"):
        return "missing", "no coordinates on this frame"
    if flags.get("imputed"):
        return "imputed", "interpolated, not measured"
    bad = [f for f in _UNRELIABLE_FLAGS if flags.get(f)]
    if bad:
        return "unreliable", f"flagged {', '.join(bad)}"
    if frame_tracking_ok is False:
        return "unreliable", "tracking_ok is False on this frame"
    if not _finite(score):
        return "unreliable", "tracker confidence not recorded or not finite"
    if flags.get("low_conf") or score < min_score:
        return "low_confidence", f"tracker confidence {score:.3g} below {min_score:.3g}"
    return "tracked", ""


def mouse_skeleton(mouse, rows, *, min_score=SUGGESTED_MIN_SCORE, frame_tracking_ok=True, edges=PROVISIONAL_EDGES):
    """Geometry of one mouse in one frame. `rows`: {keypoint name: row dict}
    with x, y, optional score and quality flags (absent keypoints are missing)."""
    points = []
    for name in KEYPOINTS:
        r = rows.get(name)
        if r is None:
            points.append(Keypoint(name, None, None, None, "missing", None, "no row for this keypoint"))
            continue
        flags = {f: bool(r[f]) for f in ("missing", "low_conf", "frozen", "jump", "out_of_frame", "imputed")
                 if f in r and r[f] is not None and not (isinstance(r[f], float) and math.isnan(r[f]))}
        status, reason = _status(r.get("x"), r.get("y"), r.get("score"), min_score, flags, frame_tracking_ok)
        x, y = (float(r["x"]), float(r["y"])) if status != "missing" else (None, None)
        score = float(r["score"]) if _finite(r.get("score")) else None
        points.append(Keypoint(name, x, y, score, status, STYLE[status], reason))
    by = {k.name: k for k in points}
    drawn, omitted = [], []
    for a, b in edges:
        bad = [n for n in (a, b) if by[n].status != "tracked"]
        if bad:
            omitted.append((a, b, ", ".join(f"{n} {by[n].status}" for n in bad)))
        else:
            drawn.append((a, b))
    return MouseSkeleton(str(mouse), tuple(points), tuple(drawn), tuple(omitted))
