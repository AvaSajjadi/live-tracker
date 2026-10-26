"""Seconds <-> frames at 30 fps for half-open windows (Contract v1, shared
surface item 2). New file.

STATUS: Ava's final rule (round half up at both window boundaries).
Written up in docs/timebase_30fps.md.

Frames are the canonical positions inside; seconds appear only at the tool
boundary. Frame f (0-based within its session) has the timestamp f / 30, the
same value as time_s in INTERFACES.md §2 and in Igor's score export.

The rule
--------
A time t maps to a frame by rounding to the nearest frame, halves up:

    frame(t) = floor(t * 30 + 0.5)

and a window [t0_s, t1_s) covers the frames [frame(t0_s), frame(t1_s)).

The expression is evaluated literally in IEEE float64 (math.floor(t * 30 +
0.5)), the same arithmetic as numpy or plain Python on the other side, so
both sides get identical frames for identical inputs. No tolerance and no
exact-rational step is added.

Equivalently, frame f is in [t0_s, t1_s) exactly when the half-way point
between frames f and f + 1 lies in (t0_s, t1_s], i.e.
30 * t0_s < f + 0.5 <= 30 * t1_s (in the float evaluation above).

Consequences (tests/test_timebase.py):
- frame(f / 30) == f for every frame f (round trip): floating-point error in a
  timestamp is far below half a frame, so no snap is needed. The monitor's
  start_s -> first_frame, fired_s -> fired_frame and exclusive end_s
  (t0 + k * (1 / 30)) -> last_frame + 1 map exactly, so
  window_frames(start_s, end_s) is exactly a bout's frames;
- an exact half-frame time rounds up: frame(0.05) == 2 (1.5 frames);
- frame is non-decreasing, so adjacent windows [a, b) and [b, c) partition
  [a, c): the shared boundary is one frame for both, nothing is missed or
  counted twice.

Validation (code "bad_window", the contract's error code): a boundary must be
a finite real number (not bool); t0_s >= 0; t1_s > t0_s; the window must
contain at least one frame (frame(t1_s) > frame(t0_s): e.g. [0.0, 0.01) is
refused, since 0.01 s rounds to frame 0); with a session length n_frames,
frame(t1_s) must not be past n_frames (the exclusive end of the last frame).

Windows given by an anchor and durations (compare_windows) are formed in
seconds first and EACH absolute endpoint is converted, e.g. the pre window is
[frame(start_s - pre_s), frame(start_s)). A window is never built by
subtracting a separately rounded duration count: frame(a - d) can differ from
frame(a) - frame(d) (start 12 s, 0.05 s before 10 s: frame(9.95) = 299, but
300 - frame(0.05) = 298). A window whose endpoints convert to the same frame
is "bad_window". duration_frames below converts a lone duration and is not
used to build windows.
"""

from __future__ import annotations

import math
import numbers

from .calms21 import FPS

__all__ = ["FPS_INT", "MAX_FRAME", "WindowError", "boundary_frame", "duration_frames", "frame_time_s",
           "window_frames", "window_s"]

FPS_INT = 30
if FPS != FPS_INT:                       # the rule is written for exactly 30 fps
    raise ImportError(f"timebase assumes calms21.FPS == {FPS_INT}, got {FPS}")
MAX_FRAME = 2 ** 53                      # largest frame index whose timestamp is exact in JSON / float64


class WindowError(ValueError):
    """A window or duration that cannot be converted; code is the contract's error code."""

    def __init__(self, message, code="bad_window"):
        super().__init__(message)
        self.code = code


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise WindowError(f"{name} must be a number of seconds")
    try:
        v = float(value)
    except OverflowError:                # an integer too large for a float
        raise WindowError(f"{name} must be finite") from None
    if not math.isfinite(v):
        raise WindowError(f"{name} must be finite")
    return v


def _round_half_up(t, name):
    """floor(t * 30 + 0.5), evaluated in float64."""
    x = t * FPS_INT + 0.5
    if not math.isfinite(x):             # t is finite, but 30 * t overflows
        raise WindowError(f"{name} {t!r} is too large")
    return math.floor(x)


def frame_time_s(frame):
    """The timestamp of frame `frame`: frame / 30, as time_s."""
    if isinstance(frame, bool) or not isinstance(frame, numbers.Integral) or not 0 <= frame <= MAX_FRAME:
        raise WindowError(f"frame must be an integer in [0, {MAX_FRAME}]")
    return int(frame) / FPS_INT


def boundary_frame(t_s, name="t_s"):
    """frame(t) = floor(t * 30 + 0.5): the nearest frame, halves rounding up."""
    return _round_half_up(_seconds(t_s, name), name)


def window_frames(t0_s, t1_s, n_frames=None):
    """(first, stop): the frames of [t0_s, t1_s) are range(first, stop).
    Raises WindowError("bad_window") for an invalid, empty or out-of-session window."""
    t0, t1 = _seconds(t0_s, "t0_s"), _seconds(t1_s, "t1_s")
    if t0 < 0:
        raise WindowError(f"t0_s {t0!r} is before the session start (0 s)")
    if t1 <= t0:
        raise WindowError(f"t1_s {t1!r} must be after t0_s {t0!r} (half-open [t0_s, t1_s))")
    first, stop = boundary_frame(t0, "t0_s"), boundary_frame(t1, "t1_s")
    if n_frames is not None:
        if isinstance(n_frames, bool) or not isinstance(n_frames, numbers.Integral) or n_frames < 1:
            raise WindowError("n_frames must be a positive integer")
        if stop > n_frames:
            raise WindowError(f"t1_s {t1!r} is past the session end ({frame_time_s(n_frames)!r} s, "
                              f"{n_frames} frames)")
    if stop <= first:
        raise WindowError(f"[{t0!r}, {t1!r}) contains no frame at {FPS_INT} fps (both ends round to frame {first})")
    return first, stop


def window_s(first, stop):
    """(t0_s, t1_s) of the frames range(first, stop): the inverse of window_frames."""
    t0, t1 = frame_time_s(first), frame_time_s(stop)        # both validated as frames first
    if stop <= first:
        raise WindowError("stop must be after first")
    return t0, t1


def duration_frames(d_s, name="duration_s", allow_zero=False):
    """A lone duration in seconds as frames, by the same rule: floor(d * 30 + 0.5).
    Without allow_zero, a duration that rounds to zero frames is refused. Not
    for building windows: convert each absolute endpoint instead (module doc)."""
    d = _seconds(d_s, name)
    if d < 0 or (d == 0 and not allow_zero):
        raise WindowError(f"{name} must be {'>= 0' if allow_zero else '> 0'} seconds")
    n = _round_half_up(d, name)
    if n == 0 and not allow_zero:            # positive, but under half a frame: no frame at all
        raise WindowError(f"{name} {d!r} is shorter than one frame at {FPS_INT} fps (rounds to 0 frames)")
    return n
