"""Synthetic per-frame classifier scores for testing the monitor (new file).

SYNTHETIC TEST DATA ONLY. These numbers are made up to exercise the monitor
and the replay page. They are not CalMS21 data and not predictions of any
model. Every session ID starts with "SYNTHETIC_" so they can never be
mistaken for real recordings.

Output format: INTERFACES.md §2, one row per session x frame:
    session, frame, time_s, attack, mount, investigation, tracking_ok

Same arguments -> exactly the same table (seeded numpy generator).

Command line (writes a CSV for the replay page):
    python -m live_tracker.fake_scores fake_scores.csv
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

SYNTHETIC_PREFIX = "SYNTHETIC_"
SCORE_COLS = ("attack", "mount", "investigation")
COLUMNS = ("session", "frame", "time_s", *SCORE_COLS, "tracking_ok")
DEFAULT_FPS = 30.0  # proposal-reported CalMS21 rate; to be confirmed by H1a
BACKGROUND = (0.0, 0.2)  # score range outside any segment


@dataclass(frozen=True)
class Segment:
    """Scores for one behavior drawn uniformly from [low, high] on [start_s, end_s)."""
    behavior: str
    start_s: float
    end_s: float
    low: float
    high: float
    note: str = ""


@dataclass(frozen=True)
class TrackingLoss:
    """tracking_ok is False on [start_s, end_s). Scores there are left as given."""
    start_s: float
    end_s: float
    note: str = ""


def build_scores(session, duration_s, segments=(), tracking_loss=(),
                 fps=DEFAULT_FPS, seed=0):
    """One synthetic session as a §2 score table.

    Frames are 0..n-1 with time_s = frame / fps, n = round(duration_s * fps).
    A frame belongs to a segment if start_s <= time_s < end_s (end exclusive,
    like the event log), so a segment of d seconds has round(d * fps) frames.
    """
    if not str(session).startswith(SYNTHETIC_PREFIX):
        raise ValueError(f"synthetic session IDs must start with {SYNTHETIC_PREFIX!r}")
    if fps <= 0 or duration_s <= 0:
        raise ValueError("fps and duration_s must be > 0")
    n = int(round(duration_s * fps))
    frame = np.arange(n)
    t = frame / fps
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"session": str(session), "frame": frame, "time_s": t})
    for col in SCORE_COLS:
        df[col] = rng.uniform(*BACKGROUND, n)
    ok = np.ones(n, dtype=bool)

    # small tolerance so a boundary like 2.0 s at 30 fps lands on frame 60 exactly
    def mask(start_s, end_s):
        return (t >= start_s - 1e-9) & (t < end_s - 1e-9)

    for seg in segments:
        if seg.behavior not in SCORE_COLS:
            raise ValueError(f"unknown behavior {seg.behavior!r}")
        if not 0.0 <= seg.low <= seg.high <= 1.0:
            raise ValueError("segment scores must satisfy 0 <= low <= high <= 1")
        m = mask(seg.start_s, seg.end_s)
        df.loc[m, seg.behavior] = rng.uniform(seg.low, seg.high, int(m.sum()))
    for loss in tracking_loss:
        ok[mask(loss.start_s, loss.end_s)] = False
    df["tracking_ok"] = ok
    for col in SCORE_COLS:
        df[col] = df[col].round(4)
    return df[list(COLUMNS)]


# The demo scenario. Expected monitor results for the example rule
# (attack, min 2 s, ask 0.5, alert 0.8, mode "alert") are in the notes.
DEMO_SESSION = "SYNTHETIC_demo_01"
DEMO_DURATION_S = 24.0
DEMO_SEGMENTS = (
    Segment("attack", 2.0, 5.0, 0.88, 0.97,
            "clear attack, 3 s, high scores -> one 'alert' that fires on frame 120 (4.0 s)"),
    Segment("attack", 7.0, 10.0, 0.58, 0.72,
            "uncertain attack, 3 s, scores between ask and alert -> one 'ask' at 9.0 s"),
    Segment("attack", 12.0, 13.5, 0.88, 0.97,
            "short attack, 1.5 s -> no event"),
    Segment("attack", 15.0, 18.5, 0.88, 0.97,
            "high attack scores, but tracking is lost on 16.0-17.0 s, "
            "leaving 1.0 s + 1.5 s of valid evidence -> no event"),
    Segment("investigation", 20.0, 23.0, 0.85, 0.95,
            "clear investigation, 3 s (for multi-rule tests)"),
)
DEMO_TRACKING_LOSS = (
    TrackingLoss(16.0, 17.0, "tracking lost in the middle of a high-score stretch"),
)


def demo_scores(fps=DEFAULT_FPS, seed=0):
    """The labeled synthetic demo session (see DEMO_SEGMENTS for what is in it)."""
    return build_scores(DEMO_SESSION, DEMO_DURATION_S, DEMO_SEGMENTS,
                        DEMO_TRACKING_LOSS, fps=fps, seed=seed)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m live_tracker.fake_scores OUTPUT.csv")
        return 2
    demo_scores().to_csv(argv[0], index=False)
    print(f"wrote SYNTHETIC test scores (not CalMS21) to {argv[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
