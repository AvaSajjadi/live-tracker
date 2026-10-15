"""Convert between the two table layouts we use.

New file for Live Tracker (written by us, not part of the starter).

Wide (starter CSV):  one row per frame, columns x_a, y_a, x_b, y_b
Long (quality.py, filters.py): one row per frame per mouse,
                     columns session, mouse, frame, time_s, x, y
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MICE = ("a", "b")


def wide_to_long(df, fps=30.0, session="upload"):
    """Turn the starter's wide CSV into the long layout.

    time_s is frame / fps, because the starter CSV has no timestamp column.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    needed = [f"{c}_{m}" for m in MICE for c in ("x", "y")]
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    frames = np.arange(len(df))
    parts = []
    for m in MICE:
        parts.append(pd.DataFrame({
            "session": session, "mouse": m, "frame": frames,
            "time_s": frames / fps,
            "x": df[f"x_{m}"].to_numpy(float), "y": df[f"y_{m}"].to_numpy(float),
        }))
    return pd.concat(parts, ignore_index=True)


def inject_tracking_failures(df, fps=30.0, gap_s=1.5, freeze_s=1.0):
    """Make a copy of a wide table with fake tracking problems, for testing.

    - mouse b disappears (NaN) for gap_s seconds in the middle,
    - mouse a is frozen at one position for freeze_s seconds at 1/4 of the way.
    This is only for showing that warnings work; it is not real data.
    """
    out = df.copy().astype(float)
    n = len(out)
    g = int(round(gap_s * fps))
    mid = n // 2
    out.loc[mid:mid + g - 1, ["x_b", "y_b"]] = np.nan
    f = int(round(freeze_s * fps))
    q = n // 4
    out.loc[q:q + f - 1, ["x_a", "y_a"]] = out.loc[q, ["x_a", "y_a"]].to_numpy()
    return out
