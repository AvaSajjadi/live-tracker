"""Look inside a CalMS21 JSON file WITHOUT assuming its structure.

New file for Live Tracker (written by us, not part of the starter).

We have not seen the real files yet, so we do not want to guess array
shapes, keypoint order, or label numbers. This script just walks the JSON
and prints what is there: keys, list lengths, and array shapes. We will use
its output plus the official readme.md to write the real loader.

Usage (from the repo root):
    python -m live_tracker.calms21_inspect data/raw/calms21/<file>.json
    python -m live_tracker.calms21_inspect <file>.json --max-keys 3 --depth 6

Note: the Task 1 JSON may be several hundred MB, so loading it needs RAM.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _array_shape(value):
    """Shape if value is a regular (rectangular) nested list of numbers."""
    try:
        arr = np.asarray(value)
    except (ValueError, TypeError):
        return None
    if arr.dtype == object or arr.ndim == 0:
        return None
    return arr.shape, str(arr.dtype)


def describe(value, depth=5, max_keys=5, indent=0, lines=None):
    """Return a list of text lines describing value's structure."""
    if lines is None:
        lines = []
    pad = "  " * indent
    if isinstance(value, dict):
        keys = list(value)
        lines.append(f"{pad}dict with {len(keys)} keys")
        for k in keys[:max_keys]:
            lines.append(f"{pad}  [{k!r}]")
            if depth > 0:
                describe(value[k], depth - 1, max_keys, indent + 2, lines)
        if len(keys) > max_keys:
            lines.append(f"{pad}  ... {len(keys) - max_keys} more keys "
                         f"(e.g. {keys[max_keys:max_keys + 3]})")
    elif isinstance(value, list):
        shape = _array_shape(value) if value else None
        if shape is not None:
            lines.append(f"{pad}numeric array shape={shape[0]} dtype={shape[1]}")
        else:
            lines.append(f"{pad}list of length {len(value)}")
            if value and depth > 0:
                lines.append(f"{pad}  first item:")
                describe(value[0], depth - 1, max_keys, indent + 2, lines)
    else:
        lines.append(f"{pad}{type(value).__name__}: {str(value)[:80]!r}")
    return lines


from live_tracker.calms21 import load_calms21  # noqa: E402,F401  (real loader lives in calms21.py)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--max-keys", type=int, default=3)
    a = p.parse_args(argv)
    with open(a.path) as f:
        data = json.load(f)
    print("\n".join(describe(data, a.depth, a.max_keys)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
