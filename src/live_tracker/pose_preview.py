"""A camera picture with the keypoints the pose model found, drawn on it (new file).

Used by the Replay step (the page reads the camera itself) and by a camera run (the run program has the camera, so it saves its
newest drawn picture in the run folder, and the Real session page shows that file). Only shown, never scored.

The picture is brought to the CalMS21 frame (1024 x 570) first, the same way the camera run does it: the screen part stretched
when corners were set, else the whole picture resized. The keypoints are in that frame, so they are drawn on it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .calms21 import IMAGE_SIZE
from .quality import SUGGESTED_MIN_SCORE
from .skeleton import PROVISIONAL_EDGES

PREVIEW_FILE = "preview.jpg"
PREVIEW_EVERY_S = 0.2
COLORS = {"a": (60, 60, 230), "b": (230, 160, 40)}   # BGR: mouse a (black mouse) red, mouse b (white mouse) blue


def to_frame(image, screen):
    import cv2
    if screen is not None:
        from . import screen_map as sm
        return sm.warp(image, screen)
    return cv2.resize(image, tuple(IMAGE_SIZE))


def draw(image, found):
    """A copy of the picture with each mouse's skeleton. A point below the score threshold is drawn hollow."""
    import cv2
    out = image.copy()
    if found is None:
        return out
    for mouse, rows in found.groupby("mouse"):
        color = COLORS.get(mouse, (255, 255, 255))
        points = {r.keypoint: (int(r.x), int(r.y), r.score) for r in rows.itertuples() if r.x == r.x and r.y == r.y}
        for a, b in PROVISIONAL_EDGES:
            if a in points and b in points:
                cv2.line(out, points[a][:2], points[b][:2], color, 2, cv2.LINE_AA)
        for x, y, score in points.values():
            cv2.circle(out, (x, y), 5, color, -1 if score >= SUGGESTED_MIN_SCORE else 2, cv2.LINE_AA)
    return out


def with_preview(estimator, path, screen=None, every_s=PREVIEW_EVERY_S, now_fn=time.monotonic):
    """A pose function that brings the picture to the CalMS21 frame, calls `estimator`, and at most every `every_s` seconds saves
    the drawn picture to `path` (written to a temporary file, then renamed, so a reader never sees half a picture)."""
    import cv2
    path = Path(path)
    last = [None]

    def estimate(image):
        frame = to_frame(image, screen)
        found = estimator(frame)
        now = now_fn()
        if last[0] is None or now - last[0] >= every_s:
            last[0] = now
            try:
                from .pose_model import check_output
                ok, data = cv2.imencode(".jpg", draw(frame, check_output(found)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(data.tobytes())
                    os.replace(tmp, path)
            except (OSError, ValueError):
                pass                             # the preview is only shown: it never stops the run
        return found
    return estimate
