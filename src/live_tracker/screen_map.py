"""Find the video inside a camera picture and stretch it to the CalMS21 frame (new file).

The real presentation: a camera on a holder films an iPad that plays a CalMS21 video. The iPad screen is only a part of
the camera picture, a little tilted and not the same shape. The researcher gives the 4 corners of the video once; every
picture is then stretched so that the video fills a 1024 x 570 frame, and only then does the pose model see it. So the
pose model's keypoints are already in CalMS21 pixels.

Corners are fractions of the camera picture (0 to 1), in this order: top-left, top-right, bottom-right, bottom-left.
OpenCV is imported only inside the functions that need it.
"""

from __future__ import annotations

import numpy as np

from .calms21 import IMAGE_SIZE

MIN_AREA = 0.02                     # the video must cover at least 2% of the picture
DEFAULT_CORNERS = ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85))


class ScreenError(ValueError):
    """The 4 corners do not describe a usable screen."""


def check_corners(corners):
    """The corners as a tuple of 4 (x, y) float pairs, or ScreenError saying what is wrong."""
    try:
        pts = [(float(x), float(y)) for x, y in corners]
    except (TypeError, ValueError) as exc:
        raise ScreenError("the corners must be 4 pairs of numbers") from exc
    if len(pts) != 4:
        raise ScreenError("a screen has exactly 4 corners")
    if any(not (0.0 <= v <= 1.0) for p in pts for v in p):
        raise ScreenError("every corner must lie inside the picture (0 to 1)")
    crosses = []
    for i in range(4):
        (x0, y0), (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        crosses.append((x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1))
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        raise ScreenError("the corners must go around the screen in order (top-left, top-right, bottom-right, bottom-left) "
                          "and make a four-sided shape that does not cross itself")
    area = 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4)))
    if area < MIN_AREA:
        raise ScreenError("the screen is too small in the picture")
    return tuple(pts)


def _matrix(corners, image_shape, size):
    import cv2
    h, w = image_shape[:2]
    src = np.float32([(x * w, y * h) for x, y in corners])
    dst = np.float32([(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])])
    return cv2.getPerspectiveTransform(src, dst)


def warp(image, corners, size=IMAGE_SIZE):
    """The screen part of the picture, stretched to `size` (width, height)."""
    import cv2
    corners = check_corners(corners)
    return cv2.warpPerspective(image, _matrix(corners, image.shape, size), tuple(size))


def outline(image, corners):
    """A copy of the picture with the screen outline drawn on it (for the setup page)."""
    import cv2
    corners = check_corners(corners)
    h, w = image.shape[:2]
    pts = np.int32([(x * w, y * h) for x, y in corners])
    out = image.copy()
    cv2.polylines(out, [pts], True, (0, 255, 0), max(2, w // 300))
    for i, p in enumerate(pts):
        cv2.circle(out, tuple(int(v) for v in p), max(4, w // 150), (0, 0, 255), -1)
        cv2.putText(out, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1000),
                    (0, 255, 255), 2)
    return out


def with_screen(estimator, corners):
    """A pose function that first stretches the screen to the CalMS21 frame, then calls `estimator`."""
    corners = check_corners(corners)
    return lambda image: estimator(warp(image, corners))
