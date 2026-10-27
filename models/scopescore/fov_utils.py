"""
Field-of-view mask helpers for endoscopic videos.

Cholec80 frames have a black border around a circular FOV. Knowing that
FOV is critical for tip/start detection: a skeleton endpoint that exits
the FOV is almost certainly a trocar entry.

`compute_fov_mask` samples N evenly-spaced frames, takes the per-pixel
max across them (so any pixel that's ever bright counts), thresholds out
the dark border, and keeps the largest bright connected component.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def compute_fov_distance(fov_mask: np.ndarray) -> np.ndarray:
    """Per-pixel distance (float32) to the nearest FOV-out pixel.

    Used by the tip detector: a skeleton endpoint that sits where
    the tool exits the visible area has a tiny distance here. The trocar
    end always wins by a wide margin over the free tip.

    If the FOV mask has no zeros (whole frame is FOV), we fall back to
    distance from the rectangular image edge so the rule still has a
    meaningful signal.
    """
    fov_uint8 = fov_mask.astype(np.uint8)
    if fov_uint8.min() == 1:
        H, W = fov_mask.shape
        rows = np.arange(H, dtype=np.int32).reshape(-1, 1)
        cols = np.arange(W, dtype=np.int32).reshape(1, -1)
        return np.minimum.reduce([rows,
                                  H - 1 - rows,
                                  cols,
                                  W - 1 - cols]).astype(np.float32)
    return cv2.distanceTransform(fov_uint8, cv2.DIST_L2, 3)


def compute_fov_mask(video_path: Union[str, Path],
                     n_sample: int = 20,
                     dark_thresh: int = 15,
                     close_radius: int = 5) -> np.ndarray:
    """Compute a boolean field-of-view mask for the given video.

    Args:
        video_path:  Path to the video file.
        n_sample:    Number of frames to sample uniformly along the video.
        dark_thresh: Grayscale threshold below which a pixel counts as
                     border (out-of-FOV).
        close_radius: Morphological closing radius applied to fill small
                     notches (text overlays, scopes glints).

    Returns:
        fov_mask: bool array of shape (H, W). True inside the visible FOV.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"can't open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if total <= 0 or H <= 0 or W <= 0:
        cap.release()
        raise RuntimeError(f"bad video metadata: {video_path}")

    # Skip the first/last 5% of the video to avoid title cards / credits
    # that may be uniformly bright and contaminate the FOV estimate.
    skip = max(1, total // 20)
    sample_indices = np.linspace(skip, max(skip, total - 1 - skip),
                                 n_sample).astype(int)
    samples = []
    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append(gray)
    cap.release()
    n_read = len(samples)

    if n_read == 0:
        return np.ones((H, W), dtype=bool)

    # Pixels OUTSIDE the FOV are black in nearly every real surgical frame.
    # Count, per pixel, how many sampled frames are dark there. If the
    # majority say "dark", it's outside the FOV.
    stacked = np.stack(samples, axis=0)
    n_dark = (stacked <= dark_thresh).sum(axis=0)
    bright = (n_dark < (n_read / 2)).astype(np.uint8)

    # Morphological closing to bridge text overlays / specular dropouts.
    if close_radius > 0:
        ksize = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    # Keep the largest connected component.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright,
                                                                  connectivity=8)
    if n_labels <= 1:
        return bright.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return bright.astype(bool)
    largest = int(np.argmax(areas)) + 1
    fov_mask = (labels == largest)

    # Fill internal holes so that bright tools/specular regions don't
    # create "FOV out" pockets inside the visible area. Any non-FOV
    # component that TOUCHES an image-border pixel is real outside-FOV
    # area (left strip, right strip, corners, top/bottom letterbox, ...).
    # Components that are FULLY INTERIOR (no border pixel) are holes
    # and get filled.
    inv = (~fov_mask).astype(np.uint8)
    inv_n, inv_labels, inv_stats, _ = cv2.connectedComponentsWithStats(inv,
                                                                       connectivity=8)
    if inv_n > 1:
        border_labels = set()
        for row in (inv_labels[0, :], inv_labels[-1, :]):
            border_labels.update(int(v) for v in np.unique(row) if v > 0)
        for col in (inv_labels[:, 0], inv_labels[:, -1]):
            border_labels.update(int(v) for v in np.unique(col) if v > 0)
        for lbl in range(1, inv_n):
            if lbl in border_labels:
                continue
            fov_mask[inv_labels == lbl] = True

    return fov_mask.astype(bool)
