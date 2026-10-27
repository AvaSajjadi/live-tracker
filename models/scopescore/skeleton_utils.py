"""
Skeleton-graph helpers + result/state dataclasses for the tool_geometry
pipeline.

This file is a vendored, byte-identical copy of the four symbols we need
from `codes/tool_tip_tracking/src/tip_detection_2d/robust_tip_detector.py`:

  - DetectionState     (dataclass)
  - DetectionResult    (dataclass)
  - clean_mask         (largest CC + close + fill + remove small)
  - build_skeleton_graph  (8-neighbour graph + endpoints + junctions)
  - calculate_confidence

The original file is left untouched (the broader ScopeScore repo is
immutable). Other pipelines outside `tool_geometry/` still import from
the original. Vendoring here lets `tool_geometry/` run without depending
on the `codes/` tree.

Do not edit the algorithms in this file — only the docstring above is
new content. Everything below is copied verbatim from the source.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

try:
    from skimage.morphology import skeletonize, closing, remove_small_objects, disk
    from skimage.measure import label, regionprops
    from skimage.segmentation import clear_border
except ImportError:
    raise ImportError("scikit-image required. Install: pip install scikit-image")


@dataclass
class DetectionState:
    """State for temporal tracking across frames."""
    tip_xy: Optional[Tuple[int, int]] = None
    start_xy: Optional[Tuple[int, int]] = None
    center_xy: Optional[Tuple[int, int]] = None
    conf: float = 0.0
    last_valid_frame_idx: int = -1


@dataclass
class DetectionResult:
    """Result from tip/start detection."""
    tip_xy: Optional[Tuple[int, int]] = None      # (x, y) = (col, row)
    start_xy: Optional[Tuple[int, int]] = None    # (x, y) = (col, row)
    center_xy: Optional[Tuple[int, int]] = None   # (x, y) = (col, row)
    conf: float = 0.0
    debug: Dict = field(default_factory=dict)


# =============================================================================
# STEP A: Mask Cleaning
# =============================================================================

def clean_mask(
    mask: np.ndarray,
    min_area: int = 500,
    close_radius: int = 3,
    max_hole_ratio: float = 0.001
) -> Tuple[np.ndarray, bool]:
    """
    Clean the binary mask to prevent false skeleton endpoints.

    Steps:
    1. Extract largest connected component
    2. Morphological closing to bridge small gaps
    3. Fill small holes
    4. Remove small objects

    Args:
        mask: Binary mask [H, W]
        min_area: Minimum area threshold (pixels)
        close_radius: Radius for morphological closing
        max_hole_ratio: Max hole area as fraction of image area

    Returns:
        cleaned_mask: Cleaned binary mask
        valid: Whether mask passes minimum area check
    """
    h, w = mask.shape
    mask = (mask > 0).astype(np.uint8)

    # Check minimum area
    total_area = np.sum(mask)
    if total_area < min_area:
        return mask, False

    # 1. Extract largest connected component
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(component_sizes) + 1
        mask = (labeled == largest_label).astype(np.uint8)

    # 2. Morphological closing (bridge small gaps)
    if close_radius > 0:
        selem = disk(close_radius)
        mask = closing(mask.astype(bool), selem).astype(np.uint8)

    # 3. Fill small holes
    max_hole_area = int(max_hole_ratio * h * w)
    filled = ndimage.binary_fill_holes(mask)
    # Only fill holes smaller than threshold
    holes = filled.astype(np.uint8) - mask
    if np.any(holes):
        labeled_holes, n_holes = ndimage.label(holes)
        for i in range(1, n_holes + 1):
            hole_area = np.sum(labeled_holes == i)
            if hole_area <= max_hole_area:
                mask[labeled_holes == i] = 1

    # 4. Remove small objects (disconnected fragments)
    # Note: remove_small_objects removes objects with area <= max_size
    mask = remove_small_objects(mask.astype(bool), max_size=min_area // 2).astype(np.uint8)

    # Re-check area after cleaning
    final_area = np.sum(mask)
    valid = final_area >= min_area

    return mask, valid


# =============================================================================
# STEP B & C: Skeleton Graph Analysis
# =============================================================================

def build_skeleton_graph(skeleton: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Build an 8-neighbor graph from skeleton pixels.

    Args:
        skeleton: Binary skeleton image

    Returns:
        graph: Dict mapping pixel coord -> list of neighbor coords
        endpoints: Array of endpoint coordinates (degree=1)
        junctions: Array of junction coordinates (degree>=3)
    """
    # Get skeleton pixel coordinates
    skel_coords = np.argwhere(skeleton > 0)
    if len(skel_coords) == 0:
        return {}, np.array([]), np.array([])

    # Create coordinate set for fast lookup
    coord_set = set(map(tuple, skel_coords))

    # 8-neighbor offsets
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

    graph = {}
    degrees = {}

    for coord in skel_coords:
        r, c = coord
        key = (r, c)
        neighbors = []

        for dr, dc in neighbors_8:
            nr, nc = r + dr, c + dc
            if (nr, nc) in coord_set:
                neighbors.append((nr, nc))

        graph[key] = neighbors
        degrees[key] = len(neighbors)

    # Classify nodes
    endpoints = np.array([k for k, d in degrees.items() if d == 1])
    junctions = np.array([k for k, d in degrees.items() if d >= 3])

    return graph, endpoints, junctions


# =============================================================================
# STEP F: Confidence Scoring
# =============================================================================

def calculate_confidence(
    path_length: int,
    num_junctions: int,
    assignment_cost: float = 0.0,
    L_min: float = 60.0,
    sigma: float = 40.0
) -> float:
    """
    Calculate detection confidence score.

    Components:
    - c_len: Path length factor (longer = more confident)
    - c_stable: Temporal stability factor (lower cost = more stable)
    - c_branch: Skeleton quality (fewer junctions = cleaner skeleton)

    Args:
        path_length: Length of main skeleton path in pixels
        num_junctions: Number of junction points in skeleton
        assignment_cost: Cost from temporal assignment (if applicable)
        L_min: Minimum path length for full confidence
        sigma: Scaling factor for temporal cost

    Returns:
        Confidence score in [0, 1]
    """
    # Path length factor
    c_len = min(path_length / L_min, 1.0) if L_min > 0 else 1.0

    # Temporal stability factor
    c_stable = np.exp(-assignment_cost / sigma) if sigma > 0 else 1.0

    # Skeleton quality factor
    c_branch = 1.0 if num_junctions == 0 else np.exp(-num_junctions)

    conf = c_len * c_stable * c_branch
    return float(np.clip(conf, 0.0, 1.0))
