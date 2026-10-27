"""
Geometric tip / start / center / secondary-jaw detector.

The tool is rendered as a polyline that follows the skeleton — so open
jaws fork at the real junction near the tip, not at S — BUT the noisy
trocar-end of the skeleton is replaced by a single straight segment, so
there is no spurious bend right after S. The skeleton's medial axis is
reliable along the shaft and at the jaw fork; it only wiggles near the
trocar where the mask is fat and noisy.

Per frame, from one tool mask:

  1. Clean the mask, skeletonize it, build the skeleton graph.
  2. tip  = skeleton endpoint FARTHEST from the FOV-out region (the
            working end sits deep in the visible field).
     start= skeleton endpoint geodesically FARTHEST from the tip (the
            end of the longest skeleton run = the main shaft's trocar
            end). A blobby protrusion near the trocar is a SHORT spur,
            so it never wins "farthest from tip" — it is ignored.
  3. T  = the real skeleton tip, snapped outward to the mask boundary.
  4. S  = PCA-fit one straight axis to the whole skeleton path, then
          walk that axis outward to the mask edge on the trocar side.
          (Local trocar-end wiggle averages out, so S lands on the
          tool's true centreline.)
  5. anchor = the skeleton point ~s_straight_len px (arc length) past
          the trocar end. The rendered polyline is:
            [S, anchor, ...real skeleton..., T]
          i.e. one straight S->anchor segment + the real skeleton from
          anchor to the tip. Jaw branches share the S->anchor->shaft
          prefix and fork at the real junction.
  6. C  = geodesic midpoint of that rebuilt polyline (a point on the line).
  7. T2 = a second jaw tip, ONLY for jawed classes (grasper, bipolar,
          scissors, clipper): another endpoint whose BFS depth from
          start is >= jaw_branch_ratio of the tip's depth AND whose
          image-space distance from the tip is >= jaw_min_tip_separation
          px. Hook / irrigator never get a T2.

`fov_dist` (distance to the FOV-out region, from
fov_utils.compute_fov_distance) is computed once per video.

Every frame's decision is fresh — no prev_state, no temporal carry-forward.
Temporal smoothing of the rendered dot positions lives in the pipeline
(run_pipeline.py), not in this module.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skeleton_utils import (  # noqa: E402
    DetectionResult,
    build_skeleton_graph,
    calculate_confidence,
    clean_mask,
)


# -----------------------------------------------------------------------------
# BFS helpers on the skeleton graph
# -----------------------------------------------------------------------------
def _bfs_farthest_endpoint_and_path(graph: Dict,
                                    source: Tuple[int, int],
                                    endpoint_set: set
                                    ) -> Tuple[Optional[Tuple[int, int]],
                                               List[Tuple[int, int]]]:
    """BFS from source. Return (farthest_endpoint, path) where farthest
    is the most-distant node in endpoint_set."""
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {source: None}
    depth: Dict[Tuple[int, int], int] = {source: 0}
    q = deque([source])
    farthest = None
    farthest_depth = -1
    while q:
        node = q.popleft()
        for n in graph.get(node, []):
            if n in parent:
                continue
            parent[n] = node
            depth[n] = depth[node] + 1
            q.append(n)
            if n in endpoint_set and n != source and depth[n] > farthest_depth:
                farthest_depth = depth[n]
                farthest = n
    if farthest is None:
        return None, []
    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = farthest
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return farthest, path


def _bfs_paths_to_all_endpoints(graph: Dict,
                                source: Tuple[int, int],
                                endpoint_set: set
                                ) -> Dict[Tuple[int, int],
                                          Tuple[int, List[Tuple[int, int]]]]:
    """One BFS from source. Return {endpoint: (depth, path)} for every
    endpoint reachable in the skeleton graph (except source itself)."""
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {source: None}
    depth: Dict[Tuple[int, int], int] = {source: 0}
    q = deque([source])
    while q:
        node = q.popleft()
        for n in graph.get(node, []):
            if n in parent:
                continue
            parent[n] = node
            depth[n] = depth[node] + 1
            q.append(n)
    out: Dict[Tuple[int, int], Tuple[int, List[Tuple[int, int]]]] = {}
    for ep in endpoint_set:
        if ep == source or ep not in parent:
            continue
        path: List[Tuple[int, int]] = []
        cur: Optional[Tuple[int, int]] = ep
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        out[ep] = (depth[ep], path)
    return out


def _bfs_inner_node(graph: Dict, source: Tuple[int, int], depth: int
                    ) -> Tuple[int, int]:
    """Return the node reached at the given BFS depth from source, or
    the deepest reachable node if depth exceeds the skeleton's reach.
    Used to compute an outward tangent at a skeleton endpoint."""
    visited = {source: 0}
    q = deque([source])
    last = source
    while q:
        node = q.popleft()
        d = visited[node]
        last = node
        if d >= depth:
            return node
        for n in graph.get(node, []):
            if n not in visited:
                visited[n] = d + 1
                q.append(n)
    return last


def _outward_tangent(endpoint: Tuple[int, int],
                     graph: Dict,
                     k: int = 5) -> Tuple[float, float]:
    inner = _bfs_inner_node(graph, endpoint, k)
    dy = float(endpoint[0] - inner[0])
    dx = float(endpoint[1] - inner[1])
    L = (dy * dy + dx * dx) ** 0.5
    if L < 1e-6:
        return 0.0, 0.0
    return dy / L, dx / L


def _snap_to_mask_boundary(endpoint: Tuple[int, int],
                           tangent: Tuple[float, float],
                           cleaned_mask: np.ndarray,
                           max_walk: int = 30) -> Tuple[int, int]:
    """Walk outward from endpoint along tangent. Return the last in-mask
    pixel — i.e. the visible tip of the tool, not the skeleton terminus."""
    H, W = cleaned_mask.shape
    dy, dx = tangent
    cy, cx = endpoint
    last = (int(cy), int(cx))
    for s in range(1, max_walk + 1):
        ny = int(round(cy + dy * s))
        nx = int(round(cx + dx * s))
        if ny < 0 or ny >= H or nx < 0 or nx >= W:
            break
        if cleaned_mask[ny, nx] == 0:
            break
        last = (ny, nx)
    return last


def _straight_axis_S(path: List[Tuple[int, int]],
                     tip_ep: Tuple[int, int],
                     cleaned_mask: np.ndarray,
                     max_walk: int = 200) -> Tuple[int, int]:
    """S = where the tool's single straight axis crosses the mask edge on
    the trocar side.

    The tool is rendered as ONE straight line S -> T. For that line to lie
    along the tool, S must sit on the tool's overall axis — not on a
    corner / protrusion of the noisy medial axis. So:

      1. PCA-fit a single straight axis to the WHOLE skeleton main path.
      2. Take the trocar-side extreme of the path projected onto that axis.
      3. Walk from there OUTWARD along the axis to the mask edge.

    Because the axis is fit over the whole shaft, local mask-boundary
    wiggle near the trocar averages out — S lands on the tool's true
    centreline, and the straight S -> T line has no bend.
    """
    n = len(path)
    if n < 3:
        return path[0]
    arr = np.asarray(path, dtype=np.float64)                # (N, 2) row,col
    m = arr.mean(axis=0)
    centered = arr - m
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return path[0]
    axis = vt[0]                                            # unit (dy, dx)
    proj = centered @ axis
    # orient the axis so +axis points toward the tip
    tip_proj = float((np.asarray(tip_ep, dtype=np.float64) - m) @ axis)
    if tip_proj < 0.0:
        axis = -axis
        proj = -proj
    # trocar-side extreme of the path, reconstructed onto the axis line
    p_start = m + axis * float(proj.min())
    dy, dx = -float(axis[0]), -float(axis[1])               # outward (toward trocar)
    H, W = cleaned_mask.shape
    cy, cx = float(p_start[0]), float(p_start[1])
    # make sure we begin inside the mask (the projected point can land a
    # pixel or two off the noisy boundary) — nudge inward if needed
    iy, ix = int(round(cy)), int(round(cx))
    if not (0 <= iy < H and 0 <= ix < W and cleaned_mask[iy, ix]):
        for s in range(1, 20):
            iy = int(round(cy - dy * s))
            ix = int(round(cx - dx * s))
            if 0 <= iy < H and 0 <= ix < W and cleaned_mask[iy, ix]:
                cy, cx = cy - dy * s, cx - dx * s
                break
    last = (int(round(cy)), int(round(cx)))
    for s in range(1, max_walk + 1):
        ny = int(round(cy + dy * s))
        nx = int(round(cx + dx * s))
        if ny < 0 or ny >= H or nx < 0 or nx >= W:
            break
        if cleaned_mask[ny, nx] == 0:
            break
        last = (ny, nx)
    return last


def _arc_anchor_index(path: List[Tuple[int, int]],
                      target_len: float,
                      max_frac: float = 0.4) -> int:
    """Index of the path point ~`target_len` px (arc length) from path[0],
    capped at `max_frac` of the total path length so short tools still
    keep most of their skeleton. Used to find the 'anchor' that ends the
    straight trocar segment and begins the real skeleton polyline."""
    n = len(path)
    if n < 3:
        return 0
    seg = []
    total = 0.0
    for i in range(1, n):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        d = (dr * dr + dc * dc) ** 0.5
        seg.append(d)
        total += d
    tgt = min(float(target_len), total * max_frac)
    cum = 0.0
    for i, d in enumerate(seg, start=1):
        cum += d
        if cum >= tgt:
            return i
    return n - 1


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
# YOLO classes that actually have a two-part jaw. Only these can produce
# a secondary jaw tip (T2) / jaw_state == 'open'. Hook (2) and irrigator
# (5) have no jaws; specimen_bag (6) is not a manipulator. A hook's
# curved tip makes a branchy skeleton, but that branch is NOT a jaw, so
# T2 detection is skipped entirely for non-jawed classes.
JAW_CLASSES = {0, 1, 3, 4}  # grasper, bipolar, scissors, clipper


def detect_tip_start(mask: np.ndarray,
                            fov_dist: np.ndarray,
                            frame_idx: int = 0,
                            image_shape: Optional[Tuple[int, int]] = None,
                            tool_class: int = -1,
                            min_area: int = 300,
                            close_radius: int = 3,
                            snap_to_boundary: bool = True,
                            tangent_k: int = 5,
                            snap_max_walk: int = 30,
                            s_max_walk: int = 80,
                            s_straight_len: float = 35.0,
                            jaw_branch_ratio: float = 0.7,
                            jaw_min_branch_px: int = 10,
                            jaw_min_tip_separation_px: float = 20.0,
                            ) -> DetectionResult:
    """Run the tip / start / center / T2 detector on a tool mask.

    Args:
        mask: (H, W) binary mask of one tool.
        fov_dist: (H, W) float, distance from each pixel to the nearest
            FOV-out pixel. Used to pick the trocar-side skeleton endpoint
            (BFS source). Precompute once per video via
            fov_utils.compute_fov_distance.
        frame_idx, image_shape, min_area, close_radius: standard.
        tool_class: YOLO class index. Only jawed classes (grasper,
            bipolar, scissors, clipper) can produce a T2 / jaw_state ==
            'open'. Hook / irrigator are skipped — a hook's curved tip
            makes a branchy skeleton, but that branch is not a jaw.
            -1 (unknown) leaves T2 detection enabled.
        snap_to_boundary: if True, walk outward from the tip endpoint to
            the last in-mask pixel. (S is placed by _straight_axis_S.)
        tangent_k: number of skeleton steps used for local tangents.
        snap_max_walk: max pixels to walk during the tip boundary snap.
        s_max_walk: max pixels to walk the midline outward when locating
            S (the tool's starting point on the mask edge).

    Returns a DetectionResult with tip_xy, start_xy, center_xy, conf,
    plus a debug dict (BFS path, jaw state, etc.).
    """
    debug: Dict = {
        'frame_idx': frame_idx,
        'mask_area_raw': int(np.sum(mask > 0)),
        'mask_area_clean': 0,
        'path_length': 0,
        'num_endpoints': 0,
        'method': 'fov_dist',
        'endpoints': None,
        'endpoint_fov_dist': None,
        'main_path': None,
        'jaw_paths': [],
        'jaw_state': 'unknown',
        'snap_applied': bool(snap_to_boundary),
    }

    if image_shape is None:
        image_shape = mask.shape[:2]

    cleaned_mask, valid = clean_mask(mask,
                                     min_area=min_area,
                                     close_radius=close_radius)
    debug['mask_area_clean'] = int(np.sum(cleaned_mask))
    if not valid:
        debug['error'] = f"Mask area below minimum {min_area}"
        return DetectionResult(conf=0.0, debug=debug)

    skeleton = skeletonize(cleaned_mask.astype(bool)).astype(np.uint8)
    if int(np.sum(skeleton)) == 0:
        debug['error'] = "Empty skeleton"
        return DetectionResult(conf=0.0, debug=debug)

    graph, endpoints_arr, _ = build_skeleton_graph(skeleton)
    if len(endpoints_arr) == 0:
        debug['error'] = "No skeleton endpoints"
        return DetectionResult(conf=0.0, debug=debug)
    endpoints = [tuple(int(v) for v in ep) for ep in endpoints_arr]
    debug['num_endpoints'] = len(endpoints)
    debug['endpoints'] = endpoints

    # Endpoint selection — robust to mask protrusions / corners near the
    # trocar (the cause of the sharp bend after S):
    #   tip   = endpoint FARTHEST from the FOV-out region. The working
    #           end is deep in the visible field.
    #   start = endpoint geodesically FARTHEST from the tip. That is the
    #           end of the longest skeleton run = the main shaft's
    #           trocar end. A blobby protrusion near the trocar is a
    #           SHORT side-spur, so it can never win "farthest from tip"
    #           against the long main shaft — the protrusion becomes an
    #           untraversed branch instead of dragging the path into a V.
    H_d, W_d = fov_dist.shape
    ep_dists = []
    for (r, c) in endpoints:
        if 0 <= r < H_d and 0 <= c < W_d:
            ep_dists.append(float(fov_dist[r, c]))
        else:
            ep_dists.append(0.0)
    debug['endpoint_fov_dist'] = ep_dists
    endpoint_set = set(endpoints)

    tip_guess = endpoints[int(np.argmax(ep_dists))]
    paths_from_tip = _bfs_paths_to_all_endpoints(graph, tip_guess,
                                                 endpoint_set)
    if not paths_from_tip:
        start_ep = tip_guess
    else:
        # geodesically farthest endpoint from the tip = main-shaft trocar end
        start_ep = max(paths_from_tip.items(),
                       key=lambda kv: kv[1][0])[0]

    # BFS from start to ALL skeleton endpoints. This lets us:
    #   - pick the primary tip (farthest endpoint = end of the longest branch)
    #   - detect open jaws: 2+ endpoints whose BFS depths are within
    #     jaw_branch_ratio of the longest branch and meet a minimum px length
    # T2 / jaw_state is only meaningful for jawed classes. For a hook or
    # irrigator the skeleton can still fork (a hook's curved tip), but
    # that fork is not a jaw — so we never emit a T2 for those.
    jaws_allowed = (tool_class in JAW_CLASSES) or (tool_class < 0)
    paths_by_ep = _bfs_paths_to_all_endpoints(graph, start_ep, endpoint_set)
    if not paths_by_ep:
        tip_ep = start_ep
        path = [start_ep]
        jaw_paths = [path]
        debug['jaw_paths'] = jaw_paths
        debug['jaw_state'] = 'unknown'
    else:
        # sort by depth descending: longest branch first
        items = sorted(paths_by_ep.items(),
                       key=lambda kv: -kv[1][0])
        max_depth = items[0][1][0]
        threshold = max(jaw_min_branch_px,
                        int(round(jaw_branch_ratio * max_depth)))
        long_paths = [p for ep, (d, p) in items if d >= threshold]
        if not long_paths:
            long_paths = [items[0][1][1]]
        # Stricter "jaw open" gate: secondary branches must terminate at
        # tips that are visibly separated from the primary tip in
        # image-space. A tiny skeleton fork at the very tip produces two
        # endpoints only a few pixels apart -- that is not really an
        # open jaw, just mask noise. Require an in-frame separation of
        # at least `jaw_min_tip_separation_px` pixels.
        primary_path = long_paths[0]
        primary_tip = primary_path[-1]
        jaw_paths = [primary_path]
        if jaws_allowed:
            for p in long_paths[1:]:
                jaw_tip = p[-1]
                sep = float(np.hypot(jaw_tip[0] - primary_tip[0],
                                     jaw_tip[1] - primary_tip[1]))
                if sep >= jaw_min_tip_separation_px:
                    jaw_paths.append(p)
        tip_ep = primary_tip
        path = primary_path
        debug['jaw_paths'] = jaw_paths
        if not jaws_allowed:
            debug['jaw_state'] = 'n/a'
        else:
            debug['jaw_state'] = ('open' if len(jaw_paths) >= 2
                                  else 'closed')

    # ---- Render: skeleton polyline, but with a STRAIGHT trocar segment --
    # The skeleton's medial axis is reliable along the shaft and at the
    # jaw fork, but wiggles near the trocar (fat, noisy mask). So we keep
    # the polyline — jaws fork at the real junction near the tip, not at
    # S — and only replace the noisy first ~s_straight_len px of arc
    # length with one straight S -> anchor segment.
    tip_ep_raw = tip_ep
    if snap_to_boundary and len(path) >= 2:
        tip_tangent = _outward_tangent(tip_ep_raw, graph, k=tangent_k)
        tip_snapped = _snap_to_mask_boundary(tip_ep_raw, tip_tangent,
                                             cleaned_mask, snap_max_walk)
    else:
        tip_snapped = tip_ep_raw

    s_border = _straight_axis_S(path, tip_ep_raw, cleaned_mask,
                                max_walk=max(s_max_walk, 200))
    debug['s_border'] = s_border

    # anchor = skeleton point ~s_straight_len px past the trocar end.
    # Everything before it (the noisy medial axis) is dropped and
    # replaced by the straight S -> anchor segment.
    anchor_idx = _arc_anchor_index(path, s_straight_len)
    anchor_pt = path[anchor_idx]

    # Main rendered polyline: straight S -> anchor, then the real
    # skeleton from anchor to the tip (last point snapped to the tip).
    main_path_r = [s_border] + list(path[anchor_idx:])
    if len(main_path_r) >= 2:
        main_path_r[-1] = tip_snapped

    # Jaw branches: rebuild the same way. They share the shaft prefix
    # with the main path, so anchor_pt appears in each — the fork stays
    # at the real junction near the tip.
    jaw_paths_r = [main_path_r]
    for jp in jaw_paths[1:]:
        try:
            j_idx = jp.index(anchor_pt)
            jaw_paths_r.append([s_border] + list(jp[j_idx:]))
        except ValueError:
            jaw_paths_r.append([s_border] + list(jp))

    path = main_path_r
    jaw_paths = jaw_paths_r
    debug['jaw_paths'] = jaw_paths
    debug['main_path'] = path
    debug['path_length'] = len(path)

    # center = geodesic midpoint of the rebuilt polyline (a point on the line)
    center_ep = path[len(path) // 2] if path else s_border

    start_snapped = s_border

    # row, col -> x, y for output
    tip_xy = (int(tip_snapped[1]), int(tip_snapped[0]))
    start_xy = (int(start_snapped[1]), int(start_snapped[0]))
    center_xy = (int(center_ep[1]), int(center_ep[0]))

    conf = calculate_confidence(
        path_length=len(path),
        num_junctions=0,         # not differentiating; full graph counts
        assignment_cost=0.0,     # no prev_state
    )

    return DetectionResult(
        tip_xy=tip_xy,
        start_xy=start_xy,
        center_xy=center_xy,
        conf=conf,
        debug=debug,
    )
