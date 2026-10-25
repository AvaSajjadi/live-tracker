"""Skeleton drawing that follows the video on the Replay page (new file).

The page shows the real video (st.video) and, next to it, a small canvas with the mouse skeletons. The canvas reads
the video's current time from the page and draws the keypoints of that frame, so the two stay together while the
video plays, pauses or jumps. A point that is missing in a frame is not drawn. Nothing is computed here: the
keypoints come from the poses file as they are.
"""

from __future__ import annotations

import json

import pandas as pd

from .calms21 import FPS, IMAGE_SIZE, KEYPOINTS
from .skeleton import PROVISIONAL_EDGES

COLOURS = {"a": "#222222", "b": "#d97a14"}


def pose_frames(poses_path):
    """The poses as a list over frames; each frame is {mouse: [[x, y] or None for each keypoint in KEYPOINTS order]}."""
    df = pd.read_parquet(poses_path, columns=["mouse", "keypoint", "frame", "x", "y"])
    n = int(df["frame"].max()) + 1
    frames = [{} for _ in range(n)]
    index = {k: i for i, k in enumerate(KEYPOINTS)}
    for mouse, keypoint, frame, x, y in df[df["keypoint"].isin(index)].itertuples(index=False):
        points = frames[int(frame)].setdefault(mouse, [None] * len(KEYPOINTS))
        if pd.notna(x) and pd.notna(y):
            points[index[keypoint]] = [round(float(x), 1), round(float(y), 1)]
    return frames


def html(frames, fps=FPS):
    """The page of the canvas, with the poses inside it."""
    edges = [[KEYPOINTS.index(a), KEYPOINTS.index(b)] for a, b in PROVISIONAL_EDGES]
    data = json.dumps({"frames": frames, "fps": fps, "edges": edges, "colours": COLOURS, "size": list(IMAGE_SIZE)},
                      separators=(",", ":"))
    return _TEMPLATE.replace("__DATA__", data)


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
html, body { margin: 0; background: transparent; font-family: sans-serif; }
canvas { width: 100%; background: #fafafa; border: 1px solid #ccc; border-radius: 6px; display: block; }
#note { font-size: 13px; color: #555; margin-top: 4px; }
</style></head><body>
<canvas id="c"></canvas><div id="note">Looking for the video...</div>
<script>
const D = __DATA__;
const cv = document.getElementById("c"), note = document.getElementById("note"), g = cv.getContext("2d");
cv.width = D.size[0]; cv.height = D.size[1];
function video() {
  try { return window.parent.document.querySelector("video"); } catch (e) { return null; }
}
function draw(t) {
  g.clearRect(0, 0, cv.width, cv.height);
  const f = Math.min(D.frames.length - 1, Math.max(0, Math.round(t * D.fps)));
  const mice = D.frames[f] || {};
  for (const m in mice) {
    const p = mice[m], col = D.colours[m] || "#000";
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 3;
    for (const [a, b] of D.edges) {
      if (p[a] && p[b]) { g.beginPath(); g.moveTo(p[a][0], p[a][1]); g.lineTo(p[b][0], p[b][1]); g.stroke(); }
    }
    for (const q of p) { if (q) { g.beginPath(); g.arc(q[0], q[1], 6, 0, 6.2832); g.fill(); } }
  }
  note.textContent = "Frame " + f + " of " + D.frames.length + ". Mouse a: dark. Mouse b: orange.";
}
let last = -1;
function tick() {
  const v = video();
  if (v) { const t = v.currentTime; if (t !== last) { last = t; draw(t); } }
  else { note.textContent = "The video was not found on this page."; }
  requestAnimationFrame(tick);
}
draw(0); tick();
</script></body></html>
"""
