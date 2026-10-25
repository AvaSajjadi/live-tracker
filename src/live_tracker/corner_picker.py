"""A live camera picture with 4 corner handles the researcher drags with the mouse (new file).

Streamlit has no widget for this, so it is a small inline custom component (st.components.v2). The page sends a new
JPEG picture about 10 times a second; the component draws it on a canvas, draws the screen outline on top, and lets the
researcher drag each corner. When a corner is let go, the 4 corners go back to Python as fractions of the picture (0 to 1).

Order of the corners: top-left, top-right, bottom-right, bottom-left (see screen_map.py).
"""

from __future__ import annotations

import base64

import streamlit as st

HTML = """<canvas></canvas>"""

CSS = """
canvas { width: 100%; display: block; border-radius: 8px; touch-action: none; cursor: crosshair; background: #111; }
"""

JS = """
const STATE = new WeakMap()
const clamp = (v) => Math.min(1, Math.max(0, v))
const GRAB_PX = 40

export default function (component) {
  const { data, parentElement, setStateValue } = component
  const canvas = parentElement.querySelector("canvas")
  if (!canvas) return

  let s = STATE.get(parentElement)
  const first = !s
  if (!s) {
    s = { corners: null, version: null, dragging: -1, img: null, frame: null }
    STATE.set(parentElement, s)
  }
  s.send = (corners) => setStateValue("corners", corners.map((p) => [p[0], p[1]]))

  // the corners are kept here while dragging; Python only replaces them when its version number changes (a reset)
  if (s.corners === null || s.version !== data.version) {
    s.corners = data.corners.map((p) => [p[0], p[1]])
    s.version = data.version
    if (!first) s.send(s.corners)
  }

  if (canvas.width !== data.width || canvas.height !== data.height) {
    canvas.width = data.width
    canvas.height = data.height
  }

  s.draw = () => {
    const ctx = canvas.getContext("2d")
    const w = canvas.width, h = canvas.height
    if (s.img) ctx.drawImage(s.img, 0, 0, w, h)
    else { ctx.fillStyle = "#111"; ctx.fillRect(0, 0, w, h) }
    const pts = s.corners.map((p) => [p[0] * w, p[1] * h])
    ctx.lineWidth = Math.max(2, w / 300)
    ctx.strokeStyle = "#00ff00"
    ctx.beginPath()
    pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])))
    ctx.closePath()
    ctx.stroke()
    const r = Math.max(7, w / 70)
    ctx.font = "bold " + Math.round(r * 2) + "px sans-serif"
    pts.forEach((p, i) => {
      ctx.beginPath()
      ctx.arc(p[0], p[1], r, 0, 2 * Math.PI)
      ctx.fillStyle = i === s.dragging ? "#ffcc00" : "#ff2d2d"
      ctx.fill()
      ctx.lineWidth = 2
      ctx.strokeStyle = "#ffffff"
      ctx.stroke()
      ctx.fillStyle = "#ffff66"
      ctx.fillText(String(i + 1), p[0] + r + 3, p[1] - r - 3)
    })
  }

  if (data.frame && s.frame !== data.frame) {
    s.frame = data.frame
    const img = new Image()
    img.onload = () => { s.img = img; s.draw() }
    img.src = "data:image/jpeg;base64," + data.frame
  }
  s.draw()

  const where = (e) => {
    const r = canvas.getBoundingClientRect()
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, r]
  }
  canvas.onpointerdown = (e) => {
    const [x, y, r] = where(e)
    let best = -1, bestD = Infinity
    s.corners.forEach((c, i) => {
      const d = Math.hypot((c[0] - x) * r.width, (c[1] - y) * r.height)
      if (d < bestD) { bestD = d; best = i }
    })
    if (bestD <= GRAB_PX) {
      s.dragging = best
      canvas.setPointerCapture(e.pointerId)
      canvas.style.cursor = "grabbing"
      e.preventDefault()
    }
  }
  canvas.onpointermove = (e) => {
    if (s.dragging < 0) return
    const [x, y] = where(e)
    s.corners[s.dragging] = [clamp(x), clamp(y)]
    s.draw()
  }
  const end = () => {
    if (s.dragging < 0) return
    s.dragging = -1
    canvas.style.cursor = "crosshair"
    s.draw()
    s.send(s.corners)
  }
  canvas.onpointerup = end
  canvas.onpointercancel = end
}
"""

_COMPONENT = st.components.v2.component("live_tracker_corner_picker", html=HTML, css=CSS, js=JS)


def jpeg_base64(frame, quality=70):
    """The picture (an OpenCV BGR image) as base64 JPEG text for the component."""
    import cv2
    done, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not done:
        raise ValueError("the picture could not be encoded")
    return base64.b64encode(jpg.tobytes()).decode("ascii")


def corner_picker(frame, corners, *, key, version=0):
    """Show `frame` with draggable corners. `corners` are the starting corners (and the ones a reset puts back when
    `version` changes). The corners the researcher chose are in current_corners(key)."""
    h, w = frame.shape[:2]
    return _COMPONENT(key=key, width="stretch", on_corners_change=lambda: None,
                      data={"frame": jpeg_base64(frame), "width": int(w), "height": int(h),
                            "corners": [[float(x), float(y)] for x, y in corners], "version": int(version)})


def current_corners(key, default):
    """The corners the researcher dragged to (a list of 4 pairs), or `default` before any drag."""
    state = st.session_state.get(key)
    try:
        corners = state["corners"] if state is not None else None
    except (KeyError, TypeError):
        corners = None
    return corners if corners else [list(p) for p in default]
