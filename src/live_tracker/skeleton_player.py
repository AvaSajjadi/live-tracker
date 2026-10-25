"""Recorded-pose display, not pose inference or a scientific result.

One component owns its video, controls and canvas. Presented-frame PTS selects
poses; overlay copies the video pixels and draws poses in the SAME canvas update.
Browsers without frame callbacks get an explicitly approximate side-by-side view.
Assumption: zero-origin constant-rate recording, pose frame n at n / fps. Pairing
and timing cannot be authenticated from a user-selected filename alone.
"""
from __future__ import annotations

import json
import math
import mimetypes

import pandas as pd

from .calms21 import FPS, IMAGE_SIZE, KEYPOINTS, MICE
from .skeleton import PROVISIONAL_EDGES, mouse_skeleton
from .quality import SUGGESTED_MIN_SCORE

COLOURS = {"a": "#27c9ed", "b": "#ff9d28"}
DISCLAIMER = "PROVISIONAL connections: inferred from keypoint names, not confirmed official CalMS21 anatomy."


def pose_frames(poses_path, session=None, *, require_observed=False):
    """Keep recorded positions/quality, including gaps. Never smooth or interpolate.

    A supplied session must match the entire file; mixed-session sources and
    ambiguous duplicate points are refused rather than merged into one animal.
    """
    df = pd.read_parquet(poses_path)
    if df.empty:
        raise ValueError("The pose file has no frames")
    if "session" in df:
        sessions = df["session"].dropna().unique()
        if len(sessions) != 1 or df["session"].isna().any() or (session is not None and sessions[0] != session):
            raise ValueError("Pose session does not match the selected source")
    elif session is not None:
        raise ValueError("Pose session identity is absent")
    if not set(df["mouse"]) <= set(MICE) or not set(df["keypoint"]) <= set(KEYPOINTS):
        raise ValueError("Unexpected mouse identity or keypoint name")
    f = pd.to_numeric(df["frame"], errors="coerce")
    if (~f.map(lambda v: math.isfinite(v) and 0 <= v < 2_000_000 and v == int(v))).any():
        raise ValueError("Pose frames must be bounded nonnegative integers")
    df["frame"] = f.astype(int)
    for coordinate in ("x", "y"):
        df[coordinate] = pd.to_numeric(df[coordinate], errors="coerce")
    if df.duplicated(["frame", "mouse", "keypoint"]).any():
        raise ValueError("Duplicate pose point on one frame")
    frames = [{} for _ in range(int(f.max()) + 1)]
    for (frame, mouse), group in df.groupby(["frame", "mouse"], sort=False):
        rows = {}
        for r in group.to_dict("records"):
            # Upload normalization historically substitutes 1 for absent scores.
            # New uploads retain a display-only provenance bit; backend scores
            # and monitoring calculations retain their previous values.
            if require_observed and "score_observed" not in r:
                r["score"] = None
            if "score_observed" in r and not pd.notna(r["score_observed"]):
                r["score"] = None
            elif "score_observed" in r and r["score_observed"] not in (True, 1):
                r["score"] = None
            x, y = r.get("x"), r.get("y")
            if pd.notna(x) and pd.notna(y) and (not 0 <= x < IMAGE_SIZE[0] or not 0 <= y < IMAGE_SIZE[1]):
                r["out_of_frame"] = True
            rows[r["keypoint"]] = r
        frame_ok = not ("tracking_ok" in group and group["tracking_ok"].eq(False).any())
        sk = mouse_skeleton(mouse, rows, frame_tracking_ok=frame_ok)
        frames[int(frame)][mouse] = [dict(x=p.x, y=p.y, status=p.status, reason=p.reason) for p in sk.keypoints]
    return frames


def payload(frames, fps=FPS, **source):
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video/pose fps must be finite and positive")
    return dict(frames=frames, fps=float(fps), size=list(IMAGE_SIZE), keypoints=list(KEYPOINTS),
                edges=[[KEYPOINTS.index(a), KEYPOINTS.index(b)] for a, b in PROVISIONAL_EDGES],
                colours=COLOURS, disclaimer=DISCLAIMER, min_score=SUGGESTED_MIN_SCORE, **source)


def html(frames, fps=FPS, video_url="", **source):
    """Standalone harness using the EXACT component JS (synthetic browser tests)."""
    data = json.dumps(payload(frames, fps, video_url=video_url, **source), allow_nan=False).replace("<", chr(92) + "u003c")
    js = JS.replace("export default function", "function mount", 1)
    return '<!doctype html><meta charset="utf-8"><style>' + CSS + '</style><div id="root">' + HTML + \
           '</div><script>' + js + '\nmount({parentElement:document.getElementById("root"),data:' + data + '});</script>'


def render(video_path, frames, fps=FPS, *, session, poses_path, original_video, key="replay_player"):
    """Serve through Streamlit's session-owned media manager (as st.video does).

    This one internal API is isolated here; no custom server, filesystem URL or
    path-based browser access is needed. Streamlit serves only registered media.
    """
    import streamlit as st
    from streamlit.runtime import get_instance
    global _component
    if _component is None:
        _component = st.components.v2.component("recorded_skeleton_player", html=HTML, css=CSS, js=JS)
    mime = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    url = get_instance().media_file_mgr.add(str(video_path), mime, f"skeleton_player.{key}")
    return _component(data=payload(frames, fps, video_url=url, session=session, poses_path=str(poses_path),
                                   original_video=str(original_video), playback_video=str(video_path)),
                      key=key, height="content", width="stretch")


_component = None
HTML = """
<section aria-label="Recorded video and poses">
<p class="provisional"></p>
<label>Presentation <select class="mode"><option value="side">Side by side</option><option value="overlay">Overlay (same canvas frame)</option></select></label>
<div class="views"><video muted playsinline preload="auto" aria-label="Recorded video"></video><canvas aria-label="Recorded skeleton"></canvas></div>
<div class="controls"><button class="play">Play</button><button class="prev">Previous frame</button><button class="next">Next frame</button>
<label>Playback speed <select class="rate"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select></label></div>
<label class="seek-label">Seek recording <input class="seek" type="range" min="0" value="0" step="0.001"></label>
<p class="note" role="status" aria-live="off">Waiting for a presented video frame; no pose shown.</p>
<p class="quality"></p><p class="legend">Mouse a (resident): cyan; mouse b (intruder): orange. Solid: recorded score meets threshold and supplied quality checks. Hollow: low confidence or unreliable/unknown confidence. Cross: explicitly imputed (not measured). Missing: not drawn. Only tracked points are connected. No display interpolation.</p>
<p class="timing">Assumes matching zero-origin constant-rate video and poses. Side-by-side callbacks can lag display by a refresh; overlay draws copied video pixels and poses together. This does not independently prove source pairing.</p>
<details><summary>Video and pose provenance</summary><pre class="provenance"></pre></details>
</section>
"""
CSS = """
:host,body {font-family:system-ui,sans-serif;color:var(--st-text-color,#222);margin:0}
section {width:100%;box-sizing:border-box} p {line-height:1.45;overflow-wrap:anywhere}
.provisional {font-weight:600} .views {display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
video,canvas {width:100%;height:auto;min-width:0;aspect-ratio:1024/570;object-fit:contain;background:#161616;border-radius:4px}
.views.overlay {grid-template-columns:1fr} .views.overlay video {display:none}
.controls {display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button,select {font:inherit;padding:6px;cursor:pointer} .seek-label {display:block;margin-top:10px}
.seek {width:100%} .legend,.timing,.quality,pre {font-size:13px} pre {white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:600px){.views{grid-template-columns:1fr}}
"""
JS = r"""
export default function(component) {
  const root = component.parentElement, D = component.data;
  if (root._replayCleanup) root._replayCleanup();
  const q = s => root.querySelector(s), v=q('video'), cv=q('canvas'), g=cv.getContext('2d');
  const note=q('.note'), quality=q('.quality'), seek=q('.seek'), mode=q('.mode');
  let stopped=false, callback=null, raf=null, lastPTS=null, lastFrame=null;
  const precise=typeof v.requestVideoFrameCallback==='function';
  cv.width=D.size[0]; cv.height=D.size[1];
  q('.provisional').textContent=D.disclaimer;
  q('.legend').textContent += ' Recorded score threshold: '+D.min_score+'. No additional tracking checks are computed here. Legacy normalized uploads without score provenance are shown as unknown; their original confidence cannot be recovered.';
  q('.provenance').textContent='Session: '+(D.session || 'not supplied')+'\nOriginal video: '+(D.original_video || D.video_url)+
    '\nPlayback video: '+(D.playback_video || D.video_url)+'\nPoses: '+(D.poses_path || 'synthetic fixture')+
    '\nPose frame rate: '+D.fps+' fps; coordinates: '+D.size.join(' × ')+' px, x right, y down.'+
    '\nSeven keypoints per mouse: '+D.keypoints.join(', ');
  if (!precise) {mode.querySelector('[value=overlay]').disabled=true; q('.timing').textContent=
    'APPROXIMATE synchronization: this browser has no presented-frame callback. Uses floor(currentTime × fps), not a decoded-frame identity. Overlay is disabled. No interpolation.';}
  const clear = message => {g.clearRect(0,0,cv.width,cv.height); note.textContent=message; quality.textContent='';
    delete cv.dataset.frame; lastFrame=null;};
  function draw(t, exact) {
    if (v.readyState<2 || (!exact && v.seeking)) {clear('Seeking/loading: no pose shown until a video frame is available.'); return;}
    const n=t*D.fps;
    // Presented PTS is a frame START, with container timebase quantization.
    // currentTime is a playhead anywhere inside a frame: it must NOT be rounded.
    const f=exact ? Math.round(n) : (v.ended ? Math.max(0,Math.ceil(v.duration*D.fps)-1) : Math.floor(n+1e-7));
    g.clearRect(0,0,cv.width,cv.height);
    const overlay=mode.value==='overlay';
    if (overlay) g.drawImage(v,0,0,cv.width,cv.height);
    const badSize=v.videoWidth!==D.size[0] || v.videoHeight!==D.size[1];
    if (badSize || (exact && Math.abs(n-f)>0.02)) {
      clear(badSize ? 'Coordinate mismatch: video dimensions differ from the pose image (1024 × 570); poses withheld.' :
        'Timing mismatch: presented timestamp is not on the declared frame grid; poses withheld.'); return;
    }
    const mice=D.frames[f];
    const counts={tracked:0,low_confidence:0,unreliable:0,imputed:0,missing:0};
    if (mice) for (const m of ['a','b']) {
      const points=mice[m] || Array(7).fill(null);
      g.strokeStyle=D.colours[m]; g.fillStyle=D.colours[m]; g.lineWidth=3;
      for (const [a,b] of D.edges) {
        if(points[a]?.status==='tracked' && points[b]?.status==='tracked') {
          g.beginPath();g.moveTo(points[a].x,points[a].y);g.lineTo(points[b].x,points[b].y);g.stroke();
        }
      }
      for (const p of points) {
        counts[p?.status || 'missing']++;
        if (!p || p.status==='missing' || !Number.isFinite(p.x) || !Number.isFinite(p.y) ||
            p.x<0 || p.y<0 || p.x>=cv.width || p.y>=cv.height) continue;
        g.beginPath();
        if(p.status==='imputed') {g.moveTo(p.x-6,p.y-6);g.lineTo(p.x+6,p.y+6);g.moveTo(p.x+6,p.y-6);g.lineTo(p.x-6,p.y+6);g.stroke();}
        else {g.arc(p.x,p.y,6,0,Math.PI*2);p.status==='tracked' ? g.fill() : g.stroke();}
      }
    }
    cv.dataset.frame=String(f);cv.dataset.timestamp=String(t);cv.dataset.clock=exact?'presented':'approximate';lastFrame=f;
    note.textContent=(exact?'Presented':'APPROXIMATE')+' frame '+f+' (zero-based), '+t.toFixed(6)+' s. '+
      (!mice || !Object.keys(mice).length ? 'No pose recorded for this frame; no position substituted.' : 'Recorded poses only.');
    quality.textContent=Object.entries(counts).map(([k,n])=>k+': '+n).join(' · ')+
      (Number.isFinite(v.duration) && Math.abs(v.duration*D.fps-D.frames.length)>1.1 ? ' · Video/pose lengths differ; no last-pose clamping.' : '');
    seek.value=String(v.currentTime); q('.play').textContent=v.paused?'Play':'Pause';
  }
  function presented(now, meta) {
    if(stopped)return;
    lastPTS=meta.mediaTime; draw(lastPTS,true);
    callback=v.requestVideoFrameCallback(presented);
  }
  function fallback() {if(stopped)return;draw(v.currentTime,false);raf=requestAnimationFrame(fallback);}
  v.onseeking=()=>{clear('Seeking: waiting for a newly presented frame; no old pose retained.');};
  v.onloadedmetadata=()=>{seek.max=String(v.duration);};
  // Do not redraw using currentTime on pause/seeked: it can refer to the NEXT
  // frame while the compositor still holds the previous one. rVFC owns identity.
  // Some decoders deliver PTS before readyState reaches HAVE_CURRENT_DATA.
  // A callback can precede seeking too. Retain it, but reuse only within
  // that presented frame's interval after decoding; never round the playhead.
  const settled=()=>{if(precise && lastPTS!==null && v.currentTime>=lastPTS-1e-7 && v.currentTime<lastPTS+1/D.fps-1e-7)draw(lastPTS,true);};
  v.onseeked=settled;v.onloadeddata=settled;
  v.onpause=()=>{q('.play').textContent='Play';};
  v.onplay=()=>{q('.play').textContent='Pause';};
  v.onerror=()=>clear('Video could not be decoded. No synchronized pose is claimed.');
  v.onended=()=>{q('.play').textContent='Play';if(!precise)draw(v.currentTime,false);};
  q('.play').onclick=()=>{if(v.paused)v.play().catch(()=>clear('Playback refused by browser; use Play again.'));else v.pause();};
  function step(delta) {if(!Number.isFinite(v.duration) || v.readyState<2)return;v.pause();const base=lastFrame===null?Math.floor(v.currentTime*D.fps):lastFrame;
    const max=Math.max(0,Math.ceil(v.duration*D.fps)-1);const target=Math.min(max,Math.max(0,base+delta));if(target===lastFrame)return;v.currentTime=(target+0.5)/D.fps;}
  q('.prev').onclick=()=>step(-1);q('.next').onclick=()=>step(1);
  seek.oninput=()=>{v.currentTime=Number(seek.value);};
  q('.rate').onchange=e=>{v.playbackRate=Number(e.target.value);};
  mode.onchange=()=>{q('.views').classList.toggle('overlay',mode.value==='overlay');
    if(lastPTS!==null)draw(lastPTS,precise);else if(!precise)draw(v.currentTime,false);};
  if(precise)callback=v.requestVideoFrameCallback(presented);else raf=requestAnimationFrame(fallback);
  v.src=D.video_url;
  const cleanup=()=>{stopped=true;v.pause();if(callback!==null && precise)v.cancelVideoFrameCallback(callback);
    if(raf!==null)cancelAnimationFrame(raf);v.removeAttribute('src');v.load();};
  root._replayCleanup=cleanup;
  return cleanup;
}
"""
