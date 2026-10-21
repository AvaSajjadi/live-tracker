"""Replay page: watch a recorded SYNTHETIC session with the approved rules (new file, D3-2).

Rendering only. What is shown comes from live_tracker.replay (causal frame
states, D3-1) through live_tracker.replay_view (labels, playback ticks,
skeleton image), so nothing appears before the replay position reached it.

State, all in st.session_state:
- the approved rules: the Rules page's workflow (rule_workflow), read only;
- the built replays: REPLAY_CACHE, keyed by replay_view.cache_identity: the
  score-bound run_id (session, rules, fps, exact score rows), the pose content
  and the configuration. The inputs are re-checked on every run before a
  cached replay can be reused, so changed scores, pose or rules rebuild (or
  fail with an error) and never show an older run's events or answers;
- per run (keys carry the run_id): the frame position, the end-of-session
  view, and playback (playing flag and replay_view.PlaybackAnchor).

Answers are written through ReplaySession.answer (journal + answer position);
a second, different answer is refused there.

The session summary (summary.py, from this run's journal only) is shown in
the end-of-session view only, never while an earlier frame is shown.
"""

import pandas as pd
import streamlit as st

from live_tracker import replay as rp
from live_tracker import replay_view as rv
from live_tracker import summary as sm
from live_tracker.rule_workflow import is_valid_workflow

WORKFLOW_KEY = "rule_workflow"
REPLAY_CACHE = "replay_cache"
SESSION_KEY = "replay_session"
SPEED_KEY = "replay_speed"
ACTIVE_RUN = "replay_active_run"
SPEEDS = {"1×": 1.0, "2×": 2.0, "5×": 5.0}
TICK_S = 0.25                                      # playback refresh while playing
WARNING_SOURCE = ("replay.tracking_warnings over Ava's features.tracking_warning_events, on this session's "
                  "SYNTHETIC pose (provisional flags and thresholds)")


def keys(run_id):
    # "frame" is the position (plain state, kept while another session is shown); "slider" is the widget
    # "anchor": replay_view.PlaybackAnchor while playing (None otherwise)
    return {k: f"replay_{k}_{run_id}" for k in ("frame", "slider", "end", "playing", "anchor")}


st.title("Replay")
st.warning(rp.REPLAY_BANNER)

# 1. approved rules (from the Rules page; never invented here)
wf = st.session_state.get(WORKFLOW_KEY)
if wf is None or not is_valid_workflow(wf) or not wf.ruleset.active():
    st.error("No approved rule: nothing can be monitored. Go to the Rules page, propose a rule and click Confirm.")
    st.stop()
rules = wf.ruleset.active()

# 2. session (SYNTHETIC only)
try:
    catalog = rv.synthetic_catalog()
except Exception as exc:                            # a broken fixture is an error, never a silent fallback
    st.error(f"The synthetic sessions could not be built ({type(exc).__name__}: {exc}).")
    st.stop()
labels = [e.label for e in catalog]
if not labels:
    st.error("No synthetic session is available.")
    st.stop()
label = st.selectbox("Session", labels, key=SESSION_KEY)
entry = catalog[labels.index(label)]
try:
    rv.check_entry(entry)
except rp.ReplayError as exc:
    st.error(f"This session cannot be replayed: {exc}")
    st.stop()
st.warning(rp.SYNTHETIC_NOTE)
st.caption(f"`{entry.session}`: {entry.description}")

pose_problem = rv.check_pose(entry)
warnings = ()
if pose_problem is not None:
    st.error(f"Skeletons and tracking warnings are unavailable for this session: {pose_problem}.")
else:
    try:
        warnings = rv.entry_warnings(entry)
    except (rp.ReplayError, KeyError, ValueError) as exc:
        st.error(f"Tracking warnings could not be computed: {exc}")

# 3. the replay run (built once per session, rules and storage directory)
directory = rv.replay_dir()
try:
    directory.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    st.error(f"The replay journal directory {directory} cannot be used ({exc.strerror}).")
    st.stop()
cache = st.session_state.get(REPLAY_CACHE)
if not isinstance(cache, dict):
    cache = st.session_state[REPLAY_CACHE] = {}
try:
    cache_key = rv.cache_identity(entry, rules, directory, pose_problem is None)
except rp.ReplayError as exc:
    st.error(f"The replay could not be built: {exc}")
    st.stop()
replay = cache.get(cache_key)
if not isinstance(replay, rp.ReplaySession):
    try:
        run_id, journal_path, answers_path = rp.run_paths(entry.scores, entry.session, rules, directory)
        with st.spinner("Running the approved rules over the recorded scores..."):
            replay = rp.build_replay(entry.scores, entry.session, rules, warnings=warnings,
                                     pose=entry.pose if pose_problem is None else None,
                                     journal_path=journal_path, answers_path=answers_path)
    except rp.ReplayError as exc:
        st.error(f"The replay could not be built: {exc}")
        st.stop()
    cache[cache_key] = replay
try:
    replay.refresh()                               # answers given in another tab since the last run
except rp.ReplayError as exc:
    st.error(f"The replay journal cannot be read: {exc}")
    st.stop()
run = replay.run_id
K = keys(run)
if st.session_state.get(ACTIVE_RUN) != run:        # switching sessions or rules: stop every other playback
    for k in [k for k in st.session_state if str(k).startswith("replay_playing_")]:
        st.session_state[k] = False
    for k in [k for k in st.session_state if str(k).startswith("replay_anchor_")]:
        st.session_state[k] = None
    st.session_state[ACTIVE_RUN] = run
st.session_state.setdefault(K["frame"], replay.frames[0])
st.session_state.setdefault(K["end"], False)
st.session_state.setdefault(K["playing"], False)

with st.expander(f"Approved rules in use ({len(replay.rules)})", expanded=False):
    st.dataframe(pd.DataFrame([{"rule": r.rule_id, "behavior": r.behavior, "min duration (s)": r.min_duration_s,
                                "ask threshold": r.ask_threshold, "alert threshold": r.alert_threshold,
                                "mode": r.alert_mode, "description": r.description} for r in replay.rules]),
                 hide_index=True, width="stretch")
    st.caption(f"Run `{run}`. The system only alerts, asks and logs; it never acts on the experiment.")

# 4. playback: the speed and the playing flag set the fragment's refresh, so changing them reruns the page
speed_label = st.radio("Playback speed (recorded time)", list(SPEEDS), key=SPEED_KEY, horizontal=True)
speed = SPEEDS[speed_label]
playing = st.session_state[K["playing"]]


def stop_playback():
    st.session_state[K["playing"]] = False
    st.session_state[K["anchor"]] = None


def seek(frame):
    st.session_state[K["frame"]] = frame
    st.session_state[K["end"]] = False
    stop_playback()


def step(rows):
    i = min(max(replay.index_of(st.session_state[K["frame"]]) + rows, 0), replay.n_frames - 1)
    seek(replay.frames[i])


def show_end():
    st.session_state[K["end"]] = True
    stop_playback()


def start_playing():
    """Play (idempotent: applying it twice only re-anchors at the same position)."""
    if st.session_state[K["playing"]]:
        return
    st.session_state[K["end"]] = False
    if replay.index_of(st.session_state[K["frame"]]) >= replay.n_frames - 1:
        st.session_state[K["frame"]] = replay.frames[0]            # play again from the start
    st.session_state[K["anchor"]] = rv.start_playback(replay, replay.index_of(st.session_state[K["frame"]]),
                                                      rv.now(), speed)
    st.session_state[K["playing"]] = True


# Play and Pause are two idempotent controls outside the auto-refreshing fragment, handled by callbacks. A
# single toggle button lost 3-4 of every 20 Pause clicks in a real browser (D4-2): a click that coincides with
# a playback tick can be applied twice, and a toggle applied twice keeps playing. A click here is a full rerun,
# which also sets the fragment's new refresh rate.
pc = st.columns([1, 1, 4])
pc[0].button("Play", key=f"replay_play_{run}", type="primary", on_click=start_playing, disabled=playing)
pc[1].button("Pause", key=f"replay_pause_{run}", on_click=stop_playback, disabled=not playing)


@st.fragment(run_every=TICK_S if playing else None)
def view():
    anchor = st.session_state.get(K["anchor"])
    if st.session_state[K["playing"]] and isinstance(anchor, rv.PlaybackAnchor):   # one tick at recorded speed
        index, anchor, still = rv.playback_position(replay, anchor, rv.now(), speed)
        st.session_state[K["frame"]] = replay.frames[index]
        st.session_state[K["anchor"]] = anchor
        if not still:
            stop_playback()
            st.rerun()                                 # the last row: stop the refresh timer
    one_s = max(1, round(1.0 / (replay.times[1] - replay.times[0]))) if replay.n_frames > 1 else 1
    c = st.columns(6)
    c[0].button("⏮ Start", key=f"replay_start_{run}", on_click=seek, args=(replay.frames[0],))
    c[1].button("−1 s", key=f"replay_back1s_{run}", on_click=step, args=(-one_s,))
    c[2].button("−1 frame", key=f"replay_back1_{run}", on_click=step, args=(-1,))
    c[3].button("+1 frame", key=f"replay_fwd1_{run}", on_click=step, args=(1,))
    c[4].button("+1 s", key=f"replay_fwd1s_{run}", on_click=step, args=(one_s,))
    c[5].button("End of session", key=f"replay_endbtn_{run}", on_click=show_end)
    st.session_state[K["slider"]] = st.session_state[K["frame"]]
    st.slider("Frame", replay.frames[0], replay.frames[-1], key=K["slider"],
              on_change=lambda: (st.session_state.update({K["frame"]: st.session_state[K["slider"]],
                                                          K["end"]: False}), stop_playback()))
    at_end = st.session_state[K["end"]]
    state = replay.end_state() if at_end else replay.state_at(st.session_state[K["frame"]])
    render(state)


def render(state):
    if state.at_end:
        st.info("End of session: everything known after the recording ended, including bouts and tracking "
                "warnings that were still going. Seek to a frame to go back into the replay.")
        m = st.columns(3)
        m[0].metric("Frame", "end")
        m[1].metric("Time", rv.fmt_s(replay.times[-1]) + " (last frame)")
        m[2].metric("Tracking", "—")
    else:
        m = st.columns(3)
        m[0].metric("Frame", f"{state.frame} ({state.index + 1}/{replay.n_frames})")
        m[1].metric("Time", rv.fmt_s(state.time_s))
        m[2].metric("Tracking", "ok" if state.tracking_ok else "LOST")
        if state.tracking_ok is False:
            st.warning(rp.TRACKING_GAP_NOTE)
        bad = rv.tracking_summary(state)
        if bad:
            st.warning("Unreliable tracking on this frame: " + "; ".join(
                f"mouse {m_}: " + ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in sorted(c.items()))
                for m_, c in sorted(bad.items())) + ". These keypoints are not drawn as tracked.")

    left, right = st.columns([3, 2])
    with left:
        img = None if state.at_end else rv.render_skeletons(state)
        if img is not None:
            st.image(img, caption="Skeletons on this frame (image pixels, scaled)")
            st.caption(rv.LEGEND)
        elif not state.at_end:
            st.caption("No skeleton for this frame (no usable pose).")
    with right:
        st.markdown("**Current scores** (classifier output on this frame; SYNTHETIC)")
        if state.scores:
            st.dataframe(pd.DataFrame([{"behavior": b, "score": "missing" if v is None else round(v, 3)}
                                       for b, v in sorted(state.scores.items())]), hide_index=True, width="stretch")
        else:
            st.caption("No frame played yet." if not state.at_end else "—")

    for e in state.events:                          # announcements: only on the frame an event fires
        if e["event_id"] in state.new_event_ids:
            text = (f"{e['action'].upper()} {e['event_id']}: {e['behavior']} (rule {e['rule_id']}), confidence "
                    f"{e['confidence']:.2f}, since {rv.fmt_s(e['start_s'])}")
            {"alert": st.error, "ask": st.warning}.get(e["action"], st.info)(text)

    st.subheader("Events so far")
    rows = rv.timeline_rows(state)
    if rows:
        st.dataframe(pd.DataFrame(rows).drop(columns=["event_id"]), hide_index=True, width="stretch")
        st.caption("Onset: first frame of the bout; fired: when the rule's minimum duration was reached; "
                   "closure: when the bout ended (known only after it ended).")
    else:
        st.caption("No event has fired yet.")

    asks = [e for e in state.events if e["action"] == "ask"]
    if asks:
        st.subheader("Questions for you")
        for e in asks:
            code, text = rv.answer_state(replay, state, e)
            st.markdown(f"**{e['event_id']}** ({e['behavior']}, confidence {e['confidence']:.2f}): {text}")
            if code == "unanswered" and not state.at_end:
                a, b = st.columns(2)
                for col, choice, label in ((a, "confirmed", "Confirm"), (b, "rejected", "Reject")):
                    if col.button(label, key=f"replay_answer_{run}_{e['event_id']}_{choice}"):
                        try:
                            replay.answer(e["event_id"], choice, at_frame=state.frame)
                        except rp.ReplayError as exc:
                            st.error(f"Not recorded: {exc}")
                        else:
                            st.rerun()
            elif code == "unanswered":
                st.caption("Seek to a frame to answer.")

    st.subheader("Tracking warnings")
    st.caption("Kept separate from the event log until Ava confirms how tracking warnings are stored.")
    wrows = rv.warning_rows(state)
    if wrows:
        st.dataframe(pd.DataFrame(wrows), hide_index=True, width="stretch")
    else:
        st.caption("No tracking warning so far.")

    if state.at_end:
        render_summary()


def render_summary():
    st.subheader("Session summary")
    st.caption("Written only from this run's event log (and the separately labelled tracking-warning source). "
               "No new detection, statistics or LLM wording.")
    source = WARNING_SOURCE if pose_problem is None else None
    try:
        summary = sm.summary_from_replay(replay, warning_source=source)
    except sm.SummaryError as exc:
        st.error(f"The session summary cannot be written: {exc}")
        return
    md = summary.to_markdown()
    with st.container(border=True):
        st.markdown(md)
    slug = entry.session.replace("/", "_")
    d1, d2 = st.columns(2)
    d1.download_button("Download summary (Markdown)", md, file_name=f"{slug}_summary.md", mime="text/markdown",
                       key=f"replay_summary_md_{run}")
    d2.download_button("Download summary (JSON)", summary.to_json(), file_name=f"{slug}_summary.json",
                       mime="application/json", key=f"replay_summary_json_{run}")


view()
