"""Motion engine on one real CalMS21 validation session (new file).

Shows what the motion engine measures on real tracked poses: tracking quality
and warnings, the causal feature table, and the keypoint features. These are
MEASUREMENTS, not behavior labels and not model output. The annotation strip is
CalMS21's own ground truth, shown only as context and never used by a feature.

Only the hash-verified validation cache is read (see live_tracker.session_view);
a test session cannot be opened from here.
"""

import pandas as pd
import streamlit as st

from live_tracker import session_view as sv
from live_tracker.calms21 import TORSO_KEYPOINTS
from live_tracker.features import KEYPOINT_COLS

st.set_page_config(page_title="Real session | Motion Engine", layout="wide")
st.title("Motion engine | real validation session")
st.caption("Real CalMS21 pose data (validation split only). Measurements of motion and tracking; "
           "no behavior is claimed. Settings are the team's quality settings, not tuned for this page.")


@st.cache_resource(show_spinner="Verifying the validation cache (hashes)...")
def _session_ids():
    return sv.list_sessions()


@st.cache_data(show_spinner="Computing features for this session...", max_entries=4)
def _view(session):
    return sv.load_view(session)


try:
    ids = _session_ids()
except sv.SessionUnavailable as exc:
    st.info("The verified validation cache is not available, so there is no real session to show.")
    st.code(sv.BUILD_COMMAND, language="bash")
    st.caption(f"Reason: {exc}")
    st.stop()

session = st.selectbox("Validation session", ids, format_func=sv.short_name)
try:
    view = _view(session)
except sv.SessionUnavailable as exc:
    st.error(f"This session cannot be shown: {exc}")
    st.stop()

cols = st.columns(4)
cols[0].metric("Frames", f"{view.n_frames:,}")
cols[1].metric("Duration", f"{view.duration_s / 60:.1f} min")
cols[2].metric("tracking_ok frames", f"{view.tracking_ok_share:.1%}")
cols[3].metric("Tracking warnings", len(view.warnings))

# --- tracking quality ----------------------------------------------------------------
st.subheader("Tracking quality")
st.caption(f"Torso keypoints only ({', '.join(TORSO_KEYPOINTS)}). A warning means the tracker, "
           "not the mouse: a frozen track is not a mouse standing still.")
share = pd.DataFrame({"flag": list(view.flag_share), "share of torso keypoint-frames":
                      [f"{v:.2%}" for v in view.flag_share.values()]})
st.dataframe(share, hide_index=True, width="stretch")
if view.warnings:
    st.dataframe(pd.DataFrame([{"warning": w["evidence"]["flag"], "mouse": w["evidence"]["mouse"],
                                "from (s)": round(w["start_s"], 2), "to (s)": round(w["end_s"], 2),
                                "duration (s)": round(w["evidence"]["duration_s"], 2)}
                               for w in view.warnings]), hide_index=True, width="stretch")
else:
    st.success("No tracking loss, frozen or out-of-frame run longer than 0.5 s.")
runs = sv.tracking_runs(view.table)
st.caption(f"{len(runs)} stretch(es) where tracking_ok is false (a needed keypoint of either mouse unusable); "
           f"longest {runs['duration_s'].max():.2f} s." if len(runs) else
           "tracking_ok is true on every frame.")

# --- features ---------------------------------------------------------------------------
st.subheader("Features (causal)")
st.caption("Each value uses only the current and past frames. NaN means not measurable, never zero.")
options = ["dist_center", "dist_rate", "rel_speed", "close_time_s", "speed_corr_w",
           *[f"{c}_{m}" for c in ("speed", "accel", "turn_rate") for m in ("a", "b")]]
options = [c for c in options if c in view.table.columns]
chosen = st.multiselect("Features", options, default=["dist_center", "speed_a", "speed_b"])
t0, t1 = st.slider("Time range (s)", 0.0, float(view.duration_s), (0.0, min(60.0, float(view.duration_s))))
window = view.table[view.table["time_s"].between(t0, t1)]
if chosen:
    st.line_chart(window.set_index("time_s")[chosen])
with st.expander("Keypoint features (nose / tail base distances, facing angle)"):
    st.caption("facing_ab is the angle in radians between mouse a's body axis (tail base to nose) and the "
               "direction to mouse b: 0 = pointing at it, pi = pointing away. Geometry only.")
    kp = view.keypoints[view.keypoints["time_s"].between(t0, t1)].set_index("time_s")
    st.line_chart(kp[list(KEYPOINT_COLS[:3])])
    st.line_chart(kp[list(KEYPOINT_COLS[3:])])
with st.expander("Feature table (first 200 rows)"):
    st.dataframe(view.table.head(200), width="stretch")

# --- context: ground-truth annotation -------------------------------------------------------
st.subheader("Annotation (ground truth, context only)")
st.caption("CalMS21's own labels for this session. No feature or warning above uses them. "
           "Not a model prediction.")
bouts = sv.label_bouts(view.labels)
st.dataframe(bouts.drop(columns="session").round(2), hide_index=True, width="stretch")
