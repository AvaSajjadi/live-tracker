"""First milestone: visualize pose-derived features from an exported CSV.

This does not yet classify behavior or call an LLM.

Starter file, modified by us:
  - added a built-in synthetic demo option (clearly labeled, not CalMS21)
  - added a "Tracking quality" section with warnings (quality.py)
  - added a "Causal smoothing" section (filters.py) next to the starter's
    offline Savitzky-Golay features
  - added an option to inject fake tracking failures, to check the warnings
The original upload + motion features part is unchanged.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from live_tracker.filters import smooth_dataframe
from live_tracker.formats import inject_tracking_failures, wide_to_long
from live_tracker.motion import frame_features
from live_tracker.params import FROZEN_MIN_FRAMES, MAX_SPEED_PX_S, ONE_EURO
from live_tracker.quality import flag_tracking, tracking_loss_events, window_quality
from live_tracker.ui_style import apply as apply_style

DEMO_CSV = Path(__file__).resolve().parents[1] / 'example_pose_demo.csv'

st.set_page_config(page_title='Live Tracker', layout='wide')
apply_style()
st.title('Live Tracker | Motion Engine')
fps = st.number_input(
    'Camera frame rate (frames per second)', min_value=1.0, value=30.0,
    help='Frames per second (fps): how many pictures of the recording there are in each second. '
         'The page uses it to turn row numbers into seconds, so speeds and times are only right if '
         'this number is right. Check your camera or recording settings. 30 is common.')

# --- added: data source choice -------------------------------------------
use_demo = st.checkbox('Try demo data', help='A small made-up recording, so you can try the page without a file.')
upload = None if use_demo else st.file_uploader(
    'Pose CSV (one row per frame)', type='csv',
    help='Four columns, in pixels: x_a, y_a = position of mouse a; x_b, y_b = position of mouse b.')
if not use_demo:
    st.caption('Columns: x_a, y_a (mouse a) and x_b, y_b (mouse b), in pixels.')

# --- fixed values: the engine does the quality check and the smoothing by itself ----------------
# (chosen on CalMS21 train data, see live_tracker/params.py; the window length is a placeholder)
frozen_min_frames, max_speed, window_s = FROZEN_MIN_FRAMES, MAX_SPEED_PX_S, 1.0
min_cutoff, beta = ONE_EURO['min_cutoff'], ONE_EURO['beta']

# --- the few settings a user may want to change -------------------------------------------------
with st.expander('Advanced settings'):
    st.caption('These settings change only this page. The model and the replay use fixed settings.')
    min_quality = st.slider('Warn if window quality below', 0.0, 1.0, 0.8)
    min_loss_s = st.number_input('Report tracking loss longer than (s)', min_value=0.0, value=0.5)
    inject = st.checkbox('Inject fake tracking failures (for testing the warnings)')

data = None
if use_demo:
    data = pd.read_csv(DEMO_CSV)
    st.warning('SYNTHETIC TEST DATA: generated for software testing. '
               'It is not CalMS21 and not real mouse behavior.')
elif upload:
    data = pd.read_csv(upload)

if data is not None:
    required = ['x_a', 'y_a', 'x_b', 'y_b']
    if not set(required).issubset(data.columns):
        st.error('Missing columns: ' + ', '.join(sorted(set(required) - set(data.columns))))
    else:
        if inject:
            data = inject_tracking_failures(data[required], fps=fps)
            st.info('Fake tracking failures injected: mouse b missing for 1.5 s, '
                    'mouse a frozen for 1 s.')

        # --- starter section (unchanged) ----------------------------------
        # OFFLINE ONLY: this section uses Savitzky-Golay smoothing, which looks at future frames.
        # Fine for offline analysis, never for live monitoring or replay (the causal sections are live-safe).
        st.subheader('Motion features (OFFLINE ONLY)')
        f = frame_features(data[['x_a', 'y_a']].to_numpy(),
                           data[['x_b', 'y_b']].to_numpy(), fps=fps)
        st.metric('Frames', len(f))
        st.metric('Both tracks usable', f'{f.tracking_ok.mean():.1%}')
        st.line_chart(f[['speed_a_px_s', 'speed_b_px_s', 'distance_px']])
        st.dataframe(f.head(50), use_container_width=True)
        st.download_button('Download motion features CSV', f.to_csv(index=False),
                           file_name='motion_features.csv', mime='text/csv')

        # --- added: tracking quality --------------------------------------
        st.subheader('Tracking quality')
        long = wide_to_long(data, fps=fps, session='demo' if use_demo else 'upload')
        flags = flag_tracking(long, frozen_min_frames=int(frozen_min_frames),
                              max_speed_px_s=max_speed)
        losses = tracking_loss_events(flags, min_duration_s=min_loss_s)
        frozen = tracking_loss_events(flags, min_duration_s=min_loss_s, flag='frozen')
        for ev in losses:
            st.error(f"Tracking lost for mouse {ev['mouse']}: "
                     f"{ev['start_s']:.2f} s to {ev['end_s']:.2f} s")
        for ev in frozen:
            st.warning(f"Tracker may be stuck for mouse {ev['mouse']}: "
                       f"{ev['start_s']:.2f} s to {ev['end_s']:.2f} s "
                       '(identical coordinates; this is not the same as the mouse being still)')
        if not losses and not frozen:
            st.success('No tracking loss or frozen tracks found with the current settings.')

        cols = st.columns(4)
        for col, name in zip(cols, ['missing', 'frozen', 'jump', 'valid']):
            col.metric(f'{name} (all rows)', f'{flags[name].mean():.1%}')
        wq = window_quality(flags, window_s=window_s, min_quality=min_quality)
        st.write(f"Windows with a quality warning: {int(wq['warning'].sum())} of {len(wq)}")
        st.dataframe(wq, use_container_width=True)

        # --- added: causal smoothing ----------------------------------------
        st.subheader('Causal smoothing (live-safe)')
        st.caption('One-Euro filter: only uses the current and past frames. '
                   'Resets after long gaps; missing points stay missing.')
        sm = smooth_dataframe(long, method='one_euro', min_cutoff=min_cutoff, beta=beta)
        mouse = st.radio('Mouse', ['a', 'b'], horizontal=True)
        part = sm[sm.mouse == mouse].set_index('time_s')
        st.line_chart(part[['x', 'x_smooth']].rename(
            columns={'x': 'x raw (px)', 'x_smooth': 'x causal smooth (px)'}))
