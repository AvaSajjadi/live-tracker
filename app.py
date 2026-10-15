"""First milestone: visualize pose-derived features from an exported CSV.

This does not yet classify behavior or call an LLM.
"""
import streamlit as st
import pandas as pd
from live_tracker.motion import frame_features

st.set_page_config(page_title='Live Tracker', layout='wide')
st.title('Live Tracker | Motion Engine')
st.caption('Milestone 1: visualize tracked positions and quality flags; no behavior claims.')
st.write('Upload a CSV with columns: x_a, y_a, x_b, y_b. One row per frame.')
fps = st.number_input('Frame rate (fps)', min_value=1.0, value=30.0)
upload = st.file_uploader('Pose CSV', type='csv')
if upload:
    data = pd.read_csv(upload)
    required = ['x_a', 'y_a', 'x_b', 'y_b']
    if not set(required).issubset(data.columns):
        st.error('Missing columns: ' + ', '.join(sorted(set(required) - set(data.columns))))
    else:
        f = frame_features(data[['x_a','y_a']].to_numpy(),
                           data[['x_b','y_b']].to_numpy(), fps=fps)
        st.metric('Frames', len(f))
        st.metric('Both tracks usable', f'{f.tracking_ok.mean():.1%}')
        st.line_chart(f[['speed_a_px_s', 'speed_b_px_s', 'distance_px']])
        st.dataframe(f.head(50), use_container_width=True)
        st.download_button('Download motion features CSV', f.to_csv(index=False),
                           file_name='motion_features.csv', mime='text/csv')
