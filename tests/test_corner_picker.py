"""Tests for corner_picker: the Python side of the draggable corners (the JavaScript runs in the browser)."""

import base64

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from live_tracker import corner_picker as cp
from live_tracker import screen_map as sm


def test_the_picture_goes_to_the_component_as_base64_jpeg():
    image = np.full((48, 64, 3), 120, dtype=np.uint8)
    text = cp.jpeg_base64(image)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(text), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (48, 64, 3) and abs(int(decoded.mean()) - 120) < 5


def app():
    import streamlit as st
    from live_tracker import corner_picker as cp
    from live_tracker import screen_map as sm
    st.session_state["answer"] = cp.current_corners("picker", sm.DEFAULT_CORNERS)


def test_before_any_drag_the_default_corners_are_used_and_after_a_drag_the_dragged_ones():
    at = AppTest.from_function(app).run()
    assert not at.exception and at.session_state["answer"] == [list(p) for p in sm.DEFAULT_CORNERS]
    dragged = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
    at.session_state["picker"] = {"corners": dragged}
    at.run()
    assert at.session_state["answer"] == dragged
    sm.check_corners(at.session_state["answer"])                  # what the dragged corners look like is accepted by screen_map
