"""One open camera shared by the pages that show live video (new file).

On Windows only one program can use a camera at a time. The Upload step (camera test and corners) and the Real session
page (staged demo) both show the live picture, so they share ONE open capture kept in the Streamlit session. It is
released with release(): by the Stop button, when the session is reset, and before a real camera run starts, because that
run opens the camera in its own program.
"""

from __future__ import annotations

import streamlit as st

from . import run_sources as rsrc

KEY_CAP, KEY_DEVICE = "camera_live_capture", "camera_live_device"


def release():
    cap = st.session_state.pop(KEY_CAP, None)
    st.session_state.pop(KEY_DEVICE, None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def capture(device):
    """The open capture for this camera number or address. Opens it on first use; a different camera replaces it."""
    device = str(device).strip()
    cap = st.session_state.get(KEY_CAP)
    if cap is not None and st.session_state.get(KEY_DEVICE) == device:
        return cap
    release()
    cap = rsrc.open_camera(rsrc.camera_device(device))          # raises CameraUnavailableError
    st.session_state[KEY_CAP], st.session_state[KEY_DEVICE] = cap, device
    return cap


def read(device):
    """(the newest picture, None), or (None, the reason in plain words)."""
    try:
        cap = capture(device)
    except rsrc.CameraUnavailableError as exc:
        return None, str(exc)
    try:
        ok, image = cap.read()
    except Exception as exc:                                    # a driver can fail in many ways
        release()
        return None, f"the camera gave an error ({type(exc).__name__})"
    if not ok or image is None:
        return None, "the camera gave no picture"
    return image, None
