"""Tests for the USB camera choice on the Upload step (fake camera, no hardware)."""

from pathlib import Path

import cv2
import numpy as np

from streamlit.testing.v1 import AppTest

from live_tracker import camera_check as cc

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "0_Upload.py")
PNG = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()


def texts(elements):
    return " ".join(e.value for e in elements)


def button(at, key):
    return next(b for b in at.button if b.key == key)


def open_usb(monkeypatch, snapshot):
    monkeypatch.setattr(cc, "snapshot", snapshot)
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    at.radio(key="upload_choice").set_value("Connect a camera").run()
    at.radio(key="upload_camera_kind").set_value("USB camera").run()
    return at


def test_a_working_camera_can_be_chosen_and_the_pose_model_gap_is_said(monkeypatch):
    at = open_usb(monkeypatch, lambda device, opener=None: (PNG, None))
    assert "Press Test the camera" in texts(at.caption)
    button(at, "upload_test_camera").click().run()
    assert "The camera works." in texts(at.success)
    assert "pose model" in texts(at.warning)
    button(at, "upload_use_camera").click().run()
    source = at.session_state["flow_source"]
    assert source.kind == "camera" and source.device == "0"


def test_a_camera_that_does_not_open_is_not_offered(monkeypatch):
    at = open_usb(monkeypatch, lambda device, opener=None: (None, "could not open the camera '0'"))
    button(at, "upload_test_camera").click().run()
    assert "cannot be used" in texts(at.error)
    assert not any(b.key == "upload_use_camera" for b in at.button)
    assert "flow_source" not in at.session_state


def test_changing_the_number_asks_for_a_new_test(monkeypatch):
    at = open_usb(monkeypatch, lambda device, opener=None: (PNG, None))
    button(at, "upload_test_camera").click().run()
    at.text_input(key="upload_device").set_value("1").run()
    assert not any(b.key == "upload_use_camera" for b in at.button)


def test_choosing_a_usb_camera_drops_the_demo_camera(monkeypatch):
    at = open_usb(monkeypatch, lambda device, opener=None: (PNG, None))
    assert "flow_source" not in at.session_state
