"""Tests for the Replay page: the real video next to the skeleton drawing, and nothing about the run."""

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import live_tracker.replay_view as rv
from live_tracker import flow
from live_tracker.rule_workflow import RuleWorkflow
from live_tracker import skeleton_player as sp
from live_tracker.calms21 import KEYPOINTS
from live_tracker.uploads import SessionSource

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "2_Replay.py")


# Helpers other test files use (the synthetic sessions and an approved rule workflow).
DEMO, TRACK = (e.label for e in rv.synthetic_catalog())


def workflow(*instructions):
    wf = RuleWorkflow()
    for text in instructions or ("Alert me when the mice attack for at least 2 seconds",):
        wf.submit(text)
        wf.confirm(wf.pending.token)
    return wf


def start(wf=None, **state):
    at = AppTest.from_file(PAGE, default_timeout=60)
    if wf is not None:
        at.session_state["rule_workflow"] = wf
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def write_poses(path, frames=5):
    rows = [{"mouse": m, "keypoint": k, "frame": f, "x": 10.0 + f, "y": 20.0 + i}
            for m in "ab" for i, k in enumerate(KEYPOINTS) for f in range(frames)
            if (m, k, f) != ("a", "nose", 2)]
    rows.append({"mouse": "a", "keypoint": "nose", "frame": 2, "x": float("nan"), "y": float("nan")})
    pd.DataFrame(rows).to_parquet(path)


def test_pose_frames_keep_points_and_leave_missing_ones_out(tmp_path):
    write_poses(tmp_path / "p.parquet")
    frames = sp.pose_frames(tmp_path / "p.parquet")
    assert len(frames) == 5 and set(frames[0]) == {"a", "b"}
    assert frames[0]["a"][0]["x"] == 10.0 and frames[0]["a"][0]["y"] == 20.0
    assert frames[0]["a"][0]["status"] == "unreliable"
    assert frames[2]["a"][0]["status"] == "missing"
    assert frames[2]["a"][0]["x"] is None                      # missing: no position is invented


def test_html_holds_the_poses_and_follows_the_video(tmp_path):
    write_poses(tmp_path / "p.parquet")
    page = sp.html(sp.pose_frames(tmp_path / "p.parquet"), 30.0)
    assert "currentTime" in page and "__DATA__" not in page
    data = sp.payload(sp.pose_frames(tmp_path / "p.parquet"), 30.0)
    assert "requestVideoFrameCallback" in page
    assert "PROVISIONAL connections" in page
    assert data["fps"] == 30.0 and len(data["frames"]) == 5 and len(data["edges"]) == 8


def test_page_without_a_source_asks_for_one():
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception
    assert any("Nothing to replay" in i.value for i in at.info)


def test_page_shows_no_run_details(tmp_path):
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state[flow.SOURCE_KEY] = SessionSource(kind="keypoints", label="x", poses="p.parquet", session="s")
    at.run()
    assert not at.exception
    assert not at.metric and not at.dataframe and not at.slider and not at.radio


def test_a_real_camera_says_it_has_no_recording_to_replay():
    at = start(workflow(), **{flow.SOURCE_KEY: SessionSource("camera", "USB camera 0", device="0")})
    text = " ".join(i.value for i in at.info)
    assert "no recording to replay" in text and "Staged demo" in text
