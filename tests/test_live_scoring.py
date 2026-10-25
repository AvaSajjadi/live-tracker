"""Tests for live_scoring: a moving buffer must give the same scores as the whole session.

Fast tests use made-up poses (SYNTHETIC session). One opt-in test uses a real CalMS21 validation
session when the data and models are present and LIVE_TRACKER_SLOW=1 is set.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from live_tracker import live_scoring as ls
from live_tracker.calms21 import FPS
from live_tracker.classify import score_frames
from pose_fixtures import SESSION, features_of, make_poses, tiny_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world():
    train = make_poses(400, seed=3)
    model, cols = tiny_model(features_of(train))
    return model, cols


def stream(scorer, poses, frames=None):
    out = []
    by_frame = {f: g for f, g in poses.groupby("frame")}
    for f in (frames if frames is not None else sorted(by_frame)):
        out += scorer.push(f, by_frame[f])
    return out + scorer.flush()


def test_buffered_scores_equal_whole_session_scores(world):
    model, cols = world
    poses = make_poses(300, seed=5)
    batch = score_frames(model, features_of(poses), [SESSION]).set_index("frame")
    live = pd.DataFrame(stream(ls.LiveScorer(model, cols, session=SESSION), poses)).set_index("frame")
    assert list(live.index) == list(batch.index)
    for c in ("attack", "mount", "investigation"):
        assert np.abs(live[c] - batch[c]).max() < 1e-9
    assert (live["tracking_ok"] == batch["tracking_ok"]).all()


def test_rows_have_the_monitor_format(world):
    model, cols = world
    rows = stream(ls.LiveScorer(model, cols, session=SESSION), make_poses(40, seed=6))
    assert len(rows) == 40
    r = rows[0]
    assert tuple(r) == ls.SCORE_ROW_KEYS and r["session"] == SESSION
    assert type(r["frame"]) is int and type(r["tracking_ok"]) is bool and type(r["attack"]) is float
    assert [x["frame"] for x in rows] == list(range(40))
    assert abs(rows[7]["time_s"] - 7 / FPS) < 1e-12


def test_scores_come_in_chunks_and_flush_gives_the_rest(world):
    model, cols = world
    scorer = ls.LiveScorer(model, cols, session=SESSION, chunk=5)
    poses = make_poses(12, seed=7)
    by_frame = {f: g for f, g in poses.groupby("frame")}
    counts = [len(scorer.push(f, by_frame[f])) for f in range(12)]
    assert counts == [0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0]
    assert [r["frame"] for r in scorer.flush()] == [10, 11]
    assert scorer.flush() == []


def test_a_row_never_depends_on_later_frames(world):
    model, cols = world
    a = make_poses(120, seed=8)
    b = a.copy()
    b.loc[b.frame >= 60, ["x", "y"]] += 40.0                       # change only the future
    ra = {r["frame"]: r for r in stream(ls.LiveScorer(model, cols, session=SESSION, chunk=1), a)}
    rb = {r["frame"]: r for r in stream(ls.LiveScorer(model, cols, session=SESSION, chunk=1), b)}
    for f in range(60):
        assert ra[f] == rb[f]


def test_a_dropped_frame_is_absent_from_the_output(world):
    model, cols = world
    poses = make_poses(60, seed=9)
    frames = [f for f in range(60) if f not in (20, 21, 22)]
    rows = stream(ls.LiveScorer(model, cols, session=SESSION), poses, frames)
    assert [r["frame"] for r in rows] == frames


def test_model_a_uses_only_the_basic_features():
    train = make_poses(300, seed=11)
    basic = features_of(train, motion=False)
    model, cols = tiny_model(basic)
    poses = make_poses(60, seed=12)
    batch = score_frames(model, features_of(poses, motion=False), [SESSION]).set_index("frame")
    live = pd.DataFrame(stream(ls.LiveScorer(model, cols, session=SESSION, features="basic"), poses)).set_index("frame")
    assert np.abs(live["attack"] - batch["attack"]).max() < 1e-9


def test_wrong_columns_are_refused(world):
    model, cols = world
    scorer = ls.LiveScorer(model, cols[:-1], session=SESSION, chunk=1)
    poses = make_poses(3, seed=13)
    with pytest.raises(ls.LiveScoringError, match="feature columns differ"):
        scorer.push(0, poses[poses.frame == 0])


def test_bad_input_is_refused(world):
    model, cols = world
    poses = make_poses(4, seed=14)
    scorer = ls.LiveScorer(model, cols, session=SESSION)
    scorer.push(0, poses[poses.frame == 0])
    with pytest.raises(ls.LiveScoringError, match="not after"):
        scorer.push(0, poses[poses.frame == 0])
    with pytest.raises(ls.LiveScoringError, match="lack columns"):
        scorer.push(1, poses[poses.frame == 1].drop(columns=["score"]))
    with pytest.raises(ls.LiveScoringError, match="this session and this frame"):
        scorer.push(2, poses[poses.frame == 1])
    with pytest.raises(ls.LiveScoringError, match="this session and this frame"):
        scorer.push(3, poses[poses.frame == 3].assign(session="OTHER"))
    for bad in ({"features": "motion"}, {"features": "basic,other"}, {"chunk": 0}, {"buffer_s": 0.01}):
        with pytest.raises(ls.LiveScoringError):
            ls.LiveScorer(model, cols, session=SESSION, **bad)


@pytest.mark.skipif(os.environ.get("LIVE_TRACKER_SLOW") != "1", reason="set LIVE_TRACKER_SLOW=1 to run on real data")
def test_real_validation_session_buffered_equals_whole_session():
    from live_tracker.calms21 import _read_json, _sessions, load_calms21
    import joblib
    train = ROOT / "data/raw/calms21/calms21_task1_train.json"
    model_json = ROOT / "models/model_b.json"
    if not (train.is_file() and model_json.is_file() and (ROOT / "models/model_b.joblib").is_file()):
        pytest.skip("CalMS21 data or model B not present locally")
    manifest = json.loads((ROOT / "splits/split_manifest.json").read_text())
    sid = manifest["validation"][0]                                  # validation only, never test
    recs = _sessions(_read_json(train))
    poses, _ = load_calms21({"g": {sid: recs[sid]}})
    poses = poses[poses.frame < 600]                                 # first 20 s keeps the test short
    info = json.loads(model_json.read_text())
    model = joblib.load(ROOT / "models/model_b.joblib")
    batch = score_frames(model, features_of(poses), [sid]).set_index("frame")
    live = pd.DataFrame(stream(ls.LiveScorer(model, info["feature_columns"], session=sid), poses)).set_index("frame")
    for c in ("attack", "mount", "investigation"):
        assert np.abs(live[c] - batch[c]).max() < 1e-6
