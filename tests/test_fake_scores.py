"""Tests for the synthetic score generator (new file)."""

import numpy as np
import pandas as pd
import pytest

from live_tracker.fake_scores import (COLUMNS, DEMO_SEGMENTS, DEMO_SESSION, SCORE_COLS,
                                      Segment, TrackingLoss, build_scores, demo_scores, main)

FPS = 30.0


def _between(df, start_s, end_s):
    return df[(df["time_s"] >= start_s - 1e-9) & (df["time_s"] < end_s - 1e-9)]


def test_columns_match_interfaces_section_2():
    df = demo_scores()
    assert tuple(df.columns) == COLUMNS
    assert df["tracking_ok"].dtype == bool
    assert df["frame"].tolist() == list(range(len(df)))
    assert np.allclose(df["time_s"], df["frame"] / FPS)
    for col in SCORE_COLS:
        assert df[col].between(0.0, 1.0).all()


def test_session_is_labeled_synthetic():
    assert DEMO_SESSION.startswith("SYNTHETIC_")
    assert (demo_scores()["session"] == DEMO_SESSION).all()
    with pytest.raises(ValueError, match="SYNTHETIC_"):
        build_scores("s01", 5.0)


def test_deterministic():
    pd.testing.assert_frame_equal(demo_scores(), demo_scores())
    pd.testing.assert_frame_equal(demo_scores(seed=3), demo_scores(seed=3))
    assert not demo_scores(seed=0)["attack"].equals(demo_scores(seed=1)["attack"])


def test_demo_segments_have_the_promised_scores():
    df = demo_scores()
    for seg in DEMO_SEGMENTS:
        part = _between(df, seg.start_s, seg.end_s)
        assert len(part) == round((seg.end_s - seg.start_s) * FPS)
        assert part[seg.behavior].between(seg.low, seg.high).all()
    clear, uncertain, short, interrupted, _ = DEMO_SEGMENTS
    assert clear.end_s - clear.start_s > 2 and clear.low >= 0.8
    assert uncertain.end_s - uncertain.start_s > 2 and 0.5 <= uncertain.low and uncertain.high < 0.8
    assert short.end_s - short.start_s < 2
    # outside the attack segments, attack scores stay low
    in_any = np.zeros(len(df), bool)
    for seg in DEMO_SEGMENTS:
        if seg.behavior == "attack":
            in_any |= (df["time_s"] >= seg.start_s - 1e-9) & (df["time_s"] < seg.end_s - 1e-9)
    assert (df.loc[~in_any, "attack"] < 0.5).all()


def test_demo_tracking_loss_inside_high_scores():
    df = demo_scores()
    lost = df[~df["tracking_ok"]]
    assert len(lost) == 30 and lost["time_s"].min() == pytest.approx(16.0)
    # scores stay high while tracking is lost: the monitor must ignore them
    assert (lost["attack"] >= 0.88).all()


def test_build_scores_custom():
    df = build_scores("SYNTHETIC_x", 2.0, [Segment("mount", 0.5, 1.0, 0.9, 0.9)],
                      [TrackingLoss(1.5, 2.0)], fps=10.0)
    assert len(df) == 20
    assert (df.loc[5:9, "mount"] == 0.9).all() and (df.loc[[4, 10], "mount"] < 0.5).all()
    assert df["tracking_ok"].tolist() == [True] * 15 + [False] * 5


@pytest.mark.parametrize("kwargs", [
    {"segments": [Segment("dance", 0, 1, 0.5, 0.6)]},
    {"segments": [Segment("attack", 0, 1, 0.8, 0.6)]},
    {"fps": 0},
])
def test_build_scores_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        build_scores("SYNTHETIC_bad", 2.0, **kwargs)


def test_cli_writes_csv(tmp_path):
    out = tmp_path / "scores.csv"
    assert main([str(out)]) == 0
    back = pd.read_csv(out)
    assert tuple(back.columns) == COLUMNS
    assert back["tracking_ok"].dtype == bool
    assert main([]) == 2
