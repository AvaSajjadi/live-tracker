"""Tests for uploads: safe saving, keypoint and centers checks, and the SessionSource description."""

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from live_tracker import uploads as up
from live_tracker.calms21 import KEYPOINTS, MICE

from pose_fixtures import make_poses


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def good_df(frames=60, **kw):
    df = make_poses(frames, seed=1, **kw)
    df["y"] = df["y"] * 0.5                     # the made-up mice wander over 950 pixels; the CalMS21 frame is 570 high
    return df


# ---- keypoints ----

def test_a_good_file_becomes_the_calms21_layout():
    out = up.normalize_keypoints(good_df(60), "run1.csv")
    assert list(out.columns) == up.POSE_COLUMNS
    assert len(out) == 60 * len(MICE) * len(KEYPOINTS)
    assert out["session"].nunique() == 1 and out["session"].iloc[0] == "SYNTHETIC_live_01"
    assert (out["time_s"] == out["frame"] / 30.0).all()
    assert out["score"].between(0, 1).all()


def test_the_session_name_defaults_to_the_file_name_and_time_is_recomputed():
    df = good_df(40).drop(columns=["session", "time_s"])
    out = up.normalize_keypoints(df, "my mice.csv")
    assert out["session"].iloc[0] == "UPLOAD_my mice"
    wrong_time = good_df(40).assign(time_s=999.0)
    assert (up.normalize_keypoints(wrong_time)["time_s"] <= 40 / 30).all()


def test_missing_keypoints_are_filled_with_nan_and_score_zero_never_dropped():
    df = good_df(40)
    df = df[~((df.frame == 5) & (df.keypoint == "nose") & (df.mouse == "a"))]
    df.loc[(df.frame == 6) & (df.keypoint == "tail_base"), ["x", "y"]] = np.nan
    out = up.normalize_keypoints(df)
    assert len(out) == 40 * 14
    gap = out[(out.frame == 5) & (out.keypoint == "nose") & (out.mouse == "a")].iloc[0]
    assert np.isnan(gap.x) and gap.score == 0.0
    assert (out[(out.frame == 6) & (out.keypoint == "tail_base")]["score"] == 0.0).all()


def test_no_score_column_means_score_one_where_there_is_a_position():
    out = up.normalize_keypoints(good_df(40).drop(columns=["score"]))
    assert (out["score"] == 1.0).all()


@pytest.mark.parametrize("change,message", [
    (lambda d: d.drop(columns=["x"]), "lacks the columns"),
    (lambda d: d.assign(mouse="c"), "mouse must be"),
    (lambda d: d.assign(keypoint="tail"), "keypoint names"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "twice"),
    (lambda d: d.assign(frame=d.frame - 1), "whole numbers"),
    (lambda d: d.assign(frame=d.frame + 0.5), "whole numbers"),
    (lambda d: d[d.frame < 10], "only 10 frames"),
    (lambda d: d.assign(score=2.0), "between 0 and 1"),
    (lambda d: d.assign(x=d.x + 5000.0), "outside the CalMS21 image frame"),
    (lambda d: d.assign(x=np.nan, y=np.nan), "no keypoint has a position"),
    (lambda d: d.assign(session="task1/test/mouse078_task1_annotator1"), "never used"),
    (lambda d: d.assign(session=np.where(d.frame % 2 == 0, "A", "B")), "several sessions"),
])
def test_bad_keypoint_files_are_refused_with_a_reason(change, message):
    with pytest.raises(up.UploadError, match=message):
        up.normalize_keypoints(change(good_df(60)))


def test_save_keypoints_writes_parquet_and_returns_the_session(tmp_path):
    path, session = up.save_keypoints(tmp_path, "mice run.csv", csv_bytes(good_df(40)))
    assert path.name == "mice_run_poses.parquet" and session == "SYNTHETIC_live_01"
    back = pd.read_parquet(path)
    assert list(back.columns) == up.POSE_COLUMNS and len(back) == 40 * 14


def test_parquet_input_works_too(tmp_path):
    buffer = io.BytesIO()
    good_df(40).to_parquet(buffer, index=False)
    path, _ = up.save_keypoints(tmp_path, "p.parquet", buffer.getvalue())
    assert path.exists()


def test_unreadable_or_wrong_type_tables_are_refused(tmp_path):
    with pytest.raises(up.UploadError, match="could not be read"):
        up.save_keypoints(tmp_path, "x.parquet", b"this is not parquet")
    with pytest.raises(up.UploadError, match=r"\.csv or \.parquet"):
        up.save_keypoints(tmp_path, "x.xlsx", b"data")


# ---- names and videos ----

@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd", "passwd"), ("C:\\Users\\x\\clip.mp4", "clip.mp4"), ("my clip (1).mp4", "my_clip__1_.mp4"),
    ("..hidden", "hidden"),
])
def test_safe_name_cannot_point_elsewhere(raw, expected):
    assert up.safe_name(raw) == expected


def test_safe_name_refuses_a_name_with_nothing_usable():
    with pytest.raises(up.UploadError):
        up.safe_name("...")


def test_save_video_checks_type_and_content(tmp_path):
    path = up.save_video(tmp_path, "clip.MP4", b"\x00\x01")
    assert path.name == "clip.MP4" and path.read_bytes() == b"\x00\x01"
    with pytest.raises(up.UploadError, match="must be one of"):
        up.save_video(tmp_path, "clip.exe", b"MZ")
    with pytest.raises(up.UploadError, match="empty"):
        up.save_video(tmp_path, "clip.mp4", b"")


def test_each_upload_gets_its_own_new_folder(tmp_path):
    now = datetime(2026, 10, 24, 23, 0, 0, tzinfo=timezone.utc)
    a, b = up.new_upload_dir(tmp_path, now), up.new_upload_dir(tmp_path, now)
    assert a.name == "20261024T230000Z" and b.name == "20261024T230000Z_2" and a != b and a.is_dir() and b.is_dir()


# ---- centers ----

def test_a_centers_file_is_checked_and_saved(tmp_path):
    df = pd.DataFrame({"x_a": range(20), "y_a": range(20), "x_b": range(20), "y_b": range(20)})
    path = up.save_centers(tmp_path, "c.csv", csv_bytes(df))
    assert path.name == "c_centers.csv" and list(pd.read_csv(path).columns) == list(up.CENTER_COLUMNS)


def test_bad_centers_files_are_refused(tmp_path):
    with pytest.raises(up.UploadError, match="lacks the columns"):
        up.check_centers("c.csv", csv_bytes(pd.DataFrame({"x_a": range(20)})))
    with pytest.raises(up.UploadError, match="fewer than 10"):
        up.check_centers("c.csv", csv_bytes(pd.DataFrame({c: range(5) for c in up.CENTER_COLUMNS})))
    with pytest.raises(up.UploadError, match="no numbers"):
        up.check_centers("c.csv", csv_bytes(pd.DataFrame({c: ["a"] * 20 for c in up.CENTER_COLUMNS})))


# ---- the source description ----

def test_source_kinds_and_what_they_need():
    assert up.SessionSource("keypoints", "k", poses="p.parquet", session="S").can_score
    assert not up.SessionSource("keypoints", "k", poses="p.parquet", session="S").has_video
    v = up.SessionSource("video_keypoints", "v", video="v.mp4", poses="p.parquet", session="S")
    assert v.can_score and v.has_video
    assert not up.SessionSource("centers", "c", centers="c.csv").can_score
    assert not up.SessionSource("camera", "cam").can_score
    for bad in (dict(kind="telepathy", label="x"), dict(kind="keypoints", label="k"),
                dict(kind="video_keypoints", label="v", poses="p", session="S"), dict(kind="centers", label="c"),
                dict(kind="demo", label="d", poses="p", session="S")):
        with pytest.raises(up.UploadError):
            up.SessionSource(**bad)


def test_source_round_trips_through_a_dict():
    s = up.SessionSource("video_keypoints", "v", video="v.mp4", poses="p.parquet", session="S", fps=25.0)
    assert up.SessionSource.from_dict(s.to_dict()) == s
