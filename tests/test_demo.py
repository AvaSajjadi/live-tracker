"""Tests for demo.py: finding the demo recording (a video with its keypoints) that the Upload step offers."""

import json

import pytest

from live_tracker import demo

from pose_fixtures import make_poses

NAME = "mouse001_task1_annotator1"
MANIFEST = {"seed": 0, "train": [f"task1/train/{NAME}"], "validation": ["task1/train/mouse002_task1_annotator1"],
            "test": ["task1/test/mouse003_task1_annotator1"]}


def make_root(tmp_path, names=(NAME,), with_poses=True, frames=300):
    (tmp_path / "splits").mkdir(parents=True, exist_ok=True)
    (tmp_path / "splits" / "split_manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    folder = tmp_path / demo.VIDEO_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / f"{n}.mp4").write_bytes(b"not really a video")
        if with_poses:
            make_poses(frames, session=f"task1/train/{n}").to_parquet(folder / f"{n}_poses.parquet", index=False)
    return tmp_path


def test_find_demos_needs_a_video_a_poses_file_and_an_allowed_session(tmp_path):
    root = make_root(tmp_path)
    (found,) = demo.find_demos(root)
    assert found.name == NAME and found.session == f"task1/train/{NAME}"
    assert found.video.name == f"{NAME}.mp4" and found.poses.name == f"{NAME}_poses.parquet"


def test_a_video_without_a_poses_file_is_not_a_demo(tmp_path):
    assert demo.find_demos(make_root(tmp_path, with_poses=False)) == []


def test_a_test_session_or_an_unknown_video_is_never_a_demo(tmp_path):
    root = make_root(tmp_path, names=("mouse003_task1_annotator1", "mouse099_task1_annotator1"))
    assert demo.find_demos(root) == []


def test_no_folder_or_no_manifest_means_no_demos(tmp_path):
    assert demo.find_demos(tmp_path) == []
    (tmp_path / demo.VIDEO_DIR).mkdir(parents=True)
    assert demo.find_demos(tmp_path) == []


def test_several_demos_are_listed_in_name_order(tmp_path):
    root = make_root(tmp_path, names=("mouse002_task1_annotator1", NAME))
    assert [d.name for d in demo.find_demos(root)] == [NAME, "mouse002_task1_annotator1"]


def test_model_available(tmp_path):
    assert not demo.model_available(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model_b.joblib").write_text("x", encoding="utf-8")
    assert not demo.model_available(tmp_path)
    (tmp_path / "models" / "model_b.json").write_text("{}", encoding="utf-8")
    assert demo.model_available(tmp_path)


def test_length_comes_from_the_frames(tmp_path):
    (found,) = demo.find_demos(make_root(tmp_path, frames=300))
    assert demo.length_s(found) == pytest.approx(10.0)
