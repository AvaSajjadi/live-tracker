"""Tests for the CalMS21 loader and split, on a tiny fake file with the real layout.

Layout copied from the real files: {group: {session_id: {keypoints (T,2,2,7),
scores (T,2,7), annotations (T,), metadata}}}. No real data is committed.
"""

import json

import numpy as np
import pytest

from live_tracker.calms21 import (FPS, KEYPOINTS, TORSO_KEYPOINTS, VOCAB, check_manifest,
                                  load_calms21, main, make_split, session_ids)

T = 20


def _record(seed, n=T):
    rng = np.random.default_rng(seed)
    kp = rng.uniform(0, 500, size=(n, 2, 2, 7))
    return {"keypoints": kp.tolist(),
            "scores": rng.uniform(0, 1, size=(n, 2, 7)).tolist(),
            "annotations": (np.arange(n) % 4).tolist(),
            "metadata": {"annotator-id": 0, "vocab": dict(VOCAB)}}


def _file(split, mice):
    return {"annotator-id_0": {f"task1/{split}/mouse{m:03d}_task1_annotator1": _record(m)
                               for m in mice}}


@pytest.fixture
def tiny():
    return _file("train", [1, 2])


def test_long_table_columns_and_size(tiny):
    poses, labels = load_calms21(tiny)
    assert list(poses.columns) == ["session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score"]
    assert len(poses) == 2 * 2 * 7 * T          # sessions x mice x keypoints x frames
    assert list(labels.columns) == ["session", "frame", "time_s", "label"]
    assert len(labels) == 2 * T


def test_values_map_to_right_axes(tiny):
    poses, _ = load_calms21(tiny)
    sid = "task1/train/mouse002_task1_annotator1"
    kp = np.asarray(tiny["annotator-id_0"][sid]["keypoints"])
    sc = np.asarray(tiny["annotator-id_0"][sid]["scores"])
    # intruder (index 1 -> "b"), left_hip (index 4), frame 7
    row = poses[(poses.session == sid) & (poses.mouse == "b")
                & (poses.keypoint == "left_hip") & (poses.frame == 7)].iloc[0]
    assert row.x == kp[7, 1, 0, 4] and row.y == kp[7, 1, 1, 4] and row.score == sc[7, 1, 4]
    row = poses[(poses.session == sid) & (poses.mouse == "a")
                & (poses.keypoint == "nose") & (poses.frame == 0)].iloc[0]
    assert row.x == kp[0, 0, 0, 0] and row.y == kp[0, 0, 1, 0]


def test_time_and_label_names(tiny):
    poses, labels = load_calms21(tiny)
    assert poses.time_s.max() == pytest.approx((T - 1) / FPS)
    assert labels.label.iloc[:4].tolist() == ["attack", "investigation", "mount", "other"]


def test_session_filter(tiny):
    sid = "task1/train/mouse001_task1_annotator1"
    poses, labels = load_calms21(tiny, sessions=[sid])
    assert set(poses.session) == {sid} and set(labels.session) == {sid}
    with pytest.raises(KeyError):
        load_calms21(tiny, sessions=["nope"])


def test_reads_file_path(tmp_path, tiny):
    p = tmp_path / "calms21_task1_train.json"
    p.write_text(json.dumps(tiny))
    poses, _ = load_calms21(p)
    assert poses.session.nunique() == 2


def test_rejects_wrong_vocab(tiny):
    rec = tiny["annotator-id_0"]["task1/train/mouse001_task1_annotator1"]
    rec["metadata"]["vocab"] = {"attack": 1, "investigation": 0, "mount": 2, "other": 3}
    with pytest.raises(ValueError, match="vocab"):
        load_calms21(tiny)


def test_rejects_wrong_shape(tiny):
    rec = tiny["annotator-id_0"]["task1/train/mouse001_task1_annotator1"]
    rec["keypoints"] = np.zeros((T, 2, 7, 2)).tolist()
    with pytest.raises(ValueError, match="shape"):
        load_calms21(tiny)


def test_torso_names_exist():
    assert set(TORSO_KEYPOINTS) <= set(KEYPOINTS)


def test_works_with_feature_body_center(tiny):
    from live_tracker.features import body_center
    poses, _ = load_calms21(tiny)
    c = body_center(poses.assign(valid=True, imputed=False), torso_keypoints=list(TORSO_KEYPOINTS))
    assert len(c) == 2 * 2 * T and c.n_keypoints.eq(3).all()


def test_split_by_session_no_overlap():
    train = [f"tr{i:02d}" for i in range(70)]
    test = [f"te{i:02d}" for i in range(19)]
    m = make_split(train, test, val_frac=0.2, seed=0)
    check_manifest(m)
    assert len(m["validation"]) == 14 and len(m["train"]) == 56
    assert m["test"] == sorted(test) and m["seed"] == 0
    assert not set(m["train"]) & set(m["validation"])


def test_split_is_reproducible():
    train, test = [f"tr{i}" for i in range(10)], ["te0"]
    assert make_split(train, test, seed=3) == make_split(train, test, seed=3)
    assert make_split(train, test, seed=3)["validation"] != make_split(train, test, seed=4)["validation"]


def test_split_rejects_overlap():
    with pytest.raises(ValueError):
        make_split(["s1", "s2", "s3"], ["s3"])
    with pytest.raises(ValueError):
        check_manifest({"train": ["a"], "validation": ["a"], "test": []})


def test_cli_writes_manifest(tmp_path):
    tr, te = tmp_path / "train.json", tmp_path / "test.json"
    tr.write_text(json.dumps(_file("train", range(1, 11))))
    te.write_text(json.dumps(_file("test", [71, 72])))
    out = tmp_path / "splits" / "split_manifest.json"
    assert main(["split", "--train", str(tr), "--test", str(te), "--out", str(out)]) == 0
    m = check_manifest(json.loads(out.read_text()))
    assert m["test"] == session_ids(te) and len(m["validation"]) == 2
