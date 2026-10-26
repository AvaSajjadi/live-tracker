"""Tests for scripts/mars_pose_to_yolo.py, scripts/eval_yolo_pose.py and the training notebook (tiny fake data, no downloads)."""

import importlib.util
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv, ev = load("mars_pose_to_yolo"), load("eval_yolo_pose")


def record(i):
    return {"filename": f"MARS_top_{i:05d}.jpg", "width": 1024, "height": 570,
            "labels": ["nose tip", "right ear", "left ear", "neck", "right side body", "left side body", "tail base", "m", "e"],
            "coords": {"black": {"x": [100.0 + 10 * k for k in range(7)], "y": [200.0 + 5 * k for k in range(7)]},
                       "white": {"x": [700.0 - 10 * k for k in range(7)], "y": [300.0 + 5 * k for k in range(7)]}}}


def test_a_label_line_has_the_box_and_7_visible_keypoints_between_0_and_1():
    rec = record(0)
    parts = conv.label_line(0, rec["coords"]["black"]["x"], rec["coords"]["black"]["y"], 1024, 570).split()
    assert len(parts) == 5 + 21 and parts[0] == "0"
    values = [float(v) for v in parts[1:]]
    assert all(0.0 <= v <= 1.0 for i, v in enumerate(values) if i < 4 or (i - 4) % 3 != 2)       # box, x and y
    assert [parts[5 + 3 * k + 2] for k in range(7)] == ["2"] * 7                                  # all keypoints visible
    assert float(parts[5]) == pytest.approx(100 / 1024, abs=1e-4) and float(parts[6]) == pytest.approx(200 / 570, abs=1e-4)
    cx, w = float(parts[1]), float(parts[3])
    assert cx - w / 2 <= 100 / 1024 and cx + w / 2 >= 160 / 1024                                   # the box holds every keypoint


def test_the_dataset_has_two_classes_a_flip_map_and_a_test_split(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    recs = [record(i) for i in range(30)]
    for r in recs:
        cv2.imwrite(str(images / r["filename"]), np.zeros((57, 102, 3), np.uint8))
    out = tmp_path / "dataset"
    n_train, n_val = conv.convert(recs, images, out)
    assert (n_train, n_val) == (27, 3)                                      # a small set: the last tenth is the test set
    train, val = (out / "train.txt").read_text().split(), (out / "val.txt").read_text().split()
    assert len(train) == 27 and len(val) == 3 and not set(train) & set(val)
    lines = (out / "labels" / "MARS_top_00000.txt").read_text().strip().split("\n")
    assert [l.split()[0] for l in lines] == ["0", "1"]                      # black first, then white
    yaml = (out / "dataset.yaml").read_text()
    assert "kpt_shape: [7, 3]" in yaml and "flip_idx: [0, 2, 1, 3, 5, 4, 6]" in yaml and "0: black" in yaml and "1: white" in yaml
    assert sorted(conv.FLIP_IDX) == list(range(7)) and all(conv.FLIP_IDX[conv.FLIP_IDX[i]] == i for i in range(7))   # a real swap map


def test_the_full_set_uses_the_papers_split_with_1500_test_pictures(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    recs = [record(i) for i in range(3200)]
    for r in recs:
        (images / r["filename"]).write_bytes(b"x")
    n_train, n_val = conv.convert(recs, images, tmp_path / "d")
    assert (n_train, n_val) == (1700, 1500)                                 # with 15,000 pictures this is 13,500 and 1,500


def test_the_evaluation_reads_labels_and_counts_misses():
    text = "0 0.5 0.5 0.2 0.2 " + " ".join(f"{0.1 * (k + 1):.3f} {0.05 * (k + 1):.3f} 2" for k in range(7)) + "\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.txt"
        p.write_text(text, encoding="utf-8")
        gt = ev.read_labels(p)
    assert set(gt) == {0} and gt[0].shape == (7, 2) and gt[0][0] == pytest.approx([0.1 * 1024, 0.05 * 570])
    row = ev.summarize([np.zeros(7), np.full(7, 8.0)], found=2, total=4)
    assert row["found_share"] == 0.5 and row["median_error_all_px"] == 4.0
    assert row["share_within_5px"] == 0.5 and row["share_within_10px"] == 1.0
    assert ev.summarize([], 0, 5)["found_share"] == 0.0


def test_the_training_notebook_is_valid_and_uses_the_three_scripts():
    nb = json.loads((ROOT / "colab" / "train_yolo_pose_colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert nb["nbformat"] == 4
    for name in ("mars_pose_to_yolo.py", "train_yolo_pose.py", "eval_yolo_pose.py"):
        assert name in code and (ROOT / "scripts" / name).is_file()
    assert "j1ww1-mdc55" in code and "files.download" in code and "SAVE_TO_DRIVE = False" in code   # Drive is opt-in
