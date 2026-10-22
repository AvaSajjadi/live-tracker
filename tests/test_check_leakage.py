"""Tests for scripts/check_leakage.py: each check passes clean input and catches a leak."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_leakage", ROOT / "scripts" / "check_leakage.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)

TR = ["task1/train/mouse001_task1_annotator1", "task1/train/mouse002_task1_annotator1"]
VA = ["task1/train/mouse003_task1_annotator1"]
TE = ["task1/test/mouse071_task1_annotator1"]
MANIFEST = {"seed": 0, "train": TR, "validation": VA, "test": TE}
FRAMES = {TR[0]: 10, TR[1]: 7, VA[0]: 5, TE[0]: 9}


def _ok(results):
    return [ok for _, ok, _ in results]


def test_split_clean_and_leaks():
    assert all(_ok(cl.check_split(MANIFEST, TR + VA, TE)))
    leaky = {**MANIFEST, "train": TR + TE}
    assert not all(_ok(cl.check_split(leaky, TR + VA, TE)))
    missing = {**MANIFEST, "test": []}
    assert not all(_ok(cl.check_split(missing, TR + VA, TE)))


def test_model_rows_catch_an_extra_session():
    # stride 3: ceil(10/3) + ceil(7/3) = 4 + 3 = 7 rows from train only
    info = {"name": "m", "train_sessions": 2, "validation_sessions": 1, "n_train_rows": 7,
            "fit_params": {"train_stride": 3}}
    assert all(_ok(cl.check_model(info, MANIFEST, FRAMES)))
    leaked = {**info, "n_train_rows": 7 + 3}         # + ceil(9/3) rows of the test session
    assert _ok(cl.check_model(leaked, MANIFEST, FRAMES)) == [True, False]


def test_scores_must_be_validation_only():
    assert all(_ok(cl.check_scores("scores_validation_x.csv", VA, MANIFEST)))
    assert not any(_ok(cl.check_scores("scores_validation_x.csv", VA + TE, MANIFEST)))


def test_repo_mentions():
    clean = {"splits/split_manifest.json": TE[0], "README.md": "nothing here"}
    assert all(_ok(cl.check_repo(clean, TE)))
    leak = {**clean, "docs/demo.md": "replay mouse071_task1_annotator1"}
    (name, ok, detail), = cl.check_repo(leak, TE)
    assert not ok and detail == "docs/demo.md"
