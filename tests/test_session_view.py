"""Tests for session_view and the real-session page (new file).

Everything here is a tiny SYNTHETIC test fixture in the CalMS21 JSON layout, run
through the real validation-cache code. It is not CalMS21 data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from live_tracker import session_view as sv
from live_tracker.calms21 import FPS
from live_tracker.features import KEYPOINT_COLS
from live_tracker.investigation_data import extract_sessions, session_filename

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "4_Real_Session.py")
VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
VAL = ["task1/train/SYNTHETIC_val_a", "task1/train/SYNTHETIC_val_b"]
TRAIN = ["task1/train/SYNTHETIC_train_a"]
TEST = ["task1/test/SYNTHETIC_test_a"]
MANIFEST = {"train": TRAIN, "validation": VAL, "test": TEST}
N = 120


def record(labels, seed):
    """Two mice walking slowly, 7 keypoints each (nose first, tail base last), 0.9 confidence."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    walk = np.cumsum(rng.normal(0, 1.0, (n, 2, 2)), axis=0)
    kp = np.zeros((n, 2, 2, 7))
    for m, cx in enumerate((300.0, 600.0)):
        for k in range(7):
            kp[:, m, 0, k] = cx + walk[:, m, 0] + 6.0 * (3 - k)
            kp[:, m, 1, k] = 250.0 + walk[:, m, 1] + 2.0 * k
    return {"keypoints": kp.tolist(), "scores": np.full((n, 2, 7), 0.9).tolist(),
            "annotations": list(labels), "metadata": {"annotator-id": 0, "vocab": dict(VOCAB)}}


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sv")
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST))
    labels_a = [3] * 30 + [0] * 40 + [3] * 20 + [1] * 30           # other, attack x40, other, investigation x30
    recs = {VAL[0]: record(labels_a, 1), VAL[1]: record([3] * N, 2),
            TRAIN[0]: record([3] * N, 3), TEST[0]: record([0] * N, 4)}
    src = tmp / "source.json"
    src.write_text(json.dumps({"annotator-id_0": {k: v for k, v in recs.items() if k not in TEST}}))
    out = tmp / "cache"
    extract_sessions(src, out, VAL, manifest)
    return out, manifest


def test_lists_only_verified_validation_sessions(cache):
    out, manifest = cache
    assert sv.list_sessions(out, manifest) == VAL


def test_view_has_the_motion_engine_tables(cache):
    out, manifest = cache
    v = sv.load_view(VAL[0], out, manifest)
    assert v.session == VAL[0] and v.n_frames == N and v.duration_s == pytest.approx(N / FPS)
    assert {"session", "frame", "time_s", "tracking_ok", "dist_center", "speed_a", "speed_b"} <= set(v.table.columns)
    assert len(v.table) == N and len(v.keypoints) == N
    assert list(v.keypoints.columns) == ["session", "frame", "time_s", *KEYPOINT_COLS]
    assert 0.0 <= v.tracking_ok_share <= 1.0
    assert v.warnings == []                         # clean fixture: no tracking warnings
    assert set(v.flag_share) >= {"missing", "frozen", "out_of_frame"}


def test_labels_are_context_only_and_never_change_features(cache):
    out, manifest = cache
    a = sv.load_view(VAL[0], out, manifest)
    b = sv.load_view(VAL[1], out, manifest)         # different labels, different motion: just load both
    assert set(a.labels["label"]) == {"other", "attack", "investigation"}
    assert set(b.labels["label"]) == {"other"}
    # the feature table has no label column at all
    assert not {"label", "annotation"} & set(a.table.columns)


def test_label_bouts_known_sequence():
    lab = pd.DataFrame({"session": "s", "frame": np.arange(8), "time_s": np.arange(8) / FPS,
                        "label": ["other", "other", "attack", "attack", "attack", "other", "investigation", "investigation"]})
    b = sv.label_bouts(lab)
    assert b["label"].tolist() == ["attack", "investigation"]
    assert b["start_s"].tolist() == pytest.approx([2 / FPS, 6 / FPS])
    assert b["end_s"].tolist() == pytest.approx([5 / FPS, 8 / FPS])           # exclusive end
    assert b["duration_s"].tolist() == pytest.approx([3 / FPS, 2 / FPS])
    assert sv.label_bouts(lab, skip=())["label"].tolist() == ["other", "attack", "other", "investigation"]


def test_label_bouts_of_a_view(cache):
    out, manifest = cache
    b = sv.label_bouts(sv.load_view(VAL[0], out, manifest).labels)
    assert b["label"].tolist() == ["attack", "investigation"]
    assert b["start_s"].tolist() == pytest.approx([30 / FPS, 90 / FPS])
    assert b["duration_s"].tolist() == pytest.approx([40 / FPS, 30 / FPS])


def test_tracking_runs():
    t = pd.DataFrame({"frame": np.arange(10), "time_s": np.arange(10) / FPS,
                      "tracking_ok": [True, False, False, True, True, True, False, True, True, False]})
    r = sv.tracking_runs(t)
    assert r["start_s"].tolist() == pytest.approx([1 / FPS, 6 / FPS, 9 / FPS])
    assert r["duration_s"].tolist() == pytest.approx([2 / FPS, 1 / FPS, 1 / FPS])
    assert sv.tracking_runs(t.assign(tracking_ok=True)).empty


def test_missing_cache_is_a_clear_unavailable_error(tmp_path):
    with pytest.raises(sv.SessionUnavailable, match="investigation_data"):
        sv.list_sessions(tmp_path / "nope", tmp_path / "m.json")
    with pytest.raises(sv.SessionUnavailable, match="investigation_data"):
        sv.load_view(VAL[0], tmp_path / "nope", tmp_path / "m.json")


def test_unknown_or_non_validation_session_is_refused(cache):
    out, manifest = cache
    for sid in (TRAIN[0], TEST[0], "task1/train/nope"):
        with pytest.raises(sv.SessionUnavailable, match="not in the verified validation cache"):
            sv.load_view(sid, out, manifest)


def test_tampered_cache_file_is_refused(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST))
    src = tmp_path / "source.json"
    src.write_text(json.dumps({"annotator-id_0": {VAL[0]: record([3] * N, 1), VAL[1]: record([3] * N, 2),
                                                  TRAIN[0]: record([3] * N, 3)}}))
    out = tmp_path / "cache"
    extract_sessions(src, out, VAL, manifest)
    target = out / session_filename(VAL[0])
    target.write_text(target.read_text().replace("0.9", "0.8", 5))      # change content, keep the hash in the index
    with pytest.raises(sv.SessionUnavailable):
        sv.list_sessions(out, manifest)


# --- the page ---------------------------------------------------------------------------

def test_page_without_cache_explains_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "DEFAULT_CACHE", tmp_path / "nope")
    monkeypatch.setattr(sv, "DEFAULT_MANIFEST", tmp_path / "m.json")
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    assert any("not available" in i.value for i in at.info)
    assert any("investigation_data" in c.value for c in at.code)


def test_page_shows_a_session_and_labels_the_annotation_as_context(cache, monkeypatch):
    out, manifest = cache
    monkeypatch.setattr(sv, "DEFAULT_CACHE", out)
    monkeypatch.setattr(sv, "DEFAULT_MANIFEST", manifest)
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [m.label for m in at.metric][:4] == ["Frames", "Duration", "tracking_ok frames", "Tracking warnings"]
    assert at.selectbox[0].options == [sv.short_name(s) for s in VAL]
    assert any("ground truth" in h.value.lower() for h in at.subheader)
    text = " ".join(c.value for c in at.caption).lower()
    assert "no behavior is claimed" in text and "not a model prediction" in text
    at.selectbox[0].select(sv.short_name(VAL[1])).run()
    assert not at.exception
