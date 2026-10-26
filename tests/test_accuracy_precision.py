"""Tests for the accuracy and precision report (tiny synthetic data; the real data runs are in the report itself)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep, mars_acc = load("accuracy_precision_report"), load("accuracy_mars_handlabels")


def synthetic_sessions(n_sessions=3, frames=400, miss_in=None):
    """An attack bout in frames 100-200 of every session; the scores say attack there (or nothing, for the sessions in miss_in)."""
    scores, labels = [], []
    for s in range(n_sessions):
        sid = f"task1/train/mouse{s:03d}_task1_annotator1"
        f = np.arange(frames)
        on = (f >= 100) & (f < 200) & (s not in (miss_in or ()))
        scores.append(pd.DataFrame({"session": sid, "frame": f, "time_s": f / 30.0, "attack": np.where(on, 0.95, 0.0), "mount": 0.0,
                                    "investigation": 0.0, "tracking_ok": True}))
        labels.append(pd.DataFrame({"session": sid, "frame": f, "time_s": f / 30.0,
                                    "label": np.where((f >= 100) & (f < 200), "attack", "other")}))
    return pd.concat(scores, ignore_index=True), pd.concat(labels, ignore_index=True)


def test_perfect_scores_give_perfect_accuracy_and_precision():
    scores, labels = synthetic_sessions()
    r = rep.behavior_section(scores, labels, "synthetic")
    assert r["frame_accuracy"] == 1.0 and r["alert_precision"] == 1.0 and r["alert_recall"] == 1.0
    assert r["alerts"] == 3 and r["true_attack_bouts"] == 3 and r["false_alerts_per_hour"] == 0.0
    assert r["repeatable"] is True and r["demo_session"] is None


def test_a_missed_bout_lowers_recall_but_not_precision():
    scores, labels = synthetic_sessions(miss_in=(1,))
    r = rep.behavior_section(scores, labels, "synthetic")
    assert r["alert_precision"] == 1.0 and r["alert_recall"] == pytest.approx(2 / 3, abs=1e-3)
    lo, hi = r["alert_recall_ci95"]
    assert 0.0 <= lo <= r["alert_recall"] <= hi <= 1.0


def test_the_bootstrap_interval_holds_the_estimate_and_is_wide_for_few_sessions():
    counts = [{"tp": 1, "events": 1}, {"tp": 0, "events": 1}, {"tp": 1, "events": 1}]
    lo, hi = rep.bootstrap_ci(counts, "tp", "events")
    assert lo <= 2 / 3 <= hi and hi - lo > 0.3
    assert rep.bootstrap_ci([{"tp": 5, "events": 5}] * 4, "tp", "events") == (1.0, 1.0)


def test_the_centers_numbers_are_exact_without_noise_and_noise_makes_a_still_mouse_move():
    c = rep.centers_section()
    clean, noisy = c["rows"][0], c["rows"][-1]
    assert clean["speed_error_median_px_s"] == 0.0 and clean["still_mouse_reads_px_s"] == 0.0 and clean["distance_error_median_px"] == 0.0
    assert noisy["still_mouse_reads_px_s"] > 50 and noisy["speed_spread_px_s"] > clean["speed_spread_px_s"]
    assert c["repeatable"] and c["gap_is_not_a_still_mouse"]


def test_the_stretch_is_accurate_to_about_a_pixel_and_the_recording_has_the_right_length():
    k = rep.camera_section()
    assert max(k["corner_error_px"]) < 2.5 and k["stretch_repeatable"]
    rec = k["recording"]
    assert rec["frames_in_video"] == 150 and rec["video_fps"] == pytest.approx(30.0, abs=0.5) and rec["length_error_s"] <= 0.05


def test_the_mars_summary_splits_touching_from_apart_and_names_the_keypoints():
    errors = np.zeros((4, 2, 7))
    errors[2:] = 12.0                                                  # the last two pictures have big errors
    touching = [False, False, True, True]
    s = mars_acc.summarize(errors, touching, with_parts=True)
    assert s["median_apart"] == 0.0 and s["median_touching"] == 12.0 and s["within_5"] == 0.5 and s["within_10"] == 0.5
    assert set(s["per_keypoint"]) == {"nose", "right_ear", "left_ear", "neck", "right_side", "left_side", "tail_base"}
    assert set(s["per_mouse"]) == {"black", "white"}


def test_the_degrade_step_keeps_the_picture_size_and_changes_it():
    img = (np.random.default_rng(0).random((570, 1024, 3)) * 255).astype(np.uint8)
    d = mars_acc.degrade(img)
    assert d.shape == img.shape and not np.array_equal(d, img)


def test_the_report_renders_every_section_and_says_what_is_not_measured():
    scores, labels = synthetic_sessions()
    res = {"behavior": rep.behavior_section(scores, labels, "synthetic"), "centers": rep.centers_section(), "camera": rep.camera_section()}
    text = rep.render(res)
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## Not measured"):
        assert heading in text
    assert "Not measured in this run" in text and "Not trained yet" in text
