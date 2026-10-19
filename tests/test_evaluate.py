"""Tests for evaluate.py on tiny hand-made scores where the answer is known."""

import numpy as np
import pandas as pd
import pytest

from live_tracker.evaluate import (EventParams, confusion, evaluate, event_metrics,
                                   frame_metrics, predict_labels, predicted_bouts,
                                   summary_metrics)

FPS = 10.0


def _tables(label_seq, attack, session="s1"):
    n = len(label_seq)
    attack = np.asarray(attack, dtype=float)
    scores = pd.DataFrame({"session": session, "frame": np.arange(n), "time_s": np.arange(n) / FPS,
                           "attack": attack, "mount": 0.0, "investigation": 0.0})
    labels = pd.DataFrame({"session": session, "frame": np.arange(n),
                           "time_s": np.arange(n) / FPS, "label": list(label_seq)})
    return scores, labels


def test_predict_labels_uses_other_as_remainder():
    s = pd.DataFrame({"attack": [0.6, 0.2, 0.1], "investigation": [0.1, 0.5, 0.1],
                      "mount": [0.1, 0.1, 0.1]})
    assert predict_labels(s).tolist() == ["attack", "investigation", "other"]


def test_frame_metrics_by_hand():
    true = ["attack", "attack", "other", "other", "mount"]
    pred = ["attack", "other", "attack", "other", "mount"]
    m = frame_metrics(true, pred).set_index("class")
    assert m.loc["attack", "precision"] == 0.5 and m.loc["attack", "recall"] == 0.5
    assert m.loc["attack", "f1"] == 0.5 and m.loc["attack", "support"] == 2
    assert m.loc["mount", "f1"] == 1.0
    assert np.isnan(m.loc["investigation", "f1"])          # never true, never predicted
    c = confusion(true, pred)
    assert c.loc["attack", "other"] == 1 and c.loc["other", "attack"] == 1 and c.values.sum() == 5


def test_bouts_threshold_min_duration_and_merge():
    p = np.array([0, .9, .9, .9, 0, .9, .9, 0, 0, 0, .9, 0])
    assert predicted_bouts(p, EventParams(0.5, 0.0, 0.0), FPS) == [(1, 4), (5, 7), (10, 11)]
    assert predicted_bouts(p, EventParams(0.5, 0.3, 0.0), FPS) == [(1, 4)]
    # a 1-frame gap is merged, the 3-frame gap is not
    assert predicted_bouts(p, EventParams(0.5, 0.0, 0.1), FPS) == [(1, 7), (10, 11)]
    assert predicted_bouts(p, EventParams(0.95, 0.0, 0.0), FPS) == []


def test_event_metrics_by_hand():
    # true attack bouts: frames 2-5 and 20-22; one predicted hit, one false alert
    truth = np.zeros(36, bool)
    truth[2:6] = truth[20:23] = True
    prob = np.zeros(36)
    prob[4:9] = 0.9          # overlaps the first true bout, starts 2 frames late
    prob[30:34] = 0.9        # no true bout here -> false alert
    r = event_metrics(prob, truth, EventParams(0.5, 0.2, 0.0), FPS)
    assert (r["n_pred"], r["n_true"], r["matched_pred"], r["detected_true"]) == (2, 2, 1, 1)
    assert r["false_alerts"] == 1
    # alert at frame 4 + 2 - 1 = 5, onset at 2 -> 0.3 s
    assert r["latencies_s"] == [pytest.approx(0.3)]
    assert r["hours"] == pytest.approx(36 / FPS / 3600)


def test_summary_metrics_perfect_and_flat():
    lab = ["other"] * 5 + ["attack"] * 10 + ["other"] * 5
    perfect = [0.0] * 5 + [1.0] * 10 + [0.0] * 5
    s, l = _tables(lab, perfect)
    m = summary_metrics(s, l, EventParams(0.5, 0.5, 0.0), fps=FPS)
    assert m["attack_f1"] == 1.0 and m["attack_event_recall"] == 1.0
    assert m["attack_event_precision"] == 1.0 and m["attack_false_alerts_per_hour"] == 0.0
    assert m["attack_latency_median_s"] == pytest.approx(0.4)   # waits min_duration (5 frames)
    assert m["macro_f1"] == 1.0                                  # only attack is defined
    s0, _ = _tables(lab, [0.0] * 20)
    m0 = summary_metrics(s0, l, EventParams(0.5, 0.5, 0.0), fps=FPS)
    assert m0["attack_event_recall"] == 0.0 and np.isnan(m0["attack_event_precision"])
    assert np.isnan(m0["attack_latency_median_s"])


def test_bouts_never_cross_sessions():
    # attack runs to the end of s1 and starts s2: must be two bouts, not one
    s1, l1 = _tables(["other"] * 5 + ["attack"] * 5, [0] * 5 + [1] * 5, "s1")
    s2, l2 = _tables(["attack"] * 5 + ["other"] * 5, [1] * 5 + [0] * 5, "s2")
    m = summary_metrics(pd.concat([s1, s2]), pd.concat([l1, l2]), EventParams(0.5, 0.0, 1.0), fps=FPS)
    assert m["attack_n_pred_bouts"] == 2 and m["attack_n_true_bouts"] == 2


def test_only_scored_sessions_are_evaluated():
    s1, l1 = _tables(["attack"] * 4, [1] * 4, "s1")
    _, l2 = _tables(["other"] * 4, [0] * 4, "s2")
    assert summary_metrics(s1, pd.concat([l1, l2]), fps=FPS)["n_frames"] == 4


def test_rejects_missing_frames():
    s, l = _tables(["other"] * 6, [0] * 6)
    with pytest.raises(ValueError, match="different frames"):
        summary_metrics(s.iloc[:-1], l, fps=FPS)


def test_evaluate_full_report():
    s, l = _tables(["other"] * 3 + ["attack"] * 3, [0, 0, 0, 1, 1, 1])
    summary, table, cm = evaluate(s, l, EventParams(0.5, 0.0, 0.0), FPS)
    assert "mount_event_recall" in summary and list(table["class"]) == ["attack", "investigation",
                                                                       "mount", "other"]
    assert cm.loc["attack", "attack"] == 3
