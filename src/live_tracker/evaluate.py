"""Evaluation of per-frame behavior scores against CalMS21 labels.

Two levels:
- frame level: per-class precision / recall / F1 and a confusion matrix.
  The predicted label of a frame is the largest of attack, investigation,
  mount and other, where other = 1 - (attack + investigation + mount).
- event level: frame scores -> bouts with events.threshold_events (the same
  rule the agent uses), per session, never across sessions. Then:
    match rule      a predicted bout and a true bout match if they share >= 1 frame
    precision       matched predicted bouts / predicted bouts
    recall          true bouts with a match / true bouts
    latency         alert time - true onset, in seconds, for each detected true bout;
                    the alert fires when the first matching predicted bout has
                    lasted min_duration_s (when the agent would fire). Negative = early.
    false alerts/h  predicted bouts matching no true bout, per hour of scored video

Event thresholds in EventParams are placeholders: choose them on VALIDATION
(Day 4), never on test.

Input formats: scores = INTERFACES.md §2 (session, frame, attack, mount,
investigation, ...), labels = calms21.load_calms21() labels (session, frame, label).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calms21 import FPS
from .events import threshold_events

BEHAVIORS = ("attack", "investigation", "mount")
CLASSES = BEHAVIORS + ("other",)
KEYS = ["session", "frame"]


@dataclass
class EventParams:
    """Bout rule. Placeholders until chosen on validation sessions.

    threshold       a frame is "on" when its score >= threshold
    min_duration_s  shorter bouts are dropped (and the alert waits this long)
    merge_gap_s     two bouts separated by a gap this short or shorter become one
    """
    threshold: float = 0.5
    min_duration_s: float = 0.5
    merge_gap_s: float = 0.0


def _align(scores, labels):
    """One row per session x frame with the scores and the true label."""
    sessions = scores["session"].unique()
    lab = labels[labels["session"].isin(sessions)]
    if scores.duplicated(KEYS).any():
        raise ValueError("scores have duplicate session/frame rows")
    m = lab[KEYS + ["label"]].merge(scores[KEYS + list(BEHAVIORS)], on=KEYS, how="outer",
                                    indicator=True)
    if not (m["_merge"] == "both").all():
        raise ValueError("scores and labels cover different frames of these sessions")
    return m.drop(columns="_merge").sort_values(KEYS).reset_index(drop=True)


def predict_labels(scores):
    """Per-frame label from the three behavior scores (other = 1 - their sum)."""
    p = scores[list(BEHAVIORS)].to_numpy(float)
    other = 1.0 - p.sum(axis=1, keepdims=True)
    return np.asarray(CLASSES)[np.argmax(np.hstack([p, other]), axis=1)]


def frame_metrics(y_true, y_pred, classes=CLASSES):
    """Per-class precision, recall, F1, support. NaN when undefined (0 / 0)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        n_pred, n_true = np.sum(y_pred == c), np.sum(y_true == c)
        prec = tp / n_pred if n_pred else np.nan
        rec = tp / n_true if n_true else np.nan
        f1 = 2 * tp / (n_pred + n_true) if n_pred + n_true else np.nan
        rows.append({"class": c, "precision": prec, "recall": rec, "f1": f1, "support": int(n_true)})
    return pd.DataFrame(rows)


def confusion(y_true, y_pred, classes=CLASSES):
    """Counts, rows = true label, columns = predicted label."""
    t = pd.Categorical(y_true, categories=classes)
    p = pd.Categorical(y_pred, categories=classes)
    return pd.crosstab(t, p, rownames=["true"], colnames=["predicted"], dropna=False)


def _runs(mask):
    """(start, end) index pairs of True runs; end is exclusive."""
    m = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    ends = np.flatnonzero(m & ~np.r_[m[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def predicted_bouts(prob, params=None, fps=FPS):
    """Bouts from one session's scores: threshold_events + merge rule + min duration."""
    p = params or EventParams()
    raw = threshold_events(prob, fps=fps, threshold=p.threshold, min_duration_s=0.0)
    gap = int(round(p.merge_gap_s * fps))
    merged = []
    for e in raw:
        if merged and e["start_frame"] - merged[-1][1] <= gap:
            merged[-1][1] = e["end_frame"]
        else:
            merged.append([e["start_frame"], e["end_frame"]])
    min_frames = int(np.ceil(p.min_duration_s * fps))
    return [(s, e) for s, e in merged if e - s >= max(min_frames, 1)]


def event_metrics(prob, is_behavior, params=None, fps=FPS):
    """Event-level counts for ONE session (frames in order). See module docstring."""
    p = params or EventParams()
    pred = predicted_bouts(prob, p, fps)
    true = _runs(is_behavior)
    min_frames = max(int(np.ceil(p.min_duration_s * fps)), 1)

    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    matched_pred = sum(any(overlaps(pb, tb) for tb in true) for pb in pred)
    latencies = []
    for tb in true:
        hits = [pb for pb in pred if overlaps(pb, tb)]
        if hits:
            alert_frame = hits[0][0] + min_frames - 1
            latencies.append((alert_frame - tb[0]) / fps)
    return {"n_pred": len(pred), "n_true": len(true), "matched_pred": matched_pred,
            "detected_true": len(latencies), "latencies_s": latencies,
            "false_alerts": len(pred) - matched_pred, "hours": len(prob) / fps / 3600}


def bout_metrics(aligned, behavior="attack", params=None, fps=FPS):
    """Event-level metrics over all sessions of an aligned table (see _align)."""
    tot = {"n_pred": 0, "n_true": 0, "matched_pred": 0, "detected_true": 0,
           "false_alerts": 0, "hours": 0.0}
    lat = []
    for _, part in aligned.groupby("session", sort=True):
        part = part.sort_values("frame")
        r = event_metrics(part[behavior].to_numpy(float),
                          (part["label"] == behavior).to_numpy(), params, fps)
        lat += r.pop("latencies_s")
        for k in tot:
            tot[k] += r[k]
    return {
        f"{behavior}_event_precision": tot["matched_pred"] / tot["n_pred"] if tot["n_pred"] else np.nan,
        f"{behavior}_event_recall": tot["detected_true"] / tot["n_true"] if tot["n_true"] else np.nan,
        f"{behavior}_latency_median_s": float(np.median(lat)) if lat else np.nan,
        f"{behavior}_false_alerts_per_hour": tot["false_alerts"] / tot["hours"] if tot["hours"] else np.nan,
        f"{behavior}_n_true_bouts": tot["n_true"],
        f"{behavior}_n_pred_bouts": tot["n_pred"],
    }


def summary_metrics(scores, labels, params=None, event_behaviors=("attack",), fps=FPS):
    """Flat dict of float metrics; the evaluate_fn for scripts/ablation.py.

    macro_f1 is the mean F1 of attack, investigation and mount (classes with
    no true and no predicted frames are left out of the mean).
    """
    a = _align(scores, labels)
    fm = frame_metrics(a["label"], predict_labels(a)).set_index("class")
    out = {"n_frames": len(a), "macro_f1": float(np.nanmean(fm.loc[list(BEHAVIORS), "f1"]))
           if fm.loc[list(BEHAVIORS), "f1"].notna().any() else np.nan}
    for b in BEHAVIORS:
        for m in ("precision", "recall", "f1"):
            out[f"{b}_{m}"] = float(fm.loc[b, m])
    for b in event_behaviors:
        out.update(bout_metrics(a, b, params, fps))
    return out


def evaluate(scores, labels, params=None, fps=FPS):
    """Full report: (summary dict, per-class frame table, confusion matrix)."""
    a = _align(scores, labels)
    pred = predict_labels(a)
    return (summary_metrics(scores, labels, params, BEHAVIORS, fps),
            frame_metrics(a["label"], pred), confusion(a["label"], pred))
