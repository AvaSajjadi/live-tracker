"""Accuracy and precision report for every type of data the app takes (new file).

Words used here (the same in the whole report):
  ACCURACY  how close the result is to the truth (a pixel error, or the share of frames or alerts that are right).
  PRECISION two meanings, both reported and named each time:
            (a) classification precision = right alerts / all alerts (TP / (TP + FP));
            (b) repeatability = how much the result moves when the input is the same or only slightly changed (noise).

Data types and what has ground truth:
  1. Keypoints + behavior labels (CalMS21 validation: the demo camera, "video + keypoints", the staged demo). Truth = the labels.
  2. Motion-only centers CSV. No labels exist for real files, so the truth is made: synthetic tracks with known speed and distance.
  3. Video -> keypoints (MARS on a recording). Truth = the MARS hand-labeled set. Run scripts/accuracy_mars_handlabels.py
     in the MARS environment first and give its JSON with --mars-json.
  4. Camera recording and the screen stretch. Truth = synthetic pictures with known corners and known timing.
  5. Fast mouse model (YOLO). Give the JSON of scripts/eval_yolo_pose.py with --yolo-json (not trained yet at the time of writing).

    python scripts/accuracy_precision_report.py --out docs/accuracy_precision_report.md --json outputs/accuracy_precision.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker import evaluate as ev  # noqa: E402
from live_tracker import live_session as ls  # noqa: E402
from live_tracker.calms21 import FPS  # noqa: E402
from live_tracker.rules import Rule  # noqa: E402

IGOR_ZIP = Path.home() / "Downloads" / "igor_scores_validation_model_b.zip"
IGOR_MEMBER = "scores_validation_model_b.csv"
IGOR_MEMBER_SHA256 = "ff00f6fe31d9348d177f7c548a21ce13cae8c30560d5234ca11f16e42a008676"
DEMO_SESSION = "mouse064"
BOOTSTRAPS = 2000


# ---------------------------------------------------------------- 1. keypoints and behavior labels

def load_scores(zip_path=IGOR_ZIP, fallback_csv=ROOT / "outputs" / "scores_validation_model_b.csv"):
    """(scores, where they came from). Igor's delivered file if its checksum matches, else our own retrained scores."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            data = z.read(IGOR_MEMBER)
        if hashlib.sha256(data).hexdigest() == IGOR_MEMBER_SHA256:
            return pd.read_csv(io.BytesIO(data)), f"Igor's delivered scores ({IGOR_MEMBER}, checksum matched)"
    except (OSError, KeyError, zipfile.BadZipFile):
        pass
    return pd.read_csv(fallback_csv), f"our retrained model B scores ({Path(fallback_csv).name}); Igor's zip was not usable"


def attack_rule():
    r = ev.ATTACK_RULE_V1
    return Rule("V1", "attack", min_duration_s=r["min_duration_s"], alert_threshold=r["alert_threshold"],
                ask_threshold=r["ask_threshold"], alert_mode=r["alert_mode"], status="approved")


def frame_accuracy(scores, labels):
    a = ev._align(scores, labels)
    pred = ev.predict_labels(a)
    ok = (a["label"] == pred)
    return float(ok.mean()), int(ok.sum()), int(len(ok))


def bootstrap_ci(counts, numerator, denominator, n=BOOTSTRAPS, seed=0):
    """95% interval of sum(numerator) / sum(denominator) when the SESSIONS are drawn again with replacement."""
    rng = np.random.default_rng(seed)
    num, den = np.array([c[numerator] for c in counts], float), np.array([c[denominator] for c in counts], float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(counts), len(counts))
        d = den[i].sum()
        if d > 0:
            vals.append(num[i].sum() / d)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def behavior_section(scores, labels, source):
    rule = attack_rule()
    labels = labels[labels["session"].isin(set(scores["session"]))]
    summary, per_class, _ = ev.evaluate(scores, labels)
    acc, n_ok, n_all = frame_accuracy(scores, labels)
    ev_all, bouts = ev.monitor_event_metrics(scores, labels, rule)
    ev_again, _ = ev.monitor_event_metrics(scores, labels, rule)
    per_session, counts = [], []
    for sid, part in scores.groupby("session"):
        lab = labels[labels["session"] == sid]
        a, ok, n = frame_accuracy(part, lab)
        m, _ = ev.monitor_event_metrics(part, lab, rule)
        tp = int(round(m["event_precision"] * m["n_events"])) if m["n_events"] else 0
        per_session.append({"session": sid.split("/")[-1].replace("_task1_annotator1", ""), "frames": n, "frame_accuracy": round(a, 4),
                            "true_attack_bouts": m["n_true_bouts"], "detected": m["detected_true"], "alerts": m["n_events"],
                            "right_alerts": tp})
        counts.append({"ok": ok, "n": n, "events": m["n_events"], "tp": tp, "true": m["n_true_bouts"], "det": m["detected_true"]})
    demo = [p for p in per_session if p["session"].startswith(DEMO_SESSION)]
    return {
        "source": source, "sessions": int(scores["session"].nunique()), "frames": n_all,
        "frame_accuracy": round(acc, 4), "frame_accuracy_ci95": [round(x, 4) for x in bootstrap_ci(counts, "ok", "n")],
        "macro_f1": round(float(summary["macro_f1"]), 4),
        "per_class": per_class.round(4).to_dict("records"),
        "rule": {k: ev.ATTACK_RULE_V1[k] for k in ("ask_threshold", "min_duration_s", "alert_threshold")},
        "alerts": ev_all["n_events"], "true_attack_bouts": ev_all["n_true_bouts"], "detected_true_bouts": ev_all["detected_true"],
        "alert_precision": round(float(ev_all["event_precision"]), 4),
        "alert_precision_ci95": [round(x, 4) for x in bootstrap_ci(counts, "tp", "events")],
        "alert_recall": round(float(ev_all["event_recall"]), 4),
        "alert_recall_ci95": [round(x, 4) for x in bootstrap_ci(counts, "det", "true")],
        "false_alerts_per_hour": round(float(ev_all["false_alerts_per_hour"]), 2),
        "start_offset_median_s": round(float(ev_all["start_offset_median_s"]), 3),
        "fire_offset_median_s": round(float(ev_all["fire_offset_median_s"]), 3),
        "repeatable": bool(all(ev_all[k] == ev_again[k] or (np.isnan(ev_all[k]) and np.isnan(ev_again[k])) for k in ev_all)),
        "per_session": per_session, "demo_session": demo[0] if demo else None,
    }


# ---------------------------------------------------------------- 2. motion-only centers CSV (synthetic truth)

def synthetic_centers(noise_px, seed=0, n=300, speed_px_per_frame=3.0):
    """Mouse a walks along x at a known speed, mouse b stands at (400, 200). Truth: speed_a = 90 px/s, speed_b = 0."""
    rng = np.random.default_rng(seed)
    f = np.arange(n)
    t = pd.DataFrame({"x_a": 100.0 + speed_px_per_frame * f, "y_a": 150.0, "x_b": 400.0, "y_b": 200.0})
    truth_dist = np.hypot(t["x_a"] - t["x_b"], t["y_a"] - t["y_b"])
    noisy = t + rng.normal(0.0, noise_px, t.shape) if noise_px else t
    return noisy, truth_dist.to_numpy()


def centers_section():
    rows = []
    for noise in (0.0, 0.5, 1.0, 2.0):
        table, truth_dist = synthetic_centers(noise)
        s = ls.centers_series(table, FPS).iloc[10:]                      # skip the start-up of the 5-frame average
        true_speed = 3.0 * FPS
        rows.append({"noise_px": noise,
                     "speed_error_median_px_s": round(float(np.median(np.abs(s["speed_a"] - true_speed))), 2),
                     "speed_error_median_percent": round(float(np.median(np.abs(s["speed_a"] - true_speed)) / true_speed * 100), 2),
                     "speed_spread_px_s": round(float(s["speed_a"].std()), 2),                 # precision (repeatability under noise)
                     "still_mouse_reads_px_s": round(float(np.median(s["speed_b"])), 2),       # a still mouse should read 0
                     "distance_error_median_px": round(float(np.median(np.abs(s["distance"].to_numpy() - truth_dist[10:]))), 2)})
    clean, _ = synthetic_centers(0.0)
    again = ls.centers_series(clean, FPS)
    gap = clean.copy()
    gap.loc[100:104, ["x_a", "y_a"]] = np.nan
    g = ls.centers_series(gap, FPS)
    return {"rows": rows, "repeatable": bool(again.equals(ls.centers_series(clean, FPS))),
            "gap_is_not_a_still_mouse": bool(g["speed_a"].iloc[100:105].isna().all() and not g["tracking_ok"].iloc[100:105].any())}


# ---------------------------------------------------------------- 4. camera recording and the screen stretch (synthetic truth)

def stretch_geometry_error():
    """Put 4 small bright squares in a 640 x 480 picture at the places where 4 chosen points of the 1024 x 570 frame come from
    (found with the inverse of the same stretch). After the stretch each square must sit on its chosen point. Returns the position
    error of each square in output pixels. The squares are inside the frame, so none is cut off by the edge."""
    import cv2
    from live_tracker import screen_map as sm
    w, h = 640, 480
    corners = ((0.20, 0.25), (0.80, 0.20), (0.85, 0.80), (0.15, 0.75))
    ow, oh = sm.IMAGE_SIZE
    src = np.float32([(x * w, y * h) for x, y in corners])
    dst = np.float32([(0, 0), (ow, 0), (ow, oh), (0, oh)])
    inverse = cv2.getPerspectiveTransform(dst, src)
    targets = np.float32([(120, 80), (ow - 120, 80), (ow - 120, oh - 80), (120, oh - 80), (ow / 2, oh / 2)])
    where = cv2.perspectiveTransform(targets.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    img = np.zeros((h, w, 3), np.uint8)
    for x, y in where:
        cx, cy = int(round(x)), int(round(y))
        img[cy - 2:cy + 3, cx - 2:cx + 3] = 255
    out = sm.warp(img, corners)
    gray = out.mean(axis=2)
    errs = []
    for ex, ey in targets:
        x0, x1, y0, y1 = int(ex) - 30, int(ex) + 31, int(ey) - 30, int(ey) + 31
        patch = gray[y0:y1, x0:x1]
        ys, xs = np.nonzero(patch > 40)
        if len(xs) == 0:
            errs.append(float("nan"))
            continue
        wgt = patch[ys, xs]
        errs.append(float(np.hypot((xs * wgt).sum() / wgt.sum() + x0 - ex, (ys * wgt).sum() / wgt.sum() + y0 - ey)))
    deterministic = bool(np.array_equal(out, sm.warp(img, corners)))
    return errs, deterministic


def recording_timing_error():
    """Feed the recorder a 15 pictures per second camera with timing jitter for 5 s: the video must have 30 fps and the right length."""
    import cv2
    from live_tracker import camera_live as cl
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.mp4"
        rec = cl.Recorder(path, None)
        img = np.full((60, 80, 3), 90, np.uint8)
        t = 100.0
        while t < 105.0:
            rec.add(img, t)
            t += 1 / 15 + rng.normal(0, 0.004)
        info = rec.close()
        cap = cv2.VideoCapture(str(path))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()                                   # Windows cannot delete a file that is still open
    expected = 5.0 * 30
    return {"frames_in_video": n, "frames_expected": expected, "length_error_s": round(abs(n - expected) / 30.0, 3),
            "video_fps": round(float(fps), 2), "recorder_reported_frames": info["frames"]}


def camera_section():
    errs, deterministic = stretch_geometry_error()
    return {"corner_error_px": [round(e, 2) for e in errs], "stretch_repeatable": deterministic, "recording": recording_timing_error()}


# ---------------------------------------------------------------- report

def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def ci(x):
    return f"{pct(x[0])} to {pct(x[1])}"


def render(res):
    L = ["# Accuracy and precision report", "",
         "Made by `scripts/accuracy_precision_report.py`. **Accuracy** = how close to the truth. **Precision** has two meanings, "
         "named each time: **(a) right alerts out of all alerts** and **(b) repeatability** (how much a result moves when the input is the "
         "same or slightly noisy). Every number below was measured; numbers that could not be measured are marked.", ""]
    b = res["behavior"]
    L += ["## 1. Keypoints with behavior labels (demo camera, video + keypoints, staged demo)", "",
          f"Scores: {b['source']}. Truth: the CalMS21 labels of the same {b['sessions']} VALIDATION sessions ({b['frames']:,} frames). "
          "The test split was not used. These are the same scores and the same rule the app uses.", "",
          "| Measure | Value | 95% interval over sessions |", "|---|---|---|",
          f"| Frame accuracy (all 4 classes) | {pct(b['frame_accuracy'])} | {ci(b['frame_accuracy_ci95'])} |",
          f"| Macro F1 (attack, investigation, mount) | {b['macro_f1']:.3f} | |",
          f"| **Alert precision** (right alerts / all alerts) | {pct(b['alert_precision'])} ({b['alerts']} alerts) | {ci(b['alert_precision_ci95'])} |",
          f"| **Alert recall** (attack bouts found / all) | {pct(b['alert_recall'])} ({b['detected_true_bouts']} of {b['true_attack_bouts']}) | {ci(b['alert_recall_ci95'])} |",
          f"| False alerts per hour | {b['false_alerts_per_hour']} | |",
          f"| Alert start versus true start, median | {b['start_offset_median_s']:+.2f} s (positive = late) | |",
          f"| Alert fires versus true start, median | {b['fire_offset_median_s']:+.2f} s | |",
          f"| Repeatability: the same input twice gives the same alerts | {'yes' if b['repeatable'] else 'NO'} | |", "",
          f"Rule used: attack, ask above {b['rule']['ask_threshold']}, alert at {b['rule']['alert_threshold']}, at least "
          f"{b['rule']['min_duration_s']} s.", "", "Per class (frame level):", "",
          "| Class | Precision | Recall | F1 |", "|---|---|---|---|"]
    for r in b["per_class"]:
        L.append(f"| {r['class']} | {pct(r['precision'])} | {pct(r['recall'])} | {r['f1']:.3f} |")
    d = b["demo_session"]
    if d:
        L += ["", f"The demo session (`{d['session']}`): frame accuracy {pct(d['frame_accuracy'])}; {d['detected']} of {d['true_attack_bouts']} "
              f"attack bouts found; {d['right_alerts']} of {d['alerts']} alerts right."]
    L += ["", "Only 4 of the 14 validation sessions have attack bouts, and two sessions hold most of them, so the intervals are wide. "
          "The rule was chosen on these same sessions, so the numbers are a little optimistic.", ""]
    c = res["centers"]
    L += ["## 2. Motion-only centers CSV", "",
          "Real centers files have no labels, so the truth is made: mouse a walks at exactly 90 px/s, mouse b stands still, the true distance is known. "
          "Noise is Gaussian jitter added to the positions.", "",
          "| Jitter (px) | Speed error, median (px/s) | Speed error (%) | **Precision (b): spread of the speed (px/s)** | A still mouse reads (px/s) | Distance error (px) |",
          "|---|---|---|---|---|---|"]
    for r in c["rows"]:
        L.append(f"| {r['noise_px']} | {r['speed_error_median_px_s']} | {r['speed_error_median_percent']} | {r['speed_spread_px_s']} | "
                 f"{r['still_mouse_reads_px_s']} | {r['distance_error_median_px']} |")
    L += ["", f"The same input twice gives the same numbers: {'yes' if c['repeatable'] else 'NO'}. A gap in tracking is shown as missing and not as a still mouse: "
          f"{'yes' if c['gap_is_not_a_still_mouse'] else 'NO'}. **Warning:** with jitter, a mouse that does not move still reads as moving "
          "(the table shows how fast), so noisy centers files show motion that is not real. There are no behavior scores or alerts for this data type.", ""]
    m = res.get("mars")
    L += ["## 3. Video to keypoints with MARS (a recording, or the demo video)", ""]
    if m:
        L += [f"Truth: the MARS hand-labeled pictures (median of 5 workers), {m['pictures']} test pictures, in pixels of the 1024 x 570 picture "
              "(a mouse body is about 195 px long).", "",
              "| Measure | Original picture | Degraded picture (small, blurry, JPEG, dim) |", "|---|---|---|",
              f"| **Accuracy**: median error, all keypoints (px) | {m['clean']['median_all']} | {m['degraded']['median_all']} |",
              f"| Share of keypoints within 5 px | {pct(m['clean']['within_5'])} | {pct(m['degraded']['within_5'])} |",
              f"| Share within 10 px | {pct(m['clean']['within_10'])} | {pct(m['degraded']['within_10'])} |",
              f"| Touching mice: median error (px) | {m['clean']['median_touching']} | {m['degraded']['median_touching']} |",
              f"| Apart: median error (px) | {m['clean']['median_apart']} | {m['degraded']['median_apart']} |",
              f"| **Precision (b)**: how far a keypoint moves when the picture is degraded, median (px) | | {m['shift_median_px']} |",
              f"| Repeatability: the same pictures twice, largest difference (px) | {m['repeat_max_px']} | |", "",
              "Median error per keypoint, original picture (px): " + ", ".join(f"{k} {v}" for k, v in m["clean"]["per_keypoint"].items()) + ".", "",
              "Per mouse (original): black " + str(m["clean"]["per_mouse"]["black"]) + " px, white " + str(m["clean"]["per_mouse"]["white"]) + " px.", ""]
    else:
        L += ["**Not measured in this run.** Run `scripts/accuracy_mars_handlabels.py` in the MARS environment and pass `--mars-json`.", ""]
    k = res["camera"]
    L += ["## 4. Camera recording and the screen stretch", "",
          "Truth is synthetic: bright squares placed at known points of a screen in a 640 x 480 picture, and a camera that gives "
          "15 pictures per second with timing jitter.", "",
          f"- **Accuracy of the stretch:** 5 test squares (4 near the corners and 1 in the middle of the frame) land within {max(k['corner_error_px']):.2f} px "
          f"of their right place in the 1024 x 570 frame (errors: {', '.join(str(e) for e in k['corner_error_px'])} px; the squares are placed to the nearest "
          "whole pixel of the 640 x 480 picture, which already costs up to about 1 px).",
          f"- **Repeatability:** the same picture stretched twice is identical: {'yes' if k['stretch_repeatable'] else 'NO'}.",
          f"- **Recording length accuracy:** a 5 s recording has {k['recording']['frames_in_video']} frames (expected {int(k['recording']['frames_expected'])}), "
          f"an error of {k['recording']['length_error_s']} s, and the video says {k['recording']['video_fps']} frames per second.", "",
          "A real camera adds blur, glare and color, which are not measured here.", ""]
    y = res.get("yolo")
    L += ["## 5. Fast mouse model (YOLO pose)", ""]
    if y:
        a = y["all"]
        L += [f"Accuracy on {y['pictures']} test pictures: both mice found in {pct(a['found_share'])}; median error {a.get('median_error_all_px')} px; "
              f"within 5 px {pct(a.get('share_within_5px'))}; within 10 px {pct(a.get('share_within_10px'))}. Touching mice: "
              f"found {pct(y['touching']['found_share'])}, median error {y['touching'].get('median_error_all_px')} px. Apart: "
              f"found {pct(y['apart']['found_share'])}, median error {y['apart'].get('median_error_all_px')} px.", ""]
    else:
        L += ["**Not trained yet, so not measured.** When `best.pt` exists, run `scripts/eval_yolo_pose.py` and pass its JSON with `--yolo-json`.", ""]
    L += ["## Not measured (and why)", "",
          "- Behavior accuracy on **your own recordings**: there are no labels for them. Only keypoint accuracy on labeled pictures is measurable.",
          "- The **live** chain (camera, fast model, scores, rules) with everything running at once: not built end to end yet.",
          "- The **staged demo's** sync with the iPad (GO and play pressed by hand): a human timing, not measurable here. "
          "The alert timing against the labels is in section 1.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "accuracy_precision_report.md"))
    ap.add_argument("--json", default=str(ROOT / "outputs" / "accuracy_precision.json"))
    ap.add_argument("--labels", default=str(ROOT / "data" / "processed" / "labels.parquet"))
    ap.add_argument("--mars-json")
    ap.add_argument("--yolo-json")
    a = ap.parse_args(argv)
    scores, source = load_scores()
    labels = pd.read_parquet(a.labels)
    res = {"behavior": behavior_section(scores, labels, source), "centers": centers_section(), "camera": camera_section()}
    if a.mars_json and Path(a.mars_json).is_file():
        res["mars"] = json.loads(Path(a.mars_json).read_text(encoding="utf-8"))
    if a.yolo_json and Path(a.yolo_json).is_file():
        res["yolo"] = json.loads(Path(a.yolo_json).read_text(encoding="utf-8"))
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    Path(a.out).write_text(render(res), encoding="utf-8")
    print("report:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
