"""Tests for scripts/investigation_feasibility.py (new file).

SYNTHETIC test fixtures only: small generated CalMS21-layout sessions and
hand-made arrays whose right answers are known. Nothing here is CalMS21
data or a scientific result.
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from live_tracker.calms21 import FPS, IMAGE_SIZE, KEYPOINTS, TORSO_KEYPOINTS, load_calms21
from live_tracker.features import compute_features
from live_tracker.investigation_data import extract_sessions
from live_tracker.quality import SUGGESTED_MIN_SCORE, flag_tracking, interpolate_short_gaps

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("feas", ROOT / "scripts" / "investigation_feasibility.py")
feas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feas)

VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
SID = "task1/train/SYNTHETIC_val_a"
N, ONSET, END = 240, 150, 170
NECK = KEYPOINTS.index("neck")


def synthetic_record(seed=0):
    """Two synthetic mice moving smoothly with 1 px jitter; attack frames
    ONSET..END-1, investigation just before, other elsewhere."""
    rng = np.random.default_rng(seed)
    kp = np.zeros((N, 2, 2, 7))
    t = np.arange(N)
    for m, (cx, cy) in enumerate(((300.0, 300.0), (700.0, 260.0))):
        x = cx + 40 * np.sin(t / 25 + m) + rng.normal(0, 1, N)
        y = cy + 30 * np.cos(t / 30 + m) + rng.normal(0, 1, N)
        for k in range(7):
            kp[:, m, 0, k] = x + 3 * k
            kp[:, m, 1, k] = y + 2 * k
    labels = np.full(N, 3)
    labels[ONSET - 40:ONSET] = 1
    labels[ONSET:END] = 0
    return {"keypoints": kp.tolist(), "scores": np.full((N, 2, 7), 0.9).tolist(),
            "annotations": labels.tolist(),
            "metadata": {"annotator-id": 0, "vocab": dict(VOCAB)}}


def session(rec):
    return {"annotator-id_0": {SID: rec}}


def pre_windows(result):
    """Pre-attack window records, minus nothing: every computed field."""
    return {L: w["retained"] for L, w in result["windows"].items()}


# ---- small helpers ----

def test_attack_runs_are_maximal_and_end_exclusive():
    assert feas.attack_runs([0, 1, 1, 0, 1, 0, 0, 1]) == [(1, 3), (4, 5), (7, 8)]
    assert feas.attack_runs([]) == [] and feas.attack_runs([1, 1]) == [(0, 2)]


def test_window_lengths_use_calms21_fps():
    assert FPS == 30.0 and feas.window_frames(1) == 30 and feas.window_frames(3) == 90
    assert feas.pre_window(30, 30) == (0, 30) and feas.pre_window(29, 30) is None


def test_usable_frames_require_both_derivative_endpoints():
    ok = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    dr = np.array([np.nan, 5.0, 5.0, 5.0, 5.0, 5.0])
    # frame 3 has a finite dist_rate but frame 2 (its other endpoint) is not ok
    assert feas.usable_frames(ok, dr, 1).tolist() == [False, True, False, False, True, True]


def test_usable_frames_with_filter_history():
    ok = np.ones(20, dtype=bool)
    ok[3] = False
    use = feas.usable_frames(ok, np.zeros(20), 15)
    assert not use[:19].any() and use[19]  # 15 clean frames needed after frame 3


def test_baseline_tiles_exclude_attack_neighbourhood_and_pre_windows():
    att = np.zeros(15, dtype=bool)
    att[[4, 10]] = True
    kept, excluded = feas.baseline_tiles(15, att, 3, 0)
    assert kept == [(0, 3), (6, 9), (12, 15)] and excluded == {(3, 6): "near_attack", (9, 12): "near_attack"}
    kept, excluded = feas.baseline_tiles(15, att, 3, 0, blocked=[(7, 9)])
    assert kept == [(0, 3), (12, 15)] and excluded[(6, 9)] == "overlaps_pre_window"
    kept, _ = feas.baseline_tiles(15, att, 3, 2)
    assert kept == []


# ---- FPS ----

def test_fps_mismatch_is_rejected():
    feas.check_fps([0, 1, 2], [0, 1 / FPS, 2 / FPS], "ok")
    with pytest.raises(feas.FPSMismatch):
        feas.check_fps([0, 1, 2], [0, 1 / 25, 2 / 25], "25 fps")


def test_loader_times_at_another_fps_stop_the_analysis(monkeypatch):
    def load_25fps(data):
        poses, labels = load_calms21(data)
        poses["time_s"] = poses["frame"] / 25.0
        labels["time_s"] = labels["frame"] / 25.0
        return poses, labels

    monkeypatch.setattr(feas, "load_calms21", load_25fps)
    with pytest.raises(feas.FPSMismatch):
        feas.analyse_session(SID, session(synthetic_record()))


# ---- causality: future coordinates must not change pre-attack evidence ----

def _perturb_future(rec, shift=80.0):
    rec = copy.deepcopy(rec)
    for f in range(ONSET, N):
        for k in range(7):
            rec["keypoints"][f][1][0][k] += shift      # mouse b jumps after onset
            rec["scores"][f][1][k] = 0.2 if f % 2 else 0.9
    return rec


def test_future_coordinates_do_not_change_pre_attack_features():
    rec = synthetic_record()
    rec["keypoints"][ONSET - 1][0][0][NECK] = float("nan")  # a gap right before onset
    before = feas.analyse_session(SID, session(rec))
    after = feas.analyse_session(SID, session(_perturb_future(rec)))
    assert pre_windows(before) == pre_windows(after)
    assert all(pre_windows(before)[L] for L in ("1s", "2s", "3s"))


def test_the_interpolation_leak_is_real_and_avoided():
    """Ava's two-sided interpolation would fill the pre-onset gap from the
    first attack frame; the truncated pipeline does not."""
    rec = synthetic_record()
    rec["keypoints"][ONSET - 1][0][0][NECK] = float("nan")

    def interpolated_x(r):
        poses, _ = load_calms21(session(r))
        flags = interpolate_short_gaps(flag_tracking(poses, frame_size=IMAGE_SIZE, score_col="score",
                                                     min_score=SUGGESTED_MIN_SCORE))
        row = flags[(flags.frame == ONSET - 1) & (flags.mouse == "a") & (flags.keypoint == "neck")]
        return float(row["x"].iloc[0])

    moved = copy.deepcopy(rec)
    for k in range(7):
        moved["keypoints"][ONSET][0][0][k] += 100.0
    assert interpolated_x(rec) != interpolated_x(moved)  # the leak exists in the default path
    a = feas.analyse_session(SID, session(rec))
    b = feas.analyse_session(SID, session(moved))
    assert pre_windows(a) == pre_windows(b)              # ...and not in ours


def test_frozen_runs_continuing_after_onset_do_not_leak():
    """flag_tracking's 'frozen' flag looks at the whole run of identical
    positions; a run that continues after onset must not mark pre-onset
    frames in our pipeline."""
    rec = synthetic_record()
    still = rec["keypoints"][ONSET - 10][0]
    for f in range(ONSET - 10, ONSET):  # 10 identical frames before onset: not frozen
        rec["keypoints"][f][0] = copy.deepcopy(still)
    later = copy.deepcopy(rec)
    for f in range(ONSET, ONSET + 10):  # ...but 20 in a row once the future is included
        later["keypoints"][f][0] = copy.deepcopy(still)

    def full_session_ok(r):
        poses, _ = load_calms21(session(r))
        feats = compute_features(flag_tracking(poses, frame_size=IMAGE_SIZE, score_col="score",
                                               min_score=SUGGESTED_MIN_SCORE),
                                 torso_keypoints=TORSO_KEYPOINTS).set_index("frame")
        return feats.loc[ONSET - 10:ONSET - 1, "tracking_ok"].tolist()

    assert full_session_ok(rec) != full_session_ok(later)   # full-session flags look ahead
    a = feas.analyse_session(SID, session(rec))
    b = feas.analyse_session(SID, session(later))
    assert pre_windows(a) == pre_windows(b)
    assert b["causality_check"]["tracking_ok_differs"] > 0  # and the report shows it


def test_missing_point_inside_a_pre_window_is_excluded_not_imputed():
    rec = synthetic_record()
    gap = ONSET - 12
    rec["keypoints"][gap][0][0][NECK] = float("nan")
    res = feas.analyse_session(SID, session(rec))
    w = res["windows"]["1s"]["retained"][0]
    assert gap in w["excluded_frames"]["not_tracking_ok"]
    assert gap in w["by_history"]["1"]["not_usable_frames"]
    assert gap + 1 in w["by_history"]["1"]["not_usable_frames"]  # its derivative crosses the gap


# ---- quality across derivative endpoints ----

def test_derivative_crossing_an_invalid_frame_is_not_usable():
    rec = synthetic_record()
    bad = 60
    rec["scores"][bad][0][NECK] = 0.05  # below min_score: neck not valid at frame 60
    poses, _ = load_calms21(session(rec))
    _, feats, _ = feas.preprocess(poses)
    ok, dr = feas.frame_arrays(feats, N)
    assert not ok[bad] and np.isfinite(dr[bad + 1])  # the centre still exists, so a rate does
    use = feas.usable_frames(ok, dr, 1)
    assert not use[bad] and not use[bad + 1] and use[bad + 2]


# ---- windows and baseline on a synthetic session ----

def test_session_windows_baseline_and_ids():
    res = feas.analyse_session(SID, session(synthetic_record()))
    assert [e["start_frame"] for e in res["attack_episodes"]] == [ONSET]
    pre = res["windows"]["1s"]["retained"][0]
    assert pre["window_id"] == "SYNTHETIC_val_a:pre:1s:150"
    assert (pre["start_frame"], pre["end_frame"]) == (120, 150)
    assert pre["labels"]["investigation"] == 30 and pre["preprocessing"] == "truncated_at_onset"
    base = res["baseline"]["1s"]["C0"]
    spans = [(r["start_frame"], r["end_frame"]) for r in base["retained"]]
    assert all(b <= 120 or a >= 150 for a, b in spans)            # no overlap with the pre-window
    assert all(b <= ONSET or a >= END for a, b in spans)         # no attack frame
    assert base["excluded_counts"] == {"overlaps_pre_window": 1, "near_attack": 1}


def test_anomaly_scan_reports_a_planted_collapse_unverified():
    rec = synthetic_record()
    f = 90
    for k in range(7):  # mouse b's keypoints land on mouse a for one frame
        rec["keypoints"][f][1][0][k] = rec["keypoints"][f][0][0][k]
        rec["keypoints"][f][1][1][k] = rec["keypoints"][f][0][1][k]
        rec["scores"][f][1][k] = 0.3
    res = feas.analyse_session(SID, session(rec))
    top = res["anomalies"][0]
    assert (top["frame"], top["mouse"]) in {(f, "b"), (f + 1, "b")}
    assert top["dist_to_other_mouse_centre_px"] < 10 and top["return_px"] < 20
    assert top["verification"] == "unverified (no video available)"


# ---- full run, provenance and the generated report ----

@pytest.fixture
def cache(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": [], "validation": [SID], "test": ["task1/test/SYNTHETIC_x"]}))
    src = tmp_path / "src.json"
    src.write_text(json.dumps(session(synthetic_record())))
    extract_sessions(src, tmp_path / "cache", [SID], manifest)
    return tmp_path / "cache", manifest


def test_run_records_inputs_settings_and_code(cache):
    res = feas.run(*cache)
    assert res["inputs"]["files"][SID]["sha256"]
    assert res["settings"]["interpolate_short_gaps"].startswith("not used")
    assert res["settings"]["fps"] == FPS and "provisional" in res["settings"]
    assert "scripts/investigation_feasibility.py" in res["code"]["files_sha256"]


def test_generated_report_is_reproducible(cache, tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(f"intro\n{feas.GEN_BEGIN}\nSTALE-GENERATED-CONTENT\n{feas.GEN_END}\noutro\n")
    assert feas.main([str(cache[0]), "--manifest", str(cache[1]), "--write-doc", str(doc)]) == 0
    text = doc.read_text()
    assert text.startswith("intro\n") and text.endswith("\noutro\n") and "STALE-GENERATED-CONTENT" not in text
    assert feas.main([str(cache[0]), "--manifest", str(cache[1]), "--check-doc", str(doc)]) == 0
    doc.write_text(text.replace("| 1 |", "| 9 |", 1))
    assert feas.main([str(cache[0]), "--manifest", str(cache[1]), "--check-doc", str(doc)]) == 1


def test_run_refuses_an_index_that_lists_a_test_session(cache):
    out, manifest = cache
    index = json.loads((out / "extraction_index.json").read_text())
    index["sessions"]["task1/test/SYNTHETIC_x"] = dict(index["sessions"][SID])
    (out / "extraction_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="held-out test"):
        feas.run(out, manifest)


# ---- second review: nullable diagnostics, smoother memory ----

def _plant_jump(rec, frame, dx=-400.0, score=0.3):  # b lands on a, inside the image
    for k in range(7):
        rec["keypoints"][frame][1][0][k] += dx
        rec["scores"][frame][1][k] = score


def test_largest_jump_at_the_final_frame_renders_safely(tmp_path):
    rec = synthetic_record()
    _plant_jump(rec, N - 1)
    res = feas.analyse_session(SID, session(rec))
    last = [a for a in res["anomalies"] if a["frame"] == N - 1]
    assert last and last[0]["return_px"] is None  # there is no frame after the last one
    assert all(v is None for k, v in last[0]["smoother_contamination_px_s"].items())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": [], "validation": [SID], "test": []}))
    src = tmp_path / "src.json"
    src.write_text(json.dumps(session(rec)))
    extract_sessions(src, tmp_path / "cache", [SID], manifest)
    text = feas.render_markdown(feas.run(tmp_path / "cache", manifest))
    row = next(line for line in text.splitlines() if f"| {N - 1} | b |" in line)
    assert "n/a" in row and "None" not in text and "nan" not in text


def test_rounding_turns_missing_and_non_finite_values_into_none():
    assert feas._r(None) is None and feas._r(float("nan")) is None and feas._r(float("inf")) is None
    assert feas._r(1.234) == 1.23 and feas._fmt(None) == "n/a" and feas._fmt(2.0) == "2"


def test_smoother_keeps_contamination_beyond_fifteen_clean_frames():
    """A valid-looking one-frame jump (scores above the cut-off) is still in the
    smoothed dist_rate more than 15 frames later, on frames the H = 15
    criterion accepts as clean. The criterion does not remove it."""
    rec = synthetic_record()
    f = 60
    _plant_jump(rec, f)
    poses, _ = load_calms21(session(rec))
    _, feats, _ = feas.preprocess(poses)
    ok, dr = feas.frame_arrays(feats, N)
    assert ok[f - 20:f + 40].all()                      # every frame passes the quality flags
    usable15 = feas.usable_frames(ok, dr, 15)
    cont = feas.smoother_contamination(poses, dr, f, "b", lags=(16, 20, 30))
    assert usable15[f + 16] and usable15[f + 20]        # counted as clean at H = 15 ...
    assert abs(cont["16"]) > 1.0 and abs(cont["20"]) > 0.1  # ... but still contaminated
    assert abs(cont["30"]) < abs(cont["20"]) < abs(cont["16"])  # decaying, not removed
