"""Tests for the experimental get_window_features / check_tracking_quality tools (new file).

Synthetic parts use SYNTHETIC test fixtures in the CalMS21 layout, published
through the real Stage 1 extraction code; nothing synthetic is CalMS21 data.
The real-data tests read the verified validation cache under data/raw and are
skipped when it is not present.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import live_tracker.investigation_windows as iw
from live_tracker.calms21 import FPS, TORSO_KEYPOINTS
from live_tracker.features import FeatureParams, dyadic_features
from live_tracker.filters import one_euro_filter
from live_tracker.investigation_contracts import (CONTRACT_VERSION, EvidencePolicy, ForbiddenEvidence,
                                                  baseline_window_id, call_tool, episode_id, pre_window_id)
from live_tracker.investigation_data import INDEX_NAME, extract_sessions, verified_sessions
from live_tracker.investigation_events import CalMS21InvestigationTools
from live_tracker.quality import flag_tracking

ROOT = Path(__file__).resolve().parents[1]
VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
VAL_A, VAL_B, VAL_C = (f"task1/train/SYNTHETIC_val_{x}" for x in "abc")
TRAIN, TEST = "task1/train/SYNTHETIC_train", "task1/test/SYNTHETIC_test"
MANIFEST = {"train": [TRAIN], "validation": [VAL_A, VAL_B, VAL_C], "test": [TEST]}
N = 400
ATTACKS = {VAL_A: [(100, 120), (130, 140), (300, 320)], VAL_B: [(200, 230)], VAL_C: []}
INVESTIGATION = {VAL_A: [(80, 100)], VAL_B: [], VAL_C: [(0, 5)]}
SEEDS = {VAL_A: 1, VAL_B: 2, VAL_C: 3, TRAIN: 4}
K = iw.SMOOTHER_MEMORY_FRAMES
TORSO_IDX = [3, 4, 5]  # neck, left_hip, right_hip in calms21.KEYPOINTS


def arrays(sid, n=N):
    """Keypoints (n, mouse, xy, kp) with sub-pixel jitter (never frozen) and an
    approach by mouse b in frames 70..100; scores 0.9."""
    rng = np.random.default_rng(SEEDS[sid])
    kp = np.zeros((n, 2, 2, 7))
    bx = np.full(n, 700.0)
    bx[70:100] = np.linspace(700, 400, 30)
    bx[100:] = 400.0
    for k in range(7):
        kp[:, 0, 0, k] = 300.0 + 3 * k
        kp[:, 0, 1, k] = 300.0 + 2 * k
        kp[:, 1, 0, k] = bx + 3 * k
        kp[:, 1, 1, k] = 300.0 + 2 * k
    kp += rng.normal(0, 0.5, kp.shape)
    return kp, np.full((n, 2, 7), 0.9)


def record(sid, kp, sc, n=N, attacks=None):
    labels = np.full(n, VOCAB["other"])
    for a, b in INVESTIGATION.get(sid, []):
        labels[a:b] = VOCAB["investigation"]
    for a, b in (attacks or ATTACKS).get(sid, [(5, 50)]):
        labels[a:b] = VOCAB["attack"]
    return {"keypoints": kp.tolist(), "scores": sc.tolist(), "annotations": labels.tolist(),
            "metadata": {"annotator-id": 0, "vocab": dict(VOCAB)}}


def build(tmp_path, change=None, sessions=(VAL_A, VAL_B, VAL_C), name="cache", attacks=None, edit=None):
    """Publish the fixture (with change(sid, kp, sc) applied, then edit(sid, record) on the JSON
    record) and return (tools, policy, cache, manifest)."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST))
    recs = {}
    for sid in (VAL_A, VAL_B, VAL_C, TRAIN):
        kp, sc = arrays(sid)
        if change is not None:
            change(sid, kp, sc)
        recs[sid] = record(sid, kp, sc, attacks=attacks)
        if edit is not None:
            edit(sid, recs[sid])
    src = tmp_path / f"{name}_source.json"
    src.write_text(json.dumps({"annotator-id_0": recs}))
    cache = tmp_path / name
    extract_sessions(src, cache, list(sessions), manifest)
    policy = EvidencePolicy(MANIFEST, allow_synthetic=True)
    return CalMS21InvestigationTools.for_validation_cache(cache, manifest, policy), policy, cache, manifest


@pytest.fixture
def env(tmp_path):
    return build(tmp_path)


def ep(sid, frame, behavior="attack", source="validation_annotation"):
    return episode_id(source, behavior, sid, frame)


def freq(episodes, length=20, gap=0, baseline=None, features=("dist_rate",)):
    return {"contract_version": CONTRACT_VERSION, "episode_ids": list(episodes),
            "pre_event": {"length_frames": length, "gap_frames": gap}, "baseline": baseline,
            "features": list(features)}


def base(length=20, clearance=10, max_windows=100):
    return {"strategy": "same_session_tiles", "length_frames": length, "clearance_frames": clearance,
            "max_windows_per_session": max_windows}


def qreq(window_ids, history=(1, 15)):
    return {"contract_version": CONTRACT_VERSION, "window_ids": list(window_ids),
            "history_frames": list(history)}


def features(env, **kw):
    return call_tool(env[0], "get_window_features", freq(**kw), env[1])


def quality(env, window_ids, history=(1, 15)):
    return call_tool(env[0], "check_tracking_quality", qreq(window_ids, history), env[1])


def windows(res, kind=None):
    return {w["window_id"]: w for w in res["measurements"]["windows"] if kind in (None, w["kind"])}


def session(env, sid):
    verified, _ = verified_sessions(env[2], env[3])
    return iw.load_session(sid, dict(verified)[sid])


def reference(sess, start, end, feature="dist_rate"):
    """Ava's pipeline on the session truncated at `end`, summarised over frames
    [start, end) with the endpoint rule (independent of the tool's shortcuts)."""
    _, ok, feats, _ = iw._pipeline(sess.torso[sess.torso["frame"] < end])
    usable = ok[start:end] & np.r_[ok[start - 1] if start else False, ok[start:end - 1]]
    v = feats[feature][start:end]
    v = v[usable & np.isfinite(v)]
    return int(v.size), float(np.median(v)), float(np.mean(v))


# ---- pure parts ----

def test_smoother_memory_bound_matches_the_one_euro_filter():
    """An input weighs < 1% after K samples, and K is the smallest such count."""
    p = FeatureParams()
    tau = 1 / (2 * np.pi * p.smooth_kwargs["min_cutoff"])
    alpha = 1 / (1 + tau * FPS)
    assert K == 25 and (1 - alpha) ** K < 0.01 <= (1 - alpha) ** (K - 1)
    x = np.zeros(80)
    x[10] = 1.0                       # impulse after a steady start
    y = one_euro_filter(x, np.arange(80) / FPS, **p.smooth_kwargs)
    assert y[10 + K] < 0.01 * y[10] / alpha and y[10 + K - 1] > 0
    with pytest.raises(ValueError):
        iw.smoother_memory_frames(FeatureParams(smooth="ema"))


@pytest.mark.parametrize("feature", sorted(iw.FEATURE_LOOKBACK))
def test_feature_lookback_matches_avas_features(feature):
    """Without smoothing, a feature at t reads exactly FEATURE_LOOKBACK frames back."""
    kp, sc = arrays(VAL_A, 120)
    rng = np.random.default_rng(0)
    kp[:, 1, 0, :] += np.cumsum(rng.normal(0, 3, 120))[:, None]   # varying speeds (correlation defined)
    from live_tracker.calms21 import _session_to_long
    poses = _session_to_long(VAL_A, {"keypoints": kp, "scores": sc})
    poses = poses[poses["keypoint"].isin(TORSO_KEYPOINTS)]
    flags = flag_tracking(poses, **iw.FLAG_SETTINGS)
    params = FeatureParams(smooth=None)
    t, lb = 100, iw.FEATURE_LOOKBACK[feature]

    def value(frame, how):
        f = flags.copy()
        b = (f["frame"] == frame) & (f["mouse"] == "b")
        if how == "collapse":     # b's points onto a's (distance ~0, a large step)
            f.loc[b, ["x", "y"]] = f.loc[(f["frame"] == frame) & (f["mouse"] == "a"), ["x", "y"]].to_numpy()
        else:
            f.loc[b, "valid"] = False
        return dyadic_features(f, params, TORSO_KEYPOINTS).set_index("frame")[feature][t]

    def same(x, y):  # running sums carry ~1e-14 rounding from every earlier frame
        return (np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9

    plain = dyadic_features(flags, params, TORSO_KEYPOINTS).set_index("frame")[feature][t]
    for how in ("collapse", "invalid"):          # nothing before the lookback is read
        assert same(value(t - lb - 1, how), plain)
    # the lookback is tight: close_time_s counts distances from t-29 (frame t-30 only
    # enters through the coverage of step t-29); every other feature reads t - lb itself
    tight = t - lb + 1 if feature == "close_time_s" else t - lb
    assert not same(value(tight, "collapse"), plain)


def frozen_across(sid_target, frame, length=30):
    """A change that copies one frame's coordinates onto the next `length`
    frames (identical positions = a frozen run starting at `frame`)."""
    def change(sid, kp, sc):
        if sid == sid_target:
            kp[frame:frame + length] = kp[frame]
    return change


def test_prefix_arrays_equal_the_truncated_pipeline(tmp_path, monkeypatch):
    env = build(tmp_path, frozen_across(VAL_A, 245))            # a frozen run crossing frame 250
    sess = session(env, VAL_A)
    calls = []
    real = iw._pipeline
    monkeypatch.setattr(iw, "_pipeline", lambda t, *a: calls.append(len(t)) or real(t, *a))
    for end in (37, 120, 250, 251, 262, 399):
        ok, feats, valid = iw.prefix_arrays(sess, end)
        _, ref_ok, ref_feats, ref_valid = real(sess.torso[sess.torso["frame"] < end])
        assert np.array_equal(ok[:end], ref_ok) and np.array_equal(valid, ref_valid)
        for f in feats:
            assert np.array_equal(feats[f][:end], ref_feats[f], equal_nan=True)
    # the frozen run 245..274 cut at 250 / 251 is shorter than 15 frames, so the truncated
    # flags differ from the full-session ones -> rerun; cut at 262 it is 17 frames long and
    # frozen either way, so the full-session values are already the truncated ones
    assert len(calls) == 2
    full_ok = sess.ok[245:250]
    assert not full_ok.any() and iw.prefix_arrays(sess, 250)[0][245:250].all()  # look-ahead removed


def test_history_start_counts_valid_samples(tmp_path):
    env = build(tmp_path)
    sess = session(env, VAL_A)
    assert iw.history_start(sess.centre_valid, 200) == 200 - K
    assert iw.history_start(sess.centre_valid, 10) == 0

    def hide(sid, kp, sc):
        if sid == VAL_A:
            sc[190:195, 1, :] = 0.01      # every keypoint of mouse b: no body centre
    sess = session(build(tmp_path, hide, name="hidden"), VAL_A)
    assert iw.history_start(sess.centre_valid, 200) == 200 - K - 5


# ---- get_window_features ----

def test_pre_event_values_equal_the_truncated_pipeline(env):
    res = features(env, episodes=[ep(VAL_A, 100), ep(VAL_A, 300), ep(VAL_B, 200)], length=20, gap=3)
    assert res["status"] == "ok"
    for sid, onset in ((VAL_A, 100), (VAL_A, 300), (VAL_B, 200)):
        w = windows(res)[pre_window_id(ep(sid, onset), 20, 3)]
        assert (w["start_frame"], w["end_frame"]) == (onset - 23, onset - 3)
        assert w["start_s"] == (onset - 23) / FPS and w["preprocessing"]["truncated_at_onset"] is True
        f = w["features"]["dist_rate"]
        assert (f["n_usable"], f["median"], f["mean"]) == reference(session(env, sid), onset - 23, onset - 3)
        assert w["status"] == "included"
    approach = windows(res)[pre_window_id(ep(VAL_A, 100), 20, 3)]["features"]["dist_rate"]
    assert approach["median"] < -200                   # b closes ~10 px / frame before onset 100


def test_future_frames_never_change_pre_event_results(tmp_path):
    """Adversarial: everything at and after onset is replaced, including a frozen
    run that starts one frame BEFORE onset (a look-ahead flag would mark that
    pre-onset frame frozen), scores collapse and positions explode."""
    onset = 300

    def poison(sid, kp, sc):
        if sid == VAL_A:
            kp[onset:onset + 20] = kp[onset - 1]          # identical to frame onset-1
            kp[onset + 20:] += 5000.0
            sc[onset + 20:] = 0.01
    clean, dirty = build(tmp_path, name="clean"), build(tmp_path, poison, name="dirty")
    for gap in (0, 4):
        req = dict(episodes=[ep(VAL_A, onset)], length=30, gap=gap,
                   features=("dist_center", "dist_rate", "rel_speed", "close_time_s", "speed_corr_w"))
        a, b = features(clean, **req), features(dirty, **req)
        assert a["measurements"] == b["measurements"]
    # a non-causal computation WOULD differ: the full-session flags see the run
    assert not session(dirty, VAL_A).ok[onset - 1] and session(clean, VAL_A).ok[onset - 1]


def test_future_frames_never_change_exclusions(tmp_path):
    """The history start counts body-centre samples from the TRUNCATED flags: a
    frozen run 290..319 crossing the window end 300 would (with look-ahead)
    invalidate 290..299, push the history back past the attack 265..270 and
    exclude the window; computed causally it is included."""
    attacks = {**ATTACKS, VAL_A: [(100, 120), (265, 270), (300, 320)]}
    env = build(tmp_path, frozen_across(VAL_A, 290), attacks=attacks)
    assert not session(env, VAL_A).ok[290:300].any()          # full-session flags look ahead
    res = features(env, episodes=[ep(VAL_A, 300)], length=1, gap=0)
    assert windows(res)[pre_window_id(ep(VAL_A, 300), 1, 0)]["status"] == "included"


def test_future_frames_never_change_baseline_results(tmp_path):
    def poison(sid, kp, sc):
        if sid == VAL_A:
            kp[379:] = kp[379]                              # frozen from inside the tile [360, 380)
    clean, dirty = build(tmp_path, name="clean"), build(tmp_path, poison, name="dirty")
    req = dict(episodes=[ep(VAL_A, 300)], length=20, baseline=base(20, 10), features=("dist_rate",))
    wid = baseline_window_id("validation_annotation", "attack", VAL_A, 360, 20)
    assert windows(features(clean, **req))[wid] == windows(features(dirty, **req))[wid]


def test_window_starting_before_session_is_clipped_and_excluded(env):
    res = features(env, episodes=[ep(VAL_A, 100)], length=150, gap=0)
    w = windows(res)[pre_window_id(ep(VAL_A, 100), 150, 0)]
    assert (w["start_frame"], w["end_frame"], w["status"]) == (0, 100, "excluded")
    assert "starts_before_session" in w["exclusion_reasons"]
    assert w["features"]["dist_rate"]["n_frames"] == 100
    res = features(env, episodes=[ep(VAL_A, 100)], length=5, gap=200)
    w = windows(res)[pre_window_id(ep(VAL_A, 100), 5, 200)]
    assert (w["start_frame"], w["end_frame"]) == (0, 0) and w["features"]["dist_rate"]["median"] is None
    assert res["status"] == "no_data"


def test_earlier_attack_in_the_feature_history_excludes_a_pre_window(env):
    """Onset 130: the window [125, 130) holds no attack frame (the contract
    validator would accept it), but its history reaches the run ending at 120."""
    res = features(env, episodes=[ep(VAL_A, 130), ep(VAL_A, 300)], length=5, gap=0)
    w = windows(res)
    assert w[pre_window_id(ep(VAL_A, 130), 5, 0)]["exclusion_reasons"] == ["overlaps_event"]
    assert w[pre_window_id(ep(VAL_A, 300), 5, 0)]["status"] == "included"
    # the history depends on the requested features: window [170, 175) before onset 300 is
    # 1 + 25 frames of history clear of the run ending at 140 for dist_rate, but the 1-s
    # trailing window of speed_corr_w (30 + 25) reaches it
    wid = pre_window_id(ep(VAL_A, 300), 5, 125)
    near = features(env, episodes=[ep(VAL_A, 300)], length=5, gap=125)
    assert windows(near)[wid]["status"] == "included"
    far = features(env, episodes=[ep(VAL_A, 300)], length=5, gap=125, features=("speed_corr_w",))
    assert windows(far)[wid]["exclusion_reasons"] == ["overlaps_event"]


def test_attack_inside_the_gap_excludes_a_pre_window(env):
    """Window [92, 95) and its history are attack-free; the gap [95, 130) holds the
    run 100..120: the window does not precede onset 130's attack alone."""
    res = features(env, episodes=[ep(VAL_A, 130)], length=3, gap=35)
    w = windows(res)[pre_window_id(ep(VAL_A, 130), 3, 35)]
    assert (w["start_frame"], w["end_frame"]) == (92, 95) and w["label_context"]["attack"] == 0
    assert w["exclusion_reasons"] == ["overlaps_event"]


def test_baseline_rules_and_history(env):
    res = features(env, episodes=[ep(VAL_A, 300)], length=20, gap=0, baseline=base(20, 0))
    base_w = {w["start_frame"]: w for w in windows(res, "baseline").values()}
    assert base_w[100]["exclusion_reasons"][0] == "overlaps_event"
    # [140, 160): no attack in the tile, clearance 0, but its history reaches the run ending at 140
    assert base_w[140]["exclusion_reasons"] == ["near_event"]
    assert base_w[160]["status"] == "excluded" and base_w[200]["status"] == "included"
    # the pre-event window [280, 300) blocks the tile and the tiles whose history reaches it
    assert "overlaps_pre_window" in base_w[280]["exclusion_reasons"]
    assert "overlaps_pre_window" in base_w[300]["exclusion_reasons"]
    for w in windows(res, "baseline").values():
        assert w["preprocessing"]["truncated_at_onset"] is False and w["episode_id"] is None
        assert w["end_frame"] - w["start_frame"] == 20 and w["start_frame"] % 20 == 0
    assert {w["session"] for w in windows(res, "baseline").values()} == {VAL_A}  # requested sessions only


def test_clearance_is_measured_from_every_behavior_frame(env):
    """Clearance counts all attack frames (run ends, unrequested episodes), not
    only the requested onsets that the contract validator can check."""
    res = features(env, episodes=[ep(VAL_A, 300)], length=20, gap=0, baseline=base(20, 60))
    for w in windows(res, "baseline").values():
        if w["status"] == "included":
            hs = iw.history_start(session(env, VAL_A).centre_valid, w["start_frame"] - 1)
            for a, b in ATTACKS[VAL_A]:
                assert a - 60 > w["end_frame"] - 1 or b - 1 + 60 < hs


def test_included_baselines_are_isolated_from_attack_frames(tmp_path):
    """Moving mouse b by 200 px on every attack frame moves an included baseline
    tile's body-centre distance by less than the 1 % residual weight the history
    rule allows (200 px * 1 % = 2 px); the influence is real but bounded."""
    def shake(sid, kp, sc):
        if sid == VAL_A:
            for a, b in ATTACKS[VAL_A]:
                kp[a:b, 1, 0] += 200.0
    req = dict(episodes=[ep(VAL_A, 300)], length=20, baseline=base(20, 20),
               features=("dist_center", "dist_rate", "rel_speed", "close_time_s", "speed_corr_w"))
    a = windows(features(build(tmp_path, name="a"), **req), "baseline")
    b = windows(features(build(tmp_path, shake, name="b"), **req), "baseline")
    included = [wid for wid, w in a.items() if w["status"] == "included"]
    assert len(included) >= 2
    diffs = []
    for wid in included:
        assert b[wid]["status"] == "included"
        for f, s in a[wid]["features"].items():
            assert s["n_usable"] == b[wid]["features"][f]["n_usable"]
        s, t = a[wid]["features"]["dist_center"], b[wid]["features"]["dist_center"]
        diffs.append(max(abs(s["median"] - t["median"]), abs(s["mean"] - t["mean"])))
    assert 0 < max(diffs) < 200 * iw.SMOOTHER_RESIDUAL


def test_low_quality_frames_exclude_without_fabrication(tmp_path):
    def degrade(sid, kp, sc):
        if sid == VAL_A:
            sc[285:288, 0, 3] = 0.01        # 3 frames + 1 lost endpoint (288) in [280, 300)
            sc[250:270, 1, 4] = 0.01        # every frame of [250, 270)
    env = build(tmp_path, degrade)
    res = features(env, episodes=[ep(VAL_A, 300)], length=20, gap=0)
    w = windows(res)[pre_window_id(ep(VAL_A, 300), 20, 0)]
    f = w["features"]["dist_rate"]
    assert (f["n_frames"], f["n_usable"]) == (20, 16)
    assert w["exclusion_reasons"] == ["insufficient_usable_frames"]
    assert (f["n_usable"], f["median"], f["mean"]) == reference(session(env, VAL_A), 280, 300)
    res = features(env, episodes=[ep(VAL_A, 300)], length=20, gap=30)   # window [250, 270)
    w = windows(res)[pre_window_id(ep(VAL_A, 300), 20, 30)]
    assert w["exclusion_reasons"] == ["missing_data"]
    assert w["features"]["dist_rate"] == {"n_frames": 20, "n_usable": 0, "median": None, "mean": None}
    assert res["status"] == "no_data"


# confidence scores that cannot show the minimum is met (all representable in the
# published JSON: null is standard JSON, NaN / Infinity are Python json tokens)
INVALID_SCORES = {"null": None, "nan": float("nan"), "inf": float("inf"), "-inf": float("-inf"),
                  "negative": -0.1, "above_range": 1.5}


def _set_scores(value, frames, mouse=None, kps=None):
    def edit(sid, rec):
        if sid == VAL_A:
            for t in frames:
                for m in (range(2) if mouse is None else [mouse]):
                    for k in (range(7) if kps is None else kps):
                        rec["scores"][t][m][k] = value
    return edit


def _measurements(env, episode=300):
    """Both tools' measurements for a pre-event window and baselines of VAL_A."""
    res = features(env, episodes=[ep(VAL_A, 100), ep(VAL_A, episode)], baseline=base())
    q = quality(env, list(windows(res)))
    return res["measurements"], q["measurements"]


@pytest.mark.parametrize("name", INVALID_SCORES)
def test_invalid_scores_never_pass_tracking(tmp_path, name):
    """Codex's Stage 4 review case: every keypoint of frames 70..99 has an invalid
    score. Through the real extractor, window [80, 100) has no usable frame."""
    env = build(tmp_path, edit=_set_scores(INVALID_SCORES[name], range(70, 100)))
    raw = json.loads((env[2] / INDEX_NAME).read_text())
    assert VAL_A in raw["sessions"]                      # Stage 1 published it
    wid = pre_window_id(ep(VAL_A, 100), 20, 0)
    w = windows(features(env, episodes=[ep(VAL_A, 100)]))[wid]
    assert w["status"] == "excluded" and w["exclusion_reasons"] == ["missing_data"]
    assert w["features"]["dist_rate"] == {"n_frames": 20, "n_usable": 0, "median": None, "mean": None}
    q = quality(env, [wid])["measurements"]["windows"][0]
    assert (q["tracking_ok_frames"], q["endpoint_valid_frames"], q["verdict"]) == (0, 0, "insufficient_data")
    assert q["excluded_frames"]["not_tracking_ok"] == list(range(80, 100))


@pytest.mark.parametrize("name", INVALID_SCORES)
def test_invalid_score_affects_only_its_keypoint_frames_like_a_low_score(tmp_path, name):
    """An invalid score of one torso keypoint on a few frames gives exactly the
    results of a below-minimum score there: only those frames fail, the rest of
    the session is kept (frame-level, not session-level)."""
    frames, kw = range(285, 288), {"mouse": 0, "kps": [TORSO_IDX[0]]}
    invalid = build(tmp_path, edit=_set_scores(INVALID_SCORES[name], frames, **kw), name="invalid")
    low = build(tmp_path, edit=_set_scores(0.01, frames, **kw), name="low")
    a, b = _measurements(invalid), _measurements(low)
    assert a == b
    q = {w["window_id"]: w for w in a[1]["windows"]}[pre_window_id(ep(VAL_A, 300), 20, 0)]
    assert q["excluded_frames"]["not_tracking_ok"] == [285, 286, 287]


def test_legitimate_scores_keep_avas_flags_exactly(tmp_path):
    """Scores in [0, 1], including both ends, give flag_tracking's flags unchanged;
    a missing point with an invalid score stays 'missing' only (as in Ava's flags)."""
    def change(sid, kp, sc):
        if sid == VAL_A:
            sc[10:20] = 1.0
            sc[20:30] = 0.0
            kp[40:42, 0, :, 3] = np.nan
            sc[40:42, 0, 3] = np.nan
    env = build(tmp_path, change)
    sess = session(env, VAL_A)
    ours, avas = iw._flags(sess.torso), flag_tracking(sess.torso, **iw.FLAG_SETTINGS)
    pd.testing.assert_frame_equal(ours, avas)
    assert ours.loc[ours["frame"].between(10, 19), "valid"].all()
    lost = ours[(ours["frame"].isin([40, 41])) & (ours["mouse"] == ours["mouse"].unique()[0])
                & (ours["keypoint"] == TORSO_KEYPOINTS[0])]
    assert lost["missing"].all() and not lost["low_conf"].any()


def test_score_validation_is_reported(env):
    res = features(env, episodes=[ep(VAL_A, 300)])
    m = {x["name"]: x for x in res["methodology"]}["score_validation"]
    assert m["value"] == "finite score in [0.0, 1.0] required"
    assert "null" in m["note"] and "infinite" in m["note"] and "out-of-range" in m["note"]


def test_distance_rate_is_pair_level(tmp_path):
    """The same closing of the distance by either mouse gives the same dist_rate:
    the feature cannot say which mouse approached."""
    def a_moves(sid, kp, sc):
        if sid == VAL_A:
            kp[:, 1, 0, :] -= (kp[:, 1, 0, :] - kp[0, 1, 0, :])      # b stays at 700
            kp[70:, 0, 0, :] += np.r_[np.linspace(0, 300, 30), np.full(N - 100, 300.0)][:, None]
    req = dict(episodes=[ep(VAL_A, 100)], length=20, gap=0)
    b_res = features(build(tmp_path, name="b"), **req)["measurements"]["windows"][0]["features"]["dist_rate"]
    a_res = features(build(tmp_path, a_moves, name="a"), **req)["measurements"]["windows"][0]["features"]["dist_rate"]
    assert b_res["median"] < -200 and abs(a_res["median"] - b_res["median"]) < 5


def test_label_context_counts_the_window(env):
    res = features(env, episodes=[ep(VAL_A, 100)], length=30, gap=0)
    assert windows(res)[pre_window_id(ep(VAL_A, 100), 30, 0)]["label_context"] == \
        {"attack": 0, "investigation": 20, "mount": 0, "other": 10}


def test_baseline_thinning_is_even_and_deterministic(env):
    req = dict(episodes=[ep(VAL_A, 300)], length=20, baseline=base(10, 0, max_windows=3))
    res = features(env, **req)
    kept = sorted(w["start_frame"] for w in windows(res, "baseline").values() if w["status"] == "included")
    full = features(env, **{**req, "baseline": base(10, 0, max_windows=1000)})
    eligible = sorted(w["start_frame"] for w in windows(full, "baseline").values() if w["status"] == "included")
    middle = int(np.floor((len(eligible) - 1) / 2 + 0.5))
    assert kept == [eligible[0], eligible[middle], eligible[-1]]
    omitted = {m["name"]: m["value"] for m in res["methodology"]}["baseline_tiles_omitted"]
    assert omitted == len(eligible) - 3
    assert features(env, **req)["evidence_id"] == res["evidence_id"]


def test_too_many_windows_is_an_error_not_a_truncation(env, monkeypatch):
    monkeypatch.setattr(iw.ic, "MAX_WINDOWS", 30)
    res = features(env, episodes=[ep(VAL_A, 300)], length=20, baseline=base(10, 0))
    assert res["status"] == "error" and res["errors"][0]["code"] == "invalid_request"


@pytest.mark.parametrize("frame", [105, 0, 399])       # inside a run, not attack, not attack
def test_unknown_episode_is_refused(env, frame):
    res = features(env, episodes=[ep(VAL_A, frame)])
    assert res["status"] == "error" and res["errors"][0]["code"] == "unknown_id"


def test_other_sources_are_not_substituted(env):
    res = features(env, episodes=[ep(VAL_A, 100, source="classifier_prediction")])
    assert res["errors"][0]["code"] == "source_unavailable"
    # even with a prediction source plugged in, its episodes are never measured as if
    # they were annotation episodes (window rules here read annotation frames)
    annotations = env[0].sources["validation_annotation"]

    class PredictionSource:
        source, policy = "classifier_prediction", env[1]

        def snapshot(self, sessions):
            return annotations.snapshot(sessions)
    tools = CalMS21InvestigationTools(env[1], [annotations, PredictionSource()])
    for episode in (ep(VAL_A, 100, source="classifier_prediction"),):
        res = call_tool(tools, "get_window_features", freq([episode]), env[1])
        assert res["errors"][0]["code"] == "source_unavailable"
    wid = baseline_window_id("classifier_prediction", "attack", VAL_A, 0, 20)
    res = call_tool(tools, "check_tracking_quality", qreq([wid]), env[1])
    assert res["errors"][0]["code"] == "source_unavailable"


@pytest.mark.parametrize("sid", [TEST, TRAIN])
def test_test_and_train_episodes_are_refused_before_reading(env, sid):
    with pytest.raises(ForbiddenEvidence):
        features(env, episodes=[ep(sid, 5)])


def test_session_missing_from_the_cache_is_unavailable(tmp_path):
    env = build(tmp_path, sessions=(VAL_A,))
    res = features(env, episodes=[ep(VAL_B, 200)])
    assert res["errors"][0]["code"] == "source_unavailable"


def test_tampered_cache_and_changed_manifest_are_unavailable(env):
    index = json.loads((env[2] / INDEX_NAME).read_text())
    index["allowed_split"] = "test"
    (env[2] / INDEX_NAME).write_text(json.dumps(index))
    assert features(env, episodes=[ep(VAL_A, 100)])["errors"][0]["code"] == "source_unavailable"
    env[3].write_text(json.dumps(MANIFEST, indent=1))
    res = call_tool(env[0], "check_tracking_quality", qreq([pre_window_id(ep(VAL_A, 100), 20, 0)]), env[1])
    assert res["errors"][0]["code"] == "source_unavailable"


def test_index_changed_during_the_query_is_refused(env, monkeypatch):
    real = iw.load_session

    def load_then_touch(sid, data):
        index = json.loads((env[2] / INDEX_NAME).read_text())
        index["extraction_run"]["parse_seconds"] = 99.0
        (env[2] / INDEX_NAME).write_text(json.dumps(index))
        return real(sid, data)
    monkeypatch.setattr(iw, "load_session", load_then_touch)
    assert features(env, episodes=[ep(VAL_A, 100)])["errors"][0]["code"] == "source_unavailable"


def _rewrite_manifest_during(env, monkeypatch, target, name):
    """Replace target.name so that, after it runs, the manifest file holds the
    same sessions in different bytes (deterministic, mid-query)."""
    real = getattr(target, name)

    def run_then_rewrite(*args, **kw):
        out = real(*args, **kw)
        env[3].write_text(json.dumps(MANIFEST, indent=1))
        return out
    monkeypatch.setattr(target, name, run_then_rewrite)


@pytest.mark.parametrize("tool", ["get_window_features", "check_tracking_quality"])
def test_manifest_changed_during_the_query_is_refused(env, monkeypatch, tool):
    """Codex's Stage 4 review case: the manifest changes while sessions are
    loaded and the index does not; the result must not be returned."""
    req = freq([ep(VAL_A, 100)]) if tool == "get_window_features" else \
        qreq([pre_window_id(ep(VAL_A, 100), 20, 0)])
    assert call_tool(env[0], tool, req, env[1])["status"] == "ok"
    index_before = (env[2] / INDEX_NAME).read_bytes()
    _rewrite_manifest_during(env, monkeypatch, iw, "load_session")
    res = call_tool(env[0], tool, req, env[1])
    assert res["status"] == "error" and res["measurements"] is None
    assert res["errors"][0]["code"] == "source_unavailable"
    assert "manifest changed" in res["errors"][0]["message"]
    assert (env[2] / INDEX_NAME).read_bytes() == index_before      # the index never changed


def test_policy_changed_during_the_query_is_refused(env, monkeypatch):
    real = iw.load_session

    def load_then_widen(sid, data):
        env[1].split_of["task1/train/SYNTHETIC_extra"] = "train"   # still a valid policy
        return real(sid, data)
    monkeypatch.setattr(iw, "load_session", load_then_widen)
    res = features(env, episodes=[ep(VAL_A, 100)])
    assert res["errors"][0]["code"] == "source_unavailable"
    assert "policy" in res["errors"][0]["message"]


def test_result_provenance_and_methodology(env):
    res = features(env, episodes=[ep(VAL_A, 300)], baseline=base())
    prov = res["provenance"]
    here = Path(iw.__file__).parent
    files = {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in iw.CODE_FILES}
    assert prov["code_sha256"] == hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    assert {d["kind"] for d in prov["data"]} == {"synthetic_fixture"}
    names = {m["name"] for m in res["methodology"]}
    assert {"causal_computation", "interpolation", "tracking_flags", "history_start", "event_overlap_rule",
            "usable_frame", "min_usable_fraction", "baseline_rules", "dist_rate_meaning"} <= names
    assert all(m["status"] == "provisional" for m in res["methodology"])
    assert any("which mouse" in x for x in res["limitations"])
    assert any("not motion accuracy" in x for x in res["limitations"])


def test_methodology_states_what_is_read_and_what_influences(env):
    """The preprocessing description matches the computation (Codex Stage 4
    finding 3): values equal the truncated run from frame 0, later frames are
    read but have no influence, and the smoother is not restarted at the
    history start (older frames keep a decaying influence)."""
    res = features(env, episodes=[ep(VAL_A, 300)], baseline=base())
    q = quality(env, [pre_window_id(ep(VAL_A, 300), 20, 0)])
    for r in (res, q):
        m = {x["name"]: x for x in r["methodology"]}
        causal = m["causal_computation"]["note"]
        assert "frames [0, end)" in causal and "no influence" in causal and "They are read" in causal
        assert "[start - history, end)" not in causal and "no frame at or after the window end is read" \
            not in causal
        smooth = m["smoothing"]["note"]
        assert "NOT restarted at the window or its history start" in smooth
        assert "decaying influence" in smooth and "restarted at the start of each window" not in smooth
        memory = m["smoother_memory_frames"]["note"]
        assert "provisional margin" in memory and "exclusion rules" in memory
        assert "not a bound on the nonlinear features" in memory
    hist = {x["name"]: x for x in res["methodology"]}["history_start"]["note"]
    assert "used only by the exclusion rules" in hist and "still read" in hist
    doc = iw.__doc__
    assert "No influence is not the same as not read" in doc
    assert "never restarted at a window or at its history start" in doc
    assert "no frame at or after the\nonset is read" not in doc


def test_smoother_is_not_restarted_at_the_history_start(env):
    """Pins the behavior the methodology describes: a value inside a window
    depends on frames before its history start (decaying, not zero)."""
    sess = session(env, VAL_A)
    end = 300
    ok, feats, valid = iw.prefix_arrays(sess, end)
    hs = iw.history_start(valid, 280 - iw.feature_lookback(["dist_center"]))
    restarted = iw._pipeline(sess.torso[(sess.torso["frame"] >= hs) & (sess.torso["frame"] < end)])[2]
    a, b = feats["dist_center"][280:end], restarted["dist_center"][280:end]   # both frame-indexed
    assert not np.array_equal(a, b)                       # older frames still influence
    assert np.nanmax(np.abs(a - b)) < 200 * iw.SMOOTHER_RESIDUAL   # ... at a small residual


# ---- check_tracking_quality ----

def test_quality_counts_frames_exactly(tmp_path):
    def degrade(sid, kp, sc):
        if sid == VAL_A:
            sc[285, 1, 5] = 0.01
    env = build(tmp_path, degrade)
    wid = pre_window_id(ep(VAL_A, 300), 20, 0)
    q = quality(env, [wid], history=(1, 3, 15))["measurements"]["windows"][0]
    assert (q["start_frame"], q["end_frame"], q["frames"]) == (280, 300, 20)
    assert q["excluded_frames"] == {"not_tracking_ok": [285], "endpoint_invalid": [286]}
    assert (q["tracking_ok_frames"], q["endpoint_valid_frames"]) == (19, 18)
    # h = 3: 285..288 fail (285 itself, then 285 within t-3..t); h = 15: 285..299 fail
    assert q["usable_by_history"] == {"1": 18, "3": 16, "15": 5}
    assert q["smoothing_contamination"]["status"] == "detected"
    assert q["smoothing_contamination"]["frames"] == list(range(287, 300))  # within K after 285
    assert q["verdict"] == "usable_with_caveats" and q["motion_accuracy_verified"] is False


def test_quality_history_reads_the_past_but_not_the_future(tmp_path):
    def degrade(sid, kp, sc):
        if sid == VAL_A:
            sc[275, 0, 3] = 0.01                      # 5 frames before the window [280, 300)
            kp[300:] = kp[299]                         # frozen from the window's last frame
            sc[310:] = 0.01
    clean = build(tmp_path, name="clean")
    env = build(tmp_path, degrade, name="dirty")
    wid = pre_window_id(ep(VAL_A, 300), 20, 0)
    q = quality(env, [wid], history=(1, 10))["measurements"]["windows"][0]
    assert q["tracking_ok_frames"] == 20 and q["endpoint_valid_frames"] == 20
    assert q["usable_by_history"] == {"1": 20, "10": 14}       # 280..285 see frame 275
    assert q["smoothing_contamination"]["frames"] == list(range(280, 300))   # 275 + K = 300
    c = quality(clean, [wid], history=(1, 10))["measurements"]["windows"][0]
    assert c["tracking_ok_frames"] == 20  # the frozen run after the window never reaches it


def test_quality_matches_the_feature_tool(env):
    wid = pre_window_id(ep(VAL_A, 300), 20, 0)
    q = quality(env, [wid])["measurements"]["windows"][0]
    f = windows(features(env, episodes=[ep(VAL_A, 300)]))[wid]["features"]["dist_rate"]
    assert q["endpoint_valid_frames"] == f["n_usable"]


@pytest.mark.parametrize("wid, code", [
    (pre_window_id(ep(VAL_A, 100), 150, 0), "invalid_request"),     # starts before frame 0
    (pre_window_id(ep(VAL_A, 105), 20, 0), "unknown_id"),           # not an episode start
    (baseline_window_id("validation_annotation", "attack", VAL_A, 15, 20), "unknown_id"),  # not a tile
    (baseline_window_id("validation_annotation", "attack", VAL_A, 400, 20), "unknown_id"),  # past the end
    (baseline_window_id("monitor_event", "attack", VAL_A, 0, 20), "source_unavailable"),
])
def test_quality_refuses_windows_it_cannot_assess(env, wid, code):
    res = quality(env, [wid])
    assert res["status"] == "error" and res["errors"][0]["code"] == code


def test_flag_passing_collapse_is_not_detected(tmp_path):
    """Documented limitation (Stage 1 report, section 4): a one-frame collapse of
    mouse b onto mouse a with scores above the cut-off passes every flag."""
    def collapse(sid, kp, sc):
        if sid == VAL_A:
            kp[365, 1] = kp[365, 0]
            sc[365, 1] = 0.2
    env = build(tmp_path, collapse)
    wid = baseline_window_id("validation_annotation", "attack", VAL_A, 360, 20)
    q = quality(env, [wid])["measurements"]["windows"][0]
    assert q["tracking_ok_frames"] == 20 and q["smoothing_contamination"]["status"] == "none_detected"
    assert q["verdict"] == "usable_with_caveats" and q["motion_accuracy_verified"] is False
    w = windows(features(env, episodes=[ep(VAL_A, 300)], baseline=base(20, 10)))[wid]
    assert w["status"] == "included"            # the artifact enters the baseline unnoticed


# ---- real data (skipped without the verified cache) ----

REAL_CACHE = ROOT / "data" / "raw" / "calms21_validation"
REAL_MANIFEST = ROOT / "splits" / "split_manifest.json"
real = pytest.mark.skipif(not (REAL_CACHE / INDEX_NAME).exists(),
                          reason="verified CalMS21 validation cache not present")
M5 = "task1/train/mouse005_task1_annotator1"
M5_ONSETS = (79, 251, 736, 934, 1147, 1969, 2072)


@pytest.fixture(scope="module")
def real_env():
    policy = EvidencePolicy.from_manifest(REAL_MANIFEST)
    return CalMS21InvestigationTools.for_validation_cache(REAL_CACHE, REAL_MANIFEST, policy), policy


@real
def test_real_pre_windows_reproduce_the_stage1_report(real_env):
    """docs/investigation_experiment_01.md, 1-s pre-attack windows of mouse005,
    per-window median dist_rate at H = 1: -211.47 / -41.55 / -3.65 / 2.09 / 9.09."""
    eids = [ep(M5, o) for o in M5_ONSETS]
    res = call_tool(real_env[0], "get_window_features", freq(eids, length=30, gap=0), real_env[1])
    meds = [w["features"]["dist_rate"]["median"] for w in res["measurements"]["windows"]]
    assert [round(float(np.percentile(meds, q)), 2) for q in (0, 25, 50, 75, 100)] == \
        [-211.47, -41.55, -3.65, 2.09, 9.09]
    by_onset = {int(w["episode_id"].rsplit(":", 1)[1]): w for w in res["measurements"]["windows"]}
    assert by_onset[1147]["exclusion_reasons"] == ["insufficient_usable_frames"]  # 26 / 30 usable
    assert by_onset[1147]["features"]["dist_rate"]["n_usable"] == 26
    assert by_onset[934]["exclusion_reasons"] == ["overlaps_event"]  # attack ends within its history
    assert res["provenance"]["data"][1]["sha256"].startswith("c66f1cae")


@real
def test_real_baselines_keep_clear_of_every_attack_frame(real_env):
    eids = [ep(M5, o) for o in M5_ONSETS]
    res = call_tool(real_env[0], "get_window_features",
                    freq(eids, length=30, gap=0, baseline=base(30, 150, 1000)), real_env[1])
    verified, _ = verified_sessions(REAL_CACHE, REAL_MANIFEST)
    sess = iw.load_session(M5, dict(verified)[M5])
    attack = np.flatnonzero(sess.labels == "attack")
    included = [w for w in windows(res, "baseline").values() if w["status"] == "included"]
    assert len(included) == 128
    for w in included:
        hs = iw.history_start(sess.centre_valid, w["start_frame"] - 1)
        assert np.all((attack < hs - 150) | (attack > w["end_frame"] - 1 + 150))
        assert w["label_context"]["attack"] == 0


@real
def test_real_pre_windows_ignore_frames_from_onset_on(real_env):
    verified, _ = verified_sessions(REAL_CACHE, REAL_MANIFEST)
    sess = iw.load_session(M5, dict(verified)[M5])
    for onset in M5_ONSETS[1:]:
        before = iw.measure_window(sess, onset - 90, onset, list(iw.FEATURE_LOOKBACK))
        poisoned = copy.deepcopy(sess)
        poisoned._prefix.clear()
        t = poisoned.torso
        prev = t[t["frame"] == onset - 1].set_index(["mouse", "keypoint"])[["x", "y"]]
        trap = (t["frame"] >= onset) & (t["frame"] < onset + 20)   # frozen from onset - 1 on
        keys = pd.MultiIndex.from_frame(t.loc[trap, ["mouse", "keypoint"]])
        t.loc[trap, ["x", "y"]] = prev.reindex(keys).to_numpy()
        t.loc[t["frame"] >= onset + 20, ["x", "y"]] += 500.0
        t.loc[t["frame"] >= onset + 20, "score"] = 0.01
        poisoned.flags, poisoned.ok, poisoned.feats, poisoned.centre_valid = iw._pipeline(t)
        assert iw.measure_window(poisoned, onset - 90, onset, list(iw.FEATURE_LOOKBACK)) == before


@real
def test_real_quality_reports_but_cannot_see_the_known_collapse(real_env):
    """mouse005 frame 2812: mouse b's keypoints land on mouse a for one frame with
    scores above the cut-off (Stage 1 report, section 4)."""
    wid = baseline_window_id("validation_annotation", "attack", M5, 2790, 30)
    res = call_tool(real_env[0], "check_tracking_quality", qreq([wid]), real_env[1])
    q = res["measurements"]["windows"][0]
    assert q["tracking_ok_frames"] == 30 and q["smoothing_contamination"]["status"] == "none_detected"
    assert q["motion_accuracy_verified"] is False and q["verdict"] == "usable_with_caveats"
    assert any("Jump detection is off" in x for x in res["limitations"])


@real
def test_real_quality_of_every_pre_window(real_env):
    wids = [pre_window_id(ep(M5, o), 30, 0) for o in M5_ONSETS]
    res = call_tool(real_env[0], "check_tracking_quality", qreq(wids, (1, 15)), real_env[1])
    by = {q["window_id"]: q for q in res["measurements"]["windows"]}
    f = call_tool(real_env[0], "get_window_features", freq([ep(M5, o) for o in M5_ONSETS], length=30),
                  real_env[1])
    for w in f["measurements"]["windows"]:
        assert by[w["window_id"]]["endpoint_valid_frames"] == w["features"]["dist_rate"]["n_usable"]
    assert by[pre_window_id(ep(M5, 1147), 30, 0)]["excluded_frames"]["not_tracking_ok"] == [1124, 1127]
