"""One-command acceptance test of the LIVE camera path (new file).

It starts the real run program (the same one the app starts) on a camera, or on a video played like a camera, with the pose model set up in
models/pose/pose_model.json, lets it run for some seconds, and checks every step. Each check says PASS, WARN or FAIL, and the exit code is 0
only when nothing FAILED. Run it when the camera is connected and the trained model is in place; the result tells whether live pose works.

    # THE REAL TEST: camera 0 (or the number of the Nikon Webcam Utility), the trained model from models/pose/pose_model.json
    python scripts/live_camera_acceptance.py --camera 0 --seconds 30 --corners 0.23,0.22,0.81,0.22,0.81,0.65,0.23,0.65

    # REHEARSAL with no camera and no trained model: a video plays like a camera and a STUB gives perfect keypoints.
    # It tests everything around the model (speed of the run program, scores, monitor, alerts) but NOT the model itself.
    python scripts/live_camera_acceptance.py --video data/raw/calms21_video/mouse064_task1_annotator1.mp4 --start 1800 ^
        --stub-poses data/raw/calms21_video/mouse064_task1_annotator1_poses.parquet --seconds 30 --expect-events

The standalone speed of the pose model is measured on 60 pictures first (the target is 33 ms or less, for 30 frames per second).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGET_FPS = 30.0
POSE_MS_PASS, POSE_MS_WARN = 33.0, 66.0
KEEP_UP_PASS, KEEP_UP_WARN = 25.0, 15.0         # processed frames per second in the run program
CAMERA_FPS_PASS, CAMERA_FPS_WARN = 25.0, 14.0
START_TIMEOUT_S, STOP_TIMEOUT_S = 240.0, 90.0


def level_for(value, passing, warning, higher_is_better=True):
    if higher_is_better:
        return "PASS" if value >= passing else "WARN" if value >= warning else "FAIL"
    return "PASS" if value <= passing else "WARN" if value <= warning else "FAIL"


def judge(m):
    """The checks, from the measurements `m` (a dict). Pure: tests feed it numbers. Returns [(name, PASS|WARN|FAIL|INFO|SKIP, detail)]."""
    out = []
    pose = m.get("pose")
    if pose is None:
        out.append(("Pose model loads", "FAIL", m.get("pose_error") or "no pose model is set up"))
    else:
        out.append(("Pose model loads", "PASS", pose["description"]))
        if pose["stub"]:
            out.append(("Pose model speed (33 ms or less)", "SKIP", "a stub is not a model: this check needs the trained model"))
        else:
            lv = level_for(pose["median_ms"], POSE_MS_PASS, POSE_MS_WARN, higher_is_better=False)
            out.append(("Pose model speed (33 ms or less)", lv, f"median {pose['median_ms']:.1f} ms, 95th percentile {pose['p95_ms']:.1f} ms "
                        f"({1000.0 / max(pose['median_ms'], 1e-6):.0f} frames per second)"))
            if m.get("expect_mice"):
                share = pose["found_share"]
                out.append(("Both mice found in the pictures", "PASS" if share >= 0.9 else "WARN" if share >= 0.6 else "FAIL",
                            f"both mice in {100 * share:.0f}% of {pose['pictures']} pictures"))
    cam = m.get("camera")
    if cam is not None:
        lv = level_for(cam["fps"], CAMERA_FPS_PASS, CAMERA_FPS_WARN)
        note = "" if lv == "PASS" else " (the model and Model B expect 30; a webcam often gives 15)"
        out.append(("Camera gives pictures", lv, f"{cam['fps']:.1f} pictures per second at {cam['size']}{note}"))
    run = m.get("run")
    if run is None:
        return out
    out.append(("Run program starts", "PASS" if run["started"] else "FAIL",
                f"running after {run['start_s']:.0f} s" if run["started"] else f"never reached 'running': {run.get('message') or 'no message'}"))
    if not run["started"]:
        return out
    ok = run["final_state"] in ("finished", "stopped") and not run["error"] and run["interruptions"] == 0
    out.append(("Runs without a crash or a camera drop-out", "PASS" if ok else "FAIL",
                f"final state {run['final_state']}, {run['interruptions']} camera interruption(s)" + (f", error: {run['error']}" if run["error"] else "")))
    lv = level_for(run["frames_per_s"], KEEP_UP_PASS, KEEP_UP_WARN)
    out.append(("Keeps up with 30 frames per second", lv, f"{run['frames_per_s']:.1f} frames per second over {run['seconds']:.0f} s "
                f"(slowest stretch of 10 s or more: {run['worst_window_fps']:.1f})"))
    fed = run["rows"] >= 0.9 * run["frames"] - 10 and run["rejected"] == 0
    out.append(("Scores reach the monitor", "PASS" if fed else "FAIL", f"{run['rows']} score rows for {run['frames']} frames, {run['rejected']} rejected"))
    events = run["alerts"] + run["asks"]
    if m.get("expect_events"):
        out.append(("Alerts or questions appear", "PASS" if events >= 1 else "FAIL", f"{run['alerts']} alerts and {run['asks']} questions"))
    else:
        out.append(("Alerts or questions appear", "INFO", f"{run['alerts']} alerts and {run['asks']} questions (not required)"))
    return out


def measure_pose(estimator, frames, stub):
    times, found = [], 0
    for f in frames:
        t = time.perf_counter()
        result = estimator(f)
        times.append((time.perf_counter() - t) * 1000.0)
        found += int(result is not None and len(result) >= 14)
    t = np.array(times[3:] if len(times) > 6 else times)         # the first calls warm the model up
    return {"description": "stub (perfect saved keypoints)" if stub else "trained model", "stub": stub, "pictures": len(frames),
            "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "found_share": found / max(1, len(frames))}


def parse_corners(text):
    if not text:
        return None
    v = [float(x) for x in text.split(",")]
    if len(v) != 8:
        raise SystemExit("--corners needs 8 numbers")
    return [[v[i], v[i + 1]] for i in range(0, 8, 2)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", help="camera number or stream address")
    src.add_argument("--video", help="a video played like a camera (rehearsal)")
    ap.add_argument("--start", type=int, default=0, help="first frame of the video")
    ap.add_argument("--seconds", type=float, default=30.0, help="how long the run is measured")
    ap.add_argument("--corners", help="8 numbers: the screen corners (see the Upload page)")
    ap.add_argument("--stub-poses", help="rehearsal: saved CalMS21 keypoints (parquet) that a stub gives back instead of a model")
    ap.add_argument("--expect-events", action="store_true", help="fail when no alert or question appears (use with the demo video)")
    ap.add_argument("--expect-mice", action="store_true", help="check that both mice are found (the camera must see them)")
    ap.add_argument("--json", help="write the measurements and results here")
    a = ap.parse_args(argv)

    from live_tracker import pose_model, run_control as rc, run_schedule as rs, run_sources as rsrc
    from live_tracker.calms21 import FPS
    from live_tracker.evaluate import ATTACK_RULE_V1
    from live_tracker.rules import Rule
    from live_tracker import screen_map as sm

    tmp = tempfile.TemporaryDirectory()
    if a.stub_poses:
        cfg = Path(tmp.name) / "pose_model.json"
        cfg.write_text(json.dumps({"backend": "stub", "poses": str(Path(a.stub_poses).resolve()), "start_frame": a.start}), encoding="utf-8")
        os.environ[pose_model.CONFIG_ENV] = str(cfg)            # the run program inherits this
    corners = parse_corners(a.corners)
    m = {"expect_events": a.expect_events, "expect_mice": a.expect_mice, "camera": None, "pose": None, "run": None}

    # 1. the pictures (a camera, or a video) and the pose model, outside the run program
    frames = []
    if a.camera is not None:
        try:
            cap = rsrc.open_camera(rsrc.camera_device(a.camera))
        except rsrc.CameraUnavailableError as exc:
            m["camera_error"] = str(exc)
            print(f"The camera cannot be opened: {exc}")
        else:
            t0, n = time.perf_counter(), 0
            for _ in range(45):
                ok, img = cap.read()
                if ok:
                    n += 1
                    if len(frames) < 60:
                        frames.append(img)
            m["camera"] = {"fps": n / (time.perf_counter() - t0), "size": f"{img.shape[1]} x {img.shape[0]}" if n else "?"}
            while len(frames) < 60 and n:
                ok, img = cap.read()
                frames.append(img) if ok else None
            cap.release()
    else:
        import cv2
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
        for _ in range(60):
            ok, img = cap.read()
            if ok:
                frames.append(img)
        cap.release()
    estimator, reason = pose_model.load(ROOT)
    if estimator is None:
        m["pose_error"] = reason
    elif frames:
        if corners is not None:
            estimator = sm.with_screen(estimator, corners)
        m["pose"] = measure_pose(estimator, frames, stub=bool(a.stub_poses))
    if a.camera is not None and m["camera"] is None:
        results = [("Camera gives pictures", "FAIL", m.get("camera_error", "no picture"))]
        results += judge({**m, "camera": None})
    elif estimator is None:
        results = judge(m)
    else:
        # 2. the real run program
        if a.camera is not None:
            spec = {"kind": "camera", "device": a.camera, "session": "camera"}
        else:
            spec = {"kind": "video_camera", "path": str(Path(a.video).resolve()), "start": a.start, "session": "camera"}
        if corners is not None:
            spec["screen"] = corners
        r = ATTACK_RULE_V1
        rule = Rule("V1", "attack", min_duration_s=r["min_duration_s"], alert_threshold=r["alert_threshold"], ask_threshold=r["ask_threshold"],
                    alert_mode=r["alert_mode"], status="approved")
        now = rs.datetime.now(rs.timezone.utc)
        schedule = rs.for_hours((a.seconds + START_TIMEOUT_S + 60) / 3600.0, now=now, tz="America/New_York", schedule_id="ACCEPT",
                                description="Live camera acceptance test").approve()
        run_dir = rc.new_run_dir("ACCEPT", now)
        config = rc.build_config(run_dir, schedule, [rule], spec, model="models/model_b", speed=1.0, fps=FPS)
        rc.start_run(config)
        t_start, started, status = time.time(), False, None
        while time.time() - t_start < START_TIMEOUT_S:
            status = rc.read_status(run_dir)
            if status and status.get("state") in ("running", "failed", "finished", "stopped"):
                started = status.get("state") == "running"
                break
            time.sleep(1.0)
        run = {"started": started, "start_s": time.time() - t_start, "message": (status or {}).get("message"), "final_state": (status or {}).get("state")}
        samples = []
        if started:
            t0 = time.time()
            first = (rc.read_status(run_dir) or {}).get("frames", 0)
            while time.time() - t0 < a.seconds:
                time.sleep(1.0)
                s = rc.read_status(run_dir) or {}
                samples.append((time.time() - t0, s.get("frames", 0) - first))
                if s.get("state") in ("failed", "finished", "stopped"):
                    break
            rc.stop_run(run_dir)
            end = time.time()
            while time.time() - end < STOP_TIMEOUT_S:
                status = rc.read_status(run_dir) or {}
                if status.get("state") in ("failed", "finished", "stopped"):
                    break
                time.sleep(1.0)
            status = rc.read_status(run_dir) or {}
            dur = samples[-1][0] if samples else a.seconds
            # the run program writes its status only every few seconds, so use only the moments the number changed and windows of 10 s or more
            beats = [samples[0]] + [b for a_, b in zip(samples, samples[1:]) if b[1] != a_[1]] if samples else []
            rates = [(y[1] - x[1]) / (y[0] - x[0]) for i, x in enumerate(beats) for y in beats[i + 1:] if y[0] - x[0] >= 10.0]
            worst = min(rates) if rates else ((samples[-1][1] / dur) if samples else 0.0)
            run.update({"final_state": status.get("state"), "frames": status.get("frames", 0), "rows": status.get("rows_to_monitor", 0),
                        "rejected": status.get("rejected_rows", 0), "alerts": status.get("alerts", 0), "asks": status.get("asks", 0),
                        "interruptions": status.get("interruptions", 0), "seconds": dur, "frames_per_s": (samples[-1][1] / dur) if samples else 0.0,
                        "worst_window_fps": worst, "error": (run_dir / "error.txt").read_text(encoding="utf-8")[-300:] if (run_dir / "error.txt").exists() else ""})
            if status.get("message") and status.get("state") == "failed":
                run["error"] = run["error"] or status["message"]
        m["run"] = run
        m["run_dir"] = str(run_dir)
        results = judge(m)

    width = max(len(r[0]) for r in results)
    print()
    for name, lv, detail in results:
        print(f"  [{lv:4s}] {name:{width}s}  {detail}")
    failed = [r for r in results if r[1] == "FAIL"]
    warned = [r for r in results if r[1] == "WARN"]
    print(f"\n  RESULT: {'FAIL' if failed else 'PASS'}  ({len(failed)} failed, {len(warned)} warnings)\n")
    if a.json:
        Path(a.json).write_text(json.dumps({"measurements": m, "results": results}, indent=1, default=str), encoding="utf-8")
    tmp.cleanup()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
