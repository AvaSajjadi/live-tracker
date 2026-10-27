"""Tests for mars_job: starting MARS in the background and reading its progress (no program is started)."""

import json
import os
import time

import pytest

from live_tracker import mars_job as mj


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv(mj.HOME_ENV, str(tmp_path / "mars_home"))
    return tmp_path / "mars_home"


def install(home):
    py = mj.python_exe()
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    for p in mj.PB_FILES:
        f = home / "models" / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")


def test_it_says_what_is_missing(home):
    ok, why = mj.available()
    assert not ok and "environment is missing" in why
    install(home)
    (home / "models" / mj.PB_FILES[2]).unlink()
    ok, why = mj.available()
    assert not ok and "MARS_top_pose.pb" in why
    (home / "models" / mj.PB_FILES[2]).write_text("x", encoding="utf-8")
    assert mj.available() == (True, "")


def test_start_writes_the_job_and_runs_the_script_with_the_models_and_30_fps(home, tmp_path):
    install(home)
    video = tmp_path / "rec.mp4"
    video.write_bytes(b"x")
    launched = []
    job = mj.start(video, tmp_path / "job", launcher=lambda cmd, d: launched.append((cmd, d)))
    cmd = launched[0][0]
    assert cmd[0] == str(mj.python_exe()) and cmd[1] == str(mj.SCRIPT) and cmd[2] == str(video)
    assert cmd[cmd.index("--models-dir") + 1] == str(home / "models") and cmd[cmd.index("--fps") + 1] == "30.0"
    assert json.loads((job / "job.json").read_text(encoding="utf-8"))["video"] == str(video)
    assert mj.status(job)["state"] == "running" and mj.status(job)["message"] == "starting"
    with pytest.raises(FileNotFoundError):
        mj.start(tmp_path / "missing.mp4", tmp_path / "job2", launcher=lambda c, d: None)


def test_start_refuses_when_mars_is_not_installed(home, tmp_path):
    with pytest.raises(RuntimeError, match="environment is missing"):
        mj.start(tmp_path / "x.mp4", tmp_path / "job", launcher=lambda c, d: None)


def make_job(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "job.json").write_text("{}", encoding="utf-8")
    return job


def test_progress_is_read_from_the_last_log_line(tmp_path):
    job = make_job(tmp_path)
    (job / "mars.log").write_text("WARNING tensorflow noise\n4/450 frames, 0.2 frames per second\n120/450 frames, 1.1 frames per second\n",
                                  encoding="utf-8")
    s = mj.status(job)
    assert s["state"] == "running" and (s["done"], s["total"]) == (120, 450) and s["message"] == "analyzing"


def test_a_finished_job_and_a_failed_job(tmp_path):
    job = make_job(tmp_path)
    (job / "keypoints.parquet.done.json").write_text(json.dumps({"ok": True, "frames": 450, "seconds": 380.0}), encoding="utf-8")
    assert mj.status(job)["state"] == "finished" and mj.status(job)["done"] == 450
    (job / "keypoints.parquet.done.json").write_text(json.dumps({"ok": False, "error": "MemoryError: x"}), encoding="utf-8")
    s = mj.status(job)
    assert s["state"] == "failed" and "MemoryError" in s["message"]


def test_a_job_that_stopped_without_a_done_file_is_failed(tmp_path):
    job = make_job(tmp_path)
    (job / "mars.log").write_text("10/450 frames\n", encoding="utf-8")
    assert mj.status(job, now=time.time())["state"] == "running"
    assert mj.status(job, now=time.time() + mj.STALE_S + 5)["state"] == "failed"
    assert mj.status(tmp_path / "nothing")["state"] == "none"


def test_the_time_estimate_matches_the_measured_speed():
    assert mj.estimate_minutes(450) == pytest.approx((mj.WARMUP_S + 450 / 1.2) / 60)
    assert 24 < mj.estimate_minutes(1800) < 27                                # one minute of video: about 25 minutes


def test_the_project_models_folder_is_used_when_the_three_files_are_there(home, tmp_path, monkeypatch):
    project = tmp_path / "project"
    monkeypatch.setattr(mj, "ROOT", project)
    assert mj.models_dir() == home / "models"                       # nothing in models/mars yet: the MARS home
    for p in mj.PB_FILES[:2]:
        f = project / "models" / "mars" / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    assert mj.models_dir() == home / "models"                       # only two of the three: still the MARS home
    f = project / "models" / "mars" / mj.PB_FILES[2]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x", encoding="utf-8")
    assert mj.models_dir() == project / "models" / "mars"


def test_the_download_script_joins_the_pieces_and_checks_the_md5(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("dl", Path(__file__).resolve().parents[1] / "scripts" / "download_mars_models.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    data = {name: name.encode() * 1000 for name in dl.FILES}
    pieces = {}
    for name, blob in data.items():
        if dl.FILES[name][2]:
            size = -(-len(blob) // len(dl.PARTS))
            pieces.update({f"{name}.part-{p}": blob[i * size:(i + 1) * size] for i, p in enumerate(dl.PARTS)})
        else:
            pieces[name] = blob
    monkeypatch.setattr(dl, "FILES", {n: (f, hashlib.md5(data[n]).hexdigest(), s) for n, (f, _, s) in dl.FILES.items()})
    fetched = []
    monkeypatch.setattr(dl, "fetch", lambda name, out: (fetched.append(name), out.write(pieces[name])))
    dl.download(tmp_path)
    assert (tmp_path / "detection" / "MARS_top_detection_black.pb").read_bytes() == data["MARS_top_detection_black.pb"]
    assert (tmp_path / "pose" / "MARS_top_pose.pb").read_bytes() == data["MARS_top_pose.pb"]
    assert len(fetched) == 11                                     # 5 + 5 pieces and the pose file
    dl.download(tmp_path)                                         # already there with the right md5: nothing downloaded
    assert len(fetched) == 11
    monkeypatch.setattr(dl, "FILES", {"MARS_top_pose.pb": ("pose2", "0" * 32, False)})
    try:
        dl.download(tmp_path)
    except SystemExit as exc:
        assert "damaged" in str(exc) and not (tmp_path / "pose2" / "MARS_top_pose.download").exists()
    else:
        raise AssertionError("a wrong md5 must stop the download")
