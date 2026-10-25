"""Tests for scripts/fetch_demo_video.py with a fake archive: nothing is downloaded from the internet."""

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_demo_video.py"
spec = importlib.util.spec_from_file_location("fetch_demo_video", SCRIPT)
fdv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdv)

VIDEO = b"fake video bytes " * 5000
MANIFEST = {"seed": 0, "train": ["task1/train/mouse001_task1_annotator1"], "validation": ["task1/train/mouse064_task1_annotator1"],
            "test": ["task1/test/mouse078_task1_annotator1"]}


def archive(extra=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("task1_videos_mp4/train/mouse064_task1_annotator1.mp4", VIDEO)
        z.writestr("task1_videos_mp4/test/mouse078_task1_annotator1.mp4", b"a test video")
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buffer.getvalue()


def reader_for(data, calls=None):
    def reader(start, end):
        if calls is not None:
            calls.append((start, end))
        return data[start:end + 1], len(data)
    return reader


# ---- who may be downloaded ----

def test_a_validation_or_train_session_gives_its_file_name():
    assert fdv.check_session("task1/train/mouse064_task1_annotator1", MANIFEST) == "mouse064_task1_annotator1.mp4"
    assert fdv.check_session("task1/train/mouse001_task1_annotator1", MANIFEST) == "mouse001_task1_annotator1.mp4"


@pytest.mark.parametrize("session", ["task1/test/mouse078_task1_annotator1", "x/test/y", "task1/train/unknown"])
def test_test_sessions_and_unknown_ones_are_refused(session):
    with pytest.raises(SystemExit, match="refused"):
        fdv.check_session(session, MANIFEST)


# ---- the download ----

def test_the_video_is_read_inflated_checked_and_saved(tmp_path):
    messages = []
    path = fdv.download(reader_for(archive()), "mouse064_task1_annotator1.mp4", tmp_path / "videos", messages.append)
    assert path == tmp_path / "videos" / "mouse064_task1_annotator1.mp4" and path.read_bytes() == VIDEO
    assert not list((tmp_path / "videos").glob("*.part")) and "saved and checked" in messages[-1]


def test_only_the_one_video_is_read_not_the_archive():
    calls = []
    data = archive({"task1_videos_mp4/train/other.mp4": b"x" * 400_000})
    fdv.download(reader_for(data, calls), "mouse064_task1_annotator1.mp4", Path(__file__).parent / "_unused_out", lambda *_: None)
    read = sum(end - start + 1 for start, end in calls)
    other_start = data.find(b"x" * 1000)
    assert not any(start <= other_start <= end for start, end in calls)          # the other video's bytes were never asked for
    assert read < len(data) + 10_000
    import shutil
    shutil.rmtree(Path(__file__).parent / "_unused_out", ignore_errors=True)


def test_a_corrupted_download_is_deleted_and_reported(tmp_path):
    good = archive()
    start = good.find(b"fake video bytes") if good.find(b"fake video bytes") >= 0 else None
    data = bytearray(good)
    # flip bytes inside the compressed stream of the first entry (after its local header)
    zi = zipfile.ZipFile(io.BytesIO(good)).getinfo("task1_videos_mp4/train/mouse064_task1_annotator1.mp4")
    middle = zi.header_offset + 30 + len(zi.filename) + zi.compress_size // 2
    for i in range(middle, middle + 8):
        data[i] ^= 0xFF
    with pytest.raises((SystemExit, Exception)):
        fdv.download(reader_for(bytes(data)), "mouse064_task1_annotator1.mp4", tmp_path / "videos", lambda *_: None)
    assert not (tmp_path / "videos" / "mouse064_task1_annotator1.mp4").exists()
    assert not list((tmp_path / "videos").glob("*.part"))


def test_a_video_that_is_already_there_is_not_downloaded_again(tmp_path):
    out = tmp_path / "videos"
    out.mkdir()
    (out / "mouse064_task1_annotator1.mp4").write_bytes(VIDEO)
    messages = []
    calls = []
    fdv.download(reader_for(archive(), calls), "mouse064_task1_annotator1.mp4", out, messages.append)
    assert messages == [f"already there: {out / 'mouse064_task1_annotator1.mp4'} ({len(VIDEO) / 1e6:.0f} MB)"]
    zi = zipfile.ZipFile(io.BytesIO(archive())).getinfo("task1_videos_mp4/train/mouse064_task1_annotator1.mp4")
    data_start = zi.header_offset + 30 + len(zi.filename)
    assert not any(start >= data_start and end < data_start + zi.compress_size for start, end in calls)   # the video's bytes were not read


def test_a_video_that_is_not_in_the_archive_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="not in the archive"):
        fdv.download(reader_for(archive()), "mouse001_task1_annotator1.mp4", tmp_path, lambda *_: None)


def test_main_refuses_a_test_video_before_any_download(tmp_path, capsys):
    called = []
    with pytest.raises(SystemExit, match="refused"):
        fdv.main(["--session", "task1/test/mouse078_task1_annotator1", "--out", str(tmp_path)], reader=lambda s, e: called.append(1))
    assert called == []
