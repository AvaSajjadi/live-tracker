"""Tests for video_web: codec detection from the file, the playable check, and making a web copy (with a fake converter)."""

from types import SimpleNamespace

import pytest

from live_tracker import video_web as vw
from video_fixtures import box, fake_mp4, mvhd


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("marker,name", [(b"avc1", "h264"), (b"avc3", "h264"), (b"vp09", "vp9"), (b"av01", "av1"),
                                         (b"hvc1", "hevc"), (b"hev1", "hevc"), (b"mp4v", "mp4v"), (b"zzzz", "other")])
def test_codec_is_read_from_the_file(tmp_path, marker, name):
    assert vw.video_codec(write(tmp_path, "v.mp4", fake_mp4(marker))) == name


def test_the_index_may_be_at_the_start_or_the_end_of_the_file(tmp_path):
    assert vw.video_codec(write(tmp_path, "a.mp4", fake_mp4(b"avc1", moov_first=True))) == "h264"
    assert vw.video_codec(write(tmp_path, "b.mp4", fake_mp4(b"avc1", moov_first=False))) == "h264"


def test_webm_and_unreadable_files(tmp_path):
    assert vw.video_codec(write(tmp_path, "v.webm", b"anything")) == "webm"
    assert vw.video_codec(write(tmp_path, "g.mp4", b"this is not a video file at all")) is None
    assert vw.video_codec(tmp_path / "missing.mp4") is None
    assert vw.video_codec(write(tmp_path, "e.mp4", b"")) is None


def test_length_comes_from_the_header(tmp_path):
    assert vw.seconds(write(tmp_path, "v.mp4", fake_mp4(seconds=238.334))) == pytest.approx(238.334)
    assert vw.seconds(write(tmp_path, "g.mp4", b"garbage garbage garbage")) is None


def test_browser_playable_check_and_the_reason(tmp_path):
    assert vw.is_browser_playable(write(tmp_path, "a.mp4", fake_mp4(b"avc1"))) == (True, "")
    assert vw.is_browser_playable(write(tmp_path, "w.webm", b"x")) == (True, "")
    ok, why = vw.is_browser_playable(write(tmp_path, "b.mp4", fake_mp4(b"mp4v")))
    assert not ok and "mp4v" in why and "browser cannot show" in why
    ok, why = vw.is_browser_playable(write(tmp_path, "c.mp4", fake_mp4(b"hvc1")))
    assert not ok and "HEVC" in why
    assert vw.is_browser_playable(write(tmp_path, "d.mp4", b"junk junk junk junk"))[1] == "the video file could not be read"


def test_web_copy_path():
    assert vw.web_copy_path("data/clip.mov").as_posix() == "data/clip_web.mp4"


# ---- making a copy ----

def converter(tmp_path, codec=b"avc1", seconds=10.0, returncode=0, write_output=True):
    def run(cmd, capture_output=True, text=True):
        if write_output:
            (tmp_path / cmd[-1].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]).write_bytes(fake_mp4(codec, seconds))
        return SimpleNamespace(returncode=returncode, stderr="boom" if returncode else "")
    return run


def test_a_good_copy_is_made_next_to_the_original(tmp_path):
    src = write(tmp_path, "clip.mp4", fake_mp4(b"mp4v", 10.0))
    out = vw.make_web_copy(src, dst=tmp_path / "clip_web.mp4", exe="ffmpeg", runner=converter(tmp_path))
    assert out == tmp_path / "clip_web.mp4" and vw.video_codec(out) == "h264"
    assert vw.video_codec(src) == "mp4v"                                           # the original is untouched


def test_a_copy_that_is_not_h264_or_has_the_wrong_length_is_deleted(tmp_path):
    src = write(tmp_path, "clip.mp4", fake_mp4(b"mp4v", 10.0))
    for run in (converter(tmp_path, codec=b"mp4v"), converter(tmp_path, seconds=3.0)):
        with pytest.raises(vw.ConversionFailed, match="not good"):
            vw.make_web_copy(src, dst=tmp_path / "clip_web.mp4", exe="ffmpeg", runner=run)
        assert not (tmp_path / "clip_web.mp4").exists()


def test_a_failing_converter_is_reported(tmp_path):
    src = write(tmp_path, "clip.mp4", fake_mp4(b"mp4v"))
    with pytest.raises(vw.ConversionFailed, match="boom"):
        vw.make_web_copy(src, exe="ffmpeg", runner=converter(tmp_path, returncode=1, write_output=False))


def test_without_any_converter_it_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(vw, "converter_exe", lambda: None)
    monkeypatch.setattr(vw, "windows_converter_available", lambda: False)
    with pytest.raises(vw.ConverterMissing, match="imageio-ffmpeg"):
        vw.make_web_copy(write(tmp_path, "clip.mp4", fake_mp4(b"mp4v")))
    assert vw.converter_available() is False


def test_the_converter_built_into_windows_is_used_when_there_is_no_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(vw, "converter_exe", lambda: None)
    monkeypatch.setattr(vw, "windows_converter_available", lambda: True)
    seen = []

    def run(cmd, capture_output=True, text=True):
        seen.append(cmd)
        (tmp_path / "clip_web.mp4").write_bytes(fake_mp4(b"avc1", 10.0))
        return SimpleNamespace(returncode=0, stderr="")

    src = write(tmp_path, "clip.mp4", fake_mp4(b"mp4v", 10.0))
    out = vw.make_web_copy(src, runner=run)
    cmd = seen[0]
    assert out == tmp_path / "clip_web.mp4" and vw.video_codec(out) == "h264" and vw.converter_available() is True
    assert "powershell" in cmd[0].lower() and "-File" in cmd and str(vw.WIN_SCRIPT) in cmd
    assert cmd[cmd.index("-Source") + 1] == str(src) and cmd[cmd.index("-Destination") + 1] == str(out)


def test_the_windows_script_is_part_of_the_project():
    assert vw.WIN_SCRIPT.name == "win_transcode.ps1" and vw.WIN_SCRIPT.exists()


# ---- which file to play ----

def test_playable_video_prefers_the_file_then_its_web_copy(tmp_path):
    good = write(tmp_path, "good.mp4", fake_mp4(b"avc1"))
    assert vw.playable_video(good) == good
    old = write(tmp_path, "old.mp4", fake_mp4(b"mp4v"))
    assert vw.playable_video(old) is None
    copy = write(tmp_path, "old_web.mp4", fake_mp4(b"avc1"))
    assert vw.playable_video(old) == copy
    bad_copy = write(tmp_path, "worse.mp4", fake_mp4(b"mp4v"))
    write(tmp_path, "worse_web.mp4", fake_mp4(b"mp4v"))
    assert vw.playable_video(bad_copy) is None
