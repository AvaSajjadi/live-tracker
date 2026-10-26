"""Can the browser play this video? And a browser-friendly copy when it cannot (new file).

Browsers play H.264, VP9 and AV1. Many research videos use older formats such as MPEG-4 Part 2 ("mp4v"),
which Windows players show but browsers do not: the picture area just stays empty. The CalMS21 videos are
"mp4v". This module reads the video's codec from its file (no extra package), and can write an H.264 copy next to
the original (`<name>_web.mp4`). Two converters are tried, nothing has to be installed on Windows:
  - the converter that is built into Windows (scripts/win_transcode.ps1, Media Foundation), or
  - the ffmpeg of the optional package imageio-ffmpeg, if it is installed (used first when present).
The original is never changed. Nothing here is imported unless a page needs it.
"""

from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
from pathlib import Path

PLAYABLE = ("h264", "vp9", "av1")
SUFFIX = "_web"
WIN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "win_transcode.ps1"


class ConverterMissing(RuntimeError):
    """No ffmpeg is available (install the optional package imageio-ffmpeg)."""


class ConversionFailed(RuntimeError):
    """The converter ran but the result is not a good copy."""


def _moov_in(f):
    """The bytes of the 'moov' box (the index) in an open binary file, wherever it sits, or None."""
    f.seek(0, 2)
    end = f.tell()
    pos = 0
    while pos + 8 <= end:
        f.seek(pos)
        size, kind = struct.unpack(">I4s", f.read(8))
        header = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            return None
        if kind == b"moov":
            f.seek(pos + header)
            return f.read(size - header)
        pos += size
    return None


def _moov(path):
    with open(path, "rb") as f:
        return _moov_in(f)


def _codec_name(moov):
    if moov is None:
        return None
    for marker, name in ((b"avc1", "h264"), (b"avc3", "h264"), (b"vp09", "vp9"), (b"av01", "av1"),
                         (b"hvc1", "hevc"), (b"hev1", "hevc"), (b"mp4v", "mp4v")):
        if marker in moov:
            return name
    return "other"


def video_codec(path):
    """'h264', 'hevc', 'vp9', 'av1', 'mp4v' or 'other' for an mp4 / mov file; 'webm' for a .webm; None if unreadable."""
    path = Path(path)
    if path.suffix.lower() == ".webm":
        return "webm"
    try:
        return _codec_name(_moov(path))
    except OSError:
        return None


def seconds(path):
    """Length of an mp4 / mov in seconds from its header, or None."""
    try:
        moov = _moov(path)
    except OSError:
        return None
    if moov is None or b"mvhd" not in moov:
        return None
    i = moov.find(b"mvhd")
    try:
        if moov[i + 4] == 1:
            timescale, duration = struct.unpack(">IQ", moov[i + 24:i + 36])
        else:
            timescale, duration = struct.unpack(">II", moov[i + 16:i + 24])
    except struct.error:
        return None
    return duration / timescale if timescale else None


def codec_of_bytes(name, data):
    """The same as video_codec, for a video that is only in memory (an upload that is not saved yet)."""
    if Path(name).suffix.lower() == ".webm":
        return "webm"
    try:
        return _codec_name(_moov_in(io.BytesIO(data)))
    except (OSError, struct.error):
        return None


def is_playable_bytes(name, data):
    """(True/False, short reason) for a video in memory."""
    return _playable(codec_of_bytes(name, data))


def is_browser_playable(path):
    """(True/False, short reason). A .webm is taken as playable; an mp4 / mov is checked by its codec."""
    return _playable(video_codec(path))


def _playable(codec):
    if codec is None:
        return False, "the video file could not be read"
    if codec == "webm" or codec in PLAYABLE:
        return True, ""
    names = {"mp4v": "an older MPEG-4 format (mp4v)", "hevc": "HEVC, which many browsers cannot play",
             "other": "a format browsers do not play"}
    return False, f"the video uses {names.get(codec, codec)}, which the browser cannot show"


def web_copy_path(path):
    path = Path(path)
    return path.with_name(f"{path.stem}{SUFFIX}.mp4")


def converter_exe():
    """Path of an ffmpeg program, or None."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                    # not installed, or no bundled program for this system
        return None


def windows_converter_available():
    """True on Windows with PowerShell: the converter built into Windows can be used, nothing to install."""
    return os.name == "nt" and WIN_SCRIPT.exists() and shutil.which("powershell") is not None


def converter_available():
    """True if some converter can make a browser-friendly copy."""
    return bool(converter_exe()) or windows_converter_available()


def make_web_copy(src, dst=None, *, exe=None, runner=subprocess.run, tolerance_s=0.5):
    """Write an H.264 copy (faststart, no audio) and check it. Returns the new path.

    The copy must be H.264 and about as long as the original, otherwise it is deleted and ConversionFailed is raised.
    """
    src = Path(src)
    dst = Path(dst) if dst is not None else web_copy_path(src)
    exe = exe or converter_exe()
    if exe:
        command = [exe, "-y", "-v", "error", "-i", str(src), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    elif windows_converter_available():
        command = [shutil.which("powershell") or "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(WIN_SCRIPT), "-Source", str(src), "-Destination", str(dst)]
    else:
        raise ConverterMissing("no video converter: this computer has no built-in one (Windows has), and the optional "
                               "package imageio-ffmpeg is not installed in the project's Python")
    result = runner(command, capture_output=True, text=True)
    if getattr(result, "returncode", 1) != 0 or not dst.exists():
        dst.unlink(missing_ok=True)
        raise ConversionFailed(f"the converter failed: {(getattr(result, 'stderr', '') or '')[-200:].strip() or 'no message'}")
    ok, why = is_browser_playable(dst)
    before, after = seconds(src), seconds(dst)
    if not ok or (before is not None and after is not None and abs(before - after) > tolerance_s):
        dst.unlink(missing_ok=True)
        raise ConversionFailed(f"the copy is not good (playable: {ok}; length {after} s, original {before} s)")
    return dst


def playable_video(path):
    """The file the browser should play: the video itself if it plays, else its web copy if one exists, else None."""
    path = Path(path)
    if is_browser_playable(path)[0]:
        return path
    copy = web_copy_path(path)
    if copy.exists() and is_browser_playable(copy)[0]:
        return copy
    return None
