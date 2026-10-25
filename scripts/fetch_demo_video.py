"""Download ONE CalMS21 video (about 200 MB) from the public dataset and check it. Nothing else is downloaded.

Source: https://data.caltech.edu/records/s0vdx-0k302 (file task1_videos_mp4.zip, 28 GB, license CC BY 4.0).
Credit: Sun et al. 2021, "The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions", arXiv:2104.02710.

Only the one video is read, with byte ranges, so the 28 GB archive is never downloaded. The download is checked against the
archive's own size and CRC32 and is deleted if it does not match. Only train and validation sessions are allowed: a test
video is refused. Certificates are checked by curl (the system's own store); this script never turns checking off.

    python scripts/fetch_demo_video.py
    python scripts/fetch_demo_video.py --session task1/train/mouse064_task1_annotator1 --out data/raw/calms21_video

The video is saved next to where the demo expects it (git-ignored: videos never go into Git). Then run
scripts/prepare_demo_session.py for its keypoints, and see docs/run_on_another_computer.md.
"""
import argparse
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.caltech.edu/records/s0vdx-0k302/files/task1_videos_mp4.zip?download=1"
DEFAULT_SESSION = "task1/train/mouse064_task1_annotator1"
CHUNK = 16 * 1024 * 1024


def fetch(url, start, end):
    """Bytes start..end (inclusive) of the file at url, and the file's total size. The link redirects to a signed
    address that lasts about a minute, so every read asks again."""
    if not shutil.which("curl"):
        raise SystemExit("curl is needed (it ships with Windows 10 and later, macOS and most Linux systems)")
    fd, header_path = tempfile.mkstemp()
    os.close(fd)
    try:
        out = subprocess.run(["curl", "-s", "-L", "--fail", "-r", f"{start}-{end}", "--max-time", "300", "-D", header_path, url],
                             capture_output=True, check=True)
        head = Path(header_path).read_bytes()
    finally:
        os.remove(header_path)
    total = int(re.findall(rb"content-range: bytes \d+-\d+/(\d+)", head, re.I)[-1])
    return out.stdout, total


class RangeFile(io.RawIOBase):
    """A read-only file over a remote one, so zipfile can read the table of contents without downloading the archive."""

    def __init__(self, reader):
        self.reader = reader
        _, self.size = reader(0, 0)
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data, _ = self.reader(self.pos, self.pos + n - 1)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def check_session(session, manifest):
    """The video's file name, or SystemExit if the session may not be used."""
    if session in manifest["test"] or re.search(r"(^|/)test(/|$)", session):
        raise SystemExit("refused: test-split videos are never used by the agent")
    if session not in manifest["train"] + manifest["validation"]:
        raise SystemExit(f"refused: {session!r} is not in the train or validation split")
    return session.rsplit("/", 1)[-1] + ".mp4"


def download(reader, name, out_dir, log=print):
    """Read one entry of the remote zip, inflate it, check size and CRC32, and save it. Returns the path."""
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(reader), buffer_size=1 << 20))
    entries = [i for i in z.infolist() if i.filename.endswith("/" + name) and "/test/" not in i.filename]
    if len(entries) != 1:
        raise SystemExit(f"the video {name} is not in the archive (found {len(entries)} matches)")
    zi = entries[0]
    target = Path(out_dir) / name
    if target.exists() and target.stat().st_size == zi.file_size:
        log(f"already there: {target} ({zi.file_size / 1e6:.0f} MB)")
        return target
    header, _ = reader(zi.header_offset, zi.header_offset + 29)
    sig, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
    if sig != 0x04034B50:
        raise SystemExit("the archive's file header is not what was expected")
    pos, remaining = zi.header_offset + 30 + name_len + extra_len, zi.compress_size
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    inflater = zlib.decompressobj(-15) if zi.compress_type == 8 else None
    crc = written = 0
    try:
        with open(part, "wb") as out:
            while remaining > 0:
                n = min(CHUNK, remaining)
                data, _ = reader(pos, pos + n - 1)
                pos, remaining = pos + n, remaining - n
                block = inflater.decompress(data) if inflater else data
                crc, written = zlib.crc32(block, crc), written + len(block)
                out.write(block)
                log(f"  {written / 1e6:.0f} / {zi.file_size / 1e6:.0f} MB")
            if inflater:
                tail = inflater.flush()
                crc, written = zlib.crc32(tail, crc), written + len(tail)
                out.write(tail)
        if written != zi.file_size or (crc & 0xFFFFFFFF) != zi.CRC:
            raise SystemExit("CHECK FAILED: the download does not match the archive's size and checksum; it was deleted")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    log(f"saved and checked: {target}")
    return target


def main(argv=None, reader=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--out", default="data/raw/calms21_video")
    ap.add_argument("--manifest", default="splits/split_manifest.json")
    a = ap.parse_args(argv)
    manifest = json.loads((ROOT / a.manifest).read_text(encoding="utf-8"))
    name = check_session(a.session, manifest)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    download(reader or (lambda s, e: fetch(URL, s, e)), name, out)
    print("Credit when you use it: Sun et al. 2021, arXiv:2104.02710 (CalMS21, CC BY 4.0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
