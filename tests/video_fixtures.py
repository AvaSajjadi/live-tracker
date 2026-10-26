"""Tiny fake mp4 files for tests: just enough boxes (ftyp, mdat, moov with a length and a codec marker)."""

import struct


def box(kind, payload=b""):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def mvhd(seconds, timescale=1000):
    return box(b"mvhd", bytes(4) + struct.pack(">IIII", 0, 0, timescale, int(seconds * timescale)) + bytes(80))


def fake_mp4(codec=b"avc1", seconds=10.0, moov_first=False):
    moov = box(b"moov", mvhd(seconds) + box(b"trak", box(b"stsd", bytes(8) + box(codec, bytes(20)))))
    body = box(b"ftyp", b"isom" + bytes(16)) + box(b"free")
    media = box(b"mdat", b"x" * 100)
    return body + (moov + media if moov_first else media + moov)
