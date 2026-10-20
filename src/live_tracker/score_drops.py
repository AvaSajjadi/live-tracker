"""EXPERIMENTAL (Stage 9, checkpoint S9-1a): verified ingestion of Igor's
classifier score drops (new file).

A score drop is a ZIP archive shared outside Git, holding per-frame score
tables (INTERFACES.md §2, PR #18 `classify scores`) and a checksum list:

    scores_<split>_<model>.csv     session, frame, time_s, attack, mount, investigation, tracking_ok
    SHA256SUMS.txt                 "<sha256>  <member>" for every CSV member

    drop = load_validation_scores("~/Downloads/igor_score_drops.zip", "scores_validation_model_b.csv")
    drop.provenance          # immutable, honest provenance (ScoreProvenance)
    drop.scores()            # a fresh pandas copy of the verified §2 table

Nothing is extracted: the selected member is read from the archive into
memory (bounded), hashed from exactly those bytes and parsed from them.

Archive integrity (ArchiveIntegrityError), checked before any score is read:
- the file is a regular ZIP within MAX_ARCHIVE_BYTES; optionally its SHA-256
  equals a separately supplied expected digest;
- at most MAX_MEMBERS members; no duplicate names; every name is flat and
  safe (no directories, absolute paths, "..", backslashes, drive letters,
  control characters); no symlinks, no encryption; stored or deflated only;
  each member within MAX_MEMBER_BYTES and MAX_COMPRESSION_RATIO;
- only SHA256SUMS.txt and scores_<split>_<model>.csv members exist;
- SHA256SUMS.txt is present, well-formed, lists every CSV member exactly
  once and nothing else; the selected member's bytes hash to its listed
  digest. The checksum list ships inside the same archive and is unsigned:
  it shows the archive is internally consistent, it does not authenticate
  who produced the scores or with which model.

Split boundary (SplitRefused): only "validation" members are loaded in this
checkpoint. A test member is refused from its name alone; its bytes are never
read (not even hashed).

Score schema (ScoreSchemaError), on the selected member:
- the header is exactly SCORE_COLUMNS, in order; every row has 7 fields;
- the sessions are exactly the tracked manifest's validation split;
- each session is one contiguous block with frames 0..T-1 in order (no
  duplicates, gaps or reordering) and time_s == frame / FPS (|error| <= 1e-9);
- scores are finite numbers in [0, 1]; tracking_ok is literally True / False
  (case-insensitive; no 1/0, no blanks).

Model provenance is not in the archive: the model label is only inferred from
the member's file name and marked unverified; the model artifact, its hash,
its feature / quality-flag configuration and the producing code commit are
recorded as unavailable. Nothing is invented.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import stat
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calms21 import FPS
from .investigation_data import load_manifest

__all__ = ["ArchiveIntegrityError", "MAX_ARCHIVE_BYTES", "MAX_MEMBER_BYTES", "SCORE_COLUMNS", "ScoreDrop",
           "ScoreDropError", "ScoreProvenance", "ScoreSchemaError", "SplitRefused", "inspect_archive",
           "load_validation_scores"]

SCORE_COLUMNS = ("session", "frame", "time_s", "attack", "mount", "investigation", "tracking_ok")
SCORE_CLASSES = ("attack", "mount", "investigation")
CHECKSUMS = "SHA256SUMS.txt"
LOADABLE_SPLIT = "validation"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_CHECKSUMS_BYTES = 64 * 1024
MAX_MEMBERS = 16
MAX_COMPRESSION_RATIO = 100            # deflated text tables compress ~3.5x; a bomb compresses far more
TIME_TOL = 1e-9
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "splits" / "split_manifest.json"

_MEMBER = re.compile(r"scores_(train|validation|test)_([A-Za-z0-9][A-Za-z0-9_.-]{0,63})\.csv")
_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64}) [ *](\S+)")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_UNSAFE = re.compile(r"[\x00-\x1f\x7f\\:]")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_BOOL = {"true": True, "false": False}


class ScoreDropError(ValueError):
    """A score drop cannot be used (base class)."""


class ArchiveIntegrityError(ScoreDropError):
    """The archive, its member structure or its checksums are not trustworthy."""


class SplitRefused(ScoreDropError):
    """The selected member belongs to a split this checkpoint never loads."""


class ScoreSchemaError(ScoreDropError):
    """The selected member is not a valid §2 score table for the validation split."""


@dataclass(frozen=True)
class ScoreProvenance:
    """What is known, and verified, about one loaded score member."""
    archive_name: str
    archive_sha256: str
    archive_sha256_check: str          # "matched expected digest" | "no expected digest supplied"
    member: str
    member_sha256: str
    member_sha256_check: str           # always "matched SHA256SUMS.txt (in-band, unsigned)"
    split: str
    manifest_sha256: str
    n_sessions: int
    n_frames: int
    model_label: str
    model_label_status: str            # never verified: inferred from the file name
    model_artifact_sha256: None        # unavailable: not supplied with the archive
    model_id: None
    feature_configuration: None
    producing_commit: None
    model_provenance_status: str

    def to_json(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class ScoreDrop:
    """A verified validation score table and its provenance."""
    provenance: ScoreProvenance
    _table: pd.DataFrame

    def scores(self):
        """A fresh copy of the verified §2 table (session, frame, time_s, scores, tracking_ok)."""
        return self._table.copy(deep=True)

    def rows(self):
        """Verified rows as dicts, in file order (for the streaming monitor)."""
        return self._table.to_dict("records")


# ======================================================================
# archive integrity
# ======================================================================

def _snapshot(zip_path):
    """The archive's bytes, read ONCE through one file descriptor (bounded by
    MAX_ARCHIVE_BYTES). Every later check and read uses only these bytes, so the
    reported digest always describes the archive actually consumed, whatever
    happens to the path afterwards."""
    path = Path(zip_path).expanduser()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError:
        raise ArchiveIntegrityError(f"archive {path.name!r} is not a regular file") from None
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        regular = False
    if not regular:
        os.close(fd)
        raise ArchiveIntegrityError(f"archive {path.name!r} is not a regular file")
    with os.fdopen(fd, "rb") as f:
        data = f.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ArchiveIntegrityError(f"archive is larger than {MAX_ARCHIVE_BYTES} bytes")
    return path.name, bytes(data)


def _member_problems(info):
    name = info.filename
    if info.orig_filename != name or _CONTROL.search(info.orig_filename):
        # zipfile truncates a stored name at NUL (and may normalise separators): the name as stored must be
        # exactly the name that is checked and used
        return f"member name {info.orig_filename[:80]!r} is altered when read (control character or separator)"
    if not name or name.startswith(("/", "~")) or "/" in name or _UNSAFE.search(name) or name in (".", ".."):
        return f"unsafe member path {name!r}"
    if info.is_dir():
        return f"directory member {name!r}"
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        return f"symlink member {name!r}"
    if info.flag_bits & 0x1:
        return f"encrypted member {name!r}"
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        return f"member {name!r} uses an unsupported compression method"
    if info.file_size > MAX_MEMBER_BYTES:
        return f"member {name!r} is larger than {MAX_MEMBER_BYTES} bytes"
    if info.file_size > MAX_COMPRESSION_RATIO * max(info.compress_size, 1) + 1024:
        return f"member {name!r} has an implausible compression ratio"
    if name != CHECKSUMS and not _MEMBER.fullmatch(name):
        return f"unexpected member {name!r} (only {CHECKSUMS} and scores_<split>_<model>.csv)"
    return None


def _read_member(z, info, limit):
    """The member's bytes, read in bounded chunks (never more than its declared
    size, never more than `limit`); zipfile checks the CRC at the end."""
    if info.file_size > limit:
        raise ArchiveIntegrityError(f"member {info.filename!r} is larger than {limit} bytes")
    out, total = [], 0
    try:
        with z.open(info) as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > info.file_size or total > limit:
                    raise ArchiveIntegrityError(f"member {info.filename!r} decompresses beyond its declared size")
                out.append(chunk)
    except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
        raise ArchiveIntegrityError(f"member {info.filename!r} is corrupt ({type(exc).__name__})") from None
    if total != info.file_size:
        raise ArchiveIntegrityError(f"member {info.filename!r} is shorter than its declared size")
    return b"".join(out)


def _parse_checksums(raw, csv_members):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ArchiveIntegrityError(f"{CHECKSUMS} is not UTF-8 text") from None
    listed = {}
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = _CHECKSUM_LINE.fullmatch(line.strip("\r"))
        if not m:
            raise ArchiveIntegrityError(f"{CHECKSUMS} line {i} is not '<sha256>  <member>'")
        digest, name = m.groups()
        if name in listed:
            raise ArchiveIntegrityError(f"{CHECKSUMS} lists {name!r} twice")
        listed[name] = digest
    if not listed:
        raise ArchiveIntegrityError(f"{CHECKSUMS} lists no member")
    unknown = sorted(set(listed) - set(csv_members))
    unlisted = sorted(set(csv_members) - set(listed))
    if unknown or unlisted:
        raise ArchiveIntegrityError(f"{CHECKSUMS} does not match the archive (listed but absent: {unknown}; "
                                    f"present but unlisted: {unlisted})")
    return listed


def inspect_archive(zip_path, *, expected_archive_sha256=None):
    """(archive sha256, {member: listed sha256}, {member: (split, model label)})
    after every structural and checksum-list check; no score is read."""
    _, archive_sha, listed, csv_members, _ = _open_snapshot(zip_path, expected_archive_sha256)
    return archive_sha, listed, csv_members


def _open_snapshot(zip_path, expected_archive_sha256):
    """(archive name, sha256, listed digests, csv members, snapshot bytes), all from one snapshot."""
    if expected_archive_sha256 is not None and not (isinstance(expected_archive_sha256, str)
                                                    and _HEX64.fullmatch(expected_archive_sha256)):
        raise ArchiveIntegrityError("the expected archive digest is not a lower-case SHA-256 hex string")
    name, data = _snapshot(zip_path)
    archive_sha = hashlib.sha256(data).hexdigest()
    if expected_archive_sha256 is not None and archive_sha != expected_archive_sha256:
        raise ArchiveIntegrityError("archive SHA-256 differs from the expected digest")
    listed, csv_members = _structure(data)
    return name, archive_sha, listed, csv_members, data


def _zip(data):
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, zipfile.LargeZipFile):
        raise ArchiveIntegrityError("not a ZIP archive") from None


def _structure(data):
    with _zip(data) as z:
        infos = z.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ArchiveIntegrityError(f"more than {MAX_MEMBERS} members")
        dupes = sorted(n for n, k in Counter(i.filename for i in infos).items() if k > 1)
        if dupes:
            raise ArchiveIntegrityError(f"duplicate member names {dupes}")
        for info in infos:
            problem = _member_problems(info)
            if problem:
                raise ArchiveIntegrityError(problem)
        names = {i.filename: i for i in infos}
        if CHECKSUMS not in names:
            raise ArchiveIntegrityError(f"{CHECKSUMS} is missing")
        csv_members = {n: _MEMBER.fullmatch(n).groups() for n in names if n != CHECKSUMS}
        listed = _parse_checksums(_read_member(z, names[CHECKSUMS], MAX_CHECKSUMS_BYTES), csv_members)
    return listed, csv_members


# ======================================================================
# the score table
# ======================================================================

def _number(text, where):
    try:
        v = float(text)
    except (TypeError, ValueError):
        raise ScoreSchemaError(f"{where}: {text!r} is not a number") from None
    if not math.isfinite(v):
        raise ScoreSchemaError(f"{where}: {text!r} is not finite")
    return v


def _rows(reader, member):
    """(line number, fields) for each record; csv.Error (unterminated quote,
    bad escaping, a field over csv.field_size_limit()) becomes ScoreSchemaError."""
    while True:
        try:
            row = next(reader)
        except StopIteration:
            return
        except csv.Error as exc:
            raise ScoreSchemaError(f"{member} line {reader.line_num}: malformed CSV ({exc})") from None
        yield reader.line_num, row


def _parse_table(raw, member, validation):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ScoreSchemaError(f"{member}: not UTF-8 text") from None
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    rows = _rows(reader, member)
    header = next(rows, (1, None))[1]
    if header is None or tuple(header) != SCORE_COLUMNS:
        raise ScoreSchemaError(f"{member}: header {str(header)[:200]} is not exactly {list(SCORE_COLUMNS)}")
    sessions, frames, times, scores, ok = [], [], [], {c: [] for c in SCORE_CLASSES}, []
    seen, current, expected = set(), None, 0
    for line, row in rows:
        where = f"{member} line {line}"
        if len(row) != len(SCORE_COLUMNS):
            raise ScoreSchemaError(f"{where}: {len(row)} fields, expected {len(SCORE_COLUMNS)}")
        sid, frame_t, time_t, a, m, inv, tok = row
        if sid not in validation:
            raise ScoreSchemaError(f"{where}: session {sid[:80]!r} is not in the manifest's validation split")
        if sid != current:
            if sid in seen:
                raise ScoreSchemaError(f"{where}: session {sid!r} is not one contiguous block")
            seen.add(sid)
            current, expected = sid, 0
        if not re.fullmatch(r"0|[1-9]\d{0,8}", frame_t):
            raise ScoreSchemaError(f"{where}: frame {frame_t!r} is not a non-negative integer")
        frame = int(frame_t)
        if frame != expected:
            raise ScoreSchemaError(f"{where}: {sid} frame {frame} (expected {expected}: frames are 0..T-1 in "
                                   "order, without duplicates or gaps)")
        expected += 1
        t = _number(time_t, f"{where} time_s")
        if t < 0:                               # the monitor's contract: observation times are never negative
            raise ScoreSchemaError(f"{where}: time_s {t!r} is negative")
        if abs(t - frame / FPS) > TIME_TOL:
            raise ScoreSchemaError(f"{where}: time_s {t!r} is not frame / {FPS:g}")
        for col, value in zip(SCORE_CLASSES, (a, m, inv)):
            v = _number(value, f"{where} {col}")
            if not 0.0 <= v <= 1.0:
                raise ScoreSchemaError(f"{where}: {col} {v!r} is outside [0, 1]")
            scores[col].append(v)
        flag = _BOOL.get(tok.strip().lower()) if tok == tok.strip() else None
        if flag is None:
            raise ScoreSchemaError(f"{where}: tracking_ok {tok[:20]!r} is not True / False")
        sessions.append(sid)
        frames.append(frame)
        times.append(t)
        ok.append(flag)
    if seen != set(validation):
        raise ScoreSchemaError(f"{member}: sessions are not exactly the manifest's validation split (missing "
                               f"{sorted(set(validation) - seen)[:5]})")
    table = pd.DataFrame({"session": sessions, "frame": np.asarray(frames, dtype=np.int64),
                          "time_s": np.asarray(times, dtype=float),
                          **{c: np.asarray(v, dtype=float) for c, v in scores.items()},
                          "tracking_ok": np.asarray(ok, dtype=bool)})
    return table[list(SCORE_COLUMNS)]


def load_validation_scores(zip_path, member, *, manifest_path=DEFAULT_MANIFEST, expected_archive_sha256=None):
    """A verified validation score table (ScoreDrop) from one archive member.

    Raises ArchiveIntegrityError (archive / checksums), SplitRefused (a
    non-validation member: never read) or ScoreSchemaError (table)."""
    archive_name, archive_sha, listed, csv_members, data = _open_snapshot(zip_path, expected_archive_sha256)
    if member not in csv_members:
        raise ArchiveIntegrityError(f"no score member {str(member)[:80]!r} in the archive")
    split, model_label = csv_members[member]
    if split != LOADABLE_SPLIT:
        raise SplitRefused(f"{member!r} is a {split}-split member; only {LOADABLE_SPLIT} members are loaded in "
                           "this checkpoint (its contents were not read)")
    try:
        manifest, manifest_sha = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise ScoreSchemaError(f"the split manifest cannot be used ({type(exc).__name__})") from None
    validation = frozenset(manifest["validation"])
    with _zip(data) as z:                       # the same snapshot: the path is never reopened
        raw = _read_member(z, z.getinfo(member), MAX_MEMBER_BYTES)
    member_sha = hashlib.sha256(raw).hexdigest()
    if member_sha != listed[member]:
        raise ArchiveIntegrityError(f"{member!r} does not match its {CHECKSUMS} digest")
    table = _parse_table(raw, member, validation)
    provenance = ScoreProvenance(
        archive_name=archive_name, archive_sha256=archive_sha,
        archive_sha256_check=("matched expected digest" if expected_archive_sha256 is not None
                              else "no expected digest supplied"),
        member=member, member_sha256=member_sha,
        member_sha256_check=f"matched {CHECKSUMS} (in-band, unsigned: archive consistency, not authorship)",
        split=split, manifest_sha256=manifest_sha, n_sessions=int(table["session"].nunique()),
        n_frames=int(len(table)), model_label=model_label,
        model_label_status="unverified: inferred from the member file name only",
        model_artifact_sha256=None, model_id=None, feature_configuration=None, producing_commit=None,
        model_provenance_status=("unavailable: the archive holds no model artifact, model metadata, feature "
                                 "or quality-flag configuration, or producing commit"))
    return ScoreDrop(provenance=provenance, _table=table)
