"""EXPERIMENTAL (Investigation mode, Stage 1): a verified local cache of a few
CalMS21 validation sessions (new file).

Why: calms21.load_calms21() always parses the whole JSON (1.2 GB for the
train file) before selecting sessions. For development we read the source
ONCE, copy the requested validation sessions out in the original layout
({group: {session_id: record}}), and later read the small files with the
unchanged load_calms21().

Two boundaries, both enforced here:

1. Extraction (extract_sessions)
   - Requests are checked against the split manifest before the source is
     touched: only sessions of the allowed split ("validation"); any test
     session, unlisted session or "test" as the allowed split is refused.
   - The source is opened read-only ("rb"), read once, hashed (sha256) from
     the same bytes that are parsed, and its size / mtime must not change
     while it is read. A source containing any test session is refused.
   - Every record (requested sessions, and the annotations of every session
     of the allowed split used for the label summary) is validated BEFORE
     anything is published: Igor's load_calms21 must accept it, and in
     addition metadata.vocab must be present and equal calms21.VOCAB, and
     every annotation must be a plain int label id.
   - Writers (and readers) of one cache directory are serialized by an
     exclusive OS lock (.cache.lock; fcntl.flock on POSIX, msvcrt.locking on
     Windows; released by the OS if the process dies).
   - Publication is atomic: each payload is serialized once, hashed from
     those exact bytes, written to a unique *.partial file, fsynced and
     renamed; the index (extraction_index.json) is written last, the same way.
     Only sessions listed in the index count as extracted, each bound to its
     payload sha256 and to the source and manifest identities. An interrupted
     run therefore never looks successful: unlisted files are ignored and
     left-over *.partial files are removed by the next locked run.
   - One policy for every path: reuse, incremental extraction, fresh
     extraction and analysis all run the same full verification
     (_verify_locked: strict index structure, manifest binding, allowed split,
     canonical confined files, payload hashes and sizes, session IDs inside
     the files, frame counts, and the label summary validated against the
     manifest and the records). A cache that fails it is never reused or
     retained: it is rebuilt from the source. A cache is reused only if it
     passes, lists every requested session, has the same source path / size /
     mtime, AND the source content still hashes to the recorded sha256 (a full
     read of the source; see the reuse_check timing). Sessions kept from an
     earlier extraction of the same source are re-derived from the source
     records, never copied from old metadata. Every publication is verified
     again before it is returned.
   - Session IDs whose cache file names would collide are refused.

2. Analysis (verified_sessions)
   - Never trusts file names or directory contents: the index must match
     the current manifest's sha256; every listed session must belong to the
     allowed split per the manifest; each file must have its canonical name
     directly inside the cache directory (no absolute paths, traversal or
     symlinks) and match its sha256; the session IDs INSIDE the file must be
     exactly the listed one. The split label summary must cover exactly the
     manifest's split, be well-formed, and agree with every extracted record.
     Anything else is refused before load_calms21 sees the annotations.

Usage:
    python -m live_tracker.investigation_data --all \
        --dataset-root "/Volumes/T7 Hackathon/Datasets/CalMS21"      # or $LIVE_TRACKER_CALMS21_ROOT
    python -m live_tracker.investigation_data \
        --source ".../calms21_task1_train.json" \
        --sessions task1/train/mouse005_task1_annotator1 ...

--all extracts every session of --split (default validation) listed in the
manifest: coverage is decided by the manifest, not by a list of sessions.
The source is the official CalMS21 Task 1 train file; --dataset-root (or
$LIVE_TRACKER_CALMS21_ROOT) may point at a directory containing it (directly
or under extracted/task1_classic_classification/). An unavailable source is
an explicit refusal; nothing is substituted.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

try:  # not available on Windows; memory figures are then reported as None
    import resource
except ImportError:  # pragma: no cover - exercised by monkeypatching in tests
    resource = None

from .calms21 import VOCAB, _sessions, load_calms21

FORMAT = "live-tracker-calms21-extraction/2"
DEFAULT_OUT = Path("data/raw/calms21_validation")
ROOT_ENV = "LIVE_TRACKER_CALMS21_ROOT"
TRAIN_FILE = "calms21_task1_train.json"
_ROOT_LAYOUTS = (Path(TRAIN_FILE), Path("extracted") / "task1_classic_classification" / TRAIN_FILE)
INDEX_NAME = "extraction_index.json"
_SPLITS = ("train", "validation", "test")


class ExtractionRefused(ValueError):
    """The request would read, publish or analyse data it must not."""


class CacheError(RuntimeError):
    """The cache is incomplete, stale or tampered with."""


# ---- identities ----

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _stat(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def peak_rss_mb():
    """Peak resident memory of this process in MB, or None where unavailable."""
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 1e6 if sys.platform == "darwin" else rss / 1e3, 1)


def load_manifest(path):
    """(manifest dict, sha256 of the manifest file)."""
    raw = Path(path).read_bytes()
    m = json.loads(raw)
    for split in _SPLITS:
        if not isinstance(m.get(split), list) or not all(isinstance(s, str) for s in m[split]):
            raise ValueError(f"manifest has no list of session ids {split!r}")
    seen = {}
    for split in _SPLITS:
        for sid in m[split]:
            if sid in seen:
                raise ValueError(f"session {sid!r} is in both {seen[sid]!r} and {split!r}")
            seen[sid] = split
    return m, hashlib.sha256(raw).hexdigest()


def session_filename(session_id):
    """task1/train/mouse005_task1_annotator1 -> task1__train__mouse005_task1_annotator1.json"""
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", session_id) or ".." in session_id:
        raise ExtractionRefused(f"unsafe session id {session_id!r}")
    return session_id.replace("/", "__") + ".json"


def _check_allowed(sessions, manifest, allowed_split):
    if allowed_split not in ("train", "validation"):
        raise ExtractionRefused(f"split {allowed_split!r} is never extracted or analysed")
    test = set(manifest["test"])
    allowed = set(manifest[allowed_split])
    bad_test = sorted(set(sessions) & test)
    if bad_test:
        raise ExtractionRefused(f"held-out test sessions refused: {bad_test}")
    not_allowed = sorted(set(sessions) - allowed)
    if not_allowed:
        raise ExtractionRefused(f"not in the {allowed_split!r} split: {not_allowed}")


# ---- record validation (before publication) ----

def validate_annotations(sid, rec):
    """Stricter than the loader: vocab present and exact, plain int label ids."""
    if not isinstance(rec, dict):
        raise ExtractionRefused(f"{sid}: record is not an object")
    vocab = rec.get("metadata", {}).get("vocab") if isinstance(rec.get("metadata"), dict) else None
    if vocab != VOCAB:
        raise ExtractionRefused(f"{sid}: metadata.vocab {vocab!r} is missing or differs from {VOCAB}")
    ann = rec.get("annotations")
    if not isinstance(ann, list) or not ann:
        raise ExtractionRefused(f"{sid}: annotations must be a non-empty list")
    ids = set(VOCAB.values())
    if not all(type(a) is int and a in ids for a in ann):
        raise ExtractionRefused(f"{sid}: annotations must be int label ids in {sorted(ids)}")
    kp = rec.get("keypoints")
    if not isinstance(kp, list) or len(kp) != len(ann):
        raise ExtractionRefused(f"{sid}: {len(ann)} annotations but "
                                f"{len(kp) if isinstance(kp, list) else 'no'} keypoint frames")


def validate_record(group, sid, rec):
    """Full check of one record: the strict annotation checks, then Igor's
    loader on that record alone (shapes, vocab, label range, lengths)."""
    validate_annotations(sid, rec)
    try:
        poses, labels = load_calms21({group: {sid: rec}})
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ExtractionRefused(f"{sid}: rejected by calms21.load_calms21: {exc}") from exc
    if len(labels) != len(rec["annotations"]):
        raise ExtractionRefused(f"{sid}: loader returned {len(labels)} labels")
    return len(labels)


def label_runs(annotations):
    """Count runs of each label in an annotation list (no thresholds)."""
    names = {v: k for k, v in VOCAB.items()}
    runs = {k: 0 for k in VOCAB}
    prev = None
    for x in annotations:
        if x != prev:
            runs[names[x]] += 1
            prev = x
    return runs


# ---- one writer / reader at a time per cache directory ----

LOCK_NAME = ".cache.lock"

# Errors msvcrt.locking raises when the region is held by someone else:
# LK_LOCK gives up after ~10 one-second attempts with EDEADLOCK (= EDEADLK);
# a locked region can also surface as EACCES. Only these mean "try again";
# anything else (EBADF, EINVAL, ...) is a real failure and propagates at once.
WINDOWS_LOCK_CONTENTION = frozenset(
    {errno.EACCES, getattr(errno, "EDEADLK", 36), getattr(errno, "EDEADLOCK", 36)})


def _windows_acquire(fd, msvcrt_module):
    """Block until byte 0 of fd is locked, retrying only on lock contention."""
    while True:
        os.lseek(fd, 0, os.SEEK_SET)  # locking() locks from the current position
        try:
            msvcrt_module.locking(fd, msvcrt_module.LK_LOCK, 1)
            return
        except OSError as exc:
            if exc.errno not in WINDOWS_LOCK_CONTENTION:
                raise


def _windows_release(fd, msvcrt_module):
    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt_module.locking(fd, msvcrt_module.LK_UNLCK, 1)


if os.name == "nt":  # pragma: no cover - Windows
    import msvcrt

    def _acquire(fd):
        _windows_acquire(fd, msvcrt)

    def _release(fd):
        _windows_release(fd, msvcrt)
else:
    import fcntl

    def _acquire(fd):
        fcntl.flock(fd, fcntl.LOCK_EX)  # released by the OS if the process dies

    def _release(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def cache_lock(out_dir):
    """Exclusive lock on out_dir/.cache.lock, held by extraction (check, read,
    write, publish) and by the analysis while it verifies the cache, so a
    reader never sees a half-published update and two writers never interleave."""
    fd = os.open(Path(out_dir) / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _acquire(fd)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


# ---- atomic writes and confined paths ----

def _write_atomic(path, payload):
    """Write bytes to a unique temporary file, fsync, then rename over `path`."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    with open(tmp, "xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cache_file(out_dir, session_id, listed_name):
    """The only file a session may be read from: its canonical name directly
    inside out_dir, not a symlink. Anything else is refused before opening."""
    expected = session_filename(session_id)
    if listed_name != expected:
        raise ExtractionRefused(f"{session_id}: index names {listed_name!r}, expected {expected!r}")
    base = Path(out_dir).resolve(strict=True)
    path = base / expected
    if path.is_symlink():
        raise ExtractionRefused(f"{expected}: symlinks are not accepted in the cache")
    if path.resolve().parent != base:
        raise ExtractionRefused(f"{expected}: resolves outside the cache directory")
    return path


def _read_index(out_dir):
    path = Path(out_dir) / INDEX_NAME
    if path.is_symlink():
        raise ExtractionRefused(f"{INDEX_NAME}: symlinks are not accepted in the cache")
    if not path.exists():
        return None
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return index if isinstance(index, dict) and index.get("format") == FORMAT else None


def _code_identity():
    return {"investigation_data.py": sha256_file(Path(__file__))}


# ---- strict index structure (checked before any field is used) ----

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_ENTRY_KEYS = {"file", "sha256", "bytes", "frames"}


def _is_hex64(v):
    return isinstance(v, str) and _HEX64_RE.fullmatch(v) is not None


def index_problem(index):
    """None if `index` has exactly the structure this module publishes, else a
    short reason. Malformed metadata is never interpreted (no .get() on
    non-objects, no missing keys)."""
    if type(index) is not dict:
        return "the index is not an object"
    if index.get("format") != FORMAT:
        return "unknown index format"
    src = index.get("source")
    if not (type(src) is dict and isinstance(src.get("path"), str) and _is_count(src.get("size"))
            and _is_count(src.get("mtime_ns")) and _is_hex64(src.get("sha256"))):
        return "malformed source block"
    man = index.get("manifest")
    if not (type(man) is dict and _is_hex64(man.get("sha256"))):
        return "malformed manifest block"
    if not isinstance(index.get("allowed_split"), str):
        return "malformed allowed_split"
    sessions = index.get("sessions")
    if type(sessions) is not dict:
        return "sessions is not an object"
    for sid, e in sessions.items():
        if not (isinstance(sid, str) and type(e) is dict and set(e) == _ENTRY_KEYS
                and isinstance(e["file"], str) and _is_hex64(e["sha256"])
                and _is_count(e["bytes"]) and _is_count(e["frames"], 1)):
            return f"malformed session entry for {sid!r}"
    if type(index.get("split_label_summary")) is not dict:
        return "split_label_summary is not an object"
    return None


def check_filenames_unique(session_ids):
    """Refuse IDs whose cache file names collide ('a/b' and 'a__b' both map to
    'a__b.json'). Real CalMS21 IDs never collide; the convention is kept for
    cache compatibility and collisions are refused instead."""
    seen = {}
    for sid in session_ids:
        name = session_filename(sid)
        if name in seen and seen[name] != sid:
            raise ExtractionRefused(f"session ids {seen[name]!r} and {sid!r} map to the same "
                                    f"cache file {name!r}")
        seen[name] = sid


# ---- extraction ----

def extract_sessions(source, out_dir, sessions, manifest_path, allowed_split="validation"):
    """Publish `sessions` from `source` into `out_dir` (see module doc).

    Returns the published, fully verified index (plus, when an intact cache
    was reused, a "reuse_check" entry with the time spent re-hashing the
    source; it is not written to disk). Raises ExtractionRefused for anything
    that must not be read or published, CacheError if the source changed while
    being read or the published cache fails verification.
    """
    manifest, manifest_sha = load_manifest(manifest_path)
    sessions = sorted(set(sessions))
    if not sessions:
        raise ExtractionRefused("no sessions requested")
    _check_allowed(sessions, manifest, allowed_split)
    check_filenames_unique(manifest[allowed_split])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()
    if not source.exists():
        raise ExtractionRefused(f"source not found: {source}")

    with cache_lock(out_dir):
        for leftover in out_dir.glob("*.partial"):  # no other writer can be active now
            leftover.unlink()
        stat_before = {"path": str(source), **_stat(source)}
        try:  # the same verification as the analysis; anything else is never reused or retained
            _, previous = _verify_locked(out_dir, manifest, manifest_sha, allowed_split)
        except (CacheError, ExtractionRefused, OSError, ValueError):
            previous = None
        if previous is not None and all(sid in previous["sessions"] for sid in sessions) and \
                {k: previous["source"][k] for k in ("path", "size", "mtime_ns")} == stat_before:
            t0 = time.perf_counter()
            if sha256_file(source) == previous["source"]["sha256"]:  # content, not only stat
                return {**previous, "reuse_check": {"source_sha256_verified": True,
                                                    "source_hash_seconds": round(time.perf_counter() - t0, 2)}}

        t0 = time.perf_counter()
        with open(source, "rb") as f:            # read-only, read once
            raw = f.read()
        source_sha = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        del raw
        parse_seconds = time.perf_counter() - t0
        if {"path": str(source), **_stat(source)} != stat_before:
            raise CacheError("the source file changed while it was being read")
        if not isinstance(data, dict) or not all(isinstance(g, dict) for g in data.values()):
            raise ExtractionRefused("source is not a CalMS21 {group: {session: record}} file")

        recs = _sessions(data)
        group_of = {sid: g for g, ss in data.items() for sid in ss}
        test_in_source = sorted(set(recs) & set(manifest["test"]))
        if test_in_source:
            raise ExtractionRefused(f"source contains held-out test sessions {test_in_source[:3]}; "
                                    "only the official train file may be used")
        missing = sorted(set(sessions) - set(recs))
        if missing:
            raise ExtractionRefused(f"sessions not in source: {missing}")
        absent = sorted(set(manifest[allowed_split]) - set(recs))
        if absent:
            raise ExtractionRefused(f"source lacks {allowed_split} sessions {absent[:3]}; "
                                    "the label summary must cover the whole split")

        # sessions already published from this same source content stay published,
        # but are re-derived from the source records like the requested ones: none of
        # their old metadata (frames, bytes, hash) is trusted or copied
        retained = []
        if previous is not None and previous["source"]["sha256"] == source_sha:
            retained = [sid for sid in previous["sessions"] if sid not in sessions]
        to_publish = sorted(set(sessions) | set(retained))

        # validate everything first; nothing is written if any record is bad
        split_sessions = sorted(manifest[allowed_split])
        for sid in split_sessions:
            validate_annotations(sid, recs[sid])
        frames = {sid: validate_record(group_of[sid], sid, recs[sid]) for sid in to_publish}

        published = {}
        for sid in to_publish:
            payload = json.dumps({group_of[sid]: {sid: recs[sid]}}).encode("utf-8")
            target = out_dir / session_filename(sid)
            _write_atomic(target, payload)
            published[sid] = {"file": target.name, "sha256": hashlib.sha256(payload).hexdigest(),
                              "bytes": len(payload), "frames": frames[sid]}

        new_index = {
            "format": FORMAT,
            "experimental": True,
            "source": {**stat_before, "sha256": source_sha},
            "manifest": {"path": str(Path(manifest_path)), "sha256": manifest_sha},
            "allowed_split": allowed_split,
            "code": _code_identity(),
            "sessions": dict(sorted(published.items())),
            "split_label_summary": {sid: {"frames": len(recs[sid]["annotations"]),
                                          "label_runs": label_runs(recs[sid]["annotations"])}
                                    for sid in split_sessions},
            "extraction_run": {"parse_seconds": round(parse_seconds, 2),
                               "peak_rss_mb": peak_rss_mb()},
        }
        _write_atomic(out_dir / INDEX_NAME, json.dumps(new_index, indent=2).encode("utf-8"))
        _, verified = _verify_locked(out_dir, manifest, manifest_sha, allowed_split)  # same policy
        return verified


# ---- analysis boundary ----

def _is_count(v, minimum=0):
    return type(v) is int and v >= minimum


def validated_split_summary(index, manifest, allowed_split, extracted):
    """The index's per-session label summary, refused unless: it covers
    exactly the manifest's `allowed_split`, every entry has a positive frame
    count and non-negative int run counts for exactly the CalMS21 labels that
    fit in the frames, and every extracted session's entry equals what its
    verified record gives. Entries of sessions that were not extracted can only
    be checked for structure and manifest membership (their records are not in
    the cache); they were computed from the source at extraction time."""
    summary = index.get("split_label_summary")
    if type(summary) is not dict:
        raise CacheError("split_label_summary is missing or not an object")
    allowed = set(manifest[allowed_split])
    keys = set(summary)
    bad = sorted(keys & set(manifest["test"]))
    if bad:
        raise ExtractionRefused(f"split_label_summary lists held-out test sessions {bad}")
    if keys != allowed:
        raise CacheError(f"split_label_summary sessions differ from the manifest's {allowed_split!r} "
                         f"split (extra {sorted(keys - allowed)}, missing {sorted(allowed - keys)})")
    out = {}
    for sid in sorted(summary):
        v = summary[sid]
        runs = v.get("label_runs") if type(v) is dict else None
        if not (type(v) is dict and set(v) == {"frames", "label_runs"} and _is_count(v["frames"], 1)
                and type(runs) is dict and set(runs) == set(VOCAB)
                and all(_is_count(x) for x in runs.values())
                and 1 <= sum(runs.values()) <= v["frames"]):
            raise CacheError(f"{sid}: malformed split_label_summary entry")
        if sid in extracted:
            ann = extracted[sid]["annotations"]
            if v["frames"] != len(ann) or runs != label_runs(ann):
                raise CacheError(f"{sid}: split_label_summary does not match the extracted record")
        out[sid] = {"frames": v["frames"], "label_runs": dict(runs)}
    return out


def _verify_locked(out_dir, manifest, manifest_sha, allowed_split):
    """Full verification of the published cache; the caller holds the lock.
    Used by the analysis (verified_sessions), by cache reuse, before sessions
    are retained, and after every publication, so all paths apply one policy.
    Returns ([(session_id, parsed_dict)], index with the validated summary)."""
    index = _read_index(out_dir)
    if index is None:
        raise CacheError(f"no published extraction index ({FORMAT}) in {out_dir}")
    problem = index_problem(index)
    if problem:
        raise CacheError(f"malformed extraction index: {problem}")
    if index["manifest"]["sha256"] != manifest_sha:
        raise CacheError("the cache was built with a different split manifest")
    if index["allowed_split"] != allowed_split:
        raise ExtractionRefused(f"cache split {index['allowed_split']!r} != {allowed_split!r}")
    check_filenames_unique(manifest[allowed_split])
    listed = index["sessions"]
    if not listed:
        raise CacheError("the index lists no sessions")
    _check_allowed(list(listed), manifest, allowed_split)
    out, records = [], {}
    for sid, entry in sorted(listed.items()):
        path = cache_file(out_dir, sid, entry["file"])  # confined before opening
        if not path.exists():
            raise CacheError(f"{sid}: published file is missing")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != entry["bytes"]:
            raise CacheError(f"{sid}: file does not match its published sha256 / size")
        data = json.loads(raw)
        inside = sorted(_sessions(data)) if isinstance(data, dict) and \
            all(isinstance(g, dict) for g in data.values()) else None
        if inside != [sid]:
            raise ExtractionRefused(f"{entry['file']}: contains sessions {inside}, expected [{sid!r}]")
        _check_allowed(inside, manifest, allowed_split)
        rec = next(iter(data.values()))[sid]
        validate_annotations(sid, rec)
        if entry["frames"] != len(rec["annotations"]):
            raise CacheError(f"{sid}: index frame count does not match the record")
        records[sid] = rec
        out.append((sid, data))
    index = {**index, "split_label_summary": validated_split_summary(index, manifest,
                                                                     allowed_split, records)}
    return out, index


def verified_sessions(out_dir, manifest_path, allowed_split="validation"):
    """Published sessions the analysis may use: ([(session_id, parsed_dict)], index).

    Refuses (raises) on any mismatch instead of skipping it; files in the
    directory that the index does not list are never read. The returned index
    carries the validated split_label_summary. Holds the cache lock while
    reading, so it never sees a half-published extraction.
    """
    manifest, manifest_sha = load_manifest(manifest_path)
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise CacheError(f"no cache directory {out_dir}")
    with cache_lock(out_dir):
        return _verify_locked(out_dir, manifest, manifest_sha, allowed_split)


def unlisted_files(out_dir, index):
    """JSON files in the cache directory that are not published (ignored).
    A malformed index is a CacheError, never an unhandled exception."""
    problem = index_problem(index)
    if problem:
        raise CacheError(f"malformed extraction index: {problem}")
    listed = {e["file"] for e in index["sessions"].values()} | {INDEX_NAME}
    return sorted(p.name for p in Path(out_dir).glob("*.json") if p.name not in listed)


def resolve_source(source=None, dataset_root=None, env=None):
    """The CalMS21 train file: `source` if given, else found under
    `dataset_root` or $LIVE_TRACKER_CALMS21_ROOT. ExtractionRefused (never a
    substitute) when it is not available."""
    if source:
        path = Path(source)
        if not path.is_file():
            raise ExtractionRefused(f"source not found: {path}")
        return path
    root = dataset_root or (env if env is not None else os.environ).get(ROOT_ENV)
    if not root:
        raise ExtractionRefused(f"no source: pass --source or --dataset-root, or set ${ROOT_ENV}")
    for rel in _ROOT_LAYOUTS:
        if (Path(root) / rel).is_file():
            return Path(root) / rel
    raise ExtractionRefused(f"{TRAIN_FILE} not found under the dataset root {root} (is the drive mounted?)")


def split_coverage(out_dir, manifest_path, allowed_split="validation"):
    """Which sessions of the manifest's split the verified cache holds:
    {eligible, cached, missing, complete}. Refuses (raises) like
    verified_sessions when the cache does not verify."""
    manifest, _ = load_manifest(manifest_path)
    verified, _ = verified_sessions(out_dir, manifest_path, allowed_split)
    eligible = sorted(manifest[allowed_split])
    cached = sorted(sid for sid, _ in verified)
    missing = sorted(set(eligible) - set(cached))
    return {"split": allowed_split, "eligible": eligible, "cached": cached, "missing": missing,
            "complete": not missing}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=None, help="the CalMS21 Task 1 train JSON file")
    ap.add_argument("--dataset-root", default=None, help=f"directory holding {TRAIN_FILE} (or ${ROOT_ENV})")
    ap.add_argument("--manifest", default="splits/split_manifest.json")
    ap.add_argument("--split", choices=("validation", "train"), default="validation")
    ap.add_argument("--out", default=None, help="cache directory (default data/raw/calms21_<split>)")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--sessions", nargs="+")
    which.add_argument("--all", action="store_true", help="every session of --split in the manifest")
    a = ap.parse_args(argv)
    out = a.out or str(DEFAULT_OUT if a.split == "validation" else Path(f"data/raw/calms21_{a.split}"))
    try:
        source = a.source or resolve_source(None, a.dataset_root)   # --source: checked after the manifest
        sessions = load_manifest(a.manifest)[0][a.split] if a.all else a.sessions
        index = extract_sessions(source, out, sessions, a.manifest, allowed_split=a.split)
    except (ExtractionRefused, CacheError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in index.items() if k != "split_label_summary"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
