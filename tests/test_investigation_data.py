"""Tests for the experimental CalMS21 session cache (new file).

All data here is a tiny SYNTHETIC test fixture in the CalMS21 JSON layout,
not CalMS21 data.
"""

import errno
import hashlib
import json
import os
import sys
import threading

import pytest

import live_tracker.investigation_data as idata
from live_tracker.investigation_data import (INDEX_NAME, CacheError, ExtractionRefused,
                                             extract_sessions, main, peak_rss_mb, session_filename,
                                             unlisted_files, verified_sessions)


def _edit_index(out, change):
    index = json.loads((out / INDEX_NAME).read_text())
    change(index)
    (out / INDEX_NAME).write_text(json.dumps(index))

VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
VAL = ["task1/train/SYNTHETIC_val_a", "task1/train/SYNTHETIC_val_b"]
TRAIN = ["task1/train/SYNTHETIC_train_a"]
TEST = ["task1/test/SYNTHETIC_test_a"]
MANIFEST = {"train": TRAIN, "validation": VAL, "test": TEST}


# filesystem symlinks need extra privileges on Windows, and POSIX permission bits do not exist there
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: filesystem symlinks / permission bits")


def record(labels, offset=0.0, vocab=VOCAB):
    n = len(labels)
    rec = {"keypoints": [[[[100.0 + 300 * m + f + i + offset for i in range(7)],
                           [200.0 + f + i for i in range(7)]] for m in range(2)] for f in range(n)],
           "scores": [[[0.9] * 7 for _ in range(2)] for _ in range(n)],
           "annotations": list(labels),
           "metadata": {"annotator-id": 0}}
    if vocab is not None:
        rec["metadata"]["vocab"] = dict(vocab)
    return rec


def source_data(offset=0.0, with_test=False, **overrides):
    recs = {VAL[0]: record([3, 3, 0, 0, 3, 0], offset), VAL[1]: record([1, 1, 3, 3], offset),
            TRAIN[0]: record([3, 0, 3], offset)}
    if with_test:
        recs[TEST[0]] = record([0, 0], offset)
    recs.update(overrides)
    return {"annotator-id_0": recs}


@pytest.fixture
def env(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST))
    src = tmp_path / "source.json"
    src.write_text(json.dumps(source_data()))
    return tmp_path, src, manifest, tmp_path / "cache"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- extraction ----

def test_extract_publishes_verified_sessions(env):
    tmp, src, manifest, out = env
    before = digest(src)
    index = extract_sessions(src, out, [VAL[0]], manifest)
    assert digest(src) == before  # source never modified
    assert index["source"]["sha256"] == before
    assert index["manifest"]["sha256"] == digest(manifest)
    assert index["sessions"][VAL[0]]["sha256"] == digest(out / session_filename(VAL[0]))
    sessions, _ = verified_sessions(out, manifest)
    assert [sid for sid, _ in sessions] == [VAL[0]]


def test_split_label_summary_is_validation_only(env):
    _, src, manifest, out = env
    summary = extract_sessions(src, out, [VAL[0]], manifest)["split_label_summary"]
    assert set(summary) == set(VAL)
    assert summary[VAL[0]] == {"frames": 6, "label_runs": {"attack": 2, "investigation": 0,
                                                           "mount": 0, "other": 2}}
    assert not (out / session_filename(VAL[1])).exists()


@pytest.mark.parametrize("sessions, match", [
    (TEST, "held-out test"), ([VAL[0], TEST[0]], "held-out test"),
    (TRAIN, "not in the 'validation' split"), (["task1/train/unknown"], "not in the 'validation'"),
    ([], "no sessions"),
])
def test_requests_are_checked_before_the_source_is_touched(env, sessions, match):
    tmp, _, manifest, out = env
    with pytest.raises(ExtractionRefused, match=match):
        extract_sessions(tmp / "missing.json", out, sessions, manifest)
    assert not (out / INDEX_NAME).exists()


def test_test_split_is_never_allowed(env):
    tmp, src, manifest, out = env
    with pytest.raises(ExtractionRefused, match="never extracted"):
        extract_sessions(src, out, TEST, manifest, allowed_split="test")


def test_source_containing_test_sessions_is_refused(env):
    tmp, src, manifest, out = env
    src.write_text(json.dumps(source_data(with_test=True)))
    with pytest.raises(ExtractionRefused, match="contains held-out test sessions"):
        extract_sessions(src, out, [VAL[0]], manifest)
    assert not (out / INDEX_NAME).exists() and not (out / session_filename(VAL[0])).exists()


# ---- record validation before publication ----

def _bad(rec_change=None, target=VAL[0], **rec_kwargs):
    rec = record([3, 3, 0, 0, 3, 0], **rec_kwargs)
    if rec_change:
        rec_change(rec)
    return {target: rec}


@pytest.mark.parametrize("override, match", [
    (_bad(vocab={"attack": 1, "investigation": 0, "mount": 2, "other": 3}), "vocab"),
    (_bad(vocab=None), "vocab"),
    (_bad(lambda r: r.pop("metadata")), "vocab"),
    (_bad(lambda r: r["metadata"].__setitem__("vocab", {"attack": 0})), "vocab"),
    (_bad(lambda r: r["annotations"].__setitem__(2, 0.0)), "int label ids"),
    (_bad(lambda r: r["annotations"].__setitem__(2, True)), "int label ids"),
    (_bad(lambda r: r["annotations"].__setitem__(2, 4)), "int label ids"),
    (_bad(lambda r: r["annotations"].__setitem__(2, -1)), "int label ids"),
    (_bad(lambda r: r["annotations"].__setitem__(2, "0")), "int label ids"),
    (_bad(lambda r: r["annotations"].pop()), "annotations but"),
    (_bad(lambda r: r["keypoints"][0].pop()), "load_calms21"),
    (_bad(lambda r: r["scores"].pop()), "load_calms21"),
    # a validation session that is only summarised must be valid too
    (_bad(vocab=None, target=VAL[1]), "vocab"),
])
def test_invalid_records_are_never_published(env, override, match):
    _, src, manifest, out = env
    src.write_text(json.dumps(source_data(**override)))
    with pytest.raises(ExtractionRefused, match=match):
        extract_sessions(src, out, [VAL[0]], manifest)
    assert not (out / INDEX_NAME).exists()
    assert not list(out.glob("task1__*"))


# ---- interruption, retries, stale caches ----

def _fail_on(monkeypatch, name_part, times=1):
    real = idata._write_atomic
    calls = {"n": 0}

    def flaky(path, text):
        if name_part in path.name and calls["n"] < times:
            calls["n"] += 1
            raise OSError("injected interruption")
        return real(path, text)

    monkeypatch.setattr(idata, "_write_atomic", flaky)


def test_interrupted_extraction_is_not_published_and_retry_recovers(env, monkeypatch):
    _, src, manifest, out = env
    _fail_on(monkeypatch, INDEX_NAME, times=2)  # the index write fails twice
    with pytest.raises(OSError):
        extract_sessions(src, out, [VAL[0]], manifest)
    with pytest.raises(CacheError):          # a session file exists, but nothing is published
        verified_sessions(out, manifest)
    (out / (INDEX_NAME + ".partial")).write_text("garbage")  # left-overs from a crash
    (out / (session_filename(VAL[1]) + ".partial")).write_text("garbage")
    with pytest.raises(OSError):             # the failed retry is still not a success
        extract_sessions(src, out, [VAL[0]], manifest)
    with pytest.raises(CacheError):
        verified_sessions(out, manifest)
    extract_sessions(src, out, [VAL[0]], manifest)  # third attempt succeeds
    assert [sid for sid, _ in verified_sessions(out, manifest)[0]] == [VAL[0]]
    assert not list(out.glob("*.partial"))


def test_interrupted_session_write_leaves_nothing_published(env, monkeypatch):
    _, src, manifest, out = env
    _fail_on(monkeypatch, session_filename(VAL[1]))
    with pytest.raises(OSError):
        extract_sessions(src, out, VAL, manifest)
    assert not (out / INDEX_NAME).exists()
    with pytest.raises(CacheError):
        verified_sessions(out, manifest)


def test_interrupted_rebuild_does_not_leave_a_stale_success(env, monkeypatch):
    """Cache from source A; a rebuild from source B dies before its index:
    the old index must not vouch for the overwritten files."""
    tmp, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    src_b = tmp / "source_b.json"
    src_b.write_text(json.dumps(source_data(offset=50.0)))
    _fail_on(monkeypatch, INDEX_NAME)
    with pytest.raises(OSError):
        extract_sessions(src_b, out, [VAL[0]], manifest)
    with pytest.raises(CacheError, match="sha256"):
        verified_sessions(out, manifest)


def test_cache_is_not_reused_for_a_different_source(env):
    tmp, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    src_b = tmp / "source_b.json"
    src_b.write_text(json.dumps(source_data(offset=50.0)))
    index = extract_sessions(src_b, out, [VAL[0]], manifest)
    assert index["source"]["sha256"] == digest(src_b) and index["source"]["path"].endswith("source_b.json")
    (sid, data), = verified_sessions(out, manifest)[0]
    assert data["annotator-id_0"][VAL[0]]["keypoints"][0][0][0][0] == 150.0  # B's content


def test_cache_is_rebuilt_when_the_same_path_changes(env):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    src.write_text(json.dumps(source_data(offset=7.0)))
    index = extract_sessions(src, out, [VAL[0]], manifest)
    assert index["source"]["sha256"] == digest(src)


def test_intact_cache_is_reused_after_rehashing_the_source(env):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    f = out / session_filename(VAL[0])
    mtime = f.stat().st_mtime_ns
    index = extract_sessions(src, out, [VAL[0]], manifest)
    assert index["reuse_check"]["source_sha256_verified"] is True
    assert index["reuse_check"]["source_hash_seconds"] >= 0
    assert f.stat().st_mtime_ns == mtime  # reused, not rewritten


def test_source_changed_with_same_path_size_and_mtime_is_not_trusted(env):
    """Codex: path + size + mtime are not proof of identity."""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    st = src.stat()
    text = src.read_text()
    changed = text.replace("100.0", "200.0", 1)
    assert len(changed) == len(text)
    src.write_text(changed)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))  # same size and mtime as before
    assert (src.stat().st_size, src.stat().st_mtime_ns) == (st.st_size, st.st_mtime_ns)
    index = extract_sessions(src, out, [VAL[0]], manifest)
    assert "reuse_check" not in index                  # rebuilt, not reused
    assert index["source"]["sha256"] == digest(src)
    (sid, data), = verified_sessions(out, manifest)[0]
    assert data["annotator-id_0"][VAL[0]]["keypoints"][0][0][0][0] == 200.0


def test_tampered_cache_is_refused_then_rebuilt(env):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    f = out / session_filename(VAL[0])
    f.write_text(f.read_text().replace("100.0", "999.0", 1))
    with pytest.raises(CacheError, match="sha256"):
        verified_sessions(out, manifest)
    extract_sessions(src, out, [VAL[0]], manifest)  # not reused: re-extracted from the source
    assert verified_sessions(out, manifest)[0][0][0] == VAL[0]


def test_changed_manifest_invalidates_the_cache(env):
    tmp, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    manifest.write_text(json.dumps({**MANIFEST, "note": "edited"}))
    with pytest.raises(CacheError, match="different split manifest"):
        verified_sessions(out, manifest)


# ---- analysis boundary ----

def test_unlisted_test_session_file_never_enters_analysis(env):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    injected = out / session_filename(TEST[0])
    injected.write_text(json.dumps({"annotator-id_0": {TEST[0]: record([0, 0])}}))
    sessions, index = verified_sessions(out, manifest)
    assert [sid for sid, _ in sessions] == [VAL[0]]
    assert unlisted_files(out, index) == [injected.name]


def _forge(out, sid, file_name, content):
    path = out / file_name
    path.write_text(json.dumps(content))
    index = json.loads((out / INDEX_NAME).read_text())
    index["sessions"][sid] = {"file": file_name, "sha256": digest(path),
                            "bytes": path.stat().st_size, "frames": 2}
    (out / INDEX_NAME).write_text(json.dumps(index))


@pytest.mark.parametrize("sid, match", [(TEST[0], "held-out test"), (TRAIN[0], "not in the 'validation'")])
def test_forged_index_entries_for_test_or_train_are_refused(env, sid, match):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _forge(out, sid, session_filename(sid), {"annotator-id_0": {sid: record([0, 0])}})
    with pytest.raises(ExtractionRefused, match=match):
        verified_sessions(out, manifest)


def test_file_content_must_match_its_listed_session(env):
    """A validation-named file that actually holds a test session is refused,
    whatever its name or hash."""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _forge(out, VAL[1], session_filename(VAL[1]), {"annotator-id_0": {TEST[0]: record([0, 0])}})
    with pytest.raises(ExtractionRefused, match="contains sessions"):
        verified_sessions(out, manifest)


def test_missing_index_or_old_format_is_refused(env):
    _, src, manifest, out = env
    out.mkdir()
    with pytest.raises(CacheError):
        verified_sessions(out, manifest)
    (out / INDEX_NAME).write_text(json.dumps({"extracted": {VAL[0]: {}}}))  # Stage 1 v1 index
    with pytest.raises(CacheError):
        verified_sessions(out, manifest)


def test_manifest_with_overlapping_splits_is_rejected(env):
    tmp, src, manifest, out = env
    manifest.write_text(json.dumps({**MANIFEST, "test": TEST + [VAL[0]]}))
    with pytest.raises(ValueError, match="both"):
        extract_sessions(src, out, [VAL[0]], manifest)


# ---- misc ----

def test_module_imports_without_resource():
    """Windows has no `resource` module; importing must still work."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['resource'] = None; "
            "import live_tracker.investigation_data as m; print(m.peak_rss_mb())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "None"


def test_memory_figure_is_optional_without_resource(monkeypatch):
    monkeypatch.setattr(idata, "resource", None)
    assert peak_rss_mb() is None


def test_unsafe_session_ids_are_refused():
    for bad in ("../etc/passwd", "a b", "x;rm"):
        with pytest.raises(ExtractionRefused):
            session_filename(bad)


def test_cli_refusal_exit_code(env, capsys):
    tmp, _, manifest, out = env
    code = main(["--source", str(tmp / "x.json"), "--manifest", str(manifest),
                 "--out", str(out), "--sessions", TEST[0]])
    assert code == 2 and "held-out test" in capsys.readouterr().err


def test_source_missing_validation_sessions_is_refused(env):
    _, src, manifest, out = env
    data = source_data()
    del data["annotator-id_0"][VAL[1]]
    src.write_text(json.dumps(data))
    with pytest.raises(ExtractionRefused, match="lacks validation sessions"):
        extract_sessions(src, out, [VAL[0]], manifest)


# ---- split_label_summary is validated, not trusted ----

@pytest.mark.parametrize("change, error", [
    (lambda s: s[VAL[0]]["label_runs"].__setitem__("attack", 9), CacheError),   # extracted: cross-checked
    (lambda s: s[VAL[0]].__setitem__("frames", 7), CacheError),
    (lambda s: s.__setitem__(TRAIN[0], {"frames": 3, "label_runs": {"attack": 1, "investigation": 0,
                                                                    "mount": 0, "other": 2}}), CacheError),
    (lambda s: s.__setitem__(TEST[0], {"frames": 2, "label_runs": {"attack": 1, "investigation": 0,
                                                                   "mount": 0, "other": 0}}), ExtractionRefused),
    (lambda s: s.pop(VAL[1]), CacheError),                                      # a session dropped
    (lambda s: s[VAL[1]].__setitem__("frames", True), CacheError),
    (lambda s: s[VAL[1]].__setitem__("frames", 0), CacheError),
    (lambda s: s[VAL[1]].__setitem__("frames", "4"), CacheError),
    (lambda s: s[VAL[1]]["label_runs"].__setitem__("attack", -1), CacheError),
    (lambda s: s[VAL[1]]["label_runs"].__setitem__("attack", 1.0), CacheError),
    (lambda s: s[VAL[1]]["label_runs"].__setitem__("chasing", 1), CacheError),
    (lambda s: s[VAL[1]]["label_runs"].__setitem__("other", 99), CacheError),  # more runs than frames
    (lambda s: s[VAL[1]].__setitem__("extra", 1), CacheError),
    (lambda s: s.__setitem__(VAL[1], [4]), CacheError),
])
def test_tampered_split_summary_is_refused(env, change, error):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _edit_index(out, lambda ix: change(ix["split_label_summary"]))
    with pytest.raises(error):
        verified_sessions(out, manifest)


def test_summary_for_unextracted_sessions_is_structurally_checked_only(env):
    """Documented limit: a well-formed but altered count for a session that was
    not extracted cannot be cross-checked without the source."""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _edit_index(out, lambda ix: ix["split_label_summary"][VAL[1]]["label_runs"].__setitem__("attack", 1))
    _, index = verified_sessions(out, manifest)
    assert index["split_label_summary"][VAL[1]]["label_runs"]["attack"] == 1


def test_index_frame_count_must_match_the_record(env):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _edit_index(out, lambda ix: ix["sessions"][VAL[0]].__setitem__("frames", 5))
    with pytest.raises(CacheError, match="frame count"):
        verified_sessions(out, manifest)


# ---- concurrent writers are serialized ----

def test_interleaved_writers_are_serialized(env, monkeypatch):
    """Writer A pauses after writing its session file (holding the lock);
    writer B (another source) starts and reaches the lock. B may only write
    after A has published, and the final index certifies B's files only."""
    tmp, src, manifest, out = env
    src_b = tmp / "source_b.json"
    src_b.write_text(json.dumps(source_data(offset=50.0)))
    log, errors = [], []
    a_paused, release_a, b_at_lock = threading.Event(), threading.Event(), threading.Event()
    real_write, real_acquire = idata._write_atomic, idata._acquire

    def write(path, payload):
        who = threading.current_thread().name
        real_write(path, payload)
        log.append((who, path.name))
        if who == "A" and path.name != INDEX_NAME and not a_paused.is_set():
            a_paused.set()
            assert release_a.wait(10)

    def acquire(fd):
        if threading.current_thread().name == "B":
            b_at_lock.set()
        real_acquire(fd)

    monkeypatch.setattr(idata, "_write_atomic", write)
    monkeypatch.setattr(idata, "_acquire", acquire)

    def run(source):
        try:
            extract_sessions(source, out, [VAL[0]], manifest)
        except Exception as exc:  # surfaced below
            errors.append(exc)

    a = threading.Thread(target=run, args=(src,), name="A")
    b = threading.Thread(target=run, args=(src_b,), name="B")
    a.start()
    assert a_paused.wait(10)          # A has written its session file and holds the lock
    b.start()
    assert b_at_lock.wait(10)         # B is at the lock while A still holds it
    assert [w for w, _ in log] == ["A"]
    release_a.set()
    a.join(10)
    b.join(10)
    assert not errors and not a.is_alive() and not b.is_alive()
    assert [w for w, _ in log] == ["A", "A", "B", "B"]  # A published completely before B wrote
    index = json.loads((out / INDEX_NAME).read_text())
    assert index["source"]["sha256"] == digest(src_b)
    (sid, data), = verified_sessions(out, manifest)[0]
    assert data["annotator-id_0"][VAL[0]]["keypoints"][0][0][0][0] == 150.0


def test_hash_comes_from_the_written_payload(env, monkeypatch):
    """A file replaced right after it is written must not be certified."""
    _, src, manifest, out = env
    real_write = idata._write_atomic

    def write_then_swap(path, payload):
        real_write(path, payload)
        if path.name != INDEX_NAME:
            path.write_text(json.dumps({"annotator-id_0": {VAL[0]: record([3, 3, 0, 0, 3, 0], 1.0)}}))

    monkeypatch.setattr(idata, "_write_atomic", write_then_swap)
    with pytest.raises(CacheError, match="sha256"):  # caught by the post-publication check
        extract_sessions(src, out, [VAL[0]], manifest)
    with pytest.raises(CacheError, match="sha256"):
        verified_sessions(out, manifest)


def test_temporary_paths_are_unique(env, monkeypatch, tmp_path):
    seen = []
    real_open = open

    def spy(path, mode="r", *a, **k):
        if str(path).endswith(".partial"):
            seen.append(str(path))
            assert mode == "xb"  # never reuses an existing temporary file
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr("builtins.open", spy)
    _, src, manifest, out = env
    extract_sessions(src, out, VAL, manifest)
    assert len(seen) == len(set(seen)) == 3  # two sessions + index


# ---- cache reads stay inside the cache directory ----

@pytest.mark.parametrize("name", [
    "../outside.json", "/tmp/outside.json", "sub/x.json", "task1__train__SYNTHETIC_val_b.json",
    "TASK1__train__SYNTHETIC_val_a.json", "task1__train__SYNTHETIC_val_a.json/", None, 7,
])
def test_non_canonical_index_file_names_are_refused_before_opening(env, monkeypatch, name):
    tmp, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    outside = tmp / "outside.json"
    outside.write_text((out / session_filename(VAL[0])).read_text())
    _edit_index(out, lambda ix: ix["sessions"][VAL[0]].__setitem__("file", name))
    opened = []
    real_read = type(outside).read_bytes
    monkeypatch.setattr(type(outside), "read_bytes",
                        lambda self: opened.append(self.name) or real_read(self))
    # a non-string name is malformed metadata (CacheError); any string name that
    # is not the canonical one is refused before opening (ExtractionRefused)
    with pytest.raises(ExtractionRefused if isinstance(name, str) else CacheError):
        verified_sessions(out, manifest)
    assert "outside.json" not in opened and not any(n.startswith("task1__") for n in opened)


@posix_only
def test_symlinked_session_file_is_refused(env):
    tmp, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    f = out / session_filename(VAL[0])
    outside = tmp / "outside.json"
    outside.write_bytes(f.read_bytes())  # same bytes, same hash, but outside the cache
    f.unlink()
    f.symlink_to(outside)
    with pytest.raises(ExtractionRefused, match="symlink"):
        verified_sessions(out, manifest)


@posix_only
def test_symlinked_index_is_refused(env):
    tmp, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    real = tmp / "index_elsewhere.json"
    (out / INDEX_NAME).rename(real)
    (out / INDEX_NAME).symlink_to(real)
    with pytest.raises(ExtractionRefused, match="symlink"):
        verified_sessions(out, manifest)


def test_reuse_check_also_rejects_escaping_entries(env):
    tmp, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _edit_index(out, lambda ix: ix["sessions"][VAL[0]].__setitem__("file", "../outside.json"))
    index = extract_sessions(src, out, [VAL[0]], manifest)  # not reusable: rebuilt cleanly
    assert "reuse_check" not in index
    assert index["sessions"][VAL[0]]["file"] == session_filename(VAL[0])


# ---- Windows lock: retry only genuine contention (simulated msvcrt) ----

class FakeMsvcrt:
    """SYNTHETIC stand-in for Windows' msvcrt: locking() raises the scripted
    errors in order, then succeeds. A failed attempt conservatively moves the
    file position, so every attempt must seek back to byte 0 itself."""
    LK_UNLCK, LK_LOCK, LK_NBLCK = 0, 1, 2

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = []

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR)))
        if self.errors:
            code = self.errors.pop(0)
            os.lseek(fd, nbytes, os.SEEK_CUR)
            raise OSError(code, os.strerror(code))


@pytest.fixture
def lock_fd(tmp_path):
    fd = os.open(tmp_path / ".cache.lock", os.O_RDWR | os.O_CREAT, 0o644)
    yield fd
    os.close(fd)


@pytest.mark.parametrize("contention", sorted(idata.WINDOWS_LOCK_CONTENTION))
def test_windows_lock_retries_on_contention(lock_fd, contention):
    fake = FakeMsvcrt(contention, contention)
    os.lseek(lock_fd, 5, os.SEEK_SET)  # every attempt must lock from byte 0
    idata._windows_acquire(lock_fd, fake)
    assert fake.calls == [(fake.LK_LOCK, 1, 0)] * 3


@pytest.mark.parametrize("permanent", [errno.EBADF, errno.EINVAL, errno.EIO, errno.ENOLCK])
def test_windows_lock_permanent_errors_propagate_at_once(lock_fd, permanent):
    fake = FakeMsvcrt(permanent, permanent)
    with pytest.raises(OSError) as err:
        idata._windows_acquire(lock_fd, fake)
    assert err.value.errno == permanent and len(fake.calls) == 1


def test_windows_lock_contention_then_permanent_error_stops(lock_fd):
    fake = FakeMsvcrt(errno.EACCES, errno.EBADF)
    with pytest.raises(OSError) as err:
        idata._windows_acquire(lock_fd, fake)
    assert err.value.errno == errno.EBADF and len(fake.calls) == 2


def test_windows_lock_on_a_closed_descriptor_fails_promptly(tmp_path):
    fd = os.open(tmp_path / ".cache.lock", os.O_RDWR | os.O_CREAT, 0o644)
    os.close(fd)
    fake = FakeMsvcrt()
    with pytest.raises(OSError) as err:
        idata._windows_acquire(fd, fake)
    assert err.value.errno == errno.EBADF and fake.calls == []


def test_windows_release_unlocks_from_byte_zero(lock_fd):
    fake = FakeMsvcrt()
    os.lseek(lock_fd, 7, os.SEEK_SET)
    idata._windows_release(lock_fd, fake)
    assert fake.calls == [(fake.LK_UNLCK, 1, 0)]


def test_contention_codes_are_only_lock_contention():
    assert errno.EBADF not in idata.WINDOWS_LOCK_CONTENTION
    assert errno.EACCES in idata.WINDOWS_LOCK_CONTENTION
    assert getattr(errno, "EDEADLK", 36) in idata.WINDOWS_LOCK_CONTENTION


# ---- Stage 1 audit regressions: one verification policy for every path ----

def test_reuse_never_returns_a_held_out_label_summary(env):
    """Finding 1: a summary entry for a held-out session must not escape through reuse."""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _edit_index(out, lambda ix: ix["split_label_summary"].__setitem__(
        TEST[0], {"frames": 2, "label_runs": [[0, 0, 2]]}))
    index = extract_sessions(src, out, [VAL[0]], manifest)
    assert "reuse_check" not in index  # not reused: rebuilt from the source
    assert sorted(index["split_label_summary"]) == sorted(VAL)
    on_disk = json.loads((out / INDEX_NAME).read_text())
    assert sorted(on_disk["split_label_summary"]) == sorted(VAL)
    sessions, verified = verified_sessions(out, manifest)
    assert verified["split_label_summary"] == index["split_label_summary"]


@pytest.mark.parametrize("split", ["test", "train", 7, None])
def test_reuse_never_returns_an_unauthorized_allowed_split(env, split):
    """Finding 1: a tampered allowed_split is never reused or returned."""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _edit_index(out, lambda ix: ix.__setitem__("allowed_split", split))
    index = extract_sessions(src, out, [VAL[0]], manifest)
    assert "reuse_check" not in index
    assert index["allowed_split"] == "validation"
    assert json.loads((out / INDEX_NAME).read_text())["allowed_split"] == "validation"


def test_reuse_of_an_intact_cache_returns_the_validated_index(env):
    _, src, manifest, out = env
    first = extract_sessions(src, out, VAL, manifest)
    again = extract_sessions(src, out, [VAL[1]], manifest)
    assert again["reuse_check"]["source_sha256_verified"] is True
    assert {k: v for k, v in again.items() if k != "reuse_check"} == first
    _, verified = verified_sessions(out, manifest)
    assert verified == first


def test_tampered_summary_values_of_extracted_sessions_are_not_reused(env):
    """(Entries of unextracted sessions remain structure-checked only; see
    test_summary_for_unextracted_sessions_is_structurally_checked_only.)"""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _edit_index(out, lambda ix: ix["split_label_summary"][VAL[0]].__setitem__("frames", 999))
    index = extract_sessions(src, out, [VAL[0]], manifest)
    assert "reuse_check" not in index and index["split_label_summary"][VAL[0]]["frames"] == 6


def _drop(key):
    return lambda ix: ix.pop(key)


MALFORMED = {
    "entry_string": lambda ix: ix["sessions"].__setitem__(VAL[0], "a string"),
    "entry_int": lambda ix: ix["sessions"].__setitem__(VAL[0], 7),
    "entry_null": lambda ix: ix["sessions"].__setitem__(VAL[0], None),
    "entry_list": lambda ix: ix["sessions"].__setitem__(VAL[0], []),
    "entry_only_sha": lambda ix: ix["sessions"].__setitem__(VAL[0], {"sha256": "x"}),
    "entry_missing_frames": lambda ix: ix["sessions"][VAL[0]].pop("frames"),
    "entry_extra_key": lambda ix: ix["sessions"][VAL[0]].__setitem__("extra", 1),
    "entry_frames_bool": lambda ix: ix["sessions"][VAL[0]].__setitem__("frames", True),
    "entry_frames_zero": lambda ix: ix["sessions"][VAL[0]].__setitem__("frames", 0),
    "entry_frames_str": lambda ix: ix["sessions"][VAL[0]].__setitem__("frames", "6"),
    "entry_bytes_negative": lambda ix: ix["sessions"][VAL[0]].__setitem__("bytes", -1),
    "entry_sha_not_hex": lambda ix: ix["sessions"][VAL[0]].__setitem__("sha256", "Z" * 64),
    "sessions_string": lambda ix: ix.__setitem__("sessions", "a string"),
    "sessions_int": lambda ix: ix.__setitem__("sessions", 5),
    "sessions_list": lambda ix: ix.__setitem__("sessions", [VAL[0]]),
    "sessions_missing": _drop("sessions"),
    "manifest_string": lambda ix: ix.__setitem__("manifest", "not-a-dict"),
    "manifest_null": lambda ix: ix.__setitem__("manifest", None),
    "manifest_missing": _drop("manifest"),
    "source_string": lambda ix: ix.__setitem__("source", "x"),
    "source_size_str": lambda ix: ix["source"].__setitem__("size", "1"),
    "source_missing": _drop("source"),
    "allowed_split_int": lambda ix: ix.__setitem__("allowed_split", 7),
    "summary_string": lambda ix: ix.__setitem__("split_label_summary", "x"),
    "summary_missing": _drop("split_label_summary"),
}


@pytest.mark.parametrize("change", MALFORMED.values(), ids=MALFORMED.keys())
def test_malformed_index_is_a_cache_error_and_is_rebuilt(env, change):
    """Finding 2: malformed metadata never raises AttributeError/TypeError/KeyError;
    analysis refuses it and extraction rebuilds a verified cache from the source."""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    _edit_index(out, change)
    with pytest.raises(CacheError, match="malformed"):
        verified_sessions(out, manifest)
    index = extract_sessions(src, out, [VAL[0]], manifest)
    assert "reuse_check" not in index
    sessions, verified = verified_sessions(out, manifest)
    assert [sid for sid, _ in sessions] == [VAL[0]] and verified == index


@pytest.mark.parametrize("content", ["[1, 2]", '"text"', "null"])
def test_non_object_index_is_a_cache_error_and_is_rebuilt(env, content):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    (out / INDEX_NAME).write_text(content)
    with pytest.raises(CacheError):
        verified_sessions(out, manifest)
    assert extract_sessions(src, out, [VAL[0]], manifest)["sessions"][VAL[0]]["frames"] == 6


@pytest.mark.parametrize("requested", [[VAL[0]], [VAL[1]]])
def test_tampered_frames_never_survive_incremental_extraction(env, requested):
    """Finding 3: a tampered frame count on a retained (or requested) session is
    never carried into the new index."""
    _, src, manifest, out = env
    extract_sessions(src, out, VAL, manifest)
    _edit_index(out, lambda ix: ix["sessions"][VAL[1]].__setitem__("frames", 999))
    index = extract_sessions(src, out, requested, manifest)
    assert all(e["frames"] != 999 for e in index["sessions"].values())
    on_disk = json.loads((out / INDEX_NAME).read_text())
    assert all(e["frames"] != 999 for e in on_disk["sessions"].values())
    if VAL[1] in on_disk["sessions"]:
        assert on_disk["sessions"][VAL[1]]["frames"] == 4
    verified_sessions(out, manifest)  # the published cache verifies


def test_retained_sessions_are_rederived_from_the_source(env, monkeypatch):
    """Sessions kept from an earlier extraction are re-validated and rewritten
    from the source records; their old metadata is not copied."""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    validated, written = [], []
    real_validate, real_write = idata.validate_record, idata._write_atomic
    monkeypatch.setattr(idata, "validate_record",
                        lambda g, sid, rec: validated.append(sid) or real_validate(g, sid, rec))
    monkeypatch.setattr(idata, "_write_atomic",
                        lambda path, payload: written.append(path.name) or real_write(path, payload))
    index = extract_sessions(src, out, [VAL[1]], manifest)
    assert sorted(index["sessions"]) == sorted(VAL)
    assert sorted(validated) == sorted(VAL)
    assert session_filename(VAL[0]) in written and session_filename(VAL[1]) in written
    assert index["sessions"][VAL[0]]["frames"] == 6 and index["sessions"][VAL[1]]["frames"] == 4


def test_sessions_from_a_changed_source_are_not_retained(env):
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    src.write_text(json.dumps(source_data(offset=1.0)))
    index = extract_sessions(src, out, [VAL[1]], manifest)
    assert sorted(index["sessions"]) == [VAL[1]]


@pytest.mark.parametrize("entry", ["x", None, 7, {}, {"file": None}, {"file": 3}])
def test_unlisted_files_rejects_malformed_entries(env, entry):
    """Finding 4: malformed metadata is a CacheError, not TypeError/KeyError."""
    _, src, manifest, out = env
    index = extract_sessions(src, out, [VAL[0]], manifest)
    index["sessions"][VAL[0]] = entry
    with pytest.raises(CacheError, match="malformed"):
        unlisted_files(out, index)


@pytest.mark.parametrize("index", [None, [], "x", {"format": "other"}])
def test_unlisted_files_rejects_a_malformed_index(env, index):
    _, _, _, out = env
    out.mkdir()
    with pytest.raises(CacheError):
        unlisted_files(out, index)


def test_unlisted_files_still_lists_unpublished_json(env):
    _, src, manifest, out = env
    index = extract_sessions(src, out, [VAL[0]], manifest)
    (out / "stray.json").write_text("{}")
    assert unlisted_files(out, index) == ["stray.json"]


@pytest.mark.parametrize("pair", [("task1/x", "task1__x"), ("a_/b", "a/_b")])
def test_colliding_cache_file_names_are_refused(tmp_path, pair):
    """Finding 5: IDs that map to one cache file name are refused before any read."""
    assert session_filename(pair[0]) == session_filename(pair[1])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": TRAIN, "validation": list(pair), "test": TEST}))
    src = tmp_path / "source.json"
    src.write_text(json.dumps({"annotator-id_0": {sid: record([3, 0, 3]) for sid in pair}}))
    out = tmp_path / "cache"
    for requested in ([pair[0]], list(pair)):
        with pytest.raises(ExtractionRefused, match="same cache file"):
            extract_sessions(src, out, requested, manifest)
    assert not out.exists() or not list(out.glob("*.json"))


def test_real_style_ids_keep_the_existing_file_names():
    assert session_filename("task1/train/mouse005_task1_annotator1") == \
        "task1__train__mouse005_task1_annotator1.json"


def test_analysis_refuses_a_forged_index_under_a_colliding_manifest(env):
    """Extraction never builds such a cache; a forged one is refused by name."""
    _, src, manifest, out = env
    extract_sessions(src, out, [VAL[0]], manifest)
    twin = VAL[0].replace("/", "__", 1)  # same cache file name as VAL[0]
    manifest.write_text(json.dumps({**MANIFEST, "validation": VAL + [twin]}))
    _edit_index(out, lambda ix: (ix["manifest"].__setitem__("sha256", digest(manifest)),
                                 ix["split_label_summary"].__setitem__(twin, ix["split_label_summary"][VAL[0]])))
    with pytest.raises(ExtractionRefused, match="same cache file"):
        verified_sessions(out, manifest)


def test_two_sessions_sharing_one_file_are_refused(env):
    """Implied by canonical file names (no separate check is needed)."""
    _, src, manifest, out = env
    extract_sessions(src, out, VAL, manifest)
    _edit_index(out, lambda ix: ix["sessions"].__setitem__(VAL[1], dict(ix["sessions"][VAL[0]])))
    with pytest.raises((CacheError, ExtractionRefused)):
        verified_sessions(out, manifest)
    assert "reuse_check" not in extract_sessions(src, out, VAL, manifest)


# ======================================================================
# Full authorized coverage: data-driven discovery, configurable source root
# ======================================================================

from pathlib import Path  # noqa: E402

from live_tracker.investigation_data import (ROOT_ENV, TRAIN_FILE, resolve_source, session_filename,  # noqa: E402
                                             sha256_file, split_coverage)

ROOT = Path(__file__).resolve().parents[1]

REAL_SOURCE_ROOT = Path("/Volumes/T7 Hackathon/Datasets/CalMS21")      # optional; tests skip without it


def wide_env(tmp_path, n_val=6, n_train=4):
    """A manifest with more validation sessions than the old three-session cache."""
    val = [f"task1/train/SYNTHETIC_wide_val_{i}" for i in range(n_val)]
    train = [f"task1/train/SYNTHETIC_wide_train_{i}" for i in range(n_train)]
    test = ["task1/test/SYNTHETIC_wide_test_0"]
    manifest = tmp_path / "wide_manifest.json"
    manifest.write_text(json.dumps({"train": train, "validation": val, "test": test}))
    recs = {sid: record([3, 0, 0, 3, 1][: 3 + i % 3], 0.1 * i) for i, sid in enumerate(val + train)}
    src = tmp_path / TRAIN_FILE
    src.write_text(json.dumps({"annotator-id_0": recs}))
    return val, train, test, manifest, src


def test_all_extracts_every_manifest_session_of_the_split(tmp_path):
    val, train, _, manifest, src = wide_env(tmp_path)
    out = tmp_path / "full"
    assert main(["--all", "--source", str(src), "--manifest", str(manifest), "--out", str(out)]) == 0
    cov = split_coverage(out, manifest)
    assert cov["complete"] and cov["cached"] == sorted(val) and len(cov["cached"]) == 6 > 3
    index = json.loads((out / INDEX_NAME).read_text())
    assert sorted(index["split_label_summary"]) == sorted(val) and not set(index["sessions"]) & set(train)


def test_train_split_is_a_separate_cache_and_never_mixes_with_validation(tmp_path):
    val, train, _, manifest, src = wide_env(tmp_path)
    out = tmp_path / "train_cache"
    assert main(["--all", "--split", "train", "--source", str(src), "--manifest", str(manifest),
                 "--out", str(out)]) == 0
    index = json.loads((out / INDEX_NAME).read_text())
    assert index["allowed_split"] == "train" and sorted(index["sessions"]) == sorted(train)
    with pytest.raises(ExtractionRefused):                       # a validation reader refuses the train cache
        verified_sessions(out, manifest, "validation")
    with pytest.raises(ExtractionRefused):                       # and train sessions cannot enter a validation cache
        extract_sessions(src, tmp_path / "v", train[:1], manifest, allowed_split="validation")


def test_incremental_extension_keeps_and_rederives_earlier_sessions(tmp_path):
    val, _, _, manifest, src = wide_env(tmp_path)
    out = tmp_path / "inc"
    extract_sessions(src, out, val[:3], manifest)                # the old three-session cache
    first = split_coverage(out, manifest)
    assert not first["complete"] and first["missing"] == sorted(val[3:])
    before = {sid: e["sha256"] for sid, e in json.loads((out / INDEX_NAME).read_text())["sessions"].items()}
    extract_sessions(src, out, val, manifest)                    # extended to the whole split
    after = json.loads((out / INDEX_NAME).read_text())["sessions"]
    assert split_coverage(out, manifest)["complete"]
    assert all(after[sid]["sha256"] == h for sid, h in before.items())   # same source: identical payloads


def test_split_coverage_refuses_corrupt_or_stale_caches(tmp_path):
    val, _, _, manifest, src = wide_env(tmp_path)
    out = tmp_path / "c"
    extract_sessions(src, out, val, manifest)
    path = out / session_filename(val[0])
    path.write_bytes(path.read_bytes() + b" ")                   # corrupt one payload
    with pytest.raises(CacheError):
        split_coverage(out, manifest)
    extract_sessions(src, out, val, manifest)                    # rebuilt from the source
    m = json.loads(manifest.read_text())
    m["validation"] = m["validation"][:-1]
    m["train"].append(val[-1])
    manifest.write_text(json.dumps(m))                           # the manifest changed: the index is stale
    with pytest.raises(CacheError):
        split_coverage(out, manifest)


def test_partial_index_is_never_published(tmp_path, monkeypatch):
    val, _, _, manifest, src = wide_env(tmp_path)
    out = tmp_path / "p"
    extract_sessions(src, out, val[:2], manifest)
    import live_tracker.investigation_data as idata
    real = idata._write_atomic
    calls = []

    def dies_midway(path, payload):
        calls.append(path.name)
        if len(calls) == 4:
            raise OSError("disk full")
        real(path, payload)
    monkeypatch.setattr(idata, "_write_atomic", dies_midway)
    with pytest.raises(OSError):
        extract_sessions(src, out, val, manifest)
    monkeypatch.setattr(idata, "_write_atomic", real)
    cov = split_coverage(out, manifest)                          # the old index still verifies; nothing half-new
    assert cov["cached"] == sorted(val[:2]) and not cov["complete"]


def test_dataset_root_resolution(tmp_path, monkeypatch):
    nested = tmp_path / "CalMS21" / "extracted" / "task1_classic_classification"
    nested.mkdir(parents=True)
    (nested / TRAIN_FILE).write_text("{}")
    assert resolve_source(None, tmp_path / "CalMS21", env={}) == nested / TRAIN_FILE
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / TRAIN_FILE).write_text("{}")
    assert resolve_source(None, None, env={ROOT_ENV: str(flat)}) == flat / TRAIN_FILE
    assert resolve_source(str(flat / TRAIN_FILE)) == flat / TRAIN_FILE
    with pytest.raises(ExtractionRefused, match="is the drive mounted"):
        resolve_source(None, tmp_path / "missing", env={})
    with pytest.raises(ExtractionRefused, match="no source"):
        resolve_source(None, None, env={})
    with pytest.raises(ExtractionRefused, match="source not found"):
        resolve_source(str(tmp_path / "nope.json"))


def test_cli_reports_an_unavailable_source(tmp_path, capsys):
    _, _, _, manifest, _ = wide_env(tmp_path)
    code = main(["--all", "--dataset-root", str(tmp_path / "unmounted"), "--manifest", str(manifest),
                 "--out", str(tmp_path / "x")])
    assert code == 2 and "not found under the dataset root" in capsys.readouterr().err
    assert not (tmp_path / "x" / INDEX_NAME).exists()            # nothing substituted or published


# the local cache is checked first, so the external drive is never even looked up without it
@pytest.mark.skipif(not (ROOT / "data" / "raw" / "calms21_validation" / INDEX_NAME).exists()
                    or not (REAL_SOURCE_ROOT / "extracted" / "task1_classic_classification" / TRAIN_FILE).exists(),
                    reason="CalMS21 source drive or verified cache not present")
def test_real_cache_covers_the_full_validation_split_and_binds_the_source():
    manifest = ROOT / "splits" / "split_manifest.json"
    cache = ROOT / "data" / "raw" / "calms21_validation"
    cov = split_coverage(cache, manifest)
    m = json.loads(manifest.read_text())
    assert cov["complete"] and cov["cached"] == sorted(m["validation"]) and len(cov["cached"]) == 14
    index = json.loads((cache / INDEX_NAME).read_text())
    source = resolve_source(None, REAL_SOURCE_ROOT, env={})
    assert index["source"]["sha256"] == sha256_file(source)       # reproducible source-to-index provenance
    assert index["source"]["size"] == source.stat().st_size and not set(index["sessions"]) & set(m["test"])
