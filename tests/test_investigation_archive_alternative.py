"""The Live Agent page also accepts Igor's original score zip (the "repacked" zip is not available), by its own checksum."""

import hashlib
import zipfile
from pathlib import Path

from live_tracker import investigation_v1_runs as runs


def make_zip(path, payload):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("scores_validation_model_b.csv", payload)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(tmp_path, repacked_sha, original_sha):
    return runs.ValidationInputs(
        archive=str(tmp_path / "repacked.zip"), archive_sha256=repacked_sha,
        archive_alternatives=(str(tmp_path / "original.zip"),), archive_alternative_sha256=(original_sha,))


def archive_problems(vi):
    return [p for p in vi.problems() if p.startswith("score archive")]


def test_the_original_zip_is_used_when_the_repacked_one_is_missing(tmp_path):
    original_sha = make_zip(tmp_path / "original.zip", "a,b\n1,2\n")
    vi = inputs(tmp_path, "0" * 64, original_sha)
    assert archive_problems(vi) == []
    assert vi.resolved_archive() == str(tmp_path / "original.zip")
    assert vi.archive_digest() == original_sha                       # the CLI is given the checksum it will check


def test_a_changed_original_zip_is_still_refused_by_the_cli(tmp_path):
    make_zip(tmp_path / "original.zip", "a,b\n1,2\n")
    vi = inputs(tmp_path, "0" * 64, "f" * 64)                         # the accepted checksum does not match this file
    assert vi.archive_digest() == "0" * 64                           # the expected one is passed, so the CLI refuses the file


def test_the_repacked_zip_wins_when_it_exists(tmp_path):
    repacked_sha = make_zip(tmp_path / "repacked.zip", "a,b\n1,2\n")
    original_sha = make_zip(tmp_path / "original.zip", "a,b\n9,9\n")
    vi = inputs(tmp_path, repacked_sha, original_sha)
    assert vi.resolved_archive() == str(tmp_path / "repacked.zip") and vi.archive_digest() == repacked_sha


def test_nothing_is_found_when_neither_zip_exists(tmp_path):
    vi = inputs(tmp_path, "0" * 64, "1" * 64)
    assert archive_problems(vi) == ["score archive not found: repacked.zip"]
    assert vi.archive_digest() == "0" * 64


def test_the_alternatives_are_not_used_by_the_synthetic_test_inputs(tmp_path):
    original_sha = make_zip(tmp_path / "original.zip", "a,b\n1,2\n")
    vi = runs.ValidationInputs(archive=str(tmp_path / "repacked.zip"), archive_sha256="0" * 64, allow_synthetic=True,
                               archive_alternatives=(str(tmp_path / "original.zip"),), archive_alternative_sha256=(original_sha,))
    assert archive_problems(vi) == ["score archive not found: repacked.zip"]


def test_the_default_accepts_the_known_original_checksum():
    vi = runs.ValidationInputs()
    assert vi.archive_alternative_sha256 == (runs.ORIGINAL_ARCHIVE_SHA256,) and len(runs.ORIGINAL_ARCHIVE_SHA256) == 64
    assert vi.archive_sha256.startswith("5c1a81e4") and vi.archive_sha256.endswith("c23427")      # the expected one is unchanged
