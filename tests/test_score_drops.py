"""Tests for Stage 9 S9-1a: verified score-drop ingestion (score_drops.py) and
the offline monitor smoke test (scripts/stage9_smoke.py). New file.

Synthetic archives are built in tmp_path from fake_scores (SYNTHETIC_
sessions). The real-archive test is opt-in: it runs only when Igor's archive
is present (LIVE_TRACKER_SCORE_DROP, default ~/Downloads/igor_score_drops.zip)
and reads only its validation member.
"""

import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import live_tracker.score_drops as sd
from live_tracker.fake_scores import Segment, TrackingLoss, build_scores
from live_tracker.monitor import run_monitor
from live_tracker.rules import RuleSet

ROOT = Path(__file__).resolve().parents[1]
VAL = ["SYNTHETIC_val_1", "SYNTHETIC_val_2"]
MANIFEST = {"train": ["SYNTHETIC_train_1"], "validation": VAL, "test": ["SYNTHETIC_test_1"]}
VAL_MEMBER, TEST_MEMBER = "scores_validation_model_b.csv", "scores_test_model_b.csv"


# filesystem symlinks need extra privileges on Windows, and POSIX permission bits do not exist there
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: filesystem symlinks / permission bits")

spec = importlib.util.spec_from_file_location("stage9_smoke", ROOT / "scripts" / "stage9_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


# ---- builders ----

def table():
    a = build_scores(VAL[0], 8.0, segments=[Segment("attack", 1.0, 4.0, 0.85, 0.95)],
                     tracking_loss=[TrackingLoss(5.0, 5.5)])
    b = build_scores(VAL[1], 6.0, segments=[Segment("attack", 2.0, 3.0, 0.6, 0.7)], seed=1)
    return pd.concat([a, b], ignore_index=True)


def csv_bytes(df=None):
    return (table() if df is None else df).to_csv(index=False).encode()


def sums_for(members):
    return "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in members.items()).encode()


def archive(tmp_path, members=None, sums=None, extra=(), name="drops.zip"):
    """A ZIP with the given CSV members, a checksum list (computed unless given)
    and extra raw (ZipInfo or name, bytes) entries."""
    members = {VAL_MEMBER: csv_bytes(), TEST_MEMBER: csv_bytes()} if members is None else members
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in members.items():
            z.writestr(n, b)
        if sums is not False:
            z.writestr(sd.CHECKSUMS, sums_for(members) if sums is None else sums)
        for info, b in extra:
            z.writestr(info, b)
    return path


@pytest.fixture
def manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(MANIFEST))
    return p


def load(path, manifest, member=VAL_MEMBER, **kw):
    return sd.load_validation_scores(path, member, manifest_path=manifest, **kw)


def edit_csv(transform):
    lines = csv_bytes().decode().splitlines()
    return ("\n".join(transform(lines)) + "\n").encode()


# ======================================================================
# valid input and provenance
# ======================================================================

def test_valid_archive_loads_with_honest_provenance(tmp_path, manifest):
    path = archive(tmp_path)
    drop = load(path, manifest)
    t = drop.scores()
    assert list(t.columns) == list(sd.SCORE_COLUMNS) and len(t) == len(table())
    assert t["tracking_ok"].dtype == bool and t["frame"].dtype == "int64"
    pd.testing.assert_frame_equal(t, table(), check_dtype=False)
    p = drop.provenance
    assert p.archive_name == "drops.zip" and p.archive_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert p.member == VAL_MEMBER and p.member_sha256 == hashlib.sha256(csv_bytes()).hexdigest()
    assert p.split == "validation" and (p.n_sessions, p.n_frames) == (2, len(table()))
    assert p.archive_sha256_check == "no expected digest supplied" and "unsigned" in p.member_sha256_check
    assert p.model_label == "model_b" and p.model_label_status.startswith("unverified")
    assert (p.model_artifact_sha256, p.model_id, p.feature_configuration, p.producing_commit) == (None,) * 4
    assert p.model_provenance_status.startswith("unavailable")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.model_id = "model_b"
    changed = drop.scores()                                          # copies: the drop is unchanged
    changed.loc[0, "attack"] = 0.0
    assert drop.scores().loc[0, "attack"] == t.loc[0, "attack"]
    assert json.loads(json.dumps(p.to_json()))["model_artifact_sha256"] is None


def test_expected_archive_digest(tmp_path, manifest):
    path = archive(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load(path, manifest, expected_archive_sha256=digest).provenance.archive_sha256_check == \
        "matched expected digest"
    with pytest.raises(sd.ArchiveIntegrityError, match="differs from the expected digest"):
        load(path, manifest, expected_archive_sha256="0" * 64)
    with pytest.raises(sd.ArchiveIntegrityError, match="not a lower-case SHA-256"):
        load(path, manifest, expected_archive_sha256=digest.upper())


# ======================================================================
# archive integrity
# ======================================================================

def symlink(name):
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return info


@pytest.mark.parametrize("extra, text", [
    ([("../escape.csv", b"x")], "unsafe member path"),
    ([("/abs.csv", b"x")], "unsafe member path"),
    ([("sub/scores_validation_x.csv", b"x")], "unsafe member path"),
    ([("C:evil.csv", b"x")], "unsafe member path"),
    ([("back\\slash.csv", b"x")], "unsafe member path"),
    ([("notes.txt", b"hello")], "unexpected member"),
    ([(symlink("scores_validation_link.csv"), b"/etc/passwd")], "symlink member"),
    ([(VAL_MEMBER, csv_bytes())], "duplicate member names"),
    ([("scores_validation_bomb.csv", b"0" * 3_000_000)], "implausible compression ratio"),
])
def test_unsafe_archive_structure_is_refused(tmp_path, manifest, extra, text):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # zipfile warns on a duplicate name
        path = archive(tmp_path, extra=extra)
    with pytest.raises(sd.ArchiveIntegrityError, match=text):
        load(path, manifest)


def test_member_checksum_mismatch(tmp_path, manifest):
    members = {VAL_MEMBER: csv_bytes(), TEST_MEMBER: csv_bytes()}
    wrong = sums_for({VAL_MEMBER: b"other bytes", TEST_MEMBER: csv_bytes()})
    with pytest.raises(sd.ArchiveIntegrityError, match="does not match its SHA256SUMS.txt digest"):
        load(archive(tmp_path, members, sums=wrong), manifest)


@pytest.mark.parametrize("sums, text", [
    (False, "SHA256SUMS.txt is missing"),
    (b"not a checksum line\n", "is not '<sha256>  <member>'"),
    (b"", "lists no member"),
    (b"\xff\xfe", "is not UTF-8"),
    ("deadbeef  " + VAL_MEMBER + "\n", "is not '<sha256>  <member>'"),
])
def test_malformed_or_missing_checksum_manifest(tmp_path, manifest, sums, text):
    with pytest.raises(sd.ArchiveIntegrityError, match=text):
        load(archive(tmp_path, sums=sums if not isinstance(sums, str) else sums.encode()), manifest)


def test_checksum_manifest_must_match_the_members(tmp_path, manifest):
    members = {VAL_MEMBER: csv_bytes(), TEST_MEMBER: csv_bytes()}
    unlisted = sums_for({VAL_MEMBER: members[VAL_MEMBER]})
    with pytest.raises(sd.ArchiveIntegrityError, match="present but unlisted"):
        load(archive(tmp_path, members, sums=unlisted), manifest)
    phantom = sums_for({**members, "scores_validation_ghost.csv": b"x"})
    with pytest.raises(sd.ArchiveIntegrityError, match="listed but absent"):
        load(archive(tmp_path, members, sums=phantom, name="b.zip"), manifest)
    twice = sums_for(members) + sums_for({VAL_MEMBER: members[VAL_MEMBER]})
    with pytest.raises(sd.ArchiveIntegrityError, match="twice"):
        load(archive(tmp_path, members, sums=twice, name="c.zip"), manifest)


def test_not_a_zip_or_missing_member(tmp_path, manifest):
    bad = tmp_path / "x.zip"
    bad.write_bytes(b"PK not really")
    with pytest.raises(sd.ArchiveIntegrityError, match="not a ZIP"):
        load(bad, manifest)
    with pytest.raises(sd.ArchiveIntegrityError, match="not a regular file"):
        load(tmp_path / "absent.zip", manifest)
    with pytest.raises(sd.ArchiveIntegrityError, match="no score member"):
        load(archive(tmp_path), manifest, member="scores_validation_model_z.csv")


def test_test_split_member_is_refused_without_being_opened(tmp_path, manifest, monkeypatch):
    path = archive(tmp_path)
    opened = []
    real_open = zipfile.ZipFile.open

    def spy(self, name, *a, **k):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return real_open(self, name, *a, **k)
    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    with pytest.raises(sd.SplitRefused, match="only validation members"):
        load(path, manifest, member=TEST_MEMBER)
    assert TEST_MEMBER not in opened and opened == [sd.CHECKSUMS]


# ======================================================================
# score schema
# ======================================================================

def drop_column(lines, col):
    i = lines[0].split(",").index(col)
    return [",".join(x for j, x in enumerate(line.split(",")) if j != i) for line in lines]


def set_field(lines, row, col, value):
    i = lines[0].split(",").index(col)
    parts = lines[row].split(",")
    parts[i] = value
    return lines[:row] + [",".join(parts)] + lines[row + 1:]


@pytest.mark.parametrize("transform, text", [
    (lambda ls: [ls[0] + ",extra"] + [x + ",1" for x in ls[1:]], "header"),
    (lambda ls: drop_column(ls, "mount"), "header"),
    (lambda ls: [",".join(reversed(ls[0].split(",")))] + ls[1:], "header"),
    (lambda ls: set_field(ls, 5, "attack", "nan"), "is not finite"),
    (lambda ls: set_field(ls, 5, "mount", "inf"), "is not finite"),
    (lambda ls: set_field(ls, 5, "investigation", "1.5"), "outside \\[0, 1\\]"),
    (lambda ls: set_field(ls, 5, "attack", "-0.01"), "outside \\[0, 1\\]"),
    (lambda ls: set_field(ls, 5, "attack", "high"), "is not a number"),
    (lambda ls: set_field(ls, 5, "attack", ""), "is not a number"),
    (lambda ls: ls[:6] + [ls[5]] + ls[6:], "frames are 0..T-1"),                   # duplicate frame
    (lambda ls: ls[:6] + ls[7:], "frames are 0..T-1"),                             # missing frame
    (lambda ls: ls[:5] + [ls[6], ls[5]] + ls[7:], "frames are 0..T-1"),            # unordered
    (lambda ls: set_field(ls, 5, "frame", "-1"), "not a non-negative integer"),
    (lambda ls: set_field(ls, 5, "frame", "4.0"), "not a non-negative integer"),
    (lambda ls: set_field(ls, 5, "time_s", "0.1400001"), "is not frame / 30"),
    (lambda ls: set_field(ls, 5, "tracking_ok", "1"), "is not True / False"),
    (lambda ls: set_field(ls, 5, "tracking_ok", "yes"), "is not True / False"),
    (lambda ls: set_field(ls, 5, "tracking_ok", ""), "is not True / False"),
    (lambda ls: set_field(ls, 5, "tracking_ok", " True"), "is not True / False"),
    (lambda ls: set_field(ls, 5, "session", "SYNTHETIC_test_1"), "not in the manifest's validation split"),
    (lambda ls: set_field(ls, 5, "session", "SYNTHETIC_train_1"), "not in the manifest's validation split"),
    (lambda ls: [x for x in ls if not x.startswith(VAL[1])], "not exactly the manifest's validation split"),
    (lambda ls: ls[:3] + [x for x in ls if x.startswith(VAL[1])] + [x for x in ls[3:] if not x.startswith(VAL[1])],
     "not one contiguous block"),
    (lambda ls: ls[:5] + [ls[5] + ",x"] + ls[6:], "fields, expected 7"),
    (lambda ls: [], "header"),
])
def test_malformed_score_tables_are_refused(tmp_path, manifest, transform, text):
    bad = edit_csv(transform)
    members = {VAL_MEMBER: bad, TEST_MEMBER: csv_bytes()}
    with pytest.raises(sd.ScoreSchemaError, match=text):
        load(archive(tmp_path, members), manifest)


def test_schema_and_integrity_errors_are_distinct(tmp_path, manifest):
    assert issubclass(sd.ScoreSchemaError, sd.ScoreDropError) and issubclass(sd.ArchiveIntegrityError,
                                                                               sd.ScoreDropError)
    assert not issubclass(sd.ScoreSchemaError, sd.ArchiveIntegrityError)
    bad_bytes = edit_csv(lambda ls: set_field(ls, 5, "attack", "nan"))
    # a schema-invalid member with a WRONG checksum: integrity is reported first
    members = {VAL_MEMBER: bad_bytes, TEST_MEMBER: csv_bytes()}
    wrong = sums_for({VAL_MEMBER: csv_bytes(), TEST_MEMBER: csv_bytes()})
    with pytest.raises(sd.ArchiveIntegrityError):
        load(archive(tmp_path, members, sums=wrong), manifest)


# ======================================================================
# monitor compatibility and the smoke-test entry point
# ======================================================================

def approved(**overrides):
    rules = RuleSet()
    rules.add({"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0, "alert_threshold": 0.8,
               "ask_threshold": 0.5, **overrides})
    rules.approve("R1")
    return rules.active()


def test_verified_rows_give_the_monitor_exactly_the_events_of_the_original_table(tmp_path, manifest):
    drop = load(archive(tmp_path), manifest)
    direct = run_monitor(table(), approved(), run_id="direct")
    loaded = run_monitor(drop.scores(), approved(), run_id="direct")
    assert loaded == direct and len(direct) == 1 and direct[0]["session"] == VAL[0]
    assert direct[0]["action"] == "alert"                     # the 3 s high bout; the 1 s bout never fires
    assert run_monitor(drop.rows(), approved(), run_id="direct") == direct


def smoke_args(path, manifest, *extra):
    return [str(path), "--member", VAL_MEMBER, *extra]


@pytest.fixture
def synthetic_manifest_default(manifest, monkeypatch):
    monkeypatch.setattr(sd, "DEFAULT_MANIFEST", manifest)
    real = sd.load_validation_scores
    monkeypatch.setattr(sd, "load_validation_scores",
                        lambda *a, **k: real(*a, **{**k, "manifest_path": manifest}))
    return manifest


def test_smoke_script_in_memory(tmp_path, synthetic_manifest_default, capsys):
    path = archive(tmp_path)
    assert smoke.main(smoke_args(path, None, "--smoke-rule")) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["sessions"], out["frames"], out["events"]) == (2, len(table()), 1)
    assert out["tracking_ok_false_frames"] == int((~table()["tracking_ok"]).sum())
    assert out["rule"]["rule_id"] == smoke.SMOKE_RULE_ID and "LOCAL SMOKE TEST ONLY" in out["rule"]["description"]
    assert "not a researcher decision" in out["rule_approval"] and out["written"] is None
    assert out["source"]["member_sha256"] == hashlib.sha256(csv_bytes()).hexdigest()
    assert out["source"]["model_id"] is None
    snap = out["journal_snapshot"]                       # S9-2: an issued ScoreDrop over its own rows is verified
    assert (snap["provenance_status"], snap["completion"], snap["n_events"]) == ("verified_score_drop", "complete", 1)
    assert not list(tmp_path.glob("**/*.jsonl"))                               # nothing written


def test_smoke_script_rule_must_be_confirmed(tmp_path, synthetic_manifest_default, capsys):
    path = archive(tmp_path)
    rule = tmp_path / "rule.json"
    rule.write_text(json.dumps({"rule_id": "R9", "behavior": "attack", "min_duration_s": 2.0}))
    assert smoke.main(smoke_args(path, None, "--rule-json", str(rule))) == 2
    assert "confirm it explicitly with --confirm R9" in capsys.readouterr().err
    assert smoke.main(smoke_args(path, None, "--rule-json", str(rule), "--confirm", "R9")) == 0
    assert "--confirm" in json.loads(capsys.readouterr().out)["rule_approval"]
    rule.write_text(json.dumps({"rule_id": "R9", "behavior": "attack", "min_duration_s": 2.0, "status": "approved"}))
    assert smoke.main(smoke_args(path, None, "--rule-json", str(rule), "--confirm", "R9")) == 2   # never pre-approved
    assert smoke.main(smoke_args(path, None)) == 2                                # no rule at all
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--rule-json", str(rule))) == 2


def test_smoke_script_writes_only_to_an_ignored_directory(tmp_path, synthetic_manifest_default, capsys):
    path = archive(tmp_path)
    tracked = ROOT / "docs" / "stage9_smoke_should_not_exist"
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(tracked))) == 2
    assert "is not ignored by Git" in capsys.readouterr().err and not tracked.exists()
    out_dir = tmp_path / "out"                                                  # outside the repository
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(out_dir))) == 0
    capsys.readouterr()
    assert sorted(p.name for p in out_dir.iterdir()) == ["events.journal.jsonl", "events.jsonl", "summary.json"]
    assert smoke.ignored_output_dir(ROOT / "outputs" / "stage9_smoke") == (ROOT / "outputs" / "stage9_smoke")


def test_smoke_script_refuses_a_test_member(tmp_path, synthetic_manifest_default, capsys):
    path = archive(tmp_path)
    assert smoke.main([str(path), "--member", TEST_MEMBER, "--smoke-rule"]) == 2
    assert "SplitRefused" in capsys.readouterr().err


# ======================================================================
# opt-in: Igor's real archive (validation member only)
# ======================================================================

REAL = Path(os.environ.get("LIVE_TRACKER_SCORE_DROP", "~/Downloads/igor_score_drops.zip")).expanduser()
REAL_VAL_B_SHA256 = "ff00f6fe31d9348d177f7c548a21ce13cae8c30560d5234ca11f16e42a008676"   # its SHA256SUMS.txt entry


@pytest.mark.skipif(not REAL.is_file(), reason="Igor's score archive is not present locally")
def test_real_validation_model_b_drop():
    drop = sd.load_validation_scores(REAL, "scores_validation_model_b.csv")
    p = drop.provenance
    assert p.member_sha256 == REAL_VAL_B_SHA256 and (p.n_sessions, p.n_frames) == (14, 94305)
    assert p.model_label == "model_b" and p.model_id is None and p.model_artifact_sha256 is None
    t = drop.scores()
    assert set(t["session"]) == set(json.loads((ROOT / "splits" / "split_manifest.json").read_text())["validation"])
    assert int((~t["tracking_ok"]).sum()) == 1067
    events = run_monitor(t, approved(), run_id="real_smoke")
    assert all(e["start_s"] < e["end_s"] for e in events)
    with pytest.raises(sd.SplitRefused):
        sd.load_validation_scores(REAL, "scores_test_model_b.csv")


# ======================================================================
# Codex 2 S9-1a review: five reproduced defects
# ======================================================================

# ---- 1. archive replacement race (A -> B -> A) ----

def test_archive_is_read_once_and_a_swap_cannot_change_what_is_consumed(tmp_path, manifest, monkeypatch):
    a_path = archive(tmp_path, name="a.zip")
    other = table()
    other["attack"] = 0.0                                                   # B: different scores, valid archive
    b_path = archive(tmp_path, {VAL_MEMBER: csv_bytes(other), TEST_MEMBER: csv_bytes()}, name="b.zip")
    a_bytes, b_bytes = a_path.read_bytes(), b_path.read_bytes()
    a_copy = tmp_path / "a_copy.zip"
    a_copy.write_bytes(a_bytes)
    swaps = [b_path, a_copy]                  # after open 1 the path holds B, after open 2 A again: A -> B -> A
    real_open, opens = os.open, []

    def swapping_open(path, flags, *a, **k):
        fd = real_open(path, flags, *a, **k)
        if Path(path) == a_path:
            opens.append(path)
            if len(opens) <= len(swaps):
                os.replace(swaps[len(opens) - 1], a_path)    # a new file at the path; open descriptors keep theirs
        return fd
    monkeypatch.setattr(os, "open", swapping_open)
    drop = load(a_path, manifest)
    assert len(opens) == 1                                                  # one snapshot, never reopened
    assert drop.provenance.archive_sha256 == hashlib.sha256(a_bytes).hexdigest()  # the bytes consumed: A
    pd.testing.assert_frame_equal(drop.scores(), table(), check_dtype=False)      # A's scores, not B's
    assert a_path.read_bytes() == b_bytes                                    # the path really did change


def test_reported_digest_always_matches_the_consumed_bytes(tmp_path, manifest, monkeypatch):
    path = archive(tmp_path)
    seen = []
    real = sd._snapshot

    def spy(p):
        name, data = real(p)
        seen.append(data)
        return name, data
    monkeypatch.setattr(sd, "_snapshot", spy)
    drop = load(path, manifest)
    assert len(seen) == 1 and drop.provenance.archive_sha256 == hashlib.sha256(seen[0]).hexdigest()
    sha, listed, members = sd.inspect_archive(path)
    assert len(seen) == 2 and sha == hashlib.sha256(seen[1]).hexdigest()


def test_oversized_or_non_regular_archive_is_refused(tmp_path, manifest, monkeypatch):
    path = archive(tmp_path)
    monkeypatch.setattr(sd, "MAX_ARCHIVE_BYTES", path.stat().st_size - 1)
    with pytest.raises(sd.ArchiveIntegrityError, match="larger than"):
        load(path, manifest)
    monkeypatch.undo()
    with pytest.raises(sd.ArchiveIntegrityError, match="not a regular file"):
        load(tmp_path, manifest)                                             # a directory


# ---- 2. output files never follow or overwrite existing paths ----

@posix_only
def test_symlinked_output_file_is_refused_and_its_target_untouched(tmp_path, synthetic_manifest_default, capsys):
    path = archive(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_text("unrelated content")
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").symlink_to(victim)
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(out))) == 2
    assert "never overwritten" in capsys.readouterr().err
    assert victim.read_text() == "unrelated content"
    assert sorted(p.name for p in out.iterdir()) == ["summary.json"]          # nothing else was created


@posix_only
@pytest.mark.parametrize("plant", ["events.journal.jsonl", "events.jsonl"])
def test_existing_outputs_and_symlinks_are_never_overwritten(tmp_path, synthetic_manifest_default, plant, capsys):
    path = archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / plant).write_text("earlier content")
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(out))) == 2
    assert (out / plant).read_text() == "earlier content"
    capsys.readouterr()
    victim = tmp_path / "victim.txt"
    victim.write_text("unrelated")
    out2 = tmp_path / "out2"
    out2.mkdir()
    (out2 / plant).symlink_to(victim)
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(out2))) == 2
    assert victim.read_text() == "unrelated"


@posix_only
def test_symlinked_or_non_directory_output_dir_is_refused(tmp_path, synthetic_manifest_default, capsys):
    path = archive(tmp_path)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir, target_is_directory=True)
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(link))) == 2
    assert "is a symlink" in capsys.readouterr().err and not list(real_dir.iterdir())
    a_file = tmp_path / "file"
    a_file.write_text("x")
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(a_file))) == 2
    assert "not a real directory" in capsys.readouterr().err and a_file.read_text() == "x"


@posix_only
def test_a_symlink_planted_during_the_run_is_not_followed(tmp_path, synthetic_manifest_default, monkeypatch, capsys):
    path = archive(tmp_path)
    out = tmp_path / "out"
    victim = tmp_path / "victim.txt"
    victim.write_text("unrelated")
    real = smoke.monitor_run

    def planting(*a, **k):
        result = real(*a, **k)
        (out / "summary.json").symlink_to(victim)                            # after the directory was claimed
        return result
    monkeypatch.setattr(smoke, "monitor_run", planting)
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(out))) == 2
    assert victim.read_text() == "unrelated" and "never overwritten" in capsys.readouterr().err


@posix_only
def test_fresh_output_directory_is_private_and_written_once(tmp_path, synthetic_manifest_default, capsys):
    path = archive(tmp_path)
    out = tmp_path / "fresh" / "run"
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(out))) == 0
    capsys.readouterr()
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    for name in smoke.OUTPUT_NAMES:
        st = os.lstat(out / name)
        assert stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o600
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    assert len(events) == 1 and events[0]["session"] == VAL[0]
    assert smoke.main(smoke_args(path, None, "--smoke-rule", "--out", str(out))) == 2   # a second run never overwrites


# ---- 3. CSV parser boundaries ----

@pytest.mark.parametrize("transform, text", [
    (lambda ls: ls + ['"SYNTHETIC_val_2,0,0.0,0.1,0.1,0.1,True'], "unexpected end of data"),     # unterminated quote
    (lambda ls: set_field(ls, 5, "session", '"SYNTHETIC_val_1"x'), "expected after"),          # malformed escaping
    (lambda ls: set_field(ls, 5, "session", "S" * 200_000), "field larger than field limit"),  # oversized field
    (lambda ls: ['"session,frame'] + ls[1:], "malformed CSV"),                                 # broken header
])
def test_malformed_csv_is_a_schema_error_with_its_line(tmp_path, manifest, transform, text):
    members = {VAL_MEMBER: edit_csv(transform), TEST_MEMBER: csv_bytes()}
    with pytest.raises(sd.ScoreSchemaError, match=text) as e:
        load(archive(tmp_path, members), manifest)
    assert " line " in str(e.value)


# ---- 4. negative timestamps ----

@pytest.mark.parametrize("value", ["-1e-10", "-0.0000000001", "-1"])
def test_negative_timestamps_are_refused(tmp_path, manifest, value):
    members = {VAL_MEMBER: edit_csv(lambda ls: set_field(ls, 1, "time_s", value)), TEST_MEMBER: csv_bytes()}
    with pytest.raises(sd.ScoreSchemaError, match="line 2: time_s .* is negative"):
        load(archive(tmp_path, members), manifest)


def test_every_ingested_timestamp_meets_the_monitor_contract(tmp_path, manifest):
    t = load(archive(tmp_path), manifest).scores()
    assert (t["time_s"] >= 0).all() and t["time_s"].map(lambda x: x == x and abs(x) != float("inf")).all()
    run_monitor(t, approved(), run_id="contract")                          # the unchanged monitor accepts every row


# ---- 5. original ZIP filename (embedded NUL) ----

def nul_archive(tmp_path):
    """A member stored as 'scores_validation_model_b.csv\\0../evil' that zipfile
    exposes as the safe-looking 'scores_validation_model_b.csv'."""
    safe, crafted = VAL_MEMBER, VAL_MEMBER + "Q../evil"
    path = archive(tmp_path, {crafted: csv_bytes()}, sums=sums_for({safe: csv_bytes()}), name="nul.zip")
    raw = path.read_bytes()
    patched = raw.replace(crafted.encode(), (VAL_MEMBER + "\x00../evil").encode())
    assert patched.count(b"\x00../evil") == 2                              # local header + central directory
    path.write_bytes(patched)
    return path


def test_nul_in_the_original_member_name_is_refused(tmp_path, manifest):
    path = nul_archive(tmp_path)
    with zipfile.ZipFile(path) as z:
        info = z.infolist()[0]
        assert info.filename == VAL_MEMBER and "\x00" in info.orig_filename     # the truncation is real
    with pytest.raises(sd.ArchiveIntegrityError, match="altered when read"):
        load(path, manifest)


def test_unsafe_paths_duplicates_and_symlinks_are_still_refused_after_the_name_fix(tmp_path, manifest):
    import warnings
    for extra, text in (([("../x.csv", b"x")], "unsafe member path"), ([(VAL_MEMBER, b"x")], "duplicate"),
                        ([(symlink("scores_validation_l.csv"), b"x")], "symlink member")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = archive(tmp_path, extra=extra, name=f"{text[:4]}.zip")
        with pytest.raises(sd.ArchiveIntegrityError, match=text):
            load(p, manifest)
