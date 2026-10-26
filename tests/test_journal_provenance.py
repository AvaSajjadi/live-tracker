"""Tests for Stage 9 S9-2: run provenance headers, source binding, run
completion and frozen journal evidence (event_log + Monitor(provenance=...)).
New file.

Synthetic score drops (fake_scores, SYNTHETIC_ sessions) by default; one
opt-in check uses Igor's real validation archive when it is present.
"""

import copy
import hashlib
import json

import pytest

import live_tracker.event_log as el
import live_tracker.score_drops as sd
from live_tracker.monitor import Monitor, MonitorInputError, run_monitor
from live_tracker.rules import RuleSet
from test_score_drops import REAL, archive, load, manifest, table  # noqa: F401

RUN = "prov_run"


def approved(*specs):
    """Approved rules: (rule_id, behavior, alert_threshold)."""
    rules = RuleSet()
    for rid, behavior, alert in specs or (("R1", "attack", 0.99),):     # 0.99: the synthetic bout is an "ask"
        rules.add({"rule_id": rid, "behavior": behavior, "min_duration_s": 2.0, "alert_threshold": alert,
                   "ask_threshold": 0.5})
        rules.approve(rid)
    return rules.active()


@pytest.fixture
def drop(tmp_path, manifest):  # noqa: F811
    return load(archive(tmp_path), manifest)


def monitored(source, rows, path=None, run_id=RUN, rules=None):
    """A finished monitor run over `rows` with `source` as its provenance (None: none)."""
    m = Monitor(rules or approved(), journal_path=path, run_id=run_id, provenance=source)
    for row in rows:
        m.step(row)
    return m, m.finish()


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def write_lines(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def status(path, run_id=RUN):
    return el.EventJournal(path).run_provenance(run_id)


# ======================================================================
# 1. the run header
# ======================================================================

def test_run_header_binds_source_and_rules(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    m, events = monitored(drop, drop.rows(), path, rules=approved(("R2", "mount", 0.8), ("R1", "attack", 0.99)))
    header = lines(path)[0]
    assert header["record"] == "run_started" and header["run_id"] == RUN
    assert header["header_version"] == "run-header/1" and header["frame_period_s"] == 1 / 30
    assert header["source_binding"] == "verified_load"
    assert [r["rule_id"] for r in header["rules"]] == ["R1", "R2"]
    assert all(r["status"] == "approved" for r in header["rules"])
    canon = json.dumps(header["rules"], sort_keys=True, separators=(",", ":")).encode()
    assert header["rules_digest"] == hashlib.sha256(canon).hexdigest()
    assert header["source"] == drop.provenance.to_json()                            # copied, never completed
    src = header["source"]
    assert (src["model_artifact_sha256"], src["model_id"], src["feature_configuration"], src["producing_commit"]) \
        == (None, None, None, None)
    assert src["model_label"] == "model_b" and src["model_label_status"].startswith("unverified")
    assert src["rows_sha256"] == sd.rows_sha256(drop.rows()) and src["n_frames"] == len(drop.rows())
    assert set(el._SOURCE_FIELDS) == set(drop.provenance.to_json())                 # in sync with S9-1a
    finished = lines(path)[-1]
    assert finished == {"record": "run_finished", "run_id": RUN, "n_rows": len(drop.rows()), "n_events": 1,
                        "events_sha256": el.events_digest(lines(path)[1:-1]),
                        "input_rows_sha256": src["rows_sha256"]}
    assert status(path) == {"status": "verified_score_drop", "completion": "complete", "header": header,
                            "finished": finished}


def test_provenance_changes_no_event(tmp_path, drop):
    plain_path, prov_path = tmp_path / "plain.jsonl", tmp_path / "prov.jsonl"
    _, plain = monitored(None, drop.rows(), plain_path)
    _, prov = monitored(drop, drop.rows(), prov_path)
    assert prov == plain and len(prov) == 1 and prov[0]["action"] == "ask"
    assert run_monitor(drop.scores(), approved(), run_id=RUN, provenance=drop) == plain
    # the same lifecycle records between the header and the completion record
    assert lines(prov_path)[1:-1] == lines(plain_path)
    assert [lines(prov_path)[0]["record"], lines(prov_path)[-1]["record"]] == ["run_started", "run_finished"]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    el.export_events(plain_path, a)
    el.export_events(prov_path, b)
    assert a.read_bytes() == b.read_bytes()                                          # §4 export unchanged


def test_replay_with_the_same_provenance_is_idempotent(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    monitored(drop, drop.rows(), path)
    before = path.read_bytes()
    monitored(drop, drop.rows(), path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change, text", [
    (lambda h: h["source"].update(model_id="model_b"), "model identity is never claimed"),
    (lambda h: h["source"].update(model_artifact_sha256="a" * 64), "model identity is never claimed"),
    (lambda h: h["source"].update(feature_configuration={"features": "basic,motion"}), "model identity"),
    (lambda h: h["source"].update(split="test"), "is not validation"),
    (lambda h: h["source"].update(model_label_status="verified"), "marked unverified"),
    (lambda h: h["source"].update(member_sha256="xyz"), "member_sha256"),
    (lambda h: h["source"].pop("manifest_sha256"), "source fields"),
    (lambda h: h.update(rules_digest="0" * 64), "rules_digest does not match"),
    (lambda h: h["rules"][0].update(status="proposed"), "is not approved"),
    (lambda h: h.update(header_version="run-header/9"), "unknown header_version"),
    (lambda h: h.update(source_binding="trusted"), "unknown source_binding"),
    (lambda h: h.update(extra=1), "fields"),
])
def test_invalid_headers_are_refused(drop, change, text):
    header = el.run_header(RUN, approved(), 1 / 30, drop.provenance)
    change(header)
    with pytest.raises(el.JournalIntegrityError, match=text):
        el.JournalState.from_records([header])


def test_different_provenance_under_the_same_run_id_conflicts(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    monitored(drop, drop.rows(), path)
    other = copy.deepcopy(drop.provenance.to_json())
    other["archive_sha256"] = "f" * 64
    with pytest.raises(el.JournalConflictError):
        Monitor(approved(), journal_path=path, run_id=RUN, provenance=other)


def test_header_after_events_and_events_outside_the_header_are_refused(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    monitored(None, drop.rows(), path)
    with pytest.raises(el.JournalConflictError, match="header must come first"):
        Monitor(approved(), journal_path=path, run_id=RUN, provenance=drop)
    path2 = tmp_path / "h.jsonl"
    monitored(drop, drop.rows(), path2)
    header, *rest = lines(path2)
    fired = next(r for r in rest if r["record"] == "event_fired")
    for change, text in ((lambda f: f.update(rule_id="R_OTHER"), "is not in the header"),
                         (lambda f: f["rule"].update(min_duration_s=1.0), "differs from the header"),
                         (lambda f: f.update(frame_period_s=1 / 25), "differs from the header")):
        bad = copy.deepcopy(fired)
        change(bad)
        with pytest.raises(el.JournalIntegrityError, match=text):
            el.JournalState.from_records([header, bad])


# ======================================================================
# 2. Codex finding 1: only a verified ScoreDrop over its own rows is verified
# ======================================================================

def fabricated_source():
    """Codex's reproduction: a well-formed source dict with plausible hashes."""
    return {"archive_name": "made_up.zip", "archive_sha256": "1" * 64, "archive_sha256_check": "matched expected digest",
            "member": "scores_validation_model_x.csv", "member_sha256": "2" * 64,
            "member_sha256_check": "matched SHA256SUMS.txt", "split": "validation", "manifest_sha256": "3" * 64,
            "n_sessions": 2, "n_frames": len(table()), "rows_sha256": sd.rows_sha256(table().to_dict("records")),
            "model_label": "model_x", "model_label_status": "unverified: inferred from the member file name only",
            "model_artifact_sha256": None, "model_id": None, "feature_configuration": None,
            "producing_commit": None, "model_provenance_status": "unavailable"}


def test_codex_fabricated_source_is_never_verified(tmp_path, drop):
    path = tmp_path / "fab.jsonl"
    # even with the matching rows digest of the rows actually monitored, a dict is only declared
    monitored(fabricated_source(), drop.rows(), path)
    s = status(path)
    assert s["header"]["source_binding"] == "declared" and s["completion"] == "complete"
    assert s["status"] == "declared_unverified"
    assert el.run_header(RUN, approved(), 1 / 30, fabricated_source())["source_binding"] == "declared"
    assert el.freeze_run(path, RUN).provenance_status == "declared_unverified"


@pytest.mark.parametrize("make", [
    lambda d: d.provenance,                                                    # the genuine provenance object
    lambda d: d.provenance.to_json(),                                          # its dict
    lambda d: sd.ScoreDrop(provenance=d.provenance, _table=d.scores()),        # a hand-built drop, no issuer
    lambda d: sd.ScoreDrop(provenance=d.provenance, _table=d.scores(), _issuer=object()),
])
def test_anything_but_an_issued_score_drop_is_declared(tmp_path, drop, make):
    path = tmp_path / "j.jsonl"
    monitored(make(drop), drop.rows(), path)
    assert status(path)["status"] == "declared_unverified"
    assert sd.trusted_rows_sha256(make(drop)) is None


def test_an_issued_drop_with_altered_rows_or_input_is_not_verified(tmp_path, drop):
    tampered = load(archive(tmp_path, name="t.zip"), tmp_path / "manifest.json")
    tampered._table.loc[0, "attack"] = 0.5                                      # the drop's table changed after load
    assert sd.trusted_rows_sha256(tampered) is None
    p1 = tmp_path / "a.jsonl"
    monitored(tampered, tampered.rows(), p1)
    assert status(p1)["header"]["source_binding"] == "declared"
    p2 = tmp_path / "b.jsonl"
    rows = drop.rows()
    monitored(drop, rows[:-1], p2)                                              # verified load, other input
    s = status(p2)
    assert s["header"]["source_binding"] == "verified_load" and s["status"] == "declared_unverified"
    p3 = tmp_path / "c.jsonl"
    other = [dict(r) for r in rows]
    other[3]["mount"] = 0.0                                                     # same row count, changed values
    monitored(drop, other, p3)
    assert status(p3)["status"] == "declared_unverified"
    p4 = tmp_path / "d.jsonl"
    monitored(drop, rows, p4)
    assert status(p4)["status"] == "verified_score_drop"


def test_a_run_with_provenance_needs_complete_rows(drop):
    m = Monitor(approved(), run_id=RUN, provenance=drop)
    row = dict(drop.rows()[0])
    row.pop("mount")
    with pytest.raises(MonitorInputError, match="complete score rows"):
        m.step(row)


# ======================================================================
# 3. Codex finding 2: completion is explicit and binds count and digest
# ======================================================================

def test_codex_unfinished_zero_event_run_is_not_frozen(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    Monitor(approved(), journal_path=path, run_id=RUN, provenance=drop)        # header only, never finished
    assert status(path)["completion"] == "unfinished" and status(path)["status"] == "declared_unverified"
    with pytest.raises(el.JournalIntegrityError, match="unfinished run is never frozen"):
        el.freeze_run(path, RUN)
    m = Monitor(approved(), journal_path=path, run_id=RUN, provenance=drop)
    for row in drop.rows()[:200]:                                               # frames seen, not finished
        m.step(row)
    with pytest.raises(el.JournalIntegrityError, match="never frozen"):
        el.freeze_run(path, RUN)


def _resealed(snap, change):
    payload = snap.payload()
    change(payload)
    data = el.canonical_bytes(payload)
    return data, hashlib.sha256(data).hexdigest()


def test_codex_deleted_event_pairs_are_refused_even_when_resealed(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    rules = approved(("R1", "attack", 0.99), ("R2", "attack", 0.5))          # two events: one ask, one alert
    monitored(drop, drop.rows(), path, rules=rules)
    snap = el.freeze_run(path, RUN)
    assert snap.n_events == 2 and snap.completion == "complete"
    eid = next(r["event_id"] for r in snap.payload()["records"] if r["record"] == "event_fired")
    data, sha = _resealed(snap, lambda p: p.update(records=[r for r in p["records"] if r["event_id"] != eid]))
    with pytest.raises(el.JournalIntegrityError, match="n_events / events_sha256 do not match"):
        el.load_snapshot(data, sha)
    data, sha = _resealed(snap, lambda p: p.update(records=[]))                 # every pair removed
    with pytest.raises(el.JournalIntegrityError, match="do not match"):
        el.load_snapshot(data, sha)
    data, sha = _resealed(snap, lambda p: p.update(finished=None))             # completion removed
    with pytest.raises(el.JournalIntegrityError, match="never frozen|not consistent"):
        el.load_snapshot(data, sha)
    write_lines(path, [r for r in lines(path) if r.get("event_id") != eid])    # the same deletion in the journal
    with pytest.raises(el.JournalIntegrityError, match="do not match"):
        el.freeze_run(path, RUN)


def test_completion_boundaries(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    monitored(drop, drop.rows(), path)
    header, *events, done = lines(path)
    fired = next(r for r in events if r["record"] == "event_fired")
    late = {**fired, "event_id": f"E-{RUN}-9", "first_frame": fired["first_frame"] + 1000,
            "fired_frame": fired["fired_frame"] + 1000}
    with pytest.raises(el.JournalIntegrityError, match="after run 'prov_run' finished"):
        el.JournalState.from_records([header, *events, done, late])
    with pytest.raises(el.JournalIntegrityError, match="no run_started header"):
        el.JournalState.from_records([*events, done])
    open_events = [r for r in events if r["record"] == "event_fired"]
    with pytest.raises(el.JournalIntegrityError, match="still open"):
        el.JournalState.from_records([header, *open_events, {**done, "events_sha256": el.events_digest(open_events)}])
    for change in (lambda d: d.update(n_events=d["n_events"] + 1), lambda d: d.update(events_sha256="0" * 64)):
        bad = dict(done)
        change(bad)
        with pytest.raises(el.JournalIntegrityError, match="do not match"):
            el.JournalState.from_records([header, *events, bad])
    with pytest.raises(el.JournalConflictError):
        el.JournalState.from_records([header, *events, done, {**done, "n_rows": done["n_rows"] + 1}])
    with pytest.raises(el.JournalIntegrityError, match="SHA-256 hex"):
        el.JournalState.from_records([header, *events, {**done, "input_rows_sha256": "x"}])


def test_answers_after_completion_are_allowed_and_excluded(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    m, events = monitored(drop, drop.rows(), path)
    snap = el.freeze_run(path, RUN)
    m.answer(events[0]["event_id"], "confirmed")                               # appended after run_finished
    assert lines(path)[-1]["record"] == "event_answered"
    assert el.verify_snapshot(snap, path).sha256 == snap.sha256
    assert el.load_snapshot(snap.canonical, snap.sha256) == snap


# ======================================================================
# 4. backward compatibility (journals without headers)
# ======================================================================

def test_journals_without_headers_stay_readable_and_never_claim_completion(tmp_path, drop):
    path = tmp_path / "old.jsonl"
    m, events = monitored(None, drop.rows(), path)
    assert {r["record"] for r in lines(path)} <= {"event_fired", "event_closed"}  # nothing new is written
    j = el.EventJournal(path)
    assert j.events(RUN) == events and j.run_ids() == [RUN]
    assert j.run_provenance(RUN) == {"status": "no_source_provenance", "completion": "legacy_unrecorded",
                                     "header": None, "finished": None}
    snap = el.freeze_run(path, RUN)
    assert (snap.provenance_status, snap.completion) == ("no_source_provenance", "legacy_unrecorded")
    assert snap.payload()["header"] is None and snap.payload()["finished"] is None
    with pytest.raises(KeyError):
        j.run_provenance("absent")


def test_a_pre_s9_2_journal_file_reads_unchanged(tmp_path):
    from live_tracker.fake_scores import demo_scores
    path = tmp_path / "day1.jsonl"
    rules = RuleSet()
    rules.add({"rule_id": "R1", "behavior": "attack", "min_duration_s": 2.0, "alert_threshold": 0.8,
               "ask_threshold": 0.5})
    rules.approve("R1")
    events = run_monitor(demo_scores(), rules.active(), journal_path=path, run_id="day1")
    raw = path.read_bytes()
    assert el.EventJournal(path).events("day1") == events and path.read_bytes() == raw
    assert el.freeze_run(path, "day1").completion == "legacy_unrecorded"


# ======================================================================
# 5. frozen evidence
# ======================================================================

def test_snapshot_is_deterministic(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    m, _ = monitored(drop, drop.rows(), path)
    first = el.freeze_run(path, RUN)
    assert first == el.freeze_run(path, RUN) == el.freeze_run(m.journal, RUN)
    p = first.payload()
    assert p["snapshot_version"] == "journal-snapshot/1" and p["provenance_status"] == "verified_score_drop"
    assert p["completion"] == "complete" and p["finished"] == lines(path)[-1]
    assert p["header"] == lines(path)[0] and [r["record"] for r in p["records"]] == ["event_fired", "event_closed"]
    assert first.sha256 == hashlib.sha256(first.canonical).hexdigest()
    other = tmp_path / "other.jsonl"
    monitored(drop, drop.rows(), other)
    assert el.freeze_run(other, RUN).sha256 == first.sha256


def test_snapshot_detects_changes_to_the_run(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    monitored(drop, drop.rows(), path)
    snap = el.freeze_run(path, RUN)
    recs = lines(path)
    closed = next(r for r in recs if r["record"] == "event_closed")
    closed["bout_max_score"] = min(1.0, closed["bout_max_score"] + 0.001)
    write_lines(path, recs)
    with pytest.raises(el.JournalIntegrityError, match="do not match"):        # the completion digest catches it
        el.verify_snapshot(snap, path)
    with pytest.raises(el.JournalIntegrityError, match="do not match their SHA-256"):
        el.load_snapshot(snap.canonical.replace(b"verified_score_drop", b"verified_score_drip"), snap.sha256)
    pretty = json.dumps(snap.payload(), indent=1).encode()
    with pytest.raises(el.JournalIntegrityError, match="canonical"):
        el.load_snapshot(pretty, hashlib.sha256(pretty).hexdigest())


@pytest.mark.parametrize("change, text", [
    (lambda p: p["records"].append(copy.deepcopy(p["records"][0])), "duplicate records"),
    (lambda p: p["records"].pop(), "still open|do not match"),
    (lambda p: p["records"].pop(0), "no earlier event_fired"),
    (lambda p: p.update(provenance_status="no_source_provenance"), "not consistent"),
    (lambda p: p.update(completion="legacy_unrecorded"), "not consistent"),
    (lambda p: p["records"][0].update(run_id="another_run"), "another run"),
    (lambda p: p["records"].append({"record": "event_answered", "event_id": "x", "researcher_answer": "confirmed"}),
     "event_fired / event_closed"),
    (lambda p: p["header"]["source"].update(model_id="model_b"), "model identity"),
    (lambda p: p["header"].update(source_binding="declared"), "not consistent"),
    (lambda p: p["finished"].update(input_rows_sha256="0" * 64), "not consistent"),
])
def test_resealed_but_inconsistent_snapshots_are_refused(tmp_path, drop, change, text):
    path = tmp_path / "j.jsonl"
    monitored(drop, drop.rows(), path)
    data, sha = _resealed(el.freeze_run(path, RUN), change)
    with pytest.raises(el.JournalIntegrityError, match=text):
        el.load_snapshot(data, sha)


def test_duplicate_and_contradictory_lifecycle_records(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    monitored(drop, drop.rows(), path)
    snap = el.freeze_run(path, RUN)
    recs = lines(path)
    write_lines(path, recs + [recs[1]])                                         # identical duplicate: dropped
    assert el.verify_snapshot(snap, path).sha256 == snap.sha256
    fired = next(r for r in recs if r["record"] == "event_fired")
    contradicting = {**fired, "action": "alert" if fired["action"] != "alert" else "log"}
    write_lines(path, recs + [contradicting])
    with pytest.raises(el.JournalConflictError):
        el.freeze_run(path, RUN)
    with pytest.raises(KeyError):
        el.freeze_run(tmp_path / "missing.jsonl", "no_such_run")


def test_runs_in_one_journal_are_isolated(tmp_path, drop):
    path = tmp_path / "j.jsonl"
    monitored(drop, drop.rows(), path, run_id="run_a")
    a = el.freeze_run(path, "run_a")
    monitored(None, drop.rows(), path, run_id="run_b")
    monitored(fabricated_source(), drop.rows(), path, run_id="run_c", rules=approved(("R1", "attack", 0.5)))
    j = el.EventJournal(path)
    assert j.run_ids() == ["run_a", "run_b", "run_c"]
    assert el.verify_snapshot(a, path).sha256 == a.sha256
    assert [j.run_provenance(r)["status"] for r in j.run_ids()] == ["verified_score_drop", "no_source_provenance",
                                                                   "declared_unverified"]
    b, c = el.freeze_run(path, "run_b"), el.freeze_run(path, "run_c")
    assert len({a.sha256, b.sha256, c.sha256}) == 3


# ======================================================================
# 6. opt-in: the real validation drop (not run by default in this checkpoint's review)
# ======================================================================

@pytest.mark.skipif(not REAL.is_file(), reason="Igor's score archive is not present locally")
def test_real_validation_drop_with_a_run_header():
    real = sd.load_validation_scores(REAL, "scores_validation_model_b.csv")
    rules = RuleSet()
    rules.add({"rule_id": "SMOKE", "behavior": "attack", "min_duration_s": 2.0, "alert_threshold": 0.8,
               "ask_threshold": 0.5})
    rules.approve("SMOKE")
    m, events = monitored(real, real.rows(), None, run_id="real_val_b", rules=rules.active())
    snap = el.freeze_run(m.journal, "real_val_b")
    assert len(events) == snap.n_events == 7 and snap.provenance_status == "verified_score_drop"
    assert snap.payload()["header"]["source"]["member_sha256"].startswith("ff00f6fe")
    assert el.load_snapshot(snap.canonical, snap.sha256) == snap
