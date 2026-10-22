"""Tests for the end-of-session summary (summary.py), D4-1. New file.

The summary is built from monitor journal records only. Journals here come
from the real Monitor run over SYNTHETIC fake_scores sessions (or from
ReplaySession, D3-1). No CalMS21, cache, score archive, model, LLM or network.
"""

import copy
import dataclasses
import json

import pytest

import live_tracker.event_log as el
import live_tracker.replay as rp
import live_tracker.replay_view as rv
import live_tracker.summary as sm
from live_tracker.fake_scores import DEMO_SESSION, Segment, TrackingLoss, build_scores, demo_scores
from live_tracker.monitor import Monitor
from test_replay import R_ALERT, R_ASK, R_LOG, R_SHORT, end_session, ruleset, S_END

SRC = "replay.tracking_warnings on SYNTHETIC pose (test)"


def run(scores, rules, run_id="t1", finish=True):
    m = Monitor(ruleset(*rules).active(), run_id=run_id)
    for row in scores.sort_values("frame").to_dict("records"):
        m.step(row)
    if finish:
        m.finish()
    return m


def summary(m, session, rules=None, **kw):
    return sm.build_summary(m.journal.records, run_id=m.run_id, session=session,
                            rules=None if rules is None else ruleset(*rules).active(), **kw)


def one(behavior, segs, session="SYNTHETIC_one", duration=10.0, loss=()):
    return build_scores(session, duration, tuple(Segment(behavior, a, b, lo, hi) for a, b, lo, hi in segs), loss)


# ======================================================================
# sessions with different histories
# ======================================================================

def test_an_empty_journal():
    s = sm.build_summary([], run_id="nothing", session="SYNTHETIC_empty")
    d = s.as_dict()
    assert d["counts"] == {"alert": 0, "ask": 0, "log": 0} and d["events"] == []
    assert d["answers"] == {"confirmed": 0, "rejected": 0, "unanswered": 0, "timing_unknown": 0}
    assert d["rules"]["rules"] == [] and d["recording"] is None and d["provenance"]["label"] == "SYNTHETIC"
    md = s.to_markdown()
    assert "No event fired in this run." in md and "- None recorded." in md and "Recording\n\n- Not supplied." in md


def test_a_session_with_rules_but_no_events():
    m = run(one("attack", [(2.0, 2.5, 0.9, 0.95)]), [R_ALERT])                 # 0.5 s: below the 2 s rule
    d = summary(m, "SYNTHETIC_one", rules=[R_ALERT]).as_dict()
    assert d["counts"] == {"alert": 0, "ask": 0, "log": 0}
    assert [r["rule_id"] for r in d["rules"]["rules"]] == ["R1"] and d["per_rule"][0]["alert"] == 0


def test_alert_only():
    d = summary(run(one("attack", [(1.0, 4.0, 0.9, 0.95)]), [R_ALERT]), "SYNTHETIC_one", [R_ALERT]).as_dict()
    assert d["counts"] == {"alert": 1, "ask": 0, "log": 0} and d["answers"]["unanswered"] == 0
    (e,) = d["events"]
    assert (e["onset_s"], e["fired_s"]) == (1.0, 3.0) and e["closure"]["end_s"] == pytest.approx(4.0)
    assert e["closure"]["duration_kind"] == "complete" and e["closure"]["duration_s"] == pytest.approx(3.0)
    assert "answer" not in e


def test_ask_only_and_unanswered_stays_unanswered():
    d = summary(run(demo_scores(), [R_ASK]), DEMO_SESSION, [R_ASK]).as_dict()
    assert d["counts"] == {"alert": 0, "ask": 2, "log": 0}
    assert d["answers"] == {"confirmed": 0, "rejected": 0, "unanswered": 2, "timing_unknown": 0}
    assert all(e["answer"] == {"researcher_answer": "unanswered", "answered_frame": None, "timing": "not answered"}
               for e in d["events"])
    assert any("2 ask event(s) were not answered" in c for c in d["caveats"])


def test_log_only():
    d = summary(run(demo_scores(), [R_LOG]), DEMO_SESSION, [R_LOG]).as_dict()
    assert d["counts"] == {"alert": 0, "ask": 0, "log": 1} and d["events"][0]["behavior"] == "investigation"


def test_mixed_history_counts_per_rule():
    d = summary(run(demo_scores(), [R_ALERT, R_SHORT, R_LOG]), DEMO_SESSION, [R_ALERT, R_SHORT, R_LOG]).as_dict()
    assert d["counts"] == {"alert": 4, "ask": 2, "log": 1}
    assert {r["rule_id"]: (r["alert"], r["ask"], r["log"]) for r in d["per_rule"]} == \
        {"R1": (1, 1, 0), "R2": (3, 1, 0), "R4": (0, 0, 1)}
    assert [e["fired_frame"] for e in d["events"]] == sorted(e["fired_frame"] for e in d["events"])


# ======================================================================
# answers
# ======================================================================

def test_confirmed_rejected_unanswered_and_timing_unknown():
    scores = build_scores("SYNTHETIC_asks", 20.0, tuple(Segment("attack", a, a + 3, 0.6, 0.7) for a in (1, 6, 11, 16)))
    m = run(scores, [R_ALERT])
    ids = [e["event_id"] for e in m.journal.events()]
    m.answer(ids[0], "confirmed")
    m.answer(ids[1], "rejected")
    m.answer(ids[2], "confirmed")
    d = summary(m, "SYNTHETIC_asks", [R_ALERT], answer_positions={ids[0]: 100, ids[1]: 250}).as_dict()
    assert d["answers"] == {"confirmed": 2, "rejected": 1, "unanswered": 1, "timing_unknown": 1}
    a = [e["answer"] for e in d["events"]]
    assert a[0] == {"researcher_answer": "confirmed", "answered_frame": 100,
                    "timing": "position supplied; not checked against a recording"}
    assert a[1]["researcher_answer"] == "rejected" and a[1]["answered_frame"] == 250
    assert a[2] == {"researcher_answer": "confirmed", "answered_frame": None, "timing": "unknown"}
    assert a[3]["researcher_answer"] == "unanswered"
    md = summary(m, "SYNTHETIC_asks", [R_ALERT], answer_positions={ids[0]: 100}).to_markdown()
    assert "confirmed (frame 100, presence unverified)" in md and "confirmed (time unknown)" in md
    assert "| unanswered |" in md


def test_a_position_for_an_unanswered_ask_is_refused():
    m = run(demo_scores(), [R_ALERT])
    ask = m.journal.events()[1]["event_id"]
    with pytest.raises(sm.SummaryError, match="not an answered ask event"):
        summary(m, DEMO_SESSION, [R_ALERT], answer_positions={ask: 300})
    d = summary(m, DEMO_SESSION, [R_ALERT], answer_positions={ask: None}).as_dict()   # unknown stays unknown
    assert d["events"][1]["answer"]["researcher_answer"] == "unanswered"


# ======================================================================
# closures
# ======================================================================

def test_open_bouts_and_stream_end_closures():
    m = run(end_session(), [R_ALERT])                        # tracking_lost, then a bout to the end
    first, last = summary(m, S_END, [R_ALERT]).as_dict()["events"]
    assert first["closure"]["end_reason"] == "tracking_lost" and first["closure"]["duration_kind"] == "observed"
    assert last["closure"]["end_reason"] == "stream_end" and last["closure"]["duration_kind"] == "observed"
    assert "recording ended" in last["closure"]["ending"]
    crashed = run(end_session(), [R_ALERT], run_id="t2", finish=False)      # stopped before closing
    d = summary(crashed, S_END, [R_ALERT]).as_dict()
    assert d["events"][1]["closure"] == {"status": "not_recorded", "end_s": None, "last_frame": None,
                                         "end_reason": None, "duration_s": None, "duration_kind": "unknown",
                                         "ending": "no closure in the journal (the run may have stopped early)"}
    md = summary(crashed, S_END, [R_ALERT]).to_markdown()
    assert "| not recorded | — |" in md and "(observed)" in md


# ======================================================================
# tracking warnings: only from a separately identified source
# ======================================================================

def _warned():
    e = rv.synthetic_catalog()[1]
    r = rp.build_replay(e.scores, e.session, ruleset(R_ALERT), warnings=rv.entry_warnings(e), pose=e.pose)
    return r


def test_tracking_warnings_come_from_their_own_source():
    r = _warned()
    without = sm.summary_from_replay(r).as_dict()["tracking_warnings"]
    assert without == {"supplied": False, "source": None, "total": None, "by_kind": None, "warnings": []}
    d = sm.summary_from_replay(r, warning_source=SRC).as_dict()
    t = d["tracking_warnings"]
    assert t["supplied"] and t["source"] == SRC and t["total"] == len(r.end_state().warnings) >= 3
    assert {"missing_coordinates", "low_confidence", "frozen_tracker"} <= set(t["by_kind"])
    assert d["counts"]["alert"] + d["counts"]["ask"] + d["counts"]["log"] == len(d["events"])   # not mixed in
    md = sm.summary_from_replay(r, warning_source=SRC).to_markdown()
    assert "## Tracking warnings (separate from monitored events)" in md and "coordinates present but unreliable" in md
    assert "No tracking-warning source was supplied" in " ".join(sm.summary_from_replay(r).as_dict()["caveats"])


def test_warnings_without_a_source_or_unfinished_are_refused():
    r = _warned()
    entries = list(r.end_state().warnings)
    args = dict(run_id=r.run_id, session=r.session)
    with pytest.raises(sm.SummaryError, match="need their named source"):
        sm.build_summary(r.journal.records, warnings=entries, **args)
    ongoing = [w.to_log_entry(end_known=False, at_end=False) for w, _, _ in r._warnings][:1]
    with pytest.raises(sm.SummaryError, match="end-of-session state only"):
        sm.build_summary(r.journal.records, warnings=ongoing, warning_source=SRC, **args)
    other = [{**entries[0], "session": "SYNTHETIC_other"}]
    with pytest.raises(sm.SummaryError, match="another session"):
        sm.build_summary(r.journal.records, warnings=other, warning_source=SRC, **args)


# ======================================================================
# refused histories and isolation
# ======================================================================

def test_corrupt_and_contradictory_records_are_refused():
    m = run(demo_scores(), [R_ALERT])
    recs = m.journal.records
    broken = copy.deepcopy(recs)
    del broken[0]["fired_frame"]
    with pytest.raises(sm.SummaryError, match="malformed"):
        sm.build_summary(broken, run_id="t1", session=DEMO_SESSION)
    closed = next(r for r in recs if r["record"] == "event_closed")
    with pytest.raises(sm.SummaryError, match="contradictory"):
        sm.build_summary(recs + [{**closed, "end_s": closed["end_s"] + 1.0}], run_id="t1", session=DEMO_SESSION)
    orphan = {"record": "event_closed", "event_id": "E-x-1", "end_s": 1.0, "end_reason": "below_threshold",
              "last_frame": 1, "bout_frames": 2, "bout_mean_score": 0.9, "bout_min_score": 0.9, "bout_max_score": 0.9}
    with pytest.raises(sm.SummaryError):
        sm.build_summary(recs + [orphan], run_id="t1", session=DEMO_SESSION)
    alert = recs[0]["event_id"]
    with pytest.raises(sm.SummaryError):
        sm.build_summary(recs + [el.answer_record(alert, "confirmed")], run_id="t1", session=DEMO_SESSION)
    with pytest.raises(sm.SummaryError, match="malformed"):
        sm.build_summary(["not a record"], run_id="t1", session=DEMO_SESSION)


def test_rules_that_do_not_match_the_journal_are_refused():
    m = run(demo_scores(), [R_ALERT])
    with pytest.raises(sm.SummaryError, match="does not match the approved rules"):
        summary(m, DEMO_SESSION, [{**R_ALERT, "alert_threshold": 0.9}])
    with pytest.raises(sm.SummaryError, match="does not match the approved rules"):
        summary(m, DEMO_SESSION, [R_LOG])
    with pytest.raises(sm.SummaryError, match="explicit run_id"):
        sm.build_summary([], run_id="", session=DEMO_SESSION)


def test_only_the_requested_run_and_session_are_read(tmp_path):
    path = tmp_path / "shared.jsonl"
    a = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT), journal_path=path)
    b = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_SHORT), journal_path=path)
    records = el.load_journal(path).records
    da = sm.build_summary(records, run_id=a.run_id, session=DEMO_SESSION, rules=a.rules).as_dict()
    db = sm.build_summary(records, run_id=b.run_id, session=DEMO_SESSION, rules=b.rules).as_dict()
    assert {e["event_id"].split("-")[1] for e in da["events"]} == {a.run_id}
    assert {e["event_id"].split("-")[1] for e in db["events"]} == {b.run_id}
    assert da["counts"] != db["counts"]
    with pytest.raises(sm.SummaryError, match="does not match"):           # the other run's rules
        sm.build_summary(records, run_id=a.run_id, session=DEMO_SESSION, rules=b.rules)
    assert sm.build_summary(records, run_id="absent", session=DEMO_SESSION).as_dict()["events"] == []


def test_a_run_holding_another_sessions_events_is_refused():
    both = demo_scores()
    other = one("attack", [(1.0, 4.0, 0.9, 0.95)], session="SYNTHETIC_second")
    m = Monitor(ruleset(R_ALERT).active(), run_id="mixed")
    for row in list(both.to_dict("records")) + list(other.to_dict("records")):
        m.step(row)
    m.finish()
    with pytest.raises(sm.SummaryError, match="another session"):
        sm.build_summary(m.journal.records, run_id="mixed", session=DEMO_SESSION)


# ======================================================================
# provenance, wording and determinism
# ======================================================================

def test_provenance_labels():
    m = run(one("attack", [(1.0, 4.0, 0.9, 0.95)], session="SYNTHETIC_one"), [R_ALERT])
    assert summary(m, "SYNTHETIC_one").as_dict()["provenance"]["label"] == "SYNTHETIC"
    real = run(build_scores("SYNTHETIC_x", 5.0).assign(session="task1/train/mouse001"), [R_ALERT], run_id="r")
    d = sm.build_summary(real.journal.records, run_id="r", session="task1/train/mouse001",
                         declared_provenance="validation score drop (not checked here)").as_dict()
    assert d["provenance"]["label"] == "unverified source"                     # declared is never verified
    assert d["provenance"]["declared"] == "validation score drop (not checked here)"
    assert not any("SYNTHETIC" in c for c in d["caveats"])


def test_wording_keeps_scores_detections_and_ground_truth_apart():
    md = summary(run(demo_scores(), [R_ALERT]), DEMO_SESSION, [R_ALERT]).to_markdown()
    assert "confidence (model score)" in md and "not scientific certainty" in md
    assert "not human annotations or ground truth" in md and "not rates, effects" in md
    for word in ("significan", "p-value", "causal", "caused", "effect size", "ground truth shows"):
        assert word not in md.lower()


def test_summaries_are_deterministic():
    a = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_ALERT, R_LOG))
    b = rp.build_replay(demo_scores(), DEMO_SESSION, ruleset(R_LOG, R_ALERT))
    sa, sb = sm.summary_from_replay(a), sm.summary_from_replay(b)
    assert sa.to_json() == sb.to_json() and sa.to_markdown() == sb.to_markdown()
    assert json.loads(sa.to_json()) == sa.as_dict()
    assert sm.summary_from_replay(a).to_json() == sa.to_json()
    d = sa.as_dict()
    d["counts"]["alert"] = 99                                                  # a copy: the summary is unchanged
    assert sa.as_dict()["counts"]["alert"] == 1


def test_the_replay_summary_is_the_end_state_never_an_earlier_frame():
    r = rp.build_replay(end_session(), S_END, ruleset(R_ALERT))
    d = sm.summary_from_replay(r).as_dict()
    assert d["events"][1]["closure"]["end_reason"] == "stream_end"          # known only after the last row
    assert d["recording"] == {"n_rows": 300, "first_frame": 0, "last_frame": 299, "first_time_s": 0.0,
                              "last_time_s": pytest.approx(299 / 30), "fps": 30.0, "source": "replayed score rows",
                              "duration_s": pytest.approx(10.0), "missing_frames": 0,
                              "frame_domain": "exact recorded frames"}


def test_missing_frames_are_reported():
    s = demo_scores()
    s = s[(s["frame"] < 100) | (s["frame"] >= 130)]
    d = sm.summary_from_replay(rp.build_replay(s, DEMO_SESSION, ruleset(R_ALERT))).as_dict()
    assert d["recording"]["missing_frames"] == 30 and any("30 frame(s) are missing" in c for c in d["caveats"])


def test_a_tracking_lost_ending_is_observed_not_complete():
    scores = one("attack", [(1.0, 5.0, 0.9, 0.95)], loss=(TrackingLoss(4.0, 4.5),))
    (e,) = summary(run(scores, [R_ALERT]), "SYNTHETIC_one").as_dict()["events"]
    assert e["closure"]["end_reason"] == "tracking_lost" and e["closure"]["duration_kind"] == "observed"
    assert "may have continued" in e["closure"]["ending"]



# ======================================================================
# Codex D4-1 review: provenance, rules, positions and recording, warnings
# ======================================================================

from live_tracker.rules import Rule                                          # noqa: E402
from test_journal_provenance import approved as prov_rules                   # noqa: E402
from test_score_drops import MANIFEST, TEST_MEMBER, VAL, VAL_MEMBER, archive, csv_bytes, load   # noqa: E402

REAL = "task1/train/mouse999"           # a non-SYNTHETIC session name, built from synthetic rows (test only)


def _manifest(tmp_path, session=VAL[0]):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({**MANIFEST, "validation": [session]}))          # one validation session
    return p


def _drop(tmp_path, name="drops.zip", seed=0, session=VAL[0]):
    rows = build_scores(VAL[0], 8.0, (Segment("attack", 1.0, 4.0, 0.6, 0.7),), seed=seed).assign(session=session)
    return load(archive(tmp_path, {VAL_MEMBER: csv_bytes(rows), TEST_MEMBER: csv_bytes(rows)}, name=name),
                _manifest(tmp_path, session))


def _verified_run(drop, run_id="v1"):
    m = Monitor(prov_rules(), run_id=run_id, provenance=drop)
    for row in drop.rows():
        m.step(row)
    m.finish()
    return m


def _fabricated(tmp_path):
    """Internally consistent run_started / run_finished records claiming a verified load, for a non-SYNTHETIC
    session, written by hand: the journal accepts them, and run_provenance reports verified_score_drop."""
    source = _drop(tmp_path).provenance.to_json()
    rows = build_scores("SYNTHETIC_x", 8.0, (Segment("attack", 1.0, 4.0, 0.6, 0.7),)).assign(session=REAL)
    m = Monitor(prov_rules(), run_id="fake", provenance=source)              # a declared source
    for row in rows.to_dict("records"):
        m.step(row)
    m.finish()
    recs = m.journal.records
    header = next(r for r in recs if r["record"] == "run_started")
    finished = next(r for r in recs if r["record"] == "run_finished")
    header["source_binding"] = "verified_load"
    header["source"] = {**header["source"], "rows_sha256": finished["input_rows_sha256"], "n_frames": finished["n_rows"]}
    assert el.JournalState.from_records(recs).run_provenance("fake")["status"] == "verified_score_drop"
    return recs


def test_b1_a_fabricated_verified_header_is_only_a_claim(tmp_path):
    recs = _fabricated(tmp_path)
    p = sm.build_summary(recs, run_id="fake", session=REAL).as_dict()["provenance"]
    assert p["label"] == sm.CLAIMED_VERIFIED and p["independently_verified"] is False
    assert p["journal_claim"] == "verified_score_drop" and p["verification"].startswith("not checked")
    md = sm.build_summary(recs, run_id="fake", session=REAL).to_markdown()
    assert "journal-reported verified loading — not independently verified" in md
    assert "verified score drop" not in md.replace("verified score-drop load", "")
    (tmp_path / "x").mkdir()
    other = _drop(tmp_path / "x")
    with pytest.raises(sm.SummaryError, match="do not match the independently loaded"):
        sm.build_summary(recs, run_id="fake", session=REAL, verified_drop=other)


def test_b1_independent_verification_with_the_loaded_drop(tmp_path):
    drop = _drop(tmp_path)
    m = _verified_run(drop)
    plain = sm.build_summary(m.journal.records, run_id="v1", session=VAL[0]).as_dict()["provenance"]
    assert plain["label"] == "SYNTHETIC" and plain["independently_verified"] is False     # still SYNTHETIC
    checked = sm.build_summary(m.journal.records, run_id="v1", session=VAL[0], verified_drop=drop).as_dict()
    assert checked["provenance"]["independently_verified"] is True and checked["provenance"]["label"] == "SYNTHETIC"
    assert "loader-issued score drop" in checked["provenance"]["verification"]
    (tmp_path / "other").mkdir()
    with pytest.raises(sm.SummaryError, match="do not match"):                  # another drop's rows
        sm.build_summary(m.journal.records, run_id="v1", session=VAL[0],
                         verified_drop=_drop(tmp_path / "other", seed=5))


@pytest.mark.parametrize("evidence, match", [
    (lambda d: d.provenance, "unsupported verification evidence"),
    (lambda d: d.provenance.to_json(), "unsupported verification evidence"),
    (lambda d: sm.score_drops.ScoreDrop(d.provenance, d.scores()), "not issued by the loader"),   # hand-built
    (lambda d: dataclasses.replace(d, _table=d.scores().assign(attack=0.0)), "rows were changed"),
])
def test_b1_unsupported_verification_evidence_is_refused(tmp_path, evidence, match):
    drop = _drop(tmp_path)
    m = _verified_run(drop)
    with pytest.raises(sm.SummaryError, match=match):
        sm.build_summary(m.journal.records, run_id="v1", session=VAL[0], verified_drop=evidence(drop))


def test_b1_a_run_without_header_cannot_be_verified(tmp_path):
    drop = _drop(tmp_path)
    m = run(drop.scores(), [R_ALERT], run_id="plain")
    with pytest.raises(sm.SummaryError, match="needs a journal run with a source header"):
        sm.build_summary(m.journal.records, run_id="plain", session=VAL[0], verified_drop=drop)
    real = run(build_scores("SYNTHETIC_x", 5.0).assign(session=REAL), [R_ALERT], run_id="r")
    assert sm.build_summary(real.journal.records, run_id="r", session=REAL).as_dict()["provenance"]["label"] == \
        "unverified source"


def _rule(status="approved", **kw):
    return Rule(**{**{k: v for k, v in R_ALERT.items() if k != "description"}, "description": "", **kw},
                status=status)


@pytest.mark.parametrize("status", ["proposed", "rejected"])
def test_b2_only_approved_rules_are_accepted(status):
    m = run(demo_scores(), [R_ALERT])
    with pytest.raises(sm.SummaryError, match="not approved"):
        sm.build_summary(m.journal.records, run_id="t1", session=DEMO_SESSION, rules=[_rule(status)])


def test_b2_duplicate_rule_ids_are_refused():
    m = run(demo_scores(), [R_ALERT])
    with pytest.raises(sm.SummaryError, match="duplicate rule_id"):
        sm.build_summary(m.journal.records, run_id="t1", session=DEMO_SESSION, rules=[_rule(), _rule()])
    with pytest.raises(sm.SummaryError, match="rules.Rule"):
        sm.build_summary(m.journal.records, run_id="t1", session=DEMO_SESSION, rules=[R_ALERT])


def test_b2_a_zero_event_run_with_a_header_needs_its_exact_rule_set(tmp_path):
    drop = _drop(tmp_path)
    m = Monitor(prov_rules(), run_id="quiet", provenance=drop)                   # R1 only, alert 0.99
    for row in drop.rows():
        m.step({**row, "attack": 0.1})
    m.finish()
    recs = m.journal.records
    r1 = prov_rules()[0]
    extra = Rule("R4", "investigation", 1.0, 0.8, 0.5, "log", "", status="approved")
    with pytest.raises(sm.SummaryError, match="not the run header's complete approved-rule set"):
        sm.build_summary(recs, run_id="quiet", session=VAL[0], rules=[r1, extra])
    with pytest.raises(sm.SummaryError, match="not the run header's complete"):
        sm.build_summary(recs, run_id="quiet", session=VAL[0], rules=[dataclasses.replace(r1, ask_threshold=0.4)])
    d = sm.build_summary(recs, run_id="quiet", session=VAL[0], rules=[r1]).as_dict()
    assert d["events"] == [] and [r["journal_check"] for r in d["rules"]["rules"]] == ["run header"]
    assert "complete rule set" in d["rules"]["source"]
    journal_only = sm.build_summary(recs, run_id="quiet", session=VAL[0]).as_dict()["rules"]
    assert [r["rule_id"] for r in journal_only["rules"]] == ["R1"] and "run header" in journal_only["source"]


def test_b2_descriptions_are_supplied_metadata_and_unfired_rules_unchecked():
    m = run(demo_scores(), [R_ALERT])
    d = sm.build_summary(m.journal.records, run_id="t1", session=DEMO_SESSION,
                         rules=[_rule(description="anything the caller says"),
                                Rule("R9", "mount", 2.0, 0.8, 0.5, "alert", "", status="approved")]).as_dict()
    r1, r9 = d["rules"]["rules"]
    assert r1["description"] == "anything the caller says" and r1["description_source"] == "supplied metadata"
    assert r1["journal_check"] == "fired events" and r9["journal_check"].startswith("not in the journal")
    assert "only rules that fired could be checked" in d["rules"]["source"]
    assert any("Rule descriptions are supplied metadata" in c for c in d["caveats"])
    md = sm.build_summary(m.journal.records, run_id="t1", session=DEMO_SESSION, rules=[_rule(description="x")]).to_markdown()
    assert "description (supplied metadata)" in md and "| fired events | x |" in md


def _answered():
    m = run(demo_scores(), [R_ALERT])
    alert, ask = [e["event_id"] for e in m.journal.events()]
    m.answer(ask, "confirmed")
    return m, alert, ask


REC = {"n_rows": 720, "first_frame": 0, "last_frame": 719, "first_time_s": 0.0, "last_time_s": 719 / 30, "fps": 30.0,
       "source": "replayed score rows"}


@pytest.mark.parametrize("frame, match", [(-1, ">= 0"), (True, "whole replay frame"), ("yesterday", "whole replay frame"),
                                          (280.0, "whole replay frame"), (200, "before the event fired"),
                                          (5000, "outside the recording")])
def test_b3_impossible_answer_positions_are_refused(frame, match):
    m, _, ask = _answered()
    with pytest.raises(sm.SummaryError, match=match):
        summary(m, DEMO_SESSION, answer_positions={ask: frame}, recording=REC)


def test_b3_positions_only_for_answered_asks_of_the_run():
    m, alert, ask = _answered()
    for eid in (alert, "E-other-1"):
        with pytest.raises(sm.SummaryError, match="not an answered ask event"):
            summary(m, DEMO_SESSION, answer_positions={eid: 300})
    with pytest.raises(sm.SummaryError, match="mapping"):
        summary(m, DEMO_SESSION, answer_positions=[(ask, 300)])
    ok = summary(m, DEMO_SESSION, answer_positions={ask: 270}, recording=REC).as_dict()     # the firing frame itself
    assert ok["events"][1]["answer"]["answered_frame"] == 270


@pytest.mark.parametrize("edit, match", [
    ({"n_rows": 1000, "first_frame": 0, "last_frame": 5}, "frame slots"),
    ({"n_rows": True}, "integers"), ({"first_frame": -1}, "integers"), ({"first_frame": 800}, "integers"),
    ({"n_rows": 0}, "integers"), ({"last_time_s": float("nan")}, "finite"), ({"fps": 0}, "fps > 0"),
    ({"first_time_s": 5.0}, "do not match its frames"), ({"last_time_s": 30.0}, "do not match its frames"),
    ({"first_time_s": -1.0}, "finite"),
    ({"source": ""}, "source label"), ({"extra": 1}, "exactly"),
    ({"last_frame": 200, "last_time_s": 200 / 30, "n_rows": 201}, "not a recorded frame"),
])
def test_b3_inconsistent_recordings_are_refused(edit, match):
    m, _, _ = _answered()
    with pytest.raises(sm.SummaryError, match=match):
        summary(m, DEMO_SESSION, recording={**REC, **edit})


def test_b3_valid_recordings_including_gaps():
    m, _, ask = _answered()
    d = summary(m, DEMO_SESSION, recording={**REC, "n_rows": 690}, answer_positions={ask: 300}).as_dict()
    assert d["recording"]["missing_frames"] == 30 and d["recording"]["duration_s"] == pytest.approx(24.0)
    assert summary(m, DEMO_SESSION, recording=REC).as_dict()["recording"]["missing_frames"] == 0


def _warning_entries():
    r = _warned()
    return r, [dict(w) for w in r.end_state().warnings]


def _with(r, entries):
    return sm.build_summary(r.journal.records, run_id=r.run_id, session=r.session, warnings=entries,
                            warning_source=SRC, recording={
                                "n_rows": r.n_frames, "first_frame": r.frames[0], "last_frame": r.frames[-1],
                                "first_time_s": r.times[0], "last_time_s": r.times[-1], "fps": r.fps,
                                "source": "replayed score rows"})


@pytest.mark.parametrize("edit, match", [
    (lambda w: {**w, "end_s": -1.0}, "ends before it starts|>= 0"),
    (lambda w: {**w, "end_s": w["start_s"] - 0.1}, "ends before it starts"),
    (lambda w: {**w, "start_s": float("nan")}, "finite"),
    (lambda w: {**w, "end_s": 999.0}, "outside the recording"),
    (lambda w: {**w, "event_id": ""}, "no identity"),
    (lambda w: {**w, "rule_id": "R1"}, "monitor-event fields"),
    (lambda w: {**w, "evidence": {**w["evidence"], "coordinates_present": "yes"}}, "malformed evidence"),
    (lambda w: {**w, "evidence": {k: v for k, v in w["evidence"].items() if k != "kind"}}, "malformed evidence"),
    (lambda w: {**w, "extra": 1}, "malformed"),
    (lambda w: {**w, "action": "alert"}, "malformed"),
])
def test_b4_backward_or_malformed_warnings_are_refused(edit, match):
    r, entries = _warning_entries()
    with pytest.raises(sm.SummaryError, match=match):
        _with(r, [edit(entries[0])] + entries[1:])


def test_b4_duplicate_warnings_are_refused_not_counted():
    r, entries = _warning_entries()
    with pytest.raises(sm.SummaryError, match="given twice$"):
        _with(r, entries + [copy.deepcopy(entries[0])])                       # an exact duplicate
    with pytest.raises(sm.SummaryError, match="different content"):
        _with(r, entries + [{**entries[0], "end_s": entries[0]["end_s"] + 0.1}])
    with pytest.raises(sm.SummaryError, match="named source"):
        sm.build_summary(r.journal.records, run_id=r.run_id, session=r.session, warnings=entries, warning_source=" ")


def test_b4_valid_multiple_warnings_stay_separate_from_events():
    r, entries = _warning_entries()
    d = _with(r, list(reversed(entries))).as_dict()                            # order of input does not matter
    assert d["tracking_warnings"]["total"] == len(entries) >= 3
    assert [w["warning_id"] for w in d["tracking_warnings"]["warnings"]] == \
        [w["event_id"] for w in sorted(entries, key=lambda w: (w["start_s"], w["event_id"]))]
    assert sum(d["counts"].values()) == len(d["events"])
    assert _with(r, entries).to_json() == _with(r, list(reversed(entries))).to_json()
    assert _with(r, entries).to_markdown() == _with(r, list(reversed(entries))).to_markdown()


def test_b1_only_independent_verification_labels_a_real_source_verified(tmp_path):
    drop = _drop(tmp_path, session=REAL)                                     # a loader-issued, non-SYNTHETIC drop
    m = _verified_run(drop)
    claim = sm.build_summary(m.journal.records, run_id="v1", session=REAL).as_dict()["provenance"]
    assert claim["label"] == sm.CLAIMED_VERIFIED                              # the genuine journal alone: a claim
    checked = sm.build_summary(m.journal.records, run_id="v1", session=REAL, verified_drop=drop).as_dict()
    assert checked["provenance"]["label"] == "verified score drop" and checked["provenance"]["independently_verified"]
    assert not any("SYNTHETIC" in c for c in checked["caveats"])



# ======================================================================
# Codex D4-1 final review: recording timing and the exact frame domain
# ======================================================================

def test_t1_genuine_30_fps_events_with_a_claimed_60_fps_recording_are_refused():
    m = run(demo_scores(), [R_ALERT])                                          # the alert's onset: frame 60 at 2.0 s
    fake = dict(n_rows=720, first_frame=0, last_frame=719, first_time_s=0.0, last_time_s=719 / 60, fps=60.0,
                source="supplied")
    with pytest.raises(sm.SummaryError, match="monitored at 30 fps, the recording claims 60 fps"):
        summary(m, DEMO_SESSION, recording=fake)


def test_t1_a_consistent_period_with_shifted_event_times_is_refused():
    m = run(demo_scores(), [R_ALERT])
    shifted = {**REC, "first_time_s": 1.0, "last_time_s": 1.0 + 719 / 30}      # same fps, another origin
    with pytest.raises(sm.SummaryError, match="onset at frame 60 is at 2 s, but the recording places that frame"):
        summary(m, DEMO_SESSION, recording=shifted)


def test_t1_the_run_header_rate_must_agree(tmp_path):
    drop = _drop(tmp_path)
    m = _verified_run(drop)
    good = {"n_rows": 240, "first_frame": 0, "last_frame": 239, "first_time_s": 0.0, "last_time_s": 239 / 30,
            "fps": 30.0, "source": "replayed score rows"}
    assert summary(m, VAL[0], recording=good).as_dict()["recording"]["fps"] == 30.0
    recs = [r for r in m.journal.records if r["record"] != "event_fired"]      # header only: no event to compare
    with pytest.raises(sm.SummaryError, match="run header samples at 30 fps, the recording claims 60 fps"):
        sm.build_summary(recs[:1], run_id="v1", session=VAL[0],
                         recording={**good, "last_time_s": 239 / 60, "fps": 60.0})


def test_t1_genuine_consistent_recordings_are_accepted():
    m = run(demo_scores(), [R_ALERT])
    d = summary(m, DEMO_SESSION, recording=REC, recorded_frames=range(720)).as_dict()
    assert d["recording"]["frame_domain"] == "exact recorded frames" and d["recording"]["missing_frames"] == 0
    assert summary(m, DEMO_SESSION, recording=REC).as_dict()["recording"]["frame_domain"].startswith("extent only")


def test_t1_nonzero_start_gaps_and_exclusive_closures():
    s = demo_scores()
    s = s[(s["frame"] >= 30) & ((s["frame"] < 125) | (s["frame"] >= 135))].reset_index(drop=True)
    r = rp.build_replay(s, DEMO_SESSION, ruleset(R_ALERT))                     # starts at frame 30 (1.0 s)
    d = sm.summary_from_replay(r).as_dict()
    alert = d["events"][0]                                                    # fired on frame 120, then the gap
    assert alert["fired_frame"] == 120 and alert["closure"]["end_reason"] == "frame_gap"
    assert alert["closure"]["last_frame"] == 124
    assert alert["closure"]["end_s"] == pytest.approx(125 / 30)               # exclusive slot: frame 125 not recorded
    assert d["recording"]["first_frame"] == 30 and d["recording"]["missing_frames"] == 10
    assert d["recording"]["frame_domain"] == "exact recorded frames"
    end = sm.summary_from_replay(rp.build_replay(end_session(), S_END, ruleset(R_ALERT))).as_dict()
    last = end["events"][1]["closure"]
    assert last["end_reason"] == "stream_end" and last["last_frame"] == 299 and last["end_s"] == pytest.approx(10.0)
    m = run(demo_scores(), [R_ALERT])
    recs = m.journal.records                                                  # (each call returns fresh copies)
    closed = next(x for x in recs if x["record"] == "event_closed")
    bad = [x if x is not closed else {**x, "end_s": x["end_s"] - 1 / 30} for x in recs]
    with pytest.raises(sm.SummaryError, match="end_s - start_s does not match bout_frames"):
        sm.build_summary(bad, run_id="t1", session=DEMO_SESSION, recording=REC)      # the journal refuses it
    moved = [x if x is not closed else {**x, "end_s": x["end_s"] + 1 / 30, "bout_frames": x["bout_frames"] + 1}
             for x in recs]
    # the journal ties end_s, bout_frames and first/last frame together, so a closure in the wrong slot never
    # reaches the summary's own end-slot check (kept as defence in depth); a shifted event fails at its onset
    with pytest.raises(sm.SummaryError, match="bout_frames does not match first/last frame"):
        sm.build_summary(moved, run_id="t1", session=DEMO_SESSION, recording=REC)


def test_t2_an_answer_inside_a_known_gap_is_refused():
    s = demo_scores()
    s = s[(s["frame"] < 280) | (s["frame"] >= 290)].reset_index(drop=True)   # frames 280-289 not recorded
    r = rp.build_replay(s, DEMO_SESSION, ruleset(R_ALERT))
    asks = [e["event_id"] for e in r.end_state().events if e["action"] == "ask"]   # the ask fires on frame 270
    r.answer(asks[0], "confirmed", at_frame=300)
    rec = {"n_rows": r.n_frames, "first_frame": r.frames[0], "last_frame": r.frames[-1], "first_time_s": r.times[0],
           "last_time_s": r.times[-1], "fps": r.fps, "source": "replayed score rows"}
    args = dict(run_id=r.run_id, session=r.session, recording=rec)
    with pytest.raises(sm.SummaryError, match="frame 285.*not a recorded frame"):
        sm.build_summary(r.journal.records, answer_positions={asks[0]: 285}, recorded_frames=r.frames, **args)
    ok = sm.build_summary(r.journal.records, answer_positions={asks[0]: 300}, recorded_frames=r.frames, **args)
    ask = next(e for e in ok.as_dict()["events"] if e["action"] == "ask")
    assert ask["answer"] == {"researcher_answer": "confirmed", "answered_frame": 300, "timing": "recorded replay frame"}
    assert "confirmed (frame 300) |" in ok.to_markdown()
    extent = sm.build_summary(r.journal.records, answer_positions={asks[0]: 285}, **args)   # endpoints only
    a = next(e for e in extent.as_dict()["events"] if e["action"] == "ask")["answer"]
    assert a["timing"] == "within the recording extent; exact frame presence not verified"
    assert "(frame 285, presence unverified)" in extent.to_markdown()
    assert extent.as_dict()["recording"]["frame_domain"] == "extent only (exact frame presence not known)"
    via = next(e for e in sm.summary_from_replay(r).as_dict()["events"] if e["action"] == "ask")
    assert via["answer"]["timing"] == "recorded replay frame" and via["answer"]["answered_frame"] == 300


@pytest.mark.parametrize("frames, match", [
    ([0, 2, 1], "strictly increasing"), ([0, 1, True], "strictly increasing"), ([0, 1, 1.5], "strictly increasing"),
    (list(range(719)), "do not match"), (list(range(1, 721)), "do not match"),
])
def test_t2_a_malformed_frame_domain_is_refused(frames, match):
    m = run(demo_scores(), [R_ALERT])
    with pytest.raises(sm.SummaryError, match=match):
        summary(m, DEMO_SESSION, recording=REC, recorded_frames=frames)
    with pytest.raises(sm.SummaryError, match="need the recording"):
        summary(m, DEMO_SESSION, recorded_frames=range(720))


def test_t2_event_frames_must_be_recorded_frames_when_the_domain_is_known():
    m = run(demo_scores(), [R_ALERT])
    holey = [f for f in range(720) if f != 120]                               # the alert fired on frame 120
    with pytest.raises(sm.SummaryError, match="firing frame 120 is not a recorded frame"):
        summary(m, DEMO_SESSION, recording={**REC, "n_rows": 719}, recorded_frames=holey)


def test_t2_untimed_answers_stay_unknown_and_output_is_deterministic():
    m, _, ask = _answered()
    d = summary(m, DEMO_SESSION, recording=REC, recorded_frames=range(720), answer_positions={ask: None})
    assert d.as_dict()["events"][1]["answer"]["timing"] == "unknown"
    again = summary(m, DEMO_SESSION, recording=REC, recorded_frames=list(range(720)), answer_positions={ask: None})
    assert d.to_json() == again.to_json() and d.to_markdown() == again.to_markdown()
