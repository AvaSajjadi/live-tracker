"""Phase 2: Igor's frozen attack rule (evaluate.ATTACK_RULE_V1) on the REAL model B validation scores, through the
existing monitor, evidence journal and Contract v1 investigation tools. Offline: no network, no model call.

    python scripts/validation_run_v1.py ARCHIVE --archive-sha256 HEX --confirm ATTACK_RULE_V1 --out DIR \\
        [--member scores_validation_model_b.csv] [--cache-dir data/raw/calms21_validation] \\
        [--manifest splits/split_manifest.json]

Nothing here is new scientific logic; every step reuses reviewed code:
1. scores      score_drops.load_validation_scores: the validation member only (a test member is refused unread),
               checked against the expected archive SHA-256, SHA256SUMS.txt and the tracked split manifest.
2. pose cache  investigation_data.split_coverage: the verified validation cache must hold every validation session
               of the manifest, each with exactly the score drop's frame count. Missing or misaligned: refused (no
               synthetic substitute).
3. monitor     monitor.Monitor with ATTACK_RULE_V1, approved only by --confirm (the researcher's explicit approval),
               journal bound to the drop; event_log.freeze_run must give a verified_score_drop / complete snapshot.
4. evaluation  evaluate.monitor_event_metrics against the validation annotations of the verified cache.
5. contract v1 InvestigationData.from_classifier (classifier probabilities, pose from the verified cache) and one
               InvestigationRunner: query_events with the rule's ask threshold and minimum duration;
               check_tracking_quality and get_window_features on one pre window chosen by a rule fixed here (the
               first queried bout whose baseline starts at or after 0 s); compare_windows at pre_s 2 s, then the two
               preselected follow-ups (1 s, 5 s), whatever the first result. overlap_v1, quality_v1, the 10-window /
               4-session gate and the 12-call budget are the tools' own, unchanged.
DIR must be new or empty and git-ignored (or outside the repository); nothing is overwritten. Exit 0 when every
step ran and verified, 2 when an input is refused (missing, unverified, misaligned or unapproved, or the monitor and
query_events fired different bouts; the summary says which step), 1 when a step failed otherwise (including any
required Contract v1 call that did not execute correctly).
"verified" means every step executed and every required Contract v1 call returned a well-formed result that agrees
with its request; a compare_windows result of insufficient_evidence is an accepted, explicitly inconclusive outcome
(scientific_outcome), not a failure. The Contract v1 record is written only when every required call was accepted.
--allow-synthetic is for tests only: SYNTHETIC fixture sessions are then
accepted and every label says SYNTHETIC.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from live_tracker import event_log as el  # noqa: E402
from live_tracker import investigation_contracts as ic  # noqa: E402
from live_tracker import investigation_tools_v1 as v1  # noqa: E402
from live_tracker import investigation_windows as iw  # noqa: E402
from live_tracker import score_drops  # noqa: E402
from live_tracker import timebase as tb  # noqa: E402
from live_tracker.evaluate import ATTACK_RULE_V1, monitor_event_metrics  # noqa: E402
from live_tracker.investigation_data import (CacheError, ExtractionRefused, split_coverage,  # noqa: E402
                                             verified_sessions)
from live_tracker.investigation_events import AnnotationEventSource  # noqa: E402
from live_tracker.monitor import Monitor  # noqa: E402
from live_tracker.rules import RuleSet  # noqa: E402

_spec = importlib.util.spec_from_file_location("stage9_smoke", Path(__file__).with_name("stage9_smoke.py"))
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)

RULE_ID = "ATTACK_RULE_V1"
RUN_ID = "validation_run_v1"
INVESTIGATION_ID = "validation_run_v1_attack"
OUTPUT_NAMES = ("events.journal.jsonl", "journal_snapshot.json", "contract_v1_record.json", "run_summary.json")
# fixed before any real value is seen (Contract v1: the grid, feature and windows are chosen in advance)
FEATURE = "dist_center"
PRE_S_ORDER = (2.0, 1.0, 5.0)                 # the first analysis, then the two preselected follow-ups
BASELINE_S, BASELINE_GAP_S = 5.0, 1.0
DATA_LABELS = {"scores": "REAL: Igor's model B classifier probabilities on the CalMS21 validation split",
               "pose": "REAL: CalMS21 validation pose from the verified extraction cache",
               "labels": "REAL: CalMS21 validation annotations (evaluation only; never given to the tools)"}
SYNTHETIC_LABELS = {k: "SYNTHETIC test fixture: not CalMS21, not model output, never evidence" for k in DATA_LABELS}


class RunRefused(RuntimeError):
    """An input is missing, unverified or inconsistent; nothing is substituted."""


def _step(summary, name, fn):
    try:
        out = fn()
    except Exception as exc:
        summary["steps"].append({"step": name, "status": "failed", "error": type(exc).__name__,
                                 "message": str(exc)[:500], "where": traceback.extract_tb(exc.__traceback__)[-1].name})
        raise
    summary["steps"].append({"step": name, "status": "ok"})
    return out


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(out):
    out = smoke.ignored_output_dir(out)
    if out.exists() and (not out.is_dir() or out.is_symlink()):
        raise RunRefused(f"{out} is not a real directory")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = [n for n in OUTPUT_NAMES if (out / n).exists() or (out / n).is_symlink()]
    if existing:
        raise RunRefused(f"{out} already contains {existing}; choose a new directory (nothing is overwritten)")
    return out


# ---- 1-2. inputs ----

def load_scores(args, manifest):
    drop = score_drops.load_validation_scores(args.archive, args.member, manifest_path=manifest,
                                              expected_archive_sha256=args.archive_sha256)
    table = drop.scores()
    frames = table.groupby("session", sort=True)["frame"].size().astype(int).to_dict()
    synthetic = any("SYNTHETIC" in sid for sid in frames)
    if synthetic and not args.allow_synthetic:
        raise RunRefused("the score drop holds SYNTHETIC sessions; this run is for real validation data")
    return drop, table, {"provenance": drop.provenance.to_json(), "rows_sha256": score_drops.trusted_rows_sha256(drop),
                         "n_sessions": len(frames), "n_frames": int(sum(frames.values())),
                         "frames_per_session": frames, "synthetic": synthetic}


def check_pose_cache(cache, manifest, score_frames, label):
    cov = split_coverage(cache, manifest)                  # full verification; refuses on any mismatch
    if not cov["complete"]:
        raise RunRefused(f"the verified validation pose cache is incomplete (missing {cov['missing']}); build it with "
                         "`python -m live_tracker.investigation_data --all` (no synthetic substitute)")
    _, index = verified_sessions(cache, manifest)
    cache_frames = {sid: e["frames"] for sid, e in index["sessions"].items()}
    if cache_frames != score_frames:
        bad = sorted(s for s in set(cache_frames) | set(score_frames) if cache_frames.get(s) != score_frames.get(s))
        raise RunRefused(f"score drop and pose cache disagree on sessions / frame counts: {bad}")
    return {"cache_dir": str(cache), "complete": True, "n_sessions": len(cache_frames),
            "n_frames": int(sum(cache_frames.values())), "source": index["source"], "manifest": index["manifest"],
            "alignment": "every session: score frames == cache frames", "label": label}


# ---- 3. monitor + journal ----

def approved_rule(confirm):
    rules = RuleSet()
    rule = rules.add({"rule_id": RULE_ID, **ATTACK_RULE_V1,
                      "description": "Igor's frozen demo attack rule (evaluate.ATTACK_RULE_V1)"})
    if confirm != RULE_ID:                                  # rules.add leaves it "proposed"
        raise RunRefused(f"{RULE_ID} is only proposed; approve it explicitly with --confirm {RULE_ID}")
    rules.approve(RULE_ID)
    return rules.active()[0]


def run_monitor(drop, rule, out):
    journal = out / OUTPUT_NAMES[0]
    smoke.create_exclusive(journal, b"")
    m = Monitor([rule], journal_path=journal, run_id=RUN_ID, provenance=drop)
    for row in drop.rows():
        m.step(row)
    events = m.finish()
    snap = el.freeze_run(journal, RUN_ID)
    snap = el.load_snapshot(snap.canonical, snap.sha256)
    if snap.completion != "complete" or snap.provenance_status != "verified_score_drop" or \
            snap.n_events != len(events):
        raise RunRefused(f"frozen run is {snap.provenance_status} / {snap.completion} with {snap.n_events} events")
    smoke.create_exclusive(out / OUTPUT_NAMES[1], snap.canonical if isinstance(snap.canonical, bytes)
                           else snap.canonical.encode("utf-8"))
    return events, snap


def describe_events(events, table, fps):
    rows = [{"event_id": e["event_id"], "session": e["session"], "action": e["action"],
             "start_s": e["start_s"], "fired_s": e["evidence"]["fired_s"], "end_s": e["end_s"],
             "duration_s": round(e["end_s"] - e["start_s"], 6), "end_reason": e["evidence"]["end_reason"],
             "confidence": e["confidence"]} for e in events]
    per_session, interruptions = {}, {}
    for sid, g in table.groupby("session", sort=True):
        ok = g.sort_values("frame")["tracking_ok"].to_numpy(bool)
        lost_runs = int(np.sum(~ok & np.r_[True, ok[:-1]]))
        interruptions[sid] = {"tracking_lost_frames": int((~ok).sum()), "tracking_lost_runs": lost_runs}
    for r in rows:
        per_session.setdefault(r["session"], 0)
        per_session[r["session"]] += 1
    reasons = pd.Series([r["end_reason"] for r in rows]).value_counts().to_dict() if rows else {}
    actions = pd.Series([r["action"] for r in rows]).value_counts().to_dict() if rows else {}
    dur = np.array([r["duration_s"] for r in rows])
    return {"n_events": len(rows), "events_per_session": per_session, "end_reasons": reasons, "actions": actions,
            "duration_s": ({"min": float(dur.min()), "median": float(np.median(dur)), "max": float(dur.max())}
                           if len(dur) else None),
            "events_ended_by_tracking_loss": int(reasons.get("tracking_lost", 0)),
            "tracking_interruptions": interruptions, "fps": fps, "events": rows}


# ---- 4. evaluation (validation annotations) ----

def validation_labels(ann, sessions):
    snap = ann.snapshot(sessions)
    parts = []
    for sid in sorted(snap.sessions):
        pose = iw.load_session(sid, snap.sessions[sid])
        parts.append(pd.DataFrame({"session": sid, "frame": np.arange(pose.n_frames), "label": pose.labels}))
    snap.finalize()
    return pd.concat(parts, ignore_index=True)


def evaluate_events(table, labels, rule, label):
    metrics, bouts = monitor_event_metrics(table, labels, rule)
    per = bouts.assign(detected=bouts["start_offset_s"].notna()).groupby("session").agg(
        true_bouts=("true_start", "size"), detected=("detected", "sum")).astype(int).reset_index()
    return {"metrics": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()},
            "true_bouts_per_session": per.to_dict("records"), "label": label}


# ---- 5. Contract v1 tools ----

class ToolFailure(RuntimeError):
    """A required Contract v1 call did not execute correctly (refused, missing, malformed or contradictory): the run
    is not verified. An insufficient_evidence comparison is NOT a failure (it executed; it is inconclusive)."""


def _event_ordinal(event_id):
    """n of an event id E-<run id>-n: the event's order of firing within its run."""
    tail = str(event_id).rsplit("-", 1)[-1]
    if not tail.isdigit():
        raise RunRefused(f"malformed event id {str(event_id)[:80]!r}")
    return int(tail)


def bout_identities(items, start, fired, end, reason):
    """The canonical identity of every fired bout, in firing order: (ordinal, session, start frame, fired frame,
    end frame, end reason), every boundary converted with the 30 fps rule (timebase.boundary_frame)."""
    return [(_event_ordinal(x["event_id"]), x["session"], tb.boundary_frame(start(x)), tb.boundary_frame(fired(x)),
             tb.boundary_frame(end(x)), reason(x)) for x in items]


def check_same_bouts(events, bouts):
    """The monitor's complete fired-bout set must be exactly the query's (identity and frame boundaries, in order);
    nothing is reconciled, dropped or added."""
    mon = bout_identities(events, lambda e: e["start_s"], lambda e: e["evidence"]["fired_s"], lambda e: e["end_s"],
                          lambda e: e["evidence"]["end_reason"])
    qry = bout_identities(bouts, lambda b: b["start_s"], lambda b: b["fired_s"], lambda b: b["end_s"],
                          lambda b: b["end_reason"])
    if mon != qry:
        n = min(len(mon), len(qry))
        where = next((k for k in range(n) if mon[k] != qry[k]), n)
        raise RunRefused(f"the monitor fired {len(mon)} bouts and query_events {len(qry)}; they first differ at "
                         f"position {where} (monitor {mon[where] if where < len(mon) else None}, query "
                         f"{qry[where] if where < len(qry) else None}); nothing is reconciled")


def _require(tool, r, allowed, expect):
    """Accept a required call only if it executed (status in `allowed`), is a well-formed Contract v1 response and
    agrees with what was requested (`expect`: field -> value)."""
    if not isinstance(r, dict):
        raise ToolFailure(f"{tool} returned no result object ({type(r).__name__})")
    if r.get("status") not in allowed:
        raise ToolFailure(f"{tool} did not execute: status {r.get('status')!r}, code {r.get('code')!r}: "
                          f"{str(r.get('message'))[:200]}")
    if r.get("association_only") is not True or r.get("contract_version") != v1.CONTRACT_VERSION:
        raise ToolFailure(f"{tool} is not a Contract v1 association-only response")
    bad = [k for k, v in expect.items() if k not in r or r[k] != v]
    if bad:
        raise ToolFailure(f"{tool} contradicts its request or the fixed policies in {', '.join(bad)}")
    return r


def _require_query(q, rule, sessions):
    query = {"session": sorted(sessions), "behavior": rule.behavior, "min_duration_s": float(rule.min_duration_s),
             "threshold": float(rule.ask_threshold)}
    _require("query_events", q, ("ok",), {"query": query, "behavior": rule.behavior,
                                          "min_duration_s": float(rule.min_duration_s),
                                          "threshold": float(rule.ask_threshold)})
    if not isinstance(q.get("bouts"), list) or q.get("n_bouts") != len(q["bouts"]) or \
            not isinstance(q.get("query_run_id"), str):
        raise ToolFailure("query_events returned malformed bouts, counts or run id")


_COMPARE_FN = v1.compare_windows                   # the adapter's own function, bound at import (not the dispatch)


def _require_compare(c, pre_s, data, request):
    """Accept a compare_windows result only if it is exactly what Contract v1 computes for this request, and its
    sufficiency is genuine: an "ok" must meet the approved gate (>= 10 used windows from >= 4 distinct sessions)
    with per-session records that are the used bouts of the verified query; an insufficient_evidence result must
    carry a genuinely unmet condition and no effect. Never trusts the declared status or a finite effect alone."""
    name = f"compare_windows (pre_s {pre_s})"
    bouts = request["bouts"]
    gate = {"min_n_windows": v1.DEFAULT_MIN_N_WINDOWS, "min_n_sessions": v1.DEFAULT_MIN_N_SESSIONS}
    expect = {"pre_s": pre_s, "feature": FEATURE, "n_input": len(bouts), "baseline_s": BASELINE_S,
              "baseline_gap_s": BASELINE_GAP_S, "quality_policy": v1.DEFAULT_QUALITY_POLICY,
              "overlap_policy": v1.DEFAULT_OVERLAP_POLICY,
              "evidence_gate": {**gate, "overridden": False, "override_reason": None}}
    _require(name, c, ("ok", "insufficient_evidence"), expect)

    def bad(why):
        raise ToolFailure(f"{name} is not acceptable: {why}")

    # exclusions: each a distinct bout of the verified query, with a known reason; the rest are the used bouts
    excluded = c.get("excluded")
    if not isinstance(excluded, list) or not all(isinstance(e, dict) for e in excluded):
        bad("malformed exclusions")
    keys = [(b["session"], b["start_s"]) for b in bouts]
    ex_keys = [(e.get("session"), e.get("start_s")) for e in excluded]
    if len(set(ex_keys)) != len(ex_keys) or not set(ex_keys) <= set(keys) or \
            any(e.get("why") not in v1.EXCLUSION_REASONS for e in excluded):
        bad("exclusions that are not distinct bouts of the verified query with a known reason")
    used = [k for k in keys if k not in set(ex_keys)]
    used_per_session = {}
    for sid, _ in used:
        used_per_session[sid] = used_per_session.get(sid, 0) + 1
    counts = {"n_used": len(used), "n_windows": len(used), "n_bouts": len(used), "n_baseline_windows": len(used),
              "n_excluded": len(excluded), "n_sessions": len(used_per_session),
              "n_sessions_input": len({b["session"] for b in bouts})}
    wrong = [k for k, v in counts.items() if c.get(k) != v]
    if wrong:
        bad(f"counts disagree with the verified bouts and exclusions in {', '.join(wrong)}")
    n_w, n_s = counts["n_windows"], counts["n_sessions"]
    if c["status"] == "ok":
        if n_w < gate["min_n_windows"] or n_s < gate["min_n_sessions"]:
            bad(f"declared ok with {n_w} usable windows from {n_s} sessions, below the approved gate "
                f"({gate['min_n_windows']} windows, {gate['min_n_sessions']} sessions)")
        per = c.get("per_session")
        if not isinstance(per, list) or not all(isinstance(p, dict) for p in per):
            bad("malformed per_session")
        sids = [p.get("session") for p in per]
        if len(set(sids)) != len(sids):
            bad("duplicated contributing sessions")
        if {sid: p.get("n_bouts") for sid, p in zip(sids, per)} != used_per_session:
            bad("per_session sessions or counts are not the used bouts of the verified query")
        diffs = [p.get("paired_diff") for p in per]
        if not all(isinstance(d, float) and np.isfinite(d) for d in diffs) or \
                c.get("mean_diff") != statistics.median(diffs):
            bad("the effect is not the median of finite per-session paired differences")
    else:
        if any(k in c for k in ("mean_diff", "per_session", "pre_mean", "baseline_mean")):
            bad("an insufficient_evidence result carries an effect")
        unmet = ("too_few_bouts", "min_n_windows", len(bouts)) if len(bouts) < gate["min_n_windows"] else \
            ("too_few_usable_windows", "min_n_windows", n_w) if n_w < gate["min_n_windows"] else \
            ("too_few_usable_windows", "min_n_sessions", n_s) if n_s < gate["min_n_sessions"] else None
        if unmet is None:
            bad(f"declared insufficient although the gate is met ({n_w} windows, {n_s} sessions)")
        reason, failed, have = unmet
        if (c.get("reason"), c.get("failed_gate"), c.get("n_available"), c.get("n_required")) != \
                (reason, failed, have, gate[failed]):
            bad(f"the insufficiency is not the unmet condition ({reason}, {failed}: {have} < {gate[failed]})")
    if _COMPARE_FN(data, copy.deepcopy(request)) != c:           # the adapter's own computation, recomputed
        bad("it differs from what Contract v1 computes for the same request")
    return c


def contract_v1(drop, ann, policy, rule, events, synthetic):
    """The required calls, each accepted only after its result is checked: query_events; check_tracking_quality and
    get_window_features when a bout qualifies as the probe; compare_windows at every preselected pre_s when there is
    a bout. The monitor and the query must have fired exactly the same bouts before any window is measured."""
    data = v1.InvestigationData.from_classifier(drop, ann, policy)
    if (data.data_source == "synthetic") != synthetic or (not synthetic and data.data_source != "classifier"):
        raise RunRefused(f"the tools label the data {data.data_source!r}, which contradicts the run's labelling")
    runner = v1.InvestigationRunner(data, INVESTIGATION_ID)
    q = runner.call("query_events", {"session": list(data.sessions), "behavior": rule.behavior,
                                     "min_duration_s": rule.min_duration_s, "threshold": rule.ask_threshold})
    _require_query(q, rule, data.sessions)
    check_same_bouts(events, q["bouts"])                    # before anything else runs
    out = {"data_source": data.data_source, "derived_from": data.derived_from, "fingerprint": data.fingerprint,
           "query_matches_monitor_bouts": True}
    probe = next((b for b in q["bouts"] if b["start_s"] - 2.0 - BASELINE_GAP_S - BASELINE_S >= 0), None)
    if probe is not None:
        win = {"session": probe["session"], "t0_s": probe["start_s"] - 2.0, "t1_s": probe["start_s"]}
        out["probe_bout"] = {k: probe[k] for k in ("event_id", "session", "start_s")}
        out["check_tracking_quality"] = _require("check_tracking_quality", runner.call(
            "check_tracking_quality", dict(win)), ("ok",), {**win, "quality_policy": v1.DEFAULT_QUALITY_POLICY})
        if out["check_tracking_quality"].get("verdict") not in ("reliable", "unreliable"):
            raise ToolFailure("check_tracking_quality returned no verdict")
        out["get_window_features"] = _require("get_window_features", runner.call(
            "get_window_features", {**win, "cols": [FEATURE]}), ("ok",), win)
        if not isinstance(out["get_window_features"].get("summary", {}).get(FEATURE), dict):
            raise ToolFailure(f"get_window_features returned no summary of {FEATURE}")
    out["compare_windows"] = []
    for pre_s in PRE_S_ORDER if q["bouts"] else ():
        request = {"query_run_id": q["query_run_id"], "query": q["query"], "bouts": q["bouts"], "pre_s": pre_s,
                   "baseline_s": BASELINE_S, "baseline_gap_s": BASELINE_GAP_S, "feature": FEATURE}
        out["compare_windows"].append(_require_compare(runner.call("compare_windows", copy.deepcopy(request)),
                                                       pre_s, data, request))
    out["query_events"] = {k: q[k] for k in q if k != "bouts"}
    out["scientific_outcome"] = [
        {"pre_s": c["pre_s"], "status": c["status"],
         "outcome": "evidence gate met (association only)" if c["status"] == "ok" else
                    f"inconclusive: insufficient_evidence ({c['reason']}, gate {c['failed_gate']})"}
        for c in out["compare_windows"]]
    return out, runner.record()


def compare_summary(c):
    keep = ("status", "reason", "failed_gate", "n_available", "n_required", "pre_s", "feature", "n_input", "n_used",
            "n_excluded", "n_sessions", "n_sessions_input", "n_windows", "quality_policy", "overlap_policy",
            "evidence_gate", "mean_diff", "pre_mean", "baseline_mean", "per_session", "code", "message")
    out = {k: c[k] for k in keep if k in c}
    if "excluded" in c:
        out["exclusions_by_reason"] = pd.Series([e["why"] for e in c["excluded"]]).value_counts().to_dict() \
            if c["excluded"] else {}
    return out


def run(args):
    archive, cache, manifest = Path(args.archive).expanduser(), Path(args.cache_dir), Path(args.manifest)
    if not archive.is_file():
        raise RunRefused(f"score archive {archive} not found")
    if not manifest.is_file():
        raise RunRefused(f"split manifest {manifest} not found")
    if not (cache / "extraction_index.json").is_file():
        raise RunRefused(f"no verified validation pose cache at {cache}; build it with "
                         "`python -m live_tracker.investigation_data --all` (no synthetic substitute)")
    out = prepare(args.out)
    args.archive = archive
    summary = {"mode": "Phase 2 validation run (offline; no network, no model call)", "steps": [],
               "test_split": "not read: validation member and validation cache only",
               "rule": {"rule_id": RULE_ID, **ATTACK_RULE_V1, "source": "evaluate.ATTACK_RULE_V1"}}
    status = "verified"
    try:
        drop, table, summary["scores"] = _step(summary, "scores", lambda: load_scores(args, manifest))
        synthetic = summary["scores"]["synthetic"]
        labels_for = SYNTHETIC_LABELS if synthetic else DATA_LABELS
        summary["data_labels"], summary["scores"]["label"] = labels_for, labels_for["scores"]
        summary["pose_cache"] = _step(summary, "pose_cache", lambda: check_pose_cache(
            cache, manifest, summary["scores"]["frames_per_session"], labels_for["pose"]))
        rule = _step(summary, "approval", lambda: approved_rule(args.confirm))
        summary["rule"]["approval"] = f"approved for this run by the operator (--confirm {RULE_ID})"
        events, snap = _step(summary, "monitor", lambda: run_monitor(drop, rule, out))
        summary["monitor"] = {**describe_events(events, table, tb.FPS_INT),
                              "journal_snapshot_sha256": snap.sha256, "provenance_status": snap.provenance_status,
                              "completion": snap.completion}
        policy = ic.EvidencePolicy.from_manifest(manifest, allow_synthetic=synthetic)
        ann = AnnotationEventSource(cache, manifest, policy)
        sessions = sorted(summary["scores"]["frames_per_session"])
        labels = _step(summary, "labels", lambda: validation_labels(ann, sessions))
        summary["evaluation"] = _step(summary, "evaluation", lambda: evaluate_events(table, labels, rule,
                                                                                      labels_for["labels"]))
        v1_out, record = _step(summary, "contract_v1", lambda: contract_v1(drop, ann, policy, rule, events,
                                                                           synthetic))
        rec_text = json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
        smoke.create_exclusive(out / OUTPUT_NAMES[2], rec_text.encode("utf-8"))
        summary["contract_v1"] = {**{k: v for k, v in v1_out.items() if k != "compare_windows"},
                                  "compare_windows": [compare_summary(c) for c in v1_out.get("compare_windows", [])],
                                  "record_sha256": _sha(rec_text.encode("utf-8")), "calls_used": record["calls_used"],
                                  "max_tool_calls": record["max_tool_calls"]}
    except (RunRefused, smoke.SmokeError, score_drops.ScoreDropError, CacheError, ExtractionRefused,
            ic.ForbiddenEvidence):
        status = "refused"                                  # an input is missing or does not verify
    except Exception:                                       # ToolFailure included: a required call did not execute
        status = "failed"
    summary["status"] = status
    if status != "verified":
        summary["investigation_evidence"] = ("none accepted: any retained journal and snapshot are monitoring records "
                                             "only, not investigation evidence, and no Contract v1 record was written")
    summary["outputs"] = {n: _sha((out / n).read_bytes()) for n in OUTPUT_NAMES[:3] if (out / n).is_file()}
    smoke.create_exclusive(out / OUTPUT_NAMES[3],
                           (json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8"))
    summary["output_dir"] = str(out)
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="the validation score-drop ZIP (read in place, never extracted)")
    ap.add_argument("--archive-sha256", required=True, help="the archive's expected SHA-256 (checked before reading)")
    ap.add_argument("--confirm", default=None, help=f"approve the rule for this run: --confirm {RULE_ID}")
    ap.add_argument("--out", required=True, help="a new, git-ignored (or outside the repository) directory")
    ap.add_argument("--member", default="scores_validation_model_b.csv")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "raw" / "calms21_validation"))
    ap.add_argument("--manifest", default=str(ROOT / "splits" / "split_manifest.json"))
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="tests only: accept SYNTHETIC fixture sessions (labelled synthetic, never real evidence)")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        summary = run(args)
    except (RunRefused, smoke.SmokeError, score_drops.ScoreDropError) as exc:
        print(f"refused ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary[k] for k in ("status", "steps", "output_dir")}, indent=2, default=str))
    return {"verified": 0, "refused": 2}.get(summary["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
