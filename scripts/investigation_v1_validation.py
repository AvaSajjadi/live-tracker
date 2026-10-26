"""Phase 3B.1: one Stage 9 investigation (investigation_agent_v1, Contract v1 tools) on the REAL CalMS21 validation
split, with a deterministic OFFLINE investigator in place of a language model. No network, no API key, no cost.

    python scripts/investigation_v1_validation.py ~/Downloads/igor_scores_validation_model_b_repacked.zip \\
        --archive-sha256 HEX --out outputs/investigation_v1/<new> [--validation-run run5] \\
        [--question TEXT] [--member scores_validation_model_b.csv] [--cache-dir DIR] [--manifest PATH]

Every step reuses reviewed code; nothing here is new scientific logic:
1. scores      validation_run_v1.load_scores (score_drops.load_validation_scores): the validation member only (a test
               member is refused unread), the expected archive SHA-256, SHA256SUMS.txt and the tracked split manifest.
2. pose cache  validation_run_v1.check_pose_cache: the verified validation cache holds every validation session of
               the manifest, each with exactly the score drop's frame count. Missing or misaligned: refused.
3. data        InvestigationData.from_classifier (Igor's model B probabilities, pose from the verified cache; the
               annotations are never given to the tools). A real run must be the expected scope (EXPECTED_SCOPE:
               14 sessions, 94,305 frames), labelled "classifier", over exactly the manifest's validation sessions.
4. compatible  with --validation-run NAME: the recorded Phase 2 run (validation_run_view.load_run, checked for
               consistency) must name the same score drop (archive, member, rows SHA-256, frames per session) and
               the same Contract v1 data fingerprint; afterwards (before any report is written) the
               investigation's initial query_events must be exactly the run's initial query, request and result
               (fail closed), every other call with a recorded request must give the recorded result, and calls
               the run never made are listed as unmatched, never as equivalent (check_query_against_run).
5. investigate investigation_agent_v1.investigate_v1 with OfflineInvestigatorV1 (below) and the default Budget: the
               loop, the runner, the 12-call budget, the follow-up limits, quality_v1, overlap_v1 and the 10-window /
               4-session gate are the agent's and the tools' own, unchanged.
6. report      investigation_report.json (InvestigationReportV1.to_json), re-read from disk and checked with
               verify_report_v1, and investigation_report.md rendered from it by render_markdown_v1; then the
               report's own data, provenance and split are checked against the inputs. run.json records the inputs,
               every check and the files' SHA-256.

The offline investigator is NOT a language model: it is a fixed decision procedure that reads only the tool results
it is sent (as a model would) and chooses each call from them. Its findings are the agent's structured, verified
finding; nothing it does is a model judgement, and the report says "LLM-planned" only because that is the agent's
canonical disclosure (run.json states that no model was used).

DIR must be new or empty and git-ignored (or outside the repository); nothing is overwritten. Exit 0 when the
investigation completed and every check passed, 2 when an input is refused (missing, unverified, misaligned, out
of scope or incompatible; nothing is substituted), 1 when a step failed otherwise. --allow-synthetic is for tests
only: SYNTHETIC fixture sessions are then accepted and labelled synthetic everywhere, never as real evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker import investigation_agent as ia  # noqa: E402
from live_tracker import investigation_agent_v1 as av1  # noqa: E402
from live_tracker import investigation_contracts as ic  # noqa: E402
from live_tracker import investigation_tools_v1 as v1  # noqa: E402
from live_tracker import score_drops  # noqa: E402
from live_tracker import validation_run_view as vrv  # noqa: E402
from live_tracker.evaluate import ATTACK_RULE_V1  # noqa: E402
from live_tracker.investigation_data import CacheError, ExtractionRefused, load_manifest  # noqa: E402
from live_tracker.investigation_events import AnnotationEventSource  # noqa: E402

_spec = importlib.util.spec_from_file_location("validation_run_v1", Path(__file__).with_name("validation_run_v1.py"))
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)
smoke = vr.smoke

EXPECTED_SCOPE = {"n_sessions": 14, "n_frames": 94_305}      # the CalMS21 validation split of the tracked manifest
OUTPUT_NAMES = ("investigation_report.json", "investigation_report.md", "run.json")
INVESTIGATION_ID = "investigation_v1_validation"
DEFAULT_QUESTION = "Does the distance between the mice change before predicted attack bouts?"
OFFLINE_MODEL = "offline-deterministic-investigator-v1"
# the offline investigator's fixed choices, made before any real value is seen
FEATURE, BASELINE_S, BASELINE_GAP_S = "dist_center", 5.0, 1.0
PRE_S_ORDER = (2, 1, 5)
RunRefused = vr.RunRefused
REFUSALS = (RunRefused, smoke.SmokeError, score_drops.ScoreDropError, CacheError, ExtractionRefused,
            ic.ForbiddenEvidence, vrv.RunUnavailable)


# ---- the deterministic offline investigator (not a language model) ----

def _transcript(messages):
    """[(tool, input, result, is_error)] of every call the investigator was answered, in order."""
    calls, out = {}, []
    for m in messages:
        if not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                calls[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in calls:
                out.append((*calls[b["tool_use_id"]], json.loads(b["content"]), bool(b.get("is_error"))))
    return out


class OfflineInvestigatorV1:
    """A fixed, deterministic decision procedure in the place of the model (investigation_agent.ModelReply over the
    v1 tools). It sees only what a model would see (the transcript of tool results) and chooses its next call from
    it: the predicted attack bouts (ATTACK_RULE_V1's ask threshold and minimum duration); tracking quality and pair
    features on the pre window of the first shown bout whose baseline starts at or after 0 s; compare_windows at
    pre_s 2 s and, only while the result is insufficient_evidence, the follow-ups 1 s then 5 s; then a structured
    finish citing the primary comparison. On any refusal it stops (ModelUnavailable): the report is incomplete."""

    model = OFFLINE_MODEL

    def __init__(self):
        self.turns = 0

    def respond(self, system, tools, messages):
        self.turns += 1
        name, args = self.decide(_transcript(messages))
        return ia.ModelReply(content=[{"type": "tool_use", "id": f"offline_{self.turns}", "name": name,
                                       "input": args}],
                             stop_reason="tool_use", input_tokens=0, output_tokens=0, model=OFFLINE_MODEL)

    def decide(self, done):
        if any(err for *_, err in done):
            raise ia.ModelUnavailable("offline_investigator_stops_on_a_refusal")

        def of(tool):
            return [r for t, _, r, _ in done if t == tool]
        queries, quals, feats, comps = (of(t) for t in ("query_events", "check_tracking_quality",
                                                         "get_window_features", "compare_windows"))
        if not queries:
            return "query_events", {"behavior": ATTACK_RULE_V1["behavior"],
                                    "min_duration_s": ATTACK_RULE_V1["min_duration_s"],
                                    "threshold": ATTACK_RULE_V1["ask_threshold"]}
        q = queries[0]
        if not q.get("n_bouts"):
            return av1.FINISH, {"finding": "inconclusive",
                                "observations": [{"evidence_id": q["evidence_id"], "field": "n_bouts"}]}
        probe = next((b for b in q.get("bouts", []) if b["start_s"] - 2.0 - BASELINE_GAP_S - BASELINE_S >= 0), None)
        window = None if probe is None else {"session": probe["session"], "t0_s": probe["start_s"] - 2.0,
                                             "t1_s": probe["start_s"]}
        if window and not quals:
            return "check_tracking_quality", window
        if window and not feats:
            return "get_window_features", {**window, "cols": [FEATURE]}
        tried = [c["pre_s"] for c in comps]
        nxt = next((p for p in PRE_S_ORDER if float(p) not in tried), None)
        if (not comps or comps[-1]["status"] == "insufficient_evidence") and nxt is not None:
            return "compare_windows", {"query_from": {"evidence_id": q["evidence_id"]}, "pre_s": nxt,
                                       "baseline_s": BASELINE_S, "baseline_gap_s": BASELINE_GAP_S,
                                       "feature": FEATURE}
        extra = [{"evidence_id": quals[0]["evidence_id"], "field": "verdict"}] if quals else []
        c = comps[-1]
        if c["status"] == "ok":
            return av1.FINISH, {"finding": "effect_described", "primary_evidence_id": c["evidence_id"],
                                "observations": [{"evidence_id": c["evidence_id"], "field": f}
                                                 for f in ("mean_diff", "n_windows", "n_sessions")] + extra}
        return av1.FINISH, {"finding": "insufficient_evidence", "primary_evidence_id": c["evidence_id"],
                            "observations": [{"evidence_id": c["evidence_id"], "field": f}
                                             for f in ("reason", "n_windows", "n_sessions")] + extra}


# ---- inputs ----

def load_data(args, manifest, cache):
    """The verified validation scores and pose cache, as InvestigationData (refused otherwise; no substitute)."""
    drop, _, scores = vr.load_scores(args, manifest)
    synthetic = scores["synthetic"]
    pose = vr.check_pose_cache(cache, manifest, scores["frames_per_session"],
                               (vr.SYNTHETIC_LABELS if synthetic else vr.DATA_LABELS)["pose"])
    policy = ic.EvidencePolicy.from_manifest(manifest, allow_synthetic=synthetic)
    data = v1.InvestigationData.from_classifier(drop, AnnotationEventSource(cache, manifest, policy), policy)
    check_scope(data, scores, load_manifest(manifest)[0]["validation"], synthetic)
    return data, scores, pose


def check_scope(data, scores, validation_sessions, synthetic):
    """The data is exactly the manifest's validation split, labelled as what it is, and (real) the expected scope."""
    if (data.data_source == "synthetic") != synthetic or (not synthetic and (
            data.data_source, data.derived_from) != ("classifier", "classifier_probabilities")):
        raise RunRefused(f"the tools label the data {data.data_source!r} / {data.derived_from!r}, which contradicts "
                         "the score drop")
    if list(data.sessions) != sorted(validation_sessions) or sorted(scores["frames_per_session"]) != list(
            data.sessions):
        raise RunRefused("the investigation's sessions are not exactly the manifest's validation sessions")
    frames = sum(data.n_frames(s) for s in data.sessions)
    if frames != scores["n_frames"]:
        raise RunRefused(f"the pose has {frames} frames, the score drop {scores['n_frames']}")
    scope = {"n_sessions": len(data.sessions), "n_frames": frames}
    if not synthetic and scope != EXPECTED_SCOPE:
        raise RunRefused(f"the validation data is {scope}, not the expected scope {EXPECTED_SCOPE}")
    return scope


def check_validation_run(name, root, manifest, data, scores, synthetic):
    """A recorded Phase 2 run (consistency-checked by validation_run_view.load_run) on the same score drop and data."""
    run = vrv.load_run(name, root=root, manifest_path=manifest, allow_synthetic=synthetic)
    s = run.summary
    keys = ("archive_sha256", "member", "member_sha256", "rows_sha256", "split", "n_sessions", "n_frames")
    mine, theirs = scores["provenance"], s["scores"]["provenance"]
    wrong = [k for k in keys if mine.get(k) != theirs.get(k)]
    for k in ("rows_sha256", "n_sessions", "n_frames", "frames_per_session", "synthetic"):
        if s["scores"].get(k) != scores[k]:
            wrong.append(f"scores.{k}")
    if s["contract_v1"]["fingerprint"] != data.fingerprint:
        wrong.append("contract_v1 data fingerprint")
    if wrong:
        raise RunRefused(f"validation run {name!r} is not on the same data ({', '.join(wrong)})")
    return run


def check_query_against_run(doc, run):
    """Compatibility with a recorded run, fail closed. The investigation's initial query (its first query_events
    evidence) must have exactly the request of the run's initial query (its first call) and exactly its result
    (the same query_run_id and bouts): otherwise nothing is compared and the run is refused. Every other call
    with exactly the request of a recorded call must have exactly that call's result (identical_results); a call
    the run never made is listed as unmatched (additional evidence of this investigation, never claimed as
    equivalent to the run)."""
    attempts = run.record["attempts"]
    initial = attempts[0] if attempts else None
    if not initial or initial.get("tool") != "query_events" or initial.get("status") != "ok":
        raise RunRefused(f"validation run {run.name!r} has no executed initial query_events to compare with")
    first = next((e for e in doc["evidence"] if e["tool"] == "query_events"), None)
    if first is None or first["request"] != initial["request"]:
        raise RunRefused(f"the investigation's initial query_events is not the request of validation run "
                         f"{run.name!r} (no run equivalence can be established)")
    if first["result"] != initial["result"]:
        raise RunRefused(f"the investigation's initial query_events disagrees with validation run {run.name!r}'s "
                         "result for the same request")
    recorded = [a for a in attempts if a.get("status") != "error"]
    matched, unmatched = [], []
    for e in doc["evidence"]:
        same = [a for a in recorded if (a["tool"], a["request"]) == (e["tool"], e["request"])]
        if any(a["result"] != e["result"] for a in same):
            raise RunRefused(f"the investigation's {e['tool']} disagrees with validation run {run.name!r}'s result "
                             "for the same request")
        (matched if same else unmatched).append(
            {"evidence_id": e["evidence_id"], "tool": e["tool"], **({"run_call": same[0]["call"]} if same else {})})
    return {"validation_run": run.name, "compatible": True, "same_query": True,
            "query_run_id": initial["result"]["query_run_id"], "identical_results": matched,
            "unmatched": unmatched, "unmatched_meaning": "evidence of this investigation for requests the "
                                                         "validation run never made: not compared, not equivalent"}


def check_report(doc, data, synthetic):
    """The saved report verifies (verify_report_v1) and is on exactly this data, split and contract."""
    problems = av1.verify_report_v1(doc)
    if problems:
        raise RuntimeError(f"the saved report does not verify: {problems[:3]}")
    d = doc["data"]
    if (d["data_source"], d["derived_from"], d["fingerprint"], d["sessions"]) != (
            data.data_source, data.derived_from, data.fingerprint, list(data.sessions)):
        raise RuntimeError("the saved report names other data")
    split = "synthetic" if synthetic else "validation"
    if any(e["provenance"]["split"] != split or e["provenance"]["data_fingerprint"] != data.fingerprint
           for e in doc["evidence"]):
        raise RuntimeError("saved evidence has another split or data fingerprint")
    sessions = set(data.sessions)
    for e in doc["evidence"]:
        req = e["request"]
        named = req.get("session")
        named = named if isinstance(named, list) else [named] if named else [b["session"] for b in req["bouts"]]
        if not set(named) <= sessions:
            raise RuntimeError("saved evidence names a session outside the validation data")
    return {"verify_report_v1": "passed", "data": "matches the inputs", "split": split}


# ---- the run ----

def prepare(out):
    out = smoke.ignored_output_dir(out)
    if out.exists() and (not out.is_dir() or out.is_symlink()):
        raise RunRefused(f"{out} is not a real directory")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = [n for n in OUTPUT_NAMES if (out / n).exists() or (out / n).is_symlink()]
    if existing:
        raise RunRefused(f"{out} already contains {existing}; choose a new directory (nothing is overwritten)")
    return out


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


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
    summary = {"mode": "Phase 3B.1 offline investigation (no network, no model call, no cost)", "steps": [],
               "model": {"adapter": OFFLINE_MODEL, "language_model": False,
                         "note": "a fixed decision procedure over the tool results it is sent; no LLM was called"},
               "test_split": "not read: validation member and validation cache only"}
    status = "verified"
    try:
        data, scores, pose = vr._step(summary, "inputs", lambda: load_data(args, manifest, cache))
        synthetic = scores["synthetic"]
        labels = vr.SYNTHETIC_LABELS if synthetic else vr.DATA_LABELS
        summary["inputs"] = {"scores": {**{k: v for k, v in scores.items() if k != "frames_per_session"},
                                        "label": labels["scores"]},
                             "pose_cache": pose, "data_source": data.data_source, "derived_from": data.derived_from,
                             "data_fingerprint": data.fingerprint, "sessions": list(data.sessions),
                             "scope": {"n_sessions": len(data.sessions),
                                       "n_frames": sum(data.n_frames(s) for s in data.sessions)}}
        recorded = None
        if args.validation_run:
            recorded = vr._step(summary, "validation_run", lambda: check_validation_run(
                args.validation_run, args.runs_dir, manifest, data, scores, synthetic))
        budget = ia.Budget()
        report = vr._step(summary, "investigate", lambda: av1.investigate_v1(
            args.question, data, OfflineInvestigatorV1(), budget=budget, investigation_id=INVESTIGATION_ID,
            allow_synthetic=synthetic))
        saved = _json_bytes(report.to_json())
        compat = None
        if recorded is not None:                          # before any report is written: a refusal leaves none
            compat = vr._step(summary, "compatibility", lambda: check_query_against_run(json.loads(saved), recorded))
        smoke.create_exclusive(out / OUTPUT_NAMES[0], saved)
        doc = json.loads((out / OUTPUT_NAMES[0]).read_bytes())             # what was saved, not what is in memory
        if doc != json.loads(saved):
            raise RuntimeError("the saved report is not the report that was checked")
        summary["report"] = vr._step(summary, "verify", lambda: check_report(doc, data, synthetic))
        smoke.create_exclusive(out / OUTPUT_NAMES[1], av1.render_markdown_v1(doc).encode("utf-8"))
        if compat is not None:
            summary["report"]["validation_run"] = compat
        summary["investigation"] = {
            "finished": doc["finished"], "stopping_reason": doc["stopping_reason"], "finding": doc["finding"],
            "primary_evidence_id": doc["primary_evidence_id"], "observations": doc["observations"],
            "fingerprint": doc["fingerprint"], "publication": doc["publication"], "usage": doc["usage"],
            "budget": doc["budget"],
            "tool_sequence": [(t["tool"], t.get("refused") or t["outcome"]) for t in doc["trace"]
                              if t["kind"] in ("tool_call", "finish")],
            "evidence": [{"evidence_id": e["evidence_id"], "tool": e["tool"], "status": e["status"]}
                         for e in doc["evidence"]]}
        if not doc["finished"]:
            status = "incomplete"
    except REFUSALS:
        status = "refused"
    except Exception:
        status = "failed"
    summary["status"] = status
    summary["outputs"] = {n: _sha((out / n).read_bytes()) for n in OUTPUT_NAMES[:2] if (out / n).is_file()}
    smoke.create_exclusive(out / OUTPUT_NAMES[2], _json_bytes(summary))
    summary["output_dir"] = str(out)
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="the validation score-drop ZIP (read in place, never extracted)")
    ap.add_argument("--archive-sha256", required=True, help="the archive's expected SHA-256 (checked before reading)")
    ap.add_argument("--out", required=True, help="a new, git-ignored (or outside the repository) directory")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--validation-run", default=None, help="a recorded Phase 2 run to check compatibility against")
    ap.add_argument("--runs-dir", default=None, help="where recorded runs live (default: validation_run_view's)")
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
    except REFUSALS as exc:
        print(f"refused ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary.get(k) for k in ("status", "steps", "output_dir")}, indent=2, default=str))
    return {"verified": 0, "refused": 2}.get(summary["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
