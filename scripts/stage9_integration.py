"""Stage 9 (S9-6) full offline integration: verified validation score drop ->
monitor journal -> frozen provenance -> prediction / monitor investigation
tools -> bounded Stage 6 investigation -> Findings v2.1 -> saved, re-verified
JSON and Markdown reports. No network, no API key, no paid call.

    python scripts/stage9_integration.py ARCHIVE --out DIR \\
        [--member scores_validation_model_b.csv] [--archive-sha256 HEX] \\
        [--cache-dir data/raw/calms21_validation] [--manifest splits/split_manifest.json]

Steps (each recorded in DIR/integration_summary.json):
1. scores     score_drops.load_validation_scores: the validation member only (a
              test member is refused unopened), checked against the archive's
              SHA256SUMS.txt and the tracked split manifest.
2. monitor    the LOCAL_SMOKE_TEST rule (placeholder thresholds; an integration
              demonstration, not a validated monitoring rule) over the verified
              rows, journal bound to the drop (run header, completion record).
3. snapshot   event_log.freeze_run + load_snapshot: verified_score_drop, complete.
4. classifier Stage 6 (investigation_agent.investigate) with the Stage 8 offline
              emulator over classifier_prediction episodes (query_events,
              get_window_features, check_tracking_quality, compare_windows).
5. monitor_event  the same over monitor_event episodes of the frozen run.
              The two investigations never share a findings chain.
6. reports    Findings v2.1 JSON and Markdown per investigation, re-verified from
              the saved JSON (verify_saved_v2); status must be not_authorized or
              record_only, never publishable.

DIR must be outside the repository or git-ignored, and must not contain any of
the output names (nothing is overwritten). Pose for the windows comes from the
verified validation cache (its annotations are never read for these runs).
Exit status 0 when every step verified, 1 otherwise, 2 for refused inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker import event_log as el  # noqa: E402
from live_tracker import investigation_agent as ia  # noqa: E402
from live_tracker import investigation_claims as cl  # noqa: E402
from live_tracker import investigation_contracts as ic  # noqa: E402
from live_tracker import investigation_live as live  # noqa: E402
from live_tracker import investigation_report as ir  # noqa: E402
from live_tracker import investigation_report_v2 as v2  # noqa: E402
from live_tracker import score_drops  # noqa: E402
from live_tracker.investigation_events import AnnotationEventSource, CalMS21InvestigationTools  # noqa: E402
from live_tracker.investigation_sources import ClassifierEventSource, MonitorEventSource  # noqa: E402
from live_tracker.monitor import Monitor  # noqa: E402

_spec = importlib.util.spec_from_file_location("stage9_smoke", Path(__file__).with_name("stage9_smoke.py"))
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)

RUN_ID = "stage9_integration"
SOURCES = ("classifier_prediction", "monitor_event")
ALLOWED_PUBLICATION = ("not_authorized", "record_only")
OUTPUT_NAMES = ("events.journal.jsonl", "journal_snapshot.json", "integration_summary.json") + tuple(
    f"{s}_{kind}" for s in SOURCES for kind in ("investigation.json", "report.json", "report.md"))


class IntegrationError(RuntimeError):
    """An integration step cannot be completed or does not verify."""


def _step(summary, name, fn):
    """Run one step; record ok / failed (with the error class and message) in the summary."""
    try:
        out = fn()
    except Exception as exc:                                   # preserved as a diagnostic record
        summary["steps"].append({"step": name, "status": "failed", "error": type(exc).__name__,
                                 "message": str(exc)[:500],
                                 "where": traceback.extract_tb(exc.__traceback__)[-1].name})
        raise
    summary["steps"].append({"step": name, "status": "ok", **(out if isinstance(out, dict) else {})})
    return out


def prepare(out):
    """The output directory (ignored by Git or outside the repository), created
    private if new; refused when it already holds any output name."""
    out = smoke.ignored_output_dir(out)
    if out.exists() and (not out.is_dir() or out.is_symlink()):
        raise IntegrationError(f"{out} is not a real directory")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = [n for n in OUTPUT_NAMES if (out / n).exists() or (out / n).is_symlink()]
    if existing:
        raise IntegrationError(f"{out} already contains {existing}; choose a new directory (nothing is overwritten)")
    return out


def write(out, name, data):
    smoke.create_exclusive(out / name, data if isinstance(data, bytes) else data.encode("utf-8"))


def run_monitor(drop, out, args):
    """LOCAL_SMOKE_TEST rule over the verified rows; journal bound to the drop."""
    rules, approval = smoke.build_rules(argparse.Namespace(
        smoke_rule=True, rule_json=None, confirm=None, behavior="attack", min_duration_s=args.min_duration_s,
        ask_threshold=args.ask_threshold, alert_threshold=args.alert_threshold, alert_mode="alert"))
    journal = out / OUTPUT_NAMES[0]
    smoke.create_exclusive(journal, b"")
    m = Monitor(rules.active(), journal_path=journal, run_id=RUN_ID, provenance=drop)
    for row in drop.rows():
        m.step(row)
    events = m.finish()
    return rules.active()[0], approval, events, journal


def frozen_snapshot(journal, run_id=RUN_ID):
    """The completed, verified frozen snapshot of the run (refused otherwise)."""
    try:
        snap = el.freeze_run(journal, run_id)
        snap = el.load_snapshot(snap.canonical, snap.sha256)
    except (el.JournalIntegrityError, KeyError) as exc:
        raise IntegrationError(f"the monitor journal does not give a verified frozen run: {exc}") from None
    if snap.completion != "complete" or snap.provenance_status != "verified_score_drop":
        raise IntegrationError(f"frozen run is {snap.provenance_status} / {snap.completion}, not verified_score_drop "
                               "/ complete")
    return snap


def offline_model(source):
    """The Stage 8 offline path: the real SDK and GuardedClient over the
    in-process emulator (no network), with the Stage 8 budget."""
    import httpx2
    plan = live.LivePlan(max_model_turns=10, max_tool_calls=9)
    live.require_clean_environment()
    # 25 baselines per session keep compare_windows within the contract's request size (64 kB of window ids)
    emulator = live._EmulatedOpenRouter(source=source, with_quality=True, max_baselines_per_session=25)
    prices = live.snapshot_prices(plan.model_id) or live.Prices(1.0, 1.0, "offline placeholder")
    budget, _ = live.plan_budget(plan, prices)
    model, ledger = live.build_model(plan, env={live.PROVIDERS[plan.provider].key_env: "offline-placeholder-not-a-key"},
                                     http_client=httpx2.Client(transport=httpx2.MockTransport(emulator)))
    return model, budget, emulator, ledger


def check_single_source(report, source):
    """Every snapshot of one investigation comes from its own source (never a mixed chain)."""
    found = sorted({cl.evidence_source(e) for e in report.evidence})
    if found and found != [source]:
        raise IntegrationError(f"the {source} investigation holds evidence of {found}")
    if report.registry_version != cl.REGISTRY_V21:
        raise IntegrationError(f"the {source} investigation used {report.registry_version}, not {cl.REGISTRY_V21}")


def verify_saved(saved_json, policy, markdown, source):
    """Rebuild the saved document from its canonical payload (tamper-resistant) and check its status."""
    doc = json.loads(saved_json)
    rebuilt = v2.verify_saved_v2(doc, policy=policy)
    if v2.render_markdown_v2(doc, policy=policy) != markdown:
        raise IntegrationError("the saved Markdown is not the rendering of the saved JSON")
    publication = rebuilt["status"]["publication"]
    if publication not in ALLOWED_PUBLICATION:
        raise IntegrationError(f"a prediction-derived report is {publication!r}, not not_authorized / record_only")
    if rebuilt.get("source", {}).get("source") != source or not rebuilt["source"]["disclosures"]:
        raise IntegrationError("the report lacks its source-specific disclosures")
    return rebuilt


def investigate(source, tools, policy, out, question):
    model, budget, emulator, ledger = offline_model(source)
    report = ia.investigate(question, tools, policy, model, budget=budget)
    check_single_source(report, source)
    write(out, f"{source}_investigation.json", json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n")
    doc, md = v2.render_v2(report, policy=policy)
    saved = ir.to_json_text(doc)
    write(out, f"{source}_report.json", saved)
    write(out, f"{source}_report.md", md)
    rebuilt = verify_saved(saved, policy, md, source)
    return {"source": source, "finished": report.finished, "stopping_reason": report.stopping_reason,
            "finding": report.finding, "registry_version": report.registry_version,
            "publication": rebuilt["status"]["publication"], "fingerprint": report.fingerprint,
            "n_evidence": len(report.evidence),
            "tools_used": [t["tool"] for t in report.trace if t["kind"] in ("tool_call", "finish")],
            "evidence_sources": sorted({cl.evidence_source(e) for e in report.evidence}),
            "emulated_requests": len(emulator.requests), "provider_reported_cost_usd": ledger.reported_cost_usd,
            "verified_from_saved_json": True}


def run(args):
    archive = Path(args.archive).expanduser()
    if not archive.is_file():
        raise IntegrationError(f"score archive {archive.name!r} not found (pass the validation score drop ZIP)")
    cache, manifest = Path(args.cache_dir), Path(args.manifest)
    for what, path in (("validation pose cache", cache), ("split manifest", manifest)):
        if not path.exists():
            raise IntegrationError(f"{what} {path} not found")
    out = prepare(args.out)
    summary = {"mode": "offline Stage 9 integration (no network, no paid call)", "steps": [],
               "rule_note": "LOCAL_SMOKE_TEST: placeholder thresholds; an integration demonstration, not a validated "
                            "monitoring rule", "investigations": {}}
    try:
        drop = _step(summary, "scores", lambda: score_drops.load_validation_scores(
            archive, args.member, manifest_path=manifest, expected_archive_sha256=args.archive_sha256))
        summary["source"] = drop.provenance.to_json()
        rule, approval, events, journal = _step(summary, "monitor", lambda: run_monitor(drop, out, args))
        summary["monitor"] = {"rule": rule.to_dict(), "rule_approval": approval, "events": len(events)}
        snap = _step(summary, "snapshot", lambda: frozen_snapshot(journal))
        write(out, OUTPUT_NAMES[1], snap.canonical)
        summary["monitor"].update(snapshot_sha256=snap.sha256, provenance_status=snap.provenance_status,
                                  completion=snap.completion, n_events=snap.n_events)
        policy = ic.EvidencePolicy.from_manifest(manifest, allow_synthetic=args.allow_synthetic)
        pose = AnnotationEventSource(cache, manifest, policy)      # pose only for these runs; labels never read
        tools = {
            "classifier_prediction": CalMS21InvestigationTools(policy, [pose, ClassifierEventSource(
                drop, policy, threshold=args.threshold, min_duration_frames=args.min_duration_frames)]),
            "monitor_event": CalMS21InvestigationTools(policy, [pose, MonitorEventSource(
                snap, drop, policy, rule_id=rule.rule_id)])}
        for source in SOURCES:
            summary["investigations"][source] = _step(
                summary, source, lambda s=source: investigate(s, tools[s], policy, out, args.question))
    except Exception:
        summary["status"] = "failed"
    else:
        summary["status"] = "verified"
    write(out, OUTPUT_NAMES[2], json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    summary["output_dir"] = str(out)
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="validation score-drop ZIP (read in place, never extracted)")
    ap.add_argument("--out", required=True, help="new or empty, git-ignored (or outside the repository) directory")
    ap.add_argument("--member", default="scores_validation_model_b.csv")
    ap.add_argument("--archive-sha256", default=None)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "raw" / "calms21_validation"))
    ap.add_argument("--manifest", default=str(ROOT / "splits" / "split_manifest.json"))
    ap.add_argument("--question", default=live.DEFAULT_QUESTION)
    ap.add_argument("--threshold", type=float, default=0.5, help="classifier source threshold (provisional)")
    ap.add_argument("--min-duration-frames", type=int, default=15, help="classifier source minimum (provisional)")
    ap.add_argument("--min-duration-s", type=float, default=2.0, help="LOCAL_SMOKE_TEST rule (placeholder)")
    ap.add_argument("--ask-threshold", type=float, default=0.5)
    ap.add_argument("--alert-threshold", type=float, default=0.8)
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="tests only: accept SYNTHETIC fixture sessions (never real evidence)")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        summary = run(args)
    except (IntegrationError, smoke.SmokeError, score_drops.ScoreDropError) as exc:
        print(f"refused ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary[k] for k in ("status", "steps", "investigations", "output_dir")}, indent=2,
                     sort_keys=True, default=str))
    return 0 if summary["status"] == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
