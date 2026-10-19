"""EXPERIMENTAL (Investigation mode, Stage 7): evidence-grounded scientific reports (new file).

Turns a Stage 6 InvestigationReport into a researcher-readable report:

    doc, markdown = render(report)            # or build_report(report) + render_markdown(doc)

Reporting only: no tool is called, no model is asked, nothing is stored.
Separate from the monitor journal (event_log.py); integration is Stage 9.

Fail closed
-----------
Nothing is rendered until the report has been verified; any problem raises
ReportIntegrityError listing every problem found:
- each evidence snapshot is a well-formed result of its request
  (check_result_schema, or validate_result when a policy is given) and its
  evidence_id is the hash of that request and result;
- the trace lists exactly the executed tool calls, with the same requests,
  in the same order as the evidence;
- every evidence_id the report cites is one of its snapshots;
- every observation value (and its statement) is what its field holds in
  the snapshot; the primary result equals its snapshot;
- the finding agrees with the evidence (effect_described: an "ok"
  comparison, with the direction of its effect; otherwise no direction; an
  incomplete run has no finding);
- every comparison is descriptive: uncertainty method "none" and no interval;
- the tool limitations are exactly those of the snapshots; investigation
  (agent) limitations are the loop's fixed operational texts; the model's
  prose passes the Stage 6 checks;
- snapshots that measured the same window and feature agree (cross-snapshot
  check). Values are compared only when comparable: the same statistic (a
  window median), identical processing methodology (PROCESSING_METHOD) and
  identical, hashed data provenance. Units take part: agreement is verified
  only when both snapshots record the same unit. Explicitly different units
  are a unit conflict (values not compared, nothing converted); a unit
  missing on either side leaves agreeing values "unit_unverified". All of
  these are disclosed as not verified, never silently trusted. Values that
  disagree are refused whenever processing and data are identical and the
  units do not explicitly conflict: the same computation on the same data
  must give the same numbers, whatever a missing unit label would have said.

Findings
--------
Only an accepted effect_described finding on an "ok" comparison presents
an aggregate effect (and a direction). For every other finding a recorded
effect stays in the appendix, marked as not a finding.

Trust
-----
Tool limitations come from validated results (verbatim); investigation
limitations are fixed texts of the Stage 6 loop; observations are values
read from snapshots; interpretations, next steps and model limitations are
model-authored and unverified beyond the lexical checks.

Facts come from the snapshots
-----------------------------
Measurements, counts, units, sessions, exclusions, sufficiency criteria and
provenance are read from the verified snapshots, never from model prose. A
unit is the one its own snapshot records (only comparisons record units);
otherwise it is unknown. Query counts distinguish the full query (summary
metadata) from the returned, possibly truncated, page.
The model's interpretations and next steps are shown as such. Metadata the
evidence does not record is null (with the reason), never inferred.

render_markdown(doc) uses nothing but the JSON document, so the two exports
cannot disagree.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter

from . import investigation_agent as ia
from . import investigation_contracts as ic

__all__ = ["REPORT_SCHEMA", "ReportIntegrityError", "build_report", "render", "render_markdown", "to_json_text"]

REPORT_SCHEMA = "live-tracker-investigation-report/0.1"

SOURCE_CLASS = {"validation_annotation": "ground_truth_annotation",
                "classifier_prediction": "classifier_prediction",
                "monitor_event": "monitor_event"}
DATA_CLASS = {"calms21_validation_cache": "dataset", "classifier_scores": "classifier_output",
              "monitor_journal": "monitor_journal", "synthetic_fixture": "synthetic"}
DESCRIPTIVE_POLICY = ("Descriptive only: no confidence interval, p-value or significance test is reported, no "
                      "population inference is made, and an association before a behavior is not a cause.")
UNIT_OF_ANALYSIS = ("Sessions are the unit of analysis. Bouts and windows within one session are repeated "
                    "measurements of that session, not independent replicates.")
# methodology items defining how a window value is computed: two snapshots'
# values are comparable only when all of these are recorded and identical
PROCESSING_METHOD = ("event_source", "causal_computation", "tracking_flags", "score_validation", "tracking_ok",
                     "interpolation", "smoothing", "smoother_memory_frames", "fps", "feature_lookback_frames")
TRUST = {"tool": "validated tool results, verbatim",
         "agent": "investigation loop, fixed operational text",
         "model": "model-authored, unverified (lexical checks only)"}
FINDING_LABEL = {"effect_described": "Effect described (descriptive)",
                 "insufficient_evidence": "Insufficient evidence",
                 "inconclusive": "Inconclusive", "incomplete": "Incomplete (no finding)"}


class ReportIntegrityError(ValueError):
    """The investigation report cannot be rendered: its evidence is tampered,
    missing or inconsistent."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("report not rendered: " + "; ".join(self.problems[:10]))


# ---- verification ----

def _resolve(result, path):
    """The value of a field path (Stage 6 observation grammar) in a result."""
    if not isinstance(path, str) or len(path) > ia.MAX_FIELD or not ia._FIELD.fullmatch(path):
        raise KeyError(path)
    value = result
    for key, index in re.findall(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", path):
        if key and isinstance(value, dict) and key in value:
            value = value[key]
        elif index and isinstance(value, list) and int(index) < len(value):
            value = value[int(index)]
        else:
            raise KeyError(path)
    return value


def _sign(v):
    return "negative" if v < 0 else "positive" if v > 0 else "zero"


def _verify(report, policy):
    try:
        return _check(report, policy)
    except ReportIntegrityError:
        raise
    except Exception as exc:                  # a malformed report: never rendered
        raise ReportIntegrityError([f"malformed report ({type(exc).__name__})"]) from None


def _check(report, policy):
    if not isinstance(report, ia.InvestigationReport):
        raise ReportIntegrityError(["not an InvestigationReport"])
    problems = []
    finding, reason = report.finding, report.stopping_reason
    if finding not in (*ia.FINDINGS, "incomplete"):
        problems.append(f"unknown finding {str(finding)[:40]!r}")
    if reason not in ia.STOP_REASONS:
        problems.append(f"unknown stopping reason {str(reason)[:40]!r}")
    if not (report.finished == (reason == "finished") == (finding != "incomplete")):
        problems.append("finished, stopping_reason and finding disagree")
    if not isinstance(report.question, str) or not report.question.strip():
        problems.append("no research question")

    # evidence snapshots
    snaps = {}
    if not isinstance(report.evidence, list):
        raise ReportIntegrityError(problems + ["evidence is not a list"])
    for i, e in enumerate(report.evidence):
        where = f"evidence[{i}]"
        if not isinstance(e, dict) or set(e) != {"evidence_id", "tool", "status", "request", "result"}:
            problems.append(f"{where}: not a request / result snapshot")
            continue
        eid = e["evidence_id"]
        if not isinstance(eid, str) or eid in snaps:
            problems.append(f"{where}: missing or duplicate evidence_id")
            continue
        try:
            if e["tool"] not in ic.TOOLS:
                raise ic.ContractError(where, "unknown tool")
            if policy is None:
                ic.check_result_schema(e["tool"], e["request"], e["result"])
            else:
                ic.validate_result(e["tool"], e["request"], e["result"], policy)
            if e["result"]["evidence_id"] != eid or e["status"] != e["result"]["status"]:
                raise ic.ContractError(where, "evidence_id / status differ from the result")
        except ic.ContractError as exc:
            problems.append(f"{where} ({str(eid)[:80]}): failed verification: {str(exc)[:200]}")
            continue
        except Exception as exc:
            problems.append(f"{where}: failed verification ({type(exc).__name__})")
            continue
        snaps[eid] = e
    if len(snaps) != len(report.evidence):
        raise ReportIntegrityError(problems)          # nothing below can be trusted
    if report.verify_evidence():
        problems.append(f"evidence hashes do not match: {report.verify_evidence()[:3]}")

    # trace <-> evidence
    trace = report.trace if isinstance(report.trace, list) else []
    calls = [t for t in trace if isinstance(t, dict) and t.get("kind") == "tool_call" and "evidence_id" in t]
    if [t["evidence_id"] for t in calls] != list(snaps):
        problems.append("the trace's executed tool calls do not match the evidence")
    elif any(t.get("request") != snaps[t["evidence_id"]]["request"] or t.get("tool") != snaps[t["evidence_id"]]["tool"]
             for t in calls):
        problems.append("a traced request differs from its evidence snapshot")
    if not trace or trace[-1].get("kind") != "stop" or trace[-1].get("stopping_reason") != reason:
        problems.append("the trace does not end with the report's stopping reason")

    # finding, direction, primary result
    p = report.primary_evidence
    if p is not None:
        pid = p.get("evidence_id") if isinstance(p, dict) else None
        if pid not in snaps:
            problems.append("primary evidence is not one of the snapshots")
        else:
            res = snaps[pid]["result"]
            m = res["measurements"]
            expect = {"evidence_id": pid, "tool": snaps[pid]["tool"], "status": res["status"]}
            if snaps[pid]["tool"] == "compare_windows" and m is not None:
                expect.update({k: m[k] for k in ("feature", "effect", "sufficiency", "n_sessions_paired",
                                                 "n_windows_pre", "n_windows_baseline")})
            if p != expect:
                problems.append("primary evidence differs from its snapshot")
    if finding == "incomplete":
        if p is not None or report.direction is not None or report.observations or report.interpretations \
                or report.next_steps:
            problems.append("an incomplete investigation carries a finding")
    elif finding == "effect_described":
        snap = snaps.get(p["evidence_id"]) if isinstance(p, dict) else None
        res = snap and snap["result"]
        if not res or snap["tool"] != "compare_windows" or res["status"] != "ok" or \
                res["measurements"]["effect"]["value"] is None or not res["measurements"]["sufficiency"]["met"]:
            problems.append("effect_described without an 'ok' comparison with an effect")
        elif report.direction != _sign(res["measurements"]["effect"]["value"]):
            problems.append("direction differs from the sign of the effect")
    elif report.direction is not None:
        problems.append(f"{finding} carries a direction")
    for eid, e in snaps.items():                       # status / effect / sufficiency agree in every comparison
        if e["tool"] == "compare_windows" and e["result"]["measurements"] is not None:
            m = e["result"]["measurements"]
            if (e["status"] == "ok") != (m["effect"]["value"] is not None) or \
                    (e["status"] == "ok") != m["sufficiency"]["met"]:
                problems.append(f"{eid}: status, effect and sufficiency disagree")

    # observations: re-derived from the snapshots
    for i, o in enumerate(report.observations):
        try:
            snap = snaps[o["evidence_id"]]
            value = _resolve(snap["result"], o["field"])
        except (KeyError, TypeError):
            problems.append(f"observations[{i}]: cites a field that is not in its evidence")
            continue
        statement = f"{snap['tool']} {o['evidence_id']}: {o['field']} = {json.dumps(value)}"
        if o.get("kind") != "observation" or o.get("tool") != snap["tool"] or o.get("value") != value or \
                o.get("statement") != statement:
            problems.append(f"observations[{i}]: value or statement differs from the evidence")

    # model prose
    for i, it in enumerate(report.interpretations):
        ids = it.get("evidence_ids") if isinstance(it, dict) else None
        if not ids or any(e not in snaps for e in ids) or it.get("kind") != "interpretation" or \
                not isinstance(it.get("statement"), str):
            problems.append(f"interpretations[{i}]: cites evidence that is not in the report")
            continue
        problems += [f"interpretations[{i}]: {x}" for x in ia._prose_problems(it["statement"], finding)]
    for i, s in enumerate(report.next_steps):
        problems += [f"next_steps[{i}]: {x}" for x in ia._prose_problems(s, finding)]

    # limitations: the tools' exactly, model ones checked
    tool_lims = {l["text"] for l in report.limitations if l.get("source") == "tool"}
    snap_lims = {t for e in snaps.values() for t in e["result"]["limitations"]}
    if tool_lims != snap_lims:
        problems.append("tool limitations differ from those of the evidence")
    agent_texts = _agent_texts(report)
    for l in report.limitations:
        if l.get("source") not in ("tool", "agent", "model") or not isinstance(l.get("text"), str):
            problems.append("a limitation has no known source")
        elif l["source"] == "model":
            if finding == "incomplete":
                problems.append("an incomplete investigation carries model-authored limitations")
            problems += [f"model limitation: {x}" for x in ia._prose_problems(l["text"], finding)]
        elif l["source"] == "agent" and l["text"] not in agent_texts:
            problems.append("an investigation limitation is not one of the loop's fixed texts")

    # descriptive-only policy (Stage 5)
    for eid, e in snaps.items():
        if e["tool"] == "compare_windows" and e["result"]["measurements"] is not None:
            eff = e["result"]["measurements"]["effect"]
            if e["request"]["uncertainty"]["method"] != "none" or \
                    any(eff[k] is not None for k in ("ci_low", "ci_high", "ci_level")):
                problems.append(f"{eid}: an interval or uncertainty method outside the descriptive policy")
    cross = _reconcile(snaps)
    problems += cross.pop("contradictions")
    if problems:
        raise ReportIntegrityError(problems)
    return snaps, cross


def _agent_texts(report):
    """The operational limitation texts the Stage 6 loop can write for this run."""
    reason = report.stopping_reason
    texts = set(ia.AGENT_LIMITATIONS)
    if reason == "usage_unavailable":
        texts.add("The last model reply reported no valid token usage, so the token / cost budget could no longer "
                  "be enforced; it was not used and the run stopped. Usage totals exclude that reply.")
    if reason in ("token_budget", "cost_budget", "timeout"):
        texts.add(f"The last model request went past the {reason} limit (a request cannot be interrupted); none "
                  "of its actions were executed.")
    if not report.finished:
        texts.add(f"The investigation stopped before a final answer ({reason}); the evidence listed is partial "
                  "and no finding was made.")
    return texts


# ---- cross-snapshot agreement ----

def _window_values(e):
    """{(window_id, feature): (statistic, value, n_frames, n_usable)} measured by a snapshot."""
    m = e["result"]["measurements"]
    if m is None:
        return {}
    if e["tool"] == "get_window_features":
        return {(w["window_id"], f): ("window_median", v["median"], v["n_frames"], v["n_usable"])
                for w in m["windows"] for f, v in w["features"].items()}
    if e["tool"] == "compare_windows":
        method = _method(e["result"], "window_value")
        statistic = "window_median" if isinstance(method, str) and \
            method.startswith("median of the feature over usable frames") else f"unrecognised: {method!r}"
        return {(w["window_id"], m["feature"]): (statistic, w["value"], w["n_frames"], w["n_usable"])
                for s in m["per_session"] for w in s["pre_windows"] + s["baseline_windows"]}
    return {}


def _comparability(a, b):
    """Reasons two snapshots' window values cannot be compared (empty: comparable)."""
    reasons = []
    for e in (a, b):
        missing = [k for k in PROCESSING_METHOD if not any(x["name"] == k for x in e["result"]["methodology"])]
        if missing:
            reasons.append(f"{e['evidence_id']} does not record processing {missing[:4]}")
    if not reasons:
        conf = [{x["name"]: (json.dumps(x["value"]), x["status"]) for x in e["result"]["methodology"]
                 if x["name"] in PROCESSING_METHOD} for e in (a, b)]
        differ = sorted(k for k in PROCESSING_METHOD if conf[0][k] != conf[1][k])
        if differ:
            reasons.append(f"processing configuration differs: {differ}")
    data = [{(d["kind"], d["sha256"]) for d in e["result"]["provenance"]["data"]} for e in (a, b)]
    if not data[0] or any(sha is None for d in data for _, sha in d):
        reasons.append("data provenance is not recorded with hashes")
    elif data[0] != data[1]:
        reasons.append("data provenance differs")
    return reasons


def _reconcile(snaps):
    """Agreement of window values measured by more than one snapshot."""
    values = {eid: _window_values(e) for eid, e in snaps.items()}
    pairs, contradictions = [], []
    ids = [e for e in snaps if values[e]]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = sorted(set(values[a]) & set(values[b]))
            if not shared:
                continue
            reasons = _comparability(snaps[a], snaps[b])
            stats = {values[x][k][0] for x in (a, b) for k in shared}
            if stats != {"window_median"}:
                reasons.append(f"different or unrecognised statistics {sorted(stats)}")
            ua, ub = _unit(snaps[a]), _unit(snaps[b])
            pair = {"evidence_ids": [a, b], "shared_window_features": len(shared),
                    "features": sorted({f for _, f in shared}), "units": [ua, ub]}
            if reasons:
                pairs.append({**pair, "outcome": "not_comparable", "reasons": reasons})
                continue
            if ua is not None and ub is not None and ua != ub:
                pairs.append({**pair, "outcome": "unit_conflict", "reasons": [
                    f"units differ ({ua!r} vs {ub!r}): values are not compared and no conversion is assumed"]})
                continue
            bad = [k for k in shared if not _agree(values[a][k], values[b][k])]
            if bad:
                w, f = bad[0]
                contradictions.append(f"{a} and {b} disagree on {f} of {len(bad)} shared window(s), e.g. {w}: "
                                      f"{values[a][bad[0]][1:]} vs {values[b][bad[0]][1:]}")
                pairs.append({**pair, "outcome": "contradiction", "reasons": []})
            elif ua is None or ub is None:
                missing = [x for x, u in ((a, ua), (b, ub)) if u is None]
                pairs.append({**pair, "outcome": "unit_unverified", "reasons": [
                    f"values agree, but no unit is recorded by {', '.join(missing)}: agreement is not verified "
                    "without explicit unit evidence"]})
            else:
                pairs.append({**pair, "outcome": "agree", "reasons": []})
    outcomes = {p["outcome"] for p in pairs}
    status = ("not_applicable" if not pairs else "verified" if outcomes == {"agree"} else
              "partially_verified" if "agree" in outcomes else "not_verified")
    return {"status": status, "pairs": pairs, "unit_conflicts": _unit_records(snaps),
            "contradictions": contradictions}


def _unit_records(snaps):
    """Per feature, the units comparisons record when they are not all the
    same: JSON-safe records (a unit may be null = not recorded)."""
    by = {}
    for eid, e in snaps.items():
        if e["tool"] == "compare_windows" and e["result"]["measurements"] is not None:
            by.setdefault(e["request"]["feature"], {}).setdefault(_unit(e), []).append(eid)
    out = []
    for feature, units in sorted(by.items()):
        if len(units) > 1:
            known = [u for u in units if u is not None]
            out.append({"feature": feature,
                        "kind": "conflicting" if len(known) > 1 else "partly_unrecorded",
                        "units": [{"unit": u, "evidence_ids": units[u]}
                                  for u in sorted(units, key=lambda u: (u is None, u or ""))]})
    return out


def _unit_label(u):
    return "not recorded" if u is None else u


def _agree(x, y):
    (_, v1, f1, u1), (_, v2, f2, u2) = x, y
    if (f1, u1) != (f2, u2) or (v1 is None) != (v2 is None):
        return False
    return v1 is None or ic._close(v1, v2)


# ---- reading the snapshots ----

def _method(result, name):
    return next((m["value"] for m in result["methodology"] if m["name"] == name), None)


def _sources_and_sessions(e):
    """(event sources, behaviors, sessions) a snapshot is about, from its IDs."""
    req, res = e["request"], e["result"]
    m = res["measurements"] or {}
    if e["tool"] == "query_events":
        sessions = set(m.get("sessions_searched", [])) | {x["session"] for x in m.get("episodes", [])}
        return {req["source"]}, {req["behavior"]}, sessions
    if e["tool"] == "get_window_features":
        parsed = [ic.parse_episode_id(x) for x in req["episode_ids"]]
    elif e["tool"] == "check_tracking_quality":
        parsed = [ic.parse_window_id(x) for x in req["window_ids"]]
    else:
        parsed = [ic.parse_window_id(x) for x in req["pre_window_ids"] + req["baseline_window_ids"]]
    return {x["source"] for x in parsed}, {x["behavior"] for x in parsed}, {x["session"] for x in parsed}


def _unit(e):
    """The unit a snapshot itself records for its feature values (only
    comparisons record one), else None: never borrowed from another snapshot."""
    if e["tool"] != "compare_windows":
        return None
    unit = _method(e["result"], "units")
    return unit if isinstance(unit, str) else None


def _unit_of(snap, field):
    tool, leaf = snap["tool"], re.split(r"[.\[]", field)[-1]
    if tool == "compare_windows":
        if leaf in ("value", "difference", "pre_summary", "baseline_summary"):
            return _unit(snap)
        return {"n_sessions_paired": "sessions", "n_windows_pre": "windows",
                "n_windows_baseline": "windows", "n_frames": "frames", "n_usable": "frames"}.get(leaf)
    if tool == "get_window_features":
        if re.search(r"\.features\.([a-z_]+)\.(median|mean)$", field):
            return None                               # get_window_features records no unit
    if leaf.endswith("_s"):
        return "s"
    if leaf.endswith("_frames") or leaf in ("frames", "start_frame", "end_frame", "n_frames", "n_usable"):
        return "frames"
    return {"n_episodes_total": "episodes"}.get(leaf)


def _compact_request(e):
    """The tool configuration of a request; ID lists become counts."""
    r = {k: v for k, v in e["request"].items() if k != "contract_version"}
    for k in ("episode_ids", "window_ids", "pre_window_ids", "baseline_window_ids"):
        if k in r:
            r[k] = {"count": len(r[k])}
    return copy.deepcopy(r)


def _summary(e):
    """A per-tool, status-aware summary of a snapshot for the appendix (facts only)."""
    tool, req, res = e["tool"], e["request"], e["result"]
    m = res["measurements"]
    if m is None:
        return {"status": res["status"], "errors": copy.deepcopy(res["errors"]), "measurements": None}
    if tool == "query_events":
        prov = [x["provenance"] for x in m["episodes"]]
        return {"source": req["source"], "source_class": SOURCE_CLASS[req["source"]], "behavior": req["behavior"],
                **_query_counts(e), "sessions_without_episodes": list(m["sessions_without_episodes"]),
                "returned_episodes_per_session": dict(sorted(Counter(x["session"] for x in m["episodes"]).items())),
                "episode_provenance": _merge_values(prov)}
    if tool == "get_window_features":
        by = Counter((w["kind"], w["status"]) for w in m["windows"])
        reasons = Counter((w["kind"], r) for w in m["windows"] for r in w["exclusion_reasons"])
        return {"features": list(req["features"]), "units": {f: None for f in req["features"]},
                "units_note": "get_window_features records no unit",
                "windows": {f"{k}_{s}": n for (k, s), n in sorted(by.items())},
                "exclusion_reasons": {f"{k}:{r}": n for (k, r), n in sorted(reasons.items())},
                "sessions_with_included_windows": len({w["session"] for w in m["windows"]
                                                       if w["status"] == "included"})}
    if tool == "check_tracking_quality":
        return {"n_windows": len(m["windows"]), "history_frames": list(req["history_frames"]),
                "verdicts": dict(sorted(Counter(w["verdict"] for w in m["windows"]).items())),
                "smoothing_contamination": dict(sorted(Counter(w["smoothing_contamination"]["status"]
                                                               for w in m["windows"]).items())),
                "motion_accuracy_verified": False}
    return _comparison(e)


def _query_counts(e):
    """Bout and session counts of a query: the full query (summary metadata)
    kept apart from the returned, possibly truncated, page."""
    m = e["result"]["measurements"]
    total, returned = m["n_episodes_total"], len(m["episodes"])
    with_bouts = set(m["sessions_searched"]) - set(m["sessions_without_episodes"])
    return {"n_bouts_total": total, "n_bouts_returned": returned, "truncated": m["truncated"],
            "n_sessions_searched": len(m["sessions_searched"]), "n_sessions_with_bouts": len(with_bouts),
            "sessions_with_bouts_basis": "query summary: sessions_searched minus sessions_without_episodes",
            "n_sessions_among_returned_bouts": len({x["session"] for x in m["episodes"]}),
            "truncation_note": (f"Truncated: {returned} of {total} bouts were returned; counts per session of "
                                "returned bouts are not the full distribution." if m["truncated"] else None)}


def _merge_values(dicts):
    """{key: sorted distinct values} over provenance dicts (file hashes counted)."""
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, set()).add(json.dumps(v))
    return {k: sorted(json.loads(v) for v in vs) if not k.endswith("sha256") else f"{len(vs)} distinct"
            for k, vs in sorted(out.items())}


def _comparison(e):
    req, res = e["request"], e["result"]
    m = res["measurements"]
    rows = []
    for s in m["per_session"]:
        bouts = {ic.parse_window_id(w["window_id"])["episode_id"] for w in s["pre_windows"]}
        rows.append({"session": s["session"], "bouts_with_used_pre_window": len(bouts),
                     "pre_windows_used": len(s["pre_windows"]), "baseline_windows_used": len(s["baseline_windows"]),
                     "pre_summary": s["pre_summary"], "baseline_summary": s["baseline_summary"],
                     "difference": s["difference"], "paired": s["difference"] is not None})
    behaviors = sorted({ic.parse_window_id(w)["behavior"] for w in req["pre_window_ids"]})
    return {"evidence_id": res["evidence_id"], "status": res["status"], "feature": m["feature"],
            "unit": _unit(e), "behavior": behaviors,
            "statistic": "median over paired sessions of (median pre-event window value - median baseline "
                         "window value)",
            "effect_value": m["effect"]["value"], "effect_reported": m["effect"]["value"] is not None,
            "effect_sign": None if m["effect"]["value"] is None else _sign(m["effect"]["value"]),
            "independent_sessions": m["n_sessions_paired"], "n_windows_pre": m["n_windows_pre"],
            "n_windows_baseline": m["n_windows_baseline"],
            "sufficiency": copy.deepcopy(m["sufficiency"]),
            "uncertainty": {"method": req["uncertainty"]["method"], "interval": None,
                            "note": "no approved session-level uncertainty method (Stage 5)"},
            "per_session": rows,
            "excluded_windows": {"n": len(m["excluded_windows"]),
                                 "by_reason": dict(sorted(Counter(x["reason"] for x in m["excluded_windows"]).items()))},
            "unit_of_analysis": UNIT_OF_ANALYSIS}


def _provenance(report, snaps):
    sources, behaviors, sessions = {}, set(), set()
    for eid, e in snaps.items():
        src, beh, ses = _sources_and_sessions(e)
        for s in src:
            sources.setdefault(s, []).append(eid)
        behaviors |= beh
        sessions |= ses

    def source_class(source, ids):
        kinds = {d["kind"] for i in ids for d in snaps[i]["result"]["provenance"]["data"]}
        if kinds == {"synthetic_fixture"}:
            return f"synthetic_fixture ({source.replace('_', ' ')} layout, not real data)"
        return SOURCE_CLASS[source] + (" (partly synthetic data)" if "synthetic_fixture" in kinds else "")
    data = {}
    for e in snaps.values():
        for d in e["result"]["provenance"]["data"]:
            key = (d["kind"], d["sha256"], d["description"])
            data.setdefault(key, {"kind": d["kind"], "class": DATA_CLASS[d["kind"]], "sha256": d["sha256"],
                                  "description": d["description"], "path": d["path"]})
    kinds = {d["kind"] for d in data.values()}
    episodes = [x for e in snaps.values() if e["tool"] == "query_events" and e["result"]["measurements"]
                for x in e["result"]["measurements"]["episodes"]]
    splits = sorted({x["provenance"]["split"] for x in episodes if "split" in x["provenance"]})
    annotators = sorted({x["provenance"]["annotator"] for x in episodes if "annotator" in x["provenance"]})
    model_ids = sorted({x["provenance"]["model_id"] for x in episodes if "model_id" in x["provenance"]})
    thresholds = sorted({x["provenance"]["threshold"] for x in episodes if "threshold" in x["provenance"]})
    if "classifier_prediction" not in sources:
        classifier = {"model_ids": None, "thresholds": None, "note": "no classifier-generated evidence"}
    elif model_ids:
        classifier = {"model_ids": model_ids, "thresholds": thresholds,
                      "note": "from the provenance of classifier-prediction episodes"}
    else:
        classifier = {"model_ids": None, "thresholds": None,
                      "note": "classifier predictions are referenced (window IDs) but the classifier's identity "
                              "is not recorded in this evidence"}
    investigators = sorted({t["model"] for t in report.trace if t.get("kind") == "model_turn" and t.get("model")})
    definitions = []
    for eid, e in snaps.items():
        if e["tool"] == "query_events":
            r = e["request"]
            definitions.append({"evidence_id": eid, "source": r["source"], "behavior": r["behavior"],
                                "episode_definition": _method(e["result"], "episode_definition"),
                                "merge_gap_frames": r["merge_gap_frames"],
                                "min_duration_frames": r["min_duration_frames"]})
    quality = [w for e in snaps.values() if e["tool"] == "check_tracking_quality" and e["result"]["measurements"]
               for w in e["result"]["measurements"]["windows"]]
    tracking = {"assessed": bool(quality), "n_window_assessments": len(quality),
                "verdicts": dict(sorted(Counter(w["verdict"] for w in quality).items())) or None,
                "smoothing_contamination": dict(sorted(Counter(w["smoothing_contamination"]["status"]
                                                               for w in quality).items())) or None,
                "tracking_flags": next((_method(e["result"], "tracking_flags") for e in snaps.values()
                                        if _method(e["result"], "tracking_flags") is not None), None),
                "motion_accuracy_verified": False}
    return {
        "event_sources": [{"source": s, "class": source_class(s, ids), "evidence_ids": ids}
                          for s, ids in sorted(sources.items())] or None,
        "data": sorted(data.values(), key=lambda d: (d["kind"], d["description"], str(d["sha256"]))) or None,
        "synthetic": None if not kinds else "synthetic_fixture" in kinds,
        "split": ({"values": splits, "basis": "provenance of query_events episodes"} if splits else
                  {"values": None, "basis": "not recorded in this evidence"}),
        "annotators": annotators or None,
        "classifier": classifier,
        "investigator_model_ids": investigators or None,
        "behaviors": sorted(behaviors) or None,
        "session_ids": sorted(sessions) or None,
        "session_id_note": "Session IDs are dataset file identifiers; the split is read from evidence provenance, "
                           "not from the ID text.",
        "event_definitions": definitions or None,
        "tracking_quality": tracking,
        "sufficiency_criteria": [copy.deepcopy(e["request"]["sufficiency"]) for e in snaps.values()
                                 if e["tool"] == "compare_windows"] or None,
        "tool_configuration": [{"evidence_id": eid, "tool": e["tool"],
                                **{k: e["result"]["provenance"][k] for k in ("implementation", "implementation_version",
                                                                             "contract_version", "code_sha256")},
                                "request": _compact_request(e)} for eid, e in snaps.items()] or None,
        "evidence_ids": list(snaps) or None,
    }


def _counts(snaps, primary):
    queries = [e for e in snaps.values() if e["tool"] == "query_events" and e["result"]["measurements"]]
    bouts = [{"evidence_id": e["evidence_id"], "behavior": e["request"]["behavior"], **_query_counts(e)}
             for e in queries]
    return {"behavior_bouts": bouts or None,
            "windows_used": None if primary is None else {"pre_event": primary["n_windows_pre"],
                                                          "baseline": primary["n_windows_baseline"]},
            "independent_sessions": None if primary is None else primary["independent_sessions"],
            "unit_of_analysis": UNIT_OF_ANALYSIS}


# ---- the document ----

def build_report(report, *, policy=None):
    """The verified report as a JSON-ready dict; raises ReportIntegrityError.
    With a policy, every snapshot is re-accepted by validate_result."""
    snaps, cross = _verify(report, policy)
    primary = None
    if report.primary_evidence and report.primary_evidence["tool"] == "compare_windows" and \
            snaps[report.primary_evidence["evidence_id"]]["result"]["measurements"] is not None:
        primary = _comparison(snaps[report.primary_evidence["evidence_id"]])
        # only an accepted effect_described finding presents the aggregate effect
        primary["presented_as_finding"] = report.finding == "effect_described"
        primary["direction"] = report.direction
    provenance = _provenance(report, snaps)
    observations = [{"evidence_id": o["evidence_id"], "tool": o["tool"], "field": o["field"],
                     "value": copy.deepcopy(o["value"]),
                     "unit": _unit_of(snaps[o["evidence_id"]], o["field"]), "statement": o["statement"]}
                    for o in report.observations]
    lims = {"confounders": [], "tool": [], "agent": [], "model": [], "trust": dict(TRUST)}
    for l in report.limitations:
        lims[l["source"]].append(l["text"])
        if l["source"] == "tool" and "confound" in l["text"].lower():
            lims["confounders"].append(l["text"])
    doc = {
        "schema": REPORT_SCHEMA,
        "verification": {"status": "verified", "n_evidence": len(snaps), "policy_revalidated": policy is not None,
                         "checks": ["evidence hashes recomputed", "result schemas", "trace matches evidence",
                                    "observations re-derived", "primary result matches evidence",
                                    "finding consistent with evidence", "descriptive-only policy",
                                    "tool limitations complete", "investigation limitations are fixed texts",
                                    "model prose checks", "cross-snapshot agreement"],
                         "cross_snapshot": cross},
        "question": report.question,
        "status": {"finding": report.finding, "finding_label": FINDING_LABEL[report.finding],
                   "finished": report.finished, "stopping_reason": report.stopping_reason,
                   "direction": report.direction,
                   "primary_evidence_id": report.primary_evidence["evidence_id"] if report.primary_evidence else None},
        "primary_result": primary,
        "counts": _counts(snaps, primary),
        "observations": observations,
        "primary_evidence": (None if not report.primary_evidence else
                             {"evidence_id": report.primary_evidence["evidence_id"],
                              "tool": report.primary_evidence["tool"], "status": report.primary_evidence["status"],
                              "error_codes": [x["code"] for x in snaps[report.primary_evidence["evidence_id"]]
                                              ["result"]["errors"]]}),
        "interpretations": [{"kind": "model_interpretation", "statement": i["statement"],
                             "evidence_ids": list(i["evidence_ids"])} for i in report.interpretations],
        "limitations": lims,
        "next_steps": [{"kind": "model_suggestion", "text": s} for s in report.next_steps],
        "statistical_policy": DESCRIPTIVE_POLICY,
        "provenance": provenance,
        "trace": {"stopping_reason": report.stopping_reason, "usage": copy.deepcopy(report.usage),
                  "steps": [_step(t) for t in report.trace],
                  "note": "model_text is the model's own reasoning, recorded unverified"},
        "evidence_appendix": [{"evidence_id": eid, "tool": e["tool"], "status": e["status"],
                               "summary": _summary(e), "configuration": _compact_request(e),
                               "methodology": copy.deepcopy(e["result"]["methodology"]),
                               "limitations": list(e["result"]["limitations"])} for eid, e in snaps.items()],
        "evidence_snapshots": [copy.deepcopy(e) for e in snaps.values()],
    }
    doc["executive_summary"] = _executive_summary(doc)
    return doc


def _step(t):
    keep = ("step", "kind", "elapsed_s", "tool", "outcome", "evidence_id", "refused", "stopping_reason", "model",
            "input_tokens", "output_tokens", "problem")
    out = {k: copy.deepcopy(t[k]) for k in keep if k in t}
    if t.get("text"):
        out["model_text"] = t["text"]
    return out


def _executive_summary(doc):
    st, p, prov = doc["status"], doc["primary_result"], doc["provenance"]
    out = []
    if st["finding"] == "incomplete":
        out.append(f"Incomplete: the investigation stopped before a final answer ({st['stopping_reason']}). "
                   f"No finding was made; {doc['verification']['n_evidence']} validated results were collected "
                   "and are listed in the appendix, none presented as a finding.")
    else:
        out.append(f"Finding: {st['finding_label']} (investigation finished; the finding was chosen by the "
                   "model and checked against the evidence).")
    pe = doc["primary_evidence"]
    if p is None and pe is not None:
        codes = ", ".join(pe["error_codes"]) or "none"
        out.append(f"Primary evidence ({pe['evidence_id']}, {pe['tool']}) has status {pe['status']} "
                   f"(error codes: {codes}); it carries no comparison measurement.")
    if p is not None:
        unit = p["unit"] or "unit not recorded"
        counts = (f"{_n(p['independent_sessions'], 'independent session')} paired, from "
                  f"{_n(p['n_windows_pre'], 'pre-event window')} and {_n(p['n_windows_baseline'], 'baseline window')}")
        if p["presented_as_finding"]:
            out.append(f"Primary comparison ({p['evidence_id']}): the median of per-session differences "
                       f"(pre-event minus baseline) in {p['feature']} was {_fmt(p['effect_value'])} ({unit}), "
                       f"direction {p['direction']}; {counts}.")
        elif p["effect_reported"]:
            out.append(f"Primary comparison ({p['evidence_id']}): status {p['status']}, but the finding is "
                       f"{st['finding_label'].lower()}, so no aggregate effect is presented as a finding (the "
                       f"recorded value is in the evidence appendix for audit); {counts}.")
        else:
            reasons = ", ".join(p["sufficiency"]["reasons_not_met"]) or "none recorded"
            out.append(f"Primary comparison ({p['evidence_id']}): status {p['status']}; sufficiency criteria not "
                       f"met ({reasons}), so no aggregate effect is reported; {counts}.")
        crit = p["sufficiency"]["criteria"]
        out.append(f"Sufficiency criteria ({crit['status']}): at least {_n(crit['min_sessions'], 'session')}, "
                   f"{_n(crit['min_windows_per_group'], 'window')} per group, usable fraction "
                   f"{_fmt(crit['min_usable_fraction'])}.")
    sources = ", ".join(f"{s['source']} ({s['class']})" for s in prov["event_sources"] or []) or "not recorded"
    synthetic = {True: "yes: SYNTHETIC fixture data, not experimental evidence", False: "no",
                 None: "unknown (no data provenance recorded)"}[prov["synthetic"]]
    out.append(f"Provenance: event source {sources}; synthetic data: {synthetic}; split: "
               f"{', '.join(prov['split']['values']) if prov['split']['values'] else 'not recorded'}.")
    out.append(DESCRIPTIVE_POLICY + " " + UNIT_OF_ANALYSIS)
    cross = doc["verification"]["cross_snapshot"]
    if cross["status"] in ("not_verified", "partially_verified"):
        why = Counter(x["outcome"] for x in cross["pairs"] if x["outcome"] != "agree")
        n = sum(why.values())
        out.append(f"Cross-snapshot agreement was not verified for {_n(n, 'pair')} of snapshots measuring the same "
                   f"windows ({', '.join(f'{k}: {v}' for k, v in sorted(why.items()))}; see Verification).")
    if cross["unit_conflicts"]:
        out.append("Comparisons record different units for the same feature: "
                   + "; ".join(f"{c['feature']} ({c['kind']}): "
                               + ", ".join(f"{_unit_label(r['unit'])} in {', '.join(r['evidence_ids'])}"
                                           for r in c["units"]) for c in cross["unit_conflicts"]) + ".")
    for b in doc["counts"]["behavior_bouts"] or []:
        if b["truncation_note"]:
            out.append(f"{b['evidence_id']}: {b['truncation_note']}")
    if doc["limitations"]["confounders"]:
        out.append(f"{_n(len(doc['limitations']['confounders']), 'unresolved confounder')} reported by the tools "
                   "(see Limitations).")
    return out


def to_json_text(doc):
    """Deterministic JSON text of a report document."""
    return json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def render(report, *, policy=None):
    """(JSON document, Markdown) of a verified report; raises ReportIntegrityError."""
    doc = build_report(report, policy=policy)
    return doc, render_markdown(doc)


# ---- Markdown (from the document only) ----

def _n(k, word):
    return f"{k} {word}" + ("" if k == 1 else "s")


def _fmt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return format(v, ".6g")
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    return str(v)


def _cell(v):
    return _fmt(v).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out


def _text(s):
    return str(s).replace("\n", " ")


def render_markdown(doc):
    """Markdown for a report document (deterministic, from the document alone)."""
    st, p, prov = doc["status"], doc["primary_result"], doc["provenance"]
    md = ["# Investigation report", "", f"**Research question:** {_text(doc['question'])}", ""]
    if prov["synthetic"]:
        md += ["> **SYNTHETIC DATA.** At least one result comes from a synthetic fixture; it is not "
               "experimental evidence.", ""]
    md += ["## Executive summary", ""] + [f"- {_text(s)}" for s in doc["executive_summary"]] + [""]
    md += ["## Finding and status", ""] + _table(["Field", "Value"], [
        ("Finding", st["finding_label"]), ("Finished", st["finished"]), ("Stopping reason", st["stopping_reason"]),
        ("Direction", st["direction"]), ("Primary evidence", st["primary_evidence_id"])]) + [""]
    if p is not None:
        md += ["## Primary comparison", "", f"Evidence `{p['evidence_id']}`, status `{p['status']}`.", ""]
        md += _table(["Quantity", "Value"], [
            ("Feature", p["feature"]), ("Unit", p["unit"] or "not recorded"), ("Statistic", p["statistic"]),
            ("Aggregate effect", p["effect_value"] if p["presented_as_finding"] else
             "recorded but not presented as a finding (see appendix)" if p["effect_reported"] else
             "not reported (null)"),
            ("Direction", p["direction"]), ("Independent sessions (paired)", p["independent_sessions"]),
            ("Pre-event windows used", p["n_windows_pre"]), ("Baseline windows used", p["n_windows_baseline"]),
            ("Sufficiency met", p["sufficiency"]["met"]),
            ("Reasons not met", ", ".join(p["sufficiency"]["reasons_not_met"]) or "none"),
            ("Criteria", p["sufficiency"]["criteria"]), ("Uncertainty", f"{p['uncertainty']['method']}: "
                                                         f"{p['uncertainty']['note']}"),
            ("Excluded windows", p["excluded_windows"]["n"]), ("Exclusions by reason", p["excluded_windows"]["by_reason"])])
        md += ["", "Per session (descriptive values; each row is one independent session):", ""]
        if not p["presented_as_finding"]:
            md += ["_No aggregate effect is presented as a finding: these per-session values are shown for audit "
                   "only and are not a finding._", ""]
        md += _table(["Session", "Bouts with a used pre-event window", "Pre-event windows", "Baseline windows",
                      "Pre summary", "Baseline summary", "Difference"],
                     [(r["session"], r["bouts_with_used_pre_window"], r["pre_windows_used"], r["baseline_windows_used"],
                       r["pre_summary"], r["baseline_summary"], r["difference"]) for r in p["per_session"]])
        md += ["", f"_{p['unit_of_analysis']}_", ""]
    c = doc["counts"]
    if c["behavior_bouts"]:
        md += ["## Bouts, windows and sessions", ""] + _table(
            ["Evidence", "Behavior", "Bouts (query total)", "Bouts returned", "Sessions searched",
             "Sessions with bouts (query total)", "Sessions among returned bouts", "Truncated"],
            [(b["evidence_id"], b["behavior"], b["n_bouts_total"], b["n_bouts_returned"], b["n_sessions_searched"],
              b["n_sessions_with_bouts"], b["n_sessions_among_returned_bouts"], b["truncated"])
             for b in c["behavior_bouts"]]) + [""]
        md += [f"- {b['truncation_note']}" for b in c["behavior_bouts"] if b["truncation_note"]]
        md += ["", "_Sessions with bouts (query total): sessions searched minus sessions without bouts, from the "
               "query summary._", "", f"_{c['unit_of_analysis']}_", ""]
    md += ["## Verified observations", "",
           "Values read from the hash-verified evidence snapshots (not written by the model).", ""]
    md += _table(["#", "Field", "Value", "Unit", "Evidence"],
                 [(i + 1, o["field"], o["value"], o["unit"] or "-", o["evidence_id"])
                  for i, o in enumerate(doc["observations"])]) if doc["observations"] else ["None."]
    md += ["", "## Model interpretations (not evidence; model-authored, unverified)", ""]
    md += [f"- _Interpretation (model):_ {_text(i['statement'])} (cites {', '.join(i['evidence_ids'])})"
           for i in doc["interpretations"]] or ["None."]
    md += ["", "## Limitations and confounders", ""]
    trust = doc["limitations"]["trust"]
    for key, title in (("confounders", "Confounders (unresolved; from the tools)"),
                       ("tool", f"Tool limitations ({trust['tool']})"),
                       ("agent", f"Investigation limitations ({trust['agent']})"),
                       ("model", f"Limitations added by the model ({trust['model']})")):
        md += [f"### {title}", ""] + ([f"- {_text(t)}" for t in doc["limitations"][key]] or ["None."]) + [""]
    md += ["## Suggested next steps (model suggestions, not evidence; unverified)", ""]
    md += [f"- {_text(s['text'])}" for s in doc["next_steps"]] or ["None."]
    md += ["", "## Statistical policy", "", doc["statistical_policy"], ""]
    md += ["## Provenance", ""] + _table(["Field", "Value"], [
        ("Event sources", "; ".join(f"{s['source']} ({s['class']})" for s in prov["event_sources"] or []) or "unknown"),
        ("Synthetic data", {True: "yes", False: "no", None: "unknown"}[prov["synthetic"]]),
        ("Data", "; ".join(f"{d['kind']} ({d['class']}), sha256 {d['sha256'] or 'unknown'}"
                           for d in prov["data"] or []) or "unknown"),
        ("Split", f"{', '.join(prov['split']['values']) if prov['split']['values'] else 'unknown'} "
                  f"({prov['split']['basis']})"),
        ("Annotators", ", ".join(prov["annotators"]) if prov["annotators"] else "unknown"),
        ("Classifier model IDs", ", ".join(prov["classifier"]["model_ids"]) if prov["classifier"]["model_ids"]
         else f"none ({prov['classifier']['note']})"),
        ("Investigator model IDs", ", ".join(prov["investigator_model_ids"]) if prov["investigator_model_ids"]
         else "unknown"),
        ("Behaviors", ", ".join(prov["behaviors"]) if prov["behaviors"] else "unknown"),
        ("Sessions", f"{len(prov['session_ids'] or [])}: " + ", ".join(prov["session_ids"] or [])),
        ("Tracking quality", "not assessed" if not prov["tracking_quality"]["assessed"] else
         f"{prov['tracking_quality']['n_window_assessments']} window assessments, verdicts "
         f"{_fmt(prov['tracking_quality']['verdicts'])}, smoothing contamination "
         f"{_fmt(prov['tracking_quality']['smoothing_contamination'])}; motion accuracy not verified"),
    ]) + ["", f"_{prov['session_id_note']}_", ""]
    if prov["event_definitions"]:
        md += _table(["Event definition", "Source", "Behavior", "Definition", "Merge gap", "Min duration"],
                     [(d["evidence_id"], d["source"], d["behavior"], d["episode_definition"], d["merge_gap_frames"],
                       d["min_duration_frames"]) for d in prov["event_definitions"]]) + [""]
    tr = doc["trace"]
    md += ["## Investigation trace", "", f"Stopping reason: `{tr['stopping_reason']}`. Usage: {_fmt(tr['usage'])}.", ""]
    md += _table(["Step", "Kind", "Tool", "Outcome", "Evidence / refusal"],
                 [(s["step"], s["kind"], s.get("tool", s.get("model", "")), s.get("outcome", s.get("stopping_reason", "")),
                   s.get("evidence_id") or s.get("refused") or "") for s in tr["steps"]])
    md += ["", "The model's own reasoning text for each turn is in the JSON export (unverified).", ""]
    md += ["## Evidence appendix", ""]
    for a in doc["evidence_appendix"]:
        md += [f"### `{a['evidence_id']}`", "", f"Tool `{a['tool']}`, status `{a['status']}`.", ""]
        summ = a["summary"]
        if summ.get("measurements", True) is None:          # error result: errors, no measurements
            md += _table(["Error code", "Message"], [(x["code"], x["message"]) for x in summ["errors"]])
        elif a["tool"] == "compare_windows":
            md += _table(["Summary", "Value"], [(k, summ[k]) for k in (
                "status", "feature", "unit", "effect_value", "effect_sign", "independent_sessions", "n_windows_pre",
                "n_windows_baseline", "excluded_windows")] + [("note", "audit record; a finding is stated only in "
                                                                      "the summary")])
        else:
            md += _table(["Summary", "Value"], sorted(summ.items()))
        md += ["", "<details><summary>Configuration, methodology and limitations</summary>", ""]
        md += _table(["Configuration", "Value"], sorted(a["configuration"].items())) + [""]
        md += _table(["Methodology", "Value", "Status"], [(m["name"], m["value"], m["status"])
                                                          for m in a["methodology"]]) + [""]
        md += [f"- {_text(t)}" for t in a["limitations"]] + ["", "</details>", ""]
    v = doc["verification"]
    cross = v["cross_snapshot"]
    md += ["## Verification", "", f"Cross-snapshot agreement: `{cross['status']}`.", ""]
    if cross["pairs"]:
        md += _table(["Snapshots", "Shared window values", "Features", "Units", "Outcome", "Reasons"],
                     [(" / ".join(x["evidence_ids"]), x["shared_window_features"], ", ".join(x["features"]),
                       " / ".join(_unit_label(u) for u in x["units"]), x["outcome"], "; ".join(x["reasons"]) or "-")
                      for x in cross["pairs"]]) + [""]
    for c_ in cross["unit_conflicts"]:
        md += [f"- Units for {c_['feature']} ({c_['kind']}): "
               + "; ".join(f"{_unit_label(r['unit'])}: {', '.join(r['evidence_ids'])}" for r in c_["units"])]
    md += ["---", "", f"Verified before rendering: {v['n_evidence']} evidence snapshots; "
           f"{', '.join(v['checks'])}. Policy re-validation: {'yes' if v['policy_revalidated'] else 'no'}. "
           f"Report schema `{doc['schema']}`.", ""]
    return "\n".join(md)
