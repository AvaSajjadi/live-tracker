"""Investigation page: presentation of existing investigation records (new file).

Pure functions that turn the record produced by replay_investigation (the
deterministic Contract v1 tool run behind the Replay page) into rows, labels
and chart data for pages/3_Investigation.py. Nothing here computes evidence,
re-runs a tool, fills a missing value or changes a result: every number shown
comes from the record (or, for tracking warnings, from replay_view's existing
warning source), and anything the record does not contain is reported as
not available.

What the record is, and is not
------------------------------
- A deterministic, scripted tool sequence (query_events, then compare_windows
  on that query) chosen by code, not by a language model: there is no model
  reasoning to show, and none is shown (RUN_KIND).
- Not a findings/2 report: those are produced only by the bounded agent loop
  (investigation_agent) over the verified validation cache, and a saved
  report is verified only under the evidence policy it was built with
  (investigation_report_v2.verify_saved_v2). REPORT_NOTE says so on the page.
"""

from __future__ import annotations

import json

from . import timebase as tb

__all__ = ["REPORT_NOTE", "RUN_KIND", "bout_label", "bout_outcome", "bout_tracking", "bout_windows",
           "evidence_status", "export_json", "timeline_rows", "trace_rows"]

RUN_KIND = ("Deterministic scripted tool sequence (no language model was called): query_events, then "
            "compare_windows on the cited query, with fixed settings. There is no model reasoning to show.")
REPORT_NOTE = ("No findings/2 report exists for this session. Reports (claims, observations, authorization) come "
               "only from the bounded agent loop over the verified CalMS21 validation cache; Replay sessions are "
               "synthetic and are investigated with the Contract v1 tools only. Live (paid) model runs are not "
               "started from this page.")


def export_json(res):
    """The investigation record exactly as the Replay page exports it."""
    return json.dumps(res, indent=2, sort_keys=True)


# ---- trace ----

def _outcome(attempt):
    if attempt["status"] == "exception":
        return "failed (programming error)"
    if attempt["status"] == "error":
        if attempt["code"] == "budget_exhausted":
            return "refused: budget exhausted"
        return f"refused: {attempt['code']}"
    return f"executed: {attempt['status']}"


def _request_summary(req):
    if not isinstance(req, dict):
        return str(req)[:120]
    parts = []
    for k in sorted(req):
        v = req[k]
        if k == "bouts" and isinstance(v, list):
            parts.append(f"bouts: {len(v)} (the cited query's, unchanged)")
        elif k == "query" and isinstance(v, dict):
            continue                                        # shown through query_run_id
        else:
            parts.append(f"{k}: {v}")
    return "; ".join(parts)


def trace_rows(record):
    """One row per recorded attempt, in order."""
    return [{"step": a["call"], "tool": a["tool"], "outcome": _outcome(a),
             "counted toward budget": a["counted"], "arguments": _request_summary(a["request"]),
             "result sha256": (a.get("result_sha256") or "")[:12]} for a in record["attempts"]]


# ---- evidence ----

def evidence_status(item):
    """(kind, text) with kind in supported / insufficient / no_bouts / unsupported / error."""
    if item["unsupported"]:
        return "unsupported", item["unsupported"]
    q, c = item["query"], item["compare"]
    if c is None:
        return "no_bouts", f"query_events found {q['n_bouts']} bout(s); compare_windows was not run."
    if c["status"] == "error":
        return "error", f"compare_windows refused ({c['code']}): {c['message']}"
    if c["status"] == "insufficient_evidence":
        return "insufficient", (f"Insufficient evidence ({c['reason']}): {c['n_available']} available, "
                                f"{c['n_required']} required (gate {c['failed_gate']}). No effect value is reported.")
    return "supported", (f"Comparison ran: {c['n_bouts']} bout(s) in {c['n_sessions']} session(s); median "
                         f"session-paired difference {c['mean_diff']:.4g} ({c['feature']}). Association only.")


def bout_label(bout):
    return (f"{bout['start_s']:.2f} s – {bout['end_s']:.2f} s · {bout['behavior']} · "
            f"ended {bout['end_reason']}")


def bout_outcome(bout, compare):
    """'used', an exclusion reason, or None when no comparison ran."""
    if compare is None or compare["status"] == "error":
        return None
    for e in compare["excluded"]:
        if e["session"] == bout["session"] and e["start_s"] == bout["start_s"]:
            return e["why"]
    return "used"


def bout_windows(bout, compare):
    """The pre-event and baseline windows of one bout, as compare_windows defines them (absolute endpoints in
    seconds, each converted by the 30 fps rule), or None when no comparison ran."""
    if compare is None or compare["status"] == "error":
        return None
    s = bout["start_s"]
    pre = (s - compare["pre_s"], s)
    b1 = pre[0] - compare["baseline_gap_s"]
    base = (b1 - compare["baseline_s"], b1)
    out = []
    for kind, (a, z) in (("pre-event", pre), ("baseline", base)):
        frames = (f"[{tb.boundary_frame(a)}, {tb.boundary_frame(z)})" if a >= 0 else "starts before 0 s")
        out.append({"window": kind, "from (s)": round(a, 4), "to (s)": round(z, 4), "frames": frames})
    return out


def _overlap(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def bout_tracking(bout, warnings, windows=None):
    """Tracking warnings (replay_view.entry_warnings) overlapping the bout or its windows."""
    spans = [("bout", bout["start_s"], bout["end_s"])]
    for w in windows or []:
        spans.append((w["window"], max(w["from (s)"], 0.0), w["to (s)"]))
    rows = []
    for w in warnings:
        hit = [name for name, a, z in spans if _overlap(w.start_s, w.end_s, a, z)]
        if hit:
            rows.append({"warning": w.kind, "mouse": w.mouse or "both", "from (s)": round(w.start_s, 3),
                         "to (s)": round(w.end_s, 3), "overlaps": ", ".join(hit)})
    return rows


def timeline_rows(item, warnings, selected=None):
    """Interval rows for the timeline chart: the query's bouts, the selected bout's windows, tracking warnings."""
    rows = [{"lane": f"{item['rule_id']} bouts", "start": b["start_s"], "end": b["end_s"], "kind": "bout",
             "selected": selected is not None and b["event_id"] == selected["event_id"]}
            for b in (item["query"].get("bouts") or [])]
    if selected is not None:
        for w in bout_windows(selected, item["compare"]) or []:
            if w["to (s)"] <= 0:                    # wholly before the session: nothing to draw (the table says so)
                continue
            rows.append({"lane": f"{w['window']} window", "start": max(w["from (s)"], 0.0), "end": w["to (s)"],
                         "kind": w["window"], "selected": True})
    for w in warnings:
        rows.append({"lane": f"tracking: {w.kind} ({w.mouse or 'both'})", "start": w.start_s, "end": w.end_s,
                     "kind": "tracking warning", "selected": False})
    return rows
