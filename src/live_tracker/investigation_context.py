"""EXPERIMENTAL (Investigation mode, Findings v2 checkpoint M0): the shared
validation foundation used by Stage 6 (investigation_agent) and Stage 7
(investigation_report). New file; docs/findings_v2_design.md sections 2, 3, 9.

What lives here (one implementation of each rule):
- the observation field grammar and resolver (MAX_FIELD, FIELD_RE,
  resolve_field), used by Stage 6 observations and Stage 7 re-derivation;
- snapshot validation per layer: content consistency (L0: the evidence_id is
  the hash of the snapshot's content; NOT proof of tool origin), schema (L1)
  and policy authorization (L2), with an AuthorizationAssessment;
- the ExecutionAssessment of a trace: "prefix" mode (at a freeze point, no
  stop event yet) and "complete" mode (exactly one final stop event that
  matches the report); an incomplete run never needs an accepted finish;
- cross-snapshot data identity, measurement comparability and agreement,
  behavior-scoped coverage, and typed tracking facts (moved unchanged from
  Stage 7);
- strict format detection (legacy/1 vs findings/2), recursive freezing, and
  the versioned canonical content fingerprint (evidence and claims only;
  execution history, stopping reason and authorization are kept separate).

No claim registry, no report prose and no finish-schema change (M1-M3).
"""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
import re
import types
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import investigation_contracts as ic
from .investigation_data import INDEX_NAME

# ---- the observation field grammar (shared by Stage 6 and Stage 7) ----

MAX_FIELD = 200
FIELD_RE = re.compile(r"(status|errors|measurements)(\.[A-Za-z_][A-Za-z0-9_]*|\[\d{1,4}\])*")


def resolve_field(result, path):
    """The value at a field path of a result; KeyError if the path is not
    well-formed or not present."""
    return _resolve(result, path)


# methodology items defining how a window value is computed: two snapshots'
# values are comparable only when all of these are recorded and identical
PROCESSING_METHOD = ("event_source", "causal_computation", "tracking_flags", "score_validation", "tracking_ok",
                     "interpolation", "smoothing", "smoother_memory_frames", "fps", "feature_lookback_frames")


_HEX64 = re.compile(r"[0-9a-f]{64}")


TRACKING_REASONS = ("insufficient_usable_frames", "missing_data")


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


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _sign(v):
    return "negative" if v < 0 else "positive" if v > 0 else "zero"


def _method(result, name):
    return next((m["value"] for m in result["methodology"] if m["name"] == name), None)


def _resolve(result, path):
    """The value of a field path (Stage 6 observation grammar) in a result."""
    if not isinstance(path, str) or len(path) > MAX_FIELD or not FIELD_RE.fullmatch(path):
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


def _provenance_entries(e):
    """(path -> (kind, sha256), None) for a snapshot's data provenance, or
    (None, problem). Every entry is validated before the mapping is built;
    paths are normalized; a path recorded twice with a different kind or
    sha256 is a conflict whatever the order; identical duplicates collapse."""
    try:
        data = e["result"]["provenance"]["data"]
    except (KeyError, TypeError):
        return None, "data provenance is missing or malformed"
    if not isinstance(data, list) or not data:
        return None, "data provenance is not recorded with hashes"
    entries = {}
    for d in data:
        if not isinstance(d, dict):
            return None, "data provenance has a malformed entry"
        kind, path, sha = d.get("kind"), d.get("path"), d.get("sha256")
        if sha is None or sha == "":
            return None, "data provenance is not recorded with hashes"
        if not (isinstance(kind, str) and kind and isinstance(path, str) and path.strip()
                and isinstance(sha, str) and _HEX64.fullmatch(sha)):
            return None, "data provenance has a malformed entry (kind, path and sha256 are required)"
        key = posixpath.normpath(path.strip())
        if key in entries and entries[key] != (kind, sha):
            return None, f"data provenance records conflicting entries for {key}"
        entries[key] = (kind, sha)
    return entries, None


def _data_identity_problem(a, b):
    """None when two snapshots provably read the same verified data, else why
    not. Shared by the cross-snapshot check and the coverage provenance chain.
    Identity = both record valid, conflict-free hashed data provenance with
    exactly one verified extraction index, the index digests (hash of the
    whole verified index: source sha256, manifest binding, every session file)
    are equal, and any path recorded by both has the same kind and sha256.
    Entries only one records (other sessions' files) are allowed. Units and
    processing are comparability, not identity: they are not looked at here."""
    recorded = []
    for e in (a, b):
        entries, problem = _provenance_entries(e)
        if problem:
            return problem
        index = [p for p in entries if p.endswith(INDEX_NAME)]
        if len(index) != 1:
            return "data provenance does not record exactly one verified extraction index"
        recorded.append((entries, entries[index[0]]))
    (paths_a, index_a), (paths_b, index_b) = recorded
    if index_a != index_b or any(paths_a[p] != paths_b[p] for p in set(paths_a) & set(paths_b)):
        return "data provenance differs"
    return None


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
    problem = _data_identity_problem(a, b)        # the same notion of identity as coverage provenance
    if problem:
        reasons.append(problem)
    return reasons


def _agree(x, y):
    (_, v1, f1, u1), (_, v2, f2, u2) = x, y
    if (f1, u1) != (f2, u2) or (v1 is None) != (v2 is None):
        return False
    return v1 is None or ic._close(v1, v2)


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


def _tracking_concerns(snaps, primary_id):
    """Tracking-quality facts of the evidence, each with its source (no
    severity is assigned: every recorded non-zero count is listed)."""
    out = []

    def add(eid, source, text, note=None, window_ids=None):
        out.append({"evidence_id": eid, "source": source, "text": text, "tool_note": note,
                    "window_ids": window_ids, "primary": eid is not None and eid == primary_id})
    for eid, e in snaps.items():
        res, m = e["result"], e["result"]["measurements"]
        if m is None:
            continue
        if e["tool"] == "compare_windows":
            for kind, label in (("pre", "pre-event window"), ("baseline", "baseline window")):
                item = next((x for x in res["methodology"] if x["name"] == f"smoothing_contamination_windows_{kind}"),
                            None)
                n = _num(item and item["value"])
                if n:
                    add(eid, f"methodology.smoothing_contamination_windows_{kind}",
                        f"{_n(n, 'used ' + label)} with smoothing contamination", item["note"])
                item = next((x for x in res["methodology"] if x["name"] == f"usable_frame_fraction_{kind}"), None)
                f = _num(item and item["value"])
                if f is not None and f < 1:
                    add(eid, f"methodology.usable_frame_fraction_{kind}",
                        f"usable-frame fraction over used {label}s {_fmt(f)}", item["note"])
            for reason, k in sorted(Counter(x["reason"] for x in m["excluded_windows"]
                                            if x["reason"] in TRACKING_REASONS).items()):
                add(eid, "measurements.excluded_windows", f"{_n(k, 'window')} excluded: {reason} (first reason)")
        elif e["tool"] == "get_window_features":
            counts = Counter((w["kind"], r) for w in m["windows"] for r in w["exclusion_reasons"]
                             if r in TRACKING_REASONS)
            for (kind, reason), k in sorted(counts.items()):
                add(eid, "measurements.windows[].exclusion_reasons",
                    f"{_n(k, kind.replace('_', '-') + ' window')} with exclusion reason {reason} (reasons can overlap)")
        elif e["tool"] == "check_tracking_quality":
            by_verdict = {}
            for w in m["windows"]:
                if w["verdict"] != "usable_with_caveats":
                    by_verdict.setdefault(w["verdict"], []).append(w["window_id"])
            for verdict, ids in sorted(by_verdict.items()):
                add(eid, "measurements.windows[].verdict", f"{_n(len(ids), 'window')} with verdict {verdict}",
                    window_ids=ids)
            # one item per distinct note, quoting only the note of the contaminated windows it counts
            by_note = {}
            for w in m["windows"]:
                if w["smoothing_contamination"]["status"] == "detected":
                    by_note.setdefault(w["smoothing_contamination"]["note"], []).append(w["window_id"])
            for note, ids in sorted(by_note.items()):
                add(eid, "measurements.windows[].smoothing_contamination",
                    f"{_n(len(ids), 'window')} with detected smoothing contamination", note, window_ids=ids)
    quality = [(eid, e) for eid, e in snaps.items() if e["tool"] == "check_tracking_quality"]
    for eid, e in quality:
        if e["result"]["measurements"] is None:             # attempted, but no assessment came back
            codes = ", ".join(x["code"] for x in e["result"]["errors"]) or "none recorded"
            add(eid, "status", f"check_tracking_quality was called but returned status {e['status']} (error codes: "
                               f"{codes}); it assessed no window")
    if not quality:
        add(None, "trace", "check_tracking_quality was never called in this investigation: per-window tracking "
                           "quality was not assessed beyond the counts other tools record")
    elif all(e["result"]["measurements"] is None for _, e in quality):
        add(None, "trace", "check_tracking_quality was called but no call returned an assessment: per-window "
                           "tracking quality was not assessed beyond the counts other tools record")
    return out


def _recorded_coverage(e):
    """(eligible, available, session_coverage) a tool recorded, or Nones."""
    r = e["result"]
    return (_num(_method(r, "eligible_sessions")), _num(_method(r, "available_sessions")),
            _method(r, "session_coverage"))


def _definition(e):
    r = e["request"]
    return {"source": r["source"], "behavior": r["behavior"], "merge_gap_frames": r["merge_gap_frames"],
            "min_duration_frames": r["min_duration_frames"]}


def _group_coverage(definition, members, policy):
    """Coverage of one compatible query group (never mixed with another)."""
    searched = sorted({s for e in members for s in e["result"]["measurements"]["sessions_searched"]})
    with_behavior = sorted({s for e in members for s in e["result"]["measurements"]["sessions_searched"]
                            if s not in e["result"]["measurements"]["sessions_without_episodes"]})
    recorded = [_recorded_coverage(e) for e in members]
    partial_cache = any(c not in (None, "complete") for _, _, c in recorded)
    eligible = None
    if policy is not None and definition["source"] == "validation_annotation":
        eligible = sorted(sid for sid, sp in policy.split_of.items() if sp == policy.annotation_split)
        eligible_n, basis = len(eligible), f"split manifest via the evidence policy ({policy.annotation_split} split)"
    elif recorded and all(n is not None for n, _, _ in recorded) and len({n for n, _, _ in recorded}) == 1:
        eligible_n, basis = recorded[0][0], "recorded by the tools (no policy given)"
    else:
        eligible_n, basis = None, "not established for this source"
    available = min((a for _, a, _ in recorded if a is not None), default=None)
    identity = next((f"{p} between {x['evidence_id']} and {y['evidence_id']}"
                     for x, y in zip(members, members[1:]) if (p := _data_identity_problem(x, y))), None)
    if eligible is not None:
        status = "complete" if searched == eligible and not partial_cache else "partial"
    elif eligible_n is not None:
        status = ("complete" if len(searched) == eligible_n and all(c == "complete" for _, _, c in recorded)
                  else "partial" if partial_cache or len(searched) < eligible_n else "unknown")
    else:
        status = "partial" if partial_cache else "unknown"
    if identity:                                 # queries over different data are never combined into coverage
        status, basis = "unknown", f"{basis}; the group's queries do not share data identity ({identity})"
    return {**definition, "query_evidence_ids": [e["evidence_id"] for e in members], "status": status,
            "basis": basis, "eligible_count": eligible_n, "eligible": eligible, "available_count": available,
            "searched": searched, "not_searched": sorted(set(eligible) - set(searched)) if eligible is not None
            else None, "with_behavior": with_behavior,
            "truncated": any(e["result"]["measurements"]["truncated"] for e in members)}


def _primary_coverage(snaps, primary, groups, query_groups):
    """The coverage that applies to the primary comparison, from its provenance."""
    comp = snaps[primary["evidence_id"]]
    rows = {r["session"]: r for r in primary["per_session"]}
    contributing = sorted(sid for sid, r in rows.items() if r["paired"])
    out = {"status": "unknown", "reason": None, "group": None, "contributing": contributing, "excluded": None,
           "features_evidence_ids": [], "query_evidence_ids": []}
    identity_failed = False
    chain_partial = [eid for eid in [comp["evidence_id"]]
                     if _recorded_coverage(comp)[2] not in (None, "complete")]
    pre_ids = comp["request"]["pre_window_ids"]
    parsed = [ic.parse_window_id(w) for w in pre_ids]
    needed = {p["episode_id"] for p in parsed}
    kinds = {(p["source"], p["behavior"]) for p in parsed}
    feats = [e for e in snaps.values() if e["tool"] == "get_window_features" and e["result"]["measurements"]
             and set(pre_ids) <= {w["window_id"] for w in e["result"]["measurements"]["windows"]}]
    out["features_evidence_ids"] = [e["evidence_id"] for e in feats]
    chain_partial += [e["evidence_id"] for e in feats if _recorded_coverage(e)[2] not in (None, "complete")]
    if not feats:
        out["reason"] = "no get_window_features result of this run produced the compared pre-event windows"
    elif len(kinds) != 1:
        out["reason"] = "the compared windows come from more than one event source or behavior"
    else:
        (source, behavior), = kinds
        covering = [i for i, (definition, members) in enumerate(query_groups)
                    if (definition["source"], definition["behavior"]) == (source, behavior)
                    and needed <= {x["episode_id"] for e in members for x in e["result"]["measurements"]["episodes"]}]
        if not covering:
            out["reason"] = (f"no compatible query_events group of this run contains every compared {behavior} "
                             "episode (missing, truncated or differently defined queries)")
        elif len(covering) > 1 and len({groups[i]["status"] for i in covering}) > 1:
            out["reason"] = "the compared episodes appear in several query groups with different coverage"
        else:
            g = groups[covering[0]]
            members = query_groups[covering[0]][1]
            links = [(comp, f) for f in feats] + [(f, q) for f in feats for q in members]
            identity = next((f"{p} between {x['evidence_id']} and {y['evidence_id']}"
                             for x, y in links if (p := _data_identity_problem(x, y))), None)
            if identity:                         # matching IDs alone never establish the same data
                identity_failed = True
                out.update(query_evidence_ids=g["query_evidence_ids"],
                           reason="the comparison's data identity is not established along its provenance chain: "
                                  + identity)
            else:
                out.update(status=g["status"], group=covering[0], query_evidence_ids=g["query_evidence_ids"])
                if g["truncated"] and g["status"] == "complete":
                    out.update(status="partial", reason="the originating query was truncated: not every bout of "
                                                        "the searched sessions was returned, so not every bout "
                                                        "could be compared")
                out["excluded"] = []
                for sid in g["with_behavior"]:
                    if sid in contributing:
                        continue
                    r = rows.get(sid)
                    reason = ("no included pre-event window" if r is None or not r["pre_windows_used"] else
                              "no eligible baseline window" if not r["baseline_windows_used"] else "not paired")
                    out["excluded"].append({"session": sid, "reason": reason})
    if chain_partial:                    # disclosed in every case; never lifts an unestablished identity
        if not identity_failed:
            out["status"] = "partial"
        out["reason"] = (out["reason"] + "; " if out["reason"] else "") + \
            f"partial cache coverage recorded by {', '.join(chain_partial)}"
    return out


def _coverage(snaps, policy, primary):
    """Per-group scientific coverage, the primary comparison's coverage, and an
    operational list of sessions touched by any tool."""
    query_groups = []
    for e in snaps.values():
        if e["tool"] != "query_events" or e["result"]["measurements"] is None:
            continue
        definition = _definition(e)
        for d, members in query_groups:
            if d == definition:
                members.append(e)
                break
        else:
            query_groups.append((definition, [e]))
    groups = [_group_coverage(d, members, policy) for d, members in query_groups]
    prim = _primary_coverage(snaps, primary, groups, query_groups) if primary is not None else None
    partial_anywhere = any(_recorded_coverage(e)[2] not in (None, "complete") for e in snaps.values())
    if prim is not None:
        status = prim["status"]
    elif len(groups) == 1:
        status = groups[0]["status"]
    elif groups:
        status = "per_query_group"
    else:
        status = "partial" if partial_anywhere else "unknown"
    touched = sorted({sid for e in snaps.values() for sid in _sources_and_sessions(e)[2]})
    return {"status": status, "primary": prim, "groups": groups,
            "sessions_touched": {"count": len(touched), "sessions": touched,
                                 "note": "operational: sessions read by any tool for any behavior; not scientific "
                                         "coverage"}}


# ======================================================================
# validation layers (L0 content consistency, L1 schema, L2 authorization)
# ======================================================================

SNAPSHOT_KEYS = frozenset({"evidence_id", "tool", "status", "request", "result"})


def content_consistency(evidence):
    """evidence_ids whose request / result snapshot does not hash to its
    evidence_id (L0; an unkeyed hash: consistency, not authentic origin)."""
    bad = []
    for e in evidence:
        try:
            ok = (e["result"]["evidence_id"] == e["evidence_id"]
                  == ic.evidence_id(e["tool"], e["request"], e["result"]))
        except Exception:
            ok = False
        if not ok:
            bad.append(e.get("evidence_id") if isinstance(e, dict) else None)
    return bad


def policy_digest(policy):
    """A digest of the evidence policy's manifest binding and settings."""
    if policy is None:
        return None
    return hashlib.sha256(canonical_bytes({"split_of": dict(sorted(policy.split_of.items())),
                                           "annotation_split": policy.annotation_split,
                                           "allow_synthetic": policy.allow_synthetic})).hexdigest()


@dataclass(frozen=True)
class AuthorizationAssessment:
    """L2 per snapshot: "rechecked" (validate_result under the policy),
    "not_rechecked" (no policy: schema only, diagnostic) or "refused"."""
    status: str
    per_snapshot: Any            # frozen mapping evidence_id -> status
    policy_digest: str | None


def validate_snapshots(evidence, policy):
    """(snapshots {evidence_id: snapshot} in evidence order, problems,
    AuthorizationAssessment). Every entry is checked for shape, uniqueness,
    schema (L1) and, with a policy, authorization (L2)."""
    snaps, problems, per = {}, [], {}
    if not isinstance(evidence, list):
        return None, ["evidence is not a list"], AuthorizationAssessment("refused", freeze({}), policy_digest(policy))
    for i, e in enumerate(evidence):
        where = f"evidence[{i}]"
        if not isinstance(e, dict) or set(e) != SNAPSHOT_KEYS:
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
            per[str(eid)] = "refused"
            problems.append(f"{where} ({str(eid)[:80]}): failed verification: {str(exc)[:200]}")
            continue
        except Exception as exc:
            per[str(eid)] = "refused"
            problems.append(f"{where}: failed verification ({type(exc).__name__})")
            continue
        snaps[eid] = e
        per[eid] = "not_rechecked" if policy is None else "rechecked"
    status = "refused" if any(v == "refused" for v in per.values()) or problems else \
        ("not_rechecked" if policy is None else "rechecked")
    return snaps, problems, AuthorizationAssessment(status, freeze(per), policy_digest(policy))


def evidence_policy_problems(snaps):
    """Stage 5's descriptive-only policy: no interval and no uncertainty method."""
    problems = []
    for eid, e in snaps.items():
        if e["tool"] == "compare_windows" and e["result"]["measurements"] is not None:
            eff = e["result"]["measurements"]["effect"]
            if e["request"]["uncertainty"]["method"] != "none" or \
                    any(eff[k] is not None for k in ("ci_low", "ci_high", "ci_level")):
                problems.append(f"{eid}: an interval or uncertainty method outside the descriptive policy")
    return problems


def policy_metadata_problems(snaps, policy):
    """Recorded coverage metadata must agree with a given policy."""
    if policy is None:
        return []
    eligible = sum(1 for sp in policy.split_of.values() if sp == policy.annotation_split)
    problems = []
    for eid, e in snaps.items():
        n = _num(_method(e["result"], "eligible_sessions"))
        if n is not None and n != eligible:
            problems.append(f"{eid}: records {n} eligible sessions, the policy has {eligible}")
    return problems


# ======================================================================
# execution assessment (separate from the content fingerprint)
# ======================================================================

EXECUTION_MODES = ("prefix", "complete")


@dataclass(frozen=True)
class ExecutionAssessment:
    mode: str                     # "prefix" (freeze point, no stop event) or "complete"
    status: str                   # "consistent" or "inconsistent"
    problems: tuple
    tool_calls: int               # executed calls (with evidence)
    refusals: tuple               # ((tool, refused kind), ...)
    model_ids: tuple
    accepted_finish: bool
    stopping_reason: str | None   # complete mode: the stop event's reason


def assess_execution(trace, snaps, *, mode, stopping_reason=None, finished=None):
    """Check a trace against the validated snapshots.

    Both modes: the executed tool calls are exactly the snapshots, in order,
    with the same tool and request; at most one accepted finish, and nothing
    executed after it.
    prefix: no stop event (the trace up to an accepted finish, or up to report
    construction for an incomplete run).
    complete: exactly one stop event, last; its reason equals
    `stopping_reason`; finished == (reason == "finished") == an accepted
    finish exists. The messages of the first three checks are the ones Stage 7
    has always reported."""
    if mode not in EXECUTION_MODES:
        raise ValueError(f"mode must be one of {EXECUTION_MODES}")
    shape = _trace_shape_problems(trace)
    if shape:                                    # nothing in a malformed trace is interpreted
        return ExecutionAssessment(mode=mode, status="inconsistent", problems=tuple(shape), tool_calls=0,
                                   refusals=(), model_ids=(), accepted_finish=False, stopping_reason=None)
    problems = []
    entries = trace
    calls = [t for t in entries if t.get("kind") == "tool_call" and "evidence_id" in t]
    if [t["evidence_id"] for t in calls] != list(snaps):
        problems.append("the trace's executed tool calls do not match the evidence")
    elif any(t.get("request") != snaps[t["evidence_id"]]["request"] or t.get("tool") != snaps[t["evidence_id"]]["tool"]
             for t in calls):
        problems.append("a traced request differs from its evidence snapshot")
    stops = [i for i, t in enumerate(entries) if t.get("kind") == "stop"]
    finishes = [i for i, t in enumerate(entries) if t.get("kind") == "finish" and t.get("outcome") == "accepted"]
    if mode == "complete":
        if not trace or trace[-1].get("kind") != "stop" or trace[-1].get("stopping_reason") != stopping_reason:
            problems.append("the trace does not end with the report's stopping reason")
        if len(stops) > 1:
            problems.append("the trace has more than one stop event")
        if not isinstance(finished, bool):
            problems.append("a completed trace is assessed against a boolean finished flag")
        else:                                    # finished <=> reason "finished" <=> one accepted finish
            if finished != (stopping_reason == "finished"):
                problems.append("finished and the stopping reason disagree")
            if finished != (len(finishes) == 1):
                problems.append("an accepted finish exists exactly when the investigation finished")
    elif stops:
        problems.append("an execution prefix has no stop event")
    if len(finishes) > 1:
        problems.append("the trace has more than one accepted finish")
    elif finishes and any(t.get("kind") == "tool_call" and "evidence_id" in t for t in entries[finishes[0] + 1:]):
        problems.append("a tool call was executed after the accepted finish")
    refusals = tuple((t.get("tool"), t.get("refused")) for t in entries if t.get("outcome") == "refused")
    models = tuple(sorted({t["model"] for t in entries if t.get("kind") == "model_turn" and t.get("model")}))
    return ExecutionAssessment(mode=mode, status="inconsistent" if problems else "consistent",
                               problems=tuple(problems), tool_calls=len(calls), refusals=refusals, model_ids=models,
                               accepted_finish=bool(finishes),
                               stopping_reason=entries[stops[-1]].get("stopping_reason") if stops else None)


def _trace_shape_problems(trace):
    """Why a trace cannot be interpreted at all (empty: every entry is an
    object with a string kind)."""
    if not isinstance(trace, list):
        return [f"the trace is not a list ({type(trace).__name__})"]
    bad = [i for i, t in enumerate(trace) if not isinstance(t, dict) or not isinstance(t.get("kind"), str)]
    return [f"malformed trace entries at positions {bad[:5]}: each entry is an object with a string kind"] \
        if bad else []


def prefix_extends(prefix, complete):
    """True when `complete` is `prefix` followed by exactly one stop event."""
    return isinstance(prefix, list) and isinstance(complete, list) and len(complete) == len(prefix) + 1 \
        and complete[:len(prefix)] == prefix and isinstance(complete[-1], dict) and complete[-1].get("kind") == "stop"


# ======================================================================
# immutability, canonical bytes, the content fingerprint
# ======================================================================

def freeze(obj):
    """A recursively immutable view of JSON-like data: dict -> MappingProxyType,
    list / tuple -> tuple. Scalars are returned as they are."""
    if isinstance(obj, (dict, types.MappingProxyType)):
        return types.MappingProxyType({k: freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(freeze(v) for v in obj)
    return obj


def thaw(obj):
    """A plain, mutable deep copy of frozen (or plain) JSON-like data."""
    if isinstance(obj, (dict, types.MappingProxyType)):
        return {k: thaw(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [thaw(v) for v in obj]
    return copy.deepcopy(obj)


def canonical_bytes(obj):
    """The canonical serialization: sorted keys, compact separators, UTF-8,
    no NaN / infinity (ValueError)."""
    return json.dumps(thaw(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


FINGERPRINT_VERSION = "fp/1"
FINGERPRINT_FINDINGS = ("effect_described", "insufficient_evidence", "inconclusive", "incomplete")


class FingerprintError(ValueError):
    """The input cannot be fingerprinted (malformed or incoherent)."""


def fingerprint_payload(*, evidence, finding, primary_evidence_id, observations=(), claims=(),
                        next_investigations=(), findings_format="findings/2", registry_version="claims/2.0",
                        grid_version="none"):
    """The canonical evidence-and-claim content F (design section 3.1).
    Evidence is sorted by evidence_id; observations, claims and next
    investigations keep their submitted order. An incomplete run has a null
    primary and empty observations, claims and next investigations. Nothing
    from the trace, stopping reason, usage, policy or authorization enters."""
    if finding not in FINGERPRINT_FINDINGS:
        raise FingerprintError(f"finding must be one of {FINGERPRINT_FINDINGS}")
    if finding == "incomplete" and (primary_evidence_id is not None or observations or claims or next_investigations):
        raise FingerprintError("an incomplete run has a null primary and no observations, claims or next "
                               "investigations")
    rows, seen = [], set()
    for e in evidence:
        if not isinstance(e, (dict, types.MappingProxyType)) or set(e) != SNAPSHOT_KEYS:
            raise FingerprintError("evidence entries are request / result snapshots")
        if e["evidence_id"] in seen:
            raise FingerprintError(f"duplicate evidence_id {e['evidence_id']!r}")
        seen.add(e["evidence_id"])
        rows.append([e["evidence_id"], e["tool"], thaw(e["request"]), thaw(e["result"])])
    if primary_evidence_id is not None and primary_evidence_id not in seen:
        raise FingerprintError("the primary evidence is not one of the snapshots")

    def shaped(items, keys, what):
        out = []
        for x in items:
            if not isinstance(x, (dict, types.MappingProxyType)) or set(x) != keys:
                raise FingerprintError(f"{what} entries have exactly the keys {sorted(keys)}")
            out.append(thaw(x))
        return out
    return {"fingerprint_version": FINGERPRINT_VERSION, "findings_format": findings_format,
            "registry_version": registry_version, "grid_version": grid_version,
            "evidence": sorted(rows, key=lambda r: r[0]), "finding": finding,
            "primary_evidence_id": primary_evidence_id,
            "observations": shaped(observations, {"evidence_id", "field"}, "observation"),
            "claims": shaped(claims, {"type", "evidence_ids", "params"}, "claim"),
            "next_investigations": shaped(next_investigations, {"action", "params", "because"},
                                          "next investigation")}


def content_fingerprint(payload):
    """sha256 (hex) of the canonical bytes of a fingerprint payload."""
    if not isinstance(payload, (dict, types.MappingProxyType)) or \
            payload.get("fingerprint_version") != FINGERPRINT_VERSION:
        raise FingerprintError("not a fingerprint payload")
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


# ======================================================================
# strict format detection (design section 9)
# ======================================================================

LEGACY_FORMAT = "legacy/1"
V2_FORMAT = "findings/2"
KNOWN_FORMATS = frozenset({V2_FORMAT})
KNOWN_REGISTRIES = frozenset({"claims/2.0"})
KNOWN_GRIDS = frozenset({"none"})
V2_ONLY_FIELDS = ("findings_format", "claims", "next_investigations", "fingerprint", "registry_version",
                  "grid_version")
LEGACY_FIELDS = frozenset({"question", "stopping_reason", "finished", "finding", "direction", "primary_evidence",
                           "observations", "interpretations", "evidence", "limitations", "next_steps", "usage",
                           "trace"})
LEGACY_REQUIRED = LEGACY_FIELDS - {"trace"}


class FormatError(ValueError):
    """A report payload in an unknown, mixed or malformed format."""


def detect_format(data):
    """"legacy/1" or "findings/2" for a report payload (a mapping of its
    fields), or FormatError. A missing format is legacy only when every
    v2-only field is absent and the fields are exactly the legacy ones."""
    if not isinstance(data, (dict, types.MappingProxyType)):
        raise FormatError("a report payload is a mapping of fields")
    keys = set(data)
    if "findings_format" not in keys:
        present = sorted(keys & set(V2_ONLY_FIELDS))
        if present:
            raise FormatError(f"mixed payload: no findings_format but v2-only fields {present}")
        unknown, missing = sorted(keys - LEGACY_FIELDS), sorted(LEGACY_REQUIRED - keys)
        if unknown or missing:
            raise FormatError(f"not a legacy report (unknown fields {unknown}, missing fields {missing})")
        return LEGACY_FORMAT
    for marker, known in (("findings_format", KNOWN_FORMATS), ("registry_version", KNOWN_REGISTRIES),
                          ("grid_version", KNOWN_GRIDS)):
        value = data.get(marker)
        # type first: only a non-empty string can be a marker (no lists, mappings, numbers, booleans, null)
        if not isinstance(value, str) or not value or value not in known:
            shown = repr(value)[:40] if isinstance(value, str) else type(value).__name__
            raise FormatError(f"unknown {marker} {shown}: must be one of {sorted(known)}")
    for prose in ("interpretations", "next_steps"):
        value = data.get(prose)
        if value is not None and not isinstance(value, (list, tuple)):
            raise FormatError(f"malformed {prose}: not a list")
        if value:
            raise FormatError("mixed payload: a findings/2 report carries legacy model-authored prose")
    limitations = data.get("limitations")
    if limitations is not None and not isinstance(limitations, (list, tuple)):
        raise FormatError("malformed limitations: not a list")
    if any(isinstance(l, (dict, types.MappingProxyType)) and l.get("source") == "model" for l in limitations or ()):
        raise FormatError("mixed payload: a findings/2 report carries legacy model-authored prose")
    return V2_FORMAT


# ======================================================================
# the validated context
# ======================================================================

@dataclass(frozen=True)
class ValidatedContext:
    """Everything evidence-level that Stage 6 and Stage 7 validate, frozen.
    Built only from snapshots that passed L0-L2; the cross-snapshot result
    excludes contradictions (those refuse before a context exists)."""
    format_version: str
    snapshots: Any               # frozen {evidence_id: snapshot}, evidence order
    authorization: AuthorizationAssessment
    execution: ExecutionAssessment
    cross_snapshot: Any          # frozen {status, pairs, unit_conflicts}
    coverage: Any                # frozen behavior-scoped coverage
    tracking: Any                # frozen typed tracking facts
    primary_id: str | None

    def snapshot_dicts(self):
        """Plain deep copies of the snapshots, in evidence order."""
        return {eid: thaw(e) for eid, e in self.snapshots.items()}


def primary_rows(snapshot):
    """The per-session facts of a comparison that coverage needs."""
    m = snapshot["result"]["measurements"]
    return {"evidence_id": snapshot["evidence_id"],
            "per_session": [{"session": s["session"], "paired": s["difference"] is not None,
                             "pre_windows_used": len(s["pre_windows"]),
                             "baseline_windows_used": len(s["baseline_windows"])} for s in m["per_session"]]}


def build_context(*, format_version, snaps, authorization, execution, cross_snapshot, policy, primary_evidence):
    """Assemble the frozen context from already validated parts. Coverage and
    tracking are computed here, on plain copies, then frozen."""
    plain = {eid: thaw(e) for eid, e in snaps.items()}
    primary_id = primary_evidence["evidence_id"] if primary_evidence else None
    coverage_primary = None
    if primary_evidence and primary_evidence["tool"] == "compare_windows" and \
            plain[primary_id]["result"]["measurements"] is not None:
        coverage_primary = primary_rows(plain[primary_id])
    cross = {k: v for k, v in thaw(cross_snapshot).items() if k != "contradictions"}
    return ValidatedContext(format_version=format_version, snapshots=freeze(plain), authorization=authorization,
                            execution=execution, cross_snapshot=freeze(cross),
                            coverage=freeze(_coverage(plain, policy, coverage_primary)),
                            tracking=freeze(_tracking_concerns(plain, primary_id)), primary_id=primary_id)


def context_from_evidence(evidence, *, policy, primary_evidence_id, trace, mode, stopping_reason=None,
                          finished=None, format_version=V2_FORMAT):
    """(ValidatedContext, []) or (None, problems): every M0 layer applied to a
    run's evidence and trace (snapshot shape, L0 content consistency, L1
    schema, L2 authorization when a policy is given, execution in `mode`,
    descriptive-only policy, policy metadata, cross-snapshot contradictions).
    Used by the claim registry (findings/2); the legacy renderer keeps its own
    sequencing for byte-identical output."""
    try:
        snaps, problems, authorization = validate_snapshots(evidence, policy)
        if snaps is None or problems:
            return None, problems or ["evidence is not valid"]
        bad = content_consistency(evidence)
        if bad:
            problems.append(f"evidence hashes do not match: {bad[:3]}")
        execution = assess_execution(trace, snaps, mode=mode, stopping_reason=stopping_reason, finished=finished)
        problems += list(execution.problems)
        problems += evidence_policy_problems(snaps) + policy_metadata_problems(snaps, policy)
        cross = _reconcile(snaps)
        problems += cross["contradictions"]
        if primary_evidence_id is not None and primary_evidence_id not in snaps:
            problems.append("the primary evidence is not one of the snapshots")
        if problems:
            return None, problems
        primary = None if primary_evidence_id is None else \
            {"evidence_id": primary_evidence_id, "tool": snaps[primary_evidence_id]["tool"]}
        return build_context(format_version=format_version, snaps=snaps, authorization=authorization,
                             execution=execution, cross_snapshot=cross, policy=policy,
                             primary_evidence=primary), []
    except Exception as exc:                     # malformed evidence never escapes as another exception
        return None, [f"the evidence could not be validated ({type(exc).__name__})"]


__all__ = ["AuthorizationAssessment", "ExecutionAssessment", "FINGERPRINT_VERSION", "FingerprintError",
           "FormatError", "KNOWN_FORMATS", "KNOWN_GRIDS", "KNOWN_REGISTRIES", "LEGACY_FORMAT", "V2_FORMAT",
           "V2_ONLY_FIELDS", "ValidatedContext", "assess_execution", "build_context", "canonical_bytes",
           "content_consistency", "content_fingerprint", "context_from_evidence", "detect_format", "fingerprint_payload", "freeze",
           "prefix_extends", "resolve_field", "thaw", "validate_snapshots"]
