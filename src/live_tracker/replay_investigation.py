"""Replay -> Contract v1 investigation tools, for a SYNTHETIC replayed session (new file).

A read-only bridge from the Replay page to the Stage 9 Contract v1 adapter
(investigation_tools_v1). The replayed session's synthetic scores and pose
become one InvestigationData (data_source "synthetic"), and for each approved
rule one investigation runs through its own InvestigationRunner (12-call
budget, every attempt recorded with its result):

  query_events     behavior, min_duration_s and threshold = the rule's
                   ask_threshold, i.e. the monitor's own evidence threshold,
                   so the bouts are the replay's monitored bouts (the same
                   monitor code, run in memory under an inv_ run id)
  compare_windows  the cited query, PRESET (pre_s 2 s: a value of the
                   preselected grid; baseline 5 s; gap 1 s; dist_center)

Nothing here is a finding. One synthetic session cannot meet the evidence
gate (>= 10 windows in >= 4 sessions), so compare_windows reports
insufficient_evidence with its counts and exclusions and no effect value;
every response is association_only and labelled synthetic. No researcher rule,
replay journal, answer or file is touched.
"""

from __future__ import annotations

import copy
import hashlib
import json

from . import investigation_tools_v1 as v1
from .fake_scores import SYNTHETIC_PREFIX
from .replay_view import content_digest

__all__ = ["PRESET", "config_digest", "effective_config", "investigate_replay"]

PRESET = {"pre_s": 2.0, "baseline_s": 5.0, "baseline_gap_s": 1.0, "feature": "dist_center"}
UNSUPPORTED_ZERO = ("ask threshold 0 is not supported by Contract v1 investigation: query_events requires a threshold "
                    "in (0, 1] and refused it (invalid_request). Deterministic monitoring still runs this rule, so "
                    "the replay shows its events; no investigation evidence exists for it.")
NOTE = ("SYNTHETIC tool demonstration, not a scientific finding: one made-up session, association-only wording, "
        "and the evidence gate (at least 10 windows in 4 sessions) cannot be met here.")


def effective_config():
    """The settings an investigation will use, resolved ONCE, now (not at import): everything besides the inputs
    (scores, pose, rules, session) that can change a result. A JSON-only deep snapshot, independent of the module
    dictionaries; investigate_replay executes with exactly these values (passed to the tools explicitly), the page
    keys its cache on config_digest() of it, and every result records a copy of it."""
    policy = v1.DEFAULT_QUALITY_POLICY
    if policy not in v1.QUALITY_POLICIES:
        raise ValueError(f"the default quality policy {policy!r} is not a Contract v1 policy")
    snap = {"preset": dict(PRESET), "threshold_from": "rule.ask_threshold", "contract_version": v1.CONTRACT_VERSION,
            "quality_policy": policy, "quality_policy_definition": v1.QUALITY_POLICIES[policy].to_json(),
            "evidence_gate": {"min_n_windows": v1.DEFAULT_MIN_N_WINDOWS, "min_n_sessions": v1.DEFAULT_MIN_N_SESSIONS},
            "max_tool_calls": v1.MAX_TOOL_CALLS}
    return json.loads(json.dumps(snap, sort_keys=True))     # JSON-only, shares nothing with module state


def config_digest(config):
    """SHA-256 of a configuration snapshot (canonical JSON)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def investigate_replay(entry, rules, run_id, config=None):
    """{"note", "data_source", "inputs", "rules": [{"rule_id", "investigation_id", "query", "compare", "record",
    "unsupported"}]} for a SYNTHETIC catalog entry (replay_view.CatalogEntry) with a pose table, and the approved
    rules. config: an effective_config() snapshot (resolved now when None); execution uses only it, and
    "inputs" records exactly what the result was computed from (score and pose digests, a copy of the
    configuration and its digest).
    A rule the contract refuses keeps the refusal as its evidence and gets an "unsupported" disclosure; nothing
    is substituted."""
    if not str(entry.session).startswith(SYNTHETIC_PREFIX):
        raise ValueError(f"{entry.session!r} is not a SYNTHETIC session; the replay investigation is synthetic-only")
    if entry.pose is None:
        raise ValueError(f"{entry.session}: no pose table, so windows and tracking quality cannot be measured")
    cfg = copy.deepcopy(effective_config() if config is None else config)
    data = v1.InvestigationData.from_synthetic(entry.scores[list(v1.SCORE_COLUMNS)], entry.pose)
    out = []
    inputs = {"session": entry.session, "replay_run_id": run_id, "scores_sha256": content_digest(entry.scores),
              "pose_sha256": content_digest(entry.pose), "config_sha256": config_digest(cfg),
              "config": copy.deepcopy(cfg)}
    compare_settings = {**cfg["preset"], "quality_policy": cfg["quality_policy"], **cfg["evidence_gate"]}
    for rule in sorted(rules, key=lambda r: r.rule_id):
        runner = v1.InvestigationRunner(data, f"{run_id}_{rule.rule_id}", max_tool_calls=cfg["max_tool_calls"])
        q = runner.call("query_events", {"session": entry.session, "behavior": rule.behavior,
                                         "min_duration_s": rule.min_duration_s, "threshold": rule.ask_threshold})
        c = None
        if q["status"] == "ok" and q["bouts"]:
            c = runner.call("compare_windows", {"query_run_id": q["query_run_id"], "query": q["query"],
                                                "bouts": q["bouts"], **compare_settings})
        unsupported = None
        if q["status"] == "error":
            unsupported = (UNSUPPORTED_ZERO if rule.ask_threshold == 0 else
                           f"not investigated: query_events refused this rule ({q['code']}: {q['message']})")
        out.append({"rule_id": rule.rule_id, "investigation_id": runner.investigation_id, "query": q, "compare": c,
                    "record": runner.record(), "unsupported": unsupported})
    return {"note": NOTE, "data_source": data.data_source, "inputs": inputs, "rules": out}
