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

__all__ = ["CACHE_KEY", "PRESET", "PolicyMismatch", "cached_investigation", "config_digest", "effective_config",
           "investigate_replay"]

CACHE_KEY = "replay_investigation_cache"     # st.session_state dict shared by the Replay and Investigation pages

PRESET = {"pre_s": 2.0, "baseline_s": 5.0, "baseline_gap_s": 1.0, "feature": "dist_center"}
UNSUPPORTED_ZERO = ("ask threshold 0 is not supported by Contract v1 investigation: query_events requires a threshold "
                    "in (0, 1] and refused it (invalid_request). Deterministic monitoring still runs this rule, so "
                    "the replay shows its events; no investigation evidence exists for it.")
NOTE = ("SYNTHETIC tool demonstration, not a scientific finding: one made-up session, association-only wording, "
        "and the evidence gate (at least 10 windows in 4 sessions) cannot be met here.")


class PolicyMismatch(ValueError):
    """A configuration snapshot is missing, stale or altered, or the tools executed something other than what it
    records. Nothing is returned, cached or exported: a result never carries two different scientific definitions."""


OVERLAP_KEYS = ("overlap_policy", "overlap_policy_definition", "exclusion_order")
THRESHOLD_FROM = "rule.ask_threshold"        # query_events runs at the rule's ask threshold (the monitor's evidence)


def _canonical_overlap():
    """The overlap policy compare_windows executes NOW (name, definition, exclusion order), as JSON."""
    name = v1.DEFAULT_OVERLAP_POLICY
    if name not in v1.OVERLAP_POLICIES:
        raise PolicyMismatch(f"the default overlap policy {name!r} is not a Contract v1 policy")
    return json.loads(json.dumps({"overlap_policy": name,
                                  "overlap_policy_definition": v1.OVERLAP_POLICIES[name].to_json(),
                                  "exclusion_order": list(v1.EXCLUSION_REASONS)}))


def _canonical_config(quality_policy):
    """The exact configuration the tools execute NOW with the selected quality policy: every field that can change
    a result, each from its one executable source (the module preset, the registries, the contract constants)."""
    if not isinstance(quality_policy, str) or quality_policy not in v1.QUALITY_POLICIES:
        raise PolicyMismatch(f"the quality policy {str(quality_policy)[:60]!r} is not a Contract v1 policy "
                             f"({sorted(v1.QUALITY_POLICIES)})")
    snap = {"preset": dict(PRESET), "threshold_from": THRESHOLD_FROM, "contract_version": v1.CONTRACT_VERSION,
            "quality_policy": quality_policy,
            "quality_policy_definition": v1.QUALITY_POLICIES[quality_policy].to_json(),
            **_canonical_overlap(),
            "evidence_gate": {"min_n_windows": v1.DEFAULT_MIN_N_WINDOWS, "min_n_sessions": v1.DEFAULT_MIN_N_SESSIONS},
            "max_tool_calls": v1.MAX_TOOL_CALLS}
    return json.loads(json.dumps(snap, sort_keys=True))     # JSON-only, shares nothing with module state


def _check_config(cfg, where="the configuration snapshot"):
    """Refuse unless `cfg` is exactly the canonical configuration of the quality policy it selects: no missing,
    extra or different field (no fallback, no repair; a matching hash proves nothing)."""
    if not isinstance(cfg, dict):
        raise PolicyMismatch(f"{where} must be a JSON object")
    canon = _canonical_config(cfg.get("quality_policy"))
    missing = [k for k in canon if k not in cfg]
    if missing:
        raise PolicyMismatch(f"{where} has no {', '.join(missing)}: what it would execute cannot be verified")
    extra = sorted(set(cfg) - set(canon))
    if extra:
        raise PolicyMismatch(f"{where} has fields the tools do not execute: {', '.join(map(str, extra))[:200]}")
    differ = [k for k in canon if cfg[k] != canon[k]]
    if differ:
        raise PolicyMismatch(f"{where} disagrees with what the tools execute ({cfg['quality_policy']}, "
                             f"{canon['overlap_policy']}, {canon['contract_version']}) in {', '.join(differ)}; "
                             "stale or altered metadata is refused")


def _query_mismatch(data, session, rule, q, c, record):
    """The first way the executed query differs from the one the approved rule defines on this data source, or
    None. The canonical query (session, behavior, min_duration_s, threshold = the rule's ask threshold), its run id
    and its bouts are recomputed here (the same in-memory monitor, deterministic), never taken from the response;
    every copy the response carries (its `query`, the top-level fields, the counts) must equal them, and the
    comparison must cite exactly that query and those bouts. A refused query is not given any."""
    expected = {"session": [session], "behavior": rule.behavior, "min_duration_s": float(rule.min_duration_s),
                "threshold": float(rule.ask_threshold)}
    if q["status"] != "ok":
        return None
    if q.get("query") != expected:
        return f"query_events ran another query than the approved rule's ({q.get('query')!r} != {expected!r})"
    run_id, bouts = v1._query_run_id(data, expected), v1._run_query(data, expected)[1]
    top = {"behavior": expected["behavior"], "min_duration_s": expected["min_duration_s"],
           "threshold": expected["threshold"], "n_sessions_searched": len(expected["session"]),
           "query_run_id": run_id, "bouts": bouts, "n_bouts": len(bouts),
           "n_sessions": len({b["session"] for b in bouts}), "data_source": data.data_source,
           "derived_from": data.derived_from, "latency_gap_s": None, "start_meaning": v1.START_MEANING}
    differ = [k for k in top if k not in q or q[k] != top[k]]
    if differ:
        return f"query_events reports another {', '.join(differ)} than the approved rule's query"
    if c is not None:
        cited = [a["request"] for a in record["attempts"] if a["tool"] == "compare_windows" and a["counted"]]
        if len(cited) != 1 or not isinstance(cited[0], dict) or cited[0].get("query_run_id") != run_id or \
                cited[0].get("query") != expected or cited[0].get("bouts") != bouts:
            return "compare_windows did not cite exactly the verified query and its bouts"
        if c["status"] != "error" and c.get("n_input") != len(bouts):
            return "compare_windows analysed another number of bouts than the verified query found"
    return None


def _executed_mismatch(cfg, rule, q, c, record):
    """The first way the executed responses differ from the accepted snapshot, or None. Only what a tool reports
    having executed is compared; a refusal (status error) reports no analysis and is not given any."""
    for a in record["attempts"]:
        if a.get("result", {}).get("contract_version") != cfg["contract_version"]:
            return f"call {a['call']} ({a['tool']}) answered under another contract version"
    if record.get("contract_version") != cfg["contract_version"] or \
            record.get("max_tool_calls") != cfg["max_tool_calls"]:
        return "the attempt record has another contract version or tool-call budget"
    if c is None or c["status"] == "error":
        return None
    expected = {**{k: cfg[k] for k in ("quality_policy", "quality_policy_definition", *OVERLAP_KEYS)},
                **{k: (float(v) if k != "feature" else v) for k, v in cfg["preset"].items()}}
    missing = [k for k in expected if k not in c]
    if missing:
        return f"compare_windows reports no {', '.join(missing)}"
    differ = [k for k in expected if c[k] != expected[k]]
    gate = {k: c["evidence_gate"].get(k) for k in cfg["evidence_gate"]}
    if gate != cfg["evidence_gate"]:
        differ.append("evidence_gate")
    if differ:
        return f"compare_windows executed another {', '.join(differ)} than the configuration snapshot"
    return None


def effective_config():
    """The settings an investigation will use, resolved ONCE, now (not at import): everything besides the inputs
    (scores, pose, rules, session) that can change a result. A JSON-only deep snapshot, independent of the module
    dictionaries; investigate_replay executes with exactly these values (passed to the tools explicitly), the page
    keys its cache on config_digest() of it, and every result records a copy of it."""
    return _canonical_config(v1.DEFAULT_QUALITY_POLICY)


def config_digest(config):
    """SHA-256 of a configuration snapshot (canonical JSON)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def cached_investigation(cache, replay_identity, entry, rules, run_id):
    """The investigation for one replay identity (replay_view.cache_identity: score-bound run_id = session, rules,
    fps, score rows; pose digest; page config; storage) under the settings resolved now (effective_config, once).
    Computed once per (replay identity, config digest) and reused from `cache`, so any change of pose, scores,
    rules, session or effective settings recomputes and an ordinary rerun does not. Every page that shows
    investigation evidence goes through this one function and one cache."""
    config = effective_config()
    ident = (replay_identity, config_digest(config))
    if ident not in cache:
        cache[ident] = investigate_replay(entry, rules, run_id, config)
    return cache[ident]


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
    _check_config(cfg)                                               # before any tool call, result or cache
    data = v1.InvestigationData.from_synthetic(entry.scores[list(v1.SCORE_COLUMNS)], entry.pose)
    out = []
    inputs = {"session": entry.session, "replay_run_id": run_id, "scores_sha256": content_digest(entry.scores),
              "pose_sha256": content_digest(entry.pose), "config_sha256": config_digest(cfg),
              "config": copy.deepcopy(cfg)}
    compare_settings = {**cfg["preset"], "quality_policy": cfg["quality_policy"], **cfg["evidence_gate"]}
    for rule in sorted(rules, key=lambda r: r.rule_id):
        runner = v1.InvestigationRunner(data, f"{run_id}_{rule.rule_id}", max_tool_calls=cfg["max_tool_calls"])
        try:
            q = runner.call("query_events", {"session": entry.session, "behavior": rule.behavior,
                                             "min_duration_s": rule.min_duration_s, "threshold": rule.ask_threshold})
            c = None
            problem = _query_mismatch(data, entry.session, rule, q, None, runner.record())   # before q is used
            if problem:
                raise PolicyMismatch(f"rule {rule.rule_id}: {problem}; the result is refused")
            if q["status"] == "ok" and q["bouts"]:
                c = runner.call("compare_windows", {"query_run_id": q["query_run_id"], "query": q["query"],
                                                    "bouts": q["bouts"], **compare_settings})
        except KeyError:                                              # a policy left a registry mid-run
            _check_config(cfg, "the configuration snapshot (re-checked after a failed call)")
            raise
        problem = (_query_mismatch(data, entry.session, rule, q, c, runner.record()) or
                   _executed_mismatch(cfg, rule, q, c, runner.record()))
        if problem:
            raise PolicyMismatch(f"rule {rule.rule_id}: {problem}; the result is refused")
        unsupported = None
        if q["status"] == "error":
            unsupported = (UNSUPPORTED_ZERO if rule.ask_threshold == 0 else
                           f"not investigated: query_events refused this rule ({q['code']}: {q['message']})")
        out.append({"rule_id": rule.rule_id, "investigation_id": runner.investigation_id, "query": q, "compare": c,
                    "record": runner.record(), "unsupported": unsupported})
    _check_config(cfg, "the configuration snapshot (re-checked after execution)")   # no drift during the run
    return {"note": NOTE, "data_source": data.data_source, "inputs": inputs, "rules": out}
