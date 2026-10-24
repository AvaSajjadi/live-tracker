"""Streamlit rendering of a verified REAL validation run on the Replay and Investigation pages (new file).

The pages offer two data modes: the existing SYNTHETIC demonstration (unchanged) and a verified real validation
run (validation_run_view.load_run). In real mode a run is chosen by its directory name from the runs directory and
re-verified on every rerun; if it does not verify, the reason is shown and nothing else (no synthetic fallback).
Everything shown is a recorded value of that run: nothing is re-run, re-scored or re-classified, and no model is
called. Inspecting a completed run is not an approval: the run's own recorded approval is shown as such.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from . import validation_run_view as vv

MODE_KEY = "data_mode"                                   # shared by the two pages (plain state, not a widget)
SYNTHETIC, REAL = "Synthetic demonstration", "Verified real validation run"
MODES = (SYNTHETIC, REAL)
RUN_KEY = "validation_run"                               # the selected run directory name, shared too
FEATURE_LABEL = vv.FEATURE


def choose_mode(widget_key):
    """The data mode radio (synthetic is the default); the choice is kept for the other page."""
    current = st.session_state.get(MODE_KEY, SYNTHETIC)
    mode = st.radio("Data", MODES, index=MODES.index(current) if current in MODES else 0, horizontal=True,
                    key=widget_key)
    st.session_state[MODE_KEY] = mode
    return mode


def choose_run(widget_key):
    """The verified run, or None after showing why there is none (never a fallback)."""
    root = vv.runs_dir()
    names = vv.list_runs(root)
    if not names:
        st.error(f"No validation run found under {root.name}/. Produce one with scripts/validation_run_v1.py "
                 "(see docs/validation_run_v1.md). Nothing is shown in its place.")
        return None
    current = st.session_state.get(RUN_KEY)
    name = st.selectbox("Validation run", names, index=names.index(current) if current in names else len(names) - 1,
                        key=widget_key)
    st.session_state[RUN_KEY] = name
    try:
        run = vv.load_run(name, root)
    except vv.RunUnavailable as exc:
        st.error(f"Run {name!r} cannot be shown: {exc}. Nothing is substituted.")
        return None
    except (OSError, ValueError) as exc:
        st.error(f"Run {name!r} cannot be read ({type(exc).__name__}: {exc}). Nothing is substituted.")
        return None
    st.success(f"Run {name!r}: the recorded execution is CONSISTENT (status 'verified'; file digests, journal, "
               "snapshot, Contract v1 record and summary agree with each other and with the rules they claim).")
    st.caption("Not independently authenticated: these are local files and no signature or registry binds them to "
               "the run that executed. Recorded-only claims are labelled as such.")
    return run


def disclosures():
    with st.expander("What this run is and is not", expanded=True):
        for d in vv.DISCLOSURES:
            st.markdown(f"- {d}")
    with st.expander("Verification levels"):
        for title, text in vv.VERIFICATION_LEVELS:
            st.markdown(f"**{title}.** {text}")


def recorded_evaluation(run):
    """The recorded event-level evaluation, labelled as an unverified claim (or why it is incomplete)."""
    ev = run.evaluation
    m = ev["metrics"]
    st.markdown("**Recorded event-level evaluation against the validation annotations: an UNVERIFIED claim** (the "
                "annotations are not re-read here; the values are only checked for arithmetic consistency with the "
                "journal's events; the rule was chosen on these same sessions, so they would be optimistic)")
    st.dataframe(pd.DataFrame([{"events": m["n_events"], "precision": round(m["event_precision"], 3),
                                "true attack bouts": m["n_true_bouts"], "detected": m["detected_true"],
                                "recall": round(m["event_recall"], 3), "F1": round(m["event_f1"], 3),
                                "false alerts / h": m.get("false_alerts_per_hour"),
                                "start offset median (s)": m.get("start_offset_median_s"),
                                "fire offset median (s)": m.get("fire_offset_median_s")}]),
                 hide_index=True, width="stretch")
    if ev["reason"]:
        st.warning(f"Per-session annotation coverage: {ev['reason']}.")


def _rule(run):
    r = run.header["rules"][0]
    return (f"{r['rule_id']}: {r['behavior']}, ask threshold {r['ask_threshold']:g}, at least "
            f"{r['min_duration_s']:g} s, alert threshold {r['alert_threshold']:g}, mode {r['alert_mode']}")


def render_replay():
    """Replay, real mode: the recorded monitoring run (no new monitoring run, no approval)."""
    st.info("REAL validation data: a COMPLETED monitoring run is browsed. Nothing is re-run; the model and the "
            "monitor are not called. Starting a new monitoring run still requires an explicit researcher approval "
            "(scripts/validation_run_v1.py --confirm); browsing is not one.")
    run = choose_run("replay_validation_run")
    if run is None:
        return
    disclosures()
    st.markdown(f"**Approved rule (as recorded):** {_rule(run)}")
    st.caption(f"Recorded approval: {run.summary['rule'].get('approval', 'not recorded')}")
    m, d = run.summary["monitor"], run.derived                   # counts derived from the journal
    cols = st.columns(5)
    for col, (label, value) in zip(cols, (("Sessions", len(run.validation_sessions)),
                                           ("Events (predicted)", d["n_events"]),
                                           ("Alerts", d["actions"].get("alert", 0)),
                                           ("Asks", d["actions"].get("ask", 0)),
                                           ("Ended by tracking loss", d["events_ended_by_tracking_loss"]))):
        col.metric(label, value)
    st.caption("Event, alert, ask and end-reason counts are derived from the journal itself.")
    st.markdown("**Sessions** (events derived from the journal; tracking interruptions and annotated attack bouts "
                "are recorded, unverified values; the events are classifier predictions)")
    st.dataframe(pd.DataFrame(vv.session_rows(run)), hide_index=True, width="stretch")
    labels = {sid.rsplit("/", 1)[-1]: sid for sid in run.validation_sessions}
    pick = st.selectbox("Validation session", list(labels), key="replay_validation_session")
    rows = vv.event_rows(run, labels[pick])
    st.markdown(f"**Recorded events in {pick}** (journal order, recorded ids)")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("The rule fired no event in this session.")
    ti = m["tracking_interruptions"].get(labels[pick], {})
    st.caption(f"Recorded tracking interruptions in {pick} (unverified here): {ti.get('tracking_lost_frames')} "
               f"frames without tracking_ok in {ti.get('tracking_lost_runs')} runs (a tracking_ok = false frame ends a "
               "bout).")
    recorded_evaluation(run)
    with st.expander("Journal and score-drop provenance"):
        st.dataframe(pd.DataFrame(vv.provenance_rows(run)), hide_index=True, width="stretch")


def render_investigation():
    """Investigation, real mode: the recorded Contract v1 tool run of the verified run."""
    st.info("REAL validation data: the recorded Contract v1 tool run of a verified validation run. The calls were a "
            "fixed, deterministic sequence chosen before any value was seen; no language model chose or "
            "interpreted them, and nothing is re-run here.")
    run = choose_run("investigation_validation_run")
    if run is None:
        return
    disclosures()
    q = run.query
    tab_q, tab_cmp, tab_trace, tab_find, tab_prov = st.tabs(["Query & bouts", "Comparisons", "Trace", "Findings",
                                                             "Provenance & export"])
    with tab_q:
        st.markdown(f"**query_events** `{q['query_run_id']}`: behavior {q['query']['behavior']}, threshold "
                    f"{q['query']['threshold']:g} (the rule's ask threshold), minimum {q['query']['min_duration_s']:g} "
                    f"s, {len(q['query']['session'])} validation sessions searched.")
        st.caption(f"{q['n_bouts']} bouts in {q['n_sessions']} sessions, exactly the journal's events (identity and "
                   f"frame boundaries checked). {q['start_meaning']}.")
        st.dataframe(pd.DataFrame([{"event id": b["event_id"], "session": b["session"].rsplit("/", 1)[-1],
                                    "start (s)": round(b["start_s"], 4), "fired (s)": round(b["fired_s"], 4),
                                    "end (s)": round(b["end_s"], 4), "end reason": b["end_reason"]}
                                   for b in q["bouts"]]), hide_index=True, width="stretch")
        quality, feats = run.probe("check_tracking_quality"), run.probe("get_window_features")
        if quality is not None:
            st.markdown(f"**Probe window** {quality['session'].rsplit('/', 1)[-1]} [{quality['t0_s']:g}, "
                        f"{quality['t1_s']:g}) s (the first bout whose baseline starts at or after 0 s): "
                        f"check_tracking_quality {quality['verdict']} under {quality['quality_policy']} (usable "
                        f"fraction {quality['usable_fraction']:.3f}, {quality['n_usable_frames']} of "
                        f"{quality['n_frames']} frames; longest frozen / out-of-frame run "
                        f"{quality['longest_veto_run_s']:g} s); get_window_features {json.dumps(feats['summary'])}.")
            st.caption("The verdict was re-decided from these recorded counts with Contract v1's own rule "
                       "(quality_reasons) and the approved policy definition, and matches. The counts and flags "
                       "themselves are recorded values: re-measuring them would need the pose data.")
    with tab_cmp:
        st.dataframe(pd.DataFrame(vv.comparison_rows(run)), hide_index=True, width="stretch")
        first = run.comparisons[0] if run.comparisons else None
        if first is not None:
            cfg = vv.canonical_configuration(q["data_source"], q["derived_from"])   # the record equals it (checked)
            w = cfg["windows"]
            st.caption(f"Windows (Contract v1's definitions; every comparison was checked to carry exactly these): "
                       f"pre {w['pre']}; baseline {w['baseline']}; anchored on {w['anchor']}; {w['truncation']}. "
                       f"Feature `{cfg['feature']}`, baseline {cfg['baseline_s']:g} s, gap "
                       f"{cfg['baseline_gap_s']:g} s. "
                       f"Quality policy `{cfg['quality_policy']}` {cfg['quality_policy_definition']}. Overlap policy "
                       f"`{cfg['overlap_policy']}` {cfg['overlap_policy_definition']}. Exclusion reasons, checked in "
                       f"this order: {', '.join(cfg['exclusion_order'])}.")
        for c in run.comparisons:
            role = "primary" if c is first else "follow-up"
            with st.expander(f"pre_s {c['pre_s']:g} s ({role}): {c['status']}"):
                if c["status"] == "ok":
                    st.success(f"Evidence gate met: {c['n_windows']} usable windows in {c['n_sessions']} sessions "
                               f"(gate >= {c['evidence_gate']['min_n_windows']} windows in >= "
                               f"{c['evidence_gate']['min_n_sessions']} sessions). Median session-paired difference "
                               f"{c['mean_diff']:.4g} ({c['feature']}). Association only.")
                    st.markdown("**Contributing sessions**")
                    st.dataframe(pd.DataFrame(vv.per_session_rows(run, c["pre_s"])), hide_index=True, width="stretch")
                else:
                    st.warning(f"Insufficient evidence ({c['reason']}): {c['n_available']} available, "
                               f"{c['n_required']} required (gate {c['failed_gate']}). No effect value is reported.")
                st.markdown(f"**Excluded windows** ({c['n_excluded']} of {c['n_input']})")
                st.dataframe(pd.DataFrame(vv.exclusion_rows(run, c["pre_s"])), hide_index=True, width="stretch")
    with tab_trace:
        st.caption("The recorded attempts, in order, with the SHA-256 of each recorded result (each verified).")
        st.dataframe(pd.DataFrame(vv.trace_rows(run)), hide_index=True, width="stretch")
        for a in run.record["attempts"]:
            with st.expander(f"Call {a['call']}: {a['tool']} ({a['status']})"):
                st.json(a["result"], expanded=False)
    with tab_find:
        st.markdown("**What the recorded evidence supports** (consistency-checked, not independently authenticated)")
        st.markdown(f"- Associations between `{FEATURE_LABEL}` before PREDICTED attack bouts and an earlier baseline "
                    "window, session-paired; no causal claim and no confidence interval.")
        for c in run.comparisons:
            if c["status"] == "ok":
                st.markdown(f"- pre_s {c['pre_s']:g} s: the evidence gate is met ({c['n_windows']} windows in "
                            f"{c['n_sessions']} sessions); median session-paired difference {c['mean_diff']:.4g}.")
            else:
                st.markdown(f"- pre_s {c['pre_s']:g} s: inconclusive (insufficient evidence; no effect).")
        st.markdown("- The bouts are classifier predictions. How many of them are real attacks is NOT established "
                    "here: the event-level evaluation and the per-session annotation counts are recorded, unverified "
                    "claims, and no conclusion about false alerts or attack prevalence is drawn from them.")
        st.markdown("- Publication status: not applicable. This is a recorded tool run, not an authorized report.")
    with tab_prov:
        st.dataframe(pd.DataFrame(vv.provenance_rows(run)), hide_index=True, width="stretch")
        st.download_button("Download the recorded Contract v1 record (JSON; consistency-checked, not authenticated)",
                           json.dumps(run.record, indent=2, sort_keys=True, default=str) + "\n",
                           file_name=f"{run.name}_contract_v1_record.json", mime="application/json",
                           key="investigation_validation_record")
