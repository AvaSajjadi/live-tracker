"""Investigation page: the Contract v1 evidence behind a synthetic Replay session (new file).

Data choice (Phase 2B): the SYNTHETIC demonstration below, unchanged and the default, or the recorded Contract v1
tool run of a verified REAL validation run, shown read only by live_tracker.validation_run_ui.

Shows, for one SYNTHETIC session and one approved rule, the record of the
deterministic Contract v1 tool run (replay_investigation): the bouts found,
each bout's windows, tracking status and comparison outcome, the recorded
tool sequence, what the evidence does and does not support, and its
provenance. The record comes from replay_investigation.cached_investigation,
the same function and session cache the Replay page uses, keyed on Replay's
own identity (score-bound run id, pose digest, page config) plus the effective
investigation settings: nothing here can show evidence computed from other
inputs. No language model is called, no file path is accepted, and nothing is
written except what the Replay pipeline already writes.
"""

import altair as alt
import pandas as pd
import streamlit as st

from live_tracker import investigation_view as iv
from live_tracker import replay as rp
from live_tracker import replay_investigation as ri
from live_tracker import replay_view as rv
from live_tracker import ui_style
from live_tracker import validation_run_ui as vui
from live_tracker.rule_workflow import is_valid_workflow
from streamlit.errors import StreamlitPageNotFoundError

WORKFLOW_KEY = "rule_workflow"
REPLAY_SESSION_KEY = "replay_session"
SESSION_KEY = "investigation_session"
STATUS_BOX = {"supported": st.success, "insufficient": st.warning, "no_bouts": st.info, "unsupported": st.info,
              "error": st.error}

ui_style.apply()
st.title("Investigation")
st.caption("Inspect bouts, tracking quality and association-only evidence behind a replay.")
nav = st.columns([2, 1, 4])
try:
    nav[0].page_link("pages/2_Replay.py", label="Back to Replay", icon=":material/arrow_back:")
    nav[1].page_link("pages/1_Rules.py", label="Rules", icon=":material/rule:")
except StreamlitPageNotFoundError:                  # this file run on its own, outside the app's page list
    nav[0].caption("Replay and Rules are in the sidebar.")

# 0. data: the SYNTHETIC demonstration (below, unchanged) or a verified REAL validation run (read only)
if vui.choose_mode("investigation_data_mode_widget") == vui.REAL:
    vui.render_investigation()
    st.stop()

# 1. approved rules (the same researcher workflow the Replay page reads)
wf = st.session_state.get(WORKFLOW_KEY)
if wf is None or not is_valid_workflow(wf) or not wf.ruleset.active():
    st.warning("No approved rule yet. Propose and confirm a rule on the Rules page; the investigation uses only "
               "rules a researcher approved.")
    st.stop()
rules = wf.ruleset.active()

# 2. session (SYNTHETIC only; preselects the session open in Replay)
try:
    catalog = rv.synthetic_catalog()
except Exception as exc:                            # a broken fixture is an error, never a silent fallback
    st.error(f"The synthetic sessions could not be built ({type(exc).__name__}: {exc}).")
    st.stop()
labels = [e.label for e in catalog]
if not labels:
    st.error("No synthetic session is available.")
    st.stop()
if st.session_state.get(SESSION_KEY) not in labels:
    replay_choice = st.session_state.get(REPLAY_SESSION_KEY)
    st.session_state[SESSION_KEY] = replay_choice if replay_choice in labels else labels[0]
label = st.selectbox("Session", labels, key=SESSION_KEY)
entry = catalog[labels.index(label)]
try:
    rv.check_entry(entry)
except rp.ReplayError as exc:
    st.error(f"This session cannot be investigated: {exc}")
    st.stop()
st.warning(ri.NOTE)
st.caption(f"`{entry.session}`: {entry.description}")
pose_problem = rv.check_pose(entry)
if pose_problem is not None:
    st.error(f"Investigation evidence needs the session's pose, which is unusable: {pose_problem}.")
    st.stop()

# 3. the investigation record: the Replay page's identity, cache and function
try:
    replay_identity = rv.cache_identity(entry, rules, rv.replay_dir(), True)
    run_id, _, _ = rp.run_paths(entry.scores, entry.session, rules, rv.replay_dir())
    warnings = rv.entry_warnings(entry)
except (rp.ReplayError, KeyError, ValueError) as exc:
    st.error(f"The session's replay identity or tracking warnings could not be computed: {exc}")
    st.stop()
cache = st.session_state.get(ri.CACHE_KEY)
if not isinstance(cache, dict):
    cache = st.session_state[ri.CACHE_KEY] = {}
try:
    res = ri.cached_investigation(cache, replay_identity, entry, rules, run_id)
except ValueError as exc:
    st.error(f"The investigation could not run: {exc}")
    st.stop()

items = {item["rule_id"]: item for item in res["rules"]}
rule_by_id = {r.rule_id: r for r in rules}
rule_id = st.radio("Approved rule", list(items), horizontal=True, key=f"investigation_rule_{run_id}",
                   format_func=lambda rid: f"{rid}: {rule_by_id[rid].behavior}, at least "
                                           f"{rule_by_id[rid].min_duration_s:g} s, ask threshold "
                                           f"{rule_by_id[rid].ask_threshold:g}")
item = items[rule_id]
kind, text = iv.evidence_status(item)
STATUS_BOX[kind](text)

tab_ev, tab_trace, tab_find, tab_prov = st.tabs(["Evidence", "Trace", "Findings", "Provenance & export"])

# ---- A + C. bouts and evidence ----
with tab_ev:
    q, c = item["query"], item["compare"]
    if item["unsupported"]:
        st.caption("No evidence exists for this rule; the Trace tab shows the refused call.")
    else:
        if c is not None and c["status"] != "error":
            m = st.columns(5)
            for col, (name, key) in zip(m, (("Bouts found", "n_input"), ("Used", "n_used"), ("Excluded", "n_excluded"),
                                            ("Paired sessions", "n_sessions"), ("Pre-event windows", "n_windows"))):
                col.metric(name, c[key])
        bouts = q["bouts"]
        if not bouts:
            st.caption("The rule finds no bout in this session.")
        else:
            opts = list(range(len(bouts)))
            pick = st.selectbox("Bout", opts, format_func=lambda i: iv.bout_label(bouts[i]),
                                key=f"investigation_bout_{run_id}_{rule_id}")
            bout = bouts[pick]
            windows = iv.bout_windows(bout, c)
            outcome = iv.bout_outcome(bout, c)
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**{bout['behavior']}** · start {bout['start_s']:.2f} s · "
                            f"fired {bout['fired_s']:.2f} s · end {bout['end_s']:.2f} s ({bout['end_reason']}) · "
                            f"confidence "
                            f"{bout['confidence']:.3f} (mean score, not a probability of the behavior)")
                st.caption(f"Source: {q['data_source']} ({q['derived_from']}). "
                           f"{q['start_meaning']}.")
            with right:
                if outcome is None:
                    st.caption("Not compared.")
                elif outcome == "used":
                    st.success("Used in the comparison (both windows passed the quality and overlap "
                               "policies).")
                else:
                    st.warning(f"Excluded from the comparison: {outcome}.")
            if windows:
                st.markdown("**Windows** (anchored on the bout start, never on the firing time)")
                st.dataframe(pd.DataFrame(windows), hide_index=True, width="stretch")
            track = iv.bout_tracking(bout, warnings, windows)
            st.markdown("**Tracking status**")
            if track:
                st.dataframe(pd.DataFrame(track), hide_index=True, width="stretch")
            else:
                st.caption("No tracking warning overlaps this bout or its windows.")
            rows = pd.DataFrame(iv.timeline_rows(item, warnings, bout))
            chart = alt.Chart(rows).mark_bar(height=14).encode(
                x=alt.X("start:Q", title="time (s)", scale=alt.Scale(domainMin=0)), x2="end:Q",
                y=alt.Y("lane:N", title=None, sort=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("kind:N", title=None, scale=alt.Scale(
                    domain=["bout", "pre-event", "baseline", "tracking warning"],
                    range=["#4c78a8", "#e45756", "#72b7b2", "#f2a65a"])),
                opacity=alt.condition("datum.selected", alt.value(1.0), alt.value(0.45)),
                tooltip=["lane", "start", "end", "kind"])
            st.altair_chart(chart, width="stretch")
        if c is not None and c["status"] != "error":
            with st.expander("All exclusions, evidence gate, quality and overlap policies"):
                if c["excluded"]:
                    st.dataframe(pd.DataFrame(c["excluded"]), hide_index=True, width="stretch")
                else:
                    st.caption("No bout was excluded.")
                st.caption(f"Windows: pre {c['windows']['pre']}; baseline {c['windows']['baseline']}; "
                           f"{c['windows']['truncation']}.")
                gate = c["evidence_gate"]
                st.caption(f"Evidence gate: at least {gate['min_n_windows']} windows in {gate['min_n_sessions']} "
                           f"paired sessions (override: {gate['overridden']}).")
                st.caption(f"Quality policy: `{c['quality_policy']}` "
                           f"{res['inputs']['config']['quality_policy_definition']}.")
                st.caption(f"Overlap policy: `{c['overlap_policy']}` {c['overlap_policy_definition']}. "
                           f"Exclusion reasons, checked in this order: {', '.join(c['exclusion_order'])}.")
            if c["status"] == "ok":
                st.markdown("**Per-session paired differences** (pre minus baseline; association only)")
                st.bar_chart(pd.DataFrame(c["per_session"]).set_index("session")["paired_diff"])

# ---- B. trace ----
with tab_trace:
    st.info(iv.RUN_KIND)
    record = item["record"]
    st.caption(f"Investigation `{record['investigation_id']}`: {record['calls_used']} of {record['max_tool_calls']} "
               "tool calls used. LLM-driven runs are not started from this page.")
    st.dataframe(pd.DataFrame(iv.trace_rows(record)), hide_index=True, width="stretch")
    for a in record["attempts"]:
        with st.expander(f"Step {a['call']}: {a['tool']} — {a['status']}"):
            st.json(a.get("result", {}), expanded=False)

# ---- D. findings ----
with tab_find:
    st.info(iv.REPORT_NOTE)
    st.markdown("**What this evidence supports**")
    st.markdown(f"- {text}")
    if not item["unsupported"]:
        st.markdown(f"- Association only: every comparison describes what precedes or is associated with a "
                    f"behavior, never a cause (`association_only: {item['query']['association_only']}`).")
        st.markdown(f"- The bouts are threshold crossings of {item['query']['data_source']} scores, not verified "
                    "behavior: no annotation is involved.")
    st.markdown("- Publication status: not applicable. This is tool evidence, not a report, and nothing here is "
                "authorized for publication.")

# ---- E. provenance and export ----
with tab_prov:
    inp = res["inputs"]
    st.dataframe(pd.DataFrame([
        {"field": "session", "value": inp["session"]},
        {"field": "replay run id", "value": inp["replay_run_id"]},
        {"field": "investigation id", "value": item["investigation_id"]},
        {"field": "query run id", "value": item["query"].get("query_run_id") or "none (refused)"},
        {"field": "scores sha256", "value": inp["scores_sha256"]},
        {"field": "pose sha256", "value": inp["pose_sha256"]},
        {"field": "configuration sha256", "value": inp["config_sha256"]},
        {"field": "data source", "value": res["data_source"]}]), hide_index=True, width="stretch")
    with st.expander("Configuration snapshot (as recorded)"):
        st.json(inp["config"], expanded=False)
    st.download_button("Download investigation record (JSON)", iv.export_json(res),
                       file_name=f"{entry.session}_investigation.json", mime="application/json",
                       key=f"investigation_json_{run_id}")
    ui_style.note("caption", "The same record the Replay page exports.",
                  "No Markdown rendering exists for Contract v1 tool records; "
                  "findings/2 Markdown exists only for agent reports.")
