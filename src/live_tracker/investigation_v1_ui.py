"""Streamlit rendering of the Live Agent page (Phase 3D, new file). All logic is investigation_v1_runs'.

A new investigation: question, verified validation inputs, model, turn and tool budgets, local USD target and
key-limit mode; then the free metadata preflight (no model request); then, only for the exact configuration that
preflight approved, the paid run, which needs an explicit confirmation of the target AND the typed authorization
text (choosing best-effort mode authorizes nothing). Each preflight issues a one-use token and the runs directory is
locked while an investigation runs, so a double click, a rerun or a second browser tab cannot start a second paid
run. A free rehearsal (the emulated endpoint, no model, no cost) exercises the same pipeline.

A saved investigation is always reloaded from disk and re-verified offline before anything of it is shown; one that
does not verify shows only why. Reopening never calls the network or the model.
"""

from __future__ import annotations

import json
import secrets

import pandas as pd
import streamlit as st

from . import investigation_agent_v1 as av1
from . import investigation_live_v1 as lv1
from . import investigation_v1_runs as runs

PREFLIGHT_KEY = "live_agent_preflight"            # {"digest", "ready", "report", "token"}
USED_KEY = "live_agent_used_tokens"
GEN_KEY = "live_agent_generation"                 # renews the confirmation widgets after every run
OPEN_KEY = "live_agent_open"                      # the saved investigation to show (its directory)
MESSAGE_KEY = "live_agent_message"
STATUS_BOX = {"verified": st.success, "incomplete": st.warning, "unverified": st.error}


def _state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---- a new investigation ----

def configure():
    """(RunConfig, ValidationInputs) from the widgets."""
    inputs = runs.default_inputs()
    st.markdown("**1. Research question and data**")
    question = st.text_area("Research question", value=runs.RunConfig.question, max_chars=500,
                            key="live_agent_question")
    st.selectbox("Validation inputs", [inputs.label], key="live_agent_inputs",
                 help="The verified Phase 3B.1 inputs: the validation member only, the expected archive SHA-256, "
                      "the verified pose cache and the tracked split manifest. The test split is never read.")
    for p in inputs.problems():
        st.error(f"Inputs: {p}. Nothing is substituted.")
    st.markdown("**2. Model and budgets**")
    c = st.columns(4)
    model = c[0].selectbox("Model (OpenRouter)", runs.MODELS, key="live_agent_model")
    turns = c[1].number_input("Max model turns", 1, 12, 8, key="live_agent_turns")
    tools = c[2].number_input("Tool-call budget", 1, 12, 8, key="live_agent_tools")
    target = c[3].number_input("Local USD target", 0.01, lv1.MAX_TARGET_USD, lv1.MAX_TARGET_USD, step=0.05,
                               format="%.2f", key="live_agent_target")
    mode = st.radio("Key-limit mode", ("strict", "best-effort"), horizontal=True, key="live_agent_key_mode",
                    help="strict (default): the OpenRouter key's own limit must be at most $0.50. best-effort: an "
                         "existing non-resetting key limited to at most $50, its limit acknowledged.")
    ack = None
    if mode == "best-effort":
        ack = st.number_input("Acknowledge the key's actual limit (USD)", 0.51, lv1.MAX_BEST_EFFORT_KEY_LIMIT_USD,
                              lv1.MAX_BEST_EFFORT_KEY_LIMIT_USD, key="live_agent_ack")
        st.warning(f"Best-effort mode: the ${target:.2f} target is NOT a spending ceiling. Spending can exceed it, "
                   f"potentially up to the available credit on the ${ack:g} key, subject to OpenRouter's enforcement. "
                   "Choosing this mode does not authorize a paid run.")
    check = st.checkbox(f"Reference compatibility check against {runs.REFERENCE_RUN} (optional, exact matching)",
                        value=True, key="live_agent_reference")
    config = runs.RunConfig(question=question.strip(), model=model, max_model_turns=int(turns),
                            max_tool_calls=int(tools), target_usd=float(target), key_mode=mode,
                            acknowledged_key_limit_usd=ack, validation_run=runs.REFERENCE_RUN if check else None)
    return config, inputs


def show_preflight(report):
    if "refused" in report:
        st.error(f"Refused before any request: {report['refused']}")
        return
    sp = report.get("spending") or {}
    (st.success if report["ready"] else st.error)(
        "Ready: every check passed (metadata only; no model request was made)." if report["ready"] else
        f"Not ready: {'; '.join(map(str, report['problems']))[:1500]}")
    m = st.columns(4)
    m[0].metric("Key-limit mode", sp.get("authorization_mode", "?"))
    m[1].metric("Local target", f"${sp.get('local_target_usd', 0):.2f}")
    m[2].metric("Provider key limit", "unverified" if sp.get("provider_key_limit_usd") is None
                else f"${sp['provider_key_limit_usd']:g}")
    m[3].metric("Remaining credit", "unverified" if sp.get("provider_key_remaining_usd") is None
                else f"${sp['provider_key_remaining_usd']:g}")
    if sp.get("exposure"):
        st.caption(f"Exposure: {sp['exposure']}")
    est = report.get("estimate") or {}
    if est:
        st.caption(f"Estimate: typical run ~${est.get('typical_run_usd')}, worst case "
                   f"${est.get('estimated_worst_case_usd')} (client-side estimate, not a cap); Stage 6 cost cap "
                   f"${est.get('stage6_cost_cap_usd')} plus a one-request reserve ${est.get('one_request_reserve_usd')}.")
    with st.expander("Preflight details (confirmed metadata, local configuration, assumptions)"):
        st.markdown("**Only a real model response can show:**")
        for a in report.get("requires_a_model_response", []):
            st.markdown(f"- {a}")
        st.json({k: report.get(k) for k in ("plan", "confirmed_provider_metadata", "credentials", "estimate",
                                            "local_configuration")}, expanded=False)


def _run(config, inputs, paid, **kw):
    try:
        with st.spinner("Investigating (the agent chooses each Contract v1 call)..."):
            name, summary = runs.execute(config, inputs, paid=paid, **kw)
    except runs.RunBusy as exc:
        st.session_state[MESSAGE_KEY] = ("error", f"Not started: {exc}.")
        return
    except Exception as exc:                       # the CLI's refusals: before any request or file
        st.session_state[MESSAGE_KEY] = ("error", f"Not started ({type(exc).__name__}): {exc}")
        return
    st.session_state[OPEN_KEY] = str(runs.live_dir() / name)
    st.session_state.pop("live_agent_saved", None)          # preselect the new run (that widget renders later)
    st.session_state[MESSAGE_KEY] = ("info", f"Investigation {name} finished with recorded status "
                                              f"`{summary.get('status')}`. Open the Saved investigations tab: the "
                                              "report is reloaded from disk and re-verified before it is shown.")


def render_new():
    config, inputs = configure()
    digest = config.digest(inputs)
    st.markdown("**3. Free metadata preflight** (model list, key status, endpoint list; never a model request)")
    st.caption(f"OPENROUTER_API_KEY is {'set' if runs.key_present() else 'NOT set'} in the app's environment "
               "(its value is never shown or written).")
    if st.button("Run preflight", key="live_agent_preflight_button"):
        ready, report = runs.preflight(config, inputs)
        st.session_state[PREFLIGHT_KEY] = {"digest": digest, "ready": ready, "report": report,
                                           "token": secrets.token_hex(8)}
    pf = st.session_state.get(PREFLIGHT_KEY)
    if pf is not None:
        if pf["digest"] != digest:
            st.info("The configuration changed since the last preflight: run the preflight again.")
        else:
            show_preflight(pf["report"])
    st.markdown("**4. Paid investigation**")
    used, gen = _state(USED_KEY, set()), _state(GEN_KEY, 0)
    busy = runs.RunLock(runs.live_dir()).holder() is not None
    fresh = pf is not None and pf["digest"] == digest and pf["ready"] and pf["token"] not in used
    confirm = st.checkbox(f"I confirm ONE paid investigation with a local target of ${config.target_usd:.2f} "
                          "(a best-effort target, not a guaranteed maximum charge)", key=f"live_agent_confirm_{gen}")
    phrase = st.text_input("Type the authorization text exactly", key=f"live_agent_phrase_{gen}",
                           placeholder=lv1.PAID_RUN_AUTHORIZATION)
    st.caption(f'Authorization text: "{lv1.PAID_RUN_AUTHORIZATION}"')
    if busy:
        st.warning("Another investigation is running: wait for it to finish.")
    elif not fresh:
        st.caption("A paid run needs a ready preflight of exactly this configuration (one paid run per preflight).")
    if st.button("Start paid investigation", type="primary", key=f"live_agent_start_{gen}",
                 disabled=not (fresh and confirm and phrase == lv1.PAID_RUN_AUTHORIZATION and not busy)):
        used.add(pf["token"])                      # consumed before anything runs: no second submission
        st.session_state[GEN_KEY] = gen + 1
        _run(config, inputs, True, confirm_budget_usd=config.target_usd, authorization=phrase)
        st.rerun()
    st.markdown("**Free rehearsal** (the same pipeline over the emulated OpenRouter endpoint: no model, no network, "
                "no cost; the offline investigator answers)")
    if st.button("Run free rehearsal", key=f"live_agent_rehearse_{gen}", disabled=busy):
        st.session_state[GEN_KEY] = gen + 1
        _run(config, inputs, False, confirm_budget_usd=None, authorization=None)
        st.rerun()


# ---- a saved investigation ----

def render_reference(saved):
    st.markdown(f"**Reference compatibility check against {runs.REFERENCE_RUN}** (separate from the investigation; "
                "exact initial request and result matching, fail closed)")
    ref = runs.reference_check(saved)
    if ref["recorded"] is not None:
        st.caption(f"Recorded at run time: {ref['recorded']['status']}"
                   + (f" ({ref['recorded']['message']})" if ref["recorded"].get("message") else ""))
    elif saved.kind != "offline":
        st.caption("No reference check was requested for this run; the check below is computed now, offline.")
    (st.success if ref["equivalent"] else st.warning)(f"Verdict: {ref['verdict']}")
    if ref.get("explanation"):
        st.markdown(ref["explanation"])
    if ref["rows"]:
        st.dataframe(pd.DataFrame(ref["rows"]), hide_index=True, width="stretch")
        st.caption("'not sent' means the model left the parameter to Contract v1's default; 'applied' is the value "
                   "the tool recorded in its result.")


def render_cost(saved):
    c = runs.cost_view(saved)
    if c is None:
        st.caption("No model was called (offline deterministic investigator): no cost.")
        return
    m = st.columns(4)
    m[0].metric("Actual spend (provider-reported)", f"${c['actual_spend_usd']:.5f}"
                + ("" if c["actual_spend_complete"] else " +unknown"))
    m[1].metric("Local target", f"${c['target_usd']:.2f}")
    m[2].metric("Provider key limit", "none (no provider)" if c["provider_key_limit_usd"] is None
                else f"${c['provider_key_limit_usd']:g}")
    m[3].metric("Over target", "yes" if c["over_target"] else "no")
    st.caption(f"Key-limit mode `{c['authorization_mode']}`. {c['exposure']}. {c['note']}")
    if saved.kind == "rehearsal":
        st.info("Rehearsal: tokens and costs are EMULATED, nothing was spent.")
    st.dataframe(pd.DataFrame(runs.ledger_rows(saved.summary)), hide_index=True, width="stretch")


def render_saved(saved):
    if not saved.verified:
        st.error("This saved investigation does NOT verify, so none of its findings, evidence or costs are shown: "
                 + "; ".join(map(str, saved.problems[:5])))
        return
    d = saved.doc
    kind, text = runs.autonomous_status(saved)
    STATUS_BOX[kind](f"**Autonomous investigation.** {text}")
    st.caption(f"Recorded run status `{saved.summary.get('status')}`"
               + (" (the reference compatibility check failed; see that tab: this concerns run5 equivalence, not the "
                  "investigation's own validity)" if saved.summary.get("status") == "refused" and kind != "unverified"
                  else "") + f". Question: {d['question']}")
    tabs = st.tabs(["Finding", "Tool sequence", "Evidence", "Cost ledger", "Reference check (run5)",
                    "Provenance & limitations", "Report"])
    with tabs[0]:
        st.markdown(f"**Finding:** `{d['finding']}`. {av1.FINDING_TEXT[d['finding']]}")
        if d["primary_evidence_id"]:
            st.caption(f"Primary evidence `{d['primary_evidence_id']}`.")
        if d["observations"]:
            st.markdown("**Observations** (values read by the investigation from the cited results; the model wrote "
                        "no number)")
            st.dataframe(pd.DataFrame(runs.observation_rows(d)), hide_index=True, width="stretch")
        st.caption("Association only: no causal claim, no confidence interval, validation data only. Publication: "
                   f"{d['publication']}.")
    with tabs[1]:
        st.caption("Every model turn and every call it asked for, including refused ones.")
        st.dataframe(pd.DataFrame(runs.trace_rows(d)), hide_index=True, width="stretch")
    with tabs[2]:
        st.dataframe(pd.DataFrame(runs.evidence_rows(d)), hide_index=True, width="stretch")
        for e in d["evidence"]:
            with st.expander(f"{e['tool']} {e['evidence_id']} ({e['status']})"):
                st.json({"request": e["request"], "result": e["result"], "provenance": e["provenance"]},
                        expanded=False)
    with tabs[3]:
        render_cost(saved)
    with tabs[4]:
        render_reference(saved)
    with tabs[5]:
        st.dataframe(pd.DataFrame(runs.provenance_rows(saved)), hide_index=True, width="stretch")
        st.markdown("**Limitations** (the report's canonical disclosures)")
        for x in av1.DISCLOSURES:
            st.markdown(f"- {x}")
    with tabs[6]:
        st.download_button("Download the verified report (Markdown)", saved.markdown,
                           file_name=f"{saved.name}_investigation_report.md", mime="text/markdown",
                           key=f"live_agent_md_{saved.name}")
        st.download_button("Download the verified report (JSON)", json.dumps(d, indent=2, sort_keys=True) + "\n",
                           file_name=f"{saved.name}_investigation_report.json", mime="application/json",
                           key=f"live_agent_json_{saved.name}")
        st.markdown(saved.markdown)


def render_open():
    """Reopen a saved investigation (offline: reload and re-verify; no network, no model)."""
    entries = runs.list_saved()
    if not entries:
        st.info("No saved investigation yet.")
        return
    labels = [label for label, _ in entries]
    want = st.session_state.get(OPEN_KEY)
    index = next((n for n, (_, p) in enumerate(entries) if str(p) == want), 0)
    pick = st.selectbox("Saved investigation", labels, index=index, key="live_agent_saved")
    path = entries[labels.index(pick)][1]
    render_saved(runs.load_saved(path))
