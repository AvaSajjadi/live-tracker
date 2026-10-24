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
OPEN_KEY = "live_agent_open"                      # a saved investigation to select once (its directory)
SAVED_KEY = "live_agent_saved"                    # the saved-investigation selectbox: its value (a directory) is
                                                  # the ONLY source of what the Saved tab shows
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
    # select the new run through the Session State API (sent to the browser), never by dropping the widget's
    # state: a dropped state falls back to a default the browser is not told about, so the selector would keep
    # showing the previous run above the new run's results
    st.session_state[OPEN_KEY] = str(runs.live_dir() / name)
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

def render_reference(saved, ref=None):
    st.markdown(f"**Reference compatibility check against {runs.REFERENCE_RUN}** (separate from the investigation; "
                "exact initial request and result matching, fail closed)")
    ref = ref or runs.reference_check(saved)
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
    m = st.columns(5 if saved.kind == "rehearsal" else 4)
    if saved.kind == "rehearsal":                 # an emulated ledger: nothing was spent
        m[0].metric("Actual API spending", "$0.00")
        m[4].metric("Simulated cost (emulated, not spent)", f"${c['actual_spend_usd']:.5f}")
        st.info(f"Simulated cost ${c['actual_spend_usd']:.5f}; actual API spending $0. This rehearsal used the "
                "emulated endpoint: its tokens and costs are emulated, no provider was called.")
    else:
        m[0].metric("Actual spend (provider-reported)", f"${c['actual_spend_usd']:.5f}"
                    + ("" if c["actual_spend_complete"] else " +unknown"))
    m[1].metric("Local target", f"${c['target_usd']:.2f}")
    m[2].metric("Provider key limit", "none (no provider)" if c["provider_key_limit_usd"] is None
                else f"${c['provider_key_limit_usd']:g}")
    m[3].metric("Over target", "yes" if c["over_target"] else "no")
    if saved.kind == "rehearsal":
        st.caption(f"Key-limit mode `{c['authorization_mode']}` (recorded; no key was used).")
    else:
        st.caption(f"Key-limit mode `{c['authorization_mode']}`. {c['exposure']}. {c['note']}")
    st.dataframe(pd.DataFrame(runs.ledger_rows(saved.summary, simulated=saved.kind == "rehearsal")),
                 hide_index=True, width="stretch")


# ---- UI-1: the scientific overview and the recorded agent activity ----

BADGES = {"accepted": ":green-badge[accepted as evidence]", "accepted_insufficient":
          ":orange-badge[accepted: insufficient evidence]", "accepted_other": ":gray-badge[accepted]",
          "finding_accepted": ":green-badge[finding accepted]", "refused": ":red-badge[refused]",
          "problem": ":red-badge[stopped]", "finished": ":green-badge[investigation finished]",
          "incomplete": ":orange-badge[incomplete]"}
ICONS = {"accepted": ":material/check_circle:", "finding_accepted": ":material/task_alt:",
         "accepted_insufficient": ":material/error:",
         "accepted_other": ":material/check:", "refused": ":material/block:", "problem": ":material/stop_circle:",
         "finished": ":material/flag:", "incomplete": ":material/hourglass_disabled:"}


MAIN_DIFFERENCE = "Main difference (pre − baseline)"
SIMULATED_COST = "Simulated cost (emulated, not spent)"
# Metric labels, qualifiers and values wrap instead of being cut off (a value only between number and unit); small
# scientific captions get more contrast and size; long identifiers wrap instead of widening the page.
READABLE = """<style>
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {white-space: normal !important;
  overflow: visible !important; text-overflow: clip !important; height: auto !important;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {white-space: normal !important;
  overflow: visible !important; text-overflow: clip !important; overflow-wrap: normal !important;
  font-size: 1.75rem; line-height: 1.2;}
[data-testid="stCaptionContainer"] {color: inherit !important; opacity: 0.86; font-size: 0.92rem; line-height: 1.45;}
[data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {overflow-wrap: anywhere;}
[data-testid="stMarkdownContainer"] code {white-space: normal; overflow-wrap: anywhere;}
</style>"""


def _num(x, digits=4):
    return f"{x:.{digits}g}" if isinstance(x, float) else ("—" if x is None else str(x))


def _seconds(x):
    return "—" if x is None else f"{x:g} s"


def render_status_panels(saved, ov):
    kind, text = runs.autonomous_status(saved)
    left, right = st.columns(2)
    with left:
        (st.success if kind == "verified" else st.warning)(
            f"**Scientific report: {'verified' if kind == 'verified' else 'verified, incomplete'}.** {text}")
    ref = ov.get("reference") or {}
    name = ref.get("reference", runs.REFERENCE_RUN)
    with right:
        if ref.get("equivalent"):
            st.success(f"**Reference comparison ({name}): equivalent.** The initial request and result are exactly "
                       f"{name}'s.")
        elif ref.get("available"):
            th = (f" Agent threshold {ref['agent_threshold']}"
                  + ("" if ref.get("agent_threshold_sent") else " (not set by the agent: Contract v1's default)")
                  + f"; reference threshold {ref['reference_threshold']}." if ref.get("agent_threshold") else "")
            st.info(f"**Reference comparison ({name}): not equivalent.**{th} Differs in: "
                    f"{', '.join(ref['differences']) or 'a result'}. This concerns {name} equivalence, not the "
                    "investigation's own validity.")
        else:
            st.info(f"**Reference comparison ({name}): not checked.** {ref.get('verdict', '')}")
    status = saved.summary.get("status")
    st.caption(f"Recorded run status `{status}`" + (
        " (set by the reference compatibility check; the scientific report itself verifies)"
        if status == "refused" and kind != "unverified" else "") + ".")


def render_overview(saved, ov):
    """The scientific overview: every value from the re-verified report or its accepted evidence."""
    st.markdown("#### Research question")
    st.markdown(f"> {ov['question']}")
    render_status_panels(saved, ov)
    st.markdown(f"### {ov['headline']}")
    st.markdown(ov["finding_text"])
    c, e, sp = ov["comparison"] or {}, ov["effect"], ov["spending"]
    m = st.columns(5 if sp["simulated_usd"] is not None else 4)
    m[0].metric(MAIN_DIFFERENCE, f"{_num(e['value'])} {e['unit']}".strip() if e else "not reported",
                help="The median, across sessions, of each session's paired difference (pre-bout window minus "
                     f"baseline). Contract v1: {c.get('mean_diff_definition')}")
    m[1].metric("Valid comparison windows", _num(c.get("n_windows")))
    m[2].metric("Contributing sessions", _num(c.get("n_sessions")))
    actual = "$0.00" if saved.kind != "live" else f"${sp['actual_usd']:.5f}" + ("" if sp["complete"] else " +unknown")
    m[3].metric("Actual API spending", actual, help=sp["basis"])
    if sp["simulated_usd"] is not None:
        m[4].metric(SIMULATED_COST, f"${sp['simulated_usd']:.5f}", help="Computed by the emulated endpoint; no "
                    "request reached a provider and nothing was spent.")
    if e:
        st.caption(f"Sign: Contract v1 defines each session's difference as pre-window minus baseline and reports their "
                   f"median; a {e['sign']} value means the median session-paired `{e['feature']}` was "
                   f"{'lower' if e['sign'] == 'negative' else 'higher' if e['sign'] == 'positive' else 'no different'} "
                   "in the pre-bout window than in the baseline. Association only: no causal claim, no confidence "
                   "interval, no significance test.")
    elif c.get("status") == "insufficient_evidence":
        st.caption(f"The comparison did not meet the evidence gate ({c.get('reason')}): {c.get('n_available')} "
                   f"available, {c.get('n_required')} required. No effect is reported.")
    if not ov["finished"]:
        st.caption(f"The investigation stopped ({ov['stopping_reason']}) before an accepted finding; the values above "
                   "come from the partial evidence and are not a finding.")
    render_compared(ov)


def plain_comparison(c, q):
    """A plain-language reading of the comparison: the investigation's own parameters put into Contract v1's window
    and effect definitions (the window offsets are only those definitions' sums). No result is interpreted."""
    out = []
    if q is not None:
        out.append(f"**Events:** the {_num(q['n_bouts'])} predicted `{q['behavior']}` bouts where the classifier's "
                   f"score stayed at or above {_num(q['threshold'])} for at least {_seconds(q['min_duration_s'])}; "
                   "each bout is timed from its start.")
    if c is None:
        return out
    pre, base, gap = c.get("pre_s"), c.get("baseline_s"), c.get("baseline_gap_s")
    unit = f", in {c['unit']}" if c["unit"] else ""
    out.append(f"**Measured:** `{c['feature']}` ({c['feature_meaning']}{unit}), averaged over each period.")
    if isinstance(pre, (int, float)):
        out.append(f"**Pre-event period:** the {_seconds(pre)} immediately before the bout starts.")
    if all(isinstance(x, (int, float)) for x in (pre, base, gap)):
        out.append(f"**Baseline period:** a {_seconds(base)} period from {_seconds(pre + gap + base)} to "
                   f"{_seconds(pre + gap)} before the bout starts"
                   + (" (directly before the pre-event period)." if gap == 0 else
                      f" (separated from the pre-event period by a {_seconds(gap)} gap)."))
    out.append("**Pre − baseline:** for each bout, the pre-event value minus the baseline value; each session "
               "averages its bouts' differences, and the main difference is the median of those session values. "
               "Negative means lower before the bout than at baseline, positive means higher. It describes an "
               "association, not a cause.")
    return out


def render_compared(ov):
    c, q = ov["comparison"], ov["query"]
    if c is None and q is None:
        return
    with st.expander("What was compared (Contract v1 definitions and the parameters this investigation used)",
                     expanded=True):
        plain = plain_comparison(c, q)
        if plain:
            st.markdown("**In plain terms**")
            for line in plain:
                st.markdown(f"- {line}")
            st.markdown("**Contract v1 definitions**")
        if q is not None:
            th = f"{_num(q['threshold'])}" + ("" if q["threshold_sent"] else " (not set by the agent: Contract v1's "
                                                                           "default applied)")
            st.markdown(f"- **Events:** predicted `{q['behavior']}` bouts from {q['data_source']} scores: score ≥ "
                        f"threshold {th}, lasting ≥ {_seconds(q['min_duration_s'])}; {q['n_bouts']} bouts in "
                        f"{q['n_sessions']} of {q['n_sessions_searched']} sessions searched.")
            st.caption(q["start_meaning"])
        if c is not None:
            w = c.get("windows") or {}
            st.markdown(f"- **Feature:** `{c['feature']}` ({c['feature_meaning']}{', ' + c['unit'] if c['unit'] else ''}).")
            st.markdown(f"- **Pre-bout window:** {w.get('pre', '—')}, with pre_s = {_seconds(c['pre_s'])}.")
            st.markdown(f"- **Baseline window:** {w.get('baseline', '—')}, with baseline_s = {_seconds(c['baseline_s'])} "
                        f"and baseline_gap_s = {_seconds(c['baseline_gap_s'])}.")
            st.markdown(f"- **Anchor:** {w.get('anchor', '—')}; {w.get('truncation', '')}.")
            gate = c.get("evidence_gate") or {}
            st.markdown(f"- **Policies (fixed by Contract v1):** quality `{c['quality_policy']}`, overlap "
                        f"`{c['overlap_policy']}`, evidence gate ≥ {gate.get('min_n_windows')} windows in ≥ "
                        f"{gate.get('min_n_sessions')} sessions. {c['n_used']} of {c['n_input']} bouts used "
                        f"({c['n_excluded']} excluded).")
            st.markdown(f"- **Effect definition:** {c['mean_diff_definition']}.")


def render_timeline(steps, who):
    st.markdown("#### Recorded investigation activity")
    st.caption(f"{who['label'].capitalize()}. The saved trace, in order (a record of a finished run, not a live "
               f"progress stream). {who['intro']}")
    for s in steps:
        with st.container(border=True):
            left, right = st.columns([1, 14])
            left.markdown(f"{ICONS.get(s['status'], '')}  \n**{s.get('number', '')}**")
            with right:
                tool = f" · `{s['tool']}`" if s.get("tool") else ""
                st.markdown(f"**{s['title']}**{tool} {BADGES.get(s['status'], '')}")
                st.caption(s["description"])
                if s.get("parameters"):
                    st.markdown(f"{who['chooser']}: " + " · ".join(f"`{k}` = {v}" for k, v in s["parameters"]))
                if s.get("results"):
                    st.markdown(" · ".join(f"{k}: **{v}**" for k, v in s["results"]))
                if s.get("supporting"):
                    st.markdown("Supporting evidence: " + " · ".join(
                        f"`{x['field']}` = **{_num(x['value']) if isinstance(x['value'], float) else x['value']}**"
                        + (f" (step {x['step']})" if x["step"] else "") for x in s["supporting"]))
                with st.expander("Technical details"):
                    if s.get("model_note"):
                        st.markdown(f"Recorded reply text: {s['model_note']}")
                    st.json({k: s.get(k) for k in ("trace_step", "tool", "arguments", "evidence_id", "details",
                                                   "supporting") if s.get(k) is not None}, expanded=False)


def render_saved(saved):
    if not saved.verified:
        st.error("This saved investigation does NOT verify, so none of its findings, evidence or costs are shown: "
                 + "; ".join(map(str, saved.problems[:5])))
        return
    d = saved.doc
    st.markdown(f"**Showing `{saved.name}`** ({runs.KIND_LABELS.get(saved.kind, saved.kind)}); everything below is "
                "this investigation's saved, re-verified files.")
    st.markdown(READABLE, unsafe_allow_html=True)
    ref, who = runs.reference_check(saved), runs.attribution(saved)
    render_overview(saved, runs.overview(saved, ref))
    render_timeline(runs.timeline(d, who["replies"]), who)
    st.markdown("#### Technical details")
    tabs = st.tabs(["Finding", "Tool sequence", "Evidence", "Cost ledger", "Reference check (run5)",
                    "Provenance & limitations", "Report"])
    with tabs[0]:
        st.markdown(f"**Finding:** `{d['finding']}`. {av1.FINDING_TEXT[d['finding']]}")
        if d["primary_evidence_id"]:
            st.caption(f"Primary evidence `{d['primary_evidence_id']}`.")
        if d["observations"]:
            st.markdown("**Observations** (values read by the investigation from the cited results; no number was "
                        "written by the investigator)")
            st.dataframe(pd.DataFrame(runs.observation_rows(d)), hide_index=True, width="stretch")
        st.caption("Association only: no causal claim, no confidence interval, validation data only. Publication: "
                   f"{d['publication']}.")
    with tabs[1]:
        st.caption(f"Every recorded turn ({who['label']}) and every call it asked for, including refused ones.")
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
        render_reference(saved, ref)
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
    label_of = {str(p): label for label, p in entries}
    want = st.session_state.pop(OPEN_KEY, None)               # a just-completed run: selected once, explicitly
    if want in label_of:
        st.session_state[SAVED_KEY] = want
    elif st.session_state.get(SAVED_KEY) not in label_of:     # first view, or the selected run is gone
        st.session_state[SAVED_KEY] = next(iter(label_of))
    path = st.selectbox("Saved investigation", list(label_of), format_func=label_of.__getitem__, key=SAVED_KEY)
    render_saved(runs.load_saved(path))
