# Live Agent: launch and demo procedure

The **Live Agent** page (`pages/5_Live_Agent.py`) puts the Stage 9 autonomous investigation agent into the dashboard.

A researcher asks a question. Claude, through the guarded OpenRouter adapter, chooses every Contract v1 tool call on
the verified CalMS21 validation split. The saved report is reloaded and re-verified before anything is shown.

The existing pages (Motion Engine, Rules, Replay, Investigation, Real Session) are unchanged.

## Launch

From the repository root (required: the data fingerprint, and so run5 compatibility, depends on the relative cache
path):

```zsh
cd ~/live-tracker-final-integration
PYTHONPATH=src ~/live-tracker/.venv/bin/streamlit run app.py
```

`PYTHONPATH=src` is required. Without it, the shared virtual environment imports the `live_tracker` package of the
other checkout (`~/live-tracker`), which has no Live Agent. Open the **Live Agent** page in the sidebar.

Reopening saved investigations and the free rehearsal need no key and make no network request. For the preflight or
a paid run, set the key first without echo (see `docs/stage9_v1_live_run.md`, step 2):
`read -rs "OPENROUTER_API_KEY?Key: " && export OPENROUTER_API_KEY`.

## Demo for industry advisors (about 10 minutes, no spending)

1. **Saved investigations → `Live Agent: live_run1`.** This is the genuine paid investigation: Claude Sonnet 5.5,
   3 model turns, $0.03224 provider-reported.
   - The green box reads *Verified autonomous investigation: finding `effect_described`*. The report was just
     re-verified offline.
   - **Finding:** the median session-paired difference in `dist_center` before predicted attack bouts, −79.04 over
     29 windows. Association only; the model wrote no number.
   - **Tool sequence:** the model's own plan, `query_events` → `compare_windows` → finish.
   - **Evidence:** each Contract v1 result with its id, request and provenance.
   - **Cost ledger:** one row per request, with tokens and provider-reported cost; local target $0.50; best-effort
     key mode, with its exposure statement.
2. **Reference check (run5) tab.** The verdict is *not equivalent*.
   - The model left `threshold` unset, so Contract v1 applied 0.5; run5 used 0.6 (103 bouts against 80).
   - The page explains that this concerns run5 equivalence only. The investigation is a verified analysis of its own
     parameters.
   - The recorded run status `refused` comes from this reference check, and the exact-matching checker is unchanged.
3. **Saved investigations → `Offline (Phase 3B.1): phase3b1_run3`.** The deterministic offline investigator, with no
   model. Its reference check is *equivalent* to run5.
4. **New investigation.**
   - Type a question and choose the model, turns, tool budget, the $0.50 target and the key mode. Choosing
     best-effort shows the exposure warning and still authorizes nothing.
   - Click **Run free rehearsal** to run the same pipeline over the emulated endpoint, with no model and no cost. The
     result opens, verified, under Saved investigations, labelled EMULATED.
   - Point out that **Start paid investigation** stays disabled. It needs a ready **preflight** of exactly this
     configuration, the confirmation checkbox, and the typed authorization text. Each preflight allows one paid run,
     and only one investigation can run at a time.
5. **Provenance & limitations tab.** The data fingerprint, archive and score hashes, contract version, report
   fingerprints, publication `not_authorized` and the canonical disclosures.

A paid run from the page is the same command as `docs/stage9_v1_live_run.md` step 5, with the same checks. Do it
only with a separate decision to spend.

## Known limits

- Verification is consistency, not authenticity: the files are local and unsigned.
- A paid run blocks the page session while it runs (about 30 s for live_run1). Other sessions see the lock.
- Every guard is still client-side. In best-effort mode with the $50 key, OpenRouter enforces only the key's own
  limit.
