# Stage 9 v1: first live OpenRouter investigation (procedure)

Status: **prepared, not executed.** Nothing in this document has been run against OpenRouter. Steps 3 and 5 contact
the provider; step 5 spends money. Do step 5 only after a separate, explicit authorization.

All commands run from the repository root, `~/live-tracker-final-integration`, with
`PY=~/live-tracker/.venv/bin/python`. Relative paths matter: the data fingerprint includes the cache path, so
`--cache-dir data/raw/calms21_validation` (relative) is what matches run5.

## 1. Create a dedicated, capped OpenRouter key

In the OpenRouter dashboard (Settings → API Keys → Create key):

- Name it for this run only, e.g. `live-tracker-stage9-v1-run1`. Do not reuse or share it, and do not use it from
  any other client while the run is in progress. The code cannot detect other use of the key.
- **Credit limit: $0.50** (or less, with a lower `--target-usd`).
- **Limit reset: none/never.** A daily, weekly or monthly reset is refused.
- The account needs at least $0.50 of credit.

The preflight refuses an unlimited key, a limit above $0.50, any reset, remaining credit below the target,
inconsistent values (remaining ≠ limit − usage) and missing fields. The key's limit is OpenRouter's to enforce. It is
recorded as `provider_key_limit_usd`, and neither it nor the local target is a guaranteed maximum charge.

### 1a. Alternative: an existing higher-limit key (best-effort mode, explicit opt-in)

The default (`--key-limit-mode strict`) refuses any key whose limit is above $0.50. To use an existing key with a
larger, non-resetting limit (for example the $50 hackathon key), add to the preflight and to the paid run:

```zsh
--key-limit-mode best-effort --acknowledge-key-limit-usd 50
```

The acknowledged value must equal the key's verified `limit` exactly. Best-effort mode still refuses:
- an unlimited key;
- a limit above $50;
- any reset;
- inconsistent metadata;
- remaining credit below the target.

The target stays at most $0.50 and every guard is unchanged: token budgets, the cost ledger, the stop once reported
spending reaches the target, retries 0, the routing pin, no fallbacks, and fail-closed handling of missing costs and
unexpected responses.

**What changes is the exposure.** The $0.50 target is then a best-effort local target, not a spending ceiling.
- If the local guards fail, or a request costs more than estimated, spending can exceed $0.50, potentially up to the
  key's available credit (`provider_key_remaining_usd`), subject to OpenRouter's enforcement.
- The provider-side limit no longer bounds spending near the target.

Choosing this mode does not authorize a paid run; step 5's confirmation and authorization text are still required.
The mode is recorded in `run.json` (`key_policy`, `cost.authorization_mode`, `cost.exposure`) and checked by
`--verify-saved`.

### 1b. Only if the endpoint list needs it: a management key (read-only use)

The preflight reads `GET /api/v1/models/{author}/{slug}/endpoints` with Bearer authentication. OpenRouter may refuse
that endpoint with HTTP 403 unless a **management key** is sent. In that case the preflight fails closed and says so;
it never skips the check and never retries without credentials.

To proceed, create a management key (Settings → Management keys) and give it to the preflight as
`OPENROUTER_MANAGEMENT_KEY`. Security implications:

- A management key can create, change and delete API keys and their limits. It is far more powerful than the capped
  inference key, is **not** limited to $0.50, and must be guarded accordingly. Create it for this procedure, keep
  it only in the shell session, and **delete it** afterwards (step 6).
- The code uses it for exactly one request, the read-only endpoint-list GET. It is never given to the model adapter,
  never sent to `/api/v1/key` or `/api/v1/messages`, never used if the inference key is missing (no fallback), and
  never written to a report or printed. A management key equal to the inference key is refused.
- The code cannot stop other software in the same shell from reading it. Use a clean shell, and do not set it at
  all if the 403 does not occur.

## 2. Set the keys locally without exposing them

```zsh
read -rs "OPENROUTER_API_KEY?OpenRouter key (capped, inference): " && export OPENROUTER_API_KEY; echo
# only if step 3 reported HTTP 403 for the endpoint list:
read -rs "OPENROUTER_MANAGEMENT_KEY?OpenRouter management key (endpoint list only): " \
    && export OPENROUTER_MANAGEMENT_KEY; echo
unset ANTHROPIC_CUSTOM_HEADERS ANTHROPIC_LOG       # refused if set (they alter or log every request)
```

The keys are read without echo or shell history. Only the inference key reaches the SDK. Neither key is ever
printed or written: every output file and the preflight's printout are checked for both, and a match is refused. Do
not put either key in a file in the repository.

## 3. Read-only preflight (free metadata GETs; no model request)

```zsh
$PY scripts/investigation_v1_live.py --preflight-only --cache-dir data/raw/calms21_validation
```

It makes exactly three GETs:
- `/api/v1/models?supported_parameters=tools` (public, no key);
- `/api/v1/key` (the capped inference key, always);
- `/api/v1/models/anthropic/claude-sonnet-5.5/endpoints` (Bearer: the management key if `OPENROUTER_MANAGEMENT_KEY`
  is set, else the inference key; `credentials.endpoint_metadata_get` names which).

With the existing $50 key, add `--key-limit-mode best-effort --acknowledge-key-limit-usd 50` (step 1a):

```zsh
$PY scripts/investigation_v1_live.py --preflight-only --cache-dir data/raw/calms21_validation \
    --key-limit-mode best-effort --acknowledge-key-limit-usd 50
```

If the third GET answers 403 without a management key, the report's `problems` says so: set the management key
(step 1b) and rerun step 3.

It also reads the local validation cache and split manifest. It does not call the model, start an investigation or
write a file. It prints one JSON report. **Exit 0** means `"ready": true` and `"problems": []`. **Exit 2** means
not ready: stop and read `problems`.

## 4. Review the estimate and configuration

In the printed report, confirm:

| field | expected |
|---|---|
| `confirmed_provider_metadata.key` | `limit` ≤ 0.5, `limit_reset` null, `limit_remaining` ≥ `plan.target_usd`, `usage` = `limit` − `limit_remaining` |
| `confirmed_provider_metadata.provider_key_limit_usd` | the key's limit |
| `confirmed_provider_metadata.prices_usd_per_mtok` | fetched this run (`prices_basis` "fetched"); the Anthropic endpoint's prices are no higher |
| `confirmed_provider_metadata.anthropic_endpoint` | exactly one Anthropic endpoint for the model; `capacity`: `context_length` ≥ `planned_input_tokens` + `planned_output_tokens`, and `max_prompt_tokens` / `max_completion_tokens` (when stated) ≥ the planned input / output |
| `estimate` | `estimated_worst_case_usd` ≤ 0.50; `typical_run_usd`; `stage6_cost_cap_usd` and `one_request_reserve_usd` |
| `local_configuration.request` | `routing` {order: [anthropic], allow_fallbacks: false}; `sdk_max_retries` 0; `retries_when_re_requested` 0; `adapter_fallbacks` false; `beta_endpoint_closed` true; `requests_sent` 0; no `betas`, `fallbacks` or `thinking` in `sent_fields` |
| `spending` | `authorization_mode` (strict / best_effort), `local_target_usd` 0.5, `provider_key_limit_usd`, `provider_key_remaining_usd`, `exposure` (read it: in best-effort mode spending can exceed the target), `paid_run_authorized` false |
| `plan` | model `anthropic/claude-sonnet-5.5`, 8 turns, 8 tool calls, 4,000 output tokens per request |

`requires_a_model_response` lists what no preflight can confirm. Each item is either guarded during the run (it stops
on a violation) or is an assumption:
- the served model string;
- `usage.cost` on every response;
- OpenRouter honouring the routing pin;
- acceptance of the effort setting and `cache_control`;
- thinking blocks round-tripping;
- the chars/3 token estimate;
- OpenRouter enforcing the key limit;
- no other use of the key.

**Stop here** if any value differs, or if the preflight is more than a few minutes old when step 5 starts. Prices
and credit are re-read at the start of step 5 anyway.

## 5. The paid run (only after separate, explicit authorization)

```zsh
$PY scripts/investigation_v1_live.py ~/Downloads/igor_scores_validation_model_b_repacked.zip \
    --archive-sha256 5c1a81e40ca12f00dea0ae511708b7d534d42e8b1feb43948af338b8b5c23427 \
    --cache-dir data/raw/calms21_validation --validation-run run5 \
    --out outputs/investigation_v1_live/live_run1 \
    --live --confirm-budget-usd 0.50 \
    --authorize-paid-run "I authorize one paid Stage 9 v1 investigation"
```

With the existing $50 key, add the same `--key-limit-mode best-effort --acknowledge-key-limit-usd 50` as in step 3.

Before any request or file, the run refuses (exit 2):
- without both the confirmed budget (equal to `--target-usd`) and the exact authorization text;
- with an output directory that is not new or empty, or not git-ignored.

It then reruns the same checks as step 3 (exit 2, `run.json` with status `refused`, nothing sent) and verifies the
inputs as Phase 3B.1 does: archive SHA-256, validation member only, pose cache, 14 sessions and 94,305 frames, and
run5 on the same data.

During the run it stops (status `incomplete`, exit 1, report and ledger still written) on any of these:
- a failed or timed-out request (one attempt, no retry);
- a missing or invalid `usage.cost`;
- known spending reaching the target;
- the Stage 6 cost or token estimate exceeded;
- an unexpected served model, stop reason or token count;
- an oversized request;
- the turn or tool budget spent;
- three consecutive rejected actions or two repeated calls.

Outcomes:
- **0 `verified`:** the investigation finished, the report and Markdown verify, the ledger matches the trace and the
  initial query matches run5's.
- **2 `refused`:** the first query differs from run5's. The report is kept but not verified as compatible.
- **1 `over_target`:** provider-reported spending passed the target, which can happen by one request.

## 6. Afterwards: verify the outputs and the charges

```zsh
$PY scripts/investigation_v1_live.py --verify-saved outputs/investigation_v1_live/live_run1
```

This is offline. Exit 0 with `"verified": true` means `verify_report_v1` passed, the Markdown matches the report byte
for byte, the cost ledger is bound to the trace, the budget and cost summary agree and the file hashes match.

In `outputs/investigation_v1_live/live_run1/run.json`, read `cost`:
- `target_usd`;
- `provider_key_limit_usd`;
- `actual_spend_usd`, the sum of provider-reported costs. When `actual_spend_complete` is false, some attempts
  reported no cost and the true spend may be higher.
- `over_target`;
- `authorization_mode`, `provider_key_remaining_usd` and `exposure` (the key-limit mode and what it meant).

If a management key was created in step 1b, delete it now (Settings → Management keys).

Compare `actual_spend_usd` with the key's usage in the OpenRouter dashboard (Activity, filtered to this key). Then
**revoke the key**. A re-run of step 3 now reports the key's `usage`, and is expected to be not ready because the
remaining credit no longer covers the target.

Unresolved until a real response is seen:
- the exact fields of OpenRouter's `/api/v1/key` and `/endpoints` responses, as assumed by `key_limit_problems`,
  `anthropic_endpoint_problems` and `capacity_problems` (`context_length` required; `max_prompt_tokens` and
  `max_completion_tokens` optional, null meaning no separate limit);
- whether the endpoint list accepts the inference key or answers 403;
- the served model string;
- where the Messages endpoint reports `usage.cost`.

Any mismatch fails closed.
