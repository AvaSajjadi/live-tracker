"""EXPERIMENTAL (Investigation mode, Stage 8): live-model provider setup, preflight and budget (new file).

Runs the unchanged Stage 6 loop (investigation_agent.investigate) with its
unchanged Claude adapter (ClaudeInvestigationModel) against a configurable
provider, and renders the result with the unchanged Stage 7 renderer.

Providers
---------
- "openrouter" (preferred, the default): OpenRouter's Anthropic-compatible
  Messages endpoint (https://openrouter.ai/api/v1/messages), key from
  $OPENROUTER_API_KEY, sent as a Bearer token.
- "anthropic" (fallback): the Anthropic API, key from $ANTHROPIC_API_KEY.
  Never chosen automatically: only when the caller names it. Live mode is
  DISABLED for it in Stage 8: the Anthropic API has no pricing endpoint and
  reports no per-request cost, so neither the fetched-price estimate nor
  the provider-reported cost guard can be applied. Its offline dry run and
  preflight work; enabling live use needs reviewed, explicit prices.
Credentials are read from the environment only, passed explicitly to the
SDK (so the SDK does not also pick up another provider's key from the
environment) and never printed, logged or written.

Ambient isolation: the SDK also reads some variables from the real process
environment whatever mapping is passed here (ANTHROPIC_CUSTOM_HEADERS is
added to every request). Before any SDK client is built, in every mode,
the real os.environ (and any injected mapping) is checked and a run with
such a variable set is refused (AmbientEnvironmentError). Request headers
and exception texts are never recorded: only header presence flags, error
class, HTTP status and error type.

What is adapted (outside Stage 6), per provider
-----------------------------------------------
The Stage 6 adapter is used with fallbacks=False and an injected client
(GuardedClient) that, on every request:
- drops Anthropic-only fields: `betas` (anthropic-beta header) and
  `fallbacks` (OpenRouter's own `fallbacks` field means something else);
- drops `output_config` (effort) or `cache_control` when the preflight
  found the model does not support them (thinking is never sent: the
  models' default adaptive thinking is used);
- pins OpenRouter routing to the Anthropic provider without provider
  fallbacks (thinking blocks are echoed back unchanged across turns);
- records each response's provider-reported cost (OpenRouter `usage.cost`)
  and, on an error, only its class, HTTP status and error type.

Budget (client-side estimate, NOT a provider-side spending cap)
---------------------------------------------------------------
The run stops at whichever comes first: the Stage 6 Budget (turns, tool
calls, input / output tokens, estimated cost from conservative prices,
wall-clock time) or the provider-reported cost reaching the target.
Provider-reported costs are accepted only as finite non-negative numbers.
Known spending is summed separately from completeness: a response with a
missing or invalid cost never erases known spending, and (OpenRouter
reports a cost for every response) makes the ledger incomplete, after which
no further request is sent (fail closed). Every request is exactly one
physical attempt: SDK retries are forced off at the client boundary, so a
timeout or other uncertain failure is not retried; it ends the run with an
incomplete report (Stage 6 "model_unavailable"). A
request cannot be interrupted and its input size is only known afterwards,
so the Stage 6 cost limit is set below the target by a reserve for one
worst-case request. The only hard cap is a credit limit on the provider key
itself (set it in the OpenRouter dashboard).

Modes
-----
dry_run() (the default of the CLI) never touches the network: the real
SDK and loop run over an in-process transport that emulates the OpenRouter
endpoint. preflight(online=True) makes only free metadata GET requests
(model list, key status). run_live() makes paid requests and is only
reachable with an explicit budget confirmation.

Data: only the verified CalMS21 validation cache, through the
manifest-backed EvidencePolicy (validation annotations only; held-out test
sessions are refused by the Stage 2 contracts). No CSV is read. Coverage is
one explicit setting shared by preflight, dry run and live run
(LivePlan.coverage, default "complete"): the preflight verifies that the
cache holds every eligible validation session of the manifest, and the tools
refuse otherwise; "partial" must be chosen explicitly and is disclosed in
every result and report.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import investigation_agent as ia
from . import investigation_contracts as ic

__all__ = ["DEFAULT_PROVIDER", "PRICE_SNAPSHOT", "PROVIDERS", "GuardedClient", "LivePlan", "Prices",
           "build_model", "dry_run", "plan_budget", "preflight", "run_live"]

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "raw" / "calms21_validation"
MANIFEST = ROOT / "splits" / "split_manifest.json"
OUT_DIR = ROOT / "outputs" / "investigation_live"          # git-ignored

DEFAULT_QUESTION = "Does approach tend to precede attack?"
DEFAULT_TARGET_USD = 0.50
CHARS_PER_TOKEN = 3.0          # conservative for JSON with long IDs (English text is nearer 4)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    auth: str                  # "bearer" (Authorization) or "api_key" (x-api-key)
    default_model: str
    models_url: str | None     # free metadata: tool-capable models and prices
    key_url: str | None        # free metadata: key credit status


PROVIDERS = {
    "openrouter": Provider("openrouter", "https://openrouter.ai/api", "OPENROUTER_API_KEY", "bearer",
                           "anthropic/claude-sonnet-5.5",
                           "https://openrouter.ai/api/v1/models?supported_parameters=tools",
                           "https://openrouter.ai/api/v1/key"),
    "anthropic": Provider("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY", "api_key",
                          "claude-sonnet-5-5", None, None),
}
DEFAULT_PROVIDER = "openrouter"

# OpenRouter /api/v1/models, 2026-10-19 (USD per million tokens). Used ONLY
# for offline estimates; a live run always uses prices fetched that run.
PRICE_SNAPSHOT = {
    "as_of": "2026-10-19", "source": "https://openrouter.ai/api/v1/models?supported_parameters=tools",
    "models": {
        "anthropic/claude-haiku-5.5": {"prompt": 0.1, "completion": 0.5, "input_cache_read": 0.01,
                                       "input_cache_write": 0.125},
        "anthropic/claude-sonnet-5.5": {"prompt": 2.0, "completion": 10.0, "input_cache_read": 0.1,
                                        "input_cache_write": 2.5},
        "anthropic/claude-opus-5.5": {"prompt": 4.0, "completion": 20.0, "input_cache_read": 0.2,
                                      "input_cache_write": 5.0},
    }}


@dataclass(frozen=True)
class LivePlan:
    """Everything a live run is allowed to do. Defaults are deliberately small."""
    provider: str = DEFAULT_PROVIDER
    model: str | None = None                 # None: the provider's default model
    question: str = DEFAULT_QUESTION
    target_usd: float = DEFAULT_TARGET_USD   # estimated spend the run must stay under
    max_model_turns: int = 8
    max_tool_calls: int = 8
    timeout_s: float = 600.0
    per_request_max_tokens: int = 4000       # output (thinking + reply) allowance per request
    request_timeout_s: float = 120.0       # one physical attempt per request (no retries)
    effort: str = "medium"
    max_result_chars: int = 10_000           # tool result shown to the model (Stage 6 rendering)
    coverage: str = "complete"               # "partial" only by explicit choice (disclosed in every result)
    cache_dir: str | None = None             # verified validation cache (default data/raw/calms21_validation)

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; choose one of {sorted(PROVIDERS)}")
        if not (0 < self.target_usd <= 5.0):
            raise ValueError("target_usd must be in (0, 5]")
        if self.coverage not in ("complete", "partial"):
            raise ValueError("coverage must be 'complete' or 'partial'")

    @property
    def model_id(self):
        return self.model or PROVIDERS[self.provider].default_model


@dataclass(frozen=True)
class Prices:
    """Conservative USD per million tokens for the Stage 6 cost estimate: the
    loop counts cache writes and reads as input, so input is charged at the
    higher of the prompt and cache-write prices."""
    input_usd_mtok: float
    output_usd_mtok: float
    source: str
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_per_mtok(cls, p, source):
        def num(k):
            v = p.get(k)
            return None if v in (None, "") else float(v)
        prompt, completion = num("prompt"), num("completion")
        if prompt is None or completion is None or not all(math.isfinite(x) and x >= 0 for x in (prompt, completion)):
            raise ValueError("prices need finite prompt and completion prices")
        return cls(max(prompt, num("input_cache_write") or 0.0),
                   max(completion, num("internal_reasoning") or 0.0), source, dict(p))

    @classmethod
    def from_openrouter(cls, pricing, source):
        """OpenRouter pricing strings are USD per token."""
        per_mtok = {k: float(v) * 1e6 for k, v in pricing.items()
                    if k in ("prompt", "completion", "input_cache_read", "input_cache_write", "internal_reasoning")
                    and v not in (None, "")}
        if float(pricing.get("request") or 0) > 0:
            raise ValueError("per-request pricing is not modelled by the budget")
        return cls.from_per_mtok(per_mtok, source)


class AmbientEnvironmentError(RuntimeError):
    """A process environment variable the SDK would apply to every request."""


# variables the SDK reads from the real process environment even when
# credentials and base URL are passed explicitly, and which could add data
# to requests or print request details
UNSAFE_AMBIENT = {"ANTHROPIC_CUSTOM_HEADERS": "added as headers to every request",
                  "ANTHROPIC_LOG": "SDK logging can print request details"}


def ambient_problems(env=None):
    """Unsafe variables set in the real process environment or in `env`
    (names and reasons only, never values)."""
    found = {}
    for source in (os.environ, env or {}):
        for name, why in UNSAFE_AMBIENT.items():
            if (source.get(name) or "").strip():
                found[name] = why
    return [f"{n} is set ({w}); unset it before running" for n, w in sorted(found.items())]


def require_clean_environment(env=None):
    problems = ambient_problems(env)
    if problems:
        raise AmbientEnvironmentError("; ".join(problems))


def _confirmation(value, target):
    """The confirmed budget as a float, or PermissionError: a finite positive
    number equal to the plan's target (no NaN, infinity, bool or text)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PermissionError(f"a live run needs --confirm-budget-usd {target:.2f} (a number)")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise PermissionError("--confirm-budget-usd must be a finite positive number")
    if abs(value - target) > 1e-9:
        raise PermissionError(f"--confirm-budget-usd {value} does not equal the plan's target {target:.2f}")
    return value


def snapshot_prices(model):
    p = PRICE_SNAPSHOT["models"].get(model)
    return None if p is None else Prices.from_per_mtok(p, f"snapshot {PRICE_SNAPSHOT['as_of']} (offline estimate)")


# ---- budget ----

def _base_tokens():
    return math.ceil((len(ia.SYSTEM_PROMPT) + len(json.dumps(ia.TOOL_SPECS))) / CHARS_PER_TOKEN) + 200


def plan_budget(plan, prices):
    """The Stage 6 Budget for a plan, and the estimate behind it."""
    base = _base_tokens()
    growth = math.ceil(plan.max_result_chars / CHARS_PER_TOKEN) + plan.per_request_max_tokens  # per turn, worst
    worst_request_in = base + (plan.max_model_turns - 1) * growth
    reserve = worst_request_in * prices.input_usd_mtok / 1e6          # one request that overruns the cap
    cost_cap = round(plan.target_usd - reserve, 6)
    if cost_cap <= 0:
        raise ValueError(f"target ${plan.target_usd:.2f} leaves no room after the one-request reserve "
                         f"(${reserve:.3f}); lower max_model_turns / max_result_chars or raise the target")
    max_out = plan.max_model_turns * plan.per_request_max_tokens
    max_in = max(1, int(cost_cap * 1e6 / prices.input_usd_mtok)) if prices.input_usd_mtok > 0 else None
    budget = ia.Budget(max_model_turns=plan.max_model_turns, max_tool_calls=plan.max_tool_calls,
                       max_input_tokens=max_in, max_output_tokens=max_out, max_cost_usd=cost_cap,
                       price_per_mtok=(prices.input_usd_mtok, prices.output_usd_mtok), timeout_s=plan.timeout_s,
                       max_result_chars=plan.max_result_chars)
    # a typical run (for planning only): 7 turns, results ~1/3 of the cap, ~1500-token replies
    t_growth = math.ceil(plan.max_result_chars / 3 / CHARS_PER_TOKEN) + 1500
    t_turns = min(7, plan.max_model_turns)
    t_in = sum(base + k * t_growth for k in range(t_turns))
    t_out = t_turns * 1500
    typical = (t_in * prices.input_usd_mtok + t_out * prices.output_usd_mtok) / 1e6
    return budget, {
        "target_usd": plan.target_usd, "stage6_cost_cap_usd": cost_cap,
        "one_request_reserve_usd": round(reserve, 6), "worst_request_input_tokens": worst_request_in,
        "estimated_worst_case_usd": round(cost_cap + reserve, 6),
        "typical_run_usd": round(typical, 4), "typical_run_tokens": {"input": t_in, "output": t_out},
        "prices_usd_per_mtok": {"input": prices.input_usd_mtok, "output": prices.output_usd_mtok,
                                "source": prices.source},
        "note": "client-side estimate from conservative prices (chars / 3 per token); not a provider-side "
                "spending cap. The run also stops when the provider-reported cost reaches the target.",
    }


# ---- the guarded client ----

class Ledger:
    """Per-request record of provider-reported usage and cost (no content).

    known_cost_usd: the sum of the finite non-negative costs reported.
    complete: every attempt has such a cost; an attempt without one (missing,
    invalid, or a failed request that may still have been billed) makes the
    ledger incomplete, and stays counted as unknown."""

    def __init__(self):
        self.entries = []

    @property
    def known_cost_usd(self):
        return round(sum(e["cost_usd"] for e in self.entries if e.get("cost_status") == "reported"), 8)

    @property
    def unknown_cost_attempts(self):
        return sum(e.get("cost_status") != "reported" for e in self.entries)

    @property
    def complete(self):
        return self.unknown_cost_attempts == 0

    @property
    def reported_cost_usd(self):
        """The provider-reported total when every attempt reported a cost, else None."""
        return self.known_cost_usd if self.entries and self.complete else None

    def to_json(self):
        return {"requests": copy.deepcopy(self.entries), "known_cost_usd": self.known_cost_usd,
                "complete": self.complete, "unknown_cost_attempts": self.unknown_cost_attempts,
                "reported_cost_usd": self.reported_cost_usd}


# a single request reporting more than this is not a plausible cost: invalid
MAX_REPORTED_COST_USD = 1000.0


def _usage_cost(response):
    """(cost, status): status "reported" for a finite cost in
    [0, MAX_REPORTED_COST_USD], "missing" when absent, "invalid" otherwise
    (negative, non-finite, unrepresentable, implausible or malformed). The
    provider's value itself is never kept. Never raises."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, "missing"
        cost = getattr(usage, "cost", None)
        if cost is None:
            extra = getattr(usage, "model_extra", None)
            cost = extra.get("cost") if isinstance(extra, dict) else None
        if cost is None:
            return None, "missing"
        if type(cost) is int:                         # exact integer check: no float conversion, no overflow
            return (float(cost), "reported") if 0 <= cost <= MAX_REPORTED_COST_USD else (None, "invalid")
        if type(cost) is float and math.isfinite(cost) and 0.0 <= cost <= MAX_REPORTED_COST_USD:
            return cost, "reported"
        return None, "invalid"                        # bool, str, other types, nan / inf, out of range
    except Exception:
        return None, "invalid"


def _count(x):
    """A token count kept only when it is a plausible integer."""
    return x if type(x) is int and 0 <= x <= 2 ** 53 else None


def _label(x):
    """A short provider label (model, stop reason) kept only when it is plain text."""
    return x if isinstance(x, str) and len(x) <= 120 and x.isprintable() else None


class GuardedClient:
    """The Anthropic SDK client the Stage 6 adapter sees: removes or adapts
    request fields the provider does not take, and records usage."""

    def __init__(self, client, provider, *, keep_output_config=True, keep_cache_control=True, ledger=None):
        self._client, self.provider = client, provider
        self.keep_output_config, self.keep_cache_control = keep_output_config, keep_cache_control
        self.ledger = ledger if ledger is not None else Ledger()
        self.messages = self

    def with_options(self, **kw):
        kw["max_retries"] = 0               # one physical attempt per request, whatever the caller asks
        return GuardedClient(self._client.with_options(**kw), self.provider, keep_output_config=self.keep_output_config,
                             keep_cache_control=self.keep_cache_control, ledger=self.ledger)

    @property
    def beta(self):              # the adapter uses beta only with fallbacks=True; never route there
        raise AttributeError("beta endpoints are not used in Stage 8")

    def adapt(self, kwargs):
        kw = dict(kwargs)
        dropped = [k for k in ("betas", "fallbacks") if kw.pop(k, None) is not None]
        if not self.keep_output_config and kw.pop("output_config", None) is not None:
            dropped.append("output_config")
        if not self.keep_cache_control and isinstance(kw.get("system"), list):
            kw["system"] = [{k: v for k, v in b.items() if k != "cache_control"} for b in kw["system"]]
            dropped.append("cache_control")
        kw.pop("thinking", None)                      # never sent (the model default is used)
        if self.provider == "openrouter":
            kw["extra_body"] = {"provider": {"order": ["anthropic"], "allow_fallbacks": False}}
        return kw, dropped

    def create(self, **kwargs):
        kw, dropped = self.adapt(kwargs)
        # recorded BEFORE the attempt: whatever happens next, the attempt is in the ledger, and its cost
        # counts as unknown until it is accounted for
        entry = {"model_requested": kw.get("model"), "max_tokens": kw.get("max_tokens"), "dropped_fields": dropped,
                 "cost_usd": None, "cost_status": "pending"}
        self.ledger.entries.append(entry)
        started = time.monotonic()
        try:
            client = self._client
            if callable(getattr(client, "with_options", None)):
                client = client.with_options(max_retries=0)            # re-forced at the call itself
            response = client.messages.create(**kw)
        except Exception as exc:
            body = getattr(exc, "body", None)
            err = body.get("error") if isinstance(body, dict) else None
            status = getattr(exc, "status_code", None)
            etype = err.get("type") if isinstance(err, dict) else None
            # only the class, status and type are kept (exception text may hold secrets); a failure
            # without an HTTP answer (timeout, connection) may still have been billed: cost unknown
            entry.update(error=type(exc).__name__, status=_count(status), error_type=_label(etype),
                         cost_usd=None, cost_status="unknown_after_failure",
                         elapsed_s=round(time.monotonic() - started, 3))
            raise
        try:
            usage = getattr(response, "usage", None)
            entry.update(model_served=_label(getattr(response, "model", None)),
                         stop_reason=_label(getattr(response, "stop_reason", None)),
                         **{k: _count(getattr(usage, k, None)) for k in (
                             "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")})
            entry["cost_usd"], entry["cost_status"] = _usage_cost(response)
        except Exception as exc:                      # accounting failed: the request happened, its cost is unknown
            entry.update(cost_usd=None, cost_status="accounting_error", accounting_error=type(exc).__name__)
        entry["elapsed_s"] = round(time.monotonic() - started, 3)
        return response


class ReportedCostGuard:
    """Before each request: stop once known provider-reported spending
    reaches the target, and (when the provider reports costs) once any
    attempt's cost is unknown (fail closed). After each request: a reply
    whose own accounting failed is withheld from Stage 6 (no text, tool call
    or final answer of it is used); the attempt stays in the ledger."""

    # statuses that withhold the reply; a provider that reports no cost
    # (require_costs=False) may answer "missing", nothing else
    _BAD_ANY = ("invalid", "accounting_error", "pending", "unknown_after_failure")

    def __init__(self, inner, ledger, target_usd, *, require_costs=True):
        self.inner, self.ledger, self.target_usd, self.require_costs = inner, ledger, target_usd, require_costs

    def respond(self, system, tools, messages, limits=None):
        if self.ledger.known_cost_usd >= self.target_usd:
            raise ia.ModelUnavailable("provider_reported_budget", "known provider-reported cost reached the target")
        if self.require_costs and not self.ledger.complete:
            raise ia.ModelUnavailable("provider_cost_unknown", "a previous attempt's cost is unknown")
        before = len(self.ledger.entries)
        reply = self.inner.respond(system, tools, messages, limits=limits)
        statuses = [e.get("cost_status") for e in self.ledger.entries[before:]]
        bad = [s for s in statuses if s != "reported" and (self.require_costs or s in self._BAD_ANY)]
        if bad or not statuses:
            # the request was made (and is recorded) but its cost is unknown: its reply is not used
            raise ia.ModelUnavailable("provider_cost_unknown", "the reply's provider cost could not be accounted")
        return reply


def _key(provider, env):
    key = (env.get(provider.key_env) or "").strip()
    if not key:
        raise ia.ModelUnavailable("missing_api_key", f"{provider.key_env} is not set")
    return key


def build_model(plan, *, env=None, http_client=None, compat=None):
    """(model for investigate(), ledger). The key comes from the plan's
    provider's variable only; nothing falls back to another provider."""
    env = os.environ if env is None else env
    require_clean_environment(env)               # the real process environment too, before any client exists
    provider = PROVIDERS[plan.provider]
    key = _key(provider, env)
    try:
        import anthropic
    except ImportError:
        raise ia.ModelUnavailable("sdk_not_installed", 'pip install -e ".[claude]"') from None
    common = {"base_url": provider.base_url, "max_retries": 0, "http_client": http_client}
    if provider.auth == "bearer":
        client = anthropic.Anthropic(auth_token=key, **common)       # explicit: env keys are not consulted
    else:
        client = anthropic.Anthropic(api_key=key, **common)
    compat = compat or {}
    guarded = GuardedClient(client, provider.name, keep_output_config=compat.get("output_config", True),
                            keep_cache_control=compat.get("cache_control", True))
    inner = ia.ClaudeInvestigationModel(guarded, model=plan.model_id, max_tokens=plan.per_request_max_tokens,
                                        effort=plan.effort, timeout_s=plan.request_timeout_s,
                                        max_retries=0, fallbacks=False)
    return (ReportedCostGuard(inner, guarded.ledger, plan.target_usd, require_costs=provider.name == "openrouter"),
            guarded.ledger)


# ---- preflight ----

COMPATIBILITY = [
    {"parameter": "model (provider slug)", "openrouter": "sent as given, e.g. anthropic/claude-sonnet-5.5",
     "anthropic": "sent as given"},
    {"parameter": "tools / tool_choice auto", "openrouter": "supported (checked per model)", "anthropic": "supported"},
    {"parameter": "system cache_control", "openrouter": "kept when the model has cache pricing, else dropped",
     "anthropic": "kept"},
    {"parameter": "output_config.effort", "openrouter": "kept when the model supports reasoning, else dropped",
     "anthropic": "kept"},
    {"parameter": "thinking", "openrouter": "not sent (model default, adaptive)", "anthropic": "not sent"},
    {"parameter": "betas / anthropic-beta, fallbacks", "openrouter": "dropped (Anthropic-only)",
     "anthropic": "dropped (fallbacks=False: predictable cost)"},
    {"parameter": "thinking blocks echoed across turns",
     "openrouter": "passed through; routing pinned to the Anthropic provider", "anthropic": "native"},
    {"parameter": "usage.cost", "openrouter": "recorded (provider-reported)", "anthropic": "not reported"},
]


def _get_json(url, headers=None, timeout=20.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:     # noqa: S310 (fixed https URLs)
        return json.loads(r.read().decode("utf-8"))


def preflight(plan, *, env=None, online=False, http_get=None):
    """Readiness checks; never a paid request. online=True makes only free GET
    requests to the provider's model list and key-status endpoints."""
    env = os.environ if env is None else env
    http_get = http_get or _get_json
    provider = PROVIDERS[plan.provider]
    checks, info = [], {"provider": provider.name, "model": plan.model_id, "online": online}

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    key_set = bool((env.get(provider.key_env) or "").strip())
    check("api_key_present", key_set, f"{provider.key_env} is {'set' if key_set else 'not set'} (value never shown)")
    if provider.name == "openrouter":
        info["anthropic_key_used"] = False         # explicit auth_token: ANTHROPIC_API_KEY is never read or sent
    ambient = ambient_problems(env)
    check("clean_ambient_environment", not ambient, "; ".join(ambient) or
          f"none of {sorted(UNSAFE_AMBIENT)} set (process environment checked)")
    check("live_mode_supported", provider.name == "openrouter",
          "OpenRouter: fetched prices and provider-reported costs" if provider.name == "openrouter" else
          "Anthropic live mode is disabled in Stage 8 (no pricing endpoint, no reported cost): use "
          "--provider openrouter; the Anthropic dry run and preflight still work")
    try:
        import anthropic
        check("sdk_installed", True, f"anthropic {anthropic.__version__}")
    except ImportError:
        check("sdk_installed", False, 'pip install -e ".[claude]"')
    cache_dir = Path(plan.cache_dir) if plan.cache_dir else CACHE_DIR
    data_ok = (cache_dir / "extraction_index.json").exists() and MANIFEST.exists()
    check("validation_cache", data_ok, "verified validation cache and split manifest present" if data_ok
          else "validation cache or manifest missing")
    if data_ok:
        from .investigation_data import split_coverage
        try:
            cov = split_coverage(cache_dir, MANIFEST)
        except Exception as exc:
            check("validation_coverage", False, f"the cache does not verify ({type(exc).__name__})")
        else:
            info["data_coverage"] = {"mode": plan.coverage, "eligible": len(cov["eligible"]),
                                     "cached": len(cov["cached"]), "missing": cov["missing"]}
            detail = (f"{len(cov['cached'])} of {len(cov['eligible'])} eligible validation sessions in the "
                      f"verified cache" + (f"; missing {len(cov['missing'])}" if cov["missing"] else ""))
            check("validation_coverage", cov["complete"] or plan.coverage == "partial",
                  detail + ("" if cov["complete"] else
                            " (coverage 'partial' chosen explicitly; disclosed in every result)"
                            if plan.coverage == "partial" else
                            "; build the full cache: python -m live_tracker.investigation_data --all "
                            "--dataset-root <CalMS21 root>"))
    if data_ok:
        try:
            policy = ic.EvidencePolicy.from_manifest(MANIFEST)
            check("evidence_policy", policy.annotation_split == "validation" and not policy.allow_synthetic,
                  "annotations: validation split only; test sessions refused; no synthetic evidence")
        except Exception as exc:
            check("evidence_policy", False, f"manifest refused ({type(exc).__name__})")

    compat = {"output_config": True, "cache_control": True}
    prices = None
    if online and provider.models_url:
        try:
            models = http_get(provider.models_url)                 # public list: no key sent
            entry = next((m for m in models.get("data", []) if m.get("id") == plan.model_id), None)
        except Exception as exc:
            entry = None
            check("model_list", False, f"model list unavailable ({type(exc).__name__})")
        else:
            check("model_listed_with_tools", entry is not None,
                  f"{plan.model_id} {'is' if entry else 'is not'} among the tool-capable models")
        if entry is not None:
            params = set(entry.get("supported_parameters") or [])
            check("tool_support", {"tools", "tool_choice"} <= params, f"supported_parameters: {sorted(params)}")
            compat["output_config"] = "reasoning" in params or "verbosity" in params
            pricing = entry.get("pricing") or {}
            compat["cache_control"] = pricing.get("input_cache_read") not in (None, "", "0")
            try:
                prices = Prices.from_openrouter(pricing, f"{provider.models_url} (fetched this run)")
                check("pricing", True, f"input ${prices.input_usd_mtok}/M (conservative), output "
                                       f"${prices.output_usd_mtok}/M")
            except (ValueError, TypeError) as exc:
                check("pricing", False, f"prices not usable: {exc}")
            top = entry.get("top_provider") or {}
            max_out = top.get("max_completion_tokens")
            check("output_allowance", max_out is None or max_out >= plan.per_request_max_tokens,
                  f"max_completion_tokens {max_out}, context {entry.get('context_length')}")
            info["model_metadata"] = {"context_length": entry.get("context_length"), "max_completion_tokens": max_out,
                                      "supported_parameters": sorted(params),
                                      "pricing_usd_per_token": {k: v for k, v in pricing.items()
                                                                if isinstance(v, str)}}
        if key_set and provider.key_url:
            try:
                status = http_get(provider.key_url, {"Authorization": f"Bearer {_key(provider, env)}"})
                d = status.get("data") or {}
                remaining = d.get("limit_remaining")
                info["key_status"] = {k: d.get(k) for k in ("limit", "limit_remaining", "limit_reset", "usage",
                                                            "usage_daily", "is_free_tier")}
                check("key_credit", remaining is None or remaining >= plan.target_usd,
                      "no key credit limit set (account balance applies)" if remaining is None
                      else f"key limit remaining ${remaining}")
            except Exception as exc:
                check("key_status", False, f"key status unavailable ({type(exc).__name__}, HTTP "
                                           f"{getattr(exc, 'code', None)})")
    elif online:
        check("pricing", False, f"{provider.name} has no pricing endpoint: pass prices explicitly")
    if prices is None:
        prices = snapshot_prices(plan.model_id)
        info["prices_basis"] = "snapshot (offline)" if prices else "none"
    else:
        info["prices_basis"] = "fetched"
    if prices is not None:
        try:
            _, estimate = plan_budget(plan, prices)
            info["estimate"] = estimate
            check("budget", estimate["estimated_worst_case_usd"] <= plan.target_usd + 1e-9,
                  f"Stage 6 cost cap ${estimate['stage6_cost_cap_usd']:.3f} + reserve "
                  f"${estimate['one_request_reserve_usd']:.3f}; typical ~${estimate['typical_run_usd']:.3f}")
            check("budget_fit", estimate["typical_run_usd"] <= estimate["stage6_cost_cap_usd"],
                  f"a typical run (~${estimate['typical_run_usd']:.3f}) "
                  + ("fits under" if estimate["typical_run_usd"] <= estimate["stage6_cost_cap_usd"] else
                     "would likely stop on") + f" the ${estimate['stage6_cost_cap_usd']:.3f} cost cap")
        except ValueError as exc:
            check("budget", False, str(exc))
    else:
        check("budget", False, "no prices for this model")
    info["compatibility"] = {"table": COMPATIBILITY, "keep_output_config": compat["output_config"],
                             "keep_cache_control": compat["cache_control"]}
    names_required = {"api_key_present", "clean_ambient_environment", "live_mode_supported", "sdk_installed",
                      "validation_cache", "validation_coverage", "evidence_policy",
                      "budget", "model_listed_with_tools", "tool_support", "pricing", "output_allowance"}
    ok = {c["name"] for c in checks if c["ok"]}
    failed = [c["name"] for c in checks if not c["ok"]]
    info.update(checks=checks, ready_for_live=online and info["prices_basis"] == "fetched"
                and names_required <= ok and not failed,
                prices=None if prices is None else asdict(prices))
    return info


# ---- runs ----

def _tools_and_policy(plan):
    from .investigation_events import CalMS21InvestigationTools
    policy = ic.EvidencePolicy.from_manifest(MANIFEST)
    cache_dir = Path(plan.cache_dir) if plan.cache_dir else CACHE_DIR
    return CalMS21InvestigationTools.for_validation_cache(cache_dir, MANIFEST, policy,
                                                          coverage=plan.coverage), policy


def _render(report, policy):
    from . import investigation_report as ir
    try:
        doc, md = ir.render(report, policy=policy)
        return {"doc": doc, "markdown": md, "error": None}
    except ir.ReportIntegrityError as exc:
        return {"doc": None, "markdown": None, "error": exc.problems}


def _summary(report, rendered, ledger, estimate):
    return {"coverage": rendered["doc"]["coverage"] if rendered["doc"] else None, "finding": report.finding, "stopping_reason": report.stopping_reason, "finished": report.finished,
            "usage": copy.deepcopy(report.usage), "n_evidence": len(report.evidence),
            "tools_used": [t["tool"] for t in report.trace if t["kind"] in ("tool_call", "finish")],
            "report_verified": rendered["error"] is None, "report_problems": rendered["error"],
            "provider_reported_cost_usd": ledger.reported_cost_usd,
            "provider_known_cost_usd": ledger.known_cost_usd, "provider_ledger_complete": ledger.complete,
            "estimated_cost_usd": report.usage.get("cost_usd"), "estimate": estimate,
            "requests": ledger.to_json()["requests"]}


class _EmulatedOpenRouter:
    """In-process stand-in for POST /api/v1/messages (no network). It records
    what the SDK sends and answers like the endpoint: one query_events call,
    then a finish citing the result."""

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        import httpx2
        body = json.loads(request.content)
        names = {k.lower() for k in request.headers}
        self.requests.append({"method": request.method, "url": str(request.url),     # never header values
                              "has_authorization": "authorization" in names, "has_x_api_key": "x-api-key" in names,
                              "has_anthropic_beta": "anthropic-beta" in names, "body_keys": sorted(body)})
        results = [b for m in body["messages"] if isinstance(m["content"], list) for b in m["content"]
                   if b.get("type") == "tool_result"]
        if not results:
            content = [{"type": "text", "text": "Dry run: find the attack episodes."},
                       {"type": "tool_use", "id": "toolu_dry_1", "name": "query_events",
                        "input": {"source": "validation_annotation", "behavior": "attack"}}]
        else:
            eid = json.loads(results[-1]["content"])["evidence_id"]
            content = [{"type": "tool_use", "id": f"toolu_dry_{len(self.requests)}", "name": ia.FINISH,
                        "input": {"finding": "inconclusive", "observations": [
                            {"evidence_id": eid, "field": "measurements.n_episodes_total"}],
                            "next_steps": ["Run the investigation with a live model."]}}]
        return httpx2.Response(200, json={
            "id": f"msg_dry_{len(self.requests)}", "type": "message", "role": "assistant", "model": body["model"],
            "content": content, "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 3000, "output_tokens": 300, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "cost": 0.0}})


def dry_run(plan=None):
    """Offline end-to-end check: real SDK, loop, tools and renderer over the
    emulated endpoint. No network, no key, no cost."""
    import httpx2
    plan = plan or LivePlan()
    require_clean_environment()
    provider = PROVIDERS[plan.provider]
    emulator = _EmulatedOpenRouter()
    prices = snapshot_prices(plan.model_id) or Prices(1.0, 1.0, "dry-run placeholder")
    budget, estimate = plan_budget(plan, prices)
    model, ledger = build_model(plan, env={provider.key_env: "dry-run-placeholder-not-a-key"},
                                http_client=httpx2.Client(transport=httpx2.MockTransport(emulator)))
    tools, policy = _tools_and_policy(plan)
    report = ia.investigate(plan.question, tools, policy, model, budget=budget)
    rendered = _render(report, policy)
    out = _summary(report, rendered, ledger, estimate)
    out["emulated_requests"] = emulator.requests
    return out


def run_live(plan, *, confirm_budget_usd, env=None, out_dir=None, http_get=None, http_client=None):
    """A paid investigation. Requires the budget to be confirmed and a fresh
    online preflight that passes; writes report.md / report.json / run.json."""
    env = os.environ if env is None else env
    _confirmation(confirm_budget_usd, plan.target_usd)             # before any request or file
    if plan.provider != "openrouter":
        raise PermissionError("Anthropic live mode is disabled in Stage 8: it has no pricing endpoint and reports "
                              "no per-request cost, so the budget cannot be checked. Use --provider openrouter.")
    try:
        require_clean_environment(env)
    except AmbientEnvironmentError as exc:
        raise PermissionError(str(exc)) from None
    pre = preflight(plan, env=env, online=True, http_get=http_get)
    if not pre["ready_for_live"]:
        raise PermissionError(f"preflight not ready: {[c['name'] for c in pre['checks'] if not c['ok']]}")
    prices = Prices(**pre["prices"])
    budget, estimate = plan_budget(plan, prices)
    model, ledger = build_model(plan, env=env, http_client=http_client,
                                compat={"output_config": pre["compatibility"]["keep_output_config"],
                                        "cache_control": pre["compatibility"]["keep_cache_control"]})
    tools, policy = _tools_and_policy(plan)
    report = ia.investigate(plan.question, tools, policy, model, budget=budget)
    rendered = _render(report, policy)
    summary = _summary(report, rendered, ledger, estimate)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(out_dir or OUT_DIR) / f"{stamp}_{plan.provider}"
    out.mkdir(parents=True, exist_ok=False)
    files = {"run.json": json.dumps({"plan": asdict(plan), "budget": asdict(budget), "preflight": pre,
                                     "summary": summary}, indent=2, sort_keys=True, default=str),
             "investigation_report.json": json.dumps(report.to_json(), indent=2, sort_keys=True)}
    if rendered["doc"] is not None:
        from . import investigation_report as ir
        files["report.json"] = ir.to_json_text(rendered["doc"])
        files["report.md"] = rendered["markdown"]
    key = (env.get(PROVIDERS[plan.provider].key_env) or "").strip()
    for name, text in files.items():
        if key and key in text:                          # never write a credential
            raise RuntimeError(f"refusing to write {name}: it contains the API key")
        (out / name).write_text(text)
    summary["output_dir"] = str(out)
    return summary
