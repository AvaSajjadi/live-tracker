"""EXPERIMENTAL Stage 8 CLI: preflight, offline dry run, and (only when confirmed) a live investigation.

    python scripts/investigation_live.py                       # offline dry run (default): no network, no cost
    python scripts/investigation_live.py --preflight           # offline readiness checks
    python scripts/investigation_live.py --preflight-online    # + free metadata GETs (model list, key status)
    python scripts/investigation_live.py --live --confirm-budget-usd 0.50   # PAID: one bounded investigation

The key is read from $OPENROUTER_API_KEY (provider openrouter, the default) or
$ANTHROPIC_API_KEY (only with --provider anthropic; dry run and preflight only:
Anthropic live mode is disabled in Stage 8). It is never printed.
Outputs of a live run go to outputs/investigation_live/ (git-ignored).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from live_tracker import investigation_live as live  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=sorted(live.PROVIDERS), default=live.DEFAULT_PROVIDER)
    ap.add_argument("--model", default=None, help="provider model id (default: the provider's default)")
    ap.add_argument("--question", default=live.DEFAULT_QUESTION)
    ap.add_argument("--target-usd", type=float, default=live.DEFAULT_TARGET_USD)
    ap.add_argument("--max-model-turns", type=int, default=8)
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--coverage", choices=("complete", "partial"), default="complete",
                    help="complete (default): every eligible validation session must be cached")
    ap.add_argument("--findings-format", choices=("findings/2", "legacy/1"), default="findings/2",
                    help="findings/2 (default): structured finish; legacy/1 only when asked for")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="offline readiness checks only")
    mode.add_argument("--preflight-online", action="store_true", help="readiness checks with free metadata GETs")
    mode.add_argument("--live", action="store_true", help="PAID: run one bounded live investigation")
    ap.add_argument("--confirm-budget-usd", type=float, default=None,
                    help="required with --live; must equal --target-usd")
    a = ap.parse_args(argv)
    plan = live.LivePlan(provider=a.provider, model=a.model, question=a.question, target_usd=a.target_usd,
                         max_model_turns=a.max_model_turns, max_tool_calls=a.max_tool_calls, timeout_s=a.timeout_s,
                         coverage=a.coverage, findings_format=a.findings_format)
    try:
        if a.preflight or a.preflight_online:
            out = live.preflight(plan, online=a.preflight_online)
        elif a.live:
            out = live.run_live(plan, confirm_budget_usd=a.confirm_budget_usd)
        else:
            out = {"mode": "dry_run (offline, emulated endpoint, no cost)", **live.dry_run(plan)}
    except (PermissionError, live.AmbientEnvironmentError) as exc:   # messages name variables, never values
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
