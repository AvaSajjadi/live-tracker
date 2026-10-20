"""Stage 9 (S9-1a) offline smoke test: a verified validation score drop through
the existing streaming monitor. No network, no API key, no paid call.

    # a deliberately labelled local smoke-test rule (placeholder thresholds)
    python scripts/stage9_smoke.py ~/Downloads/igor_score_drops.zip \\
        --member scores_validation_model_b.csv --smoke-rule

    # or a proposed rule file, confirmed explicitly by the operator
    python scripts/stage9_smoke.py ARCHIVE --member scores_validation_model_b.csv \\
        --rule-json my_rule.json --confirm R1

    # optional outputs (journal, §4 events, summary) only into a git-ignored directory;
    # the three files must not exist yet (never overwritten) and are created without following symlinks
    ... --out outputs/stage9_smoke

The archive is read in place (score_drops.load_validation_scores: archive and
checksum checks, validation split only, §2 schema). The monitor, its bout and
event semantics and the rule-approval gate are used unchanged. Without --out
everything stays in memory and nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker import score_drops  # noqa: E402
from live_tracker.event_log import event_to_json  # noqa: E402
from live_tracker.monitor import run_monitor  # noqa: E402
from live_tracker.rules import RuleSet  # noqa: E402

OUTPUT_NAMES = ("events.journal.jsonl", "events.jsonl", "summary.json")
SMOKE_RULE_ID = "LOCAL_SMOKE_TEST"
SMOKE_DESCRIPTION = ("LOCAL SMOKE TEST ONLY: placeholder thresholds, not tuned and not a researcher-approved "
                     "monitoring rule")


class SmokeError(RuntimeError):
    """The smoke test cannot run with these arguments."""


def build_rules(args):
    """(RuleSet with exactly one approved rule, approval description)."""
    rules = RuleSet()
    if args.smoke_rule == bool(args.rule_json):
        raise SmokeError("give exactly one of --smoke-rule or --rule-json (with --confirm)")
    if args.smoke_rule:
        rule = rules.add({"rule_id": SMOKE_RULE_ID, "behavior": args.behavior,
                          "min_duration_s": args.min_duration_s, "alert_threshold": args.alert_threshold,
                          "ask_threshold": args.ask_threshold, "alert_mode": args.alert_mode,
                          "description": SMOKE_DESCRIPTION})
        rules.approve(rule.rule_id)
        return rules, "local smoke-test approval (labelled; not a researcher decision)"
    try:
        data = json.loads(Path(args.rule_json).read_text())
    except (OSError, ValueError) as exc:
        raise SmokeError(f"cannot read the rule file ({type(exc).__name__})") from None
    try:
        rule = rules.add(data)                       # validate_rule: every new rule arrives "proposed"
    except ValueError as exc:
        raise SmokeError(f"invalid rule: {exc}") from None
    if args.confirm != rule.rule_id:
        raise SmokeError(f"the rule is only proposed; confirm it explicitly with --confirm {rule.rule_id}")
    rules.approve(rule.rule_id)
    return rules, "confirmed on the command line by the operator (--confirm)"


def ignored_output_dir(path):
    """The output directory, refused unless it is outside this repository or
    ignored by Git (so journals, exports and summaries never become tracked)."""
    given = Path(path).expanduser().absolute()
    if given.is_symlink():
        raise SmokeError(f"{given} is a symlink; give a real directory")
    out = given.resolve()
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        return out                                   # outside the repository
    probe = rel / "probe.json"
    done = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", "--no-index", str(probe)],
                          capture_output=True)
    if done.returncode != 0:
        raise SmokeError(f"{rel} is not ignored by Git; choose an ignored directory such as outputs/stage9_smoke")
    return out


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def prepare_output(out):
    """Create (or accept) the output directory and claim its three files.

    The directory must be a real directory (not a symlink, not another kind of
    file); a new one is created private (0o700). None of the output names may
    exist in any form (file, symlink, directory): nothing is ever overwritten.
    Each file is then created exclusively without following symlinks
    (O_CREAT | O_EXCL | O_NOFOLLOW), so an existing or planted path is refused,
    never followed. The journal file is claimed empty here and then appended to
    by the unchanged EventJournal."""
    try:
        st = os.lstat(out)
    except FileNotFoundError:
        out.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(out, 0o700)
        st = os.lstat(out)
    if not stat.S_ISDIR(st.st_mode):
        raise SmokeError(f"{out} is not a real directory")
    existing = [n for n in OUTPUT_NAMES if os.path.lexists(out / n)]
    if existing:
        raise SmokeError(f"{out} already contains {existing}; outputs are never overwritten (choose a new directory)")
    create_exclusive(out / OUTPUT_NAMES[0], b"")


def create_exclusive(path, data):
    """Write a new regular file: refused if anything (including a symlink) is there."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o600)
    except FileExistsError:
        raise SmokeError(f"{path} already exists; it is never overwritten") from None
    except OSError as exc:
        raise SmokeError(f"{path} cannot be created safely ({exc.strerror})") from None
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def run(args):
    out = ignored_output_dir(args.out) if args.out else None
    rules, approval = build_rules(args)
    drop = score_drops.load_validation_scores(args.archive, args.member,
                                              expected_archive_sha256=args.archive_sha256)
    table = drop.scores()
    journal = None
    if out is not None:                          # only after the scores verified: a refusal writes nothing
        prepare_output(out)
        journal = out / OUTPUT_NAMES[0]
    events = run_monitor(table, rules.active(), journal_path=journal, run_id=args.run_id)
    rule = rules.active()[0].to_dict()
    summary = {
        "mode": "offline smoke test (validation scores, existing monitor; no network, no paid call)",
        "sessions": int(table["session"].nunique()), "frames": int(len(table)),
        "tracking_ok_false_frames": int((~table["tracking_ok"]).sum()),
        "events": len(events), "actions": dict(sorted(Counter(e["action"] for e in events).items())),
        "sessions_with_events": len({e["session"] for e in events}),
        "rule": rule, "rule_approval": approval, "run_id": args.run_id,
        "source": drop.provenance.to_json(),
        "written": None,
    }
    if out is not None:
        summary["written"] = [str(out / n) for n in OUTPUT_NAMES]
        create_exclusive(out / OUTPUT_NAMES[1], "".join(event_to_json(e) + "\n" for e in events).encode("utf-8"))
        create_exclusive(out / OUTPUT_NAMES[2], (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="score-drop ZIP (read in place, never extracted)")
    ap.add_argument("--member", required=True, help="validation CSV member, e.g. scores_validation_model_b.csv")
    ap.add_argument("--archive-sha256", default=None, help="expected SHA-256 of the whole archive (optional)")
    ap.add_argument("--smoke-rule", action="store_true", help="use the labelled local smoke-test rule")
    ap.add_argument("--behavior", default="attack", choices=sorted(score_drops.SCORE_CLASSES))
    ap.add_argument("--min-duration-s", type=float, default=2.0)
    ap.add_argument("--ask-threshold", type=float, default=0.5)
    ap.add_argument("--alert-threshold", type=float, default=0.8)
    ap.add_argument("--alert-mode", default="alert", choices=("alert", "ask", "log"))
    ap.add_argument("--rule-json", default=None, help="a proposed rule (rules.py format)")
    ap.add_argument("--confirm", default=None, help="rule_id of --rule-json, confirming it explicitly")
    ap.add_argument("--run-id", default="stage9_smoke")
    ap.add_argument("--out", default=None, help="git-ignored directory for journal / events / summary")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        summary = run(args)
    except (SmokeError, score_drops.ScoreDropError) as exc:
        print(f"refused ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
