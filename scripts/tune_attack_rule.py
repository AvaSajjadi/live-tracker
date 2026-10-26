"""Choose the demo attack rule (ask_threshold, min_duration_s) on VALIDATION.

Runs the monitor itself (monitor.run_monitor, tracking_ok ends a bout) on model B
validation scores for a grid of rules and scores each with
evaluate.monitor_event_metrics. Never reads test sessions or test scores.

Choice: highest attack event F1; settings within TIE_F1 of the best count as
tied, and among those the one with the fewest false alerts per hour wins.
Result and caveats: docs/attack_rule_v1.md. Constant: evaluate.ATTACK_RULE_V1.

    python scripts/tune_attack_rule.py
"""
import argparse
import json

import pandas as pd

from live_tracker.calms21 import FPS
from live_tracker.evaluate import monitor_event_metrics
from live_tracker.rules import Rule

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MIN_DURATIONS_S = (0.2, 0.5, 1.0, 1.5, 2.0)
TIE_F1 = 0.01


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", default="outputs/scores_validation_model_b.csv")
    ap.add_argument("--labels", default="data/processed/labels.parquet")
    ap.add_argument("--split", default="splits/split_manifest.json")
    ap.add_argument("--out", default="outputs/attack_rule_grid.csv")
    a = ap.parse_args(argv)

    manifest = json.load(open(a.split))
    val, test = set(manifest["validation"]), set(manifest["test"])
    scores = pd.read_csv(a.scores)
    if set(scores.session) != val:
        raise SystemExit("scores must cover exactly the validation sessions")
    labels = pd.read_parquet(a.labels)
    labels = labels[labels.session.isin(val)]
    assert not (set(labels.session) | set(scores.session)) & test

    rows = []
    for thr in THRESHOLDS:
        for md in MIN_DURATIONS_S:
            rule = Rule("grid", "attack", min_duration_s=md, ask_threshold=thr,
                        alert_threshold=max(thr, 0.8), status="approved")
            m, bouts = monitor_event_metrics(scores, labels, rule, fps=FPS)
            rows.append({"ask_threshold": thr, "min_duration_s": md, **m,
                         "sessions_detected": bouts.dropna().session.nunique(),
                         "fire_offset_max_s": bouts.fire_offset_s.max()})
    grid = pd.DataFrame(rows)
    grid.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200):
        print(grid.round(3).to_string(index=False))

    tied = grid[grid.event_f1 >= grid.event_f1.max() - TIE_F1]
    best = tied.sort_values(["false_alerts_per_hour", "min_duration_s"]).iloc[0]
    print(f"\ntied (F1 within {TIE_F1}):\n{tied.round(3).to_string(index=False)}")
    print(f"\nchosen: ask_threshold {best.ask_threshold}, min_duration_s {best.min_duration_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
