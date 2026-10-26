"""Dry run on classifier scores: usable pre-attack windows for monitor-detected bouts.

Same rules as dryrun_usable_windows.py, but bouts come from the streaming monitor
(throwaway rule, threshold 0.5) on a score CSV (INTERFACES.md §2). tracking_ok is
taken from the feature table (motion parquet). Counts only, no effect estimated.

    python scripts/dryrun_classifier_bouts.py scores.csv motion.parquet
"""
import sys

import pandas as pd

from live_tracker.monitor import run_monitor
from live_tracker.rules import RuleSet
from dryrun_usable_windows import FPS, MIN_USABLE, PRE_GRID, BASELINE_S, GAP_S, usable

THRESH, MIN_DUR = 0.5, 1.0


def main(scores_csv, motion_parquet):
    sc = pd.read_csv(scores_csv)
    ok = pd.read_parquet(motion_parquet, columns=["session", "frame", "tracking_ok"])
    sc = sc.drop(columns=[c for c in ("tracking_ok",) if c in sc]).merge(
        ok, on=["session", "frame"], how="left")
    sc["tracking_ok"] = sc["tracking_ok"].fillna(False).astype(bool)
    sc = sc.sort_values(["session", "frame"]).reset_index(drop=True)
    rule = {"rule_id": "inv", "behavior": "attack", "min_duration_s": MIN_DUR,
            "alert_threshold": 0.9, "ask_threshold": THRESH, "alert_mode": "log",
            "description": "throwaway investigation rule", "status": "proposed"}
    rules = RuleSet()
    rules.add(rule)
    rules.approve("inv")  # offline script only; the agent itself can never approve
    events = run_monitor(sc, rules.active(), run_id="dryrun_cls")
    bouts = {}
    for e in events:
        if e.get("end_s") is not None:
            bouts.setdefault(e["session"], []).append((e["start_s"], e["end_s"]))
    n_bouts = sum(len(v) for v in bouts.values())
    print(f"{sc.session.nunique()} sessions, {n_bouts} classifier attack bouts >= {MIN_DUR}s")
    for pre_s in PRE_GRID:
        c = dict(overlap=0, edge=0, bad_pre=0, bad_base=0, used=0)
        for sid, bl in bouts.items():
            tok = sc[sc.session == sid]["tracking_ok"].to_numpy()
            for k, (s, e) in enumerate(bl):
                p0, b1 = s - pre_s, s - pre_s - GAP_S
                if any(e2 > p0 and s2 < s for j, (s2, e2) in enumerate(bl) if j != k):
                    c["overlap"] += 1; continue
                up, ub = usable(tok, p0, s), usable(tok, b1 - BASELINE_S, b1)
                if up is None or ub is None: c["edge"] += 1
                elif up < MIN_USABLE: c["bad_pre"] += 1
                elif ub < MIN_USABLE: c["bad_base"] += 1
                else: c["used"] += 1
        print(f"pre_s={pre_s:>4}: {c}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
