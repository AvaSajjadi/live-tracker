"""Dry run: how do the two open Contract v1 overlap choices change the usable-window counts?

Counts only; no effect is estimated and no effect value is computed (the choice must be made
before anyone looks at effect values). Uses Syed's real Contract v1 adapter
(investigation_tools_v1) on the VALIDATION sessions, so the numbers are what the tools return.

Two choices (docs/contract_v1_integration.md section 6, items 3 and 4):
  pre-window overlap   which "other bouts" exclude a pre-event window
      fired     other bouts of the cited query (a run above threshold for at least min_duration_s)
      any_run   any run of frames at or above the threshold, however short
  baseline overlap     is a baseline window that contains a bout excluded?
      none      no (the contract's wording; what the adapter does today)
      fired / any_run   yes, by the same two definitions

Variants run (V0 is what the adapter does today):
  V0 pre=fired,   baseline=none
  V1 pre=fired,   baseline=fired
  V2 pre=any_run, baseline=none       (with ground-truth labels this is the definition of
                                       scripts/compare_quality_policies.py: its counts should match)
  V3 pre=any_run, baseline=any_run    (the most conservative)

Everything else is the adapter's own: quality_v1, 30 fps rule, no interpolation, windows anchored
on start_s. Evidence gate (contract): at least 10 used windows in at least 4 sessions.

    python scripts/dryrun_contract_options.py                      # ground-truth validation labels
    python scripts/dryrun_contract_options.py --score-drop drop.zip --member scores_validation_model_b.csv

Validation sessions only: the cache and the score loader both refuse anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker import investigation_contracts as ic  # noqa: E402
from live_tracker import investigation_tools_v1 as v1  # noqa: E402
from live_tracker import score_drops  # noqa: E402
from live_tracker import timebase as tb  # noqa: E402
from live_tracker.investigation_events import AnnotationEventSource  # noqa: E402

VARIANTS = {
    "V0 pre=fired,   baseline=none (as built)": ("fired", "none"),
    "V1 pre=fired,   baseline=fired": ("fired", "fired"),
    "V2 pre=any_run, baseline=none": ("any_run", "none"),
    "V3 pre=any_run, baseline=any_run": ("any_run", "any_run"),
}
PRE_GRID = ic.CONTRACT_V1_PRE_S_GRID
BASELINE_S, GAP_S = 5.0, 1.0
GATE_WINDOWS, GATE_SESSIONS = v1.DEFAULT_MIN_N_WINDOWS, v1.DEFAULT_MIN_N_SESSIONS


def runs(mask):
    """(start, end) frame pairs of True runs; end is exclusive."""
    m = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    ends = np.flatnonzero(m & ~np.r_[m[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def overlaps(spans, a, z, skip_start=None):
    """Does any (start, end) span overlap frames [a, z)? A span starting at skip_start is the bout itself."""
    return any(e > a and s < z for s, e in spans if s != skip_start)


def count_variant(data, bouts, behavior, threshold, pre_s, mode_pre, mode_base, policy):
    """Exclusion reasons and the used windows for one variant and one pre_s."""
    fired = {}
    for b in bouts:
        fired.setdefault(b["session"], []).append((tb.boundary_frame(b["start_s"]), tb.boundary_frame(b["end_s"])))
    anyrun = {sid: runs(data._get(sid).scores[behavior].to_numpy(float) >= threshold) for sid in data.sessions}
    spans = {"fired": fired, "any_run": anyrun}
    why_n, used = Counter(), []
    for i, b in enumerate(bouts):
        sid, pose = b["session"], data._get(b["session"]).pose
        S, p0, b0, b1, before_start = v1._bout_windows(i, b, pre_s, BASELINE_S, GAP_S)
        if overlaps(spans[mode_pre].get(sid, []), p0, S, skip_start=S):
            why = "pre overlaps"
        elif before_start:
            why = "session edge"
        elif v1._assess(pose, p0, S, policy)["verdict"] != "reliable":
            why = "bad pre"
        elif v1._assess(pose, b0, b1, policy)["verdict"] != "reliable":
            why = "bad baseline"
        elif mode_base != "none" and overlaps(spans[mode_base].get(sid, []), b0, b1):
            why = "baseline overlaps"
        elif not v1._values(pose, p0, S, "dist_center").size or not v1._values(pose, b0, b1, "dist_center").size:
            why = "no values"
        else:
            why = None
        if why:
            why_n[why] += 1
        else:
            used.append(sid)
    return why_n, used


def table(data, behavior="attack", min_duration_s=1.0, threshold=0.5):
    runner = v1.InvestigationRunner(data, "dryrun-contract-options")
    q = runner.call("query_events", {"session": list(data.sessions), "behavior": behavior,
                                     "min_duration_s": min_duration_s, "threshold": threshold})
    if q["status"] != "ok":
        raise SystemExit(f"query_events refused: {q}")
    policy = v1.QUALITY_POLICIES[v1.DEFAULT_QUALITY_POLICY]
    print(f"\n{data.data_source}: {len(data.sessions)} validation sessions; {behavior} bouts >= {min_duration_s} s "
          f"at threshold {threshold}: {q['n_bouts']} in {q['n_sessions']} sessions; policy {policy.name}")
    header = f"{'variant':44s} {'pre_s':>5s} {'overlap':>7s} {'edge':>5s} {'bad':>4s} {'base ov':>7s} {'used':>5s} {'sess':>4s}  gate"
    print(header)
    print("-" * len(header))
    rows = []
    for name, (mp, mb) in VARIANTS.items():
        for pre_s in PRE_GRID:
            why, used = count_variant(data, q["bouts"], behavior, threshold, float(pre_s), mp, mb, policy)
            n_s = len(set(used))
            passes = len(used) >= GATE_WINDOWS and n_s >= GATE_SESSIONS
            bad = why["bad pre"] + why["bad baseline"] + why["no values"]
            print(f"{name:44s} {pre_s:5.1f} {why['pre overlaps']:7d} {why['session edge']:5d} {bad:4d} "
                  f"{why['baseline overlaps']:7d} {len(used):5d} {n_s:4d}  {'pass' if passes else 'FAIL'}")
            rows.append({"variant": name, "pre_s": pre_s, "used": len(used), "sessions": n_s, "gate": passes})
        print()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", default=str(ROOT / "data" / "raw" / "calms21_validation"))
    ap.add_argument("--manifest", default=str(ROOT / "splits" / "split_manifest.json"))
    ap.add_argument("--behavior", default="attack")
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--score-drop", help="verified validation score archive (ZIP); default: ground-truth labels")
    ap.add_argument("--member", help="score member inside --score-drop, e.g. scores_validation_model_b.csv")
    ap.add_argument("--out", help="write the counts as JSON here (never overwritten)")
    a = ap.parse_args(argv)

    manifest = json.loads(Path(a.manifest).read_text())
    policy = ic.EvidencePolicy(manifest)
    pose_source = AnnotationEventSource(a.cache, a.manifest, policy)
    if a.score_drop:
        if not a.member:
            raise SystemExit("--score-drop needs --member")
        drop = score_drops.load_validation_scores(a.score_drop, a.member, manifest_path=a.manifest)
        data = v1.InvestigationData.from_classifier(drop, pose_source, policy)
    else:
        data = v1.InvestigationData.from_annotations(pose_source)
    rows = table(data, a.behavior, a.min_duration, a.threshold)
    if a.out:
        out = Path(a.out)
        with open(out, "x", encoding="utf-8") as f:
            json.dump({"source": data.data_source, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
