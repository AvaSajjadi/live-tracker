"""Legacy (legacy/1) Stage 7 byte-identity check: render the 14 fixed legacy
cases and compare SHA-256 of the JSON text and Markdown with the values pinned
in tests/fixtures/legacy_golden/manifest.json.

The pinned values were computed with the pre-Findings-v2 source (commit
c62711d, before checkpoint M0). Inputs are saved Stage 6 InvestigationReport
JSON files (re-running Stage 6 is not byte-reproducible: traces record
elapsed times and synthetic caches live in temporary directories):

- 5 synthetic cases, tracked in tests/fixtures/legacy_golden/ (SYNTHETIC
  fixture data only), each rendered with the synthetic test policy and
  without a policy (10 cases);
- real_full (local only: outputs/legacy_golden_inputs/real_full.json, derived
  from the git-ignored CalMS21 validation cache), with the real policy and
  without (2 cases);
- the original Stage 8 live run (local only, read from
  outputs/investigation_live/20261019T220059Z_openrouter/investigation_report.json),
  with the real policy and without (2 cases).

Local-only cases need splits/split_manifest.json (git-ignored) and are
reported as "skipped" when an input is missing.

    python scripts/legacy_golden.py                 # check the current source
    python scripts/legacy_golden.py --src DIR       # check another source tree, e.g.
        mkdir /tmp/base && git archive c62711d src | tar -x -C /tmp/base
        python scripts/legacy_golden.py --src /tmp/base/src

Exit status 0 when every available case is identical, 1 otherwise. Nothing
is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "legacy_golden"
REAL_MANIFEST = ROOT / "splits" / "split_manifest.json"
INPUTS = {
    "effect": FIXTURES / "effect.json", "insufficient": FIXTURES / "insufficient.json",
    "incomplete": FIXTURES / "incomplete.json", "contaminated": FIXTURES / "contaminated.json",
    "partial": FIXTURES / "partial.json",
    "real_full": ROOT / "outputs" / "legacy_golden_inputs" / "real_full.json",
    "live": ROOT / "outputs" / "investigation_live" / "20261019T220059Z_openrouter" / "investigation_report.json",
}
# the synthetic test split (tests/test_investigation_windows.py MANIFEST), repeated so no test module is imported
VAL = [f"task1/train/SYNTHETIC_val_{x}" for x in "abc"]
SYNTHETIC_MANIFEST = {"train": ["task1/train/SYNTHETIC_train"], "validation": VAL, "test": ["task1/test/SYNTHETIC_test"]}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cases():
    """(case name, input name, policy kind) for the 14 cases."""
    out = []
    for name in ("effect", "insufficient", "incomplete", "contaminated", "partial"):
        out += [(name, name, "synthetic"), (f"{name}:nopolicy", name, None)]
    out += [("real_full", "real_full", "real"), ("real_full:nopolicy", "real_full", None),
            ("live", "live", "real"), ("live:nopolicy", "live", None)]
    return out


def render_all(src=None):
    """{case: (json sha256, markdown sha256) or None when its inputs are missing}."""
    if src is not None:
        sys.path.insert(0, str(Path(src).resolve()))
    import live_tracker.investigation_agent as ia
    import live_tracker.investigation_contracts as ic
    import live_tracker.investigation_report as ir
    policies = {"synthetic": ic.EvidencePolicy(SYNTHETIC_MANIFEST, allow_synthetic=True), None: None}
    if REAL_MANIFEST.exists():
        policies["real"] = ic.EvidencePolicy.from_manifest(REAL_MANIFEST)
    out = {}
    for case, name, kind in cases():
        if not INPUTS[name].exists() or kind not in policies:
            out[case] = None
            continue
        report = ia.InvestigationReport(**json.loads(INPUTS[name].read_text()))
        doc, md = ir.render(report, policy=policies[kind])
        out[case] = (sha(ir.to_json_text(doc)), sha(md))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", help="source tree whose live_tracker package renders (default: this checkout)")
    args = ap.parse_args(argv)
    pinned = json.loads((FIXTURES / "manifest.json").read_text())
    for name, digest in pinned["inputs"].items():
        path = INPUTS[name]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            print(f"INPUT CHANGED {name}: {path}")
            return 1
    bad = 0
    for case, got in render_all(args.src).items():
        if got is None:
            print(f"skipped {case} (local input or real manifest missing)")
        elif tuple(pinned["renders"][case]) == got:
            print(f"OK      {case}")
        else:
            bad += 1
            print(f"DIFF    {case}")
    print("identical" if not bad else f"{bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
