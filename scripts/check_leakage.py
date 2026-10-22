"""Final leakage check: test sessions never reach training, tuning or agent inputs.

Checks the committed split and the local (git-ignored) artifacts it can find:

  split       train / validation / test are disjoint; test = every official
              CalMS21 test session, train + validation = every official train session
  models      models/*.json were fit on the manifest's train sessions only: their
              session counts match and n_train_rows equals the row count of the
              train sessions alone (every train_stride-th frame)
  scores      outputs/scores_validation_*.csv hold validation sessions only (these
              feed threshold tuning and the demo); outputs/attack_rule_grid.csv exists
              only next to them
  repo        no git-tracked file other than the split manifest names a test session

Reads labels.parquet (session ids and frame counts only), never test poses or test
labels' contents. Exit code 1 if any check fails.

    python scripts/check_leakage.py
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ALLOWED_TEST_MENTIONS = {"splits/split_manifest.json"}


def check_split(manifest, official_train, official_test):
    tr, va, te = (set(manifest[k]) for k in ("train", "validation", "test"))
    return [
        ("split: train, validation, test are disjoint", not (tr & va or tr & te or va & te), ""),
        ("split: test = all official test sessions", te == set(official_test),
         f"{len(te)} in manifest, {len(set(official_test))} official"),
        ("split: train + validation = all official train sessions", tr | va == set(official_train),
         f"{len(tr | va)} in manifest, {len(set(official_train))} official"),
    ]


def check_model(info, manifest, frames_per_session):
    """frames_per_session: {session: n_frames}. Rows are frames with frame % stride == 0."""
    stride = info["fit_params"]["train_stride"]
    n = sum(-(-frames_per_session[s] // stride) for s in manifest["train"])
    name = info.get("name", "?")
    return [
        (f"model {name}: session counts match the manifest",
         (info["train_sessions"], info["validation_sessions"])
         == (len(manifest["train"]), len(manifest["validation"])),
         f"{info['train_sessions']} train, {info['validation_sessions']} validation"),
        (f"model {name}: n_train_rows = rows of train sessions only", info["n_train_rows"] == n,
         f"{info['n_train_rows']} recorded, {n} expected"),
    ]


def check_scores(path, sessions, manifest):
    s, va = set(sessions), set(manifest["validation"])
    return [(f"scores {Path(path).name}: validation sessions only", s <= va and not s & set(manifest["test"]),
             f"{len(s)} sessions")]


def check_repo(files_text, test_sessions):
    """files_text: {path: text} of tracked files. Session ids are matched by their short name."""
    short = {t.split("/")[-1] for t in test_sessions}
    hits = sorted(p for p, txt in files_text.items()
                  if p not in ALLOWED_TEST_MENTIONS and any(k in txt for k in short))
    return [("repo: no tracked file names a test session", not hits, ", ".join(hits))]


def tracked_text(root):
    out = {}
    paths = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    for p in paths.stdout.split():
        try:
            out[p] = (Path(root) / p).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    root = Path(a.root)
    manifest = json.loads((root / "splits/split_manifest.json").read_text())
    labels = pd.read_parquet(root / "data/processed/labels.parquet", columns=["session", "frame"])
    frames = labels.groupby("session").size().to_dict()
    official_train = [s for s in frames if s.startswith("task1/train/")]
    official_test = [s for s in frames if s.startswith("task1/test/")]

    results = check_split(manifest, official_train, official_test)
    for p in sorted(glob.glob(str(root / "models/*.json"))):
        results += check_model(json.loads(Path(p).read_text()), manifest, frames)
    for p in sorted(glob.glob(str(root / "outputs/scores_validation_*.csv"))):
        results += check_scores(p, pd.read_csv(p, usecols=["session"]).session.unique(), manifest)
    results += check_repo(tracked_text(root), manifest["test"])

    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
