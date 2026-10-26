"""Save the pose keypoints of ONE CalMS21 session as a small Parquet file, so a demo starts at once.

Reading the 1.2 GB train JSON takes about 15 s every time; this does it once. Only train and validation
sessions are allowed: a test session is refused, and so is the test file.

    python scripts/prepare_demo_session.py --session task1/train/mouse064_task1_annotator1 \
        --out data/raw/calms21_video/mouse064_poses.parquet
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_tracker.calms21 import _read_json, _sessions, check_manifest, load_calms21  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--json", default="data/raw/calms21/calms21_task1_train.json")
    ap.add_argument("--manifest", default="splits/split_manifest.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    manifest = check_manifest(json.loads((ROOT / a.manifest).read_text(encoding="utf-8")))
    if "test" in Path(a.json).name.lower() or a.session in manifest["test"]:
        raise SystemExit("refused: test-split sessions are never used by the agent")
    if a.session not in manifest["train"] + manifest["validation"]:
        raise SystemExit(f"refused: {a.session!r} is not in the train or validation split")
    records = _sessions(_read_json(ROOT / a.json))
    if a.session not in records:
        raise SystemExit(f"{a.session!r} is not in {a.json}")
    poses, _ = load_calms21({"g": {a.session: records[a.session]}})
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    poses.to_parquet(out, index=False)
    print(f"saved {len(poses)} rows ({poses['frame'].nunique()} frames, {poses['frame'].nunique() / 30:.1f} s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
