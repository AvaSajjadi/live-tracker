"""Download the three MARS model files from the GitHub release into models/mars (new file).

The two detection files (230 MB each) are stored on the release in 50 MB pieces; this script joins them and checks the md5 of
every file against the Caltech originals. A file that is already there with the right md5 is not downloaded again.

    python scripts/download_mars_models.py
    python scripts/download_mars_models.py --models-dir ~/mars_local/models

MARS: Segalin et al. 2021 (eLife), CC BY-NC 4.0, from Caltech Data record 801be-66057. See models/mars/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/AvaSajjadi/live-tracker/releases/download/mars-models-v1"
ROOT = Path(__file__).resolve().parents[1]
PARTS = "abcde"
FILES = {                                   # file: (where it goes, md5, stored in pieces)
    "MARS_top_detection_black.pb": ("detection", "79f9b55a8eff1192b899e55cc0ea570c", True),
    "MARS_top_detection_white.pb": ("detection", "4469996d5d7003c8c05d6279ed1506af", True),
    "MARS_top_pose.pb": ("pose", "d380063a5177545721103a34fa650b56", False),
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, out):
    """Append the release file `name` to the open file `out`."""
    with urllib.request.urlopen(f"{RELEASE}/{name}") as r:
        shutil.copyfileobj(r, out, 1 << 20)


def download(models_dir):
    for name, (folder, expected, pieces) in FILES.items():
        target = Path(models_dir) / folder / name
        if target.is_file() and md5(target) == expected:
            print(f"{name}: already there")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".download")
        print(f"{name}: downloading ...", flush=True)
        with open(tmp, "wb") as out:
            for part in ([f"{name}.part-{p}" for p in PARTS] if pieces else [name]):
                fetch(part, out)
        if md5(tmp) != expected:
            tmp.unlink()
            raise SystemExit(f"{name}: the download is damaged (md5 does not match {expected}); run the script again")
        tmp.replace(target)
        print(f"{name}: ok")
    print("MARS models ready in", models_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models" / "mars")
    download(ap.parse_args(argv).models_dir.expanduser())
    return 0


if __name__ == "__main__":
    sys.exit(main())
