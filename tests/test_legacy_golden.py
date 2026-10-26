"""Legacy (legacy/1) byte identity: the 14 fixed cases of
scripts/legacy_golden.py render to the JSON and Markdown SHA-256 values pinned
in tests/fixtures/legacy_golden/manifest.json (computed with the pre-M0 source,
commit c62711d). The 10 synthetic cases always run; the 4 local-only cases
(real_full, the original live run) need git-ignored inputs and are skipped
without them."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_investigation_context import LEGACY_RENDER

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("legacy_golden", ROOT / "scripts" / "legacy_golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)
PINNED = json.loads((golden.FIXTURES / "manifest.json").read_text())
RENDERS = golden.render_all()


def test_the_manifest_pins_fourteen_cases_and_the_m0_live_hashes():
    assert len(PINNED["renders"]) == 14 and PINNED["baseline_commit"] == "c62711d"
    assert tuple(PINNED["renders"]["live"]) == LEGACY_RENDER["policy"]
    assert tuple(PINNED["renders"]["live:nopolicy"]) == LEGACY_RENDER["no_policy"]


@pytest.mark.parametrize("case", sorted(PINNED["renders"]))
def test_legacy_rendering_is_byte_identical(case):
    name = next(n for c, n, _ in golden.cases() if c == case)
    path = golden.INPUTS[name]
    if RENDERS[case] is None:
        assert not path.is_relative_to(golden.FIXTURES), "a tracked synthetic fixture is missing"
        pytest.skip("local-only input or real manifest not present")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINNED["inputs"][name]
    assert RENDERS[case] == tuple(PINNED["renders"][case])
