"""Tests for scripts/dryrun_contract_options.py (new file).

Runs on the SYNTHETIC five-session data of scripts/contract_v1_demo.py. The numbers say nothing
about mice; they check the counting logic and that stricter overlap rules never keep more windows.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dry = _load("dryrun_contract_options", "scripts/dryrun_contract_options.py")
demo = _load("contract_v1_demo", "scripts/contract_v1_demo.py")


@pytest.fixture(scope="module")
def data():
    return demo.synthetic_data()


def test_runs_are_half_open_frame_pairs():
    assert dry.runs([0, 1, 1, 0, 0, 1, 0, 1, 1]) == [(1, 3), (5, 6), (7, 9)]
    assert dry.runs([0, 0, 0]) == []
    assert dry.runs([1, 1]) == [(0, 2)]


def test_overlap_is_half_open_and_skips_the_bout_itself():
    spans = [(10, 20), (40, 50)]
    assert dry.overlaps(spans, 15, 25)
    assert not dry.overlaps(spans, 20, 40)          # touching ends do not overlap
    assert dry.overlaps(spans, 49, 60)
    assert not dry.overlaps(spans, 10, 30, skip_start=10)   # the only overlapping span is the bout itself
    assert dry.overlaps(spans + [(25, 28)], 10, 30, skip_start=10)


def test_four_variants_for_each_pre_s(data, capsys):
    rows = dry.table(data, "attack", min_duration_s=1.0, threshold=0.5)
    assert len(rows) == len(dry.VARIANTS) * len(dry.PRE_GRID)
    assert {r["variant"] for r in rows} == set(dry.VARIANTS)
    assert "synthetic" in capsys.readouterr().out


def test_stricter_rules_never_keep_more_windows(data):
    rows = dry.table(data, "attack", min_duration_s=1.0, threshold=0.5)
    used = {(r["variant"], r["pre_s"]): r["used"] for r in rows}
    names = list(dry.VARIANTS)
    v0, v1, v2, v3 = names
    for pre_s in dry.PRE_GRID:
        assert used[v1, pre_s] <= used[v0, pre_s]       # adding baseline exclusion
        assert used[v2, pre_s] <= used[v0, pre_s]       # any run excludes at least what fired bouts do
        assert used[v3, pre_s] <= min(used[v1, pre_s], used[v2, pre_s])


def test_as_built_variant_matches_compare_windows(data):
    """V0 must reproduce what the adapter's own compare_windows returns."""
    runner = dry.v1.InvestigationRunner(data, "check")
    q = runner.call("query_events", {"session": list(data.sessions), "behavior": "attack",
                                     "min_duration_s": 1.0, "threshold": 0.5})
    rows = {(r["variant"], r["pre_s"]): r for r in dry.table(data, "attack", 1.0, 0.5)}
    v0 = list(dry.VARIANTS)[0]
    for pre_s in dry.PRE_GRID:
        c = runner.call("compare_windows", {"query_run_id": q["query_run_id"], "query": q["query"],
                                            "bouts": q["bouts"], "pre_s": float(pre_s), "baseline_s": 5.0,
                                            "baseline_gap_s": 1.0, "feature": "dist_center"})
        assert c["n_used"] == rows[v0, pre_s]["used"]
        assert c["n_sessions"] == rows[v0, pre_s]["sessions"]
        if pre_s != dry.PRE_GRID[-1]:
            runner = dry.v1.InvestigationRunner(data, f"check{pre_s}")      # fresh budget per analysis
            q = runner.call("query_events", {"session": list(data.sessions), "behavior": "attack",
                                             "min_duration_s": 1.0, "threshold": 0.5})


def test_a_query_with_no_bouts_gives_zero_counts_and_fails_the_gate(data):
    rows = dry.table(data, "mount", min_duration_s=1.0, threshold=0.99)    # synthetic data has no such bouts
    assert rows and all(r["used"] == 0 and r["sessions"] == 0 and not r["gate"] for r in rows)


def test_numpy_runs_accepts_boolean_arrays():
    assert dry.runs(np.array([False, True, True])) == [(1, 3)]
