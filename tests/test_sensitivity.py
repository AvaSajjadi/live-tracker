"""Tests for scripts/sensitivity.py and for keeping live paths causal (new file)."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from live_tracker.formats import wide_to_long

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sensitivity", ROOT / "scripts" / "sensitivity.py")
sens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sens)


@pytest.fixture(scope="module")
def long_df():
    return wide_to_long(pd.read_csv(ROOT / "example_pose_demo.csv"), session="SYNTHETIC_demo")


@pytest.fixture(scope="module")
def table(long_df):
    return sens.run_sensitivity(long_df, repeats=2, seed=3)


def _val(table, kind, setting, feature, col="norm_median_diff"):
    row = table[(table.disturbance == kind) & (table.setting == setting) & (table.feature == feature)]
    return float(row[col].iloc[0])


def test_table_shape_and_columns(table):
    assert {"disturbance", "setting", "feature", "n_compared", "median_abs_diff",
            "p95_abs_diff", "norm_median_diff", "frac_nan_changed"} <= set(table.columns)
    assert set(table.disturbance) == {"jitter", "gaps", "smoothing", "threshold"}
    assert "speed_a" in set(table.feature) and "dist_center" in set(table.feature)
    # every threshold gets a -20 % and a +20 % row
    thr = table[table.disturbance == "threshold"].setting.unique()
    assert len(thr) == 2 * len(sens.THRESHOLD_FIELDS)


def test_more_jitter_changes_speed_more(table):
    s1, s5 = (_val(table, "jitter", f"{s} px", "speed_a", "median_abs_diff") for s in ("1", "5"))
    assert s5 > s1 > 0


def test_unchanged_setting_gives_zero_difference(long_df):
    base = sens.compute_features(long_df)
    same = sens.compare_tables(base, sens.compute_features(long_df))
    assert (same["median_abs_diff"].dropna() == 0).all()
    assert (same["frac_nan_changed"] == 0).all()


def test_long_gaps_change_which_frames_have_values(table):
    assert _val(table, "gaps", "10 frames", "speed_a", "frac_nan_changed") > 0
    # short gaps are filled, so far fewer frames lose their value
    assert (_val(table, "gaps", "2 frames", "speed_a", "frac_nan_changed")
            < _val(table, "gaps", "10 frames", "speed_a", "frac_nan_changed"))


def test_same_seed_same_result(long_df):
    a = sens.run_sensitivity(long_df, repeats=2, seed=7)
    b = sens.run_sensitivity(long_df, repeats=2, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_heading_difference_is_wrapped():
    base = pd.DataFrame({"heading_a": [np.pi - 0.01]})
    pert = pd.DataFrame({"heading_a": [-np.pi + 0.01]})
    row = sens._compare(base, pert, "heading_a")
    assert row["median_abs_diff"] == pytest.approx(0.02)


def test_mismatched_frames_are_rejected(long_df):
    a = sens.compute_features(long_df)
    with pytest.raises(ValueError, match="different frames"):
        sens.compare_tables(a, a.iloc[:-5])


def test_main_writes_csv(tmp_path):
    out = tmp_path / "sens.csv"
    sens.main(["--repeats", "1", "--out", str(out)])
    assert len(pd.read_csv(out)) > 0


# --- live paths stay causal -------------------------------------------------------

OFFLINE_ONLY = re.compile(r"\b(clean_trajectory|frame_features|savgol_filter|motion import)\b")
# the starter module itself and the labeled offline section of the dashboard may use them
ALLOWED = {"motion.py"}


def test_live_code_never_uses_offline_smoothing():
    offenders = [p.name for p in (ROOT / "src" / "live_tracker").glob("*.py")
                 if p.name not in ALLOWED and OFFLINE_ONLY.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_summary_names_most_changed_feature(table):
    s = sens.summarize(table)
    assert len(s) == len(table[["disturbance", "setting"]].drop_duplicates())
    row = s[(s.disturbance == "gaps") & (s.setting == "10 frames")].iloc[0]
    assert row["max_frac_nan_changed"] > 0


# --- real CalMS21 sessions: the manifest guard (the JSON is never opened for a refused request) ---

def _manifest_file(tmp_path):
    import json
    m = {"seed": 0, "train": ["task1/train/mouse001"], "validation": ["task1/train/mouse002"],
         "test": ["task1/test/mouse071"]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    return path


@pytest.mark.parametrize("session", ["task1/test/mouse071", "task1/train/not_in_manifest"])
def test_real_sessions_refuse_test_and_unknown_before_opening_the_json(tmp_path, session):
    missing_json = tmp_path / "does_not_exist.json"          # opening it would raise FileNotFoundError
    with pytest.raises(SystemExit, match="refused"):
        sens.load_real_sessions(missing_json, [session], _manifest_file(tmp_path))


def test_real_sessions_one_test_session_among_valid_ones_refuses_all(tmp_path):
    with pytest.raises(SystemExit, match="mouse071"):
        sens.load_real_sessions(tmp_path / "x.json", ["task1/train/mouse002", "task1/test/mouse071"],
                                _manifest_file(tmp_path))


def test_calms21_option_needs_sessions(tmp_path):
    with pytest.raises(SystemExit, match="--sessions"):
        sens.main(["--calms21", str(tmp_path / "x.json"), "--out", str(tmp_path / "o.csv")])
