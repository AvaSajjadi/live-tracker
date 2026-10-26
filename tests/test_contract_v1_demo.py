"""Tests for scripts/contract_v1_demo.py: the deterministic offline Contract v1
demo on SYNTHETIC data. New file. No network, no API key, no real data."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("contract_v1_demo", ROOT / "scripts" / "contract_v1_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_demo_is_deterministic_and_records_every_attempt(tmp_path, capsys):
    out = tmp_path / "demo.json"
    assert demo.main(["--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert demo.main([]) == 0
    assert json.loads(capsys.readouterr().out) == record                       # same bytes, same record
    assert record["data_source"] == "synthetic" and "SYNTHETIC DATA ONLY" in record["note"]
    steps = [(a["tool"], a["status"], a["code"]) for a in record["attempts"]]
    assert steps == [("query_events", "ok", None), ("check_tracking_quality", "ok", None),
                     ("get_window_features", "ok", None), ("compare_windows", "ok", None),
                     ("compare_windows", "ok", None), ("compare_windows", "ok", None),
                     ("compare_windows", "error", "invalid_request")]          # a third follow-up is refused
    assert record["calls_used"] == 7 and record["max_tool_calls"] == 12
    assert demo.main(["--out", str(out)]) == 2                                 # never overwritten
