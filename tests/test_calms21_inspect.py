"""Tests for the schema inspector, using a small made-up JSON.

The toy structure below is NOT the real CalMS21 layout; it only checks that
the inspector reports keys, lengths, and shapes correctly.
"""

import json

import pytest

from live_tracker.calms21_inspect import describe, load_calms21, main


TOY = {"group": {"seq_a": {"poses": [[[1.0, 2.0], [3.0, 4.0]]] * 4,
                           "labels": [0, 1, 1, 2]}}}


def test_describe_reports_shapes():
    text = "\n".join(describe(TOY))
    assert "dict with 1 keys" in text
    assert "shape=(4, 2, 2)" in text
    assert "shape=(4,)" in text


def test_describe_ragged_list():
    text = "\n".join(describe([[1, 2], [3]]))
    assert "list of length 2" in text


def test_describe_truncates_keys():
    text = "\n".join(describe({str(i): i for i in range(10)}, max_keys=2))
    assert "8 more keys" in text


def test_cli_runs(tmp_path, capsys):
    p = tmp_path / "toy.json"
    p.write_text(json.dumps(TOY))
    assert main([str(p)]) == 0
    assert "shape=(4, 2, 2)" in capsys.readouterr().out


def test_loader_is_reexported():
    from live_tracker import calms21
    assert load_calms21 is calms21.load_calms21
