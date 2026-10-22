"""Contract v1 follow-up policy in the actual Stage 9 agent loop
(investigation_agent): at most 2 compare_windows analyses after the first,
12 tool calls, and with CONTRACT_V1_BUDGET pre windows of 1, 2 or 5 s only.
New file. SYNTHETIC cache only (test_investigation_windows.build); no network.
"""

import pytest

import live_tracker.investigation_agent as ia
import live_tracker.investigation_contracts as ic
from live_tracker.investigation_agent import Budget, investigate
from test_investigation_agent import QUESTION, Scripted, call, history, no_network  # noqa: F401  (blocks sockets)
from test_investigation_windows import build

FEATURES = ["dist_rate", "dist_center", "rel_speed", "close_time_s", "speed_corr_w"]


def evidence(messages, tool):
    return next(r["evidence_id"] for name, _, r in reversed(history(messages)) if name == tool and "evidence_id" in r)


def script(length, n_compares):
    turns = [[call("query_events", source="validation_annotation", behavior="attack")],
             lambda m: [call("get_window_features", episodes_from={"evidence_id": evidence(m, "query_events")},
                             pre_event={"length_frames": length, "gap_frames": 0},
                             baseline={"length_frames": length, "clearance_frames": 10, "max_windows_per_session": 500},
                             features=FEATURES)]]
    for f in FEATURES[:n_compares]:                       # distinct requests: never refused as repeats
        turns.append(lambda m, f=f: [call("compare_windows", feature=f,
                                          windows_from={"evidence_id": evidence(m, "get_window_features")})])
    return Scripted(*turns)


def compares(report):
    executed = [e for e in report.evidence if e["tool"] == "compare_windows"]
    refused = [t for t in report.trace if t["kind"] == "tool_call" and t.get("tool") == "compare_windows"
               and t.get("outcome") == "refused"]
    return executed, refused


@pytest.fixture
def world(tmp_path):
    tools, policy, _, _ = build(tmp_path)
    return tools, policy


def test_a_fourth_comparison_never_executes(world):
    report = investigate(QUESTION, *world, script(20, 5))
    executed, refused = compares(report)
    assert len(executed) == 3                                         # the first + 2 follow-ups
    assert all(e["result"]["status"] != "error" for e in executed)
    assert [t["refused"] for t in refused] == ["follow_up_limit", "follow_up_limit"]
    assert "not executed" in refused[0]["message"]
    assert len(report.evidence) <= ia.MAX_TOOL_CALLS


def test_three_comparisons_are_allowed(world):
    executed, refused = compares(investigate(QUESTION, *world, script(20, 3)))
    assert len(executed) == 3 and refused == []


def test_contract_profile_enforces_the_preselected_grid(world):
    off_grid = investigate(QUESTION, *world, script(20, 1), budget=ia.CONTRACT_V1_BUDGET)
    executed, refused = compares(off_grid)
    assert executed == [] and refused[0]["refused"] == "pre_window_not_in_grid"
    on_grid = investigate(QUESTION, *world, script(30, 4), budget=ia.CONTRACT_V1_BUDGET)  # 30 frames = 1 s
    executed, refused = compares(on_grid)
    assert len(executed) == 3 and [t["refused"] for t in refused] == ["follow_up_limit"]


def test_the_loop_and_the_v1_adapter_share_one_budget_policy():
    import live_tracker.investigation_tools_v1 as v1
    assert ia.MAX_TOOL_CALLS == v1.MAX_TOOL_CALLS == ic.CONTRACT_V1_MAX_TOOL_CALLS == 12
    assert ia.MAX_FOLLOW_UPS == v1.MAX_FOLLOW_UPS == ic.CONTRACT_V1_MAX_FOLLOW_UPS == 2
    assert v1.FOLLOW_UP_PRE_S == ic.CONTRACT_V1_PRE_S_GRID
    assert tuple(int(s * 30) for s in ic.CONTRACT_V1_PRE_S_GRID) == ic.CONTRACT_V1_PRE_WINDOW_GRID_FRAMES
    assert ia.CONTRACT_V1_BUDGET.pre_window_grid_frames == ic.CONTRACT_V1_PRE_WINDOW_GRID_FRAMES
    for bad in (dict(max_tool_calls=13), dict(max_compare_follow_ups=3), dict(pre_window_grid_frames=(0,))):
        with pytest.raises(ValueError):
            Budget(**bad)
