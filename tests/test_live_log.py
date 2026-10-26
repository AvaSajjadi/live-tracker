from types import SimpleNamespace

import pandas as pd

from live_tracker import live_log


def pose(frame, lost_b=False):
    rows = []
    for mouse, x in (("a", 100.0), ("b", 300.0)):
        for kp in ("nose", "neck", "left_hip", "right_hip"):
            lost = mouse == "b" and lost_b
            rows.append({"mouse": mouse, "keypoint": kp, "time_s": frame / 30, "x": None if lost else x + frame,
                         "y": None if lost else 50.0})
    return SimpleNamespace(frame=frame, pose=pd.DataFrame(rows))


def test_the_run_log_gives_the_page_scores_and_motion(tmp_path):
    log = live_log.LiveLog(tmp_path)
    for f in range(10):
        log.position(pose(f, lost_b=f == 9))
        log.score({"frame": f, "time_s": f / 30, "attack": 0.1 * f, "investigation": 0.2, "mount": 0.0, "tracking_ok": f != 9})
    log.close()
    scores = live_log.read_scores(tmp_path, 30)
    assert list(scores.columns) == ["frame", "time_s", "attack", "investigation", "mount", "tracking_ok"]
    assert len(scores) == 10 and scores["attack"].iloc[-1] == 0.9
    motion = live_log.read_motion(tmp_path, 30, 30.0)
    assert round(motion["speed_a"].iloc[5]) == 30                     # 1 px per frame at 30 fps
    assert motion["distance"].iloc[0] == 200
    assert not motion["tracking_ok"].iloc[-1] and pd.isna(motion["distance"].iloc[-1])   # a lost mouse is a gap, not a still mouse


def test_nothing_to_chart_before_the_first_line(tmp_path):
    assert live_log.read_scores(tmp_path, 30) is None and live_log.read_motion(tmp_path, 30, 30.0) is None
    live_log.LiveLog(tmp_path).close()
    assert live_log.read_scores(tmp_path, 30) is None
