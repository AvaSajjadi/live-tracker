"""Tests for flow (the guided steps) and the Upload step."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from live_tracker import demo
from live_tracker import flow
from live_tracker import run_control as rc
from live_tracker import upload_ui
from live_tracker import uploads as up

from pose_fixtures import make_poses
from test_demo import NAME, make_root
from video_fixtures import fake_mp4

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "0_Upload.py")


def texts(elements):
    return " ".join(e.value for e in elements)


# ---- the steps ----

def test_the_steps_run_upload_rules_replay_real_session():
    assert flow.KEYS == ["upload", "rules", "replay", "session"]
    assert [path for _, _, path in flow.STEPS] == ["pages/0_Upload.py", "pages/1_Rules.py", "pages/2_Replay.py",
                                                    "pages/4_Real_Session.py"]
    assert Path(flow.__file__).resolve().parents[2].joinpath(flow.STEPS[0][2]).exists()


def test_every_step_page_exists():
    root = Path(flow.__file__).resolve().parents[2]
    assert all((root / path).exists() for _, _, path in flow.STEPS)


def test_next_and_previous_pages():
    assert flow.next_page("upload") == "pages/1_Rules.py" and flow.previous_page("upload") is None
    assert flow.next_page("rules") == "pages/2_Replay.py" and flow.previous_page("rules") == "pages/0_Upload.py"
    assert flow.next_page("session") is None and flow.previous_page("session") == "pages/2_Replay.py"


def test_the_bar_marks_where_you_are():
    assert flow.bar_text("upload") == "**1. Upload**  >  2. Rules  >  3. Replay  >  4. Real session"
    assert flow.bar_text("replay") == "1. Upload  >  2. Rules  >  **3. Replay**  >  4. Real session"


def nav_app(key, can_next=True):
    def script(key, can_next):
        from live_tracker import flow as f
        f.render_nav(key, can_next=can_next, next_hint="Do this first.")
    return AppTest.from_function(script, args=(key, can_next))


def test_next_goes_to_the_next_page_and_back_to_the_previous(monkeypatch):
    seen = []
    monkeypatch.setattr(st, "switch_page", lambda path: seen.append(path))
    at = nav_app("rules").run()
    [b for b in at.button if b.key == "flow_next_rules"][0].click().run()
    [b for b in at.button if b.key == "flow_back_rules"][0].click().run()
    assert seen == ["pages/2_Replay.py", "pages/0_Upload.py"]


def test_next_is_disabled_with_a_hint_until_the_step_is_done():
    at = nav_app("rules", can_next=False).run()
    assert [b for b in at.button if b.key == "flow_next_rules"][0].disabled
    assert "Do this first." in texts(at.caption)


def test_the_first_step_has_no_back_and_the_last_has_no_next():
    assert not [b for b in nav_app("upload").run().button if b.key == "flow_back_upload"]
    assert not [b for b in nav_app("session").run().button if b.key == "flow_next_session"]


# ---- the Upload step ----

@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "ROOT", make_root(tmp_path / "root"))
    return tmp_path


def playable_demo(root):
    (root / "root" / demo.VIDEO_DIR / f"{NAME}.mp4").write_bytes(fake_mp4(b"avc1", seconds=10.0))


def open_page(state=None):
    at = AppTest.from_file(PAGE, default_timeout=60)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    return at


def page_text(at):
    return " ".join(e.value for kind in (at.markdown, at.caption, at.info, at.success, at.warning, at.error) for e in kind)


def next_button(at):
    return [b for b in at.button if b.key == "flow_next_upload"][0]








def test_nothing_says_connected_before_the_camera_is_clicked(root):
    playable_demo(root)
    at = open_page()
    assert "connected" not in page_text(at).lower()
    at.radio(key="upload_choice").set_value("Upload a video and its keypoints").run()
    assert "connected" not in page_text(at).lower()










def test_a_chosen_centers_source_is_shown_as_motion_only(root):
    source = up.SessionSource("centers", "c.csv", centers="c.csv")
    at = open_page({"flow_source": source})
    assert not at.success
    assert "the agent can score" not in texts(at.caption)


def test_video_state_says_whether_the_browser_can_play_an_upload():
    assert upload_ui.video_state("a.mp4", fake_mp4(b"avc1")) == ("play", None)
    assert upload_ui.video_state("a.webm", b"x") == ("play", None)
    state, why = upload_ui.video_state("a.mp4", fake_mp4(b"mp4v"))
    assert state == "blocked" and "mp4v" in why
    assert upload_ui.video_state("a.mp4", b"junk junk junk junk")[0] == "blocked"


# ---- the router (app.py) ----

ROOT = Path(__file__).resolve().parents[1]


def test_the_menu_lists_the_steps_in_order_then_the_other_pages():
    pages = flow.menu()
    assert list(pages) == ["Guided flow", "More"]
    assert [t for t, _, _ in pages["Guided flow"]] == ["1. Upload", "2. Rules", "3. Replay", "4. Real session"]
    assert [t for t, _, _ in pages["More"]] == ["Investigation", "Live Agent", "Run", "Motion tool"]
    assert [d for _, _, d in pages["Guided flow"]] == [True, False, False, False]
    assert all((ROOT / path).exists() for items in pages.values() for _, path, _ in items)
    assert len({path for items in pages.values() for _, path, _ in items}) == 8


def test_the_app_opens_on_the_upload_step_and_every_page_can_be_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception and [t.value for t in at.title] == ["Upload"]
    for _, path in [(None, p) for _, _, p in flow.STEPS[1:]] + [(None, p) for _, p in flow.OTHER_PAGES]:
        at.switch_page(path).run()
        assert not at.exception, (path, [e.value for e in at.exception])






def test_guided_navigation_preserves_source_when_returning_to_upload(root):
    playable_demo(root)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.radio(key="upload_choice").set_value(upload_ui.CAMERA).run()
    chosen = at.session_state["flow_source"].to_dict()
    next_button(at).click().run()
    assert not at.exception and at.title[0].value == "Monitoring rules"
    at.button(key="flow_back_rules").click().run()
    assert not at.exception and at.title[0].value == "Upload"
    assert at.session_state["flow_source"].to_dict() == chosen and not next_button(at).disabled


def test_the_options_are_camera_video_and_csv_and_nothing_is_chosen_at_first(root):
    at = open_page()
    assert at.radio(key="upload_choice").options == ["Connect a camera", "Upload a video and its keypoints",
                                                     "Upload a CSV of mouse positions"]
    assert at.radio(key="upload_choice").value is None
    assert "Nothing chosen yet." in texts(at.caption) and next_button(at).disabled
    assert "demo recording" not in page_text(at).lower() and "**1. Upload**" in texts(at.markdown)


def test_clicking_the_camera_sets_the_demo_camera_and_shows_the_video_at_once(root, monkeypatch):
    playable_demo(root)
    seen = []
    monkeypatch.setattr(st, "switch_page", lambda path: seen.append(path))
    at = open_page()
    at.radio(key="upload_choice").set_value("Connect a camera").run()
    assert not at.exception
    source = at.session_state["flow_source"]
    assert source.kind == "demo" and source.can_score and source.has_video and source.label.startswith("Demo camera")
    assert len(at.get("video")) == 1                                                 # the video is on the page at once
    assert [s.value for s in at.success][0] == "Camera connected successfully!"        # the green message, exactly
    assert len(at.success) == 1                                                       # no second green box at the bottom
    assert not next_button(at).disabled
    next_button(at).click().run()
    assert seen == ["pages/1_Rules.py"]


def test_the_green_message_always_comes_with_the_demo_mode_line(root):
    """The message is the one asked for; the demo-mode line was removed on request."""
    playable_demo(root)
    at = open_page()
    at.radio(key="upload_choice").set_value("Connect a camera").run()
    assert "Camera connected successfully!" in [s.value for s in at.success]
    assert "Demo mode" not in texts(at.caption)                                       # removed on request
    assert "Camera connected successfully!" not in texts(at.warning) + texts(at.error)


def test_a_video_the_browser_cannot_play_is_explained_and_next_still_works(root):
    at = open_page()                                   # the fake demo video in this root has no playable codec
    at.radio(key="upload_choice").set_value("Connect a camera").run()
    assert not at.get("video") and "cannot be shown in the browser" in texts(at.warning)
    assert not next_button(at).disabled and at.session_state["flow_source"].kind == "demo"


def test_without_a_recorded_video_the_camera_says_so_and_next_stays_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    at = open_page()
    at.radio(key="upload_choice").set_value("Connect a camera").run()
    assert "No recorded video is available" in texts(at.warning) and "mouse pose model" in texts(at.warning)
    assert next_button(at).disabled and "flow_source" not in at.session_state


def test_the_video_option_explains_that_keypoints_are_needed(root):
    at = open_page()
    at.radio(key="upload_choice").set_value("Upload a video and its keypoints").run()
    assert "keypoints of the same recording" in texts(at.markdown) and "A video alone is not enough" in texts(at.markdown)
    assert any(b.key == "upload_use_files" for b in at.button)
    assert "The video appears here as soon as you choose one." in texts(at.caption)


def test_the_csv_option_says_it_gives_motion_only(root):
    at = open_page()
    at.radio(key="upload_choice").set_value("Upload a CSV of mouse positions").run()
    assert "motion and tracking quality only" in texts(at.markdown) and "no behavior scores" in texts(at.markdown)
