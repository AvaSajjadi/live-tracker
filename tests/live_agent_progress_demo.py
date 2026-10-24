"""A SAFE local demo for watching UI-3's live progress in a browser. Not part of the app and not a test (pytest does
not collect it). Run from the repository root:

    PYTHONPATH=src:tests ~/live-tracker/.venv/bin/streamlit run tests/live_agent_progress_demo.py \\
        --server.address 127.0.0.1 --server.port 18503 --server.headless true

then open http://127.0.0.1:18503 and click "Run free rehearsal".

It builds, in a fresh temporary directory, SYNTHETIC validation inputs (five synthetic sessions; never CalMS21 data)
and a recorded synthetic reference run, and points the Live Agent page at them. Only the FREE REHEARSAL runs: its
emulated endpoint (the offline investigator) is slowed by a chosen delay per request so the live events can be
watched. The paid path cannot run here: the process holds no API key and every paid authorization is refused. The
network guard (tests/demo_network_guard.py) refuses, in this Python process, every non-loopback connection, datagram
and name resolution made through the socket module; loopback stays open for Streamlit itself (see that module for
its boundaries). Production runs are never slowed: the delay exists only in this file.
"""

import os
import tempfile
import time
from pathlib import Path

import streamlit as st

import live_tracker.investigation_live_v1 as lv1
import live_tracker.investigation_v1_runs as runs
import live_tracker.investigation_v1_ui as ui
import demo_network_guard
from test_investigation_v1_validation import recorded_run
from test_live_agent_evidence import build_gate_inputs

LR = runs.script("investigation_v1_live")
BEHAVIOURS = {"normal (the offline investigator's calls)": {},
              "one refused call first (an unknown tool)": {1: "unknown_tool"},
              "provider error at request 3 (incomplete run)": {3: "http_500"}}


class SlowEmulator(LR.EmulatedOpenRouterV1):
    """The rehearsal's emulated endpoint, answering each request after `delay_s` (demo only)."""

    def __init__(self, delay_s, faults):
        super().__init__(faults)
        self.delay_s = delay_s

    def __call__(self, request):
        time.sleep(self.delay_s)
        return super().__call__(request)


def no_network(*a, **k):
    raise RuntimeError("the progress demo makes no network connection")


def no_paid_run(*a, **k):
    raise lv1.LiveRefused("the progress demo never runs a paid investigation")


@st.cache_resource
def synthetic_world():
    """Built once per server process: SYNTHETIC inputs and a recorded synthetic reference run."""
    tmp = Path(tempfile.mkdtemp(prefix="live_agent_progress_demo_"))
    inputs = build_gate_inputs(tmp / "inputs")
    reference_runs = recorded_run(inputs, "phase2", tmp)
    return tmp, inputs, reference_runs


st.set_page_config(page_title="Live Agent progress demo", layout="wide")
tmp, inputs, reference_runs = synthetic_world()
for key in ("OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(key, None)
os.environ[runs.LIVE_DIR_ENV] = str(tmp / "live")
os.environ[runs.OFFLINE_DIR_ENV] = str(tmp / "offline")
os.environ["LIVE_TRACKER_VALIDATION_RUNS"] = str(reference_runs)
demo_network_guard.install()
lv1.get_json_no_redirects = no_network
lv1.authorize_live = no_paid_run
runs.HTTP_CLIENT_FACTORY = None
runs.REFERENCE_RUN = "phase2"
runs.default_inputs = lambda: runs.ValidationInputs(
    label="SYNTHETIC demo inputs (five synthetic sessions; not CalMS21 data)", archive=str(inputs["zip"]),
    archive_sha256=inputs["sha"], cache_dir=str(inputs["cache"]), manifest=str(inputs["manifest"]),
    runs_dir=str(reference_runs), allow_synthetic=True)

st.sidebar.warning("DEMO: synthetic inputs, emulated endpoint, no key, paid runs refused, non-loopback network "
                   "refused in this process, no spending.")
delay = st.sidebar.slider("Emulated reply delay (s)", 0.5, 6.0, 2.0, 0.5)
behaviour = st.sidebar.selectbox("Emulated endpoint behaviour", list(BEHAVIOURS))
runs.REHEARSAL_EMULATOR_FACTORY = lambda: SlowEmulator(delay, BEHAVIOURS[behaviour])
st.sidebar.caption(f"Files: {tmp}")

st.title("Live Agent (progress demo)")
st.caption("The real Live Agent page code over SYNTHETIC inputs. Click Run free rehearsal and watch the progress.")
message = st.session_state.pop(ui.MESSAGE_KEY, None)
# The one-time completion message sits in a container that is ALWAYS rendered: when the message is gone on the next
# rerun, the tabs below keep their place in the page, so Streamlit keeps the selected tab (a tabs element that moves
# up one position is a new element to the browser, which resets it to "New investigation").
notice = st.container()
if message:
    (notice.error if message[0] == "error" else notice.info)(message[1])
tab_new, tab_saved = st.tabs(["New investigation", "Saved investigations"])
with tab_new:
    ui.render_new()
with tab_saved:
    ui.render_open()
