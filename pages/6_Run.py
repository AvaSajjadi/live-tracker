"""Run page: start a run on the lab computer, watch it, answer its questions, read its report (new file).

Rendering only; see live_tracker.run_ui and live_tracker.run_control. The run is a separate program
(live_tracker.monitor_runner), so closing this page does not stop it.
"""

from live_tracker import run_ui, ui_style

ui_style.apply()
run_ui.render()
