"""Live Agent page: the Stage 9 autonomous investigation agent on the verified CalMS21 validation inputs (new file).

A researcher asks a question; the agent (Claude through the guarded OpenRouter adapter) chooses every Contract v1
call; the saved report is re-verified before it is shown. Rendering: live_tracker.investigation_v1_ui; logic:
live_tracker.investigation_v1_runs (the reviewed CLI underneath). Paid requests happen only after a ready preflight,
an explicit confirmation and the typed authorization text.
"""

from live_tracker import investigation_v1_ui as ui
from live_tracker import ui_style

ui_style.apply()
ui.render_page()
