# Live Tracker - ScopeScore-inspired motion engine

This starter implements generic 2D motion analysis, NOT a trained mouse behavior classifier.
The first dashboard accepts exported mouse-center trajectories in CSV format.

## Setup (VS Code terminal)

Windows PowerShell:

    py -3 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -e . -r requirements.txt
    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m streamlit run app.py

Linux/macOS:

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e . -r requirements.txt
    .venv/bin/python -m pytest -q
    .venv/bin/python -m streamlit run app.py

## Data

Download CalMS21 from the official Caltech dataset repository. Verify the exact JSON
schema and landmark order before writing an adapter. Do not assume pose arrays are
already shaped as two mouse-center coordinates. Use a center derived from reliable
body landmarks (e.g., neck/hips), not nose alone. Split by recording/session before
training and never expose held-out expert labels to the monitoring agent.

For a quick first run, create a CSV with columns x_a,y_a,x_b,y_b, one frame per row.
Missing points should be blank/NaN. Pixel units are intentional; no physical scale
is claimed.

## Dashboard

Tick "Use built-in synthetic demo data" to try it without uploading anything.
The demo file (`example_pose_demo.csv`) is made-up test data, not CalMS21.
Tick "Inject fake tracking failures" in the sidebar to see the tracking warnings.

## Team workflow

See `CONTRIBUTING.md` (branches, pull requests, daily routine) and
`INTERFACES.md` (data formats passed between parts of the project).
Progress is tracked in `IMPLEMENTATION_STATUS.md`.

## ScopeScore adaptations

- Short-gap interpolation, no long-gap interpolation
- Segment-aware smoothing
- Frame-to-frame velocity and acceleration
- Relative mouse distance and approach/separation speed
- Movement bout detection with configurable thresholds
- Tracking validity flags
- Deterministic duration-gated alert extraction

## Next stages

1. Inspect CalMS21 task1 JSON and build a verified dataset adapter.
2. Implement session-level train/validation/test split.
3. Build basic behavior classifier and measure event precision/recall.
4. Add temporal windows, agent rule parser with explicit human confirmation.
5. Replay held-out sessions and log timestamped alerts.
6. Add Streamlit timeline, optional video clips, and grounded summaries.
7. Compare basic pose features vs ScopeScore-inspired features on held-out sessions.

Note: Savitzky-Golay smoothing in this starter is noncausal. For true streaming,
replace it with a causal filter (e.g., One-Euro) and validate added latency.
Do not transfer ScopeScore's surgical thresholds or claim attack detection from speed alone.

## Attribution

CalMS21: Sun, J. J., et al. "The Multi-Agent Behavior Dataset: Mouse Dyadic Social
Interactions." NeurIPS Datasets and Benchmarks Track, 2021. arXiv:2104.02710.
Dataset record: https://data.caltech.edu/records/s0vdx-0k302 (CC BY 4.0).
