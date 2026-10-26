# Run the dashboard on another computer

Models, data and videos are **not in Git**. You make or download them on your own computer. These steps are for Windows (PowerShell).

## 1. Get the code

```
git clone <the repository address>
cd live-tracker
git checkout main
```

## 2. Make the project's Python

```
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Get the data (CalMS21 train file)

Download `calms21_task1_train.json` from https://data.caltech.edu/records/s0vdx-0k302 and put it in `data/raw/calms21/`.
Do not download or use the test file.

## 4. Make the features and train the models

Use the **same Python** (`.venv`) that will run the dashboard. If you train with another Python, the dashboard fails with
`'SimpleImputer' object has no attribute '_fill_dtype'`. That means the model was saved by another scikit-learn version.
The fix is to train again with the `.venv` Python:

```
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m live_tracker.classify train --name model_b --features <features file>
```

Run it again with `--name model_a` and the basic features file. The exact feature files and steps are in
`docs/motion_engine_results.md`. The model files go to `models/` (not in Git).

## 5. Get the demo video (one command, checked)

```
.\.venv\Scripts\python.exe scripts\fetch_demo_video.py
```

It downloads only one video (about 200 MB) from the public dataset, not the 28 GB archive. It checks the size and
checksum. It refuses test videos. Then make its keypoints file:

```
.\.venv\Scripts\python.exe scripts\prepare_demo_session.py --session task1/train/mouse064_task1_annotator1 --out data/raw/calms21_video/mouse064_task1_annotator1_poses.parquet
```

The video format (`mp4v`) does not play in browsers. Prepare and test a browser-friendly copy before the offline demo;
the page offers conversion when an available converter is detected (Windows built-in support or optional
`imageio-ffmpeg`). See `docs/demo_recording.md` and the artifact inventory in `docs/demo_readiness.md`.

## 6. Start the dashboard

```
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8766 --server.address localhost
```

Open http://localhost:8766 and follow the steps in `docs/demo_recording.md`.

## If something fails

| Message | What to do |
|---|---|
| `No module named live_tracker` | Set `$env:PYTHONPATH = "src"` in the same window |
| `could not be loaded ... another version` | Train the models again with the `.venv` Python (step 4) |
| The video does not play | Use the page's "Prepare the video" button (step 5) |
| `.venv` not found | You are in the wrong folder. `cd` into the project folder |
