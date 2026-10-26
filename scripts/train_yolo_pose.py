"""Train a small YOLO pose model on the MARS top-view mouse dataset (new file).

Make the dataset first with scripts/mars_pose_to_yolo.py. This needs the packages `ultralytics` (and a GPU, unless you only test).
On this laptop's CPU a full training would take about two days, so use Google Colab (colab/train_yolo_pose_colab.ipynb).

    python scripts/train_yolo_pose.py --data dataset/dataset.yaml --epochs 50 --device 0
    python scripts/train_yolo_pose.py --data dataset/dataset.yaml --epochs 1 --imgsz 320 --fraction 0.01 --device cpu   # smoke test

The best weights end up in <project>/<name>/weights/best.pt (about 6 MB).
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="dataset.yaml made by mars_pose_to_yolo.py")
    ap.add_argument("--weights", default="yolo26n-pose.pt", help="pretrained start (falls back to yolo11n-pose.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640, help="the 1024x570 pictures are shrunk to this width (square batches)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0", help="0 for the first GPU, cpu, or mps (Apple)")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="mouse_pose")
    ap.add_argument("--fraction", type=float, default=1.0, help="use only this share of the training pictures (for a quick test)")
    ap.add_argument("--patience", type=int, default=15, help="stop if the test score has not improved for this many epochs")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    model = None
    for name in (a.weights, "yolo11n-pose.pt"):
        try:
            model = YOLO(name)
            print("start weights:", name, flush=True)
            break
        except Exception as exc:
            print(f"could not load {name}: {type(exc).__name__}", flush=True)
    if model is None:
        raise SystemExit("no pretrained pose weights could be loaded")
    device = int(a.device) if a.device.isdigit() else a.device
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device, project=a.project, name=a.name,
        fraction=a.fraction, patience=a.patience, workers=a.workers, exist_ok=True,
        fliplr=0.5, flipud=0.0,          # flipud would not swap left and right keypoints in Ultralytics, so it stays off
        degrees=15.0,                    # a mouse can face any way in a top view
        plots=False, verbose=True,
    )
    print("best weights:", f"{a.project}/{a.name}/weights/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
