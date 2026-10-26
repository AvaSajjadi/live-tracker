"""Does the Iris Xe speed hold for minutes? Re-exports the 384x640 OpenVINO model, then predicts for N seconds on the GPU."""
import sys, time, statistics, cv2
from ultralytics import YOLO
video, start, seconds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
frames = [cap.read()[1] for _ in range(120)]
m = YOLO("yolo26n-pose.pt")
path = m.export(format="openvino", imgsz=(384, 640), verbose=False)
ov = YOLO(path, task="pose")
for f in frames[:10]: ov.predict(f, imgsz=(384, 640), device="intel:gpu", verbose=False)
t0 = time.time(); i = 0; window = []; wall = []
while time.time() - t0 < seconds:
    f = frames[i % len(frames)]; i += 1
    t = time.perf_counter(); ov.predict(f, imgsz=(384, 640), device="intel:gpu", verbose=False); window.append((time.perf_counter() - t) * 1000)
    if len(window) == 600:
        wall.append((time.time() - t0, statistics.median(window), sorted(window)[int(len(window) * 0.99)])); window = []
for t, med, p99 in wall: print(f"after {t:5.0f} s: median {med:5.1f} ms, 99% of frames under {p99:5.1f} ms", flush=True)
print("frames:", i, "average fps:", round(i / (time.time() - t0), 1), flush=True)
print("DONE", flush=True)
