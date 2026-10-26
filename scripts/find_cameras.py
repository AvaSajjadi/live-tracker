"""Find the cameras on this computer and say which one is the Nikon (new file).

Run: python scripts/find_cameras.py
It lists what Windows sees, then opens camera numbers 0 to 5 and reads a few pictures from each.
A dark picture (brightness under 20) that never changes is usually an infrared (face login) camera, not a real one.
Nothing is saved to disk.
"""

from __future__ import annotations

import subprocess
import sys

NUMBERS = range(6)
FRAMES = 15


def windows_devices():
    """(class, name) of camera-like devices, or [] when not on Windows."""
    if sys.platform != "win32":
        return []
    command = ("Get-PnpDevice -PresentOnly | Where-Object { $_.Class -in 'Camera','Image','WPD' -or "
               "$_.FriendlyName -match 'Nikon|D7500' } | ForEach-Object { $_.Class + '|' + $_.FriendlyName }")
    done = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
    return [tuple(line.strip().split("|", 1)) for line in done.stdout.splitlines() if "|" in line]


def probe(number):
    """(size, mean brightness, brightness change, None) or (None, None, the reason)."""
    import cv2
    capture = cv2.VideoCapture(number)
    try:
        if not capture.isOpened():
            return None, None, None, "does not open"
        levels, size = [], None
        for _ in range(FRAMES):
            ok, image = capture.read()
            if ok and image is not None:
                size = f"{image.shape[1]}x{image.shape[0]}"
                levels.append(float(image.mean()))
        if not levels:
            return None, None, None, "opens but gives no picture"
        return size, sum(levels) / len(levels), max(levels) - min(levels), None
    finally:
        capture.release()


def main():
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("OpenCV is not installed: pip install opencv-python-headless")
        return 1
    devices = windows_devices()
    print("Windows sees:")
    for kind, name in devices:
        print(f"  [{kind}] {name}")
    nikon_file_only = any("D7500" in n or "Nikon" in n for k, n in devices if k in ("WPD", "Image")) and \
        not any("Nikon" in n or "D7500" in n for k, n in devices if k == "Camera")
    print("\nOpenCV camera numbers:")
    for number in NUMBERS:
        size, mean, change, why = probe(number)
        if why:
            print(f"  {number}: {why}")
        else:
            note = "  <- flat picture, probably an infrared camera" if mean < 20 and change < 1.0 else ""
            print(f"  {number}: works, {size}, brightness {mean:.0f}, change {change:.1f}{note}")
    print()
    if any("Nikon" in n or "D7500" in n for k, n in devices if k == "Camera"):
        print("The Nikon shows as a webcam. Use its number in the camera box.")
    elif nikon_file_only:
        print("The Nikon is seen only as a file device (photos), NOT as a webcam.")
        print("Check: USB cable in the camera's USB port, camera ON, Live View/video mode, Nikon Webcam Utility")
        print("installed and the camera on its supported list. Otherwise use HDMI + a capture card.")
    else:
        print("No Nikon found. Check the cable and that the camera is ON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
