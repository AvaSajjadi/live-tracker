"""Make the poster figures (PNG, transparent background) from measured numbers.

Run: python scripts/poster_figures.py [out_dir]
The speed of the fast model is added only when it is given with --fast-ms (measured with the trained mouse model).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# poster colors
NAVY, OLIVE, SAND, CARD, BLUE, RED, GREY = "#082D60", "#948A54", "#DDD9C3", "#EEECE1", "#2F6DB5", "#B5483A", "#A9A79A"
plt.rcParams.update({"font.family": "Arial", "text.color": NAVY})


def save(fig, out, name):
    fig.savefig(out / name, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def pipeline(out):
    # the real chain of the project (README "How the parts fit together", docs/camera.md section 1)
    steps = [("Video", "camera or file"), ("Body points", "7 per mouse"), ("Quality check", "bad frames flagged"),
             ("Motion", "speed, distance"), ("Behavior score", "Model B"), ("Researcher rule", "approved by a person"),
             ("Monitor", "alert or ask"), ("Log + report", "every event saved")]
    n, w, h, notch = len(steps), 2.3, 1.6, 0.35
    fig, ax = plt.subplots(figsize=(16, 2.6))
    ax.set_xlim(0, n * (w - notch / 2) + notch)
    ax.set_ylim(-0.1, h + 0.1)
    ax.axis("off")
    for i, (title, sub) in enumerate(steps):
        x = i * (w - notch / 2)
        pts = [(x, 0), (x + w - notch, 0), (x + w, h / 2), (x + w - notch, h), (x, h)] + ([(x + notch, h / 2)] if i else [])
        ax.add_patch(Polygon(pts, closed=True, fc=NAVY if i % 2 == 0 else OLIVE, ec="none"))
        cx = x + (w + (notch if i else 0)) / 2 - notch / 2 - 0.05
        ax.text(cx, h * 0.62, title, ha="center", va="center", fontsize=14.5, fontweight="bold", color="white")
        ax.text(cx, h * 0.28, sub, ha="center", va="center", fontsize=10, color="white")
    save(fig, out, "1_pipeline.png")


def speed(out, fast_ms):
    rows = [("Research tool (MARS)", 850, RED), ("Needed for live video", 33, GREY)]
    if fast_ms:
        rows.insert(1, ("Our fast model", fast_ms, BLUE))
    fig, ax = plt.subplots(figsize=(7, 1.2 + 1.1 * len(rows)))
    for i, (name, v, c) in enumerate(rows[::-1]):
        y = i * 1.5
        ax.barh(y, 1000, height=0.5, color=CARD)
        ax.barh(y, max(v, 20), height=0.5, color=c)
        ax.text(0, y + 0.38, name, fontsize=14, fontweight="bold", va="bottom")
        inside = v > 300
        ax.text(max(v, 20) - 15 if inside else max(v, 20) + 15, y, f"{v:g} ms", fontsize=17, fontweight="bold", va="center",
                ha="right" if inside else "left", color="white" if inside else NAVY)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.5, (len(rows) - 1) * 1.5 + 1.1)
    ax.axis("off")
    ax.text(0, (len(rows) - 1) * 1.5 + 1.0, "Time to find the body points in one picture", fontsize=12.5, color=OLIVE, va="bottom")
    save(fig, out, "2_speed.png")


def quality(out):
    rows = [("84%", "of video frames\nlabeled right"), ("82%", "of alerts\nwere right"), ("64%", "of attack bouts\nwere found")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6))
    for ax, (big, small) in zip(axes, rows):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.06", fc=CARD, ec=OLIVE, lw=1.8, mutation_aspect=1))
        ax.text(0.5, 0.62, big, ha="center", va="center", fontsize=46, fontweight="bold", color=NAVY)
        ax.text(0.5, 0.2, small, ha="center", va="center", fontsize=14, color=NAVY)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "3_alert_quality.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="outputs/poster")
    ap.add_argument("--fast-ms", type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pipeline(out)
    speed(out, a.fast_ms)
    quality(out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))
