"""Make the poster figures (PNG, transparent background) from measured numbers.

Run: python scripts/poster_figures.py [out_dir]
The speed of the fast model is added only when it is given with --fast-ms (measured with the trained mouse model).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, BLUE, TEAL, RED, GREY = "#082D60", "#2F6DB5", "#3E8E86", "#B5483A", "#8A8A80"
plt.rcParams.update({"font.family": "Arial", "font.size": 16, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def pipeline(out):
    steps = ["Camera", "Straighten\nthe screen", "Find body\npoints", "Score\nbehavior", "Apply the\nrule", "Alert +\nreport"]
    fig, ax = plt.subplots(figsize=(12.9, 2.4))
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    w, gap = 1.8, 0.32
    for i, s in enumerate(steps):
        x = i * (w + gap) + 0.1
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 1.4, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#EAF2F5", ec=BLUE, lw=2))
        ax.text(x + w / 2, 1.2, s, ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(x + 0.18, 1.78, str(i + 1), ha="center", va="center", fontsize=12, color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.25", fc=BLUE, ec="none"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 1.2), xytext=(x + w + 0.03, 1.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    fig.savefig(out / "1_pipeline.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)


def speed(out, fast_ms):
    names, vals, cols = ["Research tool\n(MARS)", "Needed for\nlive video"], [850, 33], [RED, GREY]
    if fast_ms:
        names.insert(1, "Our fast\nmodel")
        vals.insert(1, fast_ms)
        cols.insert(1, TEAL)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(names, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, f"{v:g} ms", ha="center", fontweight="bold")
    ax.set_ylabel("Time for one picture (ms), lower is faster")
    ax.set_ylim(0, 1000)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Finding body points: speed" + ("" if fast_ms else " (before the fast model)"), fontweight="bold", loc="left")
    fig.savefig(out / "2_speed.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)


def quality(out):
    rows = [("Frames labeled right", 84.3, 78.7, 88.5), ("Alerts that were right", 82.5, 18.2, 92.5), ("Attack bouts found", 64.3, 21.1, 87.2)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (name, v, lo, hi) in enumerate(rows[::-1]):
        ax.barh(i, v, color=BLUE, height=0.55)
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], color=INK, capsize=5, lw=1.5)
        ax.text(2, i, f"{v:.0f}%", va="center", color="white", fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent (line = 95% range over 14 videos)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("How good are the alerts? (validation videos)", fontweight="bold", loc="left")
    fig.savefig(out / "3_alert_quality.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)


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
