"""
generate_report_figures.py — produces the two figures needed for the Task 3
report section.

Output
------
task3/report_figures/fig_detection_example.jpg   — annotated detection image
task3/report_figures/fig_results_table.png        — bar chart + results table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = Path(__file__).parent / "report_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Results data (from main.py run) ──────────────────────────────────────────
BAGS = [
    "left_1", "left_2", "right_1", "right_2",
    "multi_1", "multi_2", "multi_3",
]
DET_RATE  = [20.5, 100.0, 70.8, 100.0, 100.0, 96.4, 73.6]
MEAN_DIST = [0.624, 2.125, 1.603, 3.185, 3.203, 1.837, 1.551]
STD_DIST  = [0.381, 0.687, 0.435, 2.713, 1.457, 0.797, 0.496]


# ── Figure 1: annotated detection example ────────────────────────────────────
def make_detection_figure():
    # Use the multiple-cylinder bag — most visually informative
    candidates = sorted(
        (Path(__file__).parent / "output" / "traffic_signs_multiple_2").glob("*.jpg"))
    if not candidates:
        print("No detection images found — skipping Figure 1.")
        return

    img = cv2.imread(str(candidates[0]))
    if img is None:
        print("Could not read detection image — skipping Figure 1.")
        return

    out = OUT_DIR / "fig_detection_example.jpg"
    cv2.imwrite(str(out), img)
    print(f"Saved: {out}")


# ── Figure 2: bar chart only ──────────────────────────────────────────────────
def make_results_figure():
    fig, ax = plt.subplots(figsize=(8, 4))

    x    = np.arange(len(BAGS))
    bars = ax.bar(x, DET_RATE, color='steelblue', width=0.55, zorder=3)
    ax.axhline(y=np.mean(DET_RATE), color='tomato', linestyle='--',
               linewidth=1.2, label=f'Mean = {np.mean(DET_RATE):.1f}%')
    ax.set_xticks(x)
    ax.set_xticklabels(BAGS, fontsize=10)
    ax.set_ylabel('Detection Rate (%)', fontsize=11)
    ax.set_title('Cylinder Detection Rate per Bag', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    ax.legend(fontsize=10)

    for bar, val in zip(bars, DET_RATE):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig_detection_rate.png"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    make_detection_figure()
    make_results_figure()
    print("Done — figures in", OUT_DIR.resolve())
