"""Steering image-retrieval mAP bar chart.

Compares 3 methods (Ours, Iso-Energy Alignment, Group-Sparse) at matched
effective alpha (W_norm-corrected), with the pre-steering mAP shown as a
horizontal dashed reference line. Typography mirrors plot_lambda_sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CARROT_ORANGE, SEAWEED, BLUE_SLATE, STRAWBERRY_RED  # noqa: E402

matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

ROOT = Path("outputs/real_exp_cc3m/cross_modal_steering")
OUT = ROOT / "steering_map_bar.pdf"

VARIANTS = [
    ("ours",         "Post-hoc\nAlignment\n(Ours)", BLUE_SLATE,    1.0),
    ("iso_align",    "Iso-Energy\nAlignment",       CARROT_ORANGE, 1.0),
    ("group_sparse", "Group-\nSparse",              SEAWEED,       1.0),
]


def _save(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    print(f"wrote {out_path}, .png, .svg")


def main() -> None:
    bases, vals = [], []
    for v, _, _, a in VARIANTS:
        d = json.load(open(ROOT / v / "summary.json"))
        bases.append(d["per_alpha"]["0.0"]["map_img"])
        vals.append(d["per_alpha"][f"{a}"]["map_img"])
    base_map = float(np.mean(bases))

    fig, ax = plt.subplots(figsize=(5.5 / 2.2, 1.015 * 1.7))
    x = np.arange(len(VARIANTS))
    colors = [c for _, _, c, _ in VARIANTS]
    labels = [lab for _, lab, _, _ in VARIANTS]

    bars = ax.bar(x, vals, color=colors, width=0.7, edgecolor="white",
                  linewidth=0.5, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.5, color="#222")

    ax.axhline(base_map, color=STRAWBERRY_RED, ls="--", lw=1.1, zorder=10)
    ax.text(len(VARIANTS) - 0.5, base_map + 0.005,
            "Base", ha="right", va="bottom", fontsize=6.5,
            color=STRAWBERRY_RED, zorder=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("mAP", fontsize=8, labelpad=2)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    ax.tick_params(axis="x", labelsize=7, pad=2)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(top=0.95, bottom=0.30, left=0.22, right=0.98)
    _save(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
