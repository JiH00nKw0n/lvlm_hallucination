"""FMS@1 bar plot for 3 variants — Iso-Energy, Group-Sparse, Post-hoc Alignment.
Colors mirror lambda-sweep palette; DejaVu Serif font.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CARROT_ORANGE, SEAWEED, BLUE_SLATE  # noqa: E402

ROOT = Path("outputs/real_exp_cc3m/monosemanticity")
OUT = Path("outputs/real_exp_cc3m/mms_coco_test/fms1_bar.pdf")

VARIANTS = [
    ("iso_align", "Iso-Energy Alignment", CARROT_ORANGE),
    ("group_sparse", "Group-Sparse", SEAWEED),
    # Post-hoc Alignment shares image-side latents with separated; FMS@1 identical.
    ("separated", "Post-hoc Alignment (Ours)", BLUE_SLATE),
]


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    labels, values, colors = [], [], []
    for v, label, color in VARIANTS:
        d = json.load(open(ROOT / v / "fms_summary.json"))
        labels.append(label)
        values.append(d.get("fms_at_1", d.get("FMS_at_1", 0.0)))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    xs = list(range(len(labels)))
    ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.0008, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("FMS@1", fontsize=10)
    lo = min(values) - 0.005
    hi = max(values) + 0.004
    ax.set_ylim(max(0.0, lo), hi)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    print(f"wrote {OUT}, .png, .svg")


if __name__ == "__main__":
    main()
