"""Steering image-retrieval mAP bar chart with capped error bars.

Compares 3 methods (Ours, Iso-Energy Alignment, Group-Sparse) at α=1.0,
with the pre-steering (α=0) mAP shown as a horizontal dashed reference
line. Reads either a single-seed summary.json (map_img is a float) or
an aggregated mean+std summary (map_img is {"mean", "std", "n"}); when
N>1, draws standard capped error bars (capsize=4).
"""
from __future__ import annotations

import argparse
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

VARIANTS = [
    ("ours",         "Post-hoc\nAlignment\n(Ours)", BLUE_SLATE,    1.0),
    ("iso_align",    "Iso-Energy\nAlignment",       CARROT_ORANGE, 1.0),
    ("group_sparse", "Group-\nSparse",              SEAWEED,       1.0),
]


def _val_std(cell) -> tuple[float, float]:
    """Accept float (single seed → std=0) or {mean, std, n} (aggregated)."""
    if isinstance(cell, dict):
        return float(cell["mean"]), float(cell.get("std", 0.0))
    return float(cell), 0.0


def _save(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    print(f"wrote {out_path}, .png, .svg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/real_exp_cc3m/cross_modal_steering",
                    help="cross_modal_steering directory (single seed or aggregated mean).")
    ap.add_argument("--out", default=None,
                    help="Output PDF path (default: <root>/steering_map_bar.pdf).")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "steering_map_bar.pdf"

    bases, base_stds = [], []
    vals, val_stds = [], []
    n_seeds_seen = 1
    for v, _, _, a in VARIANTS:
        d = json.load(open(root / v / "summary.json"))
        n_seeds_seen = max(n_seeds_seen, int(d.get("n_seeds", 1)))
        bm, bs = _val_std(d["per_alpha"]["0.0"]["map_img"])
        vm, vs = _val_std(d["per_alpha"][f"{a}"]["map_img"])
        bases.append(bm); base_stds.append(bs)
        vals.append(vm); val_stds.append(vs)
    base_map = float(np.mean(bases))

    fig, ax = plt.subplots(figsize=(5.5 / 2.2, 1.015 * 1.7))
    x = np.arange(len(VARIANTS))
    colors = [c for _, _, c, _ in VARIANTS]
    labels = [lab for _, lab, _, _ in VARIANTS]

    yerr = val_stds if n_seeds_seen > 1 else None
    bars = ax.bar(x, vals, color=colors, width=0.7, edgecolor="white",
                  linewidth=0.5, zorder=3,
                  yerr=yerr, capsize=4 if yerr else 0,
                  error_kw=dict(ecolor="#222", elinewidth=0.9, capthick=0.9))
    for b, v, s in zip(bars, vals, val_stds):
        offset = (s if yerr else 0) + 0.005
        ax.text(b.get_x() + b.get_width() / 2, v + offset,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.5, color="#222")

    ax.axhline(base_map, color=STRAWBERRY_RED, ls="--", lw=1.1, zorder=10)
    ax.text(len(VARIANTS) - 0.5, base_map + 0.005,
            "Base", ha="right", va="bottom", fontsize=6.5,
            color=STRAWBERRY_RED, zorder=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("mAP", fontsize=8, labelpad=2)
    upper = max(v + s for v, s in zip(vals, val_stds)) if yerr else max(vals)
    ax.set_ylim(0, upper * 1.18)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    ax.tick_params(axis="x", labelsize=7, pad=2)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if n_seeds_seen > 1:
        ax.set_title(f"N={n_seeds_seen} seeds (mean ± std)", fontsize=7, pad=4)

    plt.subplots_adjust(top=0.92 if n_seeds_seen > 1 else 0.95,
                         bottom=0.30, left=0.22, right=0.98)
    _save(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
