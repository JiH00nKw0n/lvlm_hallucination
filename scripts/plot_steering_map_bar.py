"""Steering image-retrieval mAP + MRR stacked bar chart with capped error bars.

Two vertically stacked panels (mAP on top, MRR on bottom) sharing one set of
method x-tick labels at the very bottom. Compares 3 methods (Ours, Iso-Energy
Alignment, Group-Sparse) at α=1.0, with the pre-steering (α=0) value shown as
a horizontal dashed reference line in each panel. Reads either a single-seed
summary.json (metrics are floats) or an aggregated mean+std summary (metrics
are {"mean", "std", "n"}); when N>1, draws standard capped error bars.
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

    metrics = [("map_img", "mAP"), ("mrr_img", "MRR")]
    panel_data = []
    n_seeds_seen = 1
    for key, _ in metrics:
        bases, vals, val_stds = [], [], []
        for v, _, _, a in VARIANTS:
            d = json.load(open(root / v / "summary.json"))
            n_seeds_seen = max(n_seeds_seen, int(d.get("n_seeds", 1)))
            bm, _ = _val_std(d["per_alpha"]["0.0"][key])
            vm, vs = _val_std(d["per_alpha"][f"{a}"][key])
            bases.append(bm); vals.append(vm); val_stds.append(vs)
        panel_data.append((float(np.mean(bases)), vals, val_stds))

    x = np.arange(len(VARIANTS))
    colors = [c for _, _, c, _ in VARIANTS]
    labels = [lab for _, lab, _, _ in VARIANTS]
    yerr_on = n_seeds_seen > 1

    fig, axes = plt.subplots(2, 1, figsize=(5.5 / 2.2, 1.015 * 1.7),
                              sharex=True,
                              gridspec_kw=dict(hspace=0.18))
    for ax, (key, ylab), (base_val, vals, val_stds) in zip(axes, metrics, panel_data):
        yerr = val_stds if yerr_on else None
        bars = ax.bar(x, vals, color=colors, width=0.7, edgecolor="white",
                      linewidth=0.5, zorder=3,
                      yerr=yerr, capsize=4 if yerr else 0,
                      error_kw=dict(ecolor="#222", elinewidth=0.9, capthick=0.9))
        upper = max(v + s for v, s in zip(vals, val_stds)) if yerr_on else max(vals)
        offset_unit = upper * 0.02
        for b, v, s in zip(bars, vals, val_stds):
            offset = (s if yerr_on else 0) + offset_unit
            ax.text(b.get_x() + b.get_width() / 2, v + offset,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color="#222")

        ax.axhline(base_val, color=STRAWBERRY_RED, ls="--", lw=1.1, zorder=10)
        ax.text(len(VARIANTS) - 0.5, base_val + offset_unit,
                "Base", ha="right", va="bottom", fontsize=6.5,
                color=STRAWBERRY_RED, zorder=11)

        ax.set_ylabel(ylab, fontsize=8, labelpad=2)
        ax.set_ylim(0, upper * 1.22)
        ax.tick_params(axis="y", labelsize=6, pad=1)
        ax.grid(axis="y", alpha=0.15, linewidth=0.4, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6.5)
    axes[1].tick_params(axis="x", labelsize=7, pad=2)

    if n_seeds_seen > 1:
        axes[0].set_title(f"N={n_seeds_seen} seeds (mean ± std)", fontsize=7, pad=4)

    plt.subplots_adjust(top=0.92 if n_seeds_seen > 1 else 0.96,
                         bottom=0.22, left=0.22, right=0.98)
    _save(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
