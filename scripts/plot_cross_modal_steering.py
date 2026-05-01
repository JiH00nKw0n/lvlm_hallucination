"""Cross-modal steering figure (lambda_sweep typography).

Reads outputs/real_exp_cc3m/cross_modal_steering/{variant}/summary.json
Produces a 4-panel α-sweep:
  Δ_image (image anchor) | Δ_text (text anchor) | rank_img | rank_txt

The split between image- and text-anchor metrics is the headline:
  - single-dictionary baselines (shared/IA/GS) carry a *bisector* decoder column
    so they help text-anchor steering 'for free' but underperform on the image
    side where intervention actually happens.
  - Ours has a clean image-side direction so it wins the image-anchor metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CARROT_ORANGE, SEAWEED, BLUE_SLATE  # noqa: E402

matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

ROOT = Path("outputs/real_exp_cc3m/cross_modal_steering")
OUT = ROOT / "steering_curves.pdf"

# Order: ours last so it overplots
VARIANTS = [
    ("separated", "Modality-Specific (no align)", "#888888", "--"),
    ("iso_align", "Iso-Energy Alignment", CARROT_ORANGE, "-"),
    ("group_sparse", "Group-Sparse", SEAWEED, "-"),
    ("ours", "Post-hoc Alignment (Ours)", BLUE_SLATE, "-"),
]

PER_PANEL_W = (5.5 / 4) * 1.0
HEIGHT = 1.015 * 1.3


def _save(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    print(f"wrote {out_path}, .png, .svg")


def _draw_legend(fig: plt.Figure, handles, labels) -> None:
    leg = fig.legend(
        handles=handles, labels=labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles), frameon=False, fontsize=8,
        handlelength=1.4, columnspacing=1.2, handletextpad=0.4,
    )
    for ln in leg.get_lines():
        ln.set_linewidth(ln.get_linewidth() + 2.0)


def main() -> None:
    summaries = {}
    for v, *_ in VARIANTS:
        p = ROOT / v / "summary.json"
        if p.exists():
            summaries[v] = json.load(open(p))
    if not summaries:
        raise SystemExit(f"no summaries found under {ROOT}")
    any_s = next(iter(summaries.values()))
    alphas = list(any_s["alphas"])

    fig, axes = plt.subplots(1, 4, figsize=(PER_PANEL_W * 4, HEIGHT))
    handles, labels = [], []

    metrics = [
        ("map_img",         "M1: mAP (image)"),
        ("p10_img",         "M2: P@10 (image)"),
        ("median_rank_img", "M3: Median rank (image)"),
        ("preserve_mean",   "M4: Preserve"),
    ]

    for ax, (key, title) in zip(axes, metrics):
        for v, label, color, ls in VARIANTS:
            s = summaries.get(v)
            if s is None:
                continue
            xs, ys = [], []
            for a in alphas:
                row = s["per_alpha"].get(f"{a}")
                if row is None or row.get(key) is None:
                    continue
                xs.append(a)
                ys.append(row[key])
            if not xs:
                continue
            line, = ax.plot(xs, ys, color=color, lw=1.4, linestyle=ls,
                             marker="o", markersize=3)
            if title.startswith("M1"):
                handles.append(line)
                labels.append(label)
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlabel(r"$\alpha$ (in $\sigma_c$ units)", fontsize=9, labelpad=1)
        ax.tick_params(axis="both", labelsize=8, pad=1)
        ax.grid(alpha=0.15, linewidth=0.4)

    # rank panel: log y (only "Median rank")
    axes[2].set_yscale("log")

    _draw_legend(fig, handles, labels)
    plt.subplots_adjust(top=0.66, bottom=0.22, left=0.05, right=0.99, wspace=0.45)
    _save(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
