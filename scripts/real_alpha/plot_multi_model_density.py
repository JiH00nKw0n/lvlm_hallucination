#!/usr/bin/env python3
"""2x2 density plots of matched decoder cosine per r-bin, across 4 VLMs.

Companion to `plot_multi_model_boxplot.py` (does NOT replace it). Same
data pipeline: for each model's two-SAE, compute alive-restricted
Hungarian matching on the train-time cross-correlation matrix, then for
each matched pair collect (r, cos). Within each subplot, overlay KDEs
of decoder cosine for 5 bins of the co-activation correlation r
(width 0.2) with 50% transparency.

Usage:
    python scripts/real_alpha/plot_multi_model_density.py \\
        --out outputs/multi_model_density.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde

MODELS = [
    {"name": "CLIP ViT-B/32", "run_dir": "outputs/real_alpha_followup_1/two_sae/final"},
    {"name": "MetaCLIP B/32", "run_dir": "outputs/metaclip_b32/two_sae/final"},
    {"name": "OpenCLIP B/32", "run_dir": "outputs/datacomp_b32/two_sae/final"},
    {"name": "SigLIP2 Base", "run_dir": "outputs/siglip2_base/two_sae/final"},
]

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = [
    f"$r \\in [{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})$" for i in range(len(BIN_EDGES) - 1)
]
# Original 5-model palette reordered 빨→주→노→초→파 as r increases.
BIN_COLORS = ["#df3a3d", "#d96627", "#dfb246", "#389076", "#206987"]
ALIVE_THR = 0.001
ALPHA = 0.2
KDE_GRID = np.linspace(-0.4, 1.0, 300)


def load_model_data(run_dir: str):
    """Return (C_matched, cos_matched) for alive-restricted Hungarian matching."""
    rd = Path(run_dir)
    C = np.load(rd / "diagnostic_B_C_train.npy")
    sd = load_file(str(rd / "model.safetensors"))
    W_i = sd["image_sae.W_dec"].float().numpy()
    W_t = sd["text_sae.W_dec"].float().numpy()
    W_i_n = W_i / (np.linalg.norm(W_i, axis=1, keepdims=True) + 1e-12)
    W_t_n = W_t / (np.linalg.norm(W_t, axis=1, keepdims=True) + 1e-12)

    rates_path = rd / "diagnostic_B_firing_rates.npz"
    if rates_path.exists():
        rates = np.load(rates_path)
        alive_i = np.where(rates["rate_i"] > ALIVE_THR)[0]
        alive_t = np.where(rates["rate_t"] > ALIVE_THR)[0]
    else:
        # Fallback: nonzero-variance proxy
        var_i = np.var(C, axis=1)
        var_t = np.var(C, axis=0)
        alive_i = np.where(var_i > 1e-8)[0]
        alive_t = np.where(var_t > 1e-8)[0]

    C_sub = C[np.ix_(alive_i, alive_t)]
    row_ind, col_ind = linear_sum_assignment(-C_sub)
    C_matched = C_sub[row_ind, col_ind]
    cos_matched = (W_i_n[alive_i[row_ind]] * W_t_n[alive_t[col_ind]]).sum(axis=1)
    return C_matched, cos_matched, len(alive_i), len(alive_t)


def plot_density_subplot(ax, C_matched, cos_matched, title: str):
    for b in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        mask = (C_matched >= lo) & (C_matched < hi)
        n = int(mask.sum())
        if n < 5:
            continue
        vals = cos_matched[mask]
        if np.std(vals) < 1e-8:
            continue
        kde = gaussian_kde(vals)
        density = kde(KDE_GRID)
        ax.fill_between(KDE_GRID, density, color=BIN_COLORS[b], alpha=ALPHA, linewidth=0)
        ax.plot(KDE_GRID, density, color=BIN_COLORS[b], alpha=min(ALPHA + 0.3, 1.0), lw=0.6)

    ax.set_title(title, fontsize=8, fontweight="bold", pad=2)
    ax.set_xlim(-0.3, 1.0)
    ax.set_xticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.axvline(0.0, color="gray", lw=0.5, ls="--", zorder=0)
    ax.set_xlabel(r"$\cos(\cdot)$", fontsize=8, labelpad=1)
    ax.tick_params(axis="x", labelsize=7, pad=1)
    ax.tick_params(axis="y", labelsize=7, pad=1)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/multi_model_density.pdf")
    parser.add_argument("--models", type=str, default="",
                        help="JSON file with [{name, run_dir}, ...]. Overrides built-in MODELS.")
    args = parser.parse_args()

    models = json.load(open(args.models)) if args.models else MODELS
    assert len(models) == 4, f"this plot is 2x2, got {len(models)} models"

    # 2×2 layout, tall enough that titles + xlabels + legend don't collide.
    fig, axes = plt.subplots(2, 2, figsize=(3.6, 2.6), sharex=True)
    axes_flat = axes.flatten()
    for ax, model in zip(axes_flat, models):
        C_m, cos_m, n_ai, n_at = load_model_data(model["run_dir"])
        print(f"{model['name']}: alive_i={n_ai}, alive_t={n_at}, matched={len(C_m)}")
        plot_density_subplot(ax, C_m, cos_m, model["name"])

    # Only bottom row shows xlabel; left column shows ylabel.
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    axes[0, 0].set_ylabel("density", fontsize=8, labelpad=2)
    axes[1, 0].set_ylabel("density", fontsize=8, labelpad=2)

    # Inline legend at 8pt to match NeurIPS footnotesize/caption text.
    from matplotlib.colors import to_rgba
    legend_handles = [
        Patch(
            facecolor=to_rgba(BIN_COLORS[b], alpha=ALPHA),
            edgecolor=to_rgba(BIN_COLORS[b], alpha=1.0),
            linewidth=1.0, label=BIN_LABELS[b],
        )
        for b in range(len(BIN_EDGES) - 1)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(BIN_LABELS),
        fontsize=8, frameon=False,
        handlelength=1.0, handletextpad=0.3, columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.06),
    )
    # Leave enough room under the bottom subplots for xlabel + legend row.
    plt.subplots_adjust(left=0.11, right=0.99, bottom=0.16, top=0.92,
                        wspace=0.22, hspace=0.45)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print(f"saved {out_path}")
    if str(out_path).endswith(".pdf"):
        for ext in (".png", ".svg"):
            alt = str(out_path).replace(".pdf", ext)
            fig.savefig(alt, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.02)
            print(f"saved {alt}")
    plt.close()


if __name__ == "__main__":
    main()
