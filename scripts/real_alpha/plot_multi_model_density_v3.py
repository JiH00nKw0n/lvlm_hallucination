#!/usr/bin/env python3
"""V3: 2x2 per-model density plot of matched decoder cosine, r >= 0.2 only.

Same data pipeline as V1 (``plot_multi_model_density.py``): for each
model's two-SAE, run alive-restricted Hungarian matching on the
train-time cross-correlation matrix, collect (r, cos) for every matched
pair, then in each subplot overlay KDEs of cos for r-bins of width 0.2.

Difference vs V1: the lowest bin [0.0, 0.2) is dropped to focus on
pairs that co-activate above the weakly-matched tail.

Usage:
    python scripts/real_alpha/plot_multi_model_density_v3.py \\
        --out outputs/multi_model_density_v3.svg
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
    {"name": "SigLIP2 Base",  "run_dir": "outputs/siglip2_base/two_sae/final"},
]

# Drop [0.0, 0.2) — start bins at 0.2.
BIN_EDGES = [0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = [
    f"$r \\in [{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})$" for i in range(len(BIN_EDGES) - 1)
]
# 4 bins -> reuse the warm-to-cool section of the V1 palette (skip red).
BIN_COLORS = ["#d96627", "#dfb246", "#389076", "#206987"]
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
    parser.add_argument("--out", type=str, default="outputs/multi_model_density_v3.svg")
    parser.add_argument("--models", type=str, default="",
                        help="JSON file with [{name, run_dir}, ...]. Overrides built-in MODELS.")
    args = parser.parse_args()

    models = json.load(open(args.models)) if args.models else MODELS
    assert len(models) == 4, f"this plot is 2x2, got {len(models)} models"

    fig, axes = plt.subplots(2, 2, figsize=(3.6, 2.6), sharex=True)
    axes_flat = axes.flatten()
    for ax, model in zip(axes_flat, models):
        C_m, cos_m, n_ai, n_at = load_model_data(model["run_dir"])
        total = len(C_m)
        kept = int((C_m >= BIN_EDGES[0]).sum())
        print(f"{model['name']}: alive_i={n_ai}, alive_t={n_at}, "
              f"matched={total}, kept(r>={BIN_EDGES[0]})={kept}")
        plot_density_subplot(ax, C_m, cos_m, model["name"])

    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    axes[0, 0].set_ylabel("density", fontsize=8, labelpad=2)
    axes[1, 0].set_ylabel("density", fontsize=8, labelpad=2)

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
    plt.subplots_adjust(left=0.11, right=0.99, bottom=0.16, top=0.92,
                        wspace=0.22, hspace=0.45)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print(f"saved {out_path}")
    suffix = out_path.suffix.lower()
    for ext in (".pdf", ".png", ".svg"):
        if ext == suffix:
            continue
        alt = out_path.with_suffix(ext)
        fig.savefig(alt, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.02)
        print(f"saved {alt}")
    plt.close()


if __name__ == "__main__":
    main()
