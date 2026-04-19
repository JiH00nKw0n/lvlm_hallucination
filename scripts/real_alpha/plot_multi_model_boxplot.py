#!/usr/bin/env python3
"""3-model Diagnostic B comparison boxplot.

For each model's two-SAE, loads C matrix + decoder weights,
does alive-restricted Hungarian matching, bins by C, plots decoder cosine.

Usage:
    python scripts/real_alpha/plot_multi_model_boxplot.py \
        --out outputs/multi_model_boxplot.pdf
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
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment

MODELS = [
    {"name": "CLIP ViT-B/32",   "run_dir": "outputs/real_alpha_followup_1/two_sae/final", "color": "#df3a3d"},
    {"name": "MetaCLIP B/32",   "run_dir": "outputs/metaclip_b32/two_sae/final",          "color": "#d96627"},
    {"name": "DataComp B/32",   "run_dir": "outputs/datacomp_b32/two_sae/final",           "color": "#dfb246"},
    {"name": "MobileCLIP2-B",   "run_dir": "outputs/mobileclip2b/two_sae/final",           "color": "#389076"},
    {"name": "SigLIP2 Base",    "run_dir": "outputs/siglip2_base/two_sae/final",            "color": "#206987"},
]

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ALIVE_THR = 0.001


def load_model_data(run_dir: str):
    """Return (C_matched, cos_matched) for alive-restricted Hungarian."""
    rd = Path(run_dir)
    C = np.load(rd / "diagnostic_B_C_train.npy")
    sd = load_file(str(rd / "model.safetensors"))
    W_i = sd["image_sae.W_dec"].float().numpy()
    W_t = sd["text_sae.W_dec"].float().numpy()
    W_i_n = W_i / (np.linalg.norm(W_i, axis=1, keepdims=True) + 1e-12)
    W_t_n = W_t / (np.linalg.norm(W_t, axis=1, keepdims=True) + 1e-12)

    # Use firing rates if available, else use all latents
    rates_path = rd / "diagnostic_B_firing_rates.npz"
    if rates_path.exists():
        rates = np.load(rates_path)
        alive_i = np.where(rates["rate_i"] > ALIVE_THR)[0]
        alive_t = np.where(rates["rate_t"] > ALIVE_THR)[0]
    else:
        # No firing rates — use C diagonal as proxy (nonzero variance)
        var_i = np.var(C, axis=1)
        var_t = np.var(C, axis=0)
        alive_i = np.where(var_i > 1e-8)[0]
        alive_t = np.where(var_t > 1e-8)[0]

    C_sub = C[np.ix_(alive_i, alive_t)]
    row_ind, col_ind = linear_sum_assignment(-C_sub)
    C_matched = C_sub[row_ind, col_ind]
    cos_matched = (W_i_n[alive_i[row_ind]] * W_t_n[alive_t[col_ind]]).sum(axis=1)
    return C_matched, cos_matched, len(alive_i), len(alive_t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/multi_model_boxplot.pdf")
    parser.add_argument("--models", type=str, default="",
                        help="JSON file with [{name, run_dir, color}, ...]. Overrides built-in MODELS.")
    args = parser.parse_args()

    models = json.load(open(args.models)) if args.models else MODELS
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5, 1.8), sharey=True)

    for ax, model in zip(axes, models):
        C_m, cos_m, n_ai, n_at = load_model_data(model["run_dir"])
        print(f"{model['name']}: alive_i={n_ai}, alive_t={n_at}, matched={len(C_m)}")

        box_data = []
        box_labels = []
        for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            mask = (C_m >= lo) & (C_m < hi)
            sub = cos_m[mask]
            if len(sub) < 3:
                continue
            box_data.append(sub)
            box_labels.append(f"[{lo:.1f},\n{hi:.1f})")

        if not box_data:
            ax.set_title(model["name"], fontsize=8, fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        bp = ax.boxplot(
            box_data, widths=0.6, patch_artist=True,
            showfliers=False, medianprops=dict(color="white", lw=1.2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(model["color"])
            patch.set_alpha(0.7)
            patch.set_edgecolor(model["color"])

        # Sample counts omitted — reported in caption

        ax.set_xticklabels(box_labels, fontsize=5.5)
        ax.set_title(model["name"], fontsize=7, fontweight="bold", pad=2)
        ax.set_xlabel(r"$r$", fontsize=8, labelpad=1)
        ax.axhline(0, color="gray", lw=0.5, ls="--", zorder=0)
        ax.tick_params(axis="y", labelsize=6.5, pad=1)
        ax.grid(axis="y", alpha=0.15, linewidth=0.4)

    axes[0].set_ylabel("")  # no ylabel, use text above axis
    axes[0].text(-0.02, 1.02, r"$\cos(\cdot)$", fontsize=8, transform=axes[0].transAxes,
                 ha="right", va="bottom")
    axes[0].set_ylim(-0.35, 1.0)
    axes[0].set_yticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.85, wspace=0.08)
    # no suptitle

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    print(f"saved {args.out}")
    if args.out.endswith(".pdf"):
        png = args.out.replace(".pdf", ".png")
        plt.savefig(png, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
        print(f"saved {png}")
    plt.close()


if __name__ == "__main__":
    main()
