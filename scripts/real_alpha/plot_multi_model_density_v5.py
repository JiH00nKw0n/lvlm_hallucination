#!/usr/bin/env python3
"""V5: multi-model density plot in cosine *distance* (1 - cos).

Same pipeline / styling as V4 (``plot_multi_model_density_v4.py``), but the
horizontal axis is cosine distance ``d = 1 - cos`` instead of cosine
similarity. Pairs are still selected by r >= R_MIN (correlation on the C
matrix; that stays unchanged). Only the decoder-cosine summary is remapped.

Usage:
    python scripts/real_alpha/plot_multi_model_density_v5.py \\
        --out outputs/multi_model_density_v5.pdf
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
from scipy.stats import gaussian_kde

MODELS = [
    {"name": "CLIP",        "run_dir": "outputs/real_alpha_followup_1/two_sae/final", "color": "#df3a3d"},
    {"name": "MetaCLIP",    "run_dir": "outputs/metaclip_b32/two_sae/final",          "color": "#d96627"},
    {"name": "OpenCLIP",    "run_dir": "outputs/datacomp_b32/two_sae/final",          "color": "#dfb246"},
    {"name": "MobileCLIP",  "run_dir": "outputs/mobileclip2b/two_sae/final",          "color": "#389076"},
    {"name": "SigLIP",      "run_dir": "outputs/siglip2_base/two_sae/final",          "color": "#206987"},
]

R_MIN = 0.4
ALIVE_THR = 0.001
FILL_ALPHA = 0.2
LINE_ALPHA = 0.95
# distance ∈ [0, 2]; near-orthogonal pairs sit ~1.0, anti-aligned ~2.0
KDE_GRID = np.linspace(0.0, 1.4, 400)


def load_model_data(run_dir: str):
    """Return (C_matched, dist_matched, n_alive_i, n_alive_t)."""
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
    dist_matched = 1.0 - cos_matched
    return C_matched, dist_matched, len(alive_i), len(alive_t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/multi_model_density_v5.pdf")
    parser.add_argument("--models", type=str, default="",
                        help="JSON file with [{name, run_dir, color}, ...]. Overrides built-in MODELS.")
    parser.add_argument("--r-min", type=float, default=R_MIN,
                        help=f"minimum r to include (default {R_MIN})")
    args = parser.parse_args()

    models = json.load(open(args.models)) if args.models else MODELS

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 1.82))

    for m in models:
        C_m, dist_m, n_ai, n_at = load_model_data(m["run_dir"])
        mask = C_m >= args.r_min
        n_kept = int(mask.sum())
        print(f"{m['name']}: alive_i={n_ai}, alive_t={n_at}, "
              f"matched={len(C_m)}, kept(r>={args.r_min})={n_kept}")
        if n_kept < 5:
            print(f"  skipping {m['name']} — too few pairs")
            continue
        vals = dist_m[mask]
        if np.std(vals) < 1e-8:
            print(f"  skipping {m['name']} — zero variance")
            continue
        kde = gaussian_kde(vals)
        density = kde(KDE_GRID)
        ax.fill_between(KDE_GRID, density, color=m["color"], alpha=FILL_ALPHA, linewidth=0)
        ax.plot(KDE_GRID, density, color=m["color"], alpha=LINE_ALPHA, lw=1.0, label=m["name"])

    ax.set_xlim(0.0, 1.3)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.set_xlabel("Cosine Distance", fontsize=9, labelpad=1)
    ax.set_ylabel("Density", fontsize=9, labelpad=2)
    ax.tick_params(axis="x", labelsize=8, pad=1)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)

    leg = fig.legend(
        loc="lower center",
        ncol=len(models),
        fontsize=8, frameon=False,
        handlelength=1.2, handletextpad=0.3, columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.06),
    )
    for line in leg.get_lines():
        line.set_linewidth(2.8)

    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.26, top=0.95)

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
