#!/usr/bin/env python3
"""Figure 2: 4-panel lambda sweep (RE, GRR, ELSim, FLSim) at fixed alpha=0.5.

x-axis: lambda multiplier (1/16, 1/4, 1, 4, 16)
Lines: iso_align, group_sparse (+ baselines as hlines)

Usage:
    python scripts/plot_fig2_v2.py \
        --result outputs/theorem2_v2_lambda_sweep/runs/*/result.json \
        --out outputs/theorem2_v2_lambda_sweep/lambda_sweep.pdf
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np

from palette import STRAWBERRY_RED, CARROT_ORANGE, SEAWEED, CERULEAN

# ---- methods & style ----
MULTIPLIERS = [1/16, 1/4, 1, 4, 16]
MULT_LABELS = ["1/16", "1/4", "1", "4", "16"]

# aux_weight bases
IA_BASE = 0.03
GS_BASE = 0.05

METHODS = {
    "iso_align": {
        "base": IA_BASE,
        "color": CARROT_ORANGE,
        "label": "Iso-Energy Align",
        "marker": "^",
    },
    "group_sparse": {
        "base": GS_BASE,
        "color": SEAWEED,
        "label": "Group-Sparse",
        "marker": "D",
    },
}

BASELINES = {
    "single_recon": {"color": STRAWBERRY_RED, "label": "Shared SAE", "ls": "--"},
    "two_recon": {"color": CERULEAN, "label": "Separated SAE", "ls": ":"},
}

PANELS = [
    ("avg_eval_loss",  r"(a) Reconstruction Error ($\downarrow$)"),
    ("grr_avg_0.95",   r"(b) GT Recovery Rate ($\uparrow$)"),
    ("pair_cos_mean",  r"(c) ELSim ($\uparrow$)"),
    ("probe_vec_cos",  r"(d) FLSim ($\uparrow$)"),
]


def extract(j):
    """Return {method_key: {metric: (means, stds)}} from sweep_results."""
    sr = j["sweep_results"][0]  # single alpha
    seeds = sr["per_seed"]
    all_methods = seeds[0].keys()

    def get_vals(method_key, metric):
        if metric == "grr_avg_0.95":
            return [
                (s[method_key]["img_mgt_shared_tau0.95"]
                 + s[method_key]["txt_mgt_shared_tau0.95"]) / 2
                for s in seeds if method_key in s
            ]
        return [s[method_key][metric] for s in seeds if method_key in s]

    result = {}
    for mk in all_methods:
        if mk in ("alpha_actual_mean", "alpha_actual_std", "alpha_target",
                   "latent_size", "seed"):
            continue
        result[mk] = {}
        for metric, _ in PANELS:
            vals = get_vals(mk, metric)
            result[mk][metric] = (np.mean(vals), np.std(vals))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str,
                        default="outputs/theorem2_v2_lambda_sweep/runs/*/result.json")
    parser.add_argument("--out", type=str,
                        default="outputs/theorem2_v2_lambda_sweep/lambda_sweep.pdf")
    args = parser.parse_args()

    rp = args.result
    if "*" in rp:
        matches = sorted(glob.glob(rp))
        if not matches:
            raise FileNotFoundError(f"no match: {rp}")
        rp = matches[-1]

    j = json.load(open(rp))
    data = extract(j)

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.45))
    x = np.arange(len(MULTIPLIERS))
    all_handles, all_labels = [], []

    for ax, (metric, title) in zip(axes, PANELS):
        # baselines as hlines
        for bk, bstyle in BASELINES.items():
            m, s = data[bk][metric]
            h = ax.axhline(m, color=bstyle["color"], ls=bstyle["ls"],
                           lw=1.0, label=bstyle["label"], zorder=1)
            ax.axhspan(m - s, m + s, color=bstyle["color"], alpha=0.08, zorder=0)
            if metric == PANELS[0][0]:
                all_handles.append(h)
                all_labels.append(bstyle["label"])

        # sweep lines
        for mk, mstyle in METHODS.items():
            means, stds = [], []
            for mult in MULTIPLIERS:
                w = round(mstyle["base"] * mult, 6)
                key = f"{mk}_w{w}"
                m, s = data[key][metric]
                means.append(m)
                stds.append(s)
            means, stds = np.array(means), np.array(stds)
            h, = ax.plot(x, means, f"-{mstyle['marker']}", color=mstyle["color"],
                         lw=1.2, ms=3, label=mstyle["label"], zorder=3)
            ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=mstyle["color"],
                        capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)
            if metric == PANELS[0][0]:
                all_handles.append(h)
                all_labels.append(mstyle["label"])

        ax.set_xticks(x)
        ax.set_xticklabels(MULT_LABELS)
        ax.set_xlabel(r"$\lambda$ multiplier", fontsize=8, labelpad=1)
        ax.set_title(title, fontsize=6.5, fontweight="bold", pad=2)
        ax.tick_params(axis="both", labelsize=6, pad=1)
        ax.grid(alpha=0.15, linewidth=0.4)

    fig.legend(
        handles=all_handles, labels=all_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.97),
        ncol=4, frameon=False, fontsize=6,
        handlelength=1.5, columnspacing=1.0, handletextpad=0.3,
    )
    plt.subplots_adjust(top=0.72, bottom=0.22, left=0.05, right=0.99, wspace=0.38)

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
