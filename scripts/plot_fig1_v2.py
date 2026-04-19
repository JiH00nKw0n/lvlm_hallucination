#!/usr/bin/env python3
"""Figure 1 (v2): 3-panel (RE, GRR, CR) with std shading, from v2 result.json.

Usage:
    python scripts/plot_fig1_v2.py \
        --result outputs/theorem2_v2_1R2R_5seeds/runs/*/result.json \
        --out    outputs/theorem2_v2_1R2R_5seeds/fig1_v2.pdf
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


# ---- style ----
# Vibrant Tones palette (full):
# #f94144 #f3722c #f8961e #f9844a #f9c74f #90be6d #43aa8b #4d908e #577590 #277da1
C_NO  = "#f94144"   # Strawberry Red
C_YES = "#277da1"   # Cerulean
LBL_NO  = "Shared SAE"
LBL_YES = "Separated SAE"

PANELS = [
    ("merged_fraction", "CR",   r"(a) Collapse Rate ($\downarrow$)",                (-0.05, 1.05)),
    ("avg_eval_loss",   "RE",   r"(b) Reconstruction Error ($\downarrow$)",         None),
    ("grr_avg_0.95",    "GRR",  r"(c) GT Recovery Rate ($\uparrow$)",     None),
]


def load(j, alphas_target):
    entries = []
    for e in j["sweep_results"]:
        a = round(e["alpha_target"], 2)
        if a in alphas_target:
            entries.append((a, e["per_seed"]))
    entries.sort(key=lambda x: x[0])

    alphas = [a for a, _ in entries]

    def get(method, metric):
        means, stds = [], []
        for _, seeds in entries:
            if metric.startswith("grr_avg_"):
                tau = metric.split("grr_avg_")[1]
                vals = [
                    (s[method][f"img_mgt_shared_tau{tau}"] + s[method][f"txt_mgt_shared_tau{tau}"]) / 2
                    for s in seeds if method in s
                ]
            else:
                vals = [s[method][metric] for s in seeds if method in s]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return np.array(means), np.array(stds)

    return np.array(alphas), get


def make_fig1(json_path, out_path, alphas_target):
    j = json.load(open(json_path))
    alphas, get = load(j, alphas_target)

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.45))
    handles = []

    for i, (ax, (metric, ylab, title, ylim)) in enumerate(zip(axes, PANELS)):
        m_no, s_no = get("single_recon", metric)
        m_yes, s_yes = get("two_recon", metric)

        h1, = ax.plot(alphas, m_no, "-o", color=C_NO, lw=1.2, ms=3, label=LBL_NO, zorder=3)
        ax.errorbar(alphas, m_no, yerr=s_no, fmt="none", ecolor=C_NO, capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)

        h2, = ax.plot(alphas, m_yes, "-s", color=C_YES, lw=1.2, ms=3, label=LBL_YES, zorder=3)
        ax.errorbar(alphas, m_yes, yerr=s_yes, fmt="none", ecolor=C_YES, capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)

        if i == 0:
            handles = [h1, h2]

        ax.set_xlabel(r"$\alpha$ (alignment)", fontsize=8, labelpad=1)
        # y-axis label omitted — title already has metric name
        ax.set_title(title, fontsize=7, fontweight="bold", pad=2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0] + sorted(alphas_target))
        ax.tick_params(axis="both", labelsize=6.5, pad=1)

        if ylim is not None:
            ax.set_ylim(*ylim)

        if metric == "avg_eval_loss":
            y_all = np.concatenate([m_no, m_yes])
            lo, hi = y_all.min(), y_all.max()
            margin = (hi - lo) * 0.1
            ax.set_ylim(lo - margin, hi + margin)

        ax.grid(alpha=0.15, linewidth=0.4, which="both")

    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2, frameon=False, fontsize=7,
        handlelength=1.5, columnspacing=1.5, handletextpad=0.4,
    )
    plt.subplots_adjust(top=0.72, bottom=0.22, left=0.05, right=0.99, wspace=0.30)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for ext in [out_path]:
        plt.savefig(ext, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
        print(f"saved {ext}")
    # also save png if output is pdf
    if out_path.endswith(".pdf"):
        png = out_path.replace(".pdf", ".png")
        plt.savefig(png, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
        print(f"saved {png}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str,
                        default="outputs/theorem2_v2_1R2R_5seeds/runs/*/result.json")
    parser.add_argument("--out", type=str,
                        default="outputs/theorem2_v2_1R2R_5seeds/fig1_v2.pdf")
    parser.add_argument("--alphas", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    args = parser.parse_args()

    result_path = args.result
    if "*" in result_path:
        matches = sorted(glob.glob(result_path))
        if not matches:
            raise FileNotFoundError(f"no match: {result_path}")
        result_path = matches[-1]

    alphas = tuple(float(x) for x in args.alphas.split(","))
    make_fig1(result_path, args.out, alphas)


if __name__ == "__main__":
    main()
