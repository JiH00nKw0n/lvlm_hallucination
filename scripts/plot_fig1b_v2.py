#!/usr/bin/env python3
"""Figure 1b: 4-panel (RE, GRR, ELSim, FLSim) with std error bars.

Usage:
    python scripts/plot_fig1b_v2.py \
        --result outputs/theorem2_v2_1R2R_5seeds/runs/*/result.json \
        --out    outputs/theorem2_v2_1R2R_5seeds/fig1b_v2.pdf
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
import matplotlib.pyplot as plt
import numpy as np


C_NO  = "#c53030"
C_YES = "#2b6cb0"
LBL_NO  = "Shared SAE"
LBL_YES = "Modality-specific SAEs"

PANELS = [
    ("avg_eval_loss",  "RE",                  r"(a) Reconstruction Error (RE) $\downarrow$",              None),
    ("grr_avg_0.95",   r"GRR ($\tau$=0.95)",  r"(b) Ground-truth Recovery Rate (GRR) $\uparrow$",        None),
    ("pair_cos_mean",  "ELSim",               r"(c) Embedding-pair Latent Similarity (ELSim) $\uparrow$", None),
    ("probe_vec_cos",  "FLSim",               r"(d) Feature-pair Latent Similarity (FLSim) $\uparrow$",   None),
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


def make_fig(json_path, out_path, alphas_target):
    j = json.load(open(json_path))
    alphas, get = load(j, alphas_target)

    fig, axes = plt.subplots(1, 4, figsize=(19, 2.6))
    handles = []

    for i, (ax, (metric, ylab, title, ylim)) in enumerate(zip(axes, PANELS)):
        m_no, s_no = get("single_recon", metric)
        m_yes, s_yes = get("two_recon", metric)

        h1, = ax.plot(alphas, m_no, "-o", color=C_NO, lw=1.8, ms=3.5, label=LBL_NO, zorder=3)
        ax.errorbar(alphas, m_no, yerr=s_no, fmt="none", ecolor=C_NO, capsize=6, capthick=1.4, elinewidth=1.4, zorder=4)

        h2, = ax.plot(alphas, m_yes, "-s", color=C_YES, lw=1.8, ms=3.5, label=LBL_YES, zorder=3)
        ax.errorbar(alphas, m_yes, yerr=s_yes, fmt="none", ecolor=C_YES, capsize=6, capthick=1.4, elinewidth=1.4, zorder=4)

        if i == 0:
            handles = [h1, h2]

        ax.set_xlabel(r"Ground-truth $\alpha$", fontsize=11, labelpad=1)
        ax.set_ylabel(ylab, fontsize=11, labelpad=1)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0] + sorted(alphas_target))
        ax.tick_params(axis="both", labelsize=10, pad=1)

        if ylim is not None:
            ax.set_ylim(*ylim)

        if metric == "avg_eval_loss":
            y_all = np.concatenate([m_no, m_yes])
            lo, hi = y_all.min(), y_all.max()
            margin = (hi - lo) * 0.1
            ax.set_ylim(lo - margin, hi + margin)

        ax.grid(alpha=0.25, which="both")

    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2, frameon=False, fontsize=10,
        handlelength=1.8, columnspacing=2.2, handletextpad=0.4,
    )
    plt.subplots_adjust(top=0.76, bottom=0.22, left=0.04, right=0.99, wspace=0.28)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    print(f"saved {out_path}")
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
                        default="outputs/theorem2_v2_1R2R_5seeds/fig1b_v2.pdf")
    parser.add_argument("--alphas", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    args = parser.parse_args()

    result_path = args.result
    if "*" in result_path:
        matches = sorted(glob.glob(result_path))
        if not matches:
            raise FileNotFoundError(f"no match: {result_path}")
        result_path = matches[-1]

    alphas = tuple(float(x) for x in args.alphas.split(","))
    make_fig(result_path, args.out, alphas)


if __name__ == "__main__":
    main()
