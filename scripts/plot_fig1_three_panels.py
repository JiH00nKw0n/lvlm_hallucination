#!/usr/bin/env python3
"""Plot Figure 1 (3-panel: MR, GRR(τ=0.95), Recon Error).

Compares w/o vs w/ modality masking (1R vs 2R) across α.

Usage:
    python scripts/plot_fig1_three_panels.py \
        --result outputs/theorem2_followup_12/runs/<run_dir>/result.json \
        --out    outputs/theorem2_followup_12/fig1_three_panels.png
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ---------------- style constants ----------------
C_NO  = "#c53030"   # w/o modality masking (single decoder)
C_YES = "#2b6cb0"   # w/  modality masking (two decoders)
LBL_NO  = "w/o modality masking"
LBL_YES = "w/  modality masking"

PANELS = [
    # (metric_key,                 ylabel,               title,                                    ylim,     yticks_step)
    ("merged_fraction",            "MR",                  "(a) Merge Rate (MR)",                    (-0.05, 1.05), None),
    ("img_mgt_shared_tau0.95",     r"GRR at $\tau=0.95$", "(b) Ground-truth Recovery Rate (GRR)",   (-0.05, 1.05), None),
    ("avg_eval_loss",              "Recon Error",          "(c) Reconstruction Error",               None,          0.01),
]


def load_alphas(j, alphas_target):
    SUB = [
        (e["alpha_target"], e["aggregate"])
        for e in j["sweep_results"]
        if round(e["alpha_target"], 2) in alphas_target
    ]
    alphas = [a for a, _ in SUB]
    def get(mk, metric):
        return [ag.get(f"{mk}/{metric}_mean") for _, ag in SUB]
    return alphas, get


def make_fig1(json_path: str, out_path: str, alphas_target=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    j = json.load(open(json_path))
    alphas, get = load_alphas(j, alphas_target)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 2.16))
    handles = []
    for i, (ax, (metric, ylab, title, ylim, ystep)) in enumerate(zip(axes, PANELS)):
        y_no  = get("single_recon", metric)
        y_yes = get("two_recon",    metric)
        h1, = ax.plot(alphas, y_no,  "-o", color=C_NO,  lw=1.8, ms=6, label=LBL_NO)
        h2, = ax.plot(alphas, y_yes, "-s", color=C_YES, lw=1.8, ms=6, label=LBL_YES)
        if i == 0:
            handles = [h1, h2]
        ax.set_xlabel(r"ground-truth  $\alpha$", fontsize=9, labelpad=1)
        ax.set_ylabel(ylab, fontsize=9, labelpad=1)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=2)
        ax.set_xticks(sorted(alphas_target))
        ax.tick_params(axis="both", labelsize=8, pad=1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if ystep is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ystep))
        ax.grid(alpha=0.3)

    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2, frameon=False, fontsize=10,
        handlelength=1.8, columnspacing=2.2, handletextpad=0.4,
    )
    plt.subplots_adjust(top=0.77, bottom=0.21, left=0.05, right=0.99, wspace=0.22)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    print(f"saved {out_path}")
    plt.close()


def _resolve_result(arg: str) -> str:
    """If arg looks like a glob with *, expand and pick the latest match."""
    if "*" in arg:
        matches = sorted(glob.glob(arg))
        if not matches:
            raise FileNotFoundError(f"no result.json matches {arg}")
        return matches[-1]
    return arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", type=str,
        default="outputs/theorem2_followup_12/runs/*/result.json",
        help="Path (or glob) to result.json",
    )
    parser.add_argument(
        "--out", type=str,
        default="outputs/theorem2_followup_12/fig1_three_panels.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--alphas", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated α values to include",
    )
    args = parser.parse_args()

    result_path = _resolve_result(args.result)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x.strip())
    make_fig1(result_path, args.out, alphas_target=alphas)


if __name__ == "__main__":
    main()
