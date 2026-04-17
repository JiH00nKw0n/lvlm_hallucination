"""Publication-quality Diagnostic A figure (two panels).

Top panel: per-epoch eval reconstruction loss curves for one-SAE vs two-SAE,
log-scaled y-axis, y_min clamped slightly below the two-SAE minimum.
Bottom panel: per-epoch gap Δ_loss(t) = L_one(t) − L_two(t).

Usage:
    python scripts/real_alpha/plot_diagnostic_A_pub.py \
        --one outputs/real_alpha_followup_1/one_sae \
        --two outputs/real_alpha_followup_1/two_sae \
        --out outputs/real_alpha_followup_1/fig_diagnostic_A_pub
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def read_eval(run_dir: str) -> tuple[list[float], list[float]]:
    with open(Path(run_dir) / "loss_history.json") as f:
        log = json.load(f)
    epochs: list[float] = []
    losses: list[float] = []
    for entry in log:
        if "eval_loss" in entry and "epoch" in entry:
            epochs.append(float(entry["epoch"]))
            losses.append(float(entry["eval_loss"]))
    return epochs, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", type=str, required=True)
    parser.add_argument("--two", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ymin-scale", type=float, default=0.98,
                        help="y_min = two_sae_min * ymin_scale")
    parser.add_argument("--ymax-scale", type=float, default=1.05,
                        help="y_max = one_sae_max * ymax_scale")
    args = parser.parse_args()

    ep_one, loss_one = read_eval(args.one)
    _ep_two, loss_two = read_eval(args.two)
    n = min(len(loss_one), len(loss_two))
    ep = ep_one[:n]
    l1 = loss_one[:n]
    l2 = loss_two[:n]
    delta = [a - b for a, b in zip(l1, l2)]

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    C_ONE = "#d62728"
    C_TWO = "#1f77b4"
    C_DELTA = "#2ca02c"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12},
    )

    # --- top: log-scale loss curves ---
    ax_top.plot(ep, l1, color=C_ONE, marker="o", markerfacecolor="white",
                markeredgewidth=1.3, label="One SAE (shared decoder)")
    ax_top.plot(ep, l2, color=C_TWO, marker="s", markerfacecolor="white",
                markeredgewidth=1.3, label="Two SAE (modality-masked)")
    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$\mathcal{L}_{\mathrm{rec}}$ (eval)")
    ax_top.grid(True, alpha=0.3, which="both", linestyle="--")
    ax_top.legend(loc="upper right")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    y_min = min(l2) * args.ymin_scale
    y_max = max(l1) * args.ymax_scale
    ax_top.set_ylim(y_min, y_max)

    # --- bottom: delta ---
    ax_bot.plot(ep, delta, color=C_DELTA, marker="D", markerfacecolor="white",
                markeredgewidth=1.3)
    ax_bot.axhline(0, color="black", linewidth=0.6)
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel(
        r"$\Delta_{\mathrm{loss}}$"
    )
    ax_bot.grid(True, alpha=0.3, linestyle="--")
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    d_min, d_max = min(delta), max(delta)
    pad = 0.12 * (d_max - d_min + 1e-9)
    ax_bot.set_ylim(d_min - pad, d_max + pad)

    ax_bot.set_xticks(list(range(0, int(max(ep)) + 1, 5)))

    fig.align_ylabels([ax_top, ax_bot])
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
