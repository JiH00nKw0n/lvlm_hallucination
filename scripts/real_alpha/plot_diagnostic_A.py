"""Diagnostic A: per-epoch train/eval reconstruction loss for one-SAE vs two-SAE.

Reads `loss_history.json` (produced by HF Trainer.state.log_history) from
each variant's run directory and plots:
  - top panel: train + eval recon loss curves for both variants
  - bottom panel: Δ_loss(t) = L_one_eval - L_two_eval

Usage:
    python scripts/real_alpha/plot_diagnostic_A.py \
        --one-dir outputs/real_alpha_followup_1/one_sae \
        --two-dir outputs/real_alpha_followup_1/two_sae \
        --output outputs/real_alpha_followup_1/fig_diagnostic_A.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--one-dir", type=str, required=True)
    p.add_argument("--two-dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def read_history(run_dir: str) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (epochs_train, train_loss, epochs_eval, eval_loss)."""
    with open(Path(run_dir) / "loss_history.json", "r") as f:
        log = json.load(f)
    ep_tr, tr, ep_ev, ev = [], [], [], []
    for entry in log:
        if "loss" in entry and "eval_loss" not in entry and "epoch" in entry:
            ep_tr.append(float(entry["epoch"]))
            tr.append(float(entry["loss"]))
        if "eval_loss" in entry and "epoch" in entry:
            ep_ev.append(float(entry["epoch"]))
            ev.append(float(entry["eval_loss"]))
    return ep_tr, tr, ep_ev, ev


def main() -> None:
    args = parse_args()
    ep_tr_1, tr_1, ep_ev_1, ev_1 = read_history(args.one_dir)
    ep_tr_2, tr_2, ep_ev_2, ev_2 = read_history(args.two_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, height_ratios=[2, 1])

    ax_top.plot(ep_tr_1, tr_1, color="#4C72B0", alpha=0.5, linewidth=1, label="one SAE train")
    ax_top.plot(ep_ev_1, ev_1, color="#4C72B0", marker="o", linewidth=2, label="one SAE eval")
    ax_top.plot(ep_tr_2, tr_2, color="#DD8452", alpha=0.5, linewidth=1, label="two SAE train")
    ax_top.plot(ep_ev_2, ev_2, color="#DD8452", marker="s", linewidth=2, label="two SAE eval")
    ax_top.set_ylabel("recon loss")
    ax_top.set_title(r"Diagnostic A: $\mathcal{L}^{\mathrm{rec}}$ per epoch (one SAE vs two SAE)")
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.3)

    # Delta: only defined at epochs both variants have eval for
    n = min(len(ev_1), len(ev_2))
    delta_x = ep_ev_1[:n]
    delta_y = [ev_1[i] - ev_2[i] for i in range(n)]
    ax_bot.plot(delta_x, delta_y, color="#55A868", marker="D", linewidth=2)
    ax_bot.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_bot.set_xlabel("epoch")
    ax_bot.set_ylabel(r"$\Delta_{\mathrm{loss}} = L_{\mathrm{one}} - L_{\mathrm{two}}$")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
