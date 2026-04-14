#!/usr/bin/env python3
"""Plot Figure 2 (3-panel: MR, GRR(τ=0.95), Recon Error) for λ sweep.

Compares Iso-Energy / Group-Sparse / Ours across λ (expressed as multiplier
of each method's paper default), with w/o and w/ modality masking baselines
as horizontal reference lines. Merge Rate uses MR_geom (vector-cosine basis),
computed offline from saved SAE decoders.

Usage:
    python scripts/plot_fig2_lambda_sweep.py \
        --root outputs/theorem2_followup_15 \
        --out  outputs/theorem2_followup_15/fig2_beta_sweep.png
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter


# ---------------- style constants ----------------
C_IA, C_GS, C_OURS = "#8c564b", "#d69e2e", "#188a3c"
C_1R, C_2R = "#c53030", "#2b6cb0"

DEFAULTS = {"IA": 0.03, "GS": 0.05, "ours": 2.0}

PANELS = [
    # (metric_key,              ylabel,                title,                                     ylim,          yticks_step, ylog)
    ("MR_geom",                 "MR",                   "(a) Merge Rate (MR)",                    None,          None,        False),
    ("img_mgt_shared_tau0.95",  r"GRR at $\tau=0.95$",  "(b) Ground-truth Recovery Rate (GRR)",   None,          None,        False),
    ("avg_eval_loss",           "RE (log scale)",       "(c) Reconstruction Error (RE)",          None,          None,        True),
]


def compute_mr_geom(npz_path: str, tau: float = 0.95) -> float:
    d = np.load(npz_path)
    w_img = d["w_dec_img"].astype(np.float64)
    w_txt = d["w_dec_txt"].astype(np.float64)
    phi = d["phi_S"].astype(np.float64).T
    psi = d["psi_S"].astype(np.float64).T
    w_img /= np.linalg.norm(w_img, axis=1, keepdims=True).clip(1e-12)
    w_txt /= np.linalg.norm(w_txt, axis=1, keepdims=True).clip(1e-12)
    phi   /= np.linalg.norm(phi,   axis=1, keepdims=True).clip(1e-12)
    psi   /= np.linalg.norm(psi,   axis=1, keepdims=True).clip(1e-12)
    top_img = (w_img @ phi.T).argmax(axis=0)
    top_txt = (w_txt @ psi.T).argmax(axis=0)
    return float(((w_img[top_img] * w_txt[top_txt]).sum(axis=1) > tau).mean())


def _get_metric(j: dict, mk: str, metric: str):
    return j["sweep_results"][0]["aggregate"].get(f"{mk}/{metric}_mean")


def _find_npz(root: str, run_tag: str, method_substr: str) -> str:
    runs_dir = os.path.join(root, "runs")
    for run in os.listdir(runs_dir):
        if run_tag not in run:
            continue
        pdir = os.path.join(runs_dir, run, "params")
        if not os.path.isdir(pdir):
            continue
        for f in os.listdir(pdir):
            if method_substr in f:
                return os.path.join(pdir, f)
    raise FileNotFoundError(f"no npz matching {run_tag}/{method_substr} under {runs_dir}")


def _collect(root: str, method_tag: str, method_id_template: str, npz_method_substr: str):
    paths = sorted(glob.glob(os.path.join(root, f"runs/*{method_tag}*/result.json")))
    items = []
    for p in paths:
        m = re.search(f"{method_tag}([\\d.eE+-]+)", p)
        if not m:
            continue
        beta = float(m.group(1))
        j = json.load(open(p))
        mk = method_id_template.format(beta) if "ours" in method_id_template else method_id_template
        run_dir = os.path.dirname(p)
        pdir = os.path.join(run_dir, "params")
        npz_path = None
        if os.path.isdir(pdir):
            for f in os.listdir(pdir):
                if npz_method_substr in f:
                    npz_path = os.path.join(pdir, f)
                    break
        items.append((beta, {
            "avg_eval_loss": _get_metric(j, mk, "avg_eval_loss"),
            "img_mgt_shared_tau0.95": _get_metric(j, mk, "img_mgt_shared_tau0.95"),
            "MR_geom": compute_mr_geom(npz_path) if npz_path else None,
        }))
    items.sort()
    return items


def make_fig2(root: str, out_path: str):
    # 1R / 2R baselines
    b_p = sorted(glob.glob(os.path.join(root, "runs/*1R_2R*/result.json")))[0]
    b_j = json.load(open(b_p))
    v_1R = {m: _get_metric(b_j, "single_recon", m) for m in ["avg_eval_loss", "img_mgt_shared_tau0.95"]}
    v_2R = {m: _get_metric(b_j, "two_recon",    m) for m in ["avg_eval_loss", "img_mgt_shared_tau0.95"]}
    v_1R["MR_geom"] = compute_mr_geom(_find_npz(root, "1R_2R", "single_recon"))
    v_2R["MR_geom"] = compute_mr_geom(_find_npz(root, "1R_2R", "two_recon"))

    ia_data   = _collect(root, "IA_beta", "iso_align",    "iso_align")
    gs_data   = _collect(root, "GS_lam",  "group_sparse", "group_sparse")
    ours_data = _collect(root, "ours_lam", "ours::lam{}_mS1024_k6_normglobal", "ours")

    to_mult = lambda items, d: [(b / d, r) for b, r in items]
    ia_m   = to_mult(ia_data,   DEFAULTS["IA"])
    gs_m   = to_mult(gs_data,   DEFAULTS["GS"])
    ours_m = to_mult(ours_data, DEFAULTS["ours"])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 2.16))
    for ax, (metric, ylab, title, ylim, ystep, ylog) in zip(axes, PANELS):
        ax.axhline(v_1R[metric], color=C_1R, ls=":", lw=1.3, alpha=0.85)
        ax.axhline(v_2R[metric], color=C_2R, ls=":", lw=1.3, alpha=0.85)
        ax.plot([m for m, _ in ia_m],   [r[metric] for _, r in ia_m],   "-s", color=C_IA,   lw=1.8, ms=6)
        ax.plot([m for m, _ in gs_m],   [r[metric] for _, r in gs_m],   "-D", color=C_GS,   lw=1.8, ms=6)
        ax.plot([m for m, _ in ours_m], [r[metric] for _, r in ours_m], "-*", color=C_OURS, lw=2.0, ms=9)

        ax.set_xscale("log", base=2)
        ax.set_xticks([1/16, 1/4, 1, 4, 16])
        ax.set_xticklabels(["1/16", "1/4", "1", "4", "16"])
        ax.set_xlabel(r"$\lambda$  ($\times$ default hyperparameter)", fontsize=9, labelpad=1)
        ax.set_ylabel(ylab, fontsize=9, labelpad=1)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=2)
        if ylim:
            ax.set_ylim(*ylim)
        if ystep:
            ax.yaxis.set_major_locator(MultipleLocator(ystep))
        if ylog:
            ax.set_yscale("log")
            y_ticks = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
            ax.yaxis.set_major_locator(FixedLocator(y_ticks))
            ax.yaxis.set_minor_locator(FixedLocator([]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(0.14, 0.55)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=8, pad=1)

    handles = [
        Line2D([0], [0], color=C_1R,   ls=":",     lw=1.3, label="w/o modality masking baseline"),
        Line2D([0], [0], color=C_2R,   ls=":",     lw=1.3, label="w/ modality masking baseline"),
        Line2D([0], [0], color=C_IA,   marker="s", lw=1.8, ms=6, label="Iso-Energy"),
        Line2D([0], [0], color=C_GS,   marker="D", lw=1.8, ms=6, label="Group-Sparse"),
        Line2D([0], [0], color=C_OURS, marker="*", lw=2.0, ms=9, label="Ours"),
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=5, frameon=False, fontsize=9.5,
        handlelength=1.6, columnspacing=1.6, handletextpad=0.35,
    )
    plt.subplots_adjust(top=0.77, bottom=0.21, left=0.05, right=0.99, wspace=0.22)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    print(f"saved {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/theorem2_followup_15",
                        help="Sweep output root containing runs/ subdir")
    parser.add_argument("--out",  type=str, default="outputs/theorem2_followup_15/fig2_beta_sweep.png",
                        help="Output PNG path")
    args = parser.parse_args()
    make_fig2(args.root, args.out)


if __name__ == "__main__":
    main()
