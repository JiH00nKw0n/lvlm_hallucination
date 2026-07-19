#!/usr/bin/env python3
"""Variant of plot_alpha_sweep.py that uses x = 1 - cos(phi_i, psi_i) as
the x-axis (labelled "Degree of Heterogeneity"), so the curves are
ordered from homogeneous (cos = 1, x = 0) to heterogeneous (cos = 0,
x = 1).

CR / RE come from runs/<run>/result.json; GRE is recomputed from the npz
weights so it stays consistent with plot_alpha_sweep.py.
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/theorem2_v2_1R2R_5seeds_coarse/runs/run_20260421_011823"

C_NO = "#90be6d"
C_YES = "#f9844a"
LBL_NO = "Shared SAE"
LBL_YES = "Modality-Specific SAEs"

METHODS = [
    ("single_recon", C_NO, "-o", LBL_NO),
    ("two_recon", C_YES, "-s", LBL_YES),
]

PANELS = [("CR", None), ("RE", None), ("GRE", None)]
NPZ_RE = re.compile(r"^alpha([\d.]+)_seed(\d+)_(.+)\.npz$")


def compute_gre_top1(w_enc, b_enc, w_dec, b_dec, gt, k=1):
    if gt.shape[1] == 0 or w_enc.shape[0] == 0:
        return float("nan")
    pre = w_enc.astype(np.float64) @ gt.astype(np.float64) + b_enc.astype(np.float64)[:, None]
    pre = np.maximum(pre, 0.0)
    kk = min(k, pre.shape[0])
    idx = np.argsort(pre, axis=0)[-kk:, :]
    z = np.zeros_like(pre)
    vals = np.take_along_axis(pre, idx, axis=0)
    np.put_along_axis(z, idx, vals, axis=0)
    recon = w_dec.astype(np.float64).T @ z + b_dec.astype(np.float64)[:, None]
    err = np.sum((gt.astype(np.float64) - recon) ** 2, axis=0)
    return float(err.mean())


def gather(run_dir: Path):
    """Return {metric: {method: (alphas, means, stds)}}."""
    with (run_dir / "result.json").open() as f:
        data = json.load(f)

    re_per_alpha = {}
    cr_per_alpha = {}
    for entry in data["sweep_results"]:
        a = round(float(entry["alpha_target"]), 2)
        agg = entry["aggregate"]
        for method, *_ in METHODS:
            re_per_alpha[(a, method)] = (
                agg[f"{method}/avg_eval_loss/mean"],
                agg[f"{method}/avg_eval_loss/std"],
            )
            cr_per_alpha[(a, method)] = (
                agg[f"{method}/merged_fraction/mean"],
                agg[f"{method}/merged_fraction/std"],
            )

    gre_acc: dict[tuple[float, str], list[float]] = {}
    for p in sorted((run_dir / "params").glob("alpha*_seed*_*.npz")):
        m = NPZ_RE.match(p.name)
        if not m:
            continue
        a = round(float(m.group(1)), 2)
        method = m.group(3)
        d = np.load(p, allow_pickle=True)
        gre_i = compute_gre_top1(d["w_enc_img"], d["b_enc_img"], d["w_dec_img"], d["b_dec_img"], d["phi_S"])
        gre_t = compute_gre_top1(d["w_enc_txt"], d["b_enc_txt"], d["w_dec_txt"], d["b_dec_txt"], d["psi_S"])
        gre_acc.setdefault((a, method), []).append(0.5 * (gre_i + gre_t))

    alphas = sorted({a for (a, _) in re_per_alpha})

    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {"CR": {}, "RE": {}, "GRE": {}}
    for method, *_ in METHODS:
        re_means, re_stds = [], []
        cr_means, cr_stds = [], []
        gre_means, gre_stds = [], []
        for a in alphas:
            re_m, re_s = re_per_alpha[(a, method)]
            cr_m, cr_s = cr_per_alpha[(a, method)]
            gre_vals = gre_acc.get((a, method), [])
            re_means.append(re_m); re_stds.append(re_s)
            cr_means.append(cr_m); cr_stds.append(cr_s)
            gre_means.append(float(np.mean(gre_vals)) if gre_vals else float("nan"))
            gre_stds.append(float(np.std(gre_vals)) if gre_vals else float("nan"))
        x = np.array(alphas)
        out["CR"][method] = (x, np.array(cr_means), np.array(cr_stds))
        out["RE"][method] = (x, np.array(re_means), np.array(re_stds))
        out["GRE"][method] = (x, np.array(gre_means), np.array(gre_stds))
    return out, alphas


def make_fig(series, alphas, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.015))
    handles = []

    # x_new = 1 - alpha
    het_alphas = [round(1 - a, 2) for a in alphas]

    for i, (ax, (metric, ylim)) in enumerate(zip(axes, PANELS)):
        for method, color, style, label in METHODS:
            xs, means, stds = series[metric][method]
            xs_het = 1.0 - xs
            order = np.argsort(xs_het)
            xs_h = xs_het[order]; m = means[order]; s = stds[order]
            (h,) = ax.plot(xs_h, m, style, color=color, lw=1.2, ms=3, label=label, zorder=3)
            ax.errorbar(xs_h, m, yerr=s, fmt="none", ecolor=color,
                        capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
            if i == 0:
                handles.append(h)

        ax.set_xlabel("Degree of Heterogeneity", fontsize=8, labelpad=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(sorted(set(het_alphas)))
        ax.tick_params(axis="both", labelsize=6.5, pad=1)

        all_vals = np.concatenate([series[metric][m][1] for m, *_ in METHODS
                                   if series[metric][m][1].size > 0])
        if metric == "CR":
            ax.set_ylim(-0.05, 1.05)
        elif all_vals.size:
            lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
            margin = (hi - lo) * 0.12 + 1e-6
            ax.set_ylim(lo - margin, hi + margin)
        ax.grid(alpha=0.15, linewidth=0.4, which="both")

    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2, frameon=False, fontsize=7,
        handlelength=1.5, columnspacing=1.5, handletextpad=0.4,
    )
    plt.subplots_adjust(top=0.72, bottom=0.22, left=0.05, right=0.99, wspace=0.30)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    print(f"saved {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--out", default=str(ROOT / "outputs/svg_export/alpha_sweep_heterogeneity.svg"))
    args = ap.parse_args()
    series, alphas = gather(Path(args.run))
    make_fig(series, alphas, Path(args.out))


if __name__ == "__main__":
    main()
