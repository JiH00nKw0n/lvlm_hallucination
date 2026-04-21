#!/usr/bin/env python3
"""Figure 1 (v3): 3-panel (CR, RE, GRE) with std shading, from v2 result.json + saved npz params.

GRE = Ground-truth Recovery Error with top-1 activation (lower is better).
    GRE := mean_i ||phi_i - V*sigma(V.T phi_i)||_2^2 averaged over img/txt sides.

CR and RE are read from result.json's per-seed dict. GRE is recomputed offline
from the saved decoder npz files (existing result.json files predate this metric).

Usage:
    python scripts/plot_fig1_v3.py \
        --result outputs/theorem2_v2_1R2R_5seeds/runs/run_20260418_054632/result.json \
        --params-run outputs/theorem2_v2_1R2R_5seeds/runs/run_20260417_165325 \
        --out outputs/theorem2_v2_1R2R_5seeds/fig1_v3.pdf
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.synthetic_eval import compute_gre_top1


# ---- style ----
C_NO  = "#f94144"   # Strawberry Red
C_YES = "#277da1"   # Cerulean
LBL_NO  = "Shared SAE"
LBL_YES = "Separated SAE"

PANELS = [
    ("merged_fraction", r"(a) Collapse Rate ($\downarrow$)",          (-0.05, 1.05)),
    ("avg_eval_loss",   r"(b) Reconstruction Error ($\downarrow$)",   None),
    ("gre_avg_shared",  r"(c) GT Recovery Error ($\downarrow$)",      None),
]

METHODS = [("single_recon", C_NO, "-o", LBL_NO),
           ("two_recon",    C_YES, "-s", LBL_YES)]


def load_gre_from_params(params_dir: Path):
    """Compute GRE for every (alpha, seed, method) npz in params_dir.

    Returns dict ``{(alpha, seed, method): gre_avg}``.
    """
    out = {}
    for npz_path in sorted(params_dir.glob("alpha*_seed*_*.npz")):
        name = npz_path.stem  # e.g. "alpha0.50_seed1_single_recon"
        parts = name.split("_")
        alpha = round(float(parts[0].replace("alpha", "")), 2)
        seed = int(parts[1].replace("seed", ""))
        method = "_".join(parts[2:])
        d = np.load(npz_path, allow_pickle=True)
        img_gre = compute_gre_top1(d["w_dec_img"], d["phi_S"])
        txt_gre = compute_gre_top1(d["w_dec_txt"], d["psi_S"])
        out[(alpha, seed, method)] = 0.5 * (img_gre + txt_gre)
    return out


def load_metric_series(j, alphas_target, metric, gre_lookup=None):
    """Return (alphas, {method: (means, stds)}) for a single metric across alphas."""
    entries = []
    for e in j["sweep_results"]:
        a = round(e["alpha_target"], 2)
        if a in alphas_target:
            entries.append((a, e["per_seed"]))
    entries.sort(key=lambda x: x[0])
    alphas = np.array([a for a, _ in entries])

    per_method: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, _, _, _ in METHODS:
        means, stds = [], []
        for a, seeds in entries:
            if metric == "gre_avg_shared":
                assert gre_lookup is not None
                vals = [
                    gre_lookup[(a, s["seed"], method)]
                    for s in seeds
                    if (a, s["seed"], method) in gre_lookup
                ]
            else:
                vals = [s[method][metric] for s in seeds if method in s]
            if not vals:
                means.append(np.nan); stds.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
        per_method[method] = (np.array(means), np.array(stds))
    return alphas, per_method


def make_fig(json_path, params_dir, out_path, alphas_target):
    j = json.load(open(json_path))

    gre_lookup = None
    if params_dir is not None:
        print(f"computing GRE from {params_dir} ...")
        gre_lookup = load_gre_from_params(Path(params_dir))
        print(f"  loaded GRE for {len(gre_lookup)} (alpha, seed, method) triples")

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.45))
    handles = []

    for i, (ax, (metric, title, ylim)) in enumerate(zip(axes, PANELS)):
        alphas, per_method = load_metric_series(j, alphas_target, metric, gre_lookup)

        for method, color, style, label in METHODS:
            means, stds = per_method[method]
            mask = ~np.isnan(means)
            if not mask.any():
                continue
            h, = ax.plot(alphas[mask], means[mask], style, color=color, lw=1.2, ms=3, label=label, zorder=3)
            ax.errorbar(alphas[mask], means[mask], yerr=stds[mask], fmt="none",
                        ecolor=color, capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
            if i == 0:
                handles.append(h)

        ax.set_xlabel(r"$\alpha$ (alignment)", fontsize=8, labelpad=1)
        ax.set_title(title, fontsize=7, fontweight="bold", pad=2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0] + sorted(alphas_target))
        ax.tick_params(axis="both", labelsize=6.5, pad=1)

        if ylim is not None:
            ax.set_ylim(*ylim)

        if metric in ("avg_eval_loss", "gre_avg_shared"):
            m_all = np.concatenate([per_method[m][0] for m, *_ in METHODS])
            m_all = m_all[~np.isnan(m_all)]
            if m_all.size:
                lo, hi = m_all.min(), m_all.max()
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
                        default="outputs/theorem2_v2_1R2R_5seeds/runs/run_20260418_054632/result.json")
    parser.add_argument("--params-run", type=str,
                        default="outputs/theorem2_v2_1R2R_5seeds/runs/run_20260417_165325",
                        help="Run dir containing params/*.npz (used for GRE recompute)")
    parser.add_argument("--out", type=str,
                        default="outputs/theorem2_v2_1R2R_5seeds/fig1_v3.pdf")
    parser.add_argument("--alphas", type=str, default="0.2,0.4,0.6,0.8,1.0")
    args = parser.parse_args()

    result_path = args.result
    if "*" in result_path:
        matches = sorted(glob.glob(result_path))
        if not matches:
            raise FileNotFoundError(f"no match: {result_path}")
        result_path = matches[-1]

    params_dir = Path(args.params_run) / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"no params dir: {params_dir}")

    alphas = tuple(round(float(x), 2) for x in args.alphas.split(","))
    make_fig(result_path, params_dir, args.out, alphas)


if __name__ == "__main__":
    main()
