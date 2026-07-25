"""Compare cross-modal feature distance against same-modality training variability.

Reads the correlation matrices written by ``build_cross_pair_C.py`` and renders
them the way the paper's density figure does: every alive latent pair is grouped
by its co-activation correlation, and within each group we plot the distribution
of cosine distance between the two decoder directions.

The panels answer the reviewer's question by decomposition. Two image SAEs from
different runs read the identical embedding, so their distance is pure training
variability. Two text SAEs reading different captions of one photo add the
mismatch that comes from describing a scene two ways. The image-text panel adds
modality on top of that. Reading the three in order separates a training effect
from an input-mismatch effect from a modality effect.

Two reporting choices matter enough to state. Distances are summarized as the
median over image latents of each latent's own median, not as the median over
matrix cells, because one frequently firing latent contributes hundreds of cells
and would otherwise dominate. Confidence intervals resample latents rather than
cells for the same reason.

    python scripts/real_alpha/plot_same_modality_control.py \
        --panels outputs/rebuttal_EA/coco_k8/panels.json \
        --out outputs/rebuttal_EA/coco_k8/fig_same_modality.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

# Same conventions as the paper's density figure.
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_COLORS = ["#df3a3d", "#d96627", "#dfb246", "#389076", "#206987"]
ALIVE_THR = 1e-3
FILL_ALPHA = 0.2
KDE_GRID = np.linspace(0.0, 1.4, 400)
XTICKS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]

# Which decoder each side of a panel uses.
SIDE_OF_PANEL = {
    "img_img": ("image", "image"),
    "txt_txt": ("text", "text"),
    "txt_txt_diffcap": ("text", "text"),
    "img_txt": ("image", "text"),
}
TITLE_OF_PANEL = {
    "img_img": "image SAE, two runs",
    "txt_txt": "text SAE, two runs",
    "txt_txt_diffcap": "text SAE, two runs\ndifferent captions",
    "img_txt": "image vs text SAE\n(paper's measurement)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", required=True,
                   help='JSON list of {"panel", "npz", "ckpt_a", "ckpt_b"}')
    p.add_argument("--out", required=True)
    p.add_argument("--headline-c", type=float, default=0.6,
                   help="correlation threshold for the headline number")
    p.add_argument("--fallback-c", type=float, default=0.4,
                   help="used in the table when the headline bin is too small")
    p.add_argument("--min-bin", type=int, default=30,
                   help="below this many pairs a bin is reported but not drawn")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def unit_rows(w: np.ndarray) -> np.ndarray:
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def decoder(ckpt: str, side: str) -> np.ndarray:
    sd = load_file(str(Path(ckpt) / "model.safetensors"))
    return unit_rows(sd[f"{side}_sae.W_dec"].float().numpy())


def load_panel(spec: dict) -> dict:
    """Alive-restricted correlation and cosine-distance matrices for one panel."""
    z = np.load(spec["npz"])
    C = z["C"].astype(np.float32)
    alive_a = np.where(z["rate_a"] > ALIVE_THR)[0]
    alive_b = np.where(z["rate_b"] > ALIVE_THR)[0]

    side_a, side_b = SIDE_OF_PANEL[spec["panel"]]
    Wa = decoder(spec["ckpt_a"], side_a)[alive_a]
    Wb = decoder(spec["ckpt_b"], side_b)[alive_b]

    return {
        "panel": spec["panel"],
        "C": C[np.ix_(alive_a, alive_b)],
        "dist": 1.0 - (Wa @ Wb.T),
        "n_alive_a": len(alive_a),
        "n_alive_b": len(alive_b),
        "n_samples": int(z["n_samples"]),
    }


def per_row_medians(C: np.ndarray, dist: np.ndarray, thr: float) -> np.ndarray:
    """One distance per left-hand latent: the median over its qualifying pairs.

    Collapsing to one value per latent is what makes the summary a median over
    independent units. Latents with no pair above the threshold drop out.
    """
    mask = C >= thr
    rows = np.where(mask.any(axis=1))[0]
    return np.array([np.median(dist[r, mask[r]]) for r in rows], dtype=np.float64)


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    meds = np.median(values[draws], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summarize(p: dict, thr: float, n_boot: int, rng: np.random.Generator) -> dict:
    C, dist = p["C"], p["dist"]
    mask = C >= thr
    n_pairs = int(mask.sum())
    cell_vals = dist[mask]
    row_vals = per_row_medians(C, dist, thr)
    lo, hi = bootstrap_ci(row_vals, n_boot, rng)
    return {
        "threshold": thr,
        "n_pairs": n_pairs,
        "n_rows": int(len(row_vals)),
        "median_over_cells": float(np.median(cell_vals)) if n_pairs else float("nan"),
        "median_over_latents": float(np.median(row_vals)) if len(row_vals) else float("nan"),
        "iqr_over_latents": (
            [float(np.percentile(row_vals, 25)), float(np.percentile(row_vals, 75))]
            if len(row_vals) else [float("nan")] * 2
        ),
        "ci95_over_latents": [lo, hi],
    }


def quantile_slice(p: dict, q: float) -> dict:
    """Distance over the top-q fraction of pairs by correlation.

    An absolute threshold selects a different fraction of each panel's pairs,
    because the panels have different correlation distributions by construction.
    Taking a fixed quantile equalizes how selective the conditioning is.
    """
    C, dist = p["C"], p["dist"]
    flat_c, flat_d = C.reshape(-1), dist.reshape(-1)
    k = max(1, int(round(q * flat_c.size)))
    idx = np.argpartition(-flat_c, k - 1)[:k]
    return {
        "q": q, "n_pairs": int(k),
        "min_c": float(flat_c[idx].min()),
        "median": float(np.median(flat_d[idx])),
    }


def draw(ax, p: dict, min_bin: int, random_null: float) -> None:
    C, dist = p["C"], p["dist"]
    for b in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        m = (C >= lo) & (C < hi)
        if int(m.sum()) < min_bin:
            continue
        vals = dist[m]
        if np.std(vals) < 1e-8:
            continue
        d = gaussian_kde(vals)(KDE_GRID)
        ax.fill_between(KDE_GRID, d, color=BIN_COLORS[b], alpha=FILL_ALPHA, linewidth=0)
        ax.plot(KDE_GRID, d, color=BIN_COLORS[b], alpha=0.95, lw=0.7)

    ax.axvline(random_null, color="0.45", ls=":", lw=0.8)
    ax.set_title(TITLE_OF_PANEL[p["panel"]], fontsize=7.5, fontweight="bold", pad=3)
    ax.set_xlim(0.0, 1.4)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:g}" for t in XTICKS])
    ax.set_xlabel(r"cosine distance $1-\cos$", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=6.5, pad=1)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    specs = json.load(open(args.panels))
    panels = [load_panel(s) for s in specs]

    # Two random directions in R^d sit at cosine distance 1 with spread 1/sqrt(d).
    dim = load_file(str(Path(specs[0]["ckpt_a"]) / "model.safetensors"))["image_sae.W_dec"].shape[1]
    random_null = 1.0

    report: dict = {"dim": int(dim), "random_null_distance": random_null, "panels": {}}
    for p in panels:
        entry = {
            "n_alive_a": p["n_alive_a"], "n_alive_b": p["n_alive_b"],
            "n_samples": p["n_samples"],
            "headline": summarize(p, args.headline_c, args.bootstrap, rng),
            "fallback": summarize(p, args.fallback_c, args.bootstrap, rng),
            "quantile_1e-4": quantile_slice(p, 1e-4),
            "bins": {},
        }
        for b in range(len(BIN_EDGES) - 1):
            lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
            m = (p["C"] >= lo) & (p["C"] < hi)
            n = int(m.sum())
            entry["bins"][f"[{lo},{hi})"] = {
                "n_pairs": n,
                "median": float(np.median(p["dist"][m])) if n else None,
            }
        report["panels"][p["panel"]] = entry

    # Paired comparison: the two panels share the run-A image latents, so the
    # difference can be taken latent by latent rather than between summaries.
    by_name = {p["panel"]: p for p in panels}
    if "img_img" in by_name and "img_txt" in by_name:
        thr = args.headline_c
        a, b = by_name["img_img"], by_name["img_txt"]
        ma, mb = a["C"] >= thr, b["C"] >= thr
        shared = np.where(ma.any(axis=1) & mb.any(axis=1))[0]
        if len(shared):
            da = np.array([np.median(a["dist"][r, ma[r]]) for r in shared])
            db = np.array([np.median(b["dist"][r, mb[r]]) for r in shared])
            diff = db - da
            lo, hi = bootstrap_ci(diff, args.bootstrap, rng)
            from scipy.stats import wilcoxon  # noqa: PLC0415
            try:
                pval = float(wilcoxon(diff)[1])
            except ValueError:  # every difference is exactly zero
                pval = float("nan")
            report["paired_img_txt_minus_img_img"] = {
                "threshold": thr, "n_latents": int(len(shared)),
                "median_difference": float(np.median(diff)),
                "ci95": [lo, hi], "wilcoxon_p": float(pval),
            }
        else:
            report["paired_img_txt_minus_img_img"] = {
                "threshold": thr, "n_latents": 0,
                "note": "no image latent clears the threshold in both panels",
            }

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(1.85 * n, 1.75), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        draw(ax, p, args.min_bin, random_null)
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].set_ylabel("density", fontsize=7.5, labelpad=2)

    handles = [
        Patch(facecolor=to_rgba(BIN_COLORS[b], FILL_ALPHA),
              edgecolor=to_rgba(BIN_COLORS[b], 1.0), linewidth=1.0,
              label=rf"$c \in [{BIN_EDGES[b]:.1f},{BIN_EDGES[b+1]:.1f})$")
        for b in range(len(BIN_EDGES) - 1)
    ] + [Line2D([], [], color="0.45", ls=":", lw=0.8, label="random directions")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6.5,
               frameon=False, handlelength=1.1, handletextpad=0.3, columnspacing=0.9,
               bbox_to_anchor=(0.5, -0.13))
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.32, top=0.84, wspace=0.18)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(str(out.with_suffix(ext)), dpi=200, bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)
    plt.close(fig)

    out.with_suffix(".json").write_text(json.dumps(report, indent=2))

    # ---- console summary -----------------------------------------------------
    print()
    hdr = f"{'panel':<22}{'alive a/b':>12}{'n(c>=%.1f)' % args.headline_c:>12}" \
          f"{'median d':>11}{'95% CI':>18}"
    print(hdr)
    print("-" * len(hdr))
    for name, e in report["panels"].items():
        h = e["headline"]
        ci = f"[{h['ci95_over_latents'][0]:.3f}, {h['ci95_over_latents'][1]:.3f}]"
        print(f"{name:<22}{e['n_alive_a']:>5}/{e['n_alive_b']:<6}{h['n_pairs']:>12}"
              f"{h['median_over_latents']:>11.3f}{ci:>18}")
    if "paired_img_txt_minus_img_img" in report:
        pr = report["paired_img_txt_minus_img_img"]
        if pr.get("n_latents"):
            print(f"\npaired (img_txt - img_img) over {pr['n_latents']} shared latents: "
                  f"{pr['median_difference']:+.3f}  "
                  f"95% CI [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}]  "
                  f"Wilcoxon p={pr['wilcoxon_p']:.2e}")
    print(f"\nwrote {out.with_suffix('.pdf')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
