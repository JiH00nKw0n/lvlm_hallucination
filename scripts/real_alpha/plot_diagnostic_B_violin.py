"""Publication-quality violin plot for Diagnostic B.

Uses the Hungarian-alive matched pool: restrict C and decoder cosines to
alive×alive latents, run Hungarian on -|C|, collect matched (C, cos) pairs,
then bin the matched pairs by C into equal-width bins and show the within-bin
decoder-cosine distribution as a violin plot.

Inputs required (all produced by earlier runs):
  <run_dir>/model.safetensors                  — TwoSidedTopKSAE weights
  <run_dir>/diagnostic_B_C_train.npy           — 4096x4096 C matrix (or similar)
  <cache_dir>/image_embeddings.pt               — CachedClipPairsDataset input
  <cache_dir>/text_embeddings.pt
  <cache_dir>/splits.json

Usage:
    python scripts/real_alpha/plot_diagnostic_B_violin.py \
        --run-dir outputs/real_alpha_followup_1/two_sae/final \
        --cache-dir cache/clip_b32_coco \
        --out outputs/real_alpha_followup_1/fig_diagnostic_B_violin \
        --bin-width 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment

# Bootstrap so `from src.models...` works without triggering broken package __init__.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore  # noqa: E402
from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore  # noqa: E402


def compute_alive_masks(
    run_dir: Path, cache_dir: str, alive_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return alive latent index arrays. Caches firing rates to run_dir."""
    rates_cache = run_dir / "diagnostic_B_firing_rates.npz"
    if rates_cache.exists():
        z = np.load(rates_cache)
        rate_i = z["rate_i"]
        rate_t = z["rate_t"]
        print(f"loaded cached firing rates from {rates_cache}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TwoSidedTopKSAE.from_pretrained(str(run_dir)).to(device).eval()
        ds = CachedClipPairsDataset(cache_dir, split="train", l2_normalize=True)
        N = len(ds)
        img = torch.stack([ds[i]["image_embeds"] for i in range(N)])
        txt = torch.stack([ds[i]["text_embeds"] for i in range(N)])
        n = int(model.image_sae.latent_size)
        fire_i = np.zeros(n, dtype=np.int64)
        fire_t = np.zeros(n, dtype=np.int64)
        B = 4096
        with torch.no_grad():
            for s in range(0, N, B):
                hs_i = img[s:s + B].to(device).unsqueeze(1)
                hs_t = txt[s:s + B].to(device).unsqueeze(1)
                out_i = model.image_sae(hidden_states=hs_i, return_dense_latents=True)
                out_t = model.text_sae(hidden_states=hs_t, return_dense_latents=True)
                fire_i += (out_i.dense_latents.squeeze(1).cpu().numpy() > 0).sum(axis=0)
                fire_t += (out_t.dense_latents.squeeze(1).cpu().numpy() > 0).sum(axis=0)
        rate_i = fire_i / N
        rate_t = fire_t / N
        np.savez(rates_cache, rate_i=rate_i, rate_t=rate_t)
        print(f"saved firing rates to {rates_cache}")
    alive_i = np.where(rate_i > alive_thr)[0]
    alive_t = np.where(rate_t > alive_thr)[0]
    return alive_i, alive_t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="output basename (no ext)")
    p.add_argument("--bin-width", type=float, default=0.15)
    p.add_argument("--c-max", type=float, default=1.05,
                   help="upper edge of last bin")
    p.add_argument("--alive-thr", type=float, default=0.001,
                   help="min firing rate to count a latent as alive")
    p.add_argument("--min-bin-count", type=int, default=3,
                   help="skip bins with fewer matches than this")
    p.add_argument("--include-negative-bin", action="store_true", default=True,
                   help="prepend a (-inf, 0) bin (default: True)")
    p.add_argument("--no-negative-bin", dest="include_negative_bin",
                   action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # --- load C, decoders ---
    C = np.load(run_dir / "diagnostic_B_C_train.npy")
    sd = load_file(run_dir / "model.safetensors")
    W_i = sd["image_sae.W_dec"].float().numpy()
    W_t = sd["text_sae.W_dec"].float().numpy()
    W_i_norm = W_i / (np.linalg.norm(W_i, axis=1, keepdims=True) + 1e-12)
    W_t_norm = W_t / (np.linalg.norm(W_t, axis=1, keepdims=True) + 1e-12)
    cos_ij = W_i_norm @ W_t_norm.T

    # --- alive masks + Hungarian on alive×alive ---
    alive_i, alive_t = compute_alive_masks(run_dir, args.cache_dir, args.alive_thr)
    print(f"alive image={len(alive_i)}, alive text={len(alive_t)}")
    C_sub = C[np.ix_(alive_i, alive_t)]
    cos_sub = cos_ij[np.ix_(alive_i, alive_t)]
    # Hungarian on signed -C (maximize sum of positive correlation)
    row_ind, col_ind = linear_sum_assignment(-C_sub)
    C_matched = C_sub[row_ind, col_ind]
    cos_matched = cos_sub[row_ind, col_ind]
    print(f"n matched pairs: {len(C_matched)}")

    # --- bin by C starting at 0 (plus an optional (-inf, 0) bin prepended) ---
    bw = args.bin_width
    pos_edges = list(np.arange(0.0, args.c_max + 1e-9, bw))
    if pos_edges[-1] < args.c_max:
        pos_edges.append(pos_edges[-1] + bw)

    # Edges are a list of (lo, hi, label) tuples; -inf is written as a symbol.
    edge_specs: list[tuple[float, float, str]] = []
    if args.include_negative_bin:
        edge_specs.append((float("-inf"), 0.0, r"$(-\infty,$" + "\n" + r"$\,0)$"))
    for lo, hi in zip(pos_edges[:-1], pos_edges[1:]):
        edge_specs.append((float(lo), float(hi), f"[{lo:.2f},\n{hi:.2f})"))

    bins: list[np.ndarray] = []
    labels: list[str] = []
    counts: list[int] = []
    for lo, hi, lab in edge_specs:
        if lo == float("-inf"):
            m = C_matched < hi
        else:
            m = (C_matched >= lo) & (C_matched < hi)
        sub = cos_matched[m]
        if sub.size < args.min_bin_count:
            continue
        bins.append(sub)
        labels.append(lab)
        counts.append(int(sub.size))

    # --- publication rcParams ---
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    positions = np.arange(len(bins))

    # Box plot — standard quartile drawing
    box_face = "#4C72B0"
    box_edge = "#1f3b6b"
    mean_color = "#d62728"
    bp = ax.boxplot(
        bins,
        positions=positions.tolist(),
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=True,
        medianprops=dict(color="white", linewidth=1.6),
        meanprops=dict(color=mean_color, linewidth=1.6, linestyle="-"),
        whiskerprops=dict(color=box_edge, linewidth=0.9),
        capprops=dict(color=box_edge, linewidth=0.9),
        boxprops=dict(facecolor=box_face, edgecolor=box_edge, linewidth=0.9, alpha=0.75),
        flierprops=dict(
            marker="o", markerfacecolor="none", markeredgecolor=box_edge,
            markersize=2.8, markeredgewidth=0.5, alpha=0.7,
        ),
    )

    # sample counts inside the bottom of each box, small and gray
    for i in positions:
        ax.text(i, -0.22, f"n={counts[i]}", ha="center", va="bottom",
                fontsize=7.5, color="gray")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Co-activation bin $C$")
    ax.set_ylabel(r"Decoder cosine $\rho$")
    ax.set_ylim(-0.25, 0.85)
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)

    # Also dump the bin summary as JSON
    summary = {
        "bin_width": args.bin_width,
        "alive_i": int(len(alive_i)),
        "alive_t": int(len(alive_t)),
        "n_matched": int(len(C_matched)),
        "bins": [
            {
                "label": lab,
                "n": counts[i],
                "cos_mean": float(bins[i].mean()),
                "cos_std": float(bins[i].std()),
                "cos_median": float(np.median(bins[i])),
                "cos_q1": float(np.percentile(bins[i], 25)),
                "cos_q3": float(np.percentile(bins[i], 75)),
                "cos_min": float(bins[i].min()),
                "cos_max": float(bins[i].max()),
            }
            for i, lab in enumerate(labels)
        ],
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
