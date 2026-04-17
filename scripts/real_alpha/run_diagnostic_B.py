"""Diagnostic B for real CLIP α-diagnostic experiment.

Loads a trained TwoSidedTopKSAE checkpoint + the cached Karpathy train pairs,
computes:
  1) Signed Pearson correlation matrix C between dense image/text latents
     (streaming, reuses `_compute_latent_correlation` from synthetic_theorem2_method).
  2) Hungarian matching on -|C| to find 1-to-1 matched latent index pairs.
  3) For each matched pair (i, π(i)), the decoder-column cosine
        ρ_k = ⟨W_dec_img[i], W_dec_txt[π(i)]⟩ / (|·||·|)
     (which, with unit-norm decoders, is just the inner product).
  4) Threshold sweep over top {10, 20, 30, 50}% co-firing pairs, reporting
     median/mean/std of ρ on each subset.

Writes:
  <run_dir>/diagnostic_B.json
  <run_dir>/fig_diagnostic_B_histogram.png

Usage:
    python scripts/real_alpha/run_diagnostic_B.py \
        --run-dir outputs/real_alpha_followup_1/two_sae/final \
        --cache-dir cache/clip_b32_coco \
        --batch-size 2048 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore  # noqa: E402
from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore  # noqa: E402
from synthetic_theorem2_method import _compute_latent_correlation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True, help="dir containing saved TwoSidedTopKSAE")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--force-recompute", action="store_true",
                   help="ignore cached C / cofire and recompute from scratch")
    return p.parse_args()


def stack_tensors(ds: CachedClipPairsDataset) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info("Stacking %d pairs to tensors", len(ds))
    img = torch.stack([ds[i]["image_embeds"] for i in range(len(ds))], dim=0)
    txt = torch.stack([ds[i]["text_embeds"] for i in range(len(ds))], dim=0)
    return img, txt


def cofiring_counts(
    sae_i, sae_t, img: torch.Tensor, txt: torch.Tensor, device: torch.device, batch_size: int,
) -> np.ndarray:
    """Diagonal co-firing counts per-latent index at given matching."""
    n = int(sae_i.latent_size)
    acts_i_total = np.zeros(n, dtype=np.float64)
    acts_t_total = np.zeros(n, dtype=np.float64)
    sae_i.eval()
    sae_t.eval()
    with torch.no_grad():
        for s in range(0, img.shape[0], batch_size):
            hs_i = img[s:s + batch_size].to(device).unsqueeze(1)
            hs_t = txt[s:s + batch_size].to(device).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i, return_dense_latents=True)
            out_t = sae_t(hidden_states=hs_t, return_dense_latents=True)
            zi = out_i.dense_latents.squeeze(1).float().cpu().numpy()
            zt = out_t.dense_latents.squeeze(1).float().cpu().numpy()
            acts_i_total += (zi > 0).sum(axis=0)
            acts_t_total += (zt > 0).sum(axis=0)
    return np.minimum(acts_i_total, acts_t_total)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    logger.info("Loading TwoSidedTopKSAE from %s", run_dir)
    model = TwoSidedTopKSAE.from_pretrained(str(run_dir)).to(device)
    model.eval()
    sae_i = model.image_sae
    sae_t = model.text_sae
    logger.info("image_sae latent=%d, text_sae latent=%d", sae_i.latent_size, sae_t.latent_size)

    # 1) Correlation matrix C (n, n) — cached to disk so threshold tweaks are free
    c_cache_path = run_dir / f"diagnostic_B_C_{args.split}.npy"
    cofire_cache_path = run_dir / f"diagnostic_B_cofire_{args.split}.npy"
    need_compute = args.force_recompute or not (c_cache_path.exists() and cofire_cache_path.exists())
    if need_compute:
        logger.info("Loading cached pairs from %s split=%s", args.cache_dir, args.split)
        ds = CachedClipPairsDataset(args.cache_dir, split=args.split, l2_normalize=True)
        img, txt = stack_tensors(ds)
        logger.info("Stacked pairs: %s, %s", img.shape, txt.shape)
    else:
        logger.info("Using cached C + cofire — skipping embedding load")
        img = txt = None

    if c_cache_path.exists() and not args.force_recompute:
        logger.info("Loading cached C from %s", c_cache_path)
        C = np.load(c_cache_path)
        logger.info("C shape=%s, max|C|=%.4f", C.shape, float(np.abs(C).max()))
    else:
        logger.info("Computing correlation matrix (streaming)...")
        C = _compute_latent_correlation(sae_i, sae_t, img, txt, args.batch_size, device)
        logger.info("C shape=%s, max|C|=%.4f", C.shape, float(np.abs(C).max()))
        np.save(c_cache_path, C)
        logger.info("Saved C to %s", c_cache_path)

    # 2) Hungarian on -C (signed; maximize positive correlation sum)
    logger.info("Hungarian assignment (signed -C)...")
    row_ind, col_ind = linear_sum_assignment(-C)

    # 3) Decoder cosine per matched pair
    W_i = sae_i.W_dec.detach().cpu().float().numpy()  # (latent, hidden)
    W_t = sae_t.W_dec.detach().cpu().float().numpy()

    def row_cosine(u, v):
        nu = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        nv = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return ((u / nu) * (v / nv)).sum(axis=1)

    rho = row_cosine(W_i[row_ind], W_t[col_ind])  # (n,)

    # 4) Co-firing threshold sweep — also cached
    if cofire_cache_path.exists() and not args.force_recompute:
        logger.info("Loading cached cofire from %s", cofire_cache_path)
        cofire = np.load(cofire_cache_path)
    else:
        logger.info("Computing co-firing counts for threshold sweep...")
        cofire = cofiring_counts(sae_i, sae_t, img, txt, device, args.batch_size)
        np.save(cofire_cache_path, cofire)
        logger.info("Saved cofire to %s", cofire_cache_path)
    matched_cofire = cofire[row_ind]  # use image-side firing for pair rank

    results = {
        "all": {
            "n_pairs": int(len(rho)),
            "median": float(np.median(rho)),
            "mean": float(np.mean(rho)),
            "std": float(np.std(rho)),
        },
        "threshold_sweep": {},
    }
    for t in args.thresholds:
        keep_k = max(1, int(round(len(rho) * t)))
        top_ix = np.argsort(-matched_cofire)[:keep_k]
        sub = rho[top_ix]
        results["threshold_sweep"][f"top{int(t * 100)}pct"] = {
            "n_pairs": int(keep_k),
            "median": float(np.median(sub)),
            "mean": float(np.mean(sub)),
            "std": float(np.std(sub)),
        }

    out_path = run_dir / "diagnostic_B.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)
    logger.info("results: %s", results)

    # Histogram plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rho, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
        median_val = float(np.median(rho))
        ax.axvline(median_val, color="red", linestyle="--",
                   label=fr"median = {median_val:.3f}")
        ax.set_xlabel(r"Matched-pair decoder cosine $\rho_k$")
        ax.set_ylabel("Count")
        ax.set_title(r"Diagnostic B: $\hat{\alpha}$ distribution (Hungarian-matched decoder columns)")
        ax.legend()
        ax.set_xlim(-0.2, 1.05)
        fig.tight_layout()
        fig_path = run_dir / "fig_diagnostic_B_histogram.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", fig_path)
    except Exception as e:
        logger.warning("Histogram plot failed: %s", e)


if __name__ == "__main__":
    main()
