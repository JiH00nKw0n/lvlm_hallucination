"""ImageNet monosemanticity analysis per SAE method.

For each method, on ImageNet val (50k images, 1000 classes, 50/class):
  A) MS (MonoSemanticity score, Pach et al. 2025, Eq. 9) per alive slot
      MS^k = (sum_{n<m} r^k_nm · s_nm) / (sum_{n<m} r^k_nm)
      with r^k_nm = ã^k_n · ã^k_m (normalized activation product)
           s_nm  = cos(E(x_n), E(x_m))  (we use CLIP image embedding directly)
  B) Purity (per alive slot, slot→class): majority_class_count / total
  C) Class → Latent spread (per class): # distinct top-1 slots samples of that class land on

A dominant-slot mask (threshold=0.1) is applied before top-1 so common modality
axes don't trivially inflate/deflate metrics (matching the valprobe protocol).

Outputs JSON with:
  - alive_slots
  - ms: {mean, median, std, distribution buckets, hist values}
  - purity: {mean, median, hist, dist buckets}
  - class_spread: {mean, median, hist, dist buckets (# slots per class)}

Usage:
    python scripts/real_alpha/eval_monosemanticity.py \
        --ckpt outputs/real_exp_v1/<method>/imagenet/final \
        --method <shared|separated|aux|ours|vl_sae> \
        --cache-dir cache/clip_b32_imagenet \
        --output outputs/real_exp_v1/<method>/imagenet/mono.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np
import torch

import eval_utils  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--ms-sample-n", type=int, default=5000,
                   help="N images to subsample for MS pairwise computation")
    p.add_argument("--ms-min-fires", type=int, default=10,
                   help="Min # samples with nonzero activation in the MS subsample "
                        "for a slot to be included in MS summary (filters dead/near-dead slots).")
    p.add_argument("--dominant-threshold", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _pick_image_sae(model, method: str):
    if method in ("shared", "aux", "vl_sae"):
        return model
    return model.image_sae


@torch.no_grad()
def _encode_dense(sae, embeds, batch_size, device, method):
    sae.eval(); sae.to(device)
    if method == "vl_sae":
        L = int(sae.cfg.latent_size)
    else:
        L = int(sae.latent_size)
    N = embeds.shape[0]
    out = torch.empty(N, L, dtype=torch.float32)
    for s in range(0, N, batch_size):
        chunk = embeds[s:s + batch_size].to(device)
        if method == "vl_sae":
            z = sae.encode(chunk)
        else:
            z = sae(hidden_states=chunk.unsqueeze(1), return_dense_latents=True).dense_latents.squeeze(1)
        out[s:s + chunk.shape[0]] = z.float().cpu()
    return out


def find_dominant(z: torch.Tensor, threshold: float):
    if threshold <= 0:
        return []
    N, L = z.shape
    masked = []
    zc = z.clone()
    while True:
        top1 = zc.argmax(dim=-1)
        counts = torch.bincount(top1, minlength=L).cpu().numpy()
        s = int(counts.argmax())
        frac = counts[s] / N
        if frac < threshold:
            break
        masked.append(s)
        zc[:, s] = -float("inf")
    return masked


def compute_ms(A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Vectorized MS score for each slot.

    A: (L, N) normalized activations (one row per slot).
    S: (N, N) pairwise similarity of images (diagonal == 1).

    MS_k = [ã^T S ã − ||ã||²] / [(1^T ã)² − ||ã||²]
         = (off-diagonal activation-weighted similarity) / (off-diag activation mass)
    """
    AS = A @ S  # (L, N)
    diag_of_ASA = (AS * A).sum(dim=1)  # ã^T S ã per slot
    norm2 = (A * A).sum(dim=1)
    sum1 = A.sum(dim=1)
    numer = diag_of_ASA - norm2
    denom = sum1 * sum1 - norm2
    return numer / denom.clamp_min(1e-12)


def histogram_summary(values: np.ndarray, bins: list[float]) -> dict:
    """Return mean/median/std/hist-counts/hist-edges."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"n": 0}
    counts, edges = np.histogram(values, bins=bins)
    out = {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "bins": [float(x) for x in bins],
        "counts": [int(c) for c in counts],
    }
    return out


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = _Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("load ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)
    sae = _pick_image_sae(model, args.method)

    logger.info("load ImageNet val")
    ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")
    N = len(ds)
    img = torch.stack([ds[i]["image_embeds"] for i in range(N)], dim=0)
    y = np.array([int(ds.pairs[i][1]) for i in range(N)], dtype=np.int64)
    # CLIP image embeddings are already L2-normalized by the dataset loader.

    logger.info("encode val → dense latents")
    z_full = _encode_dense(sae, img, args.batch_size, device, args.method)

    # Dominant-slot masking (like valprobe)
    masked = find_dominant(z_full, args.dominant_threshold)
    logger.info("masked dominant slots (>= %.0f%%): %s", 100 * args.dominant_threshold, masked)
    z_m = z_full.clone()
    for s in masked:
        z_m[:, s] = 0
    top1 = z_m.argmax(dim=-1).cpu().numpy()
    L = z_m.shape[1]

    # --- PURITY (per alive slot) ---
    slot_class_count: dict[int, dict[int, int]] = {}
    for s_, c in zip(top1, y):
        d = slot_class_count.setdefault(int(s_), {})
        d[int(c)] = d.get(int(c), 0) + 1
    purities = []
    primary_classes = []
    slots_with_samples = []
    for slot, cdict in slot_class_count.items():
        total = sum(cdict.values())
        primary = max(cdict.values())
        purities.append(primary / total)
        primary_classes.append(max(cdict.items(), key=lambda kv: kv[1])[0])
        slots_with_samples.append(slot)
    purities = np.asarray(purities)
    logger.info("alive slots (have samples): %d", len(purities))

    purity_summary = histogram_summary(purities, bins=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])

    # --- MS score (per alive slot firing anywhere, not just as top-1) ---
    # For MS we use all slots where activation != 0 for at least one sample.
    # activation_alive: bool mask over L slots
    active_any = (z_m != 0).any(dim=0)
    alive_slot_idx = torch.where(active_any)[0].cpu().numpy()
    logger.info("alive (any-nonzero) slots: %d", len(alive_slot_idx))

    # Subsample N' images for MS
    ms_N = min(args.ms_sample_n, N)
    ms_idx = rng.choice(N, ms_N, replace=False)
    ms_idx = np.sort(ms_idx)
    img_sub = img[ms_idx].to(device)  # already L2-normalized
    logger.info("MS similarity matrix %dx%d on GPU", ms_N, ms_N)
    S = img_sub @ img_sub.T

    A_full = z_m[ms_idx][:, alive_slot_idx].cpu()  # (ms_N, L_alive)
    fires_per_slot = (A_full != 0).sum(dim=0).numpy()  # (L_alive,) int
    keep_mask = fires_per_slot >= args.ms_min_fires
    kept_slot_idx = alive_slot_idx[keep_mask]
    A_full = A_full[:, keep_mask]
    logger.info(
        "MS filter: %d/%d slots have >= %d activations in the MS subsample",
        int(keep_mask.sum()), len(alive_slot_idx), args.ms_min_fires,
    )

    # Min-max normalize per slot across the subsample
    a_min = A_full.min(dim=0).values
    a_max = A_full.max(dim=0).values
    a_range = (a_max - a_min).clamp_min(1e-12)
    A_norm = ((A_full - a_min[None, :]) / a_range[None, :]).float()
    A_T = A_norm.T.to(device)  # (L_kept, ms_N)
    logger.info("compute MS on %d well-fired slots", A_T.shape[0])
    ms = compute_ms(A_T, S.float().to(device)).cpu().numpy()
    ms = ms[np.isfinite(ms)]
    # Also drop extreme tails that can appear from ill-conditioned denominators
    ms_clipped = ms[(ms >= -1.0) & (ms <= 1.0)]
    ms_summary = histogram_summary(ms_clipped, bins=[-1, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.01])
    ms_summary["n_slots_evaluated"] = int(len(ms_clipped))
    ms_summary["n_slots_dropped_out_of_range"] = int(len(ms) - len(ms_clipped))
    ms_summary["min_fires_filter"] = int(args.ms_min_fires)

    # --- Class spread (# distinct top-1 slots per class) ---
    class_to_slots: dict[int, set] = {}
    for s_, c in zip(top1, y):
        class_to_slots.setdefault(int(c), set()).add(int(s_))
    spreads = np.array([len(v) for v in class_to_slots.values()], dtype=np.int64)
    spread_summary = histogram_summary(spreads.astype(np.float64), bins=[1, 2, 5, 10, 20, 50, 100, 1000])

    result = {
        "method": args.method,
        "ckpt": args.ckpt,
        "dominant_threshold": args.dominant_threshold,
        "masked_slots": masked,
        "n_alive_slots_firing_top1": int(len(purities)),
        "n_alive_slots_any_activation": int(len(alive_slot_idx)),
        "latent_size": L,
        "n_classes_evaluated": int(len(class_to_slots)),
        "purity": purity_summary,
        "ms": ms_summary,
        "class_spread": spread_summary,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)
    logger.info("  purity  mean=%.3f, median=%.3f", purity_summary["mean"], purity_summary["median"])
    logger.info("  MS      mean=%.3f, median=%.3f", ms_summary["mean"], ms_summary["median"])
    logger.info("  spread  mean=%.1f, median=%.1f", spread_summary["mean"], spread_summary["median"])


if __name__ == "__main__":
    main()
