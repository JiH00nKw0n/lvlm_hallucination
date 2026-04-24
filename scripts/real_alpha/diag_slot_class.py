"""Per-slot class-specificity analysis on ImageNet val.

For a chosen SAE (default: Separated image-side), encode val images under
variants (centered/raw × top-1/top-k=full) and for EACH slot compute:

  - n_active: # val samples where this slot is non-zero (in any of top-k)
  - primary_class, primary_count, purity: dominant class freq
  - n_classes: # distinct classes with samples on this slot

Then summarize:
  - # slots "class-specific" at purity ≥ τ ∈ {0.9, 0.7, 0.5, 0.3}
  - # distinct primary classes among those slots
  - # val samples covered by those slots
  - coverage ratio out of total activation mass (sum over top-k)

Usage:
    python scripts/real_alpha/diag_slot_class.py \
        --ckpt outputs/real_exp_v1/separated/imagenet/final \
        --cache-dir cache/clip_b32_imagenet \
        --side image --method separated
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "vl_sae"], required=True)
    p.add_argument("--side", choices=["image", "text"], default="image")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def _clip_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.sum(torch.pow(tensor, 2), dim=-1, keepdim=True), 0.5)


def _pick_sae(model, method: str, side: str):
    if method in ("shared", "aux"):
        return model
    if method == "vl_sae":
        return model
    return model.image_sae if side == "image" else model.text_sae


@torch.no_grad()
def _encode_dense(sae, embeds: torch.Tensor, batch_size: int, device: torch.device,
                  method: str, top_k_mask: int | None = None) -> torch.Tensor:
    """Dense (N, L) encoding; if top_k_mask set, zero out all but top-k per row."""
    sae.eval(); sae.to(device)
    N = embeds.shape[0]; L = int(sae.latent_size)
    out = torch.empty(N, L, dtype=torch.float32)
    for s in range(0, N, batch_size):
        chunk = embeds[s:s + batch_size].to(device)
        if method == "vl_sae":
            z = sae.encode(chunk)
        else:
            z = sae(hidden_states=chunk.unsqueeze(1), return_dense_latents=True).dense_latents.squeeze(1)
        if top_k_mask is not None and top_k_mask > 0:
            _, idx = z.abs().topk(top_k_mask, dim=-1)
            mask = torch.zeros_like(z); mask.scatter_(-1, idx, 1.0)
            z = z * mask
        out[s:s + chunk.shape[0]] = z.float().cpu()
    return out


def slot_class_report(z: torch.Tensor, y: np.ndarray, label: str) -> dict:
    """For each slot, collect class distribution of samples where slot != 0."""
    # active[s] = indices of samples where slot s fires
    active = (z != 0).cpu().numpy().astype(bool)  # (N, L)
    N, L = active.shape
    total_activations = int(active.sum())  # sum of top-k markers across all samples

    rows = []
    per_slot_primary = {}
    for s in range(L):
        mask = active[:, s]
        n_s = int(mask.sum())
        if n_s == 0:
            continue
        classes = y[mask]
        uniq, cnt = np.unique(classes, return_counts=True)
        i_max = cnt.argmax()
        primary_c = int(uniq[i_max]); primary_n = int(cnt[i_max])
        rows.append({
            "slot": s, "n_active": n_s,
            "primary_class": primary_c, "primary_count": primary_n,
            "purity": primary_n / n_s, "n_classes": len(uniq),
        })
        per_slot_primary[s] = (primary_c, primary_n / n_s)
    rows.sort(key=lambda r: -r["n_active"])

    summary = {
        "label": label,
        "n_alive_slots": len(rows),
        "total_activations": total_activations,
    }
    for tau in [0.9, 0.7, 0.5, 0.3]:
        kept = [r for r in rows if r["purity"] >= tau]
        classes_covered = {r["primary_class"] for r in kept}
        activations = sum(r["n_active"] for r in kept)
        summary[f"n_slots_purity_ge_{tau}"] = len(kept)
        summary[f"distinct_primary_classes_purity_ge_{tau}"] = len(classes_covered)
        summary[f"activation_fraction_purity_ge_{tau}"] = activations / max(1, total_activations)
    return {"summary": summary, "top_slots": rows[:30]}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("loading ckpt=%s method=%s side=%s", args.ckpt, args.method, args.side)
    model = eval_utils.load_sae(args.ckpt, args.method)
    sae = _pick_sae(model, args.method, args.side)
    logger.info("SAE L=%d, k=%d", sae.latent_size if args.method != "vl_sae" else sae.cfg.latent_size,
                getattr(sae.cfg, "k", "?"))

    logger.info("loading ImageNet val")
    ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")
    N = len(ds)
    img = torch.stack([ds[i]["image_embeds"] for i in range(N)], dim=0)
    y = np.array([int(ds.pairs[i][1]) for i in range(N)], dtype=np.int64)
    logger.info("val: N=%d, embed dim=%d", N, img.shape[1])

    # Variant 1: raw (already L2-normalized) embeddings
    logger.info("Encoding RAW embeddings → top-8 dense (train-time k)")
    z_raw_k = _encode_dense(sae, img, args.batch_size, device, args.method, top_k_mask=None)
    logger.info("Encoding RAW → top-1")
    z_raw_1 = _encode_dense(sae, img, args.batch_size, device, args.method, top_k_mask=1)

    # Variant 2: mean-centered then re-normalized embeddings
    mean = img.mean(dim=0, keepdim=True)
    img_centered = img - mean
    img_centered = img_centered / _clip_vector_norm(img_centered)
    logger.info("Encoding CENTERED embeddings → top-8 dense")
    z_c_k = _encode_dense(sae, img_centered, args.batch_size, device, args.method, top_k_mask=None)
    logger.info("Encoding CENTERED → top-1")
    z_c_1 = _encode_dense(sae, img_centered, args.batch_size, device, args.method, top_k_mask=1)

    results = {
        "raw_top1": slot_class_report(z_raw_1, y, "raw_top1"),
        "raw_topk": slot_class_report(z_raw_k, y, "raw_topk"),
        "centered_top1": slot_class_report(z_c_1, y, "centered_top1"),
        "centered_topk": slot_class_report(z_c_k, y, "centered_topk"),
    }

    print(json.dumps({k: v["summary"] for k, v in results.items()}, indent=2))

    print("\n=== raw top-1: top 20 slots by usage ===")
    for r in results["raw_top1"]["top_slots"][:20]:
        print(f"  slot {r['slot']}: n={r['n_active']}, primary=class {r['primary_class']} "
              f"({r['primary_count']}/{r['n_active']}, purity={r['purity']:.2f}), n_classes={r['n_classes']}")

    print("\n=== centered top-1: top 20 slots by usage ===")
    for r in results["centered_top1"]["top_slots"][:20]:
        print(f"  slot {r['slot']}: n={r['n_active']}, primary=class {r['primary_class']} "
              f"({r['primary_count']}/{r['n_active']}, purity={r['purity']:.2f}), n_classes={r['n_classes']}")

    print("\n=== centered top-k: top 20 slots by usage (train-time k=8) ===")
    for r in results["centered_topk"]["top_slots"][:20]:
        print(f"  slot {r['slot']}: n={r['n_active']}, primary=class {r['primary_class']} "
              f"({r['primary_count']}/{r['n_active']}, purity={r['purity']:.2f}), n_classes={r['n_classes']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
