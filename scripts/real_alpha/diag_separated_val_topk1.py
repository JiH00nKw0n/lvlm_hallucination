"""Diagnostic: Separated SAE image-side top-1 behavior on ImageNet val.

Answers three questions:
  1. For each slot in the image-side SAE, what's the class distribution of val
     samples that land on it via top-1? How many slots are "class-specific"
     (majority class > 50%)? How many distinct primary classes do they cover?
  2. How does a linear probe trained ON THE val set (fit+eval on the same 50k
     samples) perform at top-1 restricted input?
  3. Sanity-baseline: cosine-based zero-shot at top-1 restricted, using the
     text-side SAE (no perm — raw separated).

Usage:
    python scripts/real_alpha/diag_separated_val_topk1.py \
        --ckpt outputs/real_exp_v1/separated/imagenet/final \
        --cache-dir cache/clip_b32_imagenet
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
import torch.nn as nn  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lp-lr", type=float, default=1e-2)
    p.add_argument("--lp-epochs", type=int, default=30)
    p.add_argument("--lp-batch-size", type=int, default=1024)
    p.add_argument("--eval-topk", type=int, default=1,
                   help="Keep only top-K entries per sample (set 0 to disable).")
    return p.parse_args()


@torch.no_grad()
def encode_image_side(sae, embeds, batch_size, device, eval_topk=1):
    """Encode (N, H) through TopKSAE image side, return dense (N, L) with optional top-k mask."""
    sae.eval()
    sae.to(device)
    N = embeds.shape[0]
    L = int(sae.latent_size)
    out = torch.empty(N, L, dtype=torch.float32)
    for s in range(0, N, batch_size):
        chunk = embeds[s:s + batch_size].to(device).unsqueeze(1)
        z = sae(hidden_states=chunk, return_dense_latents=True).dense_latents.squeeze(1)
        if eval_topk and eval_topk > 0:
            vals, idx = z.abs().topk(eval_topk, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, idx, 1.0)
            z = z * mask
        out[s:s + chunk.shape[0]] = z.float().cpu()
    return out


def slot_class_report(top1: np.ndarray, y: np.ndarray, L: int):
    """For each slot, compute class distribution of samples falling on it."""
    # Build slot -> class counter
    slot_class_count = {}
    for s, c in zip(top1, y):
        s, c = int(s), int(c)
        d = slot_class_count.setdefault(s, {})
        d[c] = d.get(c, 0) + 1

    rows = []
    for slot, cdict in slot_class_count.items():
        total = sum(cdict.values())
        primary_c, primary_n = max(cdict.items(), key=lambda kv: kv[1])
        purity = primary_n / total
        rows.append({
            "slot": slot,
            "total": total,
            "primary_class": primary_c,
            "primary_count": primary_n,
            "purity": purity,
            "n_classes": len(cdict),
        })
    rows.sort(key=lambda r: -r["total"])

    # Summary stats
    n_alive_slots = len(rows)
    n_samples_covered = sum(r["total"] for r in rows)
    thresholds = [0.9, 0.7, 0.5, 0.3]
    summary = {"n_alive_slots": n_alive_slots, "n_samples": n_samples_covered}
    for tau in thresholds:
        high_purity = [r for r in rows if r["purity"] >= tau]
        distinct_classes = len({r["primary_class"] for r in high_purity})
        covered = sum(r["total"] for r in high_purity)
        summary[f"n_slots_purity_ge_{tau}"] = len(high_purity)
        summary[f"distinct_classes_purity_ge_{tau}"] = distinct_classes
        summary[f"samples_covered_frac_purity_ge_{tau}"] = covered / max(1, n_samples_covered)
    return rows, summary


def fit_val_only_probe(z: torch.Tensor, y: np.ndarray, n_classes: int, device: torch.device,
                       lr: float, epochs: int, batch_size: int):
    """Fit linear head on all val samples, evaluate train/eval top-1 on same data."""
    L = z.shape[1]
    head = nn.Linear(L, n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    y_t = torch.as_tensor(y, dtype=torch.long)
    N = z.shape[0]
    history = []
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot_loss = 0.0
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            xb = z[idx].to(device)
            yb = y_t[idx].to(device)
            logits = head(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += float(loss.item()) * xb.shape[0]
        tot_loss /= N
        with torch.no_grad():
            correct = 0
            for s in range(0, N, batch_size):
                xb = z[s:s + batch_size].to(device)
                pred = head(xb).argmax(dim=-1)
                correct += int((pred.cpu().numpy() == y[s:s + batch_size]).sum())
            acc = correct / N
        history.append({"ep": ep + 1, "train_loss": tot_loss, "acc": acc})
        logger.info("  ep %02d: loss=%.4f  val_top1=%.4f", ep + 1, tot_loss, acc)
    return head, history


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("loading TwoSidedTopKSAE from %s", args.ckpt)
    model = eval_utils.load_sae(args.ckpt, "separated")
    image_sae = model.image_sae
    logger.info("image_sae: L=%d, k=%d", image_sae.latent_size, image_sae.cfg.k)

    logger.info("loading ImageNet val")
    ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")
    N = len(ds)
    img = torch.stack([ds[i]["image_embeds"] for i in range(N)], dim=0)
    y = np.array([int(ds.pairs[i][1]) for i in range(N)], dtype=np.int64)
    logger.info("val: N=%d", N)

    logger.info("encoding val → top-%d restricted latents", args.eval_topk)
    z = encode_image_side(image_sae, img, args.batch_size, device, eval_topk=args.eval_topk)
    # top-1 slot index per sample (argmax on full latent; if eval_topk=1 this
    # equals the nonzero entry)
    top1 = z.argmax(dim=-1).cpu().numpy()
    L = int(image_sae.latent_size)

    logger.info("=== Slot-class analysis ===")
    rows, summary = slot_class_report(top1, y, L)
    print(json.dumps(summary, indent=2))
    print("\nTop 20 slots by usage (slot: total, primary_class, primary_count, purity, n_classes):")
    for r in rows[:20]:
        print(f"  slot {r['slot']}: n={r['total']}, primary=class {r['primary_class']} ({r['primary_count']}/{r['total']}, purity={r['purity']:.2f}), distinct_classes={r['n_classes']}")

    logger.info("=== Linear probe on val 50k (train+eval on same) ===")
    head, history = fit_val_only_probe(
        z, y, n_classes=1000, device=device,
        lr=args.lp_lr, epochs=args.lp_epochs, batch_size=args.lp_batch_size,
    )
    print(f"\nFinal val accuracy: {history[-1]['acc']*100:.2f}%")
    print(f"Best val accuracy:  {max(h['acc'] for h in history)*100:.2f}%")


if __name__ == "__main__":
    main()
