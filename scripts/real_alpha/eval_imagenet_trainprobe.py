"""ImageNet linear probe with train-set fit + val-set eval, modality-slot masked.

Like `eval_imagenet_valprobe.py` (which fits and evaluates on val 50k for a
feature-separability diagnostic), but uses the full train set (balanced via
max_per_class) to fit the probe and val 50k to evaluate — this is the proper
generalization test.

Pipeline:
  1. Encode val images through the image-side SAE to identify dominant slots
     (slots that are top-1 for ≥ threshold of val samples).
  2. Stream-encode train images, masking dominant slots, keep only top-1.
  3. Fit linear probe on train (Adam, streaming batches).
  4. Evaluate on val (same masking & top-1).

Usage:
    python scripts/real_alpha/eval_imagenet_trainprobe.py \
        --ckpt outputs/real_exp_v1/separated/imagenet/final \
        --method separated --cache-dir cache/clip_b32_imagenet \
        --output outputs/real_exp_v1/separated/imagenet/trainprobe.json \
        --dominant-threshold 0.5
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
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dominant-threshold", type=float, default=0.5)
    p.add_argument("--max-per-class", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lp-lr", type=float, default=1e-2)
    p.add_argument("--lp-epochs", type=int, default=30)
    p.add_argument("--lp-batch-size", type=int, default=1024)
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _pick_image_sae(model, method: str):
    if method in ("shared", "aux"):
        return model
    if method == "vl_sae":
        return model
    return model.image_sae


@torch.no_grad()
def _encode_dense_full(sae, embeds: torch.Tensor, batch_size: int, device: torch.device,
                       method: str) -> torch.Tensor:
    """Full dense top-k latent (no slot masking)."""
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


def find_dominant_slots(z: torch.Tensor, threshold: float) -> list[int]:
    if threshold <= 0:
        return []
    N, L = z.shape
    masked: list[int] = []
    z_curr = z.clone()
    while True:
        top1 = z_curr.argmax(dim=-1)
        counts = torch.bincount(top1, minlength=L).cpu().numpy()
        top_slot = int(counts.argmax())
        frac = counts[top_slot] / N
        if frac < threshold:
            break
        masked.append(top_slot)
        z_curr[:, top_slot] = -float("inf")
    return masked


def _mask_and_top1(z: torch.Tensor, masked: list[int]) -> torch.Tensor:
    z_m = z.clone()
    for s in masked:
        z_m[:, s] = 0
    top1 = z_m.argmax(dim=-1)
    out = torch.zeros_like(z_m)
    out.scatter_(-1, top1.unsqueeze(-1), z_m.gather(-1, top1.unsqueeze(-1)))
    return out


@torch.no_grad()
def _encode_batch_masked_top1(
    sae, embeds_batch: torch.Tensor, masked_idx: torch.Tensor,
    device: torch.device, method: str,
) -> torch.Tensor:
    """Encode a single batch, apply mask + top-1, return GPU tensor (B, L).

    No intermediate (N, L) CPU materialization — intended to be called per-batch
    from the probe training loop so memory stays bounded to `batch × L`.
    """
    chunk = embeds_batch.to(device, non_blocking=True)
    if method == "vl_sae":
        z = sae.encode(chunk)
    else:
        z = sae(hidden_states=chunk.unsqueeze(1), return_dense_latents=True).dense_latents.squeeze(1)
    if masked_idx.numel() > 0:
        z[:, masked_idx] = 0
    top1 = z.argmax(dim=-1)
    z_t1 = torch.zeros_like(z)
    z_t1.scatter_(-1, top1.unsqueeze(-1), z.gather(-1, top1.unsqueeze(-1)))
    return z_t1


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = _Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)
    image_sae = _pick_image_sae(model, args.method)
    logger.info("loading val for dominant-slot detection")
    val_ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")
    N_val = len(val_ds)
    img_val = torch.stack([val_ds[i]["image_embeds"] for i in range(N_val)], dim=0)
    y_val = np.array([int(val_ds.pairs[i][1]) for i in range(N_val)], dtype=np.int64)

    logger.info("encoding val → dense to find dominant slots")
    z_val_full = _encode_dense_full(image_sae, img_val, args.batch_size, device, args.method)
    masked = find_dominant_slots(z_val_full, args.dominant_threshold)
    logger.info("dominant slots (≥%.0f%%): %s", 100 * args.dominant_threshold, masked)

    # Free the full val encoding; we'll re-encode per-batch via the streaming path.
    del z_val_full

    # Load train (balanced). Keep only RAW embeddings in CPU (N × 512 × 4 bytes).
    logger.info("loading train (max_per_class=%d)", args.max_per_class)
    tr_ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "train",
                                         max_per_class=args.max_per_class)
    N_tr = len(tr_ds)
    img_tr = torch.stack([tr_ds[i]["image_embeds"] for i in range(N_tr)], dim=0)
    y_tr = np.array([int(tr_ds.pairs[i][1]) for i in range(N_tr)], dtype=np.int64)
    logger.info("train N=%d (streaming SAE encode per batch — no (N, L) materialization)", N_tr)

    # SAE stays on GPU; encode+mask+top-1 happens inside training batches.
    image_sae.eval()
    image_sae.to(device)
    if args.method == "vl_sae":
        L = int(image_sae.cfg.latent_size)
    else:
        L = int(image_sae.latent_size)
    masked_idx = torch.as_tensor(masked, dtype=torch.long, device=device)

    logger.info("fitting probe (lr=%.3g, epochs=%d, batch=%d, L=%d)",
                args.lp_lr, args.lp_epochs, args.lp_batch_size, L)
    head = nn.Linear(L, args.n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lp_lr)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long)
    history = []
    for ep in range(args.lp_epochs):
        perm = torch.randperm(N_tr)
        tot = 0.0
        for s in range(0, N_tr, args.lp_batch_size):
            idx = perm[s:s + args.lp_batch_size]
            xb = _encode_batch_masked_top1(image_sae, img_tr[idx], masked_idx, device, args.method)
            yb = y_tr_t[idx].to(device, non_blocking=True)
            logits = head(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * xb.shape[0]
        tot /= N_tr
        with torch.no_grad():
            correct = 0
            for s in range(0, N_val, args.lp_batch_size):
                xb = _encode_batch_masked_top1(image_sae, img_val[s:s + args.lp_batch_size], masked_idx, device, args.method)
                pred = head(xb).argmax(-1)
                correct += int((pred == y_val_t[s:s + args.lp_batch_size].to(device)).sum().item())
            acc = correct / N_val
        history.append({"ep": ep + 1, "train_loss": tot, "val_acc": acc})
        logger.info("  ep %02d: loss=%.4f  val_top1=%.4f", ep + 1, tot, acc)
    best = max(h["val_acc"] for h in history)
    final = history[-1]["val_acc"]

    result = {
        "method": args.method,
        "ckpt": args.ckpt,
        "dominant_threshold": args.dominant_threshold,
        "masked_slots": masked,
        "n_masked": len(masked),
        "linprobe_best_acc": best,
        "linprobe_final_acc": final,
        "linprobe_history": history,
        "n_train": N_tr,
        "n_val": N_val,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s  best=%.4f final=%.4f", out_path, best, final)


if __name__ == "__main__":
    main()
