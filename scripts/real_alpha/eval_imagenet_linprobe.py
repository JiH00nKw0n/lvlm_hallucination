"""ImageNet-1K linear probe on SAE image-side latents.

Encodes each train (balanced, max_per_class) and val image through the
image-side SAE, then trains a linear classifier (softmax + cross-entropy)
via mini-batch Adam on GPU. Reports val top-1 accuracy.

Supports `--eval-topk` to emulate smaller inference-time sparsity than
training-time k (e.g., train k=8, linear probe on top-1 only).

Uses only the image-side SAE — alignment is irrelevant here.

Usage:
    python scripts/real_alpha/eval_imagenet_linprobe.py \
        --ckpt outputs/real_exp_v1/shared/imagenet/final \
        --method shared --cache-dir cache/clip_b32_imagenet \
        --output outputs/real_exp_v1/shared/imagenet/linprobe.json \
        --max-per-class 1000 --eval-topk 1
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
import torch.nn as nn  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-per-class", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    # inference-time sparsity override (emulates smaller k at eval)
    p.add_argument("--eval-topk", type=int, default=None,
                   help="If set, keep only top-k entries per sample of the SAE "
                        "output at eval time. Default: use all active slots.")
    # torch linear probe training params
    # NOTE: default lr raised to 1e-2 and batch dropped to 1024 because the
    # original (1e-3, 8192) kept train_loss near log(1000) = 6.9 for 30 epochs.
    p.add_argument("--lp-lr", type=float, default=1e-2)
    p.add_argument("--lp-weight-decay", type=float, default=0.0)
    p.add_argument("--lp-epochs", type=int, default=30)
    p.add_argument("--lp-batch-size", type=int, default=1024)
    return p.parse_args()


def _select_image_sae(model, method: str):
    if method in ("shared", "aux", "vl_sae"):
        return model
    return model.image_sae


@torch.no_grad()
def _encode_batch(
    sae, embeds_batch: torch.Tensor, device: torch.device,
    method: str, eval_topk: int | None = None,
) -> torch.Tensor:
    """Encode a single batch to dense (B, L) latents on GPU.

    Called on-the-fly in the training loop to avoid materializing the full
    (N, L) tensor on CPU (which would be 24 GB for N=732k, L=8192).
    """
    chunk = embeds_batch.to(device, non_blocking=True)
    if method == "vl_sae":
        z = sae.encode(chunk)
    else:
        z = sae(hidden_states=chunk.unsqueeze(1), return_dense_latents=True) \
            .dense_latents.squeeze(1)
    if eval_topk is not None and eval_topk > 0:
        topk_vals, topk_idx = z.abs().topk(eval_topk, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_idx, 1.0)
        z = z * mask
    return z


def _collect(ds):
    img = torch.stack([ds[i]["image_embeds"] for i in range(len(ds))], dim=0)
    y = np.array([int(ds.pairs[i][1]) for i in range(len(ds))], dtype=np.int64)
    return img, y


def _train_linprobe_streaming(
    sae, img_tr: torch.Tensor, y_tr: np.ndarray,
    img_va: torch.Tensor, y_va: np.ndarray,
    n_classes: int, L: int, device: torch.device,
    method: str, eval_topk: int | None,
    lr: float, weight_decay: float, epochs: int, batch_size: int,
) -> tuple[float, float]:
    """Linear probe with on-the-fly per-batch SAE encoding.

    Keeps only raw CLIP embeddings (N, 512) in CPU memory — re-encodes each
    batch through the SAE on GPU inside the training loop. This avoids the
    (N, L) dense tensor that would be 24 GB for L=8192 and N=732k.
    """
    head = nn.Linear(L, n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long)
    y_va_t = torch.as_tensor(y_va, dtype=torch.long)
    N = img_tr.shape[0]
    best_val = 0.0
    final_val = 0.0
    for ep in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            # SAE encode is under @torch.no_grad — SAE params are frozen.
            z = _encode_batch(sae, img_tr[idx], device, method, eval_topk)
            yb = y_tr_t[idx].to(device, non_blocking=True)
            logits = head(z)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * z.shape[0]
        total_loss /= N
        with torch.no_grad():
            correct = 0
            for s in range(0, img_va.shape[0], batch_size):
                z = _encode_batch(sae, img_va[s:s + batch_size], device, method, eval_topk)
                pred = head(z).argmax(dim=-1)
                correct += int((pred == y_va_t[s:s + batch_size].to(device)).sum().item())
            val_acc = correct / img_va.shape[0]
        final_val = val_acc
        if val_acc > best_val:
            best_val = val_acc
        logger.info("  ep %02d: train_loss=%.4f  val_top1=%.4f  (best %.4f)",
                    ep + 1, total_loss, val_acc, best_val)
    return best_val, final_val


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)
    sae = _select_image_sae(model, args.method)

    logger.info("loading ImageNet train (max_per_class=%d) + val", args.max_per_class)
    train_ds = eval_utils.load_pair_dataset(
        args.cache_dir, "imagenet", "train", max_per_class=args.max_per_class,
    )
    val_ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")

    logger.info("train=%d val=%d", len(train_ds), len(val_ds))
    img_tr, y_tr = _collect(train_ds)
    img_va, y_va = _collect(val_ds)

    # Move SAE to GPU once; keep frozen.
    sae.eval()
    sae.to(device)  # type: ignore[arg-type]
    L = int(sae.latent_size)
    logger.info("SAE latent_size=%d, training streaming linear probe (eval_topk=%s)",
                L, args.eval_topk)

    logger.info("training linear probe on GPU (lr=%.3g, wd=%.3g, epochs=%d)",
                args.lp_lr, args.lp_weight_decay, args.lp_epochs)
    best_val, final_val = _train_linprobe_streaming(
        sae, img_tr, y_tr, img_va, y_va,
        n_classes=1000, L=L, device=device,
        method=args.method, eval_topk=args.eval_topk,
        lr=args.lp_lr, weight_decay=args.lp_weight_decay,
        epochs=args.lp_epochs, batch_size=args.lp_batch_size,
    )
    logger.info("best val top-1 accuracy: %.4f", best_val)

    result = {
        "method": args.method,
        "dataset": "imagenet",
        "metric": "linear_probe_top1",
        "accuracy": best_val,
        "final_epoch_accuracy": final_val,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "latent_size": L,
        "eval_topk": args.eval_topk,
        "lp_lr": args.lp_lr,
        "lp_weight_decay": args.lp_weight_decay,
        "lp_epochs": args.lp_epochs,
        "lp_batch_size": args.lp_batch_size,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
