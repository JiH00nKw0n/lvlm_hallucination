"""Reconstruction error on eval split.

Reports the paper formula:
    recon_err = 0.5 * E[ ||x - x_hat||^2 + ||y - y_hat||^2 ]

where x, y are L2-normalized CLIP image/text embeddings and x_hat, y_hat are
each SAE's reconstruction on its modality. Works for all 5 methods.

Note: for method='ours', reconstruction is identical to method='separated'
(Hungarian permutation is a no-op on reconstruction — only affects which
latent slot stores which concept). We still compute & write it for symmetry.

Usage:
    python scripts/real_alpha/eval_recon_downstream.py \
        --ckpt outputs/real_exp_v1/shared/coco/final \
        --method shared --dataset coco --cache-dir cache/clip_b32_coco \
        --split test --output outputs/real_exp_v1/shared/coco/recon.json
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

import torch  # noqa: E402

import eval_utils  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae", "shared_enc"], required=True)
    p.add_argument("--dataset", choices=["coco", "imagenet", "cc3m"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def _recon_mse_topksae(sae, embeds: torch.Tensor, batch_size: int, device: torch.device) -> float:
    """Mean per-sample squared reconstruction error for TopKSAE."""
    sae.eval()
    sae.to(device)  # type: ignore[arg-type]
    total = 0.0
    n = 0
    for s in range(0, embeds.shape[0], batch_size):
        chunk = embeds[s:s + batch_size].to(device).unsqueeze(1)
        out = sae(hidden_states=chunk)
        x_hat = out.output.squeeze(1)
        err = (chunk.squeeze(1) - x_hat).pow(2).sum(dim=-1)
        total += float(err.sum().item())
        n += chunk.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def _recon_mse_vlsae(model, embeds: torch.Tensor, modality: str, batch_size: int, device: torch.device) -> float:
    """Mean per-sample squared reconstruction error for VLSAE (image or text side)."""
    model.eval()
    model.to(device)  # type: ignore[arg-type]
    decoder = model.vision_decoder if modality == "image" else model.text_decoder
    total = 0.0
    n = 0
    for s in range(0, embeds.shape[0], batch_size):
        chunk = embeds[s:s + batch_size].to(device)
        z = model.encode(chunk)
        x_hat = decoder(z)
        err = (chunk - x_hat).pow(2).sum(dim=-1)
        total += float(err.sum().item())
        n += chunk.shape[0]
    return total / max(n, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)

    logger.info("loading %s split=%s from %s", args.dataset, args.split, args.cache_dir)
    ds = eval_utils.load_pair_dataset(args.cache_dir, args.dataset, args.split)

    # Stack all pairs once (caches are small: COCO test 25k, ImageNet val 50k)
    n = len(ds)
    img = torch.stack([ds[i]["image_embeds"] for i in range(n)], dim=0)
    txt = torch.stack([ds[i]["text_embeds"] for i in range(n)], dim=0)
    logger.info("n=%d, img=%s txt=%s", n, img.shape, txt.shape)

    if args.method in ("vl_sae", "shared_enc"):
        mse_img = _recon_mse_vlsae(model, img, "image", args.batch_size, device)
        mse_txt = _recon_mse_vlsae(model, txt, "text", args.batch_size, device)
    else:
        if args.method in ("shared", "aux"):
            sae_i = sae_t = model
        else:
            sae_i = model.image_sae
            sae_t = model.text_sae
        mse_img = _recon_mse_topksae(sae_i, img, args.batch_size, device)
        mse_txt = _recon_mse_topksae(sae_t, txt, args.batch_size, device)
    recon = 0.5 * (mse_img + mse_txt)

    result = {
        "method": args.method,
        "dataset": args.dataset,
        "split": args.split,
        "n": n,
        "recon_error": recon,
        "recon_image": mse_img,
        "recon_text": mse_txt,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s: %s", out_path, result)


if __name__ == "__main__":
    main()
