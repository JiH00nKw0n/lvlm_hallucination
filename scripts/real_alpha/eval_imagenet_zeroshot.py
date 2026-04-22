"""ImageNet-1K zero-shot classification in SAE latent space.

Pipeline (paper slide 20):
  1) For each class c (0..999): take the 80 cached template text embeddings
     in CLIP space and MEAN them. L2-normalize. Encode through text-side SAE
     (and perm for ours) → class prototype z_T^c.
  2) For each val image: encode with image-side SAE → z_I.
  3) Predict: argmax_c cos(z_I, z_T^c).

Usage:
    python scripts/real_alpha/eval_imagenet_zeroshot.py \
        --ckpt outputs/real_exp_v1/ours/imagenet/final \
        --method ours --cache-dir cache/clip_b32_imagenet \
        --perm outputs/real_exp_v1/ours/imagenet/perm.npz \
        --output outputs/real_exp_v1/ours/imagenet/zeroshot.json
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

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--perm", type=str, default=None,
                   help="Path to perm.npz (required for method='ours').")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--n-templates", type=int, default=80)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _build_class_prototypes_clip(
    text_dict: dict[str, torch.Tensor], n_classes: int, n_templates: int
) -> torch.Tensor:
    """Mean over 80 templates in CLIP embedding space, L2-normalize, return (n_classes, H)."""
    protos = []
    for c in range(n_classes):
        vecs = torch.stack(
            [text_dict[f"{c}_{t}"] for t in range(n_templates)], dim=0,
        )
        mean = vecs.mean(dim=0)
        protos.append(_l2_normalize(mean))
    return torch.stack(protos, dim=0)  # (n_classes, H)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)

    perm = None
    if args.method == "ours":
        if args.perm is None:
            raise SystemExit("--perm required for method='ours'")
        perm = np.load(args.perm)["perm"]
        logger.info("loaded perm (len=%d) from %s", perm.shape[0], args.perm)

    # 1) Build text prototypes in CLIP space, then encode through SAE.
    text_dict_raw = torch.load(
        str(Path(args.cache_dir) / "text_embeddings.pt"), map_location="cpu",
    )
    text_dict = {str(k): v.to(torch.float32) for k, v in text_dict_raw.items()}
    # L2-normalize the raw entries to match dataset convention
    for k, v in text_dict.items():
        text_dict[k] = _l2_normalize(v)

    logger.info("building %d prototypes (mean of %d templates each)", args.n_classes, args.n_templates)
    protos_clip = _build_class_prototypes_clip(text_dict, args.n_classes, args.n_templates)

    logger.info("encoding prototypes through text-side SAE")
    z_protos = eval_utils.encode_text(model, protos_clip, args.method, device, perm=perm, batch_size=args.batch_size)
    z_protos = eval_utils.normalize_rows(z_protos)  # cosine prep

    # 2) Encode val images
    val_ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")
    logger.info("val=%d", len(val_ds))
    img_va = torch.stack([val_ds[i]["image_embeds"] for i in range(len(val_ds))], dim=0)
    y_va = np.array([int(val_ds.pairs[i][1]) for i in range(len(val_ds))], dtype=np.int64)

    logger.info("encoding val images → SAE latents")
    z_val = eval_utils.encode_image(model, img_va, args.method, device, args.batch_size)
    z_val = eval_utils.normalize_rows(z_val)

    # 3) Top-1 prediction via cosine similarity
    # Chunk val-side to avoid materializing 50k × 1000 on GPU if huge.
    correct = 0
    bsz = 8192
    for s in range(0, z_val.shape[0], bsz):
        scores = z_val[s:s + bsz] @ z_protos.T  # (b, n_classes)
        pred = scores.argmax(dim=1).cpu().numpy()
        correct += int((pred == y_va[s:s + bsz]).sum())
    acc = correct / z_val.shape[0]
    logger.info("val top-1 accuracy: %.4f", acc)

    result = {
        "method": args.method,
        "dataset": "imagenet",
        "metric": "zeroshot_top1",
        "accuracy": float(acc),
        "n_val": int(z_val.shape[0]),
        "n_classes": args.n_classes,
        "n_templates": args.n_templates,
        "latent_size": int(z_val.shape[1]),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
