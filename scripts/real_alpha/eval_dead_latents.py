"""Report per-method dead-latent counts on paired train subset.

For each (method, dataset) checkpoint: stream up to `max_samples` paired
embeddings, encode image-side and text-side through the appropriate SAE,
count nonzero firings per slot. Save counts to `dead_latents.json`.

Usage:
    python scripts/real_alpha/eval_dead_latents.py \
        --ckpt outputs/real_exp_v1/separated/coco/final \
        --method separated --dataset coco \
        --cache-dir cache/clip_b32_coco \
        --output outputs/real_exp_v1/separated/coco/dead_latents.json
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

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae", "shared_enc"], required=True)
    p.add_argument("--dataset", choices=["coco", "imagenet", "cc3m"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-per-class", type=int, default=1000)
    p.add_argument("--max-samples", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--alive-min-fires", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)

    logger.info("loading %s train split", args.dataset)
    ds = eval_utils.load_pair_dataset(
        args.cache_dir, args.dataset, split="train",
        max_per_class=args.max_per_class if args.dataset == "imagenet" else None,
    )

    info = eval_utils.compute_dead_counts(
        model, ds, args.method, device,
        max_samples=args.max_samples, batch_size=args.batch_size,
        alive_min_fires=args.alive_min_fires,
    )
    result = {
        "method": args.method,
        "dataset": args.dataset,
        **{k: v for k, v in info.items() if k not in ("fire_count_image", "fire_count_text")},
    }
    logger.info(
        "%s / %s: alive image=%d/%d (%.1f%%), alive text=%d/%d (%.1f%%)",
        args.method, args.dataset,
        result["alive_image_count"], result["latent_size_image"],
        100 * result["alive_image_count"] / result["latent_size_image"],
        result["alive_text_count"], result["latent_size_text"],
        100 * result["alive_text_count"] / result["latent_size_text"],
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
