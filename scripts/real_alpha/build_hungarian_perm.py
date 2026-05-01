"""Build the post-hoc Hungarian text→image slot permutation for Ours.

Loads a trained TwoSidedTopKSAE, streams paired train embeddings, computes
the Pearson correlation matrix C between per-side dense latents, runs
linear_sum_assignment(-C), and saves `perm.npz` with fields:
    perm  — int64 (L_per_side,)
    C     — float32 (L_per_side, L_per_side)

At eval time, z_T_aligned[:, i] = z_T[:, perm[i]].

Usage:
    python scripts/real_alpha/build_hungarian_perm.py \
        --ckpt outputs/real_exp_v1/separated/coco/final \
        --dataset coco --cache-dir cache/clip_b32_coco \
        --output outputs/real_exp_v1/ours/coco/perm.npz
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--dataset", choices=["coco", "imagenet", "cc3m"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default=None,
                   help="Split to read (default 'train'; falls back to 'val' if "
                        "the train split is absent for imagenet).")
    p.add_argument("--max-per-class", type=int, default=1000,
                   help="ImageNet-only balance cap.")
    p.add_argument("--max-samples", type=int, default=50_000,
                   help="Cap on paired samples used to build C.")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _resolve_split(cache_dir: str, dataset: str, requested: str | None) -> str:
    """Return 'train' if available, else fall back to 'val' (imagenet only)."""
    import json
    if requested:
        return requested
    splits_path = _Path(cache_dir) / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            keys = set(json.load(f).keys())
        if "train" not in keys and dataset == "imagenet" and "val" in keys:
            logger.warning("imagenet train split missing in %s — falling back to val "
                           "for permutation build", splits_path)
            return "val"
    return "train"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading separated SAE from %s", args.ckpt)
    model = eval_utils.load_sae(args.ckpt, "separated")

    split = _resolve_split(args.cache_dir, args.dataset, args.split)
    ds = eval_utils.load_pair_dataset(
        args.cache_dir, args.dataset, split=split,
        max_per_class=args.max_per_class if args.dataset == "imagenet" else None,
    )
    logger.info("paired dataset: %d samples (split=%s)", len(ds), split)

    result = eval_utils.build_perm(
        model, ds, device, max_samples=args.max_samples, batch_size=args.batch_size,
    )
    perm = result["perm"]
    logger.info("saved perm len=%d → %s", perm.shape[0], out_path)
    np.savez(
        out_path,
        perm=perm,
        alive_image=result["alive_image"],
        alive_text=result["alive_text"],
        fire_count_image=result["fire_count_image"],
        fire_count_text=result["fire_count_text"],
    )


if __name__ == "__main__":
    main()
