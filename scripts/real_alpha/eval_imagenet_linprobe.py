"""ImageNet-1K linear probe on SAE image-side latents.

Encodes each train (balanced, max_per_class) and val image through the
image-side SAE, then fits a multinomial logistic regression on the (latent,
class_idx) pairs. Reports val top-1 accuracy.

This is the paper's monosemantic-feature-recovery proxy. Uses only the
image-side SAE — alignment is irrelevant here.

Usage:
    python scripts/real_alpha/eval_imagenet_linprobe.py \
        --ckpt outputs/real_exp_v1/shared/imagenet/final \
        --method shared --cache-dir cache/clip_b32_imagenet \
        --output outputs/real_exp_v1/shared/imagenet/linprobe.json \
        --max-per-class 1000
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
from sklearn.linear_model import LogisticRegression  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-per-class", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    # scikit-learn LR params
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1000)
    return p.parse_args()


def _collect(ds):
    img = torch.stack([ds[i]["image_embeds"] for i in range(len(ds))], dim=0)
    y = np.array([int(ds.pairs[i][1]) for i in range(len(ds))], dtype=np.int64)
    return img, y


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)

    logger.info("loading ImageNet train (max_per_class=%d) + val", args.max_per_class)
    train_ds = eval_utils.load_pair_dataset(
        args.cache_dir, "imagenet", "train", max_per_class=args.max_per_class,
    )
    val_ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")

    logger.info("train=%d val=%d", len(train_ds), len(val_ds))
    img_tr, y_tr = _collect(train_ds)
    img_va, y_va = _collect(val_ds)

    logger.info("encoding train images → SAE latents")
    z_tr = eval_utils.encode_image(model, img_tr, args.method, device, args.batch_size).numpy()
    logger.info("encoding val images → SAE latents")
    z_va = eval_utils.encode_image(model, img_va, args.method, device, args.batch_size).numpy()

    logger.info("fitting LogisticRegression (C=%.3g, max_iter=%d)", args.C, args.max_iter)
    clf = LogisticRegression(
        solver="lbfgs",
        multi_class="multinomial",
        C=args.C,
        max_iter=args.max_iter,
        n_jobs=-1,
    )
    clf.fit(z_tr, y_tr)
    acc = float(clf.score(z_va, y_va))
    logger.info("val top-1 accuracy: %.4f", acc)

    result = {
        "method": args.method,
        "dataset": "imagenet",
        "metric": "linear_probe_top1",
        "accuracy": acc,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "latent_size": int(z_tr.shape[1]),
        "C": args.C,
        "max_iter": args.max_iter,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
