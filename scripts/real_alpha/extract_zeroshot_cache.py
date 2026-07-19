"""Extract zero-shot classification eval caches (CIFAR-100 / Food-101 / Pets).

Rebuttal E4a: additional zero-shot datasets in the ImageNet cache schema so
`eval_imagenet_zeroshot.py` and `CachedImageNetPairsDataset` work unchanged:

  cache_dir/
    image_embeddings.pt   # {int_idx: Tensor(d,)}
    text_embeddings.pt    # {f"{class_idx}_{template_idx}": Tensor(d,)}
    splits.json           # {"val": [[int_idx, class_idx], ...]}
    meta.json

Text prototypes use the 80 OpenAI ImageNet templates for every dataset —
one uniform protocol across all zero-shot evals.

Usage:
    python scripts/real_alpha/extract_zeroshot_cache.py \
        --dataset cifar100 --model openai/clip-vit-base-patch32 \
        --cache-dir cache/clip_b32_cifar100
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
from torch.utils.data import DataLoader

from extract_common import load_model_forwards  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ("cifar100", "food101", "pets")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["transformers", "openclip"])
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--data-root", type=str, default="cache/torchvision_data",
                   help="torchvision download root (shared across datasets)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _load_tv_dataset(name: str, root: str):
    """Returns (dataset yielding (PIL, class_idx), class_names cleaned)."""
    import torchvision.datasets as tvd
    if name == "cifar100":
        ds = tvd.CIFAR100(root, train=False, download=True)
    elif name == "food101":
        ds = tvd.Food101(root, split="test", download=True)
    elif name == "pets":
        ds = tvd.OxfordIIITPet(root, split="test", download=True)
    else:
        raise ValueError(name)
    class_names = [c.replace("_", " ") for c in ds.classes]
    return ds, class_names


def _pil_collate(batch):
    return {"imgs": [b[0] for b in batch], "labels": [int(b[1]) for b in batch]}


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")

    fwd = load_model_forwards(args.model, device, args.backend, args.pretrained)
    ds, class_names = _load_tv_dataset(args.dataset, args.data_root)
    logger.info("dataset=%s: %d test images, %d classes, model=%s (dim=%d)",
                args.dataset, len(ds), len(class_names), args.model, fwd.emb_dim)

    t0 = time.time()
    # -------- Image pass --------
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_pil_collate)
    image_dict: dict[int, torch.Tensor] = {}
    pairs: list[list[int]] = []
    idx = 0
    for batch in loader:
        feats = fwd.fwd_img(batch["imgs"])
        for vec, label in zip(feats, batch["labels"]):
            image_dict[idx] = vec.clone()
            pairs.append([idx, label])
            idx += 1
        if idx % (args.batch_size * 20) == 0:
            logger.info("images %d/%d", idx, len(ds))

    # -------- Text pass: classes × 80 OpenAI templates --------
    text_dict: dict[str, torch.Tensor] = {}
    pending_keys: list[str] = []
    pending_texts: list[str] = []

    def flush_texts():
        nonlocal pending_keys, pending_texts
        if not pending_texts:
            return
        feats = fwd.fwd_txt(pending_texts)
        for k, v in zip(pending_keys, feats):
            text_dict[k] = v.clone()
        pending_keys, pending_texts = [], []

    for c, name in enumerate(class_names):
        for t, tmpl in enumerate(OPENAI_IMAGENET_TEMPLATES):
            pending_keys.append(f"{c}_{t}")
            pending_texts.append(tmpl(name))
            if len(pending_texts) >= args.batch_size:
                flush_texts()
    flush_texts()
    logger.info("texts: %d entries (%d classes × %d templates)",
                len(text_dict), len(class_names), len(OPENAI_IMAGENET_TEMPLATES))

    torch.save(image_dict, cache_dir / "image_embeddings.pt")
    torch.save(text_dict, cache_dir / "text_embeddings.pt")
    with open(cache_dir / "splits.json", "w") as f:
        json.dump({"val": pairs}, f)
    meta = {
        "clip_model": args.model,
        "dataset": args.dataset,
        "dim": fwd.emb_dim,
        "kind": fwd.kind,
        "n_images": len(image_dict),
        "n_classes": len(class_names),
        "n_templates": len(OPENAI_IMAGENET_TEMPLATES),
        "class_names": class_names,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("done: %s (%d imgs, %d texts, %.1fs)",
                cache_dir, len(image_dict), len(text_dict), time.time() - t0)


if __name__ == "__main__":
    main()
