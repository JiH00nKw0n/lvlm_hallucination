"""Extract Flickr30k (Karpathy test 1k) paired embeddings in COCO cache schema.

Rebuttal E4a: cross-dataset retrieval eval. Streams HF `nlphuji/flickr30k`
(single "test" HF split carrying ALL 31k images with a Karpathy `split`
column), keeps rows whose split matches --karpathy-split (default test =
1000 images × 5 captions), and emits the COCO-style cache so
`eval_coco_retrieval.py` runs unchanged:

  cache_dir/
    image_embeddings.pt   # {int(img_id): Tensor(d,)}
    text_embeddings.pt    # {f"{img_id}_{cap_idx}": Tensor(d,)}
    splits.json           # {"test": [[img_id, cap_idx], ...]}
    meta.json

Usage:
    python scripts/real_alpha/extract_flickr30k_cache.py \
        --model openai/clip-vit-base-patch32 \
        --cache-dir cache/clip_b32_flickr30k
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from datasets import load_dataset

from extract_common import load_model_forwards  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET = "nlphuji/flickr30k"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["transformers", "openclip"])
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--karpathy-split", type=str, default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")

    fwd = load_model_forwards(args.model, device, args.backend, args.pretrained)
    logger.info("model=%s dim=%d; streaming %s (karpathy split=%s)",
                args.model, fwd.emb_dim, HF_DATASET, args.karpathy_split)

    ds = load_dataset(HF_DATASET, split="test", streaming=True)

    image_dict: dict[int, torch.Tensor] = {}
    text_dict: dict[str, torch.Tensor] = {}
    pairs: list[list[int]] = []
    buf_imgs, buf_iids, buf_txts, buf_tkeys = [], [], [], []
    t0 = time.time()

    def flush():
        nonlocal buf_imgs, buf_iids, buf_txts, buf_tkeys
        if buf_imgs:
            feats = fwd.fwd_img(buf_imgs)
            for iid, v in zip(buf_iids, feats):
                image_dict[iid] = v.clone()
        if buf_txts:
            tfeats = fwd.fwd_txt(buf_txts)
            for k, v in zip(buf_tkeys, tfeats):
                text_dict[k] = v.clone()
        buf_imgs, buf_iids, buf_txts, buf_tkeys = [], [], [], []

    n_kept = 0
    for row in ds:
        if row["split"] != args.karpathy_split:
            continue
        iid = int(row["img_id"])
        buf_imgs.append(row["image"].convert("RGB"))
        buf_iids.append(iid)
        for ci, cap in enumerate(row["caption"]):
            buf_txts.append(cap)
            buf_tkeys.append(f"{iid}_{ci}")
            pairs.append([iid, ci])
        n_kept += 1
        if len(buf_imgs) >= args.batch_size:
            flush()
            logger.info("images %d (pairs %d)", n_kept, len(pairs))
    flush()

    torch.save(image_dict, cache_dir / "image_embeddings.pt")
    torch.save(text_dict, cache_dir / "text_embeddings.pt")
    with open(cache_dir / "splits.json", "w") as f:
        json.dump({args.karpathy_split: pairs}, f)
    meta = {
        "clip_model": args.model,
        "dataset": HF_DATASET,
        "karpathy_split": args.karpathy_split,
        "dim": fwd.emb_dim,
        "kind": fwd.kind,
        "n_images": len(image_dict),
        "n_texts": len(text_dict),
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("done: %s (%d imgs, %d texts, %.1fs)",
                cache_dir, len(image_dict), len(text_dict), time.time() - t0)


if __name__ == "__main__":
    main()
