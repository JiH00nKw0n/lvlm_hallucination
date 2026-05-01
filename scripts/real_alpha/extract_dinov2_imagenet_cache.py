"""Extract DINOv2 ViT-B/14 image embeddings for ImageNet-1K val (50K).

Used as the external image encoder E in the MonoSemanticity (MS) score
(Pach et al., NeurIPS 2025). Image IDs match the existing CLIP cache:
val ids are 2_000_000 + counter, in the order yielded by the HF stream
(counter = 0..49_999).

Saves:
    cache/dinov2_b14_imagenet/image_embeddings.pt   {int img_id: Tensor(D,)}
    cache/dinov2_b14_imagenet/splits.json            {"val": [[img_id, class_idx], ...]}
    cache/dinov2_b14_imagenet/meta.json

Usage on elice-40g:
    export HF_TOKEN=...
    python scripts/real_alpha/extract_dinov2_imagenet_cache.py \
        --cache-dir cache/dinov2_b14_imagenet --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "ILSVRC/imagenet-1k"
VAL_OFFSET = 2_000_000
VAL_TOTAL = 50_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="facebook/dinov2-base")
    p.add_argument("--cache-dir", type=str, default="cache/dinov2_b14_imagenet")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--flush-every", type=int, default=10_000)
    return p.parse_args()


def load_state(path: Path) -> dict:
    if path.exists():
        logger.info("Loading existing cache %s", path)
        return torch.load(path, map_location="cpu")
    return {}


def save_state(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def extract(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    hf_token = os.environ.get("HF_TOKEN")
    logger.info("device=%s model=%s", device, args.model)

    model = AutoModel.from_pretrained(args.model).to(device).eval()
    model.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(args.model)
    emb_dim = int(model.config.hidden_size)
    logger.info("DINOv2 hidden_size=%d", emb_dim)

    image_dict: dict[int, torch.Tensor] = {
        int(k): v for k, v in load_state(cache_dir / "image_embeddings.pt").items()
    }
    logger.info("pre-existing cached: %d", len(image_dict))

    splits_path = cache_dir / "splits.json"
    splits = json.load(open(splits_path)) if splits_path.exists() else {}

    if "val" in splits and {p[0] for p in splits["val"]}.issubset(image_dict.keys()):
        logger.info("val split already fully cached (%d), skipping image pass", len(splits["val"]))
    else:
        logger.info("=== streaming val (~%d images) ===", VAL_TOTAL)
        ds = load_dataset(DATASET_NAME, split="validation", streaming=True, token=hf_token)

        split_pairs: list[list[int]] = []
        pending_ids: list[int] = []
        pending_pils: list = []
        processed_since_flush = 0
        n_extracted = 0
        t_start = time.time()

        def flush() -> None:
            nonlocal pending_ids, pending_pils, processed_since_flush, n_extracted
            if not pending_pils:
                return
            with torch.no_grad():
                inputs = processor(images=pending_pils, return_tensors="pt")
                out = model(pixel_values=inputs["pixel_values"].to(device))
                # DINOv2 returns last_hidden_state (B, T, H); CLS is the [0] token.
                feats = out.last_hidden_state[:, 0, :].float().cpu()
            for iid, vec in zip(pending_ids, feats):
                image_dict[iid] = vec.clone()
            n_extracted += len(pending_ids)
            processed_since_flush += len(pending_ids)
            pending_ids = []
            pending_pils = []

        pbar = tqdm(ds, total=VAL_TOTAL, desc="dino-val", miniters=100, smoothing=0.1)
        for counter, row in enumerate(pbar):
            img_idx = VAL_OFFSET + counter
            class_idx = int(row["label"])
            split_pairs.append([img_idx, class_idx])

            if img_idx in image_dict:
                continue
            pending_ids.append(img_idx)
            pending_pils.append(row["image"].convert("RGB"))
            if len(pending_pils) >= args.batch_size:
                flush()
                if processed_since_flush >= args.flush_every:
                    elapsed = time.time() - t_start
                    rate = n_extracted / elapsed if elapsed > 0 else 0
                    remaining = (VAL_TOTAL - counter) / rate if rate > 0 else 0
                    logger.info(
                        "[heartbeat] dino_extract progress=%d/%d rate=%.1f img/s eta=%.0f sec",
                        n_extracted, VAL_TOTAL, rate, remaining,
                    )
                    save_state(image_dict, cache_dir / "image_embeddings.pt")
                    processed_since_flush = 0

        flush()
        save_state(image_dict, cache_dir / "image_embeddings.pt")
        splits["val"] = split_pairs
        with open(splits_path, "w") as f:
            json.dump(splits, f)
        logger.info(
            "DONE val: %d extracted in %.1f min (%.1f img/s)",
            n_extracted, (time.time() - t_start) / 60,
            n_extracted / max(1e-9, time.time() - t_start),
        )

    meta = {
        "model": args.model,
        "dataset": DATASET_NAME,
        "split": "validation",
        "dim": emb_dim,
        "n_images": len(image_dict),
        "n_val_pairs": len(splits.get("val", [])),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("meta.json written: n_images=%d dim=%d", meta["n_images"], emb_dim)


if __name__ == "__main__":
    extract(parse_args())
