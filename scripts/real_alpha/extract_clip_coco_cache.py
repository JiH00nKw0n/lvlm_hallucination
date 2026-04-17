"""Extract CLIP ViT-B/32 image/text embeddings for COCO Karpathy.

Idempotent. Saves:
  cache_dir/image_embeddings.pt   -- dict[int image_id -> Tensor(d,) float32]
  cache_dir/text_embeddings.pt    -- dict[str "<image_id>_<caption_idx>" -> Tensor(d,) float32]
  cache_dir/splits.json           -- {"train": [[image_id, caption_idx], ...], "val": [...], "test": [...]}
  cache_dir/meta.json             -- model, dim, dataset rev, counts

Reruns skip any (image_id) / (image_id, caption_idx) that already has an
embedding in the loaded dict. Flushes every N processed items.

Usage:
    python scripts/real_alpha/extract_clip_coco_cache.py \
        --cache-dir cache/clip_b32_coco --batch-size 256 --device cuda
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
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLIP_MODEL = "openai/clip-vit-base-patch32"
DATASET_NAME = "namkha1032/coco-karpathy"
# HF dataset uses "validation" but we normalize to "val" in splits.json for brevity.
HF_TO_OUR_SPLIT = {"train": "train", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default="cache/clip_b32_coco")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--flush-every", type=int, default=4096, help="flush cache dicts every N processed items")
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
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


def build_splits(cache_dir: Path, hf_splits: list[str]) -> dict[str, list[tuple[str, int]]]:
    splits_path = cache_dir / "splits.json"
    if splits_path.exists():
        logger.info("splits.json already exists, loading")
        with open(splits_path, "r") as f:
            return json.load(f)

    out: dict[str, list] = {}
    for hf_split in hf_splits:
        our_split = HF_TO_OUR_SPLIT[hf_split]
        logger.info("Building split list: %s", hf_split)
        ds = load_dataset(DATASET_NAME, split=hf_split)
        pairs: list = []
        for row in tqdm(ds, desc=f"split-{hf_split}"):
            image_id = str(row["image_id"])
            n_caps = len(row["captions"])
            for ci in range(n_caps):
                pairs.append([image_id, ci])
        out[our_split] = pairs

    with open(splits_path, "w") as f:
        json.dump(out, f)
    logger.info("Saved splits.json with %s entries", {k: len(v) for k, v in out.items()})
    return out


def extract(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading CLIP %s", CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    image_dict: dict[int, torch.Tensor] = {
        int(k): v for k, v in load_state(cache_dir / "image_embeddings.pt").items()
    }
    text_dict: dict[str, torch.Tensor] = {
        str(k): v for k, v in load_state(cache_dir / "text_embeddings.pt").items()
    }

    splits = build_splits(cache_dir, args.splits)

    # -------- Image pass --------
    for hf_split in args.splits:
        our_split = HF_TO_OUR_SPLIT[hf_split]
        # Skip entire split if every image_id in this split is already cached.
        split_pairs = splits.get(our_split, [])
        split_iids = {int(p[0]) for p in split_pairs}
        if split_iids and split_iids.issubset(image_dict.keys()):
            logger.info("=== Image pass: %s (SKIP: all %d ids already cached) ===",
                        hf_split, len(split_iids))
            continue
        logger.info("=== Image pass: %s ===", hf_split)
        ds = load_dataset(DATASET_NAME, split=hf_split)
        pending_ids: list[int] = []
        pending_pils: list = []
        processed_since_flush = 0

        def flush_images():
            nonlocal pending_ids, pending_pils, processed_since_flush
            if not pending_pils:
                return
            inputs = processor(images=pending_pils, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=pixel_values)
                pooled = vision_out.pooler_output
                feats = model.visual_projection(pooled).float().cpu()
            for iid, vec in zip(pending_ids, feats):
                image_dict[iid] = vec.clone()
            processed_since_flush += len(pending_ids)
            pending_ids = []
            pending_pils = []

        for row in tqdm(ds, desc=f"img-{hf_split}"):
            iid = int(row["image_id"])
            if iid in image_dict:
                continue
            pending_ids.append(iid)
            pending_pils.append(row["image"].convert("RGB"))
            if len(pending_pils) >= args.batch_size:
                flush_images()
                if processed_since_flush >= args.flush_every:
                    save_state(image_dict, cache_dir / "image_embeddings.pt")
                    processed_since_flush = 0
        flush_images()
        save_state(image_dict, cache_dir / "image_embeddings.pt")
        logger.info("image_dict size = %d", len(image_dict))

    # -------- Text pass --------
    for hf_split in args.splits:
        our_split = HF_TO_OUR_SPLIT[hf_split]
        # Skip entire split if every (image_id, caption_idx) is cached.
        split_pairs = splits.get(our_split, [])
        if split_pairs and all(f"{int(p[0])}_{int(p[1])}" in text_dict for p in split_pairs):
            logger.info("=== Text pass: %s (SKIP: all %d pairs already cached) ===",
                        hf_split, len(split_pairs))
            continue
        logger.info("=== Text pass: %s ===", hf_split)
        # Drop the image column so HF datasets doesn't PIL-decode every row.
        ds = load_dataset(DATASET_NAME, split=hf_split)
        if "image" in ds.column_names:
            ds = ds.remove_columns(["image"])
        pending_keys: list[str] = []
        pending_texts: list[str] = []
        processed_since_flush = 0

        def flush_texts():
            nonlocal pending_keys, pending_texts, processed_since_flush
            if not pending_texts:
                return
            tok = processor(
                text=pending_texts, return_tensors="pt", padding=True, truncation=True, max_length=77,
            )
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
            with torch.no_grad():
                text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = text_out.pooler_output
                feats = model.text_projection(pooled).float().cpu()
            for key, vec in zip(pending_keys, feats):
                text_dict[key] = vec.clone()
            processed_since_flush += len(pending_keys)
            pending_keys = []
            pending_texts = []

        for row in tqdm(ds, desc=f"txt-{hf_split}"):
            iid = int(row["image_id"])
            caps = row["captions"]
            for ci, cap in enumerate(caps):
                key = f"{iid}_{ci}"
                if key in text_dict:
                    continue
                pending_keys.append(key)
                pending_texts.append(cap)
                if len(pending_texts) >= args.batch_size:
                    flush_texts()
                    if processed_since_flush >= args.flush_every:
                        save_state(text_dict, cache_dir / "text_embeddings.pt")
                        processed_since_flush = 0
        flush_texts()
        save_state(text_dict, cache_dir / "text_embeddings.pt")
        logger.info("text_dict size = %d", len(text_dict))

    meta = {
        "clip_model": CLIP_MODEL,
        "dataset": DATASET_NAME,
        "dim": 512,
        "n_images": len(image_dict),
        "n_texts": len(text_dict),
        "n_train_pairs": len(splits.get("train", [])),
        "n_val_pairs": len(splits.get("val", [])),
        "n_test_pairs": len(splits.get("test", [])),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("meta.json written: %s", meta)


def main() -> None:
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    extract(args)


if __name__ == "__main__":
    main()
