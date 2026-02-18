"""
CLIP Contrastive Loss Recovery Score for SAE Reconstruction Quality.

Measures how well SAE reconstruction preserves CLIP's contrastive alignment:
    LR = (H_zero - H_sae) / (H_zero - H_orig)
where H = symmetric contrastive loss between image and text embeddings.

Three modes:
    1. Joint LR: reconstruct both image_embeds and text_embeds
    2. Image LR: reconstruct only image_embeds
    3. Text LR: reconstruct only text_embeds

Dataset: Multimodal-Fatima/COCO_captions_test (columns: image, sentences_raw)

Usage:
    python test_clip_loss_recovery.py --sae_path Mayfull/CLIP_TopKSAE --num_samples 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from src.models.modeling_sae import (
    BatchTopKSAE,
    MatryoshkaSAE,
    TopKSAE,
    VLBatchTopKSAE,
    VLMatryoshkaSAE,
    VLTopKSAE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SAE_ARCH_MAP = {
    "TopKSAE": TopKSAE,
    "VLTopKSAE": VLTopKSAE,
    "BatchTopKSAE": BatchTopKSAE,
    "VLBatchTopKSAE": VLBatchTopKSAE,
    "MatryoshkaSAE": MatryoshkaSAE,
    "VLMatryoshkaSAE": VLMatryoshkaSAE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP contrastive loss recovery for SAE reconstruction")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output_dir", type=str, default="./results/clip_loss_recovery")
    parser.add_argument("--dataset_name", type=str, default="Multimodal-Fatima/COCO_captions_test")
    return parser.parse_args()


def load_models(args: argparse.Namespace) -> tuple:
    """Load CLIP model, processor, and SAE. Returns (clip_model, clip_processor, sae, device)."""
    from transformers import CLIPModel, CLIPProcessor, PretrainedConfig

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    logger.info("Loading CLIP: %s", args.clip_model_name)
    clip_model = CLIPModel.from_pretrained(args.clip_model_name, torch_dtype=dtype)
    clip_model.to(device)
    clip_model.eval()
    clip_model.requires_grad_(False)

    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)

    logger.info("Loading SAE: %s", args.sae_path)
    sae_config = PretrainedConfig.from_pretrained(args.sae_path)
    arch_name = getattr(sae_config, "architectures", ["TopKSAE"])[0]
    sae_cls = _SAE_ARCH_MAP.get(arch_name, TopKSAE)
    logger.info("Resolved SAE architecture: %s -> %s", arch_name, sae_cls.__name__)
    sae = sae_cls.from_pretrained(args.sae_path)
    logger.info("Setting SAE k = %d", args.k)
    sae.cfg.k = args.k
    sae.to(device)
    sae.eval()
    sae.requires_grad_(False)

    return clip_model, clip_processor, sae, device


def clip_contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float = 0.01) -> float:
    """Symmetric CLIP contrastive loss."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)).item() / 2


def sae_reconstruct(sae, embeds: torch.Tensor) -> torch.Tensor:
    """Pass embeddings through SAE encode/decode. Input: (B, d) -> (B, d)."""
    x = embeds.unsqueeze(0) if embeds.dim() == 2 else embeds  # (1, B, d)
    top_acts, top_indices = sae.encode(x)
    recon = sae.decode(top_acts, top_indices)
    return recon.squeeze(0) if embeds.dim() == 2 else recon


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    clip_model, clip_processor, sae, device = load_models(args)

    logger.info("Loading dataset: %s", args.dataset_name)
    dataset = load_dataset(args.dataset_name, split="test")

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    # Collect all (image, caption) pairs
    images, captions = [], []
    for idx in indices:
        sample = dataset[idx]
        try:
            img = sample["image"].convert("RGB")
        except Exception:
            continue
        cap = random.choice(sample["sentences_raw"]) if isinstance(sample["sentences_raw"], list) else sample["sentences_raw"]
        images.append(img)
        captions.append(cap)

    n = len(images)
    logger.info("Collected %d valid image-caption pairs", n)

    batch_size = args.batch_size
    joint_losses = {"orig": [], "sae": [], "zero": []}
    image_losses = {"orig": [], "sae": [], "zero": []}
    text_losses = {"orig": [], "sae": [], "zero": []}

    for start in tqdm(range(0, n, batch_size), desc="Processing batches"):
        batch_imgs = images[start : start + batch_size]
        batch_caps = captions[start : start + batch_size]
        bs = len(batch_imgs)

        inputs = clip_processor(text=batch_caps, images=batch_imgs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
        img_emb = outputs.image_embeds  # (bs, d)
        txt_emb = outputs.text_embeds   # (bs, d)

        with torch.no_grad():
            img_recon = sae_reconstruct(sae, img_emb)
            txt_recon = sae_reconstruct(sae, txt_emb)

        zeros = torch.zeros_like(img_emb)

        # Joint LR
        joint_losses["orig"].append(clip_contrastive_loss(img_emb, txt_emb))
        joint_losses["sae"].append(clip_contrastive_loss(img_recon, txt_recon))
        joint_losses["zero"].append(clip_contrastive_loss(zeros, zeros))

        # Image LR (reconstruct images only)
        image_losses["orig"].append(clip_contrastive_loss(img_emb, txt_emb))
        image_losses["sae"].append(clip_contrastive_loss(img_recon, txt_emb))
        image_losses["zero"].append(clip_contrastive_loss(zeros, txt_emb))

        # Text LR (reconstruct text only)
        text_losses["orig"].append(clip_contrastive_loss(img_emb, txt_emb))
        text_losses["sae"].append(clip_contrastive_loss(img_emb, txt_recon))
        text_losses["zero"].append(clip_contrastive_loss(img_emb, zeros))

    def compute_lr(losses: dict[str, list[float]]) -> dict:
        import numpy as np
        orig = np.mean(losses["orig"])
        sae_l = np.mean(losses["sae"])
        zero = np.mean(losses["zero"])
        denom = max(zero - orig, 1e-6)
        lr = (zero - sae_l) / denom
        return {
            "loss_recovery": float(lr),
            "mean_ce_orig": float(orig),
            "mean_ce_sae": float(sae_l),
            "mean_ce_zero": float(zero),
        }

    result = {
        "metadata": {
            "clip_model": args.clip_model_name,
            "sae_path": args.sae_path,
            "k": args.k,
            "num_samples": n,
            "batch_size": batch_size,
            "dataset": args.dataset_name,
        },
        "joint_loss_recovery": compute_lr(joint_losses),
        "image_loss_recovery": compute_lr(image_losses),
        "text_loss_recovery": compute_lr(text_losses),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    sae_name = args.sae_path.rstrip("/").split("/")[-1]
    filepath = os.path.join(args.output_dir, f"CLIP_LOSS_RECOVERY_{sae_name}_k{args.k}.json")
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved results to %s", filepath)

    logger.info(
        "Joint LR: %.4f | Image LR: %.4f | Text LR: %.4f",
        result["joint_loss_recovery"]["loss_recovery"],
        result["image_loss_recovery"]["loss_recovery"],
        result["text_loss_recovery"]["loss_recovery"],
    )


if __name__ == "__main__":
    main()
