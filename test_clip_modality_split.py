"""
Modality Specificity Analysis for CLIP SAE Features.

Measures whether each SAE feature responds more to image embeddings or text embeddings
using Normalized Token-Frequency Ratio on CLIP embedding space.

Each CLIP forward yields 1 embedding per sample (CLS-pooled), so:
    n_img_tokens = n_txt_tokens = n_samples (one token per sample per modality)

Usage:
    python test_clip_modality_split.py --sae_path Mayfull/CLIP_TopKSAE --num_samples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Optional

import torch
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm

from src.models.modeling_sae import (
    BatchTopKSAE,
    MatryoshkaSAE,
    TopKSAE,
    VLBatchTopKSAE,
    VLMatryoshkaSAE,
    VLTopKSAE,
    is_vl_sae,
    vl_encode,
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
    parser = argparse.ArgumentParser(description="Modality split analysis for CLIP SAE features")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output_dir", type=str, default="./results/clip_modality_split")
    parser.add_argument("--dataset_name", type=str, default="Multimodal-Fatima/COCO_captions_test")
    parser.add_argument("--bin_width", type=float, default=0.05, help="Histogram bin width (0~1 range)")
    parser.add_argument("--weighted", action="store_true", help="Weight histogram by activation frequency")
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

    if not hasattr(sae, "latent_size"):
        if hasattr(sae, "latent_size_total"):
            sae.latent_size = sae.latent_size_total
        elif hasattr(sae, "num_latents"):
            sae.latent_size = sae.num_latents

    return clip_model, clip_processor, sae, device


def compute_modality_ratio(
    img_counts: Tensor,
    text_counts: Tensor,
    total_img_tokens: int,
    total_text_tokens: int,
    latent_size: int,
) -> dict:
    """Compute normalized token-frequency ratio: ratio = density_img / (density_img + density_text)."""
    density_img = img_counts.float() / max(total_img_tokens, 1)
    density_text = text_counts.float() / max(total_text_tokens, 1)
    denom = density_img + density_text

    alive_mask = denom > 0
    ratio = torch.full((latent_size,), float("nan"))
    ratio[alive_mask] = density_img[alive_mask] / denom[alive_mask]

    alive_ratios = ratio[alive_mask]
    num_alive = alive_mask.sum().item()
    num_dead = latent_size - num_alive

    num_shared = ((alive_ratios >= 0.2) & (alive_ratios <= 0.8)).sum().item()
    sfp = round(num_shared / max(num_alive, 1) * 100, 2) if num_alive > 0 else 0.0

    return {
        "ratio": [None if x != x else x for x in ratio.tolist()],
        "alive_mask": alive_mask.tolist(),
        "summary": {
            "num_alive_features": num_alive,
            "num_dead_features": num_dead,
            "num_image_specific": (alive_ratios > 0.8).sum().item(),
            "num_text_specific": (alive_ratios < 0.2).sum().item(),
            "num_shared": num_shared,
            "shared_feature_percentage": sfp,
            "mean_ratio": alive_ratios.mean().item() if num_alive > 0 else 0.0,
            "median_ratio": alive_ratios.median().item() if num_alive > 0 else 0.0,
            "std_ratio": alive_ratios.std().item() if num_alive > 0 else 0.0,
        },
    }


def plot_histogram(
    ratio: Tensor,
    alive_mask: Tensor,
    output_dir: str,
    sae_path: str,
    k: int,
    bin_width: float = 0.05,
    weights: Optional[Tensor] = None,
) -> str:
    """Plot modality ratio histogram for alive features. Returns the output file path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    alive_ratios = ratio[alive_mask].numpy()
    sae_name = sae_path.rstrip("/").split("/")[-1]
    n_bins = max(1, int(1.0 / bin_width))
    weighted = weights is not None

    hist_kwargs: dict = dict(bins=n_bins, range=(0.0, 1.0), color="#4C72B0", edgecolor="white", linewidth=0.3)
    if weighted:
        alive_weights = weights[alive_mask].numpy()
        hist_kwargs["weights"] = alive_weights
    else:
        hist_kwargs["weights"] = np.ones_like(alive_ratios) / max(len(alive_ratios), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(alive_ratios, **hist_kwargs)

    ax.axvline(x=0.2, color="red", linestyle="--", linewidth=1, label="Text-specific (<0.2)")
    ax.axvline(x=0.8, color="blue", linestyle="--", linewidth=1, label="Image-specific (>0.8)")
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    shared_mask = (alive_ratios >= 0.2) & (alive_ratios <= 0.8)
    if weighted:
        w = alive_weights
        n_text = int(w[alive_ratios < 0.2].sum())
        n_shared = int(w[shared_mask].sum())
        n_image = int(w[alive_ratios > 0.8].sum())
        sfp = w[shared_mask].sum() / max(w.sum(), 1e-12) * 100
    else:
        total = max(len(alive_ratios), 1)
        n_text = (alive_ratios < 0.2).sum() / total * 100
        n_shared = shared_mask.sum() / total * 100
        n_image = (alive_ratios > 0.8).sum() / total * 100
        sfp = n_shared

    ax.set_xlabel("Modality Ratio (0=Text-only, 1=Image-only)", fontsize=12)
    ylabel = "Activation-Weighted Count" if weighted else "Proportion of Features"
    ax.set_ylabel(ylabel, fontsize=12)
    weight_tag = ", weighted" if weighted else ""
    ax.set_title(
        f"CLIP SAE Feature Modality Distribution\n{sae_name} (k={k}, alive={len(alive_ratios):,}{weight_tag})",
        fontsize=13,
    )
    ax.legend(fontsize=10)

    if weighted:
        stats_text = f"Text-specific: {n_text:,}\nShared: {n_shared:,}\nImage-specific: {n_image:,}\nSFP: {sfp:.1f}%"
    else:
        stats_text = f"Text-specific: {n_text:.1f}%\nShared: {n_shared:.1f}%\nImage-specific: {n_image:.1f}%\nSFP: {sfp:.1f}%"
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_weighted" if weighted else ""
    filepath = os.path.join(output_dir, f"CLIP_MODALITY_SPLIT_{sae_name}_k{k}{suffix}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Saved histogram to %s", filepath)
    return filepath


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    clip_model, clip_processor, sae, device = load_models(args)

    logger.info("Loading dataset: %s", args.dataset_name)
    dataset = load_dataset(args.dataset_name, split="test")

    latent_size = sae.latent_size
    img_counts = torch.zeros(latent_size, dtype=torch.long, device=device)
    text_counts = torch.zeros(latent_size, dtype=torch.long, device=device)
    total_img_tokens = 0
    total_text_tokens = 0
    mse_img_sum = 0.0
    mse_txt_sum = 0.0
    n_img_passes = 0
    n_txt_passes = 0
    skipped = 0

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in tqdm(indices, desc="Processing samples"):
        sample = dataset[idx]
        try:
            image = sample["image"].convert("RGB")
        except Exception:
            skipped += 1
            continue

        caption = random.choice(sample["sentences_raw"]) if isinstance(sample["sentences_raw"], list) else sample["sentences_raw"]

        # Image pass: CLIP image_features -> SAE encode
        img_inputs = clip_processor(images=image, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**img_inputs)  # (1, d)

        img_emb_3d = img_emb.unsqueeze(0)  # (1, 1, d)
        top_acts, top_indices = vl_encode(sae, img_emb_3d,
                                          visual_mask=torch.ones(1, 1, dtype=torch.bool, device=device))
        img_counts += torch.bincount(top_indices.reshape(-1), minlength=latent_size)
        total_img_tokens += 1
        recon = sae.decode(top_acts, top_indices)
        mse_img_sum += (img_emb_3d - recon).pow(2).mean().item()
        n_img_passes += 1

        # Text pass: CLIP text_features -> SAE encode
        txt_inputs = clip_processor(text=caption, return_tensors="pt", padding=True, truncation=True)
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
        with torch.no_grad():
            txt_emb = clip_model.get_text_features(**txt_inputs)  # (1, d)

        txt_emb_3d = txt_emb.unsqueeze(0)  # (1, 1, d)
        top_acts, top_indices = vl_encode(sae, txt_emb_3d,
                                          visual_mask=torch.zeros(1, 1, dtype=torch.bool, device=device))
        text_counts += torch.bincount(top_indices.reshape(-1), minlength=latent_size)
        total_text_tokens += 1
        recon = sae.decode(top_acts, top_indices)
        mse_txt_sum += (txt_emb_3d - recon).pow(2).mean().item()
        n_txt_passes += 1

    logger.info(
        "Done: %d samples, %d skipped, %d img tokens, %d text tokens",
        num_samples, skipped, total_img_tokens, total_text_tokens,
    )

    ratio_result = compute_modality_ratio(
        img_counts.cpu(), text_counts.cpu(), total_img_tokens, total_text_tokens, latent_size,
    )
    logger.info("SFP (Shared Feature Percentage): %.2f%%", ratio_result["summary"]["shared_feature_percentage"])

    mean_mse_img = mse_img_sum / max(n_img_passes, 1)
    mean_mse_txt = mse_txt_sum / max(n_txt_passes, 1)

    result = {
        "metadata": {
            "clip_model": args.clip_model_name,
            "sae_path": args.sae_path,
            "k": args.k,
            "num_samples": num_samples,
            "total_img_tokens": total_img_tokens,
            "total_text_tokens": total_text_tokens,
            "skipped_samples": skipped,
            "dataset": args.dataset_name,
            "mean_reconstruction_mse_image": mean_mse_img,
            "mean_reconstruction_mse_text": mean_mse_txt,
            "mean_reconstruction_mse": (mean_mse_img + mean_mse_txt) / 2.0,
        },
        "summary": ratio_result["summary"],
        "ratio": ratio_result["ratio"],
        "alive_mask": ratio_result["alive_mask"],
        "raw_counts": {
            "img_counts": img_counts.cpu().tolist(),
            "text_counts": text_counts.cpu().tolist(),
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    sae_name = args.sae_path.rstrip("/").split("/")[-1]
    suffix = "_weighted" if args.weighted else ""
    filepath = os.path.join(args.output_dir, f"CLIP_MODALITY_SPLIT_{sae_name}_k{args.k}{suffix}.json")
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved results to %s", filepath)

    ratio_tensor = torch.tensor([float("nan") if x is None else x for x in ratio_result["ratio"]])
    alive_tensor = torch.tensor(ratio_result["alive_mask"], dtype=torch.bool)
    if args.weighted:
        density_img = img_counts.float() / max(total_img_tokens, 1)
        density_text = text_counts.float() / max(total_text_tokens, 1)
        weights_tensor = (density_img + density_text).cpu()
    else:
        weights_tensor = None
    plot_histogram(
        ratio_tensor, alive_tensor, args.output_dir, args.sae_path, args.k, args.bin_width,
        weights=weights_tensor,
    )


if __name__ == "__main__":
    main()
