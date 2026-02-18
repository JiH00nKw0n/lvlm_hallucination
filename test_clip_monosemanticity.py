"""
Image-Level Monosemanticity Analysis for CLIP SAE Features.

Quantifies whether each SAE feature activates on semantically similar images
using CLIP image embeddings and DINOv2 CLS token similarity.

Two scoring modes:
    --activation_weighted_score (default): Paper's MS formula (Eq. 7-9) using
        activation-weighted similarity over ALL images. Efficient O(N*d) trick.
    --no_activation_weighted_score: Top-K images + unweighted mean off-diagonal cosine sim.

CLIP operates on single image vectors, so DINOv2 uses CLS token (not patches).

Usage:
    python test_clip_monosemanticity.py --sae_path Mayfull/CLIP_TopKSAE --num_samples 5000
    python test_clip_monosemanticity.py --sae_path Mayfull/CLIP_TopKSAE --no_activation_weighted_score
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import random
from typing import Optional

import torch
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="Image-level monosemanticity analysis for CLIP SAE features")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--dinov2_name", type=str, default="facebook/dinov2-large")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--top_images", type=int, default=25, help="Number of top activating images per feature (for unweighted mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output_dir", type=str, default="./results/clip_monosemanticity")
    parser.add_argument("--dataset_name", type=str, default="Multimodal-Fatima/COCO_captions_test")
    parser.add_argument("--bin_width", type=float, default=0.05, help="Histogram bin width (0~1 range)")
    parser.add_argument("--weighted", action="store_true", help="Weight histogram by activation frequency")
    parser.add_argument("--activation_weighted_score", action=argparse.BooleanOptionalAction, default=False,
                        help="Use paper's activation-weighted MS formula (Eq. 7-9)")
    return parser.parse_args()


def load_models(args: argparse.Namespace) -> tuple:
    """Load CLIP, SAE, DINOv2. Returns (clip_model, clip_processor, sae, dinov2, dino_processor, device)."""
    from transformers import AutoImageProcessor, CLIPModel, CLIPProcessor, Dinov2Model, PretrainedConfig

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

    logger.info("Loading DINOv2: %s", args.dinov2_name)
    dinov2 = Dinov2Model.from_pretrained(args.dinov2_name, torch_dtype=dtype)
    dinov2.to(device)
    dinov2.eval()
    dinov2.requires_grad_(False)

    dino_processor = AutoImageProcessor.from_pretrained(args.dinov2_name)

    return clip_model, clip_processor, sae, dinov2, dino_processor, device


def collect_activations(
    dataset,
    clip_model,
    clip_processor,
    sae,
    args: argparse.Namespace,
    device: torch.device,
    collect_all: bool = False,
) -> tuple[dict[int, list[tuple[float, int]]], dict[int, int], float, int,
           Optional[dict[int, list[tuple[float, int]]]], Optional[set[int]]]:
    """Collect SAE activations from CLIP image embeddings.

    Returns:
        top_buffer: feature_idx -> list of (activation, dataset_idx), max-heap of top_images.
        feature_freq: feature_idx -> total positive activation count.
        mean_mse: average reconstruction MSE.
        n_passes: number of successfully processed images.
        act_per_feature: (if collect_all) feature_idx -> ALL (activation, dataset_idx).
        processed_images: (if collect_all) set of processed dataset indices.
    """
    top_buffer: dict[int, list[tuple[float, int]]] = {}
    feature_freq: dict[int, int] = {}
    act_per_feature: Optional[dict[int, list[tuple[float, int]]]] = {} if collect_all else None
    processed_images: Optional[set[int]] = set() if collect_all else None
    top_images = args.top_images
    mse_sum = 0.0
    n_passes = 0
    skipped = 0

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    for dataset_idx in tqdm(indices, desc="Collecting activations"):
        sample = dataset[dataset_idx]
        try:
            image = sample["image"].convert("RGB")
        except Exception:
            skipped += 1
            continue

        inputs = clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            img_emb = clip_model.get_image_features(**inputs)  # (1, d)

        img_emb_3d = img_emb.unsqueeze(0)  # (1, 1, d)
        top_acts, top_indices = vl_encode(sae, img_emb_3d,
                                          visual_mask=torch.ones(1, 1, dtype=torch.bool, device=device))

        recon = sae.decode(top_acts, top_indices)
        mse_sum += (img_emb_3d - recon).pow(2).mean().item()
        n_passes += 1
        if processed_images is not None:
            processed_images.add(dataset_idx)

        acts_flat = top_acts[0, 0].cpu()       # (k,)
        indices_flat = top_indices[0, 0].cpu()  # (k,)

        for act_val, feat_idx in zip(acts_flat.tolist(), indices_flat.tolist()):
            if act_val <= 0:
                continue

            feat_idx = int(feat_idx)
            feature_freq[feat_idx] = feature_freq.get(feat_idx, 0) + 1
            entry = (act_val, dataset_idx)

            # Top-K buffer
            if feat_idx not in top_buffer:
                top_buffer[feat_idx] = []
            heap = top_buffer[feat_idx]
            if len(heap) < top_images:
                heapq.heappush(heap, entry)
            elif act_val > heap[0][0]:
                heapq.heapreplace(heap, entry)

            # All activations
            if act_per_feature is not None:
                if feat_idx not in act_per_feature:
                    act_per_feature[feat_idx] = []
                act_per_feature[feat_idx].append(entry)

    mean_mse = mse_sum / max(n_passes, 1)
    logger.info(
        "Collection done: %d processed, %d skipped, %d features found, mean_mse=%.6f",
        n_passes, skipped, len(top_buffer), mean_mse,
    )
    return top_buffer, feature_freq, mean_mse, n_passes, act_per_feature, processed_images


def _build_cls_embedding_cache(
    needed_images: set[int],
    dataset,
    dinov2,
    dino_processor,
    device: torch.device,
    batch_size: int = 64,
) -> dict[int, Tensor]:
    """Run DINOv2 on images and cache CLS token embeddings.

    Returns:
        cache: dataset_idx -> (hidden_dim,) CLS embedding
    """
    cache: dict[int, Tensor] = {}
    image_list = sorted(needed_images)

    for batch_start in tqdm(range(0, len(image_list), batch_size), desc="DINOv2 CLS embeddings"):
        batch_indices = image_list[batch_start : batch_start + batch_size]
        images = [dataset[idx]["image"].convert("RGB") for idx in batch_indices]

        dino_inputs = dino_processor(images=images, return_tensors="pt")
        dino_inputs = {k: v.to(device) for k, v in dino_inputs.items()}

        with torch.no_grad():
            dino_out = dinov2(**dino_inputs)

        # CLS token is at index 0
        cls_tokens = dino_out.last_hidden_state[:, 0, :]  # (batch, hidden_dim)

        for i, idx in enumerate(batch_indices):
            cache[idx] = cls_tokens[i]  # (hidden_dim,)

    return cache


def score_features_topk(
    top_buffer: dict[int, list[tuple[float, int]]],
    dataset,
    dinov2,
    dino_processor,
    device: torch.device,
) -> dict[int, float]:
    """Score features by DINOv2 CLS pairwise cosine similarity (top-K, unweighted)."""
    scores: dict[int, float] = {}

    scorable = {k: v for k, v in top_buffer.items() if len(v) >= 2}
    logger.info("Scoring %d features (top-K unweighted, >= 2 images)", len(scorable))

    needed_images: set[int] = {ds_idx for entries in scorable.values() for _, ds_idx in entries}
    logger.info("Computing DINOv2 CLS embeddings for %d unique images", len(needed_images))

    cls_cache = _build_cls_embedding_cache(needed_images, dataset, dinov2, dino_processor, device)

    for feat_idx, entries in tqdm(scorable.items(), desc="Scoring features (top-K)"):
        embeddings = torch.stack([cls_cache[ds_idx] for _, ds_idx in entries])  # (n, d)
        embeddings = F.normalize(embeddings, dim=-1)
        sim_matrix = embeddings @ embeddings.T

        n = sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        scores[feat_idx] = sim_matrix[mask].mean().item()

    return scores


def score_features_weighted(
    act_per_feature: dict[int, list[tuple[float, int]]],
    processed_images: set[int],
    dataset,
    dinov2,
    dino_processor,
    device: torch.device,
) -> dict[int, float]:
    """Score features using paper's activation-weighted MS formula (Eq. 7-9).

    Efficient O(N*d) per feature using CLS embeddings:
        MS^k = (||sum(a_tilde_n * e_n)||^2 - ||a_tilde||^2) / ((sum(a_tilde))^2 - ||a_tilde||^2)
    """
    eps = 1e-8
    scores: dict[int, float] = {}

    scorable = {k: v for k, v in act_per_feature.items() if len(v) >= 2}
    logger.info("Weighted scoring: %d features (of %d with >= 2 activations)", len(scorable), len(act_per_feature))

    logger.info("Computing DINOv2 CLS embeddings for %d images (all processed)", len(processed_images))
    cls_cache = _build_cls_embedding_cache(processed_images, dataset, dinov2, dino_processor, device)

    for feat_idx, activations in tqdm(scorable.items(), desc="Scoring features (weighted)"):
        acts = torch.tensor([a for a, _ in activations], device=device)
        embeddings = torch.stack([cls_cache[ds_idx] for _, ds_idx in activations])  # (N, d)
        embeddings = F.normalize(embeddings, dim=-1)

        # Eq. 7: min-max normalization
        a_min, a_max = acts.min(), acts.max()
        a_tilde = (acts - a_min) / (a_max - a_min + eps)

        # Efficient Eq. 8+9
        weighted_emb = (a_tilde.unsqueeze(-1) * embeddings).sum(dim=0)  # (d,)
        aTSa = weighted_emb @ weighted_emb  # scalar
        aTa = (a_tilde ** 2).sum()
        sum_a = a_tilde.sum()

        numerator = aTSa - aTa
        denominator = (sum_a ** 2 - aTa).clamp(min=eps)
        scores[feat_idx] = (numerator / denominator).item()

    return scores


def plot_histogram(
    scores: dict[int, float],
    output_dir: str,
    sae_path: str,
    k: int,
    bin_width: float = 0.05,
    weights: Optional[dict[int, float]] = None,
    activation_weighted: bool = True,
) -> str:
    """Plot monosemanticity score histogram. Returns the output file path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    feat_indices = list(scores.keys())
    score_values = np.array([scores[fi] for fi in feat_indices])
    sae_name = sae_path.rstrip("/").split("/")[-1]
    n_bins = max(1, int(1.0 / bin_width))
    weighted = weights is not None

    hist_kwargs: dict = dict(bins=n_bins, range=(0.0, 1.0), color="#4C72B0", edgecolor="white", linewidth=0.3)
    if weighted:
        weights_np = np.array([weights.get(fi, 0.0) for fi in feat_indices])
        hist_kwargs["weights"] = weights_np

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(score_values, **hist_kwargs)

    ax.axvline(x=0.3, color="red", linestyle="--", linewidth=1, label="Low mono (<0.3)")
    ax.axvline(x=0.7, color="green", linestyle="--", linewidth=1, label="High mono (>0.7)")

    if weighted:
        w = weights_np
        n_low = int(w[score_values < 0.3].sum())
        n_mid = int(w[(score_values >= 0.3) & (score_values <= 0.7)].sum())
        n_high = int(w[score_values > 0.7].sum())
    else:
        n_low = int((score_values < 0.3).sum())
        n_mid = int(((score_values >= 0.3) & (score_values <= 0.7)).sum())
        n_high = int((score_values > 0.7).sum())

    ax.set_xlabel("DINOv2 CLS Cosine Similarity Score", fontsize=12)
    ylabel = "Activation-Weighted Count" if weighted else "Number of Features"
    ax.set_ylabel(ylabel, fontsize=12)
    weight_tag = ", weighted" if weighted else ""
    ax.set_title(
        f"CLIP SAE Feature Monosemanticity (Image-Level)\n{sae_name} (k={k}, scored={len(score_values):,}{weight_tag})",
        fontsize=13,
    )
    ax.legend(fontsize=10)

    stats_text = (
        f"Low (<0.3): {n_low:,}\n"
        f"Mid (0.3-0.7): {n_mid:,}\n"
        f"High (>0.7): {n_high:,}\n"
        f"Mean: {score_values.mean():.3f}"
    )
    ax.text(
        0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    score_suffix = "_non_weighted" if not activation_weighted else ""
    suffix = "_weighted" if weighted else ""
    filepath = os.path.join(output_dir, f"CLIP_MONOSEMANTICITY_{sae_name}_k{k}{score_suffix}{suffix}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Saved histogram to %s", filepath)
    return filepath


def save_results(
    scores: dict[int, float],
    top_buffer: dict[int, list[tuple[float, int]]],
    args: argparse.Namespace,
    mean_mse: float = 0.0,
) -> str:
    """Save results to JSON. Returns the output file path."""
    import numpy as np

    score_values = np.array(list(scores.values())) if scores else np.array([])
    sae_name = args.sae_path.rstrip("/").split("/")[-1]

    top20pct_mean = 0.0
    if len(score_values) > 0:
        threshold_idx = max(1, int(len(score_values) * 0.2))
        sorted_scores = np.sort(score_values)[::-1]
        top20pct_mean = float(sorted_scores[:threshold_idx].mean())

    result = {
        "metadata": {
            "clip_model": args.clip_model_name,
            "sae_path": args.sae_path,
            "dinov2": args.dinov2_name,
            "k": args.k,
            "top_images": args.top_images,
            "num_samples": args.num_samples,
            "dataset": args.dataset_name,
            "mean_reconstruction_mse": mean_mse,
            "activation_weighted_score": args.activation_weighted_score,
        },
        "summary": {
            "num_scored_features": len(scores),
            "num_high_mono": int((score_values > 0.7).sum()) if len(score_values) > 0 else 0,
            "num_low_mono": int((score_values < 0.3).sum()) if len(score_values) > 0 else 0,
            "mean_score": float(score_values.mean()) if len(score_values) > 0 else 0.0,
            "median_score": float(np.median(score_values)) if len(score_values) > 0 else 0.0,
            "std_score": float(score_values.std()) if len(score_values) > 0 else 0.0,
            "top20pct_mean_score": top20pct_mean,
        },
        "scores": {str(k): round(v, 6) for k, v in scores.items()},
        "top_images_meta": {
            str(feat_idx): [
                {"act": round(act, 6), "dataset_idx": ds_idx}
                for act, ds_idx in sorted(entries, key=lambda x: -x[0])
            ]
            for feat_idx, entries in top_buffer.items()
            if feat_idx in scores
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    score_suffix = "_non_weighted" if not args.activation_weighted_score else ""
    suffix = "_weighted" if args.weighted else ""
    filename = f"CLIP_MONOSEMANTICITY_{sae_name}_k{args.k}{score_suffix}{suffix}.json"
    filepath = os.path.join(args.output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved results to %s", filepath)
    return filepath


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    clip_model, clip_processor, sae, dinov2, dino_processor, device = load_models(args)

    logger.info("Loading dataset: %s", args.dataset_name)
    dataset = load_dataset(args.dataset_name, split="test")

    # Phase 1: Collect activations
    use_weighted = args.activation_weighted_score
    logger.info("=== Phase 1: Collecting activations (weighted=%s) ===", use_weighted)
    top_buffer, feature_freq, mean_mse, n_passes, act_per_feature, processed_images = collect_activations(
        dataset, clip_model, clip_processor, sae, args, device,
        collect_all=use_weighted,
    )

    # Free CLIP memory before DINOv2 scoring
    del clip_model
    torch.cuda.empty_cache()
    logger.info("Released CLIP model memory")

    # Phase 2: Score features with DINOv2
    logger.info("=== Phase 2: Scoring features with DINOv2 (weighted=%s) ===", use_weighted)
    if use_weighted:
        scores = score_features_weighted(act_per_feature, processed_images, dataset, dinov2, dino_processor, device)
    else:
        scores = score_features_topk(top_buffer, dataset, dinov2, dino_processor, device)

    # Phase 3: Save results
    logger.info("=== Phase 3: Saving results ===")
    save_results(scores, top_buffer, args, mean_mse=mean_mse)
    weights_dict = {fi: feature_freq.get(fi, 0) / max(n_passes, 1) for fi in scores} if args.weighted else None
    plot_histogram(scores, args.output_dir, args.sae_path, args.k, args.bin_width,
                   weights=weights_dict, activation_weighted=args.activation_weighted_score)

    if scores:
        import numpy as np
        vals = np.array(list(scores.values()))
        logger.info(
            "Summary: %d scored features, mean=%.3f, median=%.3f, high(>0.7)=%d, low(<0.3)=%d",
            len(vals), vals.mean(), float(np.median(vals)),
            (vals > 0.7).sum(), (vals < 0.3).sum(),
        )


if __name__ == "__main__":
    main()
