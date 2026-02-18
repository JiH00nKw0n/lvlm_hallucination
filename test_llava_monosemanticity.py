"""
Patch-Level Monosemanticity Analysis for SAE Features (LLaVA).

Quantifies whether each SAE feature activates on semantically similar image patches
by collecting activating patches, embedding with DINOv2, and scoring similarity.

Two scoring modes:
    --activation_weighted_score (default): Paper's MS formula (Eq. 7-9) using
        activation-weighted similarity over ALL patches. Efficient O(P*d) trick.
    --no_activation_weighted_score: Top-K patches + unweighted mean off-diagonal cosine sim.

Usage:
    python test_llava_monosemanticity.py --num_samples 10 --top_patches 5 --k 256
    python test_llava_monosemanticity.py --num_samples 10 --no_activation_weighted_score
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
from PIL import Image
from torch import Tensor
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

# LLaVA-Next base image: 336x336 with 14x14 patches → 24x24 grid (576 tokens)
BASE_GRID_SIZE = 24
BASE_PATCH_PX = 14
BASE_IMG_TOKENS = BASE_GRID_SIZE * BASE_GRID_SIZE  # 576


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch-level monosemanticity analysis for SAE features")
    parser.add_argument("--model_name", type=str, default="llava-hf/llama3-llava-next-8b-hf")
    parser.add_argument("--sae_path", type=str, default="lmms-lab/llama3-llava-next-8b-hf-sae-131k")
    parser.add_argument("--dinov2_name", type=str, default="facebook/dinov2-large")
    parser.add_argument("--layer_index", type=int, default=24)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--top_patches", type=int, default=25, help="Number of top activating patches per feature")
    parser.add_argument("--bin_width", type=float, default=0.05, help="Histogram bin width (0~1 range)")
    parser.add_argument("--output_dir", type=str, default="./results/monosemanticity")
    parser.add_argument("--dataset_name", type=str, default="Multimodal-Fatima/COCO_captions_test")
    parser.add_argument("--weighted", action="store_true", help="Weight histogram by activation frequency")
    parser.add_argument("--activation_weighted_score", action=argparse.BooleanOptionalAction, default=True,
                        help="Use paper's activation-weighted MS formula (Eq. 7-9)")
    return parser.parse_args()


def load_models(
    args: argparse.Namespace,
) -> tuple:
    """Load LLaVA-Next, SAE, DINOv2, and processors. Returns (model, processor, sae, dinov2, dino_processor, device)."""
    from transformers import AutoImageProcessor, AutoProcessor, Dinov2Model, LlavaNextForConditionalGeneration

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    # LLaVA-Next
    logger.info("Loading model: %s", args.model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    logger.info("Loading processor: %s", args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)

    # SAE
    logger.info("Loading SAE: %s", args.sae_path)
    if args.sae_path == "lmms-lab/llama3-llava-next-8b-hf-sae-131k":
        import sys
        _project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(_project_root, "multimodal-sae", "train", "sae"))
        from sae.sae import Sae
        hookpoint = f"model.layers.{args.layer_index}"
        sae = Sae.load_from_hub(args.sae_path, hookpoint=hookpoint, device=device)
    else:
        from transformers import PretrainedConfig

        _SAE_ARCH_MAP = {
            "TopKSAE": TopKSAE,
            "VLTopKSAE": VLTopKSAE,
            "BatchTopKSAE": BatchTopKSAE,
            "VLBatchTopKSAE": VLBatchTopKSAE,
            "MatryoshkaSAE": MatryoshkaSAE,
            "VLMatryoshkaSAE": VLMatryoshkaSAE,
        }

        sae_config = PretrainedConfig.from_pretrained(args.sae_path)
        arch_name = getattr(sae_config, "architectures", ["TopKSAE"])[0]
        sae_cls = _SAE_ARCH_MAP.get(arch_name, TopKSAE)
        logger.info("Resolved SAE architecture: %s -> %s", arch_name, sae_cls.__name__)
        sae = sae_cls.from_pretrained(args.sae_path)
    logger.info("Setting SAE k = %d (user-specified)", args.k)
    sae.cfg.k = args.k
    sae.to(device)
    sae.eval()
    sae.requires_grad_(False)

    if not hasattr(sae, "latent_size"):
        if hasattr(sae, "latent_size_total"):
            sae.latent_size = sae.latent_size_total
        elif hasattr(sae, "num_latents"):
            sae.latent_size = sae.num_latents

    # DINOv2
    logger.info("Loading DINOv2: %s", args.dinov2_name)
    dinov2 = Dinov2Model.from_pretrained(args.dinov2_name, torch_dtype=dtype)
    dinov2.to(device)
    dinov2.eval()
    dinov2.requires_grad_(False)

    dino_processor = AutoImageProcessor.from_pretrained(args.dinov2_name)

    return model, processor, sae, dinov2, dino_processor, device


def load_coco_dataset(args: argparse.Namespace):
    """Load COCO captions dataset."""
    logger.info("Loading dataset: %s", args.dataset_name)
    return load_dataset(args.dataset_name, split="test")


def get_patch_box(
    patch_pos: int, grid_size: int = BASE_GRID_SIZE, patch_px: int = BASE_PATCH_PX,
) -> tuple[int, int, int, int]:
    """Convert patch position index to crop box (left, upper, right, lower) on the base image."""
    row = patch_pos // grid_size
    col = patch_pos % grid_size
    left = col * patch_px
    upper = row * patch_px
    return (left, upper, left + patch_px, upper + patch_px)


def collect_top_patches(
    dataset,
    model,
    processor,
    sae,
    args: argparse.Namespace,
    device: torch.device,
    collect_all_activations: bool = False,
) -> tuple[dict[int, list[tuple[float, int, int]]], dict[int, int], float, int,
           Optional[dict[int, list[tuple[float, int, int]]]], Optional[set[int]]]:
    """Collect top activating patches per feature across the dataset.

    Returns:
        top_buffer: feature_idx -> list of (activation, dataset_idx, patch_pos),
                    maintained as a max-heap of size `top_patches`.
        feature_freq: feature_idx -> total positive activation count.
        mean_mse_image: average reconstruction MSE over all image passes.
        n_passes: number of successfully processed images.
        act_per_feature: (if collect_all_activations) feature_idx -> ALL (act, ds_idx, patch_pos).
        processed_images: (if collect_all_activations) set of all successfully processed dataset indices.
    """
    top_buffer: dict[int, list[tuple[float, int, int]]] = {}
    feature_freq: dict[int, int] = {}
    act_per_feature: Optional[dict[int, list[tuple[float, int, int]]]] = {} if collect_all_activations else None
    processed_images: Optional[set[int]] = set() if collect_all_activations else None
    top_patches = args.top_patches
    mse_sum = 0.0
    n_passes = 0

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    skipped = 0

    for dataset_idx in tqdm(indices, desc="Collecting patches"):
        sample = dataset[dataset_idx]
        image = sample["image"]

        try:
            image = image.convert("RGB")
        except Exception:
            skipped += 1
            continue

        conversation = [{"role": "user", "content": [{"type": "image"}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )

        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            skipped += 1
            continue

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs.get("attention_mask").to(device)
                if inputs.get("attention_mask") is not None
                else None,
                pixel_values=inputs.get("pixel_values").to(device)
                if inputs.get("pixel_values") is not None
                else None,
                image_sizes=inputs.get("image_sizes"),
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states[args.layer_index]  # (1, seq_len, hidden_size)
        visual_mask = mm_token_type_ids.bool().to(device)  # (1, seq_len)

        img_hidden = hidden_states[visual_mask].unsqueeze(0)  # (1, n_img, hidden_size)
        n_img = img_hidden.shape[1]
        if n_img == 0:
            skipped += 1
            continue

        # Restrict to base image tokens (first 576)
        if n_img > BASE_IMG_TOKENS:
            img_hidden = img_hidden[:, :BASE_IMG_TOKENS, :]
            n_img = BASE_IMG_TOKENS

        top_acts, top_indices = sae.encode(img_hidden)  # (1, n_img, k)

        recon = sae.decode(top_acts, top_indices)
        mse_sum += (img_hidden - recon).pow(2).mean().item()
        n_passes += 1
        if processed_images is not None:
            processed_images.add(dataset_idx)

        # Flatten to (n_img * k,) for processing
        acts_flat = top_acts[0].reshape(-1).cpu()       # (n_img * k,)
        indices_flat = top_indices[0].reshape(-1).cpu()  # (n_img * k,)
        k = top_acts.shape[-1]

        # Compute patch position for each entry: token_pos = flat_idx // k
        patch_positions = torch.arange(n_img).unsqueeze(1).expand(n_img, k).reshape(-1)

        for act_val, feat_idx, patch_pos in zip(
            acts_flat.tolist(), indices_flat.tolist(), patch_positions.tolist(),
        ):
            if act_val <= 0:
                continue

            feat_idx = int(feat_idx)
            feature_freq[feat_idx] = feature_freq.get(feat_idx, 0) + 1
            entry = (act_val, dataset_idx, patch_pos)

            # Always maintain top-K buffer (used for fallback and visualization)
            if feat_idx not in top_buffer:
                top_buffer[feat_idx] = []
            heap = top_buffer[feat_idx]
            if len(heap) < top_patches:
                heapq.heappush(heap, entry)
            elif act_val > heap[0][0]:
                heapq.heapreplace(heap, entry)

            # Collect ALL activations for weighted scoring
            if act_per_feature is not None:
                if feat_idx not in act_per_feature:
                    act_per_feature[feat_idx] = []
                act_per_feature[feat_idx].append(entry)

    mean_mse_image = mse_sum / max(n_passes, 1)
    logger.info(
        "Collection done: %d samples processed, %d skipped, %d features found, mean_mse=%.6f",
        num_samples, skipped, len(top_buffer), mean_mse_image,
    )
    return top_buffer, feature_freq, mean_mse_image, n_passes, act_per_feature, processed_images


def _build_patch_embedding_cache(
    needed_images: set[int],
    dataset,
    dinov2,
    dino_processor,
    device: torch.device,
    batch_size: int = 32,
) -> dict[int, Tensor]:
    """Run DINOv2 on full images at LLaVA resolution and cache patch token embeddings.

    Uses 336×336 input (matching LLaVA's 24×24 grid with patch_size=14) so that
    DINOv2 patch positions map 1:1 to LLaVA patch positions.

    Returns:
        cache: dataset_idx → (576, hidden_dim) patch embeddings
    """
    dino_img_size = BASE_GRID_SIZE * BASE_PATCH_PX  # 336 — match LLaVA base grid

    cache: dict[int, Tensor] = {}
    image_list = sorted(needed_images)

    for batch_start in tqdm(range(0, len(image_list), batch_size), desc="DINOv2 images"):
        batch_indices = image_list[batch_start : batch_start + batch_size]
        images = [dataset[idx]["image"].convert("RGB") for idx in batch_indices]

        dino_inputs = dino_processor(
            images=images,
            return_tensors="pt",
            do_resize=True,
            size={"height": dino_img_size, "width": dino_img_size},
            do_center_crop=False,
        )
        dino_inputs = {k: v.to(device) for k, v in dino_inputs.items()}

        with torch.no_grad():
            dino_out = dinov2(**dino_inputs)

        # last_hidden_state: (batch, 1 + 576, hidden_dim) — skip CLS at idx 0
        patch_tokens = dino_out.last_hidden_state[:, 1:, :]  # (batch, 576, hidden_dim)

        for i, idx in enumerate(batch_indices):
            cache[idx] = patch_tokens[i]  # (576, hidden_dim)

    return cache


def score_features(
    top_buffer: dict[int, list[tuple[float, int, int]]],
    dataset,
    dinov2,
    dino_processor,
    device: torch.device,
) -> dict[int, float]:
    """Score each feature by DINOv2 pairwise cosine similarity of its top patches.

    Processes full images through DINOv2 to obtain context-aware patch token
    embeddings (with self-attention across the entire image), then maps LLaVA
    patch positions to the corresponding DINOv2 patch tokens.

    Returns:
        scores: feature_idx → mean off-diagonal cosine similarity
    """
    scores: dict[int, float] = {}

    # Filter to features with at least 2 patches
    scorable = {k: v for k, v in top_buffer.items() if len(v) >= 2}
    logger.info("Scoring %d features (of %d total with >= 2 patches)", len(scorable), len(top_buffer))

    # Collect unique images needed across all scorable features
    needed_images: set[int] = {
        dataset_idx
        for entries in scorable.values()
        for _, dataset_idx, _ in entries
    }
    logger.info("Computing DINOv2 patch embeddings for %d unique images", len(needed_images))

    patch_cache = _build_patch_embedding_cache(
        needed_images, dataset, dinov2, dino_processor, device,
    )
    logger.info("Cached %d image embeddings (24x24 grid, 1:1 with LLaVA), scoring features", len(patch_cache))

    # Score each feature using cached patch-token embeddings
    for feat_idx, entries in tqdm(scorable.items(), desc="Scoring features"):
        embeddings: list[Tensor] = []

        for _act_val, dataset_idx, patch_pos in entries:
            # 1:1 mapping — DINOv2 at 336×336 has same 24×24 grid as LLaVA
            embeddings.append(patch_cache[dataset_idx][patch_pos])  # (hidden_dim,)

        emb_matrix = torch.stack(embeddings, dim=0)  # (n, hidden_dim)
        emb_matrix = F.normalize(emb_matrix, dim=-1)
        sim_matrix = emb_matrix @ emb_matrix.T  # (n, n)

        # Off-diagonal mean
        n = sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        score = sim_matrix[mask].mean().item()
        scores[feat_idx] = score

    return scores


def score_features_weighted(
    act_per_feature: dict[int, list[tuple[float, int, int]]],
    processed_images: set[int],
    dataset,
    dinov2,
    dino_processor,
    device: torch.device,
) -> dict[int, float]:
    """Score features using paper's activation-weighted MS formula (Eq. 7-9).

    Uses efficient O(P*d) trick per feature:
        MS^k = (||sum(a_tilde_n * e_n)||^2 - ||a_tilde||^2) / ((sum(a_tilde))^2 - ||a_tilde||^2)
    avoiding the N*N similarity matrix.

    Here each "n" is a patch (not an image), and e_n is the DINOv2 patch embedding.
    """
    eps = 1e-8
    scores: dict[int, float] = {}

    scorable = {k: v for k, v in act_per_feature.items() if len(v) >= 2}
    logger.info(
        "Weighted scoring: %d features (of %d total with >= 2 activations)",
        len(scorable), len(act_per_feature),
    )

    # Build DINOv2 cache for ALL processed images
    logger.info("Computing DINOv2 patch embeddings for %d images (all processed)", len(processed_images))
    patch_cache = _build_patch_embedding_cache(
        processed_images, dataset, dinov2, dino_processor, device,
    )
    logger.info("Cached %d image embeddings, scoring features with weighted formula", len(patch_cache))

    for feat_idx, activations in tqdm(scorable.items(), desc="Scoring features (weighted)"):
        acts = torch.tensor([a for a, _, _ in activations], device=device)
        embeddings = torch.stack([
            patch_cache[ds_idx][patch_pos]
            for _, ds_idx, patch_pos in activations
        ])  # (P, hidden_dim)
        embeddings = F.normalize(embeddings, dim=-1)

        # Eq. 7: min-max normalization
        a_min, a_max = acts.min(), acts.max()
        a_tilde = (acts - a_min) / (a_max - a_min + eps)

        # Efficient Eq. 8+9: ã^T S ã = ||Σ ã_n e_n||²
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
        low_mask = score_values < 0.3
        mid_mask = (score_values >= 0.3) & (score_values <= 0.7)
        high_mask = score_values > 0.7
        n_low = int(w[low_mask].sum())
        n_mid = int(w[mid_mask].sum())
        n_high = int(w[high_mask].sum())
    else:
        n_low = int((score_values < 0.3).sum())
        n_mid = int(((score_values >= 0.3) & (score_values <= 0.7)).sum())
        n_high = int((score_values > 0.7).sum())

    ax.set_xlabel("DINOv2 Cosine Similarity Score", fontsize=12)
    ylabel = "Activation-Weighted Count" if weighted else "Number of Features"
    ax.set_ylabel(ylabel, fontsize=12)
    weight_tag = ", weighted" if weighted else ""
    ax.set_title(
        f"SAE Feature Monosemanticity (Patch-Level)\n{sae_name} (k={k}, scored={len(score_values):,}{weight_tag})",
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
    suffix = "_weighted" if weighted else ""
    filepath = os.path.join(output_dir, f"MONOSEMANTICITY_{sae_name}_k{k}{suffix}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Saved histogram to %s", filepath)
    return filepath


def save_results(
    scores: dict[int, float],
    top_buffer: dict[int, list[tuple[float, int, int]]],
    args: argparse.Namespace,
    mean_mse_image: float = 0.0,
) -> str:
    """Save results to JSON. Returns the output file path."""
    import numpy as np

    score_values = np.array(list(scores.values())) if scores else np.array([])

    sae_name = args.sae_path.rstrip("/").split("/")[-1]

    # Compute top 20% mean
    top20pct_mean = 0.0
    if len(score_values) > 0:
        threshold_idx = max(1, int(len(score_values) * 0.2))
        sorted_scores = np.sort(score_values)[::-1]
        top20pct_mean = float(sorted_scores[:threshold_idx].mean())

    result = {
        "metadata": {
            "model": args.model_name,
            "sae_path": args.sae_path,
            "dinov2": args.dinov2_name,
            "layer_index": args.layer_index,
            "k": args.k,
            "top_patches": args.top_patches,
            "num_samples": args.num_samples,
            "dataset": args.dataset_name,
            "mean_reconstruction_mse_image": mean_mse_image,
            "mean_reconstruction_mse": mean_mse_image,
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
        "top_patches_meta": {
            str(feat_idx): [
                {"act": round(act, 6), "dataset_idx": ds_idx, "patch_pos": pp}
                for act, ds_idx, pp in sorted(entries, key=lambda x: -x[0])
            ]
            for feat_idx, entries in top_buffer.items()
            if feat_idx in scores
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_weighted" if args.weighted else ""
    filename = f"MONOSEMANTICITY_{sae_name}_k{args.k}{suffix}.json"
    filepath = os.path.join(args.output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved results to %s", filepath)
    return filepath


def visualize_top_patches(
    feat_idx: int,
    entries: list[tuple[float, int, int]],
    dataset,
    save_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """Draw red bounding boxes on top-K patches for a given feature. Returns composite image."""
    from PIL import ImageDraw

    if not entries:
        return None

    n_cols = min(5, len(entries))
    n_rows = (len(entries) + n_cols - 1) // n_cols
    cell_size = 336
    canvas = Image.new("RGB", (n_cols * cell_size, n_rows * cell_size), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (act_val, dataset_idx, patch_pos) in enumerate(sorted(entries, key=lambda x: -x[0])):
        sample = dataset[dataset_idx]
        image = sample["image"].convert("RGB").resize((cell_size, cell_size))
        box = get_patch_box(patch_pos)

        img_draw = ImageDraw.Draw(image)
        img_draw.rectangle(box, outline="red", width=2)

        row, col = divmod(i, n_cols)
        canvas.paste(image, (col * cell_size, row * cell_size))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        canvas.save(save_path)
        logger.info("Saved visualization to %s", save_path)

    return canvas


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, processor, sae, dinov2, dino_processor, device = load_models(args)
    dataset = load_coco_dataset(args)

    # Phase 2: Collect patches per feature
    use_weighted = args.activation_weighted_score
    logger.info("=== Phase 2: Collecting patches (weighted=%s) ===", use_weighted)
    top_buffer, feature_freq, mean_mse_image, n_passes, act_per_feature, processed_images = collect_top_patches(
        dataset, model, processor, sae, args, device,
        collect_all_activations=use_weighted,
    )

    # Free LLaVA memory before DINOv2 scoring
    del model
    torch.cuda.empty_cache()
    logger.info("Released LLaVA model memory")

    # Phase 3: Score features with DINOv2
    logger.info("=== Phase 3: Scoring features with DINOv2 (weighted=%s) ===", use_weighted)
    if use_weighted:
        scores = score_features_weighted(act_per_feature, processed_images, dataset, dinov2, dino_processor, device)
    else:
        scores = score_features(top_buffer, dataset, dinov2, dino_processor, device)

    # Phase 4: Save results
    logger.info("=== Phase 4: Saving results ===")
    save_results(scores, top_buffer, args, mean_mse_image=mean_mse_image)
    weights_dict = {fi: feature_freq.get(fi, 0) / max(n_passes, 1) for fi in scores} if args.weighted else None
    plot_histogram(scores, args.output_dir, args.sae_path, args.k, args.bin_width, weights=weights_dict)

    # Summary
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
