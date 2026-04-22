"""Shared helpers for real-data downstream evaluation.

Covers:
  * load_sae — dispatch one-SAE vs two-SAE checkpoints by method name.
  * build_perm — post-hoc Hungarian matching for `ours` (text → image slot map).
  * encode_image / encode_text — per-method latent extractor; for `ours`
    permutes the text-side latent so matched slots share index.
  * load_pair_dataset — dataset factory for COCO / ImageNet caches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore
from src.datasets.cached_imagenet_pairs import CachedImageNetPairsDataset  # type: ignore
from src.models.modeling_sae import TopKSAE, TwoSidedTopKSAE  # type: ignore

logger = logging.getLogger(__name__)

Method = Literal["shared", "separated", "aux", "ours"]
Dataset = Literal["coco", "imagenet"]


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_sae(ckpt_dir: str | Path, method: Method):
    """Load a saved SAE checkpoint.

    For method ∈ {shared, aux}: returns a TopKSAE.
    For method ∈ {separated, ours}: returns a TwoSidedTopKSAE.
    """
    ckpt_dir = str(ckpt_dir)
    if method in ("shared", "aux"):
        return TopKSAE.from_pretrained(ckpt_dir)
    if method in ("separated", "ours"):
        return TwoSidedTopKSAE.from_pretrained(ckpt_dir)
    raise ValueError(f"unknown method {method!r}")


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------

def load_pair_dataset(
    cache_dir: str | Path,
    dataset: Dataset,
    split: str,
    max_per_class: int | None = None,
):
    if dataset == "coco":
        return CachedClipPairsDataset(cache_dir, split=split, l2_normalize=True)
    if dataset == "imagenet":
        return CachedImageNetPairsDataset(
            cache_dir, split=split,  # type: ignore
            max_per_class=max_per_class, l2_normalize=True,
        )
    raise ValueError(f"unknown dataset {dataset!r}")


# ----------------------------------------------------------------------
# Hungarian matching (Ours)
# ----------------------------------------------------------------------

@torch.no_grad()
def _stream_dense_latents(sae: TopKSAE, embeds: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode a tensor of embeddings (N, H) to dense latents (N, L)."""
    sae.eval()
    sae.to(device)  # type: ignore[arg-type]
    out = torch.empty(embeds.shape[0], int(sae.latent_size), dtype=torch.float32)
    for s in range(0, embeds.shape[0], batch_size):
        chunk = embeds[s:s + batch_size].to(device).unsqueeze(1)
        z = sae(hidden_states=chunk, return_dense_latents=True).dense_latents.squeeze(1)
        out[s:s + chunk.shape[0]] = z.float().cpu()
    return out


def _pearson_C(z_i: torch.Tensor, z_t: torch.Tensor) -> np.ndarray:
    """Pearson correlation matrix between two dense latent tensors (N, L)."""
    zi = z_i - z_i.mean(dim=0, keepdim=True)
    zt = z_t - z_t.mean(dim=0, keepdim=True)
    norm_i = zi.norm(dim=0, keepdim=True).clamp_min(1e-12)
    norm_t = zt.norm(dim=0, keepdim=True).clamp_min(1e-12)
    C = (zi / norm_i).T @ (zt / norm_t)
    return C.cpu().numpy()


@torch.no_grad()
def build_perm(
    two_sae: TwoSidedTopKSAE,
    dataset,
    device: torch.device,
    max_samples: int = 50_000,
    batch_size: int = 2048,
) -> np.ndarray:
    """Compute Hungarian text → image slot permutation from paired dataset.

    Streams up to `max_samples` paired embeddings through the (frozen) two-sided
    SAE, builds Pearson C (L_per_side × L_per_side), runs linear_sum_assignment(-C).
    Returns an int64 array `perm` of length L_per_side, such that
    image slot i matches text slot `perm[i]` — so at eval time,
    `z_T_aligned[:, i] = z_T[:, perm[i]]`.
    """
    n = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, n, dtype=np.int64)
    img = torch.stack([dataset[int(i)]["image_embeds"] for i in indices], dim=0)
    txt = torch.stack([dataset[int(i)]["text_embeds"] for i in indices], dim=0)
    logger.info("build_perm: stacked (%d, %d)", img.shape[0], img.shape[1])

    z_i = _stream_dense_latents(two_sae.image_sae, img, batch_size, device)
    z_t = _stream_dense_latents(two_sae.text_sae, txt, batch_size, device)
    logger.info("build_perm: latents image=%s text=%s", z_i.shape, z_t.shape)

    C = _pearson_C(z_i, z_t)
    row_ind, col_ind = linear_sum_assignment(-C)
    # linear_sum_assignment returns row_ind sorted → col_ind is the permutation
    # of text slots aligned to image slots.
    perm = np.zeros_like(col_ind)
    perm[row_ind] = col_ind
    return perm.astype(np.int64)


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------

@torch.no_grad()
def encode_image(
    model,
    x: torch.Tensor,
    method: Method,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Encode image embeddings (N, H) to dense latents (N, L) for downstream eval.

    For shared/aux: single SAE.
    For separated/ours: image side of the two-sided SAE. No perm needed on image side.
    """
    if method in ("shared", "aux"):
        sae = model
    else:
        sae = model.image_sae
    return _stream_dense_latents(sae, x, batch_size, device)


@torch.no_grad()
def encode_text(
    model,
    y: torch.Tensor,
    method: Method,
    device: torch.device,
    perm: np.ndarray | None = None,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Encode text embeddings (N, H) to dense latents (N, L).

    For shared/aux: single SAE (slot alignment trivial).
    For separated: text side, raw — slots NOT aligned to image side (expected
        to be weak on retrieval/zeroshot).
    For ours: text side, then re-index columns by `perm` so matched slots
        share index with image side. `perm` must be provided.
    """
    if method in ("shared", "aux"):
        sae = model
    else:
        sae = model.text_sae
    z_t = _stream_dense_latents(sae, y, batch_size, device)
    if method == "ours":
        if perm is None:
            raise ValueError("perm required for method='ours'")
        perm_t = torch.as_tensor(perm, dtype=torch.long)
        z_t = z_t[:, perm_t]
    return z_t


# ----------------------------------------------------------------------
# Output normalization (shared by retrieval + zero-shot)
# ----------------------------------------------------------------------

def normalize_rows(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


__all__ = [
    "load_sae",
    "load_pair_dataset",
    "build_perm",
    "encode_image",
    "encode_text",
    "normalize_rows",
]
