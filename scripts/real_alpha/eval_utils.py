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
from src.models.vl_sae import VLSAE  # type: ignore
from src.models.shared_enc_sae import SharedEncSAE  # type: ignore

logger = logging.getLogger(__name__)

Method = Literal["shared", "separated", "aux", "ours", "vl_sae", "shared_enc"]
Dataset = Literal["coco", "imagenet", "cc3m"]


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_sae(ckpt_dir: str | Path, method: Method):
    """Load a saved SAE checkpoint.

    For method ∈ {shared, aux}: returns a TopKSAE.
    For method ∈ {separated, ours}: returns a TwoSidedTopKSAE.
    For method == vl_sae: returns a VLSAE.
    """
    ckpt_dir = str(ckpt_dir)
    if method in ("shared", "aux"):
        return TopKSAE.from_pretrained(ckpt_dir)
    if method in ("separated", "ours"):
        return TwoSidedTopKSAE.from_pretrained(ckpt_dir)
    if method == "vl_sae":
        return VLSAE.from_pretrained(ckpt_dir)
    if method == "shared_enc":
        return SharedEncSAE.from_pretrained(ckpt_dir)
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
    if dataset == "cc3m":
        # pixparse/cc3m-wds has both train and validation. Default historical
        # behavior used "train"; pass split through so val-set evaluation works.
        return CachedClipPairsDataset(cache_dir, split=split, l2_normalize=True)
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


@torch.no_grad()
def _stream_vl_sae_latents(sae: VLSAE, embeds: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode a tensor of embeddings (N, H) through VL-SAE's shared encoder.

    VL-SAE's encode returns a dense (batch, L) tensor directly (top-K already
    applied in-place); no scatter is needed and there is no seq dim.
    """
    sae.eval()
    sae.to(device)  # type: ignore[arg-type]
    out = torch.empty(embeds.shape[0], int(sae.latent_size), dtype=torch.float32)
    for s in range(0, embeds.shape[0], batch_size):
        chunk = embeds[s:s + batch_size].to(device)
        z = sae.encode(chunk)
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
    alive_min_fires: int = 1,
) -> dict:
    """Compute alive-restricted Hungarian text→image slot permutation.

    Streams up to `max_samples` paired embeddings through the frozen two-sided
    SAE, builds Pearson C (L_per_side × L_per_side), then runs Hungarian on a
    C matrix where dead slots (fire count < alive_min_fires) on either side
    are penalized to a large negative so they can't steal alive matches.

    Returns dict with:
        perm: (L,) int64 — text slot permutation
        alive_image, alive_text: (L,) bool — per-slot alive masks
        fire_count_image, fire_count_text: (L,) int64 — raw fire counts
    At eval time: `z_T_aligned[:, i] = z_T[:, perm[i]]`.
    """
    n = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, n, dtype=np.int64)
    img = torch.stack([dataset[int(i)]["image_embeds"] for i in indices], dim=0)
    txt = torch.stack([dataset[int(i)]["text_embeds"] for i in indices], dim=0)
    logger.info("build_perm: stacked (%d, %d)", img.shape[0], img.shape[1])

    z_i = _stream_dense_latents(two_sae.image_sae, img, batch_size, device)
    z_t = _stream_dense_latents(two_sae.text_sae, txt, batch_size, device)
    logger.info("build_perm: latents image=%s text=%s", z_i.shape, z_t.shape)

    fire_i = (z_i != 0).sum(dim=0).cpu().numpy().astype(np.int64)
    fire_t = (z_t != 0).sum(dim=0).cpu().numpy().astype(np.int64)
    alive_i = fire_i >= alive_min_fires
    alive_t = fire_t >= alive_min_fires
    logger.info(
        "build_perm: alive image=%d/%d (%.1f%%), alive text=%d/%d (%.1f%%)",
        int(alive_i.sum()), len(alive_i), 100 * alive_i.mean(),
        int(alive_t.sum()), len(alive_t), 100 * alive_t.mean(),
    )

    C = _pearson_C(z_i, z_t).astype(np.float64)
    # Penalize dead rows / columns so alive-alive pairs dominate Hungarian.
    # Dead-dead cells fall back to the penalty too — since their z=0, the
    # resulting perm entry has no effect on downstream cosine.
    BIG_NEG = -1e9
    C_masked = C.copy()
    C_masked[~alive_i, :] = BIG_NEG
    C_masked[:, ~alive_t] = BIG_NEG

    row_ind, col_ind = linear_sum_assignment(-C_masked)
    perm = np.zeros_like(col_ind)
    perm[row_ind] = col_ind
    return {
        "perm": perm.astype(np.int64),
        "alive_image": alive_i,
        "alive_text": alive_t,
        "fire_count_image": fire_i,
        "fire_count_text": fire_t,
    }


@torch.no_grad()
def compute_dead_counts(
    model,
    dataset,
    method: Method,
    device: torch.device,
    max_samples: int = 50_000,
    batch_size: int = 2048,
    alive_min_fires: int = 1,
) -> dict:
    """Count dead slots per modality for a given method.

    Streams up to `max_samples` paired embeddings, encodes each modality with
    the appropriate SAE/encoder, and counts per-slot nonzero firings.

    Returns dict:
        latent_size_image: int
        latent_size_text: int
        fire_count_image: list[int]
        fire_count_text: list[int]
        alive_image_count, alive_text_count, dead_image_count, dead_text_count: int
        alive_min_fires: int
        n_samples: int
    """
    n = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, n, dtype=np.int64)
    img = torch.stack([dataset[int(i)]["image_embeds"] for i in indices], dim=0)
    txt = torch.stack([dataset[int(i)]["text_embeds"] for i in indices], dim=0)

    z_i = encode_image(model, img, method, device, batch_size=batch_size)
    # NOTE: for ours we need perm but we're just counting dead, so use separated path
    _method_for_text = "separated" if method == "ours" else method
    z_t = encode_text(model, txt, _method_for_text, device, batch_size=batch_size)

    fire_i = (z_i != 0).sum(dim=0).cpu().numpy().astype(np.int64)
    fire_t = (z_t != 0).sum(dim=0).cpu().numpy().astype(np.int64)

    return {
        "latent_size_image": int(z_i.shape[1]),
        "latent_size_text": int(z_t.shape[1]),
        "fire_count_image": fire_i.tolist(),
        "fire_count_text": fire_t.tolist(),
        "alive_image_count": int((fire_i >= alive_min_fires).sum()),
        "alive_text_count": int((fire_t >= alive_min_fires).sum()),
        "dead_image_count": int((fire_i < alive_min_fires).sum()),
        "dead_text_count": int((fire_t < alive_min_fires).sum()),
        "alive_min_fires": int(alive_min_fires),
        "n_samples": int(n),
    }


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
    For vl_sae: shared distance-based encoder applied to image input.
    """
    if method in ("vl_sae", "shared_enc"):
        return _stream_vl_sae_latents(model, x, batch_size, device)
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
    For vl_sae: shared distance-based encoder applied to text input; slots
        align to image side by construction (shared encoder).
    """
    if method in ("vl_sae", "shared_enc"):
        return _stream_vl_sae_latents(model, y, batch_size, device)
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
    "compute_dead_counts",
    "encode_image",
    "encode_text",
    "normalize_rows",
]
