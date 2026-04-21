"""Utilities for CachedClipPairsDataset."""

from __future__ import annotations

import torch

from src.datasets.cached_clip_pairs import CachedClipPairsDataset


def stack_paired_tensors(ds: CachedClipPairsDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack all pairs into ``(img, txt)`` tensors of shape ``(N, d)``."""
    img = torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
    txt = torch.stack(
        [ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs]
    )
    return img, txt
