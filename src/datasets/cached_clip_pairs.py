"""Dataset for pre-cached CLIP image/text embedding pairs.

Loads embeddings saved by `scripts/real_alpha/extract_clip_coco_cache.py` and
exposes them as a `torch.utils.data.Dataset` whose samples are compatible
with the Hugging Face `default_data_collator` and our `TwoSidedSAETrainer`
(`{"image_embeds": Tensor(d), "text_embeds": Tensor(d)}`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]


def _clip_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """ℓ2 norm of the last dimension, exactly matching
    ``transformers.models.clip.modeling_clip._get_vector_norm``:
    a numerically-equivalent rewrite of ``tensor.norm(p=2, dim=-1, keepdim=True)``
    chosen for executorch exportability. No epsilon is added — this mirrors
    the native `CLIPModel.forward` normalization used by CLIP inference.
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    return torch.pow(sum_tensor, 0.5)


class CachedClipPairsDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: Split,
        l2_normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.split = split
        self.l2_normalize = l2_normalize

        with open(cache_dir / "splits.json", "r") as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"split '{split}' missing in {cache_dir}/splits.json")
        # Each entry is [image_id, caption_idx]
        self.pairs: list[tuple[int, int]] = [tuple(p) for p in splits[split]]  # type: ignore[misc]

        # Memory-efficient load: pop-transfer each entry from the raw dict into
        # the final dict, folding in dtype cast + L2 normalize in one pass.
        # Prevents peak = 2× data size (both dicts alive during comprehension),
        # which OOM'd on CC3M (~2.8M pairs, 24 GB RAM box). Peak stays ~1× data.
        # Use the same norm formula as transformers.models.clip.modeling_clip so
        # our SAE inputs match CLIP's native inference-space embeddings exactly
        # (no epsilon, sqrt(sum(x^2)) on the last dim).
        self._image_dict = self._load_and_prepare(
            cache_dir / "image_embeddings.pt", key_cast=int, dtype=dtype,
        )
        self._text_dict = self._load_and_prepare(
            cache_dir / "text_embeddings.pt", key_cast=str, dtype=dtype,
        )

    def _load_and_prepare(self, path: Path, key_cast, dtype: torch.dtype) -> dict:
        raw = torch.load(path, map_location="cpu")
        out: dict = {}
        for k in list(raw.keys()):
            v = raw.pop(k)
            if v.dtype != dtype:
                v = v.to(dtype)
            if self.l2_normalize:
                v = v / _clip_vector_norm(v)
            out[key_cast(k)] = v
        del raw
        return out

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_id, caption_idx = self.pairs[idx]
        img = self._image_dict[int(image_id)]
        txt = self._text_dict[f"{int(image_id)}_{int(caption_idx)}"]
        return {"image_embeds": img, "text_embeds": txt}


__all__ = ["CachedClipPairsDataset"]
