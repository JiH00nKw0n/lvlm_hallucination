"""Dataset for pre-cached CLIP image/text embedding pairs.

Loads embeddings saved by ``scripts/real_alpha/extract_clip_coco_cache.py``
(and the CC3M variant) and exposes them as a ``torch.utils.data.Dataset``
whose samples are compatible with the Hugging Face ``default_data_collator``
and our ``TwoSidedSAETrainer``
(``{"image_embeds": Tensor(d), "text_embeds": Tensor(d)}``).

Storage layout
--------------
For each modality the dataset holds a single contiguous ``(N, dim)`` tensor
plus a small key → row-index mapping. This avoids the 2.8M-entry dict of
tiny 2KB tensors that OOM'd on the CC3M cache under 24 GB RAM — storage
is ~5.6 GB per modality + ~200 MB mapping (versus ~8 GB per modality with
heavy allocator fragmentation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]


def _clip_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """ℓ2 norm along the last dim, exactly matching
    ``transformers.models.clip.modeling_clip._get_vector_norm``: a
    numerically-equivalent rewrite of
    ``tensor.norm(p=2, dim=-1, keepdim=True)`` chosen for executorch
    exportability. No epsilon is added — this mirrors the native
    ``CLIPModel.forward`` normalization.
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
        normalize_chunk: int = 100_000,
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

        self._image_table, self._image_id_to_row = self._load_and_stack(
            cache_dir / "image_embeddings.pt", key_cast=int,
            dtype=dtype, normalize_chunk=normalize_chunk,
        )
        self._text_table, self._text_key_to_row = self._load_and_stack(
            cache_dir / "text_embeddings.pt", key_cast=str,
            dtype=dtype, normalize_chunk=normalize_chunk,
        )

    def _load_and_stack(
        self, path: Path, key_cast,
        dtype: torch.dtype, normalize_chunk: int,
    ) -> tuple[torch.Tensor, dict]:
        """Load ``path`` (dict[key, Tensor(dim,)]) into a single (N, dim)
        tensor + a key → row mapping.

        Memory profile for CC3M (N≈2.82M, dim=512):
          * peak while building: ~11.2 GB (raw pickle dict + empty table
            being filled; raw drains as table fills, so peak tracks the
            moment both hold half the data each).
          * steady after return: ~5.6 GB (table) + ~200 MB (dict mapping).
          * chunked L2 normalize adds ~200 MB transient per chunk, not
            a full table copy.
        """
        raw = torch.load(path, map_location="cpu")
        n = len(raw)
        first = next(iter(raw.values()))
        dim = int(first.shape[-1])

        table = torch.empty(n, dim, dtype=dtype)
        key_to_row: dict = {}

        # Iterate over a snapshot of keys so we can ``pop`` as we go,
        # letting Python refcount free each tensor the moment it's copied
        # into ``table``. Enumerate gives us the row index.
        keys = list(raw.keys())
        for i, k in enumerate(keys):
            v = raw.pop(k)
            if v.dtype != dtype:
                v = v.to(dtype)
            table[i].copy_(v)
            key_to_row[key_cast(k)] = i
        del raw, keys

        if self.l2_normalize:
            # In-place vectorized normalize, chunked so the scratch memory
            # for the norm computation stays small (vs. allocating a full
            # (N, 1) norm tensor from the whole table at once).
            for i0 in range(0, n, normalize_chunk):
                i1 = min(i0 + normalize_chunk, n)
                chunk = table[i0:i1]  # view
                norms = _clip_vector_norm(chunk)  # (chunk, 1)
                chunk.div_(norms)

        return table, key_to_row

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_id, caption_idx = self.pairs[idx]
        img_row = self._image_id_to_row[int(image_id)]
        txt_row = self._text_key_to_row[f"{int(image_id)}_{int(caption_idx)}"]
        return {
            "image_embeds": self._image_table[img_row],
            "text_embeds": self._text_table[txt_row],
        }

    # ------------------------------------------------------------------
    # Back-compat: diagnostic scripts access ``ds._image_dict[id]`` and
    # ``ds._text_dict[key]`` directly. We expose read-only proxies over
    # the stacked tables so those scripts keep working unchanged.
    # ------------------------------------------------------------------
    @property
    def _image_dict(self):  # type: ignore[override]
        return _RowLookup(self._image_table, self._image_id_to_row)

    @property
    def _text_dict(self):  # type: ignore[override]
        return _RowLookup(self._text_table, self._text_key_to_row)


class _RowLookup:
    """Dict-like view into a stacked (N, dim) table via a key→row mapping."""
    __slots__ = ("table", "mapping")

    def __init__(self, table: torch.Tensor, mapping: dict) -> None:
        self.table = table
        self.mapping = mapping

    def __getitem__(self, key):
        return self.table[self.mapping[key]]

    def __contains__(self, key) -> bool:
        return key in self.mapping

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self):
        return iter(self.mapping)

    def keys(self):
        return self.mapping.keys()

    def items(self):
        for k, i in self.mapping.items():
            yield k, self.table[i]

    def values(self):
        for i in self.mapping.values():
            yield self.table[i]


__all__ = ["CachedClipPairsDataset"]
