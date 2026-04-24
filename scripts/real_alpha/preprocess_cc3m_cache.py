"""Pre-process a CC3M-style cache (dict of per-sample tensors) into a single
stacked tensor + JSON key→row mapping, so training can mmap-load the stack
and skip the expensive dict reassembly.

Usage (run ONCE per cache dir, per modality):

    python scripts/real_alpha/preprocess_cc3m_cache.py \\
        --cache-dir cache/clip_b32_cc3m --modality image
    python scripts/real_alpha/preprocess_cc3m_cache.py \\
        --cache-dir cache/clip_b32_cc3m --modality text

Each invocation runs in its OWN process, so peak RAM is bounded to one
modality at a time (~12 GB for CC3M). Between runs the kernel reclaims
all memory, so the 24 GB box survives.

Writes, inside ``<cache_dir>/``:
  * ``<modality>_embeddings_stack.pt``   — (N, dim) fp32 contiguous tensor
  * ``<modality>_embeddings_map.json``   — dict[key, row_index]

If the artifacts already exist and ``--force`` is not set, the script
exits early.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _clip_vector_norm(t: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.sum(torch.pow(t, 2), dim=-1, keepdim=True), 0.5)


def process_modality(cache_dir: Path, modality: str, force: bool) -> None:
    src = cache_dir / f"{modality}_embeddings.pt"
    stack_out = cache_dir / f"{modality}_embeddings_stack.pt"
    map_out = cache_dir / f"{modality}_embeddings_map.json"

    if not src.exists():
        raise SystemExit(f"missing source: {src}")
    if stack_out.exists() and map_out.exists() and not force:
        logger.info("artifacts already exist, skipping (--force to overwrite): %s", stack_out)
        return

    logger.info("loading %s (this is the heavy step)", src)
    raw = torch.load(src, map_location="cpu")
    n = len(raw)
    first_v = next(iter(raw.values()))
    dim = int(first_v.shape[-1])
    logger.info("got n=%d entries, dim=%d", n, dim)

    logger.info("allocating (%d, %d) fp32 table (~%.2f GB)", n, dim, n * dim * 4 / 2**30)
    table = torch.empty(n, dim, dtype=torch.float32)
    mapping: dict = {}

    # Pop-transfer pattern: each raw entry is copied into the table row and
    # then refcount-freed, so peak ~= table size + (shrinking) raw size.
    keys = list(raw.keys())
    for i, k in enumerate(keys):
        v = raw.pop(k)
        if v.dtype != torch.float32:
            v = v.float()
        table[i].copy_(v)
        mapping[str(k)] = i
        if i > 0 and i % 500_000 == 0:
            logger.info("  copied %d/%d (%.1f%%)", i, n, 100.0 * i / n)
    del raw, keys, first_v
    gc.collect()

    # Chunked in-place L2 normalize (small transient memory).
    logger.info("L2-normalizing in chunks")
    CHUNK = 200_000
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        chunk = table[i0:i1]
        norms = _clip_vector_norm(chunk)
        chunk.div_(norms)

    # Verify norms
    sample = table[::max(1, n // 20)]
    sample_norms = _clip_vector_norm(sample).squeeze(-1)
    logger.info(
        "sample norms after normalize: mean=%.6f min=%.6f max=%.6f",
        float(sample_norms.mean()), float(sample_norms.min()), float(sample_norms.max()),
    )

    logger.info("saving %s (~%.2f GB)", stack_out, table.numel() * 4 / 2**30)
    torch.save(table, stack_out)
    del table
    gc.collect()

    logger.info("saving %s (n=%d mapping entries)", map_out, len(mapping))
    with open(map_out, "w") as f:
        json.dump(mapping, f)
    del mapping
    gc.collect()

    logger.info("done: %s  +  %s", stack_out, map_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, required=True)
    ap.add_argument("--modality", choices=["image", "text"], required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    process_modality(Path(args.cache_dir), args.modality, args.force)


if __name__ == "__main__":
    main()
