"""Stream a web image-text dataset from HF and cache paired embeddings (memmap).

Successor to extract_clip_cc3m_cache.py for rebuttal-scale extractions:
  * `--hf-dataset` generic — works for webdataset repos (pixparse/cc3m-wds:
    `__key__`/`jpg`/`txt`) and parquet repos with embedded image bytes
    (jp1924/Laion400m-1: `image`/`caption`).
  * GPU JPEG decode (nvJPEG) + on-device preprocess — the 2-core CPU box
    only moves bytes; decode/resize/normalize run on the GPU.
  * Embeddings are written straight into per-modality `np.memmap` files and
    finalized into the *post-preprocess* artifact format
    (`{modality}_embeddings_stack.pt` + `_map.json` + splits.json + meta.json),
    so `CachedClipPairsDataset` mmap-loads it with zero extra steps and the
    24 GB RAM box never holds the dataset in memory.
  * Resumable: `extract_state.json` tracks (source_consumed, rows_written);
    on restart the source stream is `.skip()`ed accordingly.

Usage (LAION 3M, CLIP B/32):
    python scripts/real_alpha/extract_web_pairs_cache.py \
        --hf-dataset jp1924/Laion400m-1 --image-col image --caption-col caption \
        --model openai/clip-vit-base-patch32 \
        --cache-dir cache/clip_b32_laion --max-samples 3000000

Usage (CC3M, AIMv2):
    python scripts/real_alpha/extract_web_pairs_cache.py \
        --hf-dataset pixparse/cc3m-wds --image-col jpg --caption-col txt \
        --model apple/aimv2-large-patch14-224-lit \
        --cache-dir cache/aimv2_lit_cc3m --max-samples 3000000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from extract_common import (  # noqa: E402
    GpuImagePreprocessor,
    decode_to_device,
    load_model_forwards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset", type=str, required=True)
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--image-col", type=str, default="")
    p.add_argument("--caption-col", type=str, default="")
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["transformers", "openclip"])
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--max-samples", type=int, required=True,
                   help="memmap row budget AND stop condition (>0)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--flush-every", type=int, default=65536)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-gpu-decode", action="store_true",
                   help="decode via PIL in DataLoader workers instead of nvJPEG")
    p.add_argument("--half", action="store_true",
                   help="run the encoder in bfloat16 (faster on heavy ViT-L; "
                        "embeddings still stored fp32 after L2-norm)")
    p.add_argument("--no-l2-normalize", action="store_true")
    p.add_argument("--keep-raw", action="store_true",
                   help="keep the raw .f32 memmap files after finalize")
    return p.parse_args()


IMAGE_COL_CANDIDATES = ("jpg", "image", "img", "webp", "png")
CAPTION_COL_CANDIDATES = ("txt", "caption", "text")


def _resolve_cols(features: dict, image_col: str, caption_col: str) -> tuple[str, str]:
    cols = set(features.keys())
    if not image_col:
        image_col = next((c for c in IMAGE_COL_CANDIDATES if c in cols), "")
    if not caption_col:
        caption_col = next((c for c in CAPTION_COL_CANDIDATES if c in cols), "")
    if not image_col or image_col not in cols or not caption_col or caption_col not in cols:
        raise SystemExit(f"cannot resolve image/caption columns in {sorted(cols)}; "
                         f"pass --image-col/--caption-col")
    return image_col, caption_col


class _Collate:
    """Module-level (picklable for spawn-based DataLoader workers)."""

    def __init__(self, image_col: str, caption_col: str):
        self.image_col = image_col
        self.caption_col = caption_col

    def __call__(self, rows):
        out = []
        for r in rows:
            cap = r.get(self.caption_col)
            if isinstance(cap, list):
                cap = cap[0] if cap else None
            if not cap or not isinstance(cap, str):
                continue
            img = r.get(self.image_col)
            if img is None:
                continue
            out.append({"img": img, "txt": cap.strip(),
                        "key": str(r.get("__key__", ""))})
        return {"rows": out, "n_source": len(rows)}


class _State:
    """Resume bookkeeping: how many SOURCE rows were consumed (incl. failed
    decodes) and how many embedding rows were actually written."""

    def __init__(self, path: Path):
        self.path = path
        self.source_consumed = 0
        self.rows_written = 0
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            self.source_consumed = int(d["source_consumed"])
            self.rows_written = int(d["rows_written"])

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"source_consumed": self.source_consumed,
                       "rows_written": self.rows_written}, f)
        tmp.replace(self.path)


def _open_memmap(path: Path, n_rows: int, dim: int) -> np.memmap:
    mode = "r+" if path.exists() else "w+"
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(n_rows, dim))


def _finalize(cache_dir: Path, mm_img: np.memmap, mm_txt: np.memmap,
              keys: list[str], hf_split: str, l2_normalize: bool,
              meta: dict, keep_raw: bool) -> None:
    n = len(keys)
    logger.info("finalizing %d rows", n)
    for modality, mm in (("image", mm_img), ("text", mm_txt)):
        if l2_normalize:
            CHUNK = 200_000
            for i0 in range(0, n, CHUNK):
                chunk = mm[i0:min(i0 + CHUNK, n)]
                norms = np.sqrt((chunk ** 2).sum(axis=1, keepdims=True))
                np.divide(chunk, np.clip(norms, 1e-12, None), out=chunk)
            mm.flush()
        stack = torch.from_numpy(np.asarray(mm[:n]))
        out = cache_dir / f"{modality}_embeddings_stack.pt"
        logger.info("saving %s (%.2f GB)", out, stack.numel() * 4 / 2 ** 30)
        torch.save(stack, out)
        with open(cache_dir / f"{modality}_embeddings_map.json", "w") as f:
            json.dump({k: i for i, k in enumerate(keys)}, f)
    with open(cache_dir / "splits.json", "w") as f:
        json.dump({hf_split: [[k, 0] for k in keys]}, f)
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if not keep_raw:
        for modality in ("image", "text"):
            raw = cache_dir / f"{modality}_stack.f32"
            if raw.exists():
                raw.unlink()
                logger.info("removed raw %s", raw)


def extract(args: argparse.Namespace) -> None:
    from datasets import Image as HFImage
    from datasets import load_dataset

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    gpu_decode = (not args.no_gpu_decode)

    fwd = load_model_forwards(args.model, device, args.backend, args.pretrained,
                              half=args.half)
    logger.info("model=%s kind=%s dim=%d gpu_decode=%s device=%s",
                args.model, fwd.kind, fwd.emb_dim, gpu_decode, device)
    prep = None
    if gpu_decode:
        if fwd.image_processor is None or fwd.fwd_pixels is None:
            raise SystemExit("--backend openclip requires --no-gpu-decode")
        prep = GpuImagePreprocessor(fwd.image_processor, device)

    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    features = dict(ds.features or {})
    image_col, caption_col = _resolve_cols(features, args.image_col, args.caption_col)
    logger.info("columns: image=%r caption=%r (of %s)", image_col, caption_col,
                sorted(features.keys()))
    if gpu_decode:
        ds = ds.cast_column(image_col, HFImage(decode=False))

    state = _State(cache_dir / "extract_state.json")
    if state.source_consumed:
        logger.info("resuming: source_consumed=%d rows_written=%d",
                    state.source_consumed, state.rows_written)
        ds = ds.skip(state.source_consumed)

    mm_img = _open_memmap(cache_dir / "image_stack.f32", args.max_samples, fwd.emb_dim)
    mm_txt = _open_memmap(cache_dir / "text_stack.f32", args.max_samples, fwd.emb_dim)

    keys_path = cache_dir / "keys.txt"
    keys: list[str] = []
    if keys_path.exists():
        keys = keys_path.read_text().splitlines()
        assert len(keys) == state.rows_written, (
            f"keys.txt ({len(keys)}) != rows_written ({state.rows_written})")

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,  # type: ignore[arg-type]
                        collate_fn=_Collate(image_col, caption_col),
                        persistent_workers=bool(args.num_workers))

    n_failed = 0
    unflushed_keys = 0
    t0 = time.time()
    last_log = 0
    rows_at_start = len(keys)  # rate baseline: fixed at launch, NOT at flush

    def _flush():
        nonlocal unflushed_keys
        mm_img.flush(); mm_txt.flush()
        with open(keys_path, "a") as f:
            for k in keys[len(keys) - unflushed_keys:]:
                f.write(k + "\n")
        unflushed_keys = 0
        state.rows_written = len(keys)
        state.save()

    @torch.no_grad()
    def _process(rows: list[dict]) -> None:
        nonlocal n_failed
        pixel_list, texts, batch_keys = [], [], []
        for r in rows:
            key = r["key"] or str(state.source_consumed + len(batch_keys))
            if gpu_decode:
                data = r["img"]["bytes"] if isinstance(r["img"], dict) else r["img"]
                u8 = decode_to_device(data, device) if data is not None else None
                if u8 is None:
                    n_failed += 1
                    continue
                try:
                    pixel_list.append(prep(u8))
                except Exception:
                    n_failed += 1
                    continue
            else:
                try:
                    pixel_list.append(r["img"].convert("RGB"))
                except Exception:
                    n_failed += 1
                    continue
            texts.append(r["txt"])
            batch_keys.append(key)
        if not batch_keys:
            return
        room = args.max_samples - len(keys)
        if room <= 0:
            return
        pixel_list, texts, batch_keys = pixel_list[:room], texts[:room], batch_keys[:room]

        if gpu_decode:
            img_feats = fwd.fwd_pixels(torch.stack(pixel_list))
        else:
            img_feats = fwd.fwd_img(pixel_list)
        txt_feats = fwd.fwd_txt(texts)

        # Guard: a rare corrupt image can push the encoder to NaN/Inf; one such
        # row poisons batch-variance-normalized losses downstream. Drop them.
        finite = (torch.isfinite(img_feats).all(dim=1)
                  & torch.isfinite(txt_feats).all(dim=1))
        if not bool(finite.all()):
            n_failed += int((~finite).sum())
            img_feats = img_feats[finite]
            txt_feats = txt_feats[finite]
            batch_keys = [k for k, ok in zip(batch_keys, finite.tolist()) if ok]
            if not batch_keys:
                return

        s = len(keys)
        mm_img[s:s + len(batch_keys)] = img_feats.numpy()
        mm_txt[s:s + len(batch_keys)] = txt_feats.numpy()
        keys.extend(batch_keys)

    logger.info("start: target=%d rows (batch=%d workers=%d)",
                args.max_samples, args.batch_size, args.num_workers)
    first_batch_checked = len(keys) > 0
    for batch in loader:
        n_before = len(keys)
        _process(batch["rows"])
        state.source_consumed += batch["n_source"]
        unflushed_keys += len(keys) - n_before

        if not first_batch_checked and len(keys) > 0:
            # paired-cosine sanity: catches wrong pooling / preprocess bugs
            a = torch.from_numpy(np.asarray(mm_img[:len(keys)]).copy())
            b = torch.from_numpy(np.asarray(mm_txt[:len(keys)]).copy())
            cos = torch.nn.functional.cosine_similarity(a, b).mean().item()
            logger.info("first-batch paired cosine: %.4f (should be clearly > 0)", cos)
            first_batch_checked = True

        if len(keys) - last_log >= max(1, args.flush_every // 4):
            elapsed = time.time() - t0
            rate = (len(keys) - rows_at_start) / max(elapsed, 1e-6)
            eta_min = (args.max_samples - len(keys)) / max(rate, 1e-6) / 60
            logger.info("n=%d/%d (%.1f%%) rate=%.0f/s failed=%d ETA=%.0fmin (wall_end=%s)",
                        len(keys), args.max_samples,
                        100 * len(keys) / args.max_samples, rate, n_failed, eta_min,
                        time.strftime("%H:%M UTC", time.gmtime(time.time() + eta_min * 60)))
            last_log = len(keys)
        if unflushed_keys >= args.flush_every:
            _flush()
        if len(keys) >= args.max_samples:
            break
    _flush()

    meta = {
        "clip_model": args.model,
        "dataset": args.hf_dataset,
        "split": args.hf_split,
        "dim": fwd.emb_dim,
        "kind": fwd.kind,
        "n_images": len(keys),
        "n_texts": len(keys),
        "n_failed_decode": n_failed,
        "gpu_decode": gpu_decode,
        "l2_normalized": not args.no_l2_normalize,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _finalize(cache_dir, mm_img, mm_txt, keys, args.hf_split,
              not args.no_l2_normalize, meta, args.keep_raw)
    logger.info("done: %d rows (%d failed decodes) → %s", len(keys), n_failed, cache_dir)


def main() -> None:
    extract(parse_args())


if __name__ == "__main__":
    main()
