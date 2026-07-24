"""Extract LLaVA-1.5 HACL-style paired EOS embeddings into the standard cache.

Produces exactly the on-disk schema ``CachedClipPairsDataset`` /
``CachedImageNetPairsDataset`` consume, so the existing SAE trainer, density
(run_diagnostic_B / plot_multi_model_density) and Table-1 pipeline
(run_real_v2 / eval_coco_retrieval / eval_imagenet_zeroshot) work UNCHANGED --
only the encoder is swapped for the LLM EOS hidden state (dim 4096).

Two datasets:

  --dataset coco     COCO Karpathy (namkha1032/coco-karpathy). Image side is
                     deduped by image_id; text key is the COMPOSITE f"{iid}_{ci}"
                     (required by retrieval eval). Emits captions.json + splits.json
                     for the requested --splits (default: train test).

  --dataset imagenet ILSVRC/imagenet-1k val images (key = running int index) +
                     class x template text prototypes (key f"{c}_{t}", the 80
                     OpenAI templates). splits.json = {"val": [[idx, class], ...]}.

OOM-safety (the box only ever holds one batch of 4096-d vectors in RAM): each
modality streams straight into an ``np.memmap`` stack and is finalized to
``{modality}_embeddings_stack.pt`` + ``_map.json`` (mmap-loaded downstream).
Resumable via per-modality ``*_keys.txt`` (already-done keys are skipped).
Non-finite rows are dropped. Progress + ETA are logged every --flush-every.

Smoke: ``--limit N`` caps images per split (coco) / total val images (imagenet);
``--n-classes/--n-templates`` shrink the imagenet text grid.

Examples:
  # COCO train+test (full)
  python scripts/real_alpha/extract_llava_cache.py --dataset coco \
      --cache-dir cache/llava_coco --splits train test
  # smoke: 64 imgs/split
  python scripts/real_alpha/extract_llava_cache.py --dataset coco \
      --cache-dir cache/llava_coco_smoke --splits train test --limit 64
  # imagenet val smoke: 10 classes x 5 templates, 100 val imgs
  python scripts/real_alpha/extract_llava_cache.py --dataset imagenet \
      --cache-dir cache/llava_imagenet_smoke --limit 100 --n-classes 10 --n-templates 5
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from llava_forwards import DEFAULT_MODEL, LlavaForwards

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COCO_DATASET = "namkha1032/coco-karpathy"
HF_TO_OUR_SPLIT = {"train": "train", "validation": "val", "test": "test"}
OUR_TO_HF_SPLIT = {v: k for k, v in HF_TO_OUR_SPLIT.items()}
IMAGENET_DATASET = "ILSVRC/imagenet-1k"
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["coco", "imagenet"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=list(_DTYPES), default="bf16")
    p.add_argument("--image-batch", type=int, default=8)
    p.add_argument("--text-batch", type=int, default=32)
    p.add_argument("--flush-every", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0, help="0 = full; else cap images")
    p.add_argument("--splits", nargs="+", default=["train", "test"],
                   help="coco splits (our names): train val test")
    p.add_argument("--n-classes", type=int, default=1000, help="imagenet only")
    p.add_argument("--n-templates", type=int, default=80, help="imagenet only")
    p.add_argument("--no-l2-normalize", action="store_true")
    p.add_argument("--keep-raw", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- IO
class MemmapWriter:
    """One modality: memmap stack + append-only key list, resumable."""

    def __init__(self, cache_dir: Path, modality: str, dim: int, capacity: int):
        self.cache_dir = cache_dir
        self.modality = modality
        self.dim = dim
        self.capacity = capacity
        self.raw = cache_dir / f"{modality}_stack.f32"
        self.keys_path = cache_dir / f"{modality}_keys.txt"
        self.mm = np.memmap(self.raw, dtype=np.float32,
                            mode=("r+" if self.raw.exists() else "w+"),
                            shape=(capacity, dim))
        self.keys: list[str] = (
            self.keys_path.read_text().splitlines() if self.keys_path.exists() else []
        )
        self.seen = set(self.keys)
        self._unflushed = 0

    def has(self, key: str) -> bool:
        return key in self.seen

    def add(self, key: str, vec: np.ndarray) -> None:
        r = len(self.keys)
        if r >= self.capacity:
            raise RuntimeError(f"{self.modality} memmap capacity {self.capacity} exceeded")
        self.mm[r] = vec
        self.keys.append(key)
        self.seen.add(key)
        self._unflushed += 1

    def flush(self) -> None:
        self.mm.flush()
        tmp = self.keys_path.with_suffix(".tmp")
        tmp.write_text("\n".join(self.keys))
        tmp.replace(self.keys_path)
        self._unflushed = 0

    def finalize(self, l2_normalize: bool, keep_raw: bool) -> int:
        n = len(self.keys)
        self.flush()
        if l2_normalize:
            CHUNK = 200_000
            for i0 in range(0, n, CHUNK):
                chunk = self.mm[i0:min(i0 + CHUNK, n)]
                norms = np.sqrt((chunk ** 2).sum(axis=1, keepdims=True))
                np.divide(chunk, np.clip(norms, 1e-12, None), out=chunk)
            self.mm.flush()
        stack = torch.from_numpy(np.asarray(self.mm[:n]).copy())
        out = self.cache_dir / f"{self.modality}_embeddings_stack.pt"
        logger.info("saving %s (%.2f GB, n=%d)", out, stack.numel() * 4 / 2 ** 30, n)
        torch.save(stack, out)
        with open(self.cache_dir / f"{self.modality}_embeddings_map.json", "w") as f:
            json.dump({k: i for i, k in enumerate(self.keys)}, f)
        if not keep_raw:
            self.raw.unlink(missing_ok=True)
            self.keys_path.unlink(missing_ok=True)
        return n


class _Progress:
    def __init__(self, total: int, tag: str, flush_every: int):
        self.total, self.tag, self.every = total, tag, flush_every
        self.t0 = time.time()
        self.n0 = None
        self.last = 0

    def tick(self, n: int, failed: int) -> None:
        if self.n0 is None:
            self.n0 = n
        if n - self.last < self.every:
            return
        self.last = n
        el = time.time() - self.t0
        rate = (n - self.n0) / max(el, 1e-6)
        eta = (self.total - n) / max(rate, 1e-6)
        logger.info("[%s] %d/%d (%.1f%%) rate=%.1f/s failed=%d ETA=%.1fmin (end~%s)",
                    self.tag, n, self.total, 100 * n / max(self.total, 1), rate, failed,
                    eta / 60, time.strftime("%H:%M UTC", time.gmtime(time.time() + eta)))


def _write_batch(writer: MemmapWriter, keys: list[str], feats: torch.Tensor) -> int:
    finite = torch.isfinite(feats).all(dim=1)
    dropped = int((~finite).sum())
    arr = feats.numpy()
    for k, ok, i in zip(keys, finite.tolist(), range(len(keys))):
        if ok:
            writer.add(k, arr[i])
    return dropped


# ---------------------------------------------------------------------- COCO
def extract_coco(fwd: LlavaForwards, args) -> None:
    from datasets import load_dataset

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_splits = [OUR_TO_HF_SPLIT[s] for s in args.splits]

    # ---- pass 0: sizes + splits.json + captions.json (image col dropped) ----
    splits_out: dict[str, list] = {}
    captions: dict[str, str] = {}
    n_img_cap = n_txt_cap = 0
    per_split_ids: dict[str, list[int]] = {}
    for hf_split in hf_splits:
        our = HF_TO_OUR_SPLIT[hf_split]
        ds = load_dataset(COCO_DATASET, split=hf_split)
        if "image" in ds.column_names:
            ds = ds.remove_columns(["image"])
        ids, pairs = [], []
        for row in ds:
            if args.limit and len(ids) >= args.limit:
                break
            iid = int(row["image_id"])
            ids.append(iid)
            for ci, cap in enumerate(row["captions"]):
                pairs.append([str(iid), ci])
                captions[f"{iid}_{ci}"] = cap
        per_split_ids[our] = ids
        splits_out[our] = pairs
        n_img_cap += len(ids)
        n_txt_cap += len(pairs)
        logger.info("split %s: %d images, %d caption-pairs", our, len(ids), len(pairs))
    with open(cache_dir / "splits.json", "w") as f:
        json.dump(splits_out, f)
    with open(cache_dir / "captions.json", "w") as f:
        json.dump(captions, f)

    img_w = MemmapWriter(cache_dir, "image", fwd.emb_dim, n_img_cap)
    txt_w = MemmapWriter(cache_dir, "text", fwd.emb_dim, n_txt_cap)
    failed = 0

    # ---- image pass (dedup by iid, batched) ----
    prog = _Progress(n_img_cap, "img", args.flush_every)
    buf_ids, buf_pils = [], []

    def flush_img():
        nonlocal failed
        if not buf_pils:
            return
        feats = fwd.fwd_image_eos(buf_pils)
        failed += _write_batch(img_w, [str(i) for i in buf_ids], feats)
        buf_ids.clear(); buf_pils.clear()
        if img_w._unflushed >= args.flush_every:
            img_w.flush()
        prog.tick(len(img_w.keys), failed)

    want_ids = {str(i) for ids in per_split_ids.values() for i in ids}
    for hf_split in hf_splits:
        ds = load_dataset(COCO_DATASET, split=hf_split)
        # Only the (first --limit) image_ids we recorded in pass 0; stop scanning
        # the split as soon as they're all collected (crucial for --limit smoke:
        # otherwise we'd iterate all ~113k rows to find 64).
        remaining = {str(i) for i in per_split_ids[HF_TO_OUR_SPLIT[hf_split]]} - img_w.seen
        for row in ds:
            if not remaining:
                break
            iid = str(int(row["image_id"]))
            if iid not in remaining:
                continue
            remaining.discard(iid)
            buf_ids.append(int(iid))
            buf_pils.append(row["image"].convert("RGB"))
            if len(buf_pils) >= args.image_batch:
                flush_img()
        flush_img()
    img_w.flush()
    logger.info("image pass done: %d/%d (failed=%d)", len(img_w.keys), n_img_cap, failed)

    # ---- text pass (composite keys, batched) ----
    prog = _Progress(n_txt_cap, "txt", args.flush_every)
    buf_keys, buf_txt = [], []

    def flush_txt():
        nonlocal failed
        if not buf_txt:
            return
        feats = fwd.fwd_text_eos(buf_txt)
        failed += _write_batch(txt_w, list(buf_keys), feats)
        buf_keys.clear(); buf_txt.clear()
        if txt_w._unflushed >= args.flush_every:
            txt_w.flush()
        prog.tick(len(txt_w.keys), failed)

    for our in args.splits:
        for iid_s, ci in splits_out[our]:
            key = f"{iid_s}_{ci}"
            if txt_w.has(key):
                continue
            buf_keys.append(key)
            buf_txt.append(captions[key])
            if len(buf_txt) >= args.text_batch:
                flush_txt()
    flush_txt()
    txt_w.flush()
    logger.info("text pass done: %d/%d (failed=%d)", len(txt_w.keys), n_txt_cap, failed)

    _finalize_and_sanity(fwd, cache_dir, img_w, txt_w, args, failed,
                         extra_meta={"dataset": COCO_DATASET, "splits": args.splits},
                         sanity_pairs=[(s, f"{s}_0") for s in list(want_ids)[:64]])


# ------------------------------------------------------------------ ImageNet
def extract_imagenet(fwd: LlavaForwards, args) -> None:
    """ImageNet val + class x template prototypes in the PER-KEY DICT format
    (image_embeddings.pt / text_embeddings.pt) that CachedImageNetPairsDataset +
    eval_imagenet_zeroshot.py expect. ImageNet is small (<=50k imgs + a few k
    texts) so no memmap is needed."""
    from datasets import load_dataset
    from open_clip.zero_shot_metadata import (
        IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES,
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_classes = min(args.n_classes, len(IMAGENET_CLASSNAMES))
    templates = OPENAI_IMAGENET_TEMPLATES[:args.n_templates]
    class_names = list(IMAGENET_CLASSNAMES[:n_classes])
    img_cap = args.limit if args.limit else 50_000
    l2 = not args.no_l2_normalize
    failed = 0

    def _norm(v: torch.Tensor) -> torch.Tensor:
        return v / v.norm().clamp_min(1e-12) if l2 else v

    # ---- image pass: val stream, keep labels < n_classes, cap by --limit ----
    image_dict: dict[int, torch.Tensor] = {}
    pairs: list[list[int]] = []
    ds = load_dataset(IMAGENET_DATASET, split="validation", streaming=True)
    prog = _Progress(img_cap, "in-img", args.flush_every)
    buf_idx, buf_pils, buf_lab = [], [], []
    idx = 0

    def flush_in_img():
        nonlocal failed
        if not buf_pils:
            return
        feats = fwd.fwd_image_eos(buf_pils)
        finite = torch.isfinite(feats).all(dim=1)
        for j, ok in enumerate(finite.tolist()):
            if ok:
                image_dict[buf_idx[j]] = _norm(feats[j]).clone()
                pairs.append([buf_idx[j], buf_lab[j]])
            else:
                failed += 1
        buf_idx.clear(); buf_pils.clear(); buf_lab.clear()
        prog.tick(len(image_dict), failed)

    for row in ds:
        # count buffered-but-unwritten too, else the batch overshoots the cap
        if len(image_dict) + len(buf_pils) >= img_cap:
            break
        label = int(row["label"])
        if label >= n_classes:
            continue
        buf_idx.append(idx); buf_lab.append(label)
        buf_pils.append(row["image"].convert("RGB"))
        idx += 1
        if len(buf_pils) >= args.image_batch:
            flush_in_img()
    flush_in_img()
    logger.info("imagenet image pass: %d imgs (failed=%d)", len(image_dict), failed)

    # ---- text pass: class x template prototypes, key "{c}_{t}" ----
    text_dict: dict[str, torch.Tensor] = {}
    buf_keys, buf_txt = [], []

    def flush_in_txt():
        if not buf_txt:
            return
        feats = fwd.fwd_text_eos(buf_txt)
        for k, v in zip(buf_keys, feats):
            text_dict[k] = _norm(v).clone()
        buf_keys.clear(); buf_txt.clear()

    for c, name in enumerate(class_names):
        for t, tmpl in enumerate(templates):
            buf_keys.append(f"{c}_{t}")
            buf_txt.append(tmpl(name))
            if len(buf_txt) >= args.text_batch:
                flush_in_txt()
    flush_in_txt()
    logger.info("imagenet text pass: %d texts (%d classes x %d templates)",
                len(text_dict), n_classes, len(templates))

    torch.save(image_dict, cache_dir / "image_embeddings.pt")
    torch.save(text_dict, cache_dir / "text_embeddings.pt")
    with open(cache_dir / "splits.json", "w") as f:
        json.dump({"val": pairs}, f)
    meta = {
        "clip_model": fwd.model_id, "encoder": "llava_eos", "dim": fwd.emb_dim,
        "kind": "llava", "dataset": IMAGENET_DATASET, "n_images": len(image_dict),
        "n_classes": n_classes, "n_templates": len(templates),
        "class_names": class_names, "n_failed": failed, "l2_normalized": l2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("DONE: %s (%d imgs, %d texts, %d failed)",
                cache_dir, len(image_dict), len(text_dict), failed)


# ---------------------------------------------------------------- finalize
def _finalize_and_sanity(fwd, cache_dir, img_w, txt_w, args, failed,
                         extra_meta, sanity_pairs) -> None:
    l2 = not args.no_l2_normalize
    # keep image/text row maps for the sanity check before raw memmap is dropped
    img_map = {k: i for i, k in enumerate(img_w.keys)}
    txt_map = {k: i for i, k in enumerate(txt_w.keys)}
    if sanity_pairs:
        try:
            ii = [img_map[a] for a, b in sanity_pairs if a in img_map and b in txt_map][:64]
            tt = [txt_map[b] for a, b in sanity_pairs if a in img_map and b in txt_map][:64]
            if ii:
                a = torch.from_numpy(np.asarray(img_w.mm[ii]).copy())
                b = torch.from_numpy(np.asarray(txt_w.mm[tt]).copy())
                cos = torch.nn.functional.cosine_similarity(a, b).mean().item()
                logger.info("paired-cosine sanity (image vs its caption0): %.4f "
                            "(should be clearly > 0)", cos)
        except Exception as e:  # sanity must never break extraction
            logger.warning("sanity check skipped: %s", e)

    n_img = img_w.finalize(l2, args.keep_raw)
    n_txt = txt_w.finalize(l2, args.keep_raw)
    meta = {
        "clip_model": fwd.model_id,
        "encoder": "llava_eos",
        "dim": fwd.emb_dim,
        "kind": "llava",
        "dtype": args.dtype,
        "n_images": n_img,
        "n_texts": n_txt,
        "n_failed": failed,
        "l2_normalized": l2,
        "limit": args.limit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra_meta,
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("DONE: %s (%d imgs, %d texts, %d failed)", cache_dir, n_img, n_txt, failed)


def main() -> None:
    args = parse_args()
    fwd = LlavaForwards(args.model, device=args.device, dtype=_DTYPES[args.dtype])
    if args.dataset == "coco":
        extract_coco(fwd, args)
    else:
        extract_imagenet(fwd, args)


if __name__ == "__main__":
    main()
