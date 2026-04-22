"""COCO Karpathy test cross-modal retrieval in SAE latent space.

Protocol:
  - Test split has 5000 images × 5 captions = 25000 pairs.
  - I→T: for each image, rank all captions by cos(z_I, z_T); Recall@{1,5,10}
    is 1 if any of the 5 ground-truth captions sits in top-k.
  - T→I: for each caption, rank all unique images; Recall@{1,5,10} on the
    single ground-truth image.

All latents are extracted via eval_utils.encode_{image,text}; Ours applies
the saved perm on the text side.

Usage:
    python scripts/real_alpha/eval_coco_retrieval.py \
        --ckpt outputs/real_exp_v1/ours/coco/final \
        --method ours --cache-dir cache/clip_b32_coco \
        --perm outputs/real_exp_v1/ours/coco/perm.npz \
        --output outputs/real_exp_v1/ours/coco/retrieval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _recall_at_k(ranks: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    return {f"R@{k}": float((ranks < k).mean()) for k in ks}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)

    perm = None
    if args.method == "ours":
        if args.perm is None:
            raise SystemExit("--perm required for method='ours'")
        perm = np.load(args.perm)["perm"]

    ds = eval_utils.load_pair_dataset(args.cache_dir, "coco", args.split)
    logger.info("pairs=%d (=%s split)", len(ds), args.split)

    # Collect per-pair image_id, caption; dedupe images
    pairs = ds.pairs  # type: ignore[attr-defined]
    img_ids = [int(p[0]) for p in pairs]
    unique_img_ids = sorted(set(img_ids))
    id_to_idx = {iid: k for k, iid in enumerate(unique_img_ids)}
    pair_img_idx = np.array([id_to_idx[iid] for iid in img_ids], dtype=np.int64)
    logger.info("unique images=%d, captions=%d", len(unique_img_ids), len(pairs))

    # Build image tensor (one row per unique image) and caption tensor (one per pair)
    img_cache = ds._image_dict  # type: ignore[attr-defined]
    txt_cache = ds._text_dict  # type: ignore[attr-defined]
    img = torch.stack([img_cache[iid] for iid in unique_img_ids], dim=0)
    txt = torch.stack([txt_cache[f"{int(p[0])}_{int(p[1])}"] for p in pairs], dim=0)

    logger.info("encoding %d images + %d captions", img.shape[0], txt.shape[0])
    z_img = eval_utils.encode_image(model, img, args.method, device, args.batch_size)
    z_txt = eval_utils.encode_text(model, txt, args.method, device, perm=perm, batch_size=args.batch_size)
    z_img = eval_utils.normalize_rows(z_img)
    z_txt = eval_utils.normalize_rows(z_txt)

    # T → I: for each caption, rank unique images by cos sim
    #   gt = pair_img_idx[p]; rank = number of images with higher score than gt.
    t2i_ranks = np.empty(z_txt.shape[0], dtype=np.int64)
    chunk = 1024
    for s in range(0, z_txt.shape[0], chunk):
        scores = z_txt[s:s + chunk] @ z_img.T  # (b, n_img)
        gt = pair_img_idx[s:s + chunk]
        gt_scores = scores[np.arange(len(gt)), gt]
        # Rank = how many images score strictly higher than gt
        t2i_ranks[s:s + chunk] = (scores > gt_scores[:, None]).sum(dim=1).cpu().numpy()

    # I → T: for each image, rank all captions. Correct if any of 5 gt captions rank < k.
    gt_caps_per_img: list[list[int]] = [[] for _ in range(len(unique_img_ids))]
    for cap_p, img_idx in enumerate(pair_img_idx):
        gt_caps_per_img[int(img_idx)].append(cap_p)

    # For each image, the rank reported is the rank of its best-scoring GT
    # caption among all captions (lower is better; 0 = perfect top-1).
    i2t_min_rank = np.empty(z_img.shape[0], dtype=np.int64)
    for s in range(0, z_img.shape[0], chunk):
        scores = z_img[s:s + chunk] @ z_txt.T  # (b, n_cap)
        for row in range(scores.shape[0]):
            img_idx = s + row
            gt_caps = gt_caps_per_img[img_idx]
            best_gt_score = scores[row, gt_caps].max()
            rank = int((scores[row] > best_gt_score).sum().item())
            i2t_min_rank[img_idx] = rank

    t2i = _recall_at_k(t2i_ranks)
    i2t = _recall_at_k(i2t_min_rank)
    logger.info("T→I %s", t2i)
    logger.info("I→T %s", i2t)

    result = {
        "method": args.method,
        "dataset": "coco",
        "split": args.split,
        "n_images": int(len(unique_img_ids)),
        "n_captions": int(len(pairs)),
        "latent_size": int(z_img.shape[1]),
        "T2I": t2i,
        "I2T": i2t,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
