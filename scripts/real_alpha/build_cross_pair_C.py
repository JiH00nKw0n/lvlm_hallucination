"""Co-activation correlation between any two trained SAEs, on any pairing of inputs.

The paper measures cross-modal feature heterogeneity by correlating an image
SAE's latents with a text SAE's latents over paired data, then looking at the
angle between the decoder directions of highly correlated pairs. The reviewer's
objection is that two SAEs trained independently might differ that much anyway.
Answering it needs the same measurement run on pairs of SAEs that differ only by
training run, so this script generalizes the measurement to four panels:

    img_img          image SAE of run A  vs  image SAE of run B, same image
    txt_txt          text  SAE of run A  vs  text  SAE of run B, same caption
    txt_txt_diffcap  text  SAE of run A  vs  text  SAE of run B, two captions
                     of the same photo
    img_txt          image and text SAE of run A — reproduces the paper's panel

``txt_txt_diffcap`` is the one that needs explaining. In ``img_img`` both
encoders read the identical embedding vector, so a pair of latents that mean the
same thing correlates almost perfectly; ``img_txt`` instead feeds one side a
photo and the other a sentence about it, which correlates less well even if the
two dictionaries were identical. Comparing ``img_img`` against ``img_txt``
therefore mixes a modality effect with an input-mismatch effect.
``txt_txt_diffcap`` isolates them: same modality on both sides, but the two
encoders read different sentences about the same scene.

Correlations come from ``_compute_latent_correlation``, the same function that
produced the paper's matrices, so the reproduced panel is comparable by
construction rather than by argument.

    python scripts/real_alpha/build_cross_pair_C.py \
        --ckpt-a outputs/rebuttal_models/coco_k8_r1/final \
        --ckpt-b outputs/rebuttal_models/coco_k8_r2/final \
        --panel img_img --dataset coco --cache-dir cache/clip_b32_coco \
        --out outputs/rebuttal_EA/coco_k8/C_img_img.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from collections import defaultdict
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402
from synthetic_theorem2_method import _compute_latent_correlation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PANELS = ("img_img", "txt_txt", "txt_txt_diffcap", "img_txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", required=True, help="run A checkpoint dir (TwoSidedTopKSAE)")
    p.add_argument("--ckpt-b", default=None,
                   help="run B checkpoint dir; defaults to run A (only img_txt needs a single run)")
    p.add_argument("--panel", choices=PANELS, required=True)
    p.add_argument("--dataset", choices=["coco", "cc3m", "laion"], default="coco")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 = every pair; otherwise an evenly spaced subsample")
    p.add_argument("--shuffle-b", type=int, default=0, metavar="SEED",
                   help="destroy the pairing by permuting the B-side rows, so the "
                        "correlations become a noise floor. Any non-zero value is "
                        "used as the seed. Shuffling the rows and recomputing is "
                        "the only valid way to do this: permuting the columns of a "
                        "finished correlation matrix leaves each row's maximum "
                        "intact, and the assignment simply finds it again.")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    return p.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pair_rows(ds) -> tuple[np.ndarray, np.ndarray]:
    """Row index into the stacked tables for every (image, caption) pair.

    Batch gathering from the tables is far faster than per-item __getitem__ at
    the scale of these caches.
    """
    img_rows = np.empty(len(ds.pairs), dtype=np.int64)
    txt_rows = np.empty(len(ds.pairs), dtype=np.int64)
    for i, (iid, cid) in enumerate(ds.pairs):
        img_rows[i] = ds._image_id_to_row[int(iid)]
        txt_rows[i] = ds._text_key_to_row[ds._text_key_for(iid, cid)]
    return img_rows, txt_rows


def other_caption_rows(ds) -> tuple[np.ndarray, int]:
    """For each pair, the row of a *different* caption of the same image.

    Captions of one image are ordered as they appear in the split; pair with
    caption index t takes t+1 (wrapping), so every pair gets a partner and no
    caption is paired with itself. Images with a single caption keep their own
    caption — those rows are counted and reported, since for them the panel
    degenerates to the same-input case.
    """
    by_image: dict[int, list[int]] = defaultdict(list)
    for i, (iid, _cid) in enumerate(ds.pairs):
        by_image[int(iid)].append(i)

    partner = np.empty(len(ds.pairs), dtype=np.int64)
    n_singleton = 0
    for _iid, idxs in by_image.items():
        n = len(idxs)
        if n == 1:
            partner[idxs[0]] = idxs[0]
            n_singleton += 1
            continue
        for j, pair_idx in enumerate(idxs):
            partner[pair_idx] = idxs[(j + 1) % n]
    logger.info("different-caption pairing: %d/%d rows had no alternative caption",
                n_singleton, len(ds.pairs))

    _img_rows, txt_rows = pair_rows(ds)
    return txt_rows[partner], n_singleton


@torch.no_grad()
def firing_rate(sae, table: torch.Tensor, rows: np.ndarray,
                batch_size: int, device: torch.device) -> np.ndarray:
    """Fraction of the given rows on which each latent is non-zero.

    The density figure calls a latent alive when this exceeds 1e-3, so the two
    quantities have to be computed over exactly the same rows as the
    correlations.
    """
    sae.eval().to(device)
    L = int(sae.latent_size)
    fires = torch.zeros(L, dtype=torch.float64)
    n = len(rows)
    for s in range(0, n, batch_size):
        idx = torch.as_tensor(rows[s:s + batch_size])
        x = table[idx].to(device)
        z = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True).dense_latents.squeeze(1)
        # Move to CPU before widening: MPS has no float64.
        fires += (z != 0).sum(dim=0).cpu().double()
    return (fires / max(n, 1)).numpy()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    ckpt_b = args.ckpt_b or args.ckpt_a

    logger.info("run A: %s", args.ckpt_a)
    logger.info("run B: %s", ckpt_b)
    model_a = eval_utils.load_sae(args.ckpt_a, "separated")
    model_b = model_a if ckpt_b == args.ckpt_a else eval_utils.load_sae(ckpt_b, "separated")

    ds = eval_utils.load_pair_dataset(args.cache_dir, args.dataset, split=args.split)
    logger.info("dataset %s split=%s: %d pairs", args.dataset, args.split, len(ds))

    img_rows, txt_rows = pair_rows(ds)
    n_singleton = 0
    if args.panel == "txt_txt_diffcap":
        txt_rows_b, n_singleton = other_caption_rows(ds)
    else:
        txt_rows_b = txt_rows

    # Same row subset for every panel, so sample composition cannot explain a
    # difference between panels.
    n_total = len(img_rows)
    if args.max_samples and args.max_samples < n_total:
        sel = np.linspace(0, n_total - 1, args.max_samples, dtype=np.int64)
        img_rows, txt_rows, txt_rows_b = img_rows[sel], txt_rows[sel], txt_rows_b[sel]
        logger.info("subsampled to %d rows", len(img_rows))

    if args.shuffle_b:
        # A derangement of the B side. Correlations then reflect only what two
        # independent latent streams produce at this sample count.
        shuffle_rng = np.random.default_rng(args.shuffle_b)
        order = shuffle_rng.permutation(len(img_rows))
        txt_rows = txt_rows[order]
        txt_rows_b = txt_rows_b[order]
        img_rows_b_shuffled = img_rows[order]
        logger.info("pairing destroyed with seed %d (noise-floor run)", args.shuffle_b)
    else:
        img_rows_b_shuffled = img_rows

    img_table, txt_table = ds._image_table, ds._text_table

    if args.panel == "img_img":
        sae_a, sae_b = model_a.image_sae, model_b.image_sae
        rows_a, rows_b = img_rows, img_rows_b_shuffled
        table_a, table_b = img_table, img_table
    elif args.panel in ("txt_txt", "txt_txt_diffcap"):
        sae_a, sae_b = model_a.text_sae, model_b.text_sae
        rows_a, rows_b = txt_rows, txt_rows_b
        table_a, table_b = txt_table, txt_table
    else:  # img_txt
        sae_a, sae_b = model_a.image_sae, model_a.text_sae
        rows_a, rows_b = img_rows, txt_rows
        table_a, table_b = img_table, txt_table

    logger.info("gathering %d rows per side", len(rows_a))
    X = table_a[torch.as_tensor(rows_a)]
    Y = table_b[torch.as_tensor(rows_b)]
    logger.info("X=%s Y=%s device=%s", tuple(X.shape), tuple(Y.shape), device)

    logger.info("computing correlation matrix")
    C = _compute_latent_correlation(sae_a, sae_b, X, Y, args.batch_size, device)
    logger.info("C=%s  max=%.4f  min=%.4f", C.shape, float(np.nanmax(C)), float(np.nanmin(C)))

    logger.info("computing firing rates")
    rate_a = firing_rate(sae_a, table_a, rows_a, args.batch_size, device)
    rate_b = firing_rate(sae_b, table_b, rows_b, args.batch_size, device)
    logger.info("alive at rate>1e-3: A %d/%d, B %d/%d",
                int((rate_a > 1e-3).sum()), len(rate_a),
                int((rate_b > 1e-3).sum()), len(rate_b))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        C=C.astype(np.float16),          # |C| <= 1, and fp16 is well under the bin width
        rate_a=rate_a.astype(np.float32),
        rate_b=rate_b.astype(np.float32),
        n_samples=np.int64(len(rows_a)),
        n_singleton_caption=np.int64(n_singleton),
        shuffle_b=np.int64(args.shuffle_b),
    )
    meta = {
        "panel": args.panel, "ckpt_a": args.ckpt_a, "ckpt_b": ckpt_b,
        "dataset": args.dataset, "split": args.split, "n_samples": int(len(rows_a)),
        "n_singleton_caption": int(n_singleton),
        "shuffle_b": int(args.shuffle_b),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
