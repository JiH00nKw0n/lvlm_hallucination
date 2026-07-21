"""Full-dataset Hungarian perm — identical algorithm to build_perm, streamed.

`eval_utils.build_perm` (the paper pipeline) subsamples `max_samples=50_000`
paired rows and materializes their (N, L) dense latents to form the Pearson
co-activation matrix C. This script computes the SAME quantities — signed
Pearson C, per-slot fire counts, `alive = fire >= 1`, `linear_sum_assignment
(-C_masked)` with dead rows/cols penalized to BIG_NEG — over the ENTIRE train
split, using the streaming accumulator from analyze_C_multiplicity so the
(N, L) tensors are never materialized (full N would need ~26 GB/side otherwise).

Only difference from build_perm: N = full dataset instead of 50k, streamed.
Everything downstream (masking, Hungarian, perm.npz schema) is copied verbatim
from eval_utils.build_perm.

Usage:
    python scripts/real_alpha/build_hungarian_perm_full.py \
        --ckpt outputs/real_exp_cc3m_aimv2_s0/separated/ckpt/final \
        --dataset cc3m --cache-dir cache/aimv2_lit_cc3m_1p6m \
        --output outputs/real_exp_cc3m_aimv2_s0/ours/perm_full.npz
"""

from __future__ import annotations

import argparse
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402
from analyze_C_multiplicity import accumulate_C, pearson_from_stats  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", choices=["coco", "cc3m", "laion"], default="cc3m")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=0, help="0 = full dataset")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--alive-min-fires", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = eval_utils.load_sae(args.ckpt, "separated")
    ds = eval_utils.load_pair_dataset(args.cache_dir, args.dataset, split=args.split)
    logger.info("dataset=%d pairs (split=%s)", len(ds), args.split)

    # streaming C + fire counts over the FULL set (same Pearson def as _pearson_C)
    acc = accumulate_C(model, ds, device, args.batch_size, args.max_samples)
    C = pearson_from_stats(acc).astype(np.float64)
    fire_i = acc["fire_i"].numpy().astype(np.int64)
    fire_t = acc["fire_t"].numpy().astype(np.int64)

    # ---- identical to eval_utils.build_perm from here ----
    alive_i = fire_i >= args.alive_min_fires
    alive_t = fire_t >= args.alive_min_fires
    logger.info("alive image=%d/%d (%.1f%%), alive text=%d/%d (%.1f%%)  [N=%d]",
                int(alive_i.sum()), len(alive_i), 100 * alive_i.mean(),
                int(alive_t.sum()), len(alive_t), 100 * alive_t.mean(), acc["n"])

    BIG_NEG = -1e9
    C_masked = C.copy()
    C_masked[~alive_i, :] = BIG_NEG
    C_masked[:, ~alive_t] = BIG_NEG
    row_ind, col_ind = linear_sum_assignment(-C_masked)
    perm = np.zeros_like(col_ind)
    perm[row_ind] = col_ind

    np.savez(out, perm=perm.astype(np.int64),
             alive_image=alive_i, alive_text=alive_t,
             fire_count_image=fire_i, fire_count_text=fire_t)
    logger.info("saved full-data perm (len=%d, N=%d) → %s", perm.shape[0], acc["n"], out)


if __name__ == "__main__":
    main()
