"""Do the weak matches actually cost anything?

The correlation distribution shows that a large share of matched pairs have weak
co-activation. That invites the obvious follow-up: if so many matches are weak,
is the alignment usable at all? This measures cross-modal retrieval while
keeping only the strongest matches and discarding the rest, at several cutoffs.

Comparing against a random subset of coordinates would prove nothing, because
high-confidence coordinates are also the frequently firing ones and would win on
activation mass alone. The control instead keeps exactly the same coordinates
and shuffles which text latent each is paired with. Same coordinates, same
activation, wrong correspondence. Retrieval that survives that shuffle was never
measuring correspondence.

    python scripts/real_alpha/eval_confidence_ablation.py \
        --panel outputs/rebuttal_EA/coco_k8_r1r2/C_img_txt.npz \
        --ckpt outputs/rebuttal_models/coco_k8_r1/final \
        --cache-dir cache/clip_b32_coco --split test \
        --out outputs/rebuttal_EC/coco_k8_r1
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
from rebuttal_common import alive_masks, hungarian_perm, load_panel  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CUTOFFS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache-dir", default="cache/clip_b32_coco")
    p.add_argument("--split", default="test")
    p.add_argument("--alive-rule", choices=["ever", "density"], default="ever")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
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


def recall(z_img: torch.Tensor, z_txt: torch.Tensor, pair_img_idx: np.ndarray,
           gt_caps: list[list[int]]) -> dict:
    """Recall@k both ways, with pessimistic tie handling.

    Sparse latents leave many pairs tied at zero similarity, and counting a tie
    as a hit would flatter every method that produces uninformative scores.
    """
    ks = (1, 5, 10)
    t2i_rank = np.empty(z_txt.shape[0], dtype=np.int64)
    for s in range(0, z_txt.shape[0], 1024):
        sc = z_txt[s:s + 1024] @ z_img.T
        gt = pair_img_idx[s:s + 1024]
        gt_sc = sc[np.arange(len(gt)), gt]
        t2i_rank[s:s + 1024] = ((sc >= gt_sc[:, None]).sum(dim=1) - 1).cpu().numpy()

    i2t_rank = np.empty(z_img.shape[0], dtype=np.int64)
    for s in range(0, z_img.shape[0], 512):
        sc = z_img[s:s + 512] @ z_txt.T
        for row in range(sc.shape[0]):
            i = s + row
            best = sc[row, gt_caps[i]].max()
            i2t_rank[i] = int((sc[row] >= best).sum().item()) - 1

    return {
        **{f"T2I R@{k}": float((t2i_rank < k).mean()) for k in ks},
        **{f"I2T R@{k}": float((i2t_rank < k).mean()) for k in ks},
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.panel)
    alive_i, alive_t = alive_masks(panel, args.alive_rule)
    match = hungarian_perm(panel["C"], alive_i, alive_t)
    perm, usable, matched_c = match["perm"], match["usable"], match["matched_c"]

    model = eval_utils.load_sae(args.ckpt, "separated")
    ds = eval_utils.load_pair_dataset(args.cache_dir, "coco", split=args.split)
    pairs = ds.pairs
    unique_ids = sorted({int(p[0]) for p in pairs})
    id_to_pos = {iid: i for i, iid in enumerate(unique_ids)}
    pair_img_idx = np.array([id_to_pos[int(p[0])] for p in pairs], dtype=np.int64)
    gt_caps: list[list[int]] = [[] for _ in unique_ids]
    for ci, gi in enumerate(pair_img_idx):
        gt_caps[int(gi)].append(ci)

    img = torch.stack([ds._image_dict[i] for i in unique_ids], dim=0)
    txt = torch.stack([ds._text_dict[ds._text_key_for(int(p[0]), int(p[1]))] for p in pairs], dim=0)
    logger.info("%s split: %d images, %d captions", args.split, img.shape[0], txt.shape[0])

    z_img = eval_utils.encode_image(model, img, "separated", device, args.batch_size)
    z_txt_raw = eval_utils.encode_text(model, txt, "separated", device,
                                       perm=None, batch_size=args.batch_size)
    z_txt = z_txt_raw[:, torch.as_tensor(perm, dtype=torch.long)]

    results = {}
    for cut in CUTOFFS:
        keep = usable & (matched_c >= cut)
        n_keep = int(keep.sum())
        if n_keep < 10:
            results[f"c>={cut}"] = {"n_coordinates": n_keep, "note": "too few to score"}
            continue
        mask = torch.as_tensor(keep, dtype=torch.bool)
        zi = eval_utils.normalize_rows(z_img[:, mask])
        zt = eval_utils.normalize_rows(z_txt[:, mask])
        r = recall(zi, zt, pair_img_idx, gt_caps)
        r["n_coordinates"] = n_keep

        # same coordinates, correspondence destroyed
        idx = np.where(keep)[0]
        shuffled_perm = perm.copy()
        shuffled_perm[idx] = perm[rng.permutation(idx)]
        zt_shuf = z_txt_raw[:, torch.as_tensor(shuffled_perm, dtype=torch.long)]
        zt_shuf = eval_utils.normalize_rows(zt_shuf[:, mask])
        rs = recall(zi, zt_shuf, pair_img_idx, gt_caps)
        r["shuffled_partners"] = {k: v for k, v in rs.items()}
        results[f"c>={cut}"] = r
        logger.info("cut %.2f (%d coords): I2T R@1 %.4f | shuffled %.4f",
                    cut, n_keep, r["I2T R@1"], rs["I2T R@1"])

    report = {"ckpt": args.ckpt, "panel": args.panel, "split": args.split,
              "alive_rule": args.alive_rule, "by_cutoff": results}
    (out / "confidence_ablation.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"{'kept matches':<16}{'coords':>8}{'I2T R@1':>10}{'I2T R@5':>10}"
          f"{'T2I R@1':>10}{'T2I R@5':>10}{'shuffled I2T R@1':>19}")
    for name, r in results.items():
        if "I2T R@1" not in r:
            print(f"{name:<16}{r['n_coordinates']:>8}   {r.get('note', '')}")
            continue
        print(f"{name:<16}{r['n_coordinates']:>8}{r['I2T R@1'] * 100:>10.2f}"
              f"{r['I2T R@5'] * 100:>10.2f}{r['T2I R@1'] * 100:>10.2f}"
              f"{r['T2I R@5'] * 100:>10.2f}{r['shuffled_partners']['I2T R@1'] * 100:>19.2f}")
    print(f"\nwrote {out / 'confidence_ablation.json'}")


if __name__ == "__main__":
    main()
