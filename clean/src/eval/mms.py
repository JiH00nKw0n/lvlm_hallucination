"""Multimodal Monosemanticity Score (MMS) — Kaushik et al. 2026, §3.2.

For each latent k and modality pair (m, n) on the COCO test split:
  A_ij = |a_i^(m) a_j^(n)|, S_ij = cos(E(x_i^(m)), E(x_j^(n))) under a *separate*
  encoder E (different from the one used to train the SAE), and
  MMS_k(m,n) = sum_ij (A_ij / sum A) * S_ij.

For same-modality pairs we exclude the diagonal (i == j); cross-modal uses all pairs.

Outputs (per variant):
  mms_per_latent.csv      latent_idx, fire_img, fire_txt, mms_ii, mms_tt, mms_it
  mms_summary.json        sorted curves + means/medians over alive sets
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys as _sys
import time
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
    p.add_argument("--variant", type=str, required=True, help="label only, used in output path")
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours"], required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--train-cache", type=str, required=True,
                   help="cache used to train SAE (e.g. cache/clip_b32_coco)")
    p.add_argument("--ext-cache", type=str, required=True,
                   help="separate-encoder cache for similarity matrix (e.g. cache/metaclip_b32_coco)")
    p.add_argument("--dataset", choices=["coco", "cc3m"], default="coco",
                   help="determines pair convention: coco=multi-caption (image_id, cap_idx), "
                        "cc3m=single-caption (key, 0)")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--perm", type=str, default=None, help="required when method=ours")
    p.add_argument("--out", type=str, required=True, help="output dir")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-pairs-per-latent", type=int, default=0,
                   help="if >0 cap pair count per latent (random sub-sample) for memory")
    return p.parse_args()


def _load_paired_tensors(cache_dir: str, dataset: str, split: str):
    """Return (image_keys, text_keys, img_tensor, txt_tensor).

    coco: dedupe images (multi-caption) — txt is one row per pair (25010 for
    test), img is one row per unique image (5000 for test). text_keys
    follow `f"{image_id}_{cap_idx}"`. image_keys are int image_ids.

    cc3m: 1:1 image–text pairing — img/txt have same length and use the
    *same* webdataset __key__ (string like "00000005").
    """
    ds = eval_utils.load_pair_dataset(cache_dir, dataset, split)  # type: ignore[arg-type]
    pairs = ds.pairs  # type: ignore[attr-defined]
    img_cache = ds._image_dict  # type: ignore[attr-defined]
    txt_cache = ds._text_dict  # type: ignore[attr-defined]

    if dataset == "coco":
        img_ids = [int(p[0]) for p in pairs]
        unique_img_ids = sorted(set(img_ids))
        img = torch.stack([img_cache[iid] for iid in unique_img_ids], dim=0)
        txt_keys = [f"{int(p[0])}_{int(p[1])}" for p in pairs]
        txt = torch.stack([txt_cache[k] for k in txt_keys], dim=0)
        return unique_img_ids, txt_keys, img, txt

    if dataset == "cc3m":
        # Internal (CachedClipPairsDataset) loads image keys with key_cast=int
        # and text keys with key_cast=str. External raw dict keeps both as the
        # original webdataset string key (e.g. "00000005"). Ext images: try
        # str first then fall back to int via _get() in _load_ext_embeddings.
        str_keys = [str(p[0]) for p in pairs]
        int_keys = [int(p[0]) for p in pairs]
        img = torch.stack([img_cache[k] for k in int_keys], dim=0)
        txt = torch.stack([txt_cache[k] for k in str_keys], dim=0)
        return str_keys, str_keys, img, txt

    raise ValueError(f"unknown dataset {dataset!r}")


def _load_ext_embeddings(ext_cache: str, image_keys, text_keys: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    img_dict = torch.load(_Path(ext_cache) / "image_embeddings.pt", map_location="cpu", weights_only=False)
    txt_dict = torch.load(_Path(ext_cache) / "text_embeddings.pt", map_location="cpu", weights_only=False)

    def _get(d: dict, k):
        if k in d:
            return d[k]
        # Fall back to alternative type representation (str ↔ int).
        return d[str(k)] if isinstance(k, int) else d[int(k)]

    img = torch.stack([_get(img_dict, k) for k in image_keys], dim=0).float()
    txt = torch.stack([_get(txt_dict, k) for k in text_keys], dim=0).float()
    img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return img, txt


def _mms_same_modality(act: torch.Tensor, ext: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form same-modality MMS, diagonal excluded.

    For each latent k, with a = act[:, k] (>=0) and unit-norm rows e_i in ext:
        sum_{i!=j} a_i a_j <e_i, e_j> = || sum_i a_i e_i ||^2 - sum_i a_i^2 ||e_i||^2
        sum_{i!=j} a_i a_j           = (sum_i a_i)^2          - sum_i a_i^2
    Since rows of ext are unit-norm, ||e_i||^2 = 1 → second sum simplifies.

    Cost: one (L, d) = (L, N) @ (N, d) matmul.
    """
    fire = (act > 0).sum(dim=0)  # (L,)
    mu = act.T @ ext             # (L, d)
    num_full = (mu * mu).sum(-1) # || sum a_i e_i ||^2
    a_sum = act.sum(dim=0)       # (L,)
    a2_sum = (act ** 2).sum(dim=0)
    num = num_full - a2_sum
    den = a_sum * a_sum - a2_sum
    mms = torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.zeros_like(num))
    return mms.double().cpu(), fire.cpu()


def _mms_cross_modality(act_a: torch.Tensor, ext_a: torch.Tensor,
                        act_b: torch.Tensor, ext_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form cross-modal MMS — all (i, j) pairs (no diagonal exclusion).

        sum_ij a_i^(a) a_j^(b) <e_i^(a), e_j^(b)> = < sum_i a_i^(a) e_i^(a), sum_j a_j^(b) e_j^(b) >
        sum_ij a_i^(a) a_j^(b)                    = (sum_i a_i^(a)) (sum_j a_j^(b))
    """
    fire_a = (act_a > 0).sum(dim=0)
    fire_b = (act_b > 0).sum(dim=0)
    mu_a = act_a.T @ ext_a  # (L, d)
    mu_b = act_b.T @ ext_b  # (L, d)
    num = (mu_a * mu_b).sum(-1)
    den = act_a.sum(dim=0) * act_b.sum(dim=0)
    mms = torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.zeros_like(num))
    return mms.double().cpu(), fire_a.cpu(), fire_b.cpu()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)

    perm = None
    if args.method == "ours":
        if args.perm is None:
            raise SystemExit("--perm required for method='ours'")
        perm = np.load(args.perm)["perm"]

    image_keys, text_keys, img, txt = _load_paired_tensors(args.train_cache, args.dataset, args.split)
    M, N = img.shape[0], txt.shape[0]
    logger.info("%s/%s: images=%d captions=%d", args.dataset, args.split, M, N)

    t0 = time.time()
    z_img = eval_utils.encode_image(model, img, args.method, device, args.batch_size).cpu()
    z_txt = eval_utils.encode_text(model, txt, args.method, device, perm=perm, batch_size=args.batch_size).cpu()
    logger.info("SAE encode dt=%.1fs  z_img=%s z_txt=%s", time.time() - t0, z_img.shape, z_txt.shape)
    L = z_img.shape[1]

    logger.info("loading external embeddings from %s", args.ext_cache)
    ext_img, ext_txt = _load_ext_embeddings(args.ext_cache, image_keys, text_keys)
    assert ext_img.shape[0] == M and ext_txt.shape[0] == N

    # Move heavy work to CUDA if available — per-latent ops are small but L=4-8K loops add up.
    z_img_d = z_img.to(device)
    z_txt_d = z_txt.to(device)
    ext_img_d = ext_img.to(device)
    ext_txt_d = ext_txt.to(device)

    logger.info("computing MMS(img, img)")
    mms_ii, fire_i = _mms_same_modality(z_img_d, ext_img_d)
    logger.info("computing MMS(txt, txt)")
    mms_tt, fire_t = _mms_same_modality(z_txt_d, ext_txt_d)
    logger.info("computing MMS(img, txt)")
    mms_it, fire_i2, _ = _mms_cross_modality(z_img_d, ext_img_d, z_txt_d, ext_txt_d)
    assert torch.equal(fire_i, fire_i2)

    fire_i_np = fire_i.cpu().numpy()
    fire_t_np = fire_t.cpu().numpy()
    mms_ii_np = mms_ii.cpu().numpy()
    mms_tt_np = mms_tt.cpu().numpy()
    mms_it_np = mms_it.cpu().numpy()

    # Per-latent CSV
    csv_path = out_dir / "mms_per_latent.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latent_idx", "fire_img", "fire_txt", "mms_ii", "mms_tt", "mms_it"])
        for k in range(L):
            w.writerow([
                k, int(fire_i_np[k]), int(fire_t_np[k]),
                float(mms_ii_np[k]), float(mms_tt_np[k]), float(mms_it_np[k]),
            ])

    # Sorted curves: paper Fig 4 — descending, normalized neuron index in [0, 1]
    def _sorted_desc(arr: np.ndarray) -> list[float]:
        return sorted(arr.tolist(), reverse=True)

    # Aggregate over "alive in BOTH modalities" set (defined per pair)
    alive_ii = fire_i_np >= 2  # need >=2 for diagonal-excluded same-modality
    alive_tt = fire_t_np >= 2
    alive_it = (fire_i_np >= 1) & (fire_t_np >= 1)
    alive_both = (fire_i_np >= 1) & (fire_t_np >= 1)  # alias for cross

    summary: dict = {
        "variant": args.variant,
        "method": args.method,
        "split": args.split,
        "ext_cache": args.ext_cache,
        "n_images": int(M),
        "n_captions": int(N),
        "latent_size": int(L),
        "alive_img": int(alive_ii.sum()),
        "alive_txt": int(alive_tt.sum()),
        "alive_both": int(alive_both.sum()),
        "mms_ii_mean": float(mms_ii_np[alive_ii].mean()) if alive_ii.any() else 0.0,
        "mms_ii_median": float(np.median(mms_ii_np[alive_ii])) if alive_ii.any() else 0.0,
        "mms_tt_mean": float(mms_tt_np[alive_tt].mean()) if alive_tt.any() else 0.0,
        "mms_tt_median": float(np.median(mms_tt_np[alive_tt])) if alive_tt.any() else 0.0,
        "mms_it_mean": float(mms_it_np[alive_it].mean()) if alive_it.any() else 0.0,
        "mms_it_median": float(np.median(mms_it_np[alive_it])) if alive_it.any() else 0.0,
        # Full sorted curves (length L) — paper Fig 4 plots these directly
        "sorted_ii": _sorted_desc(mms_ii_np),
        "sorted_tt": _sorted_desc(mms_tt_np),
        "sorted_it": _sorted_desc(mms_it_np),
    }
    with open(out_dir / "mms_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "summary  alive(i/t/both)=%d/%d/%d  mean MMS(ii/tt/it)=%.3f/%.3f/%.3f",
        summary["alive_img"], summary["alive_txt"], summary["alive_both"],
        summary["mms_ii_mean"], summary["mms_tt_mean"], summary["mms_it_mean"],
    )
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
