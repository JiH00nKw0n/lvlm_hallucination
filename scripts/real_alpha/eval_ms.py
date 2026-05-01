"""MonoSemanticity score (MS) — Pach et al. 2025 (NeurIPS), §3.2 Eq 9.

For each SAE neuron k and a single modality m ∈ {image, text}:
  ã^k = (a^k - min a^k) / (max a^k - min a^k)            ∈ [0, 1]^N    (Eq 7)
  R^k = ã^k ⊗ ã^k                                                       (Eq 8)
  S   = pairwise cosine of E(x_n), E(x_m) from a separate encoder E    (Eq 5)
  MS^k = Σ_{n<m} R^k_nm S_nm / Σ_{n<m} R^k_nm                          (Eq 9)

Closed-form vectorized across neurons (avoids the N×N×K relevance tensor):
  num = (a^T S a) - Σ_n a_n^2          # since S_nn = 1
  den = (Σ_n a_n)^2 - Σ_n a_n^2
  MS  = num / den         (zero where den ≤ 0)
where a is the (N, K) min-max normalized activation matrix and S = E Eᵀ ∈ ℝ^{N×N}.

Outputs (per variant, written under <out>/):
  ms_per_latent.csv     latent_idx, fire_image, fire_text, ms_image, ms_text
  ms_summary.json       sorted curves, means/medians on alive sets
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
                   help="separate-encoder cache for the similarity matrix (e.g. cache/metaclip_b32_coco)")
    p.add_argument("--dataset", choices=["coco", "cc3m"], default="coco",
                   help="coco: 5:1 multi-caption (uses unique images for image-side, all captions for text-side); "
                        "cc3m: 1:1 pairing.")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--perm", type=str, default=None, help="required when method=ours")
    p.add_argument("--out", type=str, required=True, help="output dir")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="encoding batch size for SAE forward")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _load_paired_tensors(cache_dir: str, dataset: str, split: str):
    """Return (image_keys, text_keys, img_tensor, txt_tensor). Same convention as eval_mms.py."""
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
        return d[str(k)] if isinstance(k, int) else d[int(k)]

    img = torch.stack([_get(img_dict, k) for k in image_keys], dim=0).float()
    txt = torch.stack([_get(txt_dict, k) for k in text_keys], dim=0).float()
    img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return img, txt


def _ms_per_neuron(act: torch.Tensor, ext: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Pach MS score over all neurons.

    act: (N, K) raw activations (>=0 for ReLU/TopK; we min-max normalize → [0,1]).
    ext: (N, d) L2-normalized external embeddings.

    Returns:
      ms:   (K,) double tensor (0 where denominator ≤ 0)
      fire: (K,) int tensor — number of images where neuron activated (a > 0)
    """
    fire = (act > 0).sum(dim=0)
    a_min = act.min(dim=0).values
    a_max = act.max(dim=0).values
    rng = (a_max - a_min).clamp_min(1e-12)
    a = (act - a_min) / rng                  # (N, K) ∈ [0, 1]
    a = a.float()

    # S a where S = ext @ ext^T (avoid materializing N x N S):
    #   S a = ext @ (ext^T @ a)
    Sa = ext @ (ext.T @ a)                   # (N, K)
    a_dot_Sa = (a * Sa).sum(dim=0)           # Σ_{n,m} a_n a_m S_nm  (= a^T S a)
    a_sq = (a * a).sum(dim=0)                # Σ_n a_n^2 (= Σ a_n^2 S_nn since S_nn=1)
    a_sum = a.sum(dim=0)
    num = a_dot_Sa - a_sq                    # 2 × Σ_{n<m}, but factor cancels with den
    den = (a_sum * a_sum) - a_sq
    ms = torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.zeros_like(num))
    return ms.double().cpu(), fire.cpu()


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

    z_img_d = z_img.to(device)
    z_txt_d = z_txt.to(device)
    ext_img_d = ext_img.to(device)
    ext_txt_d = ext_txt.to(device)

    logger.info("computing MS(image)")
    ms_image, fire_i = _ms_per_neuron(z_img_d, ext_img_d)
    logger.info("computing MS(text)")
    ms_text, fire_t = _ms_per_neuron(z_txt_d, ext_txt_d)

    fire_i_np = fire_i.cpu().numpy()
    fire_t_np = fire_t.cpu().numpy()
    ms_image_np = ms_image.cpu().numpy()
    ms_text_np = ms_text.cpu().numpy()

    csv_path = out_dir / "ms_per_latent.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latent_idx", "fire_image", "fire_text", "ms_image", "ms_text"])
        for k in range(L):
            w.writerow([k, int(fire_i_np[k]), int(fire_t_np[k]),
                        float(ms_image_np[k]), float(ms_text_np[k])])

    def _sorted_desc(arr: np.ndarray) -> list[float]:
        return sorted(arr.tolist(), reverse=True)

    alive_i = fire_i_np >= 2
    alive_t = fire_t_np >= 2
    summary: dict = {
        "variant": args.variant,
        "method": args.method,
        "dataset": args.dataset,
        "split": args.split,
        "ext_cache": args.ext_cache,
        "n_images": int(M),
        "n_captions": int(N),
        "latent_size": int(L),
        "alive_image": int(alive_i.sum()),
        "alive_text": int(alive_t.sum()),
        "ms_image_mean": float(ms_image_np[alive_i].mean()) if alive_i.any() else 0.0,
        "ms_image_median": float(np.median(ms_image_np[alive_i])) if alive_i.any() else 0.0,
        "ms_text_mean": float(ms_text_np[alive_t].mean()) if alive_t.any() else 0.0,
        "ms_text_median": float(np.median(ms_text_np[alive_t])) if alive_t.any() else 0.0,
        "sorted_image": _sorted_desc(ms_image_np),
        "sorted_text": _sorted_desc(ms_text_np),
    }
    with open(out_dir / "ms_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "summary  alive(i/t)=%d/%d  mean MS(i/t)=%.3f/%.3f",
        summary["alive_image"], summary["alive_text"],
        summary["ms_image_mean"], summary["ms_text_mean"],
    )
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
