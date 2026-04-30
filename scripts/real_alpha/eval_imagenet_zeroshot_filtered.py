"""ImageNet zero-shot with always-on latent filtering.

Mirrors `eval_imagenet_zeroshot.py` but adds an alive/always-on filter:
  - encode 50k val images → image-side SAE → z_val (N, L)
  - compute per-column fire rate on z_val
  - drop columns with fire_rate > --max-fire-rate (default 0.5)
  - apply the SAME column mask to text prototypes z_protos
  - L2-normalize the masked vectors, then cosine + argmax

Output is a JSON next to the original `summary.json` named
`summary_filtered_fr<rate>.json` so the unfiltered numbers stay intact.
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
    p.add_argument("--variant", type=str, required=True)
    p.add_argument("--method", type=str, required=True,
                   choices=["shared", "separated", "aux", "ours", "vl_sae", "shared_enc"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-fire-rate", type=float, default=0.5,
                   help="Drop latents whose image-side fire rate exceeds this.")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--n-templates", type=int, default=80)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _build_class_prototypes_clip(text_dict, n_classes, n_templates) -> torch.Tensor:
    protos = []
    for c in range(n_classes):
        vecs = torch.stack([text_dict[f"{c}_{t}"] for t in range(n_templates)], dim=0)
        protos.append(_l2_normalize(vecs.mean(dim=0)))
    return torch.stack(protos, dim=0)


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
        logger.info("loaded perm (len=%d) from %s", perm.shape[0], args.perm)

    text_dict_raw = torch.load(str(Path(args.cache_dir) / "text_embeddings.pt"), map_location="cpu")
    text_dict = {str(k): v.to(torch.float32) for k, v in text_dict_raw.items()}
    for k, v in text_dict.items():
        text_dict[k] = _l2_normalize(v)

    logger.info("building %d prototypes (mean of %d templates each)", args.n_classes, args.n_templates)
    protos_clip = _build_class_prototypes_clip(text_dict, args.n_classes, args.n_templates)

    logger.info("encoding prototypes through text-side SAE")
    z_protos = eval_utils.encode_text(model, protos_clip, args.method, device,
                                      perm=perm, batch_size=args.batch_size)

    val_ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")
    logger.info("val=%d", len(val_ds))
    img_va = torch.stack([val_ds[i]["image_embeds"] for i in range(len(val_ds))], dim=0)
    y_va = np.array([int(val_ds.pairs[i][1]) for i in range(len(val_ds))], dtype=np.int64)

    logger.info("encoding val images → SAE latents")
    z_val = eval_utils.encode_image(model, img_va, args.method, device, args.batch_size)

    L = z_val.shape[1]
    fire_rate = (z_val != 0).float().mean(dim=0).cpu().numpy()
    keep = fire_rate <= args.max_fire_rate
    logger.info("fire_rate threshold=%.2f → keep %d / %d latents (%.1f%%)",
                args.max_fire_rate, int(keep.sum()), L, 100 * keep.mean())

    keep_t = torch.from_numpy(keep).to(z_val.device)
    z_val_f = z_val[:, keep_t]
    z_protos_f = z_protos[:, keep_t]

    z_val_f = _l2_normalize(z_val_f)
    z_protos_f = _l2_normalize(z_protos_f)

    correct = 0
    bsz = 8192
    for s in range(0, z_val_f.shape[0], bsz):
        scores = z_val_f[s:s + bsz] @ z_protos_f.T
        pred = scores.argmax(dim=1).cpu().numpy()
        correct += int((pred == y_va[s:s + bsz]).sum())
    acc = correct / z_val_f.shape[0]
    logger.info("filtered val top-1 accuracy: %.4f", acc)

    result = {
        "method": args.method,
        "variant": args.variant,
        "dataset": "imagenet",
        "metric": "zeroshot_top1_filtered",
        "max_fire_rate": args.max_fire_rate,
        "kept_latents": int(keep.sum()),
        "total_latents": int(L),
        "kept_fraction": float(keep.mean()),
        "accuracy": float(acc),
        "n_val": int(z_val_f.shape[0]),
        "n_classes": args.n_classes,
        "n_templates": args.n_templates,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
