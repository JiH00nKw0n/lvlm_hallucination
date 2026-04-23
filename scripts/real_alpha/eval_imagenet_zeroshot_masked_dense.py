"""ImageNet-1K zero-shot classification — dense + dominant-slot masking.

Middle ground between eval_imagenet_zeroshot.py (raw, no mask, all slots) and
eval_imagenet_valprobe.py masked_zs (dominant-masked + top-1 only):

  1. Encode val images and class prototypes through SAE.
  2. Find dominant slots on val image latents (top-1 for ≥ threshold of samples).
  3. Zero those slots in BOTH val latent and proto latent.
  4. Cosine argmax on remaining dense latent (no top-1 restriction).

Isolates the effect of dominant-slot masking alone, holding dense vs top-1 fixed.
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
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae", "shared_enc"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dominant-threshold", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--n-templates", type=int, default=80)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def find_dominant(z: torch.Tensor, threshold: float) -> list[int]:
    if threshold <= 0:
        return []
    N, L = z.shape
    masked: list[int] = []
    zc = z.clone()
    while True:
        top1 = zc.argmax(dim=-1)
        counts = torch.bincount(top1, minlength=L).cpu().numpy()
        s = int(counts.argmax())
        frac = counts[s] / N
        if frac < threshold:
            break
        masked.append(s)
        zc[:, s] = -float("inf")
    return masked


def _build_protos_clip(text_dict, n_classes, n_templates):
    protos = []
    for c in range(n_classes):
        vecs = torch.stack([text_dict[f"{c}_{t}"] for t in range(n_templates)], dim=0)
        protos.append(_l2_normalize(vecs.mean(dim=0)))
    return torch.stack(protos, dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("load ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)

    perm = None
    if args.method == "ours":
        if args.perm is None:
            raise SystemExit("--perm required for method='ours'")
        perm = np.load(args.perm)["perm"]

    # Class prototypes
    text_dict_raw = torch.load(str(Path(args.cache_dir) / "text_embeddings.pt"), map_location="cpu")
    text_dict = {str(k): _l2_normalize(v.to(torch.float32)) for k, v in text_dict_raw.items()}
    protos_clip = _build_protos_clip(text_dict, args.n_classes, args.n_templates)

    logger.info("encode prototypes and val images")
    z_protos = eval_utils.encode_text(model, protos_clip, args.method, device, perm=perm, batch_size=args.batch_size)
    val_ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")
    img_va = torch.stack([val_ds[i]["image_embeds"] for i in range(len(val_ds))], dim=0)
    y_va = np.array([int(val_ds.pairs[i][1]) for i in range(len(val_ds))], dtype=np.int64)
    z_val = eval_utils.encode_image(model, img_va, args.method, device, args.batch_size)

    # Find dominant slots on val latents; zero them in BOTH
    masked = find_dominant(z_val, args.dominant_threshold)
    logger.info("dominant slots masked (>=%.0f%%): %s", 100 * args.dominant_threshold, masked)
    for s in masked:
        z_val[:, s] = 0
        z_protos[:, s] = 0

    # Dense cosine argmax (no top-1 restriction)
    z_val = eval_utils.normalize_rows(z_val)
    z_protos = eval_utils.normalize_rows(z_protos)

    correct = 0
    bsz = 8192
    for s in range(0, z_val.shape[0], bsz):
        chunk = z_val[s:s + bsz].to(device)
        logits = chunk @ z_protos.to(device).T
        pred = logits.argmax(dim=-1).cpu().numpy()
        correct += int((pred == y_va[s:s + bsz]).sum())
    acc = correct / len(y_va)
    logger.info("masked-dense zeroshot: %.4f", acc)

    result = {
        "method": args.method,
        "ckpt": args.ckpt,
        "dominant_threshold": args.dominant_threshold,
        "masked_slots": [int(m) for m in masked],
        "n_classes": args.n_classes,
        "n_templates": args.n_templates,
        "accuracy": acc,
        "n_val": int(len(y_va)),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
