"""Val-only ImageNet linear probe with modality-axis (dominant slot) masking.

For a given method/ckpt:
  1. Encode all val images through the image-side SAE (dense top-k).
  2. Find "dominant slots" — slots that are top-1 for ≥ `--dominant-threshold`
     fraction of samples. These act as modality-axis and are masked out.
  3. Recompute top-1 over the masked latent (still a one-hot encoding but
     ignoring the common direction).
  4. Fit a linear probe on val 50k (fit and evaluate on same set — this is a
     feature-separability diagnostic, not a generalization test).
  5. Report linprobe accuracy, slot-class summary, and list of masked slots.

Also optionally runs the analogous masked-top-1 zero-shot: mask same slots on
text prototypes, pick top-1, match with image top-1 via cosine.

Usage:
    python scripts/real_alpha/eval_imagenet_valprobe.py \
        --ckpt outputs/real_exp_v1/separated/imagenet/final \
        --method separated --cache-dir cache/clip_b32_imagenet \
        --output outputs/real_exp_v1/separated/imagenet/valprobe.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae", "shared_enc"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--perm", type=str, default=None, help="Path to perm.npz (for ours)")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dominant-threshold", type=float, default=0.1,
                   help="Fraction threshold: any slot that is top-1 for ≥ this share "
                        "of samples is masked. 0 disables masking.")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lp-lr", type=float, default=1e-2)
    p.add_argument("--lp-epochs", type=int, default=30)
    p.add_argument("--lp-batch-size", type=int, default=1024)
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--n-templates", type=int, default=80)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _pick_image_sae(model, method: str):
    if method in ("shared", "aux"):
        return model
    if method in ("vl_sae", "shared_enc"):
        return model  # shared encoder
    return model.image_sae


def _pick_text_sae(model, method: str):
    if method in ("shared", "aux"):
        return model
    if method in ("vl_sae", "shared_enc"):
        return model
    return model.text_sae


@torch.no_grad()
def _encode_dense(sae, embeds: torch.Tensor, batch_size: int, device: torch.device,
                  method: str) -> torch.Tensor:
    """Dense (N, L) top-k latents on CPU."""
    sae.eval(); sae.to(device)
    if method in ("vl_sae", "shared_enc"):
        L = int(sae.cfg.latent_size)
    else:
        L = int(sae.latent_size)
    N = embeds.shape[0]
    out = torch.empty(N, L, dtype=torch.float32)
    for s in range(0, N, batch_size):
        chunk = embeds[s:s + batch_size].to(device)
        if method in ("vl_sae", "shared_enc"):
            z = sae.encode(chunk)
        else:
            z = sae(hidden_states=chunk.unsqueeze(1), return_dense_latents=True).dense_latents.squeeze(1)
        out[s:s + chunk.shape[0]] = z.float().cpu()
    return out


def find_dominant_slots(z: torch.Tensor, threshold: float) -> list[int]:
    """Iteratively mask slots that are top-1 for ≥ threshold of samples.

    Returns list of slot indices masked, in order of dominance.
    """
    if threshold <= 0:
        return []
    N, L = z.shape
    masked_slots: list[int] = []
    z_curr = z.clone()
    while True:
        top1 = z_curr.argmax(dim=-1)
        counts = torch.bincount(top1, minlength=L).cpu().numpy()
        top_slot = int(counts.argmax())
        frac = counts[top_slot] / N
        if frac < threshold:
            break
        masked_slots.append(top_slot)
        z_curr[:, top_slot] = -float("inf")
    return masked_slots


def apply_mask_top1(z: torch.Tensor, masked_slots: list[int]) -> torch.Tensor:
    """Zero-out masked slots, return dense (N, L) with only top-1 of remaining nonzero."""
    z_masked = z.clone()
    for s in masked_slots:
        z_masked[:, s] = 0
    top1 = z_masked.argmax(dim=-1)  # deterministic (argmax of zero tensor returns 0 but we ensure nonzero above)
    # Build one-hot-like representation with the remaining top-1 value kept
    out = torch.zeros_like(z_masked)
    out.scatter_(-1, top1.unsqueeze(-1), z_masked.gather(-1, top1.unsqueeze(-1)))
    return out, top1.cpu().numpy()


def slot_class_summary(top1: np.ndarray, y: np.ndarray) -> dict:
    slot_class = {}
    for s, c in zip(top1, y):
        d = slot_class.setdefault(int(s), {})
        d[int(c)] = d.get(int(c), 0) + 1
    rows = []
    for slot, d in slot_class.items():
        n = sum(d.values())
        pc, pn = max(d.items(), key=lambda kv: kv[1])
        rows.append({"slot": slot, "n": n, "primary_class": pc, "primary_count": pn, "purity": pn / n, "n_classes": len(d)})
    rows.sort(key=lambda r: -r["n"])
    N = sum(r["n"] for r in rows)
    summary = {"n_alive_slots": len(rows), "n_samples": N}
    for tau in [0.9, 0.7, 0.5, 0.3]:
        kept = [r for r in rows if r["purity"] >= tau]
        classes = {r["primary_class"] for r in kept}
        cov = sum(r["n"] for r in kept)
        summary[f"n_slots_purity_ge_{tau}"] = len(kept)
        summary[f"distinct_primary_classes_purity_ge_{tau}"] = len(classes)
        summary[f"samples_frac_purity_ge_{tau}"] = cov / max(1, N)
    return {"summary": summary, "top_slots": rows[:20]}


def fit_val_probe(z_top1: torch.Tensor, y: np.ndarray, n_classes: int, device: torch.device,
                  lr: float, epochs: int, batch_size: int) -> list[dict]:
    L = z_top1.shape[1]
    head = nn.Linear(L, n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    y_t = torch.as_tensor(y, dtype=torch.long)
    N = z_top1.shape[0]
    history = []
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            xb = z_top1[idx].to(device)
            yb = y_t[idx].to(device)
            logits = head(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * xb.shape[0]
        tot /= N
        with torch.no_grad():
            correct = 0
            for s in range(0, N, batch_size):
                xb = z_top1[s:s + batch_size].to(device)
                pred = head(xb).argmax(-1).cpu().numpy()
                correct += int((pred == y[s:s + batch_size]).sum())
            acc = correct / N
        history.append({"ep": ep + 1, "train_loss": tot, "acc": acc})
    return history


def run_masked_zeroshot(
    model, method: str, perm: np.ndarray | None,
    img_val: torch.Tensor, y_val: np.ndarray,
    cache_dir: str, n_classes: int, n_templates: int,
    masked_slots_img: list[int],
    batch_size: int, device: torch.device,
) -> float:
    """Top-1 zero-shot with masked dominant slots on image side.

    Text prototype: mean over 80 templates in CLIP space → SAE → masked top-1.
    Match via cosine similarity (since both are essentially one-hot, this is
    equivalent to slot-equality check weighted by activations).
    """
    # Build prototypes in CLIP space
    text_dict = torch.load(str(_Path(cache_dir) / "text_embeddings.pt"), map_location="cpu")
    text_dict = {str(k): v.to(torch.float32) for k, v in text_dict.items()}
    protos = []
    for c in range(n_classes):
        vecs = torch.stack([text_dict[f"{c}_{t}"] for t in range(n_templates)], dim=0)
        protos.append(_l2_normalize(vecs.mean(dim=0)))
    protos = torch.stack(protos, dim=0)

    text_sae = _pick_text_sae(model, method)
    z_proto = _encode_dense(text_sae, protos, batch_size, device, method)
    if method == "ours" and perm is not None:
        z_proto = z_proto[:, torch.as_tensor(perm, dtype=torch.long)]
    # Mask dominant slots on text protos (use same image-side mask set)
    for s in masked_slots_img:
        z_proto[:, s] = 0
    # Top-1 of masked protos
    top1_proto = z_proto.argmax(dim=-1)
    z_proto_top1 = torch.zeros_like(z_proto)
    z_proto_top1.scatter_(-1, top1_proto.unsqueeze(-1), z_proto.gather(-1, top1_proto.unsqueeze(-1)))
    z_proto_top1 = _l2_normalize(z_proto_top1)

    # Image side
    image_sae = _pick_image_sae(model, method)
    z_img = _encode_dense(image_sae, img_val, batch_size, device, method)
    for s in masked_slots_img:
        z_img[:, s] = 0
    top1_img = z_img.argmax(dim=-1)
    z_img_top1 = torch.zeros_like(z_img)
    z_img_top1.scatter_(-1, top1_img.unsqueeze(-1), z_img.gather(-1, top1_img.unsqueeze(-1)))
    z_img_top1 = _l2_normalize(z_img_top1)

    # cosine = z_img @ z_proto_top1.T  → (N_val, n_classes)
    correct = 0
    bsz = 8192
    for s in range(0, z_img_top1.shape[0], bsz):
        scores = z_img_top1[s:s + bsz] @ z_proto_top1.T
        pred = scores.argmax(dim=-1).numpy()
        correct += int((pred == y_val[s:s + bsz]).sum())
    return correct / z_img_top1.shape[0]


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = _Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading ckpt=%s method=%s", args.ckpt, args.method)
    model = eval_utils.load_sae(args.ckpt, args.method)
    image_sae = _pick_image_sae(model, args.method)
    perm = None
    if args.method == "ours":
        if args.perm is None:
            raise SystemExit("--perm required for method='ours'")
        perm = np.load(args.perm)["perm"]

    logger.info("loading ImageNet val")
    ds = eval_utils.load_pair_dataset(args.cache_dir, "imagenet", "val")
    N = len(ds)
    img = torch.stack([ds[i]["image_embeds"] for i in range(N)], dim=0)
    y = np.array([int(ds.pairs[i][1]) for i in range(N)], dtype=np.int64)
    logger.info("val: N=%d", N)

    logger.info("encoding val → dense top-k latents")
    z_img = _encode_dense(image_sae, img, args.batch_size, device, args.method)
    # For ours, permute text side — but linprobe only uses image side, so perm doesn't apply here.

    # Find dominant slots (common activation modes)
    masked = find_dominant_slots(z_img, args.dominant_threshold)
    logger.info("masked slots (top-1 freq ≥ %.1f%%): %s", 100 * args.dominant_threshold, masked)

    # Apply masking → top-1 on remainder
    z_top1, top1_idx = apply_mask_top1(z_img, masked)

    sc_report = slot_class_summary(top1_idx, y)

    logger.info("fitting val probe (lr=%.3g, epochs=%d, batch=%d)",
                args.lp_lr, args.lp_epochs, args.lp_batch_size)
    history = fit_val_probe(z_top1, y, args.n_classes, device,
                            args.lp_lr, args.lp_epochs, args.lp_batch_size)
    best_acc = max(h["acc"] for h in history)
    final_acc = history[-1]["acc"]
    logger.info("val linprobe: best=%.4f, final=%.4f", best_acc, final_acc)

    # Masked zero-shot
    logger.info("running masked top-1 zero-shot")
    try:
        zs_acc = run_masked_zeroshot(
            model, args.method, perm, img, y, args.cache_dir,
            args.n_classes, args.n_templates, masked, args.batch_size, device,
        )
        logger.info("masked zeroshot: %.4f", zs_acc)
    except Exception as e:
        logger.warning("zeroshot failed: %s", e)
        zs_acc = None

    result = {
        "method": args.method,
        "ckpt": args.ckpt,
        "dominant_threshold": args.dominant_threshold,
        "masked_slots": masked,
        "n_masked": len(masked),
        "linprobe_best_acc": best_acc,
        "linprobe_final_acc": final_acc,
        "linprobe_history": history,
        "masked_zeroshot_acc": zs_acc,
        "slot_class_summary": sc_report["summary"],
        "top_slots_after_mask": sc_report["top_slots"],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
