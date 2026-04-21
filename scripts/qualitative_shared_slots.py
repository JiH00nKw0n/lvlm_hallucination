"""Qualitative check: do 'shared_slot' activations match on the SAME image-caption pairs?

For a given variant with aux loss, take the top-N slots classified as 'shared'
(fire-rate ratio in [0.4, 0.6]) and for each slot:
  - Find top-K image samples with highest z_I[slot] activation
  - Find top-K text samples with highest z_T[slot] activation
  - Compute overlap via image_id (from splits.json).
  - Caption overlap = |image_ids from image-top ∩ image_ids from text-top| / K

If a slot truly represents a cross-modal concept, the SAME images
(concept visible) should have their captions (concept mentioned) in the
text-top. Overlap should be HIGH. If low, the slot is picking up
different image-world vs text-world concepts.

Also: compare against image_dominant slots (r > 0.9) — these should have
LOW overlap since text side barely fires on them.

Usage:
    python scripts/qualitative_shared_slots.py --variant naive_once --k-top 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_alpha"))
import _bootstrap  # noqa: F401

import numpy as np
import torch
from safetensors.torch import load_file

from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from src.models.configuration_sae import TwoSidedTopKSAEConfig
from src.models.modeling_sae import TwoSidedTopKSAE


def load_model(variant_dir: Path, device: torch.device) -> TwoSidedTopKSAE:
    state = load_file(str(variant_dir / "final" / "model.safetensors"))
    cfg_path = variant_dir / "config.json"
    user_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    per_side = state["image_sae.W_dec"].shape[0]
    hidden = state["image_sae.W_dec"].shape[1]
    k = int(user_cfg.get("k", 8))
    cfg = TwoSidedTopKSAEConfig(hidden_size=hidden, latent_size=per_side * 2, k=k, normalize_decoder=True)
    model = TwoSidedTopKSAE(cfg).to(device).eval()
    model.load_state_dict(state, strict=False)
    return model


def compute_activations_and_rates(model, img, txt, bs=2048, device=None):
    """Stream forward: compute per-slot firing rates and keep raw activations on disk-efficient basis.

    Returns:
        z_i_all: (N, n) CPU tensor of image-side activations
        z_t_all: (N, n) CPU tensor of text-side activations
        rates_i, rates_t: per-slot firing rates
    """
    if device is None:
        device = next(model.parameters()).device
    sae_i = model.image_sae
    sae_t = model.text_sae
    n = int(sae_i.latent_size)
    N = img.shape[0]
    z_i_all = torch.zeros(N, n, dtype=torch.float32)
    z_t_all = torch.zeros(N, n, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, N, bs):
            ib = img[s:s+bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            tb = txt[s:s+bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            zi = sae_i(hidden_states=ib, return_dense_latents=True).dense_latents.squeeze(1).cpu()
            zt = sae_t(hidden_states=tb, return_dense_latents=True).dense_latents.squeeze(1).cpu()
            z_i_all[s:s+bs] = zi
            z_t_all[s:s+bs] = zt
    rates_i = (z_i_all > 0).float().mean(dim=0).numpy()
    rates_t = (z_t_all > 0).float().mean(dim=0).numpy()
    return z_i_all, z_t_all, rates_i, rates_t


def classify_slots(rates_i: np.ndarray, rates_t: np.ndarray, tau_alive: float = 1e-3):
    alive = (rates_i > tau_alive) | (rates_t > tau_alive)
    r = rates_i / np.clip(rates_i + rates_t, 1e-12, None)
    shared = alive & (r >= 0.4) & (r <= 0.6)
    img_dom = alive & (r > 0.9)
    txt_dom = alive & (r < 0.1)
    return np.where(shared)[0], np.where(img_dom)[0], np.where(txt_dom)[0], r


def topk_for_slot(z, slot, k):
    """Return indices of top-k activations (desc) for column `slot`."""
    col = z[:, slot]
    vals, idx = torch.topk(col, k=min(k, col.numel()))
    return idx.numpy(), vals.numpy()


def analyze(variant: str, k_slots: int, k_top: int, cache_dir: str, runs_root: str) -> None:
    device = torch.device("cpu")
    runs_root_p = Path(runs_root)
    vd = runs_root_p / variant
    if not (vd / "final/model.safetensors").exists():
        raise SystemExit(f"No model at {vd}/final")

    ds = CachedClipPairsDataset(cache_dir, split="train", l2_normalize=True)
    N = len(ds)
    print(f"Loaded {N} train pairs")

    img = torch.stack([ds[i]["image_embeds"] for i in range(N)], dim=0)
    txt = torch.stack([ds[i]["text_embeds"] for i in range(N)], dim=0)

    # splits.json: train -> [[img_id, caption_idx], ...]
    splits = json.loads(Path(cache_dir, "splits.json").read_text())
    train_pairs = splits["train"]  # length N
    # map index -> image_id string
    img_ids = np.array([p[0] for p in train_pairs], dtype=object)

    model = load_model(vd, device)
    print(f"Loaded variant: {variant}")

    z_i, z_t, rates_i, rates_t = compute_activations_and_rates(model, img, txt, bs=2048, device=device)
    print(f"Activations computed. alive_i={int((rates_i>1e-3).sum())} alive_t={int((rates_t>1e-3).sum())}")

    shared_idx, img_dom_idx, txt_dom_idx, r = classify_slots(rates_i, rates_t)
    print(f"Slot classification: shared={len(shared_idx)} img_dom={len(img_dom_idx)} txt_dom={len(txt_dom_idx)}")

    # For each group, sample k_slots and compute overlap
    def analyze_group(slots: np.ndarray, tag: str, max_slots: int = k_slots) -> dict:
        if slots.size == 0:
            return {}
        # Sort shared slots by combined fire-rate (prefer slots that fire a lot → more reliable)
        combined = rates_i[slots] + rates_t[slots]
        order = np.argsort(-combined)
        picked = slots[order[:max_slots]]

        print(f"\n=== {tag} slots (top {len(picked)}): ===")
        print(f"{'slot':>6} {'r':>5} {'f_i':>6} {'f_t':>6} {'overlap@K':>10} "
              f"{'img_top_ids':>25}  {'txt_top_ids':>25}")
        results = []
        for sl in picked:
            img_idx, _ = topk_for_slot(z_i, sl, k_top)
            txt_idx, _ = topk_for_slot(z_t, sl, k_top)
            img_ids_top = set(img_ids[img_idx].tolist())
            txt_ids_top = set(img_ids[txt_idx].tolist())
            overlap = img_ids_top & txt_ids_top
            overlap_frac = len(overlap) / max(len(img_ids_top | txt_ids_top), 1)  # Jaccard
            ri_s = f"{r[sl]:.2f}"
            fi_s = f"{rates_i[sl]:.3f}"
            ft_s = f"{rates_t[sl]:.3f}"
            # Show first 3 image ids from each side (truncated)
            img_ids_str = ",".join(list(img_ids[img_idx].tolist())[:3])
            txt_ids_str = ",".join(list(img_ids[txt_idx].tolist())[:3])
            print(f"{sl:>6} {ri_s:>5} {fi_s:>6} {ft_s:>6} {overlap_frac:>10.3f} "
                  f"{img_ids_str[:25]:>25s}  {txt_ids_str[:25]:>25s}")
            results.append({
                "slot": int(sl), "r": float(r[sl]), "fire_i": float(rates_i[sl]), "fire_t": float(rates_t[sl]),
                "jaccard_overlap": float(overlap_frac),
                "img_top_ids": [str(x) for x in img_ids[img_idx].tolist()],
                "txt_top_ids": [str(x) for x in img_ids[txt_idx].tolist()],
            })
        mean_overlap = float(np.mean([r["jaccard_overlap"] for r in results]))
        median_overlap = float(np.median([r["jaccard_overlap"] for r in results]))
        print(f"{tag} overlap mean={mean_overlap:.3f} median={median_overlap:.3f}")
        return {"group": tag, "mean_overlap": mean_overlap, "median_overlap": median_overlap, "slots": results}

    shared_stats = analyze_group(shared_idx, "shared (0.4≤r≤0.6)")
    # For baseline comparison, sample from image-dominant slots (r>0.9)
    # Only image-top makes sense here since text side barely fires
    img_dom_stats = analyze_group(img_dom_idx[:k_slots], "image_dominant (r>0.9)")
    txt_dom_stats = analyze_group(txt_dom_idx[:k_slots], "text_dominant (r<0.1)")

    out_dir = Path("outputs/aux_alignment_clip_b32/qualitative")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{variant}_shared_slot_overlap.json", "w") as f:
        json.dump({"variant": variant,
                   "shared": shared_stats,
                   "image_dominant": img_dom_stats,
                   "text_dominant": txt_dom_stats}, f, indent=2)
    print(f"\nSaved to outputs/aux_alignment_clip_b32/qualitative/{variant}_shared_slot_overlap.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="naive_once")
    p.add_argument("--runs-root", default="outputs/aux_alignment_clip_b32")
    p.add_argument("--cache-dir", default="cache/clip_b32_coco")
    p.add_argument("--k-slots", type=int, default=10, help="top-N slots per group")
    p.add_argument("--k-top", type=int, default=20, help="top-K activations per slot")
    args = p.parse_args()

    analyze(args.variant, args.k_slots, args.k_top, args.cache_dir, args.runs_root)


if __name__ == "__main__":
    main()
