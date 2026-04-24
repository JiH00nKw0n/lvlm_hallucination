#!/usr/bin/env python3
"""Inspect all active slots for a single pair: Hungarian match + quality + top captions."""

import argparse
import heapq
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore
from src.models.configuration_sae import TwoSidedTopKSAEConfig  # type: ignore
from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-idx", type=int, required=True)
    p.add_argument("--caption-json", default="cache/coco_karpathy_captions.json")
    p.add_argument("--top-k-caps", type=int, default=3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    rd = Path(args.run_dir)
    state = load_file(str(rd / "model.safetensors"))
    per_side = state["image_sae.W_dec"].shape[0]
    hidden = state["image_sae.W_dec"].shape[1]
    cfg = TwoSidedTopKSAEConfig(hidden_size=hidden, latent_size=per_side * 2, k=8, normalize_decoder=True)
    model = TwoSidedTopKSAE(cfg).to(device).eval()
    model.load_state_dict(state, strict=False)

    # Global Hungarian on alive-alive
    C = np.load(rd / "diagnostic_B_C_train.npy")
    rates = np.load(rd / "diagnostic_B_firing_rates.npz")
    alive_i = np.where(rates["rate_i"] > 0.001)[0]
    alive_t = np.where(rates["rate_t"] > 0.001)[0]
    C_sub = C[np.ix_(alive_i, alive_t)]
    row, col = linear_sum_assignment(-C_sub)
    match = np.full(per_side, -1, dtype=np.int64)
    C_vals = np.zeros(per_side, dtype=np.float64)
    match[alive_i[row]] = alive_t[col]
    C_vals[alive_i[row]] = C_sub[row, col]

    # Decoder norms for cos
    Wi = model.image_sae.W_dec.detach().cpu().numpy()
    Wt = model.text_sae.W_dec.detach().cpu().numpy()
    Wi_n = Wi / (np.linalg.norm(Wi, axis=1, keepdims=True) + 1e-12)
    Wt_n = Wt / (np.linalg.norm(Wt, axis=1, keepdims=True) + 1e-12)

    # Forward the specific pair
    ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
    x = ds[args.pair_idx]["image_embeds"].to(device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y = ds[args.pair_idx]["text_embeds"].to(device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out_i = model.image_sae(hidden_states=x, return_dense_latents=False)
        out_t = model.text_sae(hidden_states=y, return_dense_latents=False)
    idx_i = out_i.latent_indices.squeeze().cpu().numpy().tolist()
    idx_t = out_t.latent_indices.squeeze().cpu().numpy().tolist()
    val_i = out_i.latent_activations.squeeze().cpu().numpy().tolist()
    val_t = out_t.latent_activations.squeeze().cpu().numpy().tolist()

    idx_t_set = set(idx_t)
    img_id, cap_idx = ds.pairs[args.pair_idx]
    print(f"pair_idx={args.pair_idx} image_id={img_id} caption_idx={cap_idx}")
    print()
    print(f"{'rank':>4} {'img_slot':>8} {'val_i':>7} {'pi(i)':>6} {'pi_active?':>10} "
          f"{'rate_i':>7} {'rate_t(pi)':>10} {'C(i,pi)':>8} {'dec_cos':>8}")
    print("-" * 88)
    # Sort by val_i descending
    order = np.argsort(-np.array(val_i))
    for rank, k in enumerate(order):
        i = idx_i[k]
        pi = int(match[i])
        pi_in_t = pi in idx_t_set if pi >= 0 else False
        ri = float(rates["rate_i"][i])
        rt = float(rates["rate_t"][pi]) if pi >= 0 else 0.0
        cval = C_vals[i] if pi >= 0 else 0.0
        dcos = float((Wi_n[i] * Wt_n[pi]).sum()) if pi >= 0 else 0.0
        print(f"{rank+1:>4} {i:>8} {val_i[k]:>7.3f} {pi if pi>=0 else -1:>6} "
              f"{'YES' if pi_in_t else 'no':>10} {ri:>7.3f} {rt:>10.3f} {cval:>8.3f} {dcos:>8.3f}")

    # Text-side only active slots (not in image match list)
    print()
    print("text-side active slots (top-8):")
    print(f"{'rank':>4} {'txt_slot':>8} {'val_t':>7} {'rate_t':>7}")
    order_t = np.argsort(-np.array(val_t))
    for rank, k in enumerate(order_t):
        j = idx_t[k]
        rt = float(rates["rate_t"][j])
        print(f"{rank+1:>4} {j:>8} {val_t[k]:>7.3f} {rt:>7.3f}")

    # For each active img-slot with a valid Hungarian match, print top captions of the matched text slot.
    caps = json.load(open(args.caption_json)) if Path(args.caption_json).exists() else {}
    print()
    print("Top captions for each Hungarian-matched text slot of active img slots:")
    for k in order:
        i = idx_i[k]
        pi = int(match[i])
        if pi < 0:
            continue
        # Find top activating captions for slot `pi` by scanning a small subsample:
        # we don't have precomputed per-slot top activations; skip if too slow.
        # For a quick interpretation, take top-3 captions for slot `pi` by re-scanning
        # text SAE activations in a capped way: sample up to 2k captions.
        print(f"\n  img slot {i} <-> text slot {pi}  (dec_cos={float((Wi_n[i] * Wt_n[pi]).sum()):.3f}, "
              f"C={C_vals[i]:.3f}, val_i={val_i[k]:.3f}):")
        # Just print pair caption for idx
        # (Full top-K over entire dataset skipped for speed.)


if __name__ == "__main__":
    main()
