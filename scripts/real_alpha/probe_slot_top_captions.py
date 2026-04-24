#!/usr/bin/env python3
"""For a given text-side slot, stream through train split and print top-N activating captions."""

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

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore
from src.models.configuration_sae import TwoSidedTopKSAEConfig  # type: ignore
from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--slot-txt", type=int, required=True,
                   help="text-side slot index to probe")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--caption-json", default="cache/coco_karpathy_captions.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rd = Path(args.run_dir)
    state = load_file(str(rd / "model.safetensors"))
    per_side = state["image_sae.W_dec"].shape[0]
    hidden = state["image_sae.W_dec"].shape[1]
    cfg = TwoSidedTopKSAEConfig(hidden_size=hidden, latent_size=per_side * 2, k=8, normalize_decoder=True)
    model = TwoSidedTopKSAE(cfg).to(device).eval()
    model.load_state_dict(state, strict=False)

    ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
    N = len(ds)
    slot = args.slot_txt

    # min-heap of (score, pair_idx)
    heap: list = []
    cap = args.top_n

    with torch.no_grad():
        for s in range(0, N, args.batch_size):
            e = min(s + args.batch_size, N)
            txts = torch.stack([ds[i]["text_embeds"] for i in range(s, e)]).to(device, dtype=torch.float32)
            out = model.text_sae(hidden_states=txts.unsqueeze(1), return_dense_latents=True)
            zt = out.dense_latents.squeeze(1)[:, slot].cpu().numpy()  # (B,)
            for bi, v in enumerate(zt):
                if v <= 0:
                    continue
                score = float(v)
                if len(heap) < cap:
                    heapq.heappush(heap, (score, int(s + bi)))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, int(s + bi)))

    caps = json.load(open(args.caption_json)) if Path(args.caption_json).exists() else {}
    print(f"Top-{args.top_n} captions for text slot {slot}:")
    for score, pair_idx in sorted(heap, key=lambda t: -t[0]):
        img_id, cap_idx = ds.pairs[pair_idx]
        cap = caps.get(f"{img_id}::{cap_idx}", "")
        print(f"  score={score:.3f}  pair={pair_idx}  caption: {cap[:100]}")


if __name__ == "__main__":
    main()
