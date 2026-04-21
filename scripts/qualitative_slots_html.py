"""Generate a human-browsable HTML viewer for shared/dominant slots.

For a given variant, find:
  - top-N "shared" slots (fire rate ratio in [0.4, 0.6])
  - top-N "image-dominant" slots (r > 0.9)
  - top-N "text-dominant" slots (r < 0.1)

For each slot, show:
  - Top-K activations from image side (image thumbnail + COCO URL + image_id)
  - Top-K activations from text side (image thumbnail + COCO URL + caption text)
  - Jaccard overlap of image_ids

Text captions are pulled from HuggingFace dataset (namkha1032/coco-karpathy)
if available; otherwise image_id + caption_idx are shown as text.

Output: outputs/aux_alignment_clip_b32/qualitative/<variant>.html

Usage:
    python scripts/qualitative_slots_html.py --variant naive_once \\
        --k-slots 10 --k-top 12
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_alpha"))
import _bootstrap  # noqa: F401

import numpy as np
import torch
from safetensors.torch import load_file

from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from src.models.configuration_sae import TwoSidedTopKSAEConfig
from src.models.modeling_sae import TwoSidedTopKSAE


def load_captions_from_hf(cap_dict_path: Path) -> dict[tuple[str, int], str]:
    """Try to load captions into a {(image_id, caption_idx): caption_text} dict.

    Uses an on-disk cache; falls back to empty dict if HF datasets can't be loaded.
    """
    if cap_dict_path.exists():
        with open(cap_dict_path) as f:
            data = json.load(f)
        return {(k.split("::")[0], int(k.split("::")[1])): v for k, v in data.items()}

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("HF datasets not available; captions will be shown as image_id + caption_idx only")
        return {}

    print("Loading HF dataset namkha1032/coco-karpathy (first time, ~1-5 min)...")
    try:
        ds = load_dataset("namkha1032/coco-karpathy", split="train")
        # Typical columns: image_id, caption, sentids, ... but schema can vary.
        cap_dict: dict[tuple[str, int], str] = {}
        # Group captions by image_id
        from collections import defaultdict
        by_img = defaultdict(list)
        for row in ds:  # type: ignore
            iid = str(row.get("image_id") or row.get("imgid") or row.get("id"))
            cap = row.get("caption") or row.get("sentence") or row.get("text") or ""
            by_img[iid].append(cap)
        for iid, caps in by_img.items():
            for cidx, cap in enumerate(caps):
                cap_dict[(iid, cidx)] = cap
        # Save cache
        dump = {f"{k[0]}::{k[1]}": v for k, v in cap_dict.items()}
        cap_dict_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cap_dict_path, "w") as f:
            json.dump(dump, f)
        print(f"Cached {len(cap_dict)} captions to {cap_dict_path}")
        return cap_dict
    except Exception as e:
        print(f"Failed to load HF dataset: {e}")
        return {}


def coco_image_url(image_id: str) -> str:
    # Karpathy uses COCO 2014 images. Standard URL pattern.
    return f"http://images.cocodataset.org/train2014/COCO_train2014_{int(image_id):012d}.jpg"


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


def compute_fire_rates(model, img, txt, bs=2048, device=None):
    """Pass 1: compute fire rates per slot (no full-tensor storage)."""
    if device is None:
        device = next(model.parameters()).device
    sae_i = model.image_sae; sae_t = model.text_sae
    n = int(sae_i.latent_size); N = img.shape[0]
    fire_i = torch.zeros(n, dtype=torch.float64)
    fire_t = torch.zeros(n, dtype=torch.float64)
    with torch.no_grad():
        for s in range(0, N, bs):
            ib = img[s:s+bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            tb = txt[s:s+bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            zi = sae_i(hidden_states=ib, return_dense_latents=True).dense_latents.squeeze(1)
            zt = sae_t(hidden_states=tb, return_dense_latents=True).dense_latents.squeeze(1)
            fire_i += (zi > 0).sum(dim=0).cpu().to(torch.float64)
            fire_t += (zt > 0).sum(dim=0).cpu().to(torch.float64)
    return (fire_i / N).numpy(), (fire_t / N).numpy()


def collect_activations_for_slots(model, img, txt, slot_ids, bs=2048, device=None):
    """Pass 2: for the given small set of slot ids, collect per-sample activations.

    Returns z_i_sel (N, |slot_ids|) and z_t_sel (N, |slot_ids|) — much smaller
    memory than full z_i (e.g., 30 slots vs 4096).
    """
    if device is None:
        device = next(model.parameters()).device
    sae_i = model.image_sae; sae_t = model.text_sae
    N = img.shape[0]
    slot_ids_t = torch.as_tensor(slot_ids, dtype=torch.long)
    K = len(slot_ids)
    z_i_sel = torch.zeros(N, K, dtype=torch.float32)
    z_t_sel = torch.zeros(N, K, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, N, bs):
            ib = img[s:s+bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            tb = txt[s:s+bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            zi = sae_i(hidden_states=ib, return_dense_latents=True).dense_latents.squeeze(1).cpu()
            zt = sae_t(hidden_states=tb, return_dense_latents=True).dense_latents.squeeze(1).cpu()
            z_i_sel[s:s+bs] = zi[:, slot_ids_t]
            z_t_sel[s:s+bs] = zt[:, slot_ids_t]
    return z_i_sel, z_t_sel


def classify_slots(rates_i, rates_t, tau_alive=1e-3):
    alive = (rates_i > tau_alive) | (rates_t > tau_alive)
    r = rates_i / np.clip(rates_i + rates_t, 1e-12, None)
    shared_idx = np.where(alive & (r >= 0.4) & (r <= 0.6))[0]
    img_dom_idx = np.where(alive & (r > 0.9))[0]
    txt_dom_idx = np.where(alive & (r < 0.1))[0]
    return shared_idx, img_dom_idx, txt_dom_idx, r


def topk_indices(z, slot, k):
    vals, idx = torch.topk(z[:, slot], k=min(k, z.shape[0]))
    return idx.numpy(), vals.numpy()


def _render_activation_cell(idx_arr, val_arr, train_pairs, cap_dict) -> str:
    """HTML for one side's top-K activations: grid of image thumbnails with captions below."""
    parts = []
    for i, v in zip(idx_arr, val_arr):
        image_id, cap_idx = train_pairs[int(i)]
        image_id = str(image_id); cap_idx = int(cap_idx)
        url = coco_image_url(image_id)
        cap_text = cap_dict.get((image_id, cap_idx), f"[img {image_id} cap {cap_idx}]")
        cap_html = escape(cap_text)
        parts.append(
            f"<div class='cell'>"
            f"<a href='{url}' target='_blank'><img src='{url}' loading='lazy'/></a>"
            f"<div class='cap'><b>{image_id}:{cap_idx}</b> (z={v:.2f})<br>{cap_html}</div>"
            f"</div>"
        )
    return "".join(parts)


def build_html(variant: str, sections: list[dict]) -> str:
    style = """
    body { font-family: -apple-system, sans-serif; margin: 20px; }
    h2 { border-bottom: 2px solid #333; padding-bottom: 4px; }
    h3 { background: #eef; padding: 6px; border-radius: 3px; }
    .slot-card { margin: 8px 0; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
    .slot-header { font-weight: bold; font-size: 14px; background: #ffe; padding: 4px; }
    .side { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .side-label { font-weight: bold; color: #555; margin-top: 6px; }
    .cell { width: 160px; font-size: 11px; }
    .cell img { width: 160px; height: 120px; object-fit: cover; border: 1px solid #ddd; }
    .cap { margin-top: 2px; color: #333; line-height: 1.2; }
    .overlap { color: #c22; font-weight: bold; }
    """
    html = [f"<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>slot qualitative — {escape(variant)}</title>",
            f"<style>{style}</style></head><body>",
            f"<h1>Slot qualitative analysis — variant: <code>{escape(variant)}</code></h1>"]
    for sec in sections:
        html.append(f"<h2>{escape(sec['group'])} slots (top {len(sec['slots'])})</h2>")
        for s in sec["slots"]:
            overlap_frac = s["overlap"]
            html.append(
                f"<div class='slot-card'>"
                f"<div class='slot-header'>slot {s['slot']} &nbsp; "
                f"r={s['r']:.2f} &nbsp; fire_i={s['fire_i']:.3f} &nbsp; fire_t={s['fire_t']:.3f} &nbsp; "
                f"<span class='overlap'>Jaccard overlap = {overlap_frac:.3f}</span></div>"
                f"<div class='side-label'>Image-side top activations</div>"
                f"<div class='side'>{s['img_cells']}</div>"
                f"<div class='side-label'>Text-side top activations</div>"
                f"<div class='side'>{s['txt_cells']}</div>"
                f"</div>"
            )
    html.append("</body></html>")
    return "\n".join(html)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="naive_once")
    p.add_argument("--runs-root", default="outputs/aux_alignment_clip_b32")
    p.add_argument("--cache-dir", default="cache/clip_b32_coco")
    p.add_argument("--k-slots", type=int, default=10)
    p.add_argument("--k-top", type=int, default=12)
    args = p.parse_args()

    device = torch.device("cpu")
    runs_root = Path(args.runs_root)
    vd = runs_root / args.variant
    if not (vd / "final/model.safetensors").exists():
        raise SystemExit(f"No model at {vd}/final")

    cap_dict_path = Path("cache/coco_karpathy_captions.json")
    cap_dict = load_captions_from_hf(cap_dict_path)

    ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
    N = len(ds)
    img = torch.stack([ds[i]["image_embeds"] for i in range(N)], dim=0)
    txt = torch.stack([ds[i]["text_embeds"] for i in range(N)], dim=0)
    train_pairs = json.loads(Path(args.cache_dir, "splits.json").read_text())["train"]

    model = load_model(vd, device)
    print("Pass 1: fire rates (streaming, low memory)...")
    rates_i, rates_t = compute_fire_rates(model, img, txt, device=device)
    shared_idx, img_dom_idx, txt_dom_idx, r = classify_slots(rates_i, rates_t)
    print(f"Slot classes: shared={len(shared_idx)} img_dom={len(img_dom_idx)} txt_dom={len(txt_dom_idx)}")

    def pick_top(slots):
        if slots.size == 0:
            return slots
        combined = rates_i[slots] + rates_t[slots]
        return slots[np.argsort(-combined)[:args.k_slots]]

    picked_shared = pick_top(shared_idx)
    picked_img_dom = pick_top(img_dom_idx)
    picked_txt_dom = pick_top(txt_dom_idx)
    interest = np.unique(np.concatenate([picked_shared, picked_img_dom, picked_txt_dom]))
    interest = interest.astype(np.int64)
    print(f"Pass 2: collecting activations for {len(interest)} slots of interest...")
    z_i_sel, z_t_sel = collect_activations_for_slots(model, img, txt, interest, device=device)
    # local index: slot -> column in z_i_sel / z_t_sel
    slot_to_col = {int(s): i for i, s in enumerate(interest)}
    img_ids = np.array([p[0] for p in train_pairs], dtype=object)

    sections = []
    for group_name, slot_arr in [
        ("shared (0.4 ≤ r ≤ 0.6)", picked_shared),
        ("image-dominant (r > 0.9)", picked_img_dom),
        ("text-dominant (r < 0.1)", picked_txt_dom),
    ]:
        group_slots = []
        for sl in slot_arr:
            col = slot_to_col[int(sl)]
            # top-K on selected columns only
            i_vals, i_idx = torch.topk(z_i_sel[:, col], k=min(args.k_top, z_i_sel.shape[0]))
            t_vals, t_idx = torch.topk(z_t_sel[:, col], k=min(args.k_top, z_t_sel.shape[0]))
            i_idx = i_idx.numpy(); t_idx = t_idx.numpy()
            i_val = i_vals.numpy(); t_val = t_vals.numpy()
            img_set = set(img_ids[i_idx].tolist())
            txt_set = set(img_ids[t_idx].tolist())
            overlap = len(img_set & txt_set) / max(len(img_set | txt_set), 1)
            group_slots.append({
                "slot": int(sl), "r": float(r[sl]),
                "fire_i": float(rates_i[sl]), "fire_t": float(rates_t[sl]),
                "overlap": overlap,
                "img_cells": _render_activation_cell(i_idx, i_val, train_pairs, cap_dict),
                "txt_cells": _render_activation_cell(t_idx, t_val, train_pairs, cap_dict),
            })
        sections.append({"group": group_name, "slots": group_slots})

    html = build_html(args.variant, sections)
    out_dir = Path(args.runs_root) / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.variant}.html"
    out_path.write_text(html)
    print(f"Wrote {out_path} (open in browser)")


if __name__ == "__main__":
    main()
