"""Inspect top-activating COCO samples for each Hungarian-alive matched pair.

For each matched (image_latent, text_latent) pair binned by co-activation C,
finds the top-N training samples that most strongly co-fire both latents,
then renders an HTML report with the actual COCO images and captions.

Usage:
    python scripts/real_alpha/inspect_top_activations.py \
        --run-dir outputs/real_alpha_followup_1/two_sae/final \
        --cache-dir cache/clip_b32_coco \
        --out outputs/real_alpha_followup_1/top_activations.html \
        --top-n 10 --bin-width 0.2 --c-min 0.2
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore
from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--bin-width", type=float, default=0.2)
    p.add_argument("--c-min", type=float, default=0.2)
    p.add_argument("--c-max", type=float, default=1.0)
    p.add_argument("--alive-thr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--single", action="store_true",
                   help="single SAE mode: same TopKSAE for both modalities")
    return p.parse_args()


def img_to_base64(pil_img: Image.Image, height: int = 100) -> str:
    w, h = pil_img.size
    new_w = int(w * height / h)
    pil_img = pil_img.resize((new_w, height), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # --- Load model + embeddings ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
    N = len(ds)
    img_embs = torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
    txt_embs = torch.stack([ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs])
    print(f"loaded {N} pairs")

    if args.single:
        # Single SAE: same model for both modalities, compute C on the fly
        from src.models.modeling_sae import TopKSAE as _TopK
        sae = _TopK.from_pretrained(str(run_dir)).to(device).eval()
        image_sae = sae
        text_sae = sae
        _L = sae.latent_size
        _si=np.zeros(_L,dtype=np.float64);_st=np.zeros(_L,dtype=np.float64)
        _sii=np.zeros(_L,dtype=np.float64);_stt=np.zeros(_L,dtype=np.float64)
        _sit=np.zeros((_L,_L),dtype=np.float64)
        _fi=np.zeros(_L,dtype=np.int64);_ft=np.zeros(_L,dtype=np.int64);_cnt=0
        print("computing C for single SAE...")
        with torch.no_grad():
            for _s in range(0,N,2048):
                _hi=img_embs[_s:_s+2048].unsqueeze(1).to(device)
                _ht=txt_embs[_s:_s+2048].unsqueeze(1).to(device)
                _oi=sae(hidden_states=_hi,return_dense_latents=True)
                _ot=sae(hidden_states=_ht,return_dense_latents=True)
                _zi=_oi.dense_latents.squeeze(1).cpu().double().numpy()
                _zt=_ot.dense_latents.squeeze(1).cpu().double().numpy()
                _B=_zi.shape[0]
                _si+=_zi.sum(0);_st+=_zt.sum(0)
                _sii+=(_zi*_zi).sum(0);_stt+=(_zt*_zt).sum(0)
                _sit+=_zi.T@_zt;_fi+=(_zi>0).sum(0);_ft+=(_zt>0).sum(0);_cnt+=_B
                if (_s//2048)%50==0: print(f"  batch {_s//2048}/{(N+2047)//2048}")
        _mi=_si/_cnt;_mt=_st/_cnt
        _cov=_sit/_cnt-np.outer(_mi,_mt)
        _den=np.sqrt(np.clip((_sii/_cnt-_mi**2)[:,None]*(_stt/_cnt-_mt**2)[None,:],1e-16,None))
        C=np.nan_to_num(_cov/_den,nan=0.0)
        W=sae.W_dec.detach().cpu().float().numpy()
        Wn=W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-12)
        cos_ij=Wn@Wn.T
        alive_i=np.where(_fi/_cnt>args.alive_thr)[0]
        alive_t=np.where(_ft/_cnt>args.alive_thr)[0]
    else:
        C = np.load(run_dir / "diagnostic_B_C_train.npy")
        sd = load_file(run_dir / "model.safetensors")
        W_i = sd["image_sae.W_dec"].float().numpy()
        W_t = sd["text_sae.W_dec"].float().numpy()
        W_i_norm = W_i / (np.linalg.norm(W_i, axis=1, keepdims=True) + 1e-12)
        W_t_norm = W_t / (np.linalg.norm(W_t, axis=1, keepdims=True) + 1e-12)
        cos_ij = W_i_norm @ W_t_norm.T
        rates = np.load(run_dir / "diagnostic_B_firing_rates.npz")
        alive_i = np.where(rates["rate_i"] > args.alive_thr)[0]
        alive_t = np.where(rates["rate_t"] > args.alive_thr)[0]
        model = TwoSidedTopKSAE.from_pretrained(str(run_dir)).to(device).eval()
        image_sae = model.image_sae
        text_sae = model.text_sae

    print(f"alive image={len(alive_i)}, text={len(alive_t)}")
    C_sub = C[np.ix_(alive_i, alive_t)]
    cos_sub = cos_ij[np.ix_(alive_i, alive_t)]
    row_ind, col_ind = linear_sum_assignment(-C_sub)
    C_matched = C_sub[row_ind, col_ind]
    cos_matched = cos_sub[row_ind, col_ind]
    orig_i = alive_i[row_ind]
    orig_t = alive_t[col_ind]
    print(f"n matched: {len(C_matched)}")

    # --- Filter to bins ---
    edges = np.arange(args.c_min, args.c_max + 1e-9, args.bin_width)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (C_matched >= lo) & (C_matched < hi)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        bins.append({
            "label": f"[{lo:.1f}, {hi:.1f})",
            "pair_indices": idxs,
            "n": len(idxs),
        })
    total_pairs = sum(b["n"] for b in bins)
    print(f"bins: {[(b['label'], b['n']) for b in bins]}, total pairs: {total_pairs}")

    # Collect all pair latent indices we care about
    all_pair_idxs = np.concatenate([b["pair_indices"] for b in bins])
    pair_img_latents = orig_i[all_pair_idxs]  # original latent indices
    pair_txt_latents = orig_t[all_pair_idxs]

    # --- Stream through data, collect top-N per pair ---
    top_n = args.top_n
    n_pairs = len(all_pair_idxs)
    # For each pair: maintain (score, sample_idx) heap
    # Score = min(z_I[sample, latent_i], z_T[sample, latent_j])
    top_scores = np.full((n_pairs, top_n), -np.inf, dtype=np.float32)
    top_sample_ids = np.full((n_pairs, top_n), -1, dtype=np.int64)

    BS = args.batch_size
    print(f"streaming {N} samples in batches of {BS} for {n_pairs} pairs...")
    with torch.no_grad():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            hs_i = img_embs[s:e].unsqueeze(1).to(device)
            hs_t = txt_embs[s:e].unsqueeze(1).to(device)
            out_i = image_sae(hidden_states=hs_i, return_dense_latents=True)
            out_t = text_sae(hidden_states=hs_t, return_dense_latents=True)
            zi = out_i.dense_latents.squeeze(1).cpu().numpy()  # (B, L_i)
            zt = out_t.dense_latents.squeeze(1).cpu().numpy()  # (B, L_t)

            for pi in range(n_pairs):
                li = int(pair_img_latents[pi])
                lt = int(pair_txt_latents[pi])
                act_i = zi[:, li]
                act_t = zt[:, lt]
                score = np.minimum(act_i, act_t)
                for bi in range(len(score)):
                    if score[bi] > top_scores[pi].min():
                        worst = top_scores[pi].argmin()
                        top_scores[pi, worst] = score[bi]
                        top_sample_ids[pi, worst] = s + bi

            if (s // BS) % 50 == 0:
                print(f"  batch {s // BS}/{(N + BS - 1) // BS}")

    # Sort each pair's top-N by descending score
    for pi in range(n_pairs):
        order = np.argsort(-top_scores[pi])
        top_scores[pi] = top_scores[pi][order]
        top_sample_ids[pi] = top_sample_ids[pi][order]

    print("done streaming. loading COCO images...")

    # --- Load COCO images ---
    from datasets import load_dataset
    splits_json = json.load(open(Path(args.cache_dir) / "splits.json"))
    train_pairs = splits_json["train"]  # list of [image_id, caption_idx]

    hf_ds = load_dataset("namkha1032/coco-karpathy", split="train")
    # Build image_id -> row index mapping
    id_to_row = {}
    for row_idx in range(len(hf_ds)):
        iid = str(hf_ds[row_idx]["image_id"])
        if iid not in id_to_row:
            id_to_row[iid] = row_idx
    print(f"built id_to_row: {len(id_to_row)} entries")

    # --- Generate HTML ---
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: 'Helvetica Neue', sans-serif; margin: 20px; }",
        "h2 { border-bottom: 2px solid #333; padding-bottom: 4px; }",
        "h3 { color: #555; margin-top: 24px; }",
        ".pair { margin-bottom: 30px; border: 1px solid #ddd; padding: 12px; border-radius: 6px; }",
        ".sample { display: inline-block; margin: 4px; text-align: center; vertical-align: top; max-width: 200px; }",
        ".sample img { height: 100px; border-radius: 3px; }",
        ".caption { font-size: 11px; color: #666; margin-top: 2px; max-width: 180px; word-wrap: break-word; }",
        ".meta { font-size: 12px; color: #888; }",
        "</style></head><body>",
        "<h1>Top-activating samples per matched latent pair (L=8192, Hungarian-alive)</h1>",
    ]

    pair_offset = 0
    for b in bins:
        html_parts.append(f"<h2>C bin {b['label']} — {b['n']} pairs</h2>")
        for local_idx, global_pair_idx in enumerate(b["pair_indices"]):
            pi = pair_offset + local_idx
            li = int(pair_img_latents[pi])
            lt = int(pair_txt_latents[pi])
            c_val = float(C_matched[global_pair_idx])
            cos_val = float(cos_matched[global_pair_idx])

            html_parts.append(f"<div class='pair'>")
            html_parts.append(
                f"<h3>Image: #{li}, Text: #{lt} "
                f"<span class='meta'>(C={c_val:.3f}, cos={cos_val:.3f})</span></h3>"
            )

            sample_ids = top_sample_ids[pi]
            scores = top_scores[pi]
            for rank in range(top_n):
                sid = int(sample_ids[rank])
                if sid < 0:
                    continue
                sc = float(scores[rank])
                image_id_str, cap_idx = train_pairs[sid]
                cap_idx = int(cap_idx)
                row_idx = id_to_row.get(str(image_id_str))
                if row_idx is None:
                    html_parts.append(f"<div class='sample'><p>img_id={image_id_str} not found</p></div>")
                    continue
                row = hf_ds[row_idx]
                pil_img = row["image"].convert("RGB")
                caption = row["captions"][cap_idx] if cap_idx < len(row["captions"]) else "(no caption)"
                b64 = img_to_base64(pil_img, height=100)
                html_parts.append(
                    f"<div class='sample'>"
                    f"<img src='data:image/jpeg;base64,{b64}'><br>"
                    f"<div class='caption'>#{rank+1} (score={sc:.3f})<br>{caption}</div>"
                    f"</div>"
                )
            html_parts.append("</div>")
        pair_offset += b["n"]

    html_parts.append("</body></html>")
    html = "\n".join(html_parts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"saved {out_path} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
