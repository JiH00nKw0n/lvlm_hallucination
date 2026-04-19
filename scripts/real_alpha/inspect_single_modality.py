"""Compare co-fire vs single-modality activation for Hungarian-matched pairs.

For each matched (image_latent, text_latent) pair with C >= c_min,
collects three sets of top-N samples:
  1. co-fire:    min(z_I[k], z_T[k]) is high  (both activate)
  2. image-only: z_I[k] is high AND z_T[k] < thr  (only image fires)
  3. text-only:  z_T[k] is high AND z_I[k] < thr  (only text fires)

Renders an HTML report with 3 x 9-image grids per pair.

Usage:
    python scripts/real_alpha/inspect_single_modality.py \
        --run-dir outputs/real_alpha_followup_1/two_sae/final \
        --cache-dir cache/clip_b32_coco \
        --out outputs/real_alpha_followup_1/single_modality.html
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

from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from src.models.modeling_sae import TwoSidedTopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--top-n", type=int, default=9)
    p.add_argument("--c-min", type=float, default=0.4,
                   help="minimum C to include a pair")
    p.add_argument("--fire-thr", type=float, default=0.01,
                   help="threshold below which a latent is considered inactive")
    p.add_argument("--alive-thr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-pairs", type=int, default=30,
                   help="cap on number of pairs to render")
    return p.parse_args()


def img_to_b64(pil_img: Image.Image, height: int = 100) -> str:
    w, h = pil_img.size
    pil_img = pil_img.resize((int(w * height / h), height), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load embeddings
    ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
    N = len(ds)
    img_embs = torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
    txt_embs = torch.stack([ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs])
    print(f"loaded {N} pairs")

    # Load two-SAE + cached diagnostics
    C = np.load(run_dir / "diagnostic_B_C_train.npy")
    rates = np.load(run_dir / "diagnostic_B_firing_rates.npz")
    alive_i = np.where(rates["rate_i"] > args.alive_thr)[0]
    alive_t = np.where(rates["rate_t"] > args.alive_thr)[0]
    model = TwoSidedTopKSAE.from_pretrained(str(run_dir)).to(device).eval()
    image_sae = model.image_sae
    text_sae = model.text_sae

    # Hungarian matching
    C_sub = C[np.ix_(alive_i, alive_t)]
    row_ind, col_ind = linear_sum_assignment(-C_sub)
    C_matched = C_sub[row_ind, col_ind]
    orig_i = alive_i[row_ind]
    orig_t = alive_t[col_ind]

    # Filter to high-C pairs
    mask = C_matched >= args.c_min
    sel = np.where(mask)[0]
    # Sort by C descending, cap at max_pairs
    sel = sel[np.argsort(-C_matched[sel])][:args.max_pairs]
    print(f"selected {len(sel)} pairs with C >= {args.c_min}")

    pair_img = orig_i[sel]
    pair_txt = orig_t[sel]
    pair_C = C_matched[sel]
    n_pairs = len(sel)
    top_n = args.top_n
    thr = args.fire_thr

    # Heaps: [n_pairs, top_n] for 3 categories
    co_scores = np.full((n_pairs, top_n), -np.inf, dtype=np.float32)
    co_ids = np.full((n_pairs, top_n), -1, dtype=np.int64)
    img_scores = np.full((n_pairs, top_n), -np.inf, dtype=np.float32)
    img_ids = np.full((n_pairs, top_n), -1, dtype=np.int64)
    txt_scores = np.full((n_pairs, top_n), -np.inf, dtype=np.float32)
    txt_ids = np.full((n_pairs, top_n), -1, dtype=np.int64)

    BS = args.batch_size
    print(f"streaming {N} samples for {n_pairs} pairs...")
    with torch.no_grad():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            zi = image_sae(
                hidden_states=img_embs[s:e].unsqueeze(1).to(device),
                return_dense_latents=True
            ).dense_latents.squeeze(1).cpu().numpy()
            zt = text_sae(
                hidden_states=txt_embs[s:e].unsqueeze(1).to(device),
                return_dense_latents=True
            ).dense_latents.squeeze(1).cpu().numpy()

            for pi in range(n_pairs):
                ai = zi[:, int(pair_img[pi])]
                at = zt[:, int(pair_txt[pi])]

                # co-fire: min(ai, at)
                co = np.minimum(ai, at)
                # image-only: ai where at < thr
                io_mask = at < thr
                io_sc = np.where(io_mask, ai, -np.inf)
                # text-only: at where ai < thr
                to_mask = ai < thr
                to_sc = np.where(to_mask, at, -np.inf)

                for bi in range(len(co)):
                    idx = s + bi
                    if co[bi] > co_scores[pi].min():
                        w = co_scores[pi].argmin()
                        co_scores[pi, w] = co[bi]
                        co_ids[pi, w] = idx
                    if io_sc[bi] > img_scores[pi].min():
                        w = img_scores[pi].argmin()
                        img_scores[pi, w] = io_sc[bi]
                        img_ids[pi, w] = idx
                    if to_sc[bi] > txt_scores[pi].min():
                        w = txt_scores[pi].argmin()
                        txt_scores[pi, w] = to_sc[bi]
                        txt_ids[pi, w] = idx

            if (s // BS) % 50 == 0:
                print(f"  batch {s // BS}/{(N + BS - 1) // BS}")

    # Sort descending
    for pi in range(n_pairs):
        for scores, ids in [(co_scores, co_ids), (img_scores, img_ids), (txt_scores, txt_ids)]:
            order = np.argsort(-scores[pi])
            scores[pi] = scores[pi][order]
            ids[pi] = ids[pi][order]

    print("loading COCO images...")
    from datasets import load_dataset
    splits = json.load(open(Path(args.cache_dir) / "splits.json"))
    train_pairs = splits["train"]
    hf_ds = load_dataset("namkha1032/coco-karpathy", split="train")
    id_to_row = {}
    for ri in range(len(hf_ds)):
        iid = str(hf_ds[ri]["image_id"])
        if iid not in id_to_row:
            id_to_row[iid] = ri

    # --- HTML ---
    def render_grid(sample_ids, scores_arr, label):
        parts = [f"<div class='section'><h4>{label}</h4><div class='grid'>"]
        seen_imgs = set()
        rendered = 0
        for rank in range(len(sample_ids)):
            sid = int(sample_ids[rank])
            if sid < 0:
                continue
            sc = float(scores_arr[rank])
            img_id_str, cap_idx = train_pairs[sid]
            cap_idx = int(cap_idx)
            if str(img_id_str) in seen_imgs:
                continue
            seen_imgs.add(str(img_id_str))
            row_idx = id_to_row.get(str(img_id_str))
            if row_idx is None:
                continue
            row = hf_ds[row_idx]
            pil = row["image"].convert("RGB")
            cap = row["captions"][cap_idx] if cap_idx < len(row["captions"]) else ""
            b64 = img_to_b64(pil)
            parts.append(
                f"<div class='sample'>"
                f"<img src='data:image/jpeg;base64,{b64}'>"
                f"<div class='caption'>#{rendered+1} ({sc:.2f})<br>{cap[:80]}</div>"
                f"</div>"
            )
            rendered += 1
            if rendered >= 9:
                break
        parts.append("</div></div>")
        return "\n".join(parts)

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:'Helvetica Neue',sans-serif;margin:20px}",
        "h2{border-bottom:2px solid #333;padding-bottom:4px}",
        ".pair{margin-bottom:40px;border:1px solid #ddd;padding:16px;border-radius:8px}",
        ".section{margin:12px 0}",
        ".section h4{margin:4px 0;font-size:14px}",
        ".grid{display:flex;flex-wrap:wrap;gap:6px}",
        ".sample{text-align:center;max-width:160px}",
        ".sample img{height:90px;border-radius:3px}",
        ".caption{font-size:10px;color:#666;margin-top:2px;word-wrap:break-word}",
        ".meta{font-size:12px;color:#888}",
        ".co{border-left:4px solid #43aa8b}",
        ".img-only{border-left:4px solid #f94144}",
        ".txt-only{border-left:4px solid #277da1}",
        "</style></head><body>",
        f"<h1>Co-fire vs Single-modality Activation ({n_pairs} pairs, C>={args.c_min})</h1>",
        f"<p>fire_thr={thr}, top-9 per category</p>",
    ]

    for pi in range(n_pairs):
        li, lt = int(pair_img[pi]), int(pair_txt[pi])
        c_val = float(pair_C[pi])
        n_co = int((co_scores[pi] > -np.inf).sum())
        n_io = int((img_scores[pi] > -np.inf).sum())
        n_to = int((txt_scores[pi] > -np.inf).sum())

        html.append(f"<div class='pair'>")
        html.append(
            f"<h3>Pair #{pi+1}: Img latent #{li}, Txt latent #{lt} "
            f"<span class='meta'>(C={c_val:.3f})</span></h3>"
        )
        html.append(f"<p class='meta'>co-fire: {n_co}, img-only: {n_io}, txt-only: {n_to}</p>")
        html.append(render_grid(co_ids[pi], co_scores[pi], "Co-fire (both activate)"))
        html.append(render_grid(img_ids[pi], img_scores[pi], "Image-only (text silent)"))
        html.append(render_grid(txt_ids[pi], txt_scores[pi], "Text-only (image silent)"))
        html.append("</div>")

    html.append("</body></html>")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"saved {out_path} ({len(''.join(html)) // 1024} KB)")


if __name__ == "__main__":
    main()
