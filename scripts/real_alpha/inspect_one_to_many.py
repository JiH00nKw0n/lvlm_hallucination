"""Qualitative inspection of 1:N co-activation groups (rebuttal E1, part 2).

Consumes the C statistics saved by analyze_C_multiplicity.py, selects the
image-side latents with the most text-side partners at C >= tau, and renders
an HTML report showing, for each group {i -> j1..jn}:

  * the image latent i's top-activating COCO images,
  * each text partner jk's top-activating COCO captions,
  * the top co-activating samples for each (i, jk) pair
    (score = min(z_I[s,i], z_T[s,jk])).

COCO is used as the probe corpus (retrievable images + captions) even though
C was computed on CC3M — the latents are the same dictionaries.

Usage:
    python scripts/real_alpha/inspect_one_to_many.py \
        --ckpt outputs/real_exp_cc3m_s0/separated/ckpt/final \
        --c-stats outputs/rebuttal_E1/s0/C_stats.npz \
        --cache-dir cache/clip_b32_coco --tau 0.3 --top-m 8 \
        --out outputs/rebuttal_E1/s0/one_to_many.html
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import eval_utils  # type: ignore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="separated ckpt final dir")
    p.add_argument("--c-stats", type=str, required=True,
                   help="C_stats.npz from analyze_C_multiplicity.py")
    p.add_argument("--cache-dir", type=str, default="cache/clip_b32_coco")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--top-m", type=int, default=8, help="number of 1:N groups to render")
    p.add_argument("--max-partners", type=int, default=4)
    p.add_argument("--top-n", type=int, default=6, help="samples shown per row")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def _img_b64(pil_img, height=100) -> str:
    from PIL import Image
    w, h = pil_img.size
    pil_img = pil_img.resize((int(w * height / h), height), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class TopN:
    """Running top-N (score, sample_idx) tracker."""

    def __init__(self, n: int):
        self.scores = np.full(n, -np.inf, dtype=np.float32)
        self.ids = np.full(n, -1, dtype=np.int64)

    def update(self, scores: np.ndarray, base_idx: int) -> None:
        for k in np.argsort(scores)[::-1][:len(self.scores)]:
            s = float(scores[k])
            if s <= self.scores.min() or s <= 0:
                break
            worst = self.scores.argmin()
            self.scores[worst] = s
            self.ids[worst] = base_idx + int(k)

    def sorted(self):
        order = np.argsort(-self.scores)
        return [(float(self.scores[o]), int(self.ids[o]))
                for o in order if self.ids[o] >= 0]


def main():
    args = parse_args()
    device = (torch.device(args.device) if args.device != "auto"
              else torch.device("cuda" if torch.cuda.is_available()
                                else ("mps" if torch.backends.mps.is_available() else "cpu")))

    st = np.load(args.c_stats)
    C = st["C"].astype(np.float32)
    alive_i, alive_t = st["alive_image"], st["alive_text"]
    C_alive = C[np.ix_(alive_i, alive_t)]
    rows_g = np.where(alive_i)[0]
    cols_g = np.where(alive_t)[0]

    counts = (C_alive >= args.tau).sum(axis=1)
    order = np.argsort(-counts)
    groups = []
    for r in order[:args.top_m]:
        if counts[r] < 2:
            break
        partners_local = np.argsort(-C_alive[r])[:args.max_partners]
        partners_local = [int(c) for c in partners_local if C_alive[r, c] >= args.tau]
        groups.append({
            "img_latent": int(rows_g[r]),
            "partners": [(int(cols_g[c]), float(C_alive[r, c])) for c in partners_local],
        })
    print(f"selected {len(groups)} groups (tau={args.tau}); "
          f"partner counts: {[len(g['partners']) for g in groups]}")
    if not groups:
        raise SystemExit("no 1:N groups at this tau — lower --tau")

    model = eval_utils.load_sae(args.ckpt, "separated")
    model.to(device).eval()
    ds = eval_utils.load_pair_dataset(args.cache_dir, "coco", args.split)
    N = len(ds)
    print(f"probe corpus: {N} pairs from {args.cache_dir} ({args.split})")

    img_latents = sorted({g["img_latent"] for g in groups})
    txt_latents = sorted({j for g in groups for j, _ in g["partners"]})
    img_top = {i: TopN(args.top_n) for i in img_latents}
    txt_top = {j: TopN(args.top_n) for j in txt_latents}
    pair_top = {(g["img_latent"], j): TopN(args.top_n)
                for g in groups for j, _ in g["partners"]}

    with torch.no_grad():
        for s in range(0, N, args.batch_size):
            e = min(s + args.batch_size, N)
            batch = [ds[k] for k in range(s, e)]
            x = torch.stack([b["image_embeds"] for b in batch]).to(device).unsqueeze(1)
            y = torch.stack([b["text_embeds"] for b in batch]).to(device).unsqueeze(1)
            zi = model.image_sae(hidden_states=x, return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
            zt = model.text_sae(hidden_states=y, return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
            for i in img_latents:
                img_top[i].update(zi[:, i], s)
            for j in txt_latents:
                txt_top[j].update(zt[:, j], s)
            for (i, j), tracker in pair_top.items():
                tracker.update(np.minimum(zi[:, i], zt[:, j]), s)
            if (s // args.batch_size) % 50 == 0:
                print(f"  batch {s // args.batch_size}/{(N + args.batch_size - 1) // args.batch_size}")

    # --- COCO images for rendering ---
    from datasets import load_dataset
    hf_ds = load_dataset("namkha1032/coco-karpathy", split="train")
    id_to_row = {}
    for ridx in range(len(hf_ds)):
        iid = str(hf_ds[ridx]["image_id"])
        id_to_row.setdefault(iid, ridx)

    pairs = ds.pairs

    def render_samples(entries, with_image: bool, with_caption: bool) -> str:
        parts = []
        for score, sid in entries:
            iid, ci = pairs[sid]
            ridx = id_to_row.get(str(iid))
            if ridx is None:
                continue
            row = hf_ds[ridx]
            cap = row["captions"][int(ci)] if int(ci) < len(row["captions"]) else ""
            cell = "<div class='sample'>"
            if with_image:
                cell += f"<img src='data:image/jpeg;base64,{_img_b64(row['image'].convert('RGB'))}'><br>"
            cell += f"<div class='caption'>score={score:.3f}"
            if with_caption:
                cell += f"<br>{cap}"
            cell += "</div></div>"
            parts.append(cell)
        return "".join(parts)

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>",
        "body{font-family:'Helvetica Neue',sans-serif;margin:20px}",
        ".group{border:1px solid #ccc;border-radius:8px;padding:14px;margin-bottom:28px}",
        ".rowlbl{font-weight:600;margin:10px 0 4px;color:#333}",
        ".sample{display:inline-block;margin:4px;text-align:center;vertical-align:top;max-width:190px}",
        ".sample img{height:100px;border-radius:3px}",
        ".caption{font-size:11px;color:#666;max-width:180px;word-wrap:break-word}",
        ".meta{color:#888;font-size:12px}",
        "</style></head><body>",
        f"<h1>1:N co-activation groups (tau={args.tau}, C from {Path(args.c_stats).parent.name})</h1>",
    ]
    for g in groups:
        i = g["img_latent"]
        plist = ", ".join(f"txt#{j} (C={c:.2f})" for j, c in g["partners"])
        html.append(f"<div class='group'><h2>Image latent #{i} → {len(g['partners'])} partners</h2>"
                    f"<div class='meta'>{plist}</div>")
        html.append("<div class='rowlbl'>Top images for img latent "
                    f"#{i}</div>" + render_samples(img_top[i].sorted(), True, False))
        for j, c in g["partners"]:
            html.append(f"<div class='rowlbl'>Top captions for txt latent #{j} (C={c:.2f})</div>"
                        + render_samples(txt_top[j].sorted(), False, True))
            html.append(f"<div class='rowlbl'>Top co-activating samples (img#{i}, txt#{j})</div>"
                        + render_samples(pair_top[(i, j)].sorted(), True, True))
        html.append("</div>")
    html.append("</body></html>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(html), encoding="utf-8")
    meta = {"tau": args.tau,
            "groups": [{"img_latent": g["img_latent"], "partners": g["partners"]} for g in groups]}
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
