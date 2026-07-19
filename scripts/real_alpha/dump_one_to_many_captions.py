"""Dump caption evidence for 1:N co-activation groups (rebuttal, caption-only).

Lighter sibling of inspect_one_to_many.py: no images, captions only, so it
runs anywhere the COCO embedding cache exists (no 25GB parquet download).

Groups = image-side latents with >=2 text-side partners at C >= tau, ranked
by the SECOND-strongest partner's correlation (surfaces the most genuinely
one-to-many cases first). For each group and each text partner j:
  * top captions by text-latent activation z_T[:, j]
  * top captions by pairwise co-activation min(z_I[:, i], z_T[:, j])
plus the captions paired with the image latent's top-activating images.

Usage:
    python scripts/real_alpha/dump_one_to_many_captions.py \
        --ckpt outputs/real_exp_cc3m_s0/separated/ckpt/final \
        --c-stats outputs/rebuttal_E1/s0/C_stats.npz \
        --cache-dir cache/clip_b32_coco --tau 0.4 --top-m 20 \
        --out outputs/rebuttal_E1/s0/one_to_many_captions_tau04.md
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--c-stats", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="cache/clip_b32_coco")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--captions", type=str, default="cache/coco_karpathy_captions.json")
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--top-m", type=int, default=20)
    p.add_argument("--max-partners", type=int, default=4)
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = (torch.device(args.device) if args.device != "auto"
              else torch.device("cuda" if torch.cuda.is_available()
                                else ("mps" if torch.backends.mps.is_available() else "cpu")))

    st = np.load(args.c_stats)
    C = st["C"].astype(np.float32)
    alive_i, alive_t = st["alive_image"], st["alive_text"]
    rows_g, cols_g = np.where(alive_i)[0], np.where(alive_t)[0]
    C_alive = C[np.ix_(alive_i, alive_t)]

    # rank groups by the 2nd-strongest partner correlation
    part_sorted = np.sort(C_alive, axis=1)
    second_best = part_sorted[:, -2]
    n_partners = (C_alive >= args.tau).sum(axis=1)
    cand = np.where(n_partners >= 2)[0]
    cand = cand[np.argsort(-second_best[cand])][:args.top_m]
    groups = []
    for r in cand:
        order = np.argsort(-C_alive[r])
        partners = [(int(cols_g[c]), float(C_alive[r, c]))
                    for c in order[:args.max_partners] if C_alive[r, c] >= args.tau]
        groups.append({"img_latent": int(rows_g[r]), "partners": partners,
                       "second_best_C": float(second_best[r])})
    print(f"{len(groups)} groups at tau={args.tau} "
          f"(2nd-partner C range {groups[-1]['second_best_C']:.2f}–{groups[0]['second_best_C']:.2f})")

    model = eval_utils.load_sae(args.ckpt, "separated")
    model.to(device).eval()
    ds = eval_utils.load_pair_dataset(args.cache_dir, "coco", split=args.split)
    caps = json.load(open(args.captions))

    def cap_of(sid: int) -> str:
        iid, ci = ds.pairs[sid]
        for k in (f"{int(iid)}::{int(ci)}", f"{int(iid)}_{int(ci)}"):
            if k in caps:
                return caps[k]
        return "(caption missing)"

    img_latents = sorted({g["img_latent"] for g in groups})
    txt_latents = sorted({j for g in groups for j, _ in g["partners"]})
    li = {v: k for k, v in enumerate(img_latents)}
    lt = {v: k for k, v in enumerate(txt_latents)}
    N = len(ds)
    zi_sel = torch.zeros(N, len(img_latents))
    zt_sel = torch.zeros(N, len(txt_latents))

    with torch.no_grad():
        for s in range(0, N, args.batch_size):
            e = min(s + args.batch_size, N)
            batch = [ds[k] for k in range(s, e)]
            x = torch.stack([b["image_embeds"] for b in batch]).to(device).unsqueeze(1)
            y = torch.stack([b["text_embeds"] for b in batch]).to(device).unsqueeze(1)
            zi = model.image_sae(hidden_states=x, return_dense_latents=True).dense_latents.squeeze(1)
            zt = model.text_sae(hidden_states=y, return_dense_latents=True).dense_latents.squeeze(1)
            zi_sel[s:e] = zi[:, img_latents].cpu()
            zt_sel[s:e] = zt[:, txt_latents].cpu()
            if (s // args.batch_size) % 40 == 0:
                print(f"  batch {s // args.batch_size}/{(N + args.batch_size - 1) // args.batch_size}")

    lines = [f"# 1:N co-activation groups — caption evidence (tau={args.tau}, "
             f"ranked by 2nd-partner C)\n"]
    for gi, g in enumerate(groups, 1):
        i = g["img_latent"]
        lines.append(f"\n## Group {gi}: image latent #{i} — "
                     + ", ".join(f"txt#{j} (C={c:.2f})" for j, c in g["partners"]))
        top_img = torch.topk(zi_sel[:, li[i]], args.top_n)
        lines.append(f"\n**img#{i} top-activating images' paired captions:**")
        for v, sid in zip(top_img.values.tolist(), top_img.indices.tolist()):
            lines.append(f"- ({v:.2f}) {cap_of(sid)}")
        for j, c in g["partners"]:
            top_txt = torch.topk(zt_sel[:, lt[j]], args.top_n)
            lines.append(f"\n**txt#{j} (C={c:.2f}) top captions:**")
            for v, sid in zip(top_txt.values.tolist(), top_txt.indices.tolist()):
                lines.append(f"- ({v:.2f}) {cap_of(sid)}")
            co = torch.minimum(zi_sel[:, li[i]], zt_sel[:, lt[j]])
            top_co = torch.topk(co, args.top_n)
            lines.append(f"\n**co-activation top (img#{i} ∧ txt#{j}):**")
            for v, sid in zip(top_co.values.tolist(), top_co.indices.tolist()):
                if v <= 0:
                    break
                lines.append(f"- ({v:.2f}) {cap_of(sid)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
