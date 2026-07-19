"""Sample-set overlap analysis for 1:N co-activation groups (rebuttal E1).

For each image-side latent i with >=2 text partners at C >= tau, build the
co-firing sample sets  A_k = { s : z_I[s,i] > 0  and  z_T[s,j_k] > 0 }  and
quantify whether partners fire on the SAME samples (redundant duplicates —
a 1:1 assignment loses nothing) or on DISJOINT samples (a genuine split —
picking one partner drops coverage of the others' samples):

  * |S_i| (image-latent firing count), |A_k| per partner
  * pairwise Jaccard |A_k ∩ A_l| / |A_k ∪ A_l| and containment
    |A_k ∩ A_l| / min(|A_k|,|A_l|)
  * union coverage |∪A_k| / |S_i| and top-partner share |A_top| / |∪A_k|
    (how much of the explained samples the Hungarian-preferred single
    partner already covers)

All statistics are accumulated online (no (N, L) mask storage), so the full
2.8M-pair CC3M corpus fits in RAM. With --captions, example captions are
dumped for shared vs exclusive samples of the top-2 partners (COCO probe).

Usage (stats on the corpus C was measured on):
    python scripts/real_alpha/analyze_partner_overlap.py \
        --ckpt outputs/real_exp_cc3m_s0/separated/ckpt/final \
        --c-stats outputs/rebuttal_E1/s0/C_stats.npz \
        --cache-dir cache/clip_b32_cc3m --dataset cc3m --tau 0.4 \
        --out outputs/rebuttal_E1/s0/partner_overlap_cc3m.json

Caption evidence (COCO probe, top groups only):
    ... --cache-dir cache/clip_b32_coco --dataset coco \
        --captions cache/coco_karpathy_captions.json --top-m 20 \
        --out outputs/rebuttal_E1/s0/partner_overlap_coco.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
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
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--dataset", choices=["cc3m", "coco", "laion"], default="cc3m")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--relative", action="store_true",
                   help="select groups by RELATIVE competition instead of the "
                        "absolute tau cut: partners are cols with C >= rel-ratio "
                        "* rowmax, groups need rowmax >= min-top1 and >=2 partners")
    p.add_argument("--rel-ratio", type=float, default=0.8)
    p.add_argument("--min-top1", type=float, default=0.3)
    p.add_argument("--top-m", type=int, default=0,
                   help="0 = ALL 1:N groups; else only the top-M by 2nd-partner C")
    p.add_argument("--max-partners", type=int, default=4)
    p.add_argument("--captions", type=str, default=None,
                   help="captions json → dump shared/exclusive caption examples")
    p.add_argument("--n-caption-examples", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8192)
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

    rowmax = C_alive.max(axis=1)
    if args.relative:
        thr = args.rel_ratio * rowmax[:, None]          # per-row cut
        n_partners = (C_alive >= thr).sum(axis=1)
        cand = np.where((n_partners >= 2) & (rowmax >= args.min_top1))[0]
    else:
        thr = np.full((C_alive.shape[0], 1), args.tau)  # absolute cut
        n_partners = (C_alive >= args.tau).sum(axis=1)
        cand = np.where(n_partners >= 2)[0]
    second_best = np.sort(C_alive, axis=1)[:, -2]
    cand = cand[np.argsort(-second_best[cand])]
    if args.top_m:
        cand = cand[:args.top_m]
    groups = []
    for r in cand:
        order = np.argsort(-C_alive[r])
        partners = [(int(cols_g[c]), float(C_alive[r, c]))
                    for c in order[:args.max_partners] if C_alive[r, c] >= thr[r, 0]]
        groups.append({"img_latent": int(rows_g[r]), "partners": partners})
    mode = (f"relative rel_ratio={args.rel_ratio} min_top1={args.min_top1}"
            if args.relative else f"absolute tau={args.tau}")
    print(f"{len(groups)} groups ({mode})")

    model = eval_utils.load_sae(args.ckpt, "separated")
    model.to(device).eval()
    ds = eval_utils.load_pair_dataset(args.cache_dir, args.dataset, split=args.split)
    N = len(ds)
    print(f"corpus: {N} pairs ({args.dataset})")

    img_latents = sorted({g["img_latent"] for g in groups})
    txt_latents = sorted({j for g in groups for j, _ in g["partners"]})
    li = {v: k for k, v in enumerate(img_latents)}
    lt = {v: k for k, v in enumerate(txt_latents)}

    # online counters per group
    for g in groups:
        K = len(g["partners"])
        g["n_img_fires"] = 0
        g["n_partner"] = [0] * K
        g["n_pair_inter"] = {f"{a}-{b}": 0 for a, b in combinations(range(K), 2)}
        g["n_union"] = 0
    want_caps = args.captions is not None
    if want_caps:
        caps = json.load(open(args.captions))
        for g in groups:
            g["ex_shared"], g["ex_only1"], g["ex_only2"] = [], [], []

    def cap_of(sid: int) -> str:
        iid, ci = ds.pairs[sid]
        for k in (f"{int(iid)}::{int(ci)}", f"{int(iid)}_{int(ci)}"):
            if k in caps:
                return caps[k]
        return "(missing)"

    use_batch_gather = hasattr(ds, "_image_table")
    with torch.no_grad():
        for s0 in range(0, N, args.batch_size):
            e = min(s0 + args.batch_size, N)
            batch = [ds[k] for k in range(s0, e)]
            x = torch.stack([b["image_embeds"] for b in batch]).to(device).unsqueeze(1)
            y = torch.stack([b["text_embeds"] for b in batch]).to(device).unsqueeze(1)
            zi = model.image_sae(hidden_states=x, return_dense_latents=True).dense_latents.squeeze(1)
            zt = model.text_sae(hidden_states=y, return_dense_latents=True).dense_latents.squeeze(1)
            fi = (zi[:, img_latents] > 0).cpu().numpy()   # (B, n_img)
            ft = (zt[:, txt_latents] > 0).cpu().numpy()   # (B, n_txt)
            for g in groups:
                m_i = fi[:, li[g["img_latent"]]]
                if not m_i.any():
                    continue
                A = [m_i & ft[:, lt[j]] for j, _ in g["partners"]]
                g["n_img_fires"] += int(m_i.sum())
                union = np.zeros_like(m_i)
                for k, a in enumerate(A):
                    g["n_partner"][k] += int(a.sum())
                    union |= a
                g["n_union"] += int(union.sum())
                for a, b in combinations(range(len(A)), 2):
                    g["n_pair_inter"][f"{a}-{b}"] += int((A[a] & A[b]).sum())
                if want_caps and len(A) >= 2:
                    sh = np.where(A[0] & A[1])[0]
                    o1 = np.where(A[0] & ~A[1])[0]
                    o2 = np.where(A[1] & ~A[0])[0]
                    for buf, idxs in (("ex_shared", sh), ("ex_only1", o1), ("ex_only2", o2)):
                        for k in idxs[:2]:
                            if len(g[buf]) < args.n_caption_examples:
                                g[buf].append(cap_of(s0 + int(k)))
            if (s0 // args.batch_size) % 40 == 0:
                print(f"  batch {s0 // args.batch_size}/{(N + args.batch_size - 1) // args.batch_size}")

    # finalize metrics
    out_groups = []
    for g in groups:
        K = len(g["partners"])
        sizes = g["n_partner"]
        jac, contain = {}, {}
        for a, b in combinations(range(K), 2):
            inter = g["n_pair_inter"][f"{a}-{b}"]
            uni = sizes[a] + sizes[b] - inter
            jac[f"{a}-{b}"] = inter / uni if uni else None
            m = min(sizes[a], sizes[b])
            contain[f"{a}-{b}"] = inter / m if m else None
        rec = {
            "img_latent": g["img_latent"],
            "partners": g["partners"],
            "n_img_fires": g["n_img_fires"],
            "n_cofire_per_partner": sizes,
            "n_union": g["n_union"],
            "union_coverage_of_img": g["n_union"] / g["n_img_fires"] if g["n_img_fires"] else None,
            "top_partner_share_of_union": (max(sizes) / g["n_union"]) if g["n_union"] else None,
            "jaccard": jac,
            "containment": contain,
        }
        if want_caps:
            rec["captions_shared_p1p2"] = g["ex_shared"]
            rec["captions_only_p1"] = g["ex_only1"]
            rec["captions_only_p2"] = g["ex_only2"]
        out_groups.append(rec)

    # aggregate distributions over groups
    j12 = [g["jaccard"].get("0-1") for g in out_groups if g["jaccard"].get("0-1") is not None]
    tps = [g["top_partner_share_of_union"] for g in out_groups
           if g["top_partner_share_of_union"] is not None]
    agg = {
        "n_groups": len(out_groups),
        "jaccard_p1p2_pct": {q: float(np.percentile(j12, int(q))) for q in ("10", "25", "50", "75", "90")} if j12 else {},
        "top_partner_share_pct": {q: float(np.percentile(tps, int(q))) for q in ("10", "25", "50", "75", "90")} if tps else {},
        "frac_groups_jaccard_p1p2_lt_0.1": float(np.mean([j < 0.1 for j in j12])) if j12 else None,
        "frac_groups_jaccard_p1p2_gt_0.5": float(np.mean([j > 0.5 for j in j12])) if j12 else None,
    }
    result = {"tau": args.tau, "dataset": args.dataset, "corpus_size": N,
              "selection": ({"mode": "relative", "rel_ratio": args.rel_ratio,
                             "min_top1": args.min_top1} if args.relative
                            else {"mode": "absolute", "tau": args.tau}),
              "aggregate": agg, "groups": out_groups}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
