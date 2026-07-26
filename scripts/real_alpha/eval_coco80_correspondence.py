"""Do matched image and text latents stand for the same object?

Co-activation correlation says two latents fire together. It does not say they
mean the same thing — chairs and tables co-occur in photographs without being
the same concept. This test uses COCO's object annotations as an outside
reference.

For each of the 80 object categories we find the image latent that best
separates photos containing that object from photos that do not, and separately
the text latent that best separates captions of such photos from other captions.
Neither search looks at the other modality, and the two draw on disjoint halves
of the photographs, so nothing links them except the concept. The permutation
learned from co-activation claims to link image latent i to text latent P(i). If
that claim is right, the two independently chosen latents should be a matched
pair.

Separation is measured by AUC — the probability that the latent is more active
on a positive sample than on a negative one. A difference of means picks
whichever latent fires most often, and a t-statistic blows up on a latent that
fires on three positives and nothing else; at a firing rate under 1% the second
failure is the more likely one. AUC is bounded, ignores scale, and a minimum
support requirement removes the ultra-rare latents outright.

A bare agreement rate would be uninterpretable, so four references come with it:
the chance rate, a random permutation, a shuffle of the category labels that
exposes the degenerate case where a few busy latents win everything, and the
rate at which the image side agrees with *itself* across two disjoint halves,
which is the highest agreement any cross-modal test could reach.

    python scripts/real_alpha/eval_coco80_correspondence.py \
        --panel outputs/rebuttal_EA/cc3m_k32_r1r2/C_img_txt.npz \
        --ckpt outputs/rebuttal_models/cc3m_k32_r1/final \
        --labels cache/coco80_labels.npz \
        --cache-dir cache/clip_b32_coco \
        --out outputs/rebuttal_EB/cc3m_k32_r1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402
from coco80_synonyms import COCO_80, matches  # type: ignore  # noqa: E402
from rebuttal_common import alive_masks, hungarian_perm, load_panel  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True, help="C_img_txt.npz, used to build the permutation")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--labels", default="cache/coco80_labels.npz")
    p.add_argument("--cache-dir", default="cache/clip_b32_coco")
    p.add_argument("--captions", default="cache/coco_karpathy_captions.json")
    p.add_argument("--alive-rule", choices=["ever", "density"], default="ever")
    p.add_argument("--min-count", type=int, default=50,
                   help="minimum positives per side for a category to be scored")
    p.add_argument("--min-support", type=float, default=0.05,
                   help="a latent must fire on at least this share of a category's positives")
    p.add_argument("--require-caption-mention", action="store_true",
                   help="text positives must also name the object; reported as a "
                        "secondary variant because it mixes a lexical criterion into "
                        "an object-level one")
    p.add_argument("--unfiltered-area", action="store_true",
                   help="use labels without the salience filter (sensitivity check)")
    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def sparse_latents(sae, table: torch.Tensor, rows: np.ndarray, batch_size: int,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode rows and keep only the non-zeros.

    A Top-K SAE leaves all but k of its 4096 coordinates exactly zero, so the
    dense matrix would be 99% zeros. Keeping (sample, latent, value) triples
    holds the same information in a fraction of the memory and is what the AUC
    computation below consumes.
    """
    sae.eval().to(device)
    samp, lat, val = [], [], []
    for s in range(0, len(rows), batch_size):
        idx = torch.as_tensor(rows[s:s + batch_size])
        x = table[idx].to(device)
        z = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True).dense_latents.squeeze(1)
        nz = torch.nonzero(z, as_tuple=False)
        samp.append((nz[:, 0] + s).cpu().numpy())
        lat.append(nz[:, 1].cpu().numpy())
        val.append(z[nz[:, 0], nz[:, 1]].float().cpu().numpy())
    return (np.concatenate(samp), np.concatenate(lat), np.concatenate(val))


def auc_matrix(samp: np.ndarray, lat: np.ndarray, val: np.ndarray,
               labels: np.ndarray, n_samples: int, n_latents: int,
               min_support: float) -> tuple[np.ndarray, np.ndarray]:
    """AUC of every (latent, category) pair, plus each pair's support.

    Exact, not approximate. All the zero-valued samples of a latent tie at the
    bottom of its ranking, so their contribution is a closed form and only the
    non-zeros need sorting.
    """
    n_cat = labels.shape[1]
    n_pos = labels.sum(axis=0).astype(np.float64)             # (n_cat,)
    n_neg = float(n_samples) - n_pos

    auc = np.full((n_latents, n_cat), 0.5, dtype=np.float64)
    support = np.zeros((n_latents, n_cat), dtype=np.float64)

    order = np.argsort(lat, kind="stable")
    lat_s, samp_s, val_s = lat[order], samp[order], val[order]
    bounds = np.searchsorted(lat_s, np.arange(n_latents + 1))

    for l in range(n_latents):
        lo, hi = bounds[l], bounds[l + 1]
        if hi == lo:
            continue
        v = val_s[lo:hi]
        s_idx = samp_s[lo:hi]
        o = np.argsort(v, kind="stable")           # ascending activation
        s_sorted = s_idx[o]

        pos = labels[s_sorted]                     # (S, n_cat) bool
        neg = ~pos
        # negatives strictly below each non-zero position
        cum_neg = np.cumsum(neg, axis=0) - neg
        u_nz = (cum_neg * pos).sum(axis=0).astype(np.float64)

        n_pos_nz = pos.sum(axis=0).astype(np.float64)
        n_neg_nz = neg.sum(axis=0).astype(np.float64)
        n_neg_zero = n_neg - n_neg_nz
        n_pos_zero = n_pos - n_pos_nz

        # every non-zero positive beats every zero negative; zeros tie with zeros
        u = u_nz + n_pos_nz * n_neg_zero + 0.5 * n_pos_zero * n_neg_zero
        denom = np.maximum(n_pos * n_neg, 1.0)
        auc[l] = u / denom
        support[l] = n_pos_nz / np.maximum(n_pos, 1.0)

    ok = support >= min_support
    auc = np.where(ok, auc, -np.inf)
    return auc, support


def rank_of(values: np.ndarray, target: int) -> int:
    """1-based rank of ``target`` when ``values`` is sorted descending.

    Used in both directions. Asking where the image side's pick lands in the
    text side's ranking is a different question from asking where the text
    side's pick lands in the image side's, and the two need not agree, so both
    are reported rather than one standing in for the other. At rank 1 they
    describe the same event: the two sides chose permutation-matched
    coordinates.
    """
    return int((values > values[target]).sum()) + 1


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- permutation and candidate coordinates -------------------------------
    panel = load_panel(args.panel)
    alive_i, alive_t = alive_masks(panel, args.alive_rule)
    match = hungarian_perm(panel["C"], alive_i, alive_t)
    perm = match["perm"]
    M_img = np.where(match["usable"])[0]      # image latents with a live partner
    M_txt = perm[M_img]
    m_eff = len(M_img)
    logger.info("candidate coordinates: %d (alive image %d, alive text %d)",
                m_eff, match["n_alive_a"], match["n_alive_b"])

    # ---- labels --------------------------------------------------------------
    lab = np.load(args.labels)
    Y = (lab["Y_unfiltered"] if args.unfiltered_area else lab["Y"]).astype(bool)
    label_ids = lab["image_ids"].astype(np.int64)
    half = lab["half"].astype(np.int8)

    ds = eval_utils.load_pair_dataset(args.cache_dir, "coco", split="train")
    # every split shares one embedding table, so any split object can resolve rows
    id_to_row = ds._image_id_to_row
    keep = np.array([int(i) in id_to_row for i in label_ids])
    Y, label_ids, half = Y[keep], label_ids[keep], half[keep]
    logger.info("labeled images resolvable in the cache: %d", len(label_ids))

    # ---- image side ----------------------------------------------------------
    img_rows_all = np.array([id_to_row[int(i)] for i in label_ids], dtype=np.int64)
    sae_i = eval_utils.load_sae(args.ckpt, "separated").image_sae
    logger.info("encoding %d images", len(img_rows_all))
    i_samp, i_lat, i_val = sparse_latents(sae_i, ds._image_table, img_rows_all,
                                          args.batch_size, device)
    n_latents = int(sae_i.latent_size)

    # ---- text side -----------------------------------------------------------
    with open(args.captions) as f:
        cap_json = json.load(f)
    cap_rows, cap_owner, cap_text = [], [], []
    text_key_to_row = ds._text_key_to_row
    for pos, iid in enumerate(label_ids):
        for ci in range(5):
            key = ds._text_key_for(int(iid), ci)
            if key is None or key not in text_key_to_row:
                continue
            cap_rows.append(text_key_to_row[key])
            cap_owner.append(pos)
            cap_text.append(cap_json.get(f"{int(iid)}::{ci}", cap_json.get(f"{int(iid)}_{ci}", "")))
    cap_rows = np.array(cap_rows, dtype=np.int64)
    cap_owner = np.array(cap_owner, dtype=np.int64)
    logger.info("captions resolvable: %d", len(cap_rows))

    sae_t = eval_utils.load_sae(args.ckpt, "separated").text_sae
    logger.info("encoding %d captions", len(cap_rows))
    t_samp, t_lat, t_val = sparse_latents(sae_t, ds._text_table, cap_rows,
                                          args.batch_size, device)

    # caption-level labels inherit their image's labels
    Y_cap = Y[cap_owner]
    if args.require_caption_mention:
        mention = np.zeros_like(Y_cap)
        for c_idx, cat in enumerate(COCO_80):
            mention[:, c_idx] = np.array([matches(t, cat) for t in cap_text])
        Y_cap = Y_cap & mention
        logger.info("caption-mention requirement applied; positives drop to %d",
                    int(Y_cap.sum()))

    # ---- split the population ------------------------------------------------
    img_half = half == 0
    txt_half = half == 1
    cap_in_txt_half = txt_half[cap_owner]
    cap_in_img_half = ~cap_in_txt_half

    def side_auc(samp, lat, val, labels, mask, n_lat):
        sel = mask[samp]
        remap = np.cumsum(mask) - 1
        return auc_matrix(remap[samp[sel]], lat[sel], val[sel],
                          labels[mask], int(mask.sum()), n_lat, args.min_support)

    auc_img, sup_img = side_auc(i_samp, i_lat, i_val, Y, img_half, n_latents)
    auc_txt, sup_txt = side_auc(t_samp, t_lat, t_val, Y_cap, cap_in_txt_half, n_latents)
    # the ceiling uses the other half of the same modality, at the same size
    auc_img_other, _ = side_auc(i_samp, i_lat, i_val, Y, txt_half, n_latents)
    auc_txt_other, _ = side_auc(t_samp, t_lat, t_val, Y_cap, cap_in_img_half, n_latents)

    n_pos_img = Y[img_half].sum(axis=0)
    n_pos_txt = Y_cap[cap_in_txt_half].sum(axis=0)
    usable_cat = [c for c in range(80)
                  if n_pos_img[c] >= args.min_count and n_pos_txt[c] >= args.min_count]
    logger.info("categories scored: %d / 80", len(usable_cat))

    # ---- the test ------------------------------------------------------------
    rows = []
    ranks_in_text, ranks_in_image, agree_img, agree_txt = [], [], [], []
    for c in usable_cat:
        a_img = auc_img[M_img, c]
        a_txt = auc_txt[M_txt, c]
        if not np.isfinite(a_img).any() or not np.isfinite(a_txt).any():
            continue
        i_star_pos = int(np.argmax(a_img))
        j_star_pos = int(np.argmax(a_txt))

        # Two directions of the same question, both reported.
        #   r_text  — where the image side's pick sits in the text side's ranking
        #   r_image — where the text side's pick sits in the image side's ranking
        # Both equal 1 exactly when the two sides agree on a matched pair.
        r_text = rank_of(a_txt, i_star_pos)
        r_image = rank_of(a_img, j_star_pos)
        ranks_in_text.append(r_text)
        ranks_in_image.append(r_image)

        ai_o = auc_img_other[M_img, c]
        at_o = auc_txt_other[M_txt, c]
        agree_img.append(int(np.isfinite(ai_o).any() and int(np.argmax(ai_o)) == i_star_pos))
        agree_txt.append(int(np.isfinite(at_o).any() and int(np.argmax(at_o)) == j_star_pos))

        rows.append({
            "category": COCO_80[c],
            "n_pos_image": int(n_pos_img[c]), "n_pos_text": int(n_pos_txt[c]),
            "image_latent": int(M_img[i_star_pos]),
            "text_latent_via_perm": int(M_txt[i_star_pos]),
            "text_latent_chosen": int(M_txt[j_star_pos]),
            "rank_in_text": r_text, "rank_in_image": r_image,
            "rank_pct_in_text": float(1.0 - (r_text - 1) / max(m_eff - 1, 1)),
            "auc_image": float(a_img[i_star_pos]), "auc_text": float(a_txt[j_star_pos]),
            "image_self_agreement": agree_img[-1], "text_self_agreement": agree_txt[-1],
        })

    n = len(rows)
    r_text_arr = np.array(ranks_in_text, dtype=np.float64)
    r_image_arr = np.array(ranks_in_image, dtype=np.float64)

    def at_k(ranks: np.ndarray, k: int) -> float:
        return float(np.mean(ranks <= k)) if ranks.size else float("nan")

    # ---- controls ------------------------------------------------------------
    scored = [c for c in usable_cat
              if np.isfinite(auc_img[M_img, c]).any() and np.isfinite(auc_txt[M_txt, c]).any()]
    i_star = np.array([int(np.argmax(auc_img[M_img, c])) for c in scored])
    j_star = np.array([int(np.argmax(auc_txt[M_txt, c])) for c in scored])

    # Random permutation: the two sides keep the coordinates they chose, only the
    # linkage between them is randomized.
    null_hits = np.array([
        float(np.mean(rng.permutation(m_eff)[i_star] == j_star))
        for _ in range(args.n_null)
    ])

    # Label shuffle: the real permutation, but each category's image latent is
    # checked against a different category's text latent. If a handful of busy
    # latents win everything, this rises to meet the real number.
    lab_shuffle_hit = float(np.mean(i_star == j_star[rng.permutation(len(scored))]))

    report = {
        "ckpt": args.ckpt, "panel": args.panel,
        "labels": args.labels,
        "area_filtered": not args.unfiltered_area,
        "require_caption_mention": args.require_caption_mention,
        "alive_rule": args.alive_rule, "min_support": args.min_support,
        "m_eff": int(m_eff), "n_categories": int(n),
        "categories_dropped": [COCO_80[c] for c in range(80) if c not in usable_cat],
        "ranking_criterion": (
            "AUC of the latent's activation separating a category's positives "
            "from its negatives, computed independently on each side. The "
            "co-activation correlation is used only to build the permutation, "
            "never to rank."
        ),
        "result": {
            "agree@1": at_k(r_text_arr, 1),
            "image_pick_in_text_ranking": {
                "top1": at_k(r_text_arr, 1), "top5": at_k(r_text_arr, 5),
                "top10": at_k(r_text_arr, 10),
                "median_rank": float(np.median(r_text_arr)) if n else float("nan"),
                "mrr": float(np.mean(1.0 / r_text_arr)) if n else float("nan"),
            },
            "text_pick_in_image_ranking": {
                "top1": at_k(r_image_arr, 1), "top5": at_k(r_image_arr, 5),
                "top10": at_k(r_image_arr, 10),
                "median_rank": float(np.median(r_image_arr)) if n else float("nan"),
                "mrr": float(np.mean(1.0 / r_image_arr)) if n else float("nan"),
            },
        },
        "controls": {
            "chance_hit@1": 1.0 / m_eff,
            "random_permutation_hit@1_mean": float(null_hits.mean()),
            "random_permutation_hit@1_p95": float(np.percentile(null_hits, 95)),
            "p_value_vs_random_permutation": float(np.mean(null_hits >= at_k(r_text_arr, 1))),
            "label_shuffle_hit@1": lab_shuffle_hit,
            "image_self_agreement": float(np.mean(agree_img)) if n else float("nan"),
            "text_self_agreement": float(np.mean(agree_txt)) if n else float("nan"),
        },
        "distinct_image_latents_chosen": int(len({r["image_latent"] for r in rows})),
        "per_category": rows,
    }
    (out / "coco80_correspondence.json").write_text(json.dumps(report, indent=2))

    import csv  # noqa: PLC0415
    with open(out / "per_category.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    res, ctl = report["result"], report["controls"]
    print()
    print(f"categories scored {n}/80, candidate coordinates {m_eff}")
    print("ranking criterion: per-side AUC on the category's positives vs negatives")
    print(f"  {'direction':<34}{'top1':>8}{'top5':>8}{'top10':>8}{'median rank':>13}{'MRR':>8}")
    for key, label in (("image_pick_in_text_ranking", "image pick, in text ranking"),
                       ("text_pick_in_image_ranking", "text pick, in image ranking")):
        d = res[key]
        print(f"  {label:<34}{100 * d['top1']:>7.1f}%{100 * d['top5']:>7.1f}%"
              f"{100 * d['top10']:>7.1f}%{d['median_rank']:>10.0f} of {m_eff}{d['mrr']:>8.3f}")
    print("  against (agreement at rank 1)")
    print(f"    chance                       {100 * ctl['chance_hit@1']:.3f}%")
    print(f"    random permutation           {100 * ctl['random_permutation_hit@1_mean']:.3f}%"
          f"   (p={ctl['p_value_vs_random_permutation']:.4f})")
    print(f"    category labels shuffled     {100 * ctl['label_shuffle_hit@1']:.1f}%")
    print(f"    image side vs its other half {100 * ctl['image_self_agreement']:.1f}%  <- ceiling")
    print(f"    text  side vs its other half {100 * ctl['text_self_agreement']:.1f}%")
    print(f"  distinct image latents chosen: {report['distinct_image_latents_chosen']} of {n}")
    print(f"\nwrote {out / 'coco80_correspondence.json'}")


if __name__ == "__main__":
    main()
