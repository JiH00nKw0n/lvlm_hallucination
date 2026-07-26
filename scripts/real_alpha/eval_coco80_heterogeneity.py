"""Quantify cross-modal heterogeneity from human labels instead of co-activation.

Reviewer PBPC's third weakness: our headline measurement reads co-activation out
of the very latent space whose heterogeneity is in question, so the noise in that
space could be manufacturing the effect. A ground truth from outside the model
would settle it.

COCO's object annotations are that ground truth. For one category, the image side
picks its best-separating coordinate using only images and labels, and the text
side picks its own using only captions and labels. Neither side sees the other,
and neither side sees a correlation matrix. The angle between the two decoder
directions is then an estimate of heterogeneity that co-activation never touched.

A raw angle means nothing on its own, so the same selection procedure runs a
second time inside a single modality: the image side picks a coordinate on one
half of the photographs and again on the other half. That pair carries the same
label noise, the same AUC-estimation noise, and the same finite-sample noise as
the cross-modal pair, and differs from it in exactly one respect -- no modality
boundary is crossed. The gap between those two numbers is the part that only
heterogeneity explains.

Two further baselines pin the scale. Pairing a category's image pick with a
*different* category's text pick gives the angle between unrelated concepts, and
random unit vectors in the embedding dimension give the floor.

Candidates are every coordinate that fires on the labelled data for its own
modality. The permutation is deliberately not used to restrict them, because
restricting by it would smuggle co-activation back in. It is reported separately,
as a bridge: if the label-selected partner and the co-activation-selected partner
sit at a similar angle, the co-activation metric was not the source of the effect.

    python scripts/real_alpha/eval_coco80_heterogeneity.py \
        --ckpt outputs/rebuttal_models/cc3m_k32_r1/final \
        --panel outputs/rebuttal_EA/cc3m_k32_r1r2/C_img_txt.npz \
        --out outputs/rebuttal_EG/cc3m_k32_r1
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
from coco80_synonyms import COCO_80  # type: ignore  # noqa: E402
from eval_coco80_correspondence import (  # type: ignore  # noqa: E402
    auc_matrix,
    pick_device,
    sparse_latents,
)
from rebuttal_common import alive_masks, hungarian_perm, load_panel  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["separated", "shared", "aux"], default="separated")
    p.add_argument("--panel", default=None,
                   help="correlation matrix, used only for the co-activation "
                        "comparison row; the label-based numbers ignore it")
    p.add_argument("--labels", default="cache/coco80_labels.npz")
    p.add_argument("--cache-dir", default="cache/clip_b32_coco")
    p.add_argument("--captions", default="cache/coco_karpathy_captions.json")
    p.add_argument("--alive-rule", choices=["ever", "density"], default="ever")
    p.add_argument("--min-count", type=int, default=50)
    p.add_argument("--min-support", type=float, default=0.05)
    p.add_argument("--unfiltered-area", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def unit_rows(w: torch.Tensor) -> np.ndarray:
    a = w.detach().float().cpu().numpy()
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def summarize(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Median and mean with a percentile bootstrap over categories."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots_med = np.median(v[idx], axis=1)
    boots_mean = np.mean(v[idx], axis=1)
    return {
        "n": int(v.size),
        "cos_median": float(np.median(v)),
        "cos_median_ci95": [float(np.percentile(boots_med, 2.5)),
                            float(np.percentile(boots_med, 97.5))],
        "cos_mean": float(np.mean(v)),
        "cos_mean_ci95": [float(np.percentile(boots_mean, 2.5)),
                          float(np.percentile(boots_mean, 97.5))],
        "distance_median": float(1.0 - np.median(v)),
        "cos_p05": float(np.percentile(v, 5)),
        "cos_p95": float(np.percentile(v, 95)),
        "share_above_0.9": float(np.mean(v > 0.9)),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shared = args.method in ("shared", "aux")

    # ---- labels --------------------------------------------------------------
    lab = np.load(args.labels)
    Y = (lab["Y_unfiltered"] if args.unfiltered_area else lab["Y"]).astype(bool)
    label_ids = lab["image_ids"].astype(np.int64)
    half = lab["half"].astype(np.int8)

    ds = eval_utils.load_pair_dataset(args.cache_dir, "coco", split="train")
    id_to_row = ds._image_id_to_row
    keep = np.array([int(i) in id_to_row for i in label_ids])
    Y, label_ids, half = Y[keep], label_ids[keep], half[keep]
    logger.info("labeled images resolvable in the cache: %d", len(label_ids))

    # ---- encode both sides ---------------------------------------------------
    model = eval_utils.load_sae(args.ckpt, args.method)
    sae_i = model if shared else model.image_sae
    sae_t = model if shared else model.text_sae
    n_latents = int(sae_i.latent_size)

    img_rows = np.array([id_to_row[int(i)] for i in label_ids], dtype=np.int64)
    logger.info("encoding %d images", len(img_rows))
    i_samp, i_lat, i_val = sparse_latents(sae_i, ds._image_table, img_rows,
                                          args.batch_size, device)

    cap_rows, cap_owner = [], []
    text_key_to_row = ds._text_key_to_row
    for pos, iid in enumerate(label_ids):
        for ci in range(5):
            key = ds._text_key_for(int(iid), ci)
            if key is None or key not in text_key_to_row:
                continue
            cap_rows.append(text_key_to_row[key])
            cap_owner.append(pos)
    cap_rows = np.array(cap_rows, dtype=np.int64)
    cap_owner = np.array(cap_owner, dtype=np.int64)
    logger.info("encoding %d captions", len(cap_rows))
    t_samp, t_lat, t_val = sparse_latents(sae_t, ds._text_table, cap_rows,
                                          args.batch_size, device)
    Y_cap = Y[cap_owner]

    # ---- candidates: fires on the labelled data, nothing to do with matching --
    fires_img = np.zeros(n_latents, dtype=bool)
    fires_img[np.unique(i_lat)] = True
    fires_txt = np.zeros(n_latents, dtype=bool)
    fires_txt[np.unique(t_lat)] = True
    cand_img = np.where(fires_img)[0]
    cand_txt = np.where(fires_txt)[0]
    logger.info("candidates: image %d, text %d", len(cand_img), len(cand_txt))

    # ---- four independent selectors ------------------------------------------
    # Photographs split by md5, captions follow their photograph. The image side
    # and the text side therefore never see the same picture, which is what stops
    # an agreement from coming out of the pairing rather than the concept.
    img_half = half == 0
    txt_half = half == 1
    cap_in_txt_half = txt_half[cap_owner]

    def side_auc(samp, lat, val, labels, mask):
        sel = mask[samp]
        remap = np.cumsum(mask) - 1
        return auc_matrix(remap[samp[sel]], lat[sel], val[sel],
                          labels[mask], int(mask.sum()), n_latents,
                          args.min_support)[0]

    auc_img_a = side_auc(i_samp, i_lat, i_val, Y, img_half)
    auc_img_b = side_auc(i_samp, i_lat, i_val, Y, txt_half)
    auc_txt_a = side_auc(t_samp, t_lat, t_val, Y_cap, cap_in_txt_half)
    auc_txt_b = side_auc(t_samp, t_lat, t_val, Y_cap, ~cap_in_txt_half)

    n_pos_img = Y[img_half].sum(axis=0)
    n_pos_txt = Y_cap[cap_in_txt_half].sum(axis=0)
    usable = [c for c in range(80)
              if n_pos_img[c] >= args.min_count and n_pos_txt[c] >= args.min_count]
    logger.info("categories scored: %d / 80", len(usable))

    W_img = unit_rows(sae_i.W_dec)
    W_txt = unit_rows(sae_t.W_dec)
    d_embed = W_img.shape[1]

    def pick(auc: np.ndarray, cand: np.ndarray, c: int) -> int | None:
        col = auc[cand, c]
        if not np.isfinite(col).any():
            return None
        return int(cand[int(np.argmax(col))])

    rows = []
    for c in usable:
        i_a = pick(auc_img_a, cand_img, c)
        i_b = pick(auc_img_b, cand_img, c)
        t_a = pick(auc_txt_a, cand_txt, c)
        t_b = pick(auc_txt_b, cand_txt, c)
        if i_a is None or t_a is None:
            continue
        rec = {
            "category": COCO_80[c],
            "image_latent": i_a,
            "text_latent": t_a,
            "image_latent_other_half": i_b,
            "text_latent_other_half": t_b,
            "cross_modal_cos": float(W_img[i_a] @ W_txt[t_a]),
            "within_image_cos": (float(W_img[i_a] @ W_img[i_b]) if i_b is not None else None),
            "within_text_cos": (float(W_txt[t_a] @ W_txt[t_b]) if t_b is not None else None),
            "n_pos_image": int(n_pos_img[c]),
            "n_pos_text": int(n_pos_txt[c]),
        }
        rows.append(rec)

    if not rows:
        raise SystemExit("no category survived; check --min-count and --min-support")

    cross = np.array([r["cross_modal_cos"] for r in rows])
    w_img = np.array([r["within_image_cos"] for r in rows if r["within_image_cos"] is not None])
    w_txt = np.array([r["within_text_cos"] for r in rows if r["within_text_cos"] is not None])

    # ---- baselines -----------------------------------------------------------
    # Unrelated concepts: this category's image pick against every other
    # category's text pick. Same coordinates, wrong correspondence.
    ii = np.array([r["image_latent"] for r in rows])
    tt = np.array([r["text_latent"] for r in rows])
    M = W_img[ii] @ W_txt[tt].T
    off = M[~np.eye(len(rows), dtype=bool)]

    g = rng.standard_normal((4000, d_embed))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    rand_cos = (g[::2] * g[1::2]).sum(axis=1)

    result = {
        "cross_modal_matched_category": summarize(cross, rng, args.n_boot),
        "within_image_two_halves": summarize(w_img, rng, args.n_boot),
        "within_text_two_halves": summarize(w_txt, rng, args.n_boot),
        "cross_modal_mismatched_category": summarize(off, rng, args.n_boot),
        "random_unit_vectors": summarize(rand_cos, rng, args.n_boot),
    }

    # paired, since both numbers exist for the same category
    both = [(r["cross_modal_cos"], r["within_image_cos"]) for r in rows
            if r["within_image_cos"] is not None]
    if both:
        a = np.array([x for x, _ in both])
        b = np.array([y for _, y in both])
        diff = b - a
        idx = rng.integers(0, diff.size, size=(args.n_boot, diff.size))
        bmean = np.mean(diff[idx], axis=1)
        from scipy.stats import wilcoxon  # noqa: PLC0415
        result["within_image_minus_cross_modal"] = {
            "n_categories": int(diff.size),
            "median": float(np.median(diff)),
            "mean": float(np.mean(diff)),
            "mean_ci95": [float(np.percentile(bmean, 2.5)),
                          float(np.percentile(bmean, 97.5))],
            "wilcoxon_p": float(wilcoxon(b, a).pvalue),
            "share_within_image_greater": float(np.mean(diff > 0)),
        }

    # ---- bridge to the co-activation metric ----------------------------------
    # Same image coordinate, but the partner chosen by correlation instead of by
    # labels. If the two agree, the correlation metric was not inventing the gap.
    if args.panel and not shared:
        panel = load_panel(args.panel)
        alive_i, alive_t = alive_masks(panel, args.alive_rule)
        match = hungarian_perm(panel["C"], alive_i, alive_t)
        perm, ok = match["perm"], match["usable"]
        vals, corr = [], []
        for r in rows:
            i = r["image_latent"]
            if not ok[i]:
                continue
            vals.append(float(W_img[i] @ W_txt[perm[i]]))
            corr.append(float(match["matched_c"][i]))
        if vals:
            result["coactivation_partner_of_the_same_image_latent"] = summarize(
                np.array(vals), rng, args.n_boot)
            result["coactivation_partner_of_the_same_image_latent"]["correlation_median"] = \
                float(np.median(corr))
            result["coactivation_partner_of_the_same_image_latent"]["n_covered"] = len(vals)

    payload = {
        "ckpt": args.ckpt,
        "method": args.method,
        "labels": args.labels,
        "area_filtered": not args.unfiltered_area,
        "min_support": args.min_support,
        "min_count": args.min_count,
        "n_categories": len(rows),
        "n_candidates_image": int(len(cand_img)),
        "n_candidates_text": int(len(cand_txt)),
        "embedding_dim": int(d_embed),
        "selection": "argmax AUC per side, computed on disjoint halves of the "
                     "photographs; co-activation is never consulted",
        "result": result,
        "per_category": rows,
    }
    (out / "coco80_heterogeneity.json").write_text(json.dumps(payload, indent=1))

    def line(name: str, key: str) -> None:
        s = result.get(key, {})
        if not s.get("n"):
            return
        ci = s["cos_mean_ci95"]
        print(f"  {name:44s} cos median {s['cos_median']:.3f}  "
              f"mean {s['cos_mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  n={s['n']}")

    print(f"\ncategories {len(rows)}, candidates image {len(cand_img)} / text {len(cand_txt)}")
    line("cross-modal, same category", "cross_modal_matched_category")
    line("within image, two halves", "within_image_two_halves")
    line("within text, two halves", "within_text_two_halves")
    line("cross-modal, different category", "cross_modal_mismatched_category")
    line("random unit vectors", "random_unit_vectors")
    line("co-activation partner (same image latent)",
         "coactivation_partner_of_the_same_image_latent")
    p = result.get("within_image_minus_cross_modal")
    if p:
        print(f"\n  within-image minus cross-modal: mean {p['mean']:+.3f} "
              f"[{p['mean_ci95'][0]:+.3f}, {p['mean_ci95'][1]:+.3f}], "
              f"Wilcoxon p={p['wilcoxon_p']:.2e}, "
              f"{100 * p['share_within_image_greater']:.0f}% of categories")
    print(f"\nwrote {out / 'coco80_heterogeneity.json'}")


if __name__ == "__main__":
    main()
