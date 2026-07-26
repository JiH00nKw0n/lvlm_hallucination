"""Is the cross-modal gap still there for concepts that two runs both recover?

If a dictionary direction is an accident of one training run, the distance
between it and anything in the other modality says nothing about the data. So
restrict attention to directions that two independently trained image SAEs both
find, and ask whether the cross-modal distance survives on those.

Reproducibility is scored the way Papadimitriou et al. (arXiv 2504.11695) score
it, following Fel et al. (2025) and Spielman et al. (2012): align the two
dictionaries' rows with the assignment that maximizes total similarity, and take
each concept's matched cosine as its stability. Concepts are then binned by that
score, from the top 1% down through the full range.

Two things are reported that the headline alone would hide. The first is the
whole decile curve, because the conclusion moves with the quantile — reporting
one cut would be choosing the answer. The second is the same measurement
restricted to pairs that actually co-activate, since a stable direction whose
cross-modal partner never fires alongside it is not a semantic correspondence
and should not be read as one.

    python scripts/real_alpha/analyze_stability_conditioned.py \
        --panel-img-img outputs/rebuttal_EA/coco_k8_r1r2/C_img_img.npz \
        --panel-img-txt outputs/rebuttal_EA/coco_k8_r1r2/C_img_txt.npz \
        --ckpt-a outputs/rebuttal_models/coco_k8_r1/final \
        --ckpt-b outputs/rebuttal_models/coco_k8_r2/final \
        --out outputs/rebuttal_ED/coco_k8_r1r2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from rebuttal_common import (  # type: ignore  # noqa: E402
    alive_masks,
    describe,
    hungarian_perm,
    load_panel,
    unit_decoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel-img-img", required=True,
                   help="only used for its alive masks over the two runs")
    p.add_argument("--panel-img-txt", required=True,
                   help="supplies the cross-modal permutation and correlations")
    p.add_argument("--panel-txt-txt", default=None,
                   help="text-side alive masks, so stability can be required of both "
                        "endpoints of a cross-modal pair rather than only the image one")
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--ckpt-b", required=True)
    p.add_argument("--alive-rule", choices=["ever", "density"], default="ever")
    p.add_argument("--co-activation-min", type=float, default=0.6,
                   help="correlation above which a cross-modal pair counts as a "
                        "genuine correspondence")
    p.add_argument("--out", required=True)
    return p.parse_args()


def geometry_stability(Wa: np.ndarray, Wb: np.ndarray,
                       alive_a: np.ndarray, alive_b: np.ndarray) -> dict:
    """Match the two dictionaries by direction similarity and score each concept.

    This is the cited definition: the assignment maximizes total cosine between
    matched concept vectors, so a concept's stability is the cosine it achieves
    with its counterpart in the other run.
    """
    ra, rb = np.where(alive_a)[0], np.where(alive_b)[0]
    S = Wa[ra] @ Wb[rb].T
    row, col = linear_sum_assignment(-S)
    return {
        "rows": ra[row],
        "partner": rb[col],
        "score": S[row, col],
        "mean_stability": float(np.mean(S[row, col])),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p_ii = load_panel(args.panel_img_img)
    p_it = load_panel(args.panel_img_txt)
    alive_a, alive_b = alive_masks(p_ii, args.alive_rule)      # image side, runs A and B
    alive_i, alive_t = alive_masks(p_it, args.alive_rule)      # image and text, run A

    Wa = unit_decoder(args.ckpt_a, "image")
    Wb = unit_decoder(args.ckpt_b, "image")
    Wt = unit_decoder(args.ckpt_a, "text")

    stab = geometry_stability(Wa, Wb, alive_a, alive_b)
    logger.info("stability over %d matched concepts: mean %.4f",
                len(stab["rows"]), stab["mean_stability"])

    C_it = p_it["C"]
    cross = hungarian_perm(C_it, alive_i, alive_t)
    perm, usable, matched_c = cross["perm"], cross["usable"], cross["matched_c"]

    # The same-modality side is matched by decoder cosine, which is the very
    # quantity being reported, while the cross-modal side is matched by
    # co-activation. Part of any gap between them is therefore the choice of
    # matching operator rather than modality. Matching the cross-modal side the
    # same way separates the two.
    cross_geom = geometry_stability(Wa, Wt, alive_i, alive_t)
    geom_partner = np.full(Wa.shape[0], -1, dtype=np.int64)
    geom_partner[cross_geom["rows"]] = cross_geom["partner"]

    # keep concepts that are alive on both sides of both comparisons
    keep = usable[stab["rows"]]
    rows = stab["rows"][keep]
    s = stab["score"][keep]
    d_same = 1.0 - s
    d_cross = 1.0 - (Wa[rows] * Wt[perm[rows]]).sum(axis=1)
    has_geom = geom_partner[rows] >= 0
    d_cross_geom = np.full(len(rows), np.nan)
    d_cross_geom[has_geom] = 1.0 - (
        Wa[rows[has_geom]] * Wt[geom_partner[rows[has_geom]]]
    ).sum(axis=1)
    c_match = matched_c[rows]
    logger.info("concepts usable in both comparisons: %d", len(rows))

    order = np.argsort(-s)                # most reproducible first
    report = {
        "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
        "alive_rule": args.alive_rule,
        "mean_stability": stab["mean_stability"],
        "n_concepts": int(len(rows)),
        "same_modality_distance_all": describe(d_same),
        "cross_modal_distance_all": describe(d_cross),
        "cross_modal_distance_all_matched_by_geometry": describe(d_cross_geom[has_geom]),
        "operator_vs_modality": {
            "note": ("same-modality matching optimizes decoder cosine, cross-modal "
                     "matching optimizes co-activation, so the two columns are not "
                     "produced the same way; matching the cross-modal side by "
                     "decoder cosine too isolates the modality component"),
            "same_modality": float(np.median(d_same)),
            "cross_modal_matched_by_geometry": float(np.nanmedian(d_cross_geom)),
            "cross_modal_matched_by_coactivation": float(np.median(d_cross)),
            "attributable_to_operator": float(np.median(d_cross) - np.nanmedian(d_cross_geom)),
            "attributable_to_modality": float(np.nanmedian(d_cross_geom) - np.median(d_same)),
        },
        "matched_correlation_all": describe(c_match),
        "by_stability_quantile": {},
        "by_decile": {},
        "co_activating_only": {},
    }

    for q in QUANTILES:
        k = max(1, int(round(q * len(rows))))
        sel = order[:k]
        report["by_stability_quantile"][f"top_{int(q * 100)}pct"] = {
            "n": int(k),
            "stability_median": float(np.median(s[sel])),
            "same_modality_distance_median": float(np.median(d_same[sel])),
            "cross_modal_distance_median": float(np.median(d_cross[sel])),
            "cross_modal_distance_median_matched_by_geometry": (
                float(np.nanmedian(d_cross_geom[sel])) if np.isfinite(d_cross_geom[sel]).any()
                else float("nan")),
            "matched_correlation_median": float(np.median(c_match[sel])),
        }

    for dcl in range(10):
        lo, hi = int(dcl * len(rows) / 10), int((dcl + 1) * len(rows) / 10)
        sel = order[lo:hi]
        if len(sel) == 0:
            continue
        report["by_decile"][f"d{dcl + 1}"] = {
            "n": int(len(sel)),
            "stability_median": float(np.median(s[sel])),
            "cross_modal_distance_median": float(np.median(d_cross[sel])),
        }

    # The cut that matters for the paper's claim: concepts whose cross-modal
    # partner genuinely co-activates with them.
    co = c_match >= args.co_activation_min
    if co.sum() >= 5:
        s_co, d_co = s[co], d_cross[co]
        o_co = np.argsort(-s_co)
        entry = {
            "threshold": args.co_activation_min,
            "n": int(co.sum()),
            "cross_modal_distance": describe(d_co),
            "same_modality_distance": describe(d_same[co]),
        }
        for q in (0.10, 0.25, 0.50, 1.00):
            k = max(1, int(round(q * len(s_co))))
            sel = o_co[:k]
            entry[f"top_{int(q * 100)}pct_by_stability"] = {
                "n": int(k),
                "stability_median": float(np.median(s_co[sel])),
                "cross_modal_distance_median": float(np.median(d_co[sel])),
            }
        report["co_activating_only"] = entry
    else:
        report["co_activating_only"] = {
            "threshold": args.co_activation_min, "n": int(co.sum()),
            "note": "too few co-activating pairs to condition on",
        }


    # ---- reproducible on both sides AND actually corresponding ---------------
    # Conditioning on stability alone breaks in both directions. Restricting to
    # stable image atoms and following the global permutation leaves the text
    # endpoint unconstrained; forcing a bijection inside the top few percent of
    # each side pairs concepts that have no reason to correspond. Either way the
    # matched correlation collapses and the pair stops being a correspondence at
    # all. The defensible cut requires both endpoints to be reproducible and the
    # pair to genuinely co-activate, and reports the matched correlation next to
    # every cell so the reader can see when that stops being true.
    if args.panel_txt_txt:
        p_tt = load_panel(args.panel_txt_txt)
        alive_ta, alive_tb = alive_masks(p_tt, args.alive_rule)
        Wtb = unit_decoder(args.ckpt_b, "text")
        stab_t = geometry_stability(Wt, Wtb, alive_ta, alive_tb)

        img_score = np.full(Wa.shape[0], -np.inf)
        img_score[stab["rows"]] = stab["score"]
        txt_score = np.full(Wt.shape[0], -np.inf)
        txt_score[stab_t["rows"]] = stab_t["score"]

        pair_rows = np.where(usable)[0]
        pair_cos = (Wa[pair_rows] * Wt[perm[pair_rows]]).sum(axis=1)
        pair_c = matched_c[pair_rows]
        pair_stab = np.minimum(img_score[pair_rows], txt_score[perm[pair_rows]])
        finite = np.isfinite(pair_stab)

        grid = {}
        for s_min in (0.0, 0.8, 0.9, 0.95):
            for c_min in (0.0, 0.4, 0.6):
                sel = finite & (pair_stab >= s_min) & (pair_c >= c_min)
                n = int(sel.sum())
                grid[f"stability>={s_min}, c>={c_min}"] = {
                    "n": n,
                    "cosine_median": float(np.median(pair_cos[sel])) if n else None,
                    "distance_median": float(np.median(1.0 - pair_cos[sel])) if n else None,
                    "matched_correlation_median": float(np.median(pair_c[sel])) if n else None,
                    "pair_stability_median": float(np.median(pair_stab[sel])) if n else None,
                }
        quant = {}
        sp_ok = pair_stab[finite]
        cos_ok = pair_cos[finite]
        c_ok = pair_c[finite]
        order_s = np.argsort(-sp_ok)
        for q in (0.01, 0.05):
            k = max(1, int(round(q * len(sp_ok))))
            sel = order_s[:k]
            quant[f"top_{int(q * 100)}pct_by_stability"] = {
                "n": int(k),
                "pair_stability_median": float(np.median(sp_ok[sel])),
                "cosine_median": float(np.median(cos_ok[sel])),
                "distance_median": float(np.median(1.0 - cos_ok[sel])),
                "matched_correlation_median": float(np.median(c_ok[sel])),
            }

        # The requested cut: keep only pairs that genuinely correspond, then take
        # the most reproducible 1% and 5% of those.
        quant_corr = {}
        corr = finite & (pair_c >= args.co_activation_min)
        sp_c, cos_c, c_c = pair_stab[corr], pair_cos[corr], pair_c[corr]
        order_c = np.argsort(-sp_c)
        for q in (0.01, 0.05, 1.00):
            if len(sp_c) == 0:
                break
            k = max(1, int(round(q * len(sp_c))))
            sel = order_c[:k]
            quant_corr[f"top_{int(q * 100)}pct"] = {
                "n": int(k),
                "pool_size": int(len(sp_c)),
                "pair_stability_median": float(np.median(sp_c[sel])),
                "cosine_median": float(np.median(cos_c[sel])),
                "distance_median": float(np.median(1.0 - cos_c[sel])),
                "matched_correlation_median": float(np.median(c_c[sel])),
            }

        report["stable_and_corresponding"] = {
            "co_activation_min": args.co_activation_min,
            "by_stability_quantile_corresponding": quant_corr,
            "by_stability_quantile": quant,
            "text_mean_stability": stab_t["mean_stability"],
            "n_pairs_scored": int(finite.sum()),
            "note": ("pair stability is the weaker of the two endpoints; every cell "
                     "is our method's own Hungarian pairs, filtered, never rematched"),
            "grid": grid,
        }

    (out / "stability_conditioned.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"concepts compared: {len(rows)}   mean stability {stab['mean_stability']:.4f}")
    print()
    print(f"{'stability cut':<16}{'n':>6}{'stability':>12}{'d same-mod':>13}"
          f"{'d cross-mod':>13}{'d cross (geom)':>16}{'match corr':>12}")
    for name, e in report["by_stability_quantile"].items():
        print(f"{name:<16}{e['n']:>6}{e['stability_median']:>12.3f}"
              f"{e['same_modality_distance_median']:>13.3f}"
              f"{e['cross_modal_distance_median']:>13.3f}"
              f"{e['cross_modal_distance_median_matched_by_geometry']:>16.3f}"
              f"{e['matched_correlation_median']:>12.3f}")
    ov = report["operator_vs_modality"]
    print(f"\nmatched the same way on both sides: same {ov['same_modality']:.3f} vs "
          f"cross {ov['cross_modal_matched_by_geometry']:.3f}")
    print(f"  of the gap to the reported cross-modal number, "
          f"{ov['attributable_to_operator']:+.3f} is the matching operator and "
          f"{ov['attributable_to_modality']:+.3f} is modality")
    print()
    print("decile curve (most reproducible first):")
    print("  " + "  ".join(f"{e['cross_modal_distance_median']:.2f}"
                           for e in report["by_decile"].values()))
    co_e = report["co_activating_only"]
    if co_e.get("n", 0) >= 5:
        print()
        print(f"restricted to pairs with correlation >= {args.co_activation_min} "
              f"(n={co_e['n']}):")
        print(f"  cross-modal distance median {co_e['cross_modal_distance']['median']:.3f}")
        for q in (10, 25, 50, 100):
            k = co_e.get(f"top_{q}pct_by_stability")
            if k:
                print(f"    top {q:>3}% by stability (n={k['n']:>4}): "
                      f"stability {k['stability_median']:.3f}, "
                      f"cross-modal distance {k['cross_modal_distance_median']:.3f}")
    if "stable_and_corresponding" in report:
        b = report["stable_and_corresponding"]
        print(f"\nour Hungarian pairs, filtered by both-endpoint stability and by "
              f"co-activation ({b['n_pairs_scored']} pairs scored)")
        print(f"  {'condition':<28}{'n':>7}{'cosine':>9}{'distance':>10}{'matched c':>11}")
        for k, e in b.get("by_stability_quantile_corresponding", {}).items():
            print(f"  {'corresponding, ' + k:<28}{e['n']:>7}{e['cosine_median']:>9.3f}"
                  f"{e['distance_median']:>10.3f}{e['matched_correlation_median']:>11.3f}")
        for k, e in b.get("by_stability_quantile", {}).items():
            print(f"  {k:<28}{e['n']:>7}{e['cosine_median']:>9.3f}"
                  f"{e['distance_median']:>10.3f}{e['matched_correlation_median']:>11.3f}")
        for k, e in b["grid"].items():
            if not e["n"]:
                print(f"  {k:<28}{0:>7}{'--':>9}{'--':>10}{'--':>11}")
                continue
            print(f"  {k:<28}{e['n']:>7}{e['cosine_median']:>9.3f}"
                  f"{e['distance_median']:>10.3f}{e['matched_correlation_median']:>11.3f}")
    print(f"\nwrote {out / 'stability_conditioned.json'}")


if __name__ == "__main__":
    main()
