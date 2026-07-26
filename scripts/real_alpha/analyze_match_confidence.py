"""How strong, and how unambiguous, are the matches the method actually makes?

The reviewer asked for the correlation distribution of the Hungarian matches and
the percentage of low-confidence ones. "Low confidence" has no standard
definition, so this reports three readings of it and says which one carries the
argument.

Strength is the primary reading: the correlation of each matched pair, reported
as the full histogram in bands of 0.1 rather than as a handful of percentiles,
so the length of the weak tail is visible. Statistical significance is
deliberately not the headline — with millions of paired samples almost any
non-zero correlation clears a significance bar, so it would answer a question
nobody is asking.

Ambiguity is the second reading: how much the assigned partner beats the runner
up. A pair whose top two candidates are nearly tied could have been matched
elsewhere without changing the objective much, and that is a real weakness even
when the correlation is high.

Reciprocity is the third: whether the two latents are each other's first choice,
rather than one settling for the other after the assignment resolved a conflict.

A noise floor comes from a separate run of build_cross_pair_C.py with
``--shuffle-b``, which destroys the pairing between images and captions and
recomputes everything. Pass it with ``--null-panel``.

    python scripts/real_alpha/analyze_match_confidence.py \
        --panel outputs/rebuttal_EA/coco_k8_r1r2/C_img_txt.npz \
        --null-panel outputs/rebuttal_EC/coco_k8_r1/C_img_txt_shuffled.npz \
        --ckpt outputs/rebuttal_models/coco_k8_r1/final \
        --out outputs/rebuttal_EC/coco_k8_r1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rebuttal_common import (  # type: ignore  # noqa: E402
    alive_masks,
    describe,
    hungarian_perm,
    load_panel,
    matched_distance,
    unit_decoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True, help="C_img_txt.npz")
    p.add_argument("--null-panel", default=None,
                   help="the same panel rebuilt with --shuffle-b (noise floor)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--alive-rule", choices=["ever", "density"], default="ever")
    p.add_argument("--out", required=True)
    return p.parse_args()


def top_two(C: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best and second-best correlation for each row, over alive columns only."""
    sub = C[np.ix_(rows, cols)]
    part = np.partition(sub, -2, axis=1)
    return part[:, -1], part[:, -2]


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.panel)
    alive_i, alive_t = alive_masks(panel, args.alive_rule)
    C = panel["C"]
    rows_alive = np.where(alive_i)[0]
    cols_alive = np.where(alive_t)[0]
    logger.info("alive image %d, alive text %d", len(rows_alive), len(cols_alive))

    match = hungarian_perm(C, alive_i, alive_t)
    usable = np.where(match["usable"])[0]
    c_matched = match["matched_c"][usable]
    logger.info("usable matched pairs: %d", len(usable))

    Wi = unit_decoder(args.ckpt, "image")
    Wt = unit_decoder(args.ckpt, "text")
    d_matched = matched_distance(Wi, Wt, match["perm"], match["usable"])

    # --- strength -------------------------------------------------------------
    # Reported as a histogram in bands of 0.1 rather than as a set of cumulative
    # thresholds: the thresholds were arbitrary, and the bands show the whole
    # shape, including how much mass sits at the weak end.
    bands = []
    for lo in np.arange(0.9, -0.1, -0.1):
        hi = lo + 0.1
        sel = (c_matched >= lo) & (c_matched < hi) if hi < 1.0 else (c_matched >= lo)
        bands.append({
            "range": f"[{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi >= 1.0 else ')'}",
            "count": int(sel.sum()),
            "share": float(sel.mean()),
        })
    neg = c_matched < 0
    bands.append({"range": "negative", "count": int(neg.sum()), "share": float(neg.mean())})
    running = 0.0
    for b in bands:
        running += b["share"]
        b["cumulative_from_top"] = running

    # --- ambiguity ------------------------------------------------------------
    t1, t2 = top_two(C, usable, cols_alive)
    margin = t1 - t2
    ratio = t2 / np.maximum(t1, 1e-9)

    # --- reciprocity ----------------------------------------------------------
    row_best_col = cols_alive[C[np.ix_(usable, cols_alive)].argmax(axis=1)]
    partner = match["perm"][usable]
    col_best_row = rows_alive[C[np.ix_(rows_alive, partner)].argmax(axis=0)]
    mutual = (row_best_col == partner) & (col_best_row == usable)

    report = {
        "ckpt": args.ckpt, "panel": args.panel, "alive_rule": args.alive_rule,
        "n_samples": panel["n_samples"],
        "n_alive_image": int(len(rows_alive)), "n_alive_text": int(len(cols_alive)),
        "n_matched_usable": int(len(usable)),
        "matched_correlation": describe(c_matched),
        "correlation_bands": bands,
        "matched_cosine_distance": describe(d_matched),
        "ambiguity": {
            "top1_minus_top2": describe(margin),
            "top2_over_top1": describe(ratio),
            "share_runner_up_within_10pct": float(np.mean(ratio > 0.9)),
            "share_runner_up_within_50pct": float(np.mean(ratio > 0.5)),
        },
        "reciprocity": {
            "share_mutual_first_choice": float(np.mean(mutual)),
            "n_mutual": int(mutual.sum()),
        },
    }

    # --- noise floor ----------------------------------------------------------
    c_null = None
    if args.null_panel:
        null = load_panel(args.null_panel)
        na, nb = alive_masks(null, args.alive_rule)
        nm = hungarian_perm(null["C"], na, nb)
        c_null = nm["matched_c"][np.where(nm["usable"])[0]]
        floor99 = float(np.percentile(c_null, 99))
        report["noise_floor"] = {
            "panel": args.null_panel,
            "matched_correlation": describe(c_null),
            "p99": floor99,
            "share_of_real_matches_below_floor": float(np.mean(c_matched < floor99)),
        }
        logger.info("noise floor p99 = %.4f", floor99)

    (out / "match_confidence.json").write_text(json.dumps(report, indent=2))

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.9))
    ax = axes[0]
    bins = np.linspace(-0.1, 1.0, 90)
    ax.hist(c_matched, bins=bins, color="#206987", alpha=0.75, label="matched pairs")
    if c_null is not None:
        ax.hist(c_null, bins=bins, color="#df3a3d", alpha=0.55, label="pairing destroyed")
    ax.set_yscale("log")
    ax.set_xlabel("co-activation correlation of the matched pair", fontsize=7.5, labelpad=1)
    ax.set_ylabel("count", fontsize=7.5, labelpad=2)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    order = np.sort(c_matched)
    ax.plot(order, np.arange(1, len(order) + 1) / len(order), color="#206987", lw=1.0)
    for t in (0.1, 0.2, 0.4):
        ax.axvline(t, color="0.6", ls=":", lw=0.7)
        ax.text(t, 0.04, f"{np.mean(c_matched < t) * 100:.0f}%", fontsize=6,
                ha="right", va="bottom", color="0.35", rotation=90)
    ax.set_xlabel("correlation", fontsize=7.5, labelpad=1)
    ax.set_ylabel("share of matches below", fontsize=7.5, labelpad=2)
    ax.set_ylim(0, 1)

    for a in axes:
        a.tick_params(labelsize=6.5, pad=1)
        a.grid(axis="y", alpha=0.15, linewidth=0.4)
    fig.tight_layout()
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(str(out / f"match_confidence{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)

    # --- console --------------------------------------------------------------
    mc = report["matched_correlation"]
    print()
    print(f"matched pairs: {len(usable)}  "
          f"(alive {len(rows_alive)} image / {len(cols_alive)} text)")
    print(f"correlation   median {mc['median']:.3f}   "
          f"quartiles [{mc['p25']:.3f}, {mc['p75']:.3f}]   "
          f"5-95% [{mc['p05']:.3f}, {mc['p95']:.3f}]")
    print(f"{'correlation':<16}{'count':>8}{'share':>9}{'cumulative':>12}")
    for b in report["correlation_bands"]:
        print(f"{b['range']:<16}{b['count']:>8}{100 * b['share']:>8.1f}%"
              f"{100 * b['cumulative_from_top']:>11.1f}%")
    amb = report["ambiguity"]
    print(f"runner-up within 10% of the winner: "
          f"{100 * amb['share_runner_up_within_10pct']:.1f}%   "
          f"within 50%: {100 * amb['share_runner_up_within_50pct']:.1f}%")
    print(f"mutual first choice: {100 * report['reciprocity']['share_mutual_first_choice']:.1f}%")
    if c_null is not None:
        nf = report["noise_floor"]
        print(f"noise floor (pairing destroyed) p99 = {nf['p99']:.4f}; "
              f"{100 * nf['share_of_real_matches_below_floor']:.1f}% of real matches fall below it")
    print(f"\nwrote {out / 'match_confidence.json'}")


if __name__ == "__main__":
    main()
