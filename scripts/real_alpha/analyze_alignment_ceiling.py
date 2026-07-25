"""How close could image and text feature directions possibly be brought?

Two objections stand between the measured cosine distance and the paper's
conclusion, and both are about whether the distance is a property of the
dictionaries or an artifact of how we compare them.

The first is that the matching might simply be bad: perhaps a better assignment
than the Hungarian one would pair each image direction with a text direction it
actually agrees with. The test discards matching entirely and asks an oracle for
the single closest text direction anywhere in the dictionary. If even that best
case sits far from zero distance, no assignment procedure — Hungarian, Sinkhorn,
CCA, optimal transport — can close the gap. The comparison is against the same
search run over random directions, which is the distance the oracle would report
by chance from taking a maximum over thousands of candidates.

The second is that the two dictionaries might agree up to a single global
transform, in which case the per-concept distance would be a coordinate artifact
rather than evidence about individual concepts. The test fits one transform on
half the matched pairs and evaluates it on the held-out half, both as a rotation
(orthogonal Procrustes) and as an unconstrained linear map. A transform fitted
on a random re-pairing gives the floor, which shows the evaluation cannot be
gamed by the extra parameters.

    python scripts/real_alpha/analyze_alignment_ceiling.py \
        --panel outputs/rebuttal_EA/coco_k8_r1r2/C_img_txt.npz \
        --ckpt outputs/rebuttal_models/coco_k8_r1/final \
        --out outputs/rebuttal_EF/coco_k8_r1
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
    p.add_argument("--panel", required=True, help="C_img_txt.npz from build_cross_pair_C.py")
    p.add_argument("--ckpt", required=True, help="the run that panel came from")
    p.add_argument("--alive-rule", choices=["ever", "density"], default="ever")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def random_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def orthogonal_map(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rotation carrying X onto Y as closely as possible (Procrustes)."""
    u, _s, vt = np.linalg.svd(X.T @ Y)
    return u @ vt


def linear_map(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Unconstrained least-squares map, lightly regularized."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ Y)


def applied_cosine(X: np.ndarray, Y: np.ndarray, M: np.ndarray | None) -> np.ndarray:
    Z = X if M is None else X @ M
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return (Z * Y).sum(axis=1)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.panel)
    alive_i, alive_t = alive_masks(panel, args.alive_rule)
    Wi = unit_decoder(args.ckpt, "image")
    Wt = unit_decoder(args.ckpt, "text")
    d = Wi.shape[1]
    logger.info("alive image %d, alive text %d, dim %d",
                int(alive_i.sum()), int(alive_t.sum()), d)

    match = hungarian_perm(panel["C"], alive_i, alive_t)
    d_matched = matched_distance(Wi, Wt, match["perm"], match["usable"])
    logger.info("matched pairs used: %d", match["n_usable"])

    # ---- 1. oracle: best possible partner, matching ignored -------------------
    rows = np.where(alive_i)[0]
    cols = np.where(alive_t)[0]
    cos_all = Wi[rows] @ Wt[cols].T
    oracle_cos = cos_all.max(axis=1)

    # Null: the same search against random directions, same count of candidates.
    rand_t = random_unit(len(cols), d, rng)
    oracle_cos_null = (Wi[rows] @ rand_t.T).max(axis=1)
    analytic_null = float(np.sqrt(2.0 * np.log(len(cols)) / d))

    # ---- 2. does one global transform explain the gap? ------------------------
    use = np.where(match["usable"])[0]
    X = Wi[use]
    Y = Wt[match["perm"][use]]
    order = rng.permutation(len(use))
    half = len(use) // 2
    tr, te = order[:half], order[half:]

    R = orthogonal_map(X[tr], Y[tr])
    A = linear_map(X[tr], Y[tr])
    shuffled = rng.permutation(len(use))
    R_null = orthogonal_map(X[tr], Y[shuffled[tr]])

    transform = {
        "n_pairs": int(len(use)),
        "n_fit": int(len(tr)),
        "n_eval": int(len(te)),
        "identity_cos": float(np.mean(applied_cosine(X[te], Y[te], None))),
        "rotation_cos": float(np.mean(applied_cosine(X[te], Y[te], R))),
        "linear_cos": float(np.mean(applied_cosine(X[te], Y[te], A))),
        "rotation_on_shuffled_pairs_cos": float(np.mean(applied_cosine(X[te], Y[te], R_null))),
    }

    report = {
        "ckpt": args.ckpt,
        "panel": args.panel,
        "alive_rule": args.alive_rule,
        "dim": int(d),
        "n_alive_image": int(alive_i.sum()),
        "n_alive_text": int(alive_t.sum()),
        "matched_distance": describe(d_matched),
        "oracle_cosine": describe(oracle_cos),
        "oracle_distance": describe(1.0 - oracle_cos),
        "oracle_cosine_against_random_directions": describe(oracle_cos_null),
        "oracle_chance_analytic": analytic_null,
        "global_transform": transform,
    }
    (out / "alignment_ceiling.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"alive: {int(alive_i.sum())} image, {int(alive_t.sum())} text  (dim {d})")
    print(f"matched pairs                          n={match['n_usable']}")
    print(f"  cosine distance, median              {report['matched_distance']['median']:.3f}")
    print()
    print("best possible partner, matching ignored")
    print(f"  cosine to closest text direction     {report['oracle_cosine']['median']:.3f}"
          f"   (distance {report['oracle_distance']['median']:.3f})")
    print(f"  same search over random directions   {report['oracle_cosine_against_random_directions']['median']:.3f}"
          f"   (analytic {analytic_null:.3f})")
    print()
    print("one global transform, fitted on half the pairs, scored on the other half")
    print(f"  identity                             {transform['identity_cos']:.3f}")
    print(f"  best rotation                        {transform['rotation_cos']:.3f}")
    print(f"  best linear map                      {transform['linear_cos']:.3f}")
    print(f"  rotation fitted on shuffled pairs    {transform['rotation_on_shuffled_pairs_cos']:.3f}")
    print()
    print(f"wrote {out / 'alignment_ceiling.json'}")


if __name__ == "__main__":
    main()
