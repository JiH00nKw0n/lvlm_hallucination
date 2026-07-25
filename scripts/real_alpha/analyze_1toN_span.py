"""Does the span of a concept's split text features explain the image direction?

One explanation for the measured cross-modal distance is feature splitting: the
text SAE may cut a concept into several latents, so no single text direction can
match the image direction even though together they cover it. If that were the
whole story, projecting the image direction onto the subspace spanned by all of
a concept's text partners would recover it.

The projection is computed on every one-to-many group — an image latent whose
co-activation correlation clears a threshold with two or more text latents — and
reported as the fraction of the image direction's energy the subspace explains.

That fraction is meaningless without a control, because a subspace of dimension
N explains roughly N/d of any direction by chance. The comparison that actually
tests the splitting hypothesis keeps the strongest partner and replaces the
others with random text atoms: it asks whether the *additional* partners
contribute more than arbitrary directions would. Two looser controls anchor the
scale — all N drawn at random from the text dictionary, and all N drawn as
random unit vectors, which should land on the analytic N/d.

    python scripts/real_alpha/analyze_1toN_span.py \
        --panel outputs/rebuttal_EA/coco_k8_r1r2/C_img_txt.npz \
        --ckpt outputs/rebuttal_models/coco_k8_r1/final \
        --tau 0.4 --out outputs/rebuttal_EE/coco_k8_r1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from collections import Counter
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from rebuttal_common import alive_masks, describe, load_panel, unit_decoder  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--alive-rule", choices=["ever", "density"], default="ever")
    p.add_argument("--tau", type=float, default=0.4,
                   help="correlation above which a text latent counts as a partner")
    p.add_argument("--n-draws", type=int, default=20, help="random draws per control")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def explained_fraction(phi: np.ndarray, Psi: np.ndarray) -> float:
    """Share of a unit direction's energy captured by the span of Psi's rows.

    QR rather than a normal-equation pseudo-inverse: the partner directions can
    be close to linearly dependent, and QR stays well behaved there.
    """
    q, _r = np.linalg.qr(Psi.T)          # (d, N) orthonormal basis of the span
    return float(np.sum((q.T @ phi) ** 2))


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

    C = panel["C"]
    text_idx = np.where(alive_t)[0]
    image_idx = np.where(alive_i)[0]
    logger.info("alive image %d, alive text %d, tau %.2f", len(image_idx), len(text_idx), args.tau)

    groups: list[tuple[int, np.ndarray]] = []
    for i in image_idx:
        partners = text_idx[C[i, text_idx] >= args.tau]
        if len(partners) >= 2:
            order = np.argsort(-C[i, partners])   # strongest partner first
            groups.append((int(i), partners[order]))
    logger.info("one-to-many groups: %d (%.1f%% of alive image latents)",
                len(groups), 100.0 * len(groups) / max(len(image_idx), 1))

    if not groups:
        (out / "one_to_many_span.json").write_text(json.dumps(
            {"tau": args.tau, "n_groups": 0,
             "note": "no image latent has two partners at this threshold"}, indent=2))
        print(f"no one-to-many groups at tau={args.tau}")
        return

    full, top1, ctrl_keep, ctrl_rand_atoms, ctrl_rand_dirs, sizes = [], [], [], [], [], []
    for i, partners in groups:
        phi = Wi[i]
        n = len(partners)
        sizes.append(n)
        full.append(explained_fraction(phi, Wt[partners]))
        top1.append(explained_fraction(phi, Wt[partners[:1]]))

        keep_draws, atom_draws, dir_draws = [], [], []
        for _ in range(args.n_draws):
            pool = np.setdiff1d(text_idx, partners, assume_unique=False)
            # the strongest partner kept, the rest replaced by random text atoms
            filler = rng.choice(pool, size=n - 1, replace=False)
            keep_draws.append(explained_fraction(
                phi, np.vstack([Wt[partners[:1]], Wt[filler]])))
            atom_draws.append(explained_fraction(
                phi, Wt[rng.choice(pool, size=n, replace=False)]))
            v = rng.standard_normal((n, d))
            dir_draws.append(explained_fraction(
                phi, v / np.linalg.norm(v, axis=1, keepdims=True)))
        ctrl_keep.append(float(np.mean(keep_draws)))
        ctrl_rand_atoms.append(float(np.mean(atom_draws)))
        ctrl_rand_dirs.append(float(np.mean(dir_draws)))

    full = np.array(full)
    top1 = np.array(top1)
    ctrl_keep = np.array(ctrl_keep)
    ctrl_rand_atoms = np.array(ctrl_rand_atoms)
    ctrl_rand_dirs = np.array(ctrl_rand_dirs)
    sizes = np.array(sizes)

    report = {
        "ckpt": args.ckpt, "panel": args.panel, "tau": args.tau,
        "alive_rule": args.alive_rule, "dim": int(d),
        "n_alive_image": int(len(image_idx)),
        "n_groups": int(len(groups)),
        "group_share_of_alive_image": float(len(groups) / len(image_idx)),
        "group_size_histogram": {str(k): int(v) for k, v in sorted(Counter(sizes.tolist()).items())},
        "explained": {
            "all_partners": describe(full),
            "strongest_partner_only": describe(top1),
            "strongest_partner_plus_random_atoms": describe(ctrl_keep),
            "random_text_atoms": describe(ctrl_rand_atoms),
            "random_unit_directions": describe(ctrl_rand_dirs),
        },
        "analytic_random_subspace": float(np.mean(sizes) / d),
        "marginal_gain_over_strongest": float(np.median(full - top1)),
        "marginal_gain_of_control": float(np.median(ctrl_keep - top1)),
        "unexplained_median": float(1.0 - np.median(full)),
        "frac_groups_explained_above_half": float(np.mean(full > 0.5)),
    }
    (out / "one_to_many_span.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"one-to-many groups at tau={args.tau}: {len(groups)} "
          f"({100 * report['group_share_of_alive_image']:.1f}% of alive image latents)")
    print(f"group sizes: {report['group_size_histogram']}")
    print()
    print("share of the image direction explained (median over groups)")
    print(f"  span of all {int(np.mean(sizes)):.0f}-ish partners       {np.median(full):.3f}")
    print(f"  strongest partner alone            {np.median(top1):.3f}")
    print(f"  strongest + random text atoms      {np.median(ctrl_keep):.3f}   <- the control")
    print(f"  random text atoms                  {np.median(ctrl_rand_atoms):.3f}")
    print(f"  random unit directions             {np.median(ctrl_rand_dirs):.3f}"
          f"   (analytic {report['analytic_random_subspace']:.3f})")
    print()
    print(f"the extra partners add {report['marginal_gain_over_strongest']:.3f} "
          f"against {report['marginal_gain_of_control']:.3f} for the control; "
          f"{100 * report['unexplained_median']:.0f}% stays unexplained")
    print(f"groups explained above 0.5: {100 * report['frac_groups_explained_above_half']:.1f}%")
    print(f"\nwrote {out / 'one_to_many_span.json'}")


if __name__ == "__main__":
    main()
