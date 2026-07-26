"""Is the 1:N structure feature splitting, or genuinely many concepts?

Reviewer PBPC's third weakness asks how a bijection copes when the true
correspondence is one-to-many. The answer has two parts, and this script
measures both on one checkpoint so they rest on the same basis.

*How much 1:N is there.* For every alive image latent, count the alive text
latents whose co-activation correlation clears a threshold. Two or more is a
one-to-many group. The threshold has no principled value, so it is swept and the
whole curve is reported rather than one flattering point.

*What the N partners are.* If a group's partners were distinct concepts, they
would fire on different inputs. If instead one concept has been split across
several text coordinates -- the same idea phrased differently, or one aspect of
it -- the partners fire on largely the same inputs. Two measurements separate
those: the Jaccard overlap between the partners' firing sets, and how much of the
group's co-firing the single strongest partner already covers.

Jaccard needs a scale to mean anything, so random pairs of alive text latents are
measured the same way. Coverage needs an honest denominator, so both are
reported: the strongest partner's share of the samples where the image latent
co-fires with *any* partner, and how much of the image latent's own firing that
co-firing set accounts for in the first place. Quoting only the first number
overstates the case, because the co-firing set is a fraction of the whole.

    python scripts/real_alpha/analyze_1toN_splitting.py \
        --ckpt outputs/rebuttal_models/cc3m_k32_r1/final \
        --panel outputs/rebuttal_EA/cc3m_k32_r1r2/C_img_txt.npz \
        --dataset cc3m --cache-dir cache/clip_b32_cc3m --max-samples 500000 \
        --out outputs/rebuttal_EH/cc3m_k32_r1
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
from build_cross_pair_C import pair_rows  # type: ignore  # noqa: E402
from eval_coco80_correspondence import pick_device, sparse_latents  # type: ignore  # noqa: E402
from rebuttal_common import alive_masks, load_panel  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--dataset", choices=["coco", "cc3m"], default="cc3m")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=500000,
                   help="must match the panel, or the samples do not line up")
    p.add_argument("--alive-rule", choices=["ever", "density"], default="ever")
    p.add_argument("--tau-sweep", default="0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--tau", type=float, default=0.4,
                   help="threshold for the detailed splitting analysis")
    p.add_argument("--n-random-pairs", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def groups_at(C: np.ndarray, alive_i: np.ndarray, alive_t: np.ndarray,
              tau: float) -> dict[int, np.ndarray]:
    """Image latent -> its alive text partners above tau, strongest first."""
    out: dict[int, np.ndarray] = {}
    tcols = np.where(alive_t)[0]
    for i in np.where(alive_i)[0]:
        row = C[i, tcols]
        hit = tcols[row >= tau]
        if hit.size >= 2:
            out[int(i)] = hit[np.argsort(-C[i, hit])]
    return out


def per_latent_sets(lat: np.ndarray, samp: np.ndarray,
                    n_latents: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample indices grouped by latent: a flat array plus its boundaries."""
    order = np.argsort(lat, kind="stable")
    lat_s, samp_s = lat[order], samp[order]
    bounds = np.searchsorted(lat_s, np.arange(n_latents + 1))
    return samp_s, bounds


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.panel)
    C = panel["C"].astype(np.float32)
    alive_i, alive_t = alive_masks(panel, args.alive_rule)
    n_alive_i = int(alive_i.sum())
    logger.info("alive: image %d, text %d, panel samples %d",
                n_alive_i, int(alive_t.sum()), panel["n_samples"])

    # ---- how much 1:N, as a function of the threshold ------------------------
    sweep = []
    for tau in [float(x) for x in args.tau_sweep.split(",")]:
        g = groups_at(C, alive_i, alive_t, tau)
        sizes = np.array([len(v) for v in g.values()]) if g else np.array([], dtype=int)
        sweep.append({
            "tau": tau,
            "n_groups": len(g),
            "share_of_alive_image": len(g) / max(n_alive_i, 1),
            "mean_group_size": float(sizes.mean()) if sizes.size else 0.0,
            "max_group_size": int(sizes.max()) if sizes.size else 0,
        })
        logger.info("tau=%.2f -> %d groups (%.1f%% of alive image latents)",
                    tau, len(g), 100 * len(g) / max(n_alive_i, 1))

    groups = groups_at(C, alive_i, alive_t, args.tau)
    if not groups:
        raise SystemExit(f"no 1:N group at tau={args.tau}")

    # ---- encode the same rows the panel used ---------------------------------
    ds = eval_utils.load_pair_dataset(args.cache_dir, args.dataset, split=args.split)
    img_rows, txt_rows = pair_rows(ds)
    n_total = len(img_rows)
    if args.max_samples and args.max_samples < n_total:
        # identical rule to build_cross_pair_C, so sample i here is sample i there
        sel = np.linspace(0, n_total - 1, args.max_samples, dtype=np.int64)
        img_rows, txt_rows = img_rows[sel], txt_rows[sel]
    if len(img_rows) != panel["n_samples"]:
        raise SystemExit(f"sample count {len(img_rows)} != panel's {panel['n_samples']}; "
                         "pass the --max-samples the panel was built with")

    model = eval_utils.load_sae(args.ckpt, "separated")
    n_latents = int(model.image_sae.latent_size)
    logger.info("encoding %d image rows", len(img_rows))
    i_samp, i_lat, _ = sparse_latents(model.image_sae, ds._image_table, img_rows,
                                      args.batch_size, device)
    logger.info("encoding %d text rows", len(txt_rows))
    t_samp, t_lat, _ = sparse_latents(model.text_sae, ds._text_table, txt_rows,
                                      args.batch_size, device)

    i_flat, i_bnd = per_latent_sets(i_lat, i_samp, n_latents)
    t_flat, t_bnd = per_latent_sets(t_lat, t_samp, n_latents)

    def fires_img(l: int) -> np.ndarray:
        return np.unique(i_flat[i_bnd[l]:i_bnd[l + 1]])

    def fires_txt(l: int) -> np.ndarray:
        return np.unique(t_flat[t_bnd[l]:t_bnd[l + 1]])

    # ---- do the N partners fire on the same inputs? --------------------------
    jac_within, cover, cofire_share, rows = [], [], [], []
    for i, partners in groups.items():
        S = fires_img(i)
        if S.size == 0:
            continue
        sets = [fires_txt(int(j)) for j in partners]
        js = []
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                inter = np.intersect1d(sets[a], sets[b], assume_unique=True).size
                uni = sets[a].size + sets[b].size - inter
                if uni:
                    js.append(inter / uni)
        union = sets[0]
        for s in sets[1:]:
            union = np.union1d(union, s)
        co_any = np.intersect1d(S, union, assume_unique=True)
        co_top = np.intersect1d(S, sets[0], assume_unique=True)
        if co_any.size == 0:
            continue
        rec = {
            "image_latent": int(i),
            "n_partners": int(len(partners)),
            "jaccard_median": float(np.median(js)) if js else None,
            "strongest_share_of_cofiring": float(co_top.size / co_any.size),
            "cofiring_share_of_image_firing": float(co_any.size / S.size),
        }
        rows.append(rec)
        if js:
            jac_within.extend(js)
        cover.append(rec["strongest_share_of_cofiring"])
        cofire_share.append(rec["cofiring_share_of_image_firing"])

    # ---- scale for Jaccard: unrelated pairs of alive text latents ------------
    tcand = np.where(alive_t)[0]
    jac_random = []
    for _ in range(args.n_random_pairs):
        a, b = rng.choice(tcand, size=2, replace=False)
        A, B = fires_txt(int(a)), fires_txt(int(b))
        inter = np.intersect1d(A, B, assume_unique=True).size
        uni = A.size + B.size - inter
        if uni:
            jac_random.append(inter / uni)

    def summ(v) -> dict:
        a = np.asarray(v, dtype=np.float64)
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "p05": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
            "share_below_0.1": float(np.mean(a < 0.1)),
        }

    payload = {
        "ckpt": args.ckpt,
        "panel": args.panel,
        "dataset": args.dataset,
        "n_samples": int(panel["n_samples"]),
        "alive_rule": args.alive_rule,
        "n_alive_image": n_alive_i,
        "n_alive_text": int(alive_t.sum()),
        "tau_sweep": sweep,
        "tau": args.tau,
        "n_groups": len(groups),
        "group_size_histogram": {str(k): int(v) for k, v in
                                 zip(*np.unique([len(v) for v in groups.values()],
                                                return_counts=True))},
        "jaccard_within_group": summ(jac_within),
        "jaccard_random_pairs": summ(jac_random),
        "strongest_share_of_cofiring": summ(cover),
        "cofiring_share_of_image_firing": summ(cofire_share),
        "per_group": rows,
    }
    (out / "one_to_many_splitting.json").write_text(json.dumps(payload, indent=1))

    print(f"\n{args.dataset}, {panel['n_samples']} samples, alive image {n_alive_i}")
    print("\n  1:N share as the threshold moves")
    for s in sweep:
        print(f"    tau >= {s['tau']:.2f}   {s['n_groups']:5d} groups   "
              f"{100 * s['share_of_alive_image']:5.1f}% of alive image latents   "
              f"mean size {s['mean_group_size']:.2f}   max {s['max_group_size']}")
    jw, jr = payload["jaccard_within_group"], payload["jaccard_random_pairs"]
    cs, cf = payload["strongest_share_of_cofiring"], payload["cofiring_share_of_image_firing"]
    print(f"\n  at tau = {args.tau}: {len(groups)} groups")
    print(f"    Jaccard, partners in a group      median {jw['median']:.3f}   "
          f"below 0.1: {100 * jw['share_below_0.1']:.1f}%   n={jw['n']}")
    print(f"    Jaccard, random text latent pairs  median {jr['median']:.3f}   "
          f"below 0.1: {100 * jr['share_below_0.1']:.1f}%   n={jr['n']}")
    print(f"    strongest partner covers           median {100 * cs['median']:.1f}% "
          f"of the group's co-firing")
    print(f"    that co-firing is                  median {100 * cf['median']:.1f}% "
          f"of the image latent's own firing")
    print(f"\nwrote {out / 'one_to_many_splitting.json'}")


if __name__ == "__main__":
    main()
