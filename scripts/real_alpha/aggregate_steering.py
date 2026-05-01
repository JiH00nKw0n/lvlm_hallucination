"""Aggregate per-seed cross-modal steering summaries into mean ± std.

For each variant + α + metric, computes mean and std across seeds. Produces:

  <out>/<variant>/summary.json    same schema as eval_cross_modal_steering's
                                   summary.json, but each per_alpha[<a>][<key>]
                                   becomes {"mean": ..., "std": ..., "n": ...}.

Usage:
  python scripts/real_alpha/aggregate_steering.py \\
    --roots outputs/real_exp_cc3m_s0/cross_modal_steering,\
outputs/real_exp_cc3m_s1/cross_modal_steering,\
outputs/real_exp_cc3m_s2/cross_modal_steering \\
    --out outputs/real_exp_cc3m_mean/cross_modal_steering
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _agg(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def aggregate_variant(roots: list[Path], variant: str) -> dict | None:
    summaries = []
    for r in roots:
        p = r / variant / "summary.json"
        if p.exists():
            summaries.append(json.load(open(p)))
    if not summaries:
        return None

    base = summaries[0]
    out = {
        "variant": variant,
        "method": base.get("method"),
        "split": base.get("split"),
        "alphas": base.get("alphas"),
        "n_seeds": len(summaries),
        "per_alpha": {},
    }
    # Union of alpha keys (must be identical across seeds)
    alpha_keys = list(base["per_alpha"].keys())
    for a in alpha_keys:
        agg_for_a: dict = {}
        # Numeric keys: mean+std. Skip non-numeric.
        sample = base["per_alpha"][a]
        for k, v in sample.items():
            if not isinstance(v, (int, float)):
                continue
            vals = [s["per_alpha"][a][k] for s in summaries
                    if a in s["per_alpha"] and k in s["per_alpha"][a]]
            if vals:
                agg_for_a[k] = _agg(vals)
        out["per_alpha"][a] = agg_for_a
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True,
                    help="Comma-separated list of cross_modal_steering dirs.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="shared,separated,iso_align,group_sparse,ours")
    args = ap.parse_args()

    roots = [Path(p) for p in args.roots.split(",")]
    for r in roots:
        if not r.exists():
            raise SystemExit(f"missing root: {r}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for v in args.variants.split(","):
        agg = aggregate_variant(roots, v)
        if agg is None:
            print(f"  SKIP {v}: no summaries found")
            continue
        out_v = out / v
        out_v.mkdir(parents=True, exist_ok=True)
        with open(out_v / "summary.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(f"  wrote {out_v}/summary.json (N={agg['n_seeds']} seeds, {len(agg['per_alpha'])} alphas)")


if __name__ == "__main__":
    main()
