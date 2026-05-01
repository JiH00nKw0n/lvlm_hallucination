"""Recompute ms_summary.json from existing ms_per_latent.csv with new τ thresholds.

Avoids re-encoding (the costly stage). Only re-aggregates the saved per-latent
table. Used after the user requested adding τ=10 (and dropping 200, since each
ImageNet val class only has 50 images).

Usage:
    python scripts/real_alpha/recompute_ms_summary.py \
        --root outputs/real_exp_cc3m/monosemanticity \
        --variants shared separated iso_align group_sparse \
        --thresholds 10 50
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--thresholds", nargs="+", type=int, default=[10, 50])
    return p.parse_args()


def _agg(ms: np.ndarray, fire: np.ndarray, mask: np.ndarray) -> dict:
    sel_ms = ms[mask]
    valid = ~np.isnan(sel_ms)
    sel_ms = sel_ms[valid]
    sel_fire = fire[mask][valid]
    if len(sel_ms) == 0:
        return {"n": 0, "mean": None, "median": None, "fire_min": None, "fire_max": None}
    return {
        "n": int(len(sel_ms)),
        "mean": float(np.mean(sel_ms)),
        "median": float(np.median(sel_ms)),
        "fire_min": int(sel_fire.min()) if len(sel_fire) else None,
        "fire_max": int(sel_fire.max()) if len(sel_fire) else None,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    for v in args.variants:
        csv_path = root / v / "ms_per_latent.csv"
        if not csv_path.exists():
            print(f"[skip] {v}: no ms_per_latent.csv")
            continue

        rows = list(csv.DictReader(open(csv_path)))
        ms_arr = np.array([float(r["ms_score"]) if r["ms_score"] != "" else np.nan
                           for r in rows], dtype=np.float64)
        fire_arr = np.array([int(r["fire_count"]) for r in rows], dtype=np.int64)

        n_alive = int((fire_arr >= 1).sum())
        out = {
            "variant": v,
            "n_total": int(len(rows)),
            "n_alive": n_alive,
            "alive_ge_1": _agg(ms_arr, fire_arr, fire_arr >= 1),
        }
        for t in args.thresholds:
            out[f"well_supp_{t}"] = _agg(ms_arr, fire_arr, fire_arr >= t)
        # sorted curve over alive
        sorted_curve = sorted([float(x) for x in ms_arr if not np.isnan(x)], reverse=True)
        out["sorted_curve"] = sorted_curve

        with open(root / v / "ms_summary.json", "w") as f:
            json.dump(out, f)

        print(f"[done] {v}: alive={n_alive} ms_alive_mean={out['alive_ge_1']['mean']:.4f}")
        for t in args.thresholds:
            agg = out[f"well_supp_{t}"]
            print(f"        well_supp≥{t}: n={agg['n']} mean={agg['mean']:.4f} median={agg['median']:.4f}"
                  if agg["mean"] is not None
                  else f"        well_supp≥{t}: n=0 (no latents fire ≥ {t})")


if __name__ == "__main__":
    main()
