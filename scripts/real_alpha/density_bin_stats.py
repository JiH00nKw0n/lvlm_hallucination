#!/usr/bin/env python3
"""Per-correlation-bin statistics behind the multi-density figure.

Venues that forbid figures in a rebuttal still allow tables, so this prints the
numbers the density plot draws: for each co-activation bin, the distribution of
matched decoder cosine distance.

It imports ``load_model_data`` from ``plot_multi_model_density`` rather than
recomputing anything, so the table and the figure cannot disagree — same alive
rule, same Hungarian matching, same bin edges. The inputs are the artifacts
Stage 3 already wrote, so no GPU and no re-extraction is needed:

    <run_dir>/diagnostic_B_C_train.npy
    <run_dir>/model.safetensors

Usage (single run):

    python scripts/real_alpha/density_bin_stats.py \\
        --run-dir outputs/real_exp_llava_coco/separated/ckpt/final \\
        --name "LLaVA-1.5-7B" \\
        --out outputs/real_exp_llava_coco/density_bin_stats

Usage (the same JSON the plot takes, for several models at once):

    python scripts/real_alpha/density_bin_stats.py \\
        --models outputs/real_exp_llava_coco/density_models.json \\
        --out outputs/real_exp_llava_coco/density_bin_stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_multi_model_density import BIN_EDGES, load_model_data  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None,
                   help="checkpoint dir holding diagnostic_B_C_train.npy and model.safetensors")
    p.add_argument("--name", default="model", help="label for the --run-dir model")
    p.add_argument("--models", default=None,
                   help="JSON list of {name, run_dir}; the same file the plot takes")
    p.add_argument("--out", default="outputs/density_bin_stats",
                   help="output prefix; writes <out>.md and <out>.json")
    return p.parse_args()


def describe(v: np.ndarray) -> dict:
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
    }


def stats_for(run_dir: str) -> dict:
    c_matched, dist_matched, n_alive_i, n_alive_t = load_model_data(run_dir)
    total = int(c_matched.size)
    bins = []
    for b in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        # The top bin is closed so a perfectly correlated pair is not dropped.
        mask = (c_matched >= lo) & (c_matched < hi) if b < len(BIN_EDGES) - 2 \
            else (c_matched >= lo) & (c_matched <= hi)
        n = int(mask.sum())
        row = {"bin": f"[{lo:.1f}, {hi:.1f})" if b < len(BIN_EDGES) - 2
                      else f"[{lo:.1f}, {hi:.1f}]",
               "lo": lo, "hi": hi, "n": n,
               "share_of_matched": n / total if total else 0.0}
        if n:
            row["cosine_distance"] = describe(dist_matched[mask])
            row["cosine"] = describe(1.0 - dist_matched[mask])
            row["correlation"] = describe(c_matched[mask])
        bins.append(row)

    # Pairs with negative correlation fall outside every bin; say so rather than
    # letting the bin counts silently fail to sum to the total.
    neg = int((c_matched < BIN_EDGES[0]).sum())
    # load_model_data uses the recorded firing rates when Stage 3 left them and a
    # nonzero-variance proxy otherwise. Which one ran changes the alive set, so
    # record it instead of leaving the reader to guess.
    has_rates = (Path(run_dir) / "diagnostic_B_firing_rates.npz").exists()
    return {
        "run_dir": run_dir,
        "alive_rule": ("firing rate > 0.001 (diagnostic_B_firing_rates.npz)" if has_rates
                       else "nonzero-variance proxy (no firing-rate file present)"),
        "n_alive_image": int(n_alive_i),
        "n_alive_text": int(n_alive_t),
        "n_matched_pairs": total,
        "n_negative_correlation_excluded": neg,
        "quantity": "cosine distance 1 - cos(phi_i, psi_j) between matched decoder rows",
        "bins": bins,
        "all_matched": {"cosine_distance": describe(dist_matched),
                        "cosine": describe(1.0 - dist_matched),
                        "correlation": describe(c_matched)},
    }


def to_markdown(name: str, s: dict) -> list[str]:
    L = [f"### {name}", ""]
    L.append(f"Alive latents: image {s['n_alive_image']}, text {s['n_alive_text']} "
             f"(alive rule: {s['alive_rule']}). "
             f"Hungarian-matched pairs: {s['n_matched_pairs']}.")
    if s["n_negative_correlation_excluded"]:
        L.append(f"{s['n_negative_correlation_excluded']} matched pairs have a negative "
                 "correlation and fall below the lowest bin; they are excluded from the "
                 "rows below and included in *all matched pairs*.")
    L.append("")
    L.append("Cosine distance `1 - cos(phi_i, psi_j)` of matched decoder directions, "
             "by co-activation correlation bin. This is the quantity the density "
             "figure plots on its x-axis.")
    L.append("")
    L.append("| correlation bin | pairs | share | mean | median | std | min | max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for b in s["bins"]:
        if not b["n"]:
            L.append(f"| {b['bin']} | 0 | 0.0% | — | — | — | — | — |")
            continue
        d = b["cosine_distance"]
        L.append(f"| {b['bin']} | {b['n']} | {100 * b['share_of_matched']:.1f}% | "
                 f"{d['mean']:.4f} | {d['median']:.4f} | {d['std']:.4f} | "
                 f"{d['min']:.4f} | {d['max']:.4f} |")
    a = s["all_matched"]["cosine_distance"]
    L.append(f"| **all matched pairs** | {s['n_matched_pairs']} | 100.0% | "
             f"{a['mean']:.4f} | {a['median']:.4f} | {a['std']:.4f} | "
             f"{a['min']:.4f} | {a['max']:.4f} |")
    L.append("")
    L.append("Same pairs, reported as cosine rather than cosine distance:")
    L.append("")
    L.append("| correlation bin | pairs | mean | median | std | min | max |")
    L.append("|---|---|---|---|---|---|---|")
    for b in s["bins"]:
        if not b["n"]:
            L.append(f"| {b['bin']} | 0 | — | — | — | — | — |")
            continue
        c = b["cosine"]
        L.append(f"| {b['bin']} | {b['n']} | {c['mean']:.4f} | {c['median']:.4f} | "
                 f"{c['std']:.4f} | {c['min']:.4f} | {c['max']:.4f} |")
    c = s["all_matched"]["cosine"]
    L.append(f"| **all matched pairs** | {s['n_matched_pairs']} | {c['mean']:.4f} | "
             f"{c['median']:.4f} | {c['std']:.4f} | {c['min']:.4f} | {c['max']:.4f} |")
    L.append("")
    return L


def main() -> None:
    args = parse_args()
    if args.models:
        models = json.load(open(args.models))
    elif args.run_dir:
        models = [{"name": args.name, "run_dir": args.run_dir}]
    else:
        raise SystemExit("pass --run-dir or --models")

    payload, lines = {}, ["# Multi-density figure — per-bin statistics", ""]
    lines.append("Generated by `scripts/real_alpha/density_bin_stats.py` from the "
                 "cached Stage 3 artifacts. No GPU, no re-extraction.")
    lines.append("")
    for m in models:
        s = stats_for(m["run_dir"])
        payload[m["name"]] = s
        lines.extend(to_markdown(m["name"], s))
        print(f"{m['name']}: matched={s['n_matched_pairs']}, "
              f"alive_i={s['n_alive_image']}, alive_t={s['n_alive_text']}")
        for b in s["bins"]:
            if b["n"]:
                d = b["cosine_distance"]
                print(f"  {b['bin']}  n={b['n']:6d}  mean={d['mean']:.4f}  "
                      f"median={d['median']:.4f}  std={d['std']:.4f}  "
                      f"min={d['min']:.4f}  max={d['max']:.4f}")
            else:
                print(f"  {b['bin']}  n=0")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out.with_suffix('.md')}")
    print(f"wrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
