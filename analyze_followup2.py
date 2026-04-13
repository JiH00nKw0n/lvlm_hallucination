#!/usr/bin/env python3
"""Analyze theorem2_followup_2 results: 8 combinations × 2 k × 6 methods.

Usage:
    python analyze_followup2.py [--alpha 0.7]

Prints markdown tables of recon / mgt_S.99 / mcc_S / uniq_S per combination per k.
"""
import argparse
import json
import os
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "outputs" / "theorem2_followup_2" / "runs"

COMBOS = [
    ("B",   (4.5, 0.5, 0.1, 0.0)),
    ("C",   (2.0, 1.0, 0.1, 0.0)),
    ("I",   (4.5, 0.5, 0.2, 0.0)),
    ("N",   (4.5, 0.5, 0.1, 0.5)),
    ("CI",  (2.0, 1.0, 0.2, 0.0)),
    ("CN",  (2.0, 1.0, 0.1, 0.5)),
    ("IN",  (4.5, 0.5, 0.2, 0.5)),
    ("CIN", (2.0, 1.0, 0.2, 0.5)),
]

METHOD_ORDER = [
    ("1R", "single_recon"),
    ("2R", "two_recon"),
    ("GS", "group_sparse"),
    ("TA", "trace_align"),
    ("IA", "iso_align"),
    ("oB", "ours::"),
]


def find_ours_key(method_ids):
    for m in method_ids:
        if m.startswith("ours::"):
            return m
    return None


def load_metrics(run_dir: Path, alpha: float) -> dict:
    rp = run_dir / "result.json"
    if not rp.exists():
        return {}
    with open(rp) as f:
        d = json.load(f)
    for entry in d.get("sweep_results", []):
        if abs(entry["alpha_target"] - alpha) < 1e-6:
            agg = entry["aggregate"]
            ours = find_ours_key(entry["method_ids"])
            out = {}
            for m in entry["method_ids"]:
                recon = agg.get(f"{m}/avg_eval_loss_mean")
                def imt(k):
                    i = agg.get(f"{m}/img_{k}_mean")
                    t = agg.get(f"{m}/txt_{k}_mean")
                    return (i + t) / 2 if i is not None and t is not None else None
                out[m] = {
                    "recon": recon,
                    "mgt_S.99": imt("mgt_shared_tau0.99"),
                    "mcc_S": imt("mcc_shared"),
                    "uniq_S": imt("uniqueness_shared_norm"),
                }
            out["_ours_key"] = ours
            return out
    return {}


def find_run_dir(k: int, tag: str) -> Path | None:
    needle = f"followup2_k{k}_{tag}_"
    for d in sorted(RUNS_DIR.glob(f"*{needle}*")):
        if (d / "result.json").exists():
            return d
    return None


def fmt(v):
    return f"{v:.3f}" if v is not None else "—"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    if not RUNS_DIR.exists():
        print(f"runs dir not found: {RUNS_DIR}")
        return

    for k in [16, 8]:
        print(f"\n## k={k}, α={args.alpha}\n")
        header = "| combo | (μ, σ, int, noise) | method | recon | mgt_S.99 | mcc_S | uniq_S |"
        sep = "|---|---|---|---|---|---|---|"
        print(header)
        print(sep)
        for tag, params in COMBOS:
            run_dir = find_run_dir(k, tag)
            if run_dir is None:
                print(f"| {tag} | {params} | (missing) | — | — | — | — |")
                continue
            metrics = load_metrics(run_dir, args.alpha)
            if not metrics:
                continue
            ours_k = metrics.get("_ours_key")
            for short, mk in METHOD_ORDER:
                if short == "oB":
                    m = metrics.get(ours_k, {}) if ours_k else {}
                else:
                    m = metrics.get(mk, {})
                row = [
                    tag if short == "1R" else "",
                    f"{params}" if short == "1R" else "",
                    short,
                    fmt(m.get("recon")),
                    fmt(m.get("mgt_S.99")),
                    fmt(m.get("mcc_S")),
                    fmt(m.get("uniq_S")),
                ]
                print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
