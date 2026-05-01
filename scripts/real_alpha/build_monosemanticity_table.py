"""Aggregate per-variant monosemanticity summaries into a cross-variant table.

Reads {root}/{variant}/{ms,fms,ccs}_summary.json and writes Markdown +
echo to stdout.

Usage:
    python scripts/real_alpha/build_monosemanticity_table.py \
        --root outputs/real_exp_cc3m/monosemanticity \
        --variants shared separated iso_align group_sparse \
        --out outputs/real_exp_cc3m/monosemanticity/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(x, prec: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    # Discover all well_supp_X thresholds used across variants (union, sorted asc).
    threshold_set = set()
    for v in args.variants:
        ms = _load(root / v / "ms_summary.json") or {}
        for k in ms.keys():
            if k.startswith("well_supp_"):
                try:
                    threshold_set.add(int(k.removeprefix("well_supp_")))
                except ValueError:
                    pass
    thresholds = sorted(threshold_set)

    rows = []
    for v in args.variants:
        vdir = root / v
        meta = _load(vdir / "meta.json") or {}
        ms = _load(vdir / "ms_summary.json") or {}
        fms = _load(vdir / "fms_summary.json") or {}
        ccs = _load(vdir / "ccs_summary.json") or {}
        fc = _load(vdir / "fire_counts.json") or {}
        row = {
            "variant": v,
            "L": meta.get("L"),
            "alive_ge_1": fc.get("A_1"),
            "ms_alive_mean": (ms.get("alive_ge_1") or {}).get("mean"),
            "ms_alive_med":  (ms.get("alive_ge_1") or {}).get("median"),
            "ms_alive_n":    (ms.get("alive_ge_1") or {}).get("n"),
            "fms_acc0":  fms.get("acc_0_mean"),
            "fms_at_1":  fms.get("fms_at_1"),
            "fms_at_5":  fms.get("fms_at_5"),
            "ccs_top1":  ccs.get("ccs_top1_mean"),
            "ccs_h":     ccs.get("ccs_entropy_mean"),
        }
        for t in thresholds:
            agg = ms.get(f"well_supp_{t}") or {}
            # Prefer fire_counts.A_t when available; otherwise derive from per-latent
            # fire list (older runs only saved A_50/A_200 to fire_counts.json).
            count = fc.get(f"A_{t}")
            if count is None:
                fl = fc.get("fire_count") or []
                if fl:
                    count = sum(1 for f in fl if f >= t)
            row[f"alive_ge_{t}"] = count
            row[f"ms_{t}_mean"] = agg.get("mean")
            row[f"ms_{t}_med"]  = agg.get("median")
            row[f"ms_{t}_n"]    = agg.get("n")
        rows.append(row)

    md_lines = []
    md_lines.append("# Monosemanticity Summary (CC3M variants, ImageNet val 50K)\n")
    md_lines.append("MS = Pach et al. 2025 (DINOv2 ViT-B). FMS = Härle et al. 2025 (5-fold CV).")
    md_lines.append("CCS = class-concentration (top-1 share / 1 − H/Hmax). Higher = more monosemantic.\n")

    md_lines.append("## Latent counts (alive at threshold τ)\n")
    head = "| Variant | L | alive ≥1 |" + "".join(f" alive ≥{t} |" for t in thresholds)
    sep = "|---|---:|---:|" + "---:|" * len(thresholds)
    md_lines.append(head); md_lines.append(sep)
    for r in rows:
        line = f"| {r['variant']} | {_fmt(r['L'], 0)} | {_fmt(r['alive_ge_1'], 0)} |"
        for t in thresholds:
            line += f" {_fmt(r.get(f'alive_ge_{t}'), 0)} |"
        md_lines.append(line)

    md_lines.append("\n## MS (mean / median) by filter — n in parens\n")
    head = "| Variant | alive ≥1 |" + "".join(f" well-supp ≥{t} |" for t in thresholds)
    sep = "|---|---|" + "---|" * len(thresholds)
    md_lines.append(head); md_lines.append(sep)
    for r in rows:
        line = (f"| {r['variant']} "
                f"| {_fmt(r['ms_alive_mean'])} / {_fmt(r['ms_alive_med'])} (n={_fmt(r['ms_alive_n'], 0)}) |")
        for t in thresholds:
            line += (f" {_fmt(r.get(f'ms_{t}_mean'))} / {_fmt(r.get(f'ms_{t}_med'))} "
                     f"(n={_fmt(r.get(f'ms_{t}_n'), 0)}) |")
        md_lines.append(line)

    md_lines.append("\n## FMS\n")
    md_lines.append("| Variant | acc₀ mean | FMS@1 | FMS@5 |")
    md_lines.append("|---|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r['variant']} | {_fmt(r['fms_acc0'])} | {_fmt(r['fms_at_1'])} | {_fmt(r['fms_at_5'])} |"
        )

    md_lines.append("\n## CCS\n")
    md_lines.append("| Variant | CCS-Top1 (mean) | CCS-Entropy (mean) |")
    md_lines.append("|---|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r['variant']} | {_fmt(r['ccs_top1'])} | {_fmt(r['ccs_h'])} |"
        )

    md = "\n".join(md_lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
