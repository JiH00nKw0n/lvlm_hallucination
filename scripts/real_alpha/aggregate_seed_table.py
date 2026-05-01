"""Aggregate per-seed run_real_v2 outputs into a mean ± std table.

Reads N seed-specific output roots, pulls each metric via the same logic as
``build_real_table.collect_matrix``, then writes:

  <out>/table.md      cells are "mean ± std (N=k)"
  <out>/table.tex     same, LaTeX-formatted

Best/2nd-best ranking is on the mean; ties are not specially handled.

Usage:
  python scripts/real_alpha/aggregate_seed_table.py \\
    --config configs/real/cc3m.yaml \\
    --roots outputs/real_exp_cc3m_s0,outputs/real_exp_cc3m_s1,outputs/real_exp_cc3m_s2 \\
    --out outputs/real_exp_cc3m_mean
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Reuse build_real_table internals.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_real_table import (  # type: ignore  # noqa: E402
    METHOD_LABELS,
    RealExperimentConfig,
    _rank_mark,
    collect_matrix,
)


def _fmt_cell_mean_std(
    mean: float | None, std: float | None, *, is_percent: bool, mark: str, latex: bool,
) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if is_percent:
        m_s, s_s = f"{100 * mean:.2f}", (f"{100 * std:.2f}" if std is not None else "—")
    else:
        m_s = f"{mean:.4f}" if abs(mean) < 10 else f"{mean:.2f}"
        s_s = (f"{std:.4f}" if std is not None and abs(std) < 10
               else (f"{std:.2f}" if std is not None else "—"))
    body = f"{m_s} ± {s_s}"
    if latex:
        if mark == "best":   return r"\textbf{" + body + "}"
        if mark == "second": return r"\underline{" + body + "}"
        return body
    if mark == "best":   return f"**{body}**"
    if mark == "second": return f"_{body}_"
    return body


def _aggregate(values_per_seed: list[dict[str, dict[int, float | None]]],
               methods: list[str], n_cols: int):
    """Returns means[m][ci], stds[m][ci]; missing seeds dropped per-cell."""
    means: dict[str, dict[int, float | None]] = {m: {} for m in methods}
    stds: dict[str, dict[int, float | None]] = {m: {} for m in methods}
    for m in methods:
        for ci in range(n_cols):
            vals: list[float] = []
            for v in values_per_seed:
                x = v[m].get(ci)
                if x is None:
                    continue
                if isinstance(x, float) and math.isnan(x):
                    continue
                vals.append(float(x))
            if not vals:
                means[m][ci] = None
                stds[m][ci] = None
                continue
            arr = np.asarray(vals, dtype=float)
            means[m][ci] = float(arr.mean())
            stds[m][ci] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return means, stds


def render_markdown(cfg, roots: list[Path], n_seeds: int) -> str:
    values_per_seed = [collect_matrix(cfg, r, latex=False) for r in roots]
    methods = values_per_seed[0][0]
    columns = values_per_seed[0][1]
    n_cols = len(columns)
    means, stds = _aggregate([v[2] for v in values_per_seed], methods, n_cols)

    marks: dict[int, dict[str, str]] = {}
    for ci, col in enumerate(columns):
        mv = [(m, means[m][ci]) for m in methods]
        marks[ci] = _rank_mark(mv, col.direction)

    lines = [
        f"# {cfg.name} — trained {cfg.training.num_epochs} epochs "
        f"(L={cfg.training.latent_size}, k={cfg.training.k}) — N={n_seeds} seeds",
        "",
    ]
    headers = ["Method"] + [f"{c.dataset}/{c.label}" for c in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for m in methods:
        row = [METHOD_LABELS.get(m, m).replace(r"\textbf{", "**").replace("}", "**")]
        for ci, col in enumerate(columns):
            row.append(_fmt_cell_mean_std(
                means[m][ci], stds[m][ci],
                is_percent=col.is_percent, mark=marks[ci][m], latex=False,
            ))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_latex(cfg, roots: list[Path], n_seeds: int) -> str:
    values_per_seed = [collect_matrix(cfg, r, latex=True) for r in roots]
    methods = values_per_seed[0][0]
    columns = values_per_seed[0][1]
    n_cols = len(columns)
    means, stds = _aggregate([v[2] for v in values_per_seed], methods, n_cols)

    marks: dict[int, dict[str, str]] = {}
    for ci, col in enumerate(columns):
        mv = [(m, means[m][ci]) for m in methods]
        marks[ci] = _rank_mark(mv, col.direction)

    n_total = 1 + len(columns)
    col_spec = "l" + "c" * len(columns)

    dataset_spans: list[tuple[str, int]] = []
    for col in columns:
        if dataset_spans and dataset_spans[-1][0] == col.dataset:
            dataset_spans[-1] = (col.dataset, dataset_spans[-1][1] + 1)
        else:
            dataset_spans.append((col.dataset, 1))

    lines = [r"\begin{tabular}{" + col_spec + "}", r"\toprule"]
    lines.append(
        r"\multicolumn{" + str(n_total) + r"}{c}{\textit{Trained on " + cfg.name
        + r", " + str(cfg.training.num_epochs) + r" epochs, $L=" + str(cfg.training.latent_size)
        + r"$, $k=" + str(cfg.training.k) + r"$, $N=" + str(n_seeds) + r"$ seeds}} \\ "
    )
    lines.append(r"\midrule")
    ds_header = ["Method"]
    for ds, span in dataset_spans:
        ds_header.append(ds if span == 1
                         else r"\multicolumn{" + str(span) + r"}{c}{" + ds + "}")
    lines.append(" & ".join(ds_header) + r" \\")

    cmids: list[str] = []
    col_idx = 1
    for _, span in dataset_spans:
        start, end = col_idx + 1, col_idx + span
        cmids.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
        col_idx += span
    lines.append(" ".join(cmids))
    lines.append(" & ".join([""] + [c.label for c in columns]) + r" \\")
    lines.append(r"\midrule")

    for m in methods:
        cells = [METHOD_LABELS.get(m, m)]
        for ci, col in enumerate(columns):
            cells.append(_fmt_cell_mean_std(
                means[m][ci], stds[m][ci],
                is_percent=col.is_percent, mark=marks[ci][m], latex=True,
            ))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--roots", required=True,
                    help="Comma-separated list of seed-specific output roots.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = RealExperimentConfig.from_yaml(args.config)
    roots = [Path(p) for p in args.roots.split(",")]
    for r in roots:
        if not r.exists():
            raise SystemExit(f"missing root: {r}")
    n_seeds = len(roots)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(cfg, roots, n_seeds)
    tex = render_latex(cfg, roots, n_seeds)
    (out / "table.md").write_text(md)
    (out / "table.tex").write_text(tex)
    print(f"wrote {out}/table.{{md,tex}} (N={n_seeds} seeds)")


if __name__ == "__main__":
    main()
