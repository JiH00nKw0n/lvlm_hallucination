"""Build the real-data experiment summary table (LaTeX + markdown).

Invoked at the end of ``run_real_v2.py``; can also be run standalone:

    python scripts/real_alpha/build_real_table.py \\
        --config configs/real/cc3m.yaml \\
        --out-root outputs/real_exp_cc3m

For each configured evaluation, pulls metrics from
``<out_root>/<method>/<eval.dataset>/<task>.json`` and tabulates them with
best (``\\textbf``) / 2nd-best (``\\underline``) marking per column.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.configs.real_experiment import RealExperimentConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display metadata
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "shared": "Shared SAE",
    "separated": "Separated SAE",
    "iso_align": "Iso-Energy Align",
    "group_sparse": "Group-Sparse",
    "ours": r"\textbf{Post-hoc Align} (Ours)",
    "vl_sae": "VL-SAE",
}

# (task, sub-key or None, display label, "min" or "max" is better, is_percent)
COLUMN_DEFS = {
    "recon":        [("recon_error", "Recon $\\downarrow$", "min", False)],
    "retrieval":    [
        (("I2T", "R@1"),  "I$\\rightarrow$T R@1",  "max", True),
        (("I2T", "R@5"),  "R@5",                    "max", True),
        (("I2T", "R@10"), "R@10",                   "max", True),
        (("T2I", "R@1"),  "T$\\rightarrow$I R@1",  "max", True),
        (("T2I", "R@5"),  "R@5",                    "max", True),
        (("T2I", "R@10"), "R@10",                   "max", True),
    ],
    "zeroshot_raw": [("accuracy", "Zero-shot $\\uparrow$", "max", True)],
    "dead_latents": [
        ("alive_image_count", "Alive (img)", "max", False),
        ("alive_text_count",  "Alive (txt)", "max", False),
    ],
}

MD_COLUMN_DEFS = {  # markdown versions drop LaTeX glyphs
    "recon":        [("recon_error", "Recon ↓", "min", False)],
    "retrieval":    [
        (("I2T", "R@1"),  "I→T R@1",  "max", True),
        (("I2T", "R@5"),  "I→T R@5",  "max", True),
        (("I2T", "R@10"), "I→T R@10", "max", True),
        (("T2I", "R@1"),  "T→I R@1",  "max", True),
        (("T2I", "R@5"),  "T→I R@5",  "max", True),
        (("T2I", "R@10"), "T→I R@10", "max", True),
    ],
    "zeroshot_raw": [("accuracy", "Zero-shot ↑", "max", True)],
    "dead_latents": [
        ("alive_image_count", "Alive (img)", "max", False),
        ("alive_text_count",  "Alive (txt)", "max", False),
    ],
}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

@dataclass
class Column:
    dataset: str
    task: str
    key: Any                  # str or tuple[str, str]
    label: str
    direction: str            # "min" or "max"
    is_percent: bool


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("failed to load %s: %s", p, e)
        return None


def _get(d: dict | None, key: Any) -> float | None:
    if d is None:
        return None
    if isinstance(key, tuple):
        cur: Any = d
        for k in key:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur if isinstance(cur, (int, float)) else None
    v = d.get(key)
    return v if isinstance(v, (int, float)) else None


def collect_matrix(
    cfg: RealExperimentConfig, out_root: Path, latex: bool,
) -> tuple[list[str], list[Column], dict[str, dict[int, float | None]]]:
    """Return (method_names, columns, values[method][col_index])."""
    col_defs = COLUMN_DEFS if latex else MD_COLUMN_DEFS

    columns: list[Column] = []
    for ev in cfg.evaluations:
        for task in ev.tasks:
            for (key, label, direction, is_pct) in col_defs.get(task, []):
                columns.append(Column(
                    dataset=ev.dataset, task=task, key=key,
                    label=label, direction=direction, is_percent=is_pct,
                ))

    methods = [m.name for m in cfg.methods]
    values: dict[str, dict[int, float | None]] = {m: {} for m in methods}
    for m in methods:
        for ci, col in enumerate(columns):
            j = _load_json(out_root / m / col.dataset / f"{col.task}.json")
            values[m][ci] = _get(j, col.key)
    return methods, columns, values


# ---------------------------------------------------------------------------
# Best / 2nd-best marking
# ---------------------------------------------------------------------------


def _rank_mark(method_vals: list[tuple[str, float | None]], direction: str) -> dict[str, str]:
    """Return {method: "" | "best" | "second"} for a single column."""
    items = [(m, v) for m, v in method_vals
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not items:
        return {m: "" for m, _ in method_vals}
    reverse = direction == "max"
    items.sort(key=lambda x: x[1], reverse=reverse)
    out: dict[str, str] = {m: "" for m, _ in method_vals}
    if items:
        out[items[0][0]] = "best"
    if len(items) > 1 and items[0][1] != items[1][1]:
        out[items[1][0]] = "second"
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_cell(
    v: float | None, col: Column, mark: str, latex: bool,
) -> str:
    if v is None:
        return "--"
    if col.is_percent:
        s = f"{100 * v:.2f}"
    elif isinstance(v, float):
        s = f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    else:
        s = str(v)
    if latex:
        if mark == "best":
            return r"\textbf{" + s + "}"
        if mark == "second":
            return r"\underline{" + s + "}"
        return s
    # markdown
    if mark == "best":
        return f"**{s}**"
    if mark == "second":
        return f"_{s}_"
    return s


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def render_latex(cfg: RealExperimentConfig, out_root: Path) -> str:
    methods, columns, values = collect_matrix(cfg, out_root, latex=True)
    n_cols = 1 + len(columns)
    col_spec = "l" + "c" * len(columns)

    # Compute best/2nd marks per column
    marks: dict[int, dict[str, str]] = {}
    for ci, col in enumerate(columns):
        mv = [(m, values[m][ci]) for m in methods]
        marks[ci] = _rank_mark(mv, col.direction)

    # Build grouped header: dataset spans + task spans on top, columns underneath
    dataset_spans: list[tuple[str, int]] = []
    for col in columns:
        if dataset_spans and dataset_spans[-1][0] == col.dataset:
            dataset_spans[-1] = (col.dataset, dataset_spans[-1][1] + 1)
        else:
            dataset_spans.append((col.dataset, 1))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(
        r"\multicolumn{" + str(n_cols) + r"}{c}{\textit{Trained on "
        + cfg.name + r", " + str(cfg.training.num_epochs) + r" epochs, "
        + r"$L=" + str(cfg.training.latent_size) + r"$, $k="
        + str(cfg.training.k) + r"$}} \\ "
    )
    lines.append(r"\midrule")

    # Dataset super-header
    ds_header = ["Method"]
    col_idx = 0
    for ds, span in dataset_spans:
        if span == 1:
            ds_header.append(ds)
        else:
            ds_header.append(r"\multicolumn{" + str(span) + r"}{c}{" + ds + "}")
        col_idx += span
    lines.append(" & ".join(ds_header) + r" \\")

    # cmidrule under dataset groups (skip first column which is "Method")
    cmids: list[str] = []
    col_idx = 1
    for _, span in dataset_spans:
        start = col_idx + 1
        end = col_idx + span
        cmids.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
        col_idx += span
    lines.append(" ".join(cmids))

    # Column labels
    col_header = [""]
    for col in columns:
        col_header.append(col.label)
    lines.append(" & ".join(col_header) + r" \\")
    lines.append(r"\midrule")

    for m in methods:
        cells = [METHOD_LABELS.get(m, m)]
        for ci, col in enumerate(columns):
            cells.append(_fmt_cell(values[m][ci], col, marks[ci][m], latex=True))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(cfg: RealExperimentConfig, out_root: Path) -> str:
    methods, columns, values = collect_matrix(cfg, out_root, latex=False)

    marks: dict[int, dict[str, str]] = {}
    for ci, col in enumerate(columns):
        mv = [(m, values[m][ci]) for m in methods]
        marks[ci] = _rank_mark(mv, col.direction)

    lines: list[str] = []
    lines.append(
        f"# {cfg.name} — trained {cfg.training.num_epochs} epochs "
        f"(L={cfg.training.latent_size}, k={cfg.training.k})"
    )
    lines.append("")

    headers = ["Method"] + [f"{c.dataset}/{c.label}" for c in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for m in methods:
        row = [METHOD_LABELS.get(m, m).replace(r"\textbf{", "**").replace("}", "**")]
        for ci, col in enumerate(columns):
            row.append(_fmt_cell(values[m][ci], col, marks[ci][m], latex=False))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None,
                    help="Overrides cfg.output.root if provided.")
    args = ap.parse_args()

    cfg = RealExperimentConfig.from_yaml(args.config)
    out_root = Path(args.out_root) if args.out_root else Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)

    tex = render_latex(cfg, out_root)
    md = render_markdown(cfg, out_root)

    (out_root / "table.tex").write_text(tex + "\n")
    (out_root / "table.md").write_text(md)
    logger.info("Wrote %s and %s", out_root / "table.tex", out_root / "table.md")
    print(md)


if __name__ == "__main__":
    main()
