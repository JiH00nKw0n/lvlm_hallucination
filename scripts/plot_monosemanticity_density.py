#!/usr/bin/env python3
"""MS sorted-density plot in the style of Group-Sparse Autoencoders Fig. 4.

For each CC3M-trained variant, loads the per-latent MS scores aggregated in
``ms_summary.json`` (``sorted_curve``: descending list over alive latents) and
plots them against a normalized neuron index x ∈ [0, 1]. Higher curve =
more monosemantic latents.

Usage:
    python scripts/plot_monosemanticity_density.py \\
        --root outputs/real_exp_cc3m/monosemanticity \\
        --variants shared separated iso_align group_sparse \\
        --out outputs/real_exp_cc3m/monosemanticity/ms_density
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from palette import STRAWBERRY_RED, CARROT_ORANGE, SEAWEED, CERULEAN

VARIANT_STYLE = {
    "shared":       {"label": "Shared SAE",             "color": STRAWBERRY_RED, "ls": "-"},
    "separated":    {"label": "Modality-Specific SAEs", "color": CERULEAN,       "ls": "-"},
    "iso_align":    {"label": "Iso-Energy Alignment",   "color": CARROT_ORANGE,  "ls": "--"},
    "group_sparse": {"label": "Group-Sparse",         "color": SEAWEED,        "ls": "--"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--variants", nargs="+",
                   default=["shared", "separated", "iso_align", "group_sparse"])
    p.add_argument("--out", type=str, required=True,
                   help="Output stem (no extension); writes .pdf .png .svg")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    fig, ax = plt.subplots(figsize=(4.0, 2.6))

    for v in args.variants:
        summ_path = root / v / "ms_summary.json"
        if not summ_path.exists():
            print(f"[skip] {v}: missing {summ_path}")
            continue
        d = json.load(open(summ_path))
        curve = np.asarray(d.get("sorted_curve", []), dtype=np.float64)
        if curve.size == 0:
            print(f"[skip] {v}: empty sorted_curve")
            continue
        x = np.linspace(0.0, 1.0, len(curve))
        st = VARIANT_STYLE.get(v, {"label": v, "color": "k", "ls": "-"})
        ax.plot(x, curve, color=st["color"], linestyle=st["ls"],
                lw=1.4, label=f"{st['label']} (n={len(curve)})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("Normalized Neuron Index")
    ax.set_ylabel("MonoSemanticity Score")
    ax.axhline(0.0, color="gray", lw=0.5, ls=":")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.tick_params(axis="both", labelsize=8)

    fig.tight_layout()

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
