"""MMS sorted-curves figure (Image-Image / Text-Text / Image-Text) on
two datasets — CC3M val (in-distribution) and COCO test (out-of-distribution).

Output: outputs/real_exp_cc3m_s0/mms_curves.{pdf,png,svg}
Layout: 2 rows (cc3m_val, coco_test) × 3 columns (ii, tt, it).
Each panel sorts the L latents in descending MMS and plots them at
normalized index x ∈ [0, 1].

Variants: Iso-Energy Alignment, Group-Sparse, Post-hoc Alignment (Ours).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CARROT_ORANGE, SEAWEED, BLUE_SLATE  # noqa: E402

matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

# Defaults; override with --prefix / --out.
ROOTS = [Path(f"outputs/real_exp_cc3m_s{i}") for i in (0, 1, 2)]
OUT = Path("outputs/real_exp_cc3m_mean/mms_curves.pdf")

DATASETS = [
    ("mms_cc3m_val",  "CC-3M"),
    ("mms_coco_test", "MS-COCO"),
]

VARIANTS = [
    ("iso_align",    "Iso-Energy Alignment",      CARROT_ORANGE),
    ("group_sparse", "Group-Sparse",            SEAWEED),
    ("separated",    "Post-hoc Alignment (Ours)", BLUE_SLATE),
]
PAIRS = [("ii", "MMS(Image, Image)"),
         ("tt", "MMS(Text, Text)"),
         ("it", "MMS(Image, Text)")]

PER_PANEL_W = (5.5 / 3) * 1.0
HEIGHT_PER_ROW = 1.015 * 1.3


def _save(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    print(f"wrote {out_path}, .png, .svg")


def _draw_legend(fig: plt.Figure, handles, labels) -> None:
    leg = fig.legend(
        handles=handles, labels=labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles), frameon=False, fontsize=8,
        handlelength=1.2, columnspacing=1.2, handletextpad=0.4,
    )
    for ln in leg.get_lines():
        ln.set_linewidth(ln.get_linewidth() + 2.0)


def _load_mean_curve(d_dir: str, v_dir: str, key: str) -> np.ndarray | None:
    """Mean of per-seed sorted MMS curves at each (normalized) latent-rank position."""
    seeds = []
    for root in ROOTS:
        p = root / d_dir / v_dir / "mms_summary.json"
        if not p.exists():
            continue
        sj = json.load(open(p))
        seeds.append(np.array(sj[f"sorted_{key}"], dtype=float))
    if not seeds:
        return None
    L = min(len(c) for c in seeds)
    arr = np.stack([c[:L] for c in seeds], axis=0)
    return arr.mean(axis=0)


def plot_mms() -> None:
    available = [(d, lab) for d, lab in DATASETS
                 if any((root / d).exists() for root in ROOTS)]
    if not available:
        raise SystemExit("no MMS dataset dirs under any seed root")
    n_rows = len(available)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(PER_PANEL_W * 3, HEIGHT_PER_ROW * n_rows),
        squeeze=False,
    )
    handles, labels = [], []
    for r, (d_dir, d_label) in enumerate(available):
        for c, (key, title) in enumerate(PAIRS):
            ax = axes[r, c]
            for v_dir, label, color in VARIANTS:
                curve = _load_mean_curve(d_dir, v_dir, key)
                if curve is None:
                    continue
                x = np.linspace(0, 1, len(curve))
                line, = ax.plot(x, curve, color=color, lw=1.2)
                if r == 0 and c == 0:
                    handles.append(line)
                    labels.append(label)
            if r == 0:
                ax.set_title(title, fontsize=9, pad=2)
            if r == n_rows - 1:
                ax.set_xlabel("Latent code index", fontsize=9, labelpad=1)
            if c == 0:
                ax.set_ylabel(d_label, fontsize=9, labelpad=2,
                              fontfamily="monospace")
            ax.set_xlim(0, 1)
            ax.set_xticks([0.0, 0.5, 1.0])
            if key == "it":
                ax.set_ylim(-0.01, 0.5)
                ax.set_yticks([0.0, 0.25, 0.5])
            else:
                ax.set_ylim(-0.02, 1.0)
                ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(axis="both", labelsize=8, pad=1)
            ax.grid(alpha=0.15, linewidth=0.4)

    _draw_legend(fig, handles, labels)
    top = 1.0 - 0.34 / n_rows
    plt.subplots_adjust(top=top, bottom=0.22 / n_rows,
                        left=0.07, right=0.99,
                        wspace=0.40, hspace=0.45)
    _save(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="outputs/real_exp_cc3m",
                    help="seed-root prefix; reads <prefix>_s{0,1,2}/ and writes <prefix>_mean/")
    ap.add_argument("--out", default=None,
                    help="override output PDF path")
    args = ap.parse_args()
    ROOTS = [Path(f"{args.prefix}_s{i}") for i in (0, 1, 2)]
    OUT = Path(args.out) if args.out else Path(f"{args.prefix}_mean/mms_curves.pdf")
    plot_mms()
