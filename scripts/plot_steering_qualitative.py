"""Steering qualitative figure (single concept).

Layout (matches kite-style example):
  Row header: [Base | Steering ("<concept>") | Top-5 Retrieved Images]
  Per method: [base_thumb] [method label] [top-1 .. top-5] each square,
              with green ✓ on hits / red ✗ on misses.

Outputs: <out>.svg (+ .pdf, .png).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"


# Hard-coded for s0 / elephant — derived from
# outputs/real_exp_cc3m_s0/cross_modal_steering/qualitative.html
ELEPHANT_S0 = {
    "concept": "elephant",
    "base_iid": 29045,
    "methods": [
        ("Post-hoc\nAlignment\n(Ours)", [
            (273118, True), (38332, True), (36501, True),
            (132776, True), (546067, True),
        ]),
        ("Iso-Energy\nAlignment", [
            (29045, False), (196989, True), (132776, True),
            (368528, False), (490878, False),
        ]),
        ("Group-\nSparse", [
            (29045, False), (490878, False), (132776, True),
            (195673, False), (273118, True),
        ]),
    ],
}


def square_crop(path: Path):
    """Return image array cropped to a centered square."""
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def add_check(ax, hit: bool):
    sym = "✓" if hit else "✗"
    color = "#2a8a2a" if hit else "#b81f1f"
    ax.text(0.06, 0.94, sym, transform=ax.transAxes,
            fontsize=18, fontweight="bold", color=color,
            ha="left", va="top",
            family="DejaVu Sans",
            bbox=dict(facecolor="white", edgecolor="none",
                      pad=0.5, alpha=0.85))


def render(case: dict, image_dir: Path, out: Path,
           figsize=(11, 5.5)) -> None:
    n_methods = len(case["methods"])
    n_top = len(case["methods"][0][1])
    # Columns: Base | Steering label | top-1 .. top-K
    n_cols = 2 + n_top

    fig = plt.figure(figsize=figsize, facecolor="white")

    # Width ratios — make Base & Steering label columns slightly narrower
    width_ratios = [1.0, 0.95] + [1.0] * n_top
    # Header row (titles) thin; method rows equal
    height_ratios = [0.32] + [1.0] * n_methods

    gs = fig.add_gridspec(
        nrows=1 + n_methods, ncols=n_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.05, hspace=0.05,
        left=0.005, right=0.995, top=0.985, bottom=0.01,
    )

    # ---------- header row ----------
    ax_h_base = fig.add_subplot(gs[0, 0]); ax_h_base.axis("off")
    ax_h_base.text(0.5, 0.18, "Base", ha="center", va="center",
                   fontsize=18, fontweight="bold", style="italic")

    ax_h_steer = fig.add_subplot(gs[0, 1]); ax_h_steer.axis("off")
    ax_h_steer.text(0.5, 0.18,
                    f'Steering\n("{case["concept"]}")',
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", style="italic",
                    linespacing=1.1)

    ax_h_top = fig.add_subplot(gs[0, 2:]); ax_h_top.axis("off")
    ax_h_top.text(0.5, 0.18, "Top-5 Retrieved Images",
                  ha="center", va="center",
                  fontsize=18, fontweight="bold", style="italic")

    # ---------- per-method rows ----------
    base_img = square_crop(image_dir / f"{case['base_iid']}.jpg")
    for r_i, (label, top5) in enumerate(case["methods"]):
        row = r_i + 1
        # Base column
        ax_b = fig.add_subplot(gs[row, 0])
        ax_b.imshow(base_img)
        ax_b.set_xticks([]); ax_b.set_yticks([])
        for sp in ax_b.spines.values():
            sp.set_visible(False)

        # Method label column
        ax_l = fig.add_subplot(gs[row, 1]); ax_l.axis("off")
        ax_l.text(0.5, 0.5, label, ha="center", va="center",
                  fontsize=15, linespacing=1.15)

        # Top-5 image columns
        for k, (iid, hit) in enumerate(top5):
            ax = fig.add_subplot(gs[row, 2 + k])
            img = square_crop(image_dir / f"{iid}.jpg")
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            add_check(ax, hit)

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".pdf", ".png"):
        fig.savefig(out.with_suffix(ext), facecolor="white",
                    bbox_inches="tight", pad_inches=0.04,
                    dpi=200 if ext == ".png" else None)
        print(f"wrote {out.with_suffix(ext)}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path,
                    default=Path("outputs/real_exp_cc3m_s0/cross_modal_steering/coco_images"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/real_exp_cc3m_s0/cross_modal_steering/elephant_qualitative.svg"))
    args = ap.parse_args()
    render(ELEPHANT_S0, args.image_dir, args.out)


if __name__ == "__main__":
    main()
