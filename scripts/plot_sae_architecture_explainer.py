#!/usr/bin/env python3
"""Schematic: single SAE vs modality-masking SAE.

Both panels show the same encoder → latent → decoder skeleton:

    (a) single SAE           — one unified encoder-latent-decoder block;
                                image (I) enters through the top, text (T)
                                through the bottom, but both flow through
                                the same shared network.
    (b) modality-masking SAE — the same encoder-latent-decoder block has
                                been physically cut through the middle:
                                the top half is an image-only sub-network
                                (red) and the bottom half is a text-only
                                sub-network (blue). There is a visible gap
                                between the two halves.

Usage:
    python scripts/plot_sae_architecture_explainer.py \
        --out outputs/theorem2_followup_12/fig_sae_architecture.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


C_IMG   = "#c53030"
C_TXT   = "#2b6cb0"
C_EDG   = "#2d3748"
C_ARROW_IMG = C_IMG
C_ARROW_TXT = C_TXT
C_NET   = "#f7fafc"
C_LAT   = "#e9d8fd"
C_LAT_EDGE = "#6b46c1"
C_I_FILL   = "#fed7d7"
C_T_FILL   = "#bee3f8"


def _init_panel(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 6.4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)


def _arrow(ax, p0, p1, color, lw=2.0, mut=15):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mut,
        color=color, lw=lw, zorder=5, shrinkA=1, shrinkB=1,
    ))


def _poly(ax, pts, face, edge, lw=1.6):
    ax.add_patch(Polygon(pts, closed=True, facecolor=face,
                         edgecolor=edge, lw=lw, zorder=2))


def _box(ax, x, y, w, h, face, edge, lw=1.6, zorder=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face,
                           edgecolor=edge, lw=lw, zorder=zorder))


# ------------------------------------------------------------------ #
# Layout parameters shared by both panels                             #
# ------------------------------------------------------------------ #
CY     = 3.2    # vertical center
IN_H   = 2.0    # input-side height of encoder / decoder
LAT_H  = 4.0    # latent height — overcomplete SAE, intentionally large
LAT_W  = 1.10
ENC_X0, ENC_X1 = 1.95, 3.95
DEC_X0, DEC_X1 = 6.05, 8.05


def _latent_pos():
    x = (ENC_X1 + DEC_X0) / 2 - LAT_W / 2
    return x


def _draw_single(ax):
    """Panel (a): one unified SAE. Arrows: I on top, T on bottom."""
    # encoder
    enc_pts = [
        (ENC_X0, CY - IN_H / 2), (ENC_X0, CY + IN_H / 2),
        (ENC_X1, CY + LAT_H / 2), (ENC_X1, CY - LAT_H / 2),
    ]
    _poly(ax, enc_pts, face=C_NET, edge=C_EDG)
    ax.text((ENC_X0 + ENC_X1) / 2, CY, "encoder",
            ha="center", va="center", fontsize=10, color=C_EDG)

    # latent (full height, mixed color)
    lat_x = _latent_pos()
    _box(ax, lat_x, CY - LAT_H / 2, LAT_W, LAT_H,
         face=C_LAT, edge=C_LAT_EDGE, lw=1.8)
    ax.text(lat_x + LAT_W / 2, CY + LAT_H / 2 + 0.2,
            r"latent $\mathbf{z}\in\mathbb{R}^L$",
            ha="center", va="bottom", fontsize=9.5, color=C_LAT_EDGE)

    # decoder
    dec_pts = [
        (DEC_X0, CY - LAT_H / 2), (DEC_X0, CY + LAT_H / 2),
        (DEC_X1, CY + IN_H / 2), (DEC_X1, CY - IN_H / 2),
    ]
    _poly(ax, dec_pts, face=C_NET, edge=C_EDG)
    ax.text((DEC_X0 + DEC_X1) / 2, CY, "decoder",
            ha="center", va="center", fontsize=10, color=C_EDG)

    # arrows: image on top (in and out), text on bottom (in and out)
    y_img = CY + IN_H / 2 - 0.55
    y_txt = CY - IN_H / 2 + 0.55
    # inputs
    _arrow(ax, (0.55, y_img), (ENC_X0 - 0.02, y_img), color=C_IMG, lw=2.2)
    _arrow(ax, (0.55, y_txt), (ENC_X0 - 0.02, y_txt), color=C_TXT, lw=2.2)
    ax.text(0.20, y_img, r"$\mathbf{x}$", fontsize=13,
            color=C_IMG, ha="right", va="center", fontweight="bold")
    ax.text(0.20, y_txt, r"$\mathbf{y}$", fontsize=13,
            color=C_TXT, ha="right", va="center", fontweight="bold")
    # outputs
    _arrow(ax, (DEC_X1 + 0.02, y_img), (9.35, y_img), color=C_IMG, lw=2.2)
    _arrow(ax, (DEC_X1 + 0.02, y_txt), (9.35, y_txt), color=C_TXT, lw=2.2)
    ax.text(9.70, y_img, r"$\tilde{\mathbf{x}}$", fontsize=13,
            color=C_IMG, ha="left", va="center", fontweight="bold")
    ax.text(9.70, y_txt, r"$\tilde{\mathbf{y}}$", fontsize=13,
            color=C_TXT, ha="left", va="center", fontweight="bold")

    # central arrows inside the block
    _arrow(ax, (ENC_X1 + 0.02, CY), (lat_x - 0.02, CY), color=C_EDG, lw=1.4, mut=12)
    _arrow(ax, (lat_x + LAT_W + 0.02, CY), (DEC_X0 - 0.02, CY), color=C_EDG, lw=1.4, mut=12)

    ax.text(5.0, 0.05,
            "single network: both modalities share every latent slot",
            ha="center", va="center", fontsize=9.5,
            color="#2d3748", style="italic")


def _draw_masked(ax):
    """Panel (b): the SAME skeleton, but physically cut through the middle."""
    gap = 0.35   # vertical gap at the cut line
    cut = CY     # cut line position

    # ----- TOP half (image) -----
    # encoder top trapezoid
    enc_top = [
        (ENC_X0, cut + gap / 2),
        (ENC_X0, CY + IN_H / 2),
        (ENC_X1, CY + LAT_H / 2),
        (ENC_X1, cut + gap / 2),
    ]
    _poly(ax, enc_top, face=C_I_FILL, edge=C_IMG, lw=1.8)
    # decoder top trapezoid
    dec_top = [
        (DEC_X0, cut + gap / 2),
        (DEC_X0, CY + LAT_H / 2),
        (DEC_X1, CY + IN_H / 2),
        (DEC_X1, cut + gap / 2),
    ]
    _poly(ax, dec_top, face=C_I_FILL, edge=C_IMG, lw=1.8)
    # latent top half — same purple as panel (a)
    lat_x = _latent_pos()
    _box(ax, lat_x, cut + gap / 2, LAT_W, LAT_H / 2 - gap / 2,
         face=C_LAT, edge=C_LAT_EDGE, lw=1.8)

    # ----- BOTTOM half (text) -----
    enc_bot = [
        (ENC_X0, CY - IN_H / 2),
        (ENC_X0, cut - gap / 2),
        (ENC_X1, cut - gap / 2),
        (ENC_X1, CY - LAT_H / 2),
    ]
    _poly(ax, enc_bot, face=C_T_FILL, edge=C_TXT, lw=1.8)
    dec_bot = [
        (DEC_X0, CY - LAT_H / 2),
        (DEC_X0, cut - gap / 2),
        (DEC_X1, cut - gap / 2),
        (DEC_X1, CY - IN_H / 2),
    ]
    _poly(ax, dec_bot, face=C_T_FILL, edge=C_TXT, lw=1.8)
    _box(ax, lat_x, CY - LAT_H / 2, LAT_W, LAT_H / 2 - gap / 2,
         face=C_LAT, edge=C_LAT_EDGE, lw=1.8)

    # ----- labels on the two halves -----
    ax.text((ENC_X0 + ENC_X1) / 2, CY + IN_H / 4 + 0.15, "encoder",
            ha="center", va="center", fontsize=9.5, color=C_IMG)
    ax.text((DEC_X0 + DEC_X1) / 2, CY + IN_H / 4 + 0.15, "decoder",
            ha="center", va="center", fontsize=9.5, color=C_IMG)
    ax.text((ENC_X0 + ENC_X1) / 2, CY - IN_H / 4 - 0.15, "encoder",
            ha="center", va="center", fontsize=9.5, color=C_TXT)
    ax.text((DEC_X0 + DEC_X1) / 2, CY - IN_H / 4 - 0.15, "decoder",
            ha="center", va="center", fontsize=9.5, color=C_TXT)

    # latent labels
    ax.text(lat_x + LAT_W / 2, CY + LAT_H / 2 + 0.2,
            r"latent $\mathbf{z}\in\mathbb{R}^L$",
            ha="center", va="bottom", fontsize=9.5, color=C_EDG)
    ax.text(lat_x + LAT_W + 0.12, CY + LAT_H / 4, "first L/2\n(I only)",
            ha="left", va="center", fontsize=8, color=C_IMG)
    ax.text(lat_x + LAT_W + 0.12, CY - LAT_H / 4, "last L/2\n(T only)",
            ha="left", va="center", fontsize=8, color=C_TXT)

    # ----- the "cut" marker across the full block -----
    # scissors effect: a horizontal dashed band through the middle
    band_top = cut + gap / 2
    band_bot = cut - gap / 2
    ax.plot([ENC_X0 - 0.1, DEC_X1 + 0.1], [band_top, band_top],
            color="#718096", lw=1.0, ls=(0, (3, 3)), zorder=3)
    ax.plot([ENC_X0 - 0.1, DEC_X1 + 0.1], [band_bot, band_bot],
            color="#718096", lw=1.0, ls=(0, (3, 3)), zorder=3)
    ax.text(DEC_X1 + 0.55, cut, "cut",
            ha="left", va="center", fontsize=8, color="#718096", style="italic")

    # ----- arrows: I on top, T on bottom (each stays in its own half) -----
    y_img = CY + IN_H / 2 - 0.55
    y_txt = CY - IN_H / 2 + 0.55
    _arrow(ax, (0.55, y_img), (ENC_X0 - 0.02, y_img), color=C_IMG, lw=2.2)
    _arrow(ax, (0.55, y_txt), (ENC_X0 - 0.02, y_txt), color=C_TXT, lw=2.2)
    ax.text(0.20, y_img, r"$\mathbf{x}$", fontsize=13,
            color=C_IMG, ha="right", va="center", fontweight="bold")
    ax.text(0.20, y_txt, r"$\mathbf{y}$", fontsize=13,
            color=C_TXT, ha="right", va="center", fontweight="bold")
    _arrow(ax, (DEC_X1 + 0.02, y_img), (9.35, y_img), color=C_IMG, lw=2.2)
    _arrow(ax, (DEC_X1 + 0.02, y_txt), (9.35, y_txt), color=C_TXT, lw=2.2)
    ax.text(9.70, y_img, r"$\tilde{\mathbf{x}}$", fontsize=13,
            color=C_IMG, ha="left", va="center", fontweight="bold")
    ax.text(9.70, y_txt, r"$\tilde{\mathbf{y}}$", fontsize=13,
            color=C_TXT, ha="left", va="center", fontweight="bold")

    ax.text(5.0, 0.05,
            "same skeleton, but cut in half — top is I-only, bottom is T-only",
            ha="center", va="center", fontsize=9.5,
            color="#2d3748", style="italic")


def make_fig(out_path: str):
    _, axes = plt.subplots(1, 2, figsize=(14.8, 4.6))
    _init_panel(axes[0], "(a) single SAE  —  w/o modality masking")
    _draw_single(axes[0])
    _init_panel(axes[1], "(b) modality-masking SAE  —  w/ modality masking")
    _draw_masked(axes[1])

    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03, wspace=0.06)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print(f"saved {out_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str,
                   default="outputs/theorem2_followup_12/fig_sae_architecture.png")
    args = p.parse_args()
    make_fig(args.out)


if __name__ == "__main__":
    main()
