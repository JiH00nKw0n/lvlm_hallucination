#!/usr/bin/env python3
"""Four-panel explainer for cross-modal alignment metrics.

Panels:
    (a) PLC — Per-pair Latent Cosine
    (b) Cross-Slot Correlation (top-m_S)
    (c) Joint Raw Recovery Rate (τ = 0.95)
    (d) Probe Latent Cosine

Each panel shows a small schematic and the math definition.

Usage:
    python scripts/plot_alignment_metric_explainer.py \
        --out outputs/theorem2_followup_15/fig_align_metric_explainer.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


C_IMG = "#c53030"
C_TXT = "#2b6cb0"
C_EDG = "#2d3748"
C_HL  = "#6b46c1"


def _style(ax, title, xlim=(0, 10), ylim=(0, 6.0)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)


def _box(ax, x, y, w, h, face, edge, lw=1.2, zorder=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face,
                           edgecolor=edge, lw=lw, zorder=zorder))


def _arrow(ax, p0, p1, color=C_EDG, lw=1.4, mut=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                 mutation_scale=mut, color=color, lw=lw, zorder=4))


def _code_column(ax, cx, cy, L_show, active_idx, color, label, cell_h=0.18, cell_w=0.32):
    """Render a sparse code column with a few slots highlighted as active."""
    total_h = L_show * cell_h
    y0 = cy - total_h / 2
    for i in range(L_show):
        y = y0 + i * cell_h
        is_active = i in active_idx
        face = color if is_active else "white"
        ec   = color if is_active else "#cbd5e0"
        _box(ax, cx - cell_w / 2, y, cell_w, cell_h - 0.015,
             face=face, edge=ec, lw=0.8, zorder=3)
    ax.text(cx, y0 + total_h + 0.15, label, ha="center", va="bottom",
            fontsize=9.5, color=color, fontweight="bold")
    return y0


# ================================================================== #
#  Panel (a) — PLC                                                     #
# ================================================================== #
def _draw_plc(ax):
    _style(ax, "(a) PLC — Per-pair Latent Cosine", ylim=(0, 6.3))

    # input header
    ax.text(1.05, 5.65, "paired eval sample", fontsize=9, color="#555",
            ha="center", va="bottom", style="italic")

    # x (image) row
    y_img = 4.60
    _box(ax, 0.80, y_img - 0.30, 0.50, 0.60, "#fed7d7", C_IMG, lw=1.4)
    ax.text(1.05, y_img, r"$\mathbf{x}$", ha="center", va="center",
            fontsize=12, color=C_IMG, fontweight="bold")
    _arrow(ax, (1.40, y_img), (2.45, y_img))
    ax.text(1.92, y_img + 0.20, "SAE$_I$", fontsize=8, color="#555",
            ha="center", va="bottom")

    # y (text) row
    y_txt = 2.90
    _box(ax, 0.80, y_txt - 0.30, 0.50, 0.60, "#bee3f8", C_TXT, lw=1.4)
    ax.text(1.05, y_txt, r"$\mathbf{y}$", ha="center", va="center",
            fontsize=12, color=C_TXT, fontweight="bold")
    _arrow(ax, (1.40, y_txt), (2.45, y_txt))
    ax.text(1.92, y_txt + 0.20, "SAE$_T$", fontsize=8, color="#555",
            ha="center", va="bottom")

    # sparse code columns
    _code_column(ax, 2.95, y_img, 12, {2, 5, 8}, C_IMG,
                 r"$\mathbf{z}_I\in\mathbb{R}^L$")
    _code_column(ax, 2.95, y_txt, 12, {2, 5, 9}, C_TXT,
                 r"$\mathbf{z}_T\in\mathbb{R}^L$")

    # merge arrow into cosine
    _arrow(ax, (3.40, y_img), (5.05, 3.95), lw=1.2)
    _arrow(ax, (3.40, y_txt), (5.05, 3.55), lw=1.2)

    # cosine formula box
    ax.text(6.30, 3.75,
            r"$\cos(\mathbf{z}_I,\mathbf{z}_T) = \frac{\mathbf{z}_I^{\top}\mathbf{z}_T}"
            r"{\|\mathbf{z}_I\|\,\|\mathbf{z}_T\|}$",
            ha="center", va="center", fontsize=11, color=C_HL)
    ax.text(6.30, 2.85, r"$\in [-1,\,1]$", ha="center", va="center",
            fontsize=9, color="#555")

    # final definition
    ax.text(5.0, 1.30,
            r"$\mathrm{PLC} \; := \; \mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\gamma}\,\cos(\mathbf{z}_I,\mathbf{z}_T)$",
            ha="center", va="center", fontsize=11)
    ax.text(5.0, 0.40,
            "mean cosine between the two dense top-$k$ sparse codes\n"
            "of each paired image–text sample over the eval set",
            ha="center", va="center", fontsize=8.5, color="#555", style="italic")


# ================================================================== #
#  Panel (b) — Cross-slot correlation                                 #
# ================================================================== #
def _draw_crossslot(ax):
    _style(ax, r"(b) Cross-Slot Correlation (top-$m_S$)")
    # Show two L-column matrices (samples × L) and correlate column j between them
    mat_w = 2.0
    mat_h = 1.8
    ax.text(1.9, 5.4, r"$\mathbf{Z}_I \in \mathbb{R}^{N\times L}$",
            ha="center", va="bottom", fontsize=9.5, color=C_IMG)
    _box(ax, 0.9, 3.2, mat_w, mat_h, "#fed7d7", C_IMG, lw=1.2)
    for j in range(6):
        _box(ax, 0.9 + (j + 1) * mat_w / 7 - 0.05, 3.3, 0.1, mat_h - 0.2,
             face="#fc8181", edge="#c53030", lw=0.5)

    ax.text(5.5, 5.4, r"$\mathbf{Z}_T \in \mathbb{R}^{N\times L}$",
            ha="center", va="bottom", fontsize=9.5, color=C_TXT)
    _box(ax, 4.5, 3.2, mat_w, mat_h, "#bee3f8", C_TXT, lw=1.2)
    for j in range(6):
        _box(ax, 4.5 + (j + 1) * mat_w / 7 - 0.05, 3.3, 0.1, mat_h - 0.2,
             face="#63b3ed", edge="#2b6cb0", lw=0.5)

    # highlight slot j column in both
    hl_j = 3
    _box(ax, 0.9 + (hl_j + 1) * mat_w / 7 - 0.07, 3.28, 0.14, mat_h - 0.16,
         face="none", edge=C_HL, lw=2.0, zorder=5)
    _box(ax, 4.5 + (hl_j + 1) * mat_w / 7 - 0.07, 3.28, 0.14, mat_h - 0.16,
         face="none", edge=C_HL, lw=2.0, zorder=5)
    ax.text(0.9 + (hl_j + 1) * mat_w / 7, 2.95, "col j",
            ha="center", va="top", fontsize=8, color=C_HL)
    ax.text(4.5 + (hl_j + 1) * mat_w / 7, 2.95, "col j",
            ha="center", va="top", fontsize=8, color=C_HL)

    # correlation arrow
    _arrow(ax, (3.0, 4.1), (4.4, 4.1))
    ax.text(3.7, 4.45, "corr", ha="center", va="bottom", fontsize=9, color=C_HL)

    ax.text(8.45, 4.1,
            r"$C_{jj} = \frac{\mathrm{Cov}(\mathbf{z}_I^{:j}, \mathbf{z}_T^{:j})}"
            r"{\sigma_{I,j} \sigma_{T,j}}$",
            ha="center", va="center", fontsize=10, color=C_HL)

    ax.text(5.0, 1.5,
            r"$\overline{C}_{\mathrm{top}} := \frac{1}{m_S}\sum_{j=1}^{m_S} C_{jj}$",
            ha="center", va="center", fontsize=11)
    ax.text(5.0, 0.65,
            "per-slot signed Pearson correlation between image and text\n"
            "latent activations, averaged over the first $m_S$ slots",
            ha="center", va="center", fontsize=8.5, color="#555", style="italic")


# ================================================================== #
#  Panel (c) — Joint raw recovery at τ=0.95                           #
# ================================================================== #
def _draw_jointraw(ax):
    _style(ax, r"(c) Joint Raw Recovery Rate ($\tau = 0.95$)")
    # For a GT shared atom g, both Phi_S[:,g] and Psi_S[:,g]
    # need to be matched by the SAME slot index i.
    ax.text(1.2, 5.0, r"GT shared atom $g$", fontsize=9, color=C_EDG,
            ha="left", va="center", style="italic")
    # phi, psi vectors
    ax.text(0.8, 4.35, r"$\mathbf{\Phi}_S[:,g]$", fontsize=10, color=C_IMG,
            ha="left", va="center")
    ax.text(0.8, 3.40, r"$\mathbf{\Psi}_S[:,g]$", fontsize=10, color=C_TXT,
            ha="left", va="center")
    _arrow(ax, (2.8, 4.35), (4.15, 4.35), color=C_IMG)
    _arrow(ax, (2.8, 3.40), (4.15, 3.40), color=C_TXT)

    # decoder columns V, W with same slot index i highlighted
    for k, (lbl, color) in enumerate([(r"$[\mathbf{V}]_{:,i}$", C_IMG),
                                       (r"$[\mathbf{W}]_{:,i}$", C_TXT)]):
        y = 4.35 - k * 0.95
        _box(ax, 4.20, y - 0.18, 0.45, 0.36, "white", color, lw=1.5)
        ax.text(4.85, y, lbl, fontsize=10, color=color, ha="left", va="center")

    # joint score
    ax.text(7.4, 3.9,
            r"$\mathrm{joint}(g) := \max_i \frac{1}{2}("
            r"|\cos(\mathbf{V}_{:,i}, \mathbf{\Phi}_S[:,g])|$"
            "\n"
            r"$\qquad\qquad + |\cos(\mathbf{W}_{:,i}, \mathbf{\Psi}_S[:,g])|)$",
            ha="center", va="center", fontsize=9, color=C_HL)

    # key: i is the SAME slot for both
    ax.text(5.0, 2.25,
            "same slot index $i$ used for BOTH image and text — no Hungarian matching",
            ha="center", va="center", fontsize=8.5, color="#555", style="italic")

    ax.text(5.0, 1.35,
            r"$\mathrm{joint\_mgt}_{\tau} := \frac{1}{n_S}\sum_{g=1}^{n_S}"
            r"\mathbf{1}\left[\mathrm{joint}(g) > \tau\right]$",
            ha="center", va="center", fontsize=10)
    ax.text(5.0, 0.65,
            r"fraction of GT atoms whose joint score exceeds $\tau$ = 0.95",
            ha="center", va="center", fontsize=8.5, color="#555", style="italic")


# ================================================================== #
#  Panel (b) — Probe latent cosine                                     #
# ================================================================== #
def _draw_probe(ax):
    _style(ax, "(b) Probe Latent Cosine", ylim=(0, 6.3))

    ax.text(1.35, 5.65, r"pure-concept probe for GT atom $g$",
            fontsize=9, color="#555", ha="center", va="bottom", style="italic")

    # image probe
    y_img = 4.60
    _box(ax, 0.55, y_img - 0.30, 1.60, 0.60, "#fed7d7", C_IMG, lw=1.4)
    ax.text(1.35, y_img, r"$\mathbf{x} := [\mathbf{\Phi}_S]_{:,g}$",
            ha="center", va="center", fontsize=10, color=C_IMG)
    _arrow(ax, (2.25, y_img), (3.05, y_img))
    ax.text(2.65, y_img + 0.20, r"SAE$_I$", fontsize=8, color="#555",
            ha="center", va="bottom")

    # text probe
    y_txt = 2.90
    _box(ax, 0.55, y_txt - 0.30, 1.60, 0.60, "#bee3f8", C_TXT, lw=1.4)
    ax.text(1.35, y_txt, r"$\mathbf{y} := [\mathbf{\Psi}_S]_{:,g}$",
            ha="center", va="center", fontsize=10, color=C_TXT)
    _arrow(ax, (2.25, y_txt), (3.05, y_txt))
    ax.text(2.65, y_txt + 0.20, r"SAE$_T$", fontsize=8, color="#555",
            ha="center", va="bottom")

    # latent columns
    _code_column(ax, 3.55, y_img, 12, {4, 7}, C_IMG, r"$\mathbf{z}_I^{(g)}$")
    _code_column(ax, 3.55, y_txt, 12, {4, 7}, C_TXT, r"$\mathbf{z}_T^{(g)}$")

    # arrow into cosine
    _arrow(ax, (4.00, y_img), (5.30, 3.95), lw=1.2)
    _arrow(ax, (4.00, y_txt), (5.30, 3.55), lw=1.2)

    ax.text(6.55, 3.75,
            r"$\cos(\mathbf{z}_I^{(g)},\mathbf{z}_T^{(g)})$",
            ha="center", va="center", fontsize=11, color=C_HL)
    ax.text(6.55, 2.85, r"$\in [-1,\,1]$", ha="center", va="center",
            fontsize=9, color="#555")

    ax.text(5.0, 1.30,
            r"$\mathrm{Probe\,cos} \; := \; \frac{1}{n_S}\sum_{g=1}^{n_S}"
            r"\cos(\mathbf{z}_I^{(g)},\mathbf{z}_T^{(g)})$",
            ha="center", va="center", fontsize=11)
    ax.text(5.0, 0.40,
            "for each GT shared atom $g$, feed its image and text\n"
            "atom directly into the SAEs, then average latent cosines",
            ha="center", va="center", fontsize=8.5, color="#555", style="italic")


def make_fig(out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.4))
    _draw_plc(axes[0])
    _draw_probe(axes[1])

    fig.suptitle(
        "Cross-modal latent-alignment metrics — definitions",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.08)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print(f"saved {out_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str,
                   default="outputs/theorem2_followup_15/fig_align_metric_explainer.png")
    args = p.parse_args()
    make_fig(args.out)


if __name__ == "__main__":
    main()
