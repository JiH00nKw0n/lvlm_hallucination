#!/usr/bin/env python3
"""Composite DDM figure: PCA panel (a) + image/caption (middle) + multi-model density (b).

Single-row layout designed to span a single paper column (figsize ~11 × 3.2 in).
All panels share the same height; fonts and styling match fig1_v3 conventions.

Inputs:
  --pca-cache   outputs/pca_plot_cache_<pair>.npz (from plot_pca_matched_concept.py)
  --image       outputs/..._pair_image.jpg (selected pair's COCO image)
  --caption     caption text (or --caption-from-info <info.txt>)

Output:
  outputs/ddm_figure.{pdf,png,svg}
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from PIL import Image
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde


# ---------- palette (matches plot_pca_matched_concept + plot_multi_model_density) ----------

STRAWBERRY_RED = "#f94144"
CERULEAN = "#277da1"


def _shade(hex_color: str, lightness: float):
    r, g, b = to_rgb(hex_color)
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, min(1.0, lightness)), s)


IMG_COLORS = [_shade(STRAWBERRY_RED, 0.30),
              _shade(STRAWBERRY_RED, 0.48),
              _shade(STRAWBERRY_RED, 0.66)]
TXT_COLORS = [_shade(CERULEAN, 0.20),
              _shade(CERULEAN, 0.38),
              _shade(CERULEAN, 0.56)]
EMB_IMG_COLOR = _shade(STRAWBERRY_RED, 0.38)
EMB_TXT_COLOR = _shade(CERULEAN, 0.28)

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = [f"$r \\in [{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})$"
              for i in range(len(BIN_EDGES) - 1)]
# Original 5-model palette reordered 빨→주→노→초→파 as r increases.
BIN_COLORS = ["#df3a3d", "#d96627", "#dfb246", "#389076", "#206987"]
ALPHA = 0.2
KDE_GRID = np.linspace(-0.4, 1.0, 300)
ALIVE_THR = 0.001

DENSITY_MODELS = [
    {"name": "CLIP ViT-B/32", "run_dir": "outputs/real_alpha_followup_1/two_sae/final"},
    {"name": "MetaCLIP B/32", "run_dir": "outputs/metaclip_b32/two_sae/final"},
    {"name": "OpenCLIP B/32", "run_dir": "outputs/datacomp_b32/two_sae/final"},
    {"name": "SigLIP2 Base",  "run_dir": "outputs/siglip2_base/two_sae/final"},
]


# ---------- PCA panel ----------

def draw_pca_panel(ax, cache_path: Path, concept_names: list[str]) -> None:
    d = np.load(cache_path, allow_pickle=True)
    x_vec = d["x_vec"]; y_vec = d["y_vec"]
    img_scaled = list(d["img_scaled"])
    txt_scaled = list(d["txt_scaled"])

    stack = np.vstack([x_vec, y_vec, *img_scaled, *txt_scaled])
    _U, S, Vt = np.linalg.svd(stack, full_matrices=False)
    basis = Vt[:2]
    pts = stack @ basis.T
    x_pt, y_pt = pts[0], pts[1]
    raw_img = pts[2:5]
    raw_txt = pts[5:8]

    # Visibility scaling (same logic as plot_pca_matched_concept.py)
    max_embed = max(float(np.linalg.norm(x_pt)), float(np.linalg.norm(y_pt)), 1e-6)
    MIN_LEN = 0.35 * max_embed
    MAX_LEN = 0.65 * max_embed

    def _rescale(row):
        lengths = np.linalg.norm(row, axis=1) + 1e-12
        ml, mn = float(lengths.max()), float(lengths.min())
        if ml - mn < 1e-6:
            target = np.full_like(lengths, 0.5 * (MIN_LEN + MAX_LEN))
        else:
            t = (lengths - mn) / (ml - mn)
            target = MIN_LEN + t * (MAX_LEN - MIN_LEN)
        return row / lengths[:, None] * target[:, None]

    img_pts = _rescale(raw_img)
    txt_pts = _rescale(raw_txt)

    FANCY = "fancy,head_length=0.35,head_width=0.35,tail_width=0.15"

    def draw_arrow(p0, p1, color):
        ax.annotate(
            "", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
            arrowprops=dict(arrowstyle=FANCY, mutation_scale=14,
                            facecolor=color, edgecolor=color, lw=0,
                            shrinkA=0, shrinkB=0),
            zorder=3,
        )

    # Concept arrows
    for k in range(3):
        draw_arrow((0, 0), img_pts[k], IMG_COLORS[k])
        draw_arrow((0, 0), txt_pts[k], TXT_COLORS[k])

    # Embedding dots
    ax.scatter([x_pt[0]], [x_pt[1]], s=32, c=EMB_IMG_COLOR, edgecolors="none", zorder=5)
    ax.scatter([y_pt[0]], [y_pt[1]], s=32, c=EMB_TXT_COLOR, edgecolors="none", zorder=5)
    ax.scatter([0], [0], s=8, c="black", zorder=6)

    # Dashed connectors: concept tip -> embedding
    for k in range(3):
        ax.plot([img_pts[k][0], x_pt[0]], [img_pts[k][1], x_pt[1]],
                color=IMG_COLORS[k], lw=0.6, linestyle=(0, (3, 2)), zorder=2, alpha=0.7)
        ax.plot([txt_pts[k][0], y_pt[0]], [txt_pts[k][1], y_pt[1]],
                color=TXT_COLORS[k], lw=0.6, linestyle=(0, (3, 2)), zorder=2, alpha=0.7)

    # Concept labels at arrow tips
    scale_ref = max_embed

    def put_tip_label(tip, text, color, y_side):
        off_along = tip / (np.linalg.norm(tip) + 1e-12) * 0.04 * scale_ref
        off_y = (+1 if y_side == "above" else -1) * 0.06 * scale_ref
        ax.text(tip[0] + off_along[0], tip[1] + off_along[1] + off_y, text,
                color=color, fontsize=7, fontweight="bold",
                ha="center", va="bottom" if y_side == "above" else "top", zorder=6)

    for k, name in enumerate(concept_names[:3]):
        put_tip_label(img_pts[k], name, IMG_COLORS[k], y_side="above")
        put_tip_label(txt_pts[k], name, TXT_COLORS[k], y_side="below")

    # Reference lines
    ax.axhline(0, color="gray", lw=0.4, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.4, ls="--", zorder=0)

    # Axis limits
    all_x = np.concatenate([img_pts[:, 0], txt_pts[:, 0], [x_pt[0], y_pt[0], 0.0]])
    all_y = np.concatenate([img_pts[:, 1], txt_pts[:, 1], [x_pt[1], y_pt[1], 0.0]])
    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) * 1.25
    cx, cy = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
    ax.set_xlim(float(cx - span / 2), float(cx + span / 2))
    ax.set_ylim(float(cy - span / 2), float(cy + span / 2))
    ax.set_aspect("equal")
    ax.set_xlabel("PC1", fontsize=8, labelpad=1)
    ax.set_ylabel("PC2", fontsize=8, labelpad=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(labelsize=6.5, pad=1)
    ax.grid(alpha=0.15, linewidth=0.4)

    # Legend below PCA
    emb_handles = [
        Line2D([0], [0], marker="o", markersize=6, color="none",
               markerfacecolor=EMB_IMG_COLOR, markeredgecolor="none",
               label="Image embedding"),
        Line2D([0], [0], marker="o", markersize=6, color="none",
               markerfacecolor=EMB_TXT_COLOR, markeredgecolor="none",
               label="Text embedding"),
    ]
    ax.legend(handles=emb_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False,
              fontsize=7, handlelength=0.6, handletextpad=0.4, columnspacing=1.5)


# ---------- density panels ----------

def load_density_data(run_dir: str):
    rd = Path(run_dir)
    C = np.load(rd / "diagnostic_B_C_train.npy")
    sd = load_file(str(rd / "model.safetensors"))
    W_i = sd["image_sae.W_dec"].float().numpy()
    W_t = sd["text_sae.W_dec"].float().numpy()
    W_i_n = W_i / (np.linalg.norm(W_i, axis=1, keepdims=True) + 1e-12)
    W_t_n = W_t / (np.linalg.norm(W_t, axis=1, keepdims=True) + 1e-12)
    rates_path = rd / "diagnostic_B_firing_rates.npz"
    if rates_path.exists():
        rates = np.load(rates_path)
        alive_i = np.where(rates["rate_i"] > ALIVE_THR)[0]
        alive_t = np.where(rates["rate_t"] > ALIVE_THR)[0]
    else:
        alive_i = np.where(np.var(C, axis=1) > 1e-8)[0]
        alive_t = np.where(np.var(C, axis=0) > 1e-8)[0]
    C_sub = C[np.ix_(alive_i, alive_t)]
    row, col = linear_sum_assignment(-C_sub)
    C_m = C_sub[row, col]
    cos_m = (W_i_n[alive_i[row]] * W_t_n[alive_t[col]]).sum(axis=1)
    return C_m, cos_m


def draw_density_subplot(ax, C_m, cos_m, title: str):
    for b in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        mask = (C_m >= lo) & (C_m < hi)
        if int(mask.sum()) < 5:
            continue
        vals = cos_m[mask]
        if np.std(vals) < 1e-8:
            continue
        kde = gaussian_kde(vals)
        density = kde(KDE_GRID)
        ax.fill_between(KDE_GRID, density, color=BIN_COLORS[b], alpha=ALPHA, linewidth=0)
        ax.plot(KDE_GRID, density, color=BIN_COLORS[b], alpha=min(ALPHA + 0.3, 1.0), lw=0.6)
    ax.set_title(title, fontsize=7, fontweight="bold", pad=2)
    ax.set_xlim(-0.3, 1.0)
    ax.set_xticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.axvline(0.0, color="gray", lw=0.5, ls="--", zorder=0)
    ax.tick_params(axis="x", labelsize=6.5, pad=1)
    ax.tick_params(axis="y", labelsize=6.5, pad=1)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)


# ---------- main composite ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pca-cache", default="outputs/pca_plot_cache_350682.npz")
    p.add_argument("--image", default="outputs/pca_matched_concept_pair_image.jpg")
    p.add_argument("--caption",
                   default="A little boy that is standing on a snowboard in the snow.")
    p.add_argument("--concept-names", default="ski,snow,little boy")
    p.add_argument("--out", default="outputs/ddm_figure.pdf")
    args = p.parse_args()

    concept_names = [s.strip() for s in args.concept_names.split(",")]

    # Single-row composite: [PCA | image+caption | 2x2 density]
    fig = plt.figure(figsize=(11, 3.2))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.35, 0.95, 1.85],
        wspace=0.15, left=0.05, right=0.99, top=0.94, bottom=0.19,
    )

    # Left: PCA
    ax_pca = fig.add_subplot(gs[0, 0])
    draw_pca_panel(ax_pca, Path(args.pca_cache), concept_names)
    ax_pca.set_title("(a) Per-pair concept decomposition",
                     fontsize=8, fontweight="bold", loc="left", pad=6, color="#333")

    # Middle: "Image" header + image thumb + "Caption" header + caption box.
    gs_mid = gs[0, 1].subgridspec(
        4, 1,
        height_ratios=[0.15, 1.4, 0.15, 0.9],
        hspace=0.08,
    )
    ax_hdr_img = fig.add_subplot(gs_mid[0]); ax_hdr_img.axis("off")
    ax_hdr_img.text(0.0, 0.5, "Image", fontsize=8, fontweight="bold",
                    ha="left", va="center", color="#333")

    ax_img = fig.add_subplot(gs_mid[1])
    if Path(args.image).exists():
        image_pil = Image.open(args.image).convert("RGB")
        ax_img.imshow(image_pil, aspect="auto")
    else:
        ax_img.text(0.5, 0.5, "(image unavailable)", ha="center", va="center", fontsize=8)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for s in ax_img.spines.values():
        s.set_edgecolor("#bbb"); s.set_linewidth(0.8)

    ax_hdr_cap = fig.add_subplot(gs_mid[2]); ax_hdr_cap.axis("off")
    ax_hdr_cap.text(0.0, 0.5, "Caption", fontsize=8, fontweight="bold",
                    ha="left", va="center", color="#333")

    ax_cap = fig.add_subplot(gs_mid[3])
    ax_cap.axis("off")
    # Rounded-corner background rectangle for the caption
    from matplotlib.patches import FancyBboxPatch
    cap_bg = FancyBboxPatch(
        (0.02, 0.08), 0.96, 0.84,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        transform=ax_cap.transAxes,
        facecolor="#f4f4f5", edgecolor="#d0d0d5", linewidth=0.6, zorder=1,
    )
    ax_cap.add_patch(cap_bg)
    ax_cap.text(
        0.5, 0.5, args.caption.strip(),
        fontsize=7.5, ha="center", va="center", wrap=True,
        family="DejaVu Serif", style="italic", color="#222", zorder=2,
    )

    # Right: 2×2 density grid
    gs_right = gs[0, 2].subgridspec(2, 2, hspace=0.38, wspace=0.22)
    for k, model in enumerate(DENSITY_MODELS):
        r, c = divmod(k, 2)
        ax_d = fig.add_subplot(gs_right[r, c])
        if not (Path(model["run_dir"]) / "diagnostic_B_C_train.npy").exists():
            ax_d.text(0.5, 0.5, "(no data)", ha="center", va="center", fontsize=7)
            ax_d.set_title(model["name"], fontsize=7, fontweight="bold", pad=2)
            continue
        C_m, cos_m = load_density_data(model["run_dir"])
        draw_density_subplot(ax_d, C_m, cos_m, model["name"])
        # Only bottom row shows xlabel; left column shows ylabel.
        if r == 1:
            ax_d.set_xlabel(r"$\cos(\cdot)$", fontsize=8, labelpad=1)
        if c == 0:
            ax_d.set_ylabel("density", fontsize=8, labelpad=1)

    # Density legend under the 2x2 grid (spans only the density region).
    legend_handles = [
        Patch(
            facecolor=to_rgba(BIN_COLORS[b], alpha=ALPHA),
            edgecolor=to_rgba(BIN_COLORS[b], alpha=1.0),
            linewidth=1.2, label=BIN_LABELS[b],
        )
        for b in range(len(BIN_EDGES) - 1)
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.77, 0.015), ncol=5, frameon=False,
        fontsize=6.5, handlelength=1.0, handletextpad=0.3, columnspacing=0.8,
    )

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    print(f"saved {out_path}")
    if str(out_path).endswith(".pdf"):
        for ext in (".png", ".svg"):
            alt = str(out_path).replace(".pdf", ext)
            fig.savefig(alt, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
            print(f"saved {alt}")
    plt.close()


if __name__ == "__main__":
    main()
