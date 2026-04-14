#!/usr/bin/env python3
"""Three-panel metric explainer for CR and GRR, with math definitions.

Panels:
    (a) CR counts +1  — same learned slot best-matches both modalities.
    (b) CR counts  0  — different learned slots; modalities do not collapse.
    (c) GRR counts    — some learned column lies inside the τ-cone of Φ_S[:,i].

Each panel shows a geometric sketch on top and the formal definition on the
bottom. Also printed: Reconstruction Error (RE) definition at the top header.

Usage:
    python scripts/plot_metric_explainer.py \
        --out outputs/theorem2_followup_12/fig_metric_explainer.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Wedge


C_PHI  = "#c0392b"
C_PSI  = "#b7791f"
C_V1   = "#2b6cb0"
C_V2   = "#38a169"
C_CONE = "#60a5fa"


def _arrow(ax, xy, color, label, label_xy, lw=2.2, label_color=None, fontsize=10):
    ax.add_patch(FancyArrowPatch(
        (0.0, 0.0), (float(xy[0]), float(xy[1])),
        arrowstyle="-|>", mutation_scale=18,
        color=color, lw=lw, zorder=3,
    ))
    ax.text(float(label_xy[0]), float(label_xy[1]), label, fontsize=fontsize,
            color=label_color or color, ha="center", va="center")


def _style(ax, title):
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.6, 1.75)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    # unit-sphere reference (upper half)
    theta = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#cbd5e0", lw=1.0, ls="--", zorder=0)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#cbd5e0",
                         lw=0.8, ls="--", zorder=0))


def make_fig(out_path: str):
    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[3.2, 1.7],
        hspace=0.05, wspace=0.10,
        left=0.02, right=0.98, top=0.88, bottom=0.03,
    )
    axes_top = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes_bot = [fig.add_subplot(gs[1, c]) for c in range(3)]
    for ax in axes_bot:
        ax.axis("off")

    # ============ (a) CR counts +1 ============
    ax = axes_top[0]
    _style(ax, r"(a) CR counts $+1$  (collapsed)")
    phi = np.array([np.cos(np.deg2rad(108)), np.sin(np.deg2rad(108))]) * 1.05
    psi = np.array([np.cos(np.deg2rad(72)),  np.sin(np.deg2rad(72))])  * 1.05
    v_shared = np.array([np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))]) * 1.22
    _arrow(ax, phi,      C_PHI, r"$[\mathbf{\Phi}_S]_{:,i}$",     phi * 1.30)
    _arrow(ax, psi,      C_PSI, r"$[\mathbf{\Psi}_S]_{:,i}$",     psi * 1.30)
    _arrow(ax, v_shared, C_V1,  r"$[\mathbf{V}]_{:,j^\star}=[\mathbf{W}]_{:,j^\star}$", v_shared * 1.20)
    ax.text(0, -0.20,
            r"$j^{\star}_{I}(i) = j^{\star}_{T}(i)$",
            ha="center", va="center", fontsize=12)
    ax.text(0, -0.33,
            "same learned slot is best-match for both modalities",
            ha="center", va="center", fontsize=8.5, style="italic", color="#666")

    axes_bot[0].text(
        0.5, 0.95,
        r"$\mathrm{CR}   :=   \frac{1}{n_{S}}\sum_{i=1}^{n_{S}} \mathbf{1}\left[  j^{\star}_{I}(i) = j^{\star}_{T}(i)  \right]   \in   [0,1]$",
        ha="center", va="top", fontsize=12,
    )
    axes_bot[0].text(
        0.5, 0.32,
        r"$j^{\star}_{I}(i) := \arg\max_{j\in[L]}\cos([\mathbf{V}]_{:,j},  [\mathbf{\Phi}_S]_{:,i})$"
        "\n"
        r"$j^{\star}_{T}(i) := \arg\max_{j\in[L]}\cos([\mathbf{W}]_{:,j},  [\mathbf{\Psi}_S]_{:,i})$",
        ha="center", va="top", fontsize=9.5, color="#444",
    )

    # ============ (b) CR counts 0 ============
    ax = axes_top[1]
    _style(ax, r"(b) CR counts $0$  (separated)")
    phi = np.array([np.cos(np.deg2rad(122)), np.sin(np.deg2rad(122))]) * 1.05
    psi = np.array([np.cos(np.deg2rad(58)),  np.sin(np.deg2rad(58))])  * 1.05
    v1  = np.array([np.cos(np.deg2rad(120)), np.sin(np.deg2rad(120))]) * 1.20
    v2  = np.array([np.cos(np.deg2rad(60)),  np.sin(np.deg2rad(60))])  * 1.20
    _arrow(ax, phi, C_PHI, r"$[\mathbf{\Phi}_S]_{:,i}$", phi * 1.32)
    _arrow(ax, psi, C_PSI, r"$[\mathbf{\Psi}_S]_{:,i}$", psi * 1.32)
    _arrow(ax, v1,  C_V1,  r"$[\mathbf{V}]_{:,j^\star_I}$",  v1 * 1.18)
    _arrow(ax, v2,  C_V2,  r"$[\mathbf{W}]_{:,j^\star_T}$",  v2 * 1.18)
    ax.text(0, -0.20,
            r"$j^{\star}_{I}(i) \neq j^{\star}_{T}(i)$",
            ha="center", va="center", fontsize=12)
    ax.text(0, -0.33,
            "each modality keeps its own dictionary column",
            ha="center", va="center", fontsize=8.5, style="italic", color="#666")

    axes_bot[1].text(
        0.5, 0.95,
        "lower CR ⇒ fewer collapsed shared atoms\n"
        "(we want CR to be small)",
        ha="center", va="top", fontsize=10, color="#2b6cb0",
    )
    axes_bot[1].text(
        0.5, 0.32,
        r"$\mathrm{RE}   :=   \mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\mathcal{D}_{eval}}"
        r"\left[  \|\mathbf{x}-\tilde{\mathbf{x}}\|_2^2 + \|\mathbf{y}-\tilde{\mathbf{y}}\|_2^2  \right]$",
        ha="center", va="top", fontsize=11, color="#444",
    )

    # ============ (c) GRR counts +1 ============
    ax = axes_top[2]
    _style(ax, r"(c) GRR counts $+1$")
    phi_ang = 90.0
    phi = np.array([np.cos(np.deg2rad(phi_ang)), np.sin(np.deg2rad(phi_ang))]) * 1.05
    cone_halfwidth = 16
    ax.add_patch(Wedge(
        (0, 0), 1.20,
        phi_ang - cone_halfwidth, phi_ang + cone_halfwidth,
        facecolor=C_CONE, alpha=0.30, edgecolor=C_CONE, lw=1.0, zorder=1,
    ))
    v_in  = np.array([np.cos(np.deg2rad(phi_ang + 7)),  np.sin(np.deg2rad(phi_ang + 7))])  * 1.13
    v_out = np.array([np.cos(np.deg2rad(phi_ang + 38)), np.sin(np.deg2rad(phi_ang + 38))]) * 1.03
    _arrow(ax, phi,   C_PHI, r"$[\mathbf{\Phi}_S]_{:,i}$", phi * 1.32)
    _arrow(ax, v_in,  C_V1,  r"$[\mathbf{V}]_{:,j^\star}$",    v_in * 1.28)
    ax.add_patch(FancyArrowPatch(
        (0.0, 0.0), (float(v_out[0]), float(v_out[1])),
        arrowstyle="-|>", mutation_scale=14,
        color="#a0aec0", lw=1.3, zorder=2,
    ))
    ax.text(float(v_out[0]) * 1.30, float(v_out[1]) * 1.05, "other\ncolumns",
            fontsize=8, color="#718096", ha="center", va="center")
    edge_ang = np.deg2rad(phi_ang + cone_halfwidth)
    ax.text(np.cos(edge_ang) * 0.60, np.sin(edge_ang) * 0.60,
            r"$\tau$-cone", fontsize=10, color="#1e40af",
            ha="left", va="center")
    ax.text(0, -0.20,
            r"$\max_{j\in[L]} |\cos([\mathbf{V}]_{:,j}, [\mathbf{\Phi}_S]_{:,i})| > \tau$",
            ha="center", va="center", fontsize=11)
    ax.text(0, -0.33,
            r"some learned column lies inside the $\tau$-cone of $\mathbf{\Phi}_S[:,i]$",
            ha="center", va="center", fontsize=8.5, style="italic", color="#666")

    axes_bot[2].text(
        0.5, 0.95,
        r"$\mathrm{GRR}_I(\tau) := \frac{1}{n_S}\sum_{i=1}^{n_S}"
        r"\mathbf{1}\left[\max_{j\in[L]}|\cos([\mathbf{V}]_{:,j},[\mathbf{\Phi}_S]_{:,i})| > \tau\right]$",
        ha="center", va="top", fontsize=11,
    )
    axes_bot[2].text(
        0.5, 0.32,
        r"$\mathrm{GRR}(\tau) := \frac{1}{2}\left(\mathrm{GRR}_I(\tau)+\mathrm{GRR}_T(\tau)\right)$",
        ha="center", va="top", fontsize=10, color="#444",
    )

    fig.suptitle(
        "Collapse Rate (CR), Ground-truth Recovery Rate (GRR), Reconstruction Error (RE)  —  metric definitions",
        fontsize=13, fontweight="bold", y=0.98,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print(f"saved {out_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str,
                   default="outputs/theorem2_followup_12/fig_metric_explainer.png")
    args = p.parse_args()
    make_fig(args.out)


if __name__ == "__main__":
    main()
