#!/usr/bin/env python3
"""Three-panel intuition figure for Diagnostic B.

Goal: show the ASSUMPTION we make about the data and the REASONING why
the masked matched-pair cosine is a direct proxy for alpha.

Panels:
    (1) Assumption    — paired shared atoms exist with geometric angle α
    (2) Theorem 1     — masked SAE recovers each modality's atom separately
    (3) Measurement   — cos between matched decoder columns ≈ α

Usage:
    python scripts/plot_diagnostic_B_intuition.py \
        --out outputs/theorem2_followup_15/fig_diagnostic_B_intuition.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

for _cand in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
              "Noto Sans CJK KR", "Malgun Gothic"):
    if any(_cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _cand
        break
plt.rcParams["axes.unicode_minus"] = False


C_IMG = "#c53030"   # image modality
C_TXT = "#2b6cb0"   # text  modality
C_NEU = "#2d3748"
C_MUT = "#718096"
C_ACC = "#6b46c1"


ALPHA_DISPLAY = 0.55             # cosine we visualise
ANG = np.degrees(np.arccos(ALPHA_DISPLAY))   # ≈ 56.6°
HALF = ANG / 2                   # half-angle from vertical


def _unit(ax, r=1.0):
    ax.add_patch(Circle((0, 0), r, fill=False,
                         edgecolor="#cbd5e0", lw=1.1, ls="--", zorder=1))


def _arrow(ax, xy, color, lw=2.4, mut=18):
    ax.add_patch(FancyArrowPatch(
        (0, 0), (float(xy[0]), float(xy[1])),
        arrowstyle="-|>", mutation_scale=mut,
        color=color, lw=lw, zorder=4,
    ))


def _style(ax, title):
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.40, 1.60)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)


def _angle_arc(ax, label, color=C_ACC, r=0.40):
    ax.add_patch(Arc(
        (0, 0), 2 * r, 2 * r,
        theta1=90 - HALF, theta2=90 + HALF,
        edgecolor=color, lw=1.5, zorder=3,
    ))
    ax.text(0, r + 0.06, label,
            ha="center", va="bottom", fontsize=10, color=color)


def _make_pair(angle_img, angle_txt):
    phi = np.array([np.cos(np.deg2rad(angle_img)), np.sin(np.deg2rad(angle_img))])
    psi = np.array([np.cos(np.deg2rad(angle_txt)), np.sin(np.deg2rad(angle_txt))])
    return phi, psi


def _draw_assumption(ax):
    _style(ax, "(1) 가정 — 공유 concept은 두 방향으로 존재")
    _unit(ax)

    phi, psi = _make_pair(90 + HALF, 90 - HALF)
    _arrow(ax, phi, C_IMG)
    _arrow(ax, psi, C_TXT)

    ax.text(phi[0] * 1.18, phi[1] * 1.12, r"$[\mathbf{\Phi}_S]_{:,i}$",
            ha="right", va="center", fontsize=11,
            color=C_IMG, fontweight="bold")
    ax.text(psi[0] * 1.18, psi[1] * 1.12, r"$[\mathbf{\Psi}_S]_{:,i}$",
            ha="left", va="center", fontsize=11,
            color=C_TXT, fontweight="bold")

    _angle_arc(ax, r"$\alpha = \cos(\angle)$")

    ax.text(0, -0.08,
            "같은 concept $i$를 나타내는",
            ha="center", va="top", fontsize=9, color=C_NEU)
    ax.text(0, -0.22,
            "image 방향과 text 방향이 각 $\\arccos \\alpha$만큼 벌어져 있다",
            ha="center", va="top", fontsize=9, color=C_NEU, style="italic")


def _draw_theorem1(ax):
    _style(ax, "(2) Theorem 1 — masked SAE는 각 방향을 따로 저장")
    _unit(ax)

    phi, psi = _make_pair(90 + HALF, 90 - HALF)

    # Slightly offset to show "learned" as close to GT (dashed = GT, solid = learned)
    ax.add_patch(FancyArrowPatch(
        (0, 0), (float(phi[0]), float(phi[1])),
        arrowstyle="-|>", mutation_scale=16,
        color=C_IMG, lw=1.3, ls="--", alpha=0.55, zorder=2,
    ))
    ax.add_patch(FancyArrowPatch(
        (0, 0), (float(psi[0]), float(psi[1])),
        arrowstyle="-|>", mutation_scale=16,
        color=C_TXT, lw=1.3, ls="--", alpha=0.55, zorder=2,
    ))

    # Learned columns (solid, essentially on top)
    _arrow(ax, phi * 0.99, C_IMG)
    _arrow(ax, psi * 0.99, C_TXT)

    ax.text(phi[0] * 1.18, phi[1] * 1.12, r"$\mathbf{V}_I[:,k]$",
            ha="right", va="center", fontsize=11,
            color=C_IMG, fontweight="bold")
    ax.text(psi[0] * 1.18, psi[1] * 1.12, r"$\mathbf{V}_T[:,\pi(k)]$",
            ha="left", va="center", fontsize=11,
            color=C_TXT, fontweight="bold")

    # Annotate the recovery
    ax.text(-0.92, 1.40, r"$\mathbf{V}_I[:,k] \to [\mathbf{\Phi}_S]_{:,k}$",
            ha="left", va="center", fontsize=9, color=C_IMG)
    ax.text(0.92, 1.40, r"$\mathbf{V}_T[:,\pi(k)] \to [\mathbf{\Psi}_S]_{:,k}$",
            ha="right", va="center", fontsize=9, color=C_TXT)

    ax.text(0, -0.08,
            "물리적으로 나뉜 두 decoder가",
            ha="center", va="top", fontsize=9, color=C_NEU)
    ax.text(0, -0.22,
            "image 방향과 text 방향을 각각 따로 복원한다 (각도 보존)",
            ha="center", va="top", fontsize=9, color=C_NEU, style="italic")


def _draw_measurement(ax):
    _style(ax, r"(3) 측정 — matched-pair cosine = $\hat{\alpha}$")
    _unit(ax)

    phi, psi = _make_pair(90 + HALF, 90 - HALF)
    _arrow(ax, phi, C_IMG)
    _arrow(ax, psi, C_TXT)

    ax.text(phi[0] * 1.18, phi[1] * 1.12, r"$\mathbf{V}_I[:,k]$",
            ha="right", va="center", fontsize=11,
            color=C_IMG, fontweight="bold")
    ax.text(psi[0] * 1.18, psi[1] * 1.12, r"$\mathbf{V}_T[:,\pi(k)]$",
            ha="left", va="center", fontsize=11,
            color=C_TXT, fontweight="bold")

    _angle_arc(ax, r"$\angle$", color=C_ACC, r=0.45)

    # Big cosine readout
    ax.text(0, -0.08,
            r"$\rho_k = \cos(\mathbf{V}_I[:,k],  \mathbf{V}_T[:,\pi(k)])  \to  \alpha$",
            ha="center", va="top", fontsize=10, color=C_ACC)
    ax.text(0, -0.22,
            "co-firing이 강한 matched pair들의 cosine 중앙값이 $\\alpha$의 추정치",
            ha="center", va="top", fontsize=9, color=C_NEU, style="italic")


def make_fig(out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    _draw_assumption(axes[0])
    _draw_theorem1(axes[1])
    _draw_measurement(axes[2])

    fig.suptitle(
        "Diagnostic B — 왜 masked matched-pair cosine이 $\\alpha$의 proxy인가",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.04, wspace=0.08)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print(f"saved {out_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str,
                   default="outputs/theorem2_followup_15/fig_diagnostic_B_intuition.png")
    args = p.parse_args()
    make_fig(args.out)


if __name__ == "__main__":
    main()
