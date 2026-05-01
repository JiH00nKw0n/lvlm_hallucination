"""Schematic embedding space (paper-ready, clean vector).

A textbook-style 3D-suggestive schematic of a unit sphere with two cones
(image cone on the left = cerulean, text cone on the right = strawberry
red). Apex of each cone sits at the sphere center; the base sits on the
sphere surface, drawn as an ellipse (perspective view of the cone's
circular cross-section). Front rim = solid, back rim = dashed (textbook
convention for hidden edges). No labels, no numbers.

Output: outputs/embedding_space_illustration.{svg,pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, PathPatch
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import STRAWBERRY_RED, CERULEAN  # noqa: E402

OUT = Path("outputs/embedding_space_illustration.svg")


def add_cone(ax, axis_sign: int, color: str, *,
             R: float = 1.0, half_angle_deg: float = 38.0,
             perspective: float = 0.32,
             fill_alpha: float = 0.30,
             line_alpha: float = 0.95) -> None:
    """Draw a textbook-style 3D cone with apex at origin.

    `axis_sign` = -1 (opens left) or +1 (opens right). The cone's circular
    base sits on the sphere of radius `R`; we render it as an ellipse using
    `perspective` as the foreshortening factor of its (horizontal) minor
    axis. The hidden back half of the rim is dashed.
    """
    half = np.deg2rad(half_angle_deg)
    base_x = axis_sign * R * np.cos(half)
    base_r = R * np.sin(half)              # 3D base radius
    minor = base_r * perspective            # foreshortened horizontal radius
    major = base_r                          # vertical radius (full)

    # Outer wall endpoints (apex → (base_x, ±base_r))
    top = (base_x, +base_r)
    bot = (base_x, -base_r)

    # ---- Filled cone region: triangle ∪ front-half base ellipse.
    # Build a Path: apex → top rim → arc along front of base ellipse →
    # bot rim → back to apex. The "front" of the base (closer to viewer
    # in this 2D schematic projection) is the half on the OPEN side of the
    # cone (i.e. away from the sphere center along the cone axis).
    front_sign = axis_sign           # +1 for right cone, -1 for left cone
    n_arc = 60
    # front-half ellipse param: theta from +pi/2 → -pi/2 going through
    # x = base_x + front_sign * minor.
    th = np.linspace(np.pi / 2, -np.pi / 2, n_arc)
    arc_x = base_x + front_sign * minor * np.cos(th)
    arc_y = major * np.sin(th)
    verts = [(0.0, 0.0), top]
    for x, y in zip(arc_x, arc_y):
        verts.append((x, y))
    verts.append(bot)
    verts.append((0.0, 0.0))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes),
                            facecolor=color, edgecolor="none",
                            alpha=fill_alpha, zorder=3))

    # ---- Cone wall lines (crisp, on top of the fill).
    ax.plot([0.0, top[0]], [0.0, top[1]],
             color=color, lw=1.4, alpha=line_alpha, zorder=5)
    ax.plot([0.0, bot[0]], [0.0, bot[1]],
             color=color, lw=1.4, alpha=line_alpha, zorder=5)

    # ---- Base rim: front half solid, back half dashed.
    th_full = np.linspace(0.0, 2.0 * np.pi, 200)
    rim_x = base_x + minor * np.cos(th_full)
    rim_y = major * np.sin(th_full)
    front_mask = (np.cos(th_full) * front_sign) >= 0
    ax.plot(rim_x[front_mask], rim_y[front_mask],
             color=color, lw=1.4, alpha=line_alpha, zorder=5)
    ax.plot(rim_x[~front_mask], rim_y[~front_mask],
             color=color, lw=1.0, alpha=line_alpha * 0.85,
             ls=(0, (4, 2.5)), zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor="white")

    R = 1.0

    # Cones (drawn first so the sphere outline overlays cleanly).
    add_cone(ax, axis_sign=-1, color=CERULEAN,       R=R)
    add_cone(ax, axis_sign=+1, color=STRAWBERRY_RED, R=R)

    # Sphere outline + equator (textbook 3D-suggestion). One dashed equator
    # is enough — adding the meridian made the figure look cluttered.
    ax.add_patch(Circle((0, 0), R, fill=False, ec="#2b2b2b", lw=1.6, zorder=6))
    ax.add_patch(Ellipse((0, 0), width=2 * R, height=2 * R * 0.32,
                          fill=False, ec="#cfcfcf", lw=0.8,
                          ls=(0, (4, 3.5)), zorder=4))

    # Apex dot (small, neutral).
    ax.plot([0], [0], marker="o", color="#222", ms=2.5, zorder=7)

    # Cosmetic
    pad = 0.18
    ax.set_xlim(-R - pad, R + pad)
    ax.set_ylim(-R - pad, R + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".pdf", ".png"):
        out = OUT.with_suffix(ext)
        fig.savefig(out, bbox_inches="tight", facecolor="white",
                    pad_inches=0.05,
                    dpi=200 if ext == ".png" else None)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
