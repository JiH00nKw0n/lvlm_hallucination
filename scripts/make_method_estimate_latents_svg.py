#!/usr/bin/env python3
"""Generate SVG for the 'Estimate Latent Codes' subpanel of the method figure.

Two stacked 6x13 grids (z_tilde_I, z_tilde_T) with continuous activation
intensities. Same red/blue palette as method.pdf so the cells can be reused
in subsequent subpanels (similarity, matching, permute).
"""

from pathlib import Path

ROWS, COLS = 7, 13
CELL = 14          # cell side (px)
GAP = 4            # gap between cells
PAD = 10           # frame padding (top/bot)
PADX = 12          # frame padding (left/right)
BORDER = 1.6       # frame border width
LABEL_FONT = 22
LABEL_GAP = 8      # space between label baseline and frame top
PANEL_GAP = 22     # space between two panels
MARGIN = 22        # outer margin

GRID_W = COLS * CELL + (COLS - 1) * GAP                    # 13*14 + 12*4 = 230
GRID_H = ROWS * CELL + (ROWS - 1) * GAP                    # 6*14  + 5*4  = 104
FRAME_W = GRID_W + 2 * PADX                                # 254
FRAME_H = GRID_H + 2 * PAD                                 # 124

PANEL_H = LABEL_FONT + LABEL_GAP + FRAME_H                 # 22+8+124 = 154
TOTAL_W = FRAME_W + 2 * MARGIN                             # 254+44 = 298
TOTAL_H = 2 * PANEL_H + PANEL_GAP + 2 * MARGIN             # 2*154+22+44 = 374

RED = (184, 52, 52)
BLUE = (46, 87, 145)
FLOOR = 0.04


def mix(rgb, t):
    """Linear blend of base rgb toward white as t -> 0."""
    r = round(255 + (rgb[0] - 255) * t)
    g = round(255 + (rgb[1] - 255) * t)
    b = round(255 + (rgb[2] - 255) * t)
    return f"rgb({r},{g},{b})"


# Activation patterns: each column lists (row, intensity) pairs.
# Sparse but continuous; each column has 1-2 active cells.
zI = [
    [(1, 0.70), (3, 0.40)],
    [(2, 0.90), (6, 0.35)],
    [(0, 0.60), (4, 0.45)],
    [(5, 0.85), (1, 0.30)],
    [(3, 0.50), (6, 0.65)],
    [(4, 0.95), (0, 0.40)],
    [(2, 0.55), (5, 0.70)],
    [(1, 0.80), (6, 0.50)],
    [(3, 0.65)],
    [(5, 0.45), (0, 0.85)],
    [(4, 0.60), (2, 0.40)],
    [(6, 0.80), (3, 0.50)],
    [(1, 0.55), (5, 0.70)],
]
zT = [
    [(1, 0.55), (6, 0.40)],
    [(2, 0.70), (4, 0.40)],
    [(0, 0.85)],
    [(5, 0.60), (3, 0.35)],
    [(3, 0.85), (1, 0.50)],
    [(4, 0.60), (0, 0.30)],
    [(2, 0.40), (6, 0.75)],
    [(1, 0.95)],
    [(3, 0.50), (5, 0.40)],
    [(5, 0.55), (0, 0.60), (6, 0.30)],
    [(4, 0.80), (1, 0.30)],
    [(2, 0.70), (6, 0.55)],
    [(1, 0.40), (5, 0.95)],
]


def matrix(columns):
    M = [[FLOOR] * COLS for _ in range(ROWS)]
    for c, col in enumerate(columns):
        for r, t in col:
            M[r][c] = max(t, FLOOR)
    return M


def render_grid(matrix_rgb_pair, x0, y0):
    """Emit <rect> elements for a (ROWS x COLS) intensity matrix."""
    M, base_rgb = matrix_rgb_pair
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            x = x0 + c * (CELL + GAP)
            y = y0 + r * (CELL + GAP)
            fill = mix(base_rgb, M[r][c])
            out.append(
                f'  <rect x="{x:.2f}" y="{y:.2f}" width="{CELL}" height="{CELL}" '
                f'rx="2" ry="2" fill="{fill}"/>'
            )
    return "\n".join(out)


def render_panel(label_letter, columns, base_rgb, top_y):
    """Emit a labeled panel (label + frame + cells). Returns the bottom y."""
    cx = MARGIN + FRAME_W / 2
    label_y = top_y + LABEL_FONT - 4  # baseline
    frame_x = MARGIN
    frame_y = top_y + LABEL_FONT + LABEL_GAP
    grid_x = frame_x + PADX
    grid_y = frame_y + PAD
    M = matrix(columns)
    cells = render_grid((M, base_rgb), grid_x, grid_y)
    label = (
        f'<text x="{cx:.2f}" y="{label_y:.2f}" text-anchor="middle" '
        f'font-size="{LABEL_FONT}" fill="#1a1a1a">'
        f'<tspan font-style="italic">z&#x0303;</tspan>'
        f'<tspan font-size="14" dy="6">{label_letter}</tspan>'
        f'</text>'
    )
    frame = (
        f'<rect x="{frame_x:.2f}" y="{frame_y:.2f}" '
        f'width="{FRAME_W}" height="{FRAME_H}" '
        f'fill="#ffffff" stroke="#3a3a3a" stroke-width="{BORDER}"/>'
    )
    return label + "\n" + frame + "\n" + cells, frame_y + FRAME_H


def main():
    panel1, bot1 = render_panel("I", zI, RED, MARGIN)
    panel2, bot2 = render_panel("T", zT, BLUE, bot1 + PANEL_GAP - LABEL_FONT - LABEL_GAP + LABEL_FONT + LABEL_GAP)

    # The above is awkward — recompute cleanly:
    top1 = MARGIN
    panel1, bot1 = render_panel("I", zI, RED, top1)
    top2 = bot1 + PANEL_GAP
    panel2, bot2 = render_panel("T", zT, BLUE, top2)

    height = bot2 + MARGIN
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {TOTAL_W:.0f} {height:.0f}" width="{TOTAL_W:.0f}" height="{height:.0f}" font-family="'CMU Serif', 'Latin Modern Roman', 'Times New Roman', serif">
  <rect width="{TOTAL_W:.0f}" height="{height:.0f}" fill="#ffffff"/>
{panel1}
{panel2}
</svg>
'''
    out = Path("outputs/method/estimate_latents.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"saved {out}  ({TOTAL_W:.0f} x {height:.0f})")


if __name__ == "__main__":
    main()
