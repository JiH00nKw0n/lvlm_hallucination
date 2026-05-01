"""Render scripts/palette.py as a single SVG swatch sheet."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import VIBRANT_TONES  # noqa: E402

NAMES = [
    "Strawberry Red", "Pumpkin Spice", "Carrot Orange", "Atomic Tangerine",
    "Tuscan Sun", "Willow Green", "Seaweed", "Dark Cyan",
    "Blue Slate", "Cerulean",
]

OUT = Path("outputs/palette.svg")

SW = 90    # swatch width
SH = 90    # swatch height
PAD = 16   # padding
LABEL_H = 38

n = len(VIBRANT_TONES)
W = PAD * 2 + SW * n + PAD * (n - 1) // 2
H = PAD * 2 + SH + LABEL_H + 28


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="DejaVu Serif, serif">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    parts.append(
        f'<text x="{PAD}" y="{PAD + 14}" font-size="13" font-weight="700" '
        f'fill="#222">Vibrant Tones</text>'
    )
    y0 = PAD + 28
    for i, (hex_, name) in enumerate(zip(VIBRANT_TONES, NAMES)):
        x = PAD + i * (SW + PAD // 2)
        parts.append(
            f'<rect x="{x}" y="{y0}" width="{SW}" height="{SH}" '
            f'rx="6" ry="6" fill="{hex_}"/>'
        )
        cx = x + SW // 2
        ty = y0 + SH + 14
        parts.append(
            f'<text x="{cx}" y="{ty}" font-size="9" text-anchor="middle" '
            f'fill="#222">{name}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{ty + 12}" font-size="8.5" font-family="monospace" '
            f'text-anchor="middle" fill="#555">{hex_}</text>'
        )
    parts.append('</svg>')
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
