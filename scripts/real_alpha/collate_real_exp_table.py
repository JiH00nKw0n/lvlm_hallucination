"""Aggregate outputs/real_exp_v1/**/results.json into LaTeX tables.

Produces:
    <out_dir>/table1_imagenet.tex  — Recon / LinProbe / ZeroShot
    <out_dir>/table2_coco.tex      — Recon / I→T R@{1,5,10} / T→I R@{1,5,10}
    <out_dir>/summary.json         — flat dump of all numbers for reuse

Usage:
    python scripts/real_alpha/collate_real_exp_table.py \
        outputs/real_exp_v1 --out outputs/real_exp_v1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METHOD_ORDER = ["shared", "separated", "iso_align", "group_sparse", "ours"]
METHOD_LABELS = {
    "shared": "Shared SAE",
    "separated": "Separated SAE",
    "iso_align": "Iso-Energy Align",
    "group_sparse": "Group-Sparse",
    "ours": "Ours (post-hoc)",
}


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _collect(root: Path) -> dict:
    data: dict = {}
    for method in METHOD_ORDER:
        data[method] = {}
        for dataset in ("imagenet", "coco"):
            d = root / method / dataset
            data[method][dataset] = {
                "recon": _load_json(d / "recon.json"),
                "linprobe": _load_json(d / "linprobe.json"),
                "zeroshot": _load_json(d / "zeroshot.json"),
                "retrieval": _load_json(d / "retrieval.json"),
            }
    return data


def _fmt(v, digits=4, missing="--"):
    if v is None:
        return missing
    return f"{v:.{digits}f}"


def _render_imagenet(data) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Recon $\downarrow$ & Linear probe $\uparrow$ & Zero-shot $\uparrow$ \\",
        r"\midrule",
    ]
    for m in METHOD_ORDER:
        cell = data[m]["imagenet"]
        recon = cell["recon"] and cell["recon"].get("recon_error")
        lp = cell["linprobe"] and cell["linprobe"].get("accuracy")
        zs = cell["zeroshot"] and cell["zeroshot"].get("accuracy")
        lines.append(
            f"{METHOD_LABELS[m]} & {_fmt(recon)} & {_fmt(lp)} & {_fmt(zs)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _render_coco(data) -> str:
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & Recon $\downarrow$ & \multicolumn{3}{c}{I$\to$T $\uparrow$} & \multicolumn{3}{c}{T$\to$I $\uparrow$} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r" & & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 \\",
        r"\midrule",
    ]
    for m in METHOD_ORDER:
        cell = data[m]["coco"]
        recon = cell["recon"] and cell["recon"].get("recon_error")
        r = cell["retrieval"]
        i2t = (r or {}).get("I2T") or {}
        t2i = (r or {}).get("T2I") or {}
        cells = [
            _fmt(recon),
            _fmt(i2t.get("R@1")), _fmt(i2t.get("R@5")), _fmt(i2t.get("R@10")),
            _fmt(t2i.get("R@1")), _fmt(t2i.get("R@5")), _fmt(t2i.get("R@10")),
        ]
        lines.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=str, help="Root dir containing <method>/<dataset>/*.json")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = _collect(root)
    (out / "summary.json").write_text(json.dumps(data, indent=2))
    (out / "table1_imagenet.tex").write_text(_render_imagenet(data) + "\n")
    (out / "table2_coco.tex").write_text(_render_coco(data) + "\n")
    logger.info("wrote %s/table{1_imagenet,2_coco}.tex and summary.json", out)


if __name__ == "__main__":
    main()
