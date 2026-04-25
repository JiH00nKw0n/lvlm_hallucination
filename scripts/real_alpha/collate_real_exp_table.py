"""Aggregate outputs/real_exp_v1/**/results.json into tables + Markdown.

Produces:
    <out_dir>/table1_imagenet.tex          — ImageNet in-domain (Recon/LinProbe/ZeroShot + alive)
    <out_dir>/table2_coco.tex              — COCO in-domain (Recon/retrieval + tie + alive)
    <out_dir>/table3_cross.tex             — COCO→ImageNet cross (LinProbe/ZeroShot)
    <out_dir>/summary.json                 — flat dump of all numbers
    <out_dir>/RESULTS.md                   — consolidated Markdown report

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

METHOD_ORDER = ["shared", "separated", "iso_align", "group_sparse", "ours", "vl_sae"]
METHOD_LABELS = {
    "shared": "Shared SAE",
    "separated": "Separated SAE",
    "iso_align": "Iso-Energy Align",
    "group_sparse": "Group-Sparse",
    "ours": "Post-hoc Align (Ours)",
    "vl_sae": "VL-SAE",
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
        for subdir in ("imagenet", "coco", "coco_to_imagenet", "imagenet_to_coco"):
            d = root / method / subdir
            if not d.exists():
                continue
            data[method][subdir] = {
                "recon": _load_json(d / "recon.json"),
                "linprobe": _load_json(d / "linprobe.json"),
                "zeroshot": _load_json(d / "zeroshot.json"),
                "retrieval": _load_json(d / "retrieval.json"),
                "dead_latents": _load_json(d / "dead_latents.json"),
            }
    return data


def _fmt(v, digits=4, missing="--"):
    if v is None:
        return missing
    if isinstance(v, int):
        return str(v)
    return f"{v:.{digits}f}"


def _pct(v, digits=2, missing="--"):
    if v is None:
        return missing
    return f"{100*v:.{digits}f}"


def _alive_frac(cell: dict | None, side: str) -> float | None:
    if not cell:
        return None
    dl = cell.get("dead_latents")
    if not dl:
        return None
    alive = dl.get(f"alive_{side}_count")
    L = dl.get(f"latent_size_{side}")
    if alive is None or not L:
        return None
    return alive / L


def _render_imagenet(data) -> str:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Recon $\downarrow$ & Lin.~probe $\uparrow$ & Zero-shot $\uparrow$ & Alive (img) & Alive (txt) \\",
        r"\midrule",
    ]
    for m in METHOD_ORDER:
        cell = data[m].get("imagenet")
        if not cell:
            continue
        recon = cell["recon"] and cell["recon"].get("recon_error")
        lp = cell["linprobe"] and cell["linprobe"].get("accuracy")
        zs = cell["zeroshot"] and cell["zeroshot"].get("accuracy")
        ai = _alive_frac(cell, "image")
        at = _alive_frac(cell, "text")
        lines.append(
            f"{METHOD_LABELS[m]} & {_fmt(recon)} & {_fmt(lp)} & {_fmt(zs)} & {_fmt(ai)} & {_fmt(at)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _render_coco(data) -> str:
    lines = [
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Method & Recon $\downarrow$ & \multicolumn{3}{c}{I$\to$T $\uparrow$} & \multicolumn{3}{c}{T$\to$I $\uparrow$} & Tie(T$\to$I) & Alive(img) & Alive(txt) \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r" & & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 & mean size & & \\",
        r"\midrule",
    ]
    for m in METHOD_ORDER:
        cell = data[m].get("coco")
        if not cell:
            continue
        recon = cell["recon"] and cell["recon"].get("recon_error")
        r = cell["retrieval"]
        i2t = (r or {}).get("I2T") or {}
        t2i = (r or {}).get("T2I") or {}
        t2i_ties = (r or {}).get("T2I_ties") or {}
        ai = _alive_frac(cell, "image")
        at = _alive_frac(cell, "text")
        cells = [
            _fmt(recon),
            _fmt(i2t.get("R@1")), _fmt(i2t.get("R@5")), _fmt(i2t.get("R@10")),
            _fmt(t2i.get("R@1")), _fmt(t2i.get("R@5")), _fmt(t2i.get("R@10")),
            _fmt(t2i_ties.get("mean_tie_size"), digits=1),
            _fmt(ai), _fmt(at),
        ]
        lines.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _render_cross(data) -> str:
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Lin.~probe $\uparrow$ & Zero-shot $\uparrow$ \\",
        r"\midrule",
    ]
    for m in METHOD_ORDER:
        cell = data[m].get("coco_to_imagenet")
        if not cell:
            continue
        lp = cell["linprobe"] and cell["linprobe"].get("accuracy")
        zs = cell["zeroshot"] and cell["zeroshot"].get("accuracy")
        lines.append(
            f"{METHOD_LABELS[m]} & {_fmt(lp)} & {_fmt(zs)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _md_imagenet(data) -> str:
    header = (
        "| Method | Recon ↓ | Lin.probe ↑ | Zero-shot ↑ | Alive (img) | Alive (txt) |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for m in METHOD_ORDER:
        cell = data[m].get("imagenet")
        if not cell:
            continue
        recon = cell["recon"] and cell["recon"].get("recon_error")
        lp = cell["linprobe"] and cell["linprobe"].get("accuracy")
        zs = cell["zeroshot"] and cell["zeroshot"].get("accuracy")
        ai = _alive_frac(cell, "image")
        at = _alive_frac(cell, "text")
        rows.append(
            f"| {METHOD_LABELS[m]} | {_fmt(recon)} | {_pct(lp)} | {_pct(zs)} | "
            f"{_pct(ai, digits=1)} | {_pct(at, digits=1)} |"
        )
    return header + "\n".join(rows)


def _md_coco(data) -> str:
    header = (
        "| Method | Recon ↓ | I→T R@1 | R@5 | R@10 | T→I R@1 | R@5 | R@10 | T→I tie | Alive (img) | Alive (txt) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for m in METHOD_ORDER:
        cell = data[m].get("coco")
        if not cell:
            continue
        recon = cell["recon"] and cell["recon"].get("recon_error")
        r = cell["retrieval"]
        i2t = (r or {}).get("I2T") or {}
        t2i = (r or {}).get("T2I") or {}
        t2i_ties = (r or {}).get("T2I_ties") or {}
        ai = _alive_frac(cell, "image")
        at = _alive_frac(cell, "text")
        rows.append(
            f"| {METHOD_LABELS[m]} | {_fmt(recon)} | "
            f"{_pct(i2t.get('R@1'))} | {_pct(i2t.get('R@5'))} | {_pct(i2t.get('R@10'))} | "
            f"{_pct(t2i.get('R@1'))} | {_pct(t2i.get('R@5'))} | {_pct(t2i.get('R@10'))} | "
            f"{_fmt(t2i_ties.get('mean_tie_size'), digits=0)} | "
            f"{_pct(ai, digits=1)} | {_pct(at, digits=1)} |"
        )
    return header + "\n".join(rows)


def _md_cross(data) -> str:
    header = (
        "| Method | Lin.probe (ImageNet) ↑ | Zero-shot (ImageNet) ↑ |\n"
        "|---|---:|---:|\n"
    )
    rows = []
    for m in METHOD_ORDER:
        cell = data[m].get("coco_to_imagenet")
        if not cell:
            continue
        lp = cell["linprobe"] and cell["linprobe"].get("accuracy")
        zs = cell["zeroshot"] and cell["zeroshot"].get("accuracy")
        rows.append(
            f"| {METHOD_LABELS[m]} | {_pct(lp)} | {_pct(zs)} |"
        )
    return header + "\n".join(rows)


def _md_params(data) -> str:
    """Render parameter counts if we can find them saved, else placeholder."""
    header = "| Method | Latent | Params |\n|---|---:|---:|\n"
    rows = []
    for m in METHOD_ORDER:
        # Look for any saved recon/linprobe to extract latent_size
        L_img = L_txt = None
        for ds in ("coco", "imagenet"):
            c = data[m].get(ds)
            if not c:
                continue
            dl = c.get("dead_latents")
            if dl:
                L_img = dl.get("latent_size_image")
                L_txt = dl.get("latent_size_text")
                break
        L_display = (
            f"{L_img}/{L_txt}" if L_img and L_txt and L_img != L_txt
            else (f"{L_img}" if L_img else "--")
        )
        rows.append(f"| {METHOD_LABELS[m]} | {L_display} | -- |")
    return header + "\n".join(rows)


def _render_markdown(data) -> str:
    return f"""# Real-Data SAE Experiment Results (CLIP ViT-B/32)

Consolidated results for 6 methods × in-domain (COCO, ImageNet) + cross-domain (COCO → ImageNet) evaluation.

## Setup

- **Backbone**: CLIP ViT-B/32 (frozen), L2-normalized embeddings (d=512)
- **Training data**: COCO Karpathy train (566k pairs) / ImageNet-1K train (max 1000 images/class, 80 OpenAI templates)
- **Eval data**: COCO Karpathy test (5000 images × 5 captions) / ImageNet-1K val (50000 images)
- **SAE sparsity**: k=8 (both datasets)
- **Training**: 30 epochs, batch=1024, AdamW lr=5e-4 (cosine, warmup 5%), wd=1e-5, seed=0
- **Latent sizes**: L=8192 for TopKSAE variants; L=4096 for VL-SAE (param-matched with Ours per-side)
- **Linear probe**: torch Adam on image-side latents, 30 epochs, `--eval-topk 1` (inference-time top-1 sparsity)
- **Retrieval ranking**: pessimistic tie handling (`ge` - 1); `tie_at_gt_rate` / `mean_tie_size` reported as diagnostic

## Methods

1. **Shared SAE**: single TopKSAE, recon loss only
2. **Separated SAE**: TwoSidedTopKSAE (independent image/text SAEs), per-side recon
3. **Iso-Energy Align**: single TopKSAE + `-β·cos(z_I, z_T)` auxiliary (masked top-1)
4. **Group-Sparse**: single TopKSAE + `λ·‖(z_I, z_T)‖_{{2,1}}` group-L2 auxiliary
5. **Post-hoc Align (Ours)**: Separated SAE ckpt + Hungarian matching to align text slots to image slots (no re-training, dead-latent-filtered assignment)
6. **VL-SAE**: shared distance-based encoder + two modality-specific decoders (Shen et al., NeurIPS 2025)

## Parameter Counts

{_md_params(data)}

---

## Table 1. In-domain — ImageNet-1K (k=8)

Training: ImageNet-1K class-template pairs.
Evaluation: ImageNet-1K val (50000 images).

{_md_imagenet(data)}

*Notes*: Recon is per-sample squared error averaged over the val split (lower is better). Linear probe is torch-trained on image-side SAE latents with inference-time top-1 sparsity (val top-1 accuracy). Zero-shot uses text-prototype ↔ image-latent cosine matching. Alive = fraction of slots firing at least once on 50k paired train samples.

---

## Table 2. In-domain — COCO Karpathy (k=8)

Training: COCO Karpathy train captions + images.
Evaluation: COCO Karpathy test (5000 images × 25010 captions).

{_md_coco(data)}

*Notes*: Retrieval uses pessimistic ranking (ties count against gt). `T→I tie` = mean number of images scoring identically with the gt for each caption — a diagnostic of how informative the cosines are. Methods with ties > 1000 have essentially uninformative retrieval; only Ours and VL-SAE produce real ranking signal.

---

## Table 3. Cross-domain — COCO → ImageNet

Training: COCO Karpathy (same ckpts as Table 2).
Evaluation: ImageNet-1K val, using COCO-trained SAE features.

{_md_cross(data)}

*Notes*: Measures transfer of COCO-learned SAE features to ImageNet classification. Ours rebuilds Hungarian perm on ImageNet train pairs (using COCO-trained Separated weights). Other methods use the COCO-trained ckpt as-is.

---

## Discussion

(TODO — fill in manually after results are final)

- VL-SAE vs Ours at matched capacity (L=4096)
- In-domain vs cross-domain performance gap
- Tie-size indicates which methods have informative cosines
- Dead-latent fraction across methods (group-sparse tends to use more slots)

## Limitations

- Single seed. Seed sweep deferred to camera-ready.
- CLIP ViT-B/32 only. Other VLMs (DataComp, MetaCLIP, SigLIP) as appendix.
- No steering experiment in this iteration.

## Artifacts

- LaTeX tables: `table{{1_imagenet,2_coco,3_cross}}.tex` in this directory.
- Full JSON dump: `summary.json`.
- Per-method outputs under `<method>/{{imagenet,coco,coco_to_imagenet}}/`.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=str, help="Root dir containing <method>/<subdir>/*.json")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = _collect(root)
    (out / "summary.json").write_text(json.dumps(data, indent=2))
    (out / "table1_imagenet.tex").write_text(_render_imagenet(data) + "\n")
    (out / "table2_coco.tex").write_text(_render_coco(data) + "\n")
    (out / "table3_cross.tex").write_text(_render_cross(data) + "\n")
    (out / "RESULTS.md").write_text(_render_markdown(data))
    logger.info(
        "wrote %s/{table1_imagenet,table2_coco,table3_cross}.tex, summary.json, RESULTS.md",
        out,
    )


if __name__ == "__main__":
    main()
