"""Render qualitative cross-modal steering examples as HTML.

For each concept, picks ONE common BASE image (shared across all 3 variants —
requires deterministic per-concept seeding in eval_cross_modal_steering.py)
and renders `[BASE | ours top-5 | iso_align top-5 | group_sparse top-5]` so
the visual narrative is "same off-concept input image; here's how each
method's steering takes you somewhere different."

Common base selection per concept: among the 100 shared base images, pick
the one whose `ours` top-5 hit count is closest to ours' per-concept median.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/Users/junekwon/Desktop/Projects/lvlm_hallucination")
CMS = ROOT / "outputs/real_exp_cc3m/cross_modal_steering"
OUT = CMS / "qualitative.html"

CONCEPTS = ["dog", "elephant", "pizza", "kite", "skateboard", "giraffe"]
VARIANTS = ["ours", "iso_align", "group_sparse"]
ALPHA = {"ours": 1.0, "iso_align": 1.0, "group_sparse": 1.0}
LABEL = {"ours": "Ours (Hungarian)",
         "iso_align": "Iso-Energy Alignment",
         "group_sparse": "Group-Sparse"}


def first_cap(captions, iid):
    for c in range(5):
        for sep in ("::", "_"):
            k = f"{iid}{sep}{c}"
            if k in captions:
                return captions[k].rstrip()
    return ""


def has_concept(captions, iid, c):
    pat = re.compile(r"\b" + re.escape(c) + r"s?\b")
    for ci in range(5):
        for sep in ("::", "_"):
            k = f"{iid}{sep}{ci}"
            if k in captions:
                if pat.search(captions[k].lower()):
                    return True
                break
    return False


def coco_url(iid: int) -> str:
    return f"coco_images/{iid}.jpg"


def main() -> None:
    img_dict = torch.load(
        ROOT / "cache/clip_b32_coco/image_embeddings.pt",
        map_location="cpu", weights_only=False,
    )
    splits = json.load(open(ROOT / "cache/clip_b32_coco/splits.json"))
    captions = json.load(open(ROOT / "cache/coco_karpathy_captions.json"))

    unique_iids = sorted(set(int(p[0]) for p in splits["test"]))
    img_t = torch.stack(
        [img_dict[i] if i in img_dict else img_dict[str(i)] for i in unique_iids],
        dim=0,
    )
    img_norm_np = (img_t / img_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)).numpy()

    caches = {}
    for v in VARIANTS:
        npz = np.load(CMS / v / "x_steered_cache.npz")
        meta = json.load(open(CMS / v / "x_steered_cache_meta.json"))
        caches[v] = {"embeds": npz["embeddings"], "base_rows": npz["base_rows"], "meta": meta}

    html = ['<!doctype html>', '<html><head><meta charset="utf-8">',
            '<title>Cross-modal SAE steering — qualitative</title>',
            '<style>',
            'body{font-family:DejaVu Serif,Georgia,serif;margin:24px;background:#fafafa;color:#222}',
            'h1{font-size:18px;margin-bottom:8px}',
            'h2{font-size:16px;margin-top:32px;border-bottom:2px solid #444;padding-bottom:4px}',
            '.meta{font-size:11px;color:#555;margin-bottom:12px}',
            '.row{display:flex;gap:8px;align-items:flex-start;margin-bottom:6px;flex-wrap:nowrap;overflow-x:auto}',
            '.thumb{width:104px;text-align:center;font-size:10px;flex:none}',
            '.thumb img{width:104px;height:104px;object-fit:cover;border:2px solid transparent;border-radius:3px}',
            '.thumb.base img{border-color:#444}',
            '.thumb.hit img{border-color:#2a7a2a}',
            '.thumb.miss img{border-color:#a02020}',
            '.cap{margin-top:3px;font-size:9px;line-height:1.15;height:48px;overflow:hidden}',
            '.iid{font-size:9px;color:#888;font-family:monospace;margin-top:2px}',
            '.score{font-size:9px;color:#666}',
            '.label{font-size:11px;font-weight:bold;margin:6px 0 4px}',
            '.divider{font-size:18px;align-self:center;margin:0 6px;color:#888}',
            '.variant-row{margin:6px 0;padding:6px 8px;background:white;border-radius:4px}',
            '.hit-rate{font-size:10px;color:#444;margin-left:8px}',
            'table{border-collapse:collapse;font-size:11px;margin-top:8px}',
            'th,td{border:1px solid #ccc;padding:3px 6px;text-align:left}',
            'th{background:#eee}',
            '</style></head><body>']
    html.append('<h1>Cross-modal SAE steering — qualitative retrieval examples</h1>')
    html.append('<p class="meta">CC3M-trained SAEs evaluated on COCO Karpathy test (5000 images, 25010 captions). '
                 'For each concept the BASE image is identical across all 3 methods (deterministic md5 per-concept seed); '
                 'after injecting α at the concept latent on the IMAGE side, top-5 nearest images '
                 '(by cosine in CLIP B/32 space) are shown.</p>')

    # mAP table
    html.append('<h2>Image-retrieval mAP @ matched effective α (78 concepts)</h2>')
    metrics_summary = []
    for v in VARIANTS:
        d = json.load(open(CMS / v / "summary.json"))
        pa = d["per_alpha"][f"{ALPHA[v]}"]
        metrics_summary.append((LABEL[v], ALPHA[v], pa["map_img"], pa["p10_img"], pa["preserve_mean"]))
    html.append('<table><tr><th>variant</th><th>raw α</th><th>mAP (image)</th><th>P@10 (image)</th><th>preserve</th></tr>')
    for label, a, m, p, pr in metrics_summary:
        html.append(f'<tr><td>{label}</td><td>{a:.2f}</td><td>{m:.4f}</td><td>{p:.4f}</td><td>{pr:.3f}</td></tr>')
    html.append('</table>')

    for concept in CONCEPTS:
        html.append(f'<h2>concept = "{concept}"</h2>')

        # Locate the (concept, alpha) record per variant
        rec = {}
        for v in VARIANTS:
            r = next((i for i, m in enumerate(caches[v]["meta"])
                      if m["concept"] == concept and abs(m["alpha"] - ALPHA[v]) < 1e-6), None)
            if r is None:
                break
            rec[v] = r
        if len(rec) != len(VARIANTS):
            html.append('<div class="meta">[skip — missing record in some variant]</div>')
            continue

        # Verify common base sets and intersect
        base_rows_per_v = {}
        for v in VARIANTS:
            br = caches[v]["base_rows"][rec[v]]
            br = br[br >= 0].astype(int)
            base_rows_per_v[v] = br
        common = set(base_rows_per_v[VARIANTS[0]].tolist())
        for v in VARIANTS[1:]:
            common &= set(base_rows_per_v[v].tolist())
        if not common:
            html.append(f'<div class="meta">[skip — no common base rows across variants for concept={concept!r}]</div>')
            continue
        common_arr = np.array(sorted(common), dtype=int)

        # Compute hit count for `ours` at each common base, pick the one nearest median
        ours_idx_by_br = {int(br): bi for bi, br in enumerate(caches["ours"]["base_rows"][rec["ours"]])
                          if br >= 0}
        ours_hits = {}
        for br in common_arr:
            bi = ours_idx_by_br[int(br)]
            x = caches["ours"]["embeds"][rec["ours"], bi]
            top5 = np.argsort(-(img_norm_np @ x))[:5]
            ours_hits[int(br)] = sum(has_concept(captions, unique_iids[int(r)], concept) for r in top5)
        med = float(np.median(list(ours_hits.values())))
        chosen_br = min(ours_hits, key=lambda b: (abs(ours_hits[b] - med), b))
        base_iid = unique_iids[chosen_br]

        html.append(
            f'<div class="meta">|common base set| = {len(common)} / 100 · '
            f'ours per-base hit median = {med:.1f}/5 · chosen base iid={base_iid} '
            f'(ours hits at this base = {ours_hits[chosen_br]}/5)</div>'
        )

        # Render BASE thumb once
        html.append('<div class="row">')
        html.append('<div class="thumb base">'
                     f'<img loading="lazy" src="{coco_url(base_iid)}" alt="{base_iid}">'
                     f'<div class="iid">iid={base_iid}</div>'
                     f'<div class="cap">BASE: {first_cap(captions, base_iid)}</div></div>')
        html.append('</div>')

        # Per variant: top-5 of the chosen common base
        for v in VARIANTS:
            bi = {int(br): k for k, br in enumerate(caches[v]["base_rows"][rec[v]]) if br >= 0}[chosen_br]
            x = caches[v]["embeds"][rec[v], bi]
            scores = img_norm_np @ x
            top5 = np.argsort(-scores)[:5]
            n_hits = sum(has_concept(captions, unique_iids[int(r)], concept) for r in top5)

            html.append('<div class="variant-row">')
            html.append(f'<div class="label">[{LABEL[v]}]'
                         f'<span class="hit-rate">α={ALPHA[v]} · top-5 hits = {n_hits}/5</span></div>')
            html.append('<div class="row">')
            html.append('<div class="divider">→</div>')
            for r in top5:
                iid = unique_iids[int(r)]
                cls = "thumb hit" if has_concept(captions, iid, concept) else "thumb miss"
                cap = first_cap(captions, iid)
                html.append(f'<div class="{cls}"><img loading="lazy" src="{coco_url(iid)}" alt="{iid}">'
                             f'<div class="iid">iid={iid}</div>'
                             f'<div class="score">cos={scores[r]:.3f}</div>'
                             f'<div class="cap">{cap}</div></div>')
            html.append('</div></div>')

    html.append('</body></html>')
    OUT.write_text("\n".join(html))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
