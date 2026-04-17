"""Render HTML comparing collapsed (merged) single SAE pairs vs their two SAE counterparts."""
from __future__ import annotations
import base64, io, json, sys
from pathlib import Path
from PIL import Image
from datasets import load_dataset

def img_to_b64(pil_img, height=100):
    w, h = pil_img.size
    pil_img = pil_img.resize((int(w * height / h), height), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def main():
    data = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]

    hf = load_dataset("namkha1032/coco-karpathy", split="train")
    id_to_row = {}
    for i in range(len(hf)):
        iid = str(hf[i]["image_id"])
        if iid not in id_to_row:
            id_to_row[iid] = i
    print(f"built id_to_row: {len(id_to_row)}")

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:'Helvetica Neue',sans-serif;margin:20px}",
        ".pair{border:1px solid #ddd;padding:12px;margin-bottom:24px;border-radius:6px}",
        ".row{margin-top:8px}",
        ".col{background:#f9f9f9;padding:8px;border-radius:4px;margin-bottom:8px}",
        ".col h4{margin:0 0 6px 0;font-size:13px}",
        ".imgs{display:flex;flex-wrap:wrap;gap:4px}",
        ".imgs img{height:100px;border-radius:3px}",
        ".meta{font-size:12px;color:#666}",
        "h1{font-size:18px} h2{font-size:15px;border-bottom:1px solid #ccc;padding-bottom:4px}",
        "</style></head><body>",
        "<h1>Collapsed (merged) pairs: Single SAE vs Two SAE (L=4096)</h1>",
        "<p class='meta'>Single SAE에서 cos=1.0 (self-match, merged) 인 latent pair와, 같은 concept을 서빙하는 Two SAE pair의 비교. 매칭 기준: top-50 activating sample의 Jaccard overlap.</p>",
    ]

    matched = [d for d in data if d["jaccard"] >= 0.1]
    matched.sort(key=lambda d: -d["jaccard"])

    for rank, d in enumerate(matched):
        html.append(f"<div class='pair'>")
        html.append(f"<h2>#{rank+1} — Jaccard={d['jaccard']:.3f}, shared samples={d['n_shared']}</h2>")
        html.append(f"<div class='row'>")

        # Single SAE side
        html.append(f"<div class='col'>")
        html.append(f"<h4>Single SAE: latent #{d['single_lat']} (self-match)</h4>")
        html.append(f"<p class='meta'>C={d['C_one']:.3f}, cos=<b>{d['cos_one']:.4f}</b> (merged → bisector)</p>")
        html.append(f"</div>")

        # Two SAE side
        html.append(f"<div class='col'>")
        if d["two_pair"]:
            html.append(f"<h4>Two SAE: Image #{d['two_pair'][0]}, Text #{d['two_pair'][1]}</h4>")
            html.append(f"<p class='meta'>C={d['C_two']:.3f}, cos=<b>{d['cos_two']:.4f}</b> (separate decoders)</p>")
        else:
            html.append(f"<h4>Two SAE: no match found</h4>")
        html.append(f"</div>")
        html.append(f"</div>")  # row

        # Shared images
        if d["top5_imgs"]:
            html.append(f"<p class='meta'>Shared top-activating images:</p>")
            html.append(f"<div class='imgs'>")
            for iid in d["top5_imgs"][:5]:
                row_idx = id_to_row.get(str(iid))
                if row_idx is None:
                    continue
                row = hf[row_idx]
                pil = row["image"].convert("RGB")
                b64 = img_to_b64(pil)
                cap = row["captions"][0] if row["captions"] else ""
                html.append(f"<div style='text-align:center;max-width:180px'>"
                            f"<img src='data:image/jpeg;base64,{b64}'>"
                            f"<div style='font-size:10px;color:#888'>{iid}<br>{cap[:80]}</div></div>")
            html.append(f"</div>")
        html.append(f"</div>")

    # Summary
    if matched:
        cos_twos = [d["cos_two"] for d in matched]
        import numpy as np
        html.append(f"<hr><p><b>Summary</b>: {len(matched)} collapsed concepts matched. "
                     f"Single cos=1.000 (all merged). "
                     f"Two SAE cos: mean={np.mean(cos_twos):.3f}, median={np.median(cos_twos):.3f}. "
                     f"Collapse rate = 16/236 = 6.8% (C≥0.2 pairs).</p>")

    html.append("</body></html>")
    Path(out_path).write_text("\n".join(html), encoding="utf-8")
    print(f"saved {out_path} ({len(''.join(html))//1024} KB)")

if __name__ == "__main__":
    main()
