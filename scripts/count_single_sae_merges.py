"""Count Single-SAE 'merge' rate over ALL alive pairs (no C threshold).

For Single SAE trained on image+text COCO pairs, we run Hungarian matching
between img-side and txt-side activations of the SAME single dictionary.
A 'merge' is a pair where Hungarian assigned the same latent to both sides
(oi == oj), meaning that one dictionary atom serves the concept on both
modalities — i.e., the Single SAE collapsed the concept into one direction.

Outputs the raw counts: merged / total alive pairs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_alpha"))
import _bootstrap  # noqa: F401

from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from src.models.modeling_sae import TopKSAE

ONE_DIR = "outputs/real_alpha_followup_2/one_sae/final"
CACHE = "cache/clip_b32_coco"
ALIVE_RATE = 0.001  # firing-rate threshold used by render_collapse_v2.py


def main() -> None:
    dev = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {dev}", flush=True)

    ds = CachedClipPairsDataset(CACHE, split="train", l2_normalize=True)
    N = len(ds)
    img = torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
    txt = torch.stack([ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs])
    print(f"N={N}  img={tuple(img.shape)}  txt={tuple(txt.shape)}", flush=True)

    model = TopKSAE.from_pretrained(ONE_DIR).to(dev).eval()
    L = model.latent_size
    print(f"L={L}", flush=True)

    si = np.zeros(L, dtype=np.float64)
    st = np.zeros(L, dtype=np.float64)
    sii = np.zeros(L, dtype=np.float64)
    stt = np.zeros(L, dtype=np.float64)
    sit = np.zeros((L, L), dtype=np.float64)
    fi = np.zeros(L, dtype=np.int64)
    ft = np.zeros(L, dtype=np.int64)
    cnt = 0
    with torch.no_grad():
        for s in range(0, N, 2048):
            e = min(s + 2048, N)
            zi = (
                model(hidden_states=img[s:e].unsqueeze(1).to(dev),
                      return_dense_latents=True)
                .dense_latents.squeeze(1).cpu().double().numpy()
            )
            zt = (
                model(hidden_states=txt[s:e].unsqueeze(1).to(dev),
                      return_dense_latents=True)
                .dense_latents.squeeze(1).cpu().double().numpy()
            )
            B = zi.shape[0]
            si += zi.sum(0); st += zt.sum(0)
            sii += (zi * zi).sum(0); stt += (zt * zt).sum(0)
            sit += zi.T @ zt
            fi += (zi > 0).sum(0); ft += (zt > 0).sum(0)
            cnt += B
            if (s // 2048) % 20 == 0:
                print(f"  batch {s//2048}/{(N+2047)//2048}", flush=True)

    mi = si / cnt; mt = st / cnt
    var_i = sii / cnt - mi ** 2
    var_t = stt / cnt - mt ** 2
    cov = sit / cnt - np.outer(mi, mt)
    denom = np.sqrt(np.clip(var_i[:, None] * var_t[None, :], 1e-16, None))
    C = np.nan_to_num(cov / denom, nan=0.0)

    rate_i = fi / cnt
    rate_t = ft / cnt
    ai = np.where(rate_i > ALIVE_RATE)[0]
    at = np.where(rate_t > ALIVE_RATE)[0]
    both = np.intersect1d(ai, at)
    union = np.union1d(ai, at)
    print(
        f"alive_img={len(ai)}  alive_txt={len(at)}  "
        f"both={len(both)}  union={len(union)}",
        flush=True,
    )

    # Non-Hungarian view: every latent alive on BOTH modalities is "merged"
    # (a single dictionary atom serves both sides), no 1-1 constraint.
    print(
        f"\nNON-HUNGARIAN: alive-on-both / alive-union = "
        f"{len(both)}/{len(union)} = {len(both)/len(union)*100:.2f}%",
        flush=True,
    )
    print(
        f"NON-HUNGARIAN: alive-on-both / alive_img = "
        f"{len(both)}/{len(ai)} = {len(both)/len(ai)*100:.2f}%",
        flush=True,
    )
    print(
        f"NON-HUNGARIAN: alive-on-both / alive_txt = "
        f"{len(both)}/{len(at)} = {len(both)/len(at)*100:.2f}%",
        flush=True,
    )

    # Diagonal-C distribution on alive-both latents (for context)
    diag_C = np.array([C[k, k] for k in both])
    print("alive-on-both diag-C quantiles:",
          {q: float(np.quantile(diag_C, q)) for q in [0.1, 0.5, 0.9]})

    Csub = C[np.ix_(ai, at)]
    r, c = linear_sum_assignment(-Csub)
    oi = ai[r]
    oj = at[c]
    n_pairs = len(oi)
    self_mask = (oi == oj)
    n_self = int(self_mask.sum())
    cm = Csub[r, c]

    # Also break down by C threshold for context.
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4]:
        mask_thr = cm >= thr
        n_thr = int(mask_thr.sum())
        n_self_thr = int((self_mask & mask_thr).sum())
        rate = (n_self_thr / n_thr * 100.0) if n_thr else float("nan")
        print(f"  C>={thr:.1f}: merge {n_self_thr}/{n_thr} = {rate:.2f}%",
              flush=True)

    overall_rate = n_self / n_pairs * 100.0
    print(
        f"\nOVERALL (no C threshold): merged {n_self}/{n_pairs} "
        f"= {overall_rate:.3f}%   ({n_self}/{n_pairs})",
        flush=True,
    )

    out = Path("outputs/real_alpha_followup_2/single_sae_merge_count.json")
    import json
    out.write_text(json.dumps({
        "n_pairs_total_alive": n_pairs,
        "n_merged_self_match": n_self,
        "merge_rate_pct": overall_rate,
        "alive_img": int(len(ai)),
        "alive_txt": int(len(at)),
        "L": int(L),
        "alive_rate_threshold": ALIVE_RATE,
        "by_C_threshold": {
            f"{thr:.1f}": {
                "n_pairs": int((cm >= thr).sum()),
                "n_merged": int((self_mask & (cm >= thr)).sum()),
            }
            for thr in [0.0, 0.1, 0.2, 0.3, 0.4]
        },
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
