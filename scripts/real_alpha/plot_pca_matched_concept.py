#!/usr/bin/env python3
"""PCA visualization of one Hungarian-matched image-caption pair.

Pick a single (image, caption) pair from the COCO train split whose TopK
Two-sided SAE reconstruction uses >= 3 Hungarian-matched shared concepts
(high correlation + high decoder cosine), and whose full TopK
reconstruction error is low. Visualize in 2D PCA:

  - image CLIP embedding x, text CLIP embedding y (dots)
  - 3 image-side decoder directions W_I[:, i_k] (warm arrows)
  - 3 text-side decoder directions W_T[:, pi(i_k)] (cool arrows)

Below the PCA panel: the actual image thumbnail + caption string.

Usage:
    python scripts/real_alpha/plot_pca_matched_concept.py \\
        --run-dir outputs/real_alpha_followup_1/two_sae/final \\
        --cache-dir cache/clip_b32_coco \\
        --out outputs/pca_matched_concept.pdf
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore
from src.models.configuration_sae import TwoSidedTopKSAEConfig  # type: ignore
from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore


# Project palette (scripts/palette.py): pick 1 color per modality,
# shade by lightness for 3 concepts (dark → base → light).
import colorsys
from matplotlib.colors import to_rgb


def _shade(hex_color: str, lightness: float) -> tuple:
    r, g, b = to_rgb(hex_color)
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, min(1.0, lightness)), s)


STRAWBERRY_RED = "#f94144"
CERULEAN = "#277da1"

# Three shades with equal spacing (step 0.18) for clear visual separation.
IMG_COLORS = [_shade(STRAWBERRY_RED, 0.30),  # darkest
              _shade(STRAWBERRY_RED, 0.48),  # mid
              _shade(STRAWBERRY_RED, 0.66)]  # lightest
TXT_COLORS = [_shade(CERULEAN, 0.20),  # darkest
              _shade(CERULEAN, 0.38),  # mid
              _shade(CERULEAN, 0.56)]  # lightest
EMB_IMG_COLOR = _shade(STRAWBERRY_RED, 0.38)
EMB_TXT_COLOR = _shade(CERULEAN, 0.28)


def load_model(run_dir: Path, device: torch.device) -> TwoSidedTopKSAE:
    state = load_file(str(run_dir / "model.safetensors"))
    cfg_path = run_dir.parent / "config.json"
    user_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    per_side = state["image_sae.W_dec"].shape[0]
    hidden = state["image_sae.W_dec"].shape[1]
    k = int(user_cfg.get("k", 8))
    cfg = TwoSidedTopKSAEConfig(hidden_size=hidden, latent_size=per_side * 2, k=k, normalize_decoder=True)
    model = TwoSidedTopKSAE(cfg).to(device).eval()
    model.load_state_dict(state, strict=False)
    return model


def alive_hungarian(run_dir: Path):
    """Return (match_arr, C_diag, alive_i, alive_t) for alive-restricted Hungarian."""
    C = np.load(run_dir / "diagnostic_B_C_train.npy")
    rates = np.load(run_dir / "diagnostic_B_firing_rates.npz")
    alive_i = np.where(rates["rate_i"] > 0.001)[0]
    alive_t = np.where(rates["rate_t"] > 0.001)[0]
    C_sub = C[np.ix_(alive_i, alive_t)]
    row_ind, col_ind = linear_sum_assignment(-C_sub)
    # match_arr[i] = pi(i) for i in alive_i[row_ind]; else -1
    n_img = C.shape[0]
    match = np.full(n_img, -1, dtype=np.int64)
    match[alive_i[row_ind]] = alive_t[col_ind]
    # C value at each matched pair
    C_vals = np.full(n_img, 0.0, dtype=np.float64)
    C_vals[alive_i[row_ind]] = C_sub[row_ind, col_ind]
    return match, C_vals, alive_i, alive_t, C


def compute_mstar_mask(
    match: np.ndarray, C_vals: np.ndarray,
    W_i: np.ndarray, W_t: np.ndarray,
    c_min: float, cos_min: float,
) -> np.ndarray:
    """Boolean mask (length n_img) marking img-slots whose matched pair passes quality thresholds."""
    W_i_n = W_i / (np.linalg.norm(W_i, axis=1, keepdims=True) + 1e-12)
    W_t_n = W_t / (np.linalg.norm(W_t, axis=1, keepdims=True) + 1e-12)
    mask = np.zeros_like(match, dtype=bool)
    for i in range(match.shape[0]):
        j = int(match[i])
        if j < 0:
            continue
        if C_vals[i] < c_min:
            continue
        dec_cos = float((W_i_n[i] * W_t_n[j]).sum())
        if dec_cos < cos_min:
            continue
        mask[i] = True
    return mask


def stream_candidates(
    model: TwoSidedTopKSAE, ds: CachedClipPairsDataset,
    mstar_mask_img: np.ndarray, match: np.ndarray,
    min_matched: int, top_m: int, batch_size: int, device: torch.device,
):
    """Streaming forward: maintain top-M pairs (by smallest E(n)) with |S(n)| >= min_matched.

    Returns list of dicts [{"pair_idx": n, "E": float, "shared": list[(i, pi(i))], ...}, ...].
    """
    k = int(model.image_sae.cfg.k)
    N = len(ds)
    # Lookup: img_slot -> pi(img_slot) or -1
    match_t = torch.from_numpy(match).to(device)
    mstar_img = torch.from_numpy(mstar_mask_img).to(device)
    # Heap of (neg_E, idx, pair_idx, shared_pairs)  — using min-heap of -E to track smallest-E candidates with cap top_m:
    # Simpler: use heapq with (E, counter, pair_idx, ...) and keep only top-M smallest. Use min-heap of NEGATIVE-E to pop largest when over cap.
    cap = top_m
    # use (-E, ...) so max-heap by E; when len > cap, pop the worst (largest E)
    heap: list = []
    counter = 0

    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            # Load batch embeddings
            imgs = torch.stack([ds[i]["image_embeds"] for i in range(s, e)]).to(device, dtype=torch.float32)
            txts = torch.stack([ds[i]["text_embeds"] for i in range(s, e)]).to(device, dtype=torch.float32)
            # Forward through each sub-SAE with return_dense_latents=False to get latent_indices+values.
            out_i = model.image_sae(hidden_states=imgs.unsqueeze(1), return_dense_latents=False)
            out_t = model.text_sae(hidden_states=txts.unsqueeze(1), return_dense_latents=False)
            # latent_indices: (B, 1, k), latent_activations: (B, 1, k)
            idx_i = out_i.latent_indices.squeeze(1)  # (B, k)
            idx_t = out_t.latent_indices.squeeze(1)  # (B, k)
            val_i = out_i.latent_activations.squeeze(1)  # (B, k)
            val_t = out_t.latent_activations.squeeze(1)  # (B, k)
            # Reconstruct x_hat, y_hat (for E(n)):
            W_i = model.image_sae.W_dec  # (n_img, d)
            W_t = model.text_sae.W_dec
            # dense -> gather reconstruction:  x_hat[b] = sum_k val_i[b,k] * W_i[idx_i[b,k]]
            #   = einsum
            x_hat = torch.einsum("bk,bkd->bd", val_i, W_i[idx_i])  # (B, d)
            y_hat = torch.einsum("bk,bkd->bd", val_t, W_t[idx_t])
            E = ((imgs - x_hat) ** 2).sum(dim=1) + ((txts - y_hat) ** 2).sum(dim=1)  # (B,)

            # For each pair b in batch, find shared-active matched pairs.
            # shared_i slots = img-side active indices where mstar_img[slot] AND match[slot] is in text-side active.
            B = imgs.shape[0]
            # active img slots per row (B, k). Check if in mstar.
            mstar_at_active_i = mstar_img[idx_i]  # (B, k) bool
            # For each (b, k) with mstar_at_active_i, find pi(slot) and check if in idx_t[b]:
            pi_at_active = match_t[idx_i]  # (B, k) int
            # Text-side: build boolean (B, n_img) on demand -- too big (B × 4096 × 4B = 32MB/batch). Use set comparison per-row instead.
            # Faster: compare pi_at_active (B, k) against idx_t (B, k). We need for each (b, ki), whether pi_at_active[b, ki] is in idx_t[b, :].
            # pi_at_active.unsqueeze(-1) == idx_t.unsqueeze(1) → (B, k, k) bool.
            match_mat = (pi_at_active.unsqueeze(-1) == idx_t.unsqueeze(1))  # (B, k_i, k_t)
            has_match = match_mat.any(dim=-1)  # (B, k_i) bool
            shared_active = mstar_at_active_i & has_match  # (B, k_i) bool
            n_shared = shared_active.sum(dim=1).cpu().numpy()  # (B,)

            for bi in range(B):
                if int(n_shared[bi]) < min_matched:
                    continue
                e_val = float(E[bi].item())
                shared_i_slots = idx_i[bi][shared_active[bi]].cpu().numpy().tolist()
                shared_pi = [int(match[ii]) for ii in shared_i_slots]
                # activation strengths (for top-3 selection later)
                k_i_pos = shared_active[bi].nonzero(as_tuple=False).flatten()
                val_i_shared = val_i[bi][k_i_pos].cpu().numpy().tolist()
                # text-side activation for pi slot (match back)
                val_t_shared: list[float] = []
                for ii in shared_i_slots:
                    pi_slot = int(match[ii])
                    t_pos = (idx_t[bi] == pi_slot).nonzero(as_tuple=False)
                    if t_pos.numel() == 0:
                        val_t_shared.append(0.0)
                    else:
                        val_t_shared.append(float(val_t[bi][t_pos[0, 0].item()].item()))

                counter += 1
                entry = {
                    "pair_idx": int(s + bi),
                    "E": e_val,
                    "n_shared": int(n_shared[bi]),
                    "shared_i": shared_i_slots,
                    "shared_pi": shared_pi,
                    "val_i": val_i_shared,
                    "val_t": val_t_shared,
                }
                if len(heap) < cap:
                    heapq.heappush(heap, (-e_val, counter, entry))
                else:
                    if -e_val > heap[0][0]:  # smaller E than current worst
                        heapq.heapreplace(heap, (-e_val, counter, entry))
    # Sort by E ascending
    return sorted([h[2] for h in heap], key=lambda e: e["E"])


_CAP_CACHE: dict = {"loaded": False, "rows": {}}


def _load_caption_cache(caption_json_path: Path) -> None:
    """Load JSON cache of {'<image_id>::<caption_idx>': caption_string}."""
    if _CAP_CACHE["loaded"]:
        return
    _CAP_CACHE["loaded"] = True
    if not caption_json_path.exists():
        print(f"Warning: caption cache not found at {caption_json_path}")
        return
    with open(caption_json_path) as f:
        data = json.load(f)
    _CAP_CACHE["rows"] = data
    print(f"caption cache: {len(data)} entries loaded")


def _coco_image_from_url(image_id: int) -> Image.Image | None:
    """Fetch COCO image via canonical URL (train2014/val2014)."""
    import urllib.request
    for split in ("train2014", "val2014"):
        url = f"http://images.cocodataset.org/{split}/COCO_{split}_{int(image_id):012d}.jpg"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return Image.open(BytesIO(resp.read()))
        except Exception:
            continue
    return None


def fetch_caption(image_id: int, caption_idx: int) -> str:
    """Return caption text from JSON cache, or empty string."""
    key = f"{image_id}::{caption_idx}"
    rows = _CAP_CACHE["rows"]
    cap = rows.get(key, "")
    if cap:
        return cap
    # Fallback: try any caption for this image_id
    for k, v in rows.items():
        if k.startswith(f"{image_id}::"):
            return v
    return ""


def fetch_image(image_id: int, caption_idx: int) -> tuple[Image.Image | None, str]:
    """Return (PIL image via URL, caption string via cache)."""
    cap = fetch_caption(image_id, caption_idx)
    img = _coco_image_from_url(image_id)
    return img, cap


def collect_concept_top_captions(
    model: TwoSidedTopKSAE, ds: CachedClipPairsDataset,
    sel_i: list[int], sel_t: list[int],
    n_top: int, batch_size: int, device: torch.device,
) -> list[list[tuple[int, float, str]]]:
    """For each selected concept (sel_i[k] / sel_t[k]), return top-n_top (pair_idx, t_act, caption) samples.

    Only the text side is used for captioning (meaningful for naming). Caption
    strings are retrieved via `fetch_image` in the caller (too slow to do per
    sample); here we return (pair_idx, score) triples and let the caller pull
    captions for only the top hits.
    """
    K = len(sel_i)
    sel_t_t = torch.tensor(sel_t, device=device)
    heaps: list[list] = [[] for _ in range(K)]  # min-heap of (score, pair_idx)
    N = len(ds)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            txts = torch.stack([ds[i]["text_embeds"] for i in range(s, e)]).to(device, dtype=torch.float32)
            out_t = model.text_sae(hidden_states=txts.unsqueeze(1), return_dense_latents=True)
            zt = out_t.dense_latents.squeeze(1)  # (B, n)
            # For each concept k, get zt[:, sel_t[k]]
            scores = zt[:, sel_t_t].cpu().numpy()  # (B, K)
            for k in range(K):
                col = scores[:, k]
                for bi, score in enumerate(col):
                    if score <= 0:
                        continue
                    if len(heaps[k]) < n_top:
                        heapq.heappush(heaps[k], (float(score), int(s + bi)))
                    elif score > heaps[k][0][0]:
                        heapq.heapreplace(heaps[k], (float(score), int(s + bi)))

    # Sort each heap descending
    results: list[list[tuple[int, float, str]]] = []
    for k in range(K):
        sorted_hits = sorted(heaps[k], key=lambda t: -t[0])
        results.append([(pair_idx, score, "") for score, pair_idx in sorted_hits])
    return results


_STOPWORDS = set("""a an the of in on at to and or with for from by is are was were be been being this that those these
it its there their his her him she he they them we us our my your you i as not but if so too very
some any all each no nor can could would should may might will shall do does did has have had
one two three four five many much more most same other another into over under up down out off
over again just also only such than then now here where when what which who whom whose why how
 """.split())


def infer_concept_name(captions: list[str]) -> str:
    """Return a short concept label from top-activating captions (most frequent content word)."""
    from collections import Counter
    counter: Counter[str] = Counter()
    for cap in captions:
        for tok in cap.lower().replace(".", " ").replace(",", " ").split():
            tok = tok.strip("\"'?!-:;()")
            if len(tok) < 3 or tok in _STOPWORDS:
                continue
            counter[tok] += 1
    if not counter:
        return "?"
    return counter.most_common(1)[0][0]


def _save_pair_context(
    pair_idx: int,
    ds: CachedClipPairsDataset | None,
    concept_names: list[str],
    dec_cos: list[float],
    sel_i: list, sel_t: list,
    val_i: list, val_t: list,
    out_path: Path,
) -> None:
    """Save image thumbnail + caption + concept info next to the PCA pdf."""
    if ds is None:
        print("pair context: CachedClipPairsDataset not available, skipping image/caption save")
        return
    image_id, caption_idx = ds.pairs[pair_idx]
    image_pil, caption = fetch_image(int(image_id), int(caption_idx))
    base = out_path.with_suffix("").as_posix()
    img_path = Path(f"{base}_pair_image.jpg")
    info_path = Path(f"{base}_pair_info.txt")

    if image_pil is not None:
        try:
            image_pil.convert("RGB").save(img_path, format="JPEG", quality=92)
            print(f"saved pair image to {img_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"failed to save pair image: {exc}")
    else:
        print("pair image unavailable (COCO URL fetch failed)")

    with open(info_path, "w") as f:
        f.write(f"pair_idx: {pair_idx}\n")
        f.write(f"image_id: {image_id}\n")
        f.write(f"caption_idx: {caption_idx}\n")
        f.write(f"caption: {caption}\n\n")
        f.write(f"concepts (name, img slot, txt slot, val_i, val_t, dec_cos):\n")
        for k in range(min(len(concept_names), 3)):
            ii = int(sel_i[k]) if k < len(sel_i) else -1
            jj = int(sel_t[k]) if k < len(sel_t) else -1
            vi = float(val_i[k]) if k < len(val_i) else 0.0
            vt = float(val_t[k]) if k < len(val_t) else 0.0
            dc = float(dec_cos[k]) if k < len(dec_cos) else 0.0
            f.write(f"  {k+1}. {concept_names[k]:<12}  "
                    f"img_slot={ii:<5} txt_slot={jj:<5}  "
                    f"val_i={vi:.3f} val_t={vt:.3f}  dec_cos={dc:.3f}\n")
    print(f"saved pair info to {info_path}")


def make_plot(
    x_vec: np.ndarray, y_vec: np.ndarray,
    img_scaled: list[np.ndarray], txt_scaled: list[np.ndarray],
    dec_cos: list[float],
    concept_names: list[str],
    image_pil: Image.Image | None, caption: str,
    out_path: Path, entry: dict,
) -> None:
    """Build figure. Each concept arrow starts at origin; a dashed connector
    runs from its tip to the embedding's tip, conveying 'these basis
    vectors together form the embedding'.

    Projection uses uncentered SVD so origin stays at (0,0) and concept /
    embedding orientations are consistent."""
    stack = np.vstack([x_vec, y_vec, *img_scaled, *txt_scaled])  # (8, 512)
    _U, S, Vt = np.linalg.svd(stack, full_matrices=False)
    basis = Vt[:2]
    pts = stack @ basis.T  # (8, 2)
    var_ratio = (S[:2] ** 2) / (S ** 2).sum()
    x_pt, y_pt = pts[0], pts[1]

    # Visibility scaling: enforce min length so weakest concept arrow still
    # readable, cap max so concepts don't visually dominate embedding.
    raw_img = pts[2:5]
    raw_txt = pts[5:8]
    max_embed = max(float(np.linalg.norm(x_pt)), float(np.linalg.norm(y_pt)), 1e-6)
    MIN_LEN = 0.35 * max_embed  # weakest concept at least this long
    MAX_LEN = 0.65 * max_embed  # strongest capped here

    def _rescale(row: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(row, axis=1) + 1e-12
        max_l, min_l = float(lengths.max()), float(lengths.min())
        if max_l - min_l < 1e-6:
            target = np.full_like(lengths, 0.5 * (MIN_LEN + MAX_LEN))
        else:
            t = (lengths - min_l) / (max_l - min_l)
            target = MIN_LEN + t * (MAX_LEN - MIN_LEN)
        return row / lengths[:, None] * target[:, None]

    img_pts = _rescale(raw_img)
    txt_pts = _rescale(raw_txt)

    # Square panel; figsize chosen so that (plot + legend row) total height
    # matches the density SVG (~2.64") after bbox_inches="tight".
    fig, ax = plt.subplots(figsize=(2.62, 2.62))

    # Concept basis arrows: filled "fancy" arrow (curved tail + triangular head).
    FANCY_STYLE = "fancy,head_length=0.35,head_width=0.35,tail_width=0.15"

    def draw_filled_arrow(ax, p0, p1, color, zorder=3):
        ax.annotate(
            "",
            xy=(p1[0], p1[1]), xycoords="data",
            xytext=(p0[0], p0[1]), textcoords="data",
            arrowprops=dict(
                arrowstyle=FANCY_STYLE,
                mutation_scale=14,
                facecolor=color, edgecolor=color,
                lw=0, shrinkA=0, shrinkB=0,
            ),
            zorder=zorder,
        )

    def draw_embedding_dot(ax, p, color, zorder=5):
        ax.scatter([p[0]], [p[1]], s=32, c=color, edgecolors="none", zorder=zorder)

    # Concept arrows (fancy filled).
    for k in range(3):
        draw_filled_arrow(ax, (0, 0), img_pts[k], IMG_COLORS[k])
        draw_filled_arrow(ax, (0, 0), txt_pts[k], TXT_COLORS[k])

    # Embedding = filled dot (no arrow, no border) at endpoint.
    draw_embedding_dot(ax, x_pt, EMB_IMG_COLOR)
    draw_embedding_dot(ax, y_pt, EMB_TXT_COLOR)

    # Dashed connectors from each concept tip -> embedding tip.
    for k in range(3):
        ax.plot([img_pts[k][0], x_pt[0]], [img_pts[k][1], x_pt[1]],
                color=IMG_COLORS[k], lw=0.6, linestyle=(0, (3, 2)), zorder=2, alpha=0.65)
        ax.plot([txt_pts[k][0], y_pt[0]], [txt_pts[k][1], y_pt[1]],
                color=TXT_COLORS[k], lw=0.6, linestyle=(0, (3, 2)), zorder=2, alpha=0.65)

    # Concept labels offset PERPENDICULAR to the arrow (not on the arrow line).
    scale_ref = max(np.linalg.norm(x_pt), np.linalg.norm(y_pt), 1e-6)

    def put_perp_label(ax, p0, p1, text, color, side=1):
        """Place label at arrow midpoint, offset perpendicular to arrow."""
        mid = (p0 + p1) / 2.0
        d = p1 - p0
        norm = float(np.linalg.norm(d)) + 1e-12
        perp = np.array([-d[1], d[0]]) / norm * side
        off = perp * 0.11 * scale_ref
        ax.text(mid[0] + off[0], mid[1] + off[1], text,
                color=color, fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=6)

    def put_tip_label(ax, tip, text, color, y_side="above", x_nudge=0.0):
        """Place label at arrow tip with a small fixed-direction offset
        (image-side ABOVE, text-side BELOW), so labels of the two modalities
        never interleave vertically. `x_nudge` (in units of scale_ref) shifts
        the label horizontally to resolve occasional text overlaps."""
        off_along = tip / (np.linalg.norm(tip) + 1e-12) * 0.04 * scale_ref
        off_y = (+1 if y_side == "above" else -1) * 0.06 * scale_ref
        off_x = x_nudge * scale_ref
        ax.text(tip[0] + off_along[0] + off_x, tip[1] + off_along[1] + off_y, text,
                color=color, fontsize=9, fontweight="bold",
                ha="center", va="bottom" if y_side == "above" else "top",
                zorder=6)

    # Per-label horizontal nudges on the text side to resolve overlaps
    # (e.g. "snow" bumping into "snowboard"). Keyed by concept name; default 0.
    TXT_X_NUDGE = {"snow": 0.14}
    for k, name in enumerate(concept_names[:3]):
        put_tip_label(ax, img_pts[k], name, IMG_COLORS[k], y_side="above")
        put_tip_label(ax, txt_pts[k], name, TXT_COLORS[k], y_side="below",
                      x_nudge=TXT_X_NUDGE.get(name, 0.0))

    # Tight inline legend at 8pt (matches NeurIPS footnotesize / caption text).
    from matplotlib.lines import Line2D
    emb_handles = [
        Line2D([0], [0], marker="o", markersize=5, color="none",
               markerfacecolor=EMB_IMG_COLOR, markeredgecolor="none",
               label="Image embedding"),
        Line2D([0], [0], marker="o", markersize=5, color="none",
               markerfacecolor=EMB_TXT_COLOR, markeredgecolor="none",
               label="Text embedding"),
    ]
    ax.legend(handles=emb_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False,
              fontsize=9, handlelength=0.6, handletextpad=0.3, columnspacing=1.2)

    ax.scatter([0], [0], s=8, c="black", zorder=6)  # origin marker
    ax.axhline(0, color="gray", lw=0.4, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.4, ls="--", zorder=0)

    # Axis limits: include all arrows + labels
    all_x = np.concatenate([img_pts[:, 0], txt_pts[:, 0], [x_pt[0], y_pt[0], 0.0]])
    all_y = np.concatenate([img_pts[:, 1], txt_pts[:, 1], [x_pt[1], y_pt[1], 0.0]])
    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) * 1.25
    cx, cy = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
    ax.set_xlim(float(cx - span / 2), float(cx + span / 2))
    ax.set_ylim(float(cy - span / 2), float(cy + span / 2))
    ax.set_aspect("equal")
    ax.set_xlabel("Principal Component 1", fontsize=9, labelpad=1)
    ax.set_ylabel("Principal Component 2", fontsize=9, labelpad=1)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(labelsize=8, pad=1)
    ax.grid(alpha=0.15, linewidth=0.4)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print(f"saved {out_path}")
    if str(out_path).endswith(".pdf"):
        for ext in (".png", ".svg"):
            alt = str(out_path).replace(".pdf", ext)
            fig.savefig(alt, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.02)
            print(f"saved {alt}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True,
                   help="dir with model.safetensors + diagnostic_B_C_train.npy + diagnostic_B_firing_rates.npz")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/pca_matched_concept.pdf")
    p.add_argument("--min-matched", type=int, default=3)
    p.add_argument("--c-min", type=float, default=0.25)
    p.add_argument("--cos-min", type=float, default=0.25)
    p.add_argument("--top-m", type=int, default=50,
                   help="keep top-M candidate pairs; selection criterion controlled below")
    p.add_argument("--select-by", type=str, default="balance",
                   choices=("minE", "balance"),
                   help="minE: smallest recon error; balance: maximize 3rd-strongest concept activation")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--pair-idx", type=int, default=-1,
                   help="override: use this pair index from train split")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--concept-names", type=str, default="",
                   help="optional comma-separated 3 names overriding auto-inferred labels, e.g. 'dog,ball,grass'")
    p.add_argument("--concept-top-n", type=int, default=5,
                   help="top-N captions per concept for auto-naming")
    p.add_argument("--caption-json", type=str,
                   default="cache/coco_karpathy_captions.json",
                   help="path to cached {image_id::caption_idx: caption_string} JSON")
    p.add_argument("--cache-file", type=str, default="",
                   help="npz with cached pair data; if exists, skip dataset/model and plot directly")
    args = p.parse_args()

    # Fast path: if cache exists, only load it + re-render plot. Takes ~1 sec.
    if args.cache_file and Path(args.cache_file).exists():
        print(f"loading plot cache from {args.cache_file} (fast path)")
        d = np.load(args.cache_file, allow_pickle=True)
        names = [str(n) for n in d["concept_names"]]
        if args.concept_names:
            names = [s.strip() for s in args.concept_names.split(",")]
            print(f"overriding concept names from CLI: {names}")
        entry_dict = {
            "pair_idx": int(d["pair_idx"]),
            "n_shared": int(d["n_shared"]),
            "E": float(d["E"]),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        make_plot(
            x_vec=d["x_vec"],
            y_vec=d["y_vec"],
            img_scaled=list(d["img_scaled"]),
            txt_scaled=list(d["txt_scaled"]),
            dec_cos=list(d["dec_cos"]),
            concept_names=names,
            image_pil=None,
            caption="",
            out_path=out_path,
            entry=entry_dict,
        )
        # Also save pair context (image + text info) side-by-side with the PCA plot.
        _load_caption_cache(Path(args.caption_json))
        ds_for_lookup = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True) \
            if Path(args.cache_dir).exists() else None
        _save_pair_context(
            pair_idx=entry_dict["pair_idx"],
            ds=ds_for_lookup,
            concept_names=names,
            dec_cos=list(d["dec_cos"]),
            sel_i=list(d["sel_i"]) if "sel_i" in d.files else [],
            sel_t=list(d["sel_t"]) if "sel_t" in d.files else [],
            val_i=list(d["val_i"]) if "val_i" in d.files else [],
            val_t=list(d["val_t"]) if "val_t" in d.files else [],
            out_path=out_path,
        )
        return

    _load_caption_cache(Path(args.caption_json))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    run_dir = Path(args.run_dir)
    model = load_model(run_dir, device)
    W_i = model.image_sae.W_dec.detach().cpu().numpy()
    W_t = model.text_sae.W_dec.detach().cpu().numpy()

    # Hungarian + M*
    match, C_vals, alive_i, alive_t, _C = alive_hungarian(run_dir)
    mstar_mask = compute_mstar_mask(match, C_vals, W_i, W_t, args.c_min, args.cos_min)
    print(f"M*: {int(mstar_mask.sum())} quality-matched pairs (out of {alive_i.size} alive_i)")

    ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
    print(f"train pairs: {len(ds)}")

    # Pick entry (either streaming search or forced pair_idx)
    if args.pair_idx >= 0:
        entry = _entry_for_pair(model, ds, args.pair_idx, match, mstar_mask, device)
        if entry["n_shared"] < args.min_matched:
            print(f"WARNING: forced pair_idx={args.pair_idx} has only {entry['n_shared']} matched concepts")
    else:
        candidates = stream_candidates(model, ds, mstar_mask, match,
                                       args.min_matched, args.top_m, args.batch_size, device)
        if not candidates:
            raise SystemExit("No pairs with >= min_matched shared concepts found.")

        def _balance_key(c: dict) -> float:
            # 3rd-strongest min(val_i, val_t) — higher is better (more balanced).
            mins = sorted([min(vi, vt) for vi, vt in zip(c["val_i"], c["val_t"])],
                          reverse=True)
            return mins[2] if len(mins) >= 3 else 0.0

        if args.select_by == "balance":
            candidates = sorted(candidates, key=lambda c: -_balance_key(c))
            print(f"Top-15 candidates by 3rd-strongest concept activation (balance):")
        else:
            candidates = sorted(candidates, key=lambda c: c["E"])
            print(f"Top-15 candidates by smallest E:")

        entry = candidates[0]
        for c in candidates[:15]:
            img_id, cap_idx = ds.pairs[c["pair_idx"]]
            cap = fetch_caption(int(img_id), int(cap_idx))[:60]
            vi = [f"{v:.2f}" for v in c["val_i"]]
            vt = [f"{v:.2f}" for v in c["val_t"]]
            print(f"  pair {c['pair_idx']} image_id={img_id}  E={c['E']:.3f}  "
                  f"n_shared={c['n_shared']}  "
                  f"3rd-min-val={_balance_key(c):.3f}")
            print(f"     i_slots={c['shared_i']} val_i={vi}")
            print(f"     t_slots={c['shared_pi']} val_t={vt}")
            print(f"     caption: {cap}")

    # Pick top-3 concepts by min(val_i, val_t) activation
    scores = [min(vi, vt) for vi, vt in zip(entry["val_i"], entry["val_t"])]
    order = np.argsort(-np.array(scores))[:3]
    sel_i = [entry["shared_i"][k] for k in order]
    sel_t = [entry["shared_pi"][k] for k in order]
    val_i_sel = [entry["val_i"][k] for k in order]
    val_t_sel = [entry["val_t"][k] for k in order]
    print(f"selected pair_idx={entry['pair_idx']}  E={entry['E']:.4f}")
    print(f"  3 concepts: img slots {sel_i}  <->  text slots {sel_t}")
    print(f"  activations img {val_i_sel}  |  txt {val_t_sel}")

    # Raw embeddings + scaled concept vectors (val * decoder column)
    x_vec = ds[entry["pair_idx"]]["image_embeds"].numpy().astype(np.float64)
    y_vec = ds[entry["pair_idx"]]["text_embeds"].numpy().astype(np.float64)
    img_dirs = np.stack([W_i[i] for i in sel_i], axis=0)
    txt_dirs = np.stack([W_t[j] for j in sel_t], axis=0)
    img_scaled = [float(val_i_sel[k]) * img_dirs[k].astype(np.float64) for k in range(3)]
    txt_scaled = [float(val_t_sel[k]) * txt_dirs[k].astype(np.float64) for k in range(3)]

    dec_cos = [
        float(np.dot(img_dirs[k], txt_dirs[k]) /
              (np.linalg.norm(img_dirs[k]) * np.linalg.norm(txt_dirs[k]) + 1e-12))
        for k in range(3)
    ]

    # Concept naming: collect top captions per concept
    if args.concept_names:
        concept_names = [s.strip() for s in args.concept_names.split(",")]
        if len(concept_names) != 3:
            raise SystemExit(f"--concept-names expects exactly 3 comma-separated labels, got {len(concept_names)}")
        print(f"using user-provided concept names: {concept_names}")
    else:
        print("collecting top captions per concept for auto-naming...")
        concept_hits = collect_concept_top_captions(
            model, ds, sel_i, sel_t, args.concept_top_n, args.batch_size, device,
        )
        concept_names = []
        for k, hits in enumerate(concept_hits):
            caps_for_concept: list[str] = []
            for pair_idx, score, _ in hits:
                img_id, cap_idx = ds.pairs[pair_idx]
                capt = fetch_caption(int(img_id), int(cap_idx))
                caps_for_concept.append(capt)
            name = infer_concept_name(caps_for_concept)
            concept_names.append(name)
            print(f"  concept {k+1} (slot {sel_i[k]} <-> {sel_t[k]}): name='{name}'")
            for (pair_idx, score, _), capt in zip(hits, caps_for_concept):
                print(f"    score={score:.3f}  pair={pair_idx}  caption: {capt[:80]}")

    # Save plot-only cache: subsequent runs skip dataset/model/forward entirely.
    cache_file = args.cache_file or f"outputs/pca_plot_cache_{entry['pair_idx']}.npz"
    np.savez(
        cache_file,
        pair_idx=entry["pair_idx"],
        n_shared=entry["n_shared"],
        E=entry["E"],
        x_vec=x_vec,
        y_vec=y_vec,
        img_scaled=np.stack(img_scaled),
        txt_scaled=np.stack(txt_scaled),
        dec_cos=np.array(dec_cos),
        sel_i=np.array(sel_i),
        sel_t=np.array(sel_t),
        val_i=np.array(val_i_sel),
        val_t=np.array(val_t_sel),
        concept_names=np.array(concept_names),
    )
    print(f"saved plot cache to {cache_file}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_plot(
        x_vec=x_vec,
        y_vec=y_vec,
        img_scaled=img_scaled,
        txt_scaled=txt_scaled,
        dec_cos=dec_cos,
        concept_names=concept_names,
        image_pil=None,
        caption="",
        out_path=out_path,
        entry={**entry, "pair_idx": entry["pair_idx"]},
    )

    # Caption tex
    cap_tex = out_path.with_suffix("").as_posix() + "_caption.tex"
    with open(cap_tex, "w") as f:
        f.write(
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{{out_path.name}}}\n"
            "  \\caption{\n"
            "    \\textbf{Dictionary direction mismatch for a single image-caption pair.}\n"
            "    A pair from COCO train is chosen such that its TopK Separated-SAE reconstruction"
            f"    activates {entry['n_shared']} Hungarian-matched shared concepts (alive-restricted\n"
            f"    matching on the train cross-correlation matrix, with both $r \\ge {args.c_min}$ and\n"
            f"    matched decoder cosine $\\ge {args.cos_min}$). The three most strongly activated\n"
            f"    matched pairs are projected to 2D via PCA together with the image and text CLIP\n"
            "    embeddings. Image-side decoder directions (warm) and text-side directions (cool)\n"
            "    are paired by shade. The two embeddings lie in the span of their respective arrows,\n"
            "    but the matched image/text arrows consistently point in different directions ---\n"
            "    direct evidence of dictionary direction mismatch even when concepts co-activate.\n"
            "  }\n"
            "  \\label{fig:pca-matched-concept}\n"
            "\\end{figure}\n"
        )
    print(f"saved {cap_tex}")


def _entry_for_pair(model, ds, pair_idx, match, mstar_mask, device):
    """Compute the same entry dict as stream_candidates for a single pair_idx."""
    with torch.no_grad():
        x = ds[pair_idx]["image_embeds"].to(device, dtype=torch.float32).unsqueeze(0)
        y = ds[pair_idx]["text_embeds"].to(device, dtype=torch.float32).unsqueeze(0)
        out_i = model.image_sae(hidden_states=x.unsqueeze(1), return_dense_latents=False)
        out_t = model.text_sae(hidden_states=y.unsqueeze(1), return_dense_latents=False)
        idx_i = out_i.latent_indices.squeeze(1)[0].cpu().numpy()
        idx_t = out_t.latent_indices.squeeze(1)[0].cpu().numpy()
        val_i = out_i.latent_activations.squeeze(1)[0].cpu().numpy()
        val_t = out_t.latent_activations.squeeze(1)[0].cpu().numpy()
        W_i = model.image_sae.W_dec
        W_t = model.text_sae.W_dec
        x_hat = (torch.from_numpy(val_i).to(device) @ W_i[torch.from_numpy(idx_i).to(device)])
        y_hat = (torch.from_numpy(val_t).to(device) @ W_t[torch.from_numpy(idx_t).to(device)])
        E = float(((x.squeeze(0) - x_hat) ** 2).sum().item() + ((y.squeeze(0) - y_hat) ** 2).sum().item())

    shared_i = [int(i) for i in idx_i if mstar_mask[i] and match[i] in set(idx_t)]
    shared_pi = [int(match[i]) for i in shared_i]
    val_i_shared = [float(val_i[list(idx_i).index(i)]) for i in shared_i]
    val_t_shared = [float(val_t[list(idx_t).index(match[i])]) for i in shared_i]
    return {
        "pair_idx": int(pair_idx),
        "E": E,
        "n_shared": len(shared_i),
        "shared_i": shared_i,
        "shared_pi": shared_pi,
        "val_i": val_i_shared,
        "val_t": val_t_shared,
    }


if __name__ == "__main__":
    main()
