"""End-to-end analysis of the real-data aux-alignment ablation.

Given the 7-variant sweep under `outputs/aux_alignment_clip_b32/`, compute
per-variant:

  A. Alive slot count (img / txt) at fire_rate > tau_alive
  B. Slot redundancy (per-GT-atom is not applicable in real; use "per-concept
     proxy via co-firing bin 99th percentile"; skip for real)
  C. Matched pair count by C-bin (10 bins over [0, 1))
  D. Decoder cosine distribution by C-bin (boxplot data)
  E. alpha_hat estimate = median decoder cos of top-X% co-firing pairs
  F. Reconstruction loss (train + eval)
  G. For recon_only: also apply post-hoc Hungarian and recompute (B–E)

Outputs:
  outputs/aux_alignment_clip_b32/diag/summary.json
  outputs/aux_alignment_clip_b32/diag/comparison_boxplot.png
  outputs/aux_alignment_clip_b32/diag/summary_table.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_alpha"))
import _bootstrap  # noqa: F401

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore
from src.models.configuration_sae import TwoSidedTopKSAEConfig  # type: ignore
from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BIN_EDGES = np.linspace(0.0, 1.0, 11)
BIN_LABELS = [f"[{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})" for i in range(10)]


def _load_variant(variant_dir: Path, device: torch.device):
    """Load TwoSidedTopKSAE checkpoint from <variant>/final/model.safetensors."""
    state = load_file(str(variant_dir / "final" / "model.safetensors"))
    cfg_path = variant_dir / "final" / "config.json"
    cfg_json = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    top_config = variant_dir / "config.json"
    if top_config.exists():
        user_cfg = json.loads(top_config.read_text())
    else:
        user_cfg = {}
    hidden = state["image_sae.W_dec"].shape[1]
    per_side_latent = state["image_sae.W_dec"].shape[0]
    k = int(user_cfg.get("k", cfg_json.get("k", 8)))
    # TwoSidedTopKSAEConfig expects TOTAL latent_size; it splits in half per side.
    sae_cfg = TwoSidedTopKSAEConfig(
        hidden_size=hidden, latent_size=per_side_latent * 2, k=k, normalize_decoder=True,
    )
    model = TwoSidedTopKSAE(sae_cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, user_cfg


def _stack_train(cache_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    ds = CachedClipPairsDataset(cache_dir, split="train", l2_normalize=True)
    img = torch.stack([ds[i]["image_embeds"] for i in range(len(ds))], dim=0)
    txt = torch.stack([ds[i]["text_embeds"] for i in range(len(ds))], dim=0)
    return img, txt


def _stack_eval(cache_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    ds = CachedClipPairsDataset(cache_dir, split="test", l2_normalize=True)
    img = torch.stack([ds[i]["image_embeds"] for i in range(len(ds))], dim=0)
    txt = torch.stack([ds[i]["text_embeds"] for i in range(len(ds))], dim=0)
    return img, txt


def _streaming_C(sae_i, sae_t, img, txt, bs, device):
    """Pearson cross-correlation matrix (n x n), streaming over batches."""
    n = int(sae_i.latent_size)
    sum_i = np.zeros(n, dtype=np.float64)
    sum_t = np.zeros(n, dtype=np.float64)
    sum_ii = np.zeros(n, dtype=np.float64)
    sum_tt = np.zeros(n, dtype=np.float64)
    sum_it = np.zeros((n, n), dtype=np.float64)
    N = 0
    fire_i = np.zeros(n, dtype=np.int64)
    fire_t = np.zeros(n, dtype=np.int64)
    with torch.no_grad():
        for s in range(0, img.shape[0], bs):
            ib = img[s : s + bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            tb = txt[s : s + bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            zi = sae_i(hidden_states=ib, return_dense_latents=True).dense_latents.squeeze(1).cpu().to(torch.float64).numpy()
            zt = sae_t(hidden_states=tb, return_dense_latents=True).dense_latents.squeeze(1).cpu().to(torch.float64).numpy()
            sum_i += zi.sum(0); sum_t += zt.sum(0)
            sum_ii += (zi * zi).sum(0); sum_tt += (zt * zt).sum(0)
            sum_it += zi.T @ zt
            fire_i += (zi > 0).sum(0); fire_t += (zt > 0).sum(0)
            N += zi.shape[0]
    mean_i = sum_i / N; mean_t = sum_t / N
    var_i = sum_ii / N - mean_i ** 2
    var_t = sum_tt / N - mean_t ** 2
    cov = sum_it / N - np.outer(mean_i, mean_t)
    denom = np.sqrt(np.clip(var_i[:, None] * var_t[None, :], 1e-16, None))
    C = np.nan_to_num(cov / denom, nan=0.0, posinf=0.0, neginf=0.0)
    rates_i = fire_i / max(N, 1)
    rates_t = fire_t / max(N, 1)
    return C, rates_i, rates_t


def _recon_loss(sae, data, bs, device):
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, data.shape[0], bs):
            x = data[s : s + bs].to(device=device, dtype=torch.float32).unsqueeze(1)
            out = sae(hidden_states=x)
            total += float(out.recon_loss.item()) * x.shape[0]
            n += x.shape[0]
    return total / max(n, 1)


def _analyze(
    variant: str,
    model: TwoSidedTopKSAE,
    train_img, train_txt, eval_img, eval_txt,
    bs, device, tau_alive: float = 0.001,
    permute_text: bool = False,
) -> dict:
    sae_i, sae_t = model.image_sae, model.text_sae
    re_i = _recon_loss(sae_i, eval_img, bs, device)
    re_t = _recon_loss(sae_t, eval_txt, bs, device)
    re_train_i = _recon_loss(sae_i, train_img, bs, device)
    re_train_t = _recon_loss(sae_t, train_txt, bs, device)
    C, rates_i, rates_t = _streaming_C(sae_i, sae_t, train_img, train_txt, bs, device)

    # Optional: post-hoc Hungarian permutation of text side (for recon_only baseline).
    posthoc_tag = ""
    if permute_text:
        _, col_ind = linear_sum_assignment(-C)
        # Apply permutation to the text decoder in memory (not to model):
        # We'll permute W_txt before computing matched_decoder_cos.
        text_perm = col_ind
        # Re-compute C w/o permuting encoder — we permute rows of W_txt and text dense latents symbolically
        # For decoder cosine, pair is (V[i], W[text_perm[i]]); use directly in the matched-pair loop below.
        posthoc_tag = "+posthoc_Hungarian"
    else:
        text_perm = None

    alive_i = np.where(rates_i > tau_alive)[0]
    alive_t = np.where(rates_t > tau_alive)[0]

    # Alive-alive Hungarian on C (or on permuted-C if posthoc)
    if text_perm is None:
        C_alive = C[np.ix_(alive_i, alive_t)]
        row_ind, col_ind_local = linear_sum_assignment(-C_alive)
        pair_img_idx = alive_i[row_ind]
        pair_txt_idx = alive_t[col_ind_local]
    else:
        # Apply permutation to text-side firing rates too (so "alive" on permuted side is consistent)
        permuted_rates_t = rates_t[text_perm]
        alive_t_perm = np.where(permuted_rates_t > tau_alive)[0]
        # Permuted C: new C[i, k] = original C[i, text_perm[k]]
        C_perm = C[:, text_perm]
        C_alive = C_perm[np.ix_(alive_i, alive_t_perm)]
        row_ind, col_ind_local = linear_sum_assignment(-C_alive)
        pair_img_idx = alive_i[row_ind]
        pair_txt_idx_perm = alive_t_perm[col_ind_local]
        pair_txt_idx = text_perm[pair_txt_idx_perm]

    matched_C = (C[pair_img_idx, pair_txt_idx] if text_perm is None
                 else C_alive[row_ind, col_ind_local])

    # Decoder cosines
    Wi = sae_i.W_dec.detach().cpu().numpy().astype(np.float32)
    Wt = sae_t.W_dec.detach().cpu().numpy().astype(np.float32)
    Wi_n = Wi / (np.linalg.norm(Wi, axis=-1, keepdims=True) + 1e-12)
    Wt_n = Wt / (np.linalg.norm(Wt, axis=-1, keepdims=True) + 1e-12)
    dec_cos = (Wi_n[pair_img_idx] * Wt_n[pair_txt_idx]).sum(axis=-1)

    bin_idx = np.clip(np.digitize(matched_C, BIN_EDGES) - 1, 0, 9)
    bin_counts = [int((bin_idx == b).sum()) for b in range(10)]
    bin_cos_values = [dec_cos[bin_idx == b].tolist() for b in range(10)]
    bin_cos_median = [
        (float(np.median(dec_cos[bin_idx == b])) if (bin_idx == b).any() else None)
        for b in range(10)
    ]

    # alpha_hat: top-10% co-firing pairs (highest C), decoder cos median
    n_top = max(1, int(len(matched_C) * 0.10))
    order = np.argsort(-matched_C)[:n_top]
    alpha_hat_10 = float(np.median(dec_cos[order]))
    n_top20 = max(1, int(len(matched_C) * 0.20))
    order20 = np.argsort(-matched_C)[:n_top20]
    alpha_hat_20 = float(np.median(dec_cos[order20]))

    # Modality ratio: r_k = fire_i / (fire_i + fire_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_k = rates_i / np.clip(rates_i + rates_t, 1e-12, None)
    alive_either = (rates_i > tau_alive) | (rates_t > tau_alive)
    img_dom = int(np.sum((r_k[alive_either] > 0.9)))
    txt_dom = int(np.sum((r_k[alive_either] < 0.1)))
    strong_shared = int(np.sum((r_k[alive_either] >= 0.4) & (r_k[alive_either] <= 0.6)))

    return {
        "variant": variant + posthoc_tag,
        "re_eval_img": re_i,
        "re_eval_txt": re_t,
        "re_eval_total": re_i + re_t,
        "re_train_img": re_train_i,
        "re_train_txt": re_train_t,
        "re_train_total": re_train_i + re_train_t,
        "n_alive_img": int(alive_i.size),
        "n_alive_txt": int(alive_t.size),
        "n_dead_img": int(sae_i.latent_size) - int(alive_i.size),
        "n_dead_txt": int(sae_t.latent_size) - int(alive_t.size),
        "n_matched": int(matched_C.size),
        "alpha_hat_top10pct": alpha_hat_10,
        "alpha_hat_top20pct": alpha_hat_20,
        "modality_img_dominant": img_dom,
        "modality_txt_dominant": txt_dom,
        "modality_strong_shared": strong_shared,
        "bin_counts": bin_counts,
        "bin_decoder_cos_median": bin_cos_median,
        "bin_decoder_cos_values": bin_cos_values,
    }


def plot_comparison(per_variant, out_path):
    import matplotlib.pyplot as plt
    variants = list(per_variant.keys())
    n_var = len(variants)
    n_bins = 10
    width = 0.8 / max(n_var, 1)

    fig, ax = plt.subplots(figsize=(18, 7))
    cmap = plt.get_cmap("tab10")
    box_data, box_pos, box_colors = [], [], []
    for b in range(n_bins):
        for i, v in enumerate(variants):
            vals = per_variant[v]["bin_decoder_cos_values"][b]
            if not vals:
                continue
            box_data.append(vals)
            box_pos.append(b + (i - (n_var - 1) / 2) * width)
            box_colors.append(cmap(i % 10))
    if box_data:
        bp = ax.boxplot(box_data, positions=box_pos, widths=width * 0.9,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(BIN_LABELS, rotation=30)
    ax.set_ylabel("decoder column cosine of matched pair")
    ax.set_xlabel("matched pair correlation bin")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_title("Decoder-cosine distribution across variants, binned by matched correlation")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=cmap(i % 10), alpha=0.5, label=v) for i, v in enumerate(variants)]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def print_summary_table(per_variant, out_path):
    lines = []
    header = (
        f"{'variant':38s} "
        f"{'RE_eval':>8} "
        f"{'RE_train':>9} "
        f"{'alive_i':>8} "
        f"{'alive_t':>8} "
        f"{'#matched':>9} "
        f"{'alpha_top10':>12} "
        f"{'alpha_top20':>12} "
        f"{'img_dom':>8} "
        f"{'txt_dom':>8} "
        f"{'shared':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for v, r in per_variant.items():
        lines.append(
            f"{v:38s} "
            f"{r['re_eval_total']:>8.4f} "
            f"{r['re_train_total']:>9.4f} "
            f"{r['n_alive_img']:>8d} "
            f"{r['n_alive_txt']:>8d} "
            f"{r['n_matched']:>9d} "
            f"{r['alpha_hat_top10pct']:>12.3f} "
            f"{r['alpha_hat_top20pct']:>12.3f} "
            f"{r['modality_img_dominant']:>8d} "
            f"{r['modality_txt_dominant']:>8d} "
            f"{r['modality_strong_shared']:>7d}"
        )
    text = "\n".join(lines)
    print(text)
    out_path.write_text(text + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", required=True,
                   help="root dir containing variant subdirs with final/model.safetensors")
    p.add_argument("--cache-dir", required=True, help="CLIP cache for train/eval embeddings")
    p.add_argument("--output", required=True, help="output dir for summary + plot")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--tau-alive", type=float, default=0.001)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    runs_root = Path(args.runs_root)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    # Preferred variant order (if present)
    order = [
        "recon_only",
        "naive_once",
        "barlow_once",
        "infonce_once",
        "naive_perepoch+revive",
        "barlow_perepoch+revive",
        "infonce_perepoch+revive",
    ]
    variant_dirs = []
    for name in order:
        vd = runs_root / name
        if (vd / "final" / "model.safetensors").exists():
            variant_dirs.append(vd)
    # also catch any other variant present
    for vd in sorted(runs_root.iterdir()):
        if vd.is_dir() and vd.name not in order and (vd / "final" / "model.safetensors").exists():
            variant_dirs.append(vd)

    if not variant_dirs:
        raise SystemExit(f"no variants with final/model.safetensors under {runs_root}")

    train_img, train_txt = _stack_train(args.cache_dir)
    eval_img, eval_txt = _stack_eval(args.cache_dir)
    logger.info("train=%d eval=%d", train_img.shape[0], eval_img.shape[0])

    per_variant: dict[str, dict] = {}
    for vd in variant_dirs:
        vname = vd.name
        logger.info("== %s", vname)
        model, _ = _load_variant(vd, device)
        r = _analyze(vname, model, train_img, train_txt, eval_img, eval_txt,
                     args.batch_size, device, args.tau_alive, permute_text=False)
        per_variant[r["variant"]] = r
        # For recon_only, also do post-hoc Hungarian version
        if vname == "recon_only":
            logger.info("-- recon_only + post-hoc Hungarian")
            r2 = _analyze(vname, model, train_img, train_txt, eval_img, eval_txt,
                          args.batch_size, device, args.tau_alive, permute_text=True)
            per_variant[r2["variant"]] = r2
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    slim = {
        v: {k: val for k, val in r.items() if k != "bin_decoder_cos_values"}
        for v, r in per_variant.items()
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(slim, f, indent=2)
    logger.info("Wrote %s", out_dir / "summary.json")

    # Full data with boxplot values
    with open(out_dir / "summary_full.json", "w") as f:
        json.dump(per_variant, f, indent=2)

    # Plot + table
    plot_comparison(per_variant, out_dir / "comparison_boxplot.png")
    print_summary_table(per_variant, out_dir / "summary_table.txt")


if __name__ == "__main__":
    main()
