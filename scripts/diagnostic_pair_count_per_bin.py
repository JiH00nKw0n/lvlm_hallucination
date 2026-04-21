"""Pair-count + decoder-cosine diagnostic for the aux-alignment ablation.

For each variant under ``--runs-root``:
  1. Load the trained TwoSidedTopKSAE (or the .npz dump for synthetic).
  2. Compute the cross-correlation matrix ``C`` on training data.
  3. Restrict to alive-alive submatrix (firing rate > tau_alive on either side).
  4. Hungarian on -|C| -> matched pair list.
  5. For each matched pair (i, pi(i)), record:
       - ``C_value``: correlation of the matched pair
       - ``decoder_cos``: cos(W_dec_img[i], W_dec_txt[pi(i)])
  6. Bin pairs by C_value into 10 bins ([0, 0.1), ..., [0.9, 1.0]).
  7. Write per-variant JSON and an overall comparison boxplot.

Synthetic mode: ``--synthetic``. Loads .npz from <run>/runs/<ts>/params/.
Real mode (default): loads model.safetensors from <variant>/final/.

Usage:
    # synthetic
    python scripts/diagnostic_pair_count_per_bin.py \\
        --runs-root outputs/aux_alignment_synthetic/runs \\
        --output    outputs/aux_alignment_synthetic/diag/ \\
        --synthetic

    # real
    python scripts/diagnostic_pair_count_per_bin.py \\
        --runs-root outputs/aux_alignment_clip_b32 \\
        --cache-dir cache/clip_b32_coco \\
        --output    outputs/aux_alignment_clip_b32/diag/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

# Bootstrap to bypass src.* __init__ chains.
_sys.path.insert(0, str(_Path(__file__).resolve().parent / "real_alpha"))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pathlib import Path  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


BIN_EDGES = np.linspace(0.0, 1.0, 11)  # 10 bins
BIN_LABELS = [f"[{BIN_EDGES[i]:.1f}, {BIN_EDGES[i+1]:.1f})" for i in range(10)]


# ---------------------------------------------------------------------------
# Real-mode loaders
# ---------------------------------------------------------------------------


def _load_real_two_sae(variant_dir: Path):
    """Load TwoSidedTopKSAE from <variant>/final/model.safetensors.

    Returns (sae_i_state_dict, sae_t_state_dict). We don't need to instantiate
    the full model class — we only need decoder weights for cosines and need
    to run forward for activations. To keep this script lightweight, return
    raw tensors and let the caller construct the model.
    """
    final_dir = variant_dir / "final"
    cfg_path = final_dir / "config.json"
    weights_path = final_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model.safetensors in {final_dir}")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    state = load_file(str(weights_path))
    return state, cfg


def _build_two_sided_from_state(state: dict[str, torch.Tensor], cfg: dict, device: torch.device):
    from src.models.configuration_sae import TwoSidedTopKSAEConfig  # type: ignore
    from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore

    hidden = state["image_sae.W_dec"].shape[1]
    latent = state["image_sae.W_dec"].shape[0]
    k = int(cfg.get("k", 8))
    sae_cfg = TwoSidedTopKSAEConfig(
        hidden_size=hidden, latent_size=latent, k=k, normalize_decoder=True,
    )
    model = TwoSidedTopKSAE(sae_cfg).to(device)
    model.load_state_dict(state, strict=False)
    return model


def _stack_real_train_tensors(cache_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore

    ds = CachedClipPairsDataset(cache_dir, split="train", l2_normalize=True)
    img = torch.stack([ds[i]["image_embeds"] for i in range(len(ds))], dim=0)
    txt = torch.stack([ds[i]["text_embeds"] for i in range(len(ds))], dim=0)
    return img, txt


# ---------------------------------------------------------------------------
# Synthetic-mode loaders
# ---------------------------------------------------------------------------


def _load_synth_npz(npz_path: Path):
    """Load synthetic dump. Returns (sae_i_W_dec, sae_t_W_dec, ...) and a small
    re-instantiated TwoSidedTopKSAE on CPU for forward calls (Hungarian needs
    activations on the train set)."""
    d = np.load(npz_path)
    return d


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _firing_rates_per_side(sae, data: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    sae.eval()
    n = int(sae.latent_size)
    fired = np.zeros(n, dtype=np.int64)
    N = 0
    with torch.no_grad():
        for s in range(0, data.shape[0], batch_size):
            chunk = data[s : s + batch_size].to(device=device, dtype=torch.float32).unsqueeze(1)
            out = sae(hidden_states=chunk, return_dense_latents=True)
            z = out.dense_latents.squeeze(1).cpu().numpy()
            fired += (z > 0).sum(axis=0)
            N += z.shape[0]
    return fired / max(N, 1)


def _cross_corr(sae_i, sae_t, img: torch.Tensor, txt: torch.Tensor,
                batch_size: int, device: torch.device) -> np.ndarray:
    """Pearson cross-correlation, streaming over batches."""
    sae_i.eval(); sae_t.eval()
    n = int(sae_i.latent_size)
    sum_i = np.zeros(n, dtype=np.float64)
    sum_t = np.zeros(n, dtype=np.float64)
    sum_ii = np.zeros(n, dtype=np.float64)
    sum_tt = np.zeros(n, dtype=np.float64)
    sum_it = np.zeros((n, n), dtype=np.float64)
    N = 0
    with torch.no_grad():
        for s in range(0, img.shape[0], batch_size):
            ib = img[s : s + batch_size].to(device=device, dtype=torch.float32).unsqueeze(1)
            tb = txt[s : s + batch_size].to(device=device, dtype=torch.float32).unsqueeze(1)
            zi = sae_i(hidden_states=ib, return_dense_latents=True).dense_latents.squeeze(1).cpu().to(torch.float64).numpy()
            zt = sae_t(hidden_states=tb, return_dense_latents=True).dense_latents.squeeze(1).cpu().to(torch.float64).numpy()
            sum_i += zi.sum(0); sum_t += zt.sum(0)
            sum_ii += (zi * zi).sum(0); sum_tt += (zt * zt).sum(0)
            sum_it += zi.T @ zt
            N += zi.shape[0]
    if N == 0:
        return np.zeros((n, n))
    mean_i = sum_i / N; mean_t = sum_t / N
    var_i = sum_ii / N - mean_i ** 2
    var_t = sum_tt / N - mean_t ** 2
    cov = sum_it / N - np.outer(mean_i, mean_t)
    denom = np.sqrt(np.clip(var_i[:, None] * var_t[None, :], 1e-16, None))
    return np.nan_to_num(cov / denom, nan=0.0, posinf=0.0, neginf=0.0)


def analyze_variant(
    variant_name: str,
    sae_i,
    sae_t,
    img: torch.Tensor,
    txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
    tau_alive: float = 0.001,
) -> dict:
    """Compute C, alive-alive Hungarian, bin matched pairs, return dict."""
    rates_i = _firing_rates_per_side(sae_i, img, batch_size, device)
    rates_t = _firing_rates_per_side(sae_t, txt, batch_size, device)
    alive_i = np.where(rates_i > tau_alive)[0]
    alive_t = np.where(rates_t > tau_alive)[0]

    C = _cross_corr(sae_i, sae_t, img, txt, batch_size, device)
    sub = C[np.ix_(alive_i, alive_t)]
    if sub.size == 0:
        return {
            "variant": variant_name,
            "n_alive_img": int(alive_i.size),
            "n_alive_txt": int(alive_t.size),
            "n_matched": 0,
            "bin_counts": [0] * 10,
            "bin_decoder_cos_median": [None] * 10,
            "bin_decoder_cos_values": [[] for _ in range(10)],
        }
    row_ind, col_ind = linear_sum_assignment(-sub)
    matched_C = sub[row_ind, col_ind]

    Wi = sae_i.W_dec.detach().cpu().numpy().astype(np.float32)
    Wt = sae_t.W_dec.detach().cpu().numpy().astype(np.float32)
    Wi_n = Wi / (np.linalg.norm(Wi, axis=-1, keepdims=True) + 1e-12)
    Wt_n = Wt / (np.linalg.norm(Wt, axis=-1, keepdims=True) + 1e-12)
    pair_img_idx = alive_i[row_ind]
    pair_txt_idx = alive_t[col_ind]
    decoder_cos = (Wi_n[pair_img_idx] * Wt_n[pair_txt_idx]).sum(axis=-1)

    bin_idx = np.clip(np.digitize(matched_C, BIN_EDGES) - 1, 0, 9)
    bin_counts = [int((bin_idx == b).sum()) for b in range(10)]
    bin_cos_values = [decoder_cos[bin_idx == b].tolist() for b in range(10)]
    bin_cos_median = [
        (float(np.median(decoder_cos[bin_idx == b])) if (bin_idx == b).any() else None)
        for b in range(10)
    ]

    return {
        "variant": variant_name,
        "n_alive_img": int(alive_i.size),
        "n_alive_txt": int(alive_t.size),
        "n_matched": int(matched_C.size),
        "bin_edges": BIN_EDGES.tolist(),
        "bin_labels": BIN_LABELS,
        "bin_counts": bin_counts,
        "bin_decoder_cos_median": bin_cos_median,
        "bin_decoder_cos_values": bin_cos_values,
        "matched_C_values": matched_C.tolist(),
        "matched_decoder_cos_values": decoder_cos.tolist(),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_pair_count_per_bin(per_variant: dict[str, dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    variants = list(per_variant.keys())
    n_var = len(variants)
    n_bins = 10
    width = 0.8 / max(n_var, 1)

    fig, (ax_count, ax_cos) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(n_bins)
    for i, v in enumerate(variants):
        counts = per_variant[v]["bin_counts"]
        offset = (i - (n_var - 1) / 2) * width
        ax_count.bar(x + offset, counts, width=width, label=v)
    ax_count.set_ylabel("# matched pairs")
    ax_count.set_yscale("log")
    ax_count.set_title("Hungarian-survived matched pair count per correlation bin")
    ax_count.legend(fontsize=8, ncol=3)

    # Decoder cosine boxplots per bin per variant
    box_data = []
    box_positions = []
    box_colors = []
    cmap = plt.get_cmap("tab10")
    for b in range(n_bins):
        for i, v in enumerate(variants):
            vals = per_variant[v]["bin_decoder_cos_values"][b]
            if not vals:
                continue
            box_data.append(vals)
            box_positions.append(b + (i - (n_var - 1) / 2) * width)
            box_colors.append(cmap(i % 10))
    if box_data:
        bp = ax_cos.boxplot(box_data, positions=box_positions, widths=width * 0.9,
                            patch_artist=True, showfliers=False, manage_ticks=False)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.5)
    ax_cos.set_xticks(x)
    ax_cos.set_xticklabels(BIN_LABELS, rotation=30)
    ax_cos.set_ylabel("decoder cosine")
    ax_cos.set_xlabel("matched pair correlation bin")
    ax_cos.axhline(0.0, color="grey", lw=0.5)
    ax_cos.set_title("Decoder column cosine of matched pairs")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def run_real(args: argparse.Namespace, device: torch.device) -> None:
    runs_root = Path(args.runs_root)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir
    if cache_dir is None:
        raise SystemExit("--cache-dir is required in real mode")

    variant_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and (p / "final" / "model.safetensors").exists()])
    if not variant_dirs:
        raise SystemExit(f"No variant subdirs with final/model.safetensors under {runs_root}")
    logger.info("Found %d variants", len(variant_dirs))

    img, txt = _stack_real_train_tensors(cache_dir)
    per_variant: dict[str, dict] = {}
    for vd in variant_dirs:
        vname = vd.name
        logger.info("== Variant: %s", vname)
        state, cfg = _load_real_two_sae(vd)
        model = _build_two_sided_from_state(state, cfg, device)
        result = analyze_variant(
            vname, model.image_sae, model.text_sae, img, txt,
            batch_size=args.batch_size, device=device, tau_alive=args.tau_alive,
        )
        per_variant[vname] = result
        out_path = out_dir / f"{vname.replace('+', '__')}_pair_count.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_dir / "all_variants_summary.json", "w") as f:
        # Strip raw value lists to keep file small; keep counts + medians.
        slim = {v: {kk: vv for kk, vv in d.items() if kk not in ("bin_decoder_cos_values", "matched_C_values", "matched_decoder_cos_values")} for v, d in per_variant.items()}
        json.dump(slim, f, indent=2)
    plot_pair_count_per_bin(per_variant, out_dir / "comparison_pair_count_per_bin.png")


def run_synthetic(args: argparse.Namespace, device: torch.device) -> None:
    """Synthetic mode: load .npz dumps and reconstruct SAEs."""
    from src.datasets.synthetic_paired_builder import SyntheticPairedBuilder  # type: ignore
    from src.models.configuration_sae import TopKSAEConfig  # type: ignore
    from src.models.modeling_sae import TopKSAE  # type: ignore

    runs_root = Path(args.runs_root)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(runs_root.glob("**/params/*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files under {runs_root}/**/params/")
    logger.info("Found %d .npz dumps", len(npz_files))

    # Group by variant_name (parsed from filename: alpha{a}_seed{s}_{variant}.npz)
    per_variant: dict[str, dict] = {}
    for npz_path in npz_files:
        d = _load_synth_npz(npz_path)
        alpha = float(d["alpha_target"])
        seed = int(d["seed"])
        # Filename format: alpha{a:.2f}_seed{s}_{variant}.npz
        stem = npz_path.stem
        prefix = f"alpha{alpha:.2f}_seed{seed}_"
        variant = stem.replace(prefix, "").replace("__", "+")
        logger.info("== Variant: %s (alpha=%.2f, seed=%d)", variant, alpha, seed)

        # Rebuild dataset to get train img/txt
        # Parameters are inferred from the .npz itself (phi_S, etc.)
        n_shared = d["phi_S"].shape[1]
        n_image = d["phi_I"].shape[1] if d["phi_I"].size > 0 else 0
        n_text = d["psi_T"].shape[1] if d["psi_T"].size > 0 else 0
        rep_dim = d["phi_S"].shape[0]
        builder = SyntheticPairedBuilder(
            n_shared=n_shared, n_image=n_image, n_text=n_text,
            representation_dim=rep_dim, sparsity=0.99, beta=1.0,
            obs_noise_std=0.05, max_interference=0.1,
            alpha_target=alpha, num_train=10000, num_eval=1000, seed=seed,
        )
        ds = builder.build()
        img = torch.from_numpy(ds["train"]["image_representation"])
        txt = torch.from_numpy(ds["train"]["text_representation"])

        # Rebuild SAEs from parameters
        latent_i = int(d["latent_size_img"])
        latent_t = int(d["latent_size_txt"])
        sae_i_cfg = TopKSAEConfig(hidden_size=rep_dim, latent_size=latent_i, k=16, normalize_decoder=True)
        sae_t_cfg = TopKSAEConfig(hidden_size=rep_dim, latent_size=latent_t, k=16, normalize_decoder=True)
        sae_i = TopKSAE(sae_i_cfg).to(device)
        sae_t = TopKSAE(sae_t_cfg).to(device)
        with torch.no_grad():
            sae_i.W_dec.data.copy_(torch.from_numpy(d["w_dec_img"]).to(device))
            sae_i.b_dec.data.copy_(torch.from_numpy(d["b_dec_img"]).to(device))
            sae_i.encoder.weight.data.copy_(torch.from_numpy(d["w_enc_img"]).to(device))
            sae_i.encoder.bias.data.copy_(torch.from_numpy(d["b_enc_img"]).to(device))
            sae_t.W_dec.data.copy_(torch.from_numpy(d["w_dec_txt"]).to(device))
            sae_t.b_dec.data.copy_(torch.from_numpy(d["b_dec_txt"]).to(device))
            sae_t.encoder.weight.data.copy_(torch.from_numpy(d["w_enc_txt"]).to(device))
            sae_t.encoder.bias.data.copy_(torch.from_numpy(d["b_enc_txt"]).to(device))

        result = analyze_variant(
            variant, sae_i, sae_t, img, txt,
            batch_size=args.batch_size, device=device, tau_alive=args.tau_alive,
        )
        per_variant[variant] = result
        out_path = out_dir / f"{variant.replace('+', '__')}_pair_count.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        del sae_i, sae_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_dir / "all_variants_summary.json", "w") as f:
        slim = {v: {kk: vv for kk, vv in d.items() if kk not in ("bin_decoder_cos_values", "matched_C_values", "matched_decoder_cos_values")} for v, d in per_variant.items()}
        json.dump(slim, f, indent=2)
    plot_pair_count_per_bin(per_variant, out_dir / "comparison_pair_count_per_bin.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", required=True, help="Root dir containing variant subdirs (real) or synthetic runs/")
    p.add_argument("--output", required=True, help="Output dir for JSON + plot")
    p.add_argument("--cache-dir", default=None, help="CLIP cache (real mode)")
    p.add_argument("--synthetic", action="store_true", help="Synthetic mode: load .npz dumps")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--tau-alive", type=float, default=0.001)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    if args.synthetic:
        run_synthetic(args, device)
    else:
        run_real(args, device)


if __name__ == "__main__":
    main()
