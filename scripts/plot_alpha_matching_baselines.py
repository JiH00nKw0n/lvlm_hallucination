#!/usr/bin/env python3
"""Matching-baseline ablation on the alpha sweep ("Why Hungarian?").

Reuses the trained ``two_recon`` checkpoints from
``outputs/theorem2_v2_1R2R_5seeds_coarse/runs/<run>/params/`` and varies *only*
the post-hoc text-side slot permutation π : [L]→[L]. Permutation is computed
once per (alpha, seed) on the training set; ESim/FSim are measured on the
held-out eval pairs (and GT atom probes).

Three matching strategies:
  - dec_cos_hung : Hungarian on -|cos(W_dec_img, W_dec_txt)| (weights only)
  - corr_greedy  : greedy on |C_alive| (alive-restricted standardized Pearson,
                   same matrix as Hungarian, but iterative argmax+mark-used)
  - hungarian    : canonical alive-restricted Hungarian on -|C_alive|
                   (proposed Post-hoc Alignment, src/metrics/canonical_perm.py)

Output:
  - outputs/<root>/matching_baselines.{pdf,png,svg} (2-panel: 1-ESim, 1-FSim)
  - outputs/<root>/.matching_baselines_cache.json   (per (alpha, seed, method))

Usage:
    python scripts/plot_alpha_matching_baselines.py \\
        --params-dir outputs/theorem2_v2_1R2R_5seeds_coarse/runs/run_20260421_011823/params \\
        --config     configs/synthetic/alpha_1R_2R_L8192_5seeds_coarse.yaml \\
        --out        outputs/theorem2_v2_1R2R_5seeds_coarse/matching_baselines.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from palette import STRAWBERRY_RED, SEAWEED, BLUE_SLATE
from src.configs.experiment import ExperimentConfig
from src.datasets.synthetic_paired_builder import SyntheticPairedBuilder
from src.metrics.normalize import normalize_rows
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE


METHODS = [
    ("dec_cos_hung", BLUE_SLATE,     "-D", "Decoder Cosine + Hungarian"),
    ("corr_greedy",  SEAWEED,        "-^", "Correlation Greedy"),
    ("hungarian",    STRAWBERRY_RED, "-o", "Hungarian (Ours)"),
]

PANELS = [
    ("ESim", r"$1 - \mathrm{ESim}$"),
    ("FSim", r"$1 - \mathrm{FSim}$"),
]

NPZ_RE = re.compile(r"^alpha([\d.]+)_seed(\d+)_two_recon\.npz$")


def _parse(name: str):
    m = NPZ_RE.match(name)
    if m is None:
        return None
    return round(float(m.group(1)), 2), int(m.group(2))


def _load_topk(w_enc, b_enc, w_dec, b_dec, d: int, L: int, k: int,
               device: torch.device) -> TopKSAE:
    cfg = TopKSAEConfig(hidden_size=d, latent_size=L, k=k,
                        normalize_decoder=True, weight_tie=False)
    m = TopKSAE(cfg)
    with torch.no_grad():
        m.encoder.weight.copy_(torch.from_numpy(w_enc))
        m.encoder.bias.copy_(torch.from_numpy(b_enc))
        m.W_dec.copy_(torch.from_numpy(w_dec))
        m.b_dec.copy_(torch.from_numpy(b_dec))
    return m.to(device).eval()


@torch.no_grad()
def _dense_latents(sae: TopKSAE, x: torch.Tensor, batch: int,
                   device: torch.device) -> torch.Tensor:
    out_chunks = []
    for i in range(0, x.shape[0], batch):
        hs = x[i:i + batch].to(device=device, dtype=torch.float32).unsqueeze(1)
        out = sae(hidden_states=hs, return_dense_latents=True)
        out_chunks.append(out.dense_latents.squeeze(1).cpu())
    return torch.cat(out_chunks, dim=0)


# ------------------------------------------------------------------
# Permutation builders (text-side slot perm, length L)
# ------------------------------------------------------------------


def perm_decoder_cosine_hungarian(w_dec_img: np.ndarray,
                                  w_dec_txt: np.ndarray) -> np.ndarray:
    """Hungarian on |cos(W_dec_img_row, W_dec_txt_row)| — weights only."""
    L = w_dec_img.shape[0]
    Vi = normalize_rows(w_dec_img.astype(np.float64))
    Vt = normalize_rows(w_dec_txt.astype(np.float64))
    cost = -np.abs(Vi @ Vt.T)
    row, col = linear_sum_assignment(cost)
    perm = np.arange(L, dtype=np.int64)
    perm[row] = col
    return perm


def _alive_corr(zi_train: np.ndarray, zt_train: np.ndarray):
    """Returns (alive_i_idx, alive_t_idx, |C_alive|) with canonical alive
    masking + standardization (same protocol as compute_canonical_perm)."""
    fire_i = (zi_train > 0).any(axis=0)
    fire_t = (zt_train > 0).any(axis=0)
    alive_i = np.where(fire_i)[0]
    alive_t = np.where(fire_t)[0]
    if alive_i.size == 0 or alive_t.size == 0:
        return alive_i, alive_t, np.zeros((0, 0), dtype=np.float64), int(fire_i.sum()), int(fire_t.sum())
    zi_a = zi_train[:, alive_i]
    zt_a = zt_train[:, alive_t]
    mu_i = zi_a.mean(0); sd_i = zi_a.std(0) + 1e-8
    mu_t = zt_a.mean(0); sd_t = zt_a.std(0) + 1e-8
    Zi = (zi_a - mu_i) / sd_i
    Zt = (zt_a - mu_t) / sd_t
    C = np.abs((Zi.T @ Zt) / Zi.shape[0])
    return alive_i, alive_t, C, int(fire_i.sum()), int(fire_t.sum())


def perm_correlation_hungarian(zi_train: np.ndarray, zt_train: np.ndarray
                               ) -> tuple[np.ndarray, int, int]:
    """Canonical alive-restricted Hungarian — same as
    ``compute_canonical_perm`` but reuses pre-computed latents (no second
    forward pass)."""
    L = zi_train.shape[1]
    perm = np.arange(L, dtype=np.int64)
    alive_i, alive_t, C, n_i, n_t = _alive_corr(zi_train, zt_train)
    if alive_i.size and alive_t.size:
        row, col = linear_sum_assignment(-C)
        perm[alive_i[row]] = alive_t[col]
    return perm, n_i, n_t


def perm_correlation_greedy(zi_train: np.ndarray, zt_train: np.ndarray
                            ) -> tuple[np.ndarray, int, int]:
    """Greedy assignment on the same alive-restricted |C_alive| matrix:
    pick global-max |C|, mark row+col used, repeat."""
    L = zi_train.shape[1]
    perm = np.arange(L, dtype=np.int64)
    alive_i, alive_t, C, n_i, n_t = _alive_corr(zi_train, zt_train)
    if alive_i.size and alive_t.size:
        nr, nc = C.shape
        flat_idx = np.argsort(-C, axis=None)
        used_r = np.zeros(nr, dtype=bool)
        used_c = np.zeros(nc, dtype=bool)
        matched = 0
        target = min(nr, nc)
        for k in flat_idx:
            r, c = divmod(int(k), nc)
            if used_r[r] or used_c[c]:
                continue
            perm[alive_i[r]] = alive_t[c]
            used_r[r] = True
            used_c[c] = True
            matched += 1
            if matched >= target:
                break
    return perm, n_i, n_t


# ------------------------------------------------------------------
# Per-(alpha, seed) computation
# ------------------------------------------------------------------


def compute_one(npz_path: Path, cfg: ExperimentConfig, device: torch.device
                ) -> dict[str, dict[str, float]]:
    """Returns {method: {esim, fsim, n_alive_i, n_alive_t}} for one ckpt."""
    npz = np.load(npz_path, allow_pickle=True)
    alpha = float(npz["alpha_target"])
    seed = int(npz["seed"])
    L = int(npz["latent_size_img"])
    d = cfg.data
    t = cfg.training

    sae_i = _load_topk(npz["w_enc_img"], npz["b_enc_img"],
                       npz["w_dec_img"], npz["b_dec_img"],
                       d.representation_dim, L, t.k, device)
    sae_t = _load_topk(npz["w_enc_txt"], npz["b_enc_txt"],
                       npz["w_dec_txt"], npz["b_dec_txt"],
                       d.representation_dim, L, t.k, device)

    # Regenerate paired data with same seed (deterministic builder)
    builder = SyntheticPairedBuilder(
        n_shared=d.n_shared, n_image=d.n_image, n_text=d.n_text,
        representation_dim=d.representation_dim, sparsity=d.sparsity,
        beta=d.beta, obs_noise_std=d.obs_noise_std,
        max_interference=d.max_interference,
        alpha_target=alpha, num_train=d.num_train, num_eval=d.num_eval,
        seed=seed,
    )
    ds = builder.build()
    if getattr(d, "l2_normalize", False):
        for sp in ("train", "eval"):
            for kk in ("image_representation", "text_representation"):
                a = ds[sp][kk]
                n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
                ds[sp][kk] = (a / n).astype(a.dtype, copy=False)
    train_img = torch.from_numpy(ds["train"]["image_representation"])
    train_txt = torch.from_numpy(ds["train"]["text_representation"])
    eval_img = torch.from_numpy(ds["eval"]["image_representation"])
    eval_txt = torch.from_numpy(ds["eval"]["text_representation"])

    # One-time encodings shared across all baselines
    zi_train = _dense_latents(sae_i, train_img, t.batch_size, device).numpy()
    zt_train = _dense_latents(sae_t, train_txt, t.batch_size, device).numpy()
    zi_eval = _dense_latents(sae_i, eval_img, t.batch_size, device)
    zt_eval = _dense_latents(sae_t, eval_txt, t.batch_size, device)

    phi_S = npz["phi_S"]
    psi_S = npz["psi_S"]
    x_phi = torch.from_numpy(phi_S.T.astype(np.float32))
    x_psi = torch.from_numpy(psi_S.T.astype(np.float32))
    zi_phi = _dense_latents(sae_i, x_phi, t.batch_size, device)
    zt_psi = _dense_latents(sae_t, x_psi, t.batch_size, device)

    # Build perms (train data only). Both Hungarian and Greedy reuse
    # the same C_alive matrix → forward only once per ckpt.
    perm_hung, n_alive_i, n_alive_t = perm_correlation_hungarian(zi_train, zt_train)
    perm_grdy, _, _ = perm_correlation_greedy(zi_train, zt_train)
    perm_dec = perm_decoder_cosine_hungarian(
        sae_i.W_dec.detach().cpu().numpy(),
        sae_t.W_dec.detach().cpu().numpy(),
    )

    perms = {
        "dec_cos_hung": perm_dec,
        "corr_greedy":  perm_grdy,
        "hungarian":    perm_hung,
    }

    out: dict[str, dict[str, float]] = {}
    for name, perm in perms.items():
        p = torch.as_tensor(perm, dtype=torch.long)
        esim = torch.nn.functional.cosine_similarity(
            zi_eval, zt_eval[:, p], dim=1, eps=1e-12,
        ).mean().item()
        if phi_S.shape[1] > 0 and psi_S.shape[1] > 0:
            fsim = torch.nn.functional.cosine_similarity(
                zi_phi, zt_psi[:, p], dim=1, eps=1e-12,
            ).mean().item()
        else:
            fsim = float("nan")
        out[name] = {
            "esim": float(esim),
            "fsim": float(fsim),
            "n_alive_i": n_alive_i,
            "n_alive_t": n_alive_t,
        }
        print(f"  α={alpha:.2f} seed={seed} {name:>14s}: "
              f"ESim={esim:.4f} FSim={fsim:.4f} "
              f"alive_i={n_alive_i} alive_t={n_alive_t}")

    del sae_i, sae_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------
# Aggregate + plot
# ------------------------------------------------------------------


def aggregate(records: dict, alphas: list[float]
              ) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Reshape to {metric: {method: (alphas_arr, means, stds)}}."""
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for metric_lc, metric_panel in (("esim", "ESim"), ("fsim", "FSim")):
        out[metric_panel] = {}
        for method, *_ in METHODS:
            xs, means, stds = [], [], []
            for a in alphas:
                vals = [v[metric_lc] for (aa, _s, m), v in records.items()
                        if aa == a and m == method and not np.isnan(v[metric_lc])]
                if not vals:
                    continue
                xs.append(a)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            out[metric_panel][method] = (
                np.array(xs), np.array(means), np.array(stds),
            )
    return out


def make_fig(series, alphas: list[float], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.4))
    handles: list = []

    for i, (ax, (metric, ylabel)) in enumerate(zip(axes, PANELS)):
        for method, color, style, label in METHODS:
            xs, means, stds = series[metric][method]
            if means.size == 0:
                continue
            # Plot 1 - mean → distance convention (lower = better)
            y = 1.0 - means
            h, = ax.plot(xs, y, style, color=color, lw=1.2, ms=3,
                         label=label, zorder=3)
            ax.errorbar(xs, y, yerr=stds, fmt="none", ecolor=color,
                        capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
            if i == 0:
                handles.append(h)

        ax.set_xlabel(r"$\cos(\phi_i, \psi_i)$", fontsize=8, labelpad=1)
        ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(sorted(alphas))
        ax.tick_params(axis="both", labelsize=6.5, pad=1)
        ax.grid(alpha=0.15, linewidth=0.4, which="both")

    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False, fontsize=7,
        handlelength=1.5, columnspacing=1.5, handletextpad=0.4,
    )
    plt.subplots_adjust(top=0.78, bottom=0.22, left=0.10, right=0.99, wspace=0.32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    print(f"saved {out_path}")
    if str(out_path).endswith(".pdf"):
        for ext in (".png", ".svg"):
            sib = str(out_path).replace(".pdf", ext)
            plt.savefig(sib, dpi=200, bbox_inches="tight",
                        facecolor="white", pad_inches=0.04)
            print(f"saved {sib}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute even when cache file exists")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    alphas = [round(float(x), 2) for x in args.alphas.split(",")]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    out_path = Path(args.out)
    cache_path = out_path.parent / ".matching_baselines_cache.json"
    records: dict[tuple[float, int, str], dict[str, float]] = {}
    if cache_path.exists() and not args.no_cache:
        try:
            raw = json.load(open(cache_path))
            for k, v in raw.items():
                a_s, seed_s, method = k.split("|")
                records[(round(float(a_s), 2), int(seed_s), method)] = v
            print(f"loaded {len(records)} cached entries from {cache_path}")
        except Exception as e:
            print(f"cache load failed ({e}); recomputing")
            records = {}

    # Walk checkpoints
    files: list[tuple[float, int, Path]] = []
    for p in sorted(Path(args.params_dir).glob("alpha*_seed*_two_recon.npz")):
        parsed = _parse(p.name)
        if parsed is None:
            continue
        a, seed = parsed
        if a not in alphas:
            continue
        files.append((a, seed, p))
    print(f"found {len(files)} two_recon checkpoints across {len(alphas)} alphas")

    # Compute (skip cached)
    needed_methods = {m for m, *_ in METHODS}
    for a, seed, p in files:
        have = {m for (aa, ss, m) in records.keys()
                if aa == a and ss == seed}
        if needed_methods.issubset(have):
            continue
        print(f"[compute] α={a:.2f} seed={seed}  ckpt={p.name}")
        out = compute_one(p, cfg, device)
        for method, vals in out.items():
            records[(a, seed, method)] = vals

        # Persist cache after every ckpt (cheap, safe under interruption)
        try:
            serial = {f"{a}|{s}|{m}": v for (a, s, m), v in records.items()}
            json.dump(serial, open(cache_path, "w"), indent=2)
        except Exception as e:
            print(f"  warn: cache save failed: {e}")

    series = aggregate(records, alphas)

    # Sanity log
    print("\n=== Aggregated (1 - metric) ===")
    for metric, _ in PANELS:
        print(f"-- {metric}")
        for method, *_ in METHODS:
            xs, means, stds = series[metric][method]
            for a, m, s in zip(xs, means, stds):
                print(f"  {method:>14s} α={a:.2f}  "
                      f"raw={m:.4f}±{s:.4f}  1-x={1-m:.4f}")

    make_fig(series, alphas, out_path)


if __name__ == "__main__":
    main()
