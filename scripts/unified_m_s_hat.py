"""Unified post-hoc Hungarian-based $\\hat m_S$ measurement.

For each saved synthetic .npz dump, run full Hungarian on the EOT
cross-correlation matrix C and report:
  - m_S_hat_rho    : count of matched-pair correlations > rho_0
  - top_k_diag     : top-k matched diagonal values (debug)
  - CR_fixed       : decoder-cosine collapse rate (using fixed metric)

Independent of how the variant reached its final state, this gives an apples-
to-apples count of "shared concepts above threshold" at training end.

Usage:
    python scripts/unified_m_s_hat.py \\
        --runs-root outputs/aux_alignment_synthetic/runs \\
        --rho0 0.2 \\
        --tau-cr 0.95 \\
        --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.synthetic_paired_builder import SyntheticPairedBuilder
from src.metrics.alignment import compute_latent_correlation
from src.metrics.synthetic_eval import compute_merged_fraction
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _build_sae_from_npz(d, side: str, device: torch.device) -> TopKSAE:
    """Reconstruct a TopKSAE from saved tensors. side in {'img','txt'}."""
    rep_dim = int(d["phi_S"].shape[0])
    latent = int(d[f"latent_size_{side}"])
    cfg = TopKSAEConfig(hidden_size=rep_dim, latent_size=latent, k=16, normalize_decoder=True)
    sae = TopKSAE(cfg).to(device)
    with torch.no_grad():
        sae.W_dec.data.copy_(torch.from_numpy(d[f"w_dec_{side}"]).to(device))
        sae.b_dec.data.copy_(torch.from_numpy(d[f"b_dec_{side}"]).to(device))
        sae.encoder.weight.data.copy_(torch.from_numpy(d[f"w_enc_{side}"]).to(device))
        sae.encoder.bias.data.copy_(torch.from_numpy(d[f"b_enc_{side}"]).to(device))
    return sae


def measure_one(npz_path: Path, rho0: float, tau_cr: float, device: torch.device, batch_size: int = 1024) -> dict:
    d = np.load(npz_path)
    alpha = float(d["alpha_target"])
    seed = int(d["seed"])

    n_shared = int(d["phi_S"].shape[1])
    n_image = int(d["phi_I"].shape[1]) if d["phi_I"].size > 0 else 0
    n_text = int(d["psi_T"].shape[1]) if d["psi_T"].size > 0 else 0
    rep_dim = int(d["phi_S"].shape[0])

    # Regenerate train data deterministically with the same seed/builder.
    builder = SyntheticPairedBuilder(
        n_shared=n_shared, n_image=n_image, n_text=n_text,
        representation_dim=rep_dim, sparsity=0.99, beta=1.0,
        obs_noise_std=0.05, max_interference=0.1,
        alpha_target=alpha, num_train=10000, num_eval=1000, seed=seed,
    )
    ds = builder.build()
    img = torch.from_numpy(ds["train"]["image_representation"])
    txt = torch.from_numpy(ds["train"]["text_representation"])

    sae_i = _build_sae_from_npz(d, "img", device)
    sae_t = _build_sae_from_npz(d, "txt", device)

    C = compute_latent_correlation(sae_i, sae_t, img, txt, batch_size, device)
    # Optimal full-matrix Hungarian: maximises sum of matched diag.
    row_ind, col_ind = linear_sum_assignment(-C)
    matched_diag = C[row_ind, col_ind]
    matched_diag_sorted = np.sort(matched_diag)[::-1]
    m_S_hat = int((matched_diag > rho0).sum())

    cr_fixed = compute_merged_fraction(d["w_dec_img"], d["w_dec_txt"], d["phi_S"], d["psi_S"], tau=tau_cr)

    return {
        "alpha": alpha,
        "seed": seed,
        "m_S_hat_unified": m_S_hat,
        "rho0": rho0,
        "matched_diag_top8": matched_diag_sorted[:8].tolist(),
        "matched_diag_above_rho0_min": float(matched_diag_sorted[m_S_hat - 1]) if m_S_hat > 0 else None,
        "matched_diag_first_below_rho0": float(matched_diag_sorted[m_S_hat]) if m_S_hat < len(matched_diag_sorted) else None,
        "CR_fixed": cr_fixed,
        "tau_cr": tau_cr,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", required=True, help="dir containing **/params/*.npz")
    p.add_argument("--rho0", type=float, default=0.2)
    p.add_argument("--tau-cr", type=float, default=0.95)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    device = _resolve_device(args.device)
    runs_root = Path(args.runs_root)
    npz_files = sorted(runs_root.glob("**/params/*.npz"))
    if not npz_files:
        raise SystemExit(f"No npz files under {runs_root}/**/params/")

    # Display variant order: parse name from filename
    rows = []
    print(f"{'variant':50s} {'m_S_unified':>12} {'top1_C':>8} {'min>ρ':>8} {'1st<ρ':>8} {'CR_fixed':>10}")
    print("-" * 100)
    for npz in npz_files:
        # filename format: alpha{a:.2f}_seed{s}_{variant}.npz
        stem = npz.stem
        alpha = float(stem.split("_")[0].replace("alpha", ""))
        seed = int(stem.split("_")[1].replace("seed", ""))
        variant = stem.replace(f"alpha{alpha:.2f}_seed{seed}_", "").replace("__", "+")
        try:
            r = measure_one(npz, args.rho0, args.tau_cr, device, args.batch_size)
        except Exception as e:
            print(f"{variant:50s} ERROR: {e}")
            continue
        r["variant"] = variant
        r["npz"] = str(npz)
        rows.append(r)
        top1 = r["matched_diag_top8"][0] if r["matched_diag_top8"] else float("nan")
        mlast = r["matched_diag_above_rho0_min"]
        first_below = r["matched_diag_first_below_rho0"]
        print(f"{variant:50s} {r['m_S_hat_unified']:>12d} {top1:>8.3f} "
              f"{(f'{mlast:.3f}' if mlast is not None else '-'):>8} "
              f"{(f'{first_below:.3f}' if first_below is not None else '-'):>8} "
              f"{r['CR_fixed']:>10.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
