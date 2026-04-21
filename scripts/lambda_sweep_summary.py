"""Compact summary of the aux-alignment lambda sweep on synthetic.

For each variant x lambda combination:
  - RE  (avg_eval_loss)
  - CR_fixed (recomputed compute_merged_fraction with tau=0.95)
  - GRR_img@0.95 / GRR_txt@0.95 (existing in result.json)
  - m_S_hat at rho=0.3 (post-hoc full Hungarian on saved decoders)

Grouped by base variant, lambdas in columns. Values printed as a flat table.
"""

from __future__ import annotations

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


RUN_DIR = Path("outputs/aux_alignment_synthetic_lambda/runs/run_20260420_032105_alpha05_seed1_lambda3")
RHO0 = 0.3
TAU_CR = 0.95


def _build_sae(d, side, device):
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


def m_S_at_rho(npz_path: Path, rho: float, device) -> int:
    d = np.load(npz_path)
    alpha = float(d["alpha_target"])
    seed = int(d["seed"])
    n_shared = int(d["phi_S"].shape[1])
    n_image = int(d["phi_I"].shape[1]) if d["phi_I"].size > 0 else 0
    n_text = int(d["psi_T"].shape[1]) if d["psi_T"].size > 0 else 0
    rep_dim = int(d["phi_S"].shape[0])
    builder = SyntheticPairedBuilder(
        n_shared=n_shared, n_image=n_image, n_text=n_text,
        representation_dim=rep_dim, sparsity=0.99, beta=1.0,
        obs_noise_std=0.05, max_interference=0.1,
        alpha_target=alpha, num_train=10000, num_eval=1000, seed=seed,
    )
    ds = builder.build()
    img = torch.from_numpy(ds["train"]["image_representation"])
    txt = torch.from_numpy(ds["train"]["text_representation"])
    sae_i = _build_sae(d, "img", device)
    sae_t = _build_sae(d, "txt", device)
    C = compute_latent_correlation(sae_i, sae_t, img, txt, 1024, device)
    _, col_ind = linear_sum_assignment(-C)
    matched = C[np.arange(C.shape[0]), col_ind]
    return int((matched > rho).sum())


def main():
    device = torch.device("cpu")
    with open(RUN_DIR / "result.json") as f:
        r = json.load(f)
    seed_data = r["sweep_results"][0]["per_seed"][0]

    base_variants = [
        "recon_only",
        "naive_once",
        "barlow_once",
        "infonce_once",
        "infonce_once+revive",
        "naive_perepoch+revive",
        "barlow_perepoch+revive",
        "infonce_perepoch",
        "infonce_perepoch+revive",
    ]
    lambdas = [0.25, 0.5, 1.0]

    rows = []
    for base in base_variants:
        if base == "recon_only":
            keys = ["recon_only"]
            lams = [None]
        else:
            keys = [f"{base}__l{l}" for l in lambdas]
            lams = lambdas
        for key, lam in zip(keys, lams):
            if key not in seed_data:
                continue
            m = seed_data[key]
            re = m["avg_eval_loss"]
            grr_i = m["img_mgt_shared_tau0.95"]
            grr_t = m["txt_mgt_shared_tau0.95"]
            # Recompute CR_fixed
            safe = key.replace("+", "__")
            npz = RUN_DIR / "params" / f"alpha0.50_seed1_{safe}.npz"
            d = np.load(npz)
            cr = compute_merged_fraction(d["w_dec_img"], d["w_dec_txt"], d["phi_S"], d["psi_S"], tau=TAU_CR)
            ms = m_S_at_rho(npz, RHO0, device)
            rows.append({
                "base": base, "lam": lam,
                "RE": re, "CR": cr,
                "GRR_img": grr_i, "GRR_txt": grr_t,
                "m_S@rho0.3": ms,
            })

    # Print grouped table
    print(f"\n=== Lambda sweep summary (alpha=0.5, seed=1, rho0=0.3, tau_CR=0.95) ===\n")
    print(f"{'variant':30s} {'lam':>6} {'RE':>7} {'CR':>6} {'GRR_i':>7} {'GRR_t':>7} {'mS@.3':>7}")
    print("-" * 80)
    last_base = None
    for row in rows:
        if row["base"] != last_base and last_base is not None:
            print()
        last_base = row["base"]
        lam_s = "-" if row["lam"] is None else f"{row['lam']:.2f}"
        print(f"{row['base']:30s} {lam_s:>6} {row['RE']:>7.4f} {row['CR']:>6.3f} "
              f"{row['GRR_img']:>7.3f} {row['GRR_txt']:>7.3f} {row['m_S@rho0.3']:>7d}")


if __name__ == "__main__":
    main()
