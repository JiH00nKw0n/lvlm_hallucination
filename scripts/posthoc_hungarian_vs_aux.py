"""Does post-hoc Hungarian on recon_only match aux-trained joint_mgt?

For each saved variant, compute joint_mgt_tau0.95 BEFORE and AFTER applying
post-hoc full Hungarian to the saved decoders (permute text side only).

If recon_only + post-hoc Hungarian achieves similar joint_mgt to aux-trained
variants, aux loss during training is redundant relative to end-of-training
permutation.
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
from src.metrics.synthetic_eval import compute_joint_mgt, compute_merged_fraction
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE


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


def analyze(npz_path: Path, device) -> dict:
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

    # Compute cross-correlation C
    C = compute_latent_correlation(sae_i, sae_t, img, txt, 1024, device)

    # ORIGINAL (no permutation)
    W_img = d["w_dec_img"]
    W_txt = d["w_dec_txt"]
    taus = (0.95,)
    mgt_orig = compute_joint_mgt(W_img, W_txt, d["phi_S"], d["psi_S"], taus)
    cr_orig = compute_merged_fraction(W_img, W_txt, d["phi_S"], d["psi_S"], tau=0.95)

    # POST-HOC HUNGARIAN: find optimal matching, permute text side
    # col_ind[i] = the text slot matched to image slot i
    row_ind, col_ind = linear_sum_assignment(-C)
    matched_corr = C[row_ind, col_ind]
    # Permute text decoder rows: new text slot i = old text slot col_ind[i]
    W_txt_perm = W_txt[col_ind]
    mgt_perm = compute_joint_mgt(W_img, W_txt_perm, d["phi_S"], d["psi_S"], taus)
    cr_perm = compute_merged_fraction(W_img, W_txt_perm, d["phi_S"], d["psi_S"], tau=0.95)

    return {
        "joint_mgt_orig": mgt_orig["joint_mgt_tau0.95"],
        "joint_mgt_posthoc": mgt_perm["joint_mgt_tau0.95"],
        "CR_orig": cr_orig,
        "CR_posthoc": cr_perm,
        "matched_C_above_0.3": int((matched_corr > 0.3).sum()),
        "matched_C_above_0.5": int((matched_corr > 0.5).sum()),
    }


def main():
    device = torch.device("cpu")
    run_dir = Path("outputs/aux_alignment_synthetic_lambda/runs/run_20260420_032105_alpha05_seed1_lambda3")

    # Set of variants we care about: recon_only + l=0.25 of each aux
    targets = [
        "recon_only",
        "naive_once__l0.25",
        "naive_once__l1.0",
        "barlow_once__l0.25",
        "infonce_once__l0.25",
        "infonce_once__l1.0",
        "naive_perepoch+revive__l0.25",
        "naive_perepoch+revive__l1.0",
        "infonce_perepoch__l0.25",
        "infonce_perepoch__l1.0",
    ]

    print(f"{'variant':35s} {'mgt_orig':>9} {'mgt_posthoc':>12} {'CR_orig':>8} {'CR_posthoc':>11}")
    print("-" * 90)
    for v in targets:
        safe = v.replace("+", "__")
        npz = run_dir / "params" / f"alpha0.50_seed1_{safe}.npz"
        if not npz.exists():
            print(f"{v:35s} missing npz")
            continue
        r = analyze(npz, device)
        print(f"{v:35s} {r['joint_mgt_orig']:>9.3f} {r['joint_mgt_posthoc']:>12.3f} "
              f"{r['CR_orig']:>8.3f} {r['CR_posthoc']:>11.3f}")


if __name__ == "__main__":
    main()
