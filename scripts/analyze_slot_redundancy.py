"""Why does post-hoc Hungarian find MORE matched pairs above rho_0 than n_S?

Hypothesis: each GT shared concept is encoded by multiple SAE slots (slot
redundancy). With N_S=1024 GT concepts and L=4096 slots, redundancy ratio
> 1 inflates the apparent matched-pair count.

Diagnostics computed per variant:
  1. Per-GT-atom decoder responses: for each phi_S[:,g], count how many
     image SAE slots have cos similarity > tau_redundancy with it.
  2. Same for psi_S[:,g] / text SAE.
  3. Histogram of matched correlation diagonal (post-Hungarian).
  4. Counts above various thresholds {0.2, 0.3, 0.5, 0.7, 0.9}.
  5. "Active GT atoms": number of GT atoms with at least one slot above
     tau_redundancy, vs n_S.

Usage:
    python scripts/analyze_slot_redundancy.py \\
        --runs-root outputs/aux_alignment_synthetic/runs \\
        --tau-redundancy 0.5 \\
        --device auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.synthetic_paired_builder import SyntheticPairedBuilder
from src.metrics.alignment import compute_latent_correlation
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


def normalize_rows(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def slot_redundancy_per_gt(W: np.ndarray, gt_atoms: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """For each GT atom, count slots whose decoder cos > tau.

    Returns (responses_per_gt[g], best_cos_per_gt[g]).
    """
    Wn = normalize_rows(W.astype(np.float64))         # (L, d)
    Gn = normalize_rows(gt_atoms.T.astype(np.float64))  # (n_gt, d)
    sim = np.abs(Wn @ Gn.T)                            # (L, n_gt)
    responses = (sim > tau).sum(axis=0)                # (n_gt,)
    best = sim.max(axis=0)                             # (n_gt,)
    return responses, best


def analyze_one(npz_path: Path, tau_red: float, device: torch.device, batch_size: int = 1024) -> dict:
    d = np.load(npz_path)
    alpha = float(d["alpha_target"])
    seed = int(d["seed"])
    n_shared = int(d["phi_S"].shape[1])
    n_image = int(d["phi_I"].shape[1]) if d["phi_I"].size > 0 else 0
    n_text = int(d["psi_T"].shape[1]) if d["psi_T"].size > 0 else 0
    rep_dim = int(d["phi_S"].shape[0])

    # 1. Slot redundancy from decoder columns alone (no data needed).
    img_resp, img_best = slot_redundancy_per_gt(d["w_dec_img"], d["phi_S"], tau_red)
    txt_resp, txt_best = slot_redundancy_per_gt(d["w_dec_txt"], d["psi_S"], tau_red)

    # 2. Hungarian on full C.
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
    C = compute_latent_correlation(sae_i, sae_t, img, txt, batch_size, device)
    _, col_ind = linear_sum_assignment(-C)
    matched = C[np.arange(C.shape[0]), col_ind]
    matched_sorted = np.sort(matched)[::-1]

    return {
        "alpha": alpha,
        "img_responses_total": int(img_resp.sum()),       # total image slots responding (cos>tau) to any GT atom
        "img_active_gt": int((img_resp > 0).sum()),       # # GT atoms with at least 1 responding slot
        "img_redundancy_avg": float(img_resp.mean()),     # avg slots per GT atom
        "img_redundancy_max": int(img_resp.max()),
        "img_best_cos_mean": float(img_best.mean()),
        "img_best_cos_above_tau": int((img_best > tau_red).sum()),
        "txt_responses_total": int(txt_resp.sum()),
        "txt_active_gt": int((txt_resp > 0).sum()),
        "txt_redundancy_avg": float(txt_resp.mean()),
        "txt_redundancy_max": int(txt_resp.max()),
        "txt_best_cos_mean": float(txt_best.mean()),
        "txt_best_cos_above_tau": int((txt_best > tau_red).sum()),
        "matched_above_0.2": int((matched > 0.2).sum()),
        "matched_above_0.3": int((matched > 0.3).sum()),
        "matched_above_0.5": int((matched > 0.5).sum()),
        "matched_above_0.7": int((matched > 0.7).sum()),
        "matched_above_0.9": int((matched > 0.9).sum()),
        "matched_top16": matched_sorted[:16].tolist(),
        "matched_percentile_50": float(np.percentile(matched, 50)),
        "matched_percentile_75": float(np.percentile(matched, 75)),
        "matched_percentile_90": float(np.percentile(matched, 90)),
        "matched_percentile_99": float(np.percentile(matched, 99)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", required=True)
    p.add_argument("--tau-redundancy", type=float, default=0.5,
                   help="cos threshold for counting a slot as 'responding' to a GT atom")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    npz_files = sorted(Path(args.runs_root).glob("**/params/*.npz"))
    if not npz_files:
        raise SystemExit("No npz files")

    print(f"\n=== Slot redundancy analysis (tau_red = {args.tau_redundancy}) ===")
    print(f"GT n_S = 1024 shared concepts, per-side L = 4096 slots\n")

    print(f"{'variant':32s} | img:active/total/avg/max | txt:active/total/avg/max | matched > {{0.2,0.3,0.5,0.7,0.9}}")
    print("-" * 150)
    for npz in npz_files:
        stem = npz.stem
        alpha = float(stem.split("_")[0].replace("alpha", ""))
        seed = int(stem.split("_")[1].replace("seed", ""))
        variant = stem.replace(f"alpha{alpha:.2f}_seed{seed}_", "").replace("__", "+")
        try:
            r = analyze_one(npz, args.tau_redundancy, device, args.batch_size)
        except Exception as e:
            print(f"{variant:32s} ERROR: {e}")
            continue
        img = f"{r['img_active_gt']:4d}/{r['img_responses_total']:5d}/{r['img_redundancy_avg']:4.2f}/{r['img_redundancy_max']:3d}"
        txt = f"{r['txt_active_gt']:4d}/{r['txt_responses_total']:5d}/{r['txt_redundancy_avg']:4.2f}/{r['txt_redundancy_max']:3d}"
        m = f"{r['matched_above_0.2']:5d} {r['matched_above_0.3']:5d} {r['matched_above_0.5']:5d} {r['matched_above_0.7']:5d} {r['matched_above_0.9']:5d}"
        print(f"{variant:32s} | {img:25s} | {txt:25s} | {m}")

    print("\nLegend:")
    print("  active = # GT atoms with at least one slot above tau_redundancy")
    print("  total  = total # responses across all GT atoms (sum)")
    print("  avg    = avg slots per GT atom (= total / n_S)")
    print("  max    = max slots responding to any single GT atom")
    print(f"  matched > X = # of post-Hungarian matched pairs whose corr > X (out of L=4096)")


if __name__ == "__main__":
    main()
