"""Evaluation metrics for synthetic Theorem 2 experiments.

All functions are stateless; they take model weights (numpy) or SAE model
objects and return dicts of scalar metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.metrics.alignment import _paired_loader, compute_latent_correlation
from src.metrics.normalize import normalize_rows


# ------------------------------------------------------------------
# Decoder-side metrics
# ------------------------------------------------------------------


def compute_mcc_and_uniqueness(
    learned_vectors: np.ndarray,
    gt_matrix: np.ndarray,
) -> dict[str, float]:
    """MCC (Hungarian 1-to-1) + Feature Uniqueness.

    Args:
        learned_vectors: ``(L, d)`` SAE decoder rows.
        gt_matrix: ``(d, N)`` GT atoms as columns.
    """
    if gt_matrix.shape[1] == 0 or learned_vectors.shape[0] == 0:
        return {"mcc": float("nan"), "uniqueness_raw": float("nan"), "uniqueness_norm": float("nan")}

    L = int(learned_vectors.shape[0])
    N = int(gt_matrix.shape[1])

    learned_norm = normalize_rows(learned_vectors.astype(np.float64))
    gt_norm = normalize_rows(gt_matrix.T.astype(np.float64))
    sim = np.abs(learned_norm @ gt_norm.T)

    row_ind, col_ind = linear_sum_assignment(-sim)
    matched = sim[row_ind, col_ind]
    mcc = float(matched.sum() / max(min(L, N), 1))

    best_gt_per_row = sim.argmax(axis=1)
    num_distinct = int(np.unique(best_gt_per_row).size)

    return {
        "mcc": mcc,
        "uniqueness_raw": float(num_distinct / max(L, 1)),
        "uniqueness_norm": float(num_distinct / max(min(L, N), 1)),
    }


def compute_merged_fraction(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    tau: float = 0.95,
) -> float:
    """Fraction of GT shared atoms whose best image-slot column is nearly
    parallel (cos > ``tau``) to its best text-slot column.

    For each shared GT concept g:
      - best_i_idx = argmax_k cos(V[:,k], phi_S[:,g])
      - best_t_idx = argmax_k cos(W[:,k], psi_S[:,g])
    Then count g where ``cos(V[:, best_i_idx], W[:, best_t_idx]) > tau``.

    Captures Theorem-2-style "soft merge" of decoder columns: if aux pressure
    is too strong and pushes the two decoders to point at the same direction
    (typically the bisector of phi_S and psi_S), this metric trips. For a
    correctly trained two-sided SAE preserving alpha < 1, the matched columns
    stay roughly alpha-apart and CR stays near zero.
    """
    if phi_S.size == 0 or psi_S.size == 0:
        return float("nan")
    v = normalize_rows(w_dec_img.astype(np.float64))
    w = normalize_rows(w_dec_txt.astype(np.float64))
    phi = normalize_rows(phi_S.T.astype(np.float64))
    psi = normalize_rows(psi_S.T.astype(np.float64))
    best_i = (v @ phi.T).argmax(axis=0)        # (n_S,)
    best_t = (w @ psi.T).argmax(axis=0)        # (n_S,)
    pair_cos = (v[best_i] * w[best_t]).sum(axis=-1)   # (n_S,)
    return float((pair_cos > tau).mean())


def compute_gre_top1(
    learned_vectors: np.ndarray,
    gt_matrix: np.ndarray,
) -> float:
    """Ground-truth Recovery Error with top-1 activation (per-modality).

    For each GT atom ``g_i`` (column of ``gt_matrix``) and decoder ``V`` whose
    rows are unit atoms in ``d``-space, define ``sigma`` as top-1 activation
    (keep only the single largest entry of the pre-activations, zero the rest):

        GRE := mean_i || g_i - V @ sigma(V.T @ g_i) ||_2^2 .

    Concretely, ``pre = V @ g_i`` has shape ``(L,)``; ``j* = argmax_j pre[j]``;
    ``recon = pre[j*] * V[j*, :]``. Decoder rows and atoms are expected unit
    norm (the function does not renormalize).

    Args:
        learned_vectors: ``(L, d)`` decoder rows.
        gt_matrix: ``(d, n_gt)`` GT atoms as columns.

    Returns:
        Mean per-atom squared reconstruction error, or ``nan`` when empty.
    """
    if gt_matrix.shape[1] == 0 or learned_vectors.shape[0] == 0:
        return float("nan")
    V = learned_vectors.astype(np.float64)
    G = gt_matrix.astype(np.float64)
    pre = V @ G  # (L, n_gt)
    idx = pre.argmax(axis=0)  # (n_gt,)
    n = G.shape[1]
    top_vals = pre[idx, np.arange(n)]  # (n_gt,)
    recon = (V[idx] * top_vals[:, None]).T  # (d, n_gt)
    err = np.sum((G - recon) ** 2, axis=0)
    return float(err.mean())


def compute_recovery_metrics_multi_tau(
    learned_vectors: np.ndarray,
    gt_matrix: np.ndarray,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Multi-threshold MGT + MIP in one pass.

    Args:
        learned_vectors: ``(L, d)`` decoder rows.
        gt_matrix: ``(d, n_gt)`` GT atoms as columns.
    """
    if gt_matrix.shape[1] == 0:
        out: dict[str, float] = {"mip": float("nan")}
        for tau in taus:
            out[f"mgt_tau{tau}"] = float("nan")
        return out
    learned_norm = normalize_rows(learned_vectors.astype(np.float64))
    gt_norm = normalize_rows(gt_matrix.T.astype(np.float64))
    sim = np.abs(learned_norm @ gt_norm.T)
    best = sim.max(axis=0)
    out = {"mip": float(best.mean())}
    for tau in taus:
        out[f"mgt_tau{tau}"] = float((best > tau).mean())
    return out


def compute_joint_mgt(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Joint cross-modal recovery: ``max_i 0.5*(|cos(W_img[i], phi_g)| + |cos(W_txt[i], psi_g)|)``."""
    if (
        phi_S.shape[1] == 0 or psi_S.shape[1] == 0
        or w_dec_img.shape[0] == 0 or w_dec_txt.shape[0] == 0
    ):
        out: dict[str, float] = {"joint_match_mean": float("nan")}
        for tau in taus:
            out[f"joint_mgt_tau{tau}"] = float("nan")
        return out
    Wi = normalize_rows(w_dec_img.astype(np.float64))
    Wt = normalize_rows(w_dec_txt.astype(np.float64))
    Pi = normalize_rows(phi_S.T.astype(np.float64))
    Pt = normalize_rows(psi_S.T.astype(np.float64))
    joint = 0.5 * (np.abs(Wi @ Pi.T) + np.abs(Wt @ Pt.T))
    best = joint.max(axis=0)
    out = {"joint_match_mean": float(best.mean())}
    for tau in taus:
        out[f"joint_mgt_tau{tau}"] = float((best > tau).mean())
    return out


# ------------------------------------------------------------------
# Encoder-side metrics (require live SAE forward pass)
# ------------------------------------------------------------------


def _dense(out: object) -> torch.Tensor:
    """``(B, latent_size)`` dense latents from a TopKSAE output."""
    return out.dense_latents.squeeze(1)  # type: ignore[union-attr]


def probe_gt_pair_activation(
    sae_i,
    sae_t,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    device: torch.device,
    *,
    txt_permutation: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Encoder-side GT probe (Cross-Modal Alignment).

    Feeds pure GT atoms through each SAE and checks agreement.
    """
    n_S = phi_S.shape[1]
    if n_S == 0 or psi_S.shape[1] == 0:
        return {
            "probe_top1_agree": float("nan"),
            "probe_vec_cos": float("nan"),
            "probe_topk_jaccard": float("nan"),
            "probe_top1_both_valid": float("nan"),
        }
    dev = next(sae_i.parameters()).device
    x_img = torch.from_numpy(phi_S.T.astype(np.float32)).unsqueeze(1).to(dev)
    x_txt = torch.from_numpy(psi_S.T.astype(np.float32)).unsqueeze(1).to(dev)
    sae_i.eval()
    sae_t.eval()
    with torch.no_grad():
        out_i = sae_i(hidden_states=x_img, return_dense_latents=True)
        out_t = sae_t(hidden_states=x_txt, return_dense_latents=True)
        z_i = _dense(out_i)
        z_t = _dense(out_t)
    if txt_permutation is not None:
        perm = torch.as_tensor(txt_permutation, dtype=torch.long, device=z_t.device)
        z_t = z_t.index_select(dim=1, index=perm)

    top1_i = z_i.argmax(dim=1)
    top1_t = z_t.argmax(dim=1)
    agree = (top1_i == top1_t).float()
    val_i = z_i.gather(1, top1_i.unsqueeze(1)).squeeze(1)
    val_t = z_t.gather(1, top1_t.unsqueeze(1)).squeeze(1)
    both_valid = ((val_i > 0) & (val_t > 0)).float()
    valid_agree = (agree * both_valid).mean().item()

    vec_cos = F.cosine_similarity(z_i, z_t, dim=1, eps=1e-12).mean().item()

    k = int(sae_i.cfg.k)
    k_eff = min(k, z_i.shape[1], z_t.shape[1])
    topk_i = z_i.topk(k_eff, dim=1).indices.cpu().numpy()
    topk_t = z_t.topk(k_eff, dim=1).indices.cpu().numpy()
    jaccards = []
    for g in range(topk_i.shape[0]):
        a = set(topk_i[g].tolist())
        b = set(topk_t[g].tolist())
        union = len(a | b)
        jaccards.append(len(a & b) / union if union > 0 else 0.0)

    return {
        "probe_top1_agree": float(valid_agree),
        "probe_vec_cos": float(vec_cos),
        "probe_topk_jaccard": float(np.mean(jaccards)),
        "probe_top1_both_valid": float(both_valid.mean().item()),
    }


def compute_posthoc_joint_mgt(
    sae_i,
    sae_t,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    same_model: bool,
    batch_size: int,
    device: torch.device,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Joint MGT after Hungarian-optimal post-hoc reindexing."""
    w_img = sae_i.W_dec.detach().cpu().numpy()
    w_txt = sae_t.W_dec.detach().cpu().numpy()
    if same_model:
        return compute_joint_mgt(w_img, w_txt, phi_S, psi_S, taus)
    C = compute_latent_correlation(sae_i, sae_t, train_img, train_txt, batch_size, device)
    _, col_ind = linear_sum_assignment(-np.abs(C))
    w_txt_aligned = w_txt[col_ind]
    return compute_joint_mgt(w_img, w_txt_aligned, phi_S, psi_S, taus)


def eval_pair_latent_cosine(
    sae_i,
    sae_t,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Mean per-sample latent cosine between paired image/text codes."""
    sae_i.eval()
    sae_t.eval()
    device = next(sae_i.parameters()).device
    loader = _paired_loader(eval_img, eval_txt, batch_size, device)
    total = 0
    sum_cos = 0.0
    with torch.no_grad():
        for img_b, txt_b in loader:
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i, return_dense_latents=True)
            out_t = sae_t(hidden_states=hs_t, return_dense_latents=True)
            zi = _dense(out_i)
            zt = _dense(out_t)
            cos = F.cosine_similarity(zi, zt, dim=1, eps=1e-12)
            sum_cos += float(cos.sum().item())
            total += cos.shape[0]
    return sum_cos / max(total, 1)


def cross_corr_mean_parts(
    sae_i,
    sae_t,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
    m_S: int,
) -> tuple[float, float]:
    """Eval-set Pearson diagonal split into top-``m_S`` and rest."""
    C = compute_latent_correlation(sae_i, sae_t, eval_img, eval_txt, batch_size, device)
    diag = np.diag(C)
    n = diag.shape[0]
    m_S_eff = max(0, min(int(m_S), n))
    top = float(diag[:m_S_eff].mean()) if m_S_eff > 0 else float("nan")
    rest = float(diag[m_S_eff:].mean()) if m_S_eff < n else float("nan")
    return top, rest
