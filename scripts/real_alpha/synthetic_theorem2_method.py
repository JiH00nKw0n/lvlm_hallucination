"""Streaming signed-Pearson latent correlation for Diagnostic B.

Restores `_compute_latent_correlation`, which `run_diagnostic_B.py` imports but
which went missing from the tree (it produced the per-model
`diagnostic_B_C_<split>.npy` matrices behind the multi-model density figure).

C[i, j] = corr_s( z_I[s, i], z_T[s, j] ) over paired samples, where z_I / z_T
are the pre-TopK dense latents of the two-sided SAE. Computed with a single
streaming pass (fp64 accumulators, on-device fp32 partials) so the full
(L x L) matrix is exact at COCO scale without materializing (N, L) tensors.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def _dense_latents(sae, x: torch.Tensor) -> torch.Tensor:
    """(B, hidden) -> (B, L) pre-TopK... actually post-TopK dense latents,
    matching the SAE's returned dense_latents (top-k scattered into L)."""
    out = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True)
    return out.dense_latents.squeeze(1).float()


@torch.no_grad()
def _compute_latent_correlation(sae_i, sae_t, img: torch.Tensor, txt: torch.Tensor,
                                batch_size: int, device: torch.device) -> np.ndarray:
    """Signed Pearson correlation matrix C (L_i, L_t) between the two SAEs'
    dense latents over the paired (img, txt) rows."""
    sae_i.eval().to(device)
    sae_t.eval().to(device)
    L_i = int(sae_i.latent_size)
    L_t = int(sae_t.latent_size)
    N = img.shape[0]

    sum_i = torch.zeros(L_i, dtype=torch.float64)
    sum_t = torch.zeros(L_t, dtype=torch.float64)
    sumsq_i = torch.zeros(L_i, dtype=torch.float64)
    sumsq_t = torch.zeros(L_t, dtype=torch.float64)
    cross = torch.zeros(L_i, L_t, dtype=torch.float64)

    # on-device fp32 partials, flushed periodically
    p_si = torch.zeros(L_i, device=device)
    p_st = torch.zeros(L_t, device=device)
    p_qi = torch.zeros(L_i, device=device)
    p_qt = torch.zeros(L_t, device=device)
    p_cross = torch.zeros(L_i, L_t, device=device)
    FLUSH = 64

    def flush():
        nonlocal sum_i, sum_t, sumsq_i, sumsq_t, cross
        sum_i += p_si.cpu().double(); p_si.zero_()
        sum_t += p_st.cpu().double(); p_st.zero_()
        sumsq_i += p_qi.cpu().double(); p_qi.zero_()
        sumsq_t += p_qt.cpu().double(); p_qt.zero_()
        cross += p_cross.cpu().double(); p_cross.zero_()

    n_batches = (N + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, N)
        zi = _dense_latents(sae_i, img[s:e].to(device))
        zt = _dense_latents(sae_t, txt[s:e].to(device))
        p_si += zi.sum(0)
        p_st += zt.sum(0)
        p_qi += (zi * zi).sum(0)
        p_qt += (zt * zt).sum(0)
        p_cross += zi.T @ zt
        if (b + 1) % FLUSH == 0:
            flush()
    flush()

    mean_i = sum_i / N
    mean_t = sum_t / N
    cov = cross / N - torch.outer(mean_i, mean_t)
    var_i = (sumsq_i / N - mean_i ** 2).clamp_min(0.0)
    var_t = (sumsq_t / N - mean_t ** 2).clamp_min(0.0)
    denom = torch.outer(var_i.sqrt(), var_t.sqrt()).clamp_min(1e-12)
    C = (cov / denom).numpy()
    return np.nan_to_num(C, nan=0.0).astype(np.float64)


__all__ = ["_compute_latent_correlation"]
