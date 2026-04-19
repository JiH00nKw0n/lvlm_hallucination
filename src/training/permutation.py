"""Greedy permutation matching (Algorithm 1) and latent reordering."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def greedy_permutation_match_full(
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper Algorithm 1 (lines 4-16) without rho early stop.

    Repeatedly pulls the largest remaining signed entry to the
    diagonal, producing an ordered row/column pairing.

    Returns:
        ``(P_I_idx, P_T_idx, ordered_diag)``
    """
    n = C.shape[0]
    assert C.shape == (n, n)

    C_work = C.copy()
    P_I = np.arange(n, dtype=np.int64)
    P_T = np.arange(n, dtype=np.int64)
    ordered = np.zeros(n, dtype=np.float64)

    for k in range(n):
        sub = C_work[k:, k:]
        flat_idx = int(np.argmax(sub))
        local_i, local_j = divmod(flat_idx, sub.shape[1])
        i_star = k + local_i
        j_star = k + local_j
        ordered[k] = float(sub[local_i, local_j])

        if i_star != k:
            C_work[[k, i_star]] = C_work[[i_star, k]]
            P_I[[k, i_star]] = P_I[[i_star, k]]
        if j_star != k:
            C_work[:, [k, j_star]] = C_work[:, [j_star, k]]
            P_T[[k, j_star]] = P_T[[j_star, k]]

    return P_I, P_T, ordered


def apply_latent_permutation(
    sae,
    perm_idx: np.ndarray,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """In-place row-permutation of encoder weight/bias and decoder weight.

    ``perm_idx[k] = j`` means new latent *k* is old latent *j*.

    When *optimizer* is supplied the Adam(W) moment buffers are reordered
    identically so they remain aligned with the new row order.
    """
    idx = torch.as_tensor(perm_idx, dtype=torch.long, device=sae.W_dec.device)
    params_to_permute = [sae.encoder.weight, sae.encoder.bias, sae.W_dec]
    with torch.no_grad():
        for p in params_to_permute:
            p.data = p.data[idx].contiguous()
            if optimizer is not None and p in optimizer.state:
                st = optimizer.state[p]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st and isinstance(st[key], torch.Tensor):
                        st[key] = st[key][idx].contiguous()
