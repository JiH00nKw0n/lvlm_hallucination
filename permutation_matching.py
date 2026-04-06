"""Post-hoc permutation matching for comparing independent SAE decoders."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "optimal_permutation",
    "posthoc_alignment_metrics",
]


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def optimal_permutation(
    decoder_a: np.ndarray,
    decoder_b: np.ndarray,
) -> np.ndarray:
    """Find permutation P maximizing sum_i cos(a_i, b_{P(i)}).

    Args:
        decoder_a: (k, d) decoder rows of SAE A.
        decoder_b: (k, d) decoder rows of SAE B.

    Returns:
        Permutation array P of shape (k,) such that decoder_b[P] best matches decoder_a.
    """
    a_norm = _normalize_rows(decoder_a.astype(np.float64))
    b_norm = _normalize_rows(decoder_b.astype(np.float64))
    cos_sim = np.abs(a_norm @ b_norm.T)  # (k, k)
    cost = -cos_sim  # linear_sum_assignment minimizes
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty(decoder_a.shape[0], dtype=np.int64)
    perm[row_ind] = col_ind
    return perm


def posthoc_alignment_metrics(
    decoder_a: np.ndarray,
    decoder_b: np.ndarray,
    permutation: np.ndarray,
    shared_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute post-hoc alignment metrics after optimal matching.

    Args:
        decoder_a: (k, d) decoder rows of SAE A.
        decoder_b: (k, d) decoder rows of SAE B.
        permutation: (k,) permutation array.
        shared_indices: if given, indices of shared latent units for separate metric.

    Returns:
        Dict with mean_cosine_all, mean_cosine_shared.
    """
    a_norm = _normalize_rows(decoder_a.astype(np.float64))
    b_norm = _normalize_rows(decoder_b.astype(np.float64))
    b_matched = b_norm[permutation]
    cos_all = np.abs((a_norm * b_matched).sum(axis=1))

    result: dict[str, float] = {
        "mean_cosine_all": float(cos_all.mean()),
    }

    if shared_indices is not None and len(shared_indices) > 0:
        cos_shared = cos_all[shared_indices]
        result["mean_cosine_shared"] = float(cos_shared.mean())
    else:
        result["mean_cosine_shared"] = float("nan")

    return result
