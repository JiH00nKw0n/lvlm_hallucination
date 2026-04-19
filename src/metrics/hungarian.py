"""Hungarian matching on alive latents and row-cosine utilities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_alive_match(
    C: np.ndarray,
    alive_i: np.ndarray,
    alive_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian matching restricted to alive latent indices.

    Args:
        C: Full correlation matrix ``(L_i, L_t)``.
        alive_i: Indices of alive image latents.
        alive_t: Indices of alive text latents.

    Returns:
        ``(orig_i, orig_t, C_matched)`` — matched index pairs in the
        original latent space and their correlation values.
    """
    C_sub = C[np.ix_(alive_i, alive_t)]
    r, c = linear_sum_assignment(-C_sub)
    orig_i = alive_i[r]
    orig_t = alive_t[c]
    C_matched = C_sub[r, c]
    return orig_i, orig_t, C_matched


def row_cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity between matrices *u* and *v*.

    Both inputs have shape ``(n, d)``; returns a length-*n* array.
    """
    nu = np.linalg.norm(u, axis=1, keepdims=True) + eps
    nv = np.linalg.norm(v, axis=1, keepdims=True) + eps
    return ((u / nu) * (v / nv)).sum(axis=1)
