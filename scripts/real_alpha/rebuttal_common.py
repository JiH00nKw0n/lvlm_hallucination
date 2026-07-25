"""Shared pieces for the rebuttal analyses, so three scripts cannot drift apart.

Two definitions live here because getting them subtly different between scripts
would silently change the numbers we report.

*Alive.* Two rules are in play and they are not interchangeable. The method's
permutation treats a latent as alive when it fires at least once, which is what
``eval_utils.build_perm`` does. The density figure is stricter and requires a
firing rate above 1e-3, roughly 570 of COCO's 567k rows. Analyses about the
permutation use the first rule; analyses that reproduce the figure use the
second. ``ALIVE_RULES`` names both so a caller has to pick one on purpose.

*Matching.* The permutation maximizes total co-activation correlation over
one-to-one assignments, restricted to alive latents on both sides, exactly as
in ``eval_utils.build_perm``: dead rows and columns are pushed to a large
negative cost so they cannot take an alive latent's partner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment

# name -> minimum firing rate for a latent to count as alive
ALIVE_RULES = {
    "ever": 0.0,        # fires at least once — the permutation's rule
    "density": 1e-3,    # the density figure's rule
}

_BIG_NEG = -1e9


def load_panel(npz_path: str | Path) -> dict:
    """Correlation matrix and per-side firing rates written by build_cross_pair_C."""
    z = np.load(npz_path)
    return {
        "C": z["C"].astype(np.float32),
        "rate_a": z["rate_a"].astype(np.float64),
        "rate_b": z["rate_b"].astype(np.float64),
        "n_samples": int(z["n_samples"]),
    }


def alive_masks(panel: dict, rule: str = "ever") -> tuple[np.ndarray, np.ndarray]:
    thr = ALIVE_RULES[rule]
    return panel["rate_a"] > thr, panel["rate_b"] > thr


def unit_decoder(ckpt: str | Path, side: str) -> np.ndarray:
    """Decoder directions as unit-norm rows. ``side`` is 'image' or 'text'."""
    sd = load_file(str(Path(ckpt) / "model.safetensors"))
    w = sd[f"{side}_sae.W_dec"].float().numpy()
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def hungarian_perm(C: np.ndarray, alive_a: np.ndarray, alive_b: np.ndarray) -> dict:
    """One-to-one assignment maximizing total correlation among alive latents.

    Returns the full-length permutation (so ``perm[i]`` is the partner of left
    latent ``i``) along with the matched correlations and a mask marking which
    rows are genuinely alive-to-alive. Rows outside that mask carry an arbitrary
    partner and must be excluded before any statistic is computed — including
    them drags every summary toward the value for unmatched noise.
    """
    Cm = np.array(C, dtype=np.float64, copy=True)
    Cm[~alive_a, :] = _BIG_NEG
    Cm[:, ~alive_b] = _BIG_NEG
    Cm = np.nan_to_num(Cm, nan=_BIG_NEG, posinf=1.0, neginf=_BIG_NEG)

    row, col = linear_sum_assignment(-Cm)
    perm = np.zeros(C.shape[0], dtype=np.int64)
    perm[row] = col

    usable = alive_a.copy()
    usable[row] &= alive_b[col]
    matched_c = np.full(C.shape[0], np.nan, dtype=np.float64)
    matched_c[row] = C[row, col]

    return {
        "perm": perm,
        "usable": usable,               # alive on both sides
        "matched_c": matched_c,         # correlation of each row's assigned partner
        "n_alive_a": int(alive_a.sum()),
        "n_alive_b": int(alive_b.sum()),
        "n_usable": int(usable.sum()),
    }


def matched_distance(Wa: np.ndarray, Wb: np.ndarray, perm: np.ndarray,
                     usable: np.ndarray) -> np.ndarray:
    """Cosine distance between each usable latent and its assigned partner."""
    rows = np.where(usable)[0]
    cos = (Wa[rows] * Wb[perm[rows]]).sum(axis=1)
    return 1.0 - cos


def describe(values: np.ndarray) -> dict:
    """Percentile summary used in every report this module feeds."""
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }
