"""Auxiliary loss functions for SAE paired training.

Includes group-sparse and trace losses (previously in
``synthetic_theory_simplified.py``) and iso/alignment losses
(previously in ``synthetic_theorem2_method.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def group_sparse_loss(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """L_{2,1} norm over paired sparse codes."""
    return torch.sqrt(z_img.pow(2) + z_txt.pow(2) + 1e-12).sum(dim=-1).mean()


def trace_alignment_loss(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """Trace-based alignment: ``-Tr(Z_img_norm @ Z_txt_norm^T) / B``."""
    b = z_img.shape[0]
    z_img_norm = F.normalize(z_img, p=2, dim=-1)
    z_txt_norm = F.normalize(z_txt, p=2, dim=-1)
    return -(z_img_norm * z_txt_norm).sum() / b


def iso_alignment_penalty(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """IsoEnergy alignment penalty (top-1-masked cosine).

    Masks each modality's top-1 latent index in *both* halves before
    computing cosine similarity.  Returns a scalar in ``[-1, 1]``.
    """
    n = z_img.shape[0]
    top_1_img = torch.topk(z_img, k=1, dim=1).indices.squeeze(1)
    top_1_txt = torch.topk(z_txt, k=1, dim=1).indices.squeeze(1)
    arange = torch.arange(n, device=z_img.device)

    mask_i = torch.ones_like(z_img)
    mask_t = torch.ones_like(z_txt)
    mask_i[arange, top_1_img] = 0.0
    mask_i[arange, top_1_txt] = 0.0
    mask_t[arange, top_1_img] = 0.0
    mask_t[arange, top_1_txt] = 0.0

    masked_i = z_img * mask_i
    masked_t = z_txt * mask_t
    cos = F.cosine_similarity(masked_i, masked_t, dim=1).mean()
    return -cos


def auxiliary_alignment_loss(
    z_i: torch.Tensor,
    z_t: torch.Tensor,
    m_S: int,
    norm: str = "group",
) -> torch.Tensor:
    """Diag-only auxiliary alignment loss (Algorithm 1).

    For the first ``m_S`` latent dims push diagonal correlation to 1;
    for the rest push to 0.

    ``norm="group"``: each term averaged over its own group size.
    ``norm="global"``: both sums divided by the total latent count.
    """
    zi = z_i - z_i.mean(dim=0, keepdim=True)
    zt = z_t - z_t.mean(dim=0, keepdim=True)
    si = zi.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    st = zt.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    diag = (zi * zt).mean(dim=0) / (si * st)

    zero = diag.new_tensor(0.0)
    n = diag.shape[0]

    if norm == "group":
        shared_term = ((diag[:m_S] - 1.0) ** 2).mean() if m_S > 0 else zero
        private_term = (diag[m_S:] ** 2).mean() if m_S < n else zero
    elif norm == "global":
        shared_sum = ((diag[:m_S] - 1.0) ** 2).sum() if m_S > 0 else zero
        private_sum = (diag[m_S:] ** 2).sum() if m_S < n else zero
        shared_term = shared_sum / max(n, 1)
        private_term = private_sum / max(n, 1)
    else:
        raise ValueError(f"Unknown norm '{norm}'; expected 'group' or 'global'.")
    return shared_term + private_term
