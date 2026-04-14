"""
Synthetic Theorem 2 / Algorithm 1 validation.

Trains five methods on the same multimodal synthetic data and compares
reconstruction quality and cross-modal latent alignment:

    1. single_recon   : one shared SAE, reconstruction only
    2. two_recon      : two independent SAEs, reconstruction only
    3. group_sparse   : one shared SAE, recon + L_{2,1} aux (paper lambda = 0.05)
    4. trace_align    : one shared SAE, recon + trace aux (paper beta = 1e-4)
    5. ours           : two SAEs, Stage 1 recon -> greedy latent permutation ->
                        Stage 2 recon + diag-only auxiliary alignment loss

Methods 1-4 reuse existing helpers from the repo; method 5 implements
Algorithm 1 from the paper with user-supplied m_S (not rho-threshold).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from synthetic_sae_theory_experiment import (
    _seed_everything,
    _resolve_device,
    _normalize_rows,
)
from synthetic_theorem1_unified import (
    _eval_loss_single_modality,
    _gt_based_shared_alignment_mismatched,
)
from synthetic_theory_simplified import group_sparse_loss, trace_alignment_loss
from src.datasets.synthetic_theory_feature import SyntheticTheoryFeatureBuilder
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Low-level helpers                                                  #
# ------------------------------------------------------------------ #


def _make_sae(args: argparse.Namespace, latent_size: int, device: torch.device) -> TopKSAE:
    cfg = TopKSAEConfig(
        hidden_size=args.representation_dim,
        latent_size=latent_size,
        k=args.k,
        normalize_decoder=True,
    )
    model = TopKSAE(cfg)
    return model.to(device)  # type: ignore[return-value]


def _paired_loader(
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(train_img, train_txt),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _sae_step(
    model: TopKSAE,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    max_grad_norm: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if model.W_dec.grad is not None:
        model.remove_gradient_parallel_to_decoder_directions()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    model.set_decoder_norm_to_unit_norm()


def _joint_step(
    models: list[TopKSAE],
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    max_grad_norm: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for m in models:
        if m.W_dec.grad is not None:
            m.remove_gradient_parallel_to_decoder_directions()
    if max_grad_norm > 0:
        params = [p for m in models for p in m.parameters()]
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    for m in models:
        m.set_decoder_norm_to_unit_norm()


def _dense_latents(out) -> torch.Tensor:
    """Return (B, latent_size) dense latents from a TopKSAE forward call made
    with `return_dense_latents=True`. Squeezes the singleton seq dim."""
    return out.dense_latents.squeeze(1)


# ------------------------------------------------------------------ #
# Loss logging helper                                                #
# ------------------------------------------------------------------ #

def _log_interval(total_batches: int) -> int:
    return max(1, total_batches // 10)


# ------------------------------------------------------------------ #
# Method 1: single recon (shared SAE)                                #
# ------------------------------------------------------------------ #


def _train_single_recon(
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    args: argparse.Namespace,
    latent_size: int,
    seed: int,
    device: torch.device,
) -> tuple[TopKSAE, list[dict[str, float]]]:
    _seed_everything(seed)
    model = _make_sae(args, latent_size, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loader = _paired_loader(
        train_img, train_txt, args.batch_size, device,
        shuffle=True, drop_last=True,
    )
    loss_history: list[dict] = []
    total_batches = len(loader)
    log_every = _log_interval(total_batches)
    pbar = tqdm(range(args.num_epochs), desc="single_recon", leave=False)
    for ep in pbar:
        last = 0.0
        for bi, (img_b, txt_b) in enumerate(loader):
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = model(hidden_states=hs_i)
            out_t = model(hidden_states=hs_t)
            loss = out_i.recon_loss + out_t.recon_loss
            _sae_step(model, optimizer, loss, args.max_grad_norm)
            last = float(loss.detach().item())
            if (bi + 1) % log_every == 0 or bi == total_batches - 1:
                loss_history.append({"t": round(ep + (bi + 1) / total_batches, 2), "rec": last})
        pbar.set_postfix(loss=f"{last:.4f}")
    pbar.close()
    return model, loss_history


# ------------------------------------------------------------------ #
# Method 2: two recon (independent SAEs)                             #
# ------------------------------------------------------------------ #


def _train_two_recon(
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    args: argparse.Namespace,
    latent_size: int,
    seed: int,
    device: torch.device,
    *,
    num_epochs: Optional[int] = None,
) -> tuple[TopKSAE, TopKSAE, list[dict]]:
    """Two independent SAEs, each with `latent_size // 2` so the total latent
    budget matches the single-SAE baselines (Theorem 1 fair-comparison
    convention).

    Joint-step training: on every batch both SAEs process their respective
    (paired) sample and we take one AdamW step over the concatenated parameter
    set. The two SAEs' parameters are disjoint so this is mathematically
    equivalent to running two independent optimizers, but it keeps the
    training structure aligned with `_train_ours` Stage 1 and the other
    methods (paired loader, `drop_last=True`, summed loss).
    """
    n_epochs: int = int(num_epochs) if num_epochs is not None else int(args.num_epochs)
    sub_latent = latent_size // 2
    _seed_everything(seed)
    sae_i = _make_sae(args, sub_latent, device)
    _seed_everything(seed + 10000)
    sae_t = _make_sae(args, sub_latent, device)
    sae_i.train()
    sae_t.train()

    optimizer = torch.optim.AdamW(
        list(sae_i.parameters()) + list(sae_t.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    loader = _paired_loader(
        train_img, train_txt, args.batch_size, device,
        shuffle=True, drop_last=True,
    )

    loss_history: list[dict] = []
    total_batches = len(loader)
    log_every = _log_interval(total_batches)
    pbar = tqdm(range(n_epochs), desc="two_recon", leave=False)
    for ep in pbar:
        last_i = last_t = 0.0
        for bi, (img_b, txt_b) in enumerate(loader):
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i)
            out_t = sae_t(hidden_states=hs_t)
            loss = out_i.recon_loss + out_t.recon_loss
            _joint_step([sae_i, sae_t], optimizer, loss, args.max_grad_norm)
            last_i = float(out_i.recon_loss.detach().item())
            last_t = float(out_t.recon_loss.detach().item())
            if (bi + 1) % log_every == 0 or bi == total_batches - 1:
                loss_history.append({"t": round(ep + (bi + 1) / total_batches, 2), "rec": last_i + last_t})
        pbar.set_postfix(img=f"{last_i:.4f}", txt=f"{last_t:.4f}")
    pbar.close()
    return sae_i, sae_t, loss_history


# ------------------------------------------------------------------ #
# Method 3/4/5: shared SAE + paired auxiliary loss                   #
# ------------------------------------------------------------------ #


def _iso_alignment_penalty(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """Parabrele/IsoEnergy `alignment_penalty(alignment_metric='cosim')` variant.

    Differs from `trace_alignment_loss` (paper Eq. 1) in two ways:
      1. Masks each modality's top-1 latent index in BOTH halves before
         computing cosine similarity (so the "modality-specific" top-1 does
         not dominate the alignment signal).
      2. Uses `F.cosine_similarity(..., dim=1).mean()` directly instead of the
         pre-normalized trace formulation.

    Returns a loss `∈ [-1, 1]` (pre-scaling), same bounds as
    `trace_alignment_loss`. The caller multiplies by `aux_weight`.

    Reference: `Parabrele/IsoEnergy/src/losses.py::alignment_penalty`.
    """
    n = z_img.shape[0]
    top_1_img = torch.topk(z_img, k=1, dim=1).indices.squeeze(1)  # (n,)
    top_1_txt = torch.topk(z_txt, k=1, dim=1).indices.squeeze(1)  # (n,)
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


def _train_single_paired_aux(
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    args: argparse.Namespace,
    latent_size: int,
    seed: int,
    device: torch.device,
    aux_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_weight: float,
    desc: str = "paired_aux",
) -> tuple[TopKSAE, list[dict[str, float]]]:
    _seed_everything(seed)
    model = _make_sae(args, latent_size, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loader = _paired_loader(
        train_img, train_txt, args.batch_size, device,
        shuffle=True, drop_last=True,
    )
    loss_history: list[dict] = []
    total_batches = len(loader)
    log_every = _log_interval(total_batches)
    pbar = tqdm(range(args.num_epochs), desc=desc, leave=False)
    for ep in pbar:
        last_rec = last_aux = 0.0
        for bi, (img_b, txt_b) in enumerate(loader):
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = model(hidden_states=hs_i, return_dense_latents=True)
            out_t = model(hidden_states=hs_t, return_dense_latents=True)
            z_i = _dense_latents(out_i)
            z_t = _dense_latents(out_t)
            aux = aux_fn(z_i, z_t)
            loss = out_i.recon_loss + out_t.recon_loss + aux_weight * aux
            _sae_step(model, optimizer, loss, args.max_grad_norm)
            last_rec = float((out_i.recon_loss + out_t.recon_loss).detach().item())
            last_aux = float(aux.detach().item())
            if (bi + 1) % log_every == 0 or bi == total_batches - 1:
                loss_history.append({"t": round(ep + (bi + 1) / total_batches, 2), "rec": last_rec, "aux": last_aux, "total": last_rec + aux_weight * last_aux})
        pbar.set_postfix(rec=f"{last_rec:.4f}", aux=f"{last_aux:.4f}")
    pbar.close()
    return model, loss_history


# ------------------------------------------------------------------ #
# Method 5: ours (Algorithm 1)                                       #
# ------------------------------------------------------------------ #


def _compute_latent_correlation(
    sae_i: TopKSAE,
    sae_t: TopKSAE,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Signed Pearson correlation between paired dense latents, computed
    over the full training set in float64. Returns (n, n) numpy array,
    where n is the per-SAE latent size (both SAEs must have the same
    latent_size)."""
    assert sae_i.latent_size == sae_t.latent_size, (
        f"latent sizes differ: {sae_i.latent_size} vs {sae_t.latent_size}"
    )
    sae_i.eval()
    sae_t.eval()
    loader = _paired_loader(
        train_img, train_txt, batch_size, device, shuffle=False, drop_last=False,
    )

    n = int(sae_i.latent_size)
    sum_i = np.zeros(n, dtype=np.float64)
    sum_t = np.zeros(n, dtype=np.float64)
    sum_ii = np.zeros(n, dtype=np.float64)
    sum_tt = np.zeros(n, dtype=np.float64)
    sum_it = np.zeros((n, n), dtype=np.float64)
    N = 0

    with torch.no_grad():
        for img_b, txt_b in loader:
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i, return_dense_latents=True)
            out_t = sae_t(hidden_states=hs_t, return_dense_latents=True)
            zi = _dense_latents(out_i).to(torch.float64).cpu().numpy()  # (B, n)
            zt = _dense_latents(out_t).to(torch.float64).cpu().numpy()  # (B, n)
            B = zi.shape[0]
            sum_i += zi.sum(axis=0)
            sum_t += zt.sum(axis=0)
            sum_ii += (zi * zi).sum(axis=0)
            sum_tt += (zt * zt).sum(axis=0)
            sum_it += zi.T @ zt
            N += B

    if N == 0:
        return np.zeros((n, n), dtype=np.float64)

    mean_i = sum_i / N
    mean_t = sum_t / N
    var_i = (sum_ii / N) - mean_i * mean_i
    var_t = (sum_tt / N) - mean_t * mean_t
    cov = (sum_it / N) - np.outer(mean_i, mean_t)
    denom = np.sqrt(np.clip(var_i[:, None] * var_t[None, :], 1e-16, None))
    corr = cov / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def _greedy_permutation_match_full(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper Algorithm 1 (lines 4-16) without the rho early stop.

    Repeatedly pulls the largest remaining signed entry to the top-left, producing
    an ordered pairing of rows and columns over all n steps.

    Returns:
        P_I_idx: permutation to apply to rows (latent indices of sae_i)
        P_T_idx: permutation to apply to cols (latent indices of sae_t)
        ordered_diag: the paired correlation values in the order they were pulled
                      (length n)

    After applying the permutations, the first `m_S` positions correspond to the
    top-`m_S` shared correlations, by construction.
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


def _apply_latent_permutation(
    sae: TopKSAE,
    perm_idx: np.ndarray,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """In-place row-permutation of encoder weight/bias and decoder weight.

    `perm_idx[k] = j` means "new latent k is old latent j", so the new
    parameter row k is the old row j.

    When `optimizer` is supplied, the Adam(W) moment buffers
    (`exp_avg`, `exp_avg_sq`) for the three permuted parameters are
    reordered identically so they remain aligned with the new row order.
    This is essential for preserving optimizer state across the
    Stage 1 → Stage 2 transition in `_train_ours`.
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


def _auxiliary_alignment_loss(
    z_i: torch.Tensor, z_t: torch.Tensor, m_S: int, norm: str = "group",
) -> torch.Tensor:
    """Paper-exact diag-only auxiliary alignment loss.

    For the first `m_S` latent dims, push batch-level diagonal correlation to 1;
    for the remaining dims, push it to 0.

    v6 supports two normalization variants (see plan `v6 — iso_align baseline +
    L_aux normalization`):

    - `norm="group"` (Option A, default): each term is averaged over its own
      group size, so the loss balances `shared vs private` 1:1 independent of
      `m_S`:
      `(1/m_S) Σ_{i≤m_S} (C_ii − 1)² + (1/(n−m_S)) Σ_{i>m_S} C_ii²`.
    - `norm="global"` (Option B): both numerators are divided by the total
      latent count `n`, giving a cleaner "sample mean over a uniformly chosen
      dim" estimator. The loss magnitude scales with `m_S / n`:
      `(1/n) [Σ_{i≤m_S} (C_ii − 1)² + Σ_{i>m_S} C_ii²]`.
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
        raise ValueError(
            f"Unknown aux norm '{norm}'; expected 'group' or 'global'."
        )
    return shared_term + private_term


def _train_ours(
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    args: argparse.Namespace,
    latent_size: int,
    seed: int,
    device: torch.device,
    k_align: int,
    lambda_aux: float,
    m_S_supplied: int,
    aux_norm: str = "group",
) -> tuple[TopKSAE, TopKSAE, dict[str, Any]]:
    """Algorithm 1 (v3): Stage 1 (recon-only) for `k_align` epochs, then greedy
    latent permutation, then Stage 2 (recon + `lambda_aux * L_aux`) for
    `num_epochs - k_align` epochs.

    Single joint AdamW optimizer spans both stages so momentum / second-moment
    state carries across the permutation. Permutation also reorders Adam moment
    buffers (`_apply_latent_permutation(..., optimizer)`).
    """

    num_epochs = args.num_epochs
    stage1_epochs = max(0, min(k_align, num_epochs))
    stage2_epochs = num_epochs - stage1_epochs

    sub_latent = latent_size // 2
    _seed_everything(seed)
    sae_i = _make_sae(args, sub_latent, device)
    _seed_everything(seed + 10000)
    sae_t = _make_sae(args, sub_latent, device)
    sae_i.train()
    sae_t.train()

    optimizer = torch.optim.AdamW(
        list(sae_i.parameters()) + list(sae_t.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    loader = _paired_loader(
        train_img, train_txt, args.batch_size, device,
        shuffle=True, drop_last=True,
    )

    loss_history: list[dict] = []
    total_batches = len(loader)
    log_every = _log_interval(total_batches)

    # -- Stage 1: joint training, recon only --
    if stage1_epochs > 0:
        pbar1 = tqdm(
            range(stage1_epochs), desc="ours_stage1(recon)", leave=False,
        )
        for ep in pbar1:
            last_rec = 0.0
            for bi, (img_b, txt_b) in enumerate(loader):
                hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
                hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
                out_i = sae_i(hidden_states=hs_i)
                out_t = sae_t(hidden_states=hs_t)
                loss = out_i.recon_loss + out_t.recon_loss
                _joint_step([sae_i, sae_t], optimizer, loss, args.max_grad_norm)
                last_rec = float(loss.detach().item())
                if (bi + 1) % log_every == 0 or bi == total_batches - 1:
                    loss_history.append({"t": round(ep + (bi + 1) / total_batches, 2), "stage": "s1", "rec": last_rec})
            pbar1.set_postfix(rec=f"{last_rec:.4f}")
        pbar1.close()

    # -- Permutation (also reorders optimizer moment buffers) --
    C = _compute_latent_correlation(
        sae_i, sae_t, train_img, train_txt, args.batch_size, device,
    )
    P_I, P_T, ordered = _greedy_permutation_match_full(C)
    _apply_latent_permutation(sae_i, P_I, optimizer=optimizer)
    _apply_latent_permutation(sae_t, P_T, optimizer=optimizer)

    rho = float(args.rho)
    m_S_hat_rho = int((ordered > rho).sum())
    first_drop = float(ordered[m_S_hat_rho]) if m_S_hat_rho < ordered.shape[0] else float("nan")
    top_mS_diag_mean = float(ordered[:m_S_supplied].mean()) if m_S_supplied > 0 else float("nan")

    diagnostics: dict[str, Any] = {
        "m_S_hat_rho": m_S_hat_rho,
        "m_S_supplied": int(m_S_supplied),
        "first_drop_below_rho": first_drop,
        "top_mS_diag_mean": top_mS_diag_mean,
        "ordered_diag_head": [float(x) for x in ordered[:8]],
        "k_align": int(stage1_epochs),
        "stage2_epochs": int(stage2_epochs),
        "lambda_aux": float(lambda_aux),
        "aux_norm": str(aux_norm),
    }

    diagnostics["loss_history"] = loss_history

    if stage2_epochs <= 0:
        return sae_i, sae_t, diagnostics

    # -- Stage 2: joint training, recon + lambda_aux * L_aux --
    pbar2 = tqdm(
        range(stage2_epochs),
        desc=f"ours_stage2(mS={m_S_supplied},lam={lambda_aux},norm={aux_norm})",
        leave=False,
    )
    for ep2 in pbar2:
        last_rec = last_aux = 0.0
        for bi, (img_b, txt_b) in enumerate(loader):
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i, return_dense_latents=True)
            out_t = sae_t(hidden_states=hs_t, return_dense_latents=True)
            z_i = _dense_latents(out_i)
            z_t = _dense_latents(out_t)
            aux = _auxiliary_alignment_loss(z_i, z_t, m_S_supplied, norm=aux_norm)
            loss = out_i.recon_loss + out_t.recon_loss + lambda_aux * aux
            _joint_step([sae_i, sae_t], optimizer, loss, args.max_grad_norm)
            last_rec = float((out_i.recon_loss + out_t.recon_loss).detach().item())
            last_aux = float(aux.detach().item())
            if (bi + 1) % log_every == 0 or bi == total_batches - 1:
                gep = stage1_epochs + ep2 + (bi + 1) / total_batches
                loss_history.append({"t": round(gep, 2), "stage": "s2", "rec": last_rec, "aux": last_aux, "total": last_rec + lambda_aux * last_aux})
        pbar2.set_postfix(rec=f"{last_rec:.4f}", aux=f"{last_aux:.4f}")
    pbar2.close()

    return sae_i, sae_t, diagnostics


# ------------------------------------------------------------------ #
# Metrics for a method                                               #
# ------------------------------------------------------------------ #


def _compute_mcc_and_uniqueness(
    learned_vectors: np.ndarray,
    gt_matrix: np.ndarray,
) -> dict[str, float]:
    """Mean Correlation Coefficient (MCC) + Feature Uniqueness.

    Reference: Chanin & Garriga-Alonso 2026, "SynthSAEBench" (arxiv 2602.14687), Equations (17)-(18).

    Given `L` SAE decoder rows `w_j` and `N` GT atoms `d_i`, compute the
    absolute cosine similarity matrix `|S_ij| = |w_j^T d_i|` and:

    - **MCC**: optimal one-to-one matching via the Hungarian algorithm.
      `MCC = (1/min(L, N)) * Σ_{(i,j) ∈ matching} |w_j^T d_i|`. Penalises
      SAE collapses onto few atoms (stricter than `MGT`/`MIP` which allow
      many-to-one mapping).
    - **Uniqueness**: for each latent `j`, find its best-matching GT
      `i*(j) = argmax_i |w_j^T d_i|`, then count distinct targets.
      `Uniqueness_raw = |{i*(j)}| / L`. Paper formula.
      `Uniqueness_norm = |{i*(j)}| / min(L, N)` is a cross-L-comparable
      variant bounded in [0, 1].

    Args:
        learned_vectors: `(L, d)` SAE decoder rows.
        gt_matrix: `(d, N)` GT atoms as columns.
    """
    if gt_matrix.shape[1] == 0 or learned_vectors.shape[0] == 0:
        return {
            "mcc": float("nan"),
            "uniqueness_raw": float("nan"),
            "uniqueness_norm": float("nan"),
        }

    L = int(learned_vectors.shape[0])
    N = int(gt_matrix.shape[1])

    learned_norm = _normalize_rows(learned_vectors.astype(np.float64))  # (L, d)
    gt_norm = _normalize_rows(gt_matrix.T.astype(np.float64))  # (N, d)
    sim = np.abs(learned_norm @ gt_norm.T)  # (L, N), abs cosine

    # --- MCC via Hungarian algorithm ---
    # linear_sum_assignment minimises cost; maximise similarity => -sim.
    row_ind, col_ind = linear_sum_assignment(-sim)  # min(L, N) pairs
    matched = sim[row_ind, col_ind]
    mcc = float(matched.sum() / max(min(L, N), 1))

    # --- Feature Uniqueness ---
    best_gt_per_row = sim.argmax(axis=1)  # (L,)
    num_distinct = int(np.unique(best_gt_per_row).size)
    uniqueness_raw = float(num_distinct / max(L, 1))
    uniqueness_norm = float(num_distinct / max(min(L, N), 1))

    return {
        "mcc": mcc,
        "uniqueness_raw": uniqueness_raw,
        "uniqueness_norm": uniqueness_norm,
    }


def _compute_merged_fraction(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
) -> float:
    """Fraction of shared GT atoms i whose best-matching image slot equals
    the best-matching text slot: argmax_j cos(V_j, phi_i) == argmax_j cos(W_j, psi_i).

    Uses signed cosine (not |cos|) because the SAE encoder is a top-k ReLU:
    a column anti-aligned with an atom (cos < 0) cannot fire on that atom
    and therefore cannot "represent" it regardless of magnitude.

    1.0 = every shared pair is merged into one shared column (single-decoder
    bisector collapse), 0.0 = the two sides use completely different slots.
    """
    if phi_S.size == 0 or psi_S.size == 0:
        return float("nan")
    v = _normalize_rows(w_dec_img.astype(np.float64))
    w = _normalize_rows(w_dec_txt.astype(np.float64))
    phi = _normalize_rows(phi_S.T.astype(np.float64))
    psi = _normalize_rows(psi_S.T.astype(np.float64))
    sim_i = v @ phi.T  # (L, n_S), signed cosine
    sim_t = w @ psi.T  # (L, n_S), signed cosine
    top_i = sim_i.argmax(axis=0)
    top_t = sim_t.argmax(axis=0)
    return float((top_i == top_t).mean())


def _compute_recovery_metrics_multi_tau(
    learned_vectors: np.ndarray,
    gt_matrix: np.ndarray,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Multi-threshold MGT + MIP in a single pass.

    `learned_vectors`: (k, d) — SAE decoder rows.
    `gt_matrix`: (d, n_gt) — ground-truth atoms as columns.
    Returns `{"mip": ..., "mgt_tau0.8": ..., "mgt_tau0.95": ..., ...}`.
    """
    if gt_matrix.shape[1] == 0:
        out: dict[str, float] = {"mip": float("nan")}
        for tau in taus:
            out[f"mgt_tau{tau}"] = float("nan")
        return out
    learned_norm = _normalize_rows(learned_vectors.astype(np.float64))
    gt_norm = _normalize_rows(gt_matrix.T.astype(np.float64))
    sim = np.abs(learned_norm @ gt_norm.T)
    best = sim.max(axis=0)
    out = {"mip": float(best.mean())}
    for tau in taus:
        out[f"mgt_tau{tau}"] = float((best > tau).mean())
    return out


def _compute_joint_mgt(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Joint cross-modal recovery metric.

    For each GT shared atom g, computes
        joint(g) = max_i ½·(|cos(W_img[i], phi_g)| + |cos(W_txt[i], psi_g)|)
    and reports the fraction with joint(g) > τ for each τ, plus the mean.

    `w_dec_img`: (L, m_img) decoder rows of img-side SAE
    `w_dec_txt`: (L, m_txt) decoder rows of txt-side SAE
    `phi_S`:     (m_img, n_S) shared GT atoms in img embedding
    `psi_S`:     (m_txt, n_S) shared GT atoms in txt embedding

    Single-SAE methods should pass the same array for both (img and txt halves
    are intrinsically tied). The metric will then ask whether the SAME feature
    is good for both modalities, which is a strictly harder requirement than
    per-modality MGT.
    """
    if (
        phi_S.shape[1] == 0 or psi_S.shape[1] == 0
        or w_dec_img.shape[0] == 0 or w_dec_txt.shape[0] == 0
    ):
        out: dict[str, float] = {"joint_match_mean": float("nan")}
        for tau in taus:
            out[f"joint_mgt_tau{tau}"] = float("nan")
        return out
    Wi = _normalize_rows(w_dec_img.astype(np.float64))
    Wt = _normalize_rows(w_dec_txt.astype(np.float64))
    Pi = _normalize_rows(phi_S.T.astype(np.float64))
    Pt = _normalize_rows(psi_S.T.astype(np.float64))
    img_cos = np.abs(Wi @ Pi.T)  # (L, n_S)
    txt_cos = np.abs(Wt @ Pt.T)  # (L, n_S)
    joint = 0.5 * (img_cos + txt_cos)
    best = joint.max(axis=0)
    out = {"joint_match_mean": float(best.mean())}
    for tau in taus:
        out[f"joint_mgt_tau{tau}"] = float((best > tau).mean())
    return out


def _probe_gt_pair_activation(
    sae_i: TopKSAE,
    sae_t: TopKSAE,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    device: torch.device,
    *,
    txt_permutation: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Encoder-side GT probe metric (XMA — Cross-Modal Alignment).

    For each shared GT atom g, construct a synthetic paired sample
    (x_img = phi_g, x_txt = psi_g) — i.e., an idealized "only atom g is active"
    sample — and encode through both SAEs. Then measure whether the two SAEs
    agree on which latent slot fires:

    - probe_top1_agree: fraction of GT atoms where argmax z_img == argmax z_txt
    - probe_vec_cos: mean cosine similarity of the dense latent vectors
    - probe_topk_jaccard: mean Jaccard overlap of the top-k fired indices
    - probe_top1_both_valid: fraction where both top-1s are non-zero

    If `txt_permutation` is given (length m, with entry i holding the original
    txt-SAE index that should occupy position i after permutation), the txt
    latent vector is re-indexed accordingly before the comparison — this is
    the "post-hoc Hungarian" variant of the probe.
    """
    n_S = phi_S.shape[1]
    if n_S == 0 or psi_S.shape[1] == 0:
        return {
            "probe_top1_agree": float("nan"),
            "probe_vec_cos": float("nan"),
            "probe_topk_jaccard": float("nan"),
            "probe_top1_both_valid": float("nan"),
        }
    x_img = torch.from_numpy(phi_S.T.astype(np.float32)).unsqueeze(1).to(device)
    x_txt = torch.from_numpy(psi_S.T.astype(np.float32)).unsqueeze(1).to(device)
    sae_i.eval()
    sae_t.eval()
    with torch.no_grad():
        out_i = sae_i(hidden_states=x_img, return_dense_latents=True)
        out_t = sae_t(hidden_states=x_txt, return_dense_latents=True)
        z_i = _dense_latents(out_i)  # (n_S, L_i)
        z_t = _dense_latents(out_t)  # (n_S, L_t)
    if txt_permutation is not None:
        perm = torch.as_tensor(txt_permutation, dtype=torch.long, device=z_t.device)
        z_t = z_t.index_select(dim=1, index=perm)
    # top-1 index agreement
    top1_i = z_i.argmax(dim=1)
    top1_t = z_t.argmax(dim=1)
    agree = (top1_i == top1_t).float()
    val_i = z_i.gather(1, top1_i.unsqueeze(1)).squeeze(1)
    val_t = z_t.gather(1, top1_t.unsqueeze(1)).squeeze(1)
    both_valid = ((val_i > 0) & (val_t > 0)).float()
    valid_agree = (agree * both_valid).mean().item()
    both_valid_rate = both_valid.mean().item()

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
    topk_jaccard = float(np.mean(jaccards))

    return {
        "probe_top1_agree": float(valid_agree),
        "probe_vec_cos": float(vec_cos),
        "probe_topk_jaccard": topk_jaccard,
        "probe_top1_both_valid": float(both_valid_rate),
    }


def _compute_posthoc_joint_mgt(
    sae_i: TopKSAE,
    sae_t: TopKSAE,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    same_model: bool,
    batch_size: int,
    device: torch.device,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Joint MGT after Hungarian-optimal post-hoc reindexing of txt features.

    Computes the L×L cross-modal correlation matrix `C` on the training set,
    finds the optimal 1-to-1 column permutation π via Hungarian on `-|C|`,
    reorders the txt-decoder rows so that row i = original row π(i), and
    recomputes joint MGT. This is the "post-hoc baseline" for assessing whether
    training-time alignment (ours) gives anything beyond simple reindexing.

    For single-SAE methods (`same_model=True`), no permutation is meaningful
    (img and txt halves of a single decoder are tied to the same row), so we
    simply return the raw joint MGT.
    """
    w_img = sae_i.W_dec.detach().cpu().numpy()
    w_txt = sae_t.W_dec.detach().cpu().numpy()
    if same_model:
        return _compute_joint_mgt(w_img, w_txt, phi_S, psi_S, taus)
    C = _compute_latent_correlation(
        sae_i, sae_t, train_img, train_txt, batch_size, device,
    )
    _, col_ind = linear_sum_assignment(-np.abs(C))
    w_txt_aligned = w_txt[col_ind]
    return _compute_joint_mgt(w_img, w_txt_aligned, phi_S, psi_S, taus)


def _eval_pair_latent_cosine(
    sae_i: TopKSAE,
    sae_t: TopKSAE,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Per-paired-sample latent cosine similarity on the eval set.

    For each eval pair (x_img[i], x_txt[i]), forwards through the respective
    SAE (same model for single-SAE methods), obtains the dense top-k latents
    `z_i`, `z_t`, and computes `cos(z_i[i], z_t[i])`. Returns the mean over
    all eval pairs.

    This is complementary to `cross_cos_top_mS_mean` which is a per-latent-
    dimension diagonal correlation across samples. The new metric measures
    per-sample alignment — "for a given paired sample, how close are its two
    modality codes?".
    """
    sae_i.eval()
    sae_t.eval()
    loader = _paired_loader(
        eval_img, eval_txt, batch_size, device,
        shuffle=False, drop_last=False,
    )
    total = 0
    sum_cos = 0.0
    with torch.no_grad():
        for img_b, txt_b in loader:
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i, return_dense_latents=True)
            out_t = sae_t(hidden_states=hs_t, return_dense_latents=True)
            zi = _dense_latents(out_i)  # (B, latent)
            zt = _dense_latents(out_t)
            # Per-sample cosine, bounded in [-1, 1]. eps handles all-zero
            # code vectors (shouldn't happen with top-k > 0 but defensive).
            cos = F.cosine_similarity(zi, zt, dim=1, eps=1e-12)
            sum_cos += float(cos.sum().item())
            total += cos.shape[0]
    return sum_cos / max(total, 1)


def _cross_corr_mean_parts(
    sae_i: TopKSAE,
    sae_t: TopKSAE,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
    m_S: int,
) -> tuple[float, float]:
    """Signed Pearson correlation on the *eval* set, then split into
    first `m_S` diag entries (cross_cos_top_mS_mean) and rest
    (cross_cos_rest_mean). `m_S` is clipped to the actual latent count.
    """
    C = _compute_latent_correlation(
        sae_i, sae_t, eval_img, eval_txt, batch_size, device,
    )
    diag = np.diag(C)
    n = diag.shape[0]
    m_S_eff = max(0, min(int(m_S), n))
    top = float(diag[:m_S_eff].mean()) if m_S_eff > 0 else float("nan")
    rest = float(diag[m_S_eff:].mean()) if m_S_eff < n else float("nan")
    return top, rest


def _evaluate_method(
    *,
    method: str,
    sae_i: TopKSAE,
    sae_t: TopKSAE,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    phi_I: Optional[np.ndarray],
    psi_T: Optional[np.ndarray],
    n_shared: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Compute a flat metrics dict for a trained (sae_i, sae_t) pair.

    For shared-decoder methods (1, 3, 4), pass the same model as sae_i and sae_t.
    The latent size is inferred from `sae_i.latent_size`.
    """
    same_model = sae_i is sae_t
    img_eval = _eval_loss_single_modality(sae_i, eval_img, args.batch_size, device)
    txt_eval = _eval_loss_single_modality(sae_t, eval_txt, args.batch_size, device)
    avg_eval = (img_eval + txt_eval) / 2

    w_dec_img = sae_i.W_dec.detach().cpu().numpy()
    w_dec_txt = sae_t.W_dec.detach().cpu().numpy()

    # v6 multi-threshold MGT: τ ∈ {0.9, 0.95, 0.99}. τ=0.8 was dropped
    # because Theorem 1 already covers loose angular thresholds; we focus
    # on strict thresholds here.
    img_mgt_multi = _compute_recovery_metrics_multi_tau(
        w_dec_img, phi_S, (0.9, 0.95, 0.99),
    )
    txt_mgt_multi = _compute_recovery_metrics_multi_tau(
        w_dec_txt, psi_S, (0.9, 0.95, 0.99),
    )

    # v7: full per-modality GT = private + shared (drops the zero-padding
    # cross-modality columns from `phi_full`/`psi_full`). Lets us tell apart
    # methods that recover only shared atoms (high mgt_shared) from methods
    # that recover both shared and private (high mgt_full).
    if phi_I is not None and phi_I.size > 0:
        phi_full = np.concatenate([phi_I, phi_S], axis=1)
    else:
        phi_full = phi_S
    if psi_T is not None and psi_T.size > 0:
        psi_full = np.concatenate([psi_S, psi_T], axis=1)
    else:
        psi_full = psi_S

    img_mgt_full = _compute_recovery_metrics_multi_tau(
        w_dec_img, phi_full, (0.9, 0.95, 0.99),
    )
    txt_mgt_full = _compute_recovery_metrics_multi_tau(
        w_dec_txt, psi_full, (0.9, 0.95, 0.99),
    )

    # MCC / Uniqueness against shared GT (v6 addition).
    # Separately per modality side since two-SAE methods have distinct decoders.
    img_mcc_unique = _compute_mcc_and_uniqueness(w_dec_img, phi_S)
    txt_mcc_unique = _compute_mcc_and_uniqueness(w_dec_txt, psi_S)

    # v7: MCC / Uniqueness against full per-modality GT.
    img_mcc_full = _compute_mcc_and_uniqueness(w_dec_img, phi_full)
    txt_mcc_full = _compute_mcc_and_uniqueness(w_dec_txt, psi_full)

    cross = _gt_based_shared_alignment_mismatched(
        w_dec_img, w_dec_txt, phi_S, psi_S,
    )

    m_S_for_diag = n_shared  # oracle for baselines
    if method == "ours":
        m_S_for_diag = int(args._current_m_S)
    top_diag, rest_diag = _cross_corr_mean_parts(
        sae_i, sae_t, eval_img, eval_txt,
        args.batch_size, device, m_S_for_diag,
    )

    # Per-pair eval latent cosine similarity (v6 new metric).
    pair_cos = _eval_pair_latent_cosine(
        sae_i, sae_t, eval_img, eval_txt, args.batch_size, device,
    )

    # v8: joint cross-modal MGT — for each GT shared atom g, the best feature
    # i must approximate phi_g (via img decoder row) AND psi_g (via txt decoder
    # row) simultaneously. Reported in two flavors:
    #   joint_mgt_raw_*: at the original feature indices
    #   joint_mgt_posthoc_*: after Hungarian-optimal txt-column reindexing
    # For single-SAE methods the two are identical by construction.
    _joint_taus = (0.9, 0.95, 0.99)
    joint_raw = _compute_joint_mgt(w_dec_img, w_dec_txt, phi_S, psi_S, _joint_taus)
    joint_posthoc = _compute_posthoc_joint_mgt(
        sae_i, sae_t, train_img, train_txt, phi_S, psi_S,
        same_model=same_model, batch_size=args.batch_size,
        device=device, taus=_joint_taus,
    )

    # v8: GT-probe encoder test — synthetic pure-atom samples, measure whether
    # both SAEs fire at the same latent slot. This directly tests functional
    # alignment (as opposed to decoder structural alignment from joint_mgt).
    merged_frac = _compute_merged_fraction(w_dec_img, w_dec_txt, phi_S, psi_S)

    probe = _probe_gt_pair_activation(sae_i, sae_t, phi_S, psi_S, device)
    # Post-hoc Hungarian re-indexed version: apply the optimal txt-SAE
    # permutation (same Hungarian assignment used for joint_mgt_posthoc) and
    # recompute the probe. For single-SAE methods the permutation is identity.
    if same_model:
        probe_ph = probe
    else:
        C_ph = _compute_latent_correlation(
            sae_i, sae_t, train_img, train_txt, args.batch_size, device,
        )
        _, col_ind_ph = linear_sum_assignment(-np.abs(C_ph))
        probe_ph = _probe_gt_pair_activation(
            sae_i, sae_t, phi_S, psi_S, device,
            txt_permutation=col_ind_ph,
        )

    return {
        "img_eval_loss": img_eval,
        "txt_eval_loss": txt_eval,
        "avg_eval_loss": avg_eval,
        "img_mip_shared": img_mgt_multi["mip"],
        "txt_mip_shared": txt_mgt_multi["mip"],
        # Multi-tau MGT (v6: drop 0.8, add 0.9).
        "img_mgt_shared_tau0.9": img_mgt_multi["mgt_tau0.9"],
        "img_mgt_shared_tau0.95": img_mgt_multi["mgt_tau0.95"],
        "img_mgt_shared_tau0.99": img_mgt_multi["mgt_tau0.99"],
        "txt_mgt_shared_tau0.9": txt_mgt_multi["mgt_tau0.9"],
        "txt_mgt_shared_tau0.95": txt_mgt_multi["mgt_tau0.95"],
        "txt_mgt_shared_tau0.99": txt_mgt_multi["mgt_tau0.99"],
        # v7: full per-modality GT MGT/MIP (private + shared).
        "img_mip_full": img_mgt_full["mip"],
        "txt_mip_full": txt_mgt_full["mip"],
        "img_mgt_full_tau0.9": img_mgt_full["mgt_tau0.9"],
        "img_mgt_full_tau0.95": img_mgt_full["mgt_tau0.95"],
        "img_mgt_full_tau0.99": img_mgt_full["mgt_tau0.99"],
        "txt_mgt_full_tau0.9": txt_mgt_full["mgt_tau0.9"],
        "txt_mgt_full_tau0.95": txt_mgt_full["mgt_tau0.95"],
        "txt_mgt_full_tau0.99": txt_mgt_full["mgt_tau0.99"],
        # MCC + Feature Uniqueness against shared GT (v6 addition).
        "img_mcc_shared": img_mcc_unique["mcc"],
        "txt_mcc_shared": txt_mcc_unique["mcc"],
        "img_uniqueness_shared_raw": img_mcc_unique["uniqueness_raw"],
        "txt_uniqueness_shared_raw": txt_mcc_unique["uniqueness_raw"],
        "img_uniqueness_shared_norm": img_mcc_unique["uniqueness_norm"],
        "txt_uniqueness_shared_norm": txt_mcc_unique["uniqueness_norm"],
        # v7: MCC + Uniqueness against full per-modality GT.
        "img_mcc_full": img_mcc_full["mcc"],
        "txt_mcc_full": txt_mcc_full["mcc"],
        "img_uniqueness_full_raw": img_mcc_full["uniqueness_raw"],
        "txt_uniqueness_full_raw": txt_mcc_full["uniqueness_raw"],
        "img_uniqueness_full_norm": img_mcc_full["uniqueness_norm"],
        "txt_uniqueness_full_norm": txt_mcc_full["uniqueness_norm"],
        "merged_fraction": merged_frac,
        "pair_cos_mean": pair_cos,
        "cross_cos_top_mS_mean": top_diag,
        "cross_cos_rest_mean": rest_diag,
        "cross_cos_gt_mean": cross["cross_cos_mean"],
        "alpha_proxy_mean": cross["alpha_proxy_mean"],
        # v8: joint cross-modal MGT (raw + post-hoc Hungarian re-index).
        "joint_match_mean_raw": joint_raw["joint_match_mean"],
        "joint_mgt_raw_tau0.9": joint_raw["joint_mgt_tau0.9"],
        "joint_mgt_raw_tau0.95": joint_raw["joint_mgt_tau0.95"],
        "joint_mgt_raw_tau0.99": joint_raw["joint_mgt_tau0.99"],
        "joint_match_mean_posthoc": joint_posthoc["joint_match_mean"],
        "joint_mgt_posthoc_tau0.9": joint_posthoc["joint_mgt_tau0.9"],
        "joint_mgt_posthoc_tau0.95": joint_posthoc["joint_mgt_tau0.95"],
        "joint_mgt_posthoc_tau0.99": joint_posthoc["joint_mgt_tau0.99"],
        # v8: encoder-side GT probe metrics (raw + post-hoc Hungarian)
        "probe_top1_agree": probe["probe_top1_agree"],
        "probe_vec_cos": probe["probe_vec_cos"],
        "probe_topk_jaccard": probe["probe_topk_jaccard"],
        "probe_top1_both_valid": probe["probe_top1_both_valid"],
        "probe_top1_agree_posthoc": probe_ph["probe_top1_agree"],
        "probe_vec_cos_posthoc": probe_ph["probe_vec_cos"],
        "probe_topk_jaccard_posthoc": probe_ph["probe_topk_jaccard"],
        "probe_top1_both_valid_posthoc": probe_ph["probe_top1_both_valid"],
        "same_model": bool(same_model),
    }


# ------------------------------------------------------------------ #
# Run one (alpha, latent_size, seed) config over all methods         #
# ------------------------------------------------------------------ #


@dataclass
class OursConfig:
    lambda_aux: float
    m_S: int
    k_align: int
    aux_norm: str = "group"  # "group" (Option A) or "global" (Option B)

    @property
    def tag(self) -> str:
        return (
            f"lam{self.lambda_aux}_mS{self.m_S}_k{self.k_align}"
            f"_norm{self.aux_norm}"
        )


def _build_dataset(
    args: argparse.Namespace, alpha: float, seed: int,
) -> tuple[SyntheticTheoryFeatureBuilder, dict[str, Any]]:
    shared_mode = "identical" if alpha >= 0.99 else "range"
    builder = SyntheticTheoryFeatureBuilder(
        n_image=args.n_image,
        n_shared=args.n_shared,
        n_text=args.n_text,
        representation_dim=args.representation_dim,
        sparsity=args.sparsity,
        min_active=args.min_active,
        cmin=args.cmin,
        beta=args.beta,
        max_interference=args.max_interference,
        coeff_dist=getattr(args, "coeff_dist", "exponential"),
        coeff_mu=getattr(args, "coeff_mu", 4.5),
        coeff_sigma=getattr(args, "coeff_sigma", 0.5),
        obs_noise_std=getattr(args, "obs_noise_std", 0.0),
        shared_coeff_mode=getattr(args, "shared_coeff_mode", "identical"),
        strategy=args.dictionary_strategy,
        shared_mode=shared_mode,
        alpha_lo=max(alpha - 0.03, 0.0) if shared_mode == "range" else 0.7,
        alpha_hi=min(alpha + 0.03, 1.0) if shared_mode == "range" else 0.9,
        calibration_lr=args.calibration_lr,
        calibration_max_iters=args.calibration_max_iters,
        calibration_tol=args.calibration_tol,
        seed=seed,
        num_train=args.num_train,
        num_eval=args.num_eval,
        num_test=args.num_eval,
        verbose=args.verbose,
    )
    ds = builder.build_numpy_dataset()
    return builder, ds


def _run_single(
    args: argparse.Namespace,
    alpha: float,
    latent_size: int,
    seed: int,
    methods: list[str],
    ours_configs: list[OursConfig],
) -> dict[str, Any]:
    """Returns one flat dict where method_id is:
        - "single_recon", "two_recon", "group_sparse", "trace_align"
        - "ours::<tag>" for each OursConfig
    """
    device = _resolve_device(args.device)
    builder, ds = _build_dataset(args, alpha, seed)
    train_img = torch.from_numpy(ds["train"]["image_representation"])
    train_txt = torch.from_numpy(ds["train"]["text_representation"])
    eval_img = torch.from_numpy(ds["eval"]["image_representation"])
    eval_txt = torch.from_numpy(ds["eval"]["text_representation"])

    result: dict[str, Any] = {
        "seed": seed,
        "alpha_target": alpha,
        "latent_size": latent_size,
        "alpha_actual_mean": float(builder.mean_shared_cosine_similarity),
        "alpha_actual_std": float(builder.std_shared_cosine_similarity),
    }
    args._current_m_S = args.n_shared  # default for baselines

    def _finish(method_id: str, sae_i: TopKSAE, sae_t: TopKSAE, extras: Optional[dict] = None) -> None:
        metrics = _evaluate_method(
            method=method_id.split("::")[0],
            sae_i=sae_i, sae_t=sae_t,
            train_img=train_img, train_txt=train_txt,
            eval_img=eval_img, eval_txt=eval_txt,
            phi_S=builder.phi_S, psi_S=builder.psi_S,
            phi_I=builder.phi_I, psi_T=builder.psi_T,
            n_shared=args.n_shared,
            args=args, device=device,
        )
        metrics_any: dict[str, Any] = {}
        for mk, mv in metrics.items():
            metrics_any[mk] = mv
        if extras:
            for k, v in extras.items():
                if isinstance(v, list):
                    metrics_any[f"diag_{k}"] = ",".join(f"{x:.4f}" for x in v)
                else:
                    metrics_any[f"diag_{k}"] = v
        result[method_id] = metrics_any
        if sae_i is not None and sae_i is not sae_t:
            del sae_i
        if sae_t is not None:
            del sae_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    loss_histories: dict[str, Any] = {}

    if "single_recon" in methods:
        model, lh = _train_single_recon(train_img, train_txt, args, latent_size, seed, device)
        _finish("single_recon", model, model)
        loss_histories["single_recon"] = lh

    if "two_recon" in methods:
        sae_i, sae_t, lh = _train_two_recon(train_img, train_txt, args, latent_size, seed, device)
        _finish("two_recon", sae_i, sae_t)
        loss_histories["two_recon"] = lh

    if "group_sparse" in methods:
        model, lh = _train_single_paired_aux(
            train_img, train_txt, args, latent_size, seed, device,
            aux_fn=group_sparse_loss, aux_weight=args.group_sparse_lambda,
            desc="group_sparse",
        )
        _finish("group_sparse", model, model)
        loss_histories["group_sparse"] = lh

    if "trace_align" in methods:
        model, lh = _train_single_paired_aux(
            train_img, train_txt, args, latent_size, seed, device,
            aux_fn=trace_alignment_loss, aux_weight=args.trace_beta,
            desc="trace_align",
        )
        _finish("trace_align", model, model)
        loss_histories["trace_align"] = lh

    if "iso_align" in methods:
        model, lh = _train_single_paired_aux(
            train_img, train_txt, args, latent_size, seed, device,
            aux_fn=_iso_alignment_penalty, aux_weight=args.iso_align_beta,
            desc="iso_align",
        )
        _finish("iso_align", model, model)
        loss_histories["iso_align"] = lh

    if "single_paired_align" in methods:
        m_S_sp = (
            args.single_paired_align_mS
            if args.single_paired_align_mS > 0
            else args.n_shared
        )
        norm_sp = args.single_paired_align_norm

        def _single_paired_aux_fn(z_i: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
            return _auxiliary_alignment_loss(z_i, z_t, m_S=m_S_sp, norm=norm_sp)

        model, lh = _train_single_paired_aux(
            train_img, train_txt, args, latent_size, seed, device,
            aux_fn=_single_paired_aux_fn,
            aux_weight=args.single_paired_align_lambda,
            desc=f"single_paired_align(lam={args.single_paired_align_lambda},mS={m_S_sp},norm={norm_sp})",
        )
        _finish("single_paired_align", model, model)
        loss_histories["single_paired_align"] = lh

    if "ours" in methods:
        for cfg in ours_configs:
            args._current_m_S = cfg.m_S
            sae_i, sae_t, diag = _train_ours(
                train_img, train_txt, args, latent_size, seed, device,
                k_align=cfg.k_align,
                lambda_aux=cfg.lambda_aux,
                m_S_supplied=cfg.m_S,
                aux_norm=cfg.aux_norm,
            )
            tag = f"ours::{cfg.tag}"
            loss_histories[tag] = diag.pop("loss_history", [])
            _finish(tag, sae_i, sae_t, extras=diag)
        args._current_m_S = args.n_shared

    result["_loss_histories"] = loss_histories
    return result


# ------------------------------------------------------------------ #
# Aggregation / metric keys                                          #
# ------------------------------------------------------------------ #


METRIC_SUFFIXES = [
    "img_eval_loss",
    "txt_eval_loss",
    "avg_eval_loss",
    "img_mip_shared",
    "txt_mip_shared",
    # v6: multi-tau MGT, dropped 0.8, added 0.9
    "img_mgt_shared_tau0.9",
    "img_mgt_shared_tau0.95",
    "img_mgt_shared_tau0.99",
    "txt_mgt_shared_tau0.9",
    "txt_mgt_shared_tau0.95",
    "txt_mgt_shared_tau0.99",
    # v7: full per-modality GT (private + shared) MGT/MIP
    "img_mip_full",
    "txt_mip_full",
    "img_mgt_full_tau0.9",
    "img_mgt_full_tau0.95",
    "img_mgt_full_tau0.99",
    "txt_mgt_full_tau0.9",
    "txt_mgt_full_tau0.95",
    "txt_mgt_full_tau0.99",
    # v6: MCC + Feature Uniqueness against shared GT
    # (Chanin & Garriga-Alonso 2026 SynthSAEBench; Hungarian 1-to-1 matching
    #  + count of distinct best-match GTs per decoder row).
    "img_mcc_shared",
    "txt_mcc_shared",
    "img_uniqueness_shared_raw",
    "txt_uniqueness_shared_raw",
    "img_uniqueness_shared_norm",
    "txt_uniqueness_shared_norm",
    # v7: MCC + Uniqueness against full per-modality GT.
    "img_mcc_full",
    "txt_mcc_full",
    "img_uniqueness_full_raw",
    "txt_uniqueness_full_raw",
    "img_uniqueness_full_norm",
    "txt_uniqueness_full_norm",
    # followup9: merged_fraction — fraction of shared pairs where the best
    # image-matching column equals the best text-matching column (direct
    # signature of Theorem-2 partition merging on partially aligned pairs).
    "merged_fraction",
    # v6: per-pair eval latent cosine similarity metric
    "pair_cos_mean",
    "cross_cos_top_mS_mean",
    "cross_cos_rest_mean",
    "cross_cos_gt_mean",
    "alpha_proxy_mean",
    # v8: joint cross-modal MGT (raw + post-hoc Hungarian re-index)
    "joint_match_mean_raw",
    "joint_mgt_raw_tau0.9",
    "joint_mgt_raw_tau0.95",
    "joint_mgt_raw_tau0.99",
    "joint_match_mean_posthoc",
    "joint_mgt_posthoc_tau0.9",
    "joint_mgt_posthoc_tau0.95",
    "joint_mgt_posthoc_tau0.99",
    # v8: encoder-side GT probe metrics
    "probe_top1_agree",
    "probe_vec_cos",
    "probe_topk_jaccard",
    "probe_top1_both_valid",
    "probe_top1_agree_posthoc",
    "probe_vec_cos_posthoc",
    "probe_topk_jaccard_posthoc",
    "probe_top1_both_valid_posthoc",
]


def _aggregate(seed_results: list[dict[str, Any]], method_ids: list[str]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    for method_id in method_ids:
        for suf in METRIC_SUFFIXES:
            vals: list[float] = []
            for r in seed_results:
                m = r.get(method_id)
                if not m:
                    continue
                v = m.get(suf)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                vals.append(float(v))
            key = f"{method_id}/{suf}"
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals, ddof=0))
            else:
                agg[f"{key}_mean"] = float("nan")
                agg[f"{key}_std"] = float("nan")
        # Ours-specific diagnostics
        if method_id.startswith("ours::"):
            for dkey in ("m_S_hat_rho", "first_drop_below_rho", "top_mS_diag_mean"):
                vals = []
                for r in seed_results:
                    m = r.get(method_id)
                    if not m:
                        continue
                    v = m.get(f"diag_{dkey}")
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    vals.append(float(v))
                key = f"{method_id}/{dkey}"
                if vals:
                    agg[f"{key}_mean"] = float(np.mean(vals))
                    agg[f"{key}_std"] = float(np.std(vals, ddof=0))
    return agg


# ------------------------------------------------------------------ #
# Experiment dispatch                                                #
# ------------------------------------------------------------------ #


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _build_ours_configs(args: argparse.Namespace) -> list[OursConfig]:
    lambdas = _parse_csv_floats(args.lambda_aux_sweep)
    m_ss = _parse_csv_ints(args.m_s_sweep)
    k_aligns = _parse_csv_ints(args.k_align_sweep)
    aux_norms = [s.strip() for s in args.aux_norm_sweep.split(",") if s.strip()]
    for nm in aux_norms:
        if nm not in ("group", "global"):
            raise ValueError(
                f"Invalid --aux-norm-sweep value '{nm}'; "
                f"expected 'group' or 'global'."
            )
    return [
        OursConfig(lambda_aux=l, m_S=m, k_align=k, aux_norm=nm)
        for l, m, k, nm in itertools.product(lambdas, m_ss, k_aligns, aux_norms)
    ]


def _run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    alphas = _parse_csv_floats(args.alpha_sweep)
    latent_sizes = _parse_csv_ints(args.latent_size_sweep)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    ours_configs = _build_ours_configs(args) if "ours" in methods else []

    # Build the list of method ids for downstream aggregation / summary.
    method_ids: list[str] = []
    for m in methods:
        if m == "ours":
            method_ids.extend([f"ours::{c.tag}" for c in ours_configs])
        else:
            method_ids.append(m)

    sweep_results = []
    total_cfgs = len(latent_sizes) * len(alphas) * int(args.num_seeds)
    outer_pbar = tqdm(
        total=total_cfgs, desc="sweep", dynamic_ncols=True, leave=True,
    )
    for latent_size in latent_sizes:
        for alpha in alphas:
            seed_results = []
            for i in range(args.num_seeds):
                seed = args.seed_base + i
                outer_pbar.set_postfix(
                    alpha=f"{alpha:.2f}", L=latent_size, seed=seed,
                )
                logger.info(
                    "alpha=%.2f latent_size=%d seed=%d methods=%s ours_cfgs=%d",
                    alpha, latent_size, seed, ",".join(methods), len(ours_configs),
                )
                result = _run_single(
                    args, alpha, latent_size, seed, methods, ours_configs,
                )
                outer_pbar.update(1)
                # Compact per-seed log
                for mid in method_ids:
                    m = result.get(mid)
                    if not m:
                        continue
                    logger.info(
                        "  %-28s avg_eval=%.5f top_mS=%.4f rest=%.4f",
                        mid,
                        m.get("avg_eval_loss", float("nan")),
                        m.get("cross_cos_top_mS_mean", float("nan")),
                        m.get("cross_cos_rest_mean", float("nan")),
                    )
                seed_results.append(result)

            agg = _aggregate(seed_results, method_ids)
            sweep_results.append({
                "alpha_target": alpha,
                "latent_size": latent_size,
                "num_seeds": args.num_seeds,
                "method_ids": method_ids,
                "aggregate": agg,
                "seed_results": seed_results,
            })

    outer_pbar.close()
    return {
        "sweep_param": "method_x_alpha_x_latent_size",
        "sweep_results": sweep_results,
        "method_ids": method_ids,
    }


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Algorithm 1 validation on synthetic data")

    # Sweep
    parser.add_argument("--alpha-sweep", type=str, default="0.3,0.6,0.9,1.0")
    parser.add_argument("--latent-size-sweep", type=str, default="2048")
    parser.add_argument(
        "--methods",
        type=str,
        default="single_recon,two_recon,group_sparse,trace_align,ours",
        help="Comma-separated list of methods to train.",
    )

    # Data dims
    parser.add_argument("--n-shared", type=int, default=512)
    parser.add_argument("--n-image", type=int, default=256)
    parser.add_argument("--n-text", type=int, default=256)
    parser.add_argument("--representation-dim", type=int, default=768)

    # SAE config
    parser.add_argument("--latent-size", type=int, default=-1)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--gt-recovery-threshold", type=float, default=0.8)

    # Data generation
    parser.add_argument("--num-train", type=int, default=50_000)
    parser.add_argument("--num-eval", type=int, default=10_000)
    parser.add_argument("--sparsity", type=float, default=0.99)
    parser.add_argument("--min-active", type=int, default=1)
    parser.add_argument("--cmin", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--coeff-dist", type=str, choices=["exponential", "relu_gaussian"],
        default="exponential",
        help="Coefficient distribution: 'exponential' (cmin + Exp(beta)) or "
             "'relu_gaussian' (ReLU(mu + sigma*N(0,1)), SynthSAEBench).",
    )
    parser.add_argument("--coeff-mu", type=float, default=4.5,
                        help="Mean for relu_gaussian coeff dist.")
    parser.add_argument("--coeff-sigma", type=float, default=0.5,
                        help="Std for relu_gaussian coeff dist.")
    parser.add_argument("--obs-noise-std", type=float, default=0.0,
                        help="Std of Gaussian noise added to x, y embeddings (0 = no noise).")
    parser.add_argument("--max-interference", type=float, default=0.1)
    parser.add_argument(
        "--shared-coeff-mode",
        type=str, choices=["identical", "independent"], default="identical",
        help="'identical': same z_S for both modalities (v7 default). "
             "'independent': same support but magnitudes sampled independently.",
    )
    parser.add_argument(
        "--dictionary-strategy",
        type=str, choices=["gradient", "random"], default="gradient",
    )

    # Alpha calibration
    parser.add_argument("--calibration-lr", type=float, default=0.01)
    parser.add_argument("--calibration-max-iters", type=int, default=3000)
    parser.add_argument("--calibration-tol", type=float, default=0.01)

    # Training
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    # Baseline lambdas (from their papers)
    parser.add_argument("--group-sparse-lambda", type=float, default=0.05)
    parser.add_argument("--trace-beta", type=float, default=1e-4)
    parser.add_argument(
        "--single-paired-align-lambda", type=float, default=1.0,
        help="Lambda for the `single_paired_align` method: applies the "
             "paper's auxiliary alignment loss (same as ours) on top of a "
             "single shared decoder, without masking/matching.",
    )
    parser.add_argument(
        "--single-paired-align-mS", type=int, default=-1,
        help="m_S for single_paired_align's aux loss. Default -1 uses n_shared.",
    )
    parser.add_argument(
        "--single-paired-align-norm", type=str, default="global",
        choices=["group", "global"],
        help="Normalization variant of the aux loss for single_paired_align.",
    )
    parser.add_argument(
        "--iso-align-beta", type=float, default=0.03,
        help="Penalty coefficient for the IsoEnergy code-variant alignment "
             "loss (`_iso_alignment_penalty`, top-1 masked cosine). Default "
             "matches Parabrele/IsoEnergy src/losses.py::alignment_penalty.",
    )

    # Ours (Algorithm 1)
    parser.add_argument("--lambda-aux-sweep", type=str, default="1.0")
    parser.add_argument("--m-s-sweep", type=str, default="512")
    parser.add_argument("--k-align-sweep", type=str, default="6")
    parser.add_argument(
        "--aux-norm-sweep", type=str, default="group,global",
        help="Comma-separated list of L_aux normalization variants for ours. "
             "'group' (Option A) averages each term over its own group size; "
             "'global' (Option B) divides both sums by the total latent count "
             "n. Default sweeps both so a single run compares the variants.",
    )
    parser.add_argument("--rho", type=float, default=0.3)

    # Self-test
    parser.add_argument("--self-test", action="store_true")

    # Output
    parser.add_argument("--output-root", type=str, default="outputs/synthetic_theorem2")
    parser.add_argument("--run-tag", type=str, default="")

    return parser.parse_args()


def _self_test() -> None:
    """Unit check for _greedy_permutation_match_full."""
    rng = np.random.default_rng(0)
    n = 6
    C = rng.uniform(-0.1, 0.1, size=(n, n))
    # Inject a clean block of high-correlation pairs: (row, col)
    gt_pairs = [(0, 3), (2, 1), (4, 5)]
    for i, j in gt_pairs:
        C[i, j] = 0.9 + 0.01 * rng.standard_normal()
    P_I, P_T, ordered = _greedy_permutation_match_full(C)
    top_pairs = {(int(P_I[k]), int(P_T[k])) for k in range(len(gt_pairs))}
    assert top_pairs == set(gt_pairs), (
        f"greedy match failed: picked {top_pairs}, expected {set(gt_pairs)}"
    )
    assert ordered[0] >= ordered[1] >= ordered[2], (
        f"greedy order should be descending in the top {len(gt_pairs)}: {ordered}"
    )
    logger.info("self-test passed: greedy permutation recovered %s", sorted(gt_pairs))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.self_test:
        _self_test()
        return

    run_name = f"method_ns{args.n_shared}_k{args.k}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Theorem 2 / Algorithm 1 experiment")
    logger.info("Output dir: %s", run_dir)

    result = _run_experiment(args)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment": "theorem2_method",
            "n_shared": args.n_shared,
            "n_image": args.n_image,
            "n_text": args.n_text,
            "representation_dim": args.representation_dim,
            "k": args.k,
            "num_epochs": args.num_epochs,
            "num_seeds": args.num_seeds,
            "seed_base": args.seed_base,
            "sparsity": args.sparsity,
            "max_interference": args.max_interference,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "alpha_sweep": args.alpha_sweep,
            "latent_size_sweep": args.latent_size_sweep,
            "methods": args.methods,
            "group_sparse_lambda": args.group_sparse_lambda,
            "trace_beta": args.trace_beta,
            "iso_align_beta": args.iso_align_beta,
            "lambda_aux_sweep": args.lambda_aux_sweep,
            "m_s_sweep": args.m_s_sweep,
            "k_align_sweep": args.k_align_sweep,
            "rho": args.rho,
            "shared_coeff_mode": getattr(args, "shared_coeff_mode", "identical"),
            "coeff_dist": getattr(args, "coeff_dist", "exponential"),
            "coeff_mu": getattr(args, "coeff_mu", 4.5),
            "coeff_sigma": getattr(args, "coeff_sigma", 0.5),
            "obs_noise_std": getattr(args, "obs_noise_std", 0.0),
            "run_name": run_name,
        },
        **result,
    }

    json_path = run_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    loss_data = {}
    for sr_entry in result.get("sweep_results", []):
        for sr in sr_entry.get("seed_results", []):
            seed_val = sr.get("seed", "?")
            lh = sr.pop("_loss_histories", {})
            if lh:
                for mid, hist in lh.items():
                    key = f"alpha={sr_entry['alpha_target']}_L={sr_entry['latent_size']}_seed={seed_val}/{mid}"
                    loss_data[key] = hist
    if loss_data:
        loss_path = run_dir / "loss.json"
        with open(loss_path, "w", encoding="utf-8") as f:
            json.dump(loss_data, f, indent=1, ensure_ascii=False, default=str)
        logger.info("Saved: %s", loss_path)
    logger.info("Saved: %s", json_path)

    # Summary table
    logger.info("=" * 90)
    for sr in result["sweep_results"]:
        alpha = sr["alpha_target"]
        L = sr["latent_size"]
        a = sr["aggregate"]
        for mid in sr["method_ids"]:
            logger.info(
                "alpha=%.2f L=%d %-28s avg=%.5f top=%.4f rest=%.4f",
                alpha, L, mid,
                a.get(f"{mid}/avg_eval_loss_mean", float("nan")),
                a.get(f"{mid}/cross_cos_top_mS_mean_mean", float("nan")),
                a.get(f"{mid}/cross_cos_rest_mean_mean", float("nan")),
            )
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
