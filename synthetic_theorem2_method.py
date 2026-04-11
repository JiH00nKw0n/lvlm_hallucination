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
# Method 1: single recon (shared SAE)                                #
# ------------------------------------------------------------------ #


def _train_single_recon(
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    args: argparse.Namespace,
    latent_size: int,
    seed: int,
    device: torch.device,
) -> TopKSAE:
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
    pbar = tqdm(range(args.num_epochs), desc="single_recon", leave=False)
    for _ in pbar:
        last = 0.0
        for img_b, txt_b in loader:
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = model(hidden_states=hs_i)
            out_t = model(hidden_states=hs_t)
            loss = out_i.recon_loss + out_t.recon_loss
            _sae_step(model, optimizer, loss, args.max_grad_norm)
            last = float(loss.detach().item())
        pbar.set_postfix(loss=f"{last:.4f}")
    pbar.close()
    return model


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
) -> tuple[TopKSAE, TopKSAE]:
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

    pbar = tqdm(range(n_epochs), desc="two_recon", leave=False)
    for _ in pbar:
        last_i = last_t = 0.0
        for img_b, txt_b in loader:
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i)
            out_t = sae_t(hidden_states=hs_t)
            loss = out_i.recon_loss + out_t.recon_loss
            _joint_step([sae_i, sae_t], optimizer, loss, args.max_grad_norm)
            last_i = float(out_i.recon_loss.detach().item())
            last_t = float(out_t.recon_loss.detach().item())
        pbar.set_postfix(img=f"{last_i:.4f}", txt=f"{last_t:.4f}")
    pbar.close()
    return sae_i, sae_t


# ------------------------------------------------------------------ #
# Method 3/4: shared SAE + paired auxiliary loss                     #
# ------------------------------------------------------------------ #


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
) -> TopKSAE:
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
    pbar = tqdm(range(args.num_epochs), desc=desc, leave=False)
    for _ in pbar:
        last_rec = last_aux = 0.0
        for img_b, txt_b in loader:
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
        pbar.set_postfix(rec=f"{last_rec:.4f}", aux=f"{last_aux:.4f}")
    pbar.close()
    return model


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
    z_i: torch.Tensor, z_t: torch.Tensor, m_S: int,
) -> torch.Tensor:
    """Paper-exact diag-only auxiliary alignment loss.

    For the first `m_S` latent dims, push batch-level diagonal correlation to 1;
    for the remaining dims, push it to 0.
    """
    zi = z_i - z_i.mean(dim=0, keepdim=True)
    zt = z_t - z_t.mean(dim=0, keepdim=True)
    si = zi.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    st = zt.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    diag = (zi * zt).mean(dim=0) / (si * st)

    zero = diag.new_tensor(0.0)
    shared_term = ((diag[:m_S] - 1.0) ** 2).sum() if m_S > 0 else zero
    private_term = (diag[m_S:] ** 2).sum() if m_S < diag.shape[0] else zero
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

    # -- Stage 1: joint training, recon only --
    if stage1_epochs > 0:
        pbar1 = tqdm(
            range(stage1_epochs), desc="ours_stage1(recon)", leave=False,
        )
        for _ in pbar1:
            last_rec = 0.0
            for img_b, txt_b in loader:
                hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
                hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
                out_i = sae_i(hidden_states=hs_i)
                out_t = sae_t(hidden_states=hs_t)
                loss = out_i.recon_loss + out_t.recon_loss
                _joint_step([sae_i, sae_t], optimizer, loss, args.max_grad_norm)
                last_rec = float(loss.detach().item())
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
    }

    if stage2_epochs <= 0:
        return sae_i, sae_t, diagnostics

    # -- Stage 2: joint training, recon + lambda_aux * L_aux --
    pbar2 = tqdm(
        range(stage2_epochs),
        desc=f"ours_stage2(mS={m_S_supplied},lam={lambda_aux})",
        leave=False,
    )
    for _ in pbar2:
        last_rec = last_aux = 0.0
        for img_b, txt_b in loader:
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i, return_dense_latents=True)
            out_t = sae_t(hidden_states=hs_t, return_dense_latents=True)
            z_i = _dense_latents(out_i)
            z_t = _dense_latents(out_t)
            aux = _auxiliary_alignment_loss(z_i, z_t, m_S_supplied)
            loss = out_i.recon_loss + out_t.recon_loss + lambda_aux * aux
            _joint_step([sae_i, sae_t], optimizer, loss, args.max_grad_norm)
            last_rec = float((out_i.recon_loss + out_t.recon_loss).detach().item())
            last_aux = float(aux.detach().item())
        pbar2.set_postfix(rec=f"{last_rec:.4f}", aux=f"{last_aux:.4f}")
    pbar2.close()

    return sae_i, sae_t, diagnostics


# ------------------------------------------------------------------ #
# Metrics for a method                                               #
# ------------------------------------------------------------------ #


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
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
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

    # Multi-threshold MGT: always measure at three τ's so the report can
    # contrast loose (0.80 ≈ 36.9°), strict (0.95 ≈ 18.2°), and very
    # strict (0.99 ≈ 8.1°) angular recovery.
    img_mgt_multi = _compute_recovery_metrics_multi_tau(
        w_dec_img, phi_S, (0.8, 0.95, 0.99),
    )
    txt_mgt_multi = _compute_recovery_metrics_multi_tau(
        w_dec_txt, psi_S, (0.8, 0.95, 0.99),
    )

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

    return {
        "img_eval_loss": img_eval,
        "txt_eval_loss": txt_eval,
        "avg_eval_loss": avg_eval,
        # Backward-compat: `img_mgt_shared` / `txt_mgt_shared` == τ=0.8 figure.
        "img_mgt_shared": img_mgt_multi["mgt_tau0.8"],
        "txt_mgt_shared": txt_mgt_multi["mgt_tau0.8"],
        "img_mip_shared": img_mgt_multi["mip"],
        "txt_mip_shared": txt_mgt_multi["mip"],
        # Multi-tau MGT (new).
        "img_mgt_shared_tau0.8": img_mgt_multi["mgt_tau0.8"],
        "img_mgt_shared_tau0.95": img_mgt_multi["mgt_tau0.95"],
        "img_mgt_shared_tau0.99": img_mgt_multi["mgt_tau0.99"],
        "txt_mgt_shared_tau0.8": txt_mgt_multi["mgt_tau0.8"],
        "txt_mgt_shared_tau0.95": txt_mgt_multi["mgt_tau0.95"],
        "txt_mgt_shared_tau0.99": txt_mgt_multi["mgt_tau0.99"],
        "cross_cos_top_mS_mean": top_diag,
        "cross_cos_rest_mean": rest_diag,
        "cross_cos_gt_mean": cross["cross_cos_mean"],
        "alpha_proxy_mean": cross["alpha_proxy_mean"],
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

    @property
    def tag(self) -> str:
        return f"lam{self.lambda_aux}_mS{self.m_S}_k{self.k_align}"


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
            eval_img=eval_img, eval_txt=eval_txt,
            phi_S=builder.phi_S, psi_S=builder.psi_S,
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

    if "single_recon" in methods:
        model = _train_single_recon(train_img, train_txt, args, latent_size, seed, device)
        _finish("single_recon", model, model)

    if "two_recon" in methods:
        sae_i, sae_t = _train_two_recon(train_img, train_txt, args, latent_size, seed, device)
        _finish("two_recon", sae_i, sae_t)

    if "group_sparse" in methods:
        model = _train_single_paired_aux(
            train_img, train_txt, args, latent_size, seed, device,
            aux_fn=group_sparse_loss, aux_weight=args.group_sparse_lambda,
            desc="group_sparse",
        )
        _finish("group_sparse", model, model)

    if "trace_align" in methods:
        model = _train_single_paired_aux(
            train_img, train_txt, args, latent_size, seed, device,
            aux_fn=trace_alignment_loss, aux_weight=args.trace_beta,
            desc="trace_align",
        )
        _finish("trace_align", model, model)

    if "ours" in methods:
        for cfg in ours_configs:
            args._current_m_S = cfg.m_S
            sae_i, sae_t, diag = _train_ours(
                train_img, train_txt, args, latent_size, seed, device,
                k_align=cfg.k_align,
                lambda_aux=cfg.lambda_aux,
                m_S_supplied=cfg.m_S,
            )
            _finish(f"ours::{cfg.tag}", sae_i, sae_t, extras=diag)
        args._current_m_S = args.n_shared

    return result


# ------------------------------------------------------------------ #
# Aggregation / metric keys                                          #
# ------------------------------------------------------------------ #


METRIC_SUFFIXES = [
    "img_eval_loss",
    "txt_eval_loss",
    "avg_eval_loss",
    "img_mgt_shared",
    "img_mip_shared",
    "txt_mgt_shared",
    "txt_mip_shared",
    "img_mgt_shared_tau0.8",
    "img_mgt_shared_tau0.95",
    "img_mgt_shared_tau0.99",
    "txt_mgt_shared_tau0.8",
    "txt_mgt_shared_tau0.95",
    "txt_mgt_shared_tau0.99",
    "cross_cos_top_mS_mean",
    "cross_cos_rest_mean",
    "cross_cos_gt_mean",
    "alpha_proxy_mean",
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
    return [
        OursConfig(lambda_aux=l, m_S=m, k_align=k)
        for l, m, k in itertools.product(lambdas, m_ss, k_aligns)
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
    parser.add_argument("--max-interference", type=float, default=0.1)
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

    # Ours (Algorithm 1)
    parser.add_argument("--lambda-aux-sweep", type=str, default="1.0")
    parser.add_argument("--m-s-sweep", type=str, default="512")
    parser.add_argument("--k-align-sweep", type=str, default="4")
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
            "lambda_aux_sweep": args.lambda_aux_sweep,
            "m_s_sweep": args.m_s_sweep,
            "k_align_sweep": args.k_align_sweep,
            "rho": args.rho,
            "run_name": run_name,
        },
        **result,
    }

    json_path = run_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
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
