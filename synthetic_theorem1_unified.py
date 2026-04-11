"""
Unified Synthetic Theorem 1 experiments with diagnostics.

Extends the original exp1a with:
  - Fair Two-SAE baseline (each latent_size // 2)
  - Per-modality eval loss decomposition (M6)
  - Compromise diagnostics for merged features (M4)
  - Extended merge/split analysis (M3)
  - GT-free metric validation via Two-SAE proxy (M5 prep)

See papers/synthetic_theorem_1_fw.md for the full framework description.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from synthetic_sae_theory_experiment import (
    _seed_everything,
    _resolve_device,
    _train_sae,
    _compute_recovery_metrics,
    _normalize_rows,
)
from synthetic_theorem1_experiment import _compute_alignment_and_mono
from src.datasets.synthetic_theory_feature import SyntheticTheoryFeatureBuilder
from src.models.modeling_sae import TopKSAE, TopKSAEConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Alternating-modality Single SAE training (Follow-up 2 variant)     #
# ------------------------------------------------------------------ #


def _train_sae_alternating(
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    eval_data: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[TopKSAE, float, float]:
    """Train one TopKSAE by alternating image-batch and text-batch grad steps."""
    from synthetic_sae_theory_experiment import _seed_everything as _seed
    _seed(seed)

    img_loader = DataLoader(
        TensorDataset(train_img),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    txt_loader = DataLoader(
        TensorDataset(train_txt),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        TensorDataset(eval_data),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    cfg = TopKSAEConfig(
        hidden_size=args.representation_dim,
        latent_size=args.latent_size,
        k=args.k,
        normalize_decoder=True,
    )
    model = TopKSAE(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    def _step(batch: torch.Tensor) -> float:
        hs = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
        out = model(hidden_states=hs)
        loss = out.recon_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if model.W_dec.grad is not None:
            model.remove_gradient_parallel_to_decoder_directions()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        model.set_decoder_norm_to_unit_norm()
        return float(loss.detach().item())

    train_loss_sum, train_steps = 0.0, 0
    for epoch in range(args.num_epochs):
        for (img_b,), (txt_b,) in zip(img_loader, txt_loader):
            train_loss_sum += _step(img_b)
            train_loss_sum += _step(txt_b)
            train_steps += 2
        if args.log_every > 0:
            logger.info(
                "  alt seed=%d epoch=%d steps=%d recon=%.6f",
                seed, epoch + 1, train_steps,
                train_loss_sum / max(train_steps, 1),
            )

    model.eval()
    eval_loss_sum, eval_steps = 0.0, 0
    with torch.no_grad():
        for (batch,) in eval_loader:
            hs = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
            out = model(hidden_states=hs)
            eval_loss_sum += float(out.recon_loss.item())
            eval_steps += 1

    return (
        model,
        train_loss_sum / max(train_steps, 1),
        eval_loss_sum / max(eval_steps, 1),
    )


# ------------------------------------------------------------------ #
# Per-modality eval loss (M6)                                        #
# ------------------------------------------------------------------ #


def _eval_loss_single_modality(
    model: TopKSAE,
    data: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute eval reconstruction loss for a single modality."""
    model.eval()
    loader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_sum, steps = 0.0, 0
    with torch.no_grad():
        for (batch,) in loader:
            hs = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
            out = model(hidden_states=hs)
            loss_sum += float(out.recon_loss.item())
            steps += 1
    return loss_sum / max(steps, 1)


# ------------------------------------------------------------------ #
# Per-feature reconstruction error by merge/split (H0 vs H1 test)   #
# ------------------------------------------------------------------ #


def _compute_directional_recon_error(
    model: TopKSAE,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    shared_mask: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    per_feature_info: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute directional reconstruction error per shared feature, grouped by merge/split.

    For each shared feature i, projects the reconstruction residual e = x - x_hat
    onto the GT direction Phi_S[:,i] (for image) or Psi_S[:,i] (for text):

        directional_error_i = <e, phi_hat_i>^2

    This measures "how much reconstruction error remains in the direction of
    feature i." If the SAE perfectly reconstructed the component along Phi_S[:,i],
    the residual would be orthogonal to it and the projection would be ~0.

    H0 predicts: merged features have higher directional error (compromise
    direction can't fully reconstruct the GT direction component).
    H1 predicts: split features have higher directional error (capacity
    shortage degrades the dedicated latent's quality).

    Args:
        model: Trained Single SAE.
        eval_img: Image eval data, shape (N, m).
        eval_txt: Text eval data, shape (N, m).
        shared_mask: Binary activation mask, shape (N, n_S).
        phi_S: Image-side shared dictionary, shape (m, n_S).
        psi_S: Text-side shared dictionary, shape (m, n_S).
        per_feature_info: Per-feature dicts with 'is_merged' and 'idx'.
        batch_size: Batch size for forward pass.
        device: Torch device.

    Returns:
        Dict with merged/split directional error comparison.
    """
    model.eval()

    # Compute per-sample residual vectors (not scalar MSE)
    def _residual_vectors(data: torch.Tensor) -> np.ndarray:
        residuals = []
        loader = DataLoader(
            TensorDataset(data), batch_size=batch_size, shuffle=False, num_workers=0,
        )
        with torch.no_grad():
            for (batch,) in loader:
                hs = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
                out = model(hidden_states=hs)
                res = (hs.squeeze(1) - out.output.squeeze(1)).cpu().numpy()
                residuals.append(res)
        return np.concatenate(residuals, axis=0)  # (N, m)

    img_residuals = _residual_vectors(eval_img)  # (N, m)
    txt_residuals = _residual_vectors(eval_txt)  # (N, m)

    # Normalize GT directions
    phi_norm = _normalize_rows(phi_S.T.astype(np.float64))  # (n_S, m)
    psi_norm = _normalize_rows(psi_S.T.astype(np.float64))  # (n_S, m)

    merged_dir_err_img: list[float] = []
    merged_dir_err_txt: list[float] = []
    split_dir_err_img: list[float] = []
    split_dir_err_txt: list[float] = []

    for feat in per_feature_info:
        i = feat["idx"]
        active = shared_mask[:, i] > 0
        if active.sum() == 0:
            continue

        # Project residuals onto GT direction: <e, phi_hat_i>^2
        proj_img = (img_residuals[active] @ phi_norm[i]) ** 2  # (n_active,)
        proj_txt = (txt_residuals[active] @ psi_norm[i]) ** 2  # (n_active,)

        err_img = float(proj_img.mean())
        err_txt = float(proj_txt.mean())

        if feat["is_merged"]:
            merged_dir_err_img.append(err_img)
            merged_dir_err_txt.append(err_txt)
        else:
            split_dir_err_img.append(err_img)
            split_dir_err_txt.append(err_txt)

    def _safe_mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "merged_recon_error_img": _safe_mean(merged_dir_err_img),
        "merged_recon_error_txt": _safe_mean(merged_dir_err_txt),
        "split_recon_error_img": _safe_mean(split_dir_err_img),
        "split_recon_error_txt": _safe_mean(split_dir_err_txt),
        "merged_recon_error_avg": (_safe_mean(merged_dir_err_img) + _safe_mean(merged_dir_err_txt)) / 2
            if merged_dir_err_img else float("nan"),
        "split_recon_error_avg": (_safe_mean(split_dir_err_img) + _safe_mean(split_dir_err_txt)) / 2
            if split_dir_err_img else float("nan"),
        "n_merged": len(merged_dir_err_img),
        "n_split": len(split_dir_err_img),
    }


# ------------------------------------------------------------------ #
# Compromise diagnostics (M4) + extended merge/split analysis (M3)   #
# ------------------------------------------------------------------ #


def _compute_compromise_diagnostics(
    w_dec: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> dict[str, Any]:
    """Per-feature diagnostics: merge/split classification, compromise angle, MGT at multi-threshold.

    Args:
        w_dec: Decoder weights, shape (latent_size, m).
        phi_S: Image-side shared dictionary, shape (m, n_S).
        psi_S: Text-side shared dictionary, shape (m, n_S).
        thresholds: Cosine thresholds for multi-threshold MGT analysis.

    Returns:
        Dict with aggregate diagnostics and per-feature breakdown.
    """
    n_S = phi_S.shape[1]

    dec_norm = _normalize_rows(w_dec.astype(np.float64))
    phi_norm = _normalize_rows(phi_S.T.astype(np.float64))   # (n_S, m)
    psi_norm = _normalize_rows(psi_S.T.astype(np.float64))   # (n_S, m)

    sim_img = np.abs(dec_norm @ phi_norm.T)  # (latent_size, n_S)
    sim_txt = np.abs(dec_norm @ psi_norm.T)  # (latent_size, n_S)

    per_feature: list[dict[str, Any]] = []
    merged_gaps_img: list[float] = []
    merged_gaps_txt: list[float] = []
    split_gaps_img: list[float] = []
    split_gaps_txt: list[float] = []
    cos_to_gt_img_all: list[float] = []
    cos_to_gt_txt_all: list[float] = []

    for i in range(n_S):
        best_img_idx = int(sim_img[:, i].argmax())
        best_txt_idx = int(sim_txt[:, i].argmax())
        cos_img = float(sim_img[best_img_idx, i])
        cos_txt = float(sim_txt[best_txt_idx, i])
        is_merged = best_img_idx == best_txt_idx

        gap_img = 1.0 - cos_img
        gap_txt = 1.0 - cos_txt

        cos_to_gt_img_all.append(cos_img)
        cos_to_gt_txt_all.append(cos_txt)

        if is_merged:
            merged_gaps_img.append(gap_img)
            merged_gaps_txt.append(gap_txt)
        else:
            split_gaps_img.append(gap_img)
            split_gaps_txt.append(gap_txt)

        per_feature.append({
            "idx": i,
            "best_img_latent": best_img_idx,
            "best_txt_latent": best_txt_idx,
            "is_merged": is_merged,
            "cos_img": cos_img,
            "cos_txt": cos_txt,
            "gap_img": gap_img,
            "gap_txt": gap_txt,
            "compromise_angle": (gap_img + gap_txt) / 2 if is_merged else float("nan"),
        })

    # Multi-threshold MGT (P7 verification)
    cos_img_arr = np.array(cos_to_gt_img_all)
    cos_txt_arr = np.array(cos_to_gt_txt_all)
    multi_thresh_mgt: dict[str, float] = {}
    for tau in thresholds:
        multi_thresh_mgt[f"mgt_img_tau{tau}"] = float((cos_img_arr > tau).mean())
        multi_thresh_mgt[f"mgt_txt_tau{tau}"] = float((cos_txt_arr > tau).mean())

    # Per-feature cos distribution stats (P5 verification)
    cos_img_std = float(cos_img_arr.std())
    cos_txt_std = float(cos_txt_arr.std())

    # Count features in threshold-sensitive zone [0.7, 0.8] (P6 verification)
    near_threshold_img = float(((cos_img_arr >= 0.7) & (cos_img_arr < 0.8)).mean())
    near_threshold_txt = float(((cos_txt_arr >= 0.7) & (cos_txt_arr < 0.8)).mean())

    n_merged = len(merged_gaps_img)
    n_split = len(split_gaps_img)

    return {
        # M3: Merge/split rates
        "merge_count": n_merged,
        "split_count": n_split,
        "merge_rate": n_merged / n_S if n_S > 0 else 0.0,
        # M4: Compromise angle (merged features only)
        "merged_gap_img_mean": float(np.mean(merged_gaps_img)) if merged_gaps_img else float("nan"),
        "merged_gap_txt_mean": float(np.mean(merged_gaps_txt)) if merged_gaps_txt else float("nan"),
        "merged_compromise_angle_mean": float(np.mean(
            [(g1 + g2) / 2 for g1, g2 in zip(merged_gaps_img, merged_gaps_txt)]
        )) if merged_gaps_img else float("nan"),
        # Split features gap
        "split_gap_img_mean": float(np.mean(split_gaps_img)) if split_gaps_img else float("nan"),
        "split_gap_txt_mean": float(np.mean(split_gaps_txt)) if split_gaps_txt else float("nan"),
        # Per-feature cos distribution (P5)
        "cos_img_mean": float(cos_img_arr.mean()),
        "cos_img_std": cos_img_std,
        "cos_txt_mean": float(cos_txt_arr.mean()),
        "cos_txt_std": cos_txt_std,
        # Near-threshold fraction (P6)
        "near_threshold_img_rate": near_threshold_img,
        "near_threshold_txt_rate": near_threshold_txt,
        # Multi-threshold MGT (P7)
        **multi_thresh_mgt,
        # Per-feature (stripped from saved JSON, kept in memory)
        "per_feature": per_feature,
    }


# ------------------------------------------------------------------ #
# Latent utilization diagnostics (M4 capacity)                       #
# ------------------------------------------------------------------ #


def _compute_latent_utilization(
    model: TopKSAE,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure per-latent activation frequency split by modality.

    Returns:
        alive_rate, img_dominant_rate, txt_dominant_rate, mixed_rate, dead_rate.
    """
    latent_size = model.latent_size
    img_counts = np.zeros(latent_size, dtype=np.int64)
    txt_counts = np.zeros(latent_size, dtype=np.int64)

    model.eval()

    def _count_activations(data: torch.Tensor, counts: np.ndarray) -> None:
        loader = DataLoader(
            TensorDataset(data),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        with torch.no_grad():
            for (batch,) in loader:
                hs = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
                out = model(hidden_states=hs)
                if out.latent_indices is not None:
                    indices = out.latent_indices.cpu().numpy().ravel()
                    np.add.at(counts, indices, 1)

    _count_activations(eval_img, img_counts)
    _count_activations(eval_txt, txt_counts)

    total = img_counts + txt_counts
    alive = total > 0
    alive_rate = float(alive.mean())
    dead_rate = 1.0 - alive_rate

    # Among alive latents, classify by dominance
    alive_mask = alive.astype(bool)
    n_alive = int(alive_mask.sum())
    if n_alive == 0:
        return {
            "alive_rate": 0.0, "dead_rate": 1.0,
            "img_dominant_rate": 0.0, "txt_dominant_rate": 0.0, "mixed_rate": 0.0,
        }

    img_frac = np.where(alive_mask, img_counts / total.clip(min=1), 0.0)
    img_dominant = alive_mask & (img_frac > 0.8)
    txt_dominant = alive_mask & (img_frac < 0.2)
    mixed = alive_mask & ~img_dominant & ~txt_dominant

    return {
        "alive_rate": alive_rate,
        "dead_rate": dead_rate,
        "img_dominant_rate": float(img_dominant.sum()) / latent_size,
        "txt_dominant_rate": float(txt_dominant.sum()) / latent_size,
        "mixed_rate": float(mixed.sum()) / latent_size,
    }


# ------------------------------------------------------------------ #
# GT-free alpha estimation via Two-SAE (M1/M5 validation)           #
# ------------------------------------------------------------------ #


def _gt_based_shared_alignment_mismatched(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
) -> dict[str, Any]:
    """Cross-SAE alignment when Phi_S != Psi_S.

    For each GT shared atom i:
      - Find best matching latent in image SAE via phi_S[:,i]
      - Find best matching latent in text SAE via psi_S[:,i]
      - Compute cosine between those two decoder rows

    Also estimates alpha from Two-SAE decoder directions (GT-free proxy).

    Returns:
        Dict with cross-alignment stats and per-feature GT-free alpha estimates.
    """
    n_S = phi_S.shape[1]

    img_norm = _normalize_rows(w_dec_img.astype(np.float64))
    txt_norm = _normalize_rows(w_dec_txt.astype(np.float64))
    phi_norm = _normalize_rows(phi_S.T.astype(np.float64))
    psi_norm = _normalize_rows(psi_S.T.astype(np.float64))

    sim_img = np.abs(img_norm @ phi_norm.T)  # (L_img, n_S)
    sim_txt = np.abs(txt_norm @ psi_norm.T)  # (L_txt, n_S)
    best_img = sim_img.argmax(axis=0)  # (n_S,)
    best_txt = sim_txt.argmax(axis=0)  # (n_S,)

    # Cross-alignment: cosine between matched decoder rows
    cross_cos = np.array([
        np.abs(img_norm[best_img[i]] @ txt_norm[best_txt[i]])
        for i in range(n_S)
    ])

    # GT-free alpha estimate: cosine between the Two-SAE decoder directions
    # These approximate cos(Phi_S[:,i], Psi_S[:,i]) without using GT
    alpha_proxy = cross_cos  # This IS the two-SAE based alpha estimate

    # Also compute MIP for each SAE against its own GT
    mip_img = np.array([float(sim_img[best_img[i], i]) for i in range(n_S)])
    mip_txt = np.array([float(sim_txt[best_txt[i], i]) for i in range(n_S)])

    return {
        "cross_cos_mean": float(cross_cos.mean()),
        "cross_cos_std": float(cross_cos.std()),
        "cross_cos_min": float(cross_cos.min()),
        "alpha_proxy_mean": float(alpha_proxy.mean()),
        "alpha_proxy_std": float(alpha_proxy.std()),
        "img_mip_mean": float(mip_img.mean()),
        "txt_mip_mean": float(mip_txt.mean()),
    }


# ------------------------------------------------------------------ #
# Main experiment runner                                             #
# ------------------------------------------------------------------ #


def _run_single(
    args: argparse.Namespace,
    seed: int,
    alpha: float,
    latent_size: int,
) -> dict[str, Any]:
    """Run one (alpha, latent_size, seed) configuration.

    Trains:
      - Single SAE on concat(img, txt) with latent_size
      - Two SAEs (img-only, txt-only) each with latent_size // 2

    Computes all unified metrics M1-M6.
    """
    device = _resolve_device(args.device)

    # -- Build data --
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
    train_img = torch.from_numpy(ds["train"]["image_representation"])
    train_txt = torch.from_numpy(ds["train"]["text_representation"])
    eval_img = torch.from_numpy(ds["eval"]["image_representation"])
    eval_txt = torch.from_numpy(ds["eval"]["text_representation"])

    # -- Single SAE --
    train_concat = torch.cat([train_img, train_txt], dim=0)
    eval_concat = torch.cat([eval_img, eval_txt], dim=0)

    args_single = argparse.Namespace(**vars(args))
    args_single.latent_size = latent_size

    if getattr(args, "alternating_single_sae", False):
        single_model, single_train, single_eval = _train_sae_alternating(
            train_img, train_txt, eval_concat, args_single, seed, device,
        )
    else:
        single_model, single_train, single_eval = _train_sae(
            train_concat, eval_concat, args_single, seed, device,
        )
    w_dec_single = single_model.W_dec.detach().cpu().numpy()

    # M6: Per-modality eval loss
    img_eval_loss = _eval_loss_single_modality(single_model, eval_img, args.batch_size, device)
    txt_eval_loss = _eval_loss_single_modality(single_model, eval_txt, args.batch_size, device)

    # M3 + M4: Compromise diagnostics
    compromise = _compute_compromise_diagnostics(w_dec_single, builder.phi_S, builder.psi_S)

    # Directional recon error by merge/split (H0 vs H1 decisive test)
    per_feature_recon = _compute_directional_recon_error(
        single_model, eval_img, eval_txt,
        ds["eval"]["shared_ground_truth"],
        builder.phi_S, builder.psi_S,
        compromise["per_feature"],
        args.batch_size, device,
    )

    # Latent utilization
    utilization = _compute_latent_utilization(
        single_model, eval_img, eval_txt, args.batch_size, device,
    )

    # C2/C3 metrics (backward compat)
    c2c3 = _compute_alignment_and_mono(w_dec_single, builder.phi_S, builder.psi_S)

    # GT recovery
    recovery_img = _compute_recovery_metrics(w_dec_single, builder.phi_S, args.gt_recovery_threshold)
    recovery_txt = _compute_recovery_metrics(w_dec_single, builder.psi_S, args.gt_recovery_threshold)

    # Private feature recovery (P9 verification)
    private_recovery: dict[str, float] = {}
    if builder.phi_I is not None and builder.phi_I.shape[1] > 0:
        prv_img = _compute_recovery_metrics(w_dec_single, builder.phi_I, args.gt_recovery_threshold)
        private_recovery["private_img_mgt"] = prv_img["gt_recovery"]
        private_recovery["private_img_mip"] = prv_img["mip"]
    if builder.psi_T is not None and builder.psi_T.shape[1] > 0:
        prv_txt = _compute_recovery_metrics(w_dec_single, builder.psi_T, args.gt_recovery_threshold)
        private_recovery["private_txt_mgt"] = prv_txt["gt_recovery"]
        private_recovery["private_txt_mip"] = prv_txt["mip"]

    del single_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Two SAEs (fair: each L/2) --
    half_latent = latent_size // 2
    args_two = argparse.Namespace(**vars(args))
    args_two.latent_size = half_latent

    _seed_everything(seed)
    img_model, img_train, img_eval = _train_sae(
        train_img, eval_img, args_two, seed, device,
    )
    w_dec_img = img_model.W_dec.detach().cpu().numpy()
    img_recovery = _compute_recovery_metrics(w_dec_img, builder.phi_S, args.gt_recovery_threshold)
    img_prv_recovery: dict[str, float] = {}
    if builder.phi_I is not None and builder.phi_I.shape[1] > 0:
        prv = _compute_recovery_metrics(w_dec_img, builder.phi_I, args.gt_recovery_threshold)
        img_prv_recovery = {"private_mgt": prv["gt_recovery"], "private_mip": prv["mip"]}

    # Two-SAE image directional error (residual vectors needed before model deletion)
    img_model.eval()
    two_img_residuals: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(eval_img), batch_size=args.batch_size, shuffle=False):
            hs = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
            out = img_model(hidden_states=hs)
            two_img_residuals.append((hs.squeeze(1) - out.output.squeeze(1)).cpu().numpy())
    two_img_residuals_arr = np.concatenate(two_img_residuals, axis=0)

    del img_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed + 10000)
    txt_model, txt_train, txt_eval = _train_sae(
        train_txt, eval_txt, args_two, seed + 10000, device,
    )
    w_dec_txt = txt_model.W_dec.detach().cpu().numpy()
    txt_recovery = _compute_recovery_metrics(w_dec_txt, builder.psi_S, args.gt_recovery_threshold)
    txt_prv_recovery: dict[str, float] = {}
    if builder.psi_T is not None and builder.psi_T.shape[1] > 0:
        prv = _compute_recovery_metrics(w_dec_txt, builder.psi_T, args.gt_recovery_threshold)
        txt_prv_recovery = {"private_mgt": prv["gt_recovery"], "private_mip": prv["mip"]}

    # Two-SAE text directional error
    txt_model.eval()
    two_txt_residuals: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(eval_txt), batch_size=args.batch_size, shuffle=False):
            hs = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
            out = txt_model(hidden_states=hs)
            two_txt_residuals.append((hs.squeeze(1) - out.output.squeeze(1)).cpu().numpy())
    two_txt_residuals_arr = np.concatenate(two_txt_residuals, axis=0)

    del txt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Two-SAE directional error per shared feature
    phi_norm = _normalize_rows(builder.phi_S.T.astype(np.float64))
    psi_norm = _normalize_rows(builder.psi_S.T.astype(np.float64))
    shared_mask = ds["eval"]["shared_ground_truth"]
    two_sae_dir_errors: list[float] = []
    for i in range(builder.n_shared):
        active = shared_mask[:, i] > 0
        if active.sum() == 0:
            continue
        proj_img = (two_img_residuals_arr[active] @ phi_norm[i]) ** 2
        proj_txt = (two_txt_residuals_arr[active] @ psi_norm[i]) ** 2
        two_sae_dir_errors.append((float(proj_img.mean()) + float(proj_txt.mean())) / 2)
    two_sae_dir_error_mean = float(np.mean(two_sae_dir_errors)) if two_sae_dir_errors else float("nan")

    # Two SAE multi-threshold MGT
    # Image SAE vs Phi_S, Text SAE vs Psi_S
    dec_img_norm = _normalize_rows(w_dec_img.astype(np.float64))
    dec_txt_norm = _normalize_rows(w_dec_txt.astype(np.float64))
    sim_two_img = np.abs(dec_img_norm @ phi_norm.T).max(axis=0)  # (n_S,) best cos per GT
    sim_two_txt = np.abs(dec_txt_norm @ psi_norm.T).max(axis=0)  # (n_S,)
    two_sae_multi_mgt: dict[str, float] = {}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        two_sae_multi_mgt[f"mgt_img_tau{tau}"] = float((sim_two_img > tau).mean())
        two_sae_multi_mgt[f"mgt_txt_tau{tau}"] = float((sim_two_txt > tau).mean())

    # M1/M5: Cross-SAE alignment (GT-free alpha proxy)
    cross_align = _gt_based_shared_alignment_mismatched(
        w_dec_img, w_dec_txt, builder.phi_S, builder.psi_S,
    )

    # M2: Loss ratio
    two_sae_avg_loss = (img_eval + txt_eval) / 2
    loss_ratio = single_eval / two_sae_avg_loss if two_sae_avg_loss > 0 else float("nan")

    # Strip per_feature before returning
    compromise_agg = {k: v for k, v in compromise.items() if k != "per_feature"}

    return {
        "seed": seed,
        "alpha_target": alpha,
        "latent_size": latent_size,
        "alpha_actual_mean": builder.mean_shared_cosine_similarity,
        "alpha_actual_std": builder.std_shared_cosine_similarity,
        # --- Single SAE ---
        "single_sae": {
            "train_loss": single_train,
            "eval_loss": single_eval,
            "img_eval_loss": img_eval_loss,
            "txt_eval_loss": txt_eval_loss,
            "mgt_shared_img": recovery_img["gt_recovery"],
            "mip_shared_img": recovery_img["mip"],
            "mgt_shared_txt": recovery_txt["gt_recovery"],
            "mip_shared_txt": recovery_txt["mip"],
            **private_recovery,
            # C2/C3 (backward compat)
            "c2_alignment_cos_mean": c2c3["c2_alignment_cos_mean"],
            "c2_same_idx_rate": c2c3["c2_same_idx_rate"],
            "c3_ratio_mean": c2c3.get("c3_ratio_mean", float("nan")),
            # M3/M4: Compromise diagnostics
            **compromise_agg,
            # Per-feature recon error by merge/split (H0 vs H1)
            **per_feature_recon,
            # Latent utilization
            **utilization,
        },
        # --- Two SAEs ---
        "two_sae": {
            "latent_size_per_sae": half_latent,
            "img_train_loss": img_train,
            "img_eval_loss": img_eval,
            "txt_train_loss": txt_train,
            "txt_eval_loss": txt_eval,
            "avg_eval_loss": two_sae_avg_loss,
            "img_mgt_shared": img_recovery["gt_recovery"],
            "img_mip_shared": img_recovery["mip"],
            "txt_mgt_shared": txt_recovery["gt_recovery"],
            "txt_mip_shared": txt_recovery["mip"],
            **{f"img_{k}": v for k, v in img_prv_recovery.items()},
            **{f"txt_{k}": v for k, v in txt_prv_recovery.items()},
            # M1/M5: Cross-alignment
            **cross_align,
            "dir_error_mean": two_sae_dir_error_mean,
            # Multi-threshold MGT
            **two_sae_multi_mgt,
        },
        # --- Comparison ---
        "loss_ratio": loss_ratio,
    }


# ------------------------------------------------------------------ #
# Aggregation                                                        #
# ------------------------------------------------------------------ #


def _extract_metric(result: dict[str, Any], key: str) -> float | None:
    """Extract metric from nested dict, supporting slash-separated keys."""
    if key in result:
        v = result[key]
        return float(v) if v is not None else None
    parts = key.split("/")
    obj = result
    for p in parts:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return None
    if obj is None or isinstance(obj, dict):
        return None
    return float(obj)


def _aggregate_seed_results(
    seed_results: list[dict[str, Any]], metric_keys: list[str],
) -> dict[str, float]:
    """Aggregate metrics across seeds: mean and std."""
    agg: dict[str, float] = {}
    for key in metric_keys:
        vals = [_extract_metric(r, key) for r in seed_results]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals, ddof=0))
        else:
            agg[f"{key}_mean"] = float("nan")
            agg[f"{key}_std"] = float("nan")
    return agg


# ------------------------------------------------------------------ #
# Experiment dispatch                                                #
# ------------------------------------------------------------------ #


METRIC_KEYS = [
    # Single SAE
    "single_sae/eval_loss",
    "single_sae/img_eval_loss",
    "single_sae/txt_eval_loss",
    "single_sae/mgt_shared_img",
    "single_sae/mip_shared_img",
    "single_sae/mgt_shared_txt",
    "single_sae/mip_shared_txt",
    "single_sae/c2_alignment_cos_mean",
    "single_sae/c2_same_idx_rate",
    # M3/M4 compromise
    "single_sae/merge_rate",
    "single_sae/merged_compromise_angle_mean",
    "single_sae/merged_gap_img_mean",
    "single_sae/merged_gap_txt_mean",
    "single_sae/split_gap_img_mean",
    "single_sae/cos_img_std",
    "single_sae/near_threshold_img_rate",
    # Multi-threshold MGT
    "single_sae/mgt_img_tau0.8",
    "single_sae/mgt_img_tau0.9",
    "single_sae/mgt_img_tau0.95",
    "single_sae/mgt_img_tau0.99",
    # Latent utilization
    "single_sae/alive_rate",
    "single_sae/img_dominant_rate",
    "single_sae/txt_dominant_rate",
    "single_sae/mixed_rate",
    # Private recovery
    "single_sae/private_img_mgt",
    "single_sae/private_txt_mgt",
    # Per-feature recon error (H0 vs H1)
    "single_sae/merged_recon_error_avg",
    "single_sae/split_recon_error_avg",
    "single_sae/merged_recon_error_img",
    "single_sae/split_recon_error_img",
    # Two SAE
    "two_sae/img_eval_loss",
    "two_sae/txt_eval_loss",
    "two_sae/avg_eval_loss",
    "two_sae/img_mgt_shared",
    "two_sae/img_mip_shared",
    "two_sae/txt_mgt_shared",
    "two_sae/txt_mip_shared",
    # Cross-alignment / GT-free proxy
    "two_sae/cross_cos_mean",
    "two_sae/alpha_proxy_mean",
    "two_sae/dir_error_mean",
    # Two SAE multi-threshold MGT
    "two_sae/mgt_img_tau0.8",
    "two_sae/mgt_img_tau0.9",
    "two_sae/mgt_img_tau0.95",
    "two_sae/mgt_img_tau0.99",
    "two_sae/mgt_txt_tau0.8",
    "two_sae/mgt_txt_tau0.9",
    "two_sae/mgt_txt_tau0.95",
    "two_sae/mgt_txt_tau0.99",
    # Loss ratio
    "loss_ratio",
]


def _run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    alpha_values = [float(x) for x in args.alpha_sweep.split(",")]
    latent_sizes = [int(x) for x in args.latent_size_sweep.split(",")]

    sweep_results = []
    for latent_size in latent_sizes:
        for alpha in alpha_values:
            seed_results = []
            for i in range(args.num_seeds):
                seed = args.seed_base + i
                logger.info(
                    "alpha=%.2f latent_size=%d seed=%d", alpha, latent_size, seed,
                )
                result = _run_single(args, seed, alpha, latent_size)
                s = result["single_sae"]
                t = result["two_sae"]
                logger.info(
                    "  single: eval=%.6f img=%.6f txt=%.6f merge=%.3f",
                    s["eval_loss"], s["img_eval_loss"], s["txt_eval_loss"],
                    s["merge_rate"],
                )
                logger.info(
                    "  two:    avg=%.6f ratio=%.3f alpha_proxy=%.3f",
                    t["avg_eval_loss"], result["loss_ratio"],
                    t["alpha_proxy_mean"],
                )
                seed_results.append(result)

            agg = _aggregate_seed_results(seed_results, METRIC_KEYS)
            sweep_results.append({
                "alpha_target": alpha,
                "latent_size": latent_size,
                "num_seeds": args.num_seeds,
                "aggregate": agg,
                "seed_results": seed_results,
            })

    return {"sweep_param": "alpha_x_latent_size", "sweep_results": sweep_results}


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Theorem 1 experiments with diagnostics")

    # Sweep
    parser.add_argument("--alpha-sweep", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--latent-size-sweep", type=str, default="2048")

    # Data dims
    parser.add_argument("--n-shared", type=int, default=512)
    parser.add_argument("--n-image", type=int, default=256)
    parser.add_argument("--n-text", type=int, default=256)
    parser.add_argument("--representation-dim", type=int, default=768)

    # SAE config
    parser.add_argument("--latent-size", type=int, default=-1, help="Overridden by sweep")
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
    parser.add_argument("--dictionary-strategy", type=str, choices=["gradient", "random"], default="gradient")

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

    # Variants
    parser.add_argument(
        "--alternating-single-sae",
        action="store_true",
        help="Train Single SAE by alternating image-batch and text-batch grad steps "
             "instead of joint concat training (default: joint).",
    )

    # Output
    parser.add_argument("--output-root", type=str, default="outputs/synthetic_theorem1")
    parser.add_argument("--run-tag", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_name = f"unified_ns{args.n_shared}_k{args.k}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Unified Theorem 1 Experiment")
    logger.info("Output dir: %s", run_dir)

    result = _run_experiment(args)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment": "unified_theorem1",
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
            "latent_size_sweep": args.latent_size_sweep,
            "alternating_single_sae": args.alternating_single_sae,
            "run_name": run_name,
        },
        **result,
    }

    json_path = run_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Saved: %s", json_path)

    # Print summary table
    logger.info("=" * 80)
    for sr in result["sweep_results"]:
        a = sr["aggregate"]
        alpha = sr["alpha_target"]
        L = sr["latent_size"]
        logger.info(
            "alpha=%.1f L=%d | single=%.4f two=%.4f ratio=%.3f | "
            "dir_err: single_merged=%.4f single_split=%.4f two=%.4f",
            alpha, L,
            a.get("single_sae/eval_loss_mean", float("nan")),
            a.get("two_sae/avg_eval_loss_mean", float("nan")),
            a.get("loss_ratio_mean", float("nan")),
            a.get("single_sae/merged_recon_error_avg_mean", float("nan")),
            a.get("single_sae/split_recon_error_avg_mean", float("nan")),
            a.get("two_sae/dir_error_mean_mean", float("nan")),
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
