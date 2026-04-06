"""
Synthetic SAE theory experiments.

Validates the paper's theoretical claims about failure modes of naive SAE
approaches in multimodal settings:

    Exp 1(i):  Single SAE -- shared-private interference
    Exp 1(ii): Single SAE -- generative mapping mismatch
    Exp 2(i):  Two SAEs   -- shared-private entanglement
    Exp 2(ii): Two SAEs   -- latent non-identifiability
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.synthetic_theory_feature import SyntheticTheoryFeatureBuilder
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _compute_recovery_metrics(
    learned_vectors: np.ndarray,
    gt_matrix: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """GT Recovery and MIP. learned_vectors: (k, d), gt_matrix: (d, n_gt)."""
    if gt_matrix.shape[1] == 0:
        return {"gt_recovery": float("nan"), "mip": float("nan")}

    learned_norm = _normalize_rows(learned_vectors.astype(np.float64))
    gt_norm = _normalize_rows(gt_matrix.T.astype(np.float64))
    sim = np.abs(learned_norm @ gt_norm.T)
    best = sim.max(axis=0)
    return {
        "gt_recovery": float((best > threshold).mean()),
        "mip": float(best.mean()),
    }


def _gt_based_shared_alignment(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    gt_shared: np.ndarray,
) -> dict[str, float]:
    """Measure alignment of two SAEs via GT shared features.

    For each GT shared atom i:
      - Find best matching latent in image SAE
      - Find best matching latent in text SAE
      - Compute cosine between those two latents

    Args:
        w_dec_img: (latent_size, m) image SAE decoder rows.
        w_dec_txt: (latent_size, m) text SAE decoder rows.
        gt_shared: (m, n_S) shared GT dictionary columns.

    Returns:
        Dict with gt_shared_cos_mean, gt_shared_cos_std.
    """
    if gt_shared.shape[1] == 0:
        return {"gt_shared_cos_mean": float("nan"), "gt_shared_cos_std": float("nan")}

    img_norm = _normalize_rows(w_dec_img.astype(np.float64))
    txt_norm = _normalize_rows(w_dec_txt.astype(np.float64))
    gt_norm = _normalize_rows(gt_shared.T.astype(np.float64))  # (n_S, m)

    # For each GT atom, find best matching latent in each SAE
    sim_img = np.abs(img_norm @ gt_norm.T)  # (latent_size, n_S)
    sim_txt = np.abs(txt_norm @ gt_norm.T)  # (latent_size, n_S)
    best_img = sim_img.argmax(axis=0)  # (n_S,)
    best_txt = sim_txt.argmax(axis=0)  # (n_S,)

    # Cosine between matched latents
    cos_values = np.array([
        np.abs(img_norm[best_img[i]] @ txt_norm[best_txt[i]])
        for i in range(gt_shared.shape[1])
    ])

    return {
        "gt_shared_cos_mean": float(cos_values.mean()),
        "gt_shared_cos_std": float(cos_values.std()),
    }


# ------------------------------------------------------------------ #
# Training functions                                                   #
# ------------------------------------------------------------------ #


def _train_sae(
    train_data: torch.Tensor,
    eval_data: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[TopKSAE, float, float]:
    """Train a single TopKSAE and return (model, train_loss, eval_loss)."""
    _seed_everything(seed)

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        TensorDataset(eval_data),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    latent_size = args.latent_size if args.latent_size > 0 else (args.n_shared + getattr(args, "n_private", 0) * 2)
    cfg = TopKSAEConfig(
        hidden_size=args.representation_dim,
        latent_size=latent_size,
        k=args.k,
        normalize_decoder=True,
    )
    model = TopKSAE(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_sum = 0.0
    train_steps = 0

    for epoch in range(args.num_epochs):
        for step, (batch,) in enumerate(train_loader):
            batch = batch.to(device=device, dtype=torch.float32)
            hs = batch.unsqueeze(1)

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

            train_loss_sum += float(loss.detach().item())
            train_steps += 1

            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                logger.info(
                    "  seed=%d epoch=%d step=%d/%d recon=%.6f",
                    seed, epoch + 1, step + 1, len(train_loader),
                    train_loss_sum / train_steps,
                )

    model.eval()
    eval_loss_sum = 0.0
    eval_steps = 0

    with torch.no_grad():
        for (batch,) in eval_loader:
            batch = batch.to(device=device, dtype=torch.float32)
            hs = batch.unsqueeze(1)
            out = model(hidden_states=hs)
            eval_loss_sum += float(out.recon_loss.item())
            eval_steps += 1

    train_loss = train_loss_sum / max(train_steps, 1)
    eval_loss = eval_loss_sum / max(eval_steps, 1)
    return model, train_loss, eval_loss


# ------------------------------------------------------------------ #
# Result dataclass                                                     #
# ------------------------------------------------------------------ #


@dataclass
class SeedResult:
    seed: int
    # Single SAE metrics
    single_train_loss: float = float("nan")
    single_eval_loss: float = float("nan")
    single_mgt_full: float = float("nan")
    single_mip_full: float = float("nan")
    single_mgt_shared: float = float("nan")
    single_mip_shared: float = float("nan")
    # Two SAE metrics
    img_train_loss: float = float("nan")
    img_eval_loss: float = float("nan")
    txt_train_loss: float = float("nan")
    txt_eval_loss: float = float("nan")
    img_mgt_full: float = float("nan")
    img_mip_full: float = float("nan")
    txt_mgt_full: float = float("nan")
    txt_mip_full: float = float("nan")
    # Post-hoc alignment (Exp 2)
    posthoc_mean_cosine_all: float = float("nan")
    posthoc_mean_cosine_shared: float = float("nan")
    # Data gen info
    alpha_max: float = float("nan")
    alpha_min: float = float("nan")
    alpha_mean: float = float("nan")
    alpha_std: float = float("nan")
    actual_cl: float = float("nan")


def _aggregate(results: list[SeedResult]) -> dict[str, float]:
    if not results:
        return {}
    keys = [k for k in results[0].__dict__ if k != "seed"]
    summary: dict[str, float] = {}
    for k in keys:
        vals = np.array([getattr(r, k) for r in results], dtype=np.float64)
        if np.all(np.isnan(vals)):
            summary[f"{k}_mean"] = float("nan")
            summary[f"{k}_std"] = float("nan")
        else:
            summary[f"{k}_mean"] = float(np.nanmean(vals))
            summary[f"{k}_std"] = float(np.nanstd(vals, ddof=0))
    return summary


# ------------------------------------------------------------------ #
# Experiment runners                                                   #
# ------------------------------------------------------------------ #


def _run_exp1i_seed(args: argparse.Namespace, seed: int, n_private: int) -> SeedResult:
    """Exp 1(i): Single SAE vs two SAEs, sweep n_private."""
    device = _resolve_device(args.device)
    # total feature_dim = n_shared; n_shared_actual = n_shared - 2*n_private
    n_shared_actual = args.n_shared - 2 * n_private
    if n_shared_actual <= 0:
        raise ValueError(
            f"n_private={n_private} too large: 2*n_private={2*n_private} >= n_shared={args.n_shared}"
        )
    logger.info("[exp1i] seed=%d n_private=%d n_shared_actual=%d", seed, n_private, n_shared_actual)

    builder = SyntheticTheoryFeatureBuilder(
        n_image=n_private,
        n_shared=n_shared_actual,
        n_text=n_private,
        representation_dim=args.representation_dim,
        sparsity=args.sparsity,
        min_active=args.min_active,
        cmin=args.cmin,
        beta=args.beta,
        max_interference=args.max_interference,
        strategy=args.dictionary_strategy,
        shared_mode="identical",
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

    # (a) Single SAE on concatenated data
    train_concat = torch.cat([train_img, train_txt], dim=0)
    eval_concat = torch.cat([eval_img, eval_txt], dim=0)
    single_model, single_train, single_eval = _train_sae(
        train_concat, eval_concat, args, seed, device,
    )
    w_dec_single = single_model.W_dec.detach().cpu().numpy()
    gt_full = np.concatenate([
        builder.phi_full,
        builder.psi_full,
    ], axis=1)  # use both Phi and Psi columns as GT
    gt_shared = builder.phi_S  # Phi_S = Psi_S in identical mode
    single_full_m = _compute_recovery_metrics(w_dec_single, gt_full, args.gt_recovery_threshold)
    single_shared_m = _compute_recovery_metrics(w_dec_single, gt_shared, args.gt_recovery_threshold)

    # (b) Two independent SAEs
    img_model, img_train, img_eval = _train_sae(train_img, eval_img, args, seed, device)
    txt_model, txt_train, txt_eval = _train_sae(train_txt, eval_txt, args, seed + 10000, device)
    w_dec_img = img_model.W_dec.detach().cpu().numpy()
    w_dec_txt = txt_model.W_dec.detach().cpu().numpy()
    img_full_m = _compute_recovery_metrics(w_dec_img, builder.phi_full, args.gt_recovery_threshold)
    txt_full_m = _compute_recovery_metrics(w_dec_txt, builder.psi_full, args.gt_recovery_threshold)

    return SeedResult(
        seed=seed,
        single_train_loss=single_train,
        single_eval_loss=single_eval,
        single_mgt_full=single_full_m["gt_recovery"],
        single_mip_full=single_full_m["mip"],
        single_mgt_shared=single_shared_m["gt_recovery"],
        single_mip_shared=single_shared_m["mip"],
        img_train_loss=img_train,
        img_eval_loss=img_eval,
        txt_train_loss=txt_train,
        txt_eval_loss=txt_eval,
        img_mgt_full=img_full_m["gt_recovery"],
        img_mip_full=img_full_m["mip"],
        txt_mgt_full=txt_full_m["gt_recovery"],
        txt_mip_full=txt_full_m["mip"],
        alpha_max=builder.max_shared_cosine_similarity,
        alpha_min=builder.min_shared_cosine_similarity,
        alpha_mean=builder.mean_shared_cosine_similarity,
        alpha_std=builder.std_shared_cosine_similarity,
    )


def _run_exp1ii_seed(args: argparse.Namespace, seed: int, alpha_lo: float, alpha_hi: float) -> SeedResult:
    """Exp 1(ii): Single SAE, sweep alpha range (no private features)."""
    device = _resolve_device(args.device)
    logger.info("[exp1ii] seed=%d alpha_range=[%.4f, %.4f]", seed, alpha_lo, alpha_hi)

    # Use "identical" for 1.0-1.0, "range" otherwise
    if alpha_lo >= 1.0 and alpha_hi >= 1.0:
        shared_mode = "identical"
    else:
        shared_mode = "range"

    builder = SyntheticTheoryFeatureBuilder(
        n_image=0,
        n_shared=args.n_shared,
        n_text=0,
        representation_dim=args.representation_dim,
        sparsity=args.sparsity,
        min_active=args.min_active,
        cmin=args.cmin,
        beta=args.beta,
        max_interference=args.max_interference,
        strategy=args.dictionary_strategy,
        shared_mode=shared_mode,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
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

    train_concat = torch.cat([train_img, train_txt], dim=0)
    eval_concat = torch.cat([eval_img, eval_txt], dim=0)
    single_model, single_train, single_eval = _train_sae(
        train_concat, eval_concat, args, seed, device,
    )
    w_dec = single_model.W_dec.detach().cpu().numpy()
    gt_shared_img = builder.phi_S
    gt_shared_txt = builder.psi_S
    gt_full = np.concatenate([gt_shared_img, gt_shared_txt], axis=1)
    full_m = _compute_recovery_metrics(w_dec, gt_full, args.gt_recovery_threshold)
    shared_m = _compute_recovery_metrics(w_dec, gt_shared_img, args.gt_recovery_threshold)

    return SeedResult(
        seed=seed,
        single_train_loss=single_train,
        single_eval_loss=single_eval,
        single_mgt_full=full_m["gt_recovery"],
        single_mip_full=full_m["mip"],
        single_mgt_shared=shared_m["gt_recovery"],
        single_mip_shared=shared_m["mip"],
        alpha_max=builder.max_shared_cosine_similarity,
        alpha_min=builder.min_shared_cosine_similarity,
        alpha_mean=builder.mean_shared_cosine_similarity,
        alpha_std=builder.std_shared_cosine_similarity,
        actual_cl=builder.actual_contrastive_loss,
    )


def _run_exp2i_seed(args: argparse.Namespace, seed: int, n_private: int) -> SeedResult:
    """Exp 2(i): Two independent SAEs, sweep n_private, post-hoc matching."""
    device = _resolve_device(args.device)
    n_shared_actual = args.n_shared - 2 * n_private
    if n_shared_actual <= 0:
        raise ValueError(
            f"n_private={n_private} too large: 2*n_private={2*n_private} >= n_shared={args.n_shared}"
        )
    logger.info("[exp2i] seed=%d n_private=%d n_shared_actual=%d", seed, n_private, n_shared_actual)

    builder = SyntheticTheoryFeatureBuilder(
        n_image=n_private,
        n_shared=n_shared_actual,
        n_text=n_private,
        representation_dim=args.representation_dim,
        sparsity=args.sparsity,
        min_active=args.min_active,
        cmin=args.cmin,
        beta=args.beta,
        max_interference=args.max_interference,
        strategy=args.dictionary_strategy,
        shared_mode="identical",
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

    img_model, img_train, img_eval = _train_sae(train_img, eval_img, args, seed, device)
    txt_model, txt_train, txt_eval = _train_sae(train_txt, eval_txt, args, seed + 10000, device)

    w_dec_img = img_model.W_dec.detach().cpu().numpy()
    w_dec_txt = txt_model.W_dec.detach().cpu().numpy()

    # GT-based shared alignment
    gt_shared = builder.phi_S  # identical mode: Phi_S = Psi_S
    gt_align = _gt_based_shared_alignment(w_dec_img, w_dec_txt, gt_shared)

    img_full_m = _compute_recovery_metrics(w_dec_img, builder.phi_full, args.gt_recovery_threshold)
    txt_full_m = _compute_recovery_metrics(w_dec_txt, builder.psi_full, args.gt_recovery_threshold)

    return SeedResult(
        seed=seed,
        img_train_loss=img_train,
        img_eval_loss=img_eval,
        txt_train_loss=txt_train,
        txt_eval_loss=txt_eval,
        img_mgt_full=img_full_m["gt_recovery"],
        img_mip_full=img_full_m["mip"],
        txt_mgt_full=txt_full_m["gt_recovery"],
        txt_mip_full=txt_full_m["mip"],
        posthoc_mean_cosine_all=gt_align["gt_shared_cos_mean"],
        posthoc_mean_cosine_shared=gt_align["gt_shared_cos_mean"],
        alpha_max=builder.max_shared_cosine_similarity,
        alpha_min=builder.min_shared_cosine_similarity,
        alpha_mean=builder.mean_shared_cosine_similarity,
        alpha_std=builder.std_shared_cosine_similarity,
    )


def _run_exp2ii_seed(args: argparse.Namespace, seed: int) -> SeedResult:
    """Exp 2(ii): Two independent SAEs, no private, different init, post-hoc matching."""
    device = _resolve_device(args.device)
    logger.info("[exp2ii] seed=%d", seed)

    builder = SyntheticTheoryFeatureBuilder(
        n_image=0,
        n_shared=args.n_shared,
        n_text=0,
        representation_dim=args.representation_dim,
        sparsity=args.sparsity,
        min_active=args.min_active,
        cmin=args.cmin,
        beta=args.beta,
        max_interference=args.max_interference,
        strategy=args.dictionary_strategy,
        shared_mode="identical",
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

    # Two SAEs with different seeds (different random init)
    img_model, img_train, img_eval = _train_sae(train_img, eval_img, args, seed, device)
    txt_model, txt_train, txt_eval = _train_sae(train_txt, eval_txt, args, seed + 10000, device)

    w_dec_img = img_model.W_dec.detach().cpu().numpy()
    w_dec_txt = txt_model.W_dec.detach().cpu().numpy()

    gt_shared = builder.phi_S
    gt_align = _gt_based_shared_alignment(w_dec_img, w_dec_txt, gt_shared)

    img_full_m = _compute_recovery_metrics(w_dec_img, builder.phi_full, args.gt_recovery_threshold)
    txt_full_m = _compute_recovery_metrics(w_dec_txt, builder.psi_full, args.gt_recovery_threshold)

    return SeedResult(
        seed=seed,
        img_train_loss=img_train,
        img_eval_loss=img_eval,
        txt_train_loss=txt_train,
        txt_eval_loss=txt_eval,
        img_mgt_full=img_full_m["gt_recovery"],
        img_mip_full=img_full_m["mip"],
        txt_mgt_full=txt_full_m["gt_recovery"],
        txt_mip_full=txt_full_m["mip"],
        posthoc_mean_cosine_all=gt_align["gt_shared_cos_mean"],
        posthoc_mean_cosine_shared=gt_align["gt_shared_cos_mean"],
        alpha_max=builder.max_shared_cosine_similarity,
        alpha_min=builder.min_shared_cosine_similarity,
        alpha_mean=builder.mean_shared_cosine_similarity,
        alpha_std=builder.std_shared_cosine_similarity,
    )


# ------------------------------------------------------------------ #
# Main experiment dispatch                                             #
# ------------------------------------------------------------------ #


def _run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    exp = args.experiment

    if exp == "1i":
        n_private_values = [int(x) for x in args.n_private_values.split(",")]
        sweep_results = []
        for n_priv in n_private_values:
            # Temporarily set n_private for latent_size calculation
            args.n_private = n_priv
            seed_results = [
                _run_exp1i_seed(args, args.seed_base + i, n_priv)
                for i in range(args.num_seeds)
            ]
            agg = _aggregate(seed_results)
            sweep_results.append({
                "n_private": n_priv,
                "seed_results": [asdict(r) for r in seed_results],
                "aggregate": agg,
            })
            logger.info(
                "[exp1i] n_private=%d | single_eval=%.6f two_eval=%.6f",
                n_priv,
                agg.get("single_eval_loss_mean", float("nan")),
                (agg.get("img_eval_loss_mean", 0) + agg.get("txt_eval_loss_mean", 0)) / 2,
            )
        return {"sweep_param": "n_private", "sweep_results": sweep_results}

    elif exp == "1ii":
        # Parse alpha ranges: "1.0-1.0,0.7-0.9,0.5-0.7,0.3-0.5,0.1-0.3"
        alpha_ranges = []
        for entry in args.alpha_values.split(","):
            if "-" in entry and not entry.startswith("-"):
                lo, hi = entry.split("-", 1)
                alpha_ranges.append((float(lo), float(hi)))
            else:
                v = float(entry)
                alpha_ranges.append((v, v))

        sweep_results = []
        for alpha_lo, alpha_hi in alpha_ranges:
            args.n_private = 0
            seed_results = [
                _run_exp1ii_seed(args, args.seed_base + i, alpha_lo, alpha_hi)
                for i in range(args.num_seeds)
            ]
            agg = _aggregate(seed_results)
            sweep_results.append({
                "alpha_range": [alpha_lo, alpha_hi],
                "seed_results": [asdict(r) for r in seed_results],
                "aggregate": agg,
            })
            logger.info(
                "[exp1ii] range=[%.2f,%.2f] | eval=%.6f alpha_mean=%.4f alpha_min=%.4f alpha_max=%.4f",
                alpha_lo, alpha_hi,
                agg.get("single_eval_loss_mean", float("nan")),
                agg.get("alpha_mean_mean", float("nan")),
                agg.get("alpha_min_mean", float("nan")),
                agg.get("alpha_max_mean", float("nan")),
            )
        return {"sweep_param": "alpha_range", "sweep_results": sweep_results}

    elif exp == "2i":
        n_private_values = [int(x) for x in args.n_private_values.split(",")]
        sweep_results = []
        for n_priv in n_private_values:
            args.n_private = n_priv
            seed_results = [
                _run_exp2i_seed(args, args.seed_base + i, n_priv)
                for i in range(args.num_seeds)
            ]
            agg = _aggregate(seed_results)
            sweep_results.append({
                "n_private": n_priv,
                "seed_results": [asdict(r) for r in seed_results],
                "aggregate": agg,
            })
            logger.info(
                "[exp2i] n_private=%d | posthoc_cos=%.4f",
                n_priv,
                agg.get("posthoc_mean_cosine_all_mean", float("nan")),
            )
        return {"sweep_param": "n_private", "sweep_results": sweep_results}

    elif exp == "2ii":
        args.n_private = 0
        seed_results = [
            _run_exp2ii_seed(args, args.seed_base + i)
            for i in range(args.num_seeds)
        ]
        agg = _aggregate(seed_results)
        return {
            "sweep_param": "seed",
            "sweep_results": [{
                "seed_results": [asdict(r) for r in seed_results],
                "aggregate": agg,
            }],
        }

    else:
        raise ValueError(f"Unknown experiment: {exp}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic SAE theory experiments")

    parser.add_argument("--experiment", type=str, choices=["1i", "1ii", "2i", "2ii"], required=True)

    # Sweep values
    parser.add_argument("--n-private-values", type=str, default="0,8,16,32,64,128")
    parser.add_argument("--alpha-values", type=str, default="1.0-1.0,0.7-0.9,0.5-0.7,0.3-0.5,0.1-0.3")

    # Data dimensions
    parser.add_argument("--n-shared", type=int, default=64)
    parser.add_argument("--representation-dim", type=int, default=768)

    # SAE config
    parser.add_argument("--latent-size", type=int, default=-1, help="-1 = n_shared + n_private")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--gt-recovery-threshold", type=float, default=0.8)

    # Data generation
    parser.add_argument("--num-train", type=int, default=50_000)
    parser.add_argument("--num-eval", type=int, default=10_000)
    parser.add_argument("--sparsity", type=float, default=0.999)
    parser.add_argument("--min-active", type=int, default=1)
    parser.add_argument("--cmin", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-interference", type=float, default=0.3)
    parser.add_argument("--dictionary-strategy", type=str, choices=["gradient", "random"], default="gradient")

    # Alpha calibration (Exp 1ii)
    parser.add_argument("--calibration-lr", type=float, default=0.005)
    parser.add_argument("--calibration-max-iters", type=int, default=2000)
    parser.add_argument("--calibration-tol", type=float, default=0.005)

    # Training
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    # Output
    parser.add_argument("--output-root", type=str, default="outputs/synthetic_theory")
    parser.add_argument("--run-tag", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_name = f"exp{args.experiment}_ns{args.n_shared}_rep{args.representation_dim}_k{args.k}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Experiment: %s", args.experiment)
    logger.info("Output dir: %s", run_dir)

    result = _run_experiment(args)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment": args.experiment,
            "n_shared": args.n_shared,
            "representation_dim": args.representation_dim,
            "latent_size": args.latent_size,
            "k": args.k,
            "num_epochs": args.num_epochs,
            "num_seeds": args.num_seeds,
            "seed_base": args.seed_base,
            "sparsity": args.sparsity,
            "max_interference": args.max_interference,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "run_name": run_name,
        },
        **result,
    }

    json_path = run_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved: %s", json_path)


if __name__ == "__main__":
    main()
