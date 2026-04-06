"""
Synthetic Theorem 1 experiments.

Validates that a single SAE cannot simultaneously achieve:
  (C1) Reconstruction  — low reconstruction error for both modalities
  (C2) Alignment       — shared features use the same latent direction
  (C3) Monosemanticity — each shared feature maps to exactly one latent

Two sub-experiments:
  Exp 1a: Mismatch sweep (α) × capacity (latent_size)
          → impossibility when Φ_S ≠ Ψ_S
  Exp 1b: Private feature sweep × capacity, Φ_S = Ψ_S
          → practical degradation even without theoretical impossibility
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

from synthetic_sae_theory_experiment import (
    _seed_everything,
    _resolve_device,
    _train_sae,
    _compute_recovery_metrics,
    _normalize_rows,
    _gt_based_shared_alignment,
)
from src.datasets.synthetic_theory_feature import SyntheticTheoryFeatureBuilder

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# New metrics: C2 (alignment) and C3 (monosemanticity)               #
# ------------------------------------------------------------------ #


def _compute_alignment_and_mono(
    w_dec: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict[str, Any]:
    """Compute C2 (alignment) and C3 (monosemanticity) for shared features.

    Args:
        w_dec: Decoder weights, shape (latent_size, m).
        phi_S: Image-side shared dictionary, shape (m, n_S).
        psi_S: Text-side shared dictionary, shape (m, n_S).
        thresholds: Cosine thresholds for C3 monosemanticity counting.

    Returns:
        Dict with aggregate and per-feature metrics.
    """
    n_S = phi_S.shape[1]

    # Normalize everything
    dec_norm = _normalize_rows(w_dec.astype(np.float64))          # (latent_size, m)
    phi_norm = _normalize_rows(phi_S.T.astype(np.float64))        # (n_S, m)
    psi_norm = _normalize_rows(psi_S.T.astype(np.float64))        # (n_S, m)

    # Cosine similarity: (latent_size, n_S)
    sim_img = np.abs(dec_norm @ phi_norm.T)
    sim_txt = np.abs(dec_norm @ psi_norm.T)

    per_feature = []
    c2_values = []

    for i in range(n_S):
        # C1: MIP (best cosine match)
        best_img_idx = int(sim_img[:, i].argmax())
        best_txt_idx = int(sim_txt[:, i].argmax())
        mip_img = float(sim_img[best_img_idx, i])
        mip_txt = float(sim_txt[best_txt_idx, i])

        # C2: cosine between best-match decoder rows
        c2_cos = float(np.abs(dec_norm[best_img_idx] @ dec_norm[best_txt_idx]))
        c2_values.append(c2_cos)

        # C3: monosemanticity — count latents above threshold, second/first ratio
        sorted_cos_img = np.sort(sim_img[:, i])[::-1]
        c3_ratio = float(sorted_cos_img[1] / max(sorted_cos_img[0], 1e-12))

        c3_counts = {}
        for tau in thresholds:
            c3_counts[f"n_match_tau{tau:.1f}"] = int((sim_img[:, i] > tau).sum())

        per_feature.append({
            "idx": i,
            "c1_mip_img": mip_img,
            "c1_mip_txt": mip_txt,
            "c2_cos": c2_cos,
            "c2_same_idx": best_img_idx == best_txt_idx,
            "c3_ratio": c3_ratio,
            **c3_counts,
        })

    c2_arr = np.array(c2_values)

    # Aggregate C3 stats per threshold
    c3_agg = {}
    for tau in thresholds:
        key = f"n_match_tau{tau:.1f}"
        counts = [f[key] for f in per_feature]
        mono_rate = float(np.mean([c == 1 for c in counts]))
        c3_agg[f"c3_mono_rate_tau{tau:.1f}"] = mono_rate
        c3_agg[f"c3_mean_n_match_tau{tau:.1f}"] = float(np.mean(counts))

    return {
        # C1 aggregates
        "c1_mip_img_mean": float(np.mean([f["c1_mip_img"] for f in per_feature])),
        "c1_mip_txt_mean": float(np.mean([f["c1_mip_txt"] for f in per_feature])),
        # C2 aggregates
        "c2_alignment_cos_mean": float(c2_arr.mean()),
        "c2_alignment_cos_std": float(c2_arr.std()),
        "c2_alignment_cos_min": float(c2_arr.min()),
        "c2_same_idx_rate": float(np.mean([f["c2_same_idx"] for f in per_feature])),
        # C3 aggregates
        "c3_ratio_mean": float(np.mean([f["c3_ratio"] for f in per_feature])),
        "c3_ratio_std": float(np.std([f["c3_ratio"] for f in per_feature])),
        **c3_agg,
        # Per-feature details
        "per_feature": per_feature,
    }


# ------------------------------------------------------------------ #
# Experiment runners                                                   #
# ------------------------------------------------------------------ #


def _run_exp1a_single(
    args: argparse.Namespace,
    seed: int,
    alpha: float,
    latent_size: int,
) -> dict[str, Any]:
    """Single run of Exp 1a: one alpha, one latent_size, one seed."""
    device = _resolve_device(args.device)

    # Build data
    if alpha >= 0.99:
        shared_mode = "identical"
    else:
        shared_mode = "range"

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

    # Train single SAE on concatenated data
    train_concat = torch.cat([train_img, train_txt], dim=0)
    eval_concat = torch.cat([eval_img, eval_txt], dim=0)

    # Override latent_size for this run
    args_copy = argparse.Namespace(**vars(args))
    args_copy.latent_size = latent_size

    model, train_loss, eval_loss = _train_sae(
        train_concat, eval_concat, args_copy, seed, device,
    )

    # Analyze decoder weights
    w_dec = model.W_dec.detach().cpu().numpy()
    analysis = _compute_alignment_and_mono(
        w_dec, builder.phi_S, builder.psi_S,
    )

    # Standard recovery metrics
    gt_shared_img = builder.phi_S
    gt_shared_txt = builder.psi_S
    recovery_img = _compute_recovery_metrics(w_dec, gt_shared_img, args.gt_recovery_threshold)
    recovery_txt = _compute_recovery_metrics(w_dec, gt_shared_txt, args.gt_recovery_threshold)

    return {
        "seed": seed,
        "alpha_target": alpha,
        "latent_size": latent_size,
        "alpha_actual_mean": builder.mean_shared_cosine_similarity,
        "alpha_actual_min": builder.min_shared_cosine_similarity,
        "alpha_actual_max": builder.max_shared_cosine_similarity,
        "alpha_actual_std": builder.std_shared_cosine_similarity,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "mgt_shared_img": recovery_img["gt_recovery"],
        "mip_shared_img": recovery_img["mip"],
        "mgt_shared_txt": recovery_txt["gt_recovery"],
        "mip_shared_txt": recovery_txt["mip"],
        **{k: v for k, v in analysis.items() if k != "per_feature"},
        "per_feature": analysis["per_feature"],
    }


def _run_exp1b_single(
    args: argparse.Namespace,
    seed: int,
    n_private: int,
    latent_size: int,
) -> dict[str, Any]:
    """Single run of Exp 1b: one n_private, one latent_size, one seed."""
    device = _resolve_device(args.device)

    builder = SyntheticTheoryFeatureBuilder(
        n_image=n_private,
        n_shared=args.n_shared,
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

    args_copy = argparse.Namespace(**vars(args))
    args_copy.latent_size = latent_size
    args_copy.n_private = n_private

    # (a) Single SAE on concatenated data
    train_concat = torch.cat([train_img, train_txt], dim=0)
    eval_concat = torch.cat([eval_img, eval_txt], dim=0)
    single_model, single_train, single_eval = _train_sae(
        train_concat, eval_concat, args_copy, seed, device,
    )
    w_dec_single = single_model.W_dec.detach().cpu().numpy()
    single_analysis = _compute_alignment_and_mono(
        w_dec_single, builder.phi_S, builder.psi_S,
    )

    # (b) Two independent SAEs
    img_model, img_train, img_eval = _train_sae(
        train_img, eval_img, args_copy, seed, device,
    )
    txt_model, txt_train, txt_eval = _train_sae(
        train_txt, eval_txt, args_copy, seed + 10000, device,
    )
    w_dec_img = img_model.W_dec.detach().cpu().numpy()
    w_dec_txt = txt_model.W_dec.detach().cpu().numpy()

    # Two SAE: cross-SAE alignment via GT
    two_sae_align = _gt_based_shared_alignment(
        w_dec_img, w_dec_txt, builder.phi_S,
    )

    # Two SAE: per-SAE monosemanticity
    img_mono = _compute_alignment_and_mono(
        w_dec_img, builder.phi_S, builder.phi_S,  # phi_S = psi_S in identical mode
    )
    txt_mono = _compute_alignment_and_mono(
        w_dec_txt, builder.psi_S, builder.psi_S,
    )

    def _strip_per_feature(d: dict) -> dict:
        return {k: v for k, v in d.items() if k != "per_feature"}

    return {
        "seed": seed,
        "n_private": n_private,
        "latent_size": latent_size,
        "single_sae": {
            "train_loss": single_train,
            "eval_loss": single_eval,
            **_strip_per_feature(single_analysis),
        },
        "two_sae": {
            "img_train_loss": img_train,
            "img_eval_loss": img_eval,
            "txt_train_loss": txt_train,
            "txt_eval_loss": txt_eval,
            "c2_cross_cos_mean": two_sae_align["gt_shared_cos_mean"],
            "c2_cross_cos_std": two_sae_align["gt_shared_cos_std"],
            "img_c3_mono_rate_tau0.5": img_mono.get("c3_mono_rate_tau0.5", float("nan")),
            "txt_c3_mono_rate_tau0.5": txt_mono.get("c3_mono_rate_tau0.5", float("nan")),
            "img_c1_mip_mean": img_mono["c1_mip_img_mean"],
            "txt_c1_mip_mean": txt_mono["c1_mip_img_mean"],
        },
    }


# ------------------------------------------------------------------ #
# Main experiment dispatch                                             #
# ------------------------------------------------------------------ #


def _aggregate_seed_results(seed_results: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, float]:
    """Aggregate metrics across seeds: compute mean and std."""
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


def _extract_metric(result: dict[str, Any], key: str) -> float | None:
    """Extract a metric from a result dict, supporting nested keys like 'single_sae/eval_loss'."""
    # First try direct key lookup (handles keys with dots like 'c3_mono_rate_tau0.5')
    if key in result:
        v = result[key]
        return float(v) if v is not None else None
    # Then try slash-separated nested access
    parts = key.split("/")
    obj = result
    for p in parts:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return None
    return float(obj) if obj is not None else None


def _run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    exp = args.experiment

    if exp == "1a":
        alpha_values = [float(x) for x in args.alpha_sweep.split(",")]
        latent_sizes = [int(x) for x in args.latent_size_sweep.split(",")]
        metric_keys = [
            "eval_loss", "c2_alignment_cos_mean", "c2_same_idx_rate",
            "c3_mono_rate_tau0.5", "c3_ratio_mean",
            "c1_mip_img_mean", "c1_mip_txt_mean",
            "mip_shared_img", "mgt_shared_img",
        ]

        sweep_results = []
        for latent_size in latent_sizes:
            for alpha in alpha_values:
                seed_results = []
                for i in range(args.num_seeds):
                    seed = args.seed_base + i
                    logger.info(
                        "[exp1a] alpha=%.2f latent_size=%d seed=%d",
                        alpha, latent_size, seed,
                    )
                    result = _run_exp1a_single(args, seed, alpha, latent_size)
                    c2 = result["c2_alignment_cos_mean"]
                    c3 = result.get("c3_mono_rate_tau0.5", float("nan"))
                    logger.info(
                        "  → eval=%.6f C2=%.4f C3_mono=%.4f",
                        result["eval_loss"], c2, c3,
                    )
                    # Strip per_feature before storing
                    result.pop("per_feature", None)
                    seed_results.append(result)

                agg = _aggregate_seed_results(seed_results, metric_keys)
                sweep_results.append({
                    "alpha_target": alpha,
                    "latent_size": latent_size,
                    "num_seeds": args.num_seeds,
                    "aggregate": agg,
                    "seed_results": seed_results,
                })

        return {"sweep_param": "alpha_x_latent_size", "sweep_results": sweep_results}

    elif exp == "1b":
        n_private_values = [int(x) for x in args.n_private_sweep.split(",")]
        latent_sizes = [int(x) for x in args.latent_size_sweep.split(",")]
        metric_keys_single = [
            "single_sae/eval_loss", "single_sae/c2_alignment_cos_mean",
            "single_sae/c3_mono_rate_tau0.5", "single_sae/c3_ratio_mean",
            "single_sae/c1_mip_img_mean",
        ]
        metric_keys_two = [
            "two_sae/img_eval_loss", "two_sae/txt_eval_loss",
            "two_sae/c2_cross_cos_mean",
            "two_sae/img_c1_mip_mean", "two_sae/txt_c1_mip_mean",
        ]

        sweep_results = []
        for latent_size in latent_sizes:
            for n_priv in n_private_values:
                seed_results = []
                for i in range(args.num_seeds):
                    seed = args.seed_base + i
                    logger.info(
                        "[exp1b] n_private=%d latent_size=%d seed=%d",
                        n_priv, latent_size, seed,
                    )
                    result = _run_exp1b_single(args, seed, n_priv, latent_size)
                    s = result["single_sae"]
                    t = result["two_sae"]
                    logger.info(
                        "  → single eval=%.6f C2=%.4f | two C2=%.4f",
                        s["eval_loss"],
                        s["c2_alignment_cos_mean"],
                        t["c2_cross_cos_mean"],
                    )
                    seed_results.append(result)

                agg = _aggregate_seed_results(
                    seed_results, metric_keys_single + metric_keys_two,
                )
                sweep_results.append({
                    "n_private": n_priv,
                    "latent_size": latent_size,
                    "num_seeds": args.num_seeds,
                    "aggregate": agg,
                    "seed_results": seed_results,
                })

        return {"sweep_param": "n_private_x_latent_size", "sweep_results": sweep_results}

    else:
        raise ValueError(f"Unknown experiment: {exp}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Theorem 1 synthetic experiments")

    parser.add_argument("--experiment", type=str, choices=["1a", "1b"], required=True)

    # Sweep values
    parser.add_argument("--alpha-sweep", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--n-private-sweep", type=str, default="0,128,256,512")
    parser.add_argument("--latent-size-sweep", type=str, default="2048,4096,8192")

    # Data dimensions
    parser.add_argument("--n-shared", type=int, default=512)
    parser.add_argument("--n-image", type=int, default=256)
    parser.add_argument("--n-text", type=int, default=256)
    parser.add_argument("--representation-dim", type=int, default=768)

    # SAE config
    parser.add_argument("--latent-size", type=int, default=-1, help="Overridden by sweep")
    parser.add_argument("--k", type=int, default=64)
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
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    # Output
    parser.add_argument("--output-root", type=str, default="outputs/synthetic_theorem1")
    parser.add_argument("--run-tag", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_name = f"exp{args.experiment}_ns{args.n_shared}_m{args.representation_dim}_k{args.k}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
            "run_name": run_name,
        },
        **result,
    }

    # Strip per_feature from saved JSON to keep file manageable
    for r in output.get("sweep_results", []):
        r.pop("per_feature", None)
        for sr in r.get("seed_results", []):
            sr.pop("per_feature", None)

    json_path = run_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved: %s", json_path)


if __name__ == "__main__":
    main()
