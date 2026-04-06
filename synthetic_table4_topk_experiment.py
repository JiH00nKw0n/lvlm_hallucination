"""
Synthetic unimodal TopK SAE sweep over feature count and TopK sparsity.

This experiment keeps the current synthetic feature/data generation family,
but narrows the scope to the unimodal TopK setting:

    - feature_dim in {800, 1000, 1200, 1400}
    - k in {32, 64, 128}
    - representation_dim = 768
    - latent_size = 16384
    - sparsity = 0.99
    - min_active = 1
    - gt_recovery_threshold = 0.9
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.synthetic_feature import SyntheticFeatureDatasetBuilder
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE

logger = logging.getLogger(__name__)


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
    """
    Compute GT Recovery and Maximum Inner Product (MIP).

    learned_vectors: (n_learned, d)
    gt_matrix: (d, n_gt) where columns are GT features
    """
    if gt_matrix.shape[1] == 0:
        return {
            "gt_recovery": float("nan"),
            "mip": float("nan"),
            "num_gt_features": 0,
            "num_learned_features": int(learned_vectors.shape[0]),
        }

    learned_norm = _normalize_rows(learned_vectors.astype(np.float64))
    gt_norm = _normalize_rows(gt_matrix.T.astype(np.float64))

    sim = np.abs(learned_norm @ gt_norm.T)
    best = sim.max(axis=0)

    return {
        "gt_recovery": float((best > threshold).mean()),
        "mip": float(best.mean()),
        "num_gt_features": int(gt_matrix.shape[1]),
        "num_learned_features": int(learned_vectors.shape[0]),
    }


def _is_non_increasing(values: list[float], atol: float = 1e-12) -> bool:
    return all(curr <= prev + atol for prev, curr in zip(values, values[1:]))


def summarize_trends(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """
    Summarize per-k monotonicity across feature counts.

    Expects rows to contain numeric feature_dim, k, mgt_full_mean, mip_full_mean.
    """
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), []).append(row)

    summary: dict[int, dict[str, Any]] = {}
    for k, k_rows in by_k.items():
        ordered = sorted(k_rows, key=lambda row: int(row["feature_dim"]))
        mgt_values = [float(row["mgt_full_mean"]) for row in ordered]
        mip_values = [float(row["mip_full_mean"]) for row in ordered]
        summary[k] = {
            "feature_dims": [int(row["feature_dim"]) for row in ordered],
            "mgt_non_increasing": _is_non_increasing(mgt_values),
            "mip_non_increasing": _is_non_increasing(mip_values),
            "mgt_values": mgt_values,
            "mip_values": mip_values,
        }
    return summary


@dataclass
class SeedRunResult:
    seed: int
    train_recon_loss: float
    eval_recon_loss: float
    mgt_full: float
    mip_full: float


def _run_one_seed(
    args: argparse.Namespace,
    seed: int,
) -> SeedRunResult:
    _seed_everything(seed)
    device = _resolve_device(args.device)
    logger.info("seed=%d device=%s feature_dim=%d k=%d", seed, device, args.feature_dim, args.k)

    builder = SyntheticFeatureDatasetBuilder(
        feature_latent_dim=args.feature_dim,
        representation_dim=args.representation_dim,
        num_train=args.num_train_pairs,
        num_val=args.num_eval_pairs,
        num_test=args.num_eval_pairs,
        sparsity=args.sparsity,
        min_active=args.min_active,
        max_interference=args.max_interference,
        strategy=args.dictionary_strategy,
        cmin=args.cmin,
        beta=args.beta,
        seed=seed,
        return_ground_truth=False,
        verbose=False,
    )
    ds = builder.build_dataset()
    train_rep = torch.tensor(ds["train"]["representation"], dtype=torch.float32)
    eval_rep = torch.tensor(ds["val"]["representation"], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_rep),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        TensorDataset(eval_rep),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
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
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
                    "seed=%d epoch=%d step=%d/%d recon=%.6f",
                    seed,
                    epoch + 1,
                    step + 1,
                    len(train_loader),
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

    w_dec = model.W_dec.detach().cpu().numpy()
    gt_wp = builder.wp
    assert gt_wp is not None, "Wp should exist after build_dataset()."

    metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=gt_wp,
        threshold=args.gt_recovery_threshold,
    )

    return SeedRunResult(
        seed=seed,
        train_recon_loss=train_loss_sum / max(train_steps, 1),
        eval_recon_loss=eval_loss_sum / max(eval_steps, 1),
        mgt_full=metrics["gt_recovery"],
        mip_full=metrics["mip"],
    )


def _aggregate_seed_results(seed_results: list[SeedRunResult]) -> dict[str, float]:
    metric_names = [
        "train_recon_loss",
        "eval_recon_loss",
        "mgt_full",
        "mip_full",
    ]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.array([getattr(row, name) for row in seed_results], dtype=np.float64)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=0))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic unimodal TopK superposition sweep")

    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--representation-dim", type=int, default=768)
    parser.add_argument("--latent-size", type=int, default=16384)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--gt-recovery-threshold", type=float, default=0.9)

    parser.add_argument("--num-train-pairs", type=int, default=50_000)
    parser.add_argument("--num-eval-pairs", type=int, default=10_000)
    parser.add_argument("--sparsity", type=float, default=0.99)
    parser.add_argument("--min-active", type=int, default=1)
    parser.add_argument("--cmin", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-interference", type=float, default=0.1)
    parser.add_argument(
        "--dictionary-strategy",
        type=str,
        choices=["gradient", "sdp"],
        default="gradient",
    )

    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument("--log-every", type=int, default=0)

    parser.add_argument("--output-root", type=str, default="outputs/synthetic_table4_topk")
    parser.add_argument("--run-tag", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_name = (
        f"unimodal_fd{args.feature_dim}_rep{args.representation_dim}_"
        f"latent{args.latent_size}_k{args.k}_s{args.sparsity}_"
        f"tau{args.gt_recovery_threshold}_minact{args.min_active}"
    )
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run: %s", run_name)
    logger.info("Output dir: %s", run_dir)

    seed_results: list[SeedRunResult] = []
    for offset in range(args.num_seeds):
        seed = args.seed_base + offset
        seed_results.append(_run_one_seed(args=args, seed=seed))

    aggregate = _aggregate_seed_results(seed_results)
    expected_active_gt = float(args.feature_dim * (1.0 - args.sparsity))
    feature_to_rep_ratio = float(args.feature_dim / args.representation_dim)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "condition": "unimodal",
            "feature_dim": args.feature_dim,
            "representation_dim": args.representation_dim,
            "latent_size": args.latent_size,
            "k": args.k,
            "sparsity": args.sparsity,
            "min_active": args.min_active,
            "cmin": args.cmin,
            "beta": args.beta,
            "max_interference": args.max_interference,
            "dictionary_strategy": args.dictionary_strategy,
            "gt_recovery_threshold": args.gt_recovery_threshold,
            "num_train_pairs": args.num_train_pairs,
            "num_eval_pairs": args.num_eval_pairs,
            "num_epochs": args.num_epochs,
            "num_seeds": args.num_seeds,
            "seed_base": args.seed_base,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "device": args.device,
            "run_name": run_name,
            "expected_active_gt": expected_active_gt,
            "feature_to_rep_ratio": feature_to_rep_ratio,
        },
        "seed_results": [
            {
                "seed": row.seed,
                "train_recon_loss": row.train_recon_loss,
                "eval_recon_loss": row.eval_recon_loss,
                "mgt_full": row.mgt_full,
                "mip_full": row.mip_full,
            }
            for row in seed_results
        ],
        "aggregate": aggregate,
    }

    json_path = run_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved result JSON: %s", json_path)
    logger.info(
        "Summary | fd=%d k=%d mgt=%.4f+/-%.4f mip=%.4f+/-%.4f eval_recon=%.6f+/-%.6f",
        args.feature_dim,
        args.k,
        aggregate["mgt_full_mean"],
        aggregate["mgt_full_std"],
        aggregate["mip_full_mean"],
        aggregate["mip_full_std"],
        aggregate["eval_recon_loss_mean"],
        aggregate["eval_recon_loss_std"],
    )


if __name__ == "__main__":
    main()
