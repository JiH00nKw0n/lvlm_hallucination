"""
Synthetic modality comparison experiment: unimodal vs multimodal.

Compares TopKSAE feature recovery under two data-generating assumptions:
    - unimodal:   x = Wp @ z  (single representation per sample)
    - multimodal: x_img = W_img @ z_img + W_shared @ z_shared
                  x_txt = W_txt @ z_txt + W_shared @ z_shared  (paired)

Model is always TopKSAE (no modality awareness).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.synthetic_feature import SyntheticFeatureDatasetBuilder
from src.datasets.synthetic_multimodal_feature import (
    SyntheticMultimodalFeatureDatasetBuilder,
)
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
    feature_indices: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    learned_vectors: (n_learned, d)
    gt_matrix: (d, n_gt) where columns are GT features
    feature_indices: if provided, only evaluate these column indices of gt_matrix
    """
    if feature_indices is not None:
        gt_matrix = gt_matrix[:, feature_indices]

    if gt_matrix.shape[1] == 0:
        return {
            "gt_recovery": float("nan"),
            "mip": float("nan"),
            "num_gt_features": 0,
            "num_learned_features": int(learned_vectors.shape[0]),
        }

    learned_norm = _normalize_rows(learned_vectors.astype(np.float64))
    gt_norm = _normalize_rows(gt_matrix.T.astype(np.float64))

    sim = np.abs(learned_norm @ gt_norm.T)  # (n_learned, n_gt)
    best = sim.max(axis=0)  # best learned match per GT feature

    return {
        "gt_recovery": float((best > threshold).mean()),
        "mip": float(best.mean()),
        "num_gt_features": int(gt_matrix.shape[1]),
        "num_learned_features": int(learned_vectors.shape[0]),
    }


def _compute_mfms(
    model: torch.nn.Module,
    img_loader: DataLoader,
    txt_loader: DataLoader,
    device: torch.device,
    latent_size: int,
    eps: float = 1e-8,
) -> float:
    """Mean Feature Modality Specificity.

    For each latent i, compute activation frequency in image vs text samples,
    then FMS_i = |p_img - p_txt| / (p_img + p_txt + eps). Returns mean over all latents.
    """
    img_counts = torch.zeros(latent_size, device=device)
    txt_counts = torch.zeros(latent_size, device=device)
    n_img = 0
    n_txt = 0

    with torch.no_grad():
        for (img_batch,) in img_loader:
            img_batch = img_batch.to(device=device, dtype=torch.float32)
            _, top_idx = model.encode(img_batch.unsqueeze(1))  # (B, 1, k)
            top_idx = top_idx.squeeze(1)  # (B, k)
            img_counts.scatter_add_(0, top_idx.reshape(-1), torch.ones(top_idx.numel(), device=device))
            n_img += img_batch.shape[0]

        for (txt_batch,) in txt_loader:
            txt_batch = txt_batch.to(device=device, dtype=torch.float32)
            _, top_idx = model.encode(txt_batch.unsqueeze(1))
            top_idx = top_idx.squeeze(1)
            txt_counts.scatter_add_(0, top_idx.reshape(-1), torch.ones(top_idx.numel(), device=device))
            n_txt += txt_batch.shape[0]

    p_img = img_counts / max(n_img, 1)
    p_txt = txt_counts / max(n_txt, 1)
    fms = (p_img - p_txt).abs() / (p_img + p_txt + eps)
    return float(fms.mean().item())


def _top10_indices(num_features: int) -> np.ndarray:
    """Return indices of top 10% features (lowest index = highest importance)."""
    k = max(1, num_features // 10)
    return np.arange(k)


# ------------------------------------------------------------------ #
# Result dataclass                                                     #
# ------------------------------------------------------------------ #


@dataclass
class SeedRunResult:
    seed: int
    condition: str
    train_recon_loss: float
    eval_recon_loss: float
    mgt_full: float
    mip_full: float
    # top 10% importance features
    mgt_top10: float = float("nan")
    mip_top10: float = float("nan")
    # multimodal-only breakdown (nan for unimodal)
    mgt_shared: float = float("nan")
    mip_shared: float = float("nan")
    mgt_image_private: float = float("nan")
    mip_image_private: float = float("nan")
    mgt_text_private: float = float("nan")
    mip_text_private: float = float("nan")
    # Mean Feature Modality Specificity (multimodal only)
    mfms: float = float("nan")


# ------------------------------------------------------------------ #
# Unimodal path                                                        #
# ------------------------------------------------------------------ #


def _run_unimodal_seed(
    args: argparse.Namespace,
    seed: int,
) -> SeedRunResult:
    _seed_everything(seed)
    device = _resolve_device(args.device)
    logger.info("[unimodal] seed=%d device=%s", seed, device)

    use_importance = args.importance_decay < 1.0
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
        scale_by_importance_probability=use_importance,
        importance_probability_decay=args.importance_decay,
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
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loss_sum = 0.0
    train_steps = 0

    for epoch in range(args.num_epochs):
        for step, (batch,) in enumerate(train_loader):
            batch = batch.to(device=device, dtype=torch.float32)
            hs = batch.unsqueeze(1)  # (B, 1, hidden_size)

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
                    "[unimodal] seed=%d epoch=%d step=%d/%d recon=%.6f",
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

    w_dec = model.W_dec.detach().cpu().numpy()  # (latent_size, repr_dim)
    gt_wp = builder.wp  # (repr_dim, feature_dim)
    assert gt_wp is not None, "Wp should have been generated after build_dataset()"

    full_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=gt_wp,
        threshold=args.gt_recovery_threshold,
    )
    top10_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=gt_wp,
        threshold=args.gt_recovery_threshold,
        feature_indices=_top10_indices(gt_wp.shape[1]),
    )

    return SeedRunResult(
        seed=seed,
        condition="unimodal",
        train_recon_loss=train_loss_sum / max(train_steps, 1),
        eval_recon_loss=eval_loss_sum / max(eval_steps, 1),
        mgt_full=full_metrics["gt_recovery"],
        mip_full=full_metrics["mip"],
        mgt_top10=top10_metrics["gt_recovery"],
        mip_top10=top10_metrics["mip"],
    )


# ------------------------------------------------------------------ #
# Multimodal path                                                      #
# ------------------------------------------------------------------ #


def _run_multimodal_seed(
    args: argparse.Namespace,
    seed: int,
) -> SeedRunResult:
    _seed_everything(seed)
    device = _resolve_device(args.device)
    logger.info("[multimodal] seed=%d device=%s", seed, device)

    builder = SyntheticMultimodalFeatureDatasetBuilder(
        feature_dim=args.feature_dim,
        representation_dim=args.representation_dim,
        vl_split_ratio=args.vl_split_ratio,
        num_train=args.num_train_pairs,
        num_eval=args.num_eval_pairs,
        num_test=args.num_eval_pairs,
        sparsity_shared=args.sparsity,
        sparsity_image=args.sparsity,
        sparsity_text=args.sparsity,
        min_active_shared=args.min_active,
        min_active_image=args.min_active,
        min_active_text=args.min_active,
        importance_probability_decay=args.importance_decay,
        importance_target=args.importance_target,
        max_interference=args.max_interference,
        strategy=args.dictionary_strategy,
        enforce_block_orthogonality=True,
        cmin=args.cmin,
        beta=args.beta,
        seed=seed,
        return_ground_truth=False,
        verbose=False,
    )
    ds = builder.build_numpy_dataset()
    train_img = torch.from_numpy(ds["train"]["image_representation"])
    train_txt = torch.from_numpy(ds["train"]["text_representation"])
    eval_img = torch.from_numpy(ds["eval"]["image_representation"])
    eval_txt = torch.from_numpy(ds["eval"]["text_representation"])

    train_loader = DataLoader(
        TensorDataset(train_img, train_txt),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        TensorDataset(eval_img, eval_txt),
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
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loss_sum = 0.0
    train_steps = 0

    for epoch in range(args.num_epochs):
        for step, (img_batch, txt_batch) in enumerate(train_loader):
            img_batch = img_batch.to(device=device, dtype=torch.float32)
            txt_batch = txt_batch.to(device=device, dtype=torch.float32)
            img_hs = img_batch.unsqueeze(1)
            txt_hs = txt_batch.unsqueeze(1)

            out_img = model(hidden_states=img_hs)
            out_txt = model(hidden_states=txt_hs)

            loss = (out_img.recon_loss + out_txt.recon_loss) / 2.0

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
                    "[multimodal] seed=%d epoch=%d step=%d/%d recon=%.6f",
                    seed, epoch + 1, step + 1, len(train_loader),
                    train_loss_sum / train_steps,
                )

    model.eval()
    eval_loss_sum = 0.0
    eval_steps = 0

    with torch.no_grad():
        for img_batch, txt_batch in eval_loader:
            img_batch = img_batch.to(device=device, dtype=torch.float32)
            txt_batch = txt_batch.to(device=device, dtype=torch.float32)
            img_hs = img_batch.unsqueeze(1)
            txt_hs = txt_batch.unsqueeze(1)

            out_img = model(hidden_states=img_hs)
            out_txt = model(hidden_states=txt_hs)

            loss = (out_img.recon_loss + out_txt.recon_loss) / 2.0
            eval_loss_sum += float(loss.item())
            eval_steps += 1

    w_dec = model.W_dec.detach().cpu().numpy()  # (latent_size, repr_dim)

    # Full GT recovery
    gt_full = builder.w_full  # (repr_dim, feature_dim)
    full_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=gt_full,
        threshold=args.gt_recovery_threshold,
    )

    # Top 10% of target block (importance-aware)
    target_w = {
        "shared": builder.w_shared,
        "image": builder.w_image,
        "text": builder.w_text,
    }[args.importance_target]
    top10_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=target_w,
        threshold=args.gt_recovery_threshold,
        feature_indices=_top10_indices(target_w.shape[1]),
    )

    # Per-block GT recovery
    shared_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=builder.w_shared,
        threshold=args.gt_recovery_threshold,
    )
    image_priv_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=builder.w_image,
        threshold=args.gt_recovery_threshold,
    )
    text_priv_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=builder.w_text,
        threshold=args.gt_recovery_threshold,
    )

    # Mean Feature Modality Specificity
    eval_img_loader = DataLoader(
        TensorDataset(eval_img),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    eval_txt_loader = DataLoader(
        TensorDataset(eval_txt),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    mfms = _compute_mfms(
        model=model,
        img_loader=eval_img_loader,
        txt_loader=eval_txt_loader,
        device=device,
        latent_size=args.latent_size,
    )

    return SeedRunResult(
        seed=seed,
        condition="multimodal",
        train_recon_loss=train_loss_sum / max(train_steps, 1),
        eval_recon_loss=eval_loss_sum / max(eval_steps, 1),
        mgt_full=full_metrics["gt_recovery"],
        mip_full=full_metrics["mip"],
        mgt_top10=top10_metrics["gt_recovery"],
        mip_top10=top10_metrics["mip"],
        mgt_shared=shared_metrics["gt_recovery"],
        mip_shared=shared_metrics["mip"],
        mgt_image_private=image_priv_metrics["gt_recovery"],
        mip_image_private=image_priv_metrics["mip"],
        mgt_text_private=text_priv_metrics["gt_recovery"],
        mip_text_private=text_priv_metrics["mip"],
        mfms=mfms,
    )


# ------------------------------------------------------------------ #
# Aggregation                                                          #
# ------------------------------------------------------------------ #


def _aggregate_seed_results(seed_results: list[SeedRunResult]) -> dict[str, float]:
    metric_names = [
        "train_recon_loss",
        "eval_recon_loss",
        "mgt_full",
        "mip_full",
        "mgt_top10",
        "mip_top10",
        "mgt_shared",
        "mip_shared",
        "mgt_image_private",
        "mip_image_private",
        "mgt_text_private",
        "mip_text_private",
        "mfms",
    ]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.array([getattr(row, name) for row in seed_results], dtype=np.float64)
        if np.all(np.isnan(values)):
            summary[f"{name}_mean"] = float("nan")
            summary[f"{name}_std"] = float("nan")
        else:
            summary[f"{name}_mean"] = float(np.nanmean(values))
            summary[f"{name}_std"] = float(np.nanstd(values, ddof=0))
    return summary


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def _parse_ratio(value: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"vl-split-ratio must have 3 integers, got: {value}"
        )
    if any(x <= 0 for x in parts):
        raise argparse.ArgumentTypeError(
            f"vl-split-ratio values must be positive, got: {value}"
        )
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic modality comparison: unimodal vs multimodal with TopKSAE"
    )

    parser.add_argument(
        "--condition",
        type=str,
        choices=["unimodal", "multimodal"],
        required=True,
    )

    # Data dimensions
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--representation-dim", type=int, default=768)
    parser.add_argument(
        "--vl-split-ratio",
        type=_parse_ratio,
        default=(1, 2, 1),
        help="Multimodal only: comma-separated ratio, e.g. 1,2,1",
    )

    # SAE config
    parser.add_argument("--latent-size", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--gt-recovery-threshold", type=float, default=0.8)

    # Data generation
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
        choices=["gradient", "sdp", "random"],
        default="gradient",
    )
    parser.add_argument(
        "--importance-decay",
        type=float,
        default=1.0,
        help="Importance probability decay per feature index. 1.0 = uniform.",
    )
    parser.add_argument(
        "--importance-target",
        type=str,
        choices=["shared", "image", "text"],
        default="shared",
        help="Which block gets importance decay (multimodal only).",
    )

    # Training
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    parser.add_argument("--log-every", type=int, default=0)

    # Output
    parser.add_argument("--output-root", type=str, default="outputs/synthetic_modality")
    parser.add_argument("--run-tag", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_name = (
        f"{args.condition}_fd{args.feature_dim}_rep{args.representation_dim}_"
        f"latent{args.latent_size}_k{args.k}_s{args.sparsity}_"
        f"decay{args.importance_decay}_tgt{args.importance_target}"
    )
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run: %s", run_name)
    logger.info("Output dir: %s", run_dir)

    run_seed = _run_unimodal_seed if args.condition == "unimodal" else _run_multimodal_seed

    seed_results: list[SeedRunResult] = []
    for offset in range(args.num_seeds):
        seed = args.seed_base + offset
        result = run_seed(args=args, seed=seed)
        seed_results.append(result)

    aggregate = _aggregate_seed_results(seed_results)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "condition": args.condition,
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
            "importance_decay": args.importance_decay,
            "importance_target": args.importance_target,
            "vl_split_ratio": list(args.vl_split_ratio) if args.condition == "multimodal" else None,
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
        },
        "seed_results": [
            {
                "seed": row.seed,
                "condition": row.condition,
                "train_recon_loss": row.train_recon_loss,
                "eval_recon_loss": row.eval_recon_loss,
                "mgt_full": row.mgt_full,
                "mip_full": row.mip_full,
                "mgt_top10": row.mgt_top10,
                "mip_top10": row.mip_top10,
                "mgt_shared": row.mgt_shared,
                "mip_shared": row.mip_shared,
                "mgt_image_private": row.mgt_image_private,
                "mip_image_private": row.mip_image_private,
                "mgt_text_private": row.mgt_text_private,
                "mip_text_private": row.mip_text_private,
                "mfms": row.mfms,
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
        "Summary | condition=%s mgt_full=%.4f+/-%.4f mip_full=%.4f+/-%.4f eval_recon=%.6f+/-%.6f",
        args.condition,
        aggregate["mgt_full_mean"],
        aggregate["mgt_full_std"],
        aggregate["mip_full_mean"],
        aggregate["mip_full_std"],
        aggregate["eval_recon_loss_mean"],
        aggregate["eval_recon_loss_std"],
    )


if __name__ == "__main__":
    main()
