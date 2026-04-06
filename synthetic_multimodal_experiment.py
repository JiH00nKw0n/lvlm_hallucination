"""
Synthetic multimodal SAE experiments (Exp 1/2/3).

Experiment definitions:
    - Exp1: No alignment loss. Sweep feature_dim.
    - Exp2: Shared-topk latent L2 alignment with lambda sweep.
    - Exp3: Compare all-topk latent L2 vs shared-topk latent L2.
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
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.synthetic_multimodal_feature import (
    SyntheticMultimodalFeatureDatasetBuilder,
)
from src.models.configuration_sae import VLTopKSAEConfig
from src.models.modeling_sae import VLTopKSAE

logger = logging.getLogger(__name__)


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


def _parse_block_top_k(value: str) -> tuple[int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"block-top-k must have 2 integers [modality_specific,shared], got: {value}"
        )
    if any(x <= 0 for x in parts):
        raise argparse.ArgumentTypeError(
            f"block-top-k values must be positive, got: {value}"
        )
    return parts[0], parts[1]


def _split_dims(total_dim: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
    denom = ratio[0] + ratio[1] + ratio[2]
    image_dim = total_dim * ratio[0] // denom
    shared_dim = total_dim * ratio[1] // denom
    text_dim = total_dim - image_dim - shared_dim
    return image_dim, shared_dim, text_dim


def _allocate_k_from_ratio(total_k: int, ratio: tuple[int, int]) -> tuple[int, int]:
    if total_k <= 0:
        raise ValueError(f"total_k must be > 0, got {total_k}")
    r_specific, r_shared = ratio
    denom = r_specific + r_shared
    raw_specific = total_k * r_specific / denom
    raw_shared = total_k * r_shared / denom
    k_specific = int(raw_specific)
    k_shared = int(raw_shared)
    remainder = total_k - k_specific - k_shared
    frac = [
        (raw_specific - k_specific, r_specific, 0),
        (raw_shared - k_shared, r_shared, 1),
    ]
    frac.sort(reverse=True)
    for i in range(remainder):
        _, _, block_idx = frac[i % 2]
        if block_idx == 0:
            k_specific += 1
        else:
            k_shared += 1
    return k_specific, k_shared


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


def _scatter_topk_to_dense(top_acts: Tensor, top_indices: Tensor, latent_size: int) -> Tensor:
    dense = top_acts.new_zeros(top_acts.shape[:-1] + (latent_size,))
    dense.scatter_(dim=-1, index=top_indices, src=top_acts)
    return dense


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
    gt_cols = gt_matrix.T.astype(np.float64)
    gt_norm = _normalize_rows(gt_cols)

    sim = np.abs(learned_norm @ gt_norm.T)  # (n_learned, n_gt)
    best = sim.max(axis=0)  # best learned match per GT feature

    return {
        "gt_recovery": float((best > threshold).mean()),
        "mip": float(best.mean()),
        "num_gt_features": int(gt_matrix.shape[1]),
        "num_learned_features": int(learned_vectors.shape[0]),
    }


def _format_lambda_tag(value: float) -> str:
    if value == 0:
        return "0"
    text = f"{value:.8g}"
    return text.replace("-", "m").replace(".", "p")


@dataclass
class SeedRunResult:
    seed: int
    train_recon_loss: float
    train_align_loss: float
    train_total_loss: float
    eval_recon_loss: float
    eval_align_loss: float
    eval_total_loss: float
    mgt_shared: float
    mip_shared: float
    mgt_full: float
    mip_full: float
    per_example_path: str | None = None


def _align_loss_from_outputs(
    out_img: Any,
    out_txt: Any,
    latent_size: int,
    shared_slice: slice,
    alignment_mode: str,
) -> Tensor:
    if alignment_mode == "none":
        return out_img.recon_loss.new_tensor(0.0)

    img_dense = _scatter_topk_to_dense(out_img.latent_activations, out_img.latent_indices, latent_size)
    txt_dense = _scatter_topk_to_dense(out_txt.latent_activations, out_txt.latent_indices, latent_size)

    if alignment_mode == "all_topk_l2":
        return F.mse_loss(img_dense, txt_dense)
    if alignment_mode == "shared_topk_l2":
        return F.mse_loss(img_dense[..., shared_slice], txt_dense[..., shared_slice])
    raise ValueError(f"Unknown alignment mode: {alignment_mode}")


def _run_one_seed(
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> SeedRunResult:
    _seed_everything(seed)
    device = _resolve_device(args.device)
    logger.info("Seed=%d | device=%s", seed, device)

    builder = SyntheticMultimodalFeatureDatasetBuilder(
        feature_dim=args.feature_dim,
        representation_dim=args.representation_dim,
        vl_split_ratio=args.vl_split_ratio,
        num_train=args.num_train_pairs,
        num_eval=args.num_eval_pairs,
        num_test=args.num_eval_pairs,
        sparsity_shared=args.sparsity_shared,
        sparsity_image=args.sparsity_image,
        sparsity_text=args.sparsity_text,
        min_active_shared=args.min_active_shared,
        min_active_image=args.min_active_image,
        min_active_text=args.min_active_text,
        max_interference=args.max_interference,
        strategy=args.dictionary_strategy,
        enforce_block_orthogonality=not args.disable_block_orthogonality,
        seed=seed,
        return_ground_truth=args.save_per_example_metrics,
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

    latent_size = args.latent_size if args.latent_size > 0 else args.feature_dim
    cfg = VLTopKSAEConfig(
        hidden_size=args.representation_dim,
        latent_size=latent_size,
        k=args.k,
        normalize_decoder=True,
        vl_split_ratio=list(args.vl_split_ratio),
        block_top_k=list(args.block_top_k) if args.block_top_k is not None else None,
    )
    model = VLTopKSAE(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    _, shared_dim, _ = _split_dims(latent_size, args.vl_split_ratio)
    image_dim, _, _ = _split_dims(latent_size, args.vl_split_ratio)
    shared_slice = slice(image_dim, image_dim + shared_dim)

    train_recon_sum = 0.0
    train_align_sum = 0.0
    train_total_sum = 0.0
    train_steps = 0

    for epoch in range(args.num_epochs):
        for step, (img_batch, txt_batch) in enumerate(train_loader):
            img_batch = img_batch.to(device=device, dtype=torch.float32)
            txt_batch = txt_batch.to(device=device, dtype=torch.float32)
            img_hs = img_batch.unsqueeze(1)
            txt_hs = txt_batch.unsqueeze(1)
            img_mask = torch.ones((img_hs.shape[0], 1), dtype=torch.bool, device=device)
            txt_mask = torch.zeros((txt_hs.shape[0], 1), dtype=torch.bool, device=device)

            out_img = model(hidden_states=img_hs, visual_mask=img_mask)
            out_txt = model(hidden_states=txt_hs, visual_mask=txt_mask)

            recon_loss = out_img.recon_loss + out_txt.recon_loss
            align_loss = _align_loss_from_outputs(
                out_img=out_img,
                out_txt=out_txt,
                latent_size=latent_size,
                shared_slice=shared_slice,
                alignment_mode=args.alignment_mode,
            )
            total_loss = recon_loss + args.lambda_align * align_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if model.W_dec.grad is not None:
                model.remove_gradient_parallel_to_decoder_directions()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.set_decoder_norm_to_unit_norm()

            train_recon_sum += float(recon_loss.detach().item())
            train_align_sum += float(align_loss.detach().item())
            train_total_sum += float(total_loss.detach().item())
            train_steps += 1

            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                logger.info(
                    "seed=%d epoch=%d step=%d/%d recon=%.6f align=%.6f total=%.6f",
                    seed,
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    train_recon_sum / train_steps,
                    train_align_sum / train_steps,
                    train_total_sum / train_steps,
                )

    model.eval()
    eval_recon_sum = 0.0
    eval_align_sum = 0.0
    eval_total_sum = 0.0
    eval_steps = 0
    per_example_rows: list[dict[str, float | int]] = []
    eval_idx_offset = 0

    with torch.no_grad():
        for img_batch, txt_batch in eval_loader:
            img_batch = img_batch.to(device=device, dtype=torch.float32)
            txt_batch = txt_batch.to(device=device, dtype=torch.float32)
            img_hs = img_batch.unsqueeze(1)
            txt_hs = txt_batch.unsqueeze(1)
            img_mask = torch.ones((img_hs.shape[0], 1), dtype=torch.bool, device=device)
            txt_mask = torch.zeros((txt_hs.shape[0], 1), dtype=torch.bool, device=device)

            out_img = model(hidden_states=img_hs, visual_mask=img_mask)
            out_txt = model(hidden_states=txt_hs, visual_mask=txt_mask)

            recon_loss = out_img.recon_loss + out_txt.recon_loss
            align_loss = _align_loss_from_outputs(
                out_img=out_img,
                out_txt=out_txt,
                latent_size=latent_size,
                shared_slice=shared_slice,
                alignment_mode=args.alignment_mode,
            )
            total_loss = recon_loss + args.lambda_align * align_loss

            eval_recon_sum += float(recon_loss.item())
            eval_align_sum += float(align_loss.item())
            eval_total_sum += float(total_loss.item())
            eval_steps += 1

            if args.save_per_example_metrics:
                img_recon = out_img.output.squeeze(1)
                txt_recon = out_txt.output.squeeze(1)
                img_mse = ((img_recon - img_batch) ** 2).mean(dim=1).cpu().numpy()
                txt_mse = ((txt_recon - txt_batch) ** 2).mean(dim=1).cpu().numpy()
                batch_size = img_batch.shape[0]
                for i in range(batch_size):
                    per_example_rows.append(
                        {
                            "idx": int(eval_idx_offset + i),
                            "image_mse": float(img_mse[i]),
                            "text_mse": float(txt_mse[i]),
                        }
                    )
                eval_idx_offset += batch_size

    w_dec = model.W_dec.detach().cpu().numpy()  # (latent_size, repr_dim)
    gt_full = builder.w_full  # (repr_dim, feature_dim)
    gt_shared = builder.w_shared  # (repr_dim, shared_dim)

    full_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec,
        gt_matrix=gt_full,
        threshold=args.gt_recovery_threshold,
    )
    shared_metrics = _compute_recovery_metrics(
        learned_vectors=w_dec[shared_slice, :],
        gt_matrix=gt_shared,
        threshold=args.gt_recovery_threshold,
    )

    per_example_path = None
    if args.save_per_example_metrics:
        per_example_path = str(output_dir / f"per_example_seed{seed}.jsonl")
        with open(per_example_path, "w", encoding="utf-8") as f:
            for row in per_example_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return SeedRunResult(
        seed=seed,
        train_recon_loss=train_recon_sum / max(train_steps, 1),
        train_align_loss=train_align_sum / max(train_steps, 1),
        train_total_loss=train_total_sum / max(train_steps, 1),
        eval_recon_loss=eval_recon_sum / max(eval_steps, 1),
        eval_align_loss=eval_align_sum / max(eval_steps, 1),
        eval_total_loss=eval_total_sum / max(eval_steps, 1),
        mgt_shared=shared_metrics["gt_recovery"],
        mip_shared=shared_metrics["mip"],
        mgt_full=full_metrics["gt_recovery"],
        mip_full=full_metrics["mip"],
        per_example_path=per_example_path,
    )


def _aggregate_seed_results(seed_results: list[SeedRunResult]) -> dict[str, float]:
    metric_names = [
        "train_recon_loss",
        "train_align_loss",
        "train_total_loss",
        "eval_recon_loss",
        "eval_align_loss",
        "eval_total_loss",
        "mgt_shared",
        "mip_shared",
        "mgt_full",
        "mip_full",
    ]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.array([getattr(row, name) for row in seed_results], dtype=np.float64)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=0))
    return summary


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_seeds <= 0:
        raise ValueError(f"num-seeds must be > 0, got {args.num_seeds}")
    if args.num_epochs <= 0:
        raise ValueError(f"num-epochs must be > 0, got {args.num_epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be > 0, got {args.batch_size}")
    if args.lambda_align < 0:
        raise ValueError(f"lambda-align must be >= 0, got {args.lambda_align}")
    if args.latent_size == 0:
        raise ValueError("latent-size cannot be 0. Use -1 to set latent-size=feature-dim.")

    if args.experiment == 1 and args.alignment_mode != "none":
        raise ValueError("experiment=1 requires alignment-mode=none")
    if args.experiment == 2 and args.alignment_mode != "shared_topk_l2":
        raise ValueError("experiment=2 requires alignment-mode=shared_topk_l2")
    if args.experiment == 3 and args.alignment_mode == "none":
        raise ValueError("experiment=3 requires alignment-mode in {shared_topk_l2, all_topk_l2}")

    latent_size = args.feature_dim if args.latent_size < 0 else args.latent_size
    image_dim, shared_dim, text_dim = _split_dims(latent_size, args.vl_split_ratio)
    if min(image_dim, shared_dim, text_dim) <= 0:
        raise ValueError(
            f"Invalid latent split with latent_size={latent_size}, ratio={args.vl_split_ratio}"
        )
    if args.k > min(image_dim + shared_dim, shared_dim + text_dim):
        raise ValueError(
            f"k={args.k} exceeds active subspace size for one modality. "
            f"Split dims(image/shared/text)={image_dim}/{shared_dim}/{text_dim}"
        )
    if args.block_top_k is not None:
        k_specific, k_shared = _allocate_k_from_ratio(args.k, args.block_top_k)
        if k_specific > min(image_dim, text_dim):
            raise ValueError(
                "block-top-k allocation for modality_specific exceeds modality block size. "
                f"allocated={k_specific}, image_dim={image_dim}, text_dim={text_dim}"
            )
        if k_shared > shared_dim:
            raise ValueError(
                "block-top-k allocation for shared exceeds shared block size. "
                f"allocated={k_shared}, shared_dim={shared_dim}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic multimodal SAE experiments")

    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument(
        "--alignment-mode",
        type=str,
        choices=["none", "shared_topk_l2", "all_topk_l2"],
        required=True,
    )
    parser.add_argument("--lambda-align", type=float, default=0.0)

    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--representation-dim", type=int, default=128)
    parser.add_argument(
        "--vl-split-ratio",
        type=_parse_ratio,
        default=(1, 2, 1),
        help="Comma-separated ratio, e.g. 1,2,1",
    )

    parser.add_argument("--latent-size", type=int, default=-1, help="-1 means use feature-dim")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument(
        "--block-top-k",
        type=_parse_block_top_k,
        default=None,
        help="Comma-separated ratio for [modality_specific,shared] top-k allocation, e.g. 1,1",
    )
    parser.add_argument("--gt-recovery-threshold", type=float, default=0.8)

    parser.add_argument("--num-train-pairs", type=int, default=50_000)
    parser.add_argument("--num-eval-pairs", type=int, default=10_000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=42)

    parser.add_argument("--sparsity-shared", type=float, default=0.999)
    parser.add_argument("--sparsity-image", type=float, default=0.999)
    parser.add_argument("--sparsity-text", type=float, default=0.999)
    parser.add_argument("--min-active-shared", type=int, default=1)
    parser.add_argument("--min-active-image", type=int, default=1)
    parser.add_argument("--min-active-text", type=int, default=1)
    parser.add_argument("--max-interference", type=float, default=0.3)
    parser.add_argument(
        "--dictionary-strategy",
        type=str,
        choices=["gradient", "sdp", "random"],
        default="gradient",
    )
    parser.add_argument(
        "--disable-block-orthogonality",
        action="store_true",
        help="Disable cross-block orthogonality constraints for GT dictionaries.",
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=0)

    parser.add_argument("--output-root", type=str, default="outputs/synthetic_multimodal")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--save-per-example-metrics", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    latent_size = args.feature_dim if args.latent_size < 0 else args.latent_size
    lambda_tag = _format_lambda_tag(args.lambda_align)
    run_name = (
        f"exp{args.experiment}_fd{args.feature_dim}_latent{latent_size}_"
        f"k{args.k}_mode{args.alignment_mode}_lam{lambda_tag}"
    )
    if args.block_top_k is not None:
        run_name += f"_blocktopk{args.block_top_k[0]}-{args.block_top_k[1]}"
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
        result = _run_one_seed(args=args, seed=seed, output_dir=run_dir)
        seed_results.append(result)

    aggregate = _aggregate_seed_results(seed_results)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment": args.experiment,
            "alignment_mode": args.alignment_mode,
            "lambda_align": args.lambda_align,
            "feature_dim": args.feature_dim,
            "representation_dim": args.representation_dim,
            "latent_size": latent_size,
            "vl_split_ratio": list(args.vl_split_ratio),
            "k": args.k,
            "block_top_k": list(args.block_top_k) if args.block_top_k is not None else None,
            "gt_recovery_threshold": args.gt_recovery_threshold,
            "num_train_pairs": args.num_train_pairs,
            "num_eval_pairs": args.num_eval_pairs,
            "num_epochs": args.num_epochs,
            "num_seeds": args.num_seeds,
            "seed_base": args.seed_base,
            "sparsity_shared": args.sparsity_shared,
            "sparsity_image": args.sparsity_image,
            "sparsity_text": args.sparsity_text,
            "min_active_shared": args.min_active_shared,
            "min_active_image": args.min_active_image,
            "min_active_text": args.min_active_text,
            "max_interference": args.max_interference,
            "dictionary_strategy": args.dictionary_strategy,
            "enforce_block_orthogonality": not args.disable_block_orthogonality,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "device": args.device,
            "run_name": run_name,
        },
        "seed_results": [
            {
                "seed": row.seed,
                "train_recon_loss": row.train_recon_loss,
                "train_align_loss": row.train_align_loss,
                "train_total_loss": row.train_total_loss,
                "eval_recon_loss": row.eval_recon_loss,
                "eval_align_loss": row.eval_align_loss,
                "eval_total_loss": row.eval_total_loss,
                "mgt_shared": row.mgt_shared,
                "mip_shared": row.mip_shared,
                "mgt_full": row.mgt_full,
                "mip_full": row.mip_full,
                "per_example_path": row.per_example_path,
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
        "Summary | mgt_shared=%.4f±%.4f mip_shared=%.4f±%.4f eval_total=%.6f±%.6f",
        aggregate["mgt_shared_mean"],
        aggregate["mgt_shared_std"],
        aggregate["mip_shared_mean"],
        aggregate["mip_shared_std"],
        aggregate["eval_total_loss_mean"],
        aggregate["eval_total_loss_std"],
    )


if __name__ == "__main__":
    main()
