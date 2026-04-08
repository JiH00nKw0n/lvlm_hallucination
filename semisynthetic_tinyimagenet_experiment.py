"""
Semi-synthetic SAE experiment on Tiny ImageNet CLIP embeddings.

Trains TopK SAEs in two configurations:
  - Single SAE   (latent_size)     on paired image+text embeddings
  - Two SAEs     (latent_size // 2) on image-only and text-only separately

Evaluates GT recovery (MGT, MIP) against class feature directions extracted
by ``semisynthetic_extract_features.py``.

Full evaluation matrix: 3 SAEs × 2 GT methods × 2 GT modalities = 12 combos.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers (shared with synthetic_table4_topk_experiment.py)          #
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
    """
    Compute GT Recovery (MGT) and Maximum Inner Product (MIP).

    learned_vectors: (n_learned, d)
    gt_matrix: (d, n_gt) where columns are GT features
    """
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


# ------------------------------------------------------------------ #
# Data loading                                                       #
# ------------------------------------------------------------------ #

def _load_features(feature_dir: Path) -> dict[str, Any]:
    """Load cached embeddings and GT features from disk."""
    data = {
        "train_img": torch.load(feature_dir / "train_image_embeds.pt", weights_only=True),
        "train_txt": torch.load(feature_dir / "train_text_embeds.pt", weights_only=True),
        "train_labels": torch.load(feature_dir / "train_labels.pt", weights_only=True),
        "val_img": torch.load(feature_dir / "val_image_embeds.pt", weights_only=True),
        "val_txt": torch.load(feature_dir / "val_text_embeds.pt", weights_only=True),
        "val_labels": torch.load(feature_dir / "val_labels.pt", weights_only=True),
    }

    gt_features = torch.load(feature_dir / "gt_features.pt", weights_only=False)
    data["gt"] = {k: v for k, v in gt_features.items()}  # lp_image, lp_text, me_image, me_text

    with open(feature_dir / "gt_features_meta.json", encoding="utf-8") as f:
        data["gt_meta"] = json.load(f)

    return data


# ------------------------------------------------------------------ #
# SAE training                                                       #
# ------------------------------------------------------------------ #

def _train_sae_paired(
    img_embeds: torch.Tensor,
    txt_embeds: torch.Tensor,
    val_img: torch.Tensor,
    val_txt: torch.Tensor,
    latent_size: int,
    k: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    batch_size: int,
    num_epochs: int,
    device: torch.device,
    log_every: int = 0,
    seed: int = 0,
) -> tuple[TopKSAE, float, float]:
    """Train a single SAE on paired image+text batches.

    Loss = (img_recon + txt_recon) / 2 per batch.
    Returns (model, train_loss_avg, eval_loss_avg).
    """
    hidden_size = img_embeds.shape[1]
    cfg = TopKSAEConfig(hidden_size=hidden_size, latent_size=latent_size, k=k, normalize_decoder=True)
    model = TopKSAE(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(img_embeds, txt_embeds),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    train_loss_sum, train_steps = 0.0, 0

    for epoch in range(num_epochs):
        for step, (img_batch, txt_batch) in enumerate(train_loader):
            img_batch = img_batch.to(device=device, dtype=torch.float32).unsqueeze(1)
            txt_batch = txt_batch.to(device=device, dtype=torch.float32).unsqueeze(1)

            img_out = model(hidden_states=img_batch)
            txt_out = model(hidden_states=txt_batch)
            loss = (img_out.recon_loss + txt_out.recon_loss) / 2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if model.W_dec.grad is not None:
                model.remove_gradient_parallel_to_decoder_directions()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            model.set_decoder_norm_to_unit_norm()

            train_loss_sum += float(loss.detach().item())
            train_steps += 1

            if log_every > 0 and (step + 1) % log_every == 0:
                logger.info(
                    "seed=%d epoch=%d step=%d/%d recon=%.6f",
                    seed, epoch + 1, step + 1, len(train_loader),
                    train_loss_sum / train_steps,
                )

    # Eval
    model.eval()
    eval_loss_sum, eval_steps = 0.0, 0
    eval_loader = DataLoader(
        TensorDataset(val_img, val_txt),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for img_batch, txt_batch in eval_loader:
            img_batch = img_batch.to(device=device, dtype=torch.float32).unsqueeze(1)
            txt_batch = txt_batch.to(device=device, dtype=torch.float32).unsqueeze(1)
            img_out = model(hidden_states=img_batch)
            txt_out = model(hidden_states=txt_batch)
            eval_loss_sum += float((img_out.recon_loss + txt_out.recon_loss).item() / 2)
            eval_steps += 1

    return (
        model,
        train_loss_sum / max(train_steps, 1),
        eval_loss_sum / max(eval_steps, 1),
    )


def _train_sae_single_modality(
    train_embeds: torch.Tensor,
    val_embeds: torch.Tensor,
    latent_size: int,
    k: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    batch_size: int,
    num_epochs: int,
    device: torch.device,
    log_every: int = 0,
    seed: int = 0,
    modality_tag: str = "",
) -> tuple[TopKSAE, float, float]:
    """Train a single-modality SAE. Returns (model, train_loss, eval_loss)."""
    hidden_size = train_embeds.shape[1]
    cfg = TopKSAEConfig(hidden_size=hidden_size, latent_size=latent_size, k=k, normalize_decoder=True)
    model = TopKSAE(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(train_embeds),
        batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    train_loss_sum, train_steps = 0.0, 0

    for epoch in range(num_epochs):
        for step, (batch,) in enumerate(train_loader):
            batch = batch.to(device=device, dtype=torch.float32)
            hs = batch.unsqueeze(1)

            out = model(hidden_states=hs)
            loss = out.recon_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if model.W_dec.grad is not None:
                model.remove_gradient_parallel_to_decoder_directions()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            model.set_decoder_norm_to_unit_norm()

            train_loss_sum += float(loss.detach().item())
            train_steps += 1

            if log_every > 0 and (step + 1) % log_every == 0:
                logger.info(
                    "[%s] seed=%d epoch=%d step=%d/%d recon=%.6f",
                    modality_tag, seed, epoch + 1, step + 1, len(train_loader),
                    train_loss_sum / train_steps,
                )

    # Eval
    model.eval()
    eval_loss_sum, eval_steps = 0.0, 0
    eval_loader = DataLoader(
        TensorDataset(val_embeds),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for (batch,) in eval_loader:
            batch = batch.to(device=device, dtype=torch.float32)
            hs = batch.unsqueeze(1)
            out = model(hidden_states=hs)
            eval_loss_sum += float(out.recon_loss.item())
            eval_steps += 1

    return (
        model,
        train_loss_sum / max(train_steps, 1),
        eval_loss_sum / max(eval_steps, 1),
    )


# ------------------------------------------------------------------ #
# Evaluation                                                         #
# ------------------------------------------------------------------ #

_GT_KEYS = ["lp_image", "lp_text", "me_image", "me_text"]


def _evaluate_sae(
    model: TopKSAE,
    gt_directions: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Evaluate one SAE against all 4 GT direction sets.

    Returns {gt_key: {"gt_recovery": ..., "mip": ...}}.
    """
    w_dec = model.W_dec.detach().cpu().numpy()  # (latent_size, hidden_size)
    results: dict[str, dict[str, float]] = {}
    for gt_key in _GT_KEYS:
        gt_matrix = gt_directions[gt_key].T  # (hidden_size, n_classes)
        results[gt_key] = _compute_recovery_metrics(w_dec, gt_matrix, threshold)
    return results


# ------------------------------------------------------------------ #
# Seed run                                                           #
# ------------------------------------------------------------------ #

@dataclass
class SeedRunResult:
    seed: int
    # Losses
    single_train_loss: float
    single_eval_loss: float
    two_img_train_loss: float
    two_img_eval_loss: float
    two_txt_train_loss: float
    two_txt_eval_loss: float
    # Single SAE recovery (4 GT combos)
    single_mgt_lp_img: float
    single_mip_lp_img: float
    single_mgt_lp_txt: float
    single_mip_lp_txt: float
    single_mgt_me_img: float
    single_mip_me_img: float
    single_mgt_me_txt: float
    single_mip_me_txt: float
    # Image SAE recovery (4 GT combos)
    img_sae_mgt_lp_img: float
    img_sae_mip_lp_img: float
    img_sae_mgt_lp_txt: float
    img_sae_mip_lp_txt: float
    img_sae_mgt_me_img: float
    img_sae_mip_me_img: float
    img_sae_mgt_me_txt: float
    img_sae_mip_me_txt: float
    # Text SAE recovery (4 GT combos)
    txt_sae_mgt_lp_img: float
    txt_sae_mip_lp_img: float
    txt_sae_mgt_lp_txt: float
    txt_sae_mip_lp_txt: float
    txt_sae_mgt_me_img: float
    txt_sae_mip_me_img: float
    txt_sae_mgt_me_txt: float
    txt_sae_mip_me_txt: float


def _flatten_eval(prefix: str, eval_results: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten evaluation results into SeedRunResult field names.

    E.g. prefix="single", gt_key="lp_image" → single_mgt_lp_img, single_mip_lp_img
    """
    short = {"lp_image": "lp_img", "lp_text": "lp_txt", "me_image": "me_img", "me_text": "me_txt"}
    flat: dict[str, float] = {}
    for gt_key, metrics in eval_results.items():
        tag = short[gt_key]
        flat[f"{prefix}_mgt_{tag}"] = metrics["gt_recovery"]
        flat[f"{prefix}_mip_{tag}"] = metrics["mip"]
    return flat


def _run_one_seed(args: argparse.Namespace, data: dict[str, Any], seed: int) -> SeedRunResult:
    """Run one seed: train Single + Two SAEs, evaluate all 12 combos."""
    device = _resolve_device(args.device)
    _seed_everything(seed)

    train_img = data["train_img"]
    train_txt = data["train_txt"]
    val_img = data["val_img"]
    val_txt = data["val_txt"]
    gt = data["gt"]

    logger.info("=== Seed %d ===", seed)

    # --- Single SAE --- #
    logger.info("Training Single SAE (latent=%d, k=%d) ...", args.latent_size, args.k)
    single_model, single_train, single_eval = _train_sae_paired(
        train_img, train_txt, val_img, val_txt,
        latent_size=args.latent_size, k=args.k,
        lr=args.lr, weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size, num_epochs=args.num_epochs,
        device=device, log_every=args.log_every, seed=seed,
    )
    single_eval_results = _evaluate_sae(single_model, gt, args.gt_recovery_threshold)
    del single_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Two SAEs --- #
    half_latent = args.latent_size // 2
    logger.info("Training Image SAE (latent=%d, k=%d) ...", half_latent, args.k)
    _seed_everything(seed)
    img_model, img_train, img_eval = _train_sae_single_modality(
        train_img, val_img,
        latent_size=half_latent, k=args.k,
        lr=args.lr, weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size, num_epochs=args.num_epochs,
        device=device, log_every=args.log_every, seed=seed,
        modality_tag="img",
    )
    img_eval_results = _evaluate_sae(img_model, gt, args.gt_recovery_threshold)
    del img_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Training Text SAE (latent=%d, k=%d) ...", half_latent, args.k)
    _seed_everything(seed)
    txt_model, txt_train, txt_eval = _train_sae_single_modality(
        train_txt, val_txt,
        latent_size=half_latent, k=args.k,
        lr=args.lr, weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size, num_epochs=args.num_epochs,
        device=device, log_every=args.log_every, seed=seed,
        modality_tag="txt",
    )
    txt_eval_results = _evaluate_sae(txt_model, gt, args.gt_recovery_threshold)
    del txt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Flatten --- #
    flat = {
        "seed": seed,
        "single_train_loss": single_train,
        "single_eval_loss": single_eval,
        "two_img_train_loss": img_train,
        "two_img_eval_loss": img_eval,
        "two_txt_train_loss": txt_train,
        "two_txt_eval_loss": txt_eval,
        **_flatten_eval("single", single_eval_results),
        **_flatten_eval("img_sae", img_eval_results),
        **_flatten_eval("txt_sae", txt_eval_results),
    }
    return SeedRunResult(**flat)


# ------------------------------------------------------------------ #
# Aggregation                                                        #
# ------------------------------------------------------------------ #

def _aggregate_seed_results(seed_results: list[SeedRunResult]) -> dict[str, float]:
    metric_names = [f.name for f in fields(SeedRunResult) if f.name != "seed"]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.array([getattr(row, name) for row in seed_results], dtype=np.float64)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=0))
    return summary


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semi-synthetic SAE experiment on Tiny ImageNet CLIP embeddings",
    )
    parser.add_argument("--feature-dir", type=str, required=True,
                        help="Directory with cached embeddings + GT features")
    parser.add_argument("--latent-size", type=int, default=4096)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--gt-recovery-threshold", type=float, default=0.8)

    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=0)

    parser.add_argument("--output-root", type=str, default="outputs/semisynthetic_tinyimagenet")
    parser.add_argument("--run-tag", type=str, default="")

    return parser.parse_args()


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    feature_dir = Path(args.feature_dir)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    # Load data
    logger.info("Loading features from %s ...", feature_dir)
    data = _load_features(feature_dir)

    n_classes = int(data["train_labels"].max().item()) + 1
    hidden_size = data["train_img"].shape[1]

    # Run name
    run_name = (
        f"semisyn_latent{args.latent_size}_k{args.k}_"
        f"tau{args.gt_recovery_threshold}_e{args.num_epochs}"
    )
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run: %s", run_name)
    logger.info("Classes=%d  hidden=%d  latent=%d  k=%d", n_classes, hidden_size, args.latent_size, args.k)

    # Seed loop
    seed_results: list[SeedRunResult] = []
    for offset in range(args.num_seeds):
        seed = args.seed_base + offset
        result = _run_one_seed(args, data, seed)
        seed_results.append(result)

    aggregate = _aggregate_seed_results(seed_results)

    # Build output JSON
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment": "semisynthetic_tinyimagenet",
            "feature_dir": str(feature_dir),
            "hidden_size": hidden_size,
            "latent_size": args.latent_size,
            "k": args.k,
            "num_classes": n_classes,
            "gt_recovery_threshold": args.gt_recovery_threshold,
            "num_epochs": args.num_epochs,
            "num_seeds": args.num_seeds,
            "seed_base": args.seed_base,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "device": args.device,
            "run_name": run_name,
            "num_train_images": int(data["train_img"].shape[0]),
            "num_val_images": int(data["val_img"].shape[0]),
        },
        "gt_features_meta": data["gt_meta"],
        "seed_results": [
            {f.name: getattr(row, f.name) for f in fields(SeedRunResult)}
            for row in seed_results
        ],
        "aggregate": aggregate,
    }

    json_path = run_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved result JSON: %s", json_path)

    # Print summary table
    logger.info("=" * 70)
    logger.info("Summary (mean ± std across %d seeds):", args.num_seeds)
    logger.info("-" * 70)
    logger.info("%-30s  %8s  %8s", "Metric", "Mean", "Std")
    logger.info("-" * 70)
    for key in sorted(aggregate.keys()):
        if key.endswith("_mean"):
            base = key[:-5]
            mean_val = aggregate[f"{base}_mean"]
            std_val = aggregate[f"{base}_std"]
            logger.info("%-30s  %8.4f  %8.4f", base, mean_val, std_val)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
