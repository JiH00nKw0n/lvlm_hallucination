"""Train one-SAE or two-SAE on cached CLIP ViT-B/32 Karpathy COCO embeddings.

Uses HF Trainer for cosine LR + warmup + grad clip + per-epoch eval + ckpt.
Dumps `loss_history.json` (from trainer.state.log_history) at end of run.

Usage:
    python scripts/real_alpha/train_real_sae.py \
        --variant one_sae --cache-dir cache/clip_b32_coco \
        --output-dir outputs/real_alpha_followup_1/one_sae \
        --latent 8192 --k 8 --epochs 30 --batch-size 1024 \
        --lr 5e-4 --warmup-ratio 0.05 --weight-decay 1e-5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

# Bootstrap: bypass heavy package __init__.py chains.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pathlib import Path  # noqa: E402

from transformers import Trainer, TrainingArguments  # noqa: E402
from transformers.data.data_collator import default_data_collator  # noqa: E402

from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore  # noqa: E402
from src.models.configuration_sae import TopKSAEConfig, TwoSidedTopKSAEConfig  # type: ignore  # noqa: E402
from src.models.modeling_sae import TopKSAE, TwoSidedTopKSAE  # type: ignore  # noqa: E402
from src.runners.trainer import (  # type: ignore  # noqa: E402
    DeadReviveCallback,
    OneSidedSAETrainer,
    TwoSidedSAETrainer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["one_sae", "two_sae"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--latent", type=int, default=8192)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="gradient accumulation steps; effective batch = batch_size * grad_accum")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataloader-num-workers", type=int, default=2)
    p.add_argument("--auxk-weight", type=float, default=0.0,
                   help="AuxK loss weight (Gao et al. 2024). 0 disables AuxK.")
    p.add_argument("--k-aux", type=int, default=None,
                   help="AuxK top-k (default: hidden_size // 2 = 256 for h=512).")
    p.add_argument("--dead-feature-threshold", type=int, default=10_000_000,
                   help="Feature tokens-since-fired threshold to classify dead.")
    p.add_argument("--revive-every-epoch", action="store_true",
                   help="Re-init per-side slots that fired 0 times in the epoch (random init).")
    return p.parse_args()


def build_one_sae_trainer(
    args: argparse.Namespace,
    train_ds: CachedClipPairsDataset,
    eval_ds: CachedClipPairsDataset,
    training_args: TrainingArguments,
) -> Trainer:
    cfg = TopKSAEConfig(
        hidden_size=args.hidden_size,
        latent_size=args.latent,
        expansion_factor=1,
        normalize_decoder=True,
        k=args.k,
        weight_tie=False,
    )
    model = TopKSAE(cfg)
    trainer = OneSidedSAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,  # type: ignore[arg-type]
        eval_dataset=eval_ds,  # type: ignore[arg-type]
        data_collator=default_data_collator,
    )
    return trainer


def build_two_sae_trainer(
    args: argparse.Namespace,
    train_ds: CachedClipPairsDataset,
    eval_ds: CachedClipPairsDataset,
    training_args: TrainingArguments,
) -> Trainer:
    cfg = TwoSidedTopKSAEConfig(
        hidden_size=args.hidden_size,
        latent_size=args.latent,
        k=args.k,
        normalize_decoder=True,
        k_aux=args.k_aux,
    )
    model = TwoSidedTopKSAE(cfg)
    trainer = TwoSidedSAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,  # type: ignore[arg-type]
        eval_dataset=eval_ds,  # type: ignore[arg-type]
        data_collator=default_data_collator,
        auxk_weight=args.auxk_weight,
        dead_feature_threshold=args.dead_feature_threshold,
        revive_every_epoch=args.revive_every_epoch,
    )
    if args.revive_every_epoch:
        trainer.add_callback(DeadReviveCallback(trainer))
    return trainer


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Loading cached embeddings from %s", args.cache_dir)
    train_ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
    eval_ds = CachedClipPairsDataset(args.cache_dir, split="test", l2_normalize=True)
    logger.info("train=%d eval=%d", len(train_ds), len(eval_ds))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=50,
        seed=args.seed,
        fp16=False,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
    )

    if args.variant == "one_sae":
        trainer = build_one_sae_trainer(args, train_ds, eval_ds, training_args)
    else:
        trainer = build_two_sae_trainer(args, train_ds, eval_ds, training_args)

    trainer.train()

    # Dump loss history
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # Final model save
    trainer.save_model(str(output_dir / "final"))
    logger.info("Done. Model saved to %s/final", output_dir)


if __name__ == "__main__":
    main()
