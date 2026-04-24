"""Train one-SAE / two-SAE / aux-SAE on cached CLIP embeddings.

Supports both COCO pair cache and ImageNet class-template cache via
`--dataset {coco, imagenet}`. Uses HF Trainer for cosine LR + warmup +
grad clip + per-epoch eval + ckpt. Dumps `loss_history.json` at end.

Usage:
    # COCO top-8 Shared SAE
    python scripts/real_alpha/train_real_sae.py \
        --variant one_sae --dataset coco --cache-dir cache/clip_b32_coco \
        --output-dir outputs/real_exp_v1/shared/coco \
        --latent 8192 --k 8 --epochs 30 --batch-size 1024 \
        --lr 5e-4 --warmup-ratio 0.05 --weight-decay 1e-5 --seed 0

    # ImageNet top-1 aux-SAE (iso_align)
    python scripts/real_alpha/train_real_sae.py \
        --variant aux_sae --aux-loss iso_align --aux-lambda 1e-4 \
        --dataset imagenet --cache-dir cache/clip_b32_imagenet \
        --max-per-class 1000 \
        --output-dir outputs/real_exp_v1/iso_align/imagenet \
        --latent 8192 --k 1 --epochs 30 --batch-size 1024 \
        --lr 5e-4 --seed 0
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
from src.datasets.cached_imagenet_pairs import CachedImageNetPairsDataset  # type: ignore  # noqa: E402
from src.models.configuration_sae import TopKSAEConfig, TwoSidedTopKSAEConfig  # type: ignore  # noqa: E402
from src.models.modeling_sae import TopKSAE, TwoSidedTopKSAE  # type: ignore  # noqa: E402
from src.models.vl_sae import VLSAE, VLSAEConfig  # type: ignore  # noqa: E402
from src.models.shared_enc_sae import SharedEncSAE, SharedEncSAEConfig  # type: ignore  # noqa: E402
from src.runners.trainer import (  # type: ignore  # noqa: E402
    DeadReviveCallback,
    OneSidedAuxSAETrainer,
    OneSidedSAETrainer,
    TwoSidedSAETrainer,
    VLSAETrainer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["one_sae", "two_sae", "aux_sae", "vl_sae", "shared_enc"], required=True)
    p.add_argument("--dataset", choices=["coco", "imagenet", "cc3m"], default="coco")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--eval-samples", type=int, default=0,
                   help="Used with --dataset cc3m: hold out this many pairs from the tail "
                        "of the train split for Trainer eval. 0 disables eval.")
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
    # aux_sae (Iso-Energy / Group-Sparse)
    p.add_argument("--aux-loss", choices=["iso_align", "group_sparse"], default=None,
                   help="Required when --variant aux_sae.")
    p.add_argument("--aux-lambda", type=float, default=None,
                   help="Aux loss weight (β or λ). Defaults: iso 1e-4, gs 0.05.")
    # imagenet subset
    p.add_argument("--max-per-class", type=int, default=1000,
                   help="ImageNet-only: cap per class for balanced subsample.")
    return p.parse_args()


def _build_topksae(args: argparse.Namespace) -> TopKSAE:
    cfg = TopKSAEConfig(
        hidden_size=args.hidden_size,
        latent_size=args.latent,
        expansion_factor=1,
        normalize_decoder=True,
        k=args.k,
        weight_tie=False,
    )
    return TopKSAE(cfg)


def build_one_sae_trainer(
    args: argparse.Namespace,
    train_ds,
    eval_ds,
    training_args: TrainingArguments,
) -> Trainer:
    model = _build_topksae(args)
    trainer = OneSidedSAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,  # type: ignore[arg-type]
        eval_dataset=eval_ds,  # type: ignore[arg-type]
        data_collator=default_data_collator,
    )
    return trainer


def build_aux_sae_trainer(
    args: argparse.Namespace,
    train_ds,
    eval_ds,
    training_args: TrainingArguments,
) -> Trainer:
    if args.aux_loss is None:
        raise SystemExit("--aux-loss required for --variant aux_sae")
    aux_weight = args.aux_lambda
    if aux_weight is None:
        aux_weight = 1e-4 if args.aux_loss == "iso_align" else 0.05
    model = _build_topksae(args)
    trainer = OneSidedAuxSAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,  # type: ignore[arg-type]
        eval_dataset=eval_ds,  # type: ignore[arg-type]
        data_collator=default_data_collator,
        aux_loss=args.aux_loss,
        aux_weight=aux_weight,
    )
    return trainer


def build_vl_sae_trainer(
    args: argparse.Namespace,
    train_ds,
    eval_ds,
    training_args: TrainingArguments,
) -> Trainer:
    cfg = VLSAEConfig(
        hidden_size=args.hidden_size,
        latent_size=args.latent,
        k=args.k,
    )
    model = VLSAE(cfg)
    trainer = VLSAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,  # type: ignore[arg-type]
        eval_dataset=eval_ds,  # type: ignore[arg-type]
        data_collator=default_data_collator,
    )
    return trainer


def build_shared_enc_trainer(
    args: argparse.Namespace,
    train_ds,
    eval_ds,
    training_args: TrainingArguments,
) -> Trainer:
    cfg = SharedEncSAEConfig(
        hidden_size=args.hidden_size,
        latent_size=args.latent,
        k=args.k,
    )
    model = SharedEncSAE(cfg)
    trainer = VLSAETrainer(  # same interface: image_embeds/text_embeds → loss
        model=model,
        args=training_args,
        train_dataset=train_ds,  # type: ignore[arg-type]
        eval_dataset=eval_ds,  # type: ignore[arg-type]
        data_collator=default_data_collator,
    )
    return trainer


def build_two_sae_trainer(
    args: argparse.Namespace,
    train_ds,
    eval_ds,
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

    logger.info("Loading cached embeddings from %s (dataset=%s)", args.cache_dir, args.dataset)
    if args.dataset == "coco":
        train_ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
        eval_ds = CachedClipPairsDataset(args.cache_dir, split="test", l2_normalize=True)
    elif args.dataset == "cc3m":
        # CC3M cache schema matches COCO, but has no "test" split — hold out
        # the tail of "train" for Trainer eval_loss logging.
        from torch.utils.data import Subset
        full_ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
        n = len(full_ds)
        if args.eval_samples > 0:
            head = n - args.eval_samples
            train_ds = Subset(full_ds, list(range(0, head)))
            eval_ds = Subset(full_ds, list(range(head, n)))
        else:
            train_ds = full_ds
            eval_ds = None
    else:  # imagenet
        train_ds = CachedImageNetPairsDataset(
            args.cache_dir, split="train",
            max_per_class=args.max_per_class, l2_normalize=True,
        )
        eval_ds = CachedImageNetPairsDataset(
            args.cache_dir, split="val", l2_normalize=True,
        )
    logger.info("train=%d eval=%s", len(train_ds), len(eval_ds) if eval_ds is not None else "none")

    # HF Trainer needs eval_strategy='no' when no eval dataset is available.
    eval_strategy = "epoch" if eval_ds is not None else "no"

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
        eval_strategy=eval_strategy,
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
        disable_tqdm=False,  # show HF Trainer ETA/progress bar
    )

    if args.variant == "one_sae":
        trainer = build_one_sae_trainer(args, train_ds, eval_ds, training_args)
    elif args.variant == "two_sae":
        trainer = build_two_sae_trainer(args, train_ds, eval_ds, training_args)
    elif args.variant == "vl_sae":
        trainer = build_vl_sae_trainer(args, train_ds, eval_ds, training_args)
    elif args.variant == "shared_enc":
        trainer = build_shared_enc_trainer(args, train_ds, eval_ds, training_args)
    else:
        trainer = build_aux_sae_trainer(args, train_ds, eval_ds, training_args)

    trainer.train()

    # Dump loss history
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # Final model save
    trainer.save_model(str(output_dir / "final"))
    logger.info("Done. Model saved to %s/final", output_dir)


if __name__ == "__main__":
    main()
