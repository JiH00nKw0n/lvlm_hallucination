"""Real-data entry script for the aux-alignment ablation (CLIP B/32, COCO).

Trains one variant on cached CLIP embeddings. Run 9 times in a shell loop
(see `run_aux_alignment_real.sh`) to cover all variants.

Variant choice via `--variant-name`. Mapping below resolves it into
``(aux_loss, hungarian_schedule, revive_dead)``.

Usage (single variant):
    python scripts/real_alpha/real_aux_alignment.py \\
        --variant-name infonce_perepoch+revive \\
        --cache-dir cache/clip_b32_coco \\
        --output-dir outputs/aux_alignment_clip_b32/infonce_perepoch+revive \\
        --hidden-size 512 --latent 8192 --k 8 \\
        --epochs 30 --batch-size 1024 \\
        --lr 5e-4 --warmup-ratio 0.05 --weight-decay 1e-5 \\
        --rho0 0.2 --lambda-aux 1.0 --seed 1
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

import torch  # noqa: E402
from transformers import TrainingArguments  # noqa: E402
from transformers.data.data_collator import default_data_collator  # noqa: E402

from src.configs.experiment import MethodConfig  # type: ignore  # noqa: E402
from src.datasets.cached_clip_pairs import CachedClipPairsDataset  # type: ignore  # noqa: E402
from src.models.configuration_sae import TwoSidedTopKSAEConfig  # type: ignore  # noqa: E402
from src.models.modeling_sae import TwoSidedTopKSAE  # type: ignore  # noqa: E402
from src.runners.synthetic_trainers import AuxAlignmentCallback  # type: ignore  # noqa: E402
from src.runners.trainer import RealAuxAlignmentTrainer  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# variant_name -> (aux_loss, hungarian_schedule, revive_dead)
VARIANT_MAP: dict[str, tuple[str, str, bool]] = {
    "recon_only":               ("none",       "none",                  False),
    "naive_once":               ("naive_diag", "once",                  False),
    "barlow_once":              ("barlow",     "once",                  False),
    "infonce_once":             ("infonce",    "once",                  False),
    "infonce_once+revive":      ("infonce",    "once",                  True),
    "naive_perepoch+revive":    ("naive_diag", "per_epoch_partitioned", True),
    "barlow_perepoch+revive":   ("barlow",     "per_epoch_partitioned", True),
    "infonce_perepoch":         ("infonce",    "per_epoch_partitioned", False),
    "infonce_perepoch+revive":  ("infonce",    "per_epoch_partitioned", True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant-name", required=True, choices=sorted(VARIANT_MAP.keys()))
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--latent", type=int, default=8192)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rho0", type=float, default=0.2)
    p.add_argument("--lambda-aux", type=float, default=1.0)
    p.add_argument("--barlow-lambda-off", type=float, default=0.005)
    p.add_argument("--k-align", type=int, default=15,
                   help="Stage-1 length for once-mode (default 15 = epochs/2 at 30).")
    p.add_argument("--dataloader-num-workers", type=int, default=2)
    return p.parse_args()


def _stack_train_tensors(ds: CachedClipPairsDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the train split into two tensors (image, text) for callback's full-data C."""
    logger.info("Stacking train pairs for full-data correlation snapshot (n=%d)", len(ds))
    img = torch.stack([ds[i]["image_embeds"] for i in range(len(ds))], dim=0)
    txt = torch.stack([ds[i]["text_embeds"] for i in range(len(ds))], dim=0)
    return img, txt


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist config (CLI args + resolved variant)
    aux_loss, schedule, revive = VARIANT_MAP[args.variant_name]
    cfg_dict = {**vars(args), "_resolved_aux_loss": aux_loss,
                "_resolved_hungarian_schedule": schedule,
                "_resolved_revive_dead": revive}
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    logger.info("Variant: %s -> (aux=%s, hung=%s, revive=%s)",
                args.variant_name, aux_loss, schedule, revive)
    logger.info("Loading cached embeddings from %s", args.cache_dir)
    train_ds = CachedClipPairsDataset(args.cache_dir, split="train", l2_normalize=True)
    eval_ds = CachedClipPairsDataset(args.cache_dir, split="test", l2_normalize=True)
    logger.info("train=%d eval=%d", len(train_ds), len(eval_ds))

    train_img, train_txt = _stack_train_tensors(train_ds)

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
        save_total_limit=2,
        logging_steps=50,
        seed=args.seed,
        fp16=False,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
    )

    sae_cfg = TwoSidedTopKSAEConfig(
        hidden_size=args.hidden_size,
        latent_size=args.latent,
        k=args.k,
        normalize_decoder=True,
    )
    model = TwoSidedTopKSAE(sae_cfg)

    if aux_loss == "none" and schedule == "none":
        # recon-only baseline -> reuse plain TwoSidedSAETrainer (no callback).
        from src.runners.trainer import TwoSidedSAETrainer  # type: ignore
        trainer = TwoSidedSAETrainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=eval_ds,  # type: ignore[arg-type]
            data_collator=default_data_collator,
        )
        trainer.train()
        with open(output_dir / "loss_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        trainer.save_model(str(output_dir / "final"))
        logger.info("Done (recon_only). Model saved to %s/final", output_dir)
        return

    variant_cfg = MethodConfig(
        name="paired_aux_alignment",
        variant_name=args.variant_name,
        aux_loss=aux_loss,
        hungarian_schedule=schedule,
        revive_dead=revive,
        rho0=args.rho0,
        lambda_aux=args.lambda_aux,
        barlow_lambda_off=args.barlow_lambda_off,
        k_align=args.k_align,
    )
    trainer = RealAuxAlignmentTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,  # type: ignore[arg-type]
        data_collator=default_data_collator,
        variant_cfg=variant_cfg,
        train_img=train_img, train_txt=train_txt,
    )
    trainer.add_callback(AuxAlignmentCallback(trainer))
    trainer.train()

    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(trainer.diagnostics, f, indent=2, default=str)
    trainer.save_model(str(output_dir / "final"))
    logger.info("Done (variant=%s). Model + diagnostics saved to %s/final",
                args.variant_name, output_dir)


if __name__ == "__main__":
    main()
