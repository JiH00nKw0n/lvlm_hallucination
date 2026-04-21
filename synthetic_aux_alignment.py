"""Synthetic-data entry script for the aux-alignment ablation.

Trains 9 method variants on the synthetic Theorem-2 generator (Figure 1
setup) for a single (alpha, seed) point. Variants vary on three axes:

  - aux_loss        : naive_diag | barlow | infonce  (or 'none' baseline)
  - hungarian_schedule : once | per_epoch_partitioned
  - revive_dead     : true | false

Driven by `configs/synthetic/aux_alignment_compare.yaml`.

Reuses the bootstrap pattern of `run_synthetic_v2.py`. This file is a
thin variant-aware extension and does NOT modify the existing v2 script.

Usage:
    python synthetic_aux_alignment.py \\
        --config configs/synthetic/aux_alignment_compare.yaml
"""

from __future__ import annotations

# --- Bootstrap (mirrors run_synthetic_v2.py) ---------------------------------
import importlib.util
import sys
import types
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parent


def _stub(name: str, path: _Path) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = m


def _load(name: str, fpath: _Path) -> None:
    if name in sys.modules and getattr(sys.modules[name], "__file__", None):
        return
    spec = importlib.util.spec_from_file_location(name, fpath)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)


_stub("src", _REPO / "src")
_stub("src.common", _REPO / "src" / "common")
_stub("src.models", _REPO / "src" / "models")
_stub("src.runners", _REPO / "src" / "runners")
_stub("src.datasets", _REPO / "src" / "datasets")
_stub("src.metrics", _REPO / "src" / "metrics")
_stub("src.training", _REPO / "src" / "training")
_stub("src.configs", _REPO / "src" / "configs")
_load("src.common.registry", _REPO / "src" / "common" / "registry.py")
_load("src.models.configuration_sae", _REPO / "src" / "models" / "configuration_sae.py")
_load("src.models.modeling_sae", _REPO / "src" / "models" / "modeling_sae.py")
# --- end bootstrap -----------------------------------------------------------

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import TrainingArguments
from transformers.data.data_collator import default_data_collator

from src.configs.experiment import ExperimentConfig, MethodConfig
from src.datasets.synthetic_paired_builder import SyntheticPairedBuilder
from src.datasets.synthetic_pairs import SyntheticPairedDataset
from src.metrics.evaluate import aggregate, evaluate_method
from src.models.configuration_sae import TwoSidedTopKSAEConfig
from src.models.modeling_sae import TwoSidedTopKSAE
from src.runners.synthetic_trainers import (
    AuxAlignmentCallback,
    AuxAlignmentTrainer,
    TwoReconTrainer,
)
from src.training.callbacks import SAENormCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _make_training_args(cfg: ExperimentConfig, output_dir: str, seed: int) -> TrainingArguments:
    t = cfg.training
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t.num_epochs,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        learning_rate=t.lr,
        lr_scheduler_type="constant",
        dataloader_drop_last=True,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        fp16=False,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )


def _save_params(
    sae_i,
    sae_t,
    builder: SyntheticPairedBuilder,
    alpha: float,
    seed: int,
    method_id: str,
    run_dir: Path,
) -> None:
    dump_dir = run_dir / "params"
    dump_dir.mkdir(parents=True, exist_ok=True)
    mid_safe = method_id.replace("::", "__").replace("/", "_").replace("+", "__")
    fn = dump_dir / f"alpha{alpha:.2f}_seed{seed}_{mid_safe}.npz"
    np.savez_compressed(
        fn,
        w_dec_img=sae_i.W_dec.detach().cpu().numpy().astype(np.float32),
        w_dec_txt=sae_t.W_dec.detach().cpu().numpy().astype(np.float32),
        b_dec_img=sae_i.b_dec.detach().cpu().numpy().astype(np.float32),
        b_dec_txt=sae_t.b_dec.detach().cpu().numpy().astype(np.float32),
        w_enc_img=sae_i.encoder.weight.detach().cpu().numpy().astype(np.float32),
        w_enc_txt=sae_t.encoder.weight.detach().cpu().numpy().astype(np.float32),
        b_enc_img=sae_i.encoder.bias.detach().cpu().numpy().astype(np.float32),
        b_enc_txt=sae_t.encoder.bias.detach().cpu().numpy().astype(np.float32),
        phi_S=builder.phi_S.astype(np.float32),
        psi_S=builder.psi_S.astype(np.float32),
        phi_I=(builder.phi_I.astype(np.float32) if builder.phi_I is not None else np.zeros((0, 0), dtype=np.float32)),
        psi_T=(builder.psi_T.astype(np.float32) if builder.psi_T is not None else np.zeros((0, 0), dtype=np.float32)),
        alpha_target=np.array(alpha, dtype=np.float32),
        seed=np.array(seed, dtype=np.int32),
        latent_size_img=np.array(sae_i.latent_size, dtype=np.int32),
        latent_size_txt=np.array(sae_t.latent_size, dtype=np.int32),
        same_model_flag=np.array(int(sae_i is sae_t), dtype=np.int32),
    )


def _train_method(
    method: MethodConfig,
    cfg: ExperimentConfig,
    train_ds: SyntheticPairedDataset,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    latent_size: int,
    seed: int,
    run_dir: Path,
    device: torch.device,
) -> tuple[Any, Any, str, dict]:
    """Train one variant. Returns ``(sae_i, sae_t, method_id, diagnostics)``."""
    d = cfg.data
    t = cfg.training
    method_id = method.variant_name or method.name

    tmp_dir = str(run_dir / "tmp" / method_id.replace("/", "_").replace("+", "__"))
    training_args = _make_training_args(cfg, tmp_dir, seed)

    if method.name == "two_recon":
        _seed_everything(seed)
        model_cfg = TwoSidedTopKSAEConfig(
            hidden_size=d.representation_dim, latent_size=latent_size,
            k=t.k, normalize_decoder=True,
        )
        model = TwoSidedTopKSAE(model_cfg).to(device)
        trainer = TwoReconTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, data_collator=default_data_collator,
            callbacks=[SAENormCallback()],
        )
        trainer.train()
        return model.image_sae, model.text_sae, method_id, {}

    if method.name == "paired_aux_alignment":
        _seed_everything(seed)
        model_cfg = TwoSidedTopKSAEConfig(
            hidden_size=d.representation_dim, latent_size=latent_size,
            k=t.k, normalize_decoder=True,
        )
        model = TwoSidedTopKSAE(model_cfg).to(device)
        trainer = AuxAlignmentTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, data_collator=default_data_collator,
            variant_cfg=method,
            train_img=train_img, train_txt=train_txt,
        )
        trainer.add_callback(AuxAlignmentCallback(trainer))
        trainer.add_callback(SAENormCallback())
        trainer.train()
        return model.image_sae, model.text_sae, method_id, dict(trainer.diagnostics)

    raise ValueError(
        f"Unknown method '{method.name}' for aux-alignment runner. "
        f"Use 'two_recon' (recon baseline) or 'paired_aux_alignment'."
    )


def run_single(
    cfg: ExperimentConfig,
    alpha: float,
    latent_size: int,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    device = _resolve_device(cfg.training.device)
    d = cfg.data

    builder = SyntheticPairedBuilder(
        n_shared=d.n_shared, n_image=d.n_image, n_text=d.n_text,
        representation_dim=d.representation_dim, sparsity=d.sparsity,
        beta=d.beta, obs_noise_std=d.obs_noise_std,
        max_interference=d.max_interference,
        alpha_target=alpha, num_train=d.num_train, num_eval=d.num_eval,
        seed=seed,
    )
    ds = builder.build()
    train_ds = SyntheticPairedDataset(ds["train"]["image_representation"], ds["train"]["text_representation"])
    train_img = torch.from_numpy(ds["train"]["image_representation"])
    train_txt = torch.from_numpy(ds["train"]["text_representation"])
    eval_img = torch.from_numpy(ds["eval"]["image_representation"])
    eval_txt = torch.from_numpy(ds["eval"]["text_representation"])

    result: dict[str, Any] = {
        "seed": seed,
        "alpha_target": alpha,
        "latent_size": latent_size,
        "alpha_actual_mean": float(builder.mean_shared_cosine_similarity),
        "alpha_actual_std": float(builder.std_shared_cosine_similarity),
    }

    for method in cfg.methods:
        sae_i, sae_t, method_id, diagnostics = _train_method(
            method, cfg, train_ds, train_img, train_txt,
            latent_size, seed, run_dir, device,
        )
        metrics = evaluate_method(
            method=method.name,
            sae_i=sae_i, sae_t=sae_t,
            train_img=train_img, train_txt=train_txt,
            eval_img=eval_img, eval_txt=eval_txt,
            phi_S=builder.phi_S, psi_S=builder.psi_S,
            phi_I=builder.phi_I, psi_T=builder.psi_T,
            n_shared=d.n_shared, m_S=method.m_S,
            batch_size=cfg.training.batch_size, device=device,
        )
        if diagnostics:
            metrics["aux_alignment_diagnostics"] = diagnostics
        result[method_id] = metrics

        if cfg.output.save_decoders:
            _save_params(sae_i, sae_t, builder, alpha, seed, method_id, run_dir)

        if sae_i is not sae_t:
            del sae_i
        del sae_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def _cfg_to_dict(cfg: ExperimentConfig) -> dict:
    return {
        "data": asdict(cfg.data),
        "training": asdict(cfg.training),
        "methods": [asdict(m) for m in cfg.methods],
        "sweep": asdict(cfg.sweep),
        "output": asdict(cfg.output),
        "diagnostic": asdict(cfg.diagnostic),
    }


def run_experiment(cfg: ExperimentConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{cfg.output.run_tag}" if cfg.output.run_tag else ""
    run_name = f"run_{timestamp}{tag}"
    run_dir = Path(cfg.output.root) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", run_dir)

    all_results: list[dict[str, Any]] = []
    method_ids: list[str] = []

    for alpha in cfg.sweep.alpha:
        for latent_size in cfg.sweep.latent_size:
            seed_results: list[dict[str, Any]] = []
            for si in range(cfg.sweep.num_seeds):
                seed = cfg.sweep.seed_base + si
                logger.info("alpha=%.2f L=%d seed=%d", alpha, latent_size, seed)
                r = run_single(cfg, alpha, latent_size, seed, run_dir)
                seed_results.append(r)
                for k in r:
                    if k not in ("seed", "alpha_target", "latent_size",
                                 "alpha_actual_mean", "alpha_actual_std"):
                        if k not in method_ids:
                            method_ids.append(k)

            agg = aggregate(seed_results, method_ids)
            all_results.append({
                "alpha_target": alpha,
                "latent_size": latent_size,
                "aggregate": agg,
                "per_seed": seed_results,
            })

    with open(run_dir / "result.json", "w") as f:
        json.dump({"sweep_results": all_results, "config": _cfg_to_dict(cfg)},
                  f, indent=2, default=str)
    logger.info("Saved result.json to %s", run_dir / "result.json")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic aux-alignment ablation runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
