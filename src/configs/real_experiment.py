"""Experiment configuration for real-data (CC3M/COCO-trained) SAE pipelines.

YAML-driven mirror of ``src.configs.experiment.ExperimentConfig``. One yaml
defines: training source (cc3m | coco | imagenet), training hyperparams,
list of methods (shared / separated / iso_align / group_sparse / ours),
and an unordered list of evaluations (dataset + tasks subset).

Load with:

    cfg = RealExperimentConfig.from_yaml("configs/real/cc3m.yaml")
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _coerce_floats(d: dict[str, Any]) -> dict[str, Any]:
    """YAML parses ``5e-4`` as str; coerce numeric-looking strings to float."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


_VALID_TASKS = {"recon", "retrieval", "zeroshot_raw", "dead_latents"}
_VALID_DATASETS = {"cc3m", "coco", "imagenet"}
_VALID_METHODS = {"shared", "separated", "iso_align", "group_sparse", "ours"}
_VALID_SAE_CLASSES = {"topk", "batch_topk"}


@dataclass
class DataConfig:
    dataset: str                         # cc3m | coco | imagenet
    cache_dir: str
    split_train: str = "train"
    split_eval: str = "test"
    eval_samples: int = 0                # 0 = use split_eval as-is; >0 = tail slice from split_train
    max_per_class: int = 1000            # imagenet-only


@dataclass
class TrainingConfig:
    lr: float = 5e-4
    num_epochs: int = 30
    batch_size: int = 1024
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    k: int = 32
    latent_size: int = 8192
    hidden_size: int = 512
    seed: int = 0
    dataloader_num_workers: int = 2
    device: str = "cuda"


@dataclass
class MethodConfig:
    name: str                            # shared | separated | iso_align | group_sparse | ours
    aux_weight: float = 0.0              # iso_align (β) / group_sparse (λ)
    base_method: str = ""                # ours: "separated" (reuses ckpt)
    sae_class: str = "topk"              # "topk" (default) | "batch_topk"


@dataclass
class EvalSpec:
    dataset: str                         # cc3m | coco | imagenet
    cache_dir: str
    split: str = "test"
    tasks: list[str] = field(default_factory=list)
    max_per_class: int = 1000            # imagenet-only


@dataclass
class OutputConfig:
    root: str


@dataclass
class RealExperimentConfig:
    name: str
    data: DataConfig
    training: TrainingConfig
    methods: list[MethodConfig]
    evaluations: list[EvalSpec]
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> RealExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> RealExperimentConfig:
        name = str(raw.get("name", "real_exp"))

        data_raw = raw.get("data") or {}
        if "dataset" not in data_raw or "cache_dir" not in data_raw:
            raise ValueError("config.data must define both 'dataset' and 'cache_dir'")
        data = DataConfig(**_coerce_floats(data_raw))
        if data.dataset not in _VALID_DATASETS:
            raise ValueError(f"data.dataset={data.dataset!r} not in {_VALID_DATASETS}")

        training = TrainingConfig(**_coerce_floats(raw.get("training", {})))

        methods = [MethodConfig(**_coerce_floats(m)) for m in raw.get("methods", [])]
        for m in methods:
            if m.name not in _VALID_METHODS:
                raise ValueError(f"method.name={m.name!r} not in {_VALID_METHODS}")
            if m.sae_class not in _VALID_SAE_CLASSES:
                raise ValueError(f"method.sae_class={m.sae_class!r} not in {_VALID_SAE_CLASSES}")

        evals_raw = raw.get("evaluations", [])
        evaluations: list[EvalSpec] = []
        for ev in evals_raw:
            ev_c = _coerce_floats(ev)
            if "tasks" not in ev_c or not ev_c["tasks"]:
                raise ValueError(f"eval {ev_c!r} must define non-empty 'tasks'")
            for t in ev_c["tasks"]:
                if t not in _VALID_TASKS:
                    raise ValueError(f"task={t!r} not in {_VALID_TASKS}")
            if ev_c.get("dataset") not in _VALID_DATASETS:
                raise ValueError(f"eval.dataset={ev_c.get('dataset')!r} not in {_VALID_DATASETS}")
            evaluations.append(EvalSpec(**ev_c))

        out_raw = raw.get("output") or {}
        if "root" not in out_raw:
            raise ValueError("config.output.root is required")
        output = OutputConfig(**out_raw)

        return cls(
            name=name,
            data=data,
            training=training,
            methods=methods,
            evaluations=evaluations,
            output=output,
        )


__all__ = [
    "DataConfig",
    "TrainingConfig",
    "MethodConfig",
    "EvalSpec",
    "OutputConfig",
    "RealExperimentConfig",
]
