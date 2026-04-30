"""Unified VLM encoder loader."""

from __future__ import annotations

import torch

from clean.src.encoders.base import Encoder
from clean.src.utils.config import ModelConfig


def load_encoder(cfg: ModelConfig, device: torch.device | str = "cuda") -> Encoder:
    if cfg.backend == "transformers":
        from clean.src.encoders.transformers_clip import TransformersEncoder
        return TransformersEncoder(cfg, device=device)
    if cfg.backend == "openclip":
        from clean.src.encoders.openclip import OpenCLIPEncoder
        return OpenCLIPEncoder(cfg, device=device)
    raise ValueError(f"Unknown encoder backend: {cfg.backend!r}")


__all__ = ["Encoder", "load_encoder"]
