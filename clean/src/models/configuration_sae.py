"""HF PretrainedConfig for TopKSAE / TwoSidedTopKSAE."""

from __future__ import annotations

from transformers import PretrainedConfig


class TopKSAEConfig(PretrainedConfig):
    model_type = "topk_sae"

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 8192,
        expansion_factor: int = 16,
        normalize_decoder: bool = True,
        k: int = 32,
        weight_tie: bool = False,
        k_aux: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.expansion_factor = expansion_factor
        self.normalize_decoder = normalize_decoder
        self.latent_size = latent_size
        self.k = k
        self.weight_tie = weight_tie
        self.k_aux = k_aux


class TwoSidedTopKSAEConfig(PretrainedConfig):
    """Two disjoint TopKSAEs (image / text). Each gets `latent_size // 2`."""
    model_type = "two_sided_topk_sae"

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 8192,
        k: int = 32,
        normalize_decoder: bool = True,
        k_aux: int | None = None,
        weight_tie: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if latent_size % 2 != 0:
            raise ValueError(f"latent_size must be even, got {latent_size}")
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.k_aux = k_aux
        self.weight_tie = weight_tie

    @property
    def latent_size_per_side(self) -> int:
        return self.latent_size // 2


__all__ = ["TopKSAEConfig", "TwoSidedTopKSAEConfig"]
