from __future__ import annotations

from transformers import PretrainedConfig


class TopKSAEConfig(PretrainedConfig):
    """
    Configuration for the TopKSAE model.

    This mirrors the settings used in the multimodal-sae implementation, while
    presenting them in a Hugging Face `PretrainedConfig` so the model can be
    saved/loaded consistently with the Transformers ecosystem.
    """

    model_type = "topk_sae"

    def __init__(
        self,
        d_in: int,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        num_latents: int = 0,
        k: int = 32,
        multi_topk: bool = False,
        **kwargs,
    ):
        """
        Args:
            d_in: Input feature width of the activations to be autoencoded.
            expansion_factor: Multiplier for latent width when num_latents is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            num_latents: Explicit latent size; if 0, use d_in * expansion_factor.
            k: Number of non-zero latent activations (top-k) to keep per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(**kwargs)
        self.d_in = d_in
        self.expansion_factor = expansion_factor
        self.normalize_decoder = normalize_decoder
        self.num_latents = num_latents
        self.k = k
        self.multi_topk = multi_topk


class BatchTopKSAEConfig(TopKSAEConfig):
    """
    Configuration for the BatchTopKSAE model.

    BatchTopK uses a batch-level top-k selection. The same activation rule is
    applied in both training and evaluation.
    """

    model_type = "batch_topk_sae"

    def __init__(
        self,
        d_in: int,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        num_latents: int = 0,
        k: int = 32,
        multi_topk: bool = False,
        use_batch_topk_in_eval: bool = True,
        **kwargs,
    ):
        """
        Args:
            d_in: Input feature width of the activations to be autoencoded.
            expansion_factor: Multiplier for latent width when num_latents is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            num_latents: Explicit latent size; if 0, use d_in * expansion_factor.
            k: Target average number of active latents per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            use_batch_topk_in_eval: Kept for backward compatibility. BatchTopK
                always uses batch-level top-k regardless of this flag.
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            d_in=d_in,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            num_latents=num_latents,
            k=k,
            multi_topk=multi_topk,
            **kwargs,
        )
        self.use_batch_topk_in_eval = use_batch_topk_in_eval


class MatryoshkaSAEConfig(BatchTopKSAEConfig):
    """
    Configuration for MatryoshkaSAE.

    Matryoshka splits the dictionary into groups, and trains each prefix of
    groups to reconstruct well without access to later groups.
    """

    model_type = "matryoshka_sae"

    def __init__(
        self,
        d_in: int,
        group_sizes: list[int],
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        num_latents: int = 0,
        k: int = 32,
        multi_topk: bool = False,
        active_groups: int | None = None,
        **kwargs,
    ):
        """
        Args:
            d_in: Input feature width of the activations to be autoencoded.
            group_sizes: Sizes of each Matryoshka group (sum = total latents).
            expansion_factor: Multiplier for latent width when num_latents is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            num_latents: Explicit latent size; if 0, use sum(group_sizes).
            k: Target average number of active latents per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            active_groups: Number of prefix groups to activate (defaults to all).
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            d_in=d_in,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            num_latents=num_latents,
            k=k,
            multi_topk=multi_topk,
            **kwargs,
        )
        self.group_sizes = group_sizes
        self.active_groups = active_groups


class VLTopKSAEConfig(TopKSAEConfig):
    """
    Configuration for VLTopKSAE.

    The dictionary is split into [visual | shared | text] with a 1:2:1 ratio.
    """

    model_type = "vl_topk_sae"

    def __init__(
        self,
        d_in: int,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        num_latents: int = 0,
        k: int = 32,
        multi_topk: bool = False,
        **kwargs,
    ):
        """
        Args:
            d_in: Input feature width of the activations to be autoencoded.
            expansion_factor: Multiplier for latent width when num_latents is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            num_latents: Explicit latent size; if 0, use d_in * expansion_factor.
            k: Number of non-zero latent activations (top-k) to keep per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            d_in=d_in,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            num_latents=num_latents,
            k=k,
            multi_topk=multi_topk,
            **kwargs,
        )


class VLBatchTopKSAEConfig(BatchTopKSAEConfig):
    """
    Configuration for VLBatchTopKSAE.

    The dictionary is split into [visual | shared | text] with a 1:2:1 ratio.
    """

    model_type = "vl_batch_topk_sae"

    def __init__(
        self,
        d_in: int,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        num_latents: int = 0,
        k: int = 32,
        multi_topk: bool = False,
        **kwargs,
    ):
        """
        Args:
            d_in: Input feature width of the activations to be autoencoded.
            expansion_factor: Multiplier for latent width when num_latents is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            num_latents: Explicit latent size; if 0, use d_in * expansion_factor.
            k: Target average number of active latents per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            d_in=d_in,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            num_latents=num_latents,
            k=k,
            multi_topk=multi_topk,
            **kwargs,
        )


class VLMatryoshkaSAEConfig(MatryoshkaSAEConfig):
    """
    Configuration for VLMatryoshkaSAE.

    The dictionary is split into [visual | shared | text] with a 1:2:1 ratio,
    and Matryoshka prefix reconstruction losses are applied.
    """

    model_type = "vl_matryoshka_sae"

    def __init__(
        self,
        d_in: int,
        group_sizes: list[int],
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        num_latents: int = 0,
        k: int = 32,
        multi_topk: bool = False,
        active_groups: int | None = None,
        shared_group_sizes: list[int] | None = None,
        shared_active_groups: int | None = None,
        **kwargs,
    ):
        """
        Args:
            d_in: Input feature width of the activations to be autoencoded.
            group_sizes: Sizes of each Matryoshka group (sum = total latents).
            expansion_factor: Multiplier for latent width when num_latents is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            num_latents: Explicit latent size; if 0, use sum(group_sizes).
            k: Target average number of active latents per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            active_groups: Number of prefix groups to activate (defaults to all).
            shared_group_sizes: Sizes of shared-subspace Matryoshka groups. If None,
                defaults to a single shared group.
            shared_active_groups: Number of shared groups to activate (defaults to all).
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            d_in=d_in,
            group_sizes=group_sizes,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            num_latents=num_latents,
            k=k,
            multi_topk=multi_topk,
            active_groups=active_groups,
            **kwargs,
        )
        self.shared_group_sizes = shared_group_sizes
        self.shared_active_groups = shared_active_groups


__all__ = [
    "TopKSAEConfig",
    "BatchTopKSAEConfig",
    "MatryoshkaSAEConfig",
    "VLTopKSAEConfig",
    "VLBatchTopKSAEConfig",
    "VLMatryoshkaSAEConfig",
]
