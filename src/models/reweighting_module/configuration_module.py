from typing import Tuple, List, Union

from transformers import PretrainedConfig
from transformers.models.auto import AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

REWEIGHT_ATTENTION_MODULE_TYPE = "reweight_attention"


class ReweightAttentionConfig(PretrainedConfig):
    r"""
    Configuration class for ReweightAttentionModule.

    This module reweights attention scores by computing block-level pooling of attention weights
    and applying learnable transformations. It supports image and assistant token detection for
    structured block boundary detection.

    Args:
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in the base model.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key-value heads for Grouped Query Attention (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        rank_dim (`int`, *optional*, defaults to 32):
            Low-rank dimension for Q/K projections (A and B matrices).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the low-rank projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        image_token_id (`int`, *optional*, defaults to 32000):
            Token ID representing image tokens for block boundary detection.
        assistant_token_ids (`list[int]`, *optional*, defaults to [22933, 9047, 13566, 29901]):
            Token IDs representing the assistant prompt (e.g., "ASSISTANT:") for block boundary detection.
        alpha_std (`float`, *optional*, defaults to 0.02):
            Standard deviation for initializing the learnable alpha scaling parameter.
        implementation_type (`str`, *optional*, defaults to "max_pool"):
            Pooling method for block attention aggregation. Options: "max_pool", "mean_pool".

    Example:
        ```python
        >>> from src.models.reweighting_module import ReweightAttentionConfig, ReweightAttentionModule
        >>> config = ReweightAttentionConfig(num_attention_heads=32, rank_dim=64)
        >>> module = ReweightAttentionModule(config)
        ```
    """
    model_type = REWEIGHT_ATTENTION_MODULE_TYPE
    _name_or_path = "reweight_attention"

    # Tensor parallel plan for reweight attention module
    reweight_module_tp_plan = {
        "reweight_attention.q_proj_a": "colwise",  # (num_heads*head_dim, num_heads*rank_dim)
        "reweight_attention.q_proj_b": "rowwise",  # (num_heads*rank_dim, num_heads*head_dim)
        "reweight_attention.k_proj_a": "colwise",  # (num_heads*head_dim, num_heads*rank_dim)
        "reweight_attention.k_proj_b": "rowwise",  # (num_heads*rank_dim, num_heads*head_dim)
    }

    def __init__(
            self,
            num_attention_heads: int = 32,
            num_key_value_heads: int = 32,
            head_dim: int = 128,
            rank_dim: int = 16,
            attention_bias: bool = False,
            attention_dropout: float = 0.0,
            image_token_id: int = 32000,
            assistant_token_ids: Union[List[int], Tuple[int]] = (22933, 9047, 13566, 29901),
            alpha_std: float = 0.02,
            implementation_type: str = "mean_pool",
            **kwargs,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rank_dim = rank_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.image_token_id = image_token_id
        self.assistant_token_ids = list(assistant_token_ids)
        self.alpha_std = alpha_std
        self.implementation_type = implementation_type

        super().__init__(**kwargs)


__all__ = [
    "ReweightAttentionConfig",
    "REWEIGHT_ATTENTION_MODULE_TYPE"
]

AutoConfig.register(REWEIGHT_ATTENTION_MODULE_TYPE, ReweightAttentionConfig)

ReweightAttentionConfig.register_for_auto_class()
