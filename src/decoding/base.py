"""
Base classes and utilities for hallucination mitigation methods.

Supports 4 VLM architectures:
- LLaVA (llava)
- LLaVA-NeXT (llava_next)
- Qwen2-VL (qwen2_vl)
- Qwen2.5-VL (qwen2_5_vl)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

TensorDict = Dict[str, torch.Tensor]
ModelKwargs = Dict[str, Any]


# =============================================================================
# Model Architecture Helpers
# =============================================================================

class ModelHelper:
    """
    Unified helper for accessing model components across different VLM architectures.

    Supported models:
        - LLaVA: model.language_model.model.layers
        - LLaVA-NeXT: model.language_model.model.layers
        - Qwen2-VL: model.model.language_model.layers
        - Qwen2.5-VL: model.model.language_model.layers
    """

    # Image kwarg names for each model (all HF models use pixel_values)
    IMAGE_KWARG_MAP = {
        'llava': 'pixel_values',
        'llava_next': 'pixel_values',
        'qwen2_vl': 'pixel_values',
        'qwen2_5_vl': 'pixel_values',
    }

    # Default image token positions
    DEFAULT_IMAGE_TOKENS = {
        'llava': (35, 611),  # 576 patches
        'llava_next': (35, 611),
        'qwen2_vl': (10, 266),  # ~256 patches, variable
        'qwen2_5_vl': (10, 266),
    }

    @staticmethod
    def normalize_model_type(model_type: str) -> str:
        """Normalize model type string."""
        return model_type.lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def get_layers(model: nn.Module, model_type: str = "llava") -> nn.ModuleList:
        """
        Get transformer layers from different model architectures.

        Args:
            model: The VLM model
            model_type: Model type string

        Returns:
            nn.ModuleList of transformer layers

        Model structures:
            - Qwen2-VL: model.model.language_model.layers
            - LLaVA/LLaVA-NeXT: model.language_model.model.layers
        """
        model_type = ModelHelper.normalize_model_type(model_type)
        model_name = type(model).__name__.lower()

        # Qwen2-VL / Qwen2.5-VL: model.model.language_model.layers
        if 'qwen' in model_name or model_type in ('qwen2_vl', 'qwen2_5_vl'):
            # Qwen2VLForConditionalGeneration structure:
            # - model.model = Qwen2VLModel
            # - model.model.language_model = Qwen2VLTextModel (has .layers)
            if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
                lm = model.model.language_model
                if hasattr(lm, 'layers'):
                    return lm.layers
            # Fallback: model.model.layers
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return model.model.layers

        # LLaVA / LLaVA-NeXT: language_model.model.layers
        if hasattr(model, 'language_model'):
            lm = model.language_model
            # Decoder-only (LLaMA-style)
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                return lm.model.layers
            # Direct layers
            if hasattr(lm, 'layers'):
                return lm.layers

        # Direct model.model.layers (LLaMA-style)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers

        # GPT-style: model.transformer.h
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h

        raise ValueError(f"Cannot find layers in model: {type(model)}")

    @staticmethod
    def get_num_layers(model: nn.Module, model_type: str = "llava") -> int:
        """Get the number of transformer layers."""
        return len(ModelHelper.get_layers(model, model_type))

    @staticmethod
    def get_embed_tokens(model: nn.Module, model_type: str = "llava") -> nn.Module:
        """
        Get the token embedding layer.

        Args:
            model: The VLM model
            model_type: Model type string

        Returns:
            Token embedding layer
        """
        model_type = ModelHelper.normalize_model_type(model_type)

        # Method 1: get_input_embeddings()
        if hasattr(model, 'get_input_embeddings'):
            embed = model.get_input_embeddings()
            if embed is not None:
                return embed

        model_name = type(model).__name__.lower()

        # Qwen2-VL / Qwen2.5-VL
        if 'qwen' in model_name or model_type in ('qwen2_vl', 'qwen2_5_vl'):
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                return model.model.embed_tokens

        # LLaVA / LLaVA-NeXT
        if hasattr(model, 'language_model'):
            lm = model.language_model
            if hasattr(lm, 'get_input_embeddings'):
                return lm.get_input_embeddings()
            if hasattr(lm, 'model') and hasattr(lm.model, 'embed_tokens'):
                return lm.model.embed_tokens
            if hasattr(lm, 'embed_tokens'):
                return lm.embed_tokens

        # Direct embed_tokens
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            return model.model.embed_tokens

        raise ValueError(f"Cannot find embed_tokens in model: {type(model)}")

    @staticmethod
    def get_norm_and_lm_head(model: nn.Module, model_type: str = "llava") -> Tuple[nn.Module, nn.Module]:
        """
        Get the final norm layer and lm_head for early exit logits.

        Args:
            model: The VLM model
            model_type: Model type string

        Returns:
            Tuple of (norm, lm_head)

        Model structures:
            - Qwen2-VL: model.model.language_model.norm, model.lm_head
            - LLaVA/LLaVA-NeXT: model.language_model.model.norm, model.language_model.lm_head
        """
        model_type = ModelHelper.normalize_model_type(model_type)
        model_name = type(model).__name__.lower()

        # Qwen2-VL / Qwen2.5-VL: model.model.language_model.norm
        if 'qwen' in model_name or model_type in ('qwen2_vl', 'qwen2_5_vl'):
            # Qwen2VLForConditionalGeneration structure:
            # - model.model = Qwen2VLModel
            # - model.model.language_model = Qwen2VLTextModel (has .norm)
            # - model.lm_head = lm_head
            if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
                lm = model.model.language_model
                if hasattr(lm, 'norm'):
                    return lm.norm, model.lm_head
            # Fallback: try model.model.norm
            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                return model.model.norm, model.lm_head

        # LLaVA / LLaVA-NeXT: model.language_model.model.norm
        if hasattr(model, 'language_model'):
            lm = model.language_model
            # Decoder-only (LLaMA-based): lm.model.norm, lm.lm_head
            if hasattr(lm, 'model') and hasattr(lm.model, 'norm'):
                if hasattr(lm, 'lm_head'):
                    return lm.model.norm, lm.lm_head
                if hasattr(model, 'lm_head'):
                    return lm.model.norm, model.lm_head
                return lm.model.norm, lm
            # Direct norm on language_model
            if hasattr(lm, 'norm'):
                if hasattr(lm, 'lm_head'):
                    return lm.norm, lm.lm_head
                if hasattr(model, 'lm_head'):
                    return lm.norm, model.lm_head
                return lm.norm, lm

        # Direct model.model.norm (LLaMA-style standalone)
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            return model.model.norm, model.lm_head

        raise ValueError(f"Cannot find norm/lm_head in model: {type(model)}")

    @staticmethod
    def get_vision_encoder(model: nn.Module, model_type: str = "llava") -> nn.Module:
        """
        Get the vision encoder from the model.

        Args:
            model: The VLM model
            model_type: Model type string

        Returns:
            Vision encoder module
        """
        model_type = ModelHelper.normalize_model_type(model_type)

        # LLaVA / LLaVA-NeXT
        if model_type in ('llava', 'llava_next'):
            if hasattr(model, 'vision_tower'):
                return model.vision_tower
            if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
                return model.model.vision_tower

        # Qwen2-VL / Qwen2.5-VL
        if model_type in ('qwen2_vl', 'qwen2_5_vl'):
            if hasattr(model, 'visual'):
                return model.visual
            if hasattr(model, 'model') and hasattr(model.model, 'visual'):
                return model.model.visual

        raise ValueError(f"Cannot find vision encoder in {model_type} model: {type(model)}")

    @staticmethod
    def get_attention_module(layer: nn.Module) -> nn.Module:
        """Get the self-attention module from a transformer layer."""
        for attr in ['self_attn', 'attention', 'attn']:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError(f"Cannot find attention module in layer: {type(layer)}")

    @staticmethod
    def get_mlp_module(layer: nn.Module) -> nn.Module:
        """Get the MLP/FFN module from a transformer layer."""
        for attr in ['mlp', 'feed_forward', 'ffn', 'ff']:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError(f"Cannot find MLP module in layer: {type(layer)}")

    @staticmethod
    def get_image_kwarg_name(model_type: str = "llava") -> str:
        """Get the correct image keyword argument name for this model."""
        model_type = ModelHelper.normalize_model_type(model_type)
        return ModelHelper.IMAGE_KWARG_MAP.get(model_type, 'pixel_values')


def get_image_token_indices(
        input_ids: torch.Tensor,
        model_type: str = "llava",
        config: Optional[PretrainedConfig] = None,
        image_token_id: int = 32000,
) -> Tuple[int, int]:
    """
    Get start and end indices of image tokens.

    Args:
        input_ids: Input token IDs tensor [B, seq_len]
        model_type: Model type - llava, llava_next, qwen2_vl, qwen2_5_vl
        config: Optional model config to get token IDs dynamically
        image_token_id: Token ID for image placeholder (default: 32000 for LLaVA)

    Returns:
        (start_idx, end_idx): Indices where image tokens are located.
    """
    model_type = ModelHelper.normalize_model_type(model_type)

    if model_type in ("llava", "llava_next"):
        # LLaVA/LLaVA-NeXT: image tokens are marked with image_token_id
        if config is not None and hasattr(config, 'image_token_index'):
            image_token_id = config.image_token_index
        elif config is not None and hasattr(config, 'image_token_id'):
            image_token_id = config.image_token_id

        mask = (input_ids[0] == image_token_id)
        if mask.any():
            indices = torch.where(mask)[0]
            return indices[0].item(), indices[-1].item() + 1
        else:
            return ModelHelper.DEFAULT_IMAGE_TOKENS[model_type]

    elif model_type in ("qwen2_vl", "qwen2_5_vl"):
        # Qwen2-VL: <|vision_start|> ... <|vision_end|>
        vision_start_id = 151652
        vision_end_id = 151653

        if config is not None:
            vision_start_id = getattr(config, 'vision_start_token_id', vision_start_id)
            vision_end_id = getattr(config, 'vision_end_token_id', vision_end_id)

        start_mask = (input_ids[0] == vision_start_id)
        end_mask = (input_ids[0] == vision_end_id)

        if start_mask.any() and end_mask.any():
            start_indices = torch.where(start_mask)[0]
            end_indices = torch.where(end_mask)[0]
            start_idx = start_indices[0].item() + 1
            end_idx = end_indices[-1].item()
            return start_idx, end_idx
        else:
            return ModelHelper.DEFAULT_IMAGE_TOKENS[model_type]

    else:
        # Fallback: try to find image_token_id
        mask = (input_ids[0] == image_token_id)
        if mask.any():
            indices = torch.where(mask)[0]
            return indices[0].item(), indices[-1].item() + 1
        raise ValueError(f"Unknown model_type: {model_type}")


# =============================================================================
# DDPM Noise Utilities (for VCD)
# =============================================================================

def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int = 500) -> torch.Tensor:
    """
    Add DDPM-style diffusion noise to image.

    Reference: VCD/vcd_utils/vcd_add_noise.py

    Args:
        image_tensor: Image tensor of shape (B, C, H, W) or (C, H, W)
        noise_step: Diffusion step (0-999), higher = more noise

    Returns:
        Noised image tensor with same shape
    """
    num_steps = 1000
    device = image_tensor.device

    # Sigmoid schedule for betas (VCD-specific)
    betas = torch.linspace(-6, 6, num_steps, device=device)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

    # Add noise at specified step
    noise = torch.randn_like(image_tensor)
    alpha_t = alphas_bar_sqrt[noise_step]
    alpha_1_m_t = one_minus_alphas_bar_sqrt[noise_step]

    noisy_image = alpha_t * image_tensor + alpha_1_m_t * noise

    return noisy_image


# =============================================================================
# Sampling Utilities
# =============================================================================

def sample_top_p(
        logits: torch.Tensor,
        top_p: float = 0.9,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """
    Top-k + top-p (nucleus) sampling.

    Args:
        logits: Logits tensor of shape (B, vocab_size)
        top_p: Cumulative probability threshold (0-1)
        temperature: Sampling temperature
        top_k: Top-k filter size (None or <=0 disables)
        filter_value: Value to replace filtered logits with

    Returns:
        Sampled token IDs of shape (B, 1)
    """
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_values, filter_value)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, filter_value)

    # Sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


# =============================================================================
# Base Configuration
# =============================================================================

@dataclass
class MitigatorConfig:
    """Base configuration for mitigators."""

    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


# =============================================================================
# Base Mitigator Classes
# =============================================================================

class BaseMitigator(ABC):
    """
    Abstract base class for all hallucination mitigation methods.

    Usage:
        with SomeMitigator(model, model_type="llava", **config) as mitigator:
            result = mitigator.generate(input_ids, pixel_values=pixel_values)
    """

    name: str = "base"
    requires_training: bool = False

    def __init__(
            self,
            model: nn.Module,
            model_type: str = "llava",
            config: Optional[MitigatorConfig] = None,
            **kwargs,
    ):
        self.model = model
        self.model_type = ModelHelper.normalize_model_type(model_type)
        self.config = config or MitigatorConfig(
            **{
                k: v for k, v in kwargs.items()
                if k in MitigatorConfig.__dataclass_fields__
            }
            )
        self.is_active = False
        self._original_training_mode = None

    @abstractmethod
    def setup(self) -> None:
        """Apply intervention to the model. Called on __enter__."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Restore model to original state. Called on __exit__."""
        pass

    @abstractmethod
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """Generate tokens with the mitigation applied."""
        pass

    def __enter__(self):
        self._original_training_mode = self.model.training
        self.model.eval()
        self.setup()
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        self.is_active = False
        if self._original_training_mode:
            self.model.train()
        return False

    # Convenience methods using ModelHelper
    def _get_layers(self) -> nn.ModuleList:
        return ModelHelper.get_layers(self.model, self.model_type)

    def _get_num_layers(self) -> int:
        return ModelHelper.get_num_layers(self.model, self.model_type)

    def _get_embed_tokens(self) -> nn.Module:
        return ModelHelper.get_embed_tokens(self.model, self.model_type)

    def _get_norm_and_lm_head(self) -> Tuple[nn.Module, nn.Module]:
        return ModelHelper.get_norm_and_lm_head(self.model, self.model_type)

    def _get_vision_encoder(self) -> nn.Module:
        return ModelHelper.get_vision_encoder(self.model, self.model_type)

    def _get_image_kwarg_name(self) -> str:
        return ModelHelper.get_image_kwarg_name(self.model_type)

    def _get_image_token_indices(
            self,
            input_ids: torch.Tensor,
            config: Optional[PretrainedConfig] = None,
    ) -> Tuple[int, int]:
        return get_image_token_indices(
            input_ids,
            model_type=self.model_type,
            config=config or getattr(self.model, 'config', None),
        )


class TrainableMitigator(BaseMitigator):
    """Base class for mitigators that require training."""

    requires_training: bool = True

    @abstractmethod
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return list of parameters to train."""
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """Compute training loss."""
        pass

    @abstractmethod
    def save_pretrained(self, path: str) -> None:
        """Save trained module."""
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model: nn.Module, path: str, **kwargs) -> "TrainableMitigator":
        """Load trained module."""
        pass
