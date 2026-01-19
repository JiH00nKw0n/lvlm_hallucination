"""
Middle Layers: Boost image attention in middle transformer layers.

Reference:
    - middle_layers_indicating_hallucinations/modify_attention.py

Key Implementation Notes:
    1. Replaces attention forward to boost image token attention
    2. Model-specific attention implementations for LLaMA and Qwen2-VL
    3. Boost formula: attn[:,-1,img_start:img_end] += alpha * mean(attn[:,-1,img_start:img_end])

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
"""

import math
import types
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, MitigatorConfig, ModelHelper


class MiddleLayersMitigator(BaseMitigator):
    """
    Middle Layers: Boost attention to image tokens.

    Reference: middle_layers_indicating_hallucinations/modify_attention.py

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        target_layers: Layers to modify (default: 5-17 for middle layers)
        alpha: Boost scaling factor (default: 0.5)
        aggregation: How to compute boost - "mean" (default: "mean")
    """

    name: str = "middle_layers"

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava",
        target_layers: Optional[List[int]] = None,
        alpha: float = 0.5,
        aggregation: str = "mean",
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)

        # Default middle layers
        if target_layers is None:
            num_layers = self._get_num_layers()
            # Typically layers 5-17 for 32-layer models
            start = max(0, num_layers // 6)
            end = min(num_layers, num_layers // 2 + 2)
            self.target_layers = list(range(start, end))
        else:
            self.target_layers = target_layers

        self.alpha = alpha
        self.aggregation = aggregation

        # Store original forwards
        self._original_forwards: List[Tuple[nn.Module, Callable]] = []
        self._img_start: int = 0
        self._img_end: int = 0

    def _switch_to_eager_attention(self) -> None:
        """Switch Qwen models to eager attention (disable SDPA/Flash)."""
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, '_attn_implementation'):
                self.model.config._attn_implementation = 'eager'
            if hasattr(self.model.config, 'attn_implementation'):
                self.model.config.attn_implementation = 'eager'

    def _get_attention_forward(self, layer_idx: int) -> Callable:
        """
        Get the appropriate modified attention forward based on model type.
        """
        if self.model_type in ('qwen2_vl', 'qwen2_5_vl'):
            return self._create_qwen_attention_forward(layer_idx)
        else:
            return self._create_llama_attention_forward(layer_idx)

    def _create_llama_attention_forward(self, layer_idx: int) -> Callable:
        """
        Create modified attention forward for LLaMA-style models.

        Reference: middle_layers_indicating_hallucinations/modify_attention.py:10-110
        """
        mitigator = self

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Handle KV cache
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if hasattr(past_key_value, 'get_usable_length'):
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                else:
                    kv_seq_len += past_key_value[0].shape[-2]

            # RoPE
            if hasattr(self, 'rotary_emb'):
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                if hasattr(self, 'config') and hasattr(self.config, 'rope_scaling'):
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                else:
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # Update KV cache
            if past_key_value is not None:
                if hasattr(past_key_value, 'update'):
                    cache_kwargs = {"sin": sin, "cos": cos} if hasattr(self, 'rotary_emb') else {}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                else:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)

            # GQA: repeat KV heads
            if self.num_key_value_heads != self.num_heads:
                n_rep = self.num_heads // self.num_key_value_heads
                key_states = key_states.repeat_interleave(n_rep, dim=1)
                value_states = value_states.repeat_interleave(n_rep, dim=1)

            # Attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # === IMAGE ATTENTION BOOST ===
            img_start = mitigator._img_start
            img_end = mitigator._img_end
            if img_end > img_start and attn_weights.shape[-1] > img_start:
                actual_end = min(img_end, attn_weights.shape[-1])
                if mitigator.aggregation == "mean":
                    boost = mitigator.alpha * attn_weights[:, :, -1, img_start:actual_end].abs().mean(dim=1, keepdim=True)
                    attn_weights[:, :, -1, img_start:actual_end] = attn_weights[:, :, -1, img_start:actual_end] + boost

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            # Prepare cache output
            if use_cache:
                if hasattr(past_key_value, 'update'):
                    new_cache = past_key_value
                else:
                    new_cache = (key_states, value_states)
            else:
                new_cache = None

            return attn_output, (attn_weights if output_attentions else None), new_cache

        return forward

    def _create_qwen_attention_forward(self, layer_idx: int) -> Callable:
        """
        Create modified attention forward for Qwen2-VL style models.

        Handles 3D RoPE and GQA specific to Qwen architecture.

        Reference: transformers/models/qwen2_vl/modeling_qwen2_vl.py:491-541
        """
        mitigator = self

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,  # Changed: past_key_value -> past_key_values
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,  # Added: cache_position
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

            # Apply RoPE
            # Reference: modeling_qwen2_vl.py:513-516
            if position_embeddings is not None:
                cos, sin = position_embeddings
                # Qwen uses multimodal 3D RoPE with mrope_section
                from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
                # Get mrope_section from rope_scaling config
                mrope_section = self.rope_scaling["mrope_section"] if hasattr(self, 'rope_scaling') and self.rope_scaling else None
                if mrope_section is not None:
                    query_states, key_states = apply_multimodal_rotary_pos_emb(
                        query_states, key_states, cos, sin, mrope_section
                    )

            # KV cache
            # Reference: modeling_qwen2_vl.py:518-520
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} if position_embeddings else {}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # GQA: repeat KV heads
            # Reference: modeling_qwen2_vl.py (via repeat_kv function)
            num_kv_heads = key_states.shape[1]
            num_q_heads = query_states.shape[1]
            if num_kv_heads != num_q_heads:
                n_rep = num_q_heads // num_kv_heads
                key_states = key_states.repeat_interleave(n_rep, dim=1)
                value_states = value_states.repeat_interleave(n_rep, dim=1)

            # Attention
            scaling = self.head_dim ** -0.5
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # === IMAGE ATTENTION BOOST ===
            img_start = mitigator._img_start
            img_end = mitigator._img_end
            if img_end > img_start and attn_weights.shape[-1] > img_start:
                actual_end = min(img_end, attn_weights.shape[-1])
                if mitigator.aggregation == "mean":
                    boost = mitigator.alpha * attn_weights[:, :, -1, img_start:actual_end].abs().mean(dim=1, keepdim=True)
                    attn_weights[:, :, -1, img_start:actual_end] = attn_weights[:, :, -1, img_start:actual_end] + boost

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            # Return 2 values (Qwen2VLAttention returns attn_output, attn_weights)
            # Reference: modeling_qwen2_vl.py:541
            return attn_output, (attn_weights if output_attentions else None)

        return forward

    def setup(self) -> None:
        """Replace attention forwards with boosted versions."""
        # Switch Qwen to eager attention if needed
        if self.model_type in ('qwen2_vl', 'qwen2_5_vl'):
            self._switch_to_eager_attention()

        layers = self._get_layers()

        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                attn = ModelHelper.get_attention_module(layers[layer_idx])
                self._original_forwards.append((attn, attn.forward))

                new_forward = self._get_attention_forward(layer_idx)
                attn.forward = types.MethodType(new_forward, attn)

    def cleanup(self) -> None:
        """Restore original attention forwards."""
        for attn, original in self._original_forwards:
            attn.forward = original
        self._original_forwards.clear()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate with boosted image attention."""
        # Detect image token indices
        config = getattr(self.model, 'config', None)
        self._img_start, self._img_end = self._get_image_token_indices(input_ids, config)

        gen_kwargs = {
            'max_new_tokens': self.config.max_new_tokens,
            'do_sample': self.config.do_sample,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
        }
        gen_kwargs.update(kwargs)

        if pixel_values is not None:
            gen_kwargs['pixel_values'] = pixel_values

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
