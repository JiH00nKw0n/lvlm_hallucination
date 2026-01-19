"""
FarSight: Upper triangular penalty register for attention.

Reference:
    - FarSight/Shell/farsight_patch.py

Key Implementation Notes:
    1. Uses upper triangular matrix as "penalty register"
    2. W = (QK^T/sqrt(d)) * C + P, where C is causal mask, P is penalty
    3. Â = softmax(W) * C to maintain causality
    4. ALiBi-style head-specific slopes optional
    5. Does NOT support KV cache (use_cache=False required)

Formula:
    P_basic = -(sigma * ReLU(j-i)) * upper_triangular
    W = attn_scores * causal_mask + P
    attn_probs = softmax(W) * causal_mask

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
"""

import math
import types
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, MitigatorConfig, ModelHelper


class FarSightMitigator(BaseMitigator):
    """
    FarSight: Upper triangular attention penalty.

    Reference: FarSight/Shell/farsight_patch.py

    Note: FarSight does NOT support KV cache. Generation will be slower.

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        target_layers: Layers to modify (default: all layers)
        decay_factor: Sigma for distance decay (default: log(256)/log(1024))
        use_alibi: Use ALiBi-style head slopes (default: True)
    """

    name: str = "farsight"

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava",
        target_layers: Optional[List[int]] = None,
        decay_factor: Optional[float] = None,
        use_alibi: bool = True,
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)

        num_layers = self._get_num_layers()
        self.target_layers = target_layers or list(range(num_layers))
        self.decay_factor = decay_factor or float(math.log(256) / math.log(1024))
        self.use_alibi = use_alibi

        self._original_forwards: List[Tuple[nn.Module, Callable]] = []

    def _switch_to_eager_attention(self) -> None:
        """Switch to eager attention."""
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, '_attn_implementation'):
                self.model.config._attn_implementation = 'eager'
            if hasattr(self.model.config, 'attn_implementation'):
                self.model.config.attn_implementation = 'eager'

    def _get_farsight_forward(self, layer_idx: int) -> Callable:
        """Get the appropriate FarSight forward based on model type."""
        if self.model_type in ('qwen2_vl', 'qwen2_5_vl'):
            return self._create_qwen_farsight_forward(layer_idx)
        else:
            return self._create_llama_farsight_forward(layer_idx)

    def _create_llama_farsight_forward(self, layer_idx: int) -> Callable:
        """
        Create FarSight attention forward for LLaMA-style models.

        Reference: FarSight/Shell/farsight_patch.py:7-126
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
            """
            FarSight: W = (QK^T/√d) ⊙ C + P, Â = softmax(W) ⊙ C
            """
            B, L, _ = hidden_states.size()
            dtype = hidden_states.dtype
            device = hidden_states.device

            # Q K V
            q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Causal mask
            C = torch.tril(torch.ones((L, L), device=device, dtype=dtype)).view(1, 1, L, L)

            # Upper triangular and distance
            idx = torch.arange(L, device=device)
            j_idx = idx.view(1, -1)
            i_idx = idx.view(-1, 1)
            delta = (j_idx - i_idx).to(dtype)
            upper = torch.triu(torch.ones((L, L), device=device, dtype=dtype), diagonal=1)

            # Basic linear decay P
            sigma = getattr(self, "sigma", mitigator.decay_factor)
            P_basic = -(sigma * F.relu(delta)) * upper

            # Progressive decay
            prog_factor = 1.0 - (i_idx.to(dtype) / max(L - 1, 1)) * 0.5
            P_prog = -(sigma * prog_factor) * F.relu(delta) * upper

            def _derive_valid_from_attention_mask(m: torch.Tensor) -> torch.Tensor:
                if m.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                    m_f = m.to(torch.float32)
                else:
                    m_f = m
                if m_f.dim() == 2 and m_f.shape[-1] == L:
                    valid = (m_f > 0).to(dtype)
                elif m_f.dim() == 3 and m_f.shape[-1] == L and m_f.shape[-2] == 1:
                    valid = (m_f.squeeze(-2) > 0).to(dtype)
                elif m_f.dim() >= 3 and m_f.shape[-1] == L and m_f.shape[-2] == L:
                    diag = torch.diagonal(m_f, dim1=-2, dim2=-1)
                    if diag.dim() == 3:
                        diag = diag.squeeze(-2)
                    valid = (diag > -1e8).to(dtype)
                else:
                    valid = torch.ones((B, L), device=device, dtype=dtype)
                return valid

            valid_1d = _derive_valid_from_attention_mask(attention_mask) if attention_mask is not None else torch.ones(
                (B, L), device=device, dtype=dtype
            )
            valid_cols = valid_1d.view(B, 1, 1, L)
            valid_rows = valid_1d.view(B, 1, L, 1)

            # Combine
            P_static = (0.5 * P_basic + 0.5 * P_prog).view(1, 1, L, L)
            P_static = (P_static * valid_cols * valid_rows).expand(B, self.num_heads, L, L)

            # ALiBi slopes
            if getattr(self, "farsight_use_alibi", mitigator.use_alibi):
                H = self.num_heads
                slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)], device=device, dtype=dtype)
                slopes = slopes.view(1, H, 1, 1)
                P_alibi = -(F.relu(delta) * upper).view(1, 1, L, L) * slopes
                P_alibi = (P_alibi * valid_cols * valid_rows).expand(B, self.num_heads, L, L)
                P_total = P_static + P_alibi
            else:
                P_total = P_static

            # Combine: W = attn_scores * C + P
            W = attn_scores * C + P_total

            # Softmax and apply causal mask again
            attn_probs = torch.softmax(W, dim=-1) * C

            # Output
            context = torch.matmul(attn_probs, v)
            context = context.transpose(1, 2).reshape(B, L, self.hidden_size)
            output = self.o_proj(context)

            attn_weights = attn_probs if output_attentions else None

            # LLaMA-style: return 3 values (output, attn_weights, past_key_value)
            return output, attn_weights, None

        return forward

    def _create_qwen_farsight_forward(self, layer_idx: int) -> Callable:
        """
        Create FarSight attention forward for Qwen2-VL models.

        Qwen2-VL attention returns 2 values instead of 3.
        Reference: transformers/models/qwen2_vl/modeling_qwen2_vl.py:491-541
        """
        mitigator = self

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ):
            """
            FarSight for Qwen2-VL: W = (QK^T/√d) ⊙ C + P, Â = softmax(W) ⊙ C
            """
            B, L, _ = hidden_states.size()
            dtype = hidden_states.dtype
            device = hidden_states.device

            # Q K V
            q = self.q_proj(hidden_states).view(B, L, -1, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, L, -1, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, L, -1, self.head_dim).transpose(1, 2)

            # Apply RoPE if position_embeddings provided
            if position_embeddings is not None:
                cos, sin = position_embeddings
                mrope_section = self.rope_scaling["mrope_section"] if hasattr(self, 'rope_scaling') and self.rope_scaling else None
                if mrope_section is not None:
                    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
                    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)

            # GQA
            num_q_heads = q.shape[1]
            num_kv_heads = k.shape[1]
            if num_kv_heads != num_q_heads:
                n_rep = num_q_heads // num_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)

            # Attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Causal mask
            C = torch.tril(torch.ones((L, L), device=device, dtype=dtype)).view(1, 1, L, L)

            # Upper triangular and distance
            idx = torch.arange(L, device=device)
            j_idx = idx.view(1, -1)
            i_idx = idx.view(-1, 1)
            delta = (j_idx - i_idx).to(dtype)
            upper = torch.triu(torch.ones((L, L), device=device, dtype=dtype), diagonal=1)

            # Basic linear decay P
            sigma = getattr(self, "sigma", mitigator.decay_factor)
            P_basic = -(sigma * F.relu(delta)) * upper

            # Progressive decay
            prog_factor = 1.0 - (i_idx.to(dtype) / max(L - 1, 1)) * 0.5
            P_prog = -(sigma * prog_factor) * F.relu(delta) * upper
            def _derive_valid_from_attention_mask(m: torch.Tensor) -> torch.Tensor:
                if m.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                    m_f = m.to(torch.float32)
                else:
                    m_f = m
                if m_f.dim() == 2 and m_f.shape[-1] == L:
                    valid = (m_f > 0).to(dtype)
                elif m_f.dim() == 3 and m_f.shape[-1] == L and m_f.shape[-2] == 1:
                    valid = (m_f.squeeze(-2) > 0).to(dtype)
                elif m_f.dim() >= 3 and m_f.shape[-1] == L and m_f.shape[-2] == L:
                    diag = torch.diagonal(m_f, dim1=-2, dim2=-1)
                    if diag.dim() == 3:
                        diag = diag.squeeze(-2)
                    valid = (diag > -1e8).to(dtype)
                else:
                    valid = torch.ones((B, L), device=device, dtype=dtype)
                return valid

            valid_1d = _derive_valid_from_attention_mask(attention_mask) if attention_mask is not None else torch.ones(
                (B, L), device=device, dtype=dtype
            )
            valid_cols = valid_1d.view(B, 1, 1, L)
            valid_rows = valid_1d.view(B, 1, L, 1)

            # Combine
            P_static = (0.5 * P_basic + 0.5 * P_prog).view(1, 1, L, L)
            P_static = (P_static * valid_cols * valid_rows).expand(B, num_q_heads, L, L)

            # ALiBi slopes
            if getattr(self, "farsight_use_alibi", mitigator.use_alibi):
                H = num_q_heads
                slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)], device=device, dtype=dtype)
                slopes = slopes.view(1, H, 1, 1)
                P_alibi = -(F.relu(delta) * upper).view(1, 1, L, L) * slopes
                P_alibi = (P_alibi * valid_cols * valid_rows).expand(B, num_q_heads, L, L)
                P_total = P_static + P_alibi
            else:
                P_total = P_static

            # Combine: W = attn_scores * C + P
            W = attn_scores * C + P_total

            # Softmax and apply causal mask again
            attn_probs = torch.softmax(W, dim=-1) * C

            # Output
            context = torch.matmul(attn_probs, v)
            context = context.transpose(1, 2).reshape(B, L, -1).contiguous()
            output = self.o_proj(context)

            attn_weights = attn_probs if output_attentions else None

            # Qwen2-VL: return 2 values (output, attn_weights)
            return output, attn_weights

        return forward

    def setup(self) -> None:
        """Replace attention forwards with FarSight versions."""
        if self.model.training:
            return
        self._switch_to_eager_attention()
        layers = self._get_layers()

        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                attn = ModelHelper.get_attention_module(layers[layer_idx])

                # Ensure num_heads is set
                if not hasattr(attn, 'num_heads'):
                    hidden_size = attn.q_proj.in_features
                    head_dim = getattr(attn, 'head_dim', 128)
                    attn.num_heads = hidden_size // head_dim
                    attn.head_dim = head_dim
                    attn.hidden_size = hidden_size

                attn.sigma = self.decay_factor
                attn.farsight_use_alibi = bool(self.use_alibi)

                self._original_forwards.append((attn, attn.forward))
                attn.forward = types.MethodType(self._get_farsight_forward(layer_idx), attn)

    def cleanup(self) -> None:
        """Restore original forwards."""
        for attn, original in self._original_forwards:
            attn.forward = original
        self._original_forwards.clear()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with FarSight.

        IMPORTANT: use_cache=False is enforced as FarSight doesn't support KV cache.
        """
        gen_kwargs = {
            'max_new_tokens': self.config.max_new_tokens,
            'do_sample': self.config.do_sample,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
            'use_cache': False,  # FarSight doesn't support KV cache
        }
        gen_kwargs.update(kwargs)
        gen_kwargs['use_cache'] = False  # Override any user setting

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
