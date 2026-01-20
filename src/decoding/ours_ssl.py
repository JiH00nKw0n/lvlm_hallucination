"""
Ours: Query-conditioned SAE gating for image key embeddings (LLaVA-NeXT).

Uses SAE activations from the last text token (query) to mask SAE activations
of image token keys before decoding them back to hidden space. Only image token
keys are modified; other tokens remain unchanged.
"""

import logging
import math
import time
import types
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, ModelHelper
from .ssl_utils import Sae

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class OursMitigator(BaseMitigator):
    """
    Query-conditioned SAE gating for image key embeddings.

    Args:
        model: The VLM model (LLaVA-NeXT only)
        model_type: llava_next
        sae_repo: HuggingFace repo id for SAE weights
        sae_hookpoint: Hookpoint subdir inside the SAE repo (e.g., "model.layers.24")
        sae_path: Optional local path to SAE layer directory (cfg.json + sae.safetensors)
        layer: Layer index to hook (default: 24)
    """

    name: str = "OursMitigator"

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava_next",
        sae_repo: Optional[str] = None,
        sae_hookpoint: Optional[str] = None,
        sae_path: Optional[str] = None,
        layer: int = 24,
        log_attn: bool = False,
        log_timing: bool = False,
        log_layer_timing: bool = True,
        log_layer_index: int = 1,
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.sae_repo = sae_repo
        self.sae_hookpoint = sae_hookpoint
        self.sae_path = sae_path
        self.layer = layer
        self.log_attn = log_attn
        self.log_timing = log_timing
        self.log_layer_timing = log_layer_timing
        self.log_layer_index = log_layer_index

        self._sae: Optional[Sae] = None
        self._img_start: int = 0
        self._img_end: int = 0
        self._attention_module: Optional[nn.Module] = None
        self._original_attn_forward: Optional[types.MethodType] = None
        self._layer_forward_overrides: list[tuple[nn.Module, types.MethodType]] = []

    def setup(self) -> None:
        if self.model_type != "llava_next":
            raise ValueError("OursMitigator currently supports llava_next only.")
        if self.log_timing or self.log_layer_timing or self.log_attn:
            logger.info(
                "OursMitigator setup: layer=%s log_timing=%s log_layer_timing=%s log_attn=%s",
                self.layer,
                self.log_timing,
                self.log_layer_timing,
                self.log_attn,
            )

        if self.sae_path:
            sae_dir = Path(self.sae_path)
            if not sae_dir.exists():
                raise FileNotFoundError(f"SAE path not found: {sae_dir}")
            sae = Sae.load_from_disk(str(sae_dir), device=self.model.device)
        else:
            if not self.sae_repo or not self.sae_hookpoint:
                raise ValueError("OursMitigator requires sae_repo and sae_hookpoint when sae_path is not set.")
            sae = Sae.load_from_hub(
                name=self.sae_repo,
                hookpoint=self.sae_hookpoint,
                device=self.model.device,
            )

        self._sae = sae.to(dtype=torch.float16)

        layers = self._get_layers()
        if self.layer >= len(layers) or self.layer < 0:
            raise ValueError(f"Layer index {self.layer} out of range (0-{len(layers)-1}).")

        attn_module = ModelHelper.get_attention_module(layers[self.layer])
        self._attention_module = attn_module
        self._original_attn_forward = attn_module.forward
        attn_module.forward = types.MethodType(self._build_attention_forward(self.layer), attn_module)

        if self.log_layer_timing:
            idx = self.log_layer_index
            if 0 <= idx < len(layers):
                layer = layers[idx]
                original_forward = layer.forward

                def _wrap_forward(module, layer_idx, forward_fn):
                    def wrapped_forward(*args, **kwargs):
                        start_layer = time.perf_counter()
                        output = forward_fn(*args, **kwargs)
                        elapsed_ms = (time.perf_counter() - start_layer) * 1000.0
                        step_idx = -1
                        cache_position = kwargs.get("cache_position")
                        if cache_position is not None:
                            try:
                                step_idx = int(cache_position[-1].item())
                            except Exception:
                                step_idx = -1
                        logger.info(
                            "OursMitigator layer_forward %s step %s time=%.3fms",
                            layer_idx,
                            step_idx,
                            elapsed_ms,
                        )
                        return output

                    return wrapped_forward

                layer.forward = types.MethodType(_wrap_forward(layer, idx, original_forward), layer)
                self._layer_forward_overrides.append((layer, original_forward))

    def cleanup(self) -> None:
        if self._attention_module is not None and self._original_attn_forward is not None:
            self._attention_module.forward = self._original_attn_forward
        self._attention_module = None
        self._original_attn_forward = None
        for layer, original_forward in self._layer_forward_overrides:
            layer.forward = original_forward
        self._layer_forward_overrides.clear()
        self._sae = None

    def _gate_image_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sae = self._sae
        if sae is None:
            return hidden_states

        img_start, img_end = self._img_start, self._img_end
        if img_end <= img_start or hidden_states.shape[1] <= img_end:
            return hidden_states

        with torch.no_grad():
            bsz, _, dim = hidden_states.shape
            img_hidden = hidden_states[:, img_start:img_end, :]
            if img_hidden.numel() == 0:
                return hidden_states

            query_hidden = hidden_states[:, -1, :]
            query_enc = sae.encode(query_hidden)
            query_indices = query_enc.top_indices  # [B, k]

            img_flat = img_hidden.reshape(-1, dim)
            img_enc = sae.encode(img_flat)
            img_top_acts = img_enc.top_acts
            img_top_indices = img_enc.top_indices  # [B*T, k]

            num_img_tokens = img_hidden.shape[1]
            batch_idx = torch.arange(bsz, device=hidden_states.device).repeat_interleave(num_img_tokens)
            query_per_token = query_indices[batch_idx]
            match = (img_top_indices.unsqueeze(-1) == query_per_token.unsqueeze(1)).any(-1)
            gate = torch.where(match, torch.ones_like(img_top_acts), torch.full_like(img_top_acts, 0.5))
            gated_acts = img_top_acts * gate

            decoded = sae.decode(gated_acts, img_top_indices)
            decoded = decoded.to(hidden_states.dtype).reshape(bsz, num_img_tokens, dim)

        gated_hidden = hidden_states.clone()
        gated_hidden[:, img_start:img_end, :] = decoded
        return gated_hidden

    def _build_attention_forward(self, layer_idx: int):
        mitigator = self

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[object] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            bsz, q_len, _ = hidden_states.size()

            if past_key_values is None and "past_key_value" in kwargs:
                past_key_values = kwargs["past_key_value"]

            timing_on = mitigator.log_timing
            start_total = time.perf_counter() if timing_on else 0.0
            start_gate = time.perf_counter() if timing_on else 0.0
            hidden_for_k = mitigator._gate_image_hidden(hidden_states)
            gate_ms = (time.perf_counter() - start_gate) * 1000.0 if timing_on else 0.0
            log_base = past_key_values is None and mitigator.log_attn

            start_proj = time.perf_counter() if timing_on else 0.0
            query_states = self.q_proj(hidden_states)
            key_states_base = self.k_proj(hidden_states) if log_base else None
            key_states = self.k_proj(hidden_for_k)
            value_states = self.v_proj(hidden_states)

            num_heads = getattr(self, "num_heads", None)
            num_kv_heads = getattr(self, "num_key_value_heads", num_heads)
            head_dim = getattr(self, "head_dim", None)
            if num_heads is None:
                raise ValueError("OursMitigator requires num_heads on attention module.")
            if head_dim is None:
                head_dim = self.q_proj.out_features // num_heads
            if num_kv_heads is None:
                num_kv_heads = num_heads

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            if log_base and key_states_base is not None:
                key_states_base = key_states_base.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            proj_ms = (time.perf_counter() - start_proj) * 1000.0 if timing_on else 0.0

            kv_seq_len = key_states.shape[-2]
            layer_idx_local = getattr(self, "layer_idx", layer_idx)
            if past_key_values is not None:
                if hasattr(past_key_values, "get_usable_length"):
                    kv_seq_len += past_key_values.get_usable_length(kv_seq_len, layer_idx_local)
                else:
                    past = past_key_values
                    if isinstance(past, tuple) and len(past) > 0 and isinstance(past[0], tuple):
                        past = past[0]
                    kv_seq_len += past[0].shape[-2]

            if hasattr(self, "rotary_emb"):
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                if log_base and key_states_base is not None:
                    query_states, key_states_base = apply_rotary_pos_emb(
                        query_states, key_states_base, cos, sin, position_ids
                    )
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_values is not None:
                if hasattr(past_key_values, "update"):
                    cache_kwargs = {"sin": sin, "cos": cos} if hasattr(self, "rotary_emb") else {}
                    if cache_position is not None:
                        cache_kwargs["cache_position"] = cache_position
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, layer_idx_local, cache_kwargs
                    )
                else:
                    key_states = torch.cat([past_key_values[0], key_states], dim=2)
                    value_states = torch.cat([past_key_values[1], value_states], dim=2)

            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                if log_base and key_states_base is not None:
                    key_states_base = key_states_base.repeat_interleave(n_rep, dim=1)
                key_states = key_states.repeat_interleave(n_rep, dim=1)
                value_states = value_states.repeat_interleave(n_rep, dim=1)

            start_attn = time.perf_counter() if timing_on else 0.0
            attn_scores_base = None
            if log_base and key_states_base is not None:
                attn_scores_base = torch.matmul(query_states, key_states_base.transpose(2, 3)) / math.sqrt(head_dim)
            attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            if attention_mask is not None:
                if attn_scores_base is not None:
                    attn_scores_base = attn_scores_base + attention_mask
                attn_scores = attn_scores + attention_mask
                if attn_scores_base is not None:
                    attn_scores_base = torch.max(
                        attn_scores_base,
                        torch.tensor(torch.finfo(attn_scores_base.dtype).min, device=attn_scores_base.device),
                    )
                attn_scores = torch.max(
                    attn_scores,
                    torch.tensor(torch.finfo(attn_scores.dtype).min, device=attn_scores.device),
                )

            attn_weights_base = None
            if attn_scores_base is not None:
                attn_weights_base = F.softmax(attn_scores_base, dim=-1, dtype=torch.float32)
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

            if attn_weights_base is not None and mitigator._img_end > mitigator._img_start:
                if cache_position is None or cache_position[0] == 0:
                    img_slice = slice(mitigator._img_start, mitigator._img_end)
                    base_sum = attn_weights_base[:, :, -1, img_slice].sum().item()
                    gated_sum = attn_weights[:, :, -1, img_slice].sum().item()
                    logger.info(
                        "OursMitigator layer %s image_attn_sum base=%.6f gated=%.6f",
                        layer_idx_local,
                        base_sum,
                        gated_sum,
                    )
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            attn_ms = (time.perf_counter() - start_attn) * 1000.0 if timing_on else 0.0

            if timing_on:
                step_idx = -1
                if cache_position is not None:
                    try:
                        step_idx = int(cache_position[-1].item())
                    except Exception:
                        step_idx = -1
                total_ms = (time.perf_counter() - start_total) * 1000.0
                logger.info(
                    "OursMitigator layer %s step %s timing gate=%.3fms proj=%.3fms attn=%.3fms total=%.3fms",
                    layer_idx_local,
                    step_idx,
                    gate_ms,
                    proj_ms,
                    attn_ms,
                    total_ms,
                )

            return attn_output, (attn_weights if output_attentions else None)

        return forward

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.log_timing or self.log_layer_timing or self.log_attn:
            logger.info("OursMitigator generate: input_ids shape=%s", tuple(input_ids.shape))
        if self._img_end == 0:
            config = getattr(self.model, "config", None)
            img_start, img_end = self._get_image_token_indices(input_ids, config)
            self._img_start = img_start
            self._img_end = img_end

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "use_cache": True,
        }
        gen_kwargs.update(kwargs)

        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
