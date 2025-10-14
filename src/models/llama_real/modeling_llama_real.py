# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Union, Callable

import torch
from torch import nn
from transformers import LlamaConfig, AutoModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, eager_mask, causal_mask_function
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

from src.integrations import ALL_ATTENTION_FUNCTIONS
from src.models.llama_real.configuration_llama_real import LLamaRealConfig
from src.models.reweighting_module.configuration_module import ReweightAttentionConfig
from src.models.reweighting_module.modeling_module import ReweightAttentionModule

logger = logging.get_logger(__name__)


@dataclass
class ReweightBaseModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base model output with reweight masks.

    Args:
        reweight_masks (`torch.FloatTensor` of shape `(num_layers, batch_size, num_heads, query_length, key_length)`, *optional*):
            Attention reweight masks from all decoder layers.
    """
    reweight_masks: Optional[torch.Tensor] = None


@dataclass
class ReweightCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Causal LM output with reweight masks.

    Args:
        reweight_masks (`torch.FloatTensor` of shape `(num_layers, batch_size, num_heads, query_length, key_length)`, *optional*):
            Attention reweight masks from all decoder layers.
    """
    reweight_masks: Optional[torch.Tensor] = None


class LlamaRealAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            config: LlamaConfig,
            additional_attention_module_config: ReweightAttentionConfig,
            layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.additional_attention_module_config = additional_attention_module_config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.reweight_attention = ReweightAttentionModule(self.additional_attention_module_config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
            self,
            input_ids: torch.Tensor,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            tensor_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        reweight_mask = self.reweight_attention(
            input_ids=input_ids,
            query_states=query_states,
            key_states=key_states,
            attention_mask=tensor_mask,
        )

        if self.config._attn_implementation != "flex_attention":
            raise ValueError("LLamaRealAttention only support `flex_attention`")
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            ## custom reweight_mask
            reweight_mask,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, reweight_mask


class LlamaRealDecoderLayer(GradientCheckpointingLayer):
    def __init__(
            self,
            config: LlamaConfig,
            additional_attention_module_config: ReweightAttentionConfig,
            layer_idx: int
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.additional_attention_module_config = additional_attention_module_config

        self.self_attn = LlamaRealAttention(
            config=config,
            additional_attention_module_config=additional_attention_module_config,
            layer_idx=layer_idx
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
            self,
            input_ids: torch.Tensor,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            tensor_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _, reweight_mask = self.self_attn(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            tensor_mask=tensor_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, reweight_mask


@auto_docstring
class LlamaRealPreTrainedModel(PreTrainedModel):
    config_class = LLamaRealConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaRealDecoderLayer,
        "attentions": LlamaRealAttention,
    }


@auto_docstring
class LlamaRealModel(LlamaRealPreTrainedModel):
    def __init__(self, config: LLamaRealConfig):
        text_config = config.text_config
        additional_attention_module_config = config.additional_attention_module_config
        super().__init__(text_config)
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size

        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaRealDecoderLayer(text_config, additional_attention_module_config, layer_idx) for layer_idx in
             range(text_config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Flex attention만 지원
        if self.config._attn_implementation != "flex_attention":
            raise NotImplementedError("LlamaRealModel only supports flex_attention")

        if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
            layer_idx = past_key_values.is_sliding.index(False)
        else:
            layer_idx = 0

        # If using a cache, it can give all information about mask sizes based on seen tokens
        if past_key_values is not None:
            kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
        # Otherwise, the sizes are simply the input sizes
        else:
            kv_length, kv_offset = inputs_embeds.shape[1], 0

        tensor_mask = eager_mask(
            batch_size=inputs_embeds.shape[0],
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=causal_mask_function,
            attention_mask=attention_mask.to(
                device=cache_position.device, dtype=torch.bool
            ) if attention_mask is not None and attention_mask.ndim == 2 else attention_mask,
            dtype=inputs_embeds.dtype,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Collect reweight masks from all layers
        all_reweight_masks = []  # List to store masks from each layer

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, reweight_mask = decoder_layer(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                tensor_mask=tensor_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # reweight_mask shape: (batch_size, num_heads, query_length, key_length)
            all_reweight_masks.append(reweight_mask)

        hidden_states = self.norm(hidden_states)

        # Stack all reweight masks: (num_layers, batch_size, num_heads, query_length, key_length)
        stacked_reweight_masks = torch.stack(all_reweight_masks, dim=0)

        return ReweightBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            reweight_masks=stacked_reweight_masks,
        )


@auto_docstring
class LlamaRealForCausalLM(LlamaRealPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: LLamaRealConfig):
        super().__init__(config)
        self.model = LlamaRealModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: ReweightBaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return ReweightCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reweight_masks=outputs.reweight_masks,
        )


__all__ = [
    "LlamaRealForCausalLM",
    "LlamaRealModel",
    "LlamaRealPreTrainedModel",
]

## AutoModel Register
AutoModel.register(LLamaRealConfig, LlamaRealModel)

## Register for auto class
LlamaRealModel.register_for_auto_class("AutoModel")
