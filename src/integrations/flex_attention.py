"""
Partially inspired by torchtune's flex attention implementation

Citation:
@software{torchtune,
  title = {torchtune: PyTorch's finetuning library},
  author = {torchtune maintainers and contributors},
  url = {https//github.com/pytorch/torchtune},
  license = {BSD-3-Clause},
  month = apr,
  year = {2024}
}
"""
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

from typing import Optional, Union

import torch
from packaging import version
from transformers.utils import is_torch_flex_attn_available, logging
from transformers.utils.import_utils import _torch_version, is_torch_less_or_equal, is_torchdynamo_compiling

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

logger = logging.get_logger(__name__)


class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self, training):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled or training != self.training:
            self.training = training
            if is_torch_less_or_equal("2.5.1"):
                self._compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
            # In PyTorch 2.6.0, there's a known issue with flex attention compilation which may
            # cause errors. The suggested fix is to compile with "max-autotune-no-cudagraphs"
            # see https://github.com/pytorch/pytorch/issues/146260 for training
            elif version.parse(_torch_version).base_version == "2.6.0" and training:
                self._compiled_flex_attention = torch.compile(
                    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
                )
            # Fallback, usually the most recent torch 2.7.x+ versions
            else:
                self._compiled_flex_attention = torch.compile(flex_attention)

            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def compile_friendly_flex_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        training=False,
        **kwargs,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    flex_attention_compiled = WrappedFlexAttention(training)() if not is_torchdynamo_compiling() else flex_attention
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


Offset = Union[torch.Tensor, int]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def flex_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        reweight_mask: Union[torch.Tensor],
        attention_mask: Union[torch.Tensor, "BlockMask"],
        scaling: Optional[float] = None,
        softcap: Optional[float] = None,
        s_aux: Optional[torch.Tensor] = None,
        **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if kwargs.get("dropout", 0.0) > 0:
        raise ValueError(
            "`flex_attention` does not support `dropout`. Please use it with inference"
            " only (`model.eval()`) or turn off the attention dropout in the respective config."
        )

    block_mask = None
    score_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask
    else:
        score_mask = attention_mask

    if score_mask is not None:
        score_mask = score_mask[:, :, :, : key.shape[-2]]
    
    if reweight_mask is not None:
        reweight_mask = reweight_mask[:, :, :, : key.shape[-2]]
    
    ### CUSTOM CODE START: MODULE FOR REWEIGHTING ATTENTION LAYER OUTPUTS ###
    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if score_mask is not None:
            score = score + score_mask[batch_idx][0][q_idx][kv_idx]
        # Note: attention sinks cannot be correctly implemented in score_mod
        # because it requires operating on the full attention matrix before softmax.
        # ==> this is done after flex attention
        if reweight_mask is not None:
            score = score + reweight_mask[batch_idx][head_idx][q_idx][kv_idx]
        return score
    ### CUSTOM CODE END: MODULE FOR REWEIGHTING ATTENTION LAYER OUTPUTS ###

    enable_gqa = True
    num_local_query_heads = query.shape[1]

    # When running TP this helps:
    if (num_local_query_heads & (num_local_query_heads - 1)) != 0:
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    kernel_options = kwargs.get("kernel_options")
    # On CPU we must skip returning LSE due to a runtime issue; elsewhere, follow PyTorch API and return it
    return_lse = query.device.type != "cpu"

    if not return_lse and s_aux is not None:
        raise ValueError(
            "Attention sinks cannot be run on CPU with flex attention. Please switch to a different device, e.g. CUDA"
        )

    flex_attention_output = compile_friendly_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=return_lse,
        training=module.training,
    )
    # lse is returned in float32
    if return_lse:
        attention_output, lse = flex_attention_output  # type: ignore[misc]
        lse = lse.to(value.dtype)

        if s_aux is not None:
            # Apply attention sinks by renormalizing using LSE
            batch_size, num_heads, seq_len_q, _ = attention_output.shape  # batch, num_heads, seq_len, head_dim
            sinks = s_aux.view(1, -1, 1, 1).expand(batch_size, num_heads, seq_len_q, 1)

            # We need to compute the normalization that includes the sinks
            # since log(sum(exp(scores))) = lse, exp(log(sum(exp(scores)))) = exp(lse)
            # NB: log(sum(exp(scores)) + exp(sink)) = log(exp(lse) + exp(sink))
            lse_expanded = lse.unsqueeze(-1)  # [batch, num_heads, seq_len, 1]
            combined_lse = torch.logsumexp(torch.cat([lse_expanded, sinks], dim=-1), dim=-1, keepdim=True)

            # Use new_norm / old_norm = exp(combined_lse - lse) to compute renorm and apply
            renorm_factor = torch.exp(lse_expanded - combined_lse)
            attention_output = attention_output * renorm_factor
    else:
        attention_output = flex_attention_output  # type: ignore[assignment]
        lse = None

    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, lse
