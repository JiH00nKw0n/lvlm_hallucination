"""
Deco: Decoding with Early Exit Calibration

Selects intermediate layers with highest candidate confidence and blends
their logits with final layer logits.

Reference:
    - Deco/transformers/generation/utils.py:2660-2682

Key Implementation Notes:
    1. Get candidates from final layer (top-k + top-p)
    2. Find early exit layer with highest confidence on any candidate
    3. Blend: final_logits + alpha * confidence * early_logits
    4. Mask non-candidate tokens

Formula:
    candidates = top_p(top_k(final_logits))
    layer, conf = argmax over layers of max(softmax(early_logits)[candidates])
    final = final_logits + alpha * conf * early_layer_logits
    final[not in candidates] = -inf

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessorList

from .base import BaseMitigator


class DecoMitigator(BaseMitigator):
    """
    Deco: Decoding with Early Exit Calibration.

    Reference: Deco/transformers/generation/utils.py:2660-2682

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        alpha: Blend strength (default: 0.6)
        early_exit_layers: Layers to inspect (default: 20-28)
        threshold_top_k: Max candidates (default: 20)
        threshold_top_p: Nucleus threshold (default: 0.9)
    """

    name: str = "deco"

    def __init__(
            self,
            model: nn.Module,
            model_type: str = "llava",
            alpha: float = 0.6,
            early_exit_layers: Optional[List[int]] = None,
            threshold_top_k: int = 20,
            threshold_top_p: float = 0.9,
            **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        if "do_sample" not in kwargs and self.config.do_sample:
            self.config.do_sample = False
        self.alpha = alpha
        self.threshold_top_k = threshold_top_k
        self.threshold_top_p = threshold_top_p

        # Default early exit layers (later layers typically work better)
        if early_exit_layers is None:
            num_layers = self._get_num_layers()
            start = max(0, int(num_layers * 0.6))
            end = num_layers - 1
            self.early_exit_layers = list(range(start, end))
        else:
            self.early_exit_layers = early_exit_layers

    def setup(self) -> None:
        """No setup needed."""
        pass

    def cleanup(self) -> None:
        """No cleanup needed."""
        pass

    def _get_early_logits(
            self,
            hidden_states: Tuple[torch.Tensor, ...],
    ) -> Dict[int, torch.Tensor]:
        """
        Compute logits from early exit layers.

        Reference: Deco/transformers/models/llama/modeling_llama.py:821-824

        Note: Reference uses hidden_states[layer_idx] directly (not layer_idx + 1).
        This matches the patched Deco model's indexing convention.

        Args:
            hidden_states: Tuple of hidden states per layer (including embedding)

        Returns:
            Dict mapping layer_idx to logits [B, vocab_size]
        """
        norm, lm_head = self._get_norm_and_lm_head()
        early_logits = {}

        for layer_idx in self.early_exit_layers:
            # Reference: hidden_states[early_exit_layer] directly
            # Note: In standard HuggingFace, hidden_states[0] is embeddings,
            # hidden_states[i] for i>=1 is layer i-1 output.
            # We use layer_idx directly to match reference behavior.
            if layer_idx < len(hidden_states):
                h = norm(hidden_states[layer_idx])
                early_logits[layer_idx] = lm_head(h)[:, -1, :]

        return early_logits

    def _select_anchor_layer(
            self,
            final_logits: torch.Tensor,
            early_logits: Dict[int, torch.Tensor],
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select anchor layer with highest candidate confidence.

        Reference: Deco/transformers/generation/utils.py:2668-2675

        Args:
            final_logits: [B, vocab_size]
            early_logits: Dict[layer_idx, [B, vocab_size]]

        Returns:
            (selected_layer_idx, max_prob, candidate_ids)
        """
        # Get candidates from final layer
        probs = F.softmax(final_logits, dim=-1).squeeze(0)
        candidate_probs, candidate_ids = torch.topk(probs, k=self.threshold_top_k, dim=-1)

        # Apply top-p filtering
        cumulative = candidate_probs.cumsum(dim=-1)
        cutoff_idx = torch.searchsorted(cumulative, self.threshold_top_p, right=False)
        cutoff_idx = torch.clamp(cutoff_idx + 1, max=self.threshold_top_k)
        candidate_ids = candidate_ids[:cutoff_idx]

        # Find layer with highest confidence on any candidate
        # Stack early logits and get probs for candidates
        layer_indices = list(early_logits.keys())
        stacked = torch.stack([early_logits[i] for i in layer_indices], dim=0)  # [L, B, vocab]
        softmaxed = F.softmax(stacked, dim=-1)  # [L, B, vocab]

        # Get probs for candidate tokens
        candidate_probs_early = softmaxed[:, :, candidate_ids].squeeze(1)  # [L, num_candidates]

        # Find max
        max_prob_flat_idx = candidate_probs_early.argmax()
        layer_idx_local = (max_prob_flat_idx // candidate_probs_early.size(1)).item()
        selected_layer = layer_indices[layer_idx_local]
        max_prob = candidate_probs_early.max()

        return selected_layer, max_prob, candidate_ids

    def _blend_logits(
            self,
            final_logits: torch.Tensor,
            early_logits: torch.Tensor,
            max_prob: torch.Tensor,
            candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend final and early exit logits.

        Reference: Deco/transformers/generation/utils.py:2680-2682

        Formula:
            blended = final_logits + alpha * max_prob * early_logits
            blended[not in candidates] = -inf
        """
        blended = final_logits + self.alpha * max_prob * early_logits

        # Mask non-candidates
        mask = torch.ones_like(blended, dtype=torch.bool)
        mask[:, candidate_ids] = False
        blended = blended.masked_fill(mask, -float("inf"))

        return blended

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Generate with Deco early exit calibration.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            pixel_values: [B, C, H, W]
            **kwargs: Additional model kwargs

        Returns:
            Generated token IDs [B, seq_len + max_new_tokens]
        """
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = bool(self.config.do_sample)
        if self.config.max_new_tokens is not None:
            generation_config.max_new_tokens = self.config.max_new_tokens
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids.shape[1]
        if generation_config.max_length is None:
            generation_config.max_length = input_ids.shape[1] + self.config.max_new_tokens
        if self.config.temperature is not None:
            generation_config.temperature = self.config.temperature
        if self.config.top_k is not None:
            generation_config.top_k = self.config.top_k
        if self.config.top_p is not None:
            generation_config.top_p = self.config.top_p

        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            logits_processor=LogitsProcessorList(),
        )
        logits_warper = None
        if generation_config.do_sample:
            logits_warper = self.model._get_logits_warper(
                generation_config=generation_config,
                logits_warper=LogitsProcessorList(),
            )

        generated = input_ids.clone()
        device = input_ids.device
        past_key_values = None

        for step in range(self.config.max_new_tokens):
            is_first_step = (past_key_values is None)

            if is_first_step:
                curr_ids = generated
                # cache_position for first step (Qwen2-VL compatibility)
                cache_position = torch.arange(curr_ids.shape[1], device=device)
                forward_kwargs = {
                    'input_ids': curr_ids,
                    'attention_mask': attention_mask,
                    'output_hidden_states': True,
                    'use_cache': True,
                    'return_dict': True,
                    'cache_position': cache_position,
                }
                if pixel_values is not None:
                    forward_kwargs['pixel_values'] = pixel_values
                for key in ['image_sizes', 'image_grid_thw', 'position_ids']:
                    if key in kwargs and kwargs[key] is not None:
                        forward_kwargs[key] = kwargs[key]
            else:
                curr_ids = generated[:, -1:]
                # cache_position for subsequent steps
                cache_position = torch.tensor([generated.shape[1] - 1], device=device)
                forward_kwargs = {
                    'input_ids': curr_ids,
                    'attention_mask': attention_mask,
                    'past_key_values': past_key_values,
                    'output_hidden_states': True,
                    'use_cache': True,
                    'return_dict': True,
                    'cache_position': cache_position,
                }

            with torch.no_grad():
                try:
                    outputs = self.model(
                        **forward_kwargs,
                        early_exit_layers=self.early_exit_layers,
                    )
                except TypeError:
                    outputs = self.model(**forward_kwargs)

            early_logits_dict = None
            final_outputs = outputs
            if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[0], dict):
                early_logits_dict, final_outputs = outputs

            past_key_values = final_outputs.past_key_values
            final_logits = final_outputs.logits[:, -1, :]

            # Get early exit logits
            if early_logits_dict is None:
                early_logits_dict = self._get_early_logits(final_outputs.hidden_states)

            if early_logits_dict:
                # Select anchor layer and blend
                selected_layer, max_prob, candidates = self._select_anchor_layer(
                    final_logits, early_logits_dict
                )
                blended = self._blend_logits(
                    final_logits,
                    early_logits_dict[selected_layer],
                    max_prob,
                    candidates,
                )
            else:
                blended = final_logits

            next_token_scores = logits_processor(generated, final_logits)
            final_token_scores = logits_processor(generated, blended)
            if generation_config.do_sample:
                if logits_warper is not None:
                    final_token_scores = logits_warper(generated, final_token_scores)
                final_probs = F.softmax(final_token_scores, dim=-1)
                next_token = torch.multinomial(final_probs, num_samples=1)
            else:
                next_token = final_token_scores.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                    ], dim=-1
                )

            # Check EOS
            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    if any((next_token == eos).all() for eos in eos_token_id):
                        break
                elif (next_token == eos_token_id).all():
                    break

        return generated
