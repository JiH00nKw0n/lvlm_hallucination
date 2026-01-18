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

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL, InstructBLIP
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, MitigatorConfig, sample_top_p, ModelHelper


class DecoMitigator(BaseMitigator):
    """
    Deco: Decoding with Early Exit Calibration.

    Reference: Deco/transformers/generation/utils.py:2660-2682

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl, instructblip
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

        Args:
            hidden_states: Tuple of hidden states per layer (including embedding)

        Returns:
            Dict mapping layer_idx to logits [B, vocab_size]
        """
        norm, lm_head = self._get_norm_and_lm_head()
        early_logits = {}

        for layer_idx in self.early_exit_layers:
            # hidden_states[0] is embedding, so layer i is at index i+1
            if layer_idx + 1 < len(hidden_states):
                h = norm(hidden_states[layer_idx + 1])
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
        generated = input_ids.clone()
        device = input_ids.device
        past_key_values = None

        for step in range(self.config.max_new_tokens):
            is_first_step = (past_key_values is None)

            if is_first_step:
                curr_ids = generated
                forward_kwargs = {
                    'input_ids': curr_ids,
                    'attention_mask': attention_mask,
                    'output_hidden_states': True,
                    'use_cache': True,
                    'return_dict': True,
                }
                if pixel_values is not None:
                    forward_kwargs['pixel_values'] = pixel_values
                for key in ['image_sizes', 'image_grid_thw', 'position_ids',
                            'qformer_input_ids', 'qformer_attention_mask']:
                    if key in kwargs and kwargs[key] is not None:
                        forward_kwargs[key] = kwargs[key]
            else:
                curr_ids = generated[:, -1:]
                forward_kwargs = {
                    'input_ids': curr_ids,
                    'attention_mask': attention_mask,
                    'past_key_values': past_key_values,
                    'output_hidden_states': True,
                    'use_cache': True,
                    'return_dict': True,
                }

            with torch.no_grad():
                outputs = self.model(**forward_kwargs)

            past_key_values = outputs.past_key_values
            final_logits = outputs.logits[:, -1, :]

            # Get early exit logits
            early_logits_dict = self._get_early_logits(outputs.hidden_states)

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

            # Sample
            if self.config.do_sample:
                next_token = sample_top_p(blended, self.config.top_p, self.config.temperature)
            else:
                next_token = blended.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)

            # Check EOS
            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    if any((next_token == eos).all() for eos in eos_token_id):
                        break
                elif (next_token == eos_token_id).all():
                    break

        return generated
