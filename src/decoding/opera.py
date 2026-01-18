"""
OPERA: Over-trust Penalty and Retrospective Allocation

Full beam search with attention-based penalty and state rollback when
hallucination patterns detected.

Reference:
    - OPERA/transformers-4.29.2/src/transformers/generation/utils.py:3400-3600

Key Implementation Notes:
    1. Full beam search with num_beams candidates
    2. Attention penalty:
       - Early tokens (≤10): Penalize low image attention
       - Later tokens: Penalize "summary tokens" with repetitive attention
    3. Rollback when consistent summary token detected threshold times
    4. Maintains separate history for rollback

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL, InstructBLIP
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, MitigatorConfig, sample_top_p


@dataclass
class OPERAState:
    """State for OPERA rollback."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[Tuple] = None
    beam_scores: Optional[torch.Tensor] = None
    attn_previous: Optional[torch.Tensor] = None


class OPERAMitigator(BaseMitigator):
    """
    OPERA: Over-trust Penalty and Retrospective Allocation.

    Reference: OPERA/transformers-4.29.2/src/transformers/generation/utils.py:3400-3600

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl, instructblip
        num_beams: Number of beams (default: 5)
        scale_factor: Attention amplification (default: 50.0)
        threshold: Rollback trigger count (default: 15)
        num_attn_candidates: Attention candidates to track (default: 5)
        penalty_weights: Penalty weight (default: 1.0)
        history_length: Max history for rollback (default: 5)
    """

    name: str = "opera"

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava",
        num_beams: int = 5,
        scale_factor: float = 50.0,
        threshold: int = 15,
        num_attn_candidates: int = 5,
        penalty_weights: float = 1.0,
        history_length: int = 5,
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.num_beams = num_beams
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.num_attn_candidates = num_attn_candidates
        self.penalty_weights = penalty_weights
        self.history_length = history_length

    def setup(self) -> None:
        """No setup needed."""
        pass

    def cleanup(self) -> None:
        """No cleanup needed."""
        pass

    def _compute_penalty(
        self,
        attentions: torch.Tensor,
        img_start: int,
        img_end: int,
        response_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute attention-based penalty.

        Reference: OPERA/transformers/.../utils.py:3430-3449

        Args:
            attentions: Last layer attention [B, H, Q, K]
            img_start: Image token start index
            img_end: Image token end index
            response_len: Current response length

        Returns:
            (penalty_scores, rollback_locs, rollback_loc)
        """
        # Get attention from last layer, max over heads
        attn = attentions.max(dim=1, keepdim=True).values  # [B, 1, Q, K]

        if response_len <= 10:
            # Early tokens: penalize LOW image attention
            img_attn = attn[:, :, -1, img_start:img_end].sum(dim=-1)  # [B, 1]
            penalty_scores = -img_attn
            rollback_locs = torch.zeros_like(penalty_scores).long()
            rollback_loc = 0
        else:
            # Later tokens: penalize summary tokens
            # Scale and compute product over diagonal
            attn_local = attn * self.scale_factor
            window_size = min(response_len, attn_local.shape[-1])

            local_scores = []
            for j in range(window_size):
                if j < attn_local.shape[-2]:
                    score = attn_local[..., j:, j].prod(dim=-1) * 1e-7
                    local_scores.append(score)

            if local_scores:
                local_scores = torch.stack(local_scores, dim=-1)  # [B, 1, window]
                penalty_scores, rollback_locs = local_scores.max(dim=-1)
                rollback_loc = rollback_locs.mode().values.item()
            else:
                penalty_scores = torch.zeros((attn.shape[0], 1), device=attn.device)
                rollback_locs = torch.zeros_like(penalty_scores).long()
                rollback_loc = 0

        return penalty_scores.squeeze(-1), rollback_locs.squeeze(-1), rollback_loc

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with OPERA beam search and rollback.

        Simplified implementation that captures the core OPERA logic:
        1. Beam search with attention penalty
        2. Track rollback locations
        3. Rollback when threshold exceeded

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            pixel_values: [B, C, H, W]

        Returns:
            Generated token IDs [B, seq_len + max_new_tokens]
        """
        if pixel_values is None:
            raise ValueError("OPERA requires pixel_values")

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Get image token indices
        config = getattr(self.model, 'config', None)
        img_start, img_end = self._get_image_token_indices(input_ids, config)

        # Expand for beam search
        input_ids = input_ids.repeat_interleave(self.num_beams, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(self.num_beams, dim=0)
        pixel_values = pixel_values.repeat_interleave(self.num_beams, dim=0)

        for key in ['image_sizes', 'image_grid_thw']:
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = kwargs[key].repeat_interleave(self.num_beams, dim=0)

        generated = input_ids.clone()
        beam_scores = torch.zeros(batch_size * self.num_beams, device=device)

        # History for rollback
        history_states: List[OPERAState] = []
        history_rollback_locs: List[torch.Tensor] = []
        rollback_counts: Dict[int, int] = {}

        past_key_values = None
        response_start = input_ids.shape[1]

        for step in range(self.config.max_new_tokens):
            is_first_step = (past_key_values is None)
            response_len = step + 1

            # Prepare inputs
            if is_first_step:
                curr_ids = generated
                forward_kwargs = {
                    'input_ids': curr_ids,
                    'attention_mask': attention_mask,
                    'pixel_values': pixel_values,
                    'output_attentions': True,
                    'use_cache': True,
                    'return_dict': True,
                }
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
                    'output_attentions': True,
                    'use_cache': True,
                    'return_dict': True,
                }

            with torch.no_grad():
                outputs = self.model(**forward_kwargs)

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            attentions = outputs.attentions[-1] if outputs.attentions else None

            # Compute penalty
            if attentions is not None:
                penalty, rollback_locs, rollback_loc = self._compute_penalty(
                    attentions, img_start, img_end, response_len
                )
                logits = logits - self.penalty_weights * penalty.unsqueeze(-1)
                history_rollback_locs.append(rollback_locs)
            else:
                rollback_loc = 0

            # Save state for potential rollback
            state = OPERAState(
                input_ids=generated.clone(),
                attention_mask=attention_mask.clone() if attention_mask is not None else None,
                past_key_values=past_key_values,
                beam_scores=beam_scores.clone(),
            )
            if len(history_states) >= self.history_length:
                history_states.pop(0)
            history_states.append(state)

            # Check for rollback
            should_rollback = False
            if len(history_rollback_locs) >= self.threshold:
                # Check if same rollback location appears threshold times
                recent_locs = history_rollback_locs[-self.threshold:]
                if all(loc == rollback_loc for loc in recent_locs):
                    if rollback_loc >= 10:  # Only rollback for non-early tokens
                        count = rollback_counts.get(rollback_loc, 0) + 1
                        rollback_counts[rollback_loc] = count
                        if count <= self.num_attn_candidates:
                            should_rollback = True

            if should_rollback and len(history_states) > 1:
                # Rollback to previous state
                rollback_idx = max(0, len(history_states) - 2)
                prev_state = history_states[rollback_idx]
                generated = prev_state.input_ids
                attention_mask = prev_state.attention_mask
                past_key_values = prev_state.past_key_values
                beam_scores = prev_state.beam_scores

                # Pop rolled back states
                while len(history_states) > rollback_idx:
                    history_states.pop()
                    if history_rollback_locs:
                        history_rollback_locs.pop()

                continue

            # Beam search scoring
            log_probs = F.log_softmax(logits, dim=-1)
            next_scores = log_probs + beam_scores.unsqueeze(-1)

            # Reshape for beam selection
            vocab_size = next_scores.shape[-1]
            next_scores = next_scores.view(batch_size, self.num_beams * vocab_size)

            # Select top beams
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # Select best beams
            beam_scores = next_scores[:, :self.num_beams].reshape(-1)
            beam_tokens = next_tokens[:, :self.num_beams].reshape(-1)
            beam_indices = next_indices[:, :self.num_beams].reshape(-1)

            # Reorder
            batch_beam_indices = (
                torch.arange(batch_size, device=device).unsqueeze(1) * self.num_beams
                + beam_indices.view(batch_size, self.num_beams)
            ).view(-1)

            generated = generated[batch_beam_indices]
            generated = torch.cat([generated, beam_tokens.unsqueeze(-1)], dim=-1)

            if attention_mask is not None:
                attention_mask = attention_mask[batch_beam_indices]
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)

            # Reorder KV cache
            if past_key_values is not None:
                past_key_values = self.model._reorder_cache(past_key_values, batch_beam_indices)

            # Check EOS
            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    done = any((beam_tokens == eos).all() for eos in eos_token_id)
                else:
                    done = (beam_tokens == eos_token_id).all()
                if done:
                    break

        # Return best beam
        best_beam_idx = beam_scores.view(batch_size, self.num_beams).argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=device) * self.num_beams + best_beam_idx
        return generated[batch_indices]
