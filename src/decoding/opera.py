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

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
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
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
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
        attn_last: torch.Tensor,
        img_start: int,
        img_end: int,
        response_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute attention-based penalty for candidate tokens.

        Reference: OPERA/transformers/.../utils.py:3430-3449

        Args:
            attn_last: Candidate attention [B, C, Q, K]
            img_start: Image token start index
            img_end: Image token end index (exclusive)
            response_start: Response token start index

        Returns:
            (penalty_scores, rollback_locs, rollback_loc)
        """
        attn_local = attn_last[:, :, response_start:, response_start:]
        attn_local = self.scale_factor * attn_local

        attn_local_scores = torch.zeros(
            (attn_local.shape[0], attn_local.shape[1], attn_local.shape[-1]),
            device=attn_local.device,
            dtype=torch.float32,
        )
        for j in range(attn_local.shape[-1]):
            local_score = 1e-7 * attn_local[..., j:, j].prod(-1)
            attn_local_scores[..., j] = local_score

        cur_response_len = attn_local.shape[-1]
        attn_scores = attn_last[:, :, -1, img_start:img_end].sum(-1)

        rollback_scores, rollback_locs = attn_local_scores.max(-1)
        rollback_loc = rollback_locs.mode().values
        rollback_loc = rollback_loc.mode().values.item()

        penalty_scores = -attn_scores if cur_response_len <= 10 else rollback_scores
        return penalty_scores, rollback_locs, rollback_loc

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with OPERA beam search and rollback.

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
        history_rollback_locs: Optional[List[torch.Tensor]] = None
        reject_token_pos_gather: Dict[int, List[torch.Tensor]] = {}

        past_key_values = None
        attn_previous = None
        response_start = input_ids.shape[1]

        beam_scores = beam_scores.view(batch_size, self.num_beams)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        for step in range(self.config.max_new_tokens):
            is_first_step = (past_key_values is None)

            if is_first_step:
                curr_ids = generated
                cache_position = torch.arange(curr_ids.shape[1], device=device)
                forward_kwargs = {
                    'input_ids': curr_ids,
                    'attention_mask': attention_mask,
                    'pixel_values': pixel_values,
                    'output_attentions': True,
                    'use_cache': True,
                    'return_dict': True,
                    'cache_position': cache_position,
                }
                for key in ['image_sizes', 'image_grid_thw', 'position_ids']:
                    if key in kwargs and kwargs[key] is not None:
                        forward_kwargs[key] = kwargs[key]
            else:
                curr_ids = generated[:, -1:]
                cache_position = torch.tensor([generated.shape[1] - 1], device=device)
                forward_kwargs = {
                    'input_ids': curr_ids,
                    'attention_mask': attention_mask,
                    'past_key_values': past_key_values,
                    'output_attentions': True,
                    'use_cache': True,
                    'return_dict': True,
                    'cache_position': cache_position,
                }

            with torch.no_grad():
                outputs = self.model(**forward_kwargs)

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            attentions = outputs.attentions[-1] if outputs.attentions else None

            if attentions is None:
                raise ValueError("OPERA requires attention outputs")

            if attn_previous is None:
                attn_previous = attentions.clone()
            else:
                attn_previous = torch.cat(
                    [attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1
                )
                attn_previous = torch.cat(
                    [attn_previous, attentions.clone().max(1, keepdim=True).values.data], -2
                )

            attn_previous = attn_previous.max(1, keepdim=True).values.data

            candidate_token_scores, candidate_tokens = torch.topk(
                logits, self.num_attn_candidates, dim=-1, largest=True, sorted=True
            )

            attn_last = []
            for candidate_id in range(self.num_attn_candidates):
                next_tokens = candidate_tokens[:, candidate_id].unsqueeze(-1)
                tmp_attention = attention_mask
                if tmp_attention is not None:
                    tmp_attention = torch.cat(
                        [tmp_attention, torch.ones_like(tmp_attention[:, :1])], dim=-1
                    )
                tmp_kwargs = {
                    'input_ids': next_tokens,
                    'attention_mask': tmp_attention,
                    'past_key_values': past_key_values,
                    'output_attentions': True,
                    'use_cache': True,
                    'return_dict': True,
                    'cache_position': torch.tensor([generated.shape[1]], device=device),
                }
                with torch.no_grad():
                    outputs_tmp = self.model(**tmp_kwargs)

                attn_output = outputs_tmp.attentions[-1].clone()
                attn_output = attn_output.max(1, keepdim=True).values.data
                attn_square = torch.cat(
                    [attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1
                )
                attn_square = torch.cat([attn_square, attn_output], -2)
                attn_last.append(attn_square)

            attn_last = torch.cat(attn_last, 1)
            attn_last = attn_last / attn_last.sum(-1, keepdim=True)

            penalty_scores, rollback_locs, rollback_loc = self._compute_penalty(
                attn_last, img_start, img_end, response_start
            )
            candidate_token_scores = candidate_token_scores - self.penalty_weights * penalty_scores

            if history_rollback_locs is None:
                history_rollback_locs = [rollback_locs.mode().values.data[:, None]]
            else:
                history_rollback_locs.append(rollback_locs.mode().values.data[:, None])

            state = OPERAState(
                input_ids=generated.clone(),
                attention_mask=attention_mask.clone() if attention_mask is not None else None,
                past_key_values=past_key_values,
                beam_scores=beam_scores.clone(),
                attn_previous=attn_previous.clone(),
            )
            if len(history_states) >= self.history_length:
                history_states.pop(0)
                if history_rollback_locs:
                    history_rollback_locs.pop(0)
            history_states.append(state)

            should_rollback = False
            rollback_pos: Optional[int] = None
            if history_rollback_locs is not None and len(history_rollback_locs) >= self.threshold:
                recent = torch.cat(history_rollback_locs[-self.threshold:], -1)
                matches = (recent == rollback_loc).long().sum(dim=-1)
                if torch.all(matches >= self.threshold) and rollback_loc >= 10:
                    should_rollback = True
                    rollback_pos = rollback_loc + 1

            next_token_logits = torch.full_like(logits, -999.0)
            next_token_logits = next_token_logits.scatter(-1, candidate_tokens, candidate_token_scores)

            log_probs = F.log_softmax(next_token_logits, dim=-1)
            next_scores = log_probs + beam_scores.unsqueeze(-1)

            vocab_size = next_scores.shape[-1]
            next_scores = next_scores.view(batch_size, self.num_beams * vocab_size)

            response_pos = generated.shape[1] - response_start + 1
            if response_pos in reject_token_pos_gather:
                reject_tokens = torch.cat(reject_token_pos_gather[response_pos], dim=-1).long()
                next_scores = next_scores.scatter(1, reject_tokens, -1e9)

            next_scores, next_tokens = torch.topk(
                next_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            beam_scores = next_scores[:, :self.num_beams].reshape(-1)
            beam_tokens = next_tokens[:, :self.num_beams].reshape(-1)
            beam_indices = next_indices[:, :self.num_beams].reshape(-1)

            if should_rollback and rollback_pos is not None and len(history_states) > 1:
                reject_token_pos = (
                    beam_indices.view(batch_size, self.num_beams) * vocab_size
                    + beam_tokens.view(batch_size, self.num_beams)
                )
                reject_token_pos_gather.setdefault(rollback_pos, []).append(reject_token_pos)
                prev_state = history_states[-2]
                generated = prev_state.input_ids
                attention_mask = prev_state.attention_mask
                past_key_values = prev_state.past_key_values
                beam_scores = prev_state.beam_scores
                attn_previous = prev_state.attn_previous
                continue

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

            if past_key_values is not None:
                if hasattr(self.model, "_reorder_cache"):
                    past_key_values = self.model._reorder_cache(past_key_values, batch_beam_indices)
                elif hasattr(past_key_values, "reorder_cache"):
                    past_key_values.reorder_cache(batch_beam_indices)

            attn_previous = attn_previous[batch_beam_indices]

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
