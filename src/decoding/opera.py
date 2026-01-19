"""
OPERA: Over-trust Penalty and Retrospective Allocation

Reference:
    - OPERA/transformers-4.29.2/src/transformers/generation/utils.py:3116-3674

This implementation ports the reference `opera_beam_search` logic and exposes
it via a mitigator-friendly `generate` wrapper.

Supports: LLaVA, LLaVA-NeXT (reference OPERA is LLaVA-based)
"""

import copy
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

from .base import BaseMitigator, ModelHelper


@dataclass
class OPERAState:
    """State snapshot for OPERA rollback."""
    input_ids: torch.Tensor
    beam_scorer: object
    beam_indices: Optional[Tuple]
    cur_len: int
    attn_previous: torch.Tensor
    candidate_token_scores: torch.Tensor
    candidate_tokens: torch.Tensor
    beam_scores: torch.Tensor
    beam_next_tokens: Optional[torch.Tensor]
    beam_idx: Optional[torch.Tensor]


class OPERAMitigator(BaseMitigator):
    """
    OPERA: Over-trust Penalty and Retrospective Allocation.

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        num_beams: Number of beams (default: 5)
        scale_factor: Attention amplification (default: 50.0)
        threshold: Rollback trigger count (default: 15)
        num_attn_candidates: Attention candidates to track (default: 5)
        penalty_weights: Penalty weight (default: 1.0)
        window_size: Local window size for OPERA (default: 512)
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
            window_size: int = 512,
            **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.num_beams = num_beams
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.num_attn_candidates = num_attn_candidates
        self.penalty_weights = penalty_weights
        self.window_size = window_size

    def setup(self) -> None:
        return

    def cleanup(self) -> None:
        return

    def _get_num_image_tokens(self) -> int:
        if hasattr(self.model, "get_vision_tower"):
            vision_tower = self.model.get_vision_tower()
            if hasattr(vision_tower, "num_patches"):
                return int(vision_tower.num_patches)
        if hasattr(self.model, "vision_tower"):
            vision_tower = self.model.vision_tower
            if hasattr(vision_tower, "num_patches"):
                return int(vision_tower.num_patches)
        return ModelHelper.DEFAULT_IMAGE_TOKENS.get(self.model_type, (35, 611))[1] - \
            ModelHelper.DEFAULT_IMAGE_TOKENS.get(self.model_type, (35, 611))[0]

    def _compute_key_position(self, input_ids: torch.Tensor) -> dict:
        img_start, _ = self._get_image_token_indices(input_ids, getattr(self.model, "config", None))
        num_image_tokens = self._get_num_image_tokens()
        return {
            "image_start": img_start,
            "image_end": img_start + num_image_tokens - 1,
            "response_start": input_ids.shape[1] + num_image_tokens - 1,
        }

    def _opera_beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamSearchScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            key_position: Optional[dict] = None,
            scale_factor: Optional[float] = 50.0,
            threshold: Optional[int] = 15,
            num_attn_candidates: Optional[int] = 5,
            window_size: Optional[int] = 512,
            penalty_weights: Optional[float] = 1.0,
            **model_kwargs,
    ) -> torch.LongTensor:
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.model.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.model.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.model.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        history_states: List[OPERAState] = []
        history_rollback_locs = None
        beam_next_tokens = None
        beam_idx = None
        rollback_pos = 0
        max_rollback_time = torch.zeros(window_size)
        history_length = window_size
        reject_token_pos_gather: List[List[torch.Tensor]] = [[] for _ in range(window_size)]
        model_kwargs_ori = model_kwargs.copy()

        while True:
            if synced_gpus:
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                if this_peer_finished_flag.item() == 0.0:
                    break

            current_state = {
                "input_ids": input_ids.clone(),
                "beam_scorer": copy.deepcopy(beam_scorer),
                "beam_indices": beam_indices.copy() if beam_indices is not None else None,
                "cur_len": cur_len,
            }

            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if "past_key_values" not in model_kwargs.keys():
                attn_previous = outputs.attentions[-1].clone()
            else:
                assert beam_idx is not None and attn_previous is not None
                attn_previous = torch.cat([attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1)
                attn_previous = torch.cat(
                    [attn_previous[beam_idx], outputs.attentions[-1].clone().max(1, keepdim=True).values.data], -2
                )

            attn_previous = attn_previous.max(1, keepdim=True).values.data
            current_state["attn_previous"] = attn_previous.data.cpu()

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue

            next_token_logits = outputs.logits[:, -1, :]

            if num_attn_candidates < 1:
                raise ValueError(
                    f"Num of candidates must be larger than 0, but it is currently {num_attn_candidates}."
                )
            candidate_token_scores, candidate_tokens = torch.topk(
                next_token_logits, num_attn_candidates, dim=-1, largest=True, sorted=True
            )
            current_state["candidate_tokens"] = candidate_tokens.clone()

            current_state["beam_scores"] = beam_scores.clone()
            current_state["beam_next_tokens"] = beam_next_tokens.clone() if beam_next_tokens is not None else None
            current_state["beam_idx"] = beam_idx.clone() if beam_idx is not None else None

            attn_last = []
            for candidate_id in range(num_attn_candidates):
                input_ids_tmp = torch.cat([input_ids, candidate_tokens[:, candidate_id].unsqueeze(-1)], dim=-1)

                model_kwargs_tmp = model_kwargs.copy()
                model_kwargs_tmp = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs_tmp, is_encoder_decoder=self.model.config.is_encoder_decoder
                )

                model_inputs_tmp = self.model.prepare_inputs_for_generation(input_ids_tmp, **model_kwargs_tmp)

                outputs_tmp = self.model(
                    **model_inputs_tmp,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                attn_output = outputs_tmp.attentions[-1].clone()
                attn_output = attn_output.max(1, keepdim=True).values.data
                attn_square = torch.cat([attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1)
                attn_square = torch.cat([attn_square, attn_output], -2)
                attn_last.append(attn_square)

            del input_ids_tmp, model_kwargs_tmp, model_inputs_tmp, outputs_tmp

            attn_last = torch.cat(attn_last, 1)
            attn_last = attn_last / attn_last.sum(-1, keepdim=True)

            attn_pos = key_position
            attn_local = attn_last[:, :, attn_pos["response_start"]:, attn_pos["response_start"]:]

            attn_local = scale_factor * attn_local
            attn_local_scores = torch.zeros(
                (
                    attn_local.shape[0], attn_local.shape[1], attn_local.shape[-1]), dtype=torch.float16
            ).to(candidate_token_scores.device)
            for j in range(attn_local.shape[-1]):
                local_score = 1e-7 * attn_local[..., j:, j].prod(-1).data
                attn_local_scores[..., j] = local_score.to(torch.float32)

            cur_response_lens = attn_local.shape[-1]
            attn_i = attn_last[:, :, -1, attn_pos["image_start"]:attn_pos["image_end"] + 1].sum(-1)
            attn_scores = attn_i

            rollback_scores, rollback_locs = attn_local_scores.max(-1)
            rollback_loc = rollback_locs.mode().values.data
            rollback_loc = rollback_loc.mode().values.data

            penalty_scores = -attn_scores if cur_response_lens <= 10 else rollback_scores

            if history_rollback_locs is None:
                history_rollback_locs = [rollback_locs.mode().values.data[:, None]]
            else:
                history_rollback_locs.append(rollback_locs.mode().values.data[:, None])
            rollback_loc_gathers = torch.cat(history_rollback_locs, -1)

            candidate_token_scores -= penalty_weights * penalty_scores
            current_state["candidate_token_scores"] = candidate_token_scores.clone()

            if len(history_states) >= history_length:
                history_states.pop(0)
            history_states.append(current_state)

            try:
                if all(
                        (rollback_loc_gather == rollback_loc).long().sum() > int(threshold)
                        for _, rollback_loc_gather in enumerate(rollback_loc_gathers)
                        ):
                    if rollback_loc < 10:
                        assert False
                    rollback_pos = rollback_loc + 1
                    if max_rollback_time[rollback_pos] >= num_attn_candidates:
                        rollback_pos = rollback_pos - 1
                        if max_rollback_time[rollback_pos] >= num_attn_candidates:
                            assert False
                        else:
                            max_rollback_time[rollback_pos] += 1
                    else:
                        max_rollback_time[rollback_pos] += 1
                    if cur_response_lens - rollback_pos > history_length + 1:
                        rollback_pos = max(1, cur_response_lens - history_length - 1)

                    for j in range(cur_response_lens - rollback_pos - 2):
                        history_states.pop(-1)
                        history_rollback_locs.pop(-1)
                        reject_token_pos_gather[-(j + 1)] = []

                    input_ids = history_states[-2]["input_ids"]
                    beam_scorer = history_states[-2]["beam_scorer"]
                    beam_indices = history_states[-2]["beam_indices"]
                    cur_len = history_states[-2]["cur_len"]

                    attn_previous = history_states[-2]["attn_previous"].to(input_ids.device)
                    candidate_token_scores = history_states[-2]["candidate_token_scores"]
                    candidate_tokens = history_states[-2]["candidate_tokens"]

                    beam_scores = history_states[-2]["beam_scores"]
                    beam_next_tokens = history_states[-1]["beam_next_tokens"]
                    beam_idx = history_states[-1]["beam_idx"]

                    if "images" in model_kwargs_ori.keys() or "pixel_values" in model_kwargs_ori.keys():
                        model_kwargs = model_kwargs_ori.copy()
                        model_kwargs["attention_mask"] = torch.cat(
                            [
                                model_kwargs["attention_mask"],
                                torch.ones(
                                    (
                                        input_ids.shape[0],
                                        input_ids[:, :-1].shape[1] - model_kwargs["attention_mask"].shape[1]
                                    )
                                ).to(input_ids.device)
                            ], 1
                        )

                        model_inputs_tmp = self.model.prepare_inputs_for_generation(input_ids[:, :-1], **model_kwargs)
                    else:
                        answer_embeds = self.model.embed_tokens(input_ids[:, 1:-1])
                        model_kwargs = model_kwargs_ori.copy()
                        model_kwargs["inputs_embeds"] = torch.cat([model_kwargs["inputs_embeds"], answer_embeds], 1)
                        model_kwargs["attention_mask"] = torch.cat(
                            [model_kwargs["attention_mask"], torch.ones_like(input_ids[:, 1:-1]).to(input_ids.device)],
                            1
                        )

                        model_inputs_tmp = self.model.prepare_inputs_for_generation(input_ids[:, 1:-1], **model_kwargs)

                    outputs_tmp = self.model(
                        **model_inputs_tmp,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    model_kwargs = self.model._update_model_kwargs_for_generation(
                        outputs_tmp, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
                    )

                    model_inputs_tmp = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

                    outputs = self.model(
                        **model_inputs_tmp,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    del outputs_tmp, model_inputs_tmp

                    history_states.pop(-1)
                    history_rollback_locs.pop(-1)
                    reject_token_pos_gather[rollback_pos + 1] = []

                    next_token_logits -= 999. + next_token_logits.min(-1, keepdim=True).values.data
                    next_token_logits = next_token_logits.view(batch_size, num_beams * vocab_size)
                    beam_idx = beam_idx.view(batch_size, num_beams)
                    beam_next_tokens = beam_next_tokens.view(batch_size, num_beams)
                    reject_token_pos = beam_idx * vocab_size + beam_next_tokens
                    if len(reject_token_pos_gather[rollback_pos]) > 0:
                        reject_token_pos = torch.cat([reject_token_pos_gather[rollback_pos], reject_token_pos], -1)
                    reject_token_pos_gather[rollback_pos] = reject_token_pos
                    next_token_logits = next_token_logits.scatter_(-1, reject_token_pos, -999.)
                    next_token_logits = next_token_logits.view(batch_size * num_beams, vocab_size)
                else:
                    assert False
            except Exception:
                next_token_logits.fill_(-999.)
                next_token_logits = next_token_logits.scatter_(-1, candidate_tokens, candidate_token_scores)

            del attn_last, attn_local, attn_local_scores

            next_token_logits = self.model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.model._reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.model.config.is_encoder_decoder:
                return sequence_outputs
            return sequence_outputs
        return sequence_outputs["sequences"]

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        if pixel_values is None:
            raise ValueError("OPERA requires pixel_values")

        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.num_beams = self.num_beams
        generation_config.do_sample = False
        generation_config.output_attentions = True
        generation_config.use_cache = True
        if self.config.max_new_tokens is not None:
            generation_config.max_new_tokens = self.config.max_new_tokens
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids.shape[1]
        if generation_config.max_length is None:
            generation_config.max_length = input_ids.shape[1] + self.config.max_new_tokens

        model_kwargs = {}
        if attention_mask is None:
            if hasattr(self.model, "_prepare_attention_mask_for_generation"):
                attention_mask = self.model._prepare_attention_mask_for_generation(
                    input_ids,
                    generation_config.pad_token_id,
                    generation_config.eos_token_id,
                )
        model_kwargs["attention_mask"] = attention_mask
        model_kwargs["pixel_values"] = pixel_values

        for key in ["image_sizes", "image_grid_thw", "position_ids"]:
            if key in kwargs and kwargs[key] is not None:
                model_kwargs[key] = kwargs[key]

        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            logits_processor=LogitsProcessorList(),
        )
        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=StoppingCriteriaList(),
        )

        beam_scorer = BeamSearchScorer(
            batch_size=input_ids.shape[0],
            num_beams=generation_config.num_beams,
            device=input_ids.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.model.config.is_encoder_decoder,
            **model_kwargs,
        )

        key_position = self._compute_key_position(input_ids)

        return self._opera_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_attentions=generation_config.output_attentions,
            output_hidden_states=generation_config.output_hidden_states,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=False,
            key_position=key_position,
            scale_factor=self.scale_factor,
            threshold=self.threshold,
            num_attn_candidates=self.num_attn_candidates,
            window_size=self.window_size,
            penalty_weights=self.penalty_weights,
            **model_kwargs,
        )
