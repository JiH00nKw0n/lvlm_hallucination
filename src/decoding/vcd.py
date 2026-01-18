"""
VCD: Visual Contrastive Decoding

Contrasts output distributions between original and DDPM-noised images
to reduce hallucination.

Reference:
    - VCD/vcd_utils/vcd_add_noise.py: DDPM noise schedule
    - VCD/vcd_utils/vcd_sample.py: Contrastive decoding logic

Formula:
    cd_logits = (1 + alpha) * logits_orig - alpha * logits_noised
    cutoff = log(beta) + max(logits_orig)
    final = cd_logits.masked_fill(logits_orig < cutoff, -inf)

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL, InstructBLIP
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, MitigatorConfig, add_diffusion_noise, sample_top_p


class VCDMitigator(BaseMitigator):
    """
    Visual Contrastive Decoding (VCD) mitigator.

    Adds DDPM noise to images and contrasts with original to reduce hallucination.

    Reference: VCD/vcd_utils/vcd_sample.py:141-159

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl, instructblip
        alpha: Contrastive weight (default: 1.0)
        beta: Adaptive plausibility cutoff (default: 0.1)
        noise_step: DDPM noise step 0-999 (default: 500)
    """

    name: str = "vcd"

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava",
        alpha: float = 1.0,
        beta: float = 0.1,
        noise_step: int = 500,
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.noise_step = noise_step

    def setup(self) -> None:
        """No setup needed - uses custom generation loop."""
        pass

    def cleanup(self) -> None:
        """No cleanup needed."""
        pass

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        is_first_step: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Prepare model inputs, handling pixel_values only on first step for cached decoding.

        Reference: VCD maintains separate KV caches for original and noised.
        """
        inputs = {
            'input_ids': input_ids,
            'use_cache': True,
        }

        if attention_mask is not None:
            inputs['attention_mask'] = attention_mask

        if past_key_values is not None:
            inputs['past_key_values'] = past_key_values

        # Only pass pixel_values on first step (before KV cache is populated)
        if is_first_step and pixel_values is not None:
            inputs['pixel_values'] = pixel_values

            # Model-specific additional kwargs
            for key in ['image_sizes', 'image_grid_thw', 'position_ids',
                        'qformer_input_ids', 'qformer_attention_mask', 'rope_deltas']:
                if key in kwargs and kwargs[key] is not None:
                    inputs[key] = kwargs[key]

        return inputs

    def _combine_logits(
        self,
        logits_orig: torch.Tensor,
        logits_noised: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine original and noised logits using VCD formula.

        Reference: VCD/vcd_utils/vcd_sample.py:150-153

        Formula:
            cutoff = log(beta) + max(logits_orig)
            cd_logits = (1 + alpha) * logits_orig - alpha * logits_noised
            final = cd_logits.masked_fill(logits_orig < cutoff, -inf)
        """
        # Adaptive plausibility cutoff (version 2 from reference)
        cutoff = torch.log(torch.tensor(self.beta, device=logits_orig.device)) + \
                 logits_orig.max(dim=-1, keepdim=True).values

        # Contrastive logits
        cd_logits = (1 + self.alpha) * logits_orig - self.alpha * logits_noised

        # Apply cutoff
        cd_logits = cd_logits.masked_fill(logits_orig < cutoff, -float("inf"))

        return cd_logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with VCD contrastive decoding.

        Reference: VCD/vcd_utils/vcd_sample.py:90-206

        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            pixel_values: Image tensor [B, C, H, W]
            **kwargs: Additional model-specific kwargs (image_sizes, image_grid_thw, etc.)

        Returns:
            Generated token IDs [B, seq_len + max_new_tokens]
        """
        if pixel_values is None:
            raise ValueError("VCD requires pixel_values")

        generated = input_ids.clone()
        device = input_ids.device

        # Create noised image
        pixel_values_noised = add_diffusion_noise(pixel_values, self.noise_step)

        # Separate KV caches for original and noised
        past_kv_orig = None
        past_kv_noised = None

        for step in range(self.config.max_new_tokens):
            is_first_step = (past_kv_orig is None)

            # Current input (full sequence on first step, last token afterwards)
            if is_first_step:
                curr_ids = generated
            else:
                curr_ids = generated[:, -1:]

            # Prepare inputs for original image
            inputs_orig = self._prepare_inputs(
                curr_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                past_key_values=past_kv_orig,
                is_first_step=is_first_step,
                **kwargs,
            )

            # Prepare inputs for noised image
            inputs_noised = self._prepare_inputs(
                curr_ids,
                pixel_values=pixel_values_noised,
                attention_mask=attention_mask,
                past_key_values=past_kv_noised,
                is_first_step=is_first_step,
                **kwargs,
            )

            # Forward passes
            with torch.no_grad():
                outputs_orig = self.model(**inputs_orig)
                outputs_noised = self.model(**inputs_noised)

            # Update KV caches
            past_kv_orig = outputs_orig.past_key_values
            past_kv_noised = outputs_noised.past_key_values

            # Get logits for last token
            logits_orig = outputs_orig.logits[:, -1, :]
            logits_noised = outputs_noised.logits[:, -1, :]

            # Combine using VCD formula
            cd_logits = self._combine_logits(logits_orig, logits_noised)

            # Sample next token
            if self.config.do_sample:
                next_token = sample_top_p(cd_logits, self.config.top_p, self.config.temperature)
            else:
                next_token = cd_logits.argmax(dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)

            # Check for EOS
            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    if any((next_token == eos).all() for eos in eos_token_id):
                        break
                elif (next_token == eos_token_id).all():
                    break

        return generated
