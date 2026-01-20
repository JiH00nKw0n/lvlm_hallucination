"""
AvisC: Attention-based Vision Calibration

Identifies "blind tokens" (image tokens with low discriminative attention)
and masks their embeddings before contrastive decoding.

Reference:
    - AvisC/avisc_utils/avisc_sample.py:160-179 (Blind token detection)
    - AvisC/avisc_utils/avisc_sample.py:206-208 (Contrastive + plausibility cutoff)

Key Implementation Notes:
    1. Blind token detection uses attention patterns to identify low-info tokens
    2. Masking is applied via forward hook AFTER multimodal merging
    3. Separate KV caches for original and masked branches
    4. Plausibility cutoff to filter implausible tokens

Formula (Reference: avisc_sample.py:206-208):
    cutoff = log(beta) + max(logits_orig)
    diffs = (1 + alpha) * logits_orig - alpha * logits_masked
    avisc_logits = diffs.masked_fill(logits_orig < cutoff, -inf)

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseMitigator, sample_top_p


class AvisCMitigator(BaseMitigator):
    """
    AvisC: Attention-based Vision Calibration.

    Reference: AvisC/avisc_utils/avisc_sample.py:160-179

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        alpha: Contrastive weight (default: 1.0)
        beta: Plausibility cutoff (default: 0.1)
        layer_gamma: Cumulative prob for layer selection (default: 0.5)
        lamb: Std multiplier for blind token threshold (default: 100.0)
        masking_scheme: How to mask - "zeros", "ones", "noise" (default: "zeros")
    """

    name: str = "avisc"

    def __init__(
            self,
            model: nn.Module,
            model_type: str = "llava",
            alpha: float = 1.0,
            beta: float = 0.1,
            layer_gamma: float = 0.5,
            lamb: Optional[float] = None,
            masking_scheme: str = "zeros",
            **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.layer_gamma = layer_gamma
        self.masking_scheme = masking_scheme

        # Default lamb
        self.lamb = 100.0 if lamb is None else lamb

        # State for masking
        self._blind_mask: Optional[torch.Tensor] = None
        self._img_start: int = 0
        self._img_end: int = 0
        self._masking_hook_handle = None
        self._enable_masking: bool = False

    def setup(self) -> None:
        """Register masking hook on first decoder layer."""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = "eager"
            if hasattr(self.model.config, "attn_implementation"):
                self.model.config.attn_implementation = "eager"
        # We'll use a forward pre-hook on the first layer to mask embeddings
        # This happens AFTER the VLM's multimodal merging
        layers = self._get_layers()
        if len(layers) > 0:
            self._masking_hook_handle = layers[0].register_forward_pre_hook(
                self._masking_hook
            )

    def cleanup(self) -> None:
        """Remove masking hook."""
        if self._masking_hook_handle is not None:
            self._masking_hook_handle.remove()
            self._masking_hook_handle = None
        self._blind_mask = None
        self._enable_masking = False

    def _masking_hook(self, module: nn.Module, args: tuple) -> object:
        """
        Forward pre-hook to mask image token embeddings.

        Reference: Octopus/eval_bench/train_token_amber.py:208-222

        This hook runs AFTER multimodal merging is complete.
        """
        if not self._enable_masking or self._blind_mask is None:
            return args

        # Get hidden_states from args (first positional argument)
        hidden_states = args[0] if isinstance(args, tuple) else args

        # Apply masking to image tokens
        masked_hidden = hidden_states.clone()
        batch_size = masked_hidden.shape[0]

        for b in range(batch_size):
            blind_indices = torch.where(self._blind_mask[b])[0]
            for idx in blind_indices:
                pos = self._img_start + idx.item()
                if pos < masked_hidden.shape[1]:
                    if self.masking_scheme == "zeros":
                        masked_hidden[b, pos] = 0.0
                    elif self.masking_scheme == "ones":
                        masked_hidden[b, pos] = 1.0
                    elif self.masking_scheme == "noise":
                        masked_hidden[b, pos] = torch.randn_like(masked_hidden[b, pos])

        # Return modified args
        if isinstance(args, tuple):
            return (masked_hidden,) + args[1:]
        return masked_hidden

    def _detect_blind_tokens(
            self,
            attentions: Tuple[torch.Tensor, ...],
            img_start: int,
            img_end: int,
    ) -> torch.Tensor:
        """
        Detect blind tokens based on attention patterns.

        Reference: AvisC/avisc_utils/avisc_sample.py:160-179

        Steps:
            1. Compute image attention per layer
            2. Select top-k layers by cumulative probability (layer_gamma)
            3. Average attention across selected layers
            4. Threshold: mean + lamb * std → tokens below are "blind"

        Args:
            attentions: Tuple of attention tensors [B, H, Q, K] per layer
            img_start: Start index of image tokens
            img_end: End index of image tokens

        Returns:
            blind_mask: Boolean tensor [B, num_img_tokens] where True = blind
        """
        # Step 1: Layer selection by image attention
        layer_img_att = []
        for attn in attentions:
            # attn: [B, H, Q, K], average over heads, take last query token
            img_attn = attn.mean(dim=1)[:, -1, img_start:img_end]  # [B, num_img]
            layer_img_att.append(img_attn.sum())

        layer_img_att = torch.stack(layer_img_att, dim=0)
        layer_probs = layer_img_att / layer_img_att.sum()

        # Count top-p layers (reference uses < top_p)
        sorted_probs = torch.sort(layer_probs, descending=True)[0]
        cumsum = torch.cumsum(sorted_probs, dim=0)
        k = (cumsum < self.layer_gamma).sum().item() + 1
        _, top_k_layers = torch.topk(layer_probs.float(), k, dim=0)
        top_k_layers = top_k_layers.tolist()

        # Step 2: Stack attention from selected layers
        att_stack = torch.stack(
            [
                attentions[i].mean(dim=1)[:, -1, img_start:img_end]
                for i in top_k_layers
            ], dim=1
        )  # [B, k, num_img_tokens]

        img_att = att_stack.mean(dim=1)  # [B, num_img_tokens]

        # Step 3: Threshold
        threshold = img_att.mean() + self.lamb * img_att.std()

        # Blind tokens: attention BELOW threshold (low discriminative value)
        blind_mask = img_att < threshold  # [B, num_img_tokens]

        return blind_mask

    def _combine_logits(
            self,
            logits_orig: torch.Tensor,
            logits_masked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine original and masked logits with plausibility cutoff.

        Reference: AvisC/avisc_utils/avisc_sample.py:206-208

        Formula:
            cutoff = log(beta) + max(logits_orig)
            diffs = (1 + alpha) * logits_orig - alpha * logits_masked
            avisc_logits = diffs.masked_fill(logits_orig < cutoff, -inf)
        """
        # Compute plausibility cutoff (Reference: line 206)
        cutoff = torch.log(torch.tensor(self.beta, device=logits_orig.device)) + \
                 logits_orig.max(dim=-1, keepdim=True).values

        # Contrastive combination (Reference: line 207)
        diffs = (1 + self.alpha) * logits_orig - self.alpha * logits_masked

        # Apply plausibility constraint (Reference: line 208)
        return diffs.masked_fill(logits_orig < cutoff, -float("inf"))

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Generate with AvisC blind token masking.

        Reference: AvisC/avisc_utils/avisc_sample.py

        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            pixel_values: Image tensor [B, C, H, W]
            **kwargs: Additional model-specific kwargs

        Returns:
            Generated token IDs [B, seq_len + max_new_tokens]
        """
        if pixel_values is None:
            raise ValueError("AvisC requires pixel_values")

        generated = input_ids.clone()
        device = input_ids.device

        # Get image token indices
        config = getattr(self.model, 'config', None)
        self._img_start, self._img_end = self._get_image_token_indices(input_ids, config)

        # cache_position for first step (Qwen2-VL compatibility)
        cache_position = torch.arange(input_ids.shape[1], device=device)

        # Step 1: Initial forward to get attention for blind token detection
        inputs_init = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'output_attentions': True,
            'use_cache': True,
            'return_dict': True,
            'cache_position': cache_position,
        }
        # Add model-specific kwargs
        for key in ['image_sizes', 'image_grid_thw', 'position_ids']:
            if key in kwargs and kwargs[key] is not None:
                inputs_init[key] = kwargs[key]

        with torch.no_grad():
            outputs_init = self.model(**inputs_init)

        # Detect blind tokens
        self._blind_mask = self._detect_blind_tokens(
            outputs_init.attentions,
            self._img_start,
            self._img_end,
        )

        # Store original KV cache
        past_kv_orig = outputs_init.past_key_values

        # Step 2: Forward with masking enabled to get masked KV cache
        self._enable_masking = True
        inputs_masked = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'use_cache': True,
            'return_dict': True,
            'cache_position': cache_position,
        }
        for key in ['image_sizes', 'image_grid_thw', 'position_ids']:
            if key in kwargs and kwargs[key] is not None:
                inputs_masked[key] = kwargs[key]

        with torch.no_grad():
            outputs_masked = self.model(**inputs_masked)
        self._enable_masking = False

        past_kv_masked = outputs_masked.past_key_values

        # Combine first logits
        logits_orig = outputs_init.logits[:, -1, :]
        logits_masked = outputs_masked.logits[:, -1, :]
        cd_logits = self._combine_logits(logits_orig, logits_masked)

        # Sample first token
        if self.config.do_sample:
            next_token = sample_top_p(
                cd_logits,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )
        else:
            next_token = cd_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1
            )

        # Step 3: Continue generation with cached KV
        for step in range(1, self.config.max_new_tokens):
            curr_ids = generated[:, -1:]
            # cache_position for subsequent steps
            cache_position = torch.tensor([generated.shape[1] - 1], device=device)

            # Original branch
            with torch.no_grad():
                outputs_orig = self.model(
                    input_ids=curr_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_kv_orig,
                    use_cache=True,
                    return_dict=True,
                    cache_position=cache_position,
                )

            # Masked branch (no need for masking hook - KV cache already has masked info)
            with torch.no_grad():
                outputs_masked = self.model(
                    input_ids=curr_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_kv_masked,
                    use_cache=True,
                    return_dict=True,
                    cache_position=cache_position,
                )

            past_kv_orig = outputs_orig.past_key_values
            past_kv_masked = outputs_masked.past_key_values

            logits_orig = outputs_orig.logits[:, -1, :]
            logits_masked = outputs_masked.logits[:, -1, :]
            cd_logits = self._combine_logits(logits_orig, logits_masked)

            if self.config.do_sample:
                next_token = sample_top_p(
                    cd_logits,
                    top_p=self.config.top_p,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                )
            else:
                next_token = cd_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                    ], dim=-1
                )

            # Check for EOS
            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    if any((next_token == eos).all() for eos in eos_token_id):
                        break
                elif (next_token == eos_token_id).all():
                    break

        return generated
