"""
Octopus: Dynamic Strategy Selection

Learns to dynamically select between hallucination mitigation strategies
per token using a trained classifier.

Reference:
    - Octopus/eval_bench/train_token_amber.py:132-256 (MyModel classifier)
    - Octopus/avisc_utils/avisc_sample.py:286-317 (M3ID formula)
    - Octopus/eval_bench/train_token_amber.py:611-631 (DPO loss)

Key Implementation Notes:
    1. Classifier: CLS token prepended to hidden states + TransformerEncoder + MLP
    2. Actions: 0=None, 1=AvisC, 2=VCD, 3=M3ID
    3. Separate KV caches for each strategy
    4. M3ID uses log-softmax formula: lc + ((1-gamma)/gamma) * (lc - lu)

Architecture (Reference):
    - cls_token: learnable [1, d_model]
    - TransformerEncoder: nhead=2, num_layers=2
    - MLP: Linear(d_model, d_model//4) -> LeakyReLU -> Linear(d_model//4, num_classes)

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    BaseMitigator, TrainableMitigator, MitigatorConfig,
    sample_top_p, add_diffusion_noise, ModelHelper
)


class OctopusClassifier(nn.Module):
    """
    Octopus action classifier.

    Reference: Octopus/eval_bench/train_token_amber.py:132-256

    Architecture matches reference exactly:
        - Single CLS token prepended to hidden states
        - TransformerEncoder (nhead=2, num_layers=2)
        - MLP: d_model -> d_model//4 (LeakyReLU) -> num_classes

    Input: Hidden states [B, seq_len, d_model]
    Output: Action logits [B, num_classes] (CLS token output)
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_classes: int = 4,
        nhead: int = 2,
        num_layers: int = 2,
        n_query: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.n_query = n_query

        # Single CLS token (Reference: line 140)
        self.cls_token = nn.Parameter(torch.randn(1, d_model))
        # Optional queries (saved in reference checkpoints)
        self.queries = nn.Parameter(torch.randn(n_query, d_model))

        # TransformerEncoder (Reference: lines 141-143)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head (Reference: lines 145-151)
        # Uses LeakyReLU and d_model//4
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LeakyReLU(),
            nn.Linear(d_model // 4, num_classes),
        )

        # Initialize weights (Reference: lines 159-162)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classifier.

        Reference: Octopus/eval_bench/train_token_amber.py:250-256

        Args:
            hidden_states: [B, seq_len, d_model]

        Returns:
            action_logits: [B, num_classes]
        """
        B = hidden_states.shape[0]

        # Prepend CLS token (Reference: lines 250-251)
        cls_expanded = self.cls_token.unsqueeze(0).expand(B, 1, -1)
        inputs = torch.cat([cls_expanded, hidden_states.float()], dim=1)

        # TransformerEncoder (Reference: lines 253)
        out = self.transformer(inputs)

        # MLP on CLS token output (Reference: lines 255-256)
        cls_out = out[:, 0, :]  # [B, d_model]
        logits = self.mlp(cls_out)  # [B, num_classes]

        return logits


class OctopusMitigator(TrainableMitigator):
    """
    Octopus: Dynamic Strategy Selection.

    Reference: Octopus/eval_bench/train_token_amber.py

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        classifier: OctopusClassifier instance (optional, can be loaded)
        lambda_decay: M3ID gamma decay (default: 0.02)
        cd_alpha: Contrastive alpha for AvisC/VCD (default: 1.0)
        cd_beta: Plausibility cutoff (default: 0.1)
        noise_step: VCD noise step (default: 500)
        layer_gamma: AvisC layer selection threshold (default: 0.5)
        lamb: AvisC blind token threshold (default: 100.0)
        masking_scheme: AvisC masking type (default: "zeros")
        n_query: Number of query tokens (default: 4)
    """

    name: str = "octopus"

    # Action mapping (Reference: train_token_amber.py actions)
    ACTION_NONE = 0
    ACTION_AVISC = 1
    ACTION_VCD = 2
    ACTION_M3ID = 3

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava",
        classifier: Optional[OctopusClassifier] = None,
        lambda_decay: float = 0.02,
        cd_alpha: float = 1.0,
        cd_beta: float = 0.1,
        noise_step: int = 500,
        layer_gamma: float = 0.5,
        lamb: float = 100.0,
        masking_scheme: str = "zeros",
        n_query: int = 4,
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.lambda_decay = lambda_decay
        self.cd_alpha = cd_alpha
        self.cd_beta = cd_beta
        self.noise_step = noise_step
        self.layer_gamma = layer_gamma
        self.lamb = lamb
        self.masking_scheme = masking_scheme

        # Initialize or set classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            # Infer d_model from model
            d_model = 4096  # Default
            if hasattr(model, 'config'):
                d_model = getattr(model.config, 'hidden_size', d_model)
            self.classifier = OctopusClassifier(d_model=d_model, n_query=n_query)

        # State for AvisC masking
        self._blind_mask = None
        self._img_start = 0
        self._img_end = 0
        self._enable_masking = False
        self._masking_hook_handle = None

    def setup(self) -> None:
        """Move classifier to correct device and setup masking hook."""
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.classifier = self.classifier.to(device)

        # Setup masking hook for AvisC
        layers = self._get_layers()
        if len(layers) > 0:
            self._masking_hook_handle = layers[0].register_forward_pre_hook(
                self._avisc_masking_hook
            )

    def cleanup(self) -> None:
        """Remove masking hook."""
        if self._masking_hook_handle is not None:
            self._masking_hook_handle.remove()
            self._masking_hook_handle = None
        self._blind_mask = None
        self._enable_masking = False

    def _avisc_masking_hook(self, module, args):
        """Forward pre-hook to mask image token embeddings for AvisC."""
        if not self._enable_masking or self._blind_mask is None:
            return args

        hidden_states = args[0] if isinstance(args, tuple) else args
        masked_hidden = hidden_states.clone()
        batch_size = masked_hidden.shape[0]

        for b in range(batch_size):
            if b < self._blind_mask.shape[0]:
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

        Reference: Octopus/avisc_utils/avisc_sample.py:160-179
        """
        # Step 1: Layer selection by image attention
        layer_img_att = []
        for attn in attentions:
            img_attn = attn.mean(dim=1)[:, -1, img_start:img_end]
            layer_img_att.append(img_attn.sum())

        layer_img_att = torch.stack(layer_img_att, dim=0)
        layer_probs = layer_img_att / layer_img_att.sum()

        # Count top-p layers
        sorted_probs = torch.sort(layer_probs, descending=True)[0]
        cumsum = torch.cumsum(sorted_probs, dim=0)
        k = (cumsum < self.layer_gamma).sum().item() + 1
        _, top_k_layers = torch.topk(layer_probs.float(), k, dim=0)
        top_k_layers = top_k_layers.tolist()

        # Step 2: Stack attention from selected layers
        att_stack = torch.stack([
            attentions[i].mean(dim=1)[:, -1, img_start:img_end]
            for i in top_k_layers
        ], dim=1)

        img_att = att_stack.mean(dim=1)

        # Step 3: Threshold
        threshold = img_att.mean() + self.lamb * img_att.std()

        blind_mask = img_att < threshold
        return blind_mask

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return classifier parameters."""
        return list(self.classifier.parameters())

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        chosen_actions: torch.Tensor,
        rejected_actions: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute DPO loss for training.

        Reference: Octopus/eval_bench/train_token_amber.py:611-631

        Args:
            hidden_states: [B, seq_len, d_model]
            chosen_actions: [B] - preferred actions
            rejected_actions: [B] - non-preferred actions
            beta: DPO temperature (default: 1.0)

        Returns:
            loss: Scalar loss tensor
        """
        # If hidden_states are already action scores [B, T, C], use them directly
        if hidden_states.dim() == 3 and hidden_states.shape[-1] == self.classifier.num_classes:
            action_scores = hidden_states
        else:
            action_logits = self.classifier(hidden_states)  # [B, num_classes]
            action_scores = action_logits.unsqueeze(1)  # [B, 1, num_classes]

        # Align label shapes to [B, T]
        if chosen_actions.dim() == 1:
            chosen_actions = chosen_actions.unsqueeze(1)
        if rejected_actions.dim() == 1:
            rejected_actions = rejected_actions.unsqueeze(1)

        len_action = action_scores.shape[1]
        len_text = min(chosen_actions.shape[1], rejected_actions.shape[1])
        if len_action < len_text:
            chosen_actions = chosen_actions[:, :len_action]
            rejected_actions = rejected_actions[:, :len_action]
        else:
            action_scores = action_scores[:, :len_text, :]
            chosen_actions = chosen_actions[:, :len_text]
            rejected_actions = rejected_actions[:, :len_text]

        # DPO loss (reference-free)
        chosen_logps = torch.gather(
            action_scores.log_softmax(-1), 2, chosen_actions[:, :, None]
        ).squeeze(2).sum(-1)
        rejected_logps = torch.gather(
            action_scores.log_softmax(-1), 2, rejected_actions[:, :, None]
        ).squeeze(2).sum(-1)

        pi_logratios = chosen_logps - rejected_logps
        logits = (pi_logratios - 0).sum(-1)
        loss = -F.logsigmoid(beta * logits)

        return loss

    def _apply_avisc(
        self,
        logits_orig: torch.Tensor,
        logits_masked: torch.Tensor,
        is_eval: bool = True,
    ) -> torch.Tensor:
        """
        Apply AvisC contrastive decoding.

        Reference: Octopus/avisc_utils/avisc_sample.py:239-248
        """
        cutoff = torch.log(torch.tensor(self.cd_beta, device=logits_orig.device)) + \
                 logits_orig.max(dim=-1, keepdim=True).values
        diffs = (1 + self.cd_alpha) * logits_orig - self.cd_alpha * logits_masked

        if is_eval:
            return diffs.masked_fill(logits_orig < cutoff, -float("inf"))
        return diffs

    def _apply_vcd(
        self,
        logits_orig: torch.Tensor,
        logits_noised: torch.Tensor,
        is_eval: bool = True,
    ) -> torch.Tensor:
        """
        Apply VCD contrastive decoding.

        Reference: Octopus/avisc_utils/avisc_sample.py:269-276
        """
        cutoff = torch.log(torch.tensor(self.cd_beta, device=logits_orig.device)) + \
                 logits_orig.max(dim=-1, keepdim=True).values
        diffs = (1 + self.cd_alpha) * logits_orig - self.cd_alpha * logits_noised

        if is_eval:
            return diffs.masked_fill(logits_orig < cutoff, -float("inf"))
        return diffs

    def _apply_m3id(
        self,
        logits_orig: torch.Tensor,
        logits_text: torch.Tensor,
        t: int,
        is_eval: bool = True,
    ) -> torch.Tensor:
        """
        Apply M3ID log-softmax reweighting.

        Reference: Octopus/avisc_utils/avisc_sample.py:286-317

        Formula:
            gamma_t = exp(-lambda * t)
            lc = log_softmax(logits_orig)
            lu = log_softmax(logits_text)
            m3id_logit = lc + ((1 - gamma_t) / gamma_t) * (lc - lu)
        """
        gamma_t = math.exp(-self.lambda_decay * t)

        # Log-softmax formulation (Reference: lines 306-308)
        lc = F.log_softmax(logits_orig, dim=-1)
        lu = F.log_softmax(logits_text, dim=-1)
        m3id_logit = lc + ((1 - gamma_t) / gamma_t) * (lc - lu)

        if is_eval:
            cutoff = torch.log(torch.tensor(self.cd_beta, device=logits_orig.device)) + \
                     logits_orig.max(dim=-1, keepdim=True).values
            m3id_logit = m3id_logit.masked_fill(logits_orig < cutoff, -float("inf"))

        return m3id_logit

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        is_eval: bool = True,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with dynamic strategy selection.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            pixel_values: [B, C, H, W]
            is_eval: Whether to apply plausibility cutoff

        Returns:
            Generated token IDs
        """
        if pixel_values is None:
            raise ValueError("Octopus requires pixel_values")

        generated = input_ids.clone()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        output_scores = bool(output_scores)
        return_dict_in_generate = bool(return_dict_in_generate)

        if batch_size > 1:
            sequences = []
            scores = []
            for i in range(batch_size):
                out = self.generate(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1] if attention_mask is not None else None,
                    pixel_values=pixel_values[i:i+1],
                    is_eval=is_eval,
                    output_scores=output_scores,
                    return_dict_in_generate=True,
                    **kwargs,
                )
                sequences.append(out["sequences"])
                scores.append(out["scores"])

            pad_token_id = getattr(self.model.config, "pad_token_id", 0)
            sequences = torch.nn.utils.rnn.pad_sequence(
                [seq.squeeze(0) for seq in sequences],
                batch_first=True,
                padding_value=pad_token_id,
            )
            if return_dict_in_generate:
                return {"sequences": sequences, "scores": scores if output_scores else None}
            return sequences

        text_scores: List[torch.Tensor] = []
        action_scores: List[torch.Tensor] = []

        # Get image token indices
        config = getattr(self.model, 'config', None)
        self._img_start, self._img_end = self._get_image_token_indices(input_ids, config)

        # Prepare perturbed images for VCD
        pixel_values_noised = add_diffusion_noise(pixel_values, self.noise_step)

        # Separate KV caches for each strategy
        past_kv_base = None
        past_kv_avisc = None
        past_kv_vcd = None
        past_kv_m3id = None

        # M3ID step counter (Reference: line 102 - starts at 1)
        t = 1

        for step in range(self.config.max_new_tokens):
            is_first_step = (past_kv_base is None)

            if is_first_step:
                curr_ids = generated
                # cache_position for first step (Qwen2-VL compatibility)
                cache_position = torch.arange(curr_ids.shape[1], device=device)
            else:
                curr_ids = generated[:, -1:]
                # cache_position for subsequent steps
                cache_position = torch.tensor([generated.shape[1] - 1], device=device)

            # Base forward with attention output for AvisC blind token detection
            base_kwargs = {
                'input_ids': curr_ids,
                'attention_mask': attention_mask,
                'output_hidden_states': True,
                'output_attentions': is_first_step,  # Only need attention on first step
                'use_cache': True,
                'return_dict': True,
                'cache_position': cache_position,
            }
            if is_first_step:
                base_kwargs['pixel_values'] = pixel_values
                for key in ['image_sizes', 'image_grid_thw', 'position_ids']:
                    if key in kwargs and kwargs[key] is not None:
                        base_kwargs[key] = kwargs[key]
            else:
                base_kwargs['past_key_values'] = past_kv_base

            with torch.no_grad():
                outputs_base = self.model(**base_kwargs)

            past_kv_base = outputs_base.past_key_values
            logits_base = outputs_base.logits[:, -1, :]

            # Detect blind tokens on first step for AvisC
            if is_first_step and outputs_base.attentions is not None:
                self._blind_mask = self._detect_blind_tokens(
                    outputs_base.attentions,
                    self._img_start,
                    self._img_end,
                )

            # Get hidden state for classifier
            hidden = outputs_base.hidden_states[-1]  # [B, seq_len, d_model]

            # Get action from classifier (per-batch, not collapsed)
            action_logits = self.classifier(hidden)  # [B, num_classes]
            actions = action_logits.argmax(dim=-1)  # [B] - per-batch actions

            # For simplicity in generation, use mode of actions across batch
            # (Reference behavior: single action per step)
            action = actions.mode().values.item() if batch_size > 1 else actions[0].item()

            # Apply selected strategy
            if action == self.ACTION_NONE:
                final_logits = logits_base

            elif action == self.ACTION_AVISC:
                # AvisC: masked embeddings
                avisc_first = past_kv_avisc is None
                avisc_ids = generated if avisc_first else generated[:, -1:]
                avisc_cache_position = (
                    torch.arange(avisc_ids.shape[1], device=device)
                    if avisc_first
                    else torch.tensor([generated.shape[1] - 1], device=device)
                )
                avisc_kwargs = {
                    'input_ids': avisc_ids,
                    'attention_mask': attention_mask,
                    'use_cache': True,
                    'return_dict': True,
                    'cache_position': avisc_cache_position,
                }
                if avisc_first:
                    self._enable_masking = True
                    avisc_kwargs['pixel_values'] = pixel_values
                    for key in ['image_sizes', 'image_grid_thw', 'position_ids']:
                        if key in kwargs and kwargs[key] is not None:
                            avisc_kwargs[key] = kwargs[key]
                else:
                    avisc_kwargs['past_key_values'] = past_kv_avisc

                with torch.no_grad():
                    outputs_avisc = self.model(**avisc_kwargs)
                if avisc_first:
                    self._enable_masking = False
                past_kv_avisc = outputs_avisc.past_key_values

                logits_avisc = outputs_avisc.logits[:, -1, :]
                final_logits = self._apply_avisc(logits_base, logits_avisc, is_eval)

            elif action == self.ACTION_VCD:
                # VCD: noised image
                vcd_first = past_kv_vcd is None
                vcd_ids = generated if vcd_first else generated[:, -1:]
                vcd_cache_position = (
                    torch.arange(vcd_ids.shape[1], device=device)
                    if vcd_first
                    else torch.tensor([generated.shape[1] - 1], device=device)
                )
                vcd_kwargs = {
                    'input_ids': vcd_ids,
                    'attention_mask': attention_mask,
                    'use_cache': True,
                    'return_dict': True,
                    'cache_position': vcd_cache_position,
                }
                if vcd_first:
                    vcd_kwargs['pixel_values'] = pixel_values_noised
                    for key in ['image_sizes', 'image_grid_thw', 'position_ids']:
                        if key in kwargs and kwargs[key] is not None:
                            vcd_kwargs[key] = kwargs[key]
                else:
                    vcd_kwargs['past_key_values'] = past_kv_vcd

                with torch.no_grad():
                    outputs_vcd = self.model(**vcd_kwargs)

                past_kv_vcd = outputs_vcd.past_key_values
                logits_vcd = outputs_vcd.logits[:, -1, :]
                final_logits = self._apply_vcd(logits_base, logits_vcd, is_eval)

            elif action == self.ACTION_M3ID:
                # M3ID: text-only (no image)
                m3id_first = past_kv_m3id is None
                m3id_ids = generated if m3id_first else generated[:, -1:]
                m3id_cache_position = (
                    torch.arange(m3id_ids.shape[1], device=device)
                    if m3id_first
                    else torch.tensor([generated.shape[1] - 1], device=device)
                )
                m3id_kwargs = {
                    'input_ids': m3id_ids,
                    'attention_mask': attention_mask,
                    'use_cache': True,
                    'return_dict': True,
                    'cache_position': m3id_cache_position,
                }
                if not m3id_first:
                    m3id_kwargs['past_key_values'] = past_kv_m3id

                with torch.no_grad():
                    outputs_m3id = self.model(**m3id_kwargs)

                past_kv_m3id = outputs_m3id.past_key_values
                logits_m3id = outputs_m3id.logits[:, -1, :]
                final_logits = self._apply_m3id(logits_base, logits_m3id, t, is_eval)
                t += 1  # Increment t after M3ID (Reference: line 290)

            else:
                final_logits = logits_base

            # Sample
            if output_scores:
                text_scores.append(final_logits)
                action_scores.append(action_logits)

            if self.config.do_sample:
                next_token = sample_top_p(
                    final_logits,
                    top_p=self.config.top_p,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                )
            else:
                next_token = final_logits.argmax(dim=-1, keepdim=True)

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

        if return_dict_in_generate:
            return {
                'sequences': generated,
                'scores': (text_scores, action_scores) if output_scores else None,
            }
        return generated

    def save_pretrained(self, path: str) -> None:
        """
        Save classifier weights.

        Reference: Octopus/eval_bench/train_token_amber.py:626-631
        """
        payload = {
            'cls': self.classifier.cls_token.data,
            'mlp_state_dict': self.classifier.mlp.state_dict(),
            'transformer': self.classifier.transformer.state_dict(),
        }
        if hasattr(self.classifier, 'queries'):
            payload['query'] = self.classifier.queries.data
        torch.save(payload, path)

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        path: str,
        model_type: str = "llava",
        **kwargs,
    ) -> "OctopusMitigator":
        """Load classifier from path."""
        checkpoint = torch.load(path, map_location='cpu')

        # Infer d_model from checkpoint
        d_model = checkpoint['cls'].shape[-1]

        n_query = checkpoint['query'].shape[0] if 'query' in checkpoint else 4
        classifier = OctopusClassifier(d_model=d_model, n_query=n_query)
        classifier.cls_token.data = checkpoint['cls']
        classifier.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        classifier.transformer.load_state_dict(checkpoint['transformer'])
        if 'query' in checkpoint and hasattr(classifier, 'queries'):
            classifier.queries.data = checkpoint['query']

        return cls(model, model_type=model_type, classifier=classifier, **kwargs)
