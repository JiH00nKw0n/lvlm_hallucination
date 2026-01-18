"""
Octopus: Dynamic Strategy Selection

Learns to dynamically select between hallucination mitigation strategies
per token using a trained classifier.

Reference:
    - Octopus/eval_bench/train_token_amber.py:132-256 (MyModel classifier)
    - Octopus/eval_bench/train_token_amber.py:611-631 (DPO loss)

Key Implementation Notes:
    1. Classifier outputs per-token action logits
    2. Actions: 0=None, 1=AvisC, 2=VCD, 3=M3ID
    3. Separate KV caches for each strategy
    4. DPO training with per-token log-probs

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL, InstructBLIP
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

    Architecture:
        - Learnable queries + CLS token
        - Small transformer (2 layers)
        - MLP head

    Input: Hidden states [B, seq_len, d_model]
    Output: Action logits [B, seq_len, num_classes]
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_classes: int = 4,
        num_queries: int = 8,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Learnable queries and CLS token
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)

        # Small transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, seq_len, d_model]

        Returns:
            action_logits: [B, seq_len, num_classes]
        """
        B, T, D = hidden_states.shape

        # Project input
        x = self.input_proj(hidden_states)

        # Process each token position
        all_logits = []
        for t in range(T):
            # Get context up to position t
            context = x[:, :t+1, :]  # [B, t+1, D]

            # Add queries and CLS
            queries = self.queries.expand(B, -1, -1)
            cls = self.cls_token.expand(B, -1, -1)
            combined = torch.cat([cls, queries, context], dim=1)  # [B, 1+Q+t+1, D]

            # Transformer
            out = self.transformer(combined)

            # Use CLS token output
            cls_out = out[:, 0, :]  # [B, D]
            logits = self.mlp(cls_out)  # [B, num_classes]
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)  # [B, T, num_classes]

    def forward_single(self, hidden_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward for single token (used during generation).

        Args:
            hidden_state: [B, d_model] - current token hidden state
            context: [B, context_len, d_model] - previous context

        Returns:
            action_logits: [B, num_classes]
        """
        B = hidden_state.shape[0]

        # Combine with context
        x = self.input_proj(hidden_state.unsqueeze(1))
        if context is not None and context.shape[1] > 0:
            context_proj = self.input_proj(context)
            x = torch.cat([context_proj, x], dim=1)

        # Add queries and CLS
        queries = self.queries.expand(B, -1, -1)
        cls = self.cls_token.expand(B, -1, -1)
        combined = torch.cat([cls, queries, x], dim=1)

        # Transformer
        out = self.transformer(combined)
        cls_out = out[:, 0, :]

        return self.mlp(cls_out)


class OctopusMitigator(TrainableMitigator):
    """
    Octopus: Dynamic Strategy Selection.

    Reference: Octopus/eval_bench/train_token_amber.py

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl, instructblip
        classifier: OctopusClassifier instance (optional, can be loaded)
        lambda_decay: M3ID gamma decay (default: 0.02)
        cd_alpha: Contrastive alpha for AvisC/VCD (default: 1.0)
        cd_beta: Plausibility cutoff (default: 0.1)
        noise_step: VCD noise step (default: 500)
    """

    name: str = "octopus"

    # Action mapping
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
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.lambda_decay = lambda_decay
        self.cd_alpha = cd_alpha
        self.cd_beta = cd_beta
        self.noise_step = noise_step

        # Initialize or set classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            # Infer d_model from model
            d_model = 4096  # Default
            if hasattr(model, 'config'):
                d_model = getattr(model.config, 'hidden_size', d_model)
            self.classifier = OctopusClassifier(d_model=d_model)

        self._masking_enabled = False
        self._blind_mask = None
        self._img_start = 0
        self._img_end = 0

    def setup(self) -> None:
        """Move classifier to correct device."""
        device = next(self.model.parameters()).device
        self.classifier = self.classifier.to(device)

    def cleanup(self) -> None:
        """No cleanup needed."""
        pass

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
            chosen_actions: [B, seq_len] - preferred actions per token
            rejected_actions: [B, seq_len] - non-preferred actions per token
            beta: DPO temperature (default: 1.0)

        Returns:
            loss: Scalar loss tensor
        """
        # Get action logits
        action_logits = self.classifier(hidden_states)  # [B, T, num_classes]

        # Align lengths
        T = min(action_logits.shape[1], chosen_actions.shape[1], rejected_actions.shape[1])
        action_logits = action_logits[:, :T, :]
        chosen_actions = chosen_actions[:, :T]
        rejected_actions = rejected_actions[:, :T]

        # Compute log probabilities
        log_probs = action_logits.log_softmax(dim=-1)  # [B, T, num_classes]

        # Gather chosen and rejected log probs
        chosen_logps = torch.gather(
            log_probs, 2, chosen_actions.unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)  # [B]

        rejected_logps = torch.gather(
            log_probs, 2, rejected_actions.unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)  # [B]

        # DPO loss (reference-free)
        logits = chosen_logps - rejected_logps
        loss = -F.logsigmoid(beta * logits).mean()

        return loss

    def _apply_avisc(
        self,
        logits_orig: torch.Tensor,
        logits_masked: torch.Tensor,
    ) -> torch.Tensor:
        """Apply AvisC contrastive decoding."""
        return (1 + self.cd_alpha) * logits_orig - self.cd_alpha * logits_masked

    def _apply_vcd(
        self,
        logits_orig: torch.Tensor,
        logits_noised: torch.Tensor,
    ) -> torch.Tensor:
        """Apply VCD contrastive decoding."""
        cutoff = torch.log(torch.tensor(self.cd_beta, device=logits_orig.device)) + \
                 logits_orig.max(dim=-1, keepdim=True).values
        cd_logits = (1 + self.cd_alpha) * logits_orig - self.cd_alpha * logits_noised
        return cd_logits.masked_fill(logits_orig < cutoff, -float("inf"))

    def _apply_m3id(
        self,
        logits_orig: torch.Tensor,
        logits_text: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Apply M3ID entropy reweighting.

        Reference: Octopus M3ID implementation
        """
        gamma = math.exp(-self.lambda_decay * (step + 1))

        # Entropy of text-only
        probs_text = F.softmax(logits_text, dim=-1)
        entropy = -(probs_text * torch.log(probs_text + 1e-10)).sum(dim=-1, keepdim=True)

        # Reweight
        alpha = gamma * entropy
        m3id_logits = logits_orig + alpha * (logits_orig - logits_text)

        return m3id_logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with dynamic strategy selection.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            pixel_values: [B, C, H, W]

        Returns:
            Generated token IDs
        """
        if pixel_values is None:
            raise ValueError("Octopus requires pixel_values")

        generated = input_ids.clone()
        device = input_ids.device

        # Get image token indices
        config = getattr(self.model, 'config', None)
        self._img_start, self._img_end = self._get_image_token_indices(input_ids, config)

        # Prepare perturbed images
        pixel_values_noised = add_diffusion_noise(pixel_values, self.noise_step)

        # Separate KV caches for each strategy
        past_kv_base = None
        past_kv_avisc = None  # Masked embeddings
        past_kv_vcd = None    # Noised image
        past_kv_m3id = None   # Text-only

        # Context for classifier
        context_hidden = None

        for step in range(self.config.max_new_tokens):
            is_first_step = (past_kv_base is None)

            if is_first_step:
                curr_ids = generated
            else:
                curr_ids = generated[:, -1:]

            # Base forward
            base_kwargs = {
                'input_ids': curr_ids,
                'attention_mask': attention_mask,
                'output_hidden_states': True,
                'use_cache': True,
                'return_dict': True,
            }
            if is_first_step:
                base_kwargs['pixel_values'] = pixel_values
                for key in ['image_sizes', 'image_grid_thw', 'position_ids',
                            'qformer_input_ids', 'qformer_attention_mask']:
                    if key in kwargs and kwargs[key] is not None:
                        base_kwargs[key] = kwargs[key]
            else:
                base_kwargs['past_key_values'] = past_kv_base

            with torch.no_grad():
                outputs_base = self.model(**base_kwargs)

            past_kv_base = outputs_base.past_key_values
            logits_base = outputs_base.logits[:, -1, :]

            # Get hidden state for classifier
            hidden = outputs_base.hidden_states[-1][:, -1, :]  # [B, d_model]

            # Get action from classifier
            action_logits = self.classifier.forward_single(hidden, context_hidden)
            action = action_logits.argmax(dim=-1).item()

            # Update context
            if context_hidden is None:
                context_hidden = hidden.unsqueeze(1)
            else:
                context_hidden = torch.cat([context_hidden, hidden.unsqueeze(1)], dim=1)
                # Limit context length
                if context_hidden.shape[1] > 32:
                    context_hidden = context_hidden[:, -32:, :]

            # Apply selected strategy
            if action == self.ACTION_NONE:
                final_logits = logits_base

            elif action == self.ACTION_AVISC:
                # AvisC: masked embeddings (simplified - zero image tokens)
                # In full implementation, would need hook-based masking
                final_logits = logits_base  # Simplified

            elif action == self.ACTION_VCD:
                # VCD: noised image
                vcd_kwargs = base_kwargs.copy()
                if is_first_step:
                    vcd_kwargs['pixel_values'] = pixel_values_noised
                else:
                    vcd_kwargs['past_key_values'] = past_kv_vcd

                with torch.no_grad():
                    outputs_vcd = self.model(**vcd_kwargs)

                past_kv_vcd = outputs_vcd.past_key_values
                logits_vcd = outputs_vcd.logits[:, -1, :]
                final_logits = self._apply_vcd(logits_base, logits_vcd)

            elif action == self.ACTION_M3ID:
                # M3ID: text-only (no image)
                m3id_kwargs = {
                    'input_ids': curr_ids,
                    'attention_mask': attention_mask,
                    'use_cache': True,
                    'return_dict': True,
                }
                if not is_first_step:
                    m3id_kwargs['past_key_values'] = past_kv_m3id

                with torch.no_grad():
                    outputs_m3id = self.model(**m3id_kwargs)

                past_kv_m3id = outputs_m3id.past_key_values
                logits_m3id = outputs_m3id.logits[:, -1, :]
                final_logits = self._apply_m3id(logits_base, logits_m3id, step)

            else:
                final_logits = logits_base

            # Sample
            if self.config.do_sample:
                next_token = sample_top_p(final_logits, self.config.top_p, self.config.temperature)
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

        return generated

    def save_pretrained(self, path: str) -> None:
        """Save classifier weights."""
        torch.save({
            'queries': self.classifier.queries.data,
            'cls_token': self.classifier.cls_token.data,
            'input_proj': self.classifier.input_proj.state_dict(),
            'transformer': self.classifier.transformer.state_dict(),
            'mlp': self.classifier.mlp.state_dict(),
        }, path)

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
        d_model = checkpoint['queries'].shape[-1]

        classifier = OctopusClassifier(d_model=d_model)
        classifier.queries.data = checkpoint['queries']
        classifier.cls_token.data = checkpoint['cls_token']
        classifier.input_proj.load_state_dict(checkpoint['input_proj'])
        classifier.transformer.load_state_dict(checkpoint['transformer'])
        classifier.mlp.load_state_dict(checkpoint['mlp'])

        return cls(model, model_type=model_type, classifier=classifier, **kwargs)
