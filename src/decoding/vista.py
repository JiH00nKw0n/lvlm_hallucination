"""
VISTA: Visual Information Steering with Attention

Applies Visual Steering Vectors (VSV) to MLP outputs to reduce hallucination.

Reference:
    - VISTA/llm_layers.py:7-34 (VSVLayer)
    - VISTA/llm_layers.py:132-150 (add_vsv_layers)
    - VISTA/steering_vector.py (VSV computation)

Key Implementation Notes:
    1. VSV direction is computed via PCA on last-token hidden states per layer
    2. lambda_sim weighting: 1 + max(0, cos_sim(x, -vsv))
    3. Supports simple_mode (no lambda_sim) and full mode

Formula (full mode):
    y = sum_i(lam[i] * lambda_sim * normalize(vsv[i])) / len(vsv)
    x' = normalize(normalize(x) + y) * ||x||

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, MitigatorConfig, ModelHelper


class VISTAMitigator(BaseMitigator):
    """
    VISTA: Visual Information Steering with Attention.

    Reference: VISTA/llm_layers.py:7-34

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        vsv: Visual Steering Vector tensor of shape (num_layers, hidden_dim)
             or list of (num_layers, hidden_dim) tensors for multi-vector mode
        target_layers: Layers to apply VSV (default: all layers)
        lam: Steering strength (scalar or list per vector, default: 0.1)
        simple_mode: If True, skip lambda_sim weighting (default: False)
        logits_layers: Layers for Self-Logits Augmentation (optional)
        logits_alpha: Blend ratio for SLA (default: 0.3)
    """

    name: str = "vista"

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava",
        vsv: Optional[torch.Tensor] = None,
        target_layers: Optional[List[int]] = None,
        lam: float = 0.1,
        simple_mode: bool = False,
        logits_layers: Optional[List[int]] = None,
        logits_alpha: float = 0.3,
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)

        num_layers = self._get_num_layers()
        self.target_layers = target_layers or list(range(num_layers))
        self.simple_mode = simple_mode
        self.logits_layers = logits_layers
        self.logits_alpha = logits_alpha

        # lam follows reference: scalar or list shared across layers (per-vector weights)
        self.lam = lam

        # Store VSV vectors
        self.vsv: Dict[int, torch.Tensor] = {}
        if vsv is not None:
            self._set_vsv(vsv)

        # Hook handles
        self._hooks: List = []

    def _set_vsv(self, vsv: torch.Tensor) -> None:
        """
        Set VSV tensor.

        Args:
            vsv: Tensor of shape (num_layers, hidden_dim) or
                 list of such tensors for multi-vector mode
        """
        device = next(self.model.parameters()).device

        if isinstance(vsv, list):
            # Multi-vector mode
            for i in self.target_layers:
                if i < len(vsv[0]):
                    self.vsv[i] = [v[i].to(device) for v in vsv]
        else:
            # Single vector mode
            for i in self.target_layers:
                if i < vsv.shape[0]:
                    self.vsv[i] = vsv[i].to(device)

    def _create_hook(self, layer_idx: int):
        """
        Create forward hook for VSV steering.

        Reference: VISTA/llm_layers.py:15-31

        The hook applies:
            if simple_mode:
                y = lam * normalize(vsv)
            else:
                lambda_sim = 1 + max(0, cos_sim(x, -vsv))
                y = lam * lambda_sim * normalize(vsv)
            x' = normalize(normalize(x) + y) * ||x||
        """
        def hook_fn(module, input, output):
            if layer_idx not in self.vsv:
                return output

            x = output
            if isinstance(output, tuple):
                x = output[0]

            x_float = x.float()
            original_norm = torch.norm(x_float, p=2, dim=-1, keepdim=True)

            vsv_list = self.vsv[layer_idx]
            if not isinstance(vsv_list, list):
                vsv_list = [vsv_list]

            lam_val = self.lam
            if isinstance(lam_val, (int, float)):
                lam_val = [lam_val] * len(vsv_list)
            else:
                lam_val = list(lam_val)
                if len(lam_val) == 1 and len(vsv_list) > 1:
                    lam_val = lam_val * len(vsv_list)

            if self.simple_mode:
                # Simple mode: no lambda_sim
                vsv = vsv_list[0].to(x_float.dtype)
                y = lam_val[0] * F.normalize(vsv, dim=-1).unsqueeze(0).unsqueeze(0)
                y = y.expand(x.shape[0], x.shape[1], -1)
            else:
                # Full mode with lambda_sim weighting
                y = torch.zeros_like(x_float)
                for i, vsv in enumerate(vsv_list):
                    vsv = vsv.to(x_float.dtype)
                    # lambda_sim = 1 + max(0, cos_sim(x, -vsv))
                    # Reference: VISTA/llm_layers.py:27
                    cos_sim = F.cosine_similarity(
                        x_float,
                        -vsv.unsqueeze(0).unsqueeze(0).expand_as(x_float),
                        dim=-1
                    )  # [B, seq]
                    lambda_sim = 1.0 + torch.clamp(cos_sim, min=0.0).unsqueeze(-1)  # [B, seq, 1]

                    vsv_norm = F.normalize(vsv, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, D]
                    vsv_expanded = vsv_norm.expand(x.shape[0], x.shape[1], -1)

                    y = y + lam_val[i] * lambda_sim * vsv_expanded

                y = y / len(vsv_list)

            # Apply steering: x' = normalize(normalize(x) + y) * ||x||
            x_norm = F.normalize(x_float, p=2, dim=-1)
            steered = F.normalize(x_norm + y, p=2, dim=-1) * original_norm

            steered = steered.half()

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        return hook_fn

    def setup(self) -> None:
        """Register forward hooks on MLP layers."""
        layers = self._get_layers()

        for layer_idx in self.target_layers:
            if layer_idx in self.vsv and layer_idx < len(layers):
                mlp = ModelHelper.get_mlp_module(layers[layer_idx])
                handle = mlp.register_forward_hook(self._create_hook(layer_idx))
                self._hooks.append(handle)

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with VISTA steering.

        If logits_layers is set, uses Self-Logits Augmentation (SLA).
        Otherwise, uses standard model.generate() with hooks active.
        """
        if self.logits_layers is None:
            # Standard generation with hooks
            gen_kwargs = {
                'max_new_tokens': self.config.max_new_tokens,
                'do_sample': self.config.do_sample,
                'temperature': self.config.temperature,
                'top_k': self.config.top_k,
                'top_p': self.config.top_p,
            }
            gen_kwargs.update(kwargs)

            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        else:
            # Self-Logits Augmentation
            return self._generate_with_sla(input_ids, attention_mask, **kwargs)

    def _generate_with_sla(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate with Self-Logits Augmentation."""
        from .base import sample_top_p

        generated = input_ids.clone()
        past_key_values = None
        device = input_ids.device

        _, lm_head = self._get_norm_and_lm_head()

        for _ in range(self.config.max_new_tokens):
            if past_key_values is None:
                curr_ids = generated
                cache_position = torch.arange(curr_ids.shape[1], device=device)
            else:
                curr_ids = generated[:, -1:]
                cache_position = torch.tensor([generated.shape[1] - 1], device=device)

            outputs = self.model(
                input_ids=curr_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
                cache_position=cache_position,
                **{k: v for k, v in kwargs.items() if past_key_values is None or k not in ['pixel_values', 'image_sizes', 'image_grid_thw']},
            )

            logits = outputs.logits[:, -1, :]

            # Self-Logits Augmentation
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
            aug_logits = []
            for layer_idx in self.logits_layers:
                if layer_idx < len(hidden_states):
                    aug_logits.append(lm_head(hidden_states[layer_idx])[:, -1, :])

            if aug_logits:
                aug_logits = torch.stack(aug_logits).mean(dim=0)
                logits = self.logits_alpha * aug_logits + (1 - self.logits_alpha) * logits

            # Sample
            if self.config.do_sample:
                next_token = sample_top_p(
                    logits,
                    top_p=self.config.top_p,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = outputs.past_key_values

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

    @classmethod
    def compute_vsv(
        cls,
        model: nn.Module,
        positive_inputs: List[Dict],
        negative_inputs: List[Dict],
        model_type: str = "llava",
        rank: int = 1,
    ) -> torch.Tensor:
        """
        Compute Visual Steering Vector from positive/negative demonstrations.

        Reference: VISTA/steering_vector.py

        The approach:
            1. Get hidden states for positive (non-hallucinating) and negative (hallucinating) inputs
            2. Extract LAST TOKEN from each layer
            3. Compute difference and apply PCA

        Args:
            model: The VLM model
            positive_inputs: List of inputs for non-hallucinating responses
            negative_inputs: List of inputs for hallucinating responses
            model_type: Model type string
            rank: PCA rank (default: 1)

        Returns:
            vsv: Tensor of shape (num_layers, hidden_dim)
        """
        def _svd_flip(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            max_abs_cols = torch.argmax(torch.abs(u), 0)
            i = torch.arange(u.shape[1], device=u.device)
            signs = torch.sign(u[max_abs_cols, i])
            u = u * signs
            v = v * signs.view(-1, 1)
            return u, v

        def _fit_pca(x: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor]:
            n, d = x.size()
            d = min(n_components, d)
            mean = x.mean(0, keepdim=True)
            z = x - mean
            u, _, vh = torch.linalg.svd(z, full_matrices=False)
            vt = vh
            u, vt = _svd_flip(u, vt)
            components = vt[:d]
            return components, mean

        all_diffs = []
        num_layers = None
        hidden_dim = None

        with torch.no_grad():
            for pos, neg in zip(positive_inputs, negative_inputs):
                pos_out = model(**pos, output_hidden_states=True)
                neg_out = model(**neg, output_hidden_states=True)

                # hidden_states: tuple of (num_layers+1) tensors
                pos_hidden = pos_out.hidden_states[1:]  # Skip embedding
                neg_hidden = neg_out.hidden_states[1:]

                if num_layers is None:
                    num_layers = len(pos_hidden)
                    hidden_dim = pos_hidden[0].shape[-1]

                # Extract LAST TOKEN from each layer
                pos_last = torch.cat([h[:, -1] for h in pos_hidden], dim=0)  # [num_layers, hidden]
                neg_last = torch.cat([h[:, -1] for h in neg_hidden], dim=0)

                # Flatten and compute difference
                pos_flat = pos_last.view(-1)
                neg_flat = neg_last.view(-1)

                diff = (pos_flat - neg_flat).cpu()
                all_diffs.append(diff)

        # PCA (reference implementation)
        stacked = torch.stack(all_diffs).float()
        components, mean = _fit_pca(stacked, rank)
        direction = (components.sum(dim=0, keepdim=True) + mean).mean(0)
        vsv = direction.view(num_layers, hidden_dim).float()

        return vsv
