"""
VISTA: Visual Information Steering with Attention

Applies Visual Steering Vectors (VSV) to MLP outputs to reduce hallucination.

Reference:
    - VISTA/llm_layers.py:7-34 (VSVLayer)
    - VISTA/llm_layers.py:132-150 (add_vsv_layers)
    - VISTA/steering_vector.py (VSV computation)

Supports: LLaVA, LLaVA-NeXT
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator
from .vista_utils.steering_vector import obtain_vsv


class VSVLayer(nn.Module):
    """Reference VSVLayer (VISTA/llm_layers.py)."""

    def __init__(
            self,
            vsv: Optional[torch.Tensor],
            lam: List[float],
            simple_mode: bool = False,
    ) -> None:
        super().__init__()
        self.vsv = vsv
        self.lam = lam
        self.simple_mode = simple_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vsv is None:
            return x
        x = x.float()
        original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        y = 0
        if self.simple_mode:
            vsv = self.vsv[0]
            lam_schedule = self.lam[0]
            y = lam_schedule * F.normalize(vsv, dim=-1).repeat(1, x.shape[1], 1)
            x = F.normalize(F.normalize(x, p=2, dim=-1) + y, p=2, dim=-1) * original_norm
        else:
            for i in range(len(self.vsv)):
                lambda_sim = 1.0 + torch.max(
                    torch.tensor([0.]).to(x.device),
                    F.cosine_similarity(x, -self.vsv[i][None, None, :], dim=-1),
                ).unsqueeze(-1)
                y += self.lam[i] * lambda_sim * F.normalize(self.vsv[i], dim=-1).repeat(1, x.shape[1], 1)
            y = y / len(self.vsv)
            x = F.normalize(F.normalize(x.float(), p=2, dim=-1) + y, p=2, dim=-1) * original_norm
        return x.half()


def _get_nested_attr(obj: nn.Module, attr_path: str) -> nn.Module:
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _find_longest_modulelist(model: nn.Module, path: str = "") -> Tuple[str, int]:
    longest_path = path
    longest_len = 0
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name
        child_path, child_len = _find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path
    return longest_path, longest_len


def _find_module(block: nn.Module, keywords: List[str]) -> nn.Module:
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def _get_layers(model: nn.Module) -> nn.ModuleList:
    longest_path, _ = _find_longest_modulelist(model)
    if not longest_path:
        raise ValueError(f"Cannot find layers in model: {type(model)}")
    return _get_nested_attr(model, longest_path)


class VISTAMitigator(BaseMitigator):
    """
    VISTA: Visual Information Steering with Attention.

    Args:
        model: The VLM model
        model_type: llava, llava_next
        vsv: Visual Steering Vector tensor (num_layers, n_dirs, hidden_dim)
        lam: Steering strength list
        simple_mode: If True, skip lambda_sim weighting
        logits_layers: "start,end" string for SLA
        logits_alpha: Blend ratio for SLA
        tar_layers: "s,e" or "s_s,s_e,t_s,t_e" range string (reference)
    """

    name: str = "vista"

    def __init__(
            self,
            model: nn.Module,
            model_type: str = "llava",
            vsv: Optional[torch.Tensor] = None,
            lam: float = 0.1,
            simple_mode: bool = False,
            logits_layers: Optional[str] = None,
            logits_alpha: float = 0.3,
            tar_layers: Optional[str] = None,
            target_layers: Optional[List[int]] = None,
            **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        if target_layers is not None:
            raise ValueError("target_layers is not supported by the reference VISTA implementation.")
        self.vsv = vsv
        self.lam = lam
        self.simple_mode = simple_mode
        self.logits_layers = logits_layers
        self.logits_alpha = logits_alpha
        self.tar_layers = tar_layers
        self._original_mlps: List[tuple] = []

    def setup(self) -> None:
        layers = _get_layers(self.model.language_model if hasattr(self.model, "language_model") else self.model)
        if self.vsv is None:
            return

        if self.tar_layers is None:
            if len(self.vsv) != len(layers):
                raise AssertionError("len(vsv) must match len(layers) in reference VISTA.")
            layer_indices = list(range(len(layers)))
            vsv_indices = list(range(len(self.vsv)))
        else:
            parts = self.tar_layers.split(",")
            if len(parts) == 2:
                s_idx, e_idx = map(int, parts)
                layer_indices = list(range(s_idx, e_idx))
                vsv_indices = list(range(s_idx, e_idx))
            elif len(parts) == 4:
                s_s_idx, s_e_idx, t_s_idx, t_e_idx = map(int, parts)
                layer_indices = list(range(s_s_idx, s_e_idx))
                vsv_indices = list(range(t_s_idx, t_e_idx))
            else:
                raise ValueError("Invalid target layers")

        mlp_keywords = ["mlp", "feedforward", "ffn"]
        for layer_idx, vsv_idx in zip(layer_indices, vsv_indices):
            layer = layers[layer_idx]
            original_mlp = _find_module(layer, mlp_keywords)
            layer.mlp = nn.Sequential(original_mlp, VSVLayer(self.vsv[vsv_idx], self.lam, self.simple_mode))
            self._original_mlps.append((layer, original_mlp))

    def cleanup(self) -> None:
        for layer, original_mlp in self._original_mlps:
            layer.mlp = original_mlp
        self._original_mlps.clear()
        for attr in ["logits_aug", "logits_layers", "logits_alpha"]:
            if hasattr(self.model, attr):
                delattr(self.model, attr)

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
        }
        gen_kwargs.update(kwargs)

        if self.logits_layers is not None:
            self.model.logits_aug = True
            self.model.logits_layers = self.logits_layers
            self.model.logits_alpha = self.logits_alpha
            gen_kwargs["output_hidden_states"] = True

        try:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        finally:
            if self.logits_layers is not None:
                for attr in ["logits_aug", "logits_layers", "logits_alpha"]:
                    if hasattr(self.model, attr):
                        delattr(self.model, attr)

    @classmethod
    def compute_vsv(
            cls,
            model: nn.Module,
            positive_inputs: List[dict],
            negative_inputs: List[dict],
            model_type: str = "llava",
            rank: int = 1,
    ) -> torch.Tensor:
        kwargs_list = [(neg, pos) for pos, neg in zip(positive_inputs, negative_inputs)]
        llm_model = model.language_model if hasattr(model, "language_model") else model
        vsv, _ = obtain_vsv(llm_model, kwargs_list, rank=rank)
        return vsv
