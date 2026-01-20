"""
VTI: Visual-Textual Intervention

Reference:
    - VTI/vti_utils/llm_layers.py
    - VTI/vti_utils/utils.py

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, ModelHelper


class VTILayer(nn.Module):
    """Reference VTILayer (VTI/vti_utils/llm_layers.py)."""

    def __init__(self, vti_direction: Optional[torch.Tensor], lam: List[float]):
        super().__init__()
        self.vti_direction = vti_direction
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vti_direction is None:
            return x
        norm = torch.norm(x.float(), dim=-1).unsqueeze(-1)
        y = 0
        for i in range(len(self.vti_direction)):
            if x.size(1) < 2:
                lambda_sim = 1.0
                y += self.lam[i] * lambda_sim * F.normalize(self.vti_direction[i], dim=-1).repeat(1, x.shape[1], 1)
            else:
                lambda_sim = 1.0
                y += self.lam[i] * lambda_sim * F.normalize(self.vti_direction[i], dim=-1)
        y = y / len(self.vti_direction)
        x = F.normalize(F.normalize(x.float(), dim=-1) + 0.1 * y, dim=-1) * norm
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


class VTIMitigator(BaseMitigator):
    """
    VTI: Visual-Textual Intervention.

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        textual_vti: (num_layers, n_dirs, hidden)
        visual_vti: (num_layers, n_dirs, num_tokens, hidden)
        alpha_text: list of lam values
        alpha_image: list of lam values
    """

    name: str = "vti"

    def __init__(
            self,
            model: nn.Module,
            model_type: str = "llava",
            textual_vti: Optional[torch.Tensor] = None,
            visual_vti: Optional[torch.Tensor] = None,
            target_layers: Optional[List[int]] = None,
            alpha_text: Union[float, List[float]] = 0.8,
            alpha_image: Union[float, List[float]] = 0.9,
            **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        if target_layers is not None:
            raise ValueError("target_layers is not supported by the reference VTI implementation.")
        self.textual_vti = textual_vti
        self.visual_vti = visual_vti
        self.alpha_text = alpha_text
        self.alpha_image = alpha_image
        self._original_text_mlps: List[tuple] = []
        self._original_visual_mlps: List[tuple] = []

    def setup(self) -> None:
        device = next(self.model.parameters()).device

        if self.textual_vti is not None:
            self.textual_vti = self.textual_vti.to(device)
            layers = _get_layers(self.model)
            if len(self.textual_vti) != len(layers):
                raise AssertionError("len(textual_vti) must match len(layers) in reference VTI.")
            mlp_keywords = ["mlp", "feedforward", "ffn"]
            for i, layer in enumerate(layers):
                original_mlp = _find_module(layer, mlp_keywords)
                layer.mlp = nn.Sequential(original_mlp, VTILayer(self.textual_vti[i], self.alpha_text))
                self._original_text_mlps.append((layer, original_mlp))

        if self.visual_vti is not None:
            self.visual_vti = self.visual_vti.to(device)
            vision_encoder = ModelHelper.get_vision_encoder(self.model, self.model_type)
            vision_model = vision_encoder.vision_model if hasattr(vision_encoder, "vision_model") else vision_encoder
            vision_layers = _get_layers(vision_model)
            if len(self.visual_vti) != len(vision_layers):
                raise AssertionError("len(visual_vti) must match len(vision_layers) in reference VTI.")
            mlp_keywords = ["mlp", "feedforward", "ffn"]
            for i, layer in enumerate(vision_layers):
                original_mlp = _find_module(layer, mlp_keywords)
                layer.mlp = nn.Sequential(original_mlp, VTILayer(self.visual_vti[i], self.alpha_image))
                self._original_visual_mlps.append((layer, original_mlp))

    def cleanup(self) -> None:
        for layer, original_mlp in self._original_text_mlps:
            layer.mlp = original_mlp
        self._original_text_mlps.clear()
        for layer, original_mlp in self._original_visual_mlps:
            layer.mlp = original_mlp
        self._original_visual_mlps.clear()

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
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    @classmethod
    def compute_textual_vti(
            cls,
            model: nn.Module,
            hallucinating_inputs: List[Dict],
            correct_inputs: List[Dict],
            model_type: str = "llava",
            rank: int = 1,
    ) -> torch.Tensor:
        def _svd_flip(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            max_abs_cols = torch.argmax(torch.abs(u), 1)
            i = torch.arange(u.shape[2], device=u.device)
            max_abs_cols = max_abs_cols.unsqueeze(-1)
            signs = torch.sign(torch.gather(u, 1, max_abs_cols))
            u = u * signs
            v = v * signs.view(v.shape[0], -1, 1)
            return u, v

        def _fit_pca(x: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            elif x.ndim != 3:
                raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")
            _, n, d = x.size()
            d = min(n_components, d)
            mean = x.mean(1, keepdim=True)
            z = x - mean
            u, _, vh = torch.linalg.svd(z, full_matrices=False)
            vt = vh
            u, vt = _svd_flip(u, vt)
            components = vt[:, :d]
            return components, mean

        all_diffs = []
        num_layers = None
        hidden_dim = None

        with torch.no_grad():
            for hall, correct in zip(hallucinating_inputs, correct_inputs):
                hall_out = model(**hall, output_hidden_states=True)
                correct_out = model(**correct, output_hidden_states=True)

                hall_hidden = hall_out.hidden_states
                correct_hidden = correct_out.hidden_states

                if num_layers is None:
                    num_layers = len(hall_hidden)
                    hidden_dim = hall_hidden[0].shape[-1]

                hall_last = torch.stack([h[:, -1].squeeze(0) for h in hall_hidden], dim=0)
                correct_last = torch.stack([h[:, -1].squeeze(0) for h in correct_hidden], dim=0)

                hall_flat = hall_last.view(-1).cpu()
                correct_flat = correct_last.view(-1).cpu()

                diff = correct_flat - hall_flat
                all_diffs.append(diff)

        stacked = torch.stack(all_diffs).float()
        components, mean = _fit_pca(stacked, rank)
        direction = (components.sum(dim=1, keepdim=True) + mean).mean(0)
        vti = direction.view(num_layers, hidden_dim).float()

        return vti

    @classmethod
    def compute_visual_vti(
            cls,
            model: nn.Module,
            clean_images: List[Union[torch.Tensor, Tuple[List[torch.Tensor], torch.Tensor]]],
            model_type: str = "llava",
            mask_ratio: float = 0.99,
            rank: int = 1,
            patch_size: int = 14,
            image_grid_thw: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        def _svd_flip(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            max_abs_cols = torch.argmax(torch.abs(u), 1)
            i = torch.arange(u.shape[2], device=u.device)
            max_abs_cols = max_abs_cols.unsqueeze(-1)
            signs = torch.sign(torch.gather(u, 1, max_abs_cols))
            u = u * signs
            v = v * signs.view(v.shape[0], -1, 1)
            return u, v

        def _fit_pca(x: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            elif x.ndim != 3:
                raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")
            _, n, d = x.size()
            d = min(n_components, d)
            mean = x.mean(1, keepdim=True)
            z = x - mean
            u, _, vh = torch.linalg.svd(z, full_matrices=False)
            vt = vh
            u, vt = _svd_flip(u, vt)
            components = vt[:, :d]
            return components, mean

        model_type = ModelHelper.normalize_model_type(model_type)
        vision_encoder = ModelHelper.get_vision_encoder(model, model_type)
        vision_model = vision_encoder.vision_model if hasattr(vision_encoder, "vision_model") else vision_encoder

        def _get_grid_thw(idx: int) -> Optional[torch.Tensor]:
            if image_grid_thw is None:
                return None
            if isinstance(image_grid_thw, torch.Tensor):
                if image_grid_thw.ndim == 1:
                    return image_grid_thw.unsqueeze(0)
                return image_grid_thw[idx:idx + 1]
            grid = image_grid_thw[idx]
            if isinstance(grid, torch.Tensor):
                return grid.unsqueeze(0) if grid.ndim == 1 else grid
            return torch.tensor(grid).unsqueeze(0)

        def _collect_hidden_states(img: torch.Tensor, grid_thw: Optional[torch.Tensor]) -> List[torch.Tensor]:
            out = vision_model(
                img,
                output_hidden_states=True,
                return_dict=True,
                **({"grid_thw": grid_thw} if grid_thw is not None else {}),
            )
            return list(out.hidden_states)

        def _average_hidden_states(states_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
            if len(states_list) == 1:
                return states_list[0]
            num_layers = len(states_list[0])
            averaged: List[torch.Tensor] = []
            for layer_idx in range(num_layers):
                stacked = torch.stack([states[layer_idx] for states in states_list], dim=0)
                averaged.append(stacked.mean(dim=0))
            return averaged

        all_diffs = []
        n_layers = None
        n_tokens = None
        feat_dim = None

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if model_type in ("qwen2_vl", "qwen2_5_vl") and image_grid_thw is None:
            raise ValueError("Qwen2-VL/2.5-VL requires image_grid_thw for visual VTI.")

        with torch.no_grad():
            for idx, item in enumerate(clean_images):
                grid_thw = _get_grid_thw(idx)
                if grid_thw is not None:
                    grid_thw = grid_thw.to(device)

                if not (isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], list)):
                    raise ValueError("Reference VTI expects [masked_trials, clean] inputs for visual VTI.")

                masked_trials = item[0]
                img = item[1]

                img = img.to(device, dtype)
                if img.dim() == 4 and img.shape[0] == 1:
                    img = img.squeeze(0)
                img_for_model = img.unsqueeze(0) if img.dim() == 3 else img

                clean_hs = _collect_hidden_states(img_for_model, grid_thw)

                masked_states = []
                for masked in masked_trials:
                    masked = masked.to(device, dtype)
                    if masked.dim() == 4 and masked.shape[0] == 1:
                        masked = masked.squeeze(0)
                    masked_for_model = masked.unsqueeze(0) if masked.dim() == 3 else masked
                    masked_states.append(_collect_hidden_states(masked_for_model, grid_thw))
                masked_hs = _average_hidden_states(masked_states)

                if n_layers is None:
                    n_layers = len(clean_hs)
                    n_tokens = clean_hs[0].shape[1]
                    feat_dim = clean_hs[0].shape[-1]

                clean_stack = torch.stack([h.cpu() for h in clean_hs], dim=0)
                masked_stack = torch.stack([h.cpu() for h in masked_hs], dim=0)

                clean_flat = clean_stack.squeeze(1).permute(1, 0, 2).reshape(n_tokens, -1)
                masked_flat = masked_stack.squeeze(1).permute(1, 0, 2).reshape(n_tokens, -1)

                diff = masked_flat - clean_flat
                all_diffs.append(diff)

        stacked = torch.stack(all_diffs, dim=1).float()
        components, mean = _fit_pca(stacked, rank)
        direction = (components.sum(dim=1, keepdim=True) + mean).mean(1)
        visual_vti = direction.view(n_layers, n_tokens, feat_dim).float()

        return visual_vti
