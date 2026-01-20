"""
SAVE: Sparse Autoencoder Steering (LLaVA-NeXT).

Uses pre-trained SAE weights (flybamboo/vlm-saes) and feature indices
from offline identification (best_separation_feature.json) to steer
hidden states at a target layer.
"""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from .base import BaseMitigator
from .save_utils import (
    SAE,
    load_feature_indices,
    remove_module_prefix,
    resolve_or_generate_feature_path,
    resolve_sae_checkpoint,
)


class SAVEMitigator(BaseMitigator):
    """
    SAVE mitigator (steer hidden states using SAE feature directions).

    Args:
        model: The VLM model (LLaVA-NeXT only)
        model_type: llava_next
        layer: Layer index to hook (default: 24)
        steer_alpha: Steering strength (default: 5.0)
        mode: "faith", "hal", or "both"
        feature_path: Path to best_separation_feature.json
        sae_repo: HuggingFace repo id for SAE weights (flybamboo/vlm-saes)
        sae_filename: SAE checkpoint filename (e.g., sae_final_layer24_8x_norm13.996978.pkl)
    """

    name: str = "SAVEMitigator"

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava_next",
        layer: int = 24,
        steer_alpha: float = 5.0,
        mode: str = "both",
        feature_path: Optional[str] = None,
        feature_save_dir: str = "src/decoding/save_utils",
        sae_repo: str = "flybamboo/vlm-saes",
        sae_filename: str = "sae_final_layer24_8x_norm13.996978.pkl",
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.layer = layer
        self.steer_alpha = steer_alpha
        self.mode = mode
        self.feature_path = feature_path
        self.feature_save_dir = feature_save_dir
        self.sae_repo = sae_repo
        self.sae_filename = sae_filename

        self._handle: Optional[RemovableHandle] = None
        self._sae: Optional[SAE] = None
        self._faith_vec: Optional[torch.Tensor] = None
        self._hal_vec: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if self.model_type != "llava_next":
            raise ValueError("SAVEMitigator currently supports llava_next only.")

        if not self.feature_path:
            raise ValueError("SAVEMitigator requires feature_path.")

        feature_dir = Path(self.feature_save_dir)
        auto_path = feature_dir / "best_separation_feature.json"
        if self.feature_path is None and not auto_path.exists():
            print("[save] best_separation_feature.json not found; generating from cached activations.")

        resolved_feature_path = resolve_or_generate_feature_path(
            self.feature_path,
            self.feature_save_dir,
            self.layer,
        )
        print(f"[save] using feature file: {resolved_feature_path}")
        faith_indices, hal_indices = load_feature_indices(resolved_feature_path)
        if not faith_indices and not hal_indices:
            raise ValueError("feature_path has no usable indices.")

        ckpt_path = resolve_sae_checkpoint(self.sae_repo, self.sae_filename)
        state = torch.load(ckpt_path, map_location="cpu")

        sae = SAE().to(self.model.device).half()
        sae.load_state_dict(remove_module_prefix(state))
        sae.eval()
        self._sae = sae

        self._faith_vec = self._build_faith_vector(faith_indices)
        self._hal_vec = self._build_hal_vector(hal_indices)

        layers = self._get_layers()
        if self.layer >= len(layers) or self.layer < 0:
            raise ValueError(f"Layer index {self.layer} out of range (0-{len(layers)-1}).")

        hooked_module = layers[self.layer]
        self._handle = hooked_module.register_forward_hook(self._hook)

    def cleanup(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._sae = None
        self._faith_vec = None
        self._hal_vec = None

    def _build_faith_vector(self, indices: List[int]) -> Optional[torch.Tensor]:
        if not indices or self._sae is None:
            return None
        vecs = []
        for idx in indices:
            direction = self._sae.fc2.weight[:, idx].to(self.model.device)
            vecs.append(F.normalize(direction, dim=0))
        if len(vecs) == 1:
            return vecs[0]
        return F.normalize(torch.stack(vecs, dim=0).mean(dim=0), dim=0)

    def _build_hal_vector(self, indices: List[int]) -> Optional[torch.Tensor]:
        if not indices or self._sae is None:
            return None
        direction = self._sae.fc2.weight[:, indices[0]].to(self.model.device)
        return F.normalize(direction, dim=0)

    def _hook(self, module: nn.Module, _, outputs: object) -> object:
        if self._faith_vec is None and self._hal_vec is None:
            return outputs

        x = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if x.dim() < 3:
            return outputs

        x_new = x
        if self.mode in ("faith", "both") and self._faith_vec is not None:
            x_new = x_new + self.steer_alpha * self._faith_vec.view(1, 1, -1)
        if self.mode in ("hal", "both") and self._hal_vec is not None:
            x_new = x_new - self.steer_alpha * self._hal_vec.view(1, 1, -1)

        if isinstance(outputs, tuple):
            return (x_new,) + outputs[1:]
        return x_new

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
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

        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
