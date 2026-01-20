"""
SSL: Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation.

Reference:
    - SSL/utils.py (hook-based steering)
    - SSL/sae/sae.py (SAE loading)

Supports: LLaVA-NeXT only (llava_next)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseMitigator
from .ssl_utils import Sae


class SSLMitigator(BaseMitigator):
    """
    SSL: SAE-based steering for hallucination mitigation.

    Args:
        model: The VLM model (LLaVA-NeXT only)
        model_type: llava_next
        sae_repo: HuggingFace repo id for SAE weights
        sae_hookpoint: Hookpoint subdir inside the SAE repo (e.g., "model.layers.24")
        sae_path: Optional local path to SAE layer directory (cfg.json + sae.safetensors)
        layer: Layer index to hook (default: 24)
        gamma: Steering strength (default: 0.2)
        hall_index: SAE latent index for hallucination direction (default: 36992)
        non_hall_index: SAE latent index for non-hallucination direction (default: 47230)
    """

    name: str = "SSLMitigator"

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava_next",
        sae_repo: Optional[str] = None,
        sae_hookpoint: Optional[str] = None,
        sae_path: Optional[str] = None,
        layer: int = 24,
        gamma: float = 0.2,
        hall_index: int = 36992,
        non_hall_index: int = 47230,
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.sae_repo = sae_repo
        self.sae_hookpoint = sae_hookpoint
        self.sae_path = sae_path
        self.layer = layer
        self.gamma = gamma
        self.hall_index = hall_index
        self.non_hall_index = non_hall_index

        self._handle = None
        self._image_start: int = 0
        self._num_img_tokens: int = 0
        self._d_hall: Optional[torch.Tensor] = None
        self._d_non_hall: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if self.model_type != "llava_next":
            raise ValueError("SSLMitigator currently supports llava_next only.")

        if self.sae_path:
            sae_dir = Path(self.sae_path)
            if not sae_dir.exists():
                raise FileNotFoundError(f"SAE path not found: {sae_dir}")
            sae = Sae.load_from_disk(str(sae_dir), device=self.model.device)
        else:
            if not self.sae_repo or not self.sae_hookpoint:
                raise ValueError("SSLMitigator requires sae_repo and sae_hookpoint when sae_path is not set.")
            sae = Sae.load_from_hub(
                name=self.sae_repo,
                hookpoint=self.sae_hookpoint,
                device=self.model.device,
            )
        self._d_non_hall = sae.W_dec.clone()[self.non_hall_index, :].detach().view(1, 1, -1)
        self._d_hall = sae.W_dec.clone()[self.hall_index, :].detach().view(1, 1, -1)

        layers = self._get_layers()
        if self.layer >= len(layers) or self.layer < 0:
            raise ValueError(f"Layer index {self.layer} out of range (0-{len(layers)-1}).")

        hooked_module = layers[self.layer]
        self._handle = hooked_module.register_forward_hook(self._hook)

    def cleanup(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._d_hall = None
        self._d_non_hall = None

    def _hook(self, module: nn.Module, _, outputs: object) -> object:
        d_hall = self._d_hall
        d_non_hall = self._d_non_hall
        if d_hall is None or d_non_hall is None:
            return outputs

        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = list(outputs)

        hidden = unpack_outputs[0]

        with torch.no_grad():
            if hidden.dim() < 3:
                return outputs
            if hidden.shape[1] != 1:
                x_img = hidden[:, self._image_start:self._image_start + self._num_img_tokens, :]
                x_norm = x_img.norm(dim=-1, keepdim=True)
                d_norm = d_non_hall.norm() + 1e-6
                alpha_img = self.gamma * x_norm / d_norm
                hidden[:, self._image_start:self._image_start + self._num_img_tokens, :] = (
                    x_img + alpha_img * d_non_hall.to(hidden.dtype)
                )
            else:
                x_gen = hidden
                x_norm = x_gen.norm()
                d_norm = d_hall.norm() + 1e-6
                alpha_gen = self.gamma * x_norm / d_norm
                hidden = x_gen - alpha_gen * d_hall.to(hidden.dtype)
                unpack_outputs[0] = hidden

        return tuple(unpack_outputs) if isinstance(outputs, tuple) else unpack_outputs[0]

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self._num_img_tokens == 0:
            config = getattr(self.model, "config", None)
            img_start, img_end = self._get_image_token_indices(input_ids, config)
            self._image_start = img_start
            self._num_img_tokens = img_end - img_start

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
