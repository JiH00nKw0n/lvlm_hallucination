"""SharedEncSAE: ablation for VL-SAE.

Single linear top-K encoder (shared across modalities) + two modality-specific
decoders. Isolates the effect of sharing the encoder from VL-SAE's distance-
based activation — here the encoder is a plain linear projection + ReLU + top-K.

Layout intentionally mirrors `vl_sae.VLSAE`, so VL-SAE vs SharedEncSAE compares
only the activation function (distance vs linear top-K), and Shared vs
SharedEncSAE compares only decoder coupling (1 vs 2 decoders).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class SharedEncSAEConfig(PretrainedConfig):
    """Configuration for SharedEncSAE."""

    model_type = "shared_enc_sae"

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 8192,
        k: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.k = k


@dataclass
class SharedEncSAEOutput(ModelOutput):
    loss: Optional[Tensor] = None
    recon_loss: Optional[Tensor] = None
    recon_loss_image: Optional[Tensor] = None
    recon_loss_text: Optional[Tensor] = None
    image_output: Optional[Tensor] = None
    text_output: Optional[Tensor] = None
    latent_image: Optional[Tensor] = None
    latent_text: Optional[Tensor] = None


def _sparsify_topk(pre_acts: Tensor, k: int) -> Tensor:
    """ReLU + top-K sparsify (matches our TopK SAE convention)."""
    acts = F.relu(pre_acts)
    if k >= acts.shape[-1]:
        return acts
    _, top_idx = acts.topk(k, dim=-1, sorted=False)
    mask = torch.zeros_like(acts)
    mask.scatter_(-1, top_idx, 1.0)
    return acts * mask


class SharedEncSAE(PreTrainedModel):
    """Shared linear encoder + two decoders + top-K."""

    config_class = SharedEncSAEConfig
    base_model_prefix = "shared_enc_sae"

    def __init__(self, config: SharedEncSAEConfig):
        super().__init__(config)
        self.cfg = config
        self.hidden_size = config.hidden_size
        self.latent_size = config.latent_size
        self.k = config.k

        self.encoder = nn.Linear(self.hidden_size, self.latent_size)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)

        self.vision_decoder = nn.Linear(self.latent_size, self.hidden_size)
        self.text_decoder = nn.Linear(self.latent_size, self.hidden_size)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        return

    def encode(self, x: Tensor) -> Tensor:
        return _sparsify_topk(self.encoder(x), self.k)

    def forward(
        self,
        image_embeds: Optional[Tensor] = None,
        text_embeds: Optional[Tensor] = None,
        **_: dict,
    ) -> SharedEncSAEOutput:
        recon_v = recon_t = None
        latent_v = latent_t = None
        loss_v = loss_t = None

        if image_embeds is not None:
            latent_v = self.encode(image_embeds)
            recon_v = self.vision_decoder(latent_v)
            loss_v = F.mse_loss(recon_v, image_embeds)

        if text_embeds is not None:
            latent_t = self.encode(text_embeds)
            recon_t = self.text_decoder(latent_t)
            loss_t = F.mse_loss(recon_t, text_embeds)

        loss_parts = [l for l in (loss_v, loss_t) if l is not None]
        if not loss_parts:
            raise ValueError("SharedEncSAE.forward requires image_embeds or text_embeds.")
        loss = loss_parts[0] if len(loss_parts) == 1 else loss_parts[0] + loss_parts[1]

        return SharedEncSAEOutput(
            loss=loss,
            recon_loss=loss,
            recon_loss_image=loss_v,
            recon_loss_text=loss_t,
            image_output=recon_v,
            text_output=recon_t,
            latent_image=latent_v,
            latent_text=latent_t,
        )


__all__ = ["SharedEncSAE", "SharedEncSAEConfig", "SharedEncSAEOutput"]
