"""VL-SAE (Shen et al., NeurIPS 2025) baseline.

Shared encoder + two modality-specific decoders. The encoder activation is a
normalized-Euclidean-distance score (2 - cdist between L2-normalized input and
L2-normalized neuron weight), which makes a semantically-close vision/text pair
activate a similar set of neurons via the triangle inequality of the distance
metric. Loss is plain MSE reconstruction on each modality; no alignment loss.

Ported from: https://github.com/ssfgunner/VL-SAE/blob/main/cvlms/sae_trainer/sae_model.py
Paper: https://arxiv.org/abs/2510.21323
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


class VLSAEConfig(PretrainedConfig):
    """Configuration for VL-SAE (paper variant).

    Attributes:
        hidden_size: Input representation dimension (e.g. 512 for CLIP B/32).
        latent_size: Number of hidden neurons (shared across both modalities).
        k: Top-K sparsity parameter applied to the shared hidden activation.
    """

    model_type = "vl_sae"

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
class VLSAEOutput(ModelOutput):
    """Output container for VL-SAE forward pass."""

    loss: Optional[Tensor] = None
    recon_loss: Optional[Tensor] = None
    recon_loss_image: Optional[Tensor] = None
    recon_loss_text: Optional[Tensor] = None
    image_output: Optional[Tensor] = None
    text_output: Optional[Tensor] = None
    latent_image: Optional[Tensor] = None
    latent_text: Optional[Tensor] = None


def _sparsify_topk(activations: Tensor, k: int) -> Tensor:
    """Keep top-K entries by absolute value; zero the rest.

    Mirrors the paper's `kthvalue`-based sparsify but uses `topk` on the
    absolute values for clarity.
    """
    if k >= activations.shape[-1]:
        return activations
    abs_vals = activations.abs()
    _, top_idx = abs_vals.topk(k, dim=-1, sorted=False)
    mask = torch.zeros_like(activations)
    mask.scatter_(-1, top_idx, 1.0)
    return activations * mask


class VLSAE(PreTrainedModel):
    """VL-SAE: shared distance-based encoder, two modality-specific decoders.

    The encoder weight is a single `(latent_size, hidden_size)` parameter used
    for both modalities. The distance-based activation for neuron `i` on input
    `x` is `2 - ||x/||x|| - w_i/||w_i|||`, which equals `2 - sqrt(2 - 2 cos)`
    and is monotonic in the cosine similarity.
    """

    config_class = VLSAEConfig
    base_model_prefix = "vl_sae"

    def __init__(self, config: VLSAEConfig):
        super().__init__(config)
        self.cfg = config
        self.hidden_size = config.hidden_size
        self.latent_size = config.latent_size
        self.k = config.k

        self.encoder = nn.Parameter(torch.empty(self.latent_size, self.hidden_size))
        nn.init.kaiming_uniform_(self.encoder, a=math.sqrt(5))

        self.vision_decoder = nn.Linear(self.latent_size, self.hidden_size)
        self.text_decoder = nn.Linear(self.latent_size, self.hidden_size)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        return

    def encode(self, x: Tensor) -> Tensor:
        """Run the shared distance-based encoder and top-K sparsify.

        Args:
            x: Input embeddings of shape (batch, hidden_size).

        Returns:
            Sparse activations of shape (batch, latent_size).
        """
        weights = F.normalize(self.encoder, p=2, dim=1)
        inputs = F.normalize(x, p=2, dim=-1)
        dist = torch.cdist(inputs, weights, p=2)
        activations = 2.0 - dist
        return _sparsify_topk(activations, self.k)

    def forward(
        self,
        image_embeds: Optional[Tensor] = None,
        text_embeds: Optional[Tensor] = None,
        **_: dict,
    ) -> VLSAEOutput:
        """Encode and reconstruct image and/or text embeddings.

        When both modalities are provided, `loss` is the sum of MSE(image) and
        MSE(text) (matching the paper's training objective).
        """
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
            raise ValueError("VLSAE.forward requires image_embeds or text_embeds.")
        loss = loss_parts[0] if len(loss_parts) == 1 else loss_parts[0] + loss_parts[1]

        return VLSAEOutput(
            loss=loss,
            recon_loss=loss,
            recon_loss_image=loss_v,
            recon_loss_text=loss_t,
            image_output=recon_v,
            text_output=recon_t,
            latent_image=latent_v,
            latent_text=latent_t,
        )


__all__ = ["VLSAE", "VLSAEConfig", "VLSAEOutput"]
