from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import einops
import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_sae import (
    BatchTopKSAEConfig,
    MatryoshkaSAEConfig,
    TopKSAEConfig,
    VLBatchTopKSAEConfig,
    VLMatryoshkaSAEConfig,
    VLTopKSAEConfig,
)


def _decoder_impl(top_indices: Tensor, top_activations: Tensor, decoder_weight: Tensor) -> Tensor:
    """
    Sparse decode helper used by the TopKSAE.

    The inputs are a compact Top-K representation (indices + activations). We
    first scatter them into a dense vector, then project with the decoder.

    Args:
        top_indices: Indices of the top-k latents. Shape: (batch, seq_len, k).
        top_activations: Values of the top-k latents. Shape: (batch, seq_len, k).
        decoder_weight: Decoder matrix with shape (latent_size, hidden_size).

    Returns:
        Dense reconstruction in the original activation space. Shape: (batch, seq_len, hidden_size).
    """
    # Dense latent buffer: (batch, seq_len, latent_size) where only top-k positions are filled.
    # Input shapes:
    #   - top_indices: (batch, seq_len, k)
    #   - top_activations: (batch, seq_len, k)
    #   - decoder_weight: (latent_size, hidden_size)
    dense_latents = top_activations.new_zeros(
        top_activations.shape[:-1] + (decoder_weight.shape[0],)
    )

    # Scatter top-k activations into the dense latent vector.
    # Output shape: (batch, seq_len, latent_size)
    dense_latents = dense_latents.scatter_(dim=-1, index=top_indices, src=top_activations)

    # Decode: dense_latents @ W_dec -> (batch, seq_len, hidden_size)
    return dense_latents @ decoder_weight


def _vl_split_indices(
    latent_size: int,
    ratio: tuple[int, int, int] = (1, 2, 1),
) -> tuple[int, int, int]:
    """
    Split total latents into (visual, shared, text) by the given ratio.

    Args:
        latent_size: Total number of latent dimensions.
        ratio: (visual, shared, text) proportions (default (1, 2, 1)).

    Returns:
        Tuple (v_size, s_size, t_size) where v+s+t == latent_size.
    """
    total = sum(ratio)
    v_size = latent_size * ratio[0] // total
    s_size = latent_size * ratio[1] // total
    t_size = latent_size - v_size - s_size
    return v_size, s_size, t_size


def _vl_masks(
    latent_size: int,
    device: torch.device,
    ratio: tuple[int, int, int] = (1, 2, 1),
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Build boolean masks for [visual | shared | text] blocks.

    Returns:
        (v_mask, s_mask, t_mask) each shaped (latent_size,)
    """
    v_size, s_size, t_size = _vl_split_indices(latent_size, ratio)
    v_mask = torch.zeros(latent_size, dtype=torch.bool, device=device)
    s_mask = torch.zeros(latent_size, dtype=torch.bool, device=device)
    t_mask = torch.zeros(latent_size, dtype=torch.bool, device=device)

    v_mask[:v_size] = True
    s_mask[v_size: v_size + s_size] = True
    t_mask[v_size + s_size:] = True
    return v_mask, s_mask, t_mask


def _masked_total_variance(x: Tensor, attention_mask: Optional[Tensor]) -> tuple[Tensor, Optional[Tensor]]:
    if attention_mask is None:
        total_variance = (x - x.mean(0)).pow(2).sum()
        eps = torch.finfo(x.dtype).eps
        return total_variance + eps, None

    mask = attention_mask.reshape(-1).bool()
    if int(mask.sum()) == 0:
        raise ValueError("attention_mask selects no valid tokens.")
    x_flat = x.reshape(-1, x.shape[-1])[mask]
    total_variance = (x_flat - x_flat.mean(0)).pow(2).sum()
    eps = torch.finfo(x.dtype).eps
    return total_variance + eps, mask


def _masked_l2_sum(residual: Tensor, flat_mask: Optional[Tensor]) -> Tensor:
    if flat_mask is None:
        return residual.pow(2).sum()
    residual_flat = residual.reshape(-1, residual.shape[-1])
    return residual_flat[flat_mask].pow(2).sum()


def _masked_mse_mean(diff: Tensor, flat_mask: Optional[Tensor]) -> Tensor:
    if flat_mask is None:
        return diff.pow(2).mean()
    diff_flat = diff.reshape(-1, diff.shape[-1])
    return diff_flat[flat_mask].pow(2).mean()


def _sanitize_topk_acts(acts: Tensor) -> Tensor:
    """Replace NaN/Inf activations with zeros to keep aux losses finite."""
    return torch.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class SAEOutput(ModelOutput):
    """
    Output container for SAE forward pass.

    Attributes:
        output: Reconstructed activations in the original space.
        latent_activations: Sparse latent activations (TopK or BatchTopK).
        latent_indices: Indices for the top-k latent activations.
        recon_loss: Normalized reconstruction loss (lower is better).
        auxk_loss: Auxiliary loss for dead features (0 if not applicable).
        dense_latents: Dense latent vector (batch, seq_len, latent_size) when requested.
    """

    output: Tensor
    latent_activations: Optional[Tensor] = None
    latent_indices: Optional[Tensor] = None
    recon_loss: Optional[Tensor] = None
    auxk_loss: Optional[Tensor] = None
    dense_latents: Optional[Tensor] = None


@dataclass
class VLSAEOutput(ModelOutput):
    """
    Output container for VL (vision-language) SAE forward pass.

    Attributes:
        output: Reconstructed activations in the original space.
        latent_activations: Sparse latent activations (TopK or BatchTopK).
        latent_indices: Indices for the top-k latent activations.
        recon_loss: Normalized reconstruction loss (lower is better).
        auxk_loss: Auxiliary loss for dead features (0 if not applicable).
        shared_recon_loss: Normalized reconstruction loss using only the shared subspace.
        dense_latents: Dense latent vector (batch, seq_len, latent_size) when requested.
    """

    output: Tensor
    latent_activations: Optional[Tensor] = None
    latent_indices: Optional[Tensor] = None
    recon_loss: Optional[Tensor] = None
    auxk_loss: Optional[Tensor] = None
    shared_recon_loss: Optional[Tensor] = None
    dense_latents: Optional[Tensor] = None


@dataclass
class MatryoshkaSAEOutput(ModelOutput):
    """
    Output container for MatryoshkaSAE forward pass.

    Attributes:
        output: Final reconstruction using all active groups.
        latent_activations: Sparse latent activations after BatchTopK selection.
        latent_indices: Indices for a per-sample TopK view of latent_activations.
        recon_loss: Normalized reconstruction loss using the final reconstruction.
        auxk_loss: Auxiliary loss for dead features (0 if not applicable).
        mean_l2_loss: Mean L2 reconstruction loss across all prefixes.
        min_l2_loss: Best (lowest) prefix L2 reconstruction loss.
        max_l2_loss: Worst (highest) prefix L2 reconstruction loss.
    """

    output: Tensor
    latent_activations: Optional[Tensor] = None
    latent_indices: Optional[Tensor] = None
    recon_loss: Optional[Tensor] = None
    auxk_loss: Optional[Tensor] = None
    mean_l2_loss: Optional[Tensor] = None
    min_l2_loss: Optional[Tensor] = None
    max_l2_loss: Optional[Tensor] = None


@dataclass
class VLMatryoshkaSAEOutput(ModelOutput):
    """
    Output container for VLMatryoshkaSAE forward pass.

    Attributes:
        output: Final reconstruction using all active groups.
        latent_activations: Sparse latent activations after BatchTopK selection.
        latent_indices: Indices for a per-sample TopK view of latent_activations.
        recon_loss: Normalized reconstruction loss using the final reconstruction.
        auxk_loss: Auxiliary loss for dead features (0 if not applicable).
        mean_l2_loss: Mean L2 reconstruction loss across all prefixes.
        min_l2_loss: Best (lowest) prefix L2 reconstruction loss.
        max_l2_loss: Worst (highest) prefix L2 reconstruction loss.
        shared_mean_l2_loss: Mean shared L2 reconstruction loss across all prefixes.
        shared_min_l2_loss: Best (lowest) shared prefix L2 reconstruction loss.
        shared_max_l2_loss: Worst (highest) shared prefix L2 reconstruction loss.
        shared_recon_loss: Normalized reconstruction loss using only the shared subspace.
    """

    output: Tensor
    latent_activations: Optional[Tensor] = None
    latent_indices: Optional[Tensor] = None
    recon_loss: Optional[Tensor] = None
    auxk_loss: Optional[Tensor] = None
    mean_l2_loss: Optional[Tensor] = None
    min_l2_loss: Optional[Tensor] = None
    max_l2_loss: Optional[Tensor] = None
    shared_mean_l2_loss: Optional[Tensor] = None
    shared_min_l2_loss: Optional[Tensor] = None
    shared_max_l2_loss: Optional[Tensor] = None
    shared_recon_loss: Optional[Tensor] = None


class TopKSAE(PreTrainedModel):
    """
    A TopK sparse autoencoder (SAE) for hidden activations.

    This is a faithful, readable implementation of the multimodal-sae
    architecture using a Hugging Face `PreTrainedModel` wrapper.
    """

    config_class = TopKSAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: TopKSAEConfig):
        super().__init__(config)
        self.cfg = config
        # hidden_size: width of input activations (feature dimension)
        self.hidden_size = config.hidden_size
        # latent_size: size of latent dictionary
        self.latent_size = config.latent_size or self.hidden_size * config.expansion_factor

        # Encoder maps input activations to a larger latent space.
        # encoder weight shape: (latent_size, hidden_size)
        self.encoder = nn.Linear(self.hidden_size, self.latent_size)
        self.encoder.bias.data.zero_()

        # Initialize decoder as a copy of encoder weights (as in multimodal-sae).
        # decoder weight shape: (latent_size, hidden_size)
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Decoder bias shifts reconstructions; it is subtracted before encoding.
        # decoder bias shape: (hidden_size,)
        self.b_dec = nn.Parameter(torch.zeros(self.hidden_size))

        # Avoid reinitializing since we already set weights explicitly.
        self.post_init()

    def _init_weights(self, module: nn.Module):
        return

    @property
    def device(self):
        """Convenience accessor for the module's device."""
        return self.encoder.weight.device

    @property
    def dtype(self):
        """Convenience accessor for the module's dtype."""
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        """
        Compute latent pre-activations (ReLU applied after encoder).

        Note: we subtract decoder bias from input, following the Anthropic SAE
        convention used in multimodal-sae.
        """
        # Subtract decoder bias before encoding.
        # Input x: (batch, seq_len, hidden_size) -> sae_in: (batch, seq_len, hidden_size)
        sae_in = x.to(self.dtype) - self.b_dec

        # Linear projection to latent space.
        # sae_in: (batch, seq_len, hidden_size) -> out: (batch, seq_len, latent_size)
        out = self.encoder(sae_in)

        # ReLU non-linearity keeps only positive activations.
        # out: (batch, seq_len, latent_size) -> pre_acts: (batch, seq_len, latent_size)
        return nn.functional.relu(out)

    def select_topk(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        """
        Select top-k activations and their indices from the latent vector.

        Input shape: (batch, seq_len, latent_size)
        Output shapes:
            - top_acts: (batch, seq_len, k)
            - top_indices: (batch, seq_len, k)
        """
        return latents.topk(self.cfg.k, sorted=False)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs and return top-k activations and indices."""
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """
        Decode a sparse (top-k) representation back to input space.

        Input shapes:
            - top_acts: (batch, seq_len, k)
            - top_indices: (batch, seq_len, k)
        Output shape:
            - recon: (batch, seq_len, hidden_size)
        """
        y = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec)
        return y + self.b_dec

    def forward(
        self,
        hidden_states: Tensor,
        dead_mask: Optional[Tensor] = None,
        return_dense_latents: bool = False,
    ) -> SAEOutput:
        """
        Run a forward pass and compute reconstruction + auxiliary metrics.

        Args:
            hidden_states: Input activations. Shape: (batch, seq_len, hidden_size).
            dead_mask: Optional boolean mask of dead features. Shape: (latent_size,).
            return_dense_latents: If True, scatter top_acts into a dense (batch, seq, latent_size)
                tensor and return it in the output.

        Returns:
            SAEOutput with reconstruction and training metrics.
        """
        x = hidden_states
        # 1) Encode to latent pre-activations.
        # x: (batch, seq_len, hidden_size) -> pre_acts: (batch, seq_len, latent_size)
        pre_acts = self.pre_acts(x)

        # 2) Main Top-K pass: select sparse latents.
        # pre_acts: (batch, seq_len, latent_size) -> top_acts/top_indices: (batch, seq_len, k)
        top_acts, top_indices = self.select_topk(pre_acts)

        # 3) Decode sparse latents back to input space.
        # top_acts/top_indices: (batch, seq_len, k) -> sae_out: (batch, seq_len, hidden_size)
        sae_out = self.decode(top_acts, top_indices)

        # 4) Reconstruction residual.
        # residual: (batch, seq_len, hidden_size)
        # residual: (batch, seq_len, hidden_size)
        residual = sae_out - x

        # 5) Total variance (scalar) used to normalize losses/metrics.
        total_variance = (x - x.mean(0)).pow(2).sum()

        # 6) Optional AuxK loss to revive dead features.
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 of the SAE paper.
            # k_aux: auxiliary top-k for dead latents (scalar)
            k_aux = x.shape[-1] // 2

            # Scale down the aux loss if only a few dead features exist.
            # scale: down-weight aux loss when few dead features exist
            scale = min(num_dead / k_aux, 1.0)
            # clamp k_aux to number of dead features
            k_aux = min(k_aux, num_dead)

            # Mask out live features so only dead ones participate.
            # pre_acts: (batch, seq_len, latent_size) -> auxk_latents: (batch, seq_len, latent_size)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Take top-k among dead latents.
            # auxk_acts/auxk_indices: (batch, seq_len, k_aux)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_acts = _sanitize_topk_acts(auxk_acts)

            # Try to predict the residual using dead features.
            # e_hat: (batch, seq_len, hidden_size)
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - residual).pow(2).sum()
            # normalize and scale aux loss (scalar)
            auxk_loss = scale * auxk_loss / total_variance
            auxk_loss = torch.nan_to_num(auxk_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        # 7) Normalized reconstruction loss (scalar).
        l2_loss = residual.pow(2).sum()
        recon_loss = l2_loss / total_variance

        # 8) Optionally build dense latent vector.
        dense_latents = None
        if return_dense_latents:
            dense_latents = top_acts.new_zeros(
                top_acts.shape[:-1] + (self.latent_size,)
            )
            dense_latents.scatter_(dim=-1, index=top_indices, src=top_acts)

        return SAEOutput(
            output=sae_out,
            latent_activations=top_acts,
            latent_indices=top_indices,
            recon_loss=recon_loss,
            auxk_loss=auxk_loss,
            dense_latents=dense_latents,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize each decoder row to unit norm for stable training."""
        # eps: scalar for numerical stability
        eps = torch.finfo(self.W_dec.dtype).eps
        # norm: (latent_size, 1)
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        # normalize each decoder row to unit length
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Project out gradient components parallel to decoder directions.

        This keeps decoder updates orthogonal to the current decoder weights.
        """
        assert self.W_dec.grad is not None
        # parallel_component: (latent_size,) projection of grad onto decoder rows
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "latent_size hidden_size, latent_size hidden_size -> latent_size",
        )
        # remove parallel component from gradient
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "latent_size, latent_size hidden_size -> latent_size hidden_size",
        )


class BatchTopKSAE(PreTrainedModel):
    """
    BatchTopK sparse autoencoder.

    BatchTopK selects the top (batch_size * seq_len * k) latent activations
    across the entire batch. This allows each sample to have a variable number
    of active latents while preserving average sparsity.
    """

    config_class = BatchTopKSAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: BatchTopKSAEConfig):
        super().__init__(config)
        self.cfg = config
        # hidden_size: width of input activations (feature dimension)
        self.hidden_size = config.hidden_size
        # latent_size: size of latent dictionary
        self.latent_size = config.latent_size or self.hidden_size * config.expansion_factor

        # Encoder maps input activations to a larger latent space.
        # encoder weight shape: (latent_size, hidden_size)
        self.encoder = nn.Linear(self.hidden_size, self.latent_size)
        self.encoder.bias.data.zero_()

        # Initialize decoder as a copy of encoder weights (as in multimodal-sae).
        # decoder weight shape: (latent_size, hidden_size)
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Decoder bias shifts reconstructions; it is subtracted before encoding.
        # decoder bias shape: (hidden_size,)
        self.b_dec = nn.Parameter(torch.zeros(self.hidden_size))

        self.post_init()

    def _init_weights(self, module: nn.Module):
        return

    @property
    def device(self):
        """Convenience accessor for the module's device."""
        return self.encoder.weight.device

    @property
    def dtype(self):
        """Convenience accessor for the module's dtype."""
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        """
        Compute latent pre-activations (ReLU applied after encoder).

        Note: we subtract decoder bias from input, following the Anthropic SAE
        convention used in multimodal-sae.
        """
        # Subtract decoder bias before encoding.
        # Input x: (batch, seq_len, hidden_size) -> sae_in: (batch, seq_len, hidden_size)
        sae_in = x.to(self.dtype) - self.b_dec

        # Linear projection to latent space.
        # sae_in: (batch, seq_len, hidden_size) -> out: (batch, seq_len, latent_size)
        out = self.encoder(sae_in)

        # ReLU non-linearity keeps only positive activations.
        # out: (batch, seq_len, latent_size) -> pre_acts: (batch, seq_len, latent_size)
        return nn.functional.relu(out)

    def _batch_topk_mask(self, pre_acts: Tensor) -> Tensor:
        """
        Compute a batch-level TopK mask.

        We flatten (batch, seq_len, latent_size) into (num_tokens, latent_size),
        then select the top (num_tokens * k) activations across the entire batch.
        This matches the reference BatchTopK implementation, which selects
        `top_k * batch` activations when inputs are shaped (batch, latent_size).

        Input shape:
            - pre_acts: (batch, seq_len, latent_size)
        Output shape:
            - mask: (batch, seq_len, latent_size) with True at selected positions.
        """
        # pre_acts: (batch, seq_len, latent_size_total)
        batch, seq_len, latent_size = pre_acts.shape
        # num_tokens: total tokens across batch
        num_tokens = batch * seq_len
        # total_k: number of activations kept across the whole batch
        total_k = min(num_tokens * self.cfg.k, num_tokens * latent_size)

        # Flatten all activations into a single vector.
        # flat_acts: (num_tokens * latent_size,)
        flat_acts = pre_acts.reshape(-1)

        # Select top total_k activations across the batch.
        # top_indices: (total_k,)
        _, top_indices = flat_acts.topk(total_k, sorted=False)

        # Create a boolean mask for selected activations.
        mask = torch.zeros_like(flat_acts, dtype=torch.bool)
        mask[top_indices] = True
        return mask.view(batch, seq_len, latent_size)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs and return top-k activations and indices."""
        return self.pre_acts(x).topk(self.cfg.k, sorted=False)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode a sparse (top-k) representation back to input space."""
        y = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec)
        return y + self.b_dec

    def forward(self, hidden_states: Tensor, dead_mask: Optional[Tensor] = None) -> SAEOutput:
        """
        Run a forward pass and compute reconstruction + auxiliary metrics.

        Args:
            hidden_states: Input activations. Shape: (batch, seq_len, hidden_size).
            dead_mask: Optional boolean mask of dead features. Shape: (latent_size,).

        Returns:
            SAEOutput with reconstruction and training metrics.
        """
        x = hidden_states
        # 1) Encode to latent pre-activations.
        # x: (batch, seq_len, hidden_size) -> pre_acts: (batch, seq_len, latent_size)
        pre_acts = self.pre_acts(x)

        # 2) BatchTopK activation across the entire batch.
        # pre_acts: (batch, seq_len, latent_size) -> sparse_acts: (batch, seq_len, latent_size)
        # mask: (batch, seq_len, latent_size)
        mask = self._batch_topk_mask(pre_acts)
        # sparse_acts: (batch, seq_len, latent_size)
        sparse_acts = torch.where(mask, pre_acts, torch.zeros_like(pre_acts))

        # 3) Decode sparse latents back to input space.
        # sparse_acts: (batch, seq_len, latent_size) -> sae_out: (batch, seq_len, hidden_size)
        sae_out = sparse_acts.to(self.dtype) @ self.W_dec + self.b_dec

        # 4) Reconstruction residual.
        # residual: (batch, seq_len, hidden_size)
        residual = sae_out - x

        # 5) Total variance (scalar) used to normalize losses/metrics.
        total_variance = (x - x.mean(0)).pow(2).sum()

        # 6) Optional AuxK loss to revive dead features.
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # k_aux: auxiliary top-k for dead latents (scalar)
            k_aux = x.shape[-1] // 2
            # scale: down-weight aux loss when few dead features exist
            scale = min(num_dead / k_aux, 1.0)
            # clamp k_aux to number of dead features
            k_aux = min(k_aux, num_dead)

            # auxk_latents: (batch, seq_len, latent_size)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            # auxk_acts/auxk_indices: (batch, seq_len, k_aux)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_acts = _sanitize_topk_acts(auxk_acts)

            # e_hat: (batch, seq_len, hidden_size)
            e_hat = _decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec) + self.b_dec
            auxk_loss = (e_hat - residual).pow(2).sum()
            # normalize and scale aux loss (scalar)
            auxk_loss = scale * auxk_loss / total_variance
            auxk_loss = torch.nan_to_num(auxk_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        # 7) Normalized reconstruction loss (scalar).
        l2_loss = residual.pow(2).sum()
        recon_loss = l2_loss / total_variance

        # 9) For logging/compatibility, expose a per-sample TopK view.
        # sparse_acts: (batch, seq_len, latent_size) -> top_acts/top_indices: (batch, seq_len, k)
        top_acts, top_indices = sparse_acts.topk(self.cfg.k, sorted=False)

        return SAEOutput(
            output=sae_out,
            latent_activations=top_acts,
            latent_indices=top_indices,
            recon_loss=recon_loss,
            auxk_loss=auxk_loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize each decoder row to unit norm for stable training."""
        # eps: scalar for numerical stability
        eps = torch.finfo(self.W_dec.dtype).eps
        # norm: (latent_size, 1)
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        # normalize each decoder row to unit length
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Project out gradient components parallel to decoder directions.

        This keeps decoder updates orthogonal to the current decoder weights.
        """
        assert self.W_dec.grad is not None
        # parallel_component: (latent_size,) projection of grad onto decoder rows
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "latent_size hidden_size, latent_size hidden_size -> latent_size",
        )
        # remove parallel component from gradient
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "latent_size, latent_size hidden_size -> latent_size hidden_size",
        )


class MatryoshkaSAE(PreTrainedModel):
    """
    Matryoshka sparse autoencoder.

    The dictionary is split into groups. We decode using the first group,
    then the first two, and so on, and average the reconstruction loss across
    all prefixes. This encourages early groups to capture higher-level features.
    """

    config_class = MatryoshkaSAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: MatryoshkaSAEConfig):
        super().__init__(config)
        self.cfg = config
        # hidden_size: width of input activations (feature dimension)
        self.hidden_size = config.hidden_size

        # total_latents: full dictionary size
        if config.latent_size and config.latent_size > 0:
            total_latents = config.latent_size
        else:
            total_latents = int(sum(config.group_sizes))
        self.latent_size_total = total_latents
        self.latent_size = total_latents

        # group_indices: prefix boundaries for each Matryoshka group
        self.group_sizes = list(config.group_sizes)
        self.group_indices = [0]
        for size in self.group_sizes:
            self.group_indices.append(self.group_indices[-1] + int(size))

        # active_groups: number of prefix groups used for reconstruction
        self.active_groups = (
            int(config.active_groups)
            if config.active_groups is not None
            else len(self.group_sizes)
        )

        # Encoder maps input activations to the full latent dictionary.
        # encoder weight shape: (latent_size_total, hidden_size)
        self.encoder = nn.Linear(self.hidden_size, total_latents)
        self.encoder.bias.data.zero_()

        # Decoder weight initialized from encoder (tied init).
        # decoder weight shape: (latent_size_total, hidden_size)
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Decoder bias shifts reconstructions; it is subtracted before encoding.
        # decoder bias shape: (hidden_size,)
        self.b_dec = nn.Parameter(torch.zeros(self.hidden_size))

        self.post_init()

    def _init_weights(self, module: nn.Module):
        return

    @property
    def device(self):
        """Convenience accessor for the module's device."""
        return self.encoder.weight.device

    @property
    def dtype(self):
        """Convenience accessor for the module's dtype."""
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        """
        Compute latent pre-activations (ReLU applied after encoder).

        Input x: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, latent_size_total)
        """
        # Subtract decoder bias before encoding.
        # x: (batch, seq_len, hidden_size) -> sae_in: (batch, seq_len, hidden_size)
        sae_in = x.to(self.dtype) - self.b_dec

        # Linear projection to full latent dictionary.
        # sae_in: (batch, seq_len, hidden_size) -> out: (batch, seq_len, latent_size_total)
        out = self.encoder(sae_in)

        # ReLU keeps positive activations only.
        # out: (batch, seq_len, latent_size_total) -> pre_acts: (batch, seq_len, latent_size_total)
        return nn.functional.relu(out)

    def _batch_topk_mask(self, pre_acts: Tensor) -> Tensor:
        """
        Batch-level TopK mask across all latents.

        Input:
            - pre_acts: (batch, seq_len, latent_size_total)
        Output:
            - mask: (batch, seq_len, latent_size_total)
        """
        batch, seq_len, latent_size = pre_acts.shape
        num_tokens = batch * seq_len
        total_k = min(num_tokens * self.cfg.k, num_tokens * latent_size)

        # Flatten and select global top-k activations.
        # flat_acts: (batch * seq_len * latent_size_total)
        flat_acts = pre_acts.reshape(-1)
        # top_indices: (total_k,)
        _, top_indices = flat_acts.topk(total_k, sorted=False)

        # mask: (batch * seq_len * latent_size_total) -> reshape to (batch, seq_len, latent_size_total)
        mask = torch.zeros_like(flat_acts, dtype=torch.bool)
        mask[top_indices] = True
        return mask.view(batch, seq_len, latent_size)

    def _compute_activations(self, pre_acts: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute raw activations and the sparse BatchTopK activations.

        Input:
            - pre_acts: (batch, seq_len, latent_size_total)
        Output:
            - acts: (batch, seq_len, latent_size_total)
            - acts_topk: (batch, seq_len, latent_size_total)
        """
        # acts: (batch, seq_len, latent_size_total)
        acts = pre_acts
        # mask: (batch, seq_len, latent_size_total)
        mask = self._batch_topk_mask(acts)
        # acts_topk: (batch, seq_len, latent_size_total)
        acts_topk = torch.where(mask, acts, torch.zeros_like(acts))
        return acts, acts_topk

    def encode_dense(self, x: Tensor) -> Tensor:
        """
        Encode inputs into dense sparse Matryoshka activations.

        Input:
            - x: (batch, seq_len, hidden_size)
        Output:
            - acts_topk: (batch, seq_len, latent_size_total) with later groups zeroed.
        """
        pre_acts = self.pre_acts(x)
        _, acts_topk = self._compute_activations(pre_acts)

        # Zero out latents beyond the active prefix of groups.
        max_index = self.group_indices[self.active_groups]
        acts_topk[:, :, max_index:] = 0
        return acts_topk

    def decode_dense(self, acts_topk: Tensor) -> Tensor:
        """
        Decode dense sparse activations into the input space.

        Input:
            - acts_topk: (batch, seq_len, latent_size_total)
        Output:
            - recon: (batch, seq_len, hidden_size)
        """
        return acts_topk.to(self.dtype) @ self.W_dec + self.b_dec

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs and return top-k activations and indices."""
        return self.pre_acts(x).topk(self.cfg.k, sorted=False)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode a sparse (top-k) representation back to input space."""
        y = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec)
        return y + self.b_dec

    def forward(self, hidden_states: Tensor, dead_mask: Optional[Tensor] = None) -> MatryoshkaSAEOutput:
        """
        Run a forward pass and compute Matryoshka reconstruction losses.

        Args:
            hidden_states: Input activations. Shape: (batch, seq_len, hidden_size).
            dead_mask: Optional boolean mask of dead features. Shape: (latent_size_total,).

        Returns:
            MatryoshkaSAEOutput with reconstruction and training metrics.
        """
        x = hidden_states
        # 1) Encode to latent pre-activations.
        # x: (batch, seq_len, hidden_size) -> pre_acts: (batch, seq_len, latent_size_total)
        pre_acts = self.pre_acts(x)

        # 2) BatchTopK to obtain sparse activations.
        # acts_topk: (batch, seq_len, latent_size_total)
        _, acts_topk = self._compute_activations(pre_acts)

        # 3) Reconstruct progressively using group prefixes.
        # Each prefix reconstruction is stored for Matryoshka loss.
        # Initialize with bias-only reconstruction.
        # b_dec: (hidden_size,) broadcasts to (batch, seq_len, hidden_size)
        reconstruct = self.b_dec
        intermediate_reconstructs: list[Tensor] = []

        for i in range(self.active_groups):
            start = self.group_indices[i]
            end = self.group_indices[i + 1]
            W_dec_slice = self.W_dec[start:end, :]
            acts_slice = acts_topk[:, :, start:end]

            # acts_slice: (batch, seq_len, group_size) @ W_dec_slice^T -> (batch, seq_len, hidden_size)
            reconstruct = acts_slice.to(self.dtype) @ W_dec_slice + reconstruct
            intermediate_reconstructs.append(reconstruct)

        # Final reconstruction uses all active groups.
        sae_out = reconstruct

        # 4) Compute Matryoshka losses (mean/min/max across prefixes).
        # Each prefix reconstruction: (batch, seq_len, hidden_size) -> normalized L2 sum.
        total_variance = (x - x.mean(0)).pow(2).sum()
        l2_losses = torch.stack(
            [(r.float() - x.float()).pow(2).sum() / total_variance for r in intermediate_reconstructs],
            dim=0,
        )
        min_l2_loss = l2_losses.min()
        max_l2_loss = l2_losses.max()
        # Baseline (bias-only) reconstruction: (hidden_size,) broadcasts to (batch, seq_len, hidden_size) -> scalar.
        base_l2_loss = (self.b_dec - x.float()).pow(2).sum() / total_variance
        mean_l2_loss = (l2_losses.sum() + base_l2_loss) / (len(intermediate_reconstructs) + 1)

        # 5) Normalized reconstruction loss based on the final reconstruction.
        # total_variance: scalar normalization term
        residual = sae_out - x
        # l2_loss: scalar
        l2_loss = residual.pow(2).sum()
        recon_loss = l2_loss / total_variance

        # 6) Optional AuxK loss for dead features.
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # k_aux: auxiliary top-k for dead latents (scalar)
            k_aux = x.shape[-1] // 2
            # scale: down-weight aux loss when few dead features exist
            scale = min(num_dead / k_aux, 1.0)
            # clamp k_aux to number of dead features
            k_aux = min(k_aux, num_dead)

            # aux_latents: (batch, seq_len, latent_size_total)
            aux_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            aux_acts, aux_indices = aux_latents.topk(k_aux, sorted=False)
            aux_acts = _sanitize_topk_acts(aux_acts)
            # aux_acts/aux_indices: (batch, seq_len, k_aux) -> e_hat: (batch, seq_len, hidden_size)
            e_hat = _decoder_impl(aux_indices, aux_acts.to(self.dtype), self.W_dec) + self.b_dec
            auxk_loss = (e_hat - residual).pow(2).sum()
            # normalize and scale aux loss (scalar)
            auxk_loss = scale * auxk_loss / total_variance
            auxk_loss = torch.nan_to_num(auxk_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        # 7) For logging/compatibility, expose a per-sample TopK view.
        # acts_topk: (batch, seq_len, latent_size_total) -> top_acts/top_indices: (batch, seq_len, k)
        top_acts, top_indices = acts_topk.topk(self.cfg.k, sorted=False)

        return MatryoshkaSAEOutput(
            output=sae_out,
            latent_activations=top_acts,
            latent_indices=top_indices,
            recon_loss=recon_loss,
            auxk_loss=auxk_loss,
            mean_l2_loss=mean_l2_loss,
            min_l2_loss=min_l2_loss,
            max_l2_loss=max_l2_loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize each decoder row to unit norm for stable training."""
        # eps: scalar for numerical stability
        eps = torch.finfo(self.W_dec.dtype).eps
        # norm: (latent_size_total, 1)
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        # normalize each decoder row to unit length
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Project out gradient components parallel to decoder directions.

        This keeps decoder updates orthogonal to the current decoder weights.
        """
        assert self.W_dec.grad is not None
        # parallel_component: (latent_size_total,) projection of grad onto decoder rows
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "latent_size hidden_size, latent_size hidden_size -> latent_size",
        )
        # remove parallel component from gradient
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "latent_size, latent_size hidden_size -> latent_size hidden_size",
        )


class VLTopKSAE(PreTrainedModel):
    """
    Vision-language TopK SAE with [visual | shared | text] subspaces.

    We split the latent dictionary into [visual | shared | text] blocks
    (ratio controlled by config.vl_split_ratio) and apply TopK within the
    active modality subspace only.
    """

    config_class = VLTopKSAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: VLTopKSAEConfig):
        super().__init__(config)
        self.cfg = config
        self.vl_ratio = tuple(config.vl_split_ratio)
        # hidden_size: width of input activations (feature dimension)
        self.hidden_size = config.hidden_size
        # latent_size: size of latent dictionary
        self.latent_size = config.latent_size or self.hidden_size * config.expansion_factor

        # Encoder maps input activations to a larger latent space.
        # encoder weight shape: (latent_size, hidden_size)
        self.encoder = nn.Linear(self.hidden_size, self.latent_size)
        self.encoder.bias.data.zero_()

        # Decoder weight initialized from encoder (tied init).
        # decoder weight shape: (latent_size, hidden_size)
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Decoder bias shifts reconstructions; it is subtracted before encoding.
        # decoder bias shape: (hidden_size,)
        self.b_dec = nn.Parameter(torch.zeros(self.hidden_size))

        self.post_init()

    def _init_weights(self, module: nn.Module):
        return

    @property
    def device(self):
        """Convenience accessor for the module's device."""
        return self.encoder.weight.device

    @property
    def dtype(self):
        """Convenience accessor for the module's dtype."""
        return self.encoder.weight.dtype

    def _modality_mask(self, visual_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Build modality and shared masks from a visual token mask.

        Returns:
            - active_mask: (batch, seq_len, latent_size) boolean mask for active subspaces.
            - shared_mask: (latent_size,) boolean mask for shared subspace only.
        """
        if visual_mask.dim() != 2:
            raise ValueError("visual_mask must be (batch, seq_len).")

        v_mask, s_mask, t_mask = _vl_masks(self.latent_size, self.device, self.vl_ratio)
        active_visual = (v_mask | s_mask).view(1, 1, -1)
        active_text = (s_mask | t_mask).view(1, 1, -1)
        visual_mask_broadcast = visual_mask.bool().unsqueeze(-1)
        active_mask = torch.where(visual_mask_broadcast, active_visual, active_text)
        return active_mask, s_mask

    def pre_acts(self, x: Tensor) -> Tensor:
        """
        Compute latent pre-activations (ReLU applied after encoder).

        Input x: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, latent_size)
        """
        # Subtract decoder bias before encoding.
        # x: (batch, seq_len, hidden_size) -> sae_in: (batch, seq_len, hidden_size)
        sae_in = x.to(self.dtype) - self.b_dec

        # Linear projection to latent space.
        # sae_in: (batch, seq_len, hidden_size) -> out: (batch, seq_len, latent_size)
        out = self.encoder(sae_in)

        # ReLU keeps positive activations only.
        # out: (batch, seq_len, latent_size) -> pre_acts: (batch, seq_len, latent_size)
        return nn.functional.relu(out)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs and return top-k activations and indices."""
        return self.pre_acts(x).topk(self.cfg.k, sorted=False)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode a sparse (top-k) representation back to input space."""
        y = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec)
        return y + self.b_dec

    def forward(
        self,
        hidden_states: Tensor,
        *,
        visual_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        dead_mask: Optional[Tensor] = None,
        return_dense_latents: bool = False,
    ) -> VLSAEOutput:
        """
        Run a forward pass with modality masking and shared reconstruction loss.

        Args:
            hidden_states: Input activations. Shape: (batch, seq_len, hidden_size).
            visual_mask: Boolean mask for visual tokens. Shape: (batch, seq_len).
            attention_mask: Boolean mask for valid tokens. Shape: (batch, seq_len).
            dead_mask: Optional boolean mask of dead features. Shape: (latent_size,).
            return_dense_latents: If True, scatter top_acts into a dense (batch, seq, latent_size)
                tensor and return it in the output.

        Returns:
            VLSAEOutput with reconstruction and training metrics.
        """
        x = hidden_states
        if logger.isEnabledFor(logging.DEBUG):
            total_tokens = visual_mask.numel()
            visual_tokens = int(visual_mask.sum().item())
            attn_tokens = int(attention_mask.sum().item()) if attention_mask is not None else None
            v_size, s_size, t_size = _vl_split_indices(self.latent_size, self.vl_ratio)
            logger.debug(
                "VLSAE stats: tokens=%d vis=%d attn=%s v/s/t=%d/%d/%d",
                total_tokens,
                visual_tokens,
                attn_tokens,
                v_size,
                s_size,
                t_size,
            )
        if logger.isEnabledFor(logging.DEBUG):
            total_tokens = visual_mask.numel()
            visual_tokens = int(visual_mask.sum().item())
            attn_tokens = int(attention_mask.sum().item()) if attention_mask is not None else None
            v_size, s_size, t_size = _vl_split_indices(self.latent_size, self.vl_ratio)
            logger.debug(
                "VLBatch stats: tokens=%d vis=%d attn=%s v/s/t=%d/%d/%d",
                total_tokens,
                visual_tokens,
                attn_tokens,
                v_size,
                s_size,
                t_size,
            )
        # 1) Encode to latent pre-activations.
        # x: (batch, seq_len, hidden_size) -> pre_acts: (batch, seq_len, latent_size)
        pre_acts = self.pre_acts(x)

        # 2) Apply modality mask before TopK.
        # active_mask: (batch, seq_len, latent_size)
        active_mask, shared_mask = self._modality_mask(visual_mask)
        if attention_mask is not None:
            if attention_mask.shape != visual_mask.shape:
                raise ValueError("attention_mask must match visual_mask shape.")
            active_mask = active_mask & attention_mask.bool().unsqueeze(-1)
        masked_pre_acts = torch.where(active_mask, pre_acts, torch.full_like(pre_acts, -torch.inf))
        if logger.isEnabledFor(logging.DEBUG):
            num_blocked = int((~active_mask).sum().item())
            logger.debug("VLBatch mask: blocked=%d", num_blocked)
        if logger.isEnabledFor(logging.DEBUG):
            num_blocked = int((~active_mask).sum().item())
            logger.debug("VLSAE mask: blocked=%d", num_blocked)

        # 3) TopK within active subspace.
        # masked_pre_acts: (batch, seq_len, latent_size) -> top_acts/top_indices: (batch, seq_len, k)
        top_acts, top_indices = masked_pre_acts.topk(self.cfg.k, sorted=False)
        valid = active_mask.gather(dim=-1, index=top_indices)
        top_acts = torch.where(valid, top_acts, torch.zeros_like(top_acts))
        if attention_mask is not None:
            top_acts = torch.where(attention_mask.bool().unsqueeze(-1), top_acts, torch.zeros_like(top_acts))

        # 4) Decode sparse latents back to input space.
        # top_acts/top_indices: (batch, seq_len, k) -> sae_out: (batch, seq_len, hidden_size)
        sae_out = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec) + self.b_dec

        # 5) Reconstruction residual.
        # residual: (batch, seq_len, hidden_size)
        residual = sae_out - x

        # 6) Total variance (scalar) used to normalize losses/metrics.
        total_variance, flat_mask = _masked_total_variance(x, attention_mask)

        # 7) Optional AuxK loss to revive dead features.
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # k_aux: auxiliary top-k for dead latents (scalar)
            k_aux = x.shape[-1] // 2
            # scale: down-weight aux loss when few dead features exist
            scale = min(num_dead / k_aux, 1.0)
            # clamp k_aux to number of dead features
            k_aux = min(k_aux, num_dead)

            # auxk_latents: (batch, seq_len, latent_size)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            if attention_mask is not None:
                auxk_latents = torch.where(
                    attention_mask.bool().unsqueeze(-1), auxk_latents, torch.full_like(auxk_latents, -torch.inf)
                )
            # auxk_acts/auxk_indices: (batch, seq_len, k_aux)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_acts = _sanitize_topk_acts(auxk_acts)

            # e_hat: (batch, seq_len, hidden_size)
            e_hat = _decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec) + self.b_dec
            auxk_loss = (e_hat - residual).pow(2).sum()
            # normalize and scale aux loss (scalar)
            auxk_loss = scale * auxk_loss / total_variance
            auxk_loss = torch.nan_to_num(auxk_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        # 8) Normalized reconstruction loss (scalar).
        l2_loss = _masked_l2_sum(residual, flat_mask)
        recon_loss = l2_loss / total_variance

        # 10) Shared-subspace-only reconstruction loss (shared-only TopK).  # shared-only loss block
        shared_mask_broadcast = shared_mask.view(1, 1, -1).expand_as(pre_acts)  # (batch, seq_len, latent_size)
        shared_pre_acts = torch.where(  # (batch, seq_len, latent_size) keep shared, else -inf
            shared_mask_broadcast, pre_acts, torch.full_like(pre_acts, -torch.inf)  # mask pre_acts
        )
        if attention_mask is not None:  # optional token mask
            shared_pre_acts = torch.where(  # (batch, seq_len, latent_size) drop masked tokens
                attention_mask.bool().unsqueeze(-1),  # (batch, seq_len, 1) token mask
                shared_pre_acts,  # (batch, seq_len, latent_size) candidate acts
                torch.full_like(shared_pre_acts, -torch.inf),  # (batch, seq_len, latent_size) masked
            )
        shared_top_acts, shared_top_indices = shared_pre_acts.topk(self.cfg.k, sorted=False)  # (batch, seq_len, k)
        shared_valid = shared_mask_broadcast.gather(dim=-1, index=shared_top_indices)  # (batch, seq_len, k) in-shared
        shared_top_acts = torch.where(shared_valid, shared_top_acts, torch.zeros_like(shared_top_acts))  # zero invalid
        if attention_mask is not None:  # optional token mask
            shared_top_acts = torch.where(  # (batch, seq_len, k) drop masked tokens
                attention_mask.bool().unsqueeze(-1), shared_top_acts, torch.zeros_like(shared_top_acts)  # mask k-acts
            )
        shared_recon = (  # (batch, seq_len, hidden_size) shared-only reconstruction
            _decoder_impl(shared_top_indices, shared_top_acts.to(self.dtype), self.W_dec)  # decode top-k
            + self.b_dec  # add decoder bias
        )
        # shared_recon_loss: scalar
        shared_recon_loss = _masked_l2_sum(shared_recon - x, flat_mask) / total_variance

        # Optionally build dense latent vector.
        dense_latents = None
        if return_dense_latents:
            dense_latents = top_acts.new_zeros(
                top_acts.shape[:-1] + (self.latent_size,)
            )
            dense_latents.scatter_(dim=-1, index=top_indices, src=top_acts)

        return VLSAEOutput(
            output=sae_out,
            latent_activations=top_acts,
            latent_indices=top_indices,
            recon_loss=recon_loss,
            auxk_loss=auxk_loss,
            shared_recon_loss=shared_recon_loss,
            dense_latents=dense_latents,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize each decoder row to unit norm for stable training."""
        # eps: scalar for numerical stability
        eps = torch.finfo(self.W_dec.dtype).eps
        # norm: (latent_size, 1)
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        # normalize each decoder row to unit length
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Project out gradient components parallel to decoder directions.

        This keeps decoder updates orthogonal to the current decoder weights.
        """
        assert self.W_dec.grad is not None
        # parallel_component: (latent_size,) projection of grad onto decoder rows
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "latent_size hidden_size, latent_size hidden_size -> latent_size",
        )
        # remove parallel component from gradient
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "latent_size, latent_size hidden_size -> latent_size hidden_size",
        )


class VLBatchTopKSAE(PreTrainedModel):
    """
    Vision-language BatchTopK SAE with [visual | shared | text] subspaces.

    We split the latent dictionary into [visual | shared | text] blocks
    (ratio controlled by config.vl_split_ratio) and apply BatchTopK within
    the active modality subspace only.
    """

    config_class = VLBatchTopKSAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: VLBatchTopKSAEConfig):
        super().__init__(config)
        self.cfg = config
        self.vl_ratio = tuple(config.vl_split_ratio)
        # hidden_size: width of input activations (feature dimension)
        self.hidden_size = config.hidden_size
        # latent_size: size of latent dictionary
        self.latent_size = config.latent_size or self.hidden_size * config.expansion_factor

        # Encoder maps input activations to a larger latent space.
        # encoder weight shape: (latent_size, hidden_size)
        self.encoder = nn.Linear(self.hidden_size, self.latent_size)
        self.encoder.bias.data.zero_()

        # Decoder weight initialized from encoder (tied init).
        # decoder weight shape: (latent_size, hidden_size)
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Decoder bias shifts reconstructions; it is subtracted before encoding.
        # decoder bias shape: (hidden_size,)
        self.b_dec = nn.Parameter(torch.zeros(self.hidden_size))

        self.post_init()

    def _init_weights(self, module: nn.Module):
        return

    @property
    def device(self):
        """Convenience accessor for the module's device."""
        return self.encoder.weight.device

    @property
    def dtype(self):
        """Convenience accessor for the module's dtype."""
        return self.encoder.weight.dtype

    def _modality_mask(self, visual_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Build modality and shared masks from a visual token mask.

        Returns:
            - active_mask: (batch, seq_len, latent_size) boolean mask for active subspaces.
            - shared_mask: (latent_size,) boolean mask for shared subspace only.
        """
        if visual_mask.dim() != 2:
            raise ValueError("visual_mask must be (batch, seq_len).")

        v_mask, s_mask, t_mask = _vl_masks(self.latent_size, self.device, self.vl_ratio)
        active_visual = (v_mask | s_mask).view(1, 1, -1)
        active_text = (s_mask | t_mask).view(1, 1, -1)
        visual_mask_broadcast = visual_mask.bool().unsqueeze(-1)
        active_mask = torch.where(visual_mask_broadcast, active_visual, active_text)
        return active_mask, s_mask

    def pre_acts(self, x: Tensor) -> Tensor:
        """
        Compute latent pre-activations (ReLU applied after encoder).

        Input x: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, latent_size)
        """
        # Subtract decoder bias before encoding.
        # x: (batch, seq_len, hidden_size) -> sae_in: (batch, seq_len, hidden_size)
        sae_in = x.to(self.dtype) - self.b_dec

        # Linear projection to latent space.
        # sae_in: (batch, seq_len, hidden_size) -> out: (batch, seq_len, latent_size)
        out = self.encoder(sae_in)

        # ReLU keeps positive activations only.
        # out: (batch, seq_len, latent_size) -> pre_acts: (batch, seq_len, latent_size)
        return nn.functional.relu(out)

    def _batch_topk_mask(self, pre_acts: Tensor) -> Tensor:
        """
        Batch-level TopK mask across all latents (already modality-masked).

        Input:
            - pre_acts: (batch, seq_len, latent_size)
        Output:
            - mask: (batch, seq_len, latent_size)
        """
        batch, seq_len, latent_size = pre_acts.shape
        num_tokens = batch * seq_len
        total_k = min(num_tokens * self.cfg.k, num_tokens * latent_size)

        # flat_acts: (batch * seq_len * latent_size)
        flat_acts = pre_acts.reshape(-1)
        # top_indices: (total_k,)
        _, top_indices = flat_acts.topk(total_k, sorted=False)

        # mask: (batch * seq_len * latent_size) -> reshape to (batch, seq_len, latent_size)
        mask = torch.zeros_like(flat_acts, dtype=torch.bool)
        mask[top_indices] = True
        return mask.view(batch, seq_len, latent_size)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs and return top-k activations and indices."""
        return self.pre_acts(x).topk(self.cfg.k, sorted=False)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode a sparse (top-k) representation back to input space."""
        y = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec)
        return y + self.b_dec

    def forward(
            self,
            hidden_states: Tensor,
            *,
            visual_mask: Tensor,
            attention_mask: Optional[Tensor] = None,
            dead_mask: Optional[Tensor] = None,
    ) -> VLSAEOutput:
        """
        Run a forward pass with modality masking and shared reconstruction loss.

        Args:
            hidden_states: Input activations. Shape: (batch, seq_len, hidden_size).
            visual_mask: Boolean mask for visual tokens. Shape: (batch, seq_len).
            attention_mask: Boolean mask for valid tokens. Shape: (batch, seq_len).
            dead_mask: Optional boolean mask of dead features. Shape: (latent_size,).

        Returns:
            VLSAEOutput with reconstruction and training metrics.
        """
        x = hidden_states
        # 1) Encode to latent pre-activations.
        # x: (batch, seq_len, hidden_size) -> pre_acts: (batch, seq_len, latent_size)
        pre_acts = self.pre_acts(x)

        # 2) Apply modality mask before BatchTopK.
        # active_mask: (batch, seq_len, latent_size)
        active_mask, shared_mask = self._modality_mask(visual_mask)
        if attention_mask is not None:
            if attention_mask.shape != visual_mask.shape:
                raise ValueError("attention_mask must match visual_mask shape.")
            active_mask = active_mask & attention_mask.bool().unsqueeze(-1)
        masked_pre_acts = torch.where(active_mask, pre_acts, torch.full_like(pre_acts, -torch.inf))

        # 3) BatchTopK within active subspace.
        # masked_pre_acts: (batch, seq_len, latent_size) -> sparse_acts: (batch, seq_len, latent_size)
        batch_mask = self._batch_topk_mask(masked_pre_acts)
        batch_mask = batch_mask & active_mask
        if attention_mask is not None:
            batch_mask = batch_mask & attention_mask.bool().unsqueeze(-1)
        sparse_acts = torch.where(batch_mask, pre_acts, torch.zeros_like(pre_acts))

        # 4) Decode sparse latents back to input space.
        # sparse_acts: (batch, seq_len, latent_size) -> sae_out: (batch, seq_len, hidden_size)
        sae_out = sparse_acts.to(self.dtype) @ self.W_dec + self.b_dec

        # 5) Reconstruction residual.
        # residual: (batch, seq_len, hidden_size)
        residual = sae_out - x

        # 6) Total variance (scalar) used to normalize losses/metrics.
        total_variance, flat_mask = _masked_total_variance(x, attention_mask)

        # 7) Optional AuxK loss to revive dead features.
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # k_aux: auxiliary top-k for dead latents (scalar)
            k_aux = x.shape[-1] // 2
            # scale: down-weight aux loss when few dead features exist
            scale = min(num_dead / k_aux, 1.0)
            # clamp k_aux to number of dead features
            k_aux = min(k_aux, num_dead)

            # auxk_latents: (batch, seq_len, latent_size)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            if attention_mask is not None:
                auxk_latents = torch.where(
                    attention_mask.bool().unsqueeze(-1), auxk_latents, torch.full_like(auxk_latents, -torch.inf)
                )
            # auxk_acts/auxk_indices: (batch, seq_len, k_aux)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_acts = _sanitize_topk_acts(auxk_acts)

            # e_hat: (batch, seq_len, hidden_size)
            e_hat = _decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec) + self.b_dec
            auxk_loss = (e_hat - residual).pow(2).sum()
            # normalize and scale aux loss (scalar)
            auxk_loss = scale * auxk_loss / total_variance
            auxk_loss = torch.nan_to_num(auxk_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        # 8) Normalized reconstruction loss (scalar).
        l2_loss = _masked_l2_sum(residual, flat_mask)
        recon_loss = l2_loss / total_variance

        # 10) Shared-subspace-only reconstruction loss (shared-only BatchTopK).  # shared-only loss block
        shared_mask_broadcast = shared_mask.view(1, 1, -1)  # (1, 1, latent_size) broadcast mask
        shared_pre_acts = torch.where(  # (batch, seq_len, latent_size) keep shared, else -inf
            shared_mask_broadcast, pre_acts, torch.full_like(pre_acts, -torch.inf)  # mask pre_acts
        )
        if attention_mask is not None:  # optional token mask
            shared_pre_acts = torch.where(  # (batch, seq_len, latent_size) drop masked tokens
                attention_mask.bool().unsqueeze(-1),  # (batch, seq_len, 1) token mask
                shared_pre_acts,  # (batch, seq_len, latent_size) candidate acts
                torch.full_like(shared_pre_acts, -torch.inf),  # (batch, seq_len, latent_size) masked
            )
        shared_batch_mask = self._batch_topk_mask(shared_pre_acts)  # (batch, seq_len, latent_size) global top-k mask
        shared_batch_mask = shared_batch_mask & shared_mask_broadcast  # (batch, seq_len, latent_size) keep shared only
        if attention_mask is not None:  # optional token mask
            shared_batch_mask = shared_batch_mask & attention_mask.bool().unsqueeze(-1)  # (batch, seq_len, latent)
        shared_acts = torch.where(shared_batch_mask, pre_acts, torch.zeros_like(pre_acts))  # (batch, seq_len, latent)
        shared_recon = shared_acts.to(self.dtype) @ self.W_dec + self.b_dec  # (batch, seq_len, hidden_size)
        # shared_recon_loss: scalar
        shared_recon_loss = _masked_l2_sum(shared_recon - x, flat_mask) / total_variance

        # 11) For logging/compatibility, expose a per-sample TopK view.
        # sparse_acts: (batch, seq_len, latent_size) -> top_acts/top_indices: (batch, seq_len, k)
        top_acts, top_indices = sparse_acts.topk(self.cfg.k, sorted=False)

        return VLSAEOutput(
            output=sae_out,
            latent_activations=top_acts,
            latent_indices=top_indices,
            recon_loss=recon_loss,
            auxk_loss=auxk_loss,
            shared_recon_loss=shared_recon_loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize each decoder row to unit norm for stable training."""
        # eps: scalar for numerical stability
        eps = torch.finfo(self.W_dec.dtype).eps
        # norm: (latent_size, 1)
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        # normalize each decoder row to unit length
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Project out gradient components parallel to decoder directions.

        This keeps decoder updates orthogonal to the current decoder weights.
        """
        assert self.W_dec.grad is not None
        # parallel_component: (latent_size,) projection of grad onto decoder rows
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "latent_size hidden_size, latent_size hidden_size -> latent_size",
        )
        # remove parallel component from gradient
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "latent_size, latent_size hidden_size -> latent_size hidden_size",
        )


class VLMatryoshkaSAE(PreTrainedModel):
    """
    Vision-language Matryoshka SAE with [visual | shared | text] subspaces.
    """

    config_class = VLMatryoshkaSAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: VLMatryoshkaSAEConfig):
        super().__init__(config)
        self.cfg = config
        self.vl_ratio = tuple(config.vl_split_ratio)
        # hidden_size: width of input activations (feature dimension)
        self.hidden_size = config.hidden_size

        # total_latents: full dictionary size
        if config.latent_size and config.latent_size > 0:
            total_latents = config.latent_size
        else:
            total_latents = int(sum(config.group_sizes))
        self.latent_size_total = total_latents
        self.latent_size = total_latents

        # group_indices: prefix boundaries for each Matryoshka group
        self.group_sizes = list(config.group_sizes)
        self.group_indices = [0]
        for size in self.group_sizes:
            self.group_indices.append(self.group_indices[-1] + int(size))

        # active_groups: number of prefix groups used for reconstruction
        self.active_groups = (
            int(config.active_groups)
            if config.active_groups is not None
            else len(self.group_sizes)
        )
        if not (0 < self.active_groups <= len(self.group_sizes)):
            raise ValueError("active_groups must be within [1, len(group_sizes)].")

        # Encoder maps input activations to the full latent dictionary.
        # encoder weight shape: (latent_size_total, hidden_size)
        self.encoder = nn.Linear(self.hidden_size, total_latents)
        self.encoder.bias.data.zero_()

        # Decoder weight initialized from encoder (tied init).
        # decoder weight shape: (latent_size_total, hidden_size)
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Decoder bias shifts reconstructions; it is subtracted before encoding.
        # decoder bias shape: (hidden_size,)
        self.b_dec = nn.Parameter(torch.zeros(self.hidden_size))

        # Shared-group configuration within the shared subspace.
        v_size, s_size, t_size = _vl_split_indices(total_latents, self.vl_ratio)
        self.shared_start = v_size
        self.shared_end = v_size + s_size
        if config.shared_group_sizes is None:
            self.shared_group_sizes = [s_size]
        else:
            self.shared_group_sizes = [int(size) for size in config.shared_group_sizes]
        if sum(self.shared_group_sizes) != s_size:
            raise ValueError(
                "Sum of shared_group_sizes must equal shared subspace size "
                f"({s_size}), got {sum(self.shared_group_sizes)}."
            )
        self.shared_group_indices = [0]
        for size in self.shared_group_sizes:
            self.shared_group_indices.append(self.shared_group_indices[-1] + int(size))
        self.shared_active_groups = (
            int(config.shared_active_groups)
            if config.shared_active_groups is not None
            else len(self.shared_group_sizes)
        )
        if not (0 < self.shared_active_groups <= len(self.shared_group_sizes)):
            raise ValueError(
                "shared_active_groups must be within [1, len(shared_group_sizes)]."
            )

        self.post_init()

    def _init_weights(self, module: nn.Module):
        return

    @property
    def device(self):
        """Convenience accessor for the module's device."""
        return self.encoder.weight.device

    @property
    def dtype(self):
        """Convenience accessor for the module's dtype."""
        return self.encoder.weight.dtype

    def _modality_mask(self, visual_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Build modality and shared masks from a visual token mask.

        Returns:
            - active_mask: (batch, seq_len, latent_size_total) boolean mask for active subspaces.
            - shared_mask: (latent_size_total,) boolean mask for shared subspace only.
        """
        if visual_mask.dim() != 2:
            raise ValueError("visual_mask must be (batch, seq_len).")

        v_mask, s_mask, t_mask = _vl_masks(self.W_dec.shape[0], self.device, self.vl_ratio)
        active_visual = (v_mask | s_mask).view(1, 1, -1)
        active_text = (s_mask | t_mask).view(1, 1, -1)
        visual_mask_broadcast = visual_mask.bool().unsqueeze(-1)
        active_mask = torch.where(visual_mask_broadcast, active_visual, active_text)
        return active_mask, s_mask

    def pre_acts(self, x: Tensor) -> Tensor:
        """
        Compute latent pre-activations (ReLU applied after encoder).

        Input x: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, latent_size_total)
        """
        # Subtract decoder bias before encoding.
        # x: (batch, seq_len, hidden_size) -> sae_in: (batch, seq_len, hidden_size)
        sae_in = x.to(self.dtype) - self.b_dec

        # Linear projection to full latent dictionary.
        # sae_in: (batch, seq_len, hidden_size) -> out: (batch, seq_len, latent_size_total)
        out = self.encoder(sae_in)

        # ReLU keeps positive activations only.
        # out: (batch, seq_len, latent_size_total) -> pre_acts: (batch, seq_len, latent_size_total)
        return nn.functional.relu(out)

    def _batch_topk_mask(self, pre_acts: Tensor) -> Tensor:
        """
        Batch-level TopK mask across all latents (already modality-masked).

        Input:
            - pre_acts: (batch, seq_len, latent_size_total)
        Output:
            - mask: (batch, seq_len, latent_size_total)
        """
        batch, seq_len, latent_size = pre_acts.shape
        num_tokens = batch * seq_len
        total_k = min(num_tokens * self.cfg.k, num_tokens * latent_size)

        # flat_acts: (batch * seq_len * latent_size_total)
        flat_acts = pre_acts.reshape(-1)
        # top_indices: (total_k,)
        _, top_indices = flat_acts.topk(total_k, sorted=False)

        # mask: (batch * seq_len * latent_size_total) -> reshape to (batch, seq_len, latent_size_total)
        mask = torch.zeros_like(flat_acts, dtype=torch.bool)
        mask[top_indices] = True
        return mask.view(batch, seq_len, latent_size)

    def _compute_activations(self, pre_acts: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute raw activations and the sparse BatchTopK activations.

        Input:
            - pre_acts: (batch, seq_len, latent_size_total)
        Output:
            - acts: (batch, seq_len, latent_size_total)
            - acts_topk: (batch, seq_len, latent_size_total)
        """
        # acts: (batch, seq_len, latent_size_total)
        acts = pre_acts
        # mask: (batch, seq_len, latent_size_total)
        mask = self._batch_topk_mask(acts)
        # acts_topk: (batch, seq_len, latent_size_total)
        acts_topk = torch.where(mask, acts, torch.zeros_like(acts))
        return acts, acts_topk

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs and return top-k activations and indices."""
        return self.pre_acts(x).topk(self.cfg.k, sorted=False)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode a sparse (top-k) representation back to input space."""
        y = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec)
        return y + self.b_dec

    def forward(
        self,
        hidden_states: Tensor,
        *,
        visual_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        dead_mask: Optional[Tensor] = None,
    ) -> VLMatryoshkaSAEOutput:
        """
        Run a forward pass and compute Matryoshka reconstruction losses.

        Args:
            hidden_states: Input activations. Shape: (batch, seq_len, hidden_size).
            visual_mask: Boolean mask for visual tokens. Shape: (batch, seq_len).
            attention_mask: Boolean mask for valid tokens. Shape: (batch, seq_len).
            dead_mask: Optional boolean mask of dead features. Shape: (latent_size_total,).

        Returns:
            VLMatryoshkaSAEOutput with reconstruction and training metrics.
        """
        x = hidden_states
        if logger.isEnabledFor(logging.DEBUG):
            total_tokens = visual_mask.numel()
            visual_tokens = int(visual_mask.sum().item())
            attn_tokens = int(attention_mask.sum().item()) if attention_mask is not None else None
            v_size, s_size, t_size = _vl_split_indices(self.W_dec.shape[0], self.vl_ratio)
            logger.debug(
                "VLMatry stats: tokens=%d vis=%d attn=%s v/s/t=%d/%d/%d",
                total_tokens,
                visual_tokens,
                attn_tokens,
                v_size,
                s_size,
                t_size,
            )
        # 1) Encode to latent pre-activations.
        # x: (batch, seq_len, hidden_size) -> pre_acts: (batch, seq_len, latent_size_total)
        pre_acts = self.pre_acts(x)

        # 2) Apply modality mask before BatchTopK.
        # active_mask: (batch, seq_len, latent_size_total)
        active_mask, shared_mask = self._modality_mask(visual_mask)
        if attention_mask is not None:
            if attention_mask.shape != visual_mask.shape:
                raise ValueError("attention_mask must match visual_mask shape.")
            active_mask = active_mask & attention_mask.bool().unsqueeze(-1)
        masked_pre_acts = torch.where(active_mask, pre_acts, torch.full_like(pre_acts, -torch.inf))
        if logger.isEnabledFor(logging.DEBUG):
            num_blocked = int((~active_mask).sum().item())
            logger.debug("VLMatry mask: blocked=%d", num_blocked)

        # 3) BatchTopK within active subspace.
        # masked_pre_acts: (batch, seq_len, latent_size_total) -> acts_topk: (batch, seq_len, latent_size_total)
        batch_mask = self._batch_topk_mask(masked_pre_acts)
        batch_mask = batch_mask & active_mask
        if attention_mask is not None:
            batch_mask = batch_mask & attention_mask.bool().unsqueeze(-1)
        acts_topk = torch.where(batch_mask, pre_acts, torch.zeros_like(pre_acts))

        # 4) Reconstruct progressively using group prefixes.
        # Initialize with bias-only reconstruction.
        # b_dec: (hidden_size,) broadcasts to (batch, seq_len, hidden_size)
        reconstruct = self.b_dec
        intermediate_reconstructs: list[Tensor] = []
        shared_reconstruct = self.b_dec
        shared_intermediate_reconstructs: list[Tensor] = []

        for i in range(self.active_groups):
            start = self.group_indices[i]
            end = self.group_indices[i + 1]
            W_dec_slice = self.W_dec[start:end, :]
            acts_slice = acts_topk[:, :, start:end]

            # acts_slice: (batch, seq_len, group_size) @ W_dec_slice^T -> (batch, seq_len, hidden_size)
            reconstruct = acts_slice.to(self.dtype) @ W_dec_slice + reconstruct
            intermediate_reconstructs.append(reconstruct)

        for i in range(self.shared_active_groups):
            start = self.shared_start + self.shared_group_indices[i]
            end = self.shared_start + self.shared_group_indices[i + 1]
            W_dec_slice = self.W_dec[start:end, :]
            acts_slice = acts_topk[:, :, start:end]
            shared_reconstruct = acts_slice.to(self.dtype) @ W_dec_slice + shared_reconstruct
            shared_intermediate_reconstructs.append(shared_reconstruct)

        # Final reconstruction uses all active groups.
        sae_out = reconstruct

        # 5) Compute Matryoshka losses (mean/min/max across prefixes).
        # Each prefix reconstruction: (batch, seq_len, hidden_size) -> normalized L2 sum.
        total_variance, flat_mask = _masked_total_variance(x, attention_mask)
        l2_losses = torch.stack(
            [_masked_l2_sum(r.float() - x.float(), flat_mask) / total_variance for r in intermediate_reconstructs],
            dim=0,
        )
        min_l2_loss = l2_losses.min()
        max_l2_loss = l2_losses.max()
        # Baseline (bias-only) reconstruction: (hidden_size,) broadcasts to (batch, seq_len, hidden_size) -> scalar.
        base_l2_loss = _masked_l2_sum(self.b_dec - x.float(), flat_mask) / total_variance
        mean_l2_loss = (l2_losses.sum() + base_l2_loss) / (len(intermediate_reconstructs) + 1)
        shared_l2_losses = torch.stack(
            [_masked_l2_sum(r.float() - x.float(), flat_mask) / total_variance for r in shared_intermediate_reconstructs],
            dim=0,
        )
        shared_min_l2_loss = shared_l2_losses.min()
        shared_max_l2_loss = shared_l2_losses.max()
        shared_mean_l2_loss = (shared_l2_losses.sum() + base_l2_loss) / (
                len(shared_intermediate_reconstructs) + 1
        )

        # 6) Fraction of variance unexplained based on the final reconstruction.
        # total_variance: scalar normalization term
        # residual: (batch, seq_len, hidden_size)
        residual = sae_out - x
        # l2_loss: scalar
        l2_loss = _masked_l2_sum(residual, flat_mask)
        recon_loss = l2_loss / total_variance

        # 7) Optional AuxK loss for dead features.
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # k_aux: auxiliary top-k for dead latents (scalar)
            k_aux = x.shape[-1] // 2
            # scale: down-weight aux loss when few dead features exist
            scale = min(num_dead / k_aux, 1.0)
            # clamp k_aux to number of dead features
            k_aux = min(k_aux, num_dead)

            # aux_latents: (batch, seq_len, latent_size_total)
            aux_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            if attention_mask is not None:
                aux_latents = torch.where(
                    attention_mask.bool().unsqueeze(-1), aux_latents, torch.full_like(aux_latents, -torch.inf)
                )
            aux_acts, aux_indices = aux_latents.topk(k_aux, sorted=False)
            aux_acts = _sanitize_topk_acts(aux_acts)
            # aux_acts/aux_indices: (batch, seq_len, k_aux) -> e_hat: (batch, seq_len, hidden_size)
            e_hat = _decoder_impl(aux_indices, aux_acts.to(self.dtype), self.W_dec) + self.b_dec
            auxk_loss = (e_hat - residual).pow(2).sum()
            # normalize and scale aux loss (scalar)
            auxk_loss = scale * auxk_loss / total_variance
            auxk_loss = torch.nan_to_num(auxk_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        # 8) Shared-subspace-only reconstruction loss (shared-only BatchTopK).  # shared-only loss block
        shared_mask_broadcast = shared_mask.view(1, 1, -1)  # (1, 1, latent_size_total) broadcast mask
        shared_pre_acts = torch.where(  # (batch, seq_len, latent_size_total) keep shared, else -inf
            shared_mask_broadcast, pre_acts, torch.full_like(pre_acts, -torch.inf)  # mask pre_acts
        )
        if attention_mask is not None:  # optional token mask
            shared_pre_acts = torch.where(  # (batch, seq_len, latent_size_total) drop masked tokens
                attention_mask.bool().unsqueeze(-1),  # (batch, seq_len, 1) token mask
                shared_pre_acts,  # (batch, seq_len, latent_size_total) candidate acts
                torch.full_like(shared_pre_acts, -torch.inf),  # (batch, seq_len, latent_size_total) masked
            )
        shared_batch_mask = self._batch_topk_mask(shared_pre_acts)  # (batch, seq_len, latent_size_total) top-k mask
        shared_batch_mask = shared_batch_mask & shared_mask_broadcast  # (batch, seq_len, latent_size_total) shared
        if attention_mask is not None:  # optional token mask
            shared_batch_mask = shared_batch_mask & attention_mask.bool().unsqueeze(-1)  # (batch, seq_len, latent)
        shared_acts = torch.where(shared_batch_mask, pre_acts, torch.zeros_like(pre_acts))  # (batch, seq_len, latent)
        shared_recon = shared_acts.to(self.dtype) @ self.W_dec + self.b_dec  # (batch, seq_len, hidden_size)
        # shared_recon_loss: scalar
        shared_recon_loss = _masked_l2_sum(shared_recon - x, flat_mask) / total_variance

        # 9) For logging/compatibility, expose a per-sample TopK view.
        # acts_topk: (batch, seq_len, latent_size_total) -> top_acts/top_indices: (batch, seq_len, k)
        top_acts, top_indices = acts_topk.topk(self.cfg.k, sorted=False)

        return VLMatryoshkaSAEOutput(
            output=sae_out,
            latent_activations=top_acts,
            latent_indices=top_indices,
            recon_loss=recon_loss,
            auxk_loss=auxk_loss,
            mean_l2_loss=mean_l2_loss,
            min_l2_loss=min_l2_loss,
            max_l2_loss=max_l2_loss,
            shared_mean_l2_loss=shared_mean_l2_loss,
            shared_min_l2_loss=shared_min_l2_loss,
            shared_max_l2_loss=shared_max_l2_loss,
            shared_recon_loss=shared_recon_loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize each decoder row to unit norm for stable training."""
        # eps: scalar for numerical stability
        eps = torch.finfo(self.W_dec.dtype).eps
        # norm: (latent_size_total, 1)
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        # normalize each decoder row to unit length
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Project out gradient components parallel to decoder directions.

        This keeps decoder updates orthogonal to the current decoder weights.
        """
        assert self.W_dec.grad is not None
        # parallel_component: (latent_size_total,) projection of grad onto decoder rows
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "latent_size hidden_size, latent_size hidden_size -> latent_size",
        )
        # remove parallel component from gradient
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "latent_size, latent_size hidden_size -> latent_size hidden_size",
        )


__all__ = [
    "TopKSAE",
    "BatchTopKSAE",
    "MatryoshkaSAE",
    "VLTopKSAE",
    "VLBatchTopKSAE",
    "VLMatryoshkaSAE",
    "SAEOutput",
    "MatryoshkaSAEOutput",
    "VLSAEOutput",
    "VLMatryoshkaSAEOutput",
]
logger = logging.getLogger(__name__)
