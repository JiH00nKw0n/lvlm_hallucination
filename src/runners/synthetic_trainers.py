"""HF Trainer subclasses for synthetic Theorem 2 experiments.

Separate from ``trainer.py`` to avoid modifying the existing file.

Trainers:
- ``SingleReconTrainer``: one shared SAE, recon only (method: single_recon)
- ``TwoReconTrainer``: TwoSidedTopKSAE, recon only (method: two_recon)
- ``PairedAuxTrainer``: shared SAE + aux loss (methods: group_sparse, trace_align, iso_align)

All include SAE-specific gradient projection in ``training_step`` and
should be used with ``SAENormCallback`` for post-step decoder normalization.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from transformers import Trainer, TrainerCallback

from src.metrics.alignment import compute_latent_correlation
from src.training.losses import auxiliary_alignment_loss
from src.training.permutation import apply_latent_permutation, greedy_permutation_match_full

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _project_sae_gradients(model) -> None:
    """Remove gradient components parallel to decoder directions."""
    if hasattr(model, "image_sae"):  # TwoSidedTopKSAE
        for sae in (model.image_sae, model.text_sae):
            if sae.W_dec.grad is not None:
                sae.remove_gradient_parallel_to_decoder_directions()
    elif hasattr(model, "W_dec") and model.W_dec.grad is not None:
        model.remove_gradient_parallel_to_decoder_directions()


def _sae_training_step(trainer: Trainer, model, inputs) -> torch.Tensor:
    """Common training_step with gradient projection."""
    model.train()
    loss = trainer.compute_loss(model, inputs)
    if trainer.args.gradient_accumulation_steps > 1:
        loss = loss / trainer.args.gradient_accumulation_steps
    loss.backward()
    _project_sae_gradients(model)
    return loss.detach()


# ------------------------------------------------------------------
# SingleReconTrainer (method: single_recon)
# ------------------------------------------------------------------


class SingleReconTrainer(Trainer):
    """Shared SAE forwarded twice (image + text), mean recon loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)
        out_i = model(hidden_states=hs_i)
        out_t = model(hidden_states=hs_t)
        loss = out_i.recon_loss + out_t.recon_loss
        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ------------------------------------------------------------------
# TwoReconTrainer (method: two_recon)
# ------------------------------------------------------------------


class TwoReconTrainer(Trainer):
    """TwoSidedTopKSAE: disjoint image/text SAEs, recon only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)
        out_i = model.image_sae(hidden_states=hs_i)
        out_t = model.text_sae(hidden_states=hs_t)
        loss = out_i.recon_loss + out_t.recon_loss
        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ------------------------------------------------------------------
# PairedAuxTrainer (methods: group_sparse, trace_align, iso_align)
# ------------------------------------------------------------------


class PairedAuxTrainer(Trainer):
    """Shared SAE + configurable auxiliary loss on dense paired latents.

    *aux_fn(z_img, z_txt) -> scalar* and *aux_weight* are passed at init.
    """

    def __init__(self, *args, aux_fn=None, aux_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_fn = aux_fn
        self.aux_weight = aux_weight
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)

        need_dense = self.aux_fn is not None
        out_i = model(hidden_states=hs_i, return_dense_latents=need_dense)
        out_t = model(hidden_states=hs_t, return_dense_latents=need_dense)

        loss = out_i.recon_loss + out_t.recon_loss
        if self.aux_fn is not None and need_dense:
            z_i = out_i.dense_latents.squeeze(1)
            z_t = out_t.dense_latents.squeeze(1)
            aux = self.aux_fn(z_i, z_t)
            loss = loss + self.aux_weight * aux

        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ------------------------------------------------------------------
# OursTrainer (method: ours — Algorithm 1, two-stage)
# ------------------------------------------------------------------


class PermutationCallback(TrainerCallback):
    """Fires once at epoch ``k_align`` to permute latent indices."""

    def __init__(self, ours_trainer: "OursTrainer") -> None:
        self.ours = ours_trainer
        self._done = False

    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        completed = round(state.epoch)
        if self._done or completed != self.ours.k_align:
            return
        self._done = True

        device = next(model.parameters()).device
        batch_size = args.per_device_train_batch_size

        logger.info("Permutation matching at epoch %d (stage 1 -> 2)", completed)
        C = compute_latent_correlation(
            model.image_sae, model.text_sae,
            self.ours.train_img, self.ours.train_txt,
            batch_size, device,
        )
        P_I, P_T, ordered = greedy_permutation_match_full(C)

        # Unwrap accelerator optimizer if needed
        opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
        apply_latent_permutation(model.image_sae, P_I, optimizer=opt)
        apply_latent_permutation(model.text_sae, P_T, optimizer=opt)

        m_S = self.ours.m_S
        rho = self.ours.rho
        m_S_hat = int((ordered > rho).sum())
        self.ours.diagnostics = {
            "m_S_hat_rho": m_S_hat,
            "m_S_supplied": m_S,
            "first_drop_below_rho": float(ordered[m_S_hat]) if m_S_hat < len(ordered) else float("nan"),
            "top_mS_diag_mean": float(ordered[:m_S].mean()) if m_S > 0 else float("nan"),
            "ordered_diag_head": [float(x) for x in ordered[:8]],
            "k_align": self.ours.k_align,
            "lambda_aux": self.ours.lambda_aux,
            "aux_norm": self.ours.aux_norm,
        }
        logger.info("Permutation done: m_S_hat=%d, top diag=%s",
                     m_S_hat, [f"{x:.3f}" for x in ordered[:4]])
        self.ours._stage = 2


class OursTrainer(Trainer):
    """Algorithm 1: stage-aware Trainer with mid-training permutation.

    Stage 1 (epoch < k_align): recon loss only on TwoSidedTopKSAE.
    Stage 2 (epoch >= k_align): recon + lambda_aux * alignment loss.

    Use with ``PermutationCallback(trainer)`` and ``SAENormCallback``.
    """

    def __init__(
        self, *args,
        k_align: int = 6,
        lambda_aux: float = 1.0,
        m_S: int = 512,
        aux_norm: str = "group",
        train_img: Optional[torch.Tensor] = None,
        train_txt: Optional[torch.Tensor] = None,
        rho: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k_align = k_align
        self.lambda_aux = lambda_aux
        self.m_S = m_S
        self.aux_norm = aux_norm
        self.train_img = train_img
        self.train_txt = train_txt
        self.rho = rho
        self._stage = 1
        self.diagnostics: dict[str, Any] = {}
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)

        need_dense = self._stage == 2
        out_i = model.image_sae(hidden_states=hs_i, return_dense_latents=need_dense)
        out_t = model.text_sae(hidden_states=hs_t, return_dense_latents=need_dense)

        loss = out_i.recon_loss + out_t.recon_loss

        if self._stage == 2 and need_dense:
            z_i = out_i.dense_latents.squeeze(1)
            z_t = out_t.dense_latents.squeeze(1)
            aux = auxiliary_alignment_loss(z_i, z_t, self.m_S, norm=self.aux_norm)
            loss = loss + self.lambda_aux * aux

        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)
