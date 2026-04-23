import logging
from typing import Union, Optional

from accelerate import PartialState
from datasets import Dataset, IterableDataset
import torch
from transformers import (
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
)
from trl import DPOConfig, maybe_extract_prompt, maybe_apply_chat_template, is_conversational, \
    prepare_multimodal_messages
from trl.trainer import SFTTrainer, DPOTrainer

from src.common.registry import registry

logger = logging.getLogger(__name__)

__all__ = [
    "CustomSFTTrainer",
    "CustomDPOTrainer",
    "SAETrainer",
    "CLIPSAETrainer",
    "OneSidedSAETrainer",
    "OneSidedAuxSAETrainer",
    "TwoSidedSAETrainer",
    "VLSAETrainer",
    "DeadReviveCallback",
    "RealAuxAlignmentTrainer",
]


@registry.register_trainer('CustomSFTTrainer')
class CustomSFTTrainer(SFTTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, output = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        return (loss, output) if return_outputs else loss


@registry.register_trainer('CustomDPOTrainer')
class CustomDPOTrainer(DPOTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, output = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        return (loss, output) if return_outputs else loss

    def _prepare_dataset(
            self,
            dataset: Union[Dataset, IterableDataset],
            processing_class: Union[
                PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
            args: DPOConfig,
            dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc nor writer_batch_size
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        with PartialState().main_process_first():
            # Prepare multimodal messages for conversational data
            def prepare_multimodal_example(example):
                if is_conversational(example):
                    prepare_multimodal_messages(example["prompt"], len(example["images"]))
                    prepare_multimodal_messages(example["chosen"], len(example["images"]))
                    prepare_multimodal_messages(example["rejected"], len(example["images"]))
                return example

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Preparing multimodal messages in {dataset_name} dataset"
            dataset = dataset.map(prepare_multimodal_example, **map_kwargs)

            # Extract prompt if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
            )

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                remove_columns=["chosen", "rejected"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset


@registry.register_trainer("SAETrainer")
class SAETrainer(Trainer):
    supports_sae_weights = True

    def __init__(
        self,
        *args,
        auxk_weight: float = 0.0,
        shared_weight: float = 0.0,
        dead_feature_threshold: int = 10_000_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.auxk_weight = float(auxk_weight)
        self.shared_weight = float(shared_weight)
        self.dead_feature_threshold = int(dead_feature_threshold)
        self.num_tokens_since_fired = None
        self._last_epoch_save_idx = None
        self.model_accepts_loss_kwargs = False
        self._sae_loss_keys = [
            "recon_loss",
            "auxk_loss",
            "shared_recon_loss",
            "mean_l2_loss",
            "min_l2_loss",
            "max_l2_loss",
            "shared_mean_l2_loss",
            "shared_min_l2_loss",
            "shared_max_l2_loss",
        ]

    def _init_dead_mask_state(self, model, device):
        latent_size = getattr(model, "latent_size", None)
        if latent_size is None:
            latent_size = getattr(model, "latent_size_total", None)
        if latent_size is None:
            raise ValueError("Model missing latent_size/latent_size_total for dead feature tracking.")
        self.num_tokens_since_fired = torch.zeros(latent_size, device=device, dtype=torch.long)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)
        visual_mask = inputs.get("visual_mask")
        if visual_mask is not None:
            visual_mask = visual_mask.to(hidden_states.device)

        use_vl = "VL" in model.__class__.__name__
        model_kwargs = {"hidden_states": hidden_states}
        if use_vl:
            if visual_mask is None:
                raise ValueError("visual_mask is required for VL SAE models.")
            model_kwargs["visual_mask"] = visual_mask
            model_kwargs["attention_mask"] = attention_mask

        dead_mask = None
        if self.auxk_weight:
            if self.num_tokens_since_fired is None:
                self._init_dead_mask_state(model, hidden_states.device)
            dead_mask = self.num_tokens_since_fired > self.dead_feature_threshold
            model_kwargs["dead_mask"] = dead_mask

        outputs = model(**model_kwargs)
        loss = getattr(outputs, "mean_l2_loss", None)
        if loss is None:
            loss = outputs.recon_loss
        if self.auxk_weight:
            auxk = outputs.auxk_loss
            if auxk is not None and not torch.isfinite(auxk):
                logger.warning("auxk_loss is non-finite; setting to 0 for this step.")
                auxk = auxk.new_tensor(0.0)
            loss = loss + self.auxk_weight * (auxk if auxk is not None else 0.0)
        if self.shared_weight and (hasattr(outputs, "shared_recon_loss") or hasattr(outputs, "shared_mean_l2_loss")):
            shared = getattr(outputs, "shared_mean_l2_loss", None)
            if shared is None:
                shared = outputs.shared_recon_loss
            if shared is not None and not torch.isfinite(shared):
                logger.warning("shared_recon_loss is non-finite; setting to 0 for this step.")
                shared = shared.new_tensor(0.0)
            loss = loss + self.shared_weight * (shared if shared is not None else 0.0)

        self._update_sae_loss_state(outputs)

        if self.auxk_weight:
            if attention_mask is not None:
                num_tokens = int(attention_mask.sum().item())
            else:
                num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            self.num_tokens_since_fired += num_tokens
            fired = None
            acts = getattr(outputs, "latent_activations", None)
            indices = getattr(outputs, "latent_indices", None)
            if acts is not None and indices is not None:
                fired = indices[acts > 0]
            elif indices is not None:
                fired = indices.flatten()
            if fired is not None and fired.numel() > 0:
                self.num_tokens_since_fired[fired] = 0

        return (loss, outputs) if return_outputs else loss

    def _update_sae_loss_state(self, outputs):
        recon_value = outputs.recon_loss
        shared_value = getattr(outputs, "shared_recon_loss", None)

        metrics = {
            "recon_loss": recon_value,
            "auxk_loss": getattr(outputs, "auxk_loss", None),
            "shared_recon_loss": shared_value,
            "mean_l2_loss": getattr(outputs, "mean_l2_loss", None),
            "min_l2_loss": getattr(outputs, "min_l2_loss", None),
            "max_l2_loss": getattr(outputs, "max_l2_loss", None),
            "shared_mean_l2_loss": getattr(outputs, "shared_mean_l2_loss", None),
            "shared_min_l2_loss": getattr(outputs, "shared_min_l2_loss", None),
            "shared_max_l2_loss": getattr(outputs, "shared_max_l2_loss", None),
        }
        for key, value in metrics.items():
            if value is None:
                continue
            current = getattr(self.state, key, torch.tensor(0.0, device=value.device))
            setattr(self.state, key, current + value.detach())

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        steps = self.state.global_step - getattr(self.state, "sae_last_logged", 0)
        if steps <= 0:
            steps = 1

        for key in self._sae_loss_keys:
            if hasattr(self.state, key):
                val = self._nested_gather(getattr(self.state, key)).mean().item()
                logs[key] = round(val / steps, 6)
                setattr(self.state, key, torch.tensor(0.0, device=self.args.device))

        self.state.sae_last_logged = self.state.global_step
        super().log(logs, start_time=start_time)


@registry.register_trainer("CLIPSAETrainer")
class CLIPSAETrainer(Trainer):
    """
    Trainer for CLIP SAE: forwards the SAE twice per batch (image + text embeddings)
    and optionally computes a group-sparse loss (L_{2,1} norm) on paired latent codes.
    """

    supports_sae_weights = True

    def __init__(
        self,
        *args,
        auxk_weight: float = 0.0,
        shared_weight: float = 0.0,
        dead_feature_threshold: int = 10_000_000,
        use_group_sparse_loss: bool = False,
        group_sparse_lambda: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.auxk_weight = float(auxk_weight)
        self.shared_weight = float(shared_weight)
        self.dead_feature_threshold = int(dead_feature_threshold)
        self.use_group_sparse_loss = use_group_sparse_loss
        self.group_sparse_lambda = float(group_sparse_lambda)
        self.num_tokens_since_fired = None
        self._last_epoch_save_idx = None
        self.model_accepts_loss_kwargs = False
        self._sae_loss_keys = [
            "recon_loss",
            "auxk_loss",
            "shared_recon_loss",
            "group_sparse_loss",
        ]

    def _init_dead_mask_state(self, model, device):
        latent_size = getattr(model, "latent_size", None)
        if latent_size is None:
            latent_size = getattr(model, "latent_size_total", None)
        if latent_size is None:
            raise ValueError("Model missing latent_size/latent_size_total for dead feature tracking.")
        self.num_tokens_since_fired = torch.zeros(latent_size, device=device, dtype=torch.long)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        image_embeds = inputs["image_embeds"]
        text_embeds = inputs["text_embeds"]

        # Unsqueeze to (batch, 1, hidden_size) for SAE seq_len dimension
        img_input = image_embeds.unsqueeze(1)
        txt_input = text_embeds.unsqueeze(1)

        use_vl = "VL" in model.__class__.__name__
        need_dense = self.use_group_sparse_loss

        # Dead feature mask
        dead_mask = None
        if self.auxk_weight:
            if self.num_tokens_since_fired is None:
                self._init_dead_mask_state(model, image_embeds.device)
            dead_mask = self.num_tokens_since_fired > self.dead_feature_threshold

        # Forward SAE for image embeddings
        if use_vl:
            visual_mask_img = inputs["visual_mask"]
            outputs_img = model(
                hidden_states=img_input,
                visual_mask=visual_mask_img,
                dead_mask=dead_mask,
                return_dense_latents=need_dense,
            )
        else:
            outputs_img = model(
                hidden_states=img_input,
                dead_mask=dead_mask,
                return_dense_latents=need_dense,
            )

        # Forward SAE for text embeddings
        if use_vl:
            text_mask = inputs["text_mask"]
            outputs_txt = model(
                hidden_states=txt_input,
                visual_mask=text_mask,
                dead_mask=dead_mask,
                return_dense_latents=need_dense,
            )
        else:
            outputs_txt = model(
                hidden_states=txt_input,
                dead_mask=dead_mask,
                return_dense_latents=need_dense,
            )

        # Reconstruction loss (sum of image + text)
        loss = outputs_img.recon_loss + outputs_txt.recon_loss

        # AuxK loss
        auxk_total = outputs_img.recon_loss.new_tensor(0.0)
        if self.auxk_weight:
            for out in (outputs_img, outputs_txt):
                auxk = out.auxk_loss
                if auxk is not None:
                    if not torch.isfinite(auxk):
                        logger.warning("auxk_loss is non-finite; setting to 0 for this step.")
                        auxk = auxk.new_tensor(0.0)
                    auxk_total = auxk_total + auxk
            loss = loss + self.auxk_weight * auxk_total

        # Shared reconstruction loss (VL models only)
        shared_total = outputs_img.recon_loss.new_tensor(0.0)
        if self.shared_weight and use_vl:
            for out in (outputs_img, outputs_txt):
                shared = getattr(out, "shared_recon_loss", None)
                if shared is not None:
                    if not torch.isfinite(shared):
                        logger.warning("shared_recon_loss is non-finite; setting to 0 for this step.")
                        shared = shared.new_tensor(0.0)
                    shared_total = shared_total + shared
            loss = loss + self.shared_weight * shared_total

        # Group-sparse loss: L_{2,1} norm over paired latent codes
        gs_loss = outputs_img.recon_loss.new_tensor(0.0)
        if self.use_group_sparse_loss and need_dense:
            z_x = outputs_img.dense_latents.squeeze(1)  # (batch, latent_size)
            z_y = outputs_txt.dense_latents.squeeze(1)   # (batch, latent_size)
            eps = torch.finfo(z_x.dtype).eps
            gs_loss = (z_x.pow(2) + z_y.pow(2) + eps).sqrt().sum()
            loss = loss + self.group_sparse_lambda * gs_loss

        # Track metrics
        self._update_clip_sae_loss_state(
            outputs_img, outputs_txt, auxk_total, shared_total, gs_loss,
        )

        # Dead feature tracking
        if self.auxk_weight:
            num_tokens = img_input.shape[0] + txt_input.shape[0]
            self.num_tokens_since_fired += num_tokens
            for out in (outputs_img, outputs_txt):
                acts = getattr(out, "latent_activations", None)
                indices = getattr(out, "latent_indices", None)
                fired = None
                if acts is not None and indices is not None:
                    fired = indices[acts > 0]
                elif indices is not None:
                    fired = indices.flatten()
                if fired is not None and fired.numel() > 0:
                    self.num_tokens_since_fired[fired] = 0

        return (loss, outputs_img) if return_outputs else loss

    def _update_clip_sae_loss_state(self, outputs_img, outputs_txt, auxk_total, shared_total, gs_loss):
        recon_value = outputs_img.recon_loss + outputs_txt.recon_loss
        metrics = {
            "recon_loss": recon_value,
            "auxk_loss": auxk_total,
            "shared_recon_loss": shared_total,
            "group_sparse_loss": gs_loss,
        }
        for key, value in metrics.items():
            if value is None:
                continue
            current = getattr(self.state, key, torch.tensor(0.0, device=value.device))
            setattr(self.state, key, current + value.detach())

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        steps = self.state.global_step - getattr(self.state, "sae_last_logged", 0)
        if steps <= 0:
            steps = 1

        for key in self._sae_loss_keys:
            if hasattr(self.state, key):
                val = self._nested_gather(getattr(self.state, key)).mean().item()
                logs[key] = round(val / steps, 6)
                setattr(self.state, key, torch.tensor(0.0, device=self.args.device))

        self.state.sae_last_logged = self.state.global_step
        super().log(logs, start_time=start_time)


@registry.register_trainer("OneSidedSAETrainer")
class OneSidedSAETrainer(Trainer):
    """
    Trainer for a single (shared-decoder) TopKSAE on pre-cached CLIP image/text
    embeddings. Forwards the model twice per batch (once for image, once for
    text) and uses the mean of the two reconstruction losses as the optimized
    objective.

    Unlike `CLIPSAETrainer`, this trainer is intentionally minimal: no AuxK,
    no group-sparse, no dead-feature tracking — matching the real-α-diagnostic
    experiment design.
    """

    supports_sae_weights = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        image_embeds = inputs["image_embeds"]
        text_embeds = inputs["text_embeds"]
        hs_i = image_embeds.unsqueeze(1)
        hs_t = text_embeds.unsqueeze(1)
        out_i = model(hidden_states=hs_i)
        out_t = model(hidden_states=hs_t)
        loss = (out_i.recon_loss + out_t.recon_loss) / 2

        for key, val in (
            ("recon_loss", loss),
            ("recon_loss_image", out_i.recon_loss),
            ("recon_loss_text", out_t.recon_loss),
        ):
            attr = f"_os_{key}"
            current = getattr(self.state, attr, torch.tensor(0.0, device=val.device))
            setattr(self.state, attr, current + val.detach())

        return (loss, (out_i, out_t)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss.detach(), None, None)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        steps = self.state.global_step - getattr(self.state, "_os_last_logged", 0)
        if steps <= 0:
            steps = 1
        for key in ("recon_loss", "recon_loss_image", "recon_loss_text"):
            attr = f"_os_{key}"
            if hasattr(self.state, attr):
                tensor = getattr(self.state, attr)
                val = float(tensor.detach().mean().item())
                logs[key] = round(val / steps, 6)
                setattr(self.state, attr, torch.tensor(0.0, device=self.args.device))
        self.state._os_last_logged = self.state.global_step
        super().log(logs, start_time=start_time)


def _iso_alignment_penalty(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """Top-1 masked cosine alignment (Parabrele/IsoEnergy paper baseline).

    Ported from `synthetic_theorem2_method._iso_alignment_penalty`. Masks each
    sample's top-1 latent in BOTH halves before computing cosine similarity,
    so the "modality-specific" top-1 slot does not dominate the alignment
    signal. Returns a scalar in [-1, 1]; the caller scales by aux_weight.
    """
    import torch.nn.functional as F
    n = z_img.shape[0]
    top_1_img = torch.topk(z_img, k=1, dim=1).indices.squeeze(1)
    top_1_txt = torch.topk(z_txt, k=1, dim=1).indices.squeeze(1)
    arange = torch.arange(n, device=z_img.device)
    mask_i = torch.ones_like(z_img)
    mask_t = torch.ones_like(z_txt)
    mask_i[arange, top_1_img] = 0.0
    mask_i[arange, top_1_txt] = 0.0
    mask_t[arange, top_1_img] = 0.0
    mask_t[arange, top_1_txt] = 0.0
    cos = F.cosine_similarity(z_img * mask_i, z_txt * mask_t, dim=1).mean()
    return -cos


def _group_sparse_penalty(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """Group-L_{2,1} penalty sum_j sqrt(z_I_j^2 + z_T_j^2), averaged over batch.

    Ported from `synthetic_theory_simplified.group_sparse_loss`.
    """
    return torch.sqrt(z_img.pow(2) + z_txt.pow(2) + 1e-12).sum(dim=-1).mean()


_AUX_LOSS_REGISTRY = {
    "iso_align": _iso_alignment_penalty,
    "group_sparse": _group_sparse_penalty,
}


@registry.register_trainer("OneSidedAuxSAETrainer")
class OneSidedAuxSAETrainer(OneSidedSAETrainer):
    """OneSidedSAETrainer + paired aux loss (iso_align or group_sparse).

    Runs a single shared TopKSAE through both modalities, computes per-modality
    reconstruction loss, and adds an aux term that couples the two dense
    latent vectors. Used for Iso-Energy Align and Group-Sparse baselines in
    the real-data downstream table.
    """

    def __init__(self, *args, aux_loss: str = "iso_align", aux_weight: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        if aux_loss not in _AUX_LOSS_REGISTRY:
            raise ValueError(f"aux_loss must be one of {list(_AUX_LOSS_REGISTRY)}, got {aux_loss}")
        self.aux_loss_name = aux_loss
        self.aux_fn = _AUX_LOSS_REGISTRY[aux_loss]
        self.aux_weight = float(aux_weight)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        image_embeds = inputs["image_embeds"]
        text_embeds = inputs["text_embeds"]
        hs_i = image_embeds.unsqueeze(1)
        hs_t = text_embeds.unsqueeze(1)
        out_i = model(hidden_states=hs_i, return_dense_latents=True)
        out_t = model(hidden_states=hs_t, return_dense_latents=True)
        # Squeeze the seq_len=1 dim on dense latents to get (B, L)
        z_i = out_i.dense_latents.squeeze(1)
        z_t = out_t.dense_latents.squeeze(1)

        recon = (out_i.recon_loss + out_t.recon_loss) / 2
        aux = self.aux_fn(z_i, z_t)
        loss = recon + self.aux_weight * aux

        for key, val in (
            ("recon_loss", recon),
            ("recon_loss_image", out_i.recon_loss),
            ("recon_loss_text", out_t.recon_loss),
            ("aux_loss", aux),
        ):
            attr = f"_os_{key}"
            current = getattr(self.state, attr, torch.tensor(0.0, device=val.device))
            setattr(self.state, attr, current + val.detach())

        return (loss, (out_i, out_t)) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        steps = self.state.global_step - getattr(self.state, "_os_last_logged", 0)
        if steps <= 0:
            steps = 1
        for key in ("recon_loss", "recon_loss_image", "recon_loss_text", "aux_loss"):
            attr = f"_os_{key}"
            if hasattr(self.state, attr):
                tensor = getattr(self.state, attr)
                val = float(tensor.detach().mean().item())
                logs[key] = round(val / steps, 6)
                setattr(self.state, attr, torch.tensor(0.0, device=self.args.device))
        self.state._os_last_logged = self.state.global_step
        # Skip OneSidedSAETrainer.log (would double-reset); go straight to Trainer.log
        Trainer.log(self, logs, start_time=start_time)


@registry.register_trainer("TwoSidedSAETrainer")
class TwoSidedSAETrainer(Trainer):
    """
    Thin Trainer for `TwoSidedTopKSAE` on pre-cached image/text embeddings.

    When ``auxk_weight > 0``, per-side dead latents are tracked via
    ``num_tokens_since_fired_{i,t}`` (threshold ``dead_feature_threshold``),
    and ``dead_mask`` is forwarded into each side's TopKSAE so its internal
    AuxK branch fires. The per-side AuxK losses are summed and added to the
    main recon with weight ``auxk_weight``.
    """

    supports_sae_weights = True

    def __init__(
        self,
        *args,
        auxk_weight: float = 0.0,
        dead_feature_threshold: int = 10_000_000,
        revive_every_epoch: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.auxk_weight = float(auxk_weight)
        self.dead_feature_threshold = int(dead_feature_threshold)
        self.revive_every_epoch = bool(revive_every_epoch)
        self.num_tokens_since_fired_i: Optional[torch.Tensor] = None
        self.num_tokens_since_fired_t: Optional[torch.Tensor] = None
        # Per-epoch fire counters used by DeadReviveCallback.
        self.fire_count_epoch_i: Optional[torch.Tensor] = None
        self.fire_count_epoch_t: Optional[torch.Tensor] = None

    def _init_dead_mask_state(self, model, device):
        n = int(model.image_sae.latent_size)
        self.num_tokens_since_fired_i = torch.zeros(n, dtype=torch.long, device=device)
        self.num_tokens_since_fired_t = torch.zeros(n, dtype=torch.long, device=device)

    def _init_epoch_fire_state(self, model, device):
        n = int(model.image_sae.latent_size)
        self.fire_count_epoch_i = torch.zeros(n, dtype=torch.long, device=device)
        self.fire_count_epoch_t = torch.zeros(n, dtype=torch.long, device=device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        image_embeds = inputs["image_embeds"]
        text_embeds = inputs["text_embeds"]

        use_direct_forward = self.auxk_weight > 0.0 or self.revive_every_epoch
        if use_direct_forward:
            # Lazy init of trackers
            if self.auxk_weight > 0.0 and self.num_tokens_since_fired_i is None:
                self._init_dead_mask_state(model, image_embeds.device)
            if self.revive_every_epoch and self.fire_count_epoch_i is None:
                self._init_epoch_fire_state(model, image_embeds.device)

            dead_i = None
            dead_t = None
            if self.auxk_weight > 0.0:
                dead_i = self.num_tokens_since_fired_i > self.dead_feature_threshold
                dead_t = self.num_tokens_since_fired_t > self.dead_feature_threshold

            hs_i = image_embeds.unsqueeze(1) if image_embeds.dim() == 2 else image_embeds
            hs_t = text_embeds.unsqueeze(1) if text_embeds.dim() == 2 else text_embeds
            out_i = model.image_sae(hidden_states=hs_i, dead_mask=dead_i)
            out_t = model.text_sae(hidden_states=hs_t, dead_mask=dead_t)
            recon = (out_i.recon_loss + out_t.recon_loss) / 2
            auxk = (out_i.auxk_loss + out_t.auxk_loss) / 2
            loss = recon + self.auxk_weight * auxk

            # Update trackers
            batch_size = image_embeds.shape[0]
            fired_i = torch.unique(out_i.latent_indices.flatten())
            fired_t = torch.unique(out_t.latent_indices.flatten())
            if self.auxk_weight > 0.0:
                self.num_tokens_since_fired_i += batch_size
                self.num_tokens_since_fired_t += batch_size
                self.num_tokens_since_fired_i[fired_i] = 0
                self.num_tokens_since_fired_t[fired_t] = 0
            if self.revive_every_epoch:
                self.fire_count_epoch_i[fired_i] += 1
                self.fire_count_epoch_t[fired_t] += 1

            # Build an output shim that matches the vanilla TwoSided API
            outputs = type("Out", (), {
                "loss": loss, "recon_loss": recon,
                "recon_loss_image": out_i.recon_loss,
                "recon_loss_text": out_t.recon_loss,
                "auxk_loss": auxk,
            })()
        else:
            outputs = model(image_embeds=image_embeds, text_embeds=text_embeds)
            loss = outputs.loss

        for key in ("recon_loss", "recon_loss_image", "recon_loss_text", "auxk_loss"):
            val = getattr(outputs, key, None)
            if val is None:
                continue
            attr = f"_ts_{key}"
            current = getattr(self.state, attr, torch.tensor(0.0, device=val.device))
            setattr(self.state, attr, current + val.detach())

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss.detach(), None, None)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        steps = self.state.global_step - getattr(self.state, "_ts_last_logged", 0)
        if steps <= 0:
            steps = 1
        for key in ("recon_loss", "recon_loss_image", "recon_loss_text"):
            attr = f"_ts_{key}"
            if hasattr(self.state, attr):
                tensor = getattr(self.state, attr)
                val = float(tensor.detach().mean().item())
                logs[key] = round(val / steps, 6)
                setattr(self.state, attr, torch.tensor(0.0, device=self.args.device))
        self.state._ts_last_logged = self.state.global_step
        super().log(logs, start_time=start_time)


class VLSAETrainer(Trainer):
    """Minimal Trainer for VL-SAE (shared encoder, two modality-specific decoders).

    The model consumes `image_embeds` and `text_embeds` directly and returns
    `VLSAEOutput` whose `.loss` is the sum of per-modality MSE. No AuxK, no
    aux alignment loss, no dead-feature tracking.
    """

    supports_sae_weights = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            image_embeds=inputs["image_embeds"],
            text_embeds=inputs["text_embeds"],
        )
        loss = outputs.loss

        for key in ("recon_loss", "recon_loss_image", "recon_loss_text"):
            val = getattr(outputs, key, None)
            if val is None:
                continue
            attr = f"_vl_{key}"
            current = getattr(self.state, attr, torch.tensor(0.0, device=val.device))
            setattr(self.state, attr, current + val.detach())

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss.detach(), None, None)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        steps = self.state.global_step - getattr(self.state, "_vl_last_logged", 0)
        if steps <= 0:
            steps = 1
        for key in ("recon_loss", "recon_loss_image", "recon_loss_text"):
            attr = f"_vl_{key}"
            if hasattr(self.state, attr):
                tensor = getattr(self.state, attr)
                val = float(tensor.detach().mean().item())
                logs[key] = round(val / steps, 6)
                setattr(self.state, attr, torch.tensor(0.0, device=self.args.device))
        self.state._vl_last_logged = self.state.global_step
        super().log(logs, start_time=start_time)


# ------------------------------------------------------------------
# DeadReviveCallback — per-epoch random re-init of dead slots
# ------------------------------------------------------------------
#
# At the end of each epoch, finds per-side slots with zero fires during
# that epoch and re-inits (encoder row, bias, decoder row) to small random
# vectors. This is the "revive (random init)" axis — a cheaper alternative
# to AuxK for rescuing dead latents in real data.
#
# Resets ``trainer.fire_count_epoch_{i,t}`` after revive.


def _reinit_two_sided_slots(sae, idx) -> None:
    """Re-init (W_enc row, bias, W_dec row) at ``idx`` to small random vectors."""
    import numpy as np  # noqa: F401 — kept for future typing consistency
    if hasattr(idx, "numel"):
        if idx.numel() == 0:
            return
    elif getattr(idx, "size", 0) == 0:
        return
    device = sae.W_dec.device
    dtype = sae.W_dec.dtype
    d = sae.W_dec.shape[1]
    n = int(idx.shape[0]) if hasattr(idx, "shape") else int(len(idx))
    with torch.no_grad():
        new_dec = torch.randn(n, d, device=device, dtype=dtype)
        new_dec = new_dec / new_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sae.W_dec.data[idx] = new_dec
        new_enc = torch.randn(n, d, device=device, dtype=dtype) * 0.01
        sae.encoder.weight.data[idx] = new_enc
        sae.encoder.bias.data[idx] = 0.0


class DeadReviveCallback(TrainerCallback):
    """Revive per-side dead slots (random init) every epoch.

    Reads the trainer's per-epoch fire counters (``fire_count_epoch_i``,
    ``fire_count_epoch_t``), re-inits slots that fired zero times during the
    epoch, and resets the counters. Use with ``TwoSidedSAETrainer(
    revive_every_epoch=True)``.
    """

    def __init__(self, trainer: Optional["TwoSidedSAETrainer"] = None) -> None:
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        trainer = kwargs.get("trainer", None) or self.trainer
        if trainer is None:
            logger.warning("DeadReviveCallback: no trainer handle, skipping revive.")
            return
        if not getattr(trainer, "revive_every_epoch", False):
            return
        fc_i = getattr(trainer, "fire_count_epoch_i", None)
        fc_t = getattr(trainer, "fire_count_epoch_t", None)
        if fc_i is None or fc_t is None:
            return

        dead_i = torch.nonzero(fc_i == 0, as_tuple=False).flatten()
        dead_t = torch.nonzero(fc_t == 0, as_tuple=False).flatten()
        _reinit_two_sided_slots(model.image_sae, dead_i)
        _reinit_two_sided_slots(model.text_sae, dead_t)
        logger.info(
            "DeadReviveCallback@epoch%s: revived img=%d / txt=%d dead slots",
            int(round(state.epoch)) if state.epoch is not None else "?",
            int(dead_i.numel()), int(dead_t.numel()),
        )
        fc_i.zero_()
        fc_t.zero_()


# ------------------------------------------------------------------
# RealAuxAlignmentTrainer (method ablation on real CLIP embeddings)
# ------------------------------------------------------------------
#
# Mirrors `src/runners/synthetic_trainers.py:AuxAlignmentTrainer` but feeds
# data via the real CLIP embedding cache (image_embeds / text_embeds keys).
#
# Same `variant_cfg` interface (loss form x hungarian schedule x revive).
# `AuxAlignmentCallback` from synthetic_trainers can be reused with a thin
# adapter that forwards the same `train_img` / `train_txt` tensors.

import numpy as np

from src.configs.experiment import MethodConfig
from src.training.losses import (
    barlow_twins_aux_loss_masked,
    naive_diag_aux_loss_masked,
    slot_infonce_loss,
)


class RealAuxAlignmentTrainer(TwoSidedSAETrainer):
    """Real-data variant-aware trainer.

    Adds variant-aware aux loss + frozen_mask + fire counters on top of the
    standard TwoSidedSAETrainer recon path. Schedule + revive logic lives in
    `AuxAlignmentCallback` (from synthetic_trainers).
    """

    def __init__(
        self,
        *args,
        variant_cfg: MethodConfig,
        train_img: Optional[torch.Tensor] = None,
        train_txt: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.variant = variant_cfg
        self.train_img = train_img
        self.train_txt = train_txt
        self.diagnostics: dict = {}

        device = next(self.model.parameters()).device
        n = int(self.model.image_sae.latent_size)
        self.frozen_mask: torch.Tensor = torch.zeros(n, dtype=torch.bool, device=device)
        self.fire_count_i: torch.Tensor = torch.zeros(n, dtype=torch.long, device=device)
        self.fire_count_t: torch.Tensor = torch.zeros(n, dtype=torch.long, device=device)
        self.alive_mask_i: torch.Tensor = torch.ones(n, dtype=torch.bool, device=device)
        self.alive_mask_t: torch.Tensor = torch.ones(n, dtype=torch.bool, device=device)
        # EMA of batch cross-correlation for on_epoch_end gate/Hungarian.
        self.ema_C: Optional[torch.Tensor] = None
        self.ema_momentum: float = 0.99

        if variant_cfg.aux_loss == "infonce":
            init_log_tau = float(np.log(1.0 / 0.07))
            self.model.log_tau = torch.nn.Parameter(
                torch.tensor(init_log_tau, dtype=torch.float32, device=device)
            )

    def _aux_loss_value(self, z_i: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        loss_kind = self.variant.aux_loss
        if loss_kind == "none":
            return z_i.new_tensor(0.0)
        if not bool(self.frozen_mask.any()):
            return z_i.new_tensor(0.0)
        if loss_kind == "naive_diag":
            return naive_diag_aux_loss_masked(z_i, z_t, self.frozen_mask)
        if loss_kind == "barlow":
            return barlow_twins_aux_loss_masked(
                z_i, z_t, self.frozen_mask,
                lambda_off=float(self.variant.barlow_lambda_off),
            )
        if loss_kind == "infonce":
            return slot_infonce_loss(
                z_i, z_t, self.frozen_mask, self.model.log_tau,
                alive_mask_i=self.alive_mask_i,
                alive_mask_t=self.alive_mask_t,
            )
        raise ValueError(f"Unknown aux_loss '{loss_kind}'")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        image_embeds = inputs["image_embeds"]
        text_embeds = inputs["text_embeds"]
        # Need dense latents for both aux loss and fire-count tracking.
        out_i = model.image_sae(hidden_states=image_embeds.unsqueeze(1), return_dense_latents=True)
        out_t = model.text_sae(hidden_states=text_embeds.unsqueeze(1), return_dense_latents=True)
        z_i = out_i.dense_latents.squeeze(1)
        z_t = out_t.dense_latents.squeeze(1)

        recon = out_i.recon_loss + out_t.recon_loss
        aux = self._aux_loss_value(z_i, z_t)
        loss = recon + float(self.variant.lambda_aux) * aux

        # Per-side recon logging (matches TwoSidedSAETrainer.log)
        for key, val in (
            ("recon_loss", recon),
            ("recon_loss_image", out_i.recon_loss),
            ("recon_loss_text", out_t.recon_loss),
        ):
            attr = f"_ts_{key}"
            current = getattr(self.state, attr, torch.tensor(0.0, device=val.device))
            setattr(self.state, attr, current + val.detach())

        with torch.no_grad():
            self.fire_count_i += (z_i > 0).sum(dim=0).to(self.fire_count_i.dtype)
            self.fire_count_t += (z_t > 0).sum(dim=0).to(self.fire_count_t.dtype)
            # EMA batch cross-correlation -- eliminates the full-data forward
            # pass previously performed at on_epoch_end for schedule=per_epoch.
            if self.variant.hungarian_schedule == "per_epoch_partitioned":
                from src.runners.synthetic_trainers import _batch_cross_corr_detached
                batch_C = _batch_cross_corr_detached(z_i, z_t)
                if self.ema_C is None:
                    self.ema_C = batch_C
                else:
                    m = self.ema_momentum
                    self.ema_C.mul_(m).add_(batch_C, alpha=1.0 - m)

        outputs = type("Out", (), {"loss": loss, "recon_loss": recon})()
        return (loss, outputs) if return_outputs else loss
