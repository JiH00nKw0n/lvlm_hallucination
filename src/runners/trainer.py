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
