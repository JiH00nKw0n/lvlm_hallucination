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
        loss = outputs.recon_loss
        if self.auxk_weight:
            loss = loss + self.auxk_weight * outputs.auxk_loss
        if self.shared_weight and hasattr(outputs, "shared_recon_loss"):
            loss = loss + self.shared_weight * outputs.shared_recon_loss

        self._update_sae_loss_state(outputs)

        if self.auxk_weight:
            if attention_mask is not None:
                num_tokens = int(attention_mask.sum().item())
            else:
                num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            self.num_tokens_since_fired += num_tokens
            fired = outputs.latent_indices.flatten()
            if fired.numel() > 0:
                self.num_tokens_since_fired[fired] = 0

        return (loss, outputs) if return_outputs else loss

    def _update_sae_loss_state(self, outputs):
        recon_value = getattr(outputs, "mean_l2_loss", None)
        if recon_value is None:
            recon_value = outputs.recon_loss
        shared_value = getattr(outputs, "shared_mean_l2_loss", None)
        if shared_value is None:
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

