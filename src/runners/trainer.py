import logging
from typing import Union

from accelerate import PartialState
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin
)
from trl import DPOConfig, maybe_extract_prompt, maybe_apply_chat_template, is_conversational, \
    prepare_multimodal_messages
from trl.trainer import SFTTrainer, DPOTrainer

from src.common.registry import registry

logger = logging.getLogger(__name__)

__all__ = [
    "CustomSFTTrainer",
    "CustomDPOTrainer",
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
