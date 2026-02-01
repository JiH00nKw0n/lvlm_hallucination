import logging
from typing import Optional, Dict, Type, TypeVar, Any

from datasets import Dataset, IterableDataset, interleave_datasets, concatenate_datasets
from omegaconf import OmegaConf
from transformers import (
    PreTrainedModel,
    ProcessorMixin,
    TrainingArguments,
    add_end_docstrings
)

from src.common import registry, TrainConfig
from src.tasks.base import BaseTrainTask, TaskWithCustomModel, TaskWithPretrainedModel, TRAIN_TASK_DOCSTRING

ModelType = Type[PreTrainedModel]
ProcessorType = Type[ProcessorMixin]
DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = [
    "SingleTrainTask",
    "SingleTrainTaskWithPretrainedModel",
    "SingleTrainTaskWithCustomModel",
    "DatasetSingleTrainTask",
    "IterableDatasetSingleTrainTask",
    "DatasetTrainTaskWithPretrainedModel",
    "IterableDatasetTrainTaskWithPretrainedModel",
    "DatasetTrainTaskWithCustomModel",
    "IterableDatasetTrainTaskWithCustomModel",
]

logger = logging.getLogger(__name__)


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTask(BaseTrainTask):
    config: TrainConfig

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        assert "runner" in self.config.run_config, "Trainer name must be provided."

        trainer_name = self.config.run_config.runner
        trainer_cls = registry.get_trainer_class(trainer_name)
        assert trainer_cls is not None, "Trainer {} not properly registered.".format(trainer_name)

        trainer_config = trainer_config if trainer_config is not None else self.config.trainer_config.copy()

        collator_cls = registry.get_collator_class(self.config.collator_config.collator_cls)

        assert collator_cls is not None, "Collator {} not properly registered.".format(collator_cls)

        train_dataset = self.build_datasets()

        collator = collator_cls(
            processor=self.build_processor(),
            **self.config.collator_config.config,
        )

        sae_auxk_weight = trainer_config.pop("sae_auxk_weight", None)
        sae_shared_weight = trainer_config.pop("sae_shared_weight", None)
        sae_dead_feature_threshold = trainer_config.pop("sae_dead_feature_threshold", None)
        extra_trainer_kwargs: dict[str, Any] = {}
        if sae_auxk_weight is not None or sae_shared_weight is not None:
            if getattr(trainer_cls, "supports_sae_weights", False):
                extra_trainer_kwargs["auxk_weight"] = sae_auxk_weight or 0.0
                extra_trainer_kwargs["shared_weight"] = sae_shared_weight or 0.0
            else:
                raise ValueError("SAE loss weights provided but trainer does not support them.")
        if sae_dead_feature_threshold is not None:
            if getattr(trainer_cls, "supports_sae_weights", False):
                extra_trainer_kwargs["dead_feature_threshold"] = int(sae_dead_feature_threshold)
            else:
                raise ValueError("SAE dead_feature_threshold provided but trainer does not support it.")

        return trainer_cls(
            model=self.build_model(),
            args=TrainingArguments(**trainer_config),
            train_dataset=train_dataset,
            data_collator=collator,
            **extra_trainer_kwargs,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class DatasetSingleTrainTask(SingleTrainTask):

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        datasets = list()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if not isinstance(dataset, Dataset):
                raise TypeError("DatasetTrainTask must build dataset with `Dataset` type.")
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return concatenate_datasets(datasets)


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class IterableDatasetSingleTrainTask(SingleTrainTask):

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        datasets = list()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if not isinstance(dataset, IterableDataset):
                raise TypeError("DatasetTrainTask must build dataset with `IterableDataset` type.")
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return interleave_datasets(datasets).with_format("torch")


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTaskWithPretrainedModel(SingleTrainTask, TaskWithPretrainedModel):

    def build_model(self, model_config: Optional[Dict] = None):
        """
        Builds a pretrained model using the provided configuration. If a LoRA configuration is specified,
        it applies the LoRA configuration to the model for parameter-efficient fine-tuning.

        Args:
            model_config (`Optional[Dict]`, *optional*):
                The model configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ModelType`: The model instance loaded with optional LoRA fine-tuning.

        Raises:
            TypeError: If `model_config.lora` is neither a string nor a valid path.
        """
        model_config = model_config if model_config is not None else self.config.model_config.copy()

        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        # Initialize the model

        model = model_cls.from_pretrained(**model_config.model_cls_config)

        # Note: PEFT/LoRA should be configured via trainer_config['peft_config']
        # The trainer will handle PEFT model wrapping automatically

        return model.train()


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTaskWithCustomModel(SingleTrainTask, TaskWithCustomModel):

    def build_model(self, model_config: Optional[Dict] = None):
        """
        Builds a custom model using the provided configuration from the registry. If a LoRA configuration
        is provided for either text or vision models, it applies the configuration for parameter-efficient fine-tuning.

        Args:
            model_config (`Optional[Dict]`, *optional*):
                The model configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ModelType`: The custom model instance with optional LoRA fine-tuning.

        Raises:
            TypeError: If `model_config.lora` is not a valid `DictConfig` object.
        """

        model_config = model_config if model_config is not None else self.config.model_config.copy()

        # Get the model configuration and model class from the registry
        model_cfg_cls = registry.get_model_config_class(model_config.config_cls)
        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert model_cfg_cls is not None, "Model config {} not properly registered.".format(model_cfg_cls)

        # Initialize the model configuration and model
        config_cls_config = OmegaConf.to_container(model_config.config_cls_config, resolve=True)
        model_cfg = model_cfg_cls(**config_cls_config)
        model_cls_config = model_config.model_cls_config.copy()
        if "pretrained_model_name_or_path" in model_cls_config:
            model = model_cls.from_pretrained(config=model_cfg, **model_cls_config)
        else:
            model = model_cls(config=model_cfg)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0

        logger.info("=" * 50)
        logger.info("Model Parameter Summary")
        logger.info("=" * 50)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        logger.info(f"Trainable percentage: {trainable_percent:.2f}%")
        logger.info("=" * 50)

        # Note: For custom models with separate text/vision components,
        # you may need custom PEFT handling. Consider using trainer_config['peft_config'] instead.

        return model.train()


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("DatasetTrainTaskWithPretrainedModel")
class DatasetTrainTaskWithPretrainedModel(DatasetSingleTrainTask, SingleTrainTaskWithPretrainedModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithPretrainedModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        return DatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithPretrainedModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return DatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("IterableDatasetTrainTaskWithPretrainedModel")
class IterableDatasetTrainTaskWithPretrainedModel(IterableDatasetSingleTrainTask, SingleTrainTaskWithPretrainedModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithPretrainedModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        return IterableDatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithPretrainedModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return IterableDatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("DatasetTrainTaskWithCustomModel")
class DatasetTrainTaskWithCustomModel(DatasetSingleTrainTask, SingleTrainTaskWithCustomModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithCustomModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        return DatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithCustomModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return DatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("IterableDatasetTrainTaskWithCustomModel")
class IterableDatasetTrainTaskWithCustomModel(IterableDatasetSingleTrainTask, SingleTrainTaskWithCustomModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithCustomModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        return IterableDatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithCustomModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return IterableDatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )
