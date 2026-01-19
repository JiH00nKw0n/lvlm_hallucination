from typing import Dict, List, Optional

from omegaconf import OmegaConf, DictConfig
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from src.common.base import BaseConfig

__all__ = ["TrainConfig", "EvaluateConfig"]


@dataclass(
    config=ConfigDict(
        extra='ignore', frozen=True, strict=True, validate_assignment=True
    )
)
class TrainConfig(BaseConfig):
    """
    Configuration class for training, which extends `BaseConfig` by adding trainer-specific settings.

    Args:
        trainer (Dict):
            Dictionary containing trainer configuration settings.
        collator (Dict):
            Dictionary containing collator configuration settings.

    Properties:
        trainer_config (Dict):
            Returns the trainer configuration as a dictionary.
        collator_config (DictConfig):
            Returns the collator configuration as an `OmegaConf` object.
    """
    collator: Dict
    trainer: Dict

    @property
    def trainer_config(self) -> Dict:
        return self.trainer

    @property
    def collator_config(self) -> DictConfig:
        return OmegaConf.create(self.collator)


@dataclass(
    config=ConfigDict(
        extra='ignore', frozen=True, strict=True, validate_assignment=True
    )
)
class EvaluateConfig(BaseConfig):
    """
    Configuration class for evaluation, which extends `BaseConfig` by adding evaluator-specific settings.

    Args:
        evaluator (List):
            List containing evaluator configuration settings.
        dataset (List):
            List containing dataset configuration settings.

    Properties:
        evaluator_config (List):
            Returns the evaluator configuration as a list.
        dataset_config (List):
            Returns the dataset configuration as a list.
    """
    dataset: List
    evaluator: List
    mitigators: Optional[List] = None

    @property
    def evaluator_config(self) -> List:
        return self.evaluator

    @property
    def dataset_config(self) -> List:
        return self.dataset
