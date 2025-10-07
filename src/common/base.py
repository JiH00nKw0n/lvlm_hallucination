import logging
from typing import Dict
from typing import Union, Optional, TypeVar, Callable

import numpy as np
from omegaconf import OmegaConf, DictConfig
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from transformers import ProcessorMixin
from transformers.utils import PaddingStrategy, add_end_docstrings

logger = logging.getLogger(__name__)

ProcessorType = TypeVar("ProcessorType", bound=ProcessorMixin)

__all__ = [
    "BaseCollator",
    "BASE_COLLATOR_DOCSTRING",
    "BaseConfig",
]

BASE_COLLATOR_DOCSTRING = """
A collator class for processing inputs with dynamic padding, truncation, and tensor conversion.

Args:
    processor ([`ProcessorMixin`]):
        The processor used to encode the data (e.g., tokenizer).
    padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `'longest'`):
        Padding strategy to apply. It determines whether to pad, and if so, how.
        Can be a boolean or a padding strategy ('longest', 'max_length', 'do_not_pad').
    truncation (`bool`, *optional*, defaults to `True`):
        Whether to truncate the inputs to the maximum length. If True, inputs will be truncated
        to the max_length specified.
    max_length (`int`, *optional*, defaults to 64):
        Maximum length of the inputs after padding or truncation. Inputs longer than this will be
        truncated, and shorter ones will be padded.
    pad_to_multiple_of (`int`, *optional*):
        If specified, pads the input to a multiple of this value. This is useful when working
        with certain models that require input sequences to have a specific length that is a multiple
        of a particular value.
    return_tensors (`str`, *optional*, defaults to `'pt'`):
        The format of the tensors to return. Can be 'np' for NumPy arrays, 'pt' for PyTorch tensors,
        or 'tf' for TensorFlow tensors.
"""


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
class BaseCollator:
    """
    An abstract base class for collators that handle dynamic padding, truncation, and tensor conversion.
    This class provides the common structure for subclasses that need to process input data with these
    features before passing the data to a model processor.

    Subclasses should implement the `__call__` method to define how input data is processed.

    Raises:
        NotImplementedError:
            If the `__call__` method is not implemented in a subclass.
    """
    processor: Union[ProcessorType | Callable]
    padding: Union[bool, str, PaddingStrategy] = "longest"
    truncation: bool = True
    max_length: Optional[int] = 4096
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    color_model: str = "RGB"
    seed: Optional[int] = 2025
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        """Initialize RNG with seed after dataclass initialization."""
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)


@dataclass(
    config=ConfigDict(
        extra='ignore', frozen=True, strict=True, validate_assignment=True
    )
)
class BaseConfig:
    """
    Base configuration class that contains settings for model, processor, dataset, collator, and run configurations.

    Args:
        model (Dict):
            Dictionary containing model configuration settings.
        processor (Dict):
            Dictionary containing processor configuration settings.
        dataset (Dict):
            Dictionary containing dataset configuration settings.
        run (Dict):
            Dictionary containing run settings.

    Properties:
        model_config (DictConfig):
            Returns the model configuration as an `OmegaConf` object.
        processor_config (DictConfig):
            Returns the processor configuration as an `OmegaConf` object.
        dataset_config (Dict):
            Returns the dataset configuration as a Python dictionary.
        run_config (DictConfig):
            Returns the run configuration as an `OmegaConf` object.
    """
    model: Dict
    processor: Dict
    dataset: Dict
    run: Dict

    @property
    def model_config(self) -> DictConfig:
        return OmegaConf.create(self.model)

    @property
    def processor_config(self) -> DictConfig:
        return OmegaConf.create(self.processor)

    @property
    def dataset_config(self) -> Dict:
        return self.dataset

    @property
    def run_config(self) -> DictConfig:
        return OmegaConf.create(self.run)
