from typing import Union, Dict

from transformers import LlamaConfig
from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.utils import logging
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

from src.models.reweighting_module.configuration_module import ReweightAttentionConfig, REWEIGHT_ATTENTION_MODULE_TYPE

logger = logging.get_logger(__name__)

LLAMA_REAL_TYPE = "llama_real"


class LLamaRealConfig(PretrainedConfig):
    model_type = LLAMA_REAL_TYPE

    def __init__(
            self,
            text_config: Union[AutoConfig, Dict] = None,
            additional_attention_module_config: Union[AutoConfig, Dict] = None,
            **kwargs
    ):
        if isinstance(additional_attention_module_config, dict):
            additional_attention_module_config["model_type"] = (
                additional_attention_module_config[
                    "model_type"] if "model_type" in additional_attention_module_config else REWEIGHT_ATTENTION_MODULE_TYPE
            )
            additional_attention_module_config = CONFIG_MAPPING[additional_attention_module_config["model_type"]](
                **additional_attention_module_config
            )
        elif additional_attention_module_config is None:
            additional_attention_module_config = CONFIG_MAPPING[REWEIGHT_ATTENTION_MODULE_TYPE]()

        self.additional_attention_module_config: ReweightAttentionConfig = additional_attention_module_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = None

        self.text_config: LlamaConfig = text_config
        super().__init__(**kwargs)


__all__ = [
    "LLamaRealConfig",
]

AutoConfig.register(LLAMA_REAL_TYPE, LLamaRealConfig)

LLamaRealConfig.register_for_auto_class()

HARDCODED_CONFIG_FOR_MODELS["llama-real"] = LLamaRealConfig
