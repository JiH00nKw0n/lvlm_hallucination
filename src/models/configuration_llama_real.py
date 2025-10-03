from typing import Union, Dict

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

LLAMA_REAL_TYPE = "llama_real"
REWEIGHT_ATTENTION_MODULE_TYPE = "reweight_attention"


class LLamaRealConfig(PretrainedConfig):
    def __init__(
            self,
            text_config: Union[AutoConfig, Dict] = None,
            additional_attention_module_config: Union[AutoConfig, Dict] = None,
            **kwargs
    ):
        if isinstance(additional_attention_module_config, dict):
            additional_attention_module_config["model_type"] = (
                additional_attention_module_config[
                    "model_type"] if "model_type" in additional_attention_module_config else ADDITIONAL_ATTENTION_MODULE_TYPE
            )
            additional_attention_module_config = CONFIG_MAPPING[additional_attention_module_config["model_type"]](
                **additional_attention_module_config
            )
        elif additional_attention_module_config is None:
            additional_attention_module_config = CONFIG_MAPPING[ADDITIONAL_ATTENTION_MODULE_TYPE]()

        self.additional_attention_module_config = additional_attention_module_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = None

        self.text_config = text_config
        super().__init__(**kwargs)


class ReweightAttentionConfig(PretrainedConfig):
    model_type = REWEIGHT_ATTENTION_MODULE_TYPE
    _name_or_path = "reweight_attention"

    def __init__(
            self,
            num_latents_value: int = 512,
            hidden_dim: int = 4096,
            latent_dim: int = 4096,
            cross_dim_head: int = 4096,
            **kwargs,
    ):
        self.num_latents_value = num_latents_value
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cross_dim_head = cross_dim_head

        super().__init__(**kwargs)


__all__ = [
    "LLamaRealConfig",
    "ReweightAttentionConfig"
]

AutoConfig.register(LLAMA_REAL_TYPE, LLamaRealConfig)
AutoConfig.register(REWEIGHT_ATTENTION_MODULE_TYPE, ReweightAttentionConfig)

LLamaRealConfig.register_for_auto_class()
ReweightAttentionConfig.register_for_auto_class()
