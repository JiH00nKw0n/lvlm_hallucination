from typing import Tuple

from transformers import PretrainedConfig
from transformers.models.auto import AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

REWEIGHT_ATTENTION_MODULE_TYPE = "reweight_attention"


class ReweightAttentionConfig(PretrainedConfig):
    model_type = REWEIGHT_ATTENTION_MODULE_TYPE
    _name_or_path = "reweight_attention"

    def __init__(
            self,
            num_latents_value: int = 512,
            hidden_dim: int = 4096,
            image_token_id: int = 32000,
            assistant_token_ids: Tuple[int] = (22933, 9047, 13566, 29901),
            **kwargs,
    ):
        self.num_latents_value = num_latents_value
        self.hidden_dim = hidden_dim
        self.image_token_id = image_token_id  # "<image>" token sequence
        self.assistant_token_ids = list(assistant_token_ids)  # "ASSISTANT:" token sequence

        super().__init__(**kwargs)


__all__ = [
    "ReweightAttentionConfig",
    "REWEIGHT_ATTENTION_MODULE_TYPE"
]

AutoConfig.register(REWEIGHT_ATTENTION_MODULE_TYPE, ReweightAttentionConfig)

ReweightAttentionConfig.register_for_auto_class()
