from transformers import LlavaConfig, LlavaForConditionalGeneration

from src.common.registry import registry
from src.models.llama_real import LLamaRealConfig, LlamaRealModel
from src.models.llava import CustomLlavaForConditionalGeneration
from src.models.reweighting_module import ReweightAttentionConfig, ReweightAttentionModule

registry.register_model("LlavaForConditionalGeneration")(LlavaForConditionalGeneration)
registry.register_model_config("LlavaConfig")(LlavaConfig)

__all__ = [
    "LLamaRealConfig",
    "LlamaRealModel",
    "ReweightAttentionConfig",
    "ReweightAttentionModule",
    "CustomLlavaForConditionalGeneration",
]
