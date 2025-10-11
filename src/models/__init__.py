from transformers import LlavaConfig, LlavaModel

from src.common.registry import registry
from src.models.llama_real import LLamaRealConfig, LlamaRealModel
from src.models.reweighting_module import ReweightAttentionConfig, ReweightAttentionModule

registry.register_model("LlavaModel")(LlavaModel)
registry.register_model_config("LlavaConfig")(LlavaConfig)

__all__ = [
    "LLamaRealConfig",
    "LlamaRealModel",
    "ReweightAttentionConfig",
    "ReweightAttentionModule",
]