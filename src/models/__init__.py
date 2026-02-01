from transformers import LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

from .configuration_sae import (
    BatchTopKSAEConfig,
    MatryoshkaSAEConfig,
    TopKSAEConfig,
    VLBatchTopKSAEConfig,
    VLMatryoshkaSAEConfig,
    VLTopKSAEConfig,
)
from .modeling_sae import (
    BatchTopKSAE,
    MatryoshkaSAE,
    TopKSAE,
    VLBatchTopKSAE,
    VLMatryoshkaSAE,
    VLTopKSAE,
)

from src.common.registry import registry

registry.register_model("LlavaForConditionalGeneration")(LlavaForConditionalGeneration)
registry.register_model("LlavaNextForConditionalGeneration")(LlavaNextForConditionalGeneration)
registry.register_model("TopKSAE")(TopKSAE)
registry.register_model("BatchTopKSAE")(BatchTopKSAE)
registry.register_model("MatryoshkaSAE")(MatryoshkaSAE)
registry.register_model("VLTopKSAE")(VLTopKSAE)
registry.register_model("VLBatchTopKSAE")(VLBatchTopKSAE)
registry.register_model("VLMatryoshkaSAE")(VLMatryoshkaSAE)

registry.register_model_config("TopKSAEConfig")(TopKSAEConfig)
registry.register_model_config("BatchTopKSAEConfig")(BatchTopKSAEConfig)
registry.register_model_config("MatryoshkaSAEConfig")(MatryoshkaSAEConfig)
registry.register_model_config("VLTopKSAEConfig")(VLTopKSAEConfig)
registry.register_model_config("VLBatchTopKSAEConfig")(VLBatchTopKSAEConfig)
registry.register_model_config("VLMatryoshkaSAEConfig")(VLMatryoshkaSAEConfig)

__all__ = [
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "TopKSAE",
    "BatchTopKSAE",
    "MatryoshkaSAE",
    "VLTopKSAE",
    "VLBatchTopKSAE",
    "VLMatryoshkaSAE",
    "TopKSAEConfig",
    "BatchTopKSAEConfig",
    "MatryoshkaSAEConfig",
    "VLTopKSAEConfig",
    "VLBatchTopKSAEConfig",
    "VLMatryoshkaSAEConfig",
]
