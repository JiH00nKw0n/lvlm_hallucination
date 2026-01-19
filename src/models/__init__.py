from transformers import LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

from src.common.registry import registry

registry.register_model("LlavaForConditionalGeneration")(LlavaForConditionalGeneration)
registry.register_model("LlavaNextForConditionalGeneration")(LlavaNextForConditionalGeneration)

__all__ = [
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
]
