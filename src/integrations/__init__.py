from transformers.modeling_utils import AttentionInterface

from .flex_attention import flex_attention_forward

# Global AttentionInterface shared by all models which do not need to overwrite any of the existing ones
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()

ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention_forward

__all__ = [
    "ALL_ATTENTION_FUNCTIONS",
]
