"""
VISTA Utils: On-the-fly VSV computation for VISTA.

Reference:
    - VISTA/steering_vector.py (obtain_vsv, get_hiddenstates)
    - VISTA/model_loader.py:364-430 (prepare_pos_prompt, prepare_neg_prompt)

VISTA computes Visual Steering Vectors per-image at inference time:
    - pos_kwargs: Full prompt with image
    - neg_kwargs: Null prompt without image (text-only)
    - Direction: pos - neg (with_image - without_image)
"""

from .steering_vector import (
    obtain_vsv,
    get_hiddenstates,
    ForwardTrace,
    ForwardTracer,
)

from .prompt_utils import (
    prepare_pos_prompt,
    prepare_neg_prompt,
    prepare_null_prompt,
    prepare_vsv_kwargs_pair,
)

__all__ = [
    "obtain_vsv",
    "get_hiddenstates",
    "ForwardTrace",
    "ForwardTracer",
    "prepare_pos_prompt",
    "prepare_neg_prompt",
    "prepare_null_prompt",
    "prepare_vsv_kwargs_pair",
]
