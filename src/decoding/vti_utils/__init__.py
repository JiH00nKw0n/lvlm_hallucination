"""
VTI Utils: Data loading and direction computation for VTI.

Reference: VTI/vti_utils/utils.py

VTI computes direction from pre-defined demos:
    - value: Correct (non-hallucinating) response
    - h_value: Hallucinating response
    - Direction: correct - hallucinating (각 layer의 last token)
"""

from .utils import (
    VTIArgs,
    process_image,
    mask_patches,
    get_prompts,
    get_demos,
    get_hiddenstates,
    obtain_textual_vti,
    get_visual_hiddenstates,
    obtain_visual_vti,
)

__all__ = [
    "VTIArgs",
    "process_image",
    "mask_patches",
    "get_prompts",
    "get_demos",
    "get_hiddenstates",
    "obtain_textual_vti",
    "get_visual_hiddenstates",
    "obtain_visual_vti",
]
