"""
Hallucination Mitigation Decoding Strategies

This module provides a unified interface for 9 hallucination mitigation methods
for Large Vision-Language Models (LVLMs).

Supported Models:
    - LLaVA (model_type="llava")
    - LLaVA-NeXT (model_type="llava_next")
    - Qwen2-VL (model_type="qwen2_vl")
    - Qwen2.5-VL (model_type="qwen2_5_vl")
    - InstructBLIP (model_type="instructblip")

Architecture:
    BaseMitigator
    ├── VCDMitigator          # Visual Contrastive Decoding
    ├── AvisCMitigator        # Attention Vision Calibration
    ├── VISTAMitigator        # Visual Steering Vector
    ├── VTIMitigator          # Visual-Textual Intervention
    ├── MiddleLayersMitigator # Middle Layer Attention Boost
    ├── FarSightMitigator     # Upper Triangular Penalty
    ├── DecoMitigator         # Early Exit Calibration
    ├── OPERAMitigator        # Beam Search with Rollback
    └── OctopusMitigator      # Dynamic Strategy Selection (Trainable)

Usage:
    from src.decoding import VCDMitigator, get_mitigator

    # Direct instantiation
    with VCDMitigator(model, model_type="llava", alpha=1.0) as mitigator:
        output = mitigator.generate(input_ids, pixel_values=pixel_values)

    # Via registry
    mitigator = get_mitigator('vcd', model, model_type="llava", alpha=1.0)
    with mitigator:
        output = mitigator.generate(input_ids, pixel_values=pixel_values)
"""

from typing import Dict, Type

from .base import (
    BaseMitigator,
    DecodeResult,
    MitigatorConfig,
    TrainableMitigator,
    ModelHelper,
    add_diffusion_noise,
    sample_top_p,
)

# Individual method imports
from .vcd import VCDMitigator
from .avisc import AvisCMitigator
from .vista import VISTAMitigator
from .vti import VTIMitigator
from .middle_layers import MiddleLayersMitigator
from .farsight import FarSightMitigator
from .deco import DecoMitigator
from .opera import OPERAMitigator
from .octopus import OctopusMitigator, OctopusClassifier


# Registry of all mitigators
MITIGATOR_REGISTRY: Dict[str, Type[BaseMitigator]] = {
    'vcd': VCDMitigator,
    'avisc': AvisCMitigator,
    'vista': VISTAMitigator,
    'vti': VTIMitigator,
    'middle_layers': MiddleLayersMitigator,
    'farsight': FarSightMitigator,
    'deco': DecoMitigator,
    'opera': OPERAMitigator,
    'octopus': OctopusMitigator,
}


def get_mitigator(
    name: str,
    model,
    **kwargs,
) -> BaseMitigator:
    """
    Get a mitigator by name from the registry.

    Args:
        name: Mitigator name (e.g., 'vcd', 'vista', 'opera')
        model: The VLM model
        **kwargs: Additional arguments including model_type

    Returns:
        Instantiated mitigator

    Raises:
        ValueError: If name is not in registry

    Example:
        >>> mitigator = get_mitigator('vcd', model, model_type="llava", alpha=1.0)
        >>> with mitigator:
        ...     output = mitigator.generate(input_ids, pixel_values=pixel_values)
    """
    name = name.lower()
    if name not in MITIGATOR_REGISTRY:
        available = ', '.join(sorted(MITIGATOR_REGISTRY.keys()))
        raise ValueError(f"Unknown mitigator: {name}. Available: {available}")

    return MITIGATOR_REGISTRY[name](model, **kwargs)


def list_mitigators() -> Dict[str, str]:
    """
    List all available mitigators with their descriptions.

    Returns:
        Dictionary mapping mitigator names to their docstrings
    """
    return {
        name: cls.__doc__.split('\n')[1].strip() if cls.__doc__ else "No description"
        for name, cls in MITIGATOR_REGISTRY.items()
    }


def register_mitigator(name: str):
    """
    Decorator to register a custom mitigator.

    Example:
        @register_mitigator('custom')
        class CustomMitigator(BaseMitigator):
            ...
    """
    def decorator(cls: Type[BaseMitigator]):
        MITIGATOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator


__all__ = [
    # Base classes
    'BaseMitigator',
    'TrainableMitigator',
    'DecodeResult',
    'MitigatorConfig',
    'ModelHelper',

    # Method implementations
    'VCDMitigator',
    'AvisCMitigator',
    'VISTAMitigator',
    'VTIMitigator',
    'MiddleLayersMitigator',
    'FarSightMitigator',
    'DecoMitigator',
    'OPERAMitigator',
    'OctopusMitigator',
    'OctopusClassifier',

    # Utilities
    'get_mitigator',
    'list_mitigators',
    'register_mitigator',
    'add_diffusion_noise',
    'sample_top_p',
    'MITIGATOR_REGISTRY',
]
