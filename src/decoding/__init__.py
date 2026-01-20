"""
Hallucination Mitigation Decoding Strategies

This module provides a unified interface for 12 hallucination mitigation methods
for Large Vision-Language Models (LVLMs).

Supported Models:
    - LLaVA (model_type="llava")
    - LLaVA-NeXT (model_type="llava_next")
    - Qwen2-VL (model_type="qwen2_vl")
    - Qwen2.5-VL (model_type="qwen2_5_vl")

Architecture:
    BaseMitigator
    ├── GreedyMitigator       # Deterministic decoding (baseline)
    ├── VCDMitigator          # Visual Contrastive Decoding
    ├── AvisCMitigator        # Attention Vision Calibration
    ├── VISTAMitigator        # Visual Steering Vector
    ├── VTIMitigator          # Visual-Textual Intervention
    ├── MiddleLayersMitigator # Middle Layer Attention Boost
    ├── FarSightMitigator     # Upper Triangular Penalty
    ├── DecoMitigator         # Early Exit Calibration
    ├── OPERAMitigator        # Beam Search with Rollback
    ├── OctopusMitigator      # Dynamic Strategy Selection (Trainable)
    ├── SSLMitigator          # SAE-based Steering (LLaVA-NeXT only)
    └── SAVEMitigator         # SAVE steering (LLaVA-NeXT only)

Usage:
    from src.decoding import VCDMitigator, get_mitigator

    # Direct instantiation
    with VCDMitigator(model, model_type="llava", alpha=1.0) as mitigator:
        output = mitigator.generate(input_ids, pixel_values=pixel_values)

    # Via registry
    mitigator = get_mitigator('VCDMitigator', model, model_type="llava", alpha=1.0)
    with mitigator:
        output = mitigator.generate(input_ids, pixel_values=pixel_values)
"""

from typing import Dict, Type

from .base import (
    BaseMitigator,
    MitigatorConfig,
    TrainableMitigator,
    ModelHelper,
    add_diffusion_noise,
    sample_top_p,
)

# Individual method imports
from .greedy import GreedyMitigator
from .vcd import VCDMitigator
from .avisc import AvisCMitigator
from .vista import VISTAMitigator
from .vti import VTIMitigator
from .middle_layers import MiddleLayersMitigator
from .farsight import FarSightMitigator
from .deco import DecoMitigator
from .opera import OPERAMitigator
from .octopus import OctopusMitigator, OctopusClassifier
from .ssl import SSLMitigator
from .save import SAVEMitigator
from src.common.registry import registry


# Register mitigators in the global registry
registry.register_mitigator('GreedyMitigator')(GreedyMitigator)
registry.register_mitigator('VCDMitigator')(VCDMitigator)
registry.register_mitigator('AvisCMitigator')(AvisCMitigator)
registry.register_mitigator('VISTAMitigator')(VISTAMitigator)
registry.register_mitigator('VTIMitigator')(VTIMitigator)
registry.register_mitigator('MiddleLayersMitigator')(MiddleLayersMitigator)
registry.register_mitigator('FarSightMitigator')(FarSightMitigator)
registry.register_mitigator('DecoMitigator')(DecoMitigator)
registry.register_mitigator('OPERAMitigator')(OPERAMitigator)
registry.register_mitigator('OctopusMitigator')(OctopusMitigator)
registry.register_mitigator('SSLMitigator')(SSLMitigator)
registry.register_mitigator('SAVEMitigator')(SAVEMitigator)

# Backward-compatible alias
MITIGATOR_REGISTRY: Dict[str, Type[BaseMitigator]] = registry.mapping["mitigator_name_mapping"]


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
        >>> mitigator = get_mitigator('VCDMitigator', model, model_type="llava", alpha=1.0)
        >>> with mitigator:
        ...     output = mitigator.generate(input_ids, pixel_values=pixel_values)
    """
    mitigator_cls = registry.get_mitigator_class(name)
    if mitigator_cls is None:
        available = ', '.join(registry.list_mitigators())
        raise ValueError(f"Unknown mitigator: {name}. Available: {available}")

    return mitigator_cls(model, **kwargs)


def list_mitigators() -> Dict[str, str]:
    """
    List all available mitigators with their descriptions.

    Returns:
        Dictionary mapping mitigator names to their docstrings
    """
    mitigators = registry.mapping["mitigator_name_mapping"]
    return {
        name: cls.__doc__.split('\n')[1].strip() if cls.__doc__ else "No description"
        for name, cls in mitigators.items()
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
        registry.register_mitigator(name)(cls)
        return cls
    return decorator


__all__ = [
    # Base classes
    'BaseMitigator',
    'TrainableMitigator',
    'MitigatorConfig',
    'ModelHelper',

    # Method implementations
    'GreedyMitigator',
    'VCDMitigator',
    'AvisCMitigator',
    'VISTAMitigator',
    'VTIMitigator',
    'MiddleLayersMitigator',
    'FarSightMitigator',
    'DecoMitigator',
    'OPERAMitigator',
    'OctopusMitigator',
    'SSLMitigator',
    'SAVEMitigator',
    'OctopusClassifier',

    # Utilities
    'get_mitigator',
    'list_mitigators',
    'register_mitigator',
    'add_diffusion_noise',
    'sample_top_p',
    'MITIGATOR_REGISTRY',
]
