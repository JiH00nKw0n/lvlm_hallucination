"""
Compatibility Test Matrix: src/decoding × src/models

This file documents and tests the compatibility between the 9 hallucination
mitigation methods and the 5 VLM model architectures.

COMPATIBILITY MATRIX:
======================

| Method         | LLaVA | LLaVA-NeXT | Qwen2-VL | Qwen2.5-VL | InstructBLIP |
|----------------|-------|------------|----------|------------|--------------|
| VCD            |   ✓   |     ✓      |    ✓     |     ✓      |      ✓       |
| AvisC          |   ✓   |     ✓      |    ✓     |     ✓      |      ✓       |
| VISTA          |   ✓   |     ✓      |    ✓     |     ✓      |      ✓       |
| VTI            |   ✓   |     ✓      |    ✓     |     ✓      |      ✓       |
| MiddleLayers   |   ✓   |     ✓      |    ✓*    |     ✓*     |      ✓       |
| FarSight       |   ✓   |     ✓      |    ✓*    |     ✓*     |      ✓       |
| Deco           |   ✓   |     ✓      |    ✓     |     ✓      |      ✓       |
| OPERA          |   ✓   |     ✓      |    ✓     |     ✓      |      ✓       |
| Octopus        |   ✓   |     ✓      |    ✓     |     ✓      |      ✓       |

* Qwen models use SDPA/Flash Attention by default. Forward replacement methods
  (MiddleLayers, FarSight) automatically switch to eager attention.

MODEL-SPECIFIC NOTES:
=====================

LLaVA/LLaVA-NeXT:
- Layer access: model.language_model.model.layers
- Image input: `images` kwarg
- Image token ID: 32000 (default)
- 576 image patches (24x24) for standard 336px images

Qwen2-VL/Qwen2.5-VL:
- Layer access: model.model.layers
- Image input: `pixel_values` kwarg + `image_grid_thw`
- Vision tokens: <|vision_start|> (151652) to <|vision_end|> (151653)
- Uses 3D rotary position embeddings (mrope)
- Dynamic image resolution support

InstructBLIP:
- Layer access: model.language_model.model.layers
- Image input: `pixel_values` kwarg
- Requires: `qformer_input_ids`, `qformer_attention_mask`
- Q-Former outputs: first 32 tokens
- Can use different LLM backends (Vicuna, OPT, etc.)

USAGE EXAMPLES:
===============

```python
from src.decoding import VCDMitigator, VISTAMitigator, get_mitigator

# LLaVA
with VCDMitigator(llava_model, model_type="llava", alpha=1.0) as mitigator:
    output = mitigator.generate(input_ids, images=pixel_values)

# Qwen2-VL
with VCDMitigator(qwen_model, model_type="qwen2_vl", alpha=1.0) as mitigator:
    output = mitigator.generate(input_ids, pixel_values=pixel_values,
                                 image_grid_thw=image_grid_thw)

# InstructBLIP
with VCDMitigator(instructblip_model, model_type="instructblip", alpha=1.0) as mitigator:
    output = mitigator.generate(input_ids, pixel_values=pixel_values,
                                 qformer_input_ids=qformer_input_ids)
```

TRAINING SUPPORT:
=================

| Method         | Requires Training | Trainable Component        |
|----------------|-------------------|----------------------------|
| VCD            | No                | -                          |
| AvisC          | No                | -                          |
| VISTA          | Optional          | VSV (Visual Steering Vec)  |
| VTI            | Optional          | VTI directions             |
| MiddleLayers   | Optional          | MLP classifier             |
| FarSight       | No                | -                          |
| Deco           | No                | -                          |
| OPERA          | No                | -                          |
| Octopus        | Yes               | Action classifier (DPO)    |
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Type

# Model types supported
SUPPORTED_MODELS = [
    "llava",
    "llava_next",
    "qwen2_vl",
    "qwen2_5_vl",
    "instructblip",
]

# Method configurations
METHOD_CONFIGS = {
    "vcd": {
        "class": "VCDMitigator",
        "module": "contrastive",
        "requires_training": False,
        "default_params": {"alpha": 1.0, "beta": 0.1, "noise_step": 500},
    },
    "avisc": {
        "class": "AvisCMitigator",
        "module": "contrastive",
        "requires_training": False,
        "default_params": {"alpha": 1.0, "layer_gamma": 0.5, "lamb": 100.0},
    },
    "vista": {
        "class": "VISTAMitigator",
        "module": "hook_based",
        "requires_training": False,  # Optional
        "default_params": {"steering_strength": 0.1},
    },
    "vti": {
        "class": "VTIMitigator",
        "module": "hook_based",
        "requires_training": False,  # Optional
        "default_params": {"alpha_text": 0.8, "alpha_image": 0.9},
    },
    "middle_layers": {
        "class": "MiddleLayersMitigator",
        "module": "forward_replacement",
        "requires_training": False,  # Optional
        "default_params": {"alpha": 0.5, "target_layers": list(range(5, 18))},
    },
    "farsight": {
        "class": "FarSightMitigator",
        "module": "forward_replacement",
        "requires_training": False,
        "default_params": {"decay_factor": 0.188, "use_alibi": True},
    },
    "deco": {
        "class": "DecoMitigator",
        "module": "contrastive",
        "requires_training": False,
        "default_params": {"alpha": 0.6, "early_exit_layers": list(range(20, 29))},
    },
    "opera": {
        "class": "OPERAMitigator",
        "module": "beam_rollback",
        "requires_training": False,
        "default_params": {"num_beams": 5, "scale_factor": 50.0, "threshold": 15},
    },
    "octopus": {
        "class": "OctopusMitigator",
        "module": "dynamic_strategy",
        "requires_training": True,
        "default_params": {"d_model": 4096},
    },
}


def get_model_specific_params(model_type: str) -> Dict:
    """Get model-specific default parameters."""
    model_type = model_type.lower().replace("-", "_")

    params = {
        "llava": {
            "image_kwarg": "images",
            "image_token_id": 32000,
            "lamb": 100.0,  # AvisC
        },
        "llava_next": {
            "image_kwarg": "images",
            "image_token_id": 32000,
            "lamb": 100.0,
        },
        "qwen2_vl": {
            "image_kwarg": "pixel_values",
            "extra_kwargs": ["image_grid_thw"],
            "lamb": 50.0,  # AvisC - may need tuning
        },
        "qwen2_5_vl": {
            "image_kwarg": "pixel_values",
            "extra_kwargs": ["image_grid_thw"],
            "lamb": 50.0,
        },
        "instructblip": {
            "image_kwarg": "pixel_values",
            "extra_kwargs": ["qformer_input_ids", "qformer_attention_mask"],
            "lamb": 0.99,  # AvisC - different threshold for InstructBLIP
        },
    }

    return params.get(model_type, params["llava"])


def check_layer_access(model: nn.Module) -> Dict[str, bool]:
    """Check if we can access the necessary model components."""
    results = {}

    # Check layer access
    try:
        from src.decoding.base import BaseMitigator

        class _TestMitigator(BaseMitigator):
            def setup(self): pass
            def cleanup(self): pass
            def generate(self, *args, **kwargs): pass

        test = _TestMitigator(model)
        layers = test._get_model_layers()
        results["layers"] = layers is not None and len(layers) > 0
    except Exception as e:
        results["layers"] = False
        results["layers_error"] = str(e)

    # Check norm and lm_head access
    try:
        norm, lm_head = test._get_norm_and_lm_head()
        results["norm_lm_head"] = norm is not None and lm_head is not None
    except Exception as e:
        results["norm_lm_head"] = False
        results["norm_lm_head_error"] = str(e)

    # Check embed_tokens access
    try:
        embed = test._get_embed_tokens()
        results["embed_tokens"] = embed is not None
    except Exception as e:
        results["embed_tokens"] = False
        results["embed_tokens_error"] = str(e)

    return results


def run_compatibility_check(model: nn.Module, model_type: str) -> Dict[str, Dict]:
    """
    Run compatibility check for all methods with a given model.

    Args:
        model: The VLM model to test
        model_type: Type of model (llava, qwen2_vl, etc.)

    Returns:
        Dictionary with compatibility results for each method
    """
    from src.decoding import (
        VCDMitigator, AvisCMitigator, VISTAMitigator, VTIMitigator,
        MiddleLayersMitigator, FarSightMitigator, DecoMitigator,
        OPERAMitigator, OctopusMitigator,
    )

    results = {}
    model_type = model_type.lower().replace("-", "_")

    # Test each method
    methods = [
        ("vcd", VCDMitigator, {"alpha": 1.0}),
        ("avisc", AvisCMitigator, {"alpha": 1.0}),
        ("vista", VISTAMitigator, {}),
        ("vti", VTIMitigator, {}),
        ("middle_layers", MiddleLayersMitigator, {"target_layers": [5, 6, 7]}),
        ("farsight", FarSightMitigator, {"target_layers": [0, 1, 2]}),
        ("deco", DecoMitigator, {"early_exit_layers": [10, 11, 12]}),
        ("opera", OPERAMitigator, {}),
        ("octopus", OctopusMitigator, {"d_model": model.config.hidden_size}),
    ]

    for name, cls, params in methods:
        try:
            mitigator = cls(model, model_type=model_type, **params)

            # Test setup
            mitigator.setup()
            setup_ok = True

            # Test cleanup
            mitigator.cleanup()
            cleanup_ok = True

            results[name] = {
                "compatible": True,
                "setup": setup_ok,
                "cleanup": cleanup_ok,
            }
        except Exception as e:
            results[name] = {
                "compatible": False,
                "error": str(e),
            }

    return results


def print_compatibility_matrix(results: Dict[str, Dict[str, Dict]]) -> None:
    """Print a formatted compatibility matrix."""
    methods = list(METHOD_CONFIGS.keys())
    models = list(results.keys())

    # Header
    print("\nCOMPATIBILITY MATRIX")
    print("=" * 80)
    header = "| Method         |" + "|".join(f" {m:^10} " for m in models) + "|"
    print(header)
    print("|" + "-" * 15 + "|" + "|".join("-" * 12 for _ in models) + "|")

    # Rows
    for method in methods:
        row = f"| {method:^13} |"
        for model in models:
            if model in results and method in results[model]:
                status = "✓" if results[model][method]["compatible"] else "✗"
            else:
                status = "?"
            row += f" {status:^10} |"
        print(row)

    print("=" * 80)


if __name__ == "__main__":
    print(__doc__)
