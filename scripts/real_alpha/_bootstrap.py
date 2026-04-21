"""Import bootstrap for the real-alpha scripts.

The repo's `src.runners`, `src.datasets`, `src.integrations`, and `src.common`
packages pull in heavy dependencies at import time (flex_attention, MMEEvaluator,
trl internals, etc.) that either don't install cleanly or break with newer
transformers releases on the training server.

We only need a handful of modules (`modeling_sae`, `configuration_sae`,
`trainer`, `cached_clip_pairs`), so instead of fighting the package `__init__`
chains, this bootstrap loads those modules directly from their file paths and
installs them into `sys.modules` under their canonical names.

Import this module first from any real-alpha script; it is idempotent.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_package(name: str, path: Path) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod
    return mod


def _load_module(name: str, file_path: Path) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None):
        return existing
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 1) Create minimal stub packages so "from src.X.Y import Z" doesn't trigger
#    the real __init__.py (which imports broken siblings).
_stub_package("src", REPO_ROOT / "src")
_stub_package("src.common", REPO_ROOT / "src" / "common")
_stub_package("src.models", REPO_ROOT / "src" / "models")
_stub_package("src.runners", REPO_ROOT / "src" / "runners")
_stub_package("src.datasets", REPO_ROOT / "src" / "datasets")

# 2) The registry module only depends on basic Python, load it so trainer.py's
#    `from src.common.registry import registry` works.
_load_module("src.common.registry", REPO_ROOT / "src" / "common" / "registry.py")

# 3) Load the SAE model modules (configuration first, then modeling).
_load_module("src.models.configuration_sae", REPO_ROOT / "src" / "models" / "configuration_sae.py")
_load_module("src.models.modeling_sae", REPO_ROOT / "src" / "models" / "modeling_sae.py")

# 4) Load the (new) trainer module.
_load_module("src.runners.trainer", REPO_ROOT / "src" / "runners" / "trainer.py")

# 5) Load the cached-pair dataset modules.
_load_module("src.datasets.cached_clip_pairs", REPO_ROOT / "src" / "datasets" / "cached_clip_pairs.py")
_load_module("src.datasets.cached_imagenet_pairs", REPO_ROOT / "src" / "datasets" / "cached_imagenet_pairs.py")
