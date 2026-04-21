"""Extended bootstrap that loads new refactored packages on top of _bootstrap.

Usage in real-alpha scripts that need ``src.metrics.*`` etc.:

    import _bootstrap_v2  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the original bootstrap runs first.
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import _bootstrap  # noqa: F401,E402
from _bootstrap import _stub_package, _load_module, REPO_ROOT  # noqa: E402

# New packages from refactoring.
_stub_package("src.metrics", REPO_ROOT / "src" / "metrics")
_load_module("src.metrics.normalize", REPO_ROOT / "src" / "metrics" / "normalize.py")
_load_module("src.metrics.alignment", REPO_ROOT / "src" / "metrics" / "alignment.py")
_load_module("src.metrics.hungarian", REPO_ROOT / "src" / "metrics" / "hungarian.py")

_stub_package("src.training", REPO_ROOT / "src" / "training")
_load_module("src.training.losses", REPO_ROOT / "src" / "training" / "losses.py")
_load_module("src.training.permutation", REPO_ROOT / "src" / "training" / "permutation.py")

_stub_package("src.configs", REPO_ROOT / "src" / "configs")
_load_module("src.configs.experiment", REPO_ROOT / "src" / "configs" / "experiment.py")

# New datasets
_load_module("src.datasets.synthetic_pairs", REPO_ROOT / "src" / "datasets" / "synthetic_pairs.py")
_load_module("src.datasets.clip_utils", REPO_ROOT / "src" / "datasets" / "clip_utils.py")

# New trainers
_load_module("src.runners.synthetic_trainers", REPO_ROOT / "src" / "runners" / "synthetic_trainers.py")
