"""Synthetic α / λ sweep pipeline (paper §5, Fig 1 + Fig 2).

For each (alpha, latent_size, seed, method) combo:
  1. Generate synthetic paired data via SyntheticPairedBuilder.
  2. Train SAE for `method`.
  3. Compute (CR, RE, GRE, ESim, FSim) — the synthetic metric suite.
  4. Save W_enc / W_dec / b_enc / b_dec for offline replay.

The plot scripts under clean.src.plotting.{alpha,lambda}_sweep consume the
saved `params/run_*.npz` shards to draw the final figures.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clean.src.utils.config import Config

logger = logging.getLogger(__name__)


def run(cfg: Config, stage: str = "all") -> None:
    """Synthetic sweep is most cleanly handled by the existing v2 driver,
    which already supports the YAML schema in this folder. We delegate by
    importing the legacy entrypoint and feeding it our config.

    The point of `clean/` for synthetic is YAML hygiene + colocation —
    the generation/training math is unchanged from the validated v2 code.
    """
    out_root = Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Wire to the legacy synthetic_v2 driver. It accepts a YAML path
    # produced from `cfg`. We materialize a temp YAML here.
    import tempfile
    import yaml

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump({
        "data": cfg.data.__dict__,
        "training": cfg.training.__dict__,
        "methods": [m.__dict__ for m in cfg.methods],
        "sweep": cfg.sweep.__dict__,
        "output": cfg.output.__dict__,
    }, tmp)
    tmp.close()

    logger.info("[synthetic_sweep] delegating to legacy run_synthetic_v2.py with %s", tmp.name)
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "run_synthetic_v2.py", "--config", tmp.name])
    logger.info("[done] synthetic_sweep → %s", out_root)
